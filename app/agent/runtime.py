"""Agent Runtime 核心循环（对应 pi-agent-core 的定位，目标 <500 行）。

循环：接收输入 → 构建 context → LLM → 解析 →（有工具调用：执行、结果
回灌、继续 loop；无：返回最终答案）。退出条件三选一：
① LLM 返回纯文本；② 达到 max_rounds 熔断兜底；③ 致命错误向上抛。

设计要点：
- assistant 的 tool_calls 消息必须原样落历史（OpenAI 协议强约束，
  tool 消息必须紧跟带对应 tool_call_id 的 assistant 消息，漏掉会 400）。
- 工具异常转字符串"喂回"LLM 而非抛给用户：错误处理的对象是"循环"，
  工具失败是 LLM 可以读取并自我修正的 Observation；自修正不能无限
  递归，所以用 max_rounds 做资源护栏（刻意偏离 pi 的无步数设计）。
"""
import logging
import time
from dataclasses import dataclass, field
from typing import List, Optional

from ..tools.base import ToolRegistry
from ..tools.todo import set_current_session
from .context import ContextBuilder, summarize_with_llm
from .memory import build_system_prompt
from .parser import BaseParser, FunctionCallingParser
from .steering import SteeringQueue

logger = logging.getLogger("mini-agent.runtime")


@dataclass
class ToolCallRecord:
    name: str
    args: dict
    result: str
    duration_ms: int
    round: int
    status: str = "ok"


@dataclass
class AgentReply:
    content: str
    rounds_used: int
    truncated: bool
    tool_calls: List[ToolCallRecord] = field(default_factory=list)


class AgentRuntime:
    def __init__(self, llm, registry: ToolRegistry, store,
                 parser: Optional[BaseParser] = None,
                 max_rounds: int = 8,
                 steering_queue: Optional[SteeringQueue] = None,
                 context_builder: Optional[ContextBuilder] = None):
        self.llm = llm
        self.registry = registry
        self.store = store
        self.parser = parser or FunctionCallingParser()
        self.max_rounds = max_rounds
        self.steering = steering_queue
        self.context = context_builder or ContextBuilder(
            summarizer=self._summarize,  # 延迟绑定 self.llm（启动时才创建真实客户端）
        )

    async def _summarize(self, dropped):
        return await summarize_with_llm(self.llm, dropped)

    async def run(self, session_id: str, user_input: str) -> AgentReply:
        session = self.store.load_session(session_id)
        if session is None:
            self.store.create_session(session_id)
            session = {"session_id": session_id, "messages": [], "memory": {}}

        session["messages"].append({"role": "user", "content": user_input})
        self.store.log_trace(session_id, "user_input", {"content": user_input})
        set_current_session(session_id)  # todo 工具需要知道当前 session

        records: List[ToolCallRecord] = []

        for round_no in range(1, self.max_rounds + 1):
            system_prompt = build_system_prompt(self.store, session_id)
            messages, meta = await self.context.build(session, system_prompt)
            self.store.log_trace(session_id, "llm_request", {
                "round": round_no,
                "n_messages": len(messages),
                "compressed": meta["compressed"],
            })

            resp = await self.llm.chat(messages, tools=self.registry.schemas())
            parsed = self.parser.parse(resp)
            self.store.log_trace(session_id, "llm_response", {
                "round": round_no,
                "tool_calls": [tc.name for tc in parsed.tool_calls],
                "thought": (parsed.thought or "")[:200],
            })

            if parsed.is_final:
                content = parsed.thought or "(空回复)"
                session["messages"].append(parsed.raw_message or {
                    "role": "assistant", "content": content,
                })
                self.store.save_session(session_id, session["messages"], session["memory"])
                self.store.log_trace(session_id, "final_reply", {
                    "round": round_no, "content": content[:500],
                })
                return AgentReply(content, round_no, False, records)

            # 分支 a：要调工具 —— assistant(tool_calls) 原样落历史（协议强约束）
            session["messages"].append(parsed.raw_message)
            for tc in parsed.tool_calls:
                record = await self._exec_tool(session_id, tc, round_no)
                records.append(record)
                session["messages"].append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": record.result,
                })

            # steering：本轮工具执行完后检查用户插入消息（简版 pi 转向机制）
            if self.steering is not None:
                for msg in self.steering.pop_all(session_id):
                    session["messages"].append({"role": "user", "content": f"[用户插话] {msg}"})
                    self.store.log_trace(session_id, "steering_injected", {"content": msg})

            # 每轮落盘：中途崩溃不丢历史，续聊安全
            self.store.save_session(session_id, session["messages"], session["memory"])

        return self._circuit_break(session_id, records)

    # ---- 内部方法 ----

    async def _exec_tool(self, session_id: str, tc, round_no: int) -> ToolCallRecord:
        """执行单个工具调用。所有异常转字符串喂回 LLM，绝不抛给用户。"""
        started = time.time()
        args: dict = {}
        result: str
        status: str

        tool = self.registry.get(tc.name)
        if tool is None:
            result = f"Error: 未注册的工具 '{tc.name}'，可用工具: {', '.join(self.registry.names())}"
            status = "unknown_tool"
        else:
            try:
                args = tc.parsed_arguments
            except ValueError as e:
                args = {}
                result = f"Error: {e}"
                status = "invalid_args"
            else:
                kwargs, err = tool.validate(args)
                if err:
                    result = err
                    status = "invalid_args"
                else:
                    try:
                        result = await tool.run(**kwargs)
                        status = "ok"
                    except Exception as e:  # noqa: BLE001 —— 异常喂回 LLM
                        logger.warning("工具 %s 执行异常: %s", tc.name, e)
                        result = f"Error: {type(e).__name__}: {e}"
                        status = "exception"

        duration = int((time.time() - started) * 1000)
        self.store.log_trace(session_id, "tool_call", {
            "round": round_no, "name": tc.name, "args": args,
            "result": result[:500], "status": status, "duration_ms": duration,
        })
        logger.info("[%s] round=%s tool=%s status=%s (%sms)",
                    session_id, round_no, tc.name, status, duration)
        return ToolCallRecord(tc.name, args, result, duration, round_no, status)

    def _circuit_break(self, session_id: str, records: List[ToolCallRecord]) -> AgentReply:
        """达到 max_rounds 仍未给出最终回答：兜底返回已有信息，而不是硬报错。"""
        if records:
            last = records[-1]
            content = (
                f"已达最大工具调用轮次（{self.max_rounds}），暂停执行。"
                f"最后一步调用了 {last.name}，结果：{last.result[:200]}"
            )
        else:
            content = f"已达最大工具调用轮次（{self.max_rounds}），未能生成最终回答。"
        session = self.store.load_session(session_id)
        session["messages"].append({"role": "assistant", "content": content})
        self.store.save_session(session_id, session["messages"], session["memory"])
        self.store.log_trace(session_id, "truncated", {"rounds": self.max_rounds})
        return AgentReply(content, self.max_rounds, True, records)
