"""Context 管理：历史组装、最大轮次护栏、超长基础压缩（pi 式 compaction）。

核心设计：
- messages 存"全量原始历史"（SQLite），context 是每次渲染出来的视图。
- 压缩策略：保头尾、丢中间。system+摘要 保留，最近 RECENT_KEEP 条保留，
  中间更早的轮次丢弃，丢弃区间调一次 LLM 生成摘要追加到 system 末尾
  （摘要结果缓存在 session.memory，避免每轮重复调用）。
- 只在 user 消息处切割，保证 assistant(tool_calls) 与 tool 消息配对
  不被切断（OpenAI 协议强约束）。
"""
import json
from typing import Awaitable, Callable, List, Optional, Tuple

MAX_SUMMARY_CHARS = 600


async def summarize_with_llm(llm, dropped: List[dict]) -> str:
    """用一次 LLM 调用把被丢弃的历史区间压缩成摘要。

    通过 llm 参数在调用时传入（而非构造时绑定），支持 Runtime 的
    LLM 客户端延迟初始化（测试注入 FakeLLM / 生产启动时创建）。
    """
    transcript = json.dumps(dropped, ensure_ascii=False)
    try:
        resp = await llm.chat(
            [
                {"role": "system", "content": (
                    "你是对话压缩器。把给定的对话历史压缩为一段不超过 600 字的摘要，"
                    "必须保留：用户的目标与偏好、已执行的工具调用及关键结果、"
                    "得出的结论。直接输出摘要正文，不要评论。"
                )},
                {"role": "user", "content": transcript},
            ],
            tools=None,  # 压缩是纯文本任务，不提供工具
        )
        text = (resp.content or "").strip()
        return text[:MAX_SUMMARY_CHARS] if text else ""
    except Exception:
        return ""  # 摘要失败不阻塞主流程，退化为无摘要


def make_summarizer(llm) -> Callable[[List[dict]], Awaitable[str]]:
    """按固定 llm 实例生成摘要函数（延迟绑定见 Runtime._summarize）。"""

    async def summarize(dropped: List[dict]) -> str:
        return await summarize_with_llm(llm, dropped)

    return summarize


def _rule_summary(dropped: List[dict]) -> str:
    """无 LLM 时的规则级摘要兜底：抽取 assistant 最终回答的前 50 字拼接。"""
    parts = []
    for m in dropped:
        if m.get("role") == "assistant" and m.get("content") and not m.get("tool_calls"):
            parts.append(m["content"].strip()[:50])
    return "；".join(parts[:5])


class ContextBuilder:
    def __init__(self, summarizer: Optional[Callable] = None,
                 max_history: int = 40, recent_keep: int = 20):
        self.summarizer = summarizer
        self.max_history = max_history
        self.recent_keep = recent_keep

    async def build(self, session: dict, system_prompt: str) -> Tuple[List[dict], dict]:
        """返回 (messages, meta)。会话内压缩摘要缓存写回 session["memory"]。"""
        messages: List[dict] = list(session["messages"])
        memory = session.setdefault("memory", {})
        meta = {"compressed": False, "summary_used": False}

        system = system_prompt
        if len(messages) > self.max_history:
            cut = self._safe_cut(messages)
            if cut > 0:
                dropped = messages[:cut]
                summary = memory.get("summary")
                if not summary or memory.get("summary_upto") != cut:
                    if self.summarizer is not None:
                        summary = await self.summarizer(dropped)
                    else:
                        summary = _rule_summary(dropped)
                    memory["summary"] = summary
                    memory["summary_upto"] = cut
                if summary:
                    system = f"{system}\n\n[此前对话摘要] {summary}"
                    meta["summary_used"] = True
                messages = messages[cut:]
                meta["compressed"] = True

        return [{"role": "system", "content": system}] + messages, meta

    def _safe_cut(self, messages: List[dict]) -> int:
        """在 user 消息处切割，避免拆散 assistant(tool_calls)↔tool 配对。"""
        limit = len(messages) - self.recent_keep
        cut = 0
        for i in range(min(limit, len(messages))):
            if messages[i].get("role") == "user":
                cut = i
        return cut
