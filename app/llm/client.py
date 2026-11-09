"""LLM 客户端层（对标 pi-ai 的定位：纯协议适配，零 agent 逻辑）。

这一层是整个系统唯一知道"怎么调 LLM"的地方，也是测试的 mock 边界：
测试用 FakeLLMClient 替换本层，Runtime 及以上全部逻辑离线可测。

为什么用 openai SDK 不算"依赖现有 agent 框架"：它只做 HTTP 协议适配，
不实现 agent loop / 工具调度 / context 管理——那些在本项目的
runtime.py / parser.py / context.py / store.py 中全部自研。
"""
import json
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class ToolCall:
    """LLM 返回的单个工具调用（OpenAI function calling 格式的归一化）。"""
    id: str
    name: str
    arguments: str  # 原始 JSON 字符串，LLM 输出可能非法，交给 runtime 容错

    @property
    def parsed_arguments(self) -> dict:
        try:
            args = json.loads(self.arguments) if self.arguments else {}
        except (json.JSONDecodeError, TypeError) as e:
            raise ValueError(f"工具参数不是合法 JSON: {e}") from e
        if not isinstance(args, dict):
            raise ValueError("工具参数 JSON 必须是对象（dict）")
        return args


@dataclass
class LLMResponse:
    """归一化的 LLM 响应。

    raw_message 是 OpenAI 协议格式的 assistant 消息，runtime 必须
    把它原样落进历史（tool 消息必须紧跟带对应 tool_call_id 的
    assistant 消息，这是协议强约束，漏掉会 400）。
    """
    content: Optional[str]
    tool_calls: List[ToolCall] = field(default_factory=list)
    raw_message: dict = field(default_factory=dict)


class LLMClient:
    """OpenAI-compatible 客户端（GLM / DeepSeek / Qwen / Kimi 等均可）。"""

    def __init__(self, settings=None):
        from ..config import Settings

        self.settings = settings or Settings.from_env()
        self.settings.validate_llm()

        from openai import AsyncOpenAI

        self._client = AsyncOpenAI(
            base_url=self.settings.base_url,
            api_key=self.settings.api_key,
        )
        self.model = self.settings.model

    async def chat(self, messages: List[dict], tools: Optional[List[dict]] = None) -> LLMResponse:
        kwargs = {"model": self.model, "messages": messages}
        if tools:
            kwargs["tools"] = tools
        resp = await self._client.chat.completions.create(**kwargs)

        msg = resp.choices[0].message
        tool_calls = [
            ToolCall(id=tc.id, name=tc.function.name, arguments=tc.function.arguments)
            for tc in (msg.tool_calls or [])
        ]

        raw_message = {"role": "assistant", "content": msg.content or ""}
        if tool_calls:
            raw_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in tool_calls
            ]
        return LLMResponse(content=msg.content, tool_calls=tool_calls, raw_message=raw_message)
