"""LLM 输出解析层（可替换接口）。

题目要求"实现 LLM 输出的解析逻辑，提取思考过程、工具调用或最终答案"。
主方案：FunctionCallingParser（OpenAI-compatible function calling）——
结构化约束下沉到 API 协议层，解析退化为"取字段"。
备选：TextParser（pi / Claude Code 风格的 XML 文本协议）——
模型无关、可观察推理过程，但要对抗格式漂移。二者可热切换。
"""
import json
import re
import uuid
from dataclasses import dataclass, field
from typing import List

from ..llm.client import LLMResponse, ToolCall

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL
)


@dataclass
class ParsedOutput:
    thought: str                 # 思考过程 / 纯文本内容
    tool_calls: List[ToolCall] = field(default_factory=list)
    is_final: bool = True        # True = 无工具调用，thought 即最终答案
    raw_message: dict = field(default_factory=dict)  # 落历史用的 assistant 消息


class BaseParser:
    """解析器接口：换协议只动这一个文件。"""

    def parse(self, response: LLMResponse) -> ParsedOutput:
        raise NotImplementedError


class FunctionCallingParser(BaseParser):
    """主方案：从 function calling 响应中提取字段（协议已保证结构）。"""

    def parse(self, response: LLMResponse) -> ParsedOutput:
        return ParsedOutput(
            thought=response.content or "",
            tool_calls=list(response.tool_calls),
            is_final=not response.tool_calls,
            raw_message=response.raw_message,
        )


class TextParser(BaseParser):
    """备选（pi 式 XML 文本协议）：

    约定 LLM 在文本中用 <tool_call>{"name": ..., "arguments": {...}}</tool_call>
    发起工具调用，标签外的文本视为思考过程。stub 级实现，演示协议可替换。
    已知限制（对比 FC 的真实代价）：依赖模型守格式；JSON 转义/换行可能破坏
    正则匹配；需要额外的重试修复逻辑。
    """

    def parse(self, response: LLMResponse) -> ParsedOutput:
        content = response.content or ""
        matches = _TOOL_CALL_RE.findall(content)
        tool_calls: List[ToolCall] = []
        for raw in matches:
            try:
                obj = json.loads(raw)
                tool_calls.append(ToolCall(
                    id=f"call_{uuid.uuid4().hex[:8]}",
                    name=obj.get("name", ""),
                    arguments=json.dumps(obj.get("arguments", {}), ensure_ascii=False),
                ))
            except json.JSONDecodeError:
                continue  # 格式漂移：跳过坏块（stub 级容错）
        thought = _TOOL_CALL_RE.sub("", content).strip()
        # 合成协议格式消息，让 runtime 下游无感知协议差异
        raw_message = {"role": "assistant", "content": thought}
        if tool_calls:
            raw_message["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in tool_calls
            ]
        return ParsedOutput(
            thought=thought,
            tool_calls=tool_calls,
            is_final=not tool_calls,
            raw_message=raw_message,
        )
