"""search 工具（mock 实现）：返回确定性结果，预留真实接入位。

笔试允许 mock；接真实搜索只需替换 run() 内部实现，Schema 不变。
"""
from pydantic import BaseModel, Field

from .base import BaseTool, register_tool

_MOCK_KNOWLEDGE = {
    "python": "Python 是一门解释型高级编程语言，官方文档 docs.python.org，3.12 为当前稳定版本。",
    "agent": "Agent（智能体）= LLM + 工具调用 + 循环控制，代表实现有 Claude Code、pi、OpenHands。",
    "rag": "RAG（检索增强生成）：先检索相关文档片段，再拼入 prompt 交给 LLM 生成答案。",
}


class SearchArgs(BaseModel):
    query: str = Field(..., description="搜索关键词或问题")


@register_tool
class SearchTool(BaseTool):
    name = "search"
    description = "联网搜索信息。当需要外部知识、时事或不确定的事实时调用。"
    args_model = SearchArgs

    async def run(self, query: str) -> str:
        # mock：确定性返回，保证测试可复现。真实接入点：调用搜索 API 并返回 top 结果。
        for keyword, fact in _MOCK_KNOWLEDGE.items():
            if keyword in query.lower():
                return f"[mock 搜索] 「{query}」的检索结果：{fact}"
        return (
            f"[mock 搜索] 「{query}」的检索结果：未找到高度相关的资料，"
            f"仅有泛化描述——该主题在 mock 数据源中无记录。"
        )
