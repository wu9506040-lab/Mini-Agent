"""Memory 层：三层记忆模型 + system prompt 组装。

三层（对应 README memory 章节）：
- 短期记忆 = session.messages 全量历史（SQLite 持久化，供指代消解）
- 结构化记忆 = todos 表（用户事实型状态的可靠存放处）
- 压缩记忆   = context 压缩产生的摘要（见 context.py）

召回时机：每次 LLM 调用前，todos 全量注入 system（规模小，
检索式召回是负优化）；历史经滑动窗口整段带入。
"""
from typing import List, Optional

BASE_SYSTEM_PROMPT = (
    "你是 Mini Agent，一个能调用工具完成任务的智能助手。\n"
    "行为规则：\n"
    "1. 需要计算、查资料、查天气、记待办时，调用对应工具，不要凭空编造结果。\n"
    "2. 工具结果会以 role=tool 消息返回；拿到结果后基于结果回答。\n"
    "3. 如果工具返回 Error，可调整参数重试；多次失败则如实告知用户。\n"
    "4. 不需要工具的闲聊直接回答。\n"
    "5. 用中文回复用户。\n"
)


def render_todos_block(todos: List[dict]) -> Optional[str]:
    """把当前待办渲染为结构化块（fenced block，LLM 定位成本最低）。"""
    if not todos:
        return None
    lines = []
    for t in todos:
        mark = "✓" if t["status"] == "done" else "·"
        lines.append(f"- [{mark}] #{t['id']} {t['content']}（{t['priority']}）")
    return "\n".join(lines)


def build_system_prompt(store, session_id: str) -> str:
    """每次 LLM 调用前组装 system prompt：基础规则 + 当前待办（全量注入）。"""
    parts = [BASE_SYSTEM_PROMPT]
    try:
        todos = store.list_todos(session_id)
    except Exception:
        todos = []
    todos_block = render_todos_block(todos)
    if todos_block:
        parts.append(
            "当前待办清单（持久状态，与 todo 工具联动，回答涉及时必须以此为准确）：\n"
            + todos_block
        )
    return "\n\n".join(parts)
