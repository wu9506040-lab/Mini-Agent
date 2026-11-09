"""todo 工具组：待办的增/查/完成，持久化到 SQLite（todos 表，挂 session_id）。

依赖注入：init_todo_store(store) 由应用启动/测试 fixture 调用，
工具实例不直接持有连接，便于多 session 隔离与测试替换。
"""
from typing import Literal, Optional

from pydantic import BaseModel, Field

from .base import BaseTool, register_tool

_todo_store = None


def init_todo_store(store) -> None:
    global _todo_store
    _todo_store = store


def _store():
    if _todo_store is None:
        raise RuntimeError("todo store 未初始化：请先调用 init_todo_store(store)")
    return _todo_store


class TodoAddArgs(BaseModel):
    content: str = Field(..., description="待办事项内容，例如 '下午三点开会'")
    priority: Literal["high", "low"] = Field("low", description="优先级：high=紧急重要，low=普通")


class TodoListArgs(BaseModel):
    include_done: bool = Field(False, description="是否包含已完成的待办")


class TodoDoneArgs(BaseModel):
    todo_id: int = Field(..., description="要标记完成的待办 ID（来自 todo_list 的输出）")


@register_tool
class TodoAddTool(BaseTool):
    name = "todo_add"
    description = "添加一条待办事项。当用户说'帮我记一下/加个待办/别忘了…'时调用。"
    args_model = TodoAddArgs

    async def run(self, content: str, priority: str = "low") -> str:
        todo_id = _store().add_todo(_current_session_id(), content, priority)
        return f"已添加待办 #{todo_id}：{content}（优先级 {priority}）"


@register_tool
class TodoListTool(BaseTool):
    name = "todo_list"
    description = "列出当前会话的全部待办事项。用户问'我的待办/记了什么'时调用。"
    args_model = TodoListArgs

    async def run(self, include_done: bool = False) -> str:
        todos = _store().list_todos(_current_session_id(), include_done=include_done)
        if not todos:
            return "当前没有待办事项。"
        lines = []
        for t in todos:
            mark = "✓" if t["status"] == "done" else " "
            lines.append(f"[{t['id']}][{mark}][{t['priority']}] {t['content']}")
        return "当前待办：\n" + "\n".join(lines)


@register_tool
class TodoDoneTool(BaseTool):
    name = "todo_done"
    description = "把指定 ID 的待办标记为已完成。用户说'完成了/做完了某事'时调用。"
    args_model = TodoDoneArgs

    async def run(self, todo_id: int) -> str:
        ok = _store().complete_todo(_current_session_id(), todo_id)
        if ok:
            return f"待办 #{todo_id} 已标记完成。"
        return f"Error: 未找到待办 #{todo_id}（或它属于其他会话）"


# ---- 运行时 session 上下文（工具需要知道当前挂在哪个 session） ----
_current_sid: Optional[str] = None


def set_current_session(session_id: str) -> None:
    global _current_sid
    _current_sid = session_id


def _current_session_id() -> str:
    if _current_sid is None:
        raise RuntimeError("当前 session 未设置")
    return _current_sid
