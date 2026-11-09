"""工具包：导入子模块即完成注册（@register_tool 装饰器在 import 时生效）。"""
from .base import BaseTool, EmptyArgs, ToolRegistry, dumps, register_tool, registry, remove_tool  # noqa: F401
from .todo import init_todo_store, set_current_session  # noqa: F401

# 导入即注册：新增工具后在此追加一行即可
from . import calculator, search, weather, todo  # noqa: F401,E402
