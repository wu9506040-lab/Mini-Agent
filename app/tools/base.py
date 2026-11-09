"""工具基类与注册机制。

设计：装饰器注册 + Pydantic 参数模型自动推导 Schema。
Pydantic 的 Field(description=...) 一处声明三处复用：
LLM 可见（Schema）、入参校验（model_validate）、文档（model_json_schema）。
新增工具只需：继承 BaseTool + 声明 name/description/args_model + @register_tool。
"""
import json
from typing import Dict, List, Optional, Type

from pydantic import BaseModel, ValidationError


class EmptyArgs(BaseModel):
    """无参数工具的默认参数模型。"""


class BaseTool:
    name: str = ""
    description: str = ""
    args_model: Type[BaseModel] = EmptyArgs

    async def run(self, **kwargs) -> str:
        raise NotImplementedError

    def to_schema(self) -> dict:
        """推导 OpenAI tools 格式的 Schema。"""
        params = self.args_model.model_json_schema()
        params.pop("title", None)
        for prop in params.get("properties", {}).values():
            prop.pop("title", None)
        params.pop("$defs", None)
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": params,
            },
        }

    def validate(self, args: Optional[dict]):
        """入参校验。返回 (kwargs, error)，错误以可读文本返回给 LLM 而非抛出。"""
        try:
            obj = self.args_model.model_validate(args or {})
            return obj.model_dump(), None
        except ValidationError as e:
            brief = "; ".join(
                f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
                for err in e.errors()[:3]
            )
            return None, f"Error: 工具 '{self.name}' 参数不合法: {brief}"


class ToolRegistry:
    """全局工具注册表。LLM 基于 schemas() 输出自主决策调用哪个工具。"""

    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}

    def register(self, tool_cls: Type[BaseTool]) -> Type[BaseTool]:
        instance = tool_cls()
        if not instance.name:
            raise ValueError(f"{tool_cls.__name__} 缺少 name")
        self._tools[instance.name] = instance
        return tool_cls

    def get(self, name: str) -> Optional[BaseTool]:
        return self._tools.get(name)

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def schemas(self) -> List[dict]:
        return [tool.to_schema() for tool in self._tools.values()]


registry = ToolRegistry()


def register_tool(tool_cls: Type[BaseTool]) -> Type[BaseTool]:
    """类装饰器：导入即注册。"""
    return registry.register(tool_cls)


def remove_tool(name: str) -> None:
    """测试辅助：移除临时注册的工具。"""
    registry._tools.pop(name, None)


def _json_default(obj):
    return str(obj)


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, default=_json_default)
