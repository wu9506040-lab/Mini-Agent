"""calculator 工具：基于 ast 白名单的安全表达式求值，禁用 eval。"""
import ast
import operator
from typing import Literal, Union

from pydantic import BaseModel, Field

from .base import BaseTool, register_tool

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.USub: operator.neg, ast.UAdd: operator.pos}

_MAX_ABS = 1e15  # 防 Pow 结果爆炸


def _eval_node(node: ast.AST) -> Union[int, float]:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("指数过大，拒绝计算")
        value = _BIN_OPS[type(node.op)](left, right)
        if abs(value) > _MAX_ABS:
            raise ValueError("计算结果过大")
        return value
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError(f"不支持的表达式元素: {type(node).__name__}")


class CalculatorArgs(BaseModel):
    expression: str = Field(..., description="数学表达式，支持 + - * / // % ** 和括号，例如 (3+5)*12/2")


@register_tool
class CalculatorTool(BaseTool):
    name = "calculator"
    description = "计算数学表达式并返回结果。当用户提出需要精确计算的问题时调用，不要心算。"
    args_model = CalculatorArgs

    async def run(self, expression: str) -> str:
        try:
            tree = ast.parse(expression, mode="eval")
            value = _eval_node(tree)
            return f"{expression} = {value}"
        except SyntaxError:
            return f"Error: 表达式语法错误: {expression}"
        except ZeroDivisionError:
            return "Error: 除数为零"
        except (ValueError, OverflowError, TypeError) as e:
            return f"Error: {e}"
