"""Safe arithmetic. Parses the expression with `ast` and evaluates only a whitelist of nodes.

`eval()` is never used, so input like `__import__('os').system(...)` is rejected.
"""

from __future__ import annotations

import ast
import math
import operator
from typing import Any

from pydantic import BaseModel, Field

from agent.tools.base import Tool, ToolError

_BINARY = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCTIONS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "abs": abs,
    "round": round,
    "min": min,
    "max": max,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}
MAX_EXPONENT = 1000


def safe_eval(expression: str) -> float | int:
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ToolError(f"Not a valid expression: {expression!r}") from exc
    return _eval(tree.body)


def _eval(node: ast.AST) -> Any:
    if isinstance(node, ast.Constant) and isinstance(node.value, int | float) and not isinstance(node.value, bool):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ToolError("Exponent too large")
        try:
            return _BINARY[type(node.op)](left, right)
        except ZeroDivisionError as exc:
            raise ToolError("Division by zero") from exc
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCTIONS
        and not node.keywords
    ):
        try:
            return _FUNCTIONS[node.func.id](*[_eval(arg) for arg in node.args])
        except (ValueError, TypeError) as exc:
            raise ToolError(f"{node.func.id}: {exc}") from exc
    raise ToolError(f"Unsupported element in expression: {ast.dump(node)[:60]}")


class CalculatorArgs(BaseModel):
    expression: str = Field(description="Arithmetic expression, e.g. '(1250 * 12) * 0.88' or 'sqrt(2) * pi'")


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate an arithmetic expression exactly. Use it for any calculation instead of mental math."
    Args = CalculatorArgs

    async def run(self, args: CalculatorArgs) -> dict[str, Any]:
        value = safe_eval(args.expression)
        if isinstance(value, float):
            value = round(value, 10)
        return {"expression": args.expression, "value": value}
