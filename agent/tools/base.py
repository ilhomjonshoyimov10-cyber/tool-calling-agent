"""Tool interface and registry.

Each tool declares its arguments as a Pydantic model. The same model is used twice:
1. to generate the JSON Schema the LLM sees, and
2. to validate the arguments the LLM sends back before any code runs.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError


class ToolError(Exception):
    """An expected failure the model should see and can react to (bad input, upstream API down...)."""


class Tool(ABC):
    name: ClassVar[str]
    description: ClassVar[str]
    Args: ClassVar[type[BaseModel]]

    @abstractmethod
    async def run(self, args: Any) -> Any:
        """Execute the tool with validated arguments. Must return JSON-serialisable data."""

    def schema(self) -> dict[str, Any]:
        parameters = _clean_schema(self.Args.model_json_schema())
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": parameters},
        }


def _clean_schema(node: Any) -> Any:
    """Drop Pydantic's cosmetic "title" keys; some providers reject unknown schema keywords."""
    if isinstance(node, dict):
        return {k: _clean_schema(v) for k, v in node.items() if k != "title"}
    if isinstance(node, list):
        return [_clean_schema(item) for item in node]
    return node


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    data: Any

    def to_message_content(self) -> str:
        payload = {"ok": self.ok, "result" if self.ok else "error": self.data}
        return json.dumps(payload, ensure_ascii=False, default=str)


class ToolRegistry:
    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools or []:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Duplicate tool name: {tool.name}")
        self._tools[tool.name] = tool

    def names(self) -> list[str]:
        return list(self._tools)

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.schema() for tool in self._tools.values()]

    async def execute(self, name: str, raw_arguments: str) -> ToolResult:
        """Run a tool. Never raises: every failure becomes a ToolResult the model can read."""
        tool = self._tools.get(name)
        if tool is None:
            return ToolResult(False, f"Unknown tool '{name}'. Available: {', '.join(self._tools)}")
        try:
            parsed = json.loads(raw_arguments or "{}")
            args = tool.Args.model_validate(parsed)
        except (json.JSONDecodeError, ValidationError) as exc:
            return ToolResult(False, f"Invalid arguments for {name}: {exc}")
        try:
            return ToolResult(True, await tool.run(args))
        except ToolError as exc:
            return ToolResult(False, str(exc))
        except Exception as exc:  # unexpected bug inside a tool: report, don't crash the agent
            return ToolResult(False, f"{type(exc).__name__}: {exc}")
