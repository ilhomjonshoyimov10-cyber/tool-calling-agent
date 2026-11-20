"""Built-in tools and a factory that wires them together."""

from __future__ import annotations

import httpx

from agent.tools.base import Tool, ToolError, ToolRegistry, ToolResult
from agent.tools.calculator import CalculatorTool
from agent.tools.clock import ClockTool
from agent.tools.currency import CurrencyTool
from agent.tools.tasks import AddTaskTool, CompleteTaskTool, ListTasksTool, TaskStore
from agent.tools.weather import WeatherTool
from agent.tools.wikipedia import WikipediaTool

USER_AGENT = "tool-calling-agent/1.0 (+https://github.com/ilhomjonshoyimov10-cyber/tool-calling-agent)"


def default_registry(client: httpx.AsyncClient, task_store: TaskStore) -> ToolRegistry:
    return ToolRegistry(
        [
            ClockTool(),
            CalculatorTool(),
            CurrencyTool(client),
            WeatherTool(client),
            WikipediaTool(client),
            AddTaskTool(task_store),
            ListTasksTool(task_store),
            CompleteTaskTool(task_store),
        ]
    )


def make_http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=15.0, headers={"User-Agent": USER_AGENT}, follow_redirects=True)


__all__ = [
    "AddTaskTool",
    "CalculatorTool",
    "ClockTool",
    "CompleteTaskTool",
    "CurrencyTool",
    "ListTasksTool",
    "TaskStore",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "ToolResult",
    "WeatherTool",
    "WikipediaTool",
    "default_registry",
    "make_http_client",
]
