"""The agent loop: think -> call tools -> observe -> answer.

Written from scratch (no LangChain) to keep every step explicit and testable:

1. Send the conversation plus the tool schemas to the model.
2. If the model asks for tools, validate and run them **in parallel**, append the results, repeat.
3. If the model answers in text, stop.
4. A step budget (`max_steps`) prevents infinite loops; when it runs out the model must answer
   with what it already has.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any, Literal

from agent.llm import ChatModel, ToolCall
from agent.tools.base import ToolRegistry, ToolResult

DEFAULT_SYSTEM_PROMPT = """You are a careful assistant that can use tools.

Rules:
- Use a tool whenever the answer depends on live data (exchange rates, weather, current date) or exact math.
- Never invent numbers. If a tool fails, say so and explain what you could not check.
- Mention the data source and date for rates and weather.
- For relative dates ("tomorrow", "next Friday") call get_current_time first.
- Reply in the user's language (Uzbek, Russian or English). Be concise."""

BUDGET_EXHAUSTED = (
    "The tool budget for this request is used up. Answer now using only the information above, "
    "and clearly say which parts you could not verify."
)


@dataclass(frozen=True)
class Step:
    tool: str
    arguments: Any
    ok: bool
    result: Any
    duration_ms: int


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    total_tokens: int = 0
    stopped_reason: Literal["answer", "max_steps"] = "answer"


class Agent:
    def __init__(
        self,
        model: ChatModel,
        registry: ToolRegistry,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_steps: int = 6,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be >= 1")
        self.model = model
        self.registry = registry
        self.system_prompt = system_prompt
        self.max_steps = max_steps

    async def run(self, user_message: str, history: list[dict[str, Any]] | None = None) -> AgentResult:
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.system_prompt}]
        messages += history or []
        messages.append({"role": "user", "content": user_message})

        result = AgentResult(answer="")
        tools = self.registry.schemas()

        for _ in range(self.max_steps):
            turn = await self.model.chat(messages, tools)
            result.total_tokens += turn.total_tokens
            if not turn.tool_calls:
                result.answer = (turn.content or "").strip()
                return result

            messages.append(turn.message or _assistant_message(turn.content, turn.tool_calls))
            outcomes = await asyncio.gather(*(self._execute(call) for call in turn.tool_calls))
            for call, (tool_result, step) in zip(turn.tool_calls, outcomes, strict=True):
                messages.append({"role": "tool", "tool_call_id": call.id, "content": tool_result.to_message_content()})
                result.steps.append(step)

        # Budget exhausted: one last call without tools forces a text answer.
        messages.append({"role": "user", "content": BUDGET_EXHAUSTED})
        turn = await self.model.chat(messages, [])
        result.total_tokens += turn.total_tokens
        result.answer = (turn.content or "").strip()
        result.stopped_reason = "max_steps"
        return result

    async def _execute(self, call: ToolCall) -> tuple[ToolResult, Step]:
        started = time.perf_counter()
        tool_result = await self.registry.execute(call.name, call.arguments)
        elapsed = int((time.perf_counter() - started) * 1000)
        try:
            arguments: Any = json.loads(call.arguments or "{}")
        except json.JSONDecodeError:
            arguments = call.arguments
        return tool_result, Step(call.name, arguments, tool_result.ok, tool_result.data, elapsed)


def _assistant_message(content: str | None, calls: list[ToolCall]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": content,
        "tool_calls": [
            {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}} for c in calls
        ],
    }
