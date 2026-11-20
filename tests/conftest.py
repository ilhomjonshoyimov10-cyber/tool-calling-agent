from __future__ import annotations

import json
from typing import Any

from agent.llm import ModelTurn, ToolCall


def tool_turn(*calls: tuple[str, dict[str, Any]]) -> ModelTurn:
    """A model turn that requests the given tools."""
    return ModelTurn(
        content=None,
        tool_calls=[
            ToolCall(id=f"call_{i}", name=name, arguments=json.dumps(args)) for i, (name, args) in enumerate(calls)
        ],
        total_tokens=10,
    )


def text_turn(text: str) -> ModelTurn:
    return ModelTurn(content=text, total_tokens=5)


class ScriptedModel:
    """Fake LLM: returns pre-scripted turns and records what it was sent."""

    def __init__(self, turns: list[ModelTurn]) -> None:
        self.turns = list(turns)
        self.requests: list[tuple[list[dict[str, Any]], list[dict[str, Any]]]] = []

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        self.requests.append(([dict(m) for m in messages], tools))
        if not self.turns:
            raise AssertionError("ScriptedModel ran out of turns")
        return self.turns.pop(0)
