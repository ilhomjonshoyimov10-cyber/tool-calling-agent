"""Chat model abstraction with tool (function) calling.

`ChatModel` is the only thing the agent loop depends on. `OpenAICompatibleModel` adapts any provider
that speaks the OpenAI Chat Completions API; tests use a scripted fake instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from openai import AsyncOpenAI


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string produced by the model


@dataclass(frozen=True)
class ModelTurn:
    content: str | None
    tool_calls: list[ToolCall] = field(default_factory=list)
    # The assistant message exactly as it must be appended to the conversation.
    # Keeping the provider's raw message preserves extra fields (e.g. Gemini thought signatures).
    message: dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0


class ChatModel(Protocol):
    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn: ...


class ModelError(RuntimeError):
    pass


class OpenAICompatibleModel:
    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        timeout: float = 60.0,
        client: AsyncOpenAI | None = None,
    ):
        self._client = client or AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout, max_retries=2)
        self._model = model
        self._temperature = temperature

    async def chat(self, messages: list[dict[str, Any]], tools: list[dict[str, Any]]) -> ModelTurn:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,  # type: ignore[arg-type]
                tools=tools or None,  # type: ignore[arg-type]
                temperature=self._temperature,
            )
        except Exception as exc:
            raise ModelError(str(exc)) from exc
        if not response.choices:
            raise ModelError("Model returned no choices")

        message = response.choices[0].message
        calls = [
            ToolCall(id=call.id, name=call.function.name, arguments=call.function.arguments or "{}")
            for call in (message.tool_calls or [])
            if getattr(call, "function", None) is not None
        ]
        raw = message.model_dump(exclude_none=True)
        raw.setdefault("role", "assistant")
        return ModelTurn(
            content=message.content,
            tool_calls=calls,
            message=raw,
            total_tokens=response.usage.total_tokens if response.usage else 0,
        )
