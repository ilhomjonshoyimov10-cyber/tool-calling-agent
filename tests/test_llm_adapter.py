"""The real OpenAI-compatible adapter, tested against a mocked HTTP transport (no network, no key)."""

from __future__ import annotations

import json

import pytest
from openai import AsyncOpenAI

from agent.core import Agent
from agent.llm import ModelError, OpenAICompatibleModel
from agent.tools import CalculatorTool, ToolRegistry

try:  # openai>=3 ships its own fork of httpx; older versions use httpx itself
    import httpx2 as sdk_httpx
except ImportError:  # pragma: no cover
    import httpx as sdk_httpx

BASE_URL = "https://llm.example.test/v1"


def completion(message: dict, tokens: int = 20) -> dict:
    return {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": tokens - 5, "completion_tokens": 5, "total_tokens": tokens},
    }


TOOL_CALL_MESSAGE = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {
            "id": "call_abc",
            "type": "function",
            "function": {"name": "calculator", "arguments": '{"expression": "19 * 21"}'},
            # Providers such as Gemini attach extra data that must be sent back unchanged.
            "extra_content": {"google": {"thought_signature": "sig-123"}},
        }
    ],
}


class FakeProvider:
    """Plays an OpenAI-compatible server: returns queued responses and records request bodies."""

    def __init__(self, *responses: tuple[int, dict]) -> None:
        self.responses = list(responses)
        self.bodies: list[dict] = []

    def handler(self, request):
        assert request.url.path.endswith("/chat/completions")
        self.bodies.append(json.loads(request.content))
        status, body = self.responses.pop(0)
        return sdk_httpx.Response(status, json=body)

    def model(self) -> OpenAICompatibleModel:
        http_client = sdk_httpx.AsyncClient(transport=sdk_httpx.MockTransport(self.handler))
        client = AsyncOpenAI(api_key="test", base_url=BASE_URL, http_client=http_client, max_retries=0)
        return OpenAICompatibleModel(api_key="test", base_url=BASE_URL, model="test-model", client=client)


async def test_full_round_trip_through_the_openai_compatible_api():
    provider = FakeProvider(
        (200, completion(TOOL_CALL_MESSAGE)),
        (200, completion({"role": "assistant", "content": "19 * 21 = 399"})),
    )
    agent = Agent(provider.model(), ToolRegistry([CalculatorTool()]), system_prompt="S")

    result = await agent.run("19*21?")

    assert result.answer == "19 * 21 = 399"
    assert result.total_tokens == 40
    assert provider.bodies[0]["tools"][0]["function"]["name"] == "calculator"

    assistant, tool = provider.bodies[1]["messages"][2], provider.bodies[1]["messages"][3]
    assert assistant["tool_calls"][0]["id"] == "call_abc"
    assert assistant["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "sig-123"}}
    assert tool["role"] == "tool"
    assert tool["tool_call_id"] == "call_abc"
    assert json.loads(tool["content"]) == {"ok": True, "result": {"expression": "19 * 21", "value": 399}}


async def test_provider_errors_become_model_error():
    provider = FakeProvider((401, {"error": {"message": "bad key"}}))
    with pytest.raises(ModelError):
        await provider.model().chat([{"role": "user", "content": "hi"}], [])
