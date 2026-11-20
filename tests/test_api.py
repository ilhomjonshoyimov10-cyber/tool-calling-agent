from __future__ import annotations

from fastapi.testclient import TestClient

from agent.api import create_app
from agent.config import Settings
from agent.core import Agent
from agent.llm import ModelError
from agent.tools import CalculatorTool, ToolRegistry
from tests.conftest import ScriptedModel, text_turn, tool_turn


def client_for(model) -> TestClient:
    agent = Agent(model, ToolRegistry([CalculatorTool()]), system_prompt="S")
    return TestClient(create_app(agent=agent, settings=Settings(_env_file=None)))


def test_chat_returns_answer_steps_and_session():
    model = ScriptedModel([tool_turn(("calculator", {"expression": "6*7"})), text_turn("42")])
    with client_for(model) as client:
        response = client.post("/api/chat", json={"message": "6*7?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"] == "42"
    assert body["steps"][0]["tool"] == "calculator"
    assert body["steps"][0]["result"]["value"] == 42
    assert len(body["session_id"]) == 32


def test_session_history_is_sent_on_the_next_turn():
    model = ScriptedModel([text_turn("first"), text_turn("second")])
    with client_for(model) as client:
        session_id = client.post("/api/chat", json={"message": "one"}).json()["session_id"]
        client.post("/api/chat", json={"message": "two", "session_id": session_id})

    contents = [m["content"] for m in model.requests[1][0]]
    assert contents == ["S", "one", "first", "two"]


def test_reset_session_forgets_history():
    model = ScriptedModel([text_turn("a"), text_turn("b")])
    with client_for(model) as client:
        session_id = client.post("/api/chat", json={"message": "one"}).json()["session_id"]
        assert client.delete(f"/api/sessions/{session_id}").status_code == 204
        client.post("/api/chat", json={"message": "two", "session_id": session_id})

    assert [m["content"] for m in model.requests[1][0]] == ["S", "two"]


def test_model_error_becomes_502():
    class Failing:
        async def chat(self, messages, tools):
            raise ModelError("quota exceeded")

    with client_for(Failing()) as client:
        response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 502
    assert "quota exceeded" in response.json()["detail"]


def test_validation_rejects_empty_message():
    with client_for(ScriptedModel([])) as client:
        assert client.post("/api/chat", json={"message": ""}).status_code == 422


def test_tools_health_and_ui():
    with client_for(ScriptedModel([])) as client:
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/api/tools").json()[0]["name"] == "calculator"
        assert "Tool-Calling Agent" in client.get("/").text


def test_without_api_key_chat_explains_what_is_missing(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with TestClient(create_app(settings=Settings(_env_file=None))) as client:
        response = client.post("/api/chat", json={"message": "hi"})
    assert response.status_code == 503
    assert "LLM_API_KEY" in response.json()["detail"]
