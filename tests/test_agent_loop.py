from __future__ import annotations

import json

import pytest
from pydantic import BaseModel

from agent.core import Agent
from agent.tools import CalculatorTool, ToolRegistry
from agent.tools.base import Tool
from tests.conftest import ScriptedModel, text_turn, tool_turn


class EchoArgs(BaseModel):
    text: str


class EchoTool(Tool):
    name = "echo"
    description = "Echo text back"
    Args = EchoArgs

    async def run(self, args: EchoArgs) -> dict:
        return {"echo": args.text}


class BrokenTool(Tool):
    name = "broken"
    description = "Always crashes"
    Args = EchoArgs

    async def run(self, args: EchoArgs) -> dict:
        raise RuntimeError("boom")


def make_agent(model, max_steps: int = 4) -> Agent:
    return Agent(
        model, ToolRegistry([CalculatorTool(), EchoTool(), BrokenTool()]), system_prompt="SYS", max_steps=max_steps
    )


def tool_messages(model: ScriptedModel, request_index: int) -> list[dict]:
    messages, _ = model.requests[request_index]
    return [json.loads(m["content"]) for m in messages if m["role"] == "tool"]


@pytest.mark.asyncio
async def test_answers_directly_when_no_tool_is_needed():
    model = ScriptedModel([text_turn("Salom!")])
    result = await make_agent(model).run("Salom")

    assert result.answer == "Salom!"
    assert result.steps == []
    assert model.requests[0][0][0] == {"role": "system", "content": "SYS"}
    assert {t["function"]["name"] for t in model.requests[0][1]} == {"calculator", "echo", "broken"}


@pytest.mark.asyncio
async def test_tool_result_is_fed_back_to_the_model():
    model = ScriptedModel([tool_turn(("calculator", {"expression": "1250 * 12"})), text_turn("15000")])
    result = await make_agent(model).run("1250*12?")

    assert result.answer == "15000"
    assert result.steps[0].tool == "calculator"
    assert result.steps[0].ok is True
    assert tool_messages(model, 1) == [{"ok": True, "result": {"expression": "1250 * 12", "value": 15000}}]
    # the assistant message with tool_calls precedes the tool result, as the API requires
    roles = [m["role"] for m in model.requests[1][0]]
    assert roles == ["system", "user", "assistant", "tool"]


@pytest.mark.asyncio
async def test_parallel_tool_calls_keep_their_ids():
    model = ScriptedModel([tool_turn(("echo", {"text": "a"}), ("echo", {"text": "b"})), text_turn("done")])
    result = await make_agent(model).run("two things")

    messages, _ = model.requests[1]
    tool_msgs = [m for m in messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_0", "call_1"]
    assert [s.result["echo"] for s in result.steps] == ["a", "b"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("call", "error_fragment"),
    [
        (("does_not_exist", {}), "Unknown tool"),
        (("calculator", {"wrong": 1}), "Invalid arguments"),
        (("calculator", {"expression": "1/0"}), "Division by zero"),
        (("broken", {"text": "x"}), "RuntimeError: boom"),
    ],
)
async def test_tool_failures_are_reported_to_the_model_not_raised(call, error_fragment):
    model = ScriptedModel([tool_turn(call), text_turn("I could not compute it")])
    result = await make_agent(model).run("q")

    assert result.answer == "I could not compute it"
    assert result.steps[0].ok is False
    [feedback] = tool_messages(model, 1)
    assert feedback["ok"] is False
    assert error_fragment in feedback["error"]


@pytest.mark.asyncio
async def test_step_budget_forces_a_final_answer_without_tools():
    looping = [tool_turn(("echo", {"text": str(i)})) for i in range(2)]
    model = ScriptedModel([*looping, text_turn("partial answer")])
    result = await make_agent(model, max_steps=2).run("loop forever")

    assert result.stopped_reason == "max_steps"
    assert result.answer == "partial answer"
    assert len(result.steps) == 2
    final_messages, final_tools = model.requests[-1]
    assert final_tools == []
    assert "budget" in final_messages[-1]["content"]


@pytest.mark.asyncio
async def test_history_is_included_before_the_new_question():
    model = ScriptedModel([text_turn("ok")])
    history = [{"role": "user", "content": "old q"}, {"role": "assistant", "content": "old a"}]
    await make_agent(model).run("new q", history=history)

    contents = [m["content"] for m in model.requests[0][0]]
    assert contents == ["SYS", "old q", "old a", "new q"]


@pytest.mark.asyncio
async def test_tokens_are_summed_over_all_turns():
    model = ScriptedModel([tool_turn(("echo", {"text": "a"})), text_turn("x")])
    result = await make_agent(model).run("q")
    assert result.total_tokens == 15


def test_max_steps_must_be_positive():
    with pytest.raises(ValueError):
        make_agent(ScriptedModel([]), max_steps=0)
