"""HTTP API and web UI.

Run: `uvicorn agent.api:app --reload` then open http://localhost:8000
"""

from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from agent.config import Settings
from agent.core import Agent
from agent.llm import ModelError, OpenAICompatibleModel
from agent.memory import SessionStore
from agent.tools import TaskStore, default_registry, make_http_client

STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    session_id: str | None = Field(default=None, max_length=64)


class StepOut(BaseModel):
    tool: str
    arguments: Any
    ok: bool
    result: Any
    duration_ms: int


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    steps: list[StepOut]
    total_tokens: int
    stopped_reason: str


def create_app(agent: Agent | None = None, settings: Settings | None = None) -> FastAPI:
    """Build the app. Pass an `agent` in tests; otherwise a real one is created from settings."""
    settings = settings or Settings()
    sessions = SessionStore(max_messages=settings.history_messages)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if agent is not None:
            app.state.agent = agent
            yield
            return
        if not settings.llm_api_key.get_secret_value():
            app.state.agent = None  # UI still loads; /api/chat explains what is missing
            yield
            return
        async with make_http_client() as client:
            model = OpenAICompatibleModel(
                api_key=settings.llm_api_key.get_secret_value(),
                base_url=settings.llm_base_url,
                model=settings.llm_model,
                temperature=settings.llm_temperature,
            )
            registry = default_registry(client, TaskStore(settings.tasks_db_path))
            app.state.agent = Agent(model, registry, max_steps=settings.max_steps)
            yield

    app = FastAPI(title="Tool-Calling Agent", version="1.0.0", lifespan=lifespan)

    def get_agent() -> Agent:
        current = app.state.agent
        if current is None:
            raise HTTPException(503, "LLM_API_KEY is not configured on the server")
        return current

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "model": settings.llm_model}

    @app.get("/api/tools")
    async def list_tools() -> list[dict[str, str]]:
        current = app.state.agent
        if current is None:
            return []
        return [
            {"name": s["function"]["name"], "description": s["function"]["description"]}
            for s in current.registry.schemas()
        ]

    @app.post("/api/chat", response_model=ChatResponse)
    async def chat(request: ChatRequest) -> ChatResponse:
        current = get_agent()
        session_id = request.session_id or uuid.uuid4().hex
        try:
            result = await current.run(request.message, history=sessions.history(session_id))
        except ModelError as exc:
            raise HTTPException(502, f"Model provider error: {exc}") from exc
        sessions.append(session_id, request.message, result.answer)
        return ChatResponse(
            session_id=session_id,
            answer=result.answer,
            steps=[StepOut(**asdict(step)) for step in result.steps],
            total_tokens=result.total_tokens,
            stopped_reason=result.stopped_reason,
        )

    @app.delete("/api/sessions/{session_id}", status_code=204)
    async def reset_session(session_id: str) -> None:
        sessions.clear(session_id)

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


app = create_app()
