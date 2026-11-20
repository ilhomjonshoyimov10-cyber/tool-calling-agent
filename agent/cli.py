"""Terminal chat: `python -m agent.cli`. Prints every tool call so you can watch the agent think."""

from __future__ import annotations

import asyncio
import json

from agent.config import Settings
from agent.core import Agent
from agent.llm import OpenAICompatibleModel
from agent.memory import SessionStore
from agent.tools import TaskStore, default_registry, make_http_client

GREY, GREEN, RED, RESET = "\033[90m", "\033[92m", "\033[91m", "\033[0m"


async def main() -> None:
    settings = Settings()
    if not settings.llm_api_key.get_secret_value():
        raise SystemExit("Set LLM_API_KEY in .env first (see .env.example).")

    async with make_http_client() as client:
        model = OpenAICompatibleModel(
            settings.llm_api_key.get_secret_value(), settings.llm_base_url, settings.llm_model
        )
        agent = Agent(model, default_registry(client, TaskStore(settings.tasks_db_path)), max_steps=settings.max_steps)
        memory = SessionStore(settings.history_messages)
        print(f"Agent ready ({settings.llm_model}). Tools: {', '.join(agent.registry.names())}. Ctrl+C to exit.\n")

        while True:
            try:
                question = input("you> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return
            if not question:
                continue
            result = await agent.run(question, history=memory.history("cli"))
            for step in result.steps:
                color = GREEN if step.ok else RED
                args = json.dumps(step.arguments, ensure_ascii=False)
                print(f"{GREY}  -> {color}{step.tool}{GREY}({args}) {step.duration_ms} ms{RESET}")
            print(f"agent> {result.answer}\n")
            memory.append("cli", question, result.answer)


if __name__ == "__main__":
    asyncio.run(main())
