"""A persistent to-do list (SQLite) the agent can manage: the "task automation" part of the agent."""

from __future__ import annotations

import asyncio
import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from agent.tools.base import Tool, ToolError

SCHEMA = """
CREATE TABLE IF NOT EXISTS tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    title TEXT NOT NULL,
    due_date TEXT,
    priority TEXT NOT NULL DEFAULT 'medium',
    done INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    completed_at TEXT
)
"""


class TaskStore:
    """Synchronous sqlite3 wrapped with asyncio.to_thread, so the event loop is never blocked."""

    def __init__(self, path: str | Path) -> None:
        self._path = str(path)
        if self._path != ":memory:":
            Path(self._path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._path)
        conn.row_factory = sqlite3.Row
        return conn

    def _add(self, title: str, due_date: date | None, priority: str) -> dict[str, Any]:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO tasks (title, due_date, priority, created_at) VALUES (?, ?, ?, ?)",
                (title, due_date.isoformat() if due_date else None, priority, now),
            )
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (cursor.lastrowid,)).fetchone()
        return dict(row)

    def _list(self, status: str) -> list[dict[str, Any]]:
        where = {"open": "WHERE done = 0", "done": "WHERE done = 1", "all": ""}[status]
        order = (
            "ORDER BY done, due_date IS NULL, due_date, "
            "CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END, id"
        )
        with self._connect() as conn:
            return [dict(row) for row in conn.execute(f"SELECT * FROM tasks {where} {order}")]

    def _complete(self, task_id: int) -> dict[str, Any] | None:
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with self._connect() as conn:
            conn.execute("UPDATE tasks SET done = 1, completed_at = ? WHERE id = ? AND done = 0", (now, task_id))
            row = conn.execute("SELECT * FROM tasks WHERE id = ?", (task_id,)).fetchone()
        return dict(row) if row else None

    async def add(self, title: str, due_date: date | None = None, priority: str = "medium") -> dict[str, Any]:
        return await asyncio.to_thread(self._add, title, due_date, priority)

    async def list(self, status: str = "open") -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._list, status)

    async def complete(self, task_id: int) -> dict[str, Any] | None:
        return await asyncio.to_thread(self._complete, task_id)


class AddTaskArgs(BaseModel):
    title: str = Field(min_length=1, max_length=300, description="What needs to be done")
    due_date: date | None = Field(default=None, description="Deadline YYYY-MM-DD, if any")
    priority: Literal["low", "medium", "high"] = "medium"


class ListTasksArgs(BaseModel):
    status: Literal["open", "done", "all"] = "open"


class CompleteTaskArgs(BaseModel):
    task_id: int = Field(gt=0)


class AddTaskTool(Tool):
    name = "add_task"
    description = "Add a task to the user's persistent to-do list."
    Args = AddTaskArgs

    def __init__(self, store: TaskStore) -> None:
        self._store = store

    async def run(self, args: AddTaskArgs) -> dict[str, Any]:
        return await self._store.add(args.title, args.due_date, args.priority)


class ListTasksTool(Tool):
    name = "list_tasks"
    description = "List the user's tasks (open, done or all), sorted by due date and priority."
    Args = ListTasksArgs

    def __init__(self, store: TaskStore) -> None:
        self._store = store

    async def run(self, args: ListTasksArgs) -> dict[str, Any]:
        tasks = await self._store.list(args.status)
        return {"count": len(tasks), "tasks": tasks}


class CompleteTaskTool(Tool):
    name = "complete_task"
    description = "Mark a task as done by its id (use list_tasks first to find the id)."
    Args = CompleteTaskArgs

    def __init__(self, store: TaskStore) -> None:
        self._store = store

    async def run(self, args: CompleteTaskArgs) -> dict[str, Any]:
        task = await self._store.complete(args.task_id)
        if task is None:
            raise ToolError(f"Task {args.task_id} does not exist")
        return task
