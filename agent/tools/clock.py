"""Current date and time. LLMs do not know "now", so relative dates ("tomorrow") need this tool."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pydantic import BaseModel, Field

from agent.tools.base import Tool, ToolError


class ClockArgs(BaseModel):
    timezone: str = Field(default="Asia/Tashkent", description="IANA time zone, e.g. 'Asia/Tashkent', 'Europe/London'")


class ClockTool(Tool):
    name = "get_current_time"
    description = "Get the current date, time and weekday in a time zone. Use before reasoning about relative dates."
    Args = ClockArgs

    def __init__(self, now: Any = None) -> None:
        self._now = now or (lambda: datetime.now(UTC))

    async def run(self, args: ClockArgs) -> dict[str, Any]:
        try:
            tz = ZoneInfo(args.timezone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ToolError(f"Unknown time zone: {args.timezone}") from exc
        local = self._now().astimezone(tz)
        return {
            "timezone": args.timezone,
            "datetime": local.isoformat(timespec="seconds"),
            "date": local.date().isoformat(),
            "weekday": local.strftime("%A"),
        }
