"""Encyclopedic lookup through the public Wikipedia APIs (uz, en, ru)."""

from __future__ import annotations

from typing import Any, Literal
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field

from agent.tools.base import Tool, ToolError


class WikipediaArgs(BaseModel):
    query: str = Field(description="What to look up, e.g. 'Amir Temur' or 'Transformer (machine learning)'")
    lang: Literal["uz", "en", "ru"] = Field(default="en", description="Wikipedia language edition")


class WikipediaTool(Tool):
    name = "search_wikipedia"
    description = "Look up facts on Wikipedia. Returns the best-matching article summary with its URL."
    Args = WikipediaArgs

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def run(self, args: WikipediaArgs) -> dict[str, Any]:
        base = f"https://{args.lang}.wikipedia.org"
        try:
            search = await self._client.get(
                f"{base}/w/api.php",
                params={
                    "action": "query",
                    "list": "search",
                    "srsearch": args.query,
                    "srlimit": 3,
                    "format": "json",
                },
            )
            search.raise_for_status()
            hits = search.json().get("query", {}).get("search", [])
            if not hits:
                raise ToolError(f"No Wikipedia ({args.lang}) article found for: {args.query}")

            title = hits[0]["title"]
            summary = await self._client.get(f"{base}/api/rest_v1/page/summary/{quote(title.replace(' ', '_'))}")
            summary.raise_for_status()
            page = summary.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ToolError(f"Wikipedia is unavailable: {exc}") from exc

        return {
            "title": page.get("title", title),
            "summary": page.get("extract", ""),
            "url": page.get("content_urls", {}).get("desktop", {}).get("page", f"{base}/wiki/{quote(title)}"),
            "other_matches": [hit["title"] for hit in hits[1:]],
        }
