"""Official exchange rates of the Central Bank of Uzbekistan (cbu.uz), no API key needed."""

from __future__ import annotations

from datetime import date
from typing import Any

import httpx
from pydantic import BaseModel, Field, field_validator

from agent.tools.base import Tool, ToolError

CBU_URL = "https://cbu.uz/uz/arkhiv-kursov-valyut/json/"


class CurrencyArgs(BaseModel):
    amount: float = Field(default=1, gt=0, description="Amount to convert")
    from_currency: str = Field(description="ISO code, e.g. USD, EUR, RUB, UZS")
    to_currency: str = Field(default="UZS", description="ISO code, e.g. UZS, USD")
    on_date: date | None = Field(default=None, description="Historical date YYYY-MM-DD; omit for the latest rate")

    @field_validator("from_currency", "to_currency")
    @classmethod
    def _upper(cls, value: str) -> str:
        value = value.strip().upper()
        if len(value) != 3 or not value.isalpha():
            raise ValueError("currency must be a 3-letter ISO code")
        return value


class CurrencyTool(Tool):
    name = "convert_currency"
    description = (
        "Convert money between currencies using the official Central Bank of Uzbekistan rates. "
        "Supports UZS and ~75 foreign currencies, latest or a historical date."
    )
    Args = CurrencyArgs

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def _fetch_rates(self, on_date: date | None) -> tuple[dict[str, float], str]:
        url = CBU_URL if on_date is None else f"{CBU_URL}all/{on_date.isoformat()}/"
        try:
            response = await self._client.get(url)
            response.raise_for_status()
            rows = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ToolError(f"Central Bank API is unavailable: {exc}") from exc
        if not rows:
            raise ToolError("Central Bank API returned no rates")
        # CBU gives "Rate" UZS for "Nominal" units of the currency.
        rates = {row["Ccy"]: float(row["Rate"]) / float(row.get("Nominal") or 1) for row in rows}
        rates["UZS"] = 1.0
        return rates, str(rows[0].get("Date", ""))

    async def run(self, args: CurrencyArgs) -> dict[str, Any]:
        rates, rate_date = await self._fetch_rates(args.on_date)
        for code in (args.from_currency, args.to_currency):
            if code not in rates:
                raise ToolError(f"Currency {code} is not published by the Central Bank of Uzbekistan")
        uzs = args.amount * rates[args.from_currency]
        result = uzs / rates[args.to_currency]
        return {
            "amount": args.amount,
            "from": args.from_currency,
            "to": args.to_currency,
            "result": round(result, 4),
            "rate": round(rates[args.from_currency] / rates[args.to_currency], 6),
            "rate_date": rate_date,
            "source": "Central Bank of the Republic of Uzbekistan (cbu.uz)",
        }
