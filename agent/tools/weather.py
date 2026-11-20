"""Current weather and short forecast from Open-Meteo (free, no API key)."""

from __future__ import annotations

from typing import Any

import httpx
from pydantic import BaseModel, Field

from agent.tools.base import Tool, ToolError

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# WMO weather interpretation codes (subset used by Open-Meteo)
WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "rime fog",
    51: "light drizzle",
    53: "drizzle",
    55: "dense drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    80: "rain showers",
    81: "heavy rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "thunderstorm with heavy hail",
}


class WeatherArgs(BaseModel):
    city: str = Field(description="City name, e.g. 'Tashkent', 'Samarqand', 'London'")
    days: int = Field(default=3, ge=1, le=7, description="Forecast days including today")


class WeatherTool(Tool):
    name = "get_weather"
    description = "Get current weather and a daily forecast (up to 7 days) for a city."
    Args = WeatherArgs

    def __init__(self, client: httpx.AsyncClient) -> None:
        self._client = client

    async def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        try:
            response = await self._client.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise ToolError(f"Weather service is unavailable: {exc}") from exc

    async def run(self, args: WeatherArgs) -> dict[str, Any]:
        geo = await self._get(GEOCODING_URL, {"name": args.city, "count": 1, "format": "json"})
        if not geo.get("results"):
            raise ToolError(f"City not found: {args.city}")
        place = geo["results"][0]

        data = await self._get(
            FORECAST_URL,
            {
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "daily": "temperature_2m_max,temperature_2m_min,precipitation_probability_max,weather_code",
                "timezone": "auto",
                "forecast_days": args.days,
            },
        )
        current, daily = data["current"], data["daily"]
        return {
            "location": f"{place['name']}, {place.get('country', '')}".strip(", "),
            "current": {
                "time": current["time"],
                "temperature_c": current["temperature_2m"],
                "humidity_percent": current["relative_humidity_2m"],
                "wind_kmh": current["wind_speed_10m"],
                "conditions": WEATHER_CODES.get(current["weather_code"], "unknown"),
            },
            "forecast": [
                {
                    "date": day,
                    "min_c": daily["temperature_2m_min"][i],
                    "max_c": daily["temperature_2m_max"][i],
                    "precipitation_chance_percent": daily["precipitation_probability_max"][i],
                    "conditions": WEATHER_CODES.get(daily["weather_code"][i], "unknown"),
                }
                for i, day in enumerate(daily["time"])
            ],
            "source": "open-meteo.com",
        }
