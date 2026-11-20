from __future__ import annotations

from datetime import UTC, datetime

import httpx
import pytest
import respx

from agent.tools import (
    AddTaskTool,
    CalculatorTool,
    ClockTool,
    CompleteTaskTool,
    CurrencyTool,
    ListTasksTool,
    TaskStore,
    ToolRegistry,
    WeatherTool,
    WikipediaTool,
)
from agent.tools.calculator import safe_eval
from agent.tools.currency import CBU_URL
from agent.tools.weather import FORECAST_URL, GEOCODING_URL

CBU_SAMPLE = [
    {
        "id": 69,
        "Code": "840",
        "Ccy": "USD",
        "CcyNm_EN": "US Dollar",
        "Nominal": "1",
        "Rate": "12000.00",
        "Date": "20.11.2026",
    },
    {
        "id": 21,
        "Code": "978",
        "Ccy": "EUR",
        "CcyNm_EN": "Euro",
        "Nominal": "1",
        "Rate": "13000.00",
        "Date": "20.11.2026",
    },
    {
        "id": 31,
        "Code": "392",
        "Ccy": "JPY",
        "CcyNm_EN": "Japan Yen",
        "Nominal": "10",
        "Rate": "800.00",
        "Date": "20.11.2026",
    },
]


@pytest.fixture
async def client():
    async with httpx.AsyncClient() as c:
        yield c


# ---------- calculator ----------
@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("2 + 3 * 4", 14),
        ("(1250 * 12) * 0.88", 13200.0),
        ("2 ** 10", 1024),
        ("-5 + abs(-3)", -2),
        ("round(sqrt(2), 3)", 1.414),
    ],
)
def test_calculator_evaluates_math(expression, expected):
    assert safe_eval(expression) == pytest.approx(expected)


@pytest.mark.parametrize(
    "expression",
    ["__import__('os').system('ls')", "open('/etc/passwd')", "[x for x in range(9)]", "2 ** 100000", "a + 1", "1 +"],
)
@pytest.mark.asyncio
async def test_calculator_rejects_unsafe_or_invalid_input(expression):
    result = await ToolRegistry([CalculatorTool()]).execute("calculator", f'{{"expression": "{expression}"}}')
    assert result.ok is False


# ---------- currency (Central Bank of Uzbekistan) ----------
@respx.mock
@pytest.mark.asyncio
async def test_currency_converts_through_uzs(client):
    respx.get(CBU_URL).mock(return_value=httpx.Response(200, json=CBU_SAMPLE))
    result = await CurrencyTool(client).run(CurrencyTool.Args(amount=100, from_currency="usd", to_currency="EUR"))

    assert result["result"] == pytest.approx(100 * 12000 / 13000, rel=1e-4)
    assert result["rate_date"] == "20.11.2026"


@respx.mock
@pytest.mark.asyncio
async def test_currency_respects_nominal(client):
    respx.get(CBU_URL).mock(return_value=httpx.Response(200, json=CBU_SAMPLE))
    result = await CurrencyTool(client).run(CurrencyTool.Args(amount=1000, from_currency="JPY"))
    assert result["result"] == pytest.approx(1000 * 80.0)  # 800 UZS per 10 yen


@respx.mock
@pytest.mark.asyncio
async def test_currency_historical_date_uses_archive_url(client):
    route = respx.get(f"{CBU_URL}all/2025-01-15/").mock(return_value=httpx.Response(200, json=CBU_SAMPLE))
    await CurrencyTool(client).run(CurrencyTool.Args(from_currency="USD", on_date="2025-01-15"))
    assert route.called


@respx.mock
@pytest.mark.asyncio
async def test_currency_errors_are_tool_errors(client):
    respx.get(CBU_URL).mock(return_value=httpx.Response(503))
    registry = ToolRegistry([CurrencyTool(client)])
    result = await registry.execute("convert_currency", '{"from_currency": "USD"}')
    assert result.ok is False
    assert "unavailable" in result.data


@respx.mock
@pytest.mark.asyncio
async def test_currency_unknown_code(client):
    respx.get(CBU_URL).mock(return_value=httpx.Response(200, json=CBU_SAMPLE))
    result = await ToolRegistry([CurrencyTool(client)]).execute("convert_currency", '{"from_currency": "XYZ"}')
    assert "not published" in result.data


# ---------- weather ----------
@respx.mock
@pytest.mark.asyncio
async def test_weather_geocodes_then_forecasts(client):
    respx.get(GEOCODING_URL).mock(
        return_value=httpx.Response(
            200,
            json={"results": [{"name": "Samarkand", "country": "Uzbekistan", "latitude": 39.65, "longitude": 66.96}]},
        )
    )
    forecast = respx.get(FORECAST_URL).mock(
        return_value=httpx.Response(
            200,
            json={
                "current": {
                    "time": "2026-11-20T10:00",
                    "temperature_2m": 9.5,
                    "relative_humidity_2m": 60,
                    "wind_speed_10m": 7.2,
                    "weather_code": 2,
                },
                "daily": {
                    "time": ["2026-11-20", "2026-11-21"],
                    "temperature_2m_min": [3.1, 2.0],
                    "temperature_2m_max": [12.4, 10.0],
                    "precipitation_probability_max": [10, 70],
                    "weather_code": [2, 61],
                },
            },
        )
    )
    result = await WeatherTool(client).run(WeatherTool.Args(city="Samarqand", days=2))

    assert forecast.calls.last.request.url.params["latitude"] == "39.65"
    assert result["location"] == "Samarkand, Uzbekistan"
    assert result["current"]["conditions"] == "partly cloudy"
    assert result["forecast"][1] == {
        "date": "2026-11-21",
        "min_c": 2.0,
        "max_c": 10.0,
        "precipitation_chance_percent": 70,
        "conditions": "light rain",
    }


@respx.mock
@pytest.mark.asyncio
async def test_weather_unknown_city(client):
    respx.get(GEOCODING_URL).mock(return_value=httpx.Response(200, json={}))
    result = await ToolRegistry([WeatherTool(client)]).execute("get_weather", '{"city": "Nowhereville"}')
    assert result.ok is False and "not found" in result.data


# ---------- wikipedia ----------
@respx.mock
@pytest.mark.asyncio
async def test_wikipedia_search_then_summary(client):
    respx.get("https://uz.wikipedia.org/w/api.php").mock(
        return_value=httpx.Response(200, json={"query": {"search": [{"title": "Amir Temur"}, {"title": "Temuriylar"}]}})
    )
    respx.get("https://uz.wikipedia.org/api/rest_v1/page/summary/Amir_Temur").mock(
        return_value=httpx.Response(
            200,
            json={
                "title": "Amir Temur",
                "extract": "Amir Temur (1336-1405) ...",
                "content_urls": {"desktop": {"page": "https://uz.wikipedia.org/wiki/Amir_Temur"}},
            },
        )
    )
    result = await WikipediaTool(client).run(WikipediaTool.Args(query="Amir Temur", lang="uz"))

    assert result["title"] == "Amir Temur"
    assert result["url"].endswith("/wiki/Amir_Temur")
    assert result["other_matches"] == ["Temuriylar"]


# ---------- tasks ----------
@pytest.mark.asyncio
async def test_task_lifecycle(tmp_path):
    store = TaskStore(tmp_path / "tasks.db")
    registry = ToolRegistry([AddTaskTool(store), ListTasksTool(store), CompleteTaskTool(store)])

    low = await registry.execute("add_task", '{"title": "Read a paper", "priority": "low"}')
    high = await registry.execute(
        "add_task", '{"title": "Submit report", "due_date": "2026-11-27", "priority": "high"}'
    )
    assert low.ok and high.ok

    listed = await registry.execute("list_tasks", "{}")
    assert [t["title"] for t in listed.data["tasks"]] == ["Submit report", "Read a paper"]

    done = await registry.execute("complete_task", f'{{"task_id": {high.data["id"]}}}')
    assert done.data["done"] == 1
    assert (await registry.execute("list_tasks", '{"status": "open"}')).data["count"] == 1
    assert (await registry.execute("complete_task", '{"task_id": 999}')).ok is False


@pytest.mark.asyncio
async def test_tasks_persist_across_store_instances(tmp_path):
    path = tmp_path / "tasks.db"
    await TaskStore(path).add("persisted")
    assert (await TaskStore(path).list())[0]["title"] == "persisted"


# ---------- clock ----------
@pytest.mark.asyncio
async def test_clock_converts_to_requested_timezone():
    fixed = ClockTool(now=lambda: datetime(2026, 11, 20, 20, 30, tzinfo=UTC))
    result = await fixed.run(ClockTool.Args(timezone="Asia/Tashkent"))
    assert result["date"] == "2026-11-21"
    assert result["weekday"] == "Saturday"


def test_schemas_have_no_titles_and_declare_required_fields():
    schema = CurrencyTool(httpx.AsyncClient()).schema()["function"]["parameters"]
    assert "title" not in schema
    assert schema["required"] == ["from_currency"]
