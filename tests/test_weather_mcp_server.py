"""Tests mock httpx.get exactly like test_weather.py does — this server adds
no HTTP logic of its own, it only wraps app/providers/weather.py's
fetch_weather_for_place behind an MCP tool. Exercised through
`mcp.call_tool(...)` directly (no transport needed), the same way a real MCP
client would invoke it."""

import json

import pytest
from conftest import FakeHTTPResponse
from mcp.server.mcpserver.exceptions import ToolError
from test_weather import FORECAST_SAMPLE, GEOCODE_SAMPLE

from app.mcp_server.weather import mcp


def _texts(result) -> list[str]:
    return [block.text for block in result.content if hasattr(block, "text")]


async def test_lists_get_weather_forecast():
    tools = await mcp.list_tools()

    names = [t.name for t in tools]
    assert "get_weather_forecast" in names


async def test_get_weather_forecast_returns_the_normalized_forecast(monkeypatch):
    responses = iter([FakeHTTPResponse(GEOCODE_SAMPLE), FakeHTTPResponse(FORECAST_SAMPLE)])
    monkeypatch.setattr("httpx.get", lambda *a, **k: next(responses))

    result = await mcp.call_tool("get_weather_forecast", {"place": "Kyoto", "days": 2})

    assert result.is_error is False
    (forecast,) = [json.loads(t) for t in _texts(result)]
    assert forecast["location"] == "Kyoto"
    assert len(forecast["days"]) == 2
    assert forecast["days"][0]["conditions"] == "light drizzle"


async def test_get_weather_forecast_surfaces_a_geocode_miss_as_a_tool_error(monkeypatch):
    """See test_aviationstack_mcp_server.py's equivalent test for why this
    raises ToolError rather than returning `CallToolResult(is_error=True)`
    when called in-process, as here."""
    monkeypatch.setattr("httpx.get", lambda *a, **k: FakeHTTPResponse({}))

    with pytest.raises(ToolError, match="No location found"):
        await mcp.call_tool("get_weather_forecast", {"place": "zzxxqqnowhereplace123"})
