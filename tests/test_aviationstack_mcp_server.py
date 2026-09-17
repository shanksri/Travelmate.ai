"""Tests mock httpx.get exactly like test_aviationstack.py does — this server
adds no HTTP logic of its own, it only wraps app/providers/aviationstack.py's
fetch_flights/normalize_flights behind an MCP tool. Exercised through
`mcp.call_tool(...)` directly (no transport needed), the same way a real MCP
client would invoke it."""

import json

import pytest
from conftest import FakeHTTPResponse
from mcp.server.mcpserver.exceptions import ToolError
from test_aviationstack import FUTURE_SAMPLE_RESPONSE, SAMPLE_RESPONSE

from app.mcp_server.aviationstack import mcp


def _texts(result) -> list[str]:
    return [block.text for block in result.content if hasattr(block, "text")]


async def test_lists_both_tools():
    tools = await mcp.list_tools()

    names = [t.name for t in tools]
    assert "search_flights" in names
    assert "future_flight_schedule" in names


async def test_search_flights_returns_normalized_flights(monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: FakeHTTPResponse(SAMPLE_RESPONSE))
    monkeypatch.setenv("AVIATION_API_KEY", "fake-key")

    result = await mcp.call_tool("search_flights", {"flight_status": "active", "limit": 2})

    assert result.is_error is False
    flights = [json.loads(t) for t in _texts(result)]
    assert len(flights) == 1
    assert flights[0]["airline"] == "K-Mile Air"
    assert flights[0]["flight_number"] == "8K804"


async def test_search_flights_surfaces_a_missing_api_key_as_a_tool_error(monkeypatch):
    """`MCPServer.call_tool` called in-process (no transport, as here) *raises*
    `ToolError` rather than returning `CallToolResult(is_error=True)` — that
    translation only happens in the JSON-RPC dispatch layer a real transport
    goes through (confirmed live against Tavily's remote server in
    test_tavily_mcp.py, where `result.is_error` is exactly how it surfaces to
    an actual client). `search_flights` deliberately raises `ToolError`, not a
    plain exception, so this failure is reported instead of crashing as an
    `UnexpectedToolError`."""
    monkeypatch.delenv("AVIATION_API_KEY", raising=False)

    with pytest.raises(ToolError, match="not set"):
        await mcp.call_tool("search_flights", {})


async def test_search_flights_on_no_results_returns_an_empty_list(monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: FakeHTTPResponse({"data": []}))
    monkeypatch.setenv("AVIATION_API_KEY", "fake-key")

    result = await mcp.call_tool("search_flights", {"dep_iata": "JFK"})

    assert result.is_error is False
    assert _texts(result) == []


async def test_search_flights_passes_airline_filters_through(monkeypatch):
    captured = {}

    def fake_get(url, params, **kwargs):
        captured.update(params)
        return FakeHTTPResponse(SAMPLE_RESPONSE)

    monkeypatch.setattr("httpx.get", fake_get)
    monkeypatch.setenv("AVIATION_API_KEY", "fake-key")

    await mcp.call_tool(
        "search_flights", {"airline_name": "American Airlines", "airline_iata": "AA"}
    )

    assert captured["airline_name"] == "American Airlines"
    assert captured["airline_iata"] == "AA"


async def test_future_flight_schedule_returns_normalized_flights(monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: FakeHTTPResponse(FUTURE_SAMPLE_RESPONSE))
    monkeypatch.setenv("AVIATION_API_KEY", "fake-key")

    result = await mcp.call_tool(
        "future_flight_schedule",
        {"iata_code": "JFK", "schedule_type": "departure", "flight_date": "2026-10-01"},
    )

    assert result.is_error is False
    flights = [json.loads(t) for t in _texts(result)]
    assert len(flights) == 1
    assert flights[0]["airline"] == "royal air maroc"
    assert flights[0]["flight_number"] == "at5027"


async def test_future_flight_schedule_truncates_to_limit(monkeypatch):
    many = {"data": FUTURE_SAMPLE_RESPONSE["data"] * 5}
    monkeypatch.setattr("httpx.get", lambda *a, **k: FakeHTTPResponse(many))
    monkeypatch.setenv("AVIATION_API_KEY", "fake-key")

    result = await mcp.call_tool(
        "future_flight_schedule",
        {"iata_code": "JFK", "schedule_type": "departure", "flight_date": "2026-10-01", "limit": 2},
    )

    assert len(_texts(result)) == 2


async def test_future_flight_schedule_surfaces_a_plan_restriction_as_a_tool_error(monkeypatch):
    error_payload = {"error": {"code": "function_access_restricted", "message": "Not on plan."}}
    monkeypatch.setattr("httpx.get", lambda *a, **k: FakeHTTPResponse(error_payload))
    monkeypatch.setenv("AVIATION_API_KEY", "fake-key")

    with pytest.raises(ToolError, match="function_access_restricted"):
        await mcp.call_tool(
            "future_flight_schedule",
            {"iata_code": "JFK", "schedule_type": "departure", "flight_date": "2026-10-01"},
        )
