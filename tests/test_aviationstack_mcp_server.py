"""Tests mock httpx.get exactly like test_aviationstack.py does — this server
adds no HTTP logic of its own, it only wraps app/providers/aviationstack.py's
fetch_flights/normalize_flights behind an MCP tool. Exercised through
`mcp.call_tool(...)` directly (no transport needed), the same way a real MCP
client would invoke it."""

import json

import pytest
from conftest import FakeHTTPResponse
from mcp.server.mcpserver.exceptions import ToolError
from test_aviationstack import SAMPLE_RESPONSE

from app.mcp_server.aviationstack import mcp


def _texts(result) -> list[str]:
    return [block.text for block in result.content if hasattr(block, "text")]


async def test_lists_search_flights():
    tools = await mcp.list_tools()

    names = [t.name for t in tools]
    assert "search_flights" in names


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
