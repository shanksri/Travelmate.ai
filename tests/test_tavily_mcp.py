"""Tests mock `_call_tool` directly (see conftest.FakeMCPToolResult) — no real
MCP round trip, no API quota spent. `tavily_mcp` reuses `tavily.normalize_search`
unchanged, so normalization itself is already covered by test_tavily.py; these
tests only cover the parts unique to this module: auth, error surfacing, and
that the tool's JSON text block round-trips correctly."""

import pytest
from conftest import FakeMCPToolResult
from test_tavily import SAMPLE_RESPONSE

from app.providers.tavily_mcp import TavilyMCPError, fetch_search_mcp, search_via_mcp


async def test_fetch_search_mcp_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(TavilyMCPError, match="not set"):
        await fetch_search_mcp("Lisbon")


async def test_fetch_search_mcp_returns_the_parsed_payload(monkeypatch):
    async def fake_call_tool(url, name, arguments):
        assert name == "tavily_search"
        assert arguments["query"] == "Lisbon"
        return FakeMCPToolResult(SAMPLE_RESPONSE)

    monkeypatch.setattr("app.providers.tavily_mcp._call_tool", fake_call_tool)

    result = await fetch_search_mcp("Lisbon", api_key="fake-key")

    assert result == SAMPLE_RESPONSE


async def test_fetch_search_mcp_raises_on_a_tool_error(monkeypatch):
    async def fake_call_tool(url, name, arguments):
        return FakeMCPToolResult(is_error=True)

    monkeypatch.setattr("app.providers.tavily_mcp._call_tool", fake_call_tool)

    with pytest.raises(TavilyMCPError, match="returned an error"):
        await fetch_search_mcp("Lisbon", api_key="fake-key")


async def test_fetch_search_mcp_raises_on_no_text_content(monkeypatch):
    async def fake_call_tool(url, name, arguments):
        return FakeMCPToolResult()

    monkeypatch.setattr("app.providers.tavily_mcp._call_tool", fake_call_tool)

    with pytest.raises(TavilyMCPError, match="no text content"):
        await fetch_search_mcp("Lisbon", api_key="fake-key")


async def test_search_via_mcp_normalizes_the_result(monkeypatch):
    async def fake_call_tool(url, name, arguments):
        return FakeMCPToolResult(SAMPLE_RESPONSE)

    monkeypatch.setattr("app.providers.tavily_mcp._call_tool", fake_call_tool)

    normalized = await search_via_mcp("Lisbon", api_key="fake-key")

    assert normalized.query == SAMPLE_RESPONSE["query"]
    assert len(normalized.results) == 1
