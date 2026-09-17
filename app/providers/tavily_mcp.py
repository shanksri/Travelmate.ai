"""An MCP client for Tavily's hosted remote MCP server.

Same job as `tavily.py` (a thin, fetch-only client for search results,
normalized separately, standalone-runnable) but reached over MCP instead of
Tavily's REST API — `https://mcp.tavily.com/mcp/`, streamable-HTTP transport,
authenticated via `?tavilyApiKey=...` on the URL (no OAuth flow). Its
`tavily_search` tool returns the exact same JSON shape as the REST
`/search` endpoint, so `tavily.normalize_search` applies unchanged — this
module never redefines its own result models.

Not wired into `TravelProvider`; run directly with
`python -m app.providers.tavily_mcp` for a real end-to-end call.

The MCP Python SDK ships its own vendored HTTP client (`httpx2`), a separate
package from `httpx` — mocking `httpx.post` (as `tavily.py`'s tests do) does
NOT intercept this module's network calls. Tests here mock `_call_tool`
instead; `tests/conftest.py`'s network-blocking fixture also patches
`httpx2.get`/`post` as a backstop.
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult

from app.providers.tavily import SearchResults, normalize_search

load_dotenv()

MCP_URL = "https://mcp.tavily.com/mcp/"


class TavilyMCPError(RuntimeError):
    """The MCP server reported a tool error instead of search results."""


async def _call_tool(url: str, name: str, arguments: dict) -> CallToolResult:
    async with streamable_http_client(url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            return await session.call_tool(name, arguments)


async def fetch_search_mcp(
    query: str,
    *,
    max_results: int = 5,
    search_depth: str = "basic",
    api_key: str | None = None,
) -> dict:
    key = api_key or os.environ.get("TAVILY_API_KEY")
    if not key:
        raise TavilyMCPError("TAVILY_API_KEY is not set")

    url = f"{MCP_URL}?tavilyApiKey={key}"
    result = await _call_tool(
        url,
        "tavily_search",
        {"query": query, "max_results": max_results, "search_depth": search_depth},
    )

    if result.is_error:
        raise TavilyMCPError(f"tavily_search returned an error: {result.content}")

    text_blocks = [block.text for block in result.content if hasattr(block, "text")]
    if not text_blocks:
        raise TavilyMCPError(f"tavily_search returned no text content: {result.content}")

    return json.loads(text_blocks[0])


async def search_via_mcp(query: str, **kwargs: object) -> SearchResults:
    """Fetch + normalize in one call, reusing `tavily.normalize_search`."""
    return normalize_search(await fetch_search_mcp(query, **kwargs))


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")

    async def _demo() -> None:
        raw = await fetch_search_mcp("best time of year to visit Lisbon, Portugal")
        print("=== raw ===")
        print(json.dumps(raw, indent=2))
        print("\n=== normalized ===")
        print(normalize_search(raw).model_dump_json(indent=2))

    asyncio.run(_demo())
