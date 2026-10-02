"""One search on SerpApi's hosted MCP server, with its failures classified.

Shared by Google Flights (app/providers/google_flights.py) and Google Hotels
(app/providers/google_hotels.py): same server, same `search` tool, a
different `engine` and API key. Talks MCP to `https://mcp.serpapi.com/mcp`
over the shared session (app/providers/mcp_runtime.py). The key goes in an
`Authorization: Bearer` header rather than the URL path SerpApi also accepts,
because the MCP SDK logs every request URL.

Failures are classified for the retry and fallback chain
(app/providers/resilience.py):

- transient: timeouts and dropped connections. Worth retrying.
- account problem: an invalid key or an exhausted monthly quota. Not
  retried, and it opens the circuit breaker at once: every search would
  fail the same way until the key or the month changes.
- anything else, e.g. an error the server returned: not retried, but counts
  towards opening the breaker.

"No results" is not a failure: Google found nothing for that query, and the
payload comes back with empty result lists.
"""

import json
from collections.abc import Iterator

import httpx2
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult

from app.providers.mcp_runtime import get_mcp_runtime

MCP_URL = "https://mcp.serpapi.com/mcp"

_NO_RESULTS = ("hasn't returned any results", "returned no results")
# Real wording, live: "Error: Invalid SerpApi API key. Check the key in the
# request path or Authorization header…" — so "invalid api key" alone missed it.
_ACCOUNT_PROBLEMS = (
    "run out of searches",
    "invalid serpapi api key",
    "invalid api key",
    "monthly search",
)


class SerpApiError(RuntimeError):
    def __init__(
        self, message: str, *, transient: bool = False, account_problem: bool = False
    ) -> None:
        super().__init__(message)
        self.transient = transient
        self.account_problem = account_problem


async def _call_search(params: dict, api_key: str) -> CallToolResult:
    """One search on the shared, long-lived session."""
    return await get_mcp_runtime().call_tool(
        MCP_URL,
        {"Authorization": f"Bearer {api_key}"},
        "search",
        {"params": params, "mode": "compact"},
    )


def _leaves(exc: BaseException) -> Iterator[BaseException]:
    """The SDK's task groups wrap failures in (nested) ExceptionGroups."""
    if isinstance(exc, BaseExceptionGroup):
        for inner in exc.exceptions:
            yield from _leaves(inner)
    else:
        yield exc


def _is_account_problem(message: str) -> bool:
    lowered = message.lower()
    return any(marker in lowered for marker in _ACCOUNT_PROBLEMS)


def search(params: dict, api_key: str) -> dict:
    """The parsed JSON of one SerpApi search. Raises SerpApiError."""
    try:
        result = get_mcp_runtime().run(_call_search(params, api_key))
    except Exception as exc:
        leaves = list(_leaves(exc))
        if any(isinstance(leaf, MCPError) for leaf in leaves):
            raise SerpApiError(f"SerpApi refused the search: {exc}") from exc
        transient = any(
            isinstance(leaf, TimeoutError | ConnectionError | OSError | httpx2.TransportError)
            for leaf in leaves
        )
        raise SerpApiError(f"SerpApi MCP call failed: {exc!r}", transient=transient) from exc

    text = next((block.text for block in result.content if hasattr(block, "text")), "")
    if result.is_error:
        message = text or "SerpApi search failed"
        raise SerpApiError(message, account_problem=_is_account_problem(message))

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SerpApiError(f"SerpApi returned non-JSON output: {text[:200]}") from exc
    # Some responses wrap the JSON as a string under "result".
    if isinstance(payload.get("result"), str):
        payload = json.loads(payload["result"])

    error = payload.get("error")
    if error:
        message = str(error)
        if any(marker in message.lower() for marker in _NO_RESULTS):
            return {k: v for k, v in payload.items() if k != "error"}
        raise SerpApiError(message, account_problem=_is_account_problem(message))
    return payload
