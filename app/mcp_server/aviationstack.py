"""A custom MCP server exposing AviationStack flight search as a tool.

Wraps the existing `app/providers/aviationstack.py` fetch client — this
module owns no HTTP logic of its own, just the MCP surface over it. Mirrors
`app/providers/tavily_mcp.py` in reverse: that module is a *client* to
someone else's remote MCP server; this module *is* a server, for our own
AviationStack data.

Run standalone:
    python -m app.mcp_server.aviationstack
        stdio transport — what most MCP hosts (Claude Desktop, Claude Code)
        launch a local server with.
    python -m app.mcp_server.aviationstack --transport streamable-http --port 8001
        runs it as its own network server instead, the same shape as Tavily's
        remote server that app/providers/tavily_mcp.py talks to.
"""

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from app.providers.aviationstack import AviationStackError, fetch_flights, normalize_flights

mcp = MCPServer(
    name="aviationstack",
    instructions="Search real-time and scheduled flight data via AviationStack.",
)


@mcp.tool()
def search_flights(
    dep_iata: str | None = None,
    arr_iata: str | None = None,
    flight_status: str | None = None,
    limit: int = 5,
) -> list[dict]:
    """Search real-time flight data from AviationStack.

    Args:
        dep_iata: Departure airport IATA code, e.g. "JFK". Omit for any.
        arr_iata: Arrival airport IATA code, e.g. "LHR". Omit for any.
        flight_status: One of "scheduled", "active", "landed", "cancelled",
            "incident", "diverted". Omit for any status.
        limit: Maximum number of flights to return (1-100).

    At least one filter should be given — an unfiltered call returns
    whatever AviationStack currently has in the air worldwide, which is
    rarely what's useful for a trip-planning query.
    """
    try:
        raw = fetch_flights(
            dep_iata=dep_iata, arr_iata=arr_iata, flight_status=flight_status, limit=limit
        )
    except AviationStackError as exc:
        # `ToolError`, specifically — not a plain exception. Over a real
        # transport the SDK reports both a deliberate `ToolError` and any
        # other exception back to the client as `CallToolResult(is_error=True)`
        # (confirmed live), but code calling `MCPServer.call_tool()` directly
        # in-process — this project's own tests, or another module composing
        # this server — only gets a result back for `ToolError`/`ResourceError`;
        # anything else *raises* as `UnexpectedToolError`, logged as a server
        # crash. AviationStackError (bad/missing key, API error payload) is an
        # expected, callable-facing failure, not a crash, so it's ToolError.
        raise ToolError(str(exc)) from exc

    return [flight.model_dump(mode="json") for flight in normalize_flights(raw)]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the AviationStack MCP server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
    )
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if args.transport == "streamable-http":
        mcp.run(transport="streamable-http", port=args.port)
    else:
        mcp.run(transport="stdio")
