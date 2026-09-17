"""A custom MCP server exposing weather forecasts as a tool.

Wraps the existing `app/providers/weather.py` fetch client — this module
owns no HTTP logic of its own, just the MCP surface over it. Same shape as
`app/mcp_server/aviationstack.py`, but for Open-Meteo, which needs no API
key at all.

Run standalone:
    python -m app.mcp_server.weather
        stdio transport — what most MCP hosts (Claude Desktop, Claude Code)
        launch a local server with.
    python -m app.mcp_server.weather --transport streamable-http --port 8002
        runs it as its own network server instead.
"""

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from app.providers.weather import WeatherError, fetch_weather_for_place

mcp = MCPServer(
    name="weather",
    instructions="Look up a daily weather forecast for any place name, via Open-Meteo.",
)


@mcp.tool()
def get_weather_forecast(place: str, days: int = 7) -> dict:
    """Get a daily weather forecast for a place.

    Args:
        place: A place name, e.g. "Kyoto" or "Lisbon, Portugal". Resolved to
            coordinates first; the closest matching place is used.
        days: Number of days to forecast, starting today (1-16 — Open-Meteo's
            free forecast horizon caps at 16 days out; longer values are
            clamped rather than rejected).
    """
    try:
        forecast = fetch_weather_for_place(place, days=days)
    except WeatherError as exc:
        # `ToolError`, specifically — see app/mcp_server/aviationstack.py's
        # `_run` helper for why a plain exception isn't enough here: code
        # calling `MCPServer.call_tool()` in-process only gets a result back
        # for a deliberately-raised ToolError/ResourceError, not for a bare
        # exception, which raises as UnexpectedToolError (a server crash)
        # instead. A place that doesn't geocode, or an API error, is an
        # expected, callable-facing failure, not a crash.
        raise ToolError(str(exc)) from exc

    return forecast.model_dump(mode="json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the weather MCP server.")
    parser.add_argument(
        "--transport",
        choices=["stdio", "streamable-http"],
        default="stdio",
    )
    parser.add_argument("--port", type=int, default=8002)
    args = parser.parse_args()

    if args.transport == "streamable-http":
        mcp.run(transport="streamable-http", port=args.port)
    else:
        mcp.run(transport="stdio")
