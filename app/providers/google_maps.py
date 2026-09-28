"""Places and routes from Google Maps, via Google's hosted MCP server.

Talks MCP to Maps Grounding Lite at `https://mapstools.googleapis.com/mcp`
(streamable HTTP), with the key in an `X-Goog-Api-Key` header. Two of its
tools are used:

- `search_places`: "best restaurants in Bhubaneswar" comes back as a written
  summary citing places as [0], [1]…, plus a Google Maps link per place.
  There are no names, ratings or prices as fields — those live in the summary.
- `compute_routes`: driving or walking distance and time between two places.
  No trains, buses or flights, and no turn-by-turn directions.

Google's terms for this API: results must not be stored or cached, and each
result's attribution must be shown with it. So nothing here is cached (unlike
flight searches) and the page shows the attribution.

Free usage is 10,000 requests a month; each tool call is one.

Calls go over one long-lived session (app/providers/mcp_runtime.py) rather
than a new connection each, and `search_places_many` runs several searches at
once over it — a trip's per-city lookups.
"""

import asyncio
import json
import logging
import os
from typing import Any
from urllib.parse import urlencode

from dotenv import load_dotenv
from mcp.shared.exceptions import MCPError
from mcp.types import CallToolResult

from app.models.maps import Attribution, PlaceLink, PlacesAnswer, RouteAnswer, TravelMode
from app.providers.mcp_runtime import get_mcp_runtime

load_dotenv()

logger = logging.getLogger(__name__)

MCP_URL = "https://mapstools.googleapis.com/mcp"

# The MCP SDK turns Google's HTTP errors into a bare JSON-RPC -32603 and drops
# the body, which is where Google says what's wrong. Seen live: the key was
# valid but the API wasn't enabled on its project, and all that surfaced was
# "Server returned an error response".
_REFUSED_HINT = (
    "Google Maps refused the request. Check that the Maps Grounding Lite API is "
    "enabled on the Google Cloud project this key belongs to, that billing is set "
    "up there, and that the key isn't restricted to other APIs."
)


class GoogleMapsError(RuntimeError):
    """The Maps call failed, or returned an error instead of a result."""


def _api_key(api_key: str | None = None) -> str:
    key = api_key or os.environ.get("GOOGLE_MAPS_API_KEY")
    if not key:
        raise GoogleMapsError("GOOGLE_MAPS_API_KEY is not set")
    return key.strip()


async def _call_tool(tool: str, arguments: dict, api_key: str) -> CallToolResult:
    """One tool call on the shared, long-lived session (app/providers/mcp_runtime.py)."""
    return await get_mcp_runtime().call_tool(
        MCP_URL, {"X-Goog-Api-Key": api_key}, tool, arguments
    )


def _contains_mcp_error(exc: BaseException) -> bool:
    """The SDK's task groups wrap a failed call in (nested) ExceptionGroups."""
    if isinstance(exc, MCPError):
        return True
    if isinstance(exc, BaseExceptionGroup):
        return any(_contains_mcp_error(inner) for inner in exc.exceptions)
    return False


async def _arun(tool: str, arguments: dict, api_key: str) -> dict[str, Any]:
    try:
        result = await _call_tool(tool, arguments, api_key)
    except Exception as exc:  # network/transport failures surface uniformly
        if _contains_mcp_error(exc):
            raise GoogleMapsError(_REFUSED_HINT) from exc
        raise GoogleMapsError(f"Google Maps MCP call failed: {exc}") from exc

    text = next((block.text for block in result.content if hasattr(block, "text")), "")
    if result.is_error:
        raise GoogleMapsError(text or f"Google Maps {tool} failed")
    if result.structured_content:
        return result.structured_content
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        raise GoogleMapsError(f"Google Maps returned non-JSON output: {text[:200]}") from exc


def _attribution(raw: dict | None) -> Attribution | None:
    if not raw or not raw.get("title"):
        return None
    return Attribution(title=raw["title"], url=raw.get("url"))


def _run(tool: str, arguments: dict, api_key: str | None) -> dict[str, Any]:
    return get_mcp_runtime().run(_arun(tool, arguments, _api_key(api_key)))


def _places_answer(query: str, raw: dict[str, Any]) -> PlacesAnswer:
    places = []
    for index, place in enumerate(raw.get("places") or []):
        links = place.get("googleMapsLinks") or {}
        places.append(
            PlaceLink(
                index=index,
                place_url=links.get("placeUrl"),
                directions_url=links.get("directionsUrl"),
                reviews_url=links.get("reviewsUrl"),
                attribution=_attribution(place.get("attribution")),
            )
        )
    summary = (raw.get("summary") or "").strip()
    if not summary and not places:
        raise GoogleMapsError(f"Google Maps found nothing for {query!r}")
    return PlacesAnswer(query=query, summary=summary, places=places)


def search_places(query: str, *, api_key: str | None = None) -> PlacesAnswer:
    """Places matching `query`, e.g. "best restaurants in Bhubaneswar"."""
    return _places_answer(query, _run("search_places", {"textQuery": query}, api_key))


def search_places_many(
    queries: list[str], *, api_key: str | None = None
) -> list[PlacesAnswer | GoogleMapsError]:
    """Every query at once, over the one shared session. Each result is the
    answer or that query's error, in the order given, so one failure doesn't
    lose the others."""
    key = _api_key(api_key)

    async def one(query: str) -> PlacesAnswer | GoogleMapsError:
        try:
            return _places_answer(query, await _arun("search_places", {"textQuery": query}, key))
        except GoogleMapsError as exc:
            return exc

    async def every() -> list[PlacesAnswer | GoogleMapsError]:
        return list(await asyncio.gather(*(one(query) for query in queries)))

    return get_mcp_runtime().run(every())


def _seconds(duration: str | None) -> int | None:
    """Google's durations are strings like "12345s"."""
    if not duration:
        return None
    try:
        return round(float(duration.removesuffix("s")))
    except ValueError:
        return None


def directions_url(origin: str, destination: str, travel_mode: TravelMode) -> str:
    """A Google Maps directions link — built, not returned by the API."""
    mode = "walking" if travel_mode == "WALK" else "driving"
    query = urlencode(
        {"api": "1", "origin": origin, "destination": destination, "travelmode": mode}
    )
    return f"https://www.google.com/maps/dir/?{query}"


def compute_route(
    origin: str,
    destination: str,
    travel_mode: TravelMode = "DRIVE",
    *,
    api_key: str | None = None,
) -> RouteAnswer:
    """Distance and time from `origin` to `destination`, by car or on foot."""
    raw = _run(
        "compute_routes",
        {
            "origin": {"address": origin},
            "destination": {"address": destination},
            "travelMode": travel_mode,
        },
        api_key,
    )
    routes = raw.get("routes") or []
    if not routes:
        raise GoogleMapsError(f"Google Maps found no route from {origin} to {destination}")
    route = routes[0]
    return RouteAnswer(
        origin=origin,
        destination=destination,
        travel_mode=travel_mode,
        distance_meters=route.get("distanceMeters"),
        duration_seconds=_seconds(route.get("duration")),
        maps_url=directions_url(origin, destination, travel_mode),
        attribution=_attribution(route.get("attribution")),
    )
