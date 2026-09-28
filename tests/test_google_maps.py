"""Tests mock `_call_tool` (the MCP round trip to Maps Grounding Lite) — no
real call is made, and no Maps quota is spent."""

import pytest
from conftest import FakeMCPToolResult
from mcp.shared.exceptions import MCPError
from mcp.types import ErrorData

from app.providers import google_maps
from app.providers.google_maps import GoogleMapsError, compute_route, search_places

PLACES = {
    "summary": "Top picks: **Dalma** [0] for Odia thalis, and **Truptee** [1] for seafood.",
    "places": [
        {
            "id": "a",
            "googleMapsLinks": {
                "placeUrl": "https://maps.google.com/?cid=1",
                "reviewsUrl": "https://maps.google.com/?cid=1&reviews",
            },
            "attribution": {"title": "Google Maps", "url": "https://maps.google.com"},
        },
        {"id": "b", "googleMapsLinks": {"placeUrl": "https://maps.google.com/?cid=2"}},
    ],
}

ROUTE = {
    "routes": [
        {
            "distanceMeters": 172345,
            "duration": "12330s",
            "attribution": {"title": "Google Maps", "url": "https://maps.google.com"},
        }
    ]
}


@pytest.fixture
def fake_maps(monkeypatch):
    """Returns a function that makes `_call_tool` answer with `result`, and
    records every call made."""
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "fake")
    calls = []

    def install(result):
        async def fake_call_tool(tool, arguments, api_key):
            calls.append({"tool": tool, "arguments": arguments, "api_key": api_key})
            if isinstance(result, BaseException):
                raise result
            return result

        monkeypatch.setattr(google_maps, "_call_tool", fake_call_tool)
        return calls

    return install


def test_search_places_reads_the_summary_and_links(fake_maps):
    calls = fake_maps(FakeMCPToolResult(structured=PLACES))

    answer = search_places("best restaurants in Bhubaneswar")

    assert calls[0]["tool"] == "search_places"
    assert calls[0]["arguments"] == {"textQuery": "best restaurants in Bhubaneswar"}
    assert answer.query == "best restaurants in Bhubaneswar"
    assert answer.summary.startswith("Top picks")
    assert [p.index for p in answer.places] == [0, 1]
    assert answer.places[0].place_url == "https://maps.google.com/?cid=1"
    assert answer.places[0].reviews_url.endswith("&reviews")
    assert answer.places[0].attribution.title == "Google Maps"
    assert answer.places[1].attribution is None


def test_search_places_falls_back_to_the_text_block(fake_maps):
    """Some MCP servers only send JSON as text, not as structured content."""
    fake_maps(FakeMCPToolResult(PLACES))

    assert search_places("cafes in Puri").summary.startswith("Top picks")


def test_an_empty_search_is_an_error(fake_maps):
    fake_maps(FakeMCPToolResult(structured={"summary": "", "places": []}))

    with pytest.raises(GoogleMapsError, match="found nothing"):
        search_places("restaurants on the moon")


def test_compute_route_reads_distance_and_duration(fake_maps):
    calls = fake_maps(FakeMCPToolResult(structured=ROUTE))

    route = compute_route("Madurai", "Rameswaram")

    assert calls[0]["tool"] == "compute_routes"
    assert calls[0]["arguments"] == {
        "origin": {"address": "Madurai"},
        "destination": {"address": "Rameswaram"},
        "travelMode": "DRIVE",
    }
    assert route.distance_meters == 172345
    assert route.duration_seconds == 12330
    assert route.maps_url.startswith("https://www.google.com/maps/dir/?api=1")
    assert "travelmode=driving" in route.maps_url
    assert route.attribution.title == "Google Maps"


def test_a_walking_route_asks_for_walk_and_links_walking_directions(fake_maps):
    calls = fake_maps(FakeMCPToolResult(structured=ROUTE))

    route = compute_route("Puri beach", "Jagannath Temple", "WALK")

    assert calls[0]["arguments"]["travelMode"] == "WALK"
    assert "travelmode=walking" in route.maps_url


def test_no_route_is_an_error(fake_maps):
    fake_maps(FakeMCPToolResult(structured={"routes": []}))

    with pytest.raises(GoogleMapsError, match="no route"):
        compute_route("Chennai", "Colombo")


def test_a_tool_error_carries_googles_message(fake_maps):
    fake_maps(FakeMCPToolResult({"message": "ignored"}, is_error=True))
    # The error text is whatever the text block says.
    with pytest.raises(GoogleMapsError, match="ignored"):
        search_places("anything")


def test_a_refused_request_explains_what_to_check(fake_maps):
    """Live, an API that wasn't enabled surfaced only as JSON-RPC -32603,
    wrapped in ExceptionGroups by the SDK's task groups."""
    refused = MCPError.from_error_data(
        ErrorData(code=-32603, message="Server returned an error response")
    )
    fake_maps(ExceptionGroup("task group", [ExceptionGroup("inner", [refused])]))

    with pytest.raises(GoogleMapsError, match="Maps Grounding Lite API is enabled"):
        search_places("restaurants in Puri")


def test_a_transport_failure_becomes_a_google_maps_error(fake_maps):
    fake_maps(ConnectionError("network down"))

    with pytest.raises(GoogleMapsError, match="network down"):
        search_places("restaurants in Puri")


def test_a_missing_key_is_an_error(monkeypatch):
    monkeypatch.delenv("GOOGLE_MAPS_API_KEY", raising=False)

    with pytest.raises(GoogleMapsError, match="GOOGLE_MAPS_API_KEY is not set"):
        search_places("restaurants in Puri")


def test_search_places_many_answers_each_query_in_order(fake_maps, monkeypatch):
    from app.providers.google_maps import search_places_many

    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "fake")

    async def fake_call_tool(tool, arguments, api_key):
        query = arguments["textQuery"]
        if "Munnar" in query:
            return FakeMCPToolResult({"message": "quota exceeded"}, is_error=True)
        return FakeMCPToolResult(structured={"summary": f"about {query}", "places": []})

    monkeypatch.setattr(google_maps, "_call_tool", fake_call_tool)

    kochi, munnar, alleppey = search_places_many(
        ["best restaurants in Kochi", "best restaurants in Munnar", "best restaurants in Alleppey"]
    )

    assert kochi.summary == "about best restaurants in Kochi"
    assert isinstance(munnar, GoogleMapsError) and "quota exceeded" in str(munnar)
    assert alleppey.summary == "about best restaurants in Alleppey"


def test_an_unmocked_call_is_blocked_by_the_test_suite(monkeypatch):
    """The guard this file relies on: with nothing mocked, the offline MCP
    runtime refuses to connect, so no test can reach Google by accident."""
    monkeypatch.setenv("GOOGLE_MAPS_API_KEY", "fake")

    with pytest.raises(GoogleMapsError, match="A test attempted a real MCP session"):
        search_places("restaurants in Puri")


def test_durations_are_read_from_googles_format():
    assert google_maps._seconds("12330s") == 12330
    assert google_maps._seconds("59.6s") == 60
    assert google_maps._seconds(None) is None
    assert google_maps._seconds("soon") is None
