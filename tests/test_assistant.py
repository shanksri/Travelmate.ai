"""`answer_prompt` sends each request where the parser says it belongs.
Google Maps is faked at the module boundary, so no MCP call is made."""

import json
from datetime import date

import pytest
from conftest import FakeLLM, sample_planned_trip

from app.agent import assistant
from app.agent.assistant import answer_prompt
from app.agent.prompts import PARSE_REQUEST_SYSTEM
from app.core.config import Settings
from app.models.itinerary import DayPlan
from app.models.maps import PlaceLink, PlacesAnswer, RouteAnswer


@pytest.fixture
def settings() -> Settings:
    return Settings(model="gpt-4o-mini", max_itinerary_retries=2)


def _parse_llm(**fields) -> FakeLLM:
    payload = {"intent": "trip", "destination": None, "origin": None} | fields
    return FakeLLM(by_system={PARSE_REQUEST_SYSTEM: [json.dumps(payload)]})


@pytest.fixture
def fake_maps(monkeypatch):
    seen = {}

    def search_places(query):
        seen["places"] = query
        return PlacesAnswer(query=query, summary="Dalma [0].", places=[PlaceLink(index=0)])

    def compute_route(origin, destination, travel_mode="DRIVE"):
        seen["route"] = (origin, destination, travel_mode)
        return RouteAnswer(
            origin=origin, destination=destination, travel_mode=travel_mode, maps_url="u"
        )

    monkeypatch.setattr(assistant.google_maps, "search_places", search_places)
    monkeypatch.setattr(assistant.google_maps, "compute_route", compute_route)
    return seen


def test_a_places_request_goes_to_google_maps(fake_maps, settings):
    llm = _parse_llm(intent="places", places_query="best restaurants in Bhubaneswar")

    answer = answer_prompt("best places to eat in bhuvneshwar", llm=llm, settings=settings)

    assert answer.kind == "places"
    assert answer.places.summary == "Dalma [0]."
    assert answer.trip is None
    assert fake_maps["places"] == "best restaurants in Bhubaneswar"
    # Only the parser ran: no agent graph, no other LLM call.
    assert [c["system"] for c in llm.calls] == [PARSE_REQUEST_SYSTEM]


def test_a_route_request_goes_to_google_maps(fake_maps, settings):
    llm = _parse_llm(intent="route", origin="Madurai", destination="Rameswaram")

    answer = answer_prompt("how far is rameshwaram from madurai", llm=llm, settings=settings)

    assert answer.kind == "route"
    assert fake_maps["route"] == ("Madurai", "Rameswaram", "DRIVE")


def test_a_walking_route_keeps_its_mode(fake_maps, settings):
    llm = _parse_llm(
        intent="route", origin="Puri beach", destination="Jagannath Temple", travel_mode="WALK"
    )

    answer_prompt("walk from puri beach to the temple", llm=llm, settings=settings)

    assert fake_maps["route"][2] == "WALK"


def _trip_with_cities(trip_request, cities, *, places_per_city=None):
    """A planned trip whose days are based in `cities`, in order."""
    request = trip_request.model_copy(update={"places_per_city": places_per_city})
    days = [
        DayPlan(day=i + 1, date=date(2027, 1, 10 + i), summary=f"Day {i + 1}", city=city)
        for i, city in enumerate(cities)
    ]
    return sample_planned_trip(request, days=days)


@pytest.fixture
def fake_plan(monkeypatch):
    """Stops answer_prompt at the planner, returning `seen["trip"]`."""
    seen = {}

    def plan(parsed, **kwargs):
        seen["parsed"] = parsed
        seen.update(kwargs)
        return seen["trip"]

    monkeypatch.setattr(assistant, "plan_parsed_trip", plan)
    return seen


def test_a_trip_request_is_planned_with_the_pages_choices(
    fake_plan, fake_maps, trip_request, settings
):
    fake_plan["trip"] = _trip_with_cities(trip_request, ["Goa"])
    llm = _parse_llm(intent="trip", destination="Goa")

    answer = answer_prompt(
        "4 days in Goa",
        include_flights=False,
        include_hotels=True,
        start_date=date(2027, 1, 10),
        end_date=date(2027, 1, 13),
        llm=llm,
        settings=settings,
    )

    assert answer.kind == "trip"
    assert answer.trip is fake_plan["trip"]
    assert fake_plan["parsed"].destination == "Goa"
    assert fake_plan["include_flights"] is False
    assert fake_plan["start_date"] == date(2027, 1, 10)
    # Nothing asked for places, so Maps isn't touched.
    assert answer.places_by_city == []
    assert "places" not in fake_maps and "route" not in fake_maps


def test_a_trip_that_asks_for_places_gets_one_search_per_city(
    fake_plan, monkeypatch, trip_request, settings
):
    queries = []

    def search_places(query):
        queries.append(query)
        return PlacesAnswer(query=query, summary="s", places=[])

    monkeypatch.setattr(assistant.google_maps, "search_places", search_places)
    fake_plan["trip"] = _trip_with_cities(
        trip_request,
        ["Kochi", "Kochi", "Alleppey", "Munnar", "munnar", None],
        places_per_city="restaurants",
    )
    llm = _parse_llm(intent="trip", destination="Kerala", places_per_city="restaurants")

    answer = answer_prompt(
        "6 day kerala itinerary, best places to eat in each city", llm=llm, settings=settings
    )

    assert [c.city for c in answer.places_by_city] == ["Kochi", "Alleppey", "Munnar"]
    assert sorted(queries) == sorted(
        ["best restaurants in Kochi", "best restaurants in Alleppey", "best restaurants in Munnar"]
    )
    assert answer.places_by_city[0].places.query == "best restaurants in Kochi"


def test_one_city_failing_still_returns_the_trip(fake_plan, monkeypatch, trip_request, settings):
    from app.providers.google_maps import GoogleMapsError

    def search_places(query):
        if "Munnar" in query:
            raise GoogleMapsError("quota exceeded")
        return PlacesAnswer(query=query, summary="s", places=[])

    monkeypatch.setattr(assistant.google_maps, "search_places", search_places)
    fake_plan["trip"] = _trip_with_cities(
        trip_request, ["Kochi", "Munnar"], places_per_city="restaurants"
    )

    answer = answer_prompt(
        "kerala, places to eat in each city", llm=_parse_llm(), settings=settings
    )

    assert answer.trip is fake_plan["trip"]
    kochi, munnar = answer.places_by_city
    assert kochi.places is not None and kochi.error is None
    assert munnar.places is None and munnar.error == "quota exceeded"


def test_cities_fall_back_to_the_destination_when_days_have_none(trip_request):
    trip = _trip_with_cities(trip_request, [None, None])

    assert assistant.trip_cities(trip) == [trip.itinerary.destination]


def test_a_long_trip_searches_at_most_max_cities(trip_request):
    cities = [f"Town {i}" for i in range(assistant.MAX_CITIES + 3)]

    assert len(assistant.trip_cities(_trip_with_cities(trip_request, cities))) == (
        assistant.MAX_CITIES
    )


def test_half_a_date_pair_is_refused(settings):
    with pytest.raises(ValueError, match="both start_date and end_date"):
        answer_prompt("Goa", start_date=date(2027, 1, 1), llm=_parse_llm(), settings=settings)
