"""The ask graph end to end, through `answer_prompt`: the sentence is routed
to a trip, places or a route, and a trip that asked for places gets one
parallel lookup per city. The LLM is scripted (FakeLLM), travel data is the
mock provider, and Google Maps is faked at the module boundary — no network.
"""

import json
from datetime import date

import pytest
from conftest import FakeLLM

from app.agent import assistant
from app.agent.assistant import answer_prompt
from app.agent.prompt_parser import PromptParseError
from app.agent.prompts import (
    FINAL_RESPONSE_AGENT_SYSTEM,
    FLIGHT_AGENT_SYSTEM,
    HOTEL_AGENT_SYSTEM,
    ITINERARY_AGENT_SYSTEM,
    PARSE_REQUEST_SYSTEM,
)
from app.core.config import Settings
from app.models.maps import PlaceLink, PlacesAnswer, RouteAnswer
from app.providers import google_maps
from app.providers.google_maps import GoogleMapsError
from app.providers.mock import MockTravelProvider

START, END = date(2027, 1, 10), date(2027, 1, 12)


@pytest.fixture
def settings() -> Settings:
    return Settings(model="gpt-4o-mini", max_itinerary_retries=2)


@pytest.fixture
def maps(monkeypatch):
    """Records what was asked of Google Maps; per-query answers can be set
    with `maps["answer"] = callable(query)`."""
    seen: dict = {"queries": []}

    def search_places(query):
        seen["queries"].append(query)
        if "answer" in seen:
            return seen["answer"](query)
        return PlacesAnswer(query=query, summary=f"About {query} [0].", places=[PlaceLink(index=0)])

    def compute_route(origin, destination, travel_mode="DRIVE"):
        seen["route"] = (origin, destination, travel_mode)
        return RouteAnswer(
            origin=origin, destination=destination, travel_mode=travel_mode, maps_url="u"
        )

    monkeypatch.setattr(google_maps, "search_places", search_places)
    monkeypatch.setattr(google_maps, "compute_route", compute_route)
    return seen


def _parse(**fields) -> str:
    return json.dumps({"intent": "trip", "destination": None, "origin": None} | fields)


def _days(cities: list[str]) -> str:
    return json.dumps(
        {
            "notes": [],
            "days": [
                {
                    "day": i + 1,
                    "date": date.fromordinal(START.toordinal() + i).isoformat(),
                    "summary": f"Day {i + 1} in {city}",
                    "city": city,
                    "activities": [{"time": "09:00", "title": f"Walk {city}"}],
                }
                for i, city in enumerate(cities)
            ],
        }
    )


def trip_llm(cities=("Kochi", "Munnar", "Munnar"), **parse_fields) -> FakeLLM:
    """Everything a full trip run asks the LLM, scripted."""
    parse = {
        "destination": "Kerala",
        "origin": "Delhi",
        "start_date": START.isoformat(),
        "end_date": date.fromordinal(START.toordinal() + len(cities) - 1).isoformat(),
    } | parse_fields
    return FakeLLM(
        by_system={
            PARSE_REQUEST_SYSTEM: [_parse(**parse)],
            FLIGHT_AGENT_SYSTEM: ["Cheapest on the day."],
            HOTEL_AGENT_SYSTEM: ["Cheapest is fine."],
            ITINERARY_AGENT_SYSTEM: [_days(list(cities))],
            FINAL_RESPONSE_AGENT_SYSTEM: ["Three days in Kerala."],
        }
    )


def ask(prompt, llm, settings, **choices):
    return answer_prompt(
        prompt, llm=llm, provider=MockTravelProvider(), settings=settings, **choices
    )


# --- routing -----------------------------------------------------------------


def test_a_places_request_goes_to_google_maps_and_nothing_else(maps, settings):
    llm = FakeLLM(
        by_system={
            PARSE_REQUEST_SYSTEM: [
                _parse(intent="places", places_query="best restaurants in Bhubaneswar")
            ]
        }
    )

    answer = ask("best places to eat in bhuvneshwar", llm, settings)

    assert answer.kind == "places"
    assert answer.places.query == "best restaurants in Bhubaneswar"
    assert answer.trip is None
    assert [c["system"] for c in llm.calls] == [PARSE_REQUEST_SYSTEM]


def test_a_route_request_goes_to_google_maps(maps, settings):
    llm = FakeLLM(
        by_system={
            PARSE_REQUEST_SYSTEM: [
                _parse(
                    intent="route", origin="Madurai", destination="Rameswaram", travel_mode="WALK"
                )
            ]
        }
    )

    answer = ask("walk from madurai to rameswaram", llm, settings)

    assert answer.kind == "route"
    assert maps["route"] == ("Madurai", "Rameswaram", "WALK")


def test_a_trip_request_runs_the_whole_trip_graph(maps, settings):
    answer = ask("3 days in Kerala from Delhi", trip_llm(), settings)

    assert answer.kind == "trip"
    trip = answer.trip
    assert trip.version == 1 and trip.thread_id == trip.id
    assert [d.city for d in trip.itinerary.days] == ["Kochi", "Munnar", "Munnar"]
    assert trip.summary == "Three days in Kerala."
    # Every step leaves its mark in the trace — the routing step included.
    assert trip.agent_trace[0] == "interpreter: read as a trip request"
    assert any(m.startswith("itinerary_agent") for m in trip.agent_trace)
    assert answer.places_by_city == [] and maps["queries"] == []


def test_the_pages_choices_shape_the_trip(maps, settings):
    answer = ask(
        "3 days in Kerala from Delhi",
        trip_llm(),
        settings,
        include_flights=False,
        include_hotels=True,
        start_date=START,
        end_date=END,
    )

    request = answer.trip.request
    assert request.include_flights is False and request.include_hotels is True
    assert (request.start_date, request.end_date) == (START, END)
    assert answer.trip.itinerary.outbound_options == []  # flights unticked: not searched
    assert answer.trip.itinerary.lodging_source.status == "sample"


def test_a_trip_with_no_destination_is_refused_before_planning(maps, settings):
    llm = trip_llm(destination=None)

    with pytest.raises(PromptParseError, match="no destination was named"):
        ask("plan me a trip from Delhi", llm, settings)

    assert [c["system"] for c in llm.calls] == [PARSE_REQUEST_SYSTEM]


def test_half_a_date_pair_is_refused(settings):
    with pytest.raises(ValueError, match="both start_date and end_date"):
        ask("Goa", trip_llm(), settings, start_date=START)


# --- places in each city: the Send fan-out ------------------------------------


def test_restaurants_are_looked_up_once_per_city_in_trip_order(maps, settings):
    llm = trip_llm(cities=("Kochi", "Munnar", "Alleppey"))

    answer = ask("3 days in Kerala", llm, settings, include_restaurants=True)

    assert sorted(maps["queries"]) == sorted(
        ["best restaurants in Kochi", "best restaurants in Munnar", "best restaurants in Alleppey"]
    )
    # The parallel searches finish in any order; the answer is in trip order.
    assert [c.city for c in answer.places_by_city] == ["Kochi", "Munnar", "Alleppey"]
    assert answer.trip.request.places_per_city == "restaurants"


def test_the_sentence_picks_the_kind_of_place(maps, settings):
    llm = trip_llm(cities=("Kolkata",), places_per_city="street food")

    ask("kolkata, with street food", llm, settings, include_restaurants=True)

    assert maps["queries"] == ["best street food in Kolkata"]


def test_without_the_restaurants_box_nothing_is_looked_up(maps, settings):
    llm = trip_llm(places_per_city="restaurants")

    answer = ask("kerala, best places to eat in each city", llm, settings)

    assert maps["queries"] == [] and answer.places_by_city == []
    assert answer.trip.request.places_per_city is None


def test_one_city_failing_keeps_the_others_and_the_trip(maps, settings):
    def answer_for(query):
        if "Munnar" in query:
            raise GoogleMapsError("quota exceeded")
        return PlacesAnswer(query=query, summary="s", places=[])

    maps["answer"] = answer_for

    answer = ask(
        "kerala", trip_llm(cities=("Kochi", "Munnar")), settings, include_restaurants=True
    )

    kochi, munnar = answer.places_by_city
    assert kochi.places is not None and kochi.error is None
    assert munnar.places is None and munnar.error == "quota exceeded"
    assert answer.trip is not None


def test_cities_fall_back_to_the_destination_when_days_have_none(trip_request):
    from conftest import sample_planned_trip

    from app.models.itinerary import DayPlan

    days = [DayPlan(day=1, date=START, summary="1"), DayPlan(day=2, date=END, summary="2")]
    trip = sample_planned_trip(trip_request, days=days)

    assert assistant.trip_cities(trip) == [trip.itinerary.destination]


def test_a_long_trip_searches_at_most_max_cities(maps, settings):
    cities = tuple(f"Town {i}" for i in range(assistant.MAX_CITIES + 3))

    ask("a long trip", trip_llm(cities=cities), settings, include_restaurants=True)

    assert len(maps["queries"]) == assistant.MAX_CITIES
