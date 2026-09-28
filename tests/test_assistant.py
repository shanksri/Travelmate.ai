"""`answer_prompt` sends each request where the parser says it belongs.
Google Maps is faked at the module boundary, so no MCP call is made."""

import json
from datetime import date

import pytest
from conftest import FakeLLM

from app.agent import assistant
from app.agent.assistant import answer_prompt
from app.agent.prompts import PARSE_REQUEST_SYSTEM
from app.core.config import Settings
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


def test_a_trip_request_is_planned_with_the_pages_choices(monkeypatch, fake_maps, settings):
    seen = {}

    def fake_plan(parsed, **kwargs):
        seen["parsed"] = parsed
        seen.update(kwargs)
        return "the trip"

    monkeypatch.setattr(assistant, "plan_parsed_trip", fake_plan)
    monkeypatch.setattr(assistant, "Answer", lambda **kw: kw)
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

    assert answer == {"kind": "trip", "trip": "the trip"}
    assert seen["parsed"].destination == "Goa"
    assert seen["include_flights"] is False
    assert seen["start_date"] == date(2027, 1, 10)
    assert "places" not in fake_maps and "route" not in fake_maps


def test_half_a_date_pair_is_refused(settings):
    with pytest.raises(ValueError, match="both start_date and end_date"):
        answer_prompt("Goa", start_date=date(2027, 1, 1), llm=_parse_llm(), settings=settings)
