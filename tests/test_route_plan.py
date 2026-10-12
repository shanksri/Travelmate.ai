"""The route planner (app/agent/route_plan.py): its answer is checked in code
and retried with the reason, and a trip still plans if it never comes right."""

import json
from datetime import date

from conftest import FakeLLM, route_json

from app.agent.prompts import ROUTE_PLAN_SYSTEM
from app.agent.route_plan import MAX_STOPS, build_route_plan_node, describe_route, plan_route
from app.agent.state import initial_state
from app.providers.mock import MockTravelProvider


class _Airports(MockTravelProvider):
    """Knows airports only in the towns it's given."""

    def __init__(self, *towns):
        self.towns = set(towns)

    def has_airport(self, city):
        return city in self.towns


def _llm(*answers) -> FakeLLM:
    return FakeLLM(by_system={ROUTE_PLAN_SYSTEM: list(answers)})


def _request(trip_request, **changes):
    # Kerala and Tamil Nadu, 26 Oct to 4 Nov: 9 nights.
    return trip_request.model_copy(
        update={
            "destination": "Kerala and Tamil Nadu",
            "origin": "Varanasi",
            "start_date": date(2026, 10, 26),
            "end_date": date(2026, 11, 4),
        }
        | changes
    )


AIRPORTS = _Airports("Thiruvananthapuram", "Madurai")

KERALA_TN = route_json(
    ("Thiruvananthapuram", 2), ("Kochi", 3), ("Madurai", 4),
    fly_into="Thiruvananthapuram", fly_out_of="Madurai",
)


def test_a_good_route_is_taken_as_it_is(trip_request):
    request = _request(trip_request)
    llm = _llm(KERALA_TN)

    plan, note = plan_route(request, request.destination, llm, AIRPORTS)

    assert [(s.city, s.nights) for s in plan.stops] == [
        ("Thiruvananthapuram", 2), ("Kochi", 3), ("Madurai", 4)]
    assert (plan.fly_into, plan.fly_out_of) == ("Thiruvananthapuram", "Madurai")
    assert note == "route_planner: Thiruvananthapuram 2 nights → Kochi 3 → Madurai 4 (attempt 1)"
    assert llm.calls[0]["schema"].__name__ == "RoutePlan"
    assert "Flights: needed, to and from Varanasi." in llm.calls[0]["user"]


def test_nights_that_dont_add_up_are_sent_back(trip_request):
    request = _request(trip_request)
    llm = _llm(route_json(("Kochi", 3), ("Madurai", 3)), KERALA_TN)

    plan, _ = plan_route(request, request.destination, llm, AIRPORTS)

    assert len(plan.stops) == 3
    assert "The nights add up to 6, but the trip has 9 nights." in llm.calls[1]["user"]


def test_too_many_stops_are_sent_back(trip_request):
    request = _request(trip_request)
    rushed = route_json(*[(f"Town {i}", 1) for i in range(8)], ("Last", 1))
    llm = _llm(rushed, KERALA_TN)

    plan_route(request, request.destination, llm, AIRPORTS)

    assert f"use at most {MAX_STOPS}" in llm.calls[1]["user"]


def test_an_airport_that_cant_be_found_is_sent_back(trip_request):
    """"Trivandrum" isn't in the airport list; "Thiruvananthapuram" is."""
    request = _request(trip_request)
    nickname = route_json(
        ("Thiruvananthapuram", 2), ("Kochi", 3), ("Madurai", 4),
        fly_into="Trivandrum", fly_out_of="Madurai",
    )
    llm = _llm(nickname, KERALA_TN)

    plan, _ = plan_route(request, request.destination, llm, AIRPORTS)

    assert plan.fly_into == "Thiruvananthapuram"
    assert "No airport found for 'Trivandrum'" in llm.calls[1]["user"]


def test_airports_arent_checked_when_there_are_no_flights(trip_request):
    request = _request(trip_request, include_flights=False)
    no_flights = json.loads(KERALA_TN) | {"fly_into": None, "fly_out_of": None}
    llm = _llm(json.dumps(no_flights))

    plan, _ = plan_route(request, request.destination, llm, _Airports())

    assert plan.fly_into is None and len(llm.calls) == 1
    assert "Flights: not needed." in llm.calls[0]["user"]


def test_a_route_that_never_comes_right_falls_back_to_one_stop(trip_request):
    request = _request(trip_request)
    wrong = route_json(("Kochi", 1))
    llm = _llm(wrong, wrong, wrong)

    plan, note = plan_route(request, request.destination, llm, _Airports("Kochi"))

    assert [(s.city, s.nights) for s in plan.stops] == [("Kerala and Tamil Nadu", 9)]
    assert note.startswith("route_planner: fell back to one stop")
    assert len(llm.calls) == 3


def test_a_day_trip_needs_no_route(trip_request):
    request = _request(trip_request, end_date=date(2026, 10, 26))
    llm = _llm()  # nothing scripted: a call would fail the test

    plan, _ = plan_route(request, "Kochi", llm, MockTravelProvider())

    assert [(s.city, s.nights) for s in plan.stops] == [("Kochi", 0)]
    assert llm.calls == []


def test_the_step_runs_on_the_route_model_when_it_can(trip_request):
    """`OpenAILLM.with_model` gives the same client on gpt-4o; a fake without
    it answers as itself."""

    class _Switchable(FakeLLM):
        def with_model(self, model):
            self.switched_to = model
            return self

    llm = _Switchable(by_system={ROUTE_PLAN_SYSTEM: [KERALA_TN]})
    node = build_route_plan_node(_Airports("Thiruvananthapuram", "Madurai"), llm, "gpt-4o")
    request = _request(trip_request)

    result = node(initial_state(request) | {"resolved_destination": request.destination})

    assert llm.switched_to == "gpt-4o"
    assert describe_route(result["route_plan"]) == (
        "Thiruvananthapuram 2 nights → Kochi 3 → Madurai 4")
