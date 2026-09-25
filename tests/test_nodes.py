from conftest import FakeLLM, draft_itinerary_json, draft_itinerary_payload

from app.agent.nodes import (
    build_final_response_node,
    build_flight_node,
    build_hotel_node,
    build_itinerary_node,
    build_resolve_destination_node,
)
from app.agent.state import initial_state
from app.models.itinerary import Itinerary


def test_resolve_destination_passes_through_a_named_destination(provider, trip_request):
    node = build_resolve_destination_node(provider)

    result = node(initial_state(trip_request))

    assert result == {"resolved_destination": "Lisbon, Portugal"}


def test_resolve_destination_picks_one_when_unset(provider, trip_request):
    request = trip_request.model_copy(update={"destination": None})
    node = build_resolve_destination_node(provider)

    result = node(initial_state(request))

    assert result["resolved_destination"]
    assert "coordinator:" in result["messages"][0]


def test_flight_node_skips_without_an_origin(provider, trip_request):
    request = trip_request.model_copy(update={"origin": None})
    llm = FakeLLM()
    node = build_flight_node(provider, llm)
    state = initial_state(request) | {"resolved_destination": "Lisbon, Portugal"}

    result = node(state)

    assert result["flight_results"]["outbound_options"] == []
    assert result["flight_results"]["return_options"] == []
    assert llm.calls == []  # no point asking the LLM to pick from nothing


def test_flight_node_searches_both_legs_and_asks_the_llm_to_pick(provider, trip_request):
    llm = FakeLLM(responses=["Take Meridian Air both ways — cheapest and direct."])
    node = build_flight_node(provider, llm)
    state = initial_state(trip_request) | {"resolved_destination": "Lisbon, Portugal"}

    result = node(state)

    outbound = result["flight_results"]["outbound_options"]
    return_leg = result["flight_results"]["return_options"]
    assert len(outbound) > 0
    assert len(return_leg) > 0
    # Return leg runs the reverse route, departing on the trip's end date.
    assert return_leg[0]["origin"] == "Lisbon, Portugal"
    assert return_leg[0]["destination"] == trip_request.origin
    assert result["flight_results"]["recommendation"].startswith("Take Meridian")
    assert llm.calls[0]["json_mode"] is False


def test_flight_agent_is_told_what_is_booked_rather_than_asked_to_guess(
    trip_request, monkeypatch
):
    """Asked to predict the booking, gpt-4o-mini kept naming the cheapest
    flight — a day early — as "chosen" when the one on the travel date was
    actually booked. Handing it the booked flight removes the guess."""
    import json

    from app.providers.mock import MockTravelProvider

    start, end = trip_request.start_date, trip_request.end_date
    day_before = start.fromordinal(start.toordinal() - 1).isoformat()
    legs = {
        "outbound": [_flight(day_before, 5000), _flight(start.isoformat(), 6000)],
        "return": [_flight(end.isoformat(), 7000)],
    }

    class WindowedProvider(MockTravelProvider):
        def search_flights(self, origin, destination, depart, travelers):
            return legs["outbound"] if depart == start else legs["return"]

    llm = FakeLLM(responses=["Booked as shown."])
    node = build_flight_node(WindowedProvider(), llm)

    node(initial_state(trip_request) | {"resolved_destination": "Lisbon, Portugal"})

    sent = json.loads(llm.calls[0]["user"])
    assert sent["booked_outbound"]["depart_date"] == start.isoformat()
    assert sent["booked_outbound"]["total"] == 6000
    assert sent["booked_return"]["total"] == 7000


def test_hotel_node_asks_the_llm_to_pick_an_option(provider, trip_request):
    llm = FakeLLM(responses=["The Ardent House — best rated, worth the splurge."])
    node = build_hotel_node(provider, llm)
    state = initial_state(trip_request) | {"resolved_destination": "Lisbon, Portugal"}

    result = node(state)

    assert len(result["hotel_results"]["options"]) > 0
    assert "Ardent" in result["hotel_results"]["recommendation"]


def _real_flight_and_hotel_state(provider, trip_request, destination="Lisbon, Portugal"):
    """Flight/hotel state shaped exactly like the real nodes produce it, so
    itinerary-assembly tests exercise the same FlightLeg/LodgingOption mapping
    the pipeline actually uses."""
    outbound = provider.search_flights(
        origin=trip_request.origin,
        destination=destination,
        depart=trip_request.start_date,
        travelers=trip_request.travelers,
    )
    return_leg = provider.search_flights(
        origin=destination,
        destination=trip_request.origin,
        depart=trip_request.end_date,
        travelers=trip_request.travelers,
    )
    hotels = provider.search_lodging(
        destination=destination,
        check_in=trip_request.start_date,
        nights=trip_request.nights,
        travelers=trip_request.travelers,
    )
    return (
        {
            "flight_results": {
                "outbound_options": outbound,
                "return_options": return_leg,
                "recommendation": "Meridian Air both ways.",
            },
            "hotel_results": {"options": hotels, "recommendation": "The Ardent House."},
        },
        outbound,
        return_leg,
        hotels,
    )


def test_itinerary_node_assembles_flights_and_lodging_from_real_search_data(
    provider, trip_request
):
    llm = FakeLLM(responses=[draft_itinerary_json(trip_request)])
    node = build_itinerary_node(provider, llm, max_retries=2)
    extra_state, outbound, return_leg, hotels = _real_flight_and_hotel_state(
        provider, trip_request
    )
    state = initial_state(trip_request) | {"resolved_destination": "Lisbon, Portugal"} | extra_state

    result = node(state)
    itinerary = result["itinerary"]

    assert itinerary is not None
    assert itinerary.destination == "Lisbon, Portugal"
    assert "attempt 1" in result["messages"][0]
    assert llm.calls[0]["json_mode"] is True

    # Cheapest (first, already sorted by the provider) option of each becomes
    # the structured pick — not whatever the LLM's free-text recommendation
    # said, so these numbers can't be hallucinated.
    assert itinerary.outbound_flight.carrier == outbound[0]["carrier"]
    assert itinerary.outbound_flight.total == outbound[0]["total"]
    assert itinerary.return_flight.origin == "Lisbon, Portugal"
    assert itinerary.return_flight.total == return_leg[0]["total"]
    assert [h.name for h in itinerary.lodging_options] == [h["name"] for h in hotels]

    activity_cost = sum(
        act["estimated_cost"]
        for day in draft_itinerary_payload(trip_request)["days"]
        for act in day["activities"]
    )
    expected_total = (
        outbound[0]["total"]
        + return_leg[0]["total"]
        + hotels[0]["total"]
        + activity_cost
    )
    assert itinerary.total_estimated_cost == expected_total


def test_itinerary_node_handles_no_flights_or_lodging_found(provider, trip_request):
    llm = FakeLLM(responses=[draft_itinerary_json(trip_request)])
    node = build_itinerary_node(provider, llm, max_retries=2)
    state = initial_state(trip_request) | {
        "resolved_destination": "Lisbon, Portugal",
        "flight_results": {},
        "hotel_results": {},
    }

    result = node(state)
    itinerary = result["itinerary"]

    assert itinerary.outbound_flight is None
    assert itinerary.return_flight is None
    assert itinerary.lodging_options == []
    assert itinerary.total_estimated_cost == sum(
        act["estimated_cost"]
        for day in draft_itinerary_payload(trip_request)["days"]
        for act in day["activities"]
    )


def _flight(on: str, total: float) -> dict:
    return {
        "carrier": "IX",
        "origin": "DEL",
        "destination": "LIS",
        "depart_date": on,
        "stops": 0,
        "duration_hours": 3.0,
        "price_per_person": total,
        "total": total,
    }


def test_itinerary_node_books_on_the_travel_date_over_a_cheaper_day_nearby(
    provider, trip_request
):
    """A live search covers a day either side of the travel date. The flight
    that actually gets booked must still be on the travel date itself — a
    cheaper one the day before isn't the trip that was asked for."""
    start, end = trip_request.start_date, trip_request.end_date
    day_before = start.fromordinal(start.toordinal() - 1).isoformat()
    llm = FakeLLM(responses=[draft_itinerary_json(trip_request)])
    node = build_itinerary_node(provider, llm, max_retries=2)
    state = initial_state(trip_request) | {
        "resolved_destination": "Lisbon, Portugal",
        # Sorted cheapest-first, as providers return them.
        "flight_results": {
            "outbound_options": [_flight(day_before, 5000), _flight(start.isoformat(), 6000)],
            "return_options": [_flight(end.isoformat(), 7000)],
        },
        "hotel_results": {},
    }

    itinerary = node(state)["itinerary"]

    assert itinerary.outbound_flight.depart_date == start
    assert itinerary.outbound_flight.total == 6000
    assert len(itinerary.outbound_options) == 2  # the cheaper day is still listed


def test_itinerary_node_falls_back_to_a_nearby_day_when_none_on_the_date(
    provider, trip_request
):
    start = trip_request.start_date
    next_day = start.fromordinal(start.toordinal() + 1).isoformat()
    llm = FakeLLM(responses=[draft_itinerary_json(trip_request)])
    node = build_itinerary_node(provider, llm, max_retries=2)
    state = initial_state(trip_request) | {
        "resolved_destination": "Lisbon, Portugal",
        "flight_results": {"outbound_options": [_flight(next_day, 6500)]},
        "hotel_results": {},
    }

    itinerary = node(state)["itinerary"]

    assert itinerary.outbound_flight.depart_date.isoformat() == next_day


def test_itinerary_node_retries_after_an_invalid_response(provider, trip_request):
    llm = FakeLLM(responses=["not json at all", draft_itinerary_json(trip_request)])
    node = build_itinerary_node(provider, llm, max_retries=2)
    state = initial_state(trip_request) | {
        "resolved_destination": "Lisbon, Portugal",
        "flight_results": {},
        "hotel_results": {},
    }

    result = node(state)

    assert result["itinerary"] is not None
    assert "attempt 2" in result["messages"][0]
    assert "rejected" in llm.calls[1]["user"]


def test_itinerary_node_gives_up_after_max_retries(provider, trip_request):
    llm = FakeLLM(responses=["nope", "still nope", "nope again"])
    node = build_itinerary_node(provider, llm, max_retries=2)
    state = initial_state(trip_request) | {
        "resolved_destination": "Lisbon, Portugal",
        "flight_results": {},
        "hotel_results": {},
    }

    result = node(state)

    assert result["itinerary"] is None
    assert len(result["errors"]) == 1
    assert len(llm.calls) == 3  # one initial attempt + 2 retries


def test_final_response_node_summarises_the_itinerary(trip_request):
    itinerary = Itinerary(
        destination="Lisbon, Portugal",
        start_date=trip_request.start_date,
        end_date=trip_request.end_date,
        travelers=trip_request.travelers,
        **draft_itinerary_payload(trip_request),
    )
    llm = FakeLLM(responses=["Four sunny days in Lisbon, food-forward and easy."])
    node = build_final_response_node(llm)
    state = initial_state(trip_request) | {
        "itinerary": itinerary,
        "flight_results": {"recommendation": "Meridian Air"},
        "hotel_results": {"recommendation": "The Ardent House"},
    }

    result = node(state)

    assert result["final_response"] == "Four sunny days in Lisbon, food-forward and easy."


def test_final_response_node_skips_when_there_is_no_itinerary(trip_request):
    llm = FakeLLM()
    node = build_final_response_node(llm)
    state = initial_state(trip_request)

    result = node(state)

    assert result["final_response"] == ""
    assert llm.calls == []


# --- flights / hotels checkboxes ---------------------------------------------


class _NoSearchProvider:
    """Fails loudly if asked to search — proves a skipped agent really
    skipped, rather than searching and discarding (which would still spend a
    SerpApi search on live data)."""

    def search_flights(self, *args, **kwargs):
        raise AssertionError("flight search ran although flights weren't requested")

    def search_lodging(self, *args, **kwargs):
        raise AssertionError("hotel search ran although hotels weren't requested")


def test_flight_node_skips_search_and_llm_when_flights_are_unchecked(trip_request):
    request = trip_request.model_copy(update={"include_flights": False})
    llm = FakeLLM()  # no scripted responses: any LLM call would fail the test
    node = build_flight_node(_NoSearchProvider(), llm)

    result = node(initial_state(request) | {"resolved_destination": "Lisbon, Portugal"})

    assert result["flight_results"]["outbound_options"] == []
    assert result["flight_results"]["return_options"] == []
    assert "not requested" in result["messages"][0]
    assert llm.calls == []


def test_hotel_node_skips_search_and_llm_when_hotels_are_unchecked(trip_request):
    request = trip_request.model_copy(update={"include_hotels": False})
    llm = FakeLLM()
    node = build_hotel_node(_NoSearchProvider(), llm)

    result = node(initial_state(request) | {"resolved_destination": "Lisbon, Portugal"})

    assert result["hotel_results"]["options"] == []
    assert "not requested" in result["messages"][0]
    assert llm.calls == []


def test_itinerary_agent_is_told_flights_and_hotels_were_not_requested(provider, trip_request):
    """Otherwise it's told "No lodging options were found", which invites it
    to improvise a hotel into the plan."""
    request = trip_request.model_copy(update={"include_flights": False, "include_hotels": False})
    llm = FakeLLM(responses=[draft_itinerary_json(request)])
    node = build_itinerary_node(provider, llm, max_retries=2)
    state = initial_state(request) | {
        "resolved_destination": "Lisbon, Portugal",
        "flight_results": {"outbound_options": [], "return_options": []},
        "hotel_results": {"options": []},
    }

    itinerary = node(state)["itinerary"]

    prompt = llm.calls[0]["user"]
    assert "didn't ask for flights" in prompt
    assert "didn't ask for a hotel" in prompt
    assert itinerary.outbound_flight is None
    assert itinerary.lodging_options == []


def test_both_are_searched_by_default(provider, trip_request):
    """Every caller that predates the checkboxes keeps the full plan."""
    assert trip_request.include_flights is True
    assert trip_request.include_hotels is True
