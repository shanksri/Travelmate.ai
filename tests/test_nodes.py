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
    assert itinerary.outbound_flight.total_usd == outbound[0]["total_usd"]
    assert itinerary.return_flight.origin == "Lisbon, Portugal"
    assert itinerary.return_flight.total_usd == return_leg[0]["total_usd"]
    assert [h.name for h in itinerary.lodging_options] == [h["name"] for h in hotels]

    activity_cost = sum(
        act["estimated_cost_usd"]
        for day in draft_itinerary_payload(trip_request)["days"]
        for act in day["activities"]
    )
    expected_total = (
        outbound[0]["total_usd"]
        + return_leg[0]["total_usd"]
        + hotels[0]["total_usd"]
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
        act["estimated_cost_usd"]
        for day in draft_itinerary_payload(trip_request)["days"]
        for act in day["activities"]
    )


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
