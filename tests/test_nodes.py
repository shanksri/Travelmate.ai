from conftest import FakeLLM, itinerary_json

from app.agent.nodes import (
    build_final_response_node,
    build_flight_node,
    build_hotel_node,
    build_itinerary_node,
    build_resolve_destination_node,
)
from app.agent.state import initial_state


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

    assert result["flight_results"]["options"] == []
    assert llm.calls == []  # no point asking the LLM to pick from nothing


def test_flight_node_asks_the_llm_to_pick_an_option(provider, trip_request):
    llm = FakeLLM(responses=["Take Meridian Air — cheapest and direct."])
    node = build_flight_node(provider, llm)
    state = initial_state(trip_request) | {"resolved_destination": "Lisbon, Portugal"}

    result = node(state)

    assert len(result["flight_results"]["options"]) > 0
    assert result["flight_results"]["recommendation"].startswith("Take Meridian")
    assert llm.calls[0]["json_mode"] is False


def test_hotel_node_asks_the_llm_to_pick_an_option(provider, trip_request):
    llm = FakeLLM(responses=["The Ardent House — best rated, worth the splurge."])
    node = build_hotel_node(provider, llm)
    state = initial_state(trip_request) | {"resolved_destination": "Lisbon, Portugal"}

    result = node(state)

    assert len(result["hotel_results"]["options"]) > 0
    assert "Ardent" in result["hotel_results"]["recommendation"]


def test_itinerary_node_accepts_a_valid_response_on_the_first_try(provider, trip_request):
    llm = FakeLLM(responses=[itinerary_json(trip_request)])
    node = build_itinerary_node(provider, llm, max_retries=2)
    state = initial_state(trip_request) | {
        "resolved_destination": "Lisbon, Portugal",
        "flight_results": {"recommendation": "Meridian Air", "options": []},
        "hotel_results": {"recommendation": "The Ardent House", "options": []},
    }

    result = node(state)

    assert result["itinerary"] is not None
    assert result["itinerary"].destination == "Lisbon, Portugal"
    assert "attempt 1" in result["messages"][0]
    assert llm.calls[0]["json_mode"] is True


def test_itinerary_node_retries_after_an_invalid_response(provider, trip_request):
    llm = FakeLLM(responses=["not json at all", itinerary_json(trip_request)])
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


def test_final_response_node_summarises_the_itinerary(provider, trip_request):
    from app.agent.itinerary_json import parse_itinerary

    itinerary = parse_itinerary(itinerary_json(trip_request), trip_request)
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
