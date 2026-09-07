import pytest
from conftest import FakeLLM, draft_itinerary_json

from app.agent.planner import PlanningError, plan_trip
from app.agent.prompts import (
    FINAL_RESPONSE_AGENT_SYSTEM,
    FLIGHT_AGENT_SYSTEM,
    HOTEL_AGENT_SYSTEM,
    ITINERARY_AGENT_SYSTEM,
)
from app.core.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(model="gpt-4o-mini", max_itinerary_retries=2)


def happy_path_llm(trip_request) -> FakeLLM:
    """Routes by system prompt so it doesn't care which of flight/hotel — run
    in parallel by the graph — happens to call first."""
    return FakeLLM(
        by_system={
            FLIGHT_AGENT_SYSTEM: ["Meridian Air — cheapest, no stops."],
            HOTEL_AGENT_SYSTEM: ["The Ardent House — best rated."],
            ITINERARY_AGENT_SYSTEM: [draft_itinerary_json(trip_request)],
            FINAL_RESPONSE_AGENT_SYSTEM: ["Four easy, food-forward days in Lisbon."],
        }
    )


def test_plan_trip_runs_the_full_agent_graph(provider, trip_request, settings):
    llm = happy_path_llm(trip_request)

    trip = plan_trip(trip_request, llm=llm, provider=provider, settings=settings)

    assert trip.itinerary.destination == "Lisbon, Portugal"
    assert len(trip.itinerary.days) == trip_request.nights + 1
    assert trip.summary == "Four easy, food-forward days in Lisbon."
    # Flights and lodging make it all the way to the final itinerary now,
    # not just into the prompt the itinerary agent saw.
    assert trip.itinerary.outbound_flight is not None
    assert trip.itinerary.return_flight is not None
    assert len(trip.itinerary.lodging_options) > 0
    assert trip.itinerary.total_estimated_cost > 0
    assert any("flight_agent" in m for m in trip.agent_trace)
    assert any("hotel_agent" in m for m in trip.agent_trace)
    assert any("itinerary_agent" in m for m in trip.agent_trace)
    assert any("final_response_agent" in m for m in trip.agent_trace)
    assert trip.id


def test_plan_trip_resolves_a_destination_when_none_is_given(provider, settings):
    from datetime import date

    from app.models.itinerary import TripRequest

    request = TripRequest(
        start_date=date(2026, 6, 1),
        end_date=date(2026, 6, 4),
        travelers=1,
        interests=["food", "history"],
    )
    llm = happy_path_llm(request.model_copy(update={"destination": "Lisbon, Portugal"}))

    trip = plan_trip(request, llm=llm, provider=provider, settings=settings)

    assert trip.itinerary is not None
    assert any("coordinator:" in m for m in trip.agent_trace)


def test_plan_trip_raises_when_the_itinerary_agent_never_recovers(
    provider, trip_request, settings
):
    llm = FakeLLM(
        by_system={
            FLIGHT_AGENT_SYSTEM: ["Meridian Air."],
            HOTEL_AGENT_SYSTEM: ["The Ardent House."],
            ITINERARY_AGENT_SYSTEM: ["nope", "still nope", "nope again"],
        }
    )

    with pytest.raises(PlanningError, match="gave up"):
        plan_trip(trip_request, llm=llm, provider=provider, settings=settings)


def test_plan_trip_works_without_an_origin(provider, trip_request, settings):
    request = trip_request.model_copy(update={"origin": None})
    llm = happy_path_llm(request)

    trip = plan_trip(request, llm=llm, provider=provider, settings=settings)

    assert trip.itinerary is not None
    assert trip.itinerary.outbound_flight is None
    assert trip.itinerary.return_flight is None
    assert any("skipped" in m for m in trip.agent_trace)
