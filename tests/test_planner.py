import pytest
from conftest import FakeClient, submit_call, text_block, tool_use_block

from app.agent.planner import PlanningError, plan_trip
from app.core.config import Settings


@pytest.fixture
def settings() -> Settings:
    return Settings(model="claude-opus-5", max_tokens=4096, max_tool_iterations=8)


def test_plan_trip_returns_the_submitted_itinerary(provider, trip_request, settings):
    lookup = tool_use_block(
        "search_attractions",
        {"destination": "Lisbon, Portugal", "interests": "food", "limit": 4},
    )
    client = FakeClient(
        [
            [
                [lookup],
                [submit_call(trip_request)],
                [text_block("Four days in Lisbon, food-forward and easy-paced.")],
            ]
        ]
    )

    trip = plan_trip(trip_request, client=client, provider=provider, settings=settings)

    assert trip.itinerary.destination == "Lisbon, Portugal"
    assert len(trip.itinerary.days) == trip_request.nights + 1
    assert trip.tool_calls == ["search_attractions", "submit_itinerary"]
    assert trip.summary.startswith("Four days")
    assert trip.id


def test_planner_passes_thinking_and_effort_to_the_api(provider, trip_request, settings):
    client = FakeClient([[[submit_call(trip_request)], [text_block("done")]]])

    plan_trip(trip_request, client=client, provider=provider, settings=settings)

    call = client.calls[0]
    assert call["model"] == "claude-opus-5"
    assert call["thinking"] == {"type": "adaptive"}
    assert call["output_config"] == {"effort": settings.effort}
    assert call["max_iterations"] == settings.max_tool_iterations


def test_planner_nudges_once_when_the_agent_forgets_to_submit(
    provider, trip_request, settings
):
    client = FakeClient(
        [
            [[text_block("Here is a lovely plan, in prose.")]],
            [[submit_call(trip_request)], [text_block("Submitted.")]],
        ]
    )

    trip = plan_trip(trip_request, client=client, provider=provider, settings=settings)

    assert trip.itinerary is not None
    assert len(client.calls) == 2
    # The nudge is carried in the replayed history, not a fresh conversation.
    assert "submit_itinerary" in client.calls[1]["messages"][-1]["content"]


def test_planner_raises_when_no_itinerary_is_ever_submitted(
    provider, trip_request, settings
):
    client = FakeClient([[[text_block("thinking about it")]], [[text_block("still no")]]])

    with pytest.raises(PlanningError, match="never submitted"):
        plan_trip(trip_request, client=client, provider=provider, settings=settings)


def test_a_rejected_itinerary_can_be_resubmitted(provider, trip_request, settings):
    """A malformed submission is an error the agent sees, not a crash."""
    bad = submit_call(trip_request)
    bad.input = {"itinerary_json": "{ not json"}

    client = FakeClient([[[bad], [submit_call(trip_request)], [text_block("Fixed.")]]])

    trip = plan_trip(trip_request, client=client, provider=provider, settings=settings)

    assert trip.itinerary is not None
    assert trip.tool_calls.count("submit_itinerary") == 2
