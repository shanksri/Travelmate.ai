"""The checkpointer behind the revision agent's memory (app/agent/memory.py).
The memory itself is exercised through `revise_trip` in test_reviser.py."""

import json
from typing import get_args, get_type_hints

import pytest
from conftest import FakeLLM, draft_itinerary_payload, sample_planned_trip
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import BaseModel

from app.agent.memory import (
    CHECKPOINTED_TYPES,
    _psycopg_url,
    close_checkpointer,
    get_checkpointer,
)
from app.agent.prompts import FINAL_RESPONSE_AGENT_SYSTEM, REVISE_ITINERARY_SYSTEM
from app.agent.reviser import RevisionState, remembered_changes, revise_trip
from app.core.config import Settings
from app.models.itinerary import DayPlan


def test_the_tests_keep_memory_in_process():
    assert isinstance(get_checkpointer(), InMemorySaver)


def test_one_checkpointer_is_shared_until_closed():
    first = get_checkpointer()
    assert get_checkpointer() is first

    close_checkpointer()

    assert get_checkpointer() is not first


def test_the_database_url_is_rewritten_for_psycopg():
    url = "postgresql+psycopg://travelmate:secret@db:5432/travelmate"

    assert _psycopg_url(url) == "postgresql://travelmate:secret@db:5432/travelmate"


def _models_in(annotation) -> set[type[BaseModel]]:
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return {annotation}
    return set().union(*(_models_in(arg) for arg in get_args(annotation)))


def test_every_type_the_revision_state_holds_may_be_loaded_back():
    """A type missing here is refused (in future: today, warned about) when
    a checkpoint is loaded, and the agent would lose its memory."""
    hints = get_type_hints(RevisionState, include_extras=True)
    models = set().union(*(_models_in(hint) for hint in hints.values()))

    assert {(m.__module__, m.__name__) for m in models} == set(CHECKPOINTED_TYPES)


def test_the_memory_comes_back_from_a_checkpoint_as_real_objects(planned_with_memory):
    """The in-memory checkpointer stores serialized bytes too, so reading the
    memory back goes through the same allowlist the Postgres one does. A type
    left off it would come back blocked, not as itself."""
    thread_id, checkpointer = planned_with_memory

    turns = remembered_changes(thread_id, checkpointer)

    assert [type(t).__name__ for t in turns] == ["EditTurn"]
    assert turns[0].request == "more on day 1"
    assert isinstance(turns[0].days_before[0], DayPlan)


@pytest.fixture
def planned_with_memory(trip_request):
    days = [DayPlan(**day) for day in draft_itinerary_payload(trip_request)["days"]]
    trip = sample_planned_trip(trip_request, trip_id="m1", days=days)
    changed = draft_itinerary_payload(trip_request)
    changed["days"][0]["summary"] = "Day 1 (revised)"
    llm = FakeLLM(
        by_system={
            REVISE_ITINERARY_SYSTEM: [json.dumps(changed)],
            FINAL_RESPONSE_AGENT_SYSTEM: ["Summary."],
        }
    )
    checkpointer = get_checkpointer()
    revise_trip(trip, "more on day 1", llm=llm, settings=Settings(), checkpointer=checkpointer)
    return trip.thread_id, checkpointer
