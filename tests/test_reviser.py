"""A revision re-runs only the itinerary step, so these tests drive
`revise_trip` with a scripted FakeLLM and assert on what carries over from the
previous version versus what the model is allowed to change."""

import json

import pytest
from conftest import FakeLLM, draft_itinerary_payload, sample_planned_trip

from app.agent.prompts import FINAL_RESPONSE_AGENT_SYSTEM, REVISE_ITINERARY_SYSTEM
from app.agent.reviser import DECLINE_PREFIX, RevisionDeclined, RevisionError, revise_trip
from app.core.config import Settings
from app.models.itinerary import DayPlan, FlightLeg, LodgingOption


@pytest.fixture
def settings() -> Settings:
    return Settings(model="gpt-4o-mini", max_itinerary_retries=2)


@pytest.fixture
def planned(trip_request):
    """A version-1 trip with a real flight, hotel and day plan — the things a
    revision must carry over untouched."""
    days = [DayPlan(**day) for day in draft_itinerary_payload(trip_request)["days"]]
    trip = sample_planned_trip(trip_request, trip_id="v1", days=days)
    trip.itinerary.outbound_flight = FlightLeg(
        carrier="Meridian Air",
        origin="BOS",
        destination="LIS",
        depart_date=trip_request.start_date,
        total=45_000.0,
    )
    trip.itinerary.return_flight = FlightLeg(
        carrier="Meridian Air",
        origin="LIS",
        destination="BOS",
        depart_date=trip_request.end_date,
        total=38_000.0,
    )
    trip.itinerary.lodging_options = [
        LodgingOption(name="Casa Vista Hostel", nightly=8_000.0, total=24_000.0)
    ]
    return trip


def changed_payload(trip_request, tag: str = "revised") -> dict:
    """A draft whose days differ from the `planned` fixture's. One that comes
    back identical is refused as a declined change."""
    payload = draft_itinerary_payload(trip_request)
    payload["days"][0]["summary"] = f"Day 1 ({tag})"
    return payload


def revising_llm(trip_request, tag: str = "revised", **overrides) -> FakeLLM:
    payload = changed_payload(trip_request, tag) | overrides
    return FakeLLM(
        by_system={
            REVISE_ITINERARY_SYSTEM: [json.dumps(payload)],
            FINAL_RESPONSE_AGENT_SYSTEM: ["Now with a fuller day three."],
        }
    )


def test_revision_becomes_the_next_version_of_the_same_thread(planned, trip_request, settings):
    revised = revise_trip(
        planned, "more activities on day 3", llm=revising_llm(trip_request), settings=settings
    )

    assert revised.thread_id == planned.thread_id
    assert revised.version == planned.version + 1
    assert revised.id != planned.id


def test_revision_records_what_was_asked_for(planned, trip_request, settings):
    revised = revise_trip(
        planned, "more activities on day 3", llm=revising_llm(trip_request), settings=settings
    )

    assert revised.change_note == "more activities on day 3"


def test_the_original_version_is_not_mutated(planned, trip_request, settings):
    original_days = len(planned.itinerary.days)

    revise_trip(planned, "swap day 2", llm=revising_llm(trip_request), settings=settings)

    assert planned.version == 1
    assert planned.change_note is None
    assert len(planned.itinerary.days) == original_days


def test_flights_and_lodging_carry_over_untouched(planned, trip_request, settings):
    revised = revise_trip(
        planned, "more activities on day 3", llm=revising_llm(trip_request), settings=settings
    )

    assert revised.itinerary.outbound_flight == planned.itinerary.outbound_flight
    assert revised.itinerary.return_flight == planned.itinerary.return_flight
    assert revised.itinerary.lodging_options == planned.itinerary.lodging_options


def test_the_model_is_never_shown_the_flights_or_hotel(planned, trip_request, settings):
    """It cannot change what it cannot see — the revise prompt deliberately
    carries only the days."""
    llm = revising_llm(trip_request)

    revise_trip(planned, "cheaper hotel please", llm=llm, settings=settings)

    revise_call = next(c for c in llm.calls if c["system"] == REVISE_ITINERARY_SYSTEM)
    assert "Meridian Air" not in revise_call["user"]
    assert "Casa Vista Hostel" not in revise_call["user"]


def test_cost_is_recomputed_from_the_revised_activities(planned, trip_request, settings):
    payload = draft_itinerary_payload(trip_request)
    for day in payload["days"]:
        day["activities"][0]["estimated_cost"] = 5_000
    llm = FakeLLM(
        by_system={
            REVISE_ITINERARY_SYSTEM: [json.dumps(payload)],
            FINAL_RESPONSE_AGENT_SYSTEM: ["Summary."],
        }
    )

    revised = revise_trip(planned, "pricier activities", llm=llm, settings=settings)

    # 45,000 + 38,000 flights, 24,000 hotel, then 5,000 per day of activities.
    expected = 45_000 + 38_000 + 24_000 + 5_000 * len(revised.itinerary.days)
    assert revised.itinerary.total_estimated_cost == expected


def test_the_summary_is_rewritten_for_the_new_version(planned, trip_request, settings):
    revised = revise_trip(
        planned, "more activities on day 3", llm=revising_llm(trip_request), settings=settings
    )

    assert revised.summary == "Now with a fuller day three."


def test_an_invalid_response_is_retried_with_feedback(planned, trip_request, settings):
    llm = FakeLLM(
        by_system={
            REVISE_ITINERARY_SYSTEM: [
                "not json at all",
                json.dumps(changed_payload(trip_request)),
            ],
            FINAL_RESPONSE_AGENT_SYSTEM: ["Summary."],
        }
    )

    revised = revise_trip(planned, "more on day 3", llm=llm, settings=settings)

    assert revised.version == 2
    retry = [c for c in llm.calls if c["system"] == REVISE_ITINERARY_SYSTEM][1]
    assert "was rejected" in retry["user"]


def test_giving_up_raises_rather_than_saving_a_broken_version(planned, trip_request, settings):
    llm = FakeLLM(
        by_system={REVISE_ITINERARY_SYSTEM: ["nope", "still nope", "nope again"]}
    )

    with pytest.raises(RevisionError, match="gave up after 3 attempt"):
        revise_trip(planned, "more on day 3", llm=llm, settings=settings)


def test_earlier_notes_are_not_shown_to_the_model(planned, trip_request, settings):
    """An earlier refusal note kept the model refusing the same place."""
    planned.itinerary.notes = ["Rameshwaram is not included as it needs travel changes."]
    llm = revising_llm(trip_request)

    revise_trip(planned, "add rameshwaram", llm=llm, settings=settings)

    revise_call = next(c for c in llm.calls if c["system"] == REVISE_ITINERARY_SYSTEM)
    assert "is not included" not in revise_call["user"]


def test_a_revision_can_itself_be_revised(planned, trip_request, settings):
    second = revise_trip(
        planned, "more on day 3", llm=revising_llm(trip_request), settings=settings
    )
    third = revise_trip(
        second, "now day 4", llm=revising_llm(trip_request, tag="again"), settings=settings
    )

    assert [t.version for t in (planned, second, third)] == [1, 2, 3]
    assert third.thread_id == planned.thread_id
    assert third.change_note == "now day 4"


# --- a change the model didn't make ------------------------------------------


def unchanged_llm(trip_request, notes: list[str]) -> FakeLLM:
    """Returns the `planned` fixture's days exactly as they were."""
    payload = draft_itinerary_payload(trip_request) | {"notes": notes}
    return FakeLLM(by_system={REVISE_ITINERARY_SYSTEM: [json.dumps(payload)]})


def test_unchanged_days_are_refused_with_the_models_reason(planned, trip_request, settings):
    llm = unchanged_llm(
        trip_request,
        ["Book the castle ahead.", "Couldn't apply: Rameshwaram is too far for one day."],
    )

    with pytest.raises(RevisionDeclined) as declined:
        revise_trip(planned, "add rameshwaram", llm=llm, settings=settings)

    assert str(declined.value) == "Rameshwaram is too far for one day."


def test_a_declined_change_skips_the_summary_and_isnt_retried(planned, trip_request, settings):
    llm = unchanged_llm(trip_request, ["Couldn't apply: too far."])

    with pytest.raises(RevisionDeclined):
        revise_trip(planned, "add rameshwaram", llm=llm, settings=settings)

    assert [c["system"] for c in llm.calls] == [REVISE_ITINERARY_SYSTEM]


def test_a_newly_added_note_is_the_reason_when_the_prefix_is_missing(
    planned, trip_request, settings
):
    """The Rameshwaram case as it actually happened: the model declined in a
    plain note, alongside the note the trip already had."""
    planned.itinerary.notes = ["Book the castle ahead."]
    llm = unchanged_llm(
        trip_request,
        ["Book the castle ahead.", "Rameshwaram is not included as it needs travel changes."],
    )

    with pytest.raises(RevisionDeclined, match="Rameshwaram is not included"):
        revise_trip(planned, "add rameshwaram", llm=llm, settings=settings)


def test_a_silent_decline_still_gets_a_reason(planned, trip_request, settings):
    planned.itinerary.notes = ["Book the castle ahead."]
    llm = unchanged_llm(trip_request, ["Book the castle ahead."])

    with pytest.raises(RevisionDeclined, match="returned the plan unchanged"):
        revise_trip(planned, "add rameshwaram", llm=llm, settings=settings)


def test_the_prompt_asks_for_the_prefix_the_code_looks_for():
    assert f'"{DECLINE_PREFIX}"' in REVISE_ITINERARY_SYSTEM


def test_the_prompt_allows_reshaping_days_for_a_new_place():
    assert "is NOT a reason to decline" in REVISE_ITINERARY_SYSTEM
    assert "adding a place" in REVISE_ITINERARY_SYSTEM


def test_revisions_use_their_own_model(monkeypatch):
    """Planning stays on `model`; only revisions use `revise_model`."""
    from types import SimpleNamespace

    from app.agent import reviser

    sent = {}

    def create(**kwargs):
        sent.update(kwargs)
        message = SimpleNamespace(content="{}")
        return SimpleNamespace(choices=[SimpleNamespace(message=message)])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(reviser.openai, "OpenAI", lambda: client)

    llm = reviser._build_llm(Settings(model="gpt-4o-mini", revise_model="gpt-4o"))
    llm.complete(system="s", user="u")

    assert sent["model"] == "gpt-4o"


def test_the_prompt_keeps_the_trips_start_and_end_fixed():
    """Without it the model agreed to "3 days in Ladakh" by ending a Kochi
    trip in Leh, two days short."""
    assert "still starts and ends where it does now" in REVISE_ITINERARY_SYSTEM
    assert "never quietly give fewer" in REVISE_ITINERARY_SYSTEM
