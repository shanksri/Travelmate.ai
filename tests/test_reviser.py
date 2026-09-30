"""A revision routes the change, then re-picks flights or the hotel from the
trip's own options and/or re-plans the days. These tests drive `revise_trip`
with a scripted FakeLLM and assert on what carries over from the previous
version versus what the change is allowed to touch."""

import json

import pytest
from conftest import FakeLLM, draft_itinerary_payload, sample_planned_trip

from app.agent.prompts import (
    FINAL_RESPONSE_AGENT_SYSTEM,
    REVISE_ITINERARY_SYSTEM,
    ROUTE_CHANGE_SYSTEM,
)
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


def route(**fields) -> str:
    """A ROUTE_CHANGE_SYSTEM answer; anything not given is null."""
    answer = {"outbound": None, "return": None, "hotel": None, "itinerary": None,
              "declined": None, "reasoning": "test routing"}
    return json.dumps(answer | fields)


def route_to_days(instruction: str = "the change") -> str:
    return route(itinerary=instruction)


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
            ROUTE_CHANGE_SYSTEM: [route_to_days()],
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
            ROUTE_CHANGE_SYSTEM: [route_to_days()],
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
            ROUTE_CHANGE_SYSTEM: [route_to_days()],
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
        by_system={
            ROUTE_CHANGE_SYSTEM: [route_to_days()],
            REVISE_ITINERARY_SYSTEM: ["nope", "still nope", "nope again"],
        }
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
    return FakeLLM(
        by_system={
            ROUTE_CHANGE_SYSTEM: [route_to_days()],
            REVISE_ITINERARY_SYSTEM: [json.dumps(payload)],
        }
    )


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

    assert [c["system"] for c in llm.calls] == [ROUTE_CHANGE_SYSTEM, REVISE_ITINERARY_SYSTEM]


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


# --- changes routed to flights and the hotel ---------------------------------


def _leg(carrier, origin, destination, on, *, stops, hours, total) -> FlightLeg:
    return FlightLeg(
        carrier=carrier,
        origin=origin,
        destination=destination,
        depart_date=on,
        stops=stops,
        duration_hours=hours,
        total=total,
    )


@pytest.fixture
def with_options(planned, trip_request):
    """`planned`, plus real option lists: the cheapest of each is booked and
    the first hotel is selected, as the planner leaves them."""
    it = planned.itinerary
    it.outbound_options = [
        _leg("Budget Air", "BOS", "LIS", trip_request.start_date, stops=2, hours=14, total=30_000),
        _leg("Direct Jet", "BOS", "LIS", trip_request.start_date, stops=0, hours=7, total=52_000),
    ]
    it.return_options = [
        _leg("Budget Air", "LIS", "BOS", trip_request.end_date, stops=1, hours=11, total=28_000),
        _leg("Direct Jet", "LIS", "BOS", trip_request.end_date, stops=0, hours=8, total=47_000),
    ]
    it.outbound_flight = it.outbound_options[0].model_copy(update={"rationale": "cheapest"})
    it.return_flight = it.return_options[0]
    it.lodging_options = [
        LodgingOption(name="Casa Vista Hostel", nightly=8_000.0, total=24_000.0),
        LodgingOption(name="Palacio Grande", nightly=20_000.0, total=60_000.0),
    ]
    return planned


def test_a_flight_change_rebooks_from_the_existing_options(with_options, settings):
    llm = FakeLLM(
        by_system={
            ROUTE_CHANGE_SYSTEM: [route(outbound=1, **{"return": 1})],
            FINAL_RESPONSE_AGENT_SYSTEM: ["Nonstop both ways."],
        }
    )

    revised = revise_trip(with_options, "nonstop flights please", llm=llm, settings=settings)

    it = revised.itinerary
    assert it.outbound_flight.carrier == "Direct Jet" and it.outbound_flight.stops == 0
    assert it.return_flight.carrier == "Direct Jet"
    assert "nonstop flights please" in it.outbound_flight.rationale
    # The option lists themselves are untouched; only which one is booked moved.
    assert it.outbound_options == with_options.itinerary.outbound_options


def test_a_flight_only_change_does_not_replan_the_days(with_options, settings):
    llm = FakeLLM(
        by_system={
            ROUTE_CHANGE_SYSTEM: [route(outbound=1)],
            FINAL_RESPONSE_AGENT_SYSTEM: ["Summary."],
        }
    )

    revised = revise_trip(with_options, "nonstop outbound", llm=llm, settings=settings)

    assert [c["system"] for c in llm.calls] == [ROUTE_CHANGE_SYSTEM, FINAL_RESPONSE_AGENT_SYSTEM]
    assert revised.itinerary.days == with_options.itinerary.days


def test_a_rebooked_flight_is_reflected_in_the_total(with_options, settings):
    llm = FakeLLM(
        by_system={
            ROUTE_CHANGE_SYSTEM: [route(outbound=1)],
            FINAL_RESPONSE_AGENT_SYSTEM: ["Summary."],
        }
    )

    revised = revise_trip(with_options, "nonstop outbound", llm=llm, settings=settings)

    activities = 2_000 * len(revised.itinerary.days)
    # 52,000 outbound (was 30,000) + 28,000 return + 24,000 hotel + activities.
    assert revised.itinerary.total_estimated_cost == 52_000 + 28_000 + 24_000 + activities


def test_a_hotel_change_moves_the_pick_to_the_front(with_options, settings):
    llm = FakeLLM(
        by_system={
            ROUTE_CHANGE_SYSTEM: [route(hotel=1)],
            FINAL_RESPONSE_AGENT_SYSTEM: ["Summary."],
        }
    )

    revised = revise_trip(with_options, "somewhere fancier", llm=llm, settings=settings)

    names = [h.name for h in revised.itinerary.lodging_options]
    assert names == ["Palacio Grande", "Casa Vista Hostel"]
    activities = 2_000 * len(revised.itinerary.days)
    assert revised.itinerary.total_estimated_cost == 30_000 + 28_000 + 60_000 + activities


def test_one_change_can_touch_flights_hotel_and_days_together(
    with_options, trip_request, settings
):
    llm = FakeLLM(
        by_system={
            ROUTE_CHANGE_SYSTEM: [route(outbound=1, hotel=1, itinerary="more on day 2")],
            REVISE_ITINERARY_SYSTEM: [json.dumps(changed_payload(trip_request))],
            FINAL_RESPONSE_AGENT_SYSTEM: ["Summary."],
        }
    )

    revised = revise_trip(
        with_options, "nonstop, nicer hotel, more on day 2", llm=llm, settings=settings
    )

    assert revised.itinerary.outbound_flight.carrier == "Direct Jet"
    assert revised.itinerary.lodging_options[0].name == "Palacio Grande"
    assert revised.itinerary.days[0].summary == "Day 1 (revised)"
    revise_call = next(c for c in llm.calls if c["system"] == REVISE_ITINERARY_SYSTEM)
    assert "The traveller asks: more on day 2" in revise_call["user"]


def test_the_router_sees_the_options_with_the_booked_ones_marked(with_options, settings):
    llm = FakeLLM(
        by_system={
            ROUTE_CHANGE_SYSTEM: [route(outbound=1)],
            FINAL_RESPONSE_AGENT_SYSTEM: ["Summary."],
        }
    )

    revise_trip(with_options, "nonstop outbound", llm=llm, settings=settings)

    sent = json.loads(llm.calls[0]["user"])
    assert [o["booked"] for o in sent["outbound_options"]] == [True, False]
    assert [h["selected"] for h in sent["hotel_options"]] == [True, False]
    assert sent["change_request"] == "nonstop outbound"


def test_a_change_the_router_declines_is_refused_with_its_reason(with_options, settings):
    llm = FakeLLM(
        by_system={ROUTE_CHANGE_SYSTEM: [route(declined="None of the flights is before 6am.")]}
    )

    with pytest.raises(RevisionDeclined, match="before 6am"):
        revise_trip(with_options, "a 5am flight", llm=llm, settings=settings)

    assert len(llm.calls) == 1


def test_picking_what_is_already_booked_is_refused(with_options, settings):
    llm = FakeLLM(
        by_system={
            ROUTE_CHANGE_SYSTEM: [
                route(outbound=0, hotel=0, reasoning="The cheapest is already booked.")
            ]
        }
    )

    with pytest.raises(RevisionDeclined, match="already booked"):
        revise_trip(with_options, "cheapest flight please", llm=llm, settings=settings)


def test_an_index_that_isnt_there_is_retried_with_feedback(with_options, settings):
    llm = FakeLLM(
        by_system={
            ROUTE_CHANGE_SYSTEM: [route(outbound=7), route(outbound=1)],
            FINAL_RESPONSE_AGENT_SYSTEM: ["Summary."],
        }
    )

    revised = revise_trip(with_options, "nonstop outbound", llm=llm, settings=settings)

    assert revised.itinerary.outbound_flight.carrier == "Direct Jet"
    assert "only 2 outbound option(s)" in llm.calls[1]["user"]


def test_a_router_that_never_names_a_real_option_gives_up(with_options, settings):
    llm = FakeLLM(by_system={ROUTE_CHANGE_SYSTEM: [route(hotel=9)] * 3})

    with pytest.raises(RevisionError, match="couldn't work out"):
        revise_trip(with_options, "different hotel", llm=llm, settings=settings)


def test_the_router_prompt_forbids_picking_an_option_that_doesnt_fit():
    assert "doesn't actually satisfy the request" in ROUTE_CHANGE_SYSTEM


def test_the_router_prompt_keeps_day_numbers_in_the_instruction():
    """Live, "add a houseboat trip on day 3" reached the day planner as "add a
    backwater houseboat trip", and it went on day 2."""
    assert '"on day 3"' in ROUTE_CHANGE_SYSTEM
    assert "never shortened" in ROUTE_CHANGE_SYSTEM


def test_the_router_prompt_declines_a_change_that_is_only_partly_possible():
    """Live, "a cheaper return flight (already the cheapest) and a houseboat on
    day 3" once applied only the houseboat and dropped the flight silently."""
    assert "not at all, so if any part can't be done, decline it all" in ROUTE_CHANGE_SYSTEM


def test_the_router_and_the_day_planner_send_their_schemas(planned, trip_request, settings):
    from app.agent.reviser import ChangeRoute
    from app.models.itinerary import DraftItinerary

    llm = revising_llm(trip_request)
    revise_trip(planned, "more on day 3", llm=llm, settings=settings)

    schemas = {c["system"]: c["schema"] for c in llm.calls}
    assert schemas[ROUTE_CHANGE_SYSTEM] is ChangeRoute
    assert schemas[REVISE_ITINERARY_SYSTEM] is DraftItinerary
    assert schemas[FINAL_RESPONSE_AGENT_SYSTEM] is None  # plain text


# --- trips planned without flights or a hotel ------------------------------


@pytest.fixture
def days_only(planned):
    """A trip planned with Flights and Hotels unticked: nothing to re-pick."""
    it = planned.itinerary
    it.outbound_flight = it.return_flight = None
    it.outbound_options, it.return_options, it.lodging_options = [], [], []
    return planned


def test_a_days_only_trip_skips_the_router(days_only, trip_request, settings):
    """Live, the router refused "adjust more days to tamil nadu" on such a
    trip 2 times in 5, because "the trip has no hotel"."""
    llm = FakeLLM(
        by_system={
            REVISE_ITINERARY_SYSTEM: [json.dumps(changed_payload(trip_request))],
            FINAL_RESPONSE_AGENT_SYSTEM: ["Summary."],
        }
    )

    revised = revise_trip(days_only, "adjust more days to tamil nadu", llm=llm, settings=settings)

    assert [c["system"] for c in llm.calls] == [
        REVISE_ITINERARY_SYSTEM,
        FINAL_RESPONSE_AGENT_SYSTEM,
    ]
    revise_call = llm.calls[0]
    assert "The traveller asks: adjust more days to tamil nadu" in revise_call["user"]
    assert revised.itinerary.days[0].summary == "Day 1 (revised)"
    assert revised.agent_trace[0].startswith("change_router: skipped")


def test_a_flight_request_on_a_days_only_trip_is_declined_by_the_day_planner(
    days_only, trip_request, settings
):
    llm = unchanged_llm(
        trip_request, ["Couldn't apply: this plan doesn't include flights to change."]
    )

    with pytest.raises(RevisionDeclined, match="doesn't include flights"):
        revise_trip(days_only, "switch me to a nonstop flight", llm=llm, settings=settings)

    assert ROUTE_CHANGE_SYSTEM not in [c["system"] for c in llm.calls]


def test_a_trip_with_flights_but_no_hotel_still_routes(with_options, settings):
    with_options.itinerary.lodging_options = []
    llm = FakeLLM(
        by_system={
            ROUTE_CHANGE_SYSTEM: [route(outbound=1)],
            FINAL_RESPONSE_AGENT_SYSTEM: ["Summary."],
        }
    )

    revise_trip(with_options, "nonstop outbound", llm=llm, settings=settings)

    sent = json.loads(llm.calls[0]["user"])
    # Worded as the traveller's choice: "none — this trip has no hotel" read
    # to the router as a reason to refuse changes to the days.
    assert sent["hotel_options"] == "none: the traveller planned this trip without a hotel"


def test_the_router_prompt_says_a_missing_hotel_never_blocks_a_day_change():
    flat = " ".join(ROUTE_CHANGE_SYSTEM.split())  # the prompt wraps mid-sentence
    assert "their absence is never a reason to decline it" in flat
    assert "only when the request itself asks to change one" in flat
