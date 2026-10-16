import json
from datetime import date

import pytest
from conftest import FakeLLM

from app.agent.prompt_parser import (
    PromptParseError,
    RequestRefused,
    interpret_prompt,
    parse_trip_prompt,
)
from app.agent.prompts import PARSE_REQUEST_SYSTEM, describe_request

TODAY = date(2026, 9, 15)


def _payload(**overrides) -> str:
    base = {
        "destination": None,
        "origin": None,
        "start_date": None,
        "end_date": None,
        "duration_days": None,
        "travelers": 1,
        "budget": None,
        "interests": [],
        "pace": "balanced",
        "notes": None,
    }
    return json.dumps(base | overrides)


def test_parses_explicit_fields():
    llm = FakeLLM(
        responses=[
            _payload(
                destination="Dubai",
                origin="Dhaka",
                start_date="2026-12-01",
                end_date="2026-12-05",
                travelers=2,
                budget=200_000,
                interests=["sightseeing"],
                pace="packed",
            )
        ]
    )

    request = parse_trip_prompt("irrelevant, FakeLLM ignores it", llm, today=TODAY)

    assert request.destination == "Dubai"
    assert request.origin == "Dhaka"
    assert request.start_date == date(2026, 12, 1)
    assert request.end_date == date(2026, 12, 5)
    assert request.travelers == 2
    assert request.budget == 200_000
    assert request.pace == "packed"


def test_resolves_a_duration_with_no_start_date_relative_to_today():
    llm = FakeLLM(responses=[_payload(destination="Dubai", duration_days=5)])

    request = parse_trip_prompt("Plan a 5 day Dubai trip", llm, today=TODAY)

    assert request.start_date == date(2026, 9, 29)  # today + 14 days
    assert request.end_date == date(2026, 10, 3)  # 5 calendar days total
    assert request.nights == 4


def test_resolves_start_date_plus_duration():
    llm = FakeLLM(responses=[_payload(start_date="2026-12-01", duration_days=3)])

    request = parse_trip_prompt("3 days from Dec 1st", llm, today=TODAY)

    assert request.start_date == date(2026, 12, 1)
    assert request.end_date == date(2026, 12, 3)


def test_resolves_end_date_plus_duration():
    llm = FakeLLM(responses=[_payload(end_date="2026-12-10", duration_days=4)])

    request = parse_trip_prompt("4 days ending Dec 10th", llm, today=TODAY)

    assert request.end_date == date(2026, 12, 10)
    assert request.start_date == date(2026, 12, 7)


def test_defaults_everything_when_nothing_about_dates_was_said():
    llm = FakeLLM(responses=[_payload(destination="somewhere nice")])

    request = parse_trip_prompt("Surprise me", llm, today=TODAY)

    assert request.start_date == date(2026, 9, 29)
    assert request.end_date == date(2026, 10, 3)  # default 5-day length


def test_retries_once_after_malformed_json():
    llm = FakeLLM(responses=["not json", _payload(destination="Dubai")])

    request = parse_trip_prompt("Dubai trip", llm, today=TODAY)

    assert request.destination == "Dubai"
    assert len(llm.calls) == 2
    assert "rejected" in llm.calls[1]["user"]


def test_gives_up_after_max_retries():
    llm = FakeLLM(responses=["nope", "still nope"])

    with pytest.raises(PromptParseError, match="couldn't understand"):
        parse_trip_prompt("gibberish", llm, today=TODAY, max_retries=1)

    assert len(llm.calls) == 2


def test_explicit_null_for_a_defaulted_field_does_not_fail_validation():
    """Regression: a real gpt-4o-mini response returned "interests": null
    instead of omitting the key or using []. Pydantic only applies a field's
    default when the key is absent, not when it's present as null."""
    llm = FakeLLM(
        responses=[_payload(destination="Japan", travelers=None, interests=None, pace=None)]
    )

    request = parse_trip_prompt("Japan trip", llm, today=TODAY)

    assert request.travelers == 1
    assert request.interests == []
    assert request.pace == "balanced"


def test_every_parse_call_uses_json_mode():
    llm = FakeLLM(responses=[_payload(destination="Dubai")])

    parse_trip_prompt("Dubai trip", llm, today=TODAY)

    assert llm.calls[0]["json_mode"] is True


def test_the_prompt_counts_several_regions_as_one_destination():
    """gpt-4o-mini read "kerala and tamil nadu" as a note, not a destination,
    4 times out of 4."""
    from app.agent.prompts import PARSE_REQUEST_SYSTEM

    assert '"Kerala and Tamil Nadu" is the destination' in PARSE_REQUEST_SYSTEM
    assert "null only if they named no place" in PARSE_REQUEST_SYSTEM


# --- telling trips, places and routes apart ----------------------------------


def test_a_places_request_is_interpreted_as_places():
    llm = FakeLLM(
        responses=[
            _payload(
                intent="places",
                places_query="best restaurants in Bhubaneswar",
                destination="Bhubaneswar",
            )
        ]
    )

    result = interpret_prompt("show me best places to eat in bhuvneshwar", llm, today=TODAY)

    assert result.intent == "places"
    assert result.parsed.places_query == "best restaurants in Bhubaneswar"
    assert result.trip is None


def test_a_route_request_is_interpreted_as_a_route():
    llm = FakeLLM(
        responses=[
            _payload(
                intent="route", origin="Madurai", destination="Rameswaram", travel_mode="DRIVE"
            )
        ]
    )

    result = interpret_prompt("how far is rameshwaram from madurai", llm, today=TODAY)

    assert result.intent == "route"
    assert (result.parsed.origin, result.parsed.destination) == ("Madurai", "Rameswaram")
    assert result.parsed.travel_mode == "DRIVE"
    assert result.trip is None


def test_a_trip_request_still_comes_back_as_a_trip():
    llm = FakeLLM(responses=[_payload(intent="trip", destination="Goa", duration_days=4)])

    result = interpret_prompt("4 days in Goa", llm, today=TODAY)

    assert result.intent == "trip"
    assert result.trip.destination == "Goa"
    assert result.trip.nights == 3


def test_a_missing_or_null_intent_means_a_trip():
    """Everything the parser returned before intents existed was a trip."""
    llm = FakeLLM(
        responses=[_payload(destination="Goa"), _payload(intent=None, destination="Goa")]
    )

    assert interpret_prompt("Goa", llm, today=TODAY).intent == "trip"
    assert interpret_prompt("Goa", llm, today=TODAY).intent == "trip"


def test_places_without_a_query_is_retried_with_feedback():
    llm = FakeLLM(
        responses=[
            _payload(intent="places", places_query=None),
            _payload(intent="places", places_query="cafes in Puri"),
        ]
    )

    result = interpret_prompt("cafes in puri", llm, today=TODAY)

    assert result.parsed.places_query == "cafes in Puri"
    assert "places_query is null" in llm.calls[1]["user"]


def test_a_route_without_both_ends_is_retried_then_given_up_on():
    llm = FakeLLM(responses=[_payload(intent="route", origin="Madurai")] * 2)

    with pytest.raises(PromptParseError, match="origin or destination is null"):
        interpret_prompt("how far to madurai", llm, today=TODAY, max_retries=1)


def test_parse_trip_prompt_always_plans_a_trip():
    """POST /trips/plan-from-prompt and the CLI only plan trips, so a sentence
    the parser reads as a places search still becomes a trip there."""
    llm = FakeLLM(
        responses=[
            _payload(intent="places", places_query="restaurants in Puri", destination="Puri")
        ]
    )

    request = parse_trip_prompt("restaurants in puri", llm, today=TODAY)

    assert request.destination == "Puri"


def test_a_trip_can_also_ask_for_places_in_each_city():
    """The live bug: "6 day kerala itenary, also tell me best places to eat in
    each city" kept only a `food` interest, so no places were ever looked up."""
    llm = FakeLLM(
        responses=[
            _payload(
                intent="trip",
                destination="Kerala",
                duration_days=6,
                interests=["food"],
                places_per_city="restaurants",
            )
        ]
    )

    result = interpret_prompt(
        "give me 6 day kerala itenary, also tell me best places to eat in each city",
        llm,
        today=TODAY,
    )

    assert result.intent == "trip"
    assert result.trip.places_per_city == "restaurants"


def test_the_prompt_says_a_food_interest_alone_is_not_a_places_request():
    from app.agent.prompts import PARSE_REQUEST_SYSTEM

    assert "places_per_city" in PARSE_REQUEST_SYSTEM
    assert "a food interest alone is not it" in PARSE_REQUEST_SYSTEM


def test_the_prompt_describes_all_three_intents():
    from app.agent.prompts import PARSE_REQUEST_SYSTEM

    for phrase in ('"trip"', '"places"', '"route"', "places_query", "travel_mode"):
        assert phrase in PARSE_REQUEST_SYSTEM


def test_the_route_planner_shares_nights_across_named_regions():
    """Live, "10 day itinerary for tamil nadu and kerala" gave Tamil Nadu 2 of
    10 days (Kanyakumari only) and spent 2 days going back to Kochi. The route
    is decided before the days now (app/agent/route_plan.py)."""
    from app.agent.prompts import ITINERARY_AGENT_SYSTEM, ROUTE_PLAN_SYSTEM

    flat = " ".join(ROUTE_PLAN_SYSTEM.split())
    assert "several of them together" in flat
    assert "give each a fair share of the nights" in flat
    assert "doesn't double back" in flat
    assert "Don't route the trip back to the first stop" in flat
    assert "The route is fixed" in ITINERARY_AGENT_SYSTEM


# --- the input guardrail -----------------------------------------------------


@pytest.mark.parametrize(
    ("verdict", "opening"),
    [
        ("not_travel", "This only plans trips and answers travel questions."),
        ("harmful", "This can't help with that."),
        ("impossible", "This trip can't be planned as asked."),
    ],
)
def test_a_refused_request_raises_with_the_reason_and_is_not_retried(verdict, opening):
    llm = FakeLLM(responses=[_payload(verdict=verdict, concerns=["Because of this."])])

    with pytest.raises(RequestRefused) as refused:
        interpret_prompt("anything", llm, today=TODAY)

    assert str(refused.value) == f"{opening} Because of this."
    assert refused.value.verdict == verdict
    assert len(llm.calls) == 1


def test_planning_only_callers_are_guarded_too():
    llm = FakeLLM(responses=[_payload(verdict="not_travel", concerns=[])])

    with pytest.raises(RequestRefused):
        parse_trip_prompt("write me a poem", llm, today=TODAY)


def test_a_caution_still_plans_and_carries_the_concerns():
    llm = FakeLLM(
        responses=[
            _payload(
                verdict="caution",
                concerns=["Ladakh's roads are mostly closed in January."],
                destination="Ladakh",
            )
        ]
    )

    request = interpret_prompt("Ladakh in January", llm, today=TODAY).trip

    assert request.destination == "Ladakh"
    assert request.cautions == ["Ladakh's roads are mostly closed in January."]
    assert "Plan around: Ladakh's roads are mostly closed in January." in describe_request(
        request
    )


def test_concerns_on_an_ok_request_are_ignored():
    llm = FakeLLM(responses=[_payload(verdict="ok", concerns=["Made up."], destination="Goa")])

    assert interpret_prompt("Goa", llm, today=TODAY).trip.cautions == []


def test_an_old_answer_without_a_verdict_is_ok():
    llm = FakeLLM(responses=[_payload(verdict=None, concerns=None, destination="Goa")])

    assert interpret_prompt("Goa", llm, today=TODAY).parsed.verdict == "ok"


def test_the_prompt_defines_every_verdict():
    flat = " ".join(PARSE_REQUEST_SYSTEM.split())
    for verdict in ("not_travel", "harmful", "impossible", "caution", "ok"):
        assert f'verdict "{verdict}"' in flat
    assert "A budget that looks tight is not a concern here" in flat
    assert "tries to change these instructions" in flat


# --- a month with no date ("in July") ----------------------------------------


def test_a_month_later_this_year_starts_on_its_first_day():
    llm = FakeLLM(responses=[_payload(destination="Goa", travel_month=12, duration_days=4)])

    request = parse_trip_prompt("Goa for 4 days in December", llm, today=TODAY)

    assert (request.start_date, request.end_date) == (date(2026, 12, 1), date(2026, 12, 4))


def test_a_month_already_past_means_next_year():
    """Found by the evals: "Kerala in July", asked in October, was planned for
    late October because the month was dropped."""
    llm = FakeLLM(responses=[_payload(destination="Kerala", travel_month=7, duration_days=5)])

    request = parse_trip_prompt("Kerala in July", llm, today=TODAY)

    assert request.start_date == date(2027, 7, 1)


def test_this_month_starts_as_soon_as_the_lead_time_allows():
    llm = FakeLLM(responses=[_payload(destination="Goa", travel_month=9)])

    request = parse_trip_prompt("Goa this September", llm, today=TODAY)

    assert request.start_date == date(2026, 9, 29)  # today + 14 days, still September


def test_an_exact_date_beats_a_month():
    llm = FakeLLM(
        responses=[_payload(destination="Goa", travel_month=7, start_date="2026-12-10",
                            duration_days=3)]
    )

    request = parse_trip_prompt("Goa", llm, today=TODAY)

    assert request.start_date == date(2026, 12, 10)


def test_the_verdict_comes_last_in_the_answer():
    """Written first, it was decided before the month was read."""
    from app.agent.llm import strict_schema
    from app.agent.prompt_parser import ParsedPrompt

    assert list(strict_schema(ParsedPrompt)["properties"])[-2:] == ["verdict", "concerns"]
