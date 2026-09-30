import json
from datetime import date

import pytest
from conftest import FakeLLM

from app.agent.prompt_parser import PromptParseError, interpret_prompt, parse_trip_prompt

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


def test_the_itinerary_prompt_shares_days_across_named_regions():
    """Live, "10 day itinerary for tamil nadu and kerala" gave Tamil Nadu 2 of
    10 days (Kanyakumari only) and spent 2 days going back to Kochi."""
    from app.agent.prompts import ITINERARY_AGENT_SYSTEM

    assert "Several states, regions or countries named together" in ITINERARY_AGENT_SYSTEM
    assert "fair share of the days" in ITINERARY_AGENT_SYSTEM
    assert "Don't spend a day travelling back to the first city" in ITINERARY_AGENT_SYSTEM
