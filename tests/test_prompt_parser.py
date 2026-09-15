import json
from datetime import date

import pytest
from conftest import FakeLLM

from app.agent.prompt_parser import PromptParseError, parse_trip_prompt

TODAY = date(2026, 9, 15)


def _payload(**overrides) -> str:
    base = {
        "destination": None,
        "origin": None,
        "start_date": None,
        "end_date": None,
        "duration_days": None,
        "travelers": 1,
        "budget_usd": None,
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
                budget_usd=2400,
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
    assert request.budget_usd == 2400
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
