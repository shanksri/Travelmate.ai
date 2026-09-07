import json

import pytest
from conftest import draft_itinerary_json

from app.agent.itinerary_json import ItineraryValidationError, parse_itinerary


def test_parses_a_valid_draft(trip_request):
    draft = parse_itinerary(draft_itinerary_json(trip_request), trip_request)

    assert len(draft.days) == trip_request.nights + 1
    assert draft.notes == ["Book the castle ahead."]


def test_rejects_malformed_json(trip_request):
    with pytest.raises(ItineraryValidationError, match="not valid JSON"):
        parse_itinerary("{ not json", trip_request)


def test_rejects_a_missing_required_field(trip_request):
    payload = json.loads(draft_itinerary_json(trip_request))
    del payload["days"]

    with pytest.raises(ItineraryValidationError, match="days"):
        parse_itinerary(json.dumps(payload), trip_request)


def test_rejects_a_day_count_that_does_not_match_the_dates(trip_request):
    payload = json.loads(draft_itinerary_json(trip_request))
    payload["days"] = payload["days"][:2]

    with pytest.raises(ItineraryValidationError, match="entries"):
        parse_itinerary(json.dumps(payload), trip_request)
