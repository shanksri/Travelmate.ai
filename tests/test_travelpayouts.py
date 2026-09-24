"""Fetch tests mock httpx entirely (see the autouse block in conftest.py) —
no real call, no quota spent. Normalization tests run against a sample
shaped exactly like a real /aviasales/v3/prices_for_dates response captured
during development."""

from datetime import date, timedelta

import pytest
from conftest import FakeHTTPResponse

from app.models.itinerary import FlightLeg
from app.providers import travelpayouts
from app.providers.travelpayouts import (
    TravelPayoutsError,
    fetch_prices,
    normalize_prices,
    resolve_iata,
)


def _row(departure_at, price, *, airline="IX", number=1165, transfers=1, duration=215):
    return {
        "origin": "DEL",
        "destination": "BOM",
        "origin_airport": "DEL",
        "destination_airport": "BOM",
        "price": price,
        "airline": airline,
        "flight_number": str(number),
        "departure_at": departure_at,
        "transfers": transfers,
        "return_transfers": 0,
        "duration": duration,
        "duration_to": duration,
        "duration_back": 0,
        "link": "/search/DEL1010BOM1",
    }


SAMPLE_RESPONSE = {
    "success": True,
    "currency": "inr",
    "data": [_row("2026-10-10T06:00:00+05:30", 6899)],
}

SAMPLE_CITIES = [
    {"name": "New Delhi", "code": "DEL", "country_code": "IN", "has_flightable_airport": True},
    {"name": "Mumbai", "code": "BOM", "country_code": "IN", "has_flightable_airport": True},
    {"name": "Delhi", "code": "XXX", "country_code": "US", "has_flightable_airport": False},
    {"name": "Lisbon", "code": "LIS", "country_code": "PT", "has_flightable_airport": True},
]


@pytest.fixture(autouse=True)
def _fake_city_directory(monkeypatch):
    """resolve_iata is lru_cached, so the cache has to be cleared around each
    test or the first one's data leaks into the rest."""
    resolve_iata.cache_clear()
    monkeypatch.setattr(travelpayouts, "_cities", lambda *a, **k: SAMPLE_CITIES)
    yield
    resolve_iata.cache_clear()


# --- resolve_iata ----------------------------------------------------------


def test_resolves_an_exact_city_name():
    assert resolve_iata("Mumbai") == "BOM"


def test_ignores_anything_after_a_comma():
    assert resolve_iata("Lisbon, Portugal") == "LIS"


def test_prefers_a_flightable_airport_over_an_exact_name_match():
    """"Delhi" matches a non-flightable city exactly and New Delhi partially.
    Searching from a city with no flightable airport returns nothing, so the
    flightable one wins."""
    assert resolve_iata("Delhi") == "DEL"


def test_unknown_place_resolves_to_none():
    assert resolve_iata("Atlantis") is None


# --- fetch_prices ----------------------------------------------------------


def test_fetch_prices_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("TRAVELPAYOUTS_API_KEY", raising=False)

    with pytest.raises(TravelPayoutsError, match="not set"):
        fetch_prices(origin="DEL", destination="BOM", departure_at="2026-10-10")


def test_fetch_prices_sends_the_exact_date(monkeypatch):
    """The bug this endpoint replaced: the old one ignored travel dates
    entirely and returned the year's cheapest fares, on any day."""
    captured = {}

    def fake_get(url, params=None, **kwargs):
        captured.update(params or {})
        return FakeHTTPResponse(SAMPLE_RESPONSE)

    monkeypatch.setattr("httpx.get", fake_get)

    fetch_prices(origin="DEL", destination="BOM", departure_at="2026-10-10", api_key="k")

    assert captured["departure_at"] == "2026-10-10"
    assert captured["one_way"] == "true"
    assert captured["currency"] == "inr"


def test_a_401_explains_the_two_likely_causes(monkeypatch):
    """Sending no token returns 401 too, so the message has to cover both the
    wrong-token-type and program-not-connected cases."""
    monkeypatch.setattr(
        "httpx.get", lambda *a, **k: FakeHTTPResponse({"detail": "nope"}, status_code=401)
    )

    with pytest.raises(TravelPayoutsError, match="32-character hex"):
        fetch_prices(origin="DEL", destination="BOM", departure_at="2026-10-10", api_key="k")


def test_a_400_surfaces_the_apis_own_message(monkeypatch):
    body = {"error": "bad request: unknown location code `ZZZ`", "data": None, "success": False}
    monkeypatch.setattr("httpx.get", lambda *a, **k: FakeHTTPResponse(body, status_code=400))

    with pytest.raises(TravelPayoutsError, match="unknown location code"):
        fetch_prices(origin="ZZZ", destination="BOM", departure_at="2026-10-10", api_key="k")


def test_an_unsuccessful_200_still_raises(monkeypatch):
    body = {"success": False, "error": "something broke", "data": None}
    monkeypatch.setattr("httpx.get", lambda *a, **k: FakeHTTPResponse(body))

    with pytest.raises(TravelPayoutsError, match="something broke"):
        fetch_prices(origin="DEL", destination="BOM", departure_at="2026-10-10", api_key="k")


# --- normalize_prices ------------------------------------------------------


def test_normalize_keeps_the_real_airline_and_departure_time():
    (option,) = normalize_prices(SAMPLE_RESPONSE)

    assert option["carrier"] == "IX"
    assert option["departure_at"] == "2026-10-10T06:00:00+05:30"
    assert option["depart_date"] == "2026-10-10"


def test_normalize_converts_minutes_to_hours():
    (option,) = normalize_prices(SAMPLE_RESPONSE)

    assert option["duration_hours"] == 3.6  # 215 minutes
    assert option["stops"] == 1


def test_normalize_keeps_a_row_with_no_duration():
    """Still a valid cheapest option; only the fastest ranking excludes it."""
    raw = {"data": [_row("2026-10-10T06:00:00+05:30", 6899, duration=0)]}

    (option,) = normalize_prices(raw)

    assert option["duration_hours"] is None


def test_normalize_scales_total_by_party_size():
    (option,) = normalize_prices(SAMPLE_RESPONSE, travelers=3)

    assert option["price_per_person"] == 6899.0
    assert option["total"] == 20697.0


def test_normalize_output_builds_a_valid_FlightLeg():
    """The whole point of this shape: it drops straight into the model the
    rest of the pipeline already uses."""
    leg = FlightLeg(**normalize_prices(SAMPLE_RESPONSE)[0])

    assert leg.carrier == "IX"
    assert leg.departure_at.hour == 6
    assert leg.total == 6899.0


def test_normalize_on_no_data_returns_an_empty_list():
    assert normalize_prices({"data": None}) == []


# --- search_flights --------------------------------------------------------


def _by_date(responses: dict[str, dict]):
    """A fake httpx.get answering each departure_at from `responses`."""
    asked = []

    def fake_get(url, params=None, **kwargs):
        asked.append(params["departure_at"])
        return FakeHTTPResponse(responses.get(params["departure_at"], {"data": []}))

    return fake_get, asked


def test_search_flights_covers_the_day_either_side(monkeypatch):
    monkeypatch.setenv("TRAVELPAYOUTS_API_KEY", "k")
    trip_day = date.today() + timedelta(days=20)
    fake_get, asked = _by_date({})
    monkeypatch.setattr("httpx.get", fake_get)

    travelpayouts.search_flights("Delhi", "Mumbai", trip_day, 1)

    assert asked == [
        (trip_day - timedelta(days=1)).isoformat(),
        trip_day.isoformat(),
        (trip_day + timedelta(days=1)).isoformat(),
    ]


def test_search_flights_never_asks_about_a_day_already_past(monkeypatch):
    monkeypatch.setenv("TRAVELPAYOUTS_API_KEY", "k")
    fake_get, asked = _by_date({})
    monkeypatch.setattr("httpx.get", fake_get)

    travelpayouts.search_flights("Delhi", "Mumbai", date.today(), 1)

    assert (date.today() - timedelta(days=1)).isoformat() not in asked


def test_search_flights_merges_the_window_cheapest_first(monkeypatch):
    monkeypatch.setenv("TRAVELPAYOUTS_API_KEY", "k")
    trip_day = date.today() + timedelta(days=20)
    before, after = trip_day - timedelta(days=1), trip_day + timedelta(days=1)
    fake_get, _ = _by_date(
        {
            before.isoformat(): {"data": [_row(f"{before}T09:00:00+05:30", 7000)]},
            trip_day.isoformat(): {"data": [_row(f"{trip_day}T06:00:00+05:30", 6899)]},
            after.isoformat(): {"data": [_row(f"{after}T19:55:00+05:30", 6000, airline="SG")]},
        }
    )
    monkeypatch.setattr("httpx.get", fake_get)

    options = travelpayouts.search_flights("Delhi", "Mumbai", trip_day, 1)

    assert [o["price_per_person"] for o in options] == [6000.0, 6899.0, 7000.0]


def test_search_flights_returns_empty_when_a_place_cannot_be_resolved(monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: FakeHTTPResponse(SAMPLE_RESPONSE))

    assert travelpayouts.search_flights("Atlantis", "Mumbai", date.today(), 1) == []


def test_search_flights_returns_empty_for_a_same_city_route(monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: FakeHTTPResponse(SAMPLE_RESPONSE))

    assert travelpayouts.search_flights("Mumbai", "Mumbai", date.today(), 1) == []
