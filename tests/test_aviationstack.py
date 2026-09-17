"""Fetch tests mock httpx entirely (see the autouse block in conftest.py) —
no real call, no API quota spent. Normalization tests need no network at all;
they run a saved sample shaped exactly like the real response fetched during
development."""

import pytest
from conftest import FakeHTTPResponse

from app.providers.aviationstack import (
    AviationStackError,
    fetch_flights,
    fetch_flights_future,
    normalize_flights,
    normalize_flights_future,
)

# One real record's shape from `/v1/flightsFuture`, captured from a live
# fetch during development.
FUTURE_SAMPLE_RESPONSE = {
    "data": [
        {
            "weekday": "4",
            "departure": {
                "iataCode": "jfk",
                "icaoCode": "kjfk",
                "terminal": "8",
                "gate": "35",
                "scheduledTime": "2026-10-01 20:00:00",
            },
            "arrival": {
                "iataCode": "lax",
                "icaoCode": "klax",
                "terminal": "4",
                "gate": "",
                "scheduledTime": "2026-10-01 23:13:00",
            },
            "aircraft": {"modelCode": "a321", "modelText": "airbus a321-231"},
            "airline": {"name": "royal air maroc", "iataCode": "at", "icaoCode": "ram"},
            "flight": {"number": "5027", "iataNumber": "at5027", "icaoNumber": "ram5027"},
            "codeshared": {
                "airline": {"name": "american airlines", "iataCode": "aa", "icaoCode": "aal"},
                "flight": {"number": "117", "iataNumber": "aa117", "icaoNumber": "aal117"},
            },
        }
    ],
}

# One real record's shape, trimmed to what actually varies between flights —
# captured from a live fetch during development.
SAMPLE_RESPONSE = {
    "pagination": {"limit": 2, "offset": 0, "count": 2, "total": 10000},
    "data": [
        {
            "flight_date": "2026-09-08",
            "flight_status": "active",
            "departure": {
                "airport": "Suvarnabhumi International",
                "timezone": "Asia/Bangkok",
                "iata": "BKK",
                "icao": "VTBS",
                "terminal": None,
                "gate": None,
                "delay": None,
                "scheduled": "2026-09-08T01:00:00+00:00",
                "estimated": "2026-09-08T01:00:00+00:00",
                "actual": "2026-09-07T23:21:00+00:00",
                "estimated_runway": "2026-09-07T23:21:00+00:00",
                "actual_runway": "2026-09-07T23:21:00+00:00",
            },
            "arrival": {
                "airport": "Singapore Changi",
                "timezone": "Asia/Singapore",
                "iata": "SIN",
                "icao": "WSSS",
                "terminal": "3",
                "gate": "A5",
                "baggage": None,
                "delay": 12,
                "scheduled": "2026-09-08T04:30:00+00:00",
                "estimated": "2026-09-08T02:21:00+00:00",
                "actual": None,
                "estimated_runway": "2026-09-08T02:21:00+00:00",
                "actual_runway": None,
            },
            "airline": {"name": "K-Mile Air", "iata": "8K", "icao": "KMI"},
            "flight": {"number": "804", "iata": "8K804", "icao": "KMI804", "codeshared": None},
            "aircraft": {"registration": None, "iata": None, "icao": None, "icao24": "882DA2"},
            "live": None,
        }
    ],
}


# --- fetch_flights --------------------------------------------------------


def test_fetch_flights_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("AVIATION_API_KEY", raising=False)

    with pytest.raises(AviationStackError, match="not set"):
        fetch_flights()


def test_fetch_flights_returns_the_parsed_payload(monkeypatch):
    monkeypatch.setattr(
        "httpx.get", lambda *a, **k: FakeHTTPResponse(SAMPLE_RESPONSE)
    )

    result = fetch_flights(flight_status="active", limit=2, api_key="fake-key")

    assert result == SAMPLE_RESPONSE


def test_fetch_flights_raises_on_an_api_error_payload(monkeypatch):
    error_payload = {"error": {"code": "invalid_access_key", "message": "Bad key."}}
    monkeypatch.setattr(
        "httpx.get", lambda *a, **k: FakeHTTPResponse(error_payload)
    )

    with pytest.raises(AviationStackError, match="invalid_access_key"):
        fetch_flights(api_key="fake-key")


def test_fetch_flights_passes_airline_filters_through(monkeypatch):
    captured = {}

    def fake_get(url, params, **kwargs):
        captured.update(params)
        return FakeHTTPResponse(SAMPLE_RESPONSE)

    monkeypatch.setattr("httpx.get", fake_get)

    fetch_flights(airline_name="American Airlines", airline_iata="AA", api_key="fake-key")

    assert captured["airline_name"] == "American Airlines"
    assert captured["airline_iata"] == "AA"


def test_fetch_flights_raises_on_a_real_http_error_status(monkeypatch):
    """AviationStack doesn't always report a bad key via a 200-with-error-body
    — a well-formed but unrecognized key came back as a genuine 401 in a live
    test. That should raise AviationStackError like every other failure here,
    not crash with an unhandled httpx.HTTPStatusError."""
    monkeypatch.setattr(
        "httpx.get", lambda *a, **k: FakeHTTPResponse({"message": "Unauthorized"}, status_code=401)
    )

    with pytest.raises(AviationStackError, match="401"):
        fetch_flights(api_key="fake-key")


# --- normalize_flights ------------------------------------------------------


def test_normalize_flights_flattens_the_nested_fields():
    flights = normalize_flights(SAMPLE_RESPONSE)

    assert len(flights) == 1
    flight = flights[0]
    assert flight.airline == "K-Mile Air"  # was airline.name
    assert flight.flight_number == "8K804"  # was flight.iata
    assert flight.status == "active"  # was flight_status
    assert str(flight.date) == "2026-09-08"  # was flight_date


def test_normalize_flights_converts_timestamps_to_real_datetimes():
    flight = normalize_flights(SAMPLE_RESPONSE)[0]

    assert flight.departure.scheduled.year == 2026
    assert flight.departure.scheduled.hour == 1
    assert flight.arrival.delay_minutes == 12


def test_normalize_flights_drops_fields_a_planner_does_not_need():
    flight = normalize_flights(SAMPLE_RESPONSE)[0]
    dumped = flight.model_dump()

    for field in ("timezone", "icao", "codeshared", "estimated_runway", "actual_runway"):
        assert field not in dumped["departure"]
        assert field not in dumped["arrival"]
    assert "aircraft" not in dumped
    assert "live" not in dumped


def test_normalize_flights_on_no_results_returns_an_empty_list():
    assert normalize_flights({"data": []}) == []


# --- fetch_flights_future ---------------------------------------------------


def test_fetch_flights_future_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("AVIATION_API_KEY", raising=False)

    with pytest.raises(AviationStackError, match="not set"):
        fetch_flights_future(iata_code="JFK", schedule_type="departure", flight_date="2026-10-01")


def test_fetch_flights_future_returns_the_parsed_payload(monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: FakeHTTPResponse(FUTURE_SAMPLE_RESPONSE))

    result = fetch_flights_future(
        iata_code="JFK", schedule_type="departure", flight_date="2026-10-01", api_key="fake-key"
    )

    assert result == FUTURE_SAMPLE_RESPONSE


def test_fetch_flights_future_raises_on_an_api_error_payload(monkeypatch):
    error_payload = {"error": {"code": "function_access_restricted", "message": "Not on plan."}}
    monkeypatch.setattr("httpx.get", lambda *a, **k: FakeHTTPResponse(error_payload))

    with pytest.raises(AviationStackError, match="function_access_restricted"):
        fetch_flights_future(
            iata_code="JFK", schedule_type="departure", flight_date="2026-10-01", api_key="fake-key"
        )


# --- normalize_flights_future -----------------------------------------------


def test_normalize_flights_future_flattens_the_nested_fields():
    flights = normalize_flights_future(FUTURE_SAMPLE_RESPONSE)

    assert len(flights) == 1
    flight = flights[0]
    assert flight.weekday == 4
    assert flight.airline == "royal air maroc"
    assert flight.flight_number == "at5027"  # was flight.iataNumber
    assert flight.aircraft_type == "airbus a321-231"
    assert flight.departure.iata == "jfk"
    assert flight.departure.scheduled_time == "2026-10-01 20:00:00"


def test_normalize_flights_future_keeps_codeshare_info():
    flight = normalize_flights_future(FUTURE_SAMPLE_RESPONSE)[0]

    assert flight.codeshare_airline == "american airlines"
    assert flight.codeshare_flight_number == "aa117"


def test_normalize_flights_future_turns_an_empty_gate_into_none():
    flight = normalize_flights_future(FUTURE_SAMPLE_RESPONSE)[0]

    assert flight.arrival.gate is None


def test_normalize_flights_future_on_no_results_returns_an_empty_list():
    assert normalize_flights_future({"data": []}) == []
