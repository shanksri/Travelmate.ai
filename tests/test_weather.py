"""Fetch tests mock httpx entirely (see the autouse block in conftest.py) —
no real call. Normalization tests need no network at all; they run a saved
sample shaped exactly like the real response fetched during development.

Unlike aviationstack.py/tavily.py, this client needs no API key — Open-Meteo
is free and keyless — so there's no "key not set" test here."""

import pytest
from conftest import FakeHTTPResponse

from app.providers.weather import (
    GeoLocation,
    WeatherError,
    fetch_forecast,
    fetch_weather_for_place,
    geocode,
    normalize_forecast,
)

GEOCODE_SAMPLE = {
    "results": [
        {
            "id": 1857910,
            "name": "Kyoto",
            "latitude": 35.02107,
            "longitude": 135.75385,
            "elevation": 50.0,
            "country_code": "JP",
            "timezone": "Asia/Tokyo",
            "population": 1463723,
            "country": "Japan",
            "admin1": "Kyoto",
        }
    ],
}

# Trimmed to 2 days — captured from a live fetch during development.
FORECAST_SAMPLE = {
    "latitude": 35.0,
    "longitude": 135.75,
    "timezone": "Asia/Tokyo",
    "daily_units": {"time": "iso8601"},
    "daily": {
        "time": ["2026-09-18", "2026-09-19"],
        "weather_code": [51, 3],
        "temperature_2m_max": [27.1, 29.4],
        "temperature_2m_min": [20.9, 21.4],
        "precipitation_sum": [0.6, 0.0],
        "precipitation_probability_max": [17, 26],
        "wind_speed_10m_max": [4.3, 5.2],
    },
}

KYOTO = GeoLocation(name="Kyoto", country="Japan", latitude=35.02107, longitude=135.75385)


# --- geocode -----------------------------------------------------------


def test_geocode_returns_the_top_match(monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: FakeHTTPResponse(GEOCODE_SAMPLE))

    location = geocode("Kyoto")

    assert location.name == "Kyoto"
    assert location.country == "Japan"
    assert location.latitude == 35.02107


def test_geocode_raises_when_nothing_matches(monkeypatch):
    # Open-Meteo reports "no results" as a 200 with no `results` key at all.
    monkeypatch.setattr("httpx.get", lambda *a, **k: FakeHTTPResponse({}))

    with pytest.raises(WeatherError, match="No location found"):
        geocode("zzxxqqnowhereplace123")


def test_geocode_raises_on_an_api_error(monkeypatch):
    monkeypatch.setattr(
        "httpx.get",
        lambda *a, **k: FakeHTTPResponse(
            {"error": True, "reason": "Bad request."}, status_code=400
        ),
    )

    with pytest.raises(WeatherError, match="Bad request"):
        geocode("Kyoto")


# --- fetch_forecast ------------------------------------------------------


def test_fetch_forecast_returns_the_parsed_payload(monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: FakeHTTPResponse(FORECAST_SAMPLE))

    result = fetch_forecast(35.02107, 135.75385)

    assert result == FORECAST_SAMPLE


def test_fetch_forecast_clamps_days_into_the_16_day_horizon(monkeypatch):
    captured = {}

    def fake_get(url, params, **kwargs):
        captured.update(params)
        return FakeHTTPResponse(FORECAST_SAMPLE)

    monkeypatch.setattr("httpx.get", fake_get)

    fetch_forecast(35.0, 135.0, days=30)

    assert captured["forecast_days"] == 16


def test_fetch_forecast_raises_on_an_api_error(monkeypatch):
    monkeypatch.setattr(
        "httpx.get",
        lambda *a, **k: FakeHTTPResponse(
            {"error": True, "reason": "Latitude must be in range of -90 to 90°."}, status_code=400
        ),
    )

    with pytest.raises(WeatherError, match="Latitude must be in range"):
        fetch_forecast(999, 999)


# --- normalize_forecast --------------------------------------------------


def test_normalize_forecast_pivots_the_columnar_shape_into_one_entry_per_day():
    forecast = normalize_forecast(FORECAST_SAMPLE, location=KYOTO)

    assert forecast.location == "Kyoto"
    assert forecast.country == "Japan"
    assert len(forecast.days) == 2
    assert str(forecast.days[0].date) == "2026-09-18"
    assert str(forecast.days[1].date) == "2026-09-19"


def test_normalize_forecast_maps_wmo_codes_to_descriptions():
    forecast = normalize_forecast(FORECAST_SAMPLE, location=KYOTO)

    assert forecast.days[0].conditions == "light drizzle"  # was weather_code 51
    assert forecast.days[1].conditions == "overcast"  # was weather_code 3


def test_normalize_forecast_on_an_unknown_code_falls_back_gracefully():
    raw = {**FORECAST_SAMPLE, "daily": {**FORECAST_SAMPLE["daily"], "weather_code": [12345, 3]}}

    forecast = normalize_forecast(raw, location=KYOTO)

    assert forecast.days[0].conditions == "unknown"


def test_normalize_forecast_on_no_days_returns_an_empty_list():
    forecast = normalize_forecast({"daily": {"time": []}}, location=KYOTO)

    assert forecast.days == []


# --- fetch_weather_for_place ----------------------------------------------


def test_fetch_weather_for_place_chains_geocode_and_forecast(monkeypatch):
    responses = iter([FakeHTTPResponse(GEOCODE_SAMPLE), FakeHTTPResponse(FORECAST_SAMPLE)])
    monkeypatch.setattr("httpx.get", lambda *a, **k: next(responses))

    forecast = fetch_weather_for_place("Kyoto")

    assert forecast.location == "Kyoto"
    assert len(forecast.days) == 2


def test_fetch_weather_for_place_surfaces_a_geocode_miss(monkeypatch):
    monkeypatch.setattr("httpx.get", lambda *a, **k: FakeHTTPResponse({}))

    with pytest.raises(WeatherError, match="No location found"):
        fetch_weather_for_place("zzxxqqnowhereplace123")
