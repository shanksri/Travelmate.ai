"""A thin client for Open-Meteo's weather API.

Unlike every other client in this package, Open-Meteo needs **no API key** —
it's a free, keyless service, so there's no `_API_KEY` env var, no
`load_dotenv()`, and no "key not set" error path here.

Two real HTTP calls, chained: geocode a place name to coordinates
(`geocode`), then fetch its daily forecast (`fetch_forecast`). Kept separate
because the forecast endpoint takes lat/lon, not a place name, and each half
is independently useful (a caller that already has coordinates can skip
geocoding). `fetch_weather_for_place` chains both for the common case.

Fetch-only for now — this does not yet implement
`TravelProvider.get_weather_outlook`; wiring it in is a separate step.
Standalone-runnable: `python -m app.providers.weather` does one real
geocode + fetch, normalizes it, and prints both.
"""

from datetime import date

import httpx
from pydantic import BaseModel

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

DAILY_FIELDS = (
    "weather_code,temperature_2m_max,temperature_2m_min,"
    "precipitation_sum,precipitation_probability_max,wind_speed_10m_max"
)

# WMO weather interpretation codes, per Open-Meteo's own documentation —
# the raw API returns only the numeric code, not a description.
WEATHER_CODES: dict[int, str] = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow fall",
    73: "moderate snow fall",
    75: "heavy snow fall",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


class WeatherError(RuntimeError):
    """No place matched, or the API responded with an error."""


def _raise_for_status_with_reason(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        try:
            reason = response.json().get("reason")
        except Exception:
            reason = None
        raise WeatherError(
            reason or f"Open-Meteo returned {response.status_code}: {response.text}"
        ) from exc


class GeoLocation(BaseModel):
    name: str
    country: str | None = None
    latitude: float
    longitude: float
    timezone: str | None = None


def geocode(place: str, *, timeout: float = 10.0) -> GeoLocation:
    """Resolve a place name to coordinates. Raises WeatherError if nothing matches —
    Open-Meteo reports "no results" as a 200 with no `results` key at all,
    not an empty list or an error, so that's checked explicitly here."""
    response = httpx.get(GEOCODING_URL, params={"name": place, "count": 1}, timeout=timeout)
    _raise_for_status_with_reason(response)
    results = response.json().get("results") or []
    if not results:
        raise WeatherError(f"No location found matching {place!r}")

    top = results[0]
    return GeoLocation(
        name=top["name"],
        country=top.get("country"),
        latitude=top["latitude"],
        longitude=top["longitude"],
        timezone=top.get("timezone"),
    )


def fetch_forecast(
    latitude: float, longitude: float, *, days: int = 7, timeout: float = 10.0
) -> dict:
    """Fetch a daily forecast for a coordinate. Open-Meteo's free forecast
    horizon caps at 16 days out; `days` is clamped into range rather than
    left to fail with a 400 for an out-of-range request."""
    days = max(1, min(days, 16))

    response = httpx.get(
        FORECAST_URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "daily": DAILY_FIELDS,
            "timezone": "auto",
            "forecast_days": days,
        },
        timeout=timeout,
    )
    _raise_for_status_with_reason(response)
    return response.json()


# --- Normalization ------------------------------------------------------
#
# The raw forecast is columnar — one array per field, aligned by index to
# `daily.time` — not a list of per-day records. normalize_forecast pivots
# that into one DailyForecast per date and swaps the bare WMO code for a
# human-readable description.


class DailyForecast(BaseModel):
    date: date
    conditions: str
    temp_max_c: float | None = None
    temp_min_c: float | None = None
    precipitation_mm: float | None = None
    precipitation_chance_pct: int | None = None
    wind_speed_max_kmh: float | None = None


class WeatherForecast(BaseModel):
    location: str
    country: str | None = None
    latitude: float
    longitude: float
    timezone: str | None = None
    days: list[DailyForecast]


def normalize_forecast(raw: dict, *, location: GeoLocation) -> WeatherForecast:
    """Turn a raw `fetch_forecast` response into a `WeatherForecast`.

    Takes the `GeoLocation` alongside the raw response because the forecast
    endpoint itself never echoes back the place name it was resolved from —
    only latitude/longitude, which alone isn't worth showing a traveller.
    """
    daily = raw.get("daily", {})
    times = daily.get("time", [])
    codes = daily.get("weather_code", [])
    highs = daily.get("temperature_2m_max", [])
    lows = daily.get("temperature_2m_min", [])
    precip = daily.get("precipitation_sum", [])
    precip_chance = daily.get("precipitation_probability_max", [])
    wind = daily.get("wind_speed_10m_max", [])

    def _at(series: list, i: int):
        return series[i] if i < len(series) else None

    days = []
    for i, day in enumerate(times):
        code = _at(codes, i)
        days.append(
            DailyForecast(
                date=day,
                conditions=WEATHER_CODES.get(code, "unknown") if code is not None else "unknown",
                temp_max_c=_at(highs, i),
                temp_min_c=_at(lows, i),
                precipitation_mm=_at(precip, i),
                precipitation_chance_pct=_at(precip_chance, i),
                wind_speed_max_kmh=_at(wind, i),
            )
        )

    return WeatherForecast(
        location=location.name,
        country=location.country,
        latitude=raw.get("latitude", location.latitude),
        longitude=raw.get("longitude", location.longitude),
        timezone=raw.get("timezone", location.timezone),
        days=days,
    )


def fetch_weather_for_place(place: str, *, days: int = 7, timeout: float = 10.0) -> WeatherForecast:
    """Geocode + fetch + normalize in one call — the shape most callers want."""
    location = geocode(place, timeout=timeout)
    raw = fetch_forecast(location.latitude, location.longitude, days=days, timeout=timeout)
    return normalize_forecast(raw, location=location)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    forecast = fetch_weather_for_place("Kyoto")
    print(forecast.model_dump_json(indent=2))
