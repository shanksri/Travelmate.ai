"""A thin client for AviationStack's real-time flights API.

Fetch-only for now — this does not yet implement `TravelProvider`; wiring it
in as the "live" flight source is a separate step once this is tested.
Standalone-runnable: `python -m app.providers.aviationstack` does one real
fetch, normalizes it, and prints both.
"""

import json
import os
from datetime import date, datetime

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

# Idempotent, and doesn't override a var already set in the real environment
# — consistent with how the rest of this project treats .env as a fallback,
# not an authority.
load_dotenv()

BASE_URL = "http://api.aviationstack.com/v1"


class AviationStackError(RuntimeError):
    """The API responded with a structured error instead of flight data."""


def fetch_flights(
    *,
    dep_iata: str | None = None,
    arr_iata: str | None = None,
    flight_status: str | None = None,
    limit: int = 5,
    api_key: str | None = None,
    timeout: float = 10.0,
) -> dict:
    """Fetch real-time flight data.

    All of the search filters are optional — AviationStack only requires the
    access key — but an unfiltered call returns whatever is currently in the
    air worldwide, so pass at least one filter for a meaningful result.
    """
    key = api_key or os.environ.get("AVIATION_API_KEY")
    if not key:
        raise AviationStackError("AVIATION_API_KEY is not set")

    params: dict = {"access_key": key, "limit": limit}
    if dep_iata:
        params["dep_iata"] = dep_iata
    if arr_iata:
        params["arr_iata"] = arr_iata
    if flight_status:
        params["flight_status"] = flight_status

    response = httpx.get(f"{BASE_URL}/flights", params=params, timeout=timeout)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        # AviationStack reports some failures (e.g. a well-formed but
        # unrecognized key) as a real HTTP error status rather than its usual
        # 200-with-an-error-body shape below — seen live via a bad key, which
        # came back as a genuine 401. Without this, that case crashed instead
        # of raising AviationStackError like every other failure here does.
        raise AviationStackError(
            f"AviationStack returned {response.status_code}: {response.text}"
        ) from exc
    payload = response.json()

    if "error" in payload:
        error = payload["error"]
        message = error.get("message") or error.get("info")
        raise AviationStackError(f"{error.get('code')}: {message}")

    return payload


# --- Normalization ------------------------------------------------------
#
# AviationStack's raw response nests departure/arrival/airline/flight/aircraft
# objects, most of whose fields (terminal, gate, codeshare, aircraft
# registration...) are noise for a trip planner and are null far more often
# than not. `normalize_flights` flattens each record to what an agent would
# actually reason over, with real types (datetime, not ISO strings) instead of
# whatever shape the wire happened to use.
#
# Kept separate from `fetch_flights` on purpose: normalization is pure and
# needs no network access, so it can be unit-tested against a saved sample
# response without spending any of the 60-calls-a-month budget.


class FlightEndpoint(BaseModel):
    airport: str | None
    iata: str | None
    terminal: str | None = None
    gate: str | None = None
    scheduled: datetime | None = None
    estimated: datetime | None = None
    actual: datetime | None = None
    delay_minutes: int | None = None


class Flight(BaseModel):
    date: date
    status: str
    airline: str | None
    flight_number: str | None
    departure: FlightEndpoint
    arrival: FlightEndpoint


def _normalize_endpoint(raw: dict) -> FlightEndpoint:
    return FlightEndpoint(
        airport=raw.get("airport"),
        iata=raw.get("iata"),
        terminal=raw.get("terminal"),
        gate=raw.get("gate"),
        scheduled=raw.get("scheduled"),
        estimated=raw.get("estimated"),
        actual=raw.get("actual"),
        delay_minutes=raw.get("delay"),
    )


def normalize_flights(raw: dict) -> list[Flight]:
    """Turn a raw `fetch_flights` response into a list of `Flight`."""
    flights = []
    for item in raw.get("data", []):
        airline = item.get("airline") or {}
        flight = item.get("flight") or {}
        flights.append(
            Flight(
                date=item["flight_date"],
                status=item.get("flight_status", "unknown"),
                airline=airline.get("name"),
                flight_number=flight.get("iata") or flight.get("icao"),
                departure=_normalize_endpoint(item.get("departure") or {}),
                arrival=_normalize_endpoint(item.get("arrival") or {}),
            )
        )
    return flights


if __name__ == "__main__":
    raw = fetch_flights(flight_status="active", limit=2)
    print("=== raw ===")
    print(json.dumps(raw, indent=2))
    print("\n=== normalized ===")
    for flight in normalize_flights(raw):
        print(flight.model_dump_json(indent=2))
