"""A thin client for AviationStack's real-time flights API.

Fetch-only for now — this does not yet implement `TravelProvider`; wiring it
in as the "live" flight source is a separate step once this is tested.
Standalone-runnable: `python -m app.providers.aviationstack` does one real
fetch and prints it.
"""

import json
import os

import httpx
from dotenv import load_dotenv

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
    response.raise_for_status()
    payload = response.json()

    if "error" in payload:
        error = payload["error"]
        message = error.get("message") or error.get("info")
        raise AviationStackError(f"{error.get('code')}: {message}")

    return payload


if __name__ == "__main__":
    result = fetch_flights(flight_status="active", limit=2)
    print(json.dumps(result, indent=2))
