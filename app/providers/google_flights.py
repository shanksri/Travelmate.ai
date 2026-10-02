"""Live flight fares from Google Flights, via SerpApi's hosted MCP server.

This is the planner's live flight source. It replaces Travelpayouts, whose
cached fares were too sparse to use: Varanasi -> Cochin had two fares in all
of October and none near the travel date, while this same search returns
several real flights on the exact day.

Searches go through SerpApi's MCP `search` tool with `engine=google_flights`
(app/providers/serpapi.py). Each uncached call is one search against the
flights account's quota (250/month on the free plan), and a trip makes two —
one per leg.

The planner calls `search_flights_with_source`, which runs the retry and
fallback chain (`fetch_flights_with_fallback`): fresh cache, live search with
retries and a circuit breaker, stale cache, then "unavailable" with a link to
search Google Flights. It never falls back to sample data.
"""

import json
import logging
import os
from datetime import UTC, date, datetime
from typing import Any
from urllib.parse import urlencode

from dotenv import load_dotenv

from app.models.itinerary import DataSource
from app.providers import serpapi
from app.providers.flight_cache import get_flight_cache
from app.providers.iata import airports_for, resolve_route
from app.providers.resilience import RETRY_DELAYS_SECONDS, CircuitBreaker, with_retries
from app.providers.serpapi import SerpApiError

load_dotenv()

logger = logging.getLogger(__name__)

PROVIDER = "Google Flights"


class GoogleFlightsError(SerpApiError):
    """The search failed, or returned an error instead of flights."""


# One per process: shared by both legs and every trip, so a source that keeps
# failing is skipped everywhere until its cooldown is over.
breaker = CircuitBreaker("Google Flights")

# Retry waits; tests set these to zero.
RETRY_DELAYS = RETRY_DELAYS_SECONDS


def _api_key(api_key: str | None = None) -> str:
    # SERPAPI_FLIGHTS_API_KEY pairs with SERPAPI_HOTEL_API_KEY: flights and
    # hotels use separate SerpApi accounts, each with its own monthly quota.
    # SERPAPI_API_KEY is the older single-key name, still accepted.
    key = (
        api_key
        or os.environ.get("SERPAPI_FLIGHTS_API_KEY")
        or os.environ.get("SERPAPI_API_KEY")
    )
    if not key:
        raise GoogleFlightsError("SERPAPI_FLIGHTS_API_KEY is not set", account_problem=True)
    return key.strip()


def _cache_key(departure_ids: str, arrival_ids: str, outbound_date: str) -> str:
    return f"google_flights:v1:{departure_ids}:{arrival_ids}:{outbound_date}"


def _live_search(
    departure_ids: str, arrival_ids: str, outbound_date: str, api_key: str | None
) -> dict:
    """One SerpApi search, retried while it fails transiently. Raises
    GoogleFlightsError, carrying the classification the breaker needs."""
    key = _api_key(api_key)
    params = {
        "engine": "google_flights",
        "departure_id": departure_ids,
        "arrival_id": arrival_ids,
        "outbound_date": outbound_date,
        "type": "2",  # one way
        "adults": "1",
        "currency": "INR",
        "hl": "en",
        "gl": "in",
    }

    def once() -> dict:
        try:
            return serpapi.search(params, key)
        except SerpApiError as exc:
            raise GoogleFlightsError(
                str(exc), transient=exc.transient, account_problem=exc.account_problem
            ) from exc

    return with_retries(
        once, is_transient=lambda exc: getattr(exc, "transient", False), delays=RETRY_DELAYS
    )


def fetch_flights(
    *,
    departure_ids: str,
    arrival_ids: str,
    outbound_date: str,
    api_key: str | None = None,
) -> dict:
    """One-way Google Flights results for one day, as SerpApi returns them.

    `departure_ids` / `arrival_ids` are airport codes, comma-separated for a
    multi-airport city ("NRT,HND"). Prices are per adult, in rupees.

    A search for the same airports and date within `flight_cache_ttl_hours`
    is served from the cache instead of spending another SerpApi search.
    Failed searches are never cached. Raises GoogleFlightsError; the planner
    uses `fetch_flights_with_fallback`, which doesn't.
    """
    cache_key = _cache_key(departure_ids, arrival_ids, outbound_date)
    cache = get_flight_cache()
    cached = cache.get(cache_key)
    if cached is not None:
        logger.info(
            "flight search cache hit: %s -> %s on %s", departure_ids, arrival_ids, outbound_date
        )
        return cached

    payload = _live_search(departure_ids, arrival_ids, outbound_date, api_key)
    cache.put(cache_key, payload)
    return payload


def google_flights_link(origin: str, destination: str, depart: date) -> str:
    """A Google Flights search for the traveller to run themselves."""
    query = f"Flights from {origin} to {destination} on {depart.isoformat()}"
    return f"https://www.google.com/travel/flights?{urlencode({'q': query})}"


def _when(epoch_seconds: float) -> datetime:
    return datetime.fromtimestamp(epoch_seconds, tz=UTC)


def fetch_flights_with_fallback(
    *,
    departure_ids: str,
    arrival_ids: str,
    outbound_date: str,
    link: str,
    api_key: str | None = None,
) -> tuple[dict | None, DataSource]:
    """The retry and fallback chain, in order:

    1. a fresh cached search (under `flight_cache_ttl_hours`);
    2. a live search, retried on transient failures, unless the circuit
       breaker is open after repeated failures;
    3. an older cached search (up to `stale_cache_hours`), labelled stale;
    4. nothing, with the reason and a link to search Google Flights.

    Never raises, and never falls back to sample data.
    """
    cache_key = _cache_key(departure_ids, arrival_ids, outbound_date)
    cache = get_flight_cache()
    kept = cache.lookup(cache_key)
    if kept is not None and cache.get(cache_key) is not None:
        return kept.payload, DataSource(
            status="cached", provider=PROVIDER, fetched_at=_when(kept.fetched_at), link=link
        )

    if breaker.allow():
        try:
            payload = _live_search(departure_ids, arrival_ids, outbound_date, api_key)
        except GoogleFlightsError as exc:
            breaker.record_failure(str(exc), open_now=exc.account_problem)
            failure = str(exc)
            logger.warning("live flight search failed: %s", failure)
        else:
            breaker.record_success()
            cache.put(cache_key, payload)
            return payload, DataSource(
                status="live", provider=PROVIDER, fetched_at=datetime.now(UTC), link=link
            )
    else:
        failure = f"skipped after repeated failures ({breaker.reason})"

    if kept is not None:
        return kept.payload, DataSource(
            status="stale",
            provider=PROVIDER,
            fetched_at=_when(kept.fetched_at),
            detail=f"the live search failed: {failure}",
            link=link,
        )
    return None, DataSource(status="unavailable", provider=PROVIDER, detail=failure, link=link)


def normalize_flights(raw: dict, *, travelers: int = 1) -> list[dict[str, Any]]:
    """Turn raw Google Flights results into the flight-option shape the rest
    of this app uses (the same keys `MockTravelProvider.search_flights`
    returns, so `FlightLeg(**option)` works unchanged). Cheapest first.

    Google occasionally lists a flight with no price; those are skipped,
    since an unpriced option can't sit in a cheapest-first list.
    """
    options: list[dict[str, Any]] = []
    for itinerary in (raw.get("best_flights") or []) + (raw.get("other_flights") or []):
        price = itinerary.get("price")
        segments = itinerary.get("flights") or []
        if price is None or not segments:
            continue

        departure = segments[0].get("departure_airport") or {}
        arrival = segments[-1].get("arrival_airport") or {}
        departs = departure.get("time") or ""  # "2026-10-08 11:55", local time
        if not departs:
            continue
        airlines = list(dict.fromkeys(s.get("airline") for s in segments if s.get("airline")))
        minutes = itinerary.get("total_duration")

        options.append(
            {
                "carrier": " + ".join(airlines) or "unknown",
                "origin": departure.get("id"),
                "destination": arrival.get("id"),
                "depart_date": departs[:10],
                "departure_at": departs.replace(" ", "T"),
                "stops": len(segments) - 1,
                "duration_hours": round(minutes / 60, 1) if minutes else None,
                "price_per_person": float(price),
                "total": round(float(price) * travelers, 2),
            }
        )
    options.sort(key=lambda o: o["total"])
    return options


def search_flights(
    origin: str, destination: str, depart: date, travelers: int
) -> list[dict[str, Any]]:
    """Flights for one leg on `depart`, taking place names, cheapest first.

    Returns `[]` rather than raising when either place can't be resolved;
    a trip should still plan without flights, as it does with no origin.
    Raises GoogleFlightsError when the search fails; the planner uses
    `search_flights_with_source`, which falls back instead.
    """
    origin_code, destination_code = resolve_route(origin, destination)
    if not origin_code or not destination_code or origin_code == destination_code:
        return []

    raw = fetch_flights(
        departure_ids=",".join(airports_for(origin_code)),
        arrival_ids=",".join(airports_for(destination_code)),
        outbound_date=depart.isoformat(),
    )
    return normalize_flights(raw, travelers=travelers)


def search_flights_with_source(
    origin: str, destination: str, depart: date, travelers: int
) -> tuple[list[dict[str, Any]], DataSource]:
    """`search_flights` through the fallback chain, with where the result
    came from. Never raises."""
    link = google_flights_link(origin, destination, depart)
    origin_code, destination_code = resolve_route(origin, destination)
    if not origin_code or not destination_code:
        unmatched = origin if not origin_code else destination
        return [], DataSource(
            status="unavailable",
            provider=PROVIDER,
            detail=f"couldn't match {unmatched!r} to an airport",
            link=link,
        )
    if origin_code == destination_code:
        return [], DataSource(
            status="unavailable",
            provider=PROVIDER,
            detail=f"{origin} and {destination} share an airport",
            link=link,
        )

    raw, source = fetch_flights_with_fallback(
        departure_ids=",".join(airports_for(origin_code)),
        arrival_ids=",".join(airports_for(destination_code)),
        outbound_date=depart.isoformat(),
        link=link,
    )
    return (normalize_flights(raw, travelers=travelers) if raw else []), source


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    flights = search_flights("Varanasi", "Kochi", date(2026, 10, 8), travelers=1)
    print(f"{len(flights)} option(s) for Varanasi -> Kochi on 2026-10-08")
    print(json.dumps(flights[:3], indent=2))
