"""Live hotel prices from Google Hotels, via SerpApi's hosted MCP server.

Searches go through SerpApi's MCP `search` tool with `engine=google_hotels`
(app/providers/serpapi.py) on a second SerpApi account, `SERPAPI_HOTEL_API_KEY`,
so hotel searches don't eat into the flights account's quota. One search per
trip; repeats within `flight_cache_ttl_hours` come from the shared search
cache under their own key prefix.

The planner calls `search_hotels_with_source`, which runs the same retry and
fallback chain as flights (`fetch_with_fallback` in
app/providers/resilience.py): fresh cache, live search with retries and a
circuit breaker, stale cache, then "unavailable" with a link to search Google
Hotels. It never falls back to sample data.

Google Maps (places) is deliberately *not* a fallback here, though it was
in the first sketch: Google's terms for Maps Grounding Lite forbid storing
its results, and hotels are saved with every trip.

Which hotels are offered: priced ones rated at least `MIN_RATING`, cheapest
first, at most `MAX_OPTIONS`. The planner books the first (cheapest) by
default, as it always has; the rating floor keeps "cheapest" from meaning a
poorly reviewed place, and "a better rated hotel" in a revision can switch
among the rest.
"""

import logging
import os
from datetime import date, timedelta
from typing import Any
from urllib.parse import urlencode

from dotenv import load_dotenv

from app.models.itinerary import DataSource
from app.providers import serpapi
from app.providers.flight_cache import get_flight_cache
from app.providers.resilience import (
    RETRY_DELAYS_SECONDS,
    CircuitBreaker,
    fetch_with_fallback,
    with_retries,
)
from app.providers.serpapi import SerpApiError

load_dotenv()

logger = logging.getLogger(__name__)

PROVIDER = "Google Hotels"
MIN_RATING = 3.5
MAX_OPTIONS = 5

breaker = CircuitBreaker("Google Hotels")

# Retry waits; tests set these to zero.
RETRY_DELAYS = RETRY_DELAYS_SECONDS


class GoogleHotelsError(SerpApiError):
    """The search failed, or returned an error instead of hotels."""


def _api_key(api_key: str | None = None) -> str:
    key = api_key or os.environ.get("SERPAPI_HOTEL_API_KEY")
    if not key:
        raise GoogleHotelsError("SERPAPI_HOTEL_API_KEY is not set", account_problem=True)
    return key.strip()


def google_hotels_link(destination: str) -> str:
    """A Google Hotels search for the traveller to run themselves."""
    return f"https://www.google.com/travel/hotels?{urlencode({'q': f'hotels in {destination}'})}"


def _live_search(
    destination: str, check_in: date, check_out: date, adults: int, api_key: str | None
) -> dict:
    key = _api_key(api_key)
    params = {
        "engine": "google_hotels",
        "q": destination,
        "check_in_date": check_in.isoformat(),
        "check_out_date": check_out.isoformat(),
        "adults": str(adults),
        "currency": "INR",
        "gl": "in",
        "hl": "en",
    }

    def once() -> dict:
        try:
            return serpapi.search(params, key)
        except SerpApiError as exc:
            raise GoogleHotelsError(
                str(exc), transient=exc.transient, account_problem=exc.account_problem
            ) from exc

    return with_retries(
        once, is_transient=lambda exc: getattr(exc, "transient", False), delays=RETRY_DELAYS
    )


def normalize_hotels(raw: dict) -> list[dict[str, Any]]:
    """Google Hotels properties in the lodging-option shape the rest of the
    app uses (`LodgingOption(**option)`): priced, rated at least
    `MIN_RATING`, cheapest first, at most `MAX_OPTIONS`.

    `rating` is Google's 0-5 stars. `total` is Google's price for the whole
    stay for the party searched, not a nightly price times nights.
    """
    options: list[dict[str, Any]] = []
    for prop in raw.get("properties") or []:
        nightly = (prop.get("rate_per_night") or {}).get("extracted_lowest")
        total = (prop.get("total_rate") or {}).get("extracted_lowest")
        rating = prop.get("overall_rating")
        if not prop.get("name") or nightly is None or total is None:
            continue
        if rating is None or rating < MIN_RATING:
            continue
        options.append(
            {
                "name": prop["name"],
                "tier": prop.get("hotel_class") or prop.get("type"),
                "rating": rating,
                "neighbourhood": prop.get("address"),
                "nightly": float(nightly),
                "total": float(total),
            }
        )
    options.sort(key=lambda o: o["total"])
    return options[:MAX_OPTIONS]


def search_hotels_with_source(
    destination: str, check_in: date, nights: int, travelers: int
) -> tuple[list[dict[str, Any]], DataSource]:
    """Hotels for the stay, through the fallback chain, with where the
    result came from. Never raises."""
    check_out = check_in + timedelta(days=max(nights, 1))
    adults = max(travelers, 1)
    raw, source = fetch_with_fallback(
        cache=get_flight_cache(),
        cache_key=(
            f"google_hotels:v1:{destination.strip().lower()}:"
            f"{check_in.isoformat()}:{check_out.isoformat()}:{adults}"
        ),
        live=lambda: _live_search(destination, check_in, check_out, adults, None),
        breaker=breaker,
        provider=PROVIDER,
        link=google_hotels_link(destination),
    )
    return (normalize_hotels(raw) if raw else []), source
