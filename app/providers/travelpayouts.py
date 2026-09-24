"""A thin client for Travelpayouts' (Aviasales) cached flight-fare API.

This is the first provider here that returns **real prices**. AviationStack —
the other flight API in this project — has no fare data at all, only
schedules and status, which is why it can't drive a "cheapest flight" list.

Endpoint used: `/aviasales/v3/prices_for_dates`. It takes an exact
departure date and returns the operating airline, flight number, departure
time, per-leg duration (`duration_to`), stops (`transfers`) and price. An
earlier version used `/v2/prices/latest`, which ignores travel dates — it
returns the cheapest fares found across a whole year, so a trip for 10-13
October got an outbound flight on 5 October and a return on 29 September.

Known limits of this data, all measured against the live API:

- **Cached, not live.** Travelpayouts' own docs say to treat it as static
  content. These are indicative prices, not bookable quotes.
- **Sparse per day.** The cache holds roughly one fare per route per day —
  a ±1-day window around 10 October on DEL→BOM, one of India's busiest
  routes, returned a single flight. `DATE_WINDOW_DAYS` is the one place to
  widen that.
"""

import os
from datetime import date, timedelta
from functools import lru_cache
from typing import Any

import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.travelpayouts.com"
CITIES_URL = f"{BASE_URL}/data/en/cities.json"
PRICES_URL = f"{BASE_URL}/aviasales/v3/prices_for_dates"

# Fares are searched from this many days before the requested departure to
# this many days after, so the cheapest/fastest comparison has more than the
# single flight a one-day search usually returns.
DATE_WINDOW_DAYS = 1


class TravelPayoutsError(RuntimeError):
    """The API responded with an error instead of fares."""


def _api_key(api_key: str | None = None) -> str:
    key = api_key or os.environ.get("TRAVELPAYOUTS_API_KEY")
    if not key:
        raise TravelPayoutsError("TRAVELPAYOUTS_API_KEY is not set")
    return key.strip()


# --- Place name -> IATA ---------------------------------------------------
#
# Every fare endpoint takes IATA codes, but this app deals in place names
# ("Delhi", "Kyoto, Japan"). Travelpayouts publishes a keyless city directory,
# which is fetched once per process and cached.


@lru_cache(maxsize=1)
def _cities(timeout: float = 60.0) -> list[dict]:
    response = httpx.get(CITIES_URL, timeout=timeout)
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=256)
def _matches(place: str) -> tuple[dict, ...]:
    """The best tier of city matches for a place name, best first; possibly
    several when the name is shared (there's a Kochi in India and in Japan).

    Takes the part before any comma, so "Kyoto, Japan" resolves like "Kyoto".
    Prefers an exact name match, then a city whose name contains the query —
    which is what turns "Delhi" into New Delhi's DEL.
    """
    name = place.split(",")[0].strip().lower()
    if not name:
        return ()

    exact, partial = [], []
    for city in _cities():
        city_name = (city.get("name") or "").lower()
        if not city_name:
            continue
        if city_name == name:
            exact.append(city)
        elif name in city_name:
            partial.append(city)

    def flightable(cities: list[dict]) -> list[dict]:
        return [c for c in cities if c.get("has_flightable_airport")]

    # A flightable partial match beats a non-flightable exact one: searching
    # from a city with no flightable airport returns nothing at all, so an
    # exactly-named dead end is worse than a nearby usable airport.
    for candidates in (flightable(exact), flightable(partial), exact, partial):
        if candidates:
            return tuple(candidates)
    return ()


def resolve_iata(place: str) -> str | None:
    """Best-effort IATA city code for one place name on its own, or None."""
    matches = _matches(place)
    return matches[0].get("code") if matches else None


def resolve_route(origin: str, destination: str) -> tuple[str | None, str | None]:
    """IATA codes for both ends of a route, resolving shared names together.

    "Kochi" alone is ambiguous — Kochi, Japan (KCZ) and Kochi/Cochin, India
    (COK) are both flightable, and Japan's comes first in the directory. A
    real "Varanasi -> Kochi" search went to Japan and found nothing. So when
    a name is shared, prefer the match in the same country as the other end
    of the route; if no pair shares a country, each end keeps its best match.
    """
    from_matches, to_matches = _matches(origin), _matches(destination)
    if not from_matches or not to_matches:
        return (
            from_matches[0].get("code") if from_matches else None,
            to_matches[0].get("code") if to_matches else None,
        )

    for a in from_matches:
        for b in to_matches:
            if a.get("country_code") and a.get("country_code") == b.get("country_code"):
                return a.get("code"), b.get("code")
    return from_matches[0].get("code"), to_matches[0].get("code")


# --- Fares ----------------------------------------------------------------


def fetch_prices(
    *,
    origin: str,
    destination: str,
    departure_at: str,
    currency: str = "inr",
    limit: int = 30,
    api_key: str | None = None,
    timeout: float = 30.0,
) -> dict:
    """Fetch cached one-way fares for one route on one day.

    `origin`/`destination` are IATA codes; `departure_at` is YYYY-MM-DD.
    """
    response = httpx.get(
        PRICES_URL,
        params={
            "origin": origin,
            "destination": destination,
            "departure_at": departure_at,
            "one_way": "true",
            "direct": "false",
            "unique": "false",
            "sorting": "price",
            "currency": currency,
            "limit": limit,
        },
        headers={"x-access-token": _api_key(api_key)},
        timeout=timeout,
    )
    if response.status_code == 401:
        # Sending no token at all returns this too, so it means "this account
        # isn't entitled to fare data" as often as it means "bad token" —
        # check the Aviasales program is connected before blaming the key.
        raise TravelPayoutsError(
            "401 Unauthorized — the token must be the 32-character hex one from the "
            "Aviasales program's API section, and that program must be connected."
        )
    if response.status_code >= 400:
        try:
            detail = response.json().get("error")
        except ValueError:
            detail = None
        raise TravelPayoutsError(
            detail or f"Travelpayouts returned {response.status_code}: {response.text[:200]}"
        )

    payload = response.json()
    if payload.get("success") is False:
        raise TravelPayoutsError(str(payload.get("error") or "request was not successful"))
    return payload


def normalize_prices(raw: dict, *, travelers: int = 1) -> list[dict[str, Any]]:
    """Turn a raw `fetch_prices` response into the flight-option shape the
    rest of this app uses (the same keys `MockTravelProvider.search_flights`
    returns, so `FlightLeg(**option)` works unchanged).

    A row with no duration is kept with `duration_hours=None` rather than
    dropped — it's still a valid cheapest option, and the fastest ranking
    excludes it explicitly instead of letting a missing value sort first.
    """
    options: list[dict[str, Any]] = []
    for row in raw.get("data") or []:
        price = row.get("price")
        departure_at = row.get("departure_at")
        if price is None or not departure_at:
            continue
        minutes = row.get("duration_to") or row.get("duration")
        options.append(
            {
                "carrier": row.get("airline") or "unknown",
                "origin": row.get("origin_airport") or row.get("origin"),
                "destination": row.get("destination_airport") or row.get("destination"),
                "depart_date": departure_at[:10],
                "departure_at": departure_at,
                "stops": row.get("transfers") or 0,
                "duration_hours": round(minutes / 60, 1) if minutes else None,
                "price_per_person": float(price),
                "total": round(float(price) * travelers, 2),
            }
        )
    return options


def search_flights(
    origin: str, destination: str, depart: date, travelers: int
) -> list[dict[str, Any]]:
    """Fares for one leg across `depart` ± `DATE_WINDOW_DAYS`, cheapest first.

    Takes place names rather than IATA codes. Returns `[]` rather than
    raising when either place can't be resolved — a trip should still plan
    without flights, exactly as it does when no origin was given.
    """
    origin_code, destination_code = resolve_route(origin, destination)
    if not origin_code or not destination_code or origin_code == destination_code:
        return []

    today = date.today()
    days = [
        depart + timedelta(days=offset)
        for offset in range(-DATE_WINDOW_DAYS, DATE_WINDOW_DAYS + 1)
    ]

    options: list[dict[str, Any]] = []
    seen: set[tuple] = set()
    for day in days:
        if day < today:  # a flight that already left can't be booked
            continue
        raw = fetch_prices(
            origin=origin_code, destination=destination_code, departure_at=day.isoformat()
        )
        for option in normalize_prices(raw, travelers=travelers):
            key = (option["carrier"], option["departure_at"], option["price_per_person"])
            if key not in seen:
                seen.add(key)
                options.append(option)

    options.sort(key=lambda o: o["total"])
    return options


if __name__ == "__main__":
    import json
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    departure = date.today() + timedelta(days=16)
    flights = search_flights("Delhi", "Mumbai", departure, travelers=2)
    print(f"{len(flights)} option(s) for Delhi -> Mumbai around {departure}")
    print(json.dumps(flights[:3], indent=2))
