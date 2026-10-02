"""The "live" provider: real flights and hotels, mock everything else.

Flights are live Google Flights results and hotels live Google Hotels
results, both via SerpApi's MCP server (`app/providers/google_flights.py`,
`app/providers/google_hotels.py`), each through the same retry and fallback
chain: fresh cache, live search, stale cache, then "unavailable" with a link
to search for themselves. Attractions, destinations and weather are still
the deterministic mock data, so this subclasses `MockTravelProvider` and
overrides only flights and lodging, rather than pretending the rest is real.

A failed or empty search never falls back to mock flights or hotels:
invented ones shown alongside real ones would be indistinguishable from
them. A trip without them still plans, exactly as it does when no origin
was given.
"""

import logging
from datetime import date
from typing import Any

from app.models.itinerary import DataSource
from app.providers.google_flights import search_flights_with_source
from app.providers.google_hotels import search_hotels_with_source
from app.providers.mock import MockTravelProvider

logger = logging.getLogger(__name__)


class LiveTravelProvider(MockTravelProvider):
    def search_flights_with_source(
        self, origin: str, destination: str, depart: date, travelers: int
    ) -> tuple[list[dict[str, Any]], DataSource]:
        flights, source = search_flights_with_source(origin, destination, depart, travelers)
        if source.status == "unavailable":
            logger.warning(
                "no flights for %s -> %s on %s: %s", origin, destination, depart, source.detail
            )
        return flights, source

    def search_flights(
        self, origin: str, destination: str, depart: date, travelers: int
    ) -> list[dict[str, Any]]:
        return self.search_flights_with_source(origin, destination, depart, travelers)[0]

    def search_lodging_with_source(
        self, destination: str, check_in: date, nights: int, travelers: int
    ) -> tuple[list[dict[str, Any]], DataSource]:
        hotels, source = search_hotels_with_source(destination, check_in, nights, travelers)
        if source.status == "unavailable":
            logger.warning("no hotels for %s from %s: %s", destination, check_in, source.detail)
        return hotels, source

    def search_lodging(
        self, destination: str, check_in: date, nights: int, travelers: int
    ) -> list[dict[str, Any]]:
        return self.search_lodging_with_source(destination, check_in, nights, travelers)[0]
