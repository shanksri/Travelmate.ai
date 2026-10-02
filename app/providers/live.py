"""The "live" provider: real flights, mock everything else.

Flights are live Google Flights results via SerpApi's MCP server
(`app/providers/google_flights.py`), through its retry and fallback chain:
fresh cache, live search, stale cache, then "unavailable" with a link to
search Google Flights. Lodging, attractions, destinations and weather are
still the deterministic mock data, so this subclasses `MockTravelProvider`
and overrides only flights, rather than pretending the rest is real. Sample
data reports itself as "sample", so the page labels it.

A failed or empty flight search never falls back to mock flights: invented
flights shown alongside real ones would be indistinguishable from them. A
trip without flights still plans, exactly as it does when no origin was
given.
"""

import logging
from datetime import date
from typing import Any

from app.models.itinerary import DataSource
from app.providers.google_flights import GoogleFlightsError, search_flights_with_source
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
        try:
            flights, _ = self.search_flights_with_source(origin, destination, depart, travelers)
        except GoogleFlightsError as exc:  # pragma: no cover - the chain doesn't raise
            logger.warning("flight search failed for %s -> %s: %s", origin, destination, exc)
            return []
        return flights
