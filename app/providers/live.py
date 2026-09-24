"""The "live" provider: real flights, mock everything else.

Only flights have a real data source in this project so far — live Google
Flights results via SerpApi's MCP server (see
`app/providers/google_flights.py`). Lodging, attractions, destinations and
weather are still the deterministic mock data, so this subclasses
`MockTravelProvider` and overrides exactly one method rather than pretending
the rest is real.

When a search finds nothing, or fails, this returns no flights rather than
falling back to mock ones: invented flights shown alongside real ones would
be indistinguishable from them. A trip without flights still plans, exactly
as it does when no origin was given.
"""

import logging
from datetime import date
from typing import Any

from app.providers.google_flights import GoogleFlightsError
from app.providers.google_flights import search_flights as search_real_flights
from app.providers.mock import MockTravelProvider

logger = logging.getLogger(__name__)


class LiveTravelProvider(MockTravelProvider):
    def search_flights(
        self, origin: str, destination: str, depart: date, travelers: int
    ) -> list[dict[str, Any]]:
        try:
            flights = search_real_flights(origin, destination, depart, travelers)
        except GoogleFlightsError as exc:
            logger.warning("flight search failed for %s -> %s: %s", origin, destination, exc)
            return []

        if not flights:
            logger.info("no flights found for %s -> %s on %s", origin, destination, depart)
        return flights
