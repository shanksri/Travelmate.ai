"""The "live" provider: real flights, mock everything else.

Only flights have a real data source in this project so far — Travelpayouts
returns actual fares (see `app/providers/travelpayouts.py`). Lodging,
attractions, destinations and weather are still the deterministic mock data,
so this subclasses `MockTravelProvider` and overrides exactly one method
rather than pretending the rest is real.

When a route has no cached fares, or the API fails, this returns no flights
rather than falling back to mock ones: invented flights shown alongside real
ones would be indistinguishable from them. A trip without flights still
plans, exactly as it does when no origin was given.
"""

import logging
from datetime import date
from typing import Any

from app.providers.mock import MockTravelProvider
from app.providers.travelpayouts import TravelPayoutsError
from app.providers.travelpayouts import search_flights as search_real_flights

logger = logging.getLogger(__name__)


class LiveTravelProvider(MockTravelProvider):
    def search_flights(
        self, origin: str, destination: str, depart: date, travelers: int
    ) -> list[dict[str, Any]]:
        try:
            flights = search_real_flights(origin, destination, depart, travelers)
        except TravelPayoutsError as exc:
            logger.warning("travelpayouts lookup failed for %s -> %s: %s", origin, destination, exc)
            return []

        if not flights:
            logger.info("no cached fares for %s -> %s around %s", origin, destination, depart)
        return flights
