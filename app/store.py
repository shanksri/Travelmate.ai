"""Where planned trips live between requests.

In-memory today, behind a Protocol so swapping in Postgres is a one-file
change rather than a refactor.
"""

import threading
from typing import Protocol

from app.models.itinerary import PlannedTrip


class TripStore(Protocol):
    def save(self, trip: PlannedTrip) -> None: ...

    def get(self, trip_id: str) -> PlannedTrip | None: ...

    def list_all(self) -> list[PlannedTrip]: ...


class InMemoryTripStore:
    """Process-local and lost on restart — fine for a single instance."""

    def __init__(self) -> None:
        self._trips: dict[str, PlannedTrip] = {}
        self._lock = threading.Lock()

    def save(self, trip: PlannedTrip) -> None:
        with self._lock:
            self._trips[trip.id] = trip

    def get(self, trip_id: str) -> PlannedTrip | None:
        with self._lock:
            return self._trips.get(trip_id)

    def list_all(self) -> list[PlannedTrip]:
        with self._lock:
            return list(self._trips.values())


_store = InMemoryTripStore()


def get_store() -> TripStore:
    return _store
