"""Where planned trips live between requests.

`InMemoryTripStore` needs no setup and is what the tests run against.
`SqlTripStore` persists through SQLAlchemy — Postgres in production (see
docker-compose.yml), any SQLAlchemy-supported engine in tests. Both sit behind
the same `TripStore` Protocol, so `get_store()` is the only place that knows
which one is active.
"""

import threading
from functools import lru_cache
from typing import Protocol

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    func,
    insert,
    select,
)
from sqlalchemy.engine import Engine

from app.core.config import get_settings
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


_metadata = MetaData()

_trips_table = Table(
    "trips",
    _metadata,
    Column("id", String, primary_key=True),
    Column("payload", JSON, nullable=False),
    Column("created_at", DateTime, server_default=func.now()),
)


class SqlTripStore:
    """Each trip is stored as its full JSON dump under one row — there is no
    relational schema to migrate when `PlannedTrip`'s shape changes."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        _metadata.create_all(self._engine)

    @classmethod
    def from_url(cls, database_url: str) -> "SqlTripStore":
        return cls(create_engine(database_url))

    def save(self, trip: PlannedTrip) -> None:
        payload = trip.model_dump(mode="json")
        # Delete-then-insert rather than a dialect-specific upsert, so the
        # same code path works against Postgres in production and SQLite in
        # tests.
        with self._engine.begin() as conn:
            conn.execute(delete(_trips_table).where(_trips_table.c.id == trip.id))
            conn.execute(insert(_trips_table).values(id=trip.id, payload=payload))

    def get(self, trip_id: str) -> PlannedTrip | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                select(_trips_table.c.payload).where(_trips_table.c.id == trip_id)
            ).first()
        return PlannedTrip.model_validate(row.payload) if row else None

    def list_all(self) -> list[PlannedTrip]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(_trips_table.c.payload).order_by(_trips_table.c.created_at)
            ).all()
        return [PlannedTrip.model_validate(row.payload) for row in rows]


@lru_cache
def get_store() -> TripStore:
    settings = get_settings()
    if settings.store == "postgres":
        return SqlTripStore.from_url(settings.database_url)
    return InMemoryTripStore()
