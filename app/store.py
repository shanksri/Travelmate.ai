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
    Integer,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    func,
    insert,
    inspect,
    select,
    text,
)
from sqlalchemy.engine import Engine

from app.core.config import get_settings
from app.models.itinerary import PlannedTrip


class TripStore(Protocol):
    def save(self, trip: PlannedTrip) -> None: ...

    def get(self, trip_id: str) -> PlannedTrip | None: ...

    def list_all(self) -> list[PlannedTrip]: ...

    def list_versions(self, thread_id: str) -> list[PlannedTrip]:
        """Every version of one trip, oldest first. Empty if the thread is unknown."""
        ...

    def get_latest(self, thread_id: str) -> PlannedTrip | None:
        """The newest version of one trip — what a revision builds on."""
        ...


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

    def list_versions(self, thread_id: str) -> list[PlannedTrip]:
        with self._lock:
            versions = [t for t in self._trips.values() if t.thread_id == thread_id]
        return sorted(versions, key=lambda t: t.version)

    def get_latest(self, thread_id: str) -> PlannedTrip | None:
        versions = self.list_versions(thread_id)
        return versions[-1] if versions else None


_metadata = MetaData()

_trips_table = Table(
    "trips",
    _metadata,
    Column("id", String, primary_key=True),
    # Denormalized out of `payload` purely so the version queries below can be
    # real SQL instead of deserializing every row to filter in Python.
    Column("thread_id", String, index=True),
    Column("version", Integer),
    Column("payload", JSON, nullable=False),
    Column("created_at", DateTime, server_default=func.now()),
)


class SqlTripStore:
    """Each version is one row holding its full JSON dump — there is no
    relational schema to migrate when `PlannedTrip`'s shape changes, only the
    two columns the version queries need."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        _metadata.create_all(self._engine)
        self._add_versioning_columns_if_missing()

    def _add_versioning_columns_if_missing(self) -> None:
        """`create_all` creates missing *tables*, never missing columns — a
        `trips` table created before versioning existed would keep its old
        three columns and every insert below would fail. Add them and backfill
        instead of dropping the table, so trips planned before this still load
        (each becomes version 1 of its own thread, matching PlannedTrip's own
        fallback for a payload with no thread_id).
        """
        existing = {column["name"] for column in inspect(self._engine).get_columns("trips")}
        missing = {"thread_id": "VARCHAR", "version": "INTEGER"}.keys() - existing
        if not missing:
            return

        with self._engine.begin() as conn:
            for name, sql_type in (("thread_id", "VARCHAR"), ("version", "INTEGER")):
                if name in missing:
                    conn.execute(text(f"ALTER TABLE trips ADD COLUMN {name} {sql_type}"))
            conn.execute(text("UPDATE trips SET thread_id = id WHERE thread_id IS NULL"))
            conn.execute(text("UPDATE trips SET version = 1 WHERE version IS NULL"))
            # create_all() builds this index for a new table but skips an
            # existing one, so the migrated table would otherwise never get it.
            conn.execute(
                text("CREATE INDEX IF NOT EXISTS ix_trips_thread_id ON trips (thread_id)")
            )

    @classmethod
    def from_url(cls, database_url: str) -> "SqlTripStore":
        return cls(create_engine(database_url))

    def save(self, trip: PlannedTrip) -> None:
        payload = trip.model_dump(mode="json")
        # Delete-then-insert rather than a dialect-specific upsert, so the
        # same code path works against Postgres in production and SQLite in
        # tests. Only ever matches this one version's id — earlier versions of
        # the same thread have their own ids and are left alone, which is what
        # makes the store a history rather than a latest-only snapshot.
        with self._engine.begin() as conn:
            conn.execute(delete(_trips_table).where(_trips_table.c.id == trip.id))
            conn.execute(
                insert(_trips_table).values(
                    id=trip.id,
                    thread_id=trip.thread_id,
                    version=trip.version,
                    payload=payload,
                )
            )

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

    def list_versions(self, thread_id: str) -> list[PlannedTrip]:
        with self._engine.connect() as conn:
            rows = conn.execute(
                select(_trips_table.c.payload)
                .where(_trips_table.c.thread_id == thread_id)
                .order_by(_trips_table.c.version)
            ).all()
        return [PlannedTrip.model_validate(row.payload) for row in rows]

    def get_latest(self, thread_id: str) -> PlannedTrip | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                select(_trips_table.c.payload)
                .where(_trips_table.c.thread_id == thread_id)
                .order_by(_trips_table.c.version.desc())
                .limit(1)
            ).first()
        return PlannedTrip.model_validate(row.payload) if row else None


@lru_cache
def get_store() -> TripStore:
    settings = get_settings()
    if settings.store == "postgres":
        return SqlTripStore.from_url(settings.database_url)
    return InMemoryTripStore()
