"""Reuse recent flight searches instead of spending quota on them again.

Every live search costs one SerpApi search, and the free plan allows 250 a
month. Planning the same route on the same date again — re-planning a trip,
or a second trip to the same place — would otherwise search afresh each
time. A result counts as fresh for `flight_cache_ttl_hours`.

It's kept longer than that, up to `stale_cache_hours`, as a fallback: when a
live search fails, an older result labelled with its age beats nothing at all
(see app/providers/resilience.py). `get` returns only fresh results;
`lookup` returns anything still kept, with when it was fetched.

Hotel searches share this cache under their own key prefix.

What's cached is the raw, per-adult search result, keyed by airports and
date — not the normalized options — so trips with different party sizes on
the same route and date share one search.

Follows `TRAVELMATE_STORE`: in memory for "memory", a `flight_search_cache`
table for "postgres". Postgres matters in development: `uvicorn --reload`
restarts the process on every code change, which would wipe an in-memory
cache constantly.
"""

import threading
import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Protocol

from sqlalchemy import (
    JSON,
    Column,
    Float,
    MetaData,
    String,
    Table,
    create_engine,
    delete,
    insert,
    select,
)
from sqlalchemy.engine import Engine

from app.core.config import get_settings


@dataclass(frozen=True)
class CachedSearch:
    payload: dict
    fetched_at: float  # epoch seconds

    @property
    def age_seconds(self) -> float:
        return time.time() - self.fetched_at


class FlightCache(Protocol):
    def get(self, key: str) -> dict | None:
        """The cached payload, or None if absent or older than the TTL."""
        ...

    def lookup(self, key: str) -> CachedSearch | None:
        """The cached search if still kept (up to the stale limit), however
        old, with when it was fetched; None if absent or expired."""
        ...

    def put(self, key: str, payload: dict) -> None: ...


class InMemoryFlightCache:
    def __init__(self, ttl_seconds: float, stale_seconds: float | None = None) -> None:
        self._ttl = ttl_seconds
        self._keep = max(ttl_seconds, stale_seconds or 0)
        self._entries: dict[str, tuple[float, dict]] = {}
        self._lock = threading.Lock()

    def lookup(self, key: str) -> CachedSearch | None:
        with self._lock:
            entry = self._entries.get(key)
        if entry is None or time.time() - entry[0] >= self._keep:
            return None
        return CachedSearch(payload=entry[1], fetched_at=entry[0])

    def get(self, key: str) -> dict | None:
        found = self.lookup(key)
        if found is None or found.age_seconds >= self._ttl:
            return None
        return found.payload

    def put(self, key: str, payload: dict) -> None:
        if self._ttl <= 0:
            return
        now = time.time()
        with self._lock:
            self._entries[key] = (now, payload)
            for stale_key in [k for k, (t, _) in self._entries.items() if now - t >= self._keep]:
                del self._entries[stale_key]


_metadata = MetaData()

_cache_table = Table(
    "flight_search_cache",
    _metadata,
    Column("key", String, primary_key=True),
    Column("payload", JSON, nullable=False),
    # Epoch seconds rather than a DateTime: avoids naive-vs-aware timestamp
    # comparisons differing between Postgres and SQLite (which the tests use).
    Column("fetched_at", Float, nullable=False),
)


class SqlFlightCache:
    def __init__(
        self, engine: Engine, ttl_seconds: float, stale_seconds: float | None = None
    ) -> None:
        self._engine = engine
        self._ttl = ttl_seconds
        self._keep = max(ttl_seconds, stale_seconds or 0)
        _metadata.create_all(self._engine)

    def lookup(self, key: str) -> CachedSearch | None:
        with self._engine.connect() as conn:
            row = conn.execute(
                select(_cache_table.c.payload, _cache_table.c.fetched_at).where(
                    _cache_table.c.key == key
                )
            ).first()
        if row is None or time.time() - row.fetched_at >= self._keep:
            return None
        return CachedSearch(payload=row.payload, fetched_at=row.fetched_at)

    def get(self, key: str) -> dict | None:
        found = self.lookup(key)
        if found is None or found.age_seconds >= self._ttl:
            return None
        return found.payload

    def put(self, key: str, payload: dict) -> None:
        if self._ttl <= 0:
            return
        now = time.time()
        with self._engine.begin() as conn:
            # Replace this key, and prune anything past the stale limit while
            # we're here so the table can't grow without bound.
            conn.execute(
                delete(_cache_table).where(
                    (_cache_table.c.key == key) | (_cache_table.c.fetched_at < now - self._keep)
                )
            )
            conn.execute(insert(_cache_table).values(key=key, payload=payload, fetched_at=now))


@lru_cache
def get_flight_cache() -> FlightCache:
    settings = get_settings()
    ttl_seconds = settings.flight_cache_ttl_hours * 3600
    stale_seconds = settings.stale_cache_hours * 3600
    if settings.store == "postgres":
        return SqlFlightCache(create_engine(settings.database_url), ttl_seconds, stale_seconds)
    return InMemoryFlightCache(ttl_seconds, stale_seconds)
