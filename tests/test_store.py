"""Exercises SqlTripStore's actual SQL against an in-memory SQLite engine —
no Docker or Postgres needed to test the query logic. docker-compose.yml is
what points this same class at a real Postgres in production."""

import json

from conftest import sample_planned_trip
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine
from sqlalchemy.pool import StaticPool

from app.store import InMemoryTripStore, SqlTripStore


def _sqlite_engine() -> Engine:
    # StaticPool + check_same_thread=False: keeps one shared in-memory SQLite
    # database alive across the multiple connections SqlTripStore opens,
    # instead of each connection getting its own throwaway database.
    return create_engine(
        "sqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )


def test_get_on_an_empty_store_returns_none():
    store = SqlTripStore(_sqlite_engine())

    assert store.get("nope") is None


def test_save_and_get_round_trip(trip_request):
    store = SqlTripStore(_sqlite_engine())
    trip = sample_planned_trip(trip_request)

    store.save(trip)
    fetched = store.get(trip.id)

    assert fetched is not None
    assert fetched.itinerary.destination == trip.itinerary.destination
    assert fetched.request == trip_request


def test_save_overwrites_a_trip_with_the_same_id(trip_request):
    store = SqlTripStore(_sqlite_engine())
    trip = sample_planned_trip(trip_request)
    store.save(trip)

    updated = trip.model_copy(update={"summary": "Actually a longer stay."})
    store.save(updated)

    assert store.get(trip.id).summary == "Actually a longer stay."
    assert len(store.list_all()) == 1


def test_list_all_returns_every_saved_trip(trip_request):
    store = SqlTripStore(_sqlite_engine())
    store.save(sample_planned_trip(trip_request, trip_id="a"))
    store.save(sample_planned_trip(trip_request, trip_id="b"))

    ids = {trip.id for trip in store.list_all()}

    assert ids == {"a", "b"}


def test_in_memory_store_still_works_unmodified(trip_request):
    """The default store is untouched by adding SqlTripStore alongside it."""
    store = InMemoryTripStore()
    trip = sample_planned_trip(trip_request)

    store.save(trip)

    assert store.get(trip.id) == trip
    assert store.list_all() == [trip]
    assert store.get("missing") is None


# --- version history --------------------------------------------------------


def _versions(trip_request, thread_id="thread-1", count=3):
    return [
        sample_planned_trip(
            trip_request,
            trip_id=f"v{n}",
            thread_id=thread_id,
            version=n,
            change_note=None if n == 1 else f"change {n}",
        )
        for n in range(1, count + 1)
    ]


def test_saving_a_revision_keeps_the_earlier_versions(trip_request):
    """The point of the whole feature: a new version appends to the thread's
    history instead of replacing what came before."""
    store = SqlTripStore(_sqlite_engine())
    for trip in _versions(trip_request):
        store.save(trip)

    history = store.list_versions("thread-1")

    assert [t.version for t in history] == [1, 2, 3]
    assert store.get("v1") is not None  # the original is still retrievable


def test_list_versions_orders_by_version_not_insertion(trip_request):
    store = SqlTripStore(_sqlite_engine())
    first, second, third = _versions(trip_request)
    for trip in (third, first, second):
        store.save(trip)

    assert [t.version for t in store.list_versions("thread-1")] == [1, 2, 3]


def test_list_versions_only_returns_that_thread(trip_request):
    store = SqlTripStore(_sqlite_engine())
    for trip in _versions(trip_request, thread_id="thread-1"):
        store.save(trip)
    store.save(sample_planned_trip(trip_request, trip_id="other", thread_id="thread-2"))

    assert len(store.list_versions("thread-1")) == 3
    assert len(store.list_versions("thread-2")) == 1


def test_list_versions_on_an_unknown_thread_is_empty(trip_request):
    store = SqlTripStore(_sqlite_engine())

    assert store.list_versions("nope") == []


def test_get_latest_returns_the_highest_version(trip_request):
    store = SqlTripStore(_sqlite_engine())
    for trip in _versions(trip_request):
        store.save(trip)

    latest = store.get_latest("thread-1")

    assert latest.version == 3
    assert latest.change_note == "change 3"


def test_get_latest_on_an_unknown_thread_is_none(trip_request):
    store = SqlTripStore(_sqlite_engine())

    assert store.get_latest("nope") is None


def test_in_memory_store_tracks_versions_too(trip_request):
    store = InMemoryTripStore()
    for trip in _versions(trip_request):
        store.save(trip)

    assert [t.version for t in store.list_versions("thread-1")] == [1, 2, 3]
    assert store.get_latest("thread-1").version == 3
    assert store.get_latest("nope") is None


def test_a_row_stored_before_versioning_existed_still_loads(trip_request):
    """Trips saved before thread_id/version existed have neither the columns
    nor the payload fields. The table gets the columns added and backfilled,
    and PlannedTrip treats a payload with no thread_id as version 1 of its
    own thread — so old rows keep working instead of failing validation."""
    engine = _sqlite_engine()
    legacy_payload = sample_planned_trip(trip_request, trip_id="old").model_dump(mode="json")
    del legacy_payload["thread_id"]
    del legacy_payload["version"]

    with engine.begin() as conn:
        conn.execute(
            text("CREATE TABLE trips (id VARCHAR PRIMARY KEY, payload JSON NOT NULL, "
                 "created_at TIMESTAMP)")
        )
        conn.execute(
            text("INSERT INTO trips (id, payload) VALUES ('old', :payload)"),
            {"payload": json.dumps(legacy_payload)},
        )

    store = SqlTripStore(engine)  # runs the column migration on an existing table

    recovered = store.get("old")
    assert recovered is not None
    assert recovered.thread_id == "old"
    assert recovered.version == 1
    assert [t.id for t in store.list_versions("old")] == ["old"]
