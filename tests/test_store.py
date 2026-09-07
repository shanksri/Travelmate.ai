"""Exercises SqlTripStore's actual SQL against an in-memory SQLite engine —
no Docker or Postgres needed to test the query logic. docker-compose.yml is
what points this same class at a real Postgres in production."""

from conftest import sample_planned_trip
from sqlalchemy import create_engine
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
