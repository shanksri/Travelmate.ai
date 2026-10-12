"""Short-term memory that outlives one request: LangGraph checkpoints.

A graph compiled with a checkpointer saves its state after every step under a
`thread_id`. Invoking it again with the same thread_id starts from that saved
state, so a channel the new input doesn't set still holds what it held at the
end of the last run. That's how the revision agent remembers the earlier
changes to a trip (app/agent/reviser.py).

Follows `TRAVELMATE_STORE` like the trip store: in memory for "memory" (what
the tests use), Postgres for "postgres" — LangGraph's own `checkpoint*`
tables in the same database, created on first use by `setup()`.

Only the revision graph is checkpointed. The ask graph isn't: its state holds
Google Maps results, which Google's terms don't let us store.
"""

from functools import lru_cache

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from sqlalchemy.engine import make_url

from app.core.config import get_settings

# Our own types that the revision graph's state holds, as (module, class).
# Loading a checkpoint rebuilds them by name, and LangGraph only does that
# for types it's been told about: anything else gets a warning today and
# will be refused in a future version, since a stored name could point at
# any class. tests/test_memory.py checks this list against RevisionState.
CHECKPOINTED_TYPES = (
    ("app.models.itinerary", "PlannedTrip"),
    ("app.models.itinerary", "DayPlan"),
    ("app.models.itinerary", "FlightLeg"),
    ("app.models.itinerary", "LodgingOption"),
    ("app.agent.reviser", "ChangeRoute"),
    ("app.agent.reviser", "EditTurn"),
)


def _serializer() -> JsonPlusSerializer:
    return JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINTED_TYPES)


def _psycopg_url(database_url: str) -> str:
    """`postgresql+psycopg://…` (SQLAlchemy's form) as psycopg itself takes it."""
    url = make_url(database_url).set(drivername="postgresql")
    return url.render_as_string(hide_password=False)


@lru_cache
def get_checkpointer() -> BaseCheckpointSaver:
    settings = get_settings()
    if settings.store != "postgres":
        return InMemorySaver(serde=_serializer())

    from langgraph.checkpoint.postgres import PostgresSaver
    from psycopg.rows import dict_row
    from psycopg_pool import ConnectionPool

    # The settings PostgresSaver requires of its connections (see its docs).
    pool = ConnectionPool(
        conninfo=_psycopg_url(settings.database_url),
        max_size=4,
        kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row},
        open=True,
    )
    saver = PostgresSaver(pool, serde=_serializer())
    saver.setup()
    return saver


def close_checkpointer() -> None:
    """Close the Postgres connection pool, if one was opened."""
    if get_checkpointer.cache_info().currsize:
        pool = getattr(get_checkpointer(), "conn", None)
        if hasattr(pool, "close"):
            pool.close()
    get_checkpointer.cache_clear()
