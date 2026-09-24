"""The flight cache exists to save SerpApi quota, so the tests that matter
most are the ones proving a repeat search never reaches the MCP server —
and that an expired or failed one does."""

import pytest
from conftest import FakeMCPToolResult
from sqlalchemy import create_engine
from sqlalchemy.pool import StaticPool

from app.providers import flight_cache, google_flights
from app.providers.flight_cache import InMemoryFlightCache, SqlFlightCache
from app.providers.google_flights import GoogleFlightsError, fetch_flights

PAYLOAD = {"other_flights": [{"price": 9131}]}
HOUR = 3600


def _sqlite():
    return create_engine(
        "sqlite:///:memory:", poolclass=StaticPool, connect_args={"check_same_thread": False}
    )


@pytest.fixture(params=["memory", "sql"])
def cache(request):
    if request.param == "memory":
        return InMemoryFlightCache(ttl_seconds=6 * HOUR)
    return SqlFlightCache(_sqlite(), ttl_seconds=6 * HOUR)


@pytest.fixture
def clock(monkeypatch):
    """Controls what time the cache thinks it is."""
    now = {"t": 1_000_000.0}
    monkeypatch.setattr(flight_cache.time, "time", lambda: now["t"])
    return now


# --- the cache itself, both backends -----------------------------------------


def test_a_miss_returns_none(cache):
    assert cache.get("k") is None


def test_a_stored_search_is_returned(cache, clock):
    cache.put("k", PAYLOAD)

    assert cache.get("k") == PAYLOAD


def test_a_search_expires_after_the_ttl(cache, clock):
    cache.put("k", PAYLOAD)
    clock["t"] += 6 * HOUR

    assert cache.get("k") is None


def test_a_search_just_inside_the_ttl_is_still_served(cache, clock):
    cache.put("k", PAYLOAD)
    clock["t"] += 6 * HOUR - 1

    assert cache.get("k") == PAYLOAD


def test_putting_the_same_key_again_refreshes_it(cache, clock):
    cache.put("k", {"old": True})
    clock["t"] += 5 * HOUR
    cache.put("k", PAYLOAD)
    clock["t"] += 5 * HOUR  # 10h after the first put, 5h after the second

    assert cache.get("k") == PAYLOAD


def test_a_zero_ttl_disables_caching(clock):
    cache = InMemoryFlightCache(ttl_seconds=0)
    cache.put("k", PAYLOAD)

    assert cache.get("k") is None


def test_sql_cache_prunes_expired_rows_on_write(clock):
    engine = _sqlite()
    cache = SqlFlightCache(engine, ttl_seconds=6 * HOUR)
    cache.put("stale", PAYLOAD)
    clock["t"] += 7 * HOUR

    cache.put("fresh", PAYLOAD)

    with engine.connect() as conn:
        keys = {row.key for row in conn.execute(flight_cache._cache_table.select())}
    assert keys == {"fresh"}


def test_sql_cache_survives_a_new_instance_on_the_same_database(clock):
    """The reason it exists: `uvicorn --reload` restarts the process, and the
    cache must outlive that."""
    engine = _sqlite()
    SqlFlightCache(engine, ttl_seconds=6 * HOUR).put("k", PAYLOAD)

    assert SqlFlightCache(engine, ttl_seconds=6 * HOUR).get("k") == PAYLOAD


# --- wired into fetch_flights ------------------------------------------------


class _FakeMCP(list):
    """Records each MCP round trip — each would be one SerpApi search."""

    def __init__(self) -> None:
        super().__init__()
        self.result = FakeMCPToolResult(PAYLOAD)

    def respond_with(self, result) -> None:
        self.result = result


@pytest.fixture
def mcp_calls(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "fake")
    calls = _FakeMCP()

    async def fake_call_search(params, api_key):
        calls.append(params)
        return calls.result

    monkeypatch.setattr(google_flights, "_call_search", fake_call_search)
    return calls


def _search(date="2026-10-08"):
    return fetch_flights(departure_ids="VNS", arrival_ids="COK", outbound_date=date)


def test_a_repeat_search_does_not_spend_another_search(mcp_calls):
    first = _search()
    second = _search()

    assert first == second == PAYLOAD
    assert len(mcp_calls) == 1


def test_a_different_date_is_a_different_search(mcp_calls):
    _search("2026-10-08")
    _search("2026-10-09")

    assert len(mcp_calls) == 2


def test_a_failed_search_is_not_cached(mcp_calls):
    mcp_calls.respond_with(FakeMCPToolResult({"error": "temporarily unavailable"}))
    with pytest.raises(GoogleFlightsError):
        _search()

    mcp_calls.respond_with(FakeMCPToolResult(PAYLOAD))
    assert _search() == PAYLOAD
    assert len(mcp_calls) == 2
