import json
import os
from datetime import date

import httpx
import pytest

# Force safe defaults before any app module can call get_settings() and cache
# a Settings instance for the rest of the run. Without this, a developer's
# local .env (e.g. TRAVELMATE_STORE=postgres pointing at a real database) leaks
# into the test session — pydantic-settings reads .env directly, so it can't
# be avoided just by not exporting these as real environment variables.
# setdefault, not assignment: an intentional CI override still wins.
os.environ.setdefault("TRAVELMATE_STORE", "memory")
os.environ.setdefault("TRAVELMATE_PROVIDER", "mock")

# Same problem, different mechanism: app/providers/{aviationstack,tavily}.py
# call load_dotenv() at import time, which — unlike pydantic-settings — does
# write into the real os.environ. A developer's .env with LANGSMITH_TRACING=
# true would otherwise make LangGraph try to phone home to LangSmith with a
# real API key on every graph invocation in the test suite. Setting this
# first means load_dotenv()'s default no-override behavior leaves it alone.
os.environ.setdefault("LANGSMITH_TRACING", "false")

from app.models.itinerary import Itinerary, PlannedTrip, TripRequest  # noqa: E402
from app.providers.mock import MockTravelProvider  # noqa: E402


@pytest.fixture
def provider() -> MockTravelProvider:
    return MockTravelProvider()


@pytest.fixture
def trip_request() -> TripRequest:
    return TripRequest(
        start_date=date(2026, 4, 10),
        end_date=date(2026, 4, 13),
        travelers=2,
        destination="Lisbon, Portugal",
        origin="Boston, MA",
        budget_usd=3000,
        interests=["food", "history"],
        pace="balanced",
    )


def itinerary_payload(request: TripRequest) -> dict:
    """A valid itinerary for `request`, one day per calendar day."""
    days = []
    for offset in range(request.nights + 1):
        day_date = date.fromordinal(request.start_date.toordinal() + offset)
        days.append(
            {
                "day": offset + 1,
                "date": day_date.isoformat(),
                "summary": f"Day {offset + 1}",
                "activities": [
                    {
                        "time": "09:00",
                        "title": "Walk the old town",
                        "description": "Morning stroll.",
                        "location": "Alfama",
                        "category": "sightseeing",
                        "estimated_cost_usd": 0,
                    }
                ],
            }
        )
    return {
        "destination": request.destination or "Lisbon, Portugal",
        "start_date": request.start_date.isoformat(),
        "end_date": request.end_date.isoformat(),
        "travelers": request.travelers,
        "currency": "USD",
        "total_estimated_cost": 2400.0,
        "notes": ["Book the castle ahead."],
        "days": days,
    }


def itinerary_json(request: TripRequest, **overrides) -> str:
    return json.dumps(itinerary_payload(request) | overrides)


def sample_planned_trip(request: TripRequest, trip_id: str = "abc123") -> PlannedTrip:
    """A minimal but valid `PlannedTrip`, for tests that need one stored
    without running the whole agent graph."""
    return PlannedTrip(
        id=trip_id,
        request=request,
        itinerary=Itinerary(
            destination=request.destination or "Lisbon, Portugal",
            start_date=request.start_date,
            end_date=request.end_date,
            travelers=request.travelers,
            days=[],
        ),
        summary="A short stay.",
    )


class BlockedNetworkCall(RuntimeError):
    """A test tried to reach the network without mocking httpx first."""


@pytest.fixture(autouse=True)
def _block_real_network_calls(monkeypatch):
    """app/providers/aviationstack.py and tavily.py call load_dotenv() at
    import time, so real API keys are sitting in os.environ during every test
    run. A test that forgets to mock httpx would otherwise silently spend real
    API quota — fail loudly instead.

    Patches the top-level httpx.get/post convenience functions our fetch
    clients call, not httpx.Client's instance methods — so FastAPI's
    TestClient (which drives requests through its own Client instance) is
    unaffected.
    """

    def _blocked(*args, **kwargs):
        raise BlockedNetworkCall("A test attempted a real httpx.get/post call — mock it.")

    monkeypatch.setattr("httpx.get", _blocked)
    monkeypatch.setattr("httpx.post", _blocked)


class FakeHTTPResponse:
    """Stands in for an `httpx.Response` returned by a mocked get/post."""

    def __init__(self, json_data: dict, status_code: int = 200) -> None:
        self._json = json_data
        self.status_code = status_code
        self.text = json.dumps(json_data)

    def json(self) -> dict:
        return self._json

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise httpx.HTTPStatusError("error", request=None, response=self)


class FakeLLM:
    """A scripted stand-in for `app.agent.llm.LLM`.

    Responses can be queued generically (consumed in call order) or targeted
    at one agent by a substring of its system prompt — the latter is what lets
    a single fake drive the whole graph without caring what order the parallel
    flight/hotel nodes happen to run in.
    """

    def __init__(
        self,
        responses: list[str] | None = None,
        by_system: dict[str, list[str]] | None = None,
    ) -> None:
        self._queue = list(responses or [])
        self._by_system = {key: list(value) for key, value in (by_system or {}).items()}
        self.calls: list[dict] = []

    def complete(self, *, system: str, user: str, json_mode: bool = False) -> str:
        self.calls.append({"system": system, "user": user, "json_mode": json_mode})

        for key, queue in self._by_system.items():
            if key in system:
                if not queue:
                    raise AssertionError(f"FakeLLM ran out of responses for {key!r}")
                return queue.pop(0)

        if not self._queue:
            raise AssertionError("FakeLLM ran out of scripted responses")
        return self._queue.pop(0)
