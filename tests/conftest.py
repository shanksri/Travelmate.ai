import json
from datetime import date

import pytest

from app.models.itinerary import TripRequest
from app.providers.mock import MockTravelProvider


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
