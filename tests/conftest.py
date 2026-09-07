import json
from datetime import date
from types import SimpleNamespace

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


def text_block(text: str) -> SimpleNamespace:
    return SimpleNamespace(type="text", text=text)


def tool_use_block(name: str, tool_input: dict) -> SimpleNamespace:
    return SimpleNamespace(
        type="tool_use", name=name, id=f"toolu_{name}", input=tool_input
    )


class FakeRunner:
    """Stands in for the SDK tool runner.

    It yields the scripted messages and, for any `tool_use` block, calls the
    real tool the planner built — so the tools, the sink and the loop wiring
    are all exercised without touching the network.
    """

    def __init__(self, tools: list, script: list[list[SimpleNamespace]]) -> None:
        self._tools = {tool.name: tool for tool in tools}
        self._script = script
        self._pending: dict | None = None

    def __iter__(self):
        for blocks in self._script:
            stop = "tool_use" if any(b.type == "tool_use" for b in blocks) else "end_turn"
            results = []
            for block in blocks:
                if block.type == "tool_use":
                    output = self._tools[block.name].call(block.input)
                    results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": output,
                        }
                    )
            self._pending = {"role": "user", "content": results} if results else None
            yield SimpleNamespace(content=blocks, stop_reason=stop)

    def generate_tool_call_response(self) -> dict | None:
        return self._pending


class FakeClient:
    """A stub `anthropic.Anthropic` whose tool_runner replays a script."""

    def __init__(self, scripts: list[list[list[SimpleNamespace]]]) -> None:
        self._scripts = list(scripts)
        self.calls: list[dict] = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(tool_runner=self._runner))

    def _runner(self, **kwargs):
        # Snapshot the history: the planner keeps appending to the same list,
        # and the real SDK reads it once, at call time.
        self.calls.append(kwargs | {"messages": list(kwargs["messages"])})
        script = self._scripts.pop(0) if self._scripts else [[text_block("done")]]
        return FakeRunner(kwargs["tools"], script)


def submit_call(request: TripRequest, **overrides) -> SimpleNamespace:
    payload = itinerary_payload(request) | overrides
    return tool_use_block("submit_itinerary", {"itinerary_json": json.dumps(payload)})
