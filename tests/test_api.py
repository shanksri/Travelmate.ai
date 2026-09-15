from datetime import date

import pytest
from conftest import sample_planned_trip
from fastapi.testclient import TestClient

from app.main import app
from app.models.itinerary import TripRequest
from app.store import get_store


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


def test_health_reports_the_configured_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["provider"] == "mock"
    assert body["store"] == "memory"


def test_root_serves_the_frontend(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "prompt-form" in response.text


def test_static_assets_are_served(client):
    css = client.get("/static/style.css")
    js = client.get("/static/app.js")
    assert css.status_code == 200
    assert js.status_code == 200
    assert "text/css" in css.headers["content-type"]


def test_root_is_not_in_the_openapi_schema(client):
    schema = client.get("/openapi.json").json()
    assert "/" not in schema["paths"]


def test_plan_rejects_backwards_dates(client):
    response = client.post(
        "/trips/plan",
        json={"start_date": "2026-04-13", "end_date": "2026-04-10", "travelers": 2},
    )
    assert response.status_code == 422


def test_unknown_trip_is_a_404(client):
    assert client.get("/trips/nope").status_code == 404


def test_saved_trips_are_retrievable(client, trip_request):
    get_store().save(sample_planned_trip(trip_request))

    body = client.get("/trips/abc123").json()
    assert body["itinerary"]["destination"] == "Lisbon, Portugal"
    assert any(t["id"] == "abc123" for t in client.get("/trips").json()["trips"])


def test_planning_failure_surfaces_as_502(client, monkeypatch):
    from app.agent.planner import PlanningError

    def boom(_payload):
        raise PlanningError("never submitted an itinerary")

    monkeypatch.setattr("app.api.routes.trips.plan_trip", boom)

    response = client.post(
        "/trips/plan",
        json={
            "start_date": "2026-04-10",
            "end_date": "2026-04-12",
            "travelers": 1,
            "destination": "Rome, Italy",
        },
    )
    assert response.status_code == 502
    assert "never submitted" in response.json()["detail"]


def test_trip_request_rejects_absurd_durations():
    with pytest.raises(ValueError, match="60 days"):
        TripRequest(start_date=date(2026, 1, 1), end_date=date(2026, 6, 1))


def test_plan_from_prompt_delegates_to_the_parser_and_planner(client, trip_request, monkeypatch):
    trip = sample_planned_trip(trip_request)
    monkeypatch.setattr(
        "app.api.routes.trips.plan_trip_from_prompt", lambda prompt: trip if prompt else None
    )

    response = client.post("/trips/plan-from-prompt", json={"prompt": "Dubai for 5 days"})

    assert response.status_code == 200
    assert response.json()["trip"]["id"] == trip.id


def test_plan_from_prompt_rejects_an_unparseable_request(client, monkeypatch):
    from app.agent.prompt_parser import PromptParseError

    def boom(_prompt):
        raise PromptParseError("no destination or dates in that at all")

    monkeypatch.setattr("app.api.routes.trips.plan_trip_from_prompt", boom)

    response = client.post("/trips/plan-from-prompt", json={"prompt": "???"})

    assert response.status_code == 422
    assert "couldn't understand" in response.json()["detail"].lower()


def test_plan_from_prompt_rejects_an_empty_prompt(client):
    response = client.post("/trips/plan-from-prompt", json={"prompt": ""})
    assert response.status_code == 422
