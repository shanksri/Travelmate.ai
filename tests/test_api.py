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
        "app.api.routes.trips.plan_trip_from_prompt",
        lambda prompt, **kwargs: trip if prompt else None,
    )

    response = client.post("/trips/plan-from-prompt", json={"prompt": "Dubai for 5 days"})

    assert response.status_code == 200
    assert response.json()["trip"]["id"] == trip.id


def test_plan_from_prompt_rejects_an_unparseable_request(client, monkeypatch):
    from app.agent.prompt_parser import PromptParseError

    def boom(_prompt, **kwargs):
        raise PromptParseError("no destination or dates in that at all")

    monkeypatch.setattr("app.api.routes.trips.plan_trip_from_prompt", boom)

    response = client.post("/trips/plan-from-prompt", json={"prompt": "???"})

    assert response.status_code == 422
    assert "couldn't understand" in response.json()["detail"].lower()


def test_plan_from_prompt_passes_the_checkboxes_through(client, trip_request, monkeypatch):
    seen = {}

    def fake(prompt, **kwargs):
        seen.update(kwargs)
        return sample_planned_trip(trip_request)

    monkeypatch.setattr("app.api.routes.trips.plan_trip_from_prompt", fake)

    client.post(
        "/trips/plan-from-prompt",
        json={"prompt": "Goa for 3 days", "include_flights": True, "include_hotels": False},
    )

    assert seen == {"include_flights": True, "include_hotels": False}


def test_plan_from_prompt_includes_both_when_a_caller_does_not_say(
    client, trip_request, monkeypatch
):
    """Callers that predate the checkboxes still get the full plan."""
    seen = {}

    def fake(prompt, **kwargs):
        seen.update(kwargs)
        return sample_planned_trip(trip_request)

    monkeypatch.setattr("app.api.routes.trips.plan_trip_from_prompt", fake)

    client.post("/trips/plan-from-prompt", json={"prompt": "Goa for 3 days"})

    assert seen == {"include_flights": True, "include_hotels": True}


def test_plan_from_prompt_rejects_an_empty_prompt(client):
    response = client.post("/trips/plan-from-prompt", json={"prompt": ""})
    assert response.status_code == 422


# --- revisions and history ---------------------------------------------------


def test_revise_saves_a_new_version_and_keeps_the_old_one(client, trip_request, monkeypatch):
    original = sample_planned_trip(trip_request, trip_id="rev-v1", thread_id="rev")
    get_store().save(original)

    def fake_revise(previous, change_request):
        return sample_planned_trip(
            trip_request,
            trip_id="rev-v2",
            thread_id=previous.thread_id,
            version=previous.version + 1,
            change_note=change_request,
        )

    monkeypatch.setattr("app.api.routes.trips.revise_trip", fake_revise)

    response = client.post("/trips/rev/revise", json={"change_request": "more on day 3"})

    assert response.status_code == 200
    body = response.json()["trip"]
    assert body["version"] == 2
    assert body["thread_id"] == "rev"
    assert body["change_note"] == "more on day 3"
    assert client.get("/trips/rev-v1").status_code == 200  # original still there


def test_revise_builds_on_the_latest_version(client, trip_request, monkeypatch):
    for n in (1, 2):
        get_store().save(
            sample_planned_trip(
                trip_request, trip_id=f"l-v{n}", thread_id="latest", version=n
            )
        )
    seen = {}

    def fake_revise(previous, change_request):
        seen["version"] = previous.version
        return sample_planned_trip(
            trip_request, trip_id="l-v3", thread_id="latest", version=previous.version + 1
        )

    monkeypatch.setattr("app.api.routes.trips.revise_trip", fake_revise)

    client.post("/trips/latest/revise", json={"change_request": "again"})

    assert seen["version"] == 2


def test_revise_on_an_unknown_thread_is_a_404(client):
    response = client.post("/trips/nope/revise", json={"change_request": "more on day 3"})
    assert response.status_code == 404


def test_revise_rejects_an_empty_change_request(client, trip_request):
    get_store().save(sample_planned_trip(trip_request, trip_id="e-v1", thread_id="empty"))

    response = client.post("/trips/empty/revise", json={"change_request": ""})

    assert response.status_code == 422


def test_revision_failure_surfaces_as_502(client, trip_request, monkeypatch):
    from app.agent.reviser import RevisionError

    get_store().save(sample_planned_trip(trip_request, trip_id="f-v1", thread_id="fails"))

    def boom(previous, change_request):
        raise RevisionError("gave up after 3 attempt(s)")

    monkeypatch.setattr("app.api.routes.trips.revise_trip", boom)

    response = client.post("/trips/fails/revise", json={"change_request": "more on day 3"})

    assert response.status_code == 502
    assert "could not be applied" in response.json()["detail"]


def test_a_declined_revision_is_a_422_with_the_reason_and_saves_nothing(
    client, trip_request, monkeypatch
):
    from app.agent.reviser import RevisionDeclined

    get_store().save(sample_planned_trip(trip_request, trip_id="d-v1", thread_id="declined"))

    def decline(previous, change_request):
        raise RevisionDeclined("Rameshwaram is too far for the days left.")

    monkeypatch.setattr("app.api.routes.trips.revise_trip", decline)

    response = client.post("/trips/declined/revise", json={"change_request": "add rameshwaram"})

    assert response.status_code == 422
    assert response.json()["detail"] == (
        "Couldn't apply this change: Rameshwaram is too far for the days left."
    )
    versions = client.get("/trips/declined/history").json()["versions"]
    assert [v["version"] for v in versions] == [1]


def test_history_returns_every_version_oldest_first(client, trip_request):
    for n in (1, 2, 3):
        get_store().save(
            sample_planned_trip(
                trip_request,
                trip_id=f"h-v{n}",
                thread_id="hist",
                version=n,
                change_note=None if n == 1 else f"change {n}",
            )
        )

    body = client.get("/trips/hist/history").json()

    assert body["thread_id"] == "hist"
    assert [v["version"] for v in body["versions"]] == [1, 2, 3]
    assert body["versions"][0]["change_note"] is None
    assert body["versions"][2]["change_note"] == "change 3"


def test_history_on_an_unknown_thread_is_a_404(client):
    assert client.get("/trips/nope/history").status_code == 404
