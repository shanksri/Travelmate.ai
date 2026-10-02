"""Jobs: the ask graph streamed into events, and the SSE endpoint serving
them. The LLM is scripted and streams its answers in small chunks, as the
real one does; travel data is the mock provider; Google Maps is faked."""

import json
from datetime import date

import pytest
from conftest import FakeLLM
from fastapi.testclient import TestClient

from app import jobs
from app.agent.prompts import (
    FINAL_RESPONSE_AGENT_SYSTEM,
    FLIGHT_AGENT_SYSTEM,
    HOTEL_AGENT_SYSTEM,
    ITINERARY_AGENT_SYSTEM,
    PARSE_REQUEST_SYSTEM,
)
from app.agent.state import PageChoices
from app.main import app
from app.models.maps import PlacesAnswer
from app.providers import google_maps
from app.providers.mock import MockTravelProvider

START = date(2027, 1, 10)


class StreamingFakeLLM(FakeLLM):
    """A FakeLLM that also streams, a few characters at a time."""

    def stream(self, **kwargs):
        text = self.complete(**kwargs)
        for i in range(0, len(text), 7):
            yield text[i : i + 7]


def _days(cities, start=START):
    return json.dumps(
        {
            "days": [
                {
                    "day": i + 1,
                    "date": date.fromordinal(start.toordinal() + i).isoformat(),
                    "summary": f"Day in {city}",
                    "city": city,
                    "activities": [{"time": "09:00", "title": f"Walk {city}"}],
                }
                for i, city in enumerate(cities)
            ],
            "notes": [],
        }
    )


def _parse(**fields):
    return json.dumps({"intent": "trip", "destination": None, "origin": None} | fields)


def trip_llm(cities=("Kochi", "Munnar"), itinerary_answers=None, **parse_fields):
    parse = {"destination": "Kerala", "origin": "Delhi", "start_date": START.isoformat(),
             "end_date": date.fromordinal(START.toordinal() + len(cities) - 1).isoformat()}
    return StreamingFakeLLM(
        by_system={
            PARSE_REQUEST_SYSTEM: [_parse(**(parse | parse_fields))],
            FLIGHT_AGENT_SYSTEM: ["ok"],
            HOTEL_AGENT_SYSTEM: ["ok"],
            ITINERARY_AGENT_SYSTEM: itinerary_answers or [_days(cities)],
            FINAL_RESPONSE_AGENT_SYSTEM: ["Two days in Kerala."],
        }
    )


def choices(**overrides) -> PageChoices:
    return PageChoices(include_flights=True, include_hotels=True, include_restaurants=False,
                       start_date=None, end_date=None) | overrides


@pytest.fixture
def maps(monkeypatch):
    monkeypatch.setattr(
        google_maps,
        "search_places",
        lambda query: PlacesAnswer(query=query, summary=f"About {query}", places=[]),
    )


def run(llm, prompt="2 days in Kerala from Delhi", **choice_overrides):
    job = jobs.Job(id="test")
    saved = []

    def save(trip):
        saved.append((trip, len(job.events)))

    jobs.run_ask(job, prompt, choices(**choice_overrides), provider=MockTravelProvider(),
                 llm=llm, max_itinerary_retries=2, save_trip=save)
    return job, saved


def types(job):
    return [e["type"] for e in job.events]


def test_a_trip_streams_each_step_then_the_saved_trip(maps):
    job, saved = run(trip_llm(), include_restaurants=True)
    kinds = types(job)

    assert kinds[-1] == "done" and job.done
    assert "error" not in kinds
    for expected in ("status", "intent", "flights", "hotels", "day", "trip", "city"):
        assert expected in kinds, expected
    # The days stream out before the finished trip, the cities after it.
    assert kinds.index("day") < kinds.index("trip") < kinds.index("city")
    assert [e["day"]["city"] for e in job.events if e["type"] == "day"] == ["Kochi", "Munnar"]
    trip_event = next(e for e in job.events if e["type"] == "trip")
    # Saved before it was announced, so a reload right then finds it.
    (trip, events_before_save), = saved
    assert trip.id == trip_event["trip"]["id"]
    assert job.events[events_before_save]["type"] == "trip"
    assert sorted(e["city"]["city"] for e in job.events if e["type"] == "city") == [
        "Kochi", "Munnar"]


def test_flights_and_hotels_arrive_with_their_sources(maps):
    job, _ = run(trip_llm())

    flights = next(e for e in job.events if e["type"] == "flights")
    hotels = next(e for e in job.events if e["type"] == "hotels")
    assert flights["outbound_options"] and flights["outbound_source"]["status"] == "sample"
    assert hotels["options"] and hotels["source"]["status"] == "sample"


def test_every_event_is_plain_json(maps):
    job, _ = run(trip_llm(), include_restaurants=True)

    assert json.loads(json.dumps(job.events)) == job.events


def test_a_rejected_attempt_resets_the_days_already_shown(maps):
    wrong = _days(["Kochi"])  # one day for a two-day trip: rejected
    llm = trip_llm(itinerary_answers=[wrong, _days(["Kochi", "Munnar"])])

    job, _ = run(llm)
    kinds = types(job)

    assert "days_reset" in kinds
    after_reset = kinds[kinds.index("days_reset"):]
    assert after_reset.count("day") == 2


def test_a_places_request_streams_its_answer(maps):
    llm = StreamingFakeLLM(by_system={
        PARSE_REQUEST_SYSTEM: [_parse(intent="places", places_query="cafes in Puri")]
    })

    job, saved = run(llm, prompt="cafes in puri")

    assert [e for e in job.events if e["type"] == "intent"][0]["kind"] == "places"
    places = next(e for e in job.events if e["type"] == "places")
    assert places["places"]["summary"] == "About cafes in Puri"
    assert saved == [] and types(job)[-1] == "done"


def test_a_known_failure_becomes_an_error_event_worded_like_the_api(maps):
    job, saved = run(trip_llm(destination=None), prompt="a trip from Delhi")

    error = next(e for e in job.events if e["type"] == "error")
    assert error["status"] == 422
    assert "no destination was named" in error["detail"]
    assert saved == [] and types(job)[-1] == "done"


def test_an_unexpected_failure_is_still_reported_and_finishes_the_job():
    class Broken(StreamingFakeLLM):
        def complete(self, **kwargs):
            raise KeyError("bug")

    job, _ = run(Broken())

    error = next(e for e in job.events if e["type"] == "error")
    assert error["status"] == 500
    assert job.done


# --- the HTTP side ---------------------------------------------------------------


@pytest.fixture
def client():
    return TestClient(app)


def _job_with(events, done=True):
    job = jobs.registry.create()
    for event in events:
        job.add(event)
    if done:
        job.add({"type": "done"})
    return job


def _read(client, url, headers=None):
    with client.stream("GET", url, headers=headers or {}) as response:
        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        body = "".join(response.iter_text())
    blocks = [b for b in body.split("\n\n") if b.startswith("id:")]
    return [
        (int(b.split("\n")[0][4:]), json.loads(b.split("\n")[1][6:])) for b in blocks
    ]


def test_the_event_stream_replays_every_event_with_its_position(client):
    job = _job_with([{"type": "status", "message": "a"}, {"type": "intent", "kind": "trip"}])

    events = _read(client, f"/jobs/{job.id}/events")

    assert [i for i, _ in events] == [0, 1, 2]
    assert [e["type"] for _, e in events] == ["status", "intent", "done"]


def test_a_reconnect_picks_up_after_the_last_event_it_saw(client):
    job = _job_with([{"type": "status", "message": "a"}, {"type": "intent", "kind": "trip"}])

    events = _read(client, f"/jobs/{job.id}/events", headers={"Last-Event-ID": "0"})

    assert [e["type"] for _, e in events] == ["intent", "done"]


def test_an_unknown_job_is_a_404(client):
    response = client.get("/jobs/nope/events")

    assert response.status_code == 404


def test_starting_a_job_returns_its_id_at_once(client, monkeypatch):
    started = {}

    def fake_start(prompt, choices):
        started.update(prompt=prompt, choices=choices)
        return jobs.Job(id="abc123")

    monkeypatch.setattr("app.api.routes.jobs.start_ask_job", fake_start)

    response = client.post(
        "/jobs", json={"prompt": "Goa", "include_flights": False, "include_restaurants": True}
    )

    assert response.status_code == 202
    assert response.json() == {"job_id": "abc123"}
    assert started["prompt"] == "Goa"
    assert started["choices"]["include_flights"] is False
    assert started["choices"]["include_restaurants"] is True


def test_starting_a_job_validates_like_ask(client):
    response = client.post("/jobs", json={"prompt": "Goa", "start_date": "2027-01-10"})

    assert response.status_code == 422
