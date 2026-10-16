"""Jobs: the ask graph streamed into events, and the SSE endpoint serving
them. The LLM is scripted and streams its answers in small chunks, as the
real one does; travel data is the mock provider; Google Maps is faked."""

import json
import threading
from datetime import date

import pytest
from conftest import FakeLLM, route_json
from fastapi.testclient import TestClient

from app import jobs
from app.agent.prompts import (
    FINAL_RESPONSE_AGENT_SYSTEM,
    FLIGHT_AGENT_SYSTEM,
    ITINERARY_AGENT_SYSTEM,
    PARSE_REQUEST_SYSTEM,
    ROUTE_PLAN_SYSTEM,
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
            ROUTE_PLAN_SYSTEM: [route_json((cities[0], max(len(cities) - 1, 1)))],
            FLIGHT_AGENT_SYSTEM: ["ok"],
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
    (kochi,) = hotels["stays"]  # one town on this route
    assert kochi["city"] == "Kochi"
    assert kochi["options"] and kochi["source"]["status"] == "sample"


def test_the_route_arrives_before_flights_and_hotels(maps):
    job, _ = run(trip_llm())
    kinds = types(job)

    stops = next(e for e in job.events if e["type"] == "stops")
    assert [s["city"] for s in stops["route"]["stops"]] == ["Kochi"]
    assert kinds.index("stops") < kinds.index("flights")
    assert kinds.index("stops") < kinds.index("hotels")


def test_each_step_reports_starting_and_finishing(maps):
    """What the page's step list is drawn from."""
    job, _ = run(trip_llm(), include_restaurants=True)

    states = {}
    for e in job.events:
        if e["type"] == "status":
            states.setdefault(e["stage"], []).append(e["state"])
    for stage in ("interpret", "route_plan", "flights", "hotels", "itinerary", "summary"):
        assert states[stage][0] == "active" and states[stage][-1] == "done", stage
    assert states["restaurants"][0] == "active"


def test_unticked_steps_report_themselves_skipped(maps):
    job, _ = run(trip_llm(), include_flights=False, include_hotels=False)

    skipped = {e["stage"] for e in job.events if e["type"] == "status" and e["state"] == "skipped"}
    assert skipped == {"flights", "hotels", "budget", "restaurants"}  # no budget given


def test_a_budget_is_checked_before_the_summary(maps):
    job, saved = run(trip_llm(budget=5_000))

    budget = [e for e in job.events if e["type"] == "status" and e["stage"] == "budget"]
    assert budget[-1]["state"] == "warn" and budget[-1]["message"].startswith("Over by ₹")
    (trip, _), = saved
    assert trip.itinerary.budget_check.within is False
    assert any(m.startswith("budget_check: over") for m in trip.agent_trace)


def test_a_refused_request_ends_with_the_guardrails_message(maps):
    llm = StreamingFakeLLM(by_system={
        PARSE_REQUEST_SYSTEM: [_parse(verdict="not_travel", concerns=["It asks for a poem."])]
    })

    job, saved = run(llm, prompt="write me a poem about cats")

    error = next(e for e in job.events if e["type"] == "error")
    assert error["status"] == 422
    assert error["detail"] == (
        "This only plans trips and answers travel questions. It asks for a poem.")
    assert saved == [] and types(job)[-1] == "done"


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


# --- stopping a job ----------------------------------------------------------


class _Paused(StreamingFakeLLM):
    """Waits inside the route planner's call until told to go on, so a test
    can stop the job while a step is running."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inside = threading.Event()
        self.go_on = threading.Event()

    def complete(self, **kwargs):
        if kwargs["system"] == ROUTE_PLAN_SYSTEM:
            self.inside.set()
            self.go_on.wait(5)
        return super().complete(**kwargs)


def _start(llm, **choice_overrides):
    job = jobs.Job(id="test")
    saved = []
    thread = threading.Thread(
        target=jobs.run_ask,
        args=(job, "10 days in Kerala", choices(**choice_overrides)),
        kwargs={"provider": MockTravelProvider(), "llm": llm, "max_itinerary_retries": 2,
                "save_trip": saved.append},
    )
    thread.start()
    return job, saved, thread


def test_a_job_stopped_mid_step_ends_after_that_step_and_saves_nothing(maps):
    llm = _Paused(by_system=trip_llm().by_system_queues())
    job, saved, thread = _start(llm)

    assert llm.inside.wait(5)
    assert job.cancel() is True
    llm.go_on.set()  # the call already sent comes back...
    thread.join(5)

    # ...and nothing after it runs.
    kinds = types(job)
    assert kinds[-2:] == ["cancelled", "done"]
    assert job.events[-2] == {"type": "cancelled", "trip_saved": False}
    assert "error" not in kinds and "trip" not in kinds and saved == []
    asked = {c["system"] for c in llm.calls}
    assert ITINERARY_AGENT_SYSTEM not in asked and FINAL_RESPONSE_AGENT_SYSTEM not in asked


def test_a_stop_during_the_itinerary_closes_its_stream_early(maps):
    class _StopsMidAnswer(StreamingFakeLLM):
        job = None
        chunks_read = 0

        def stream(self, **kwargs):
            for chunk in super().stream(**kwargs):
                self.chunks_read += 1
                if self.chunks_read == 3:
                    self.job.cancel()  # the traveller presses Stop here
                yield chunk

    llm = _StopsMidAnswer(by_system=trip_llm(cities=["Kochi"] * 4).by_system_queues())
    job = jobs.Job(id="test")
    llm.job = job
    jobs.run_ask(job, "4 days in Kochi", choices(), provider=MockTravelProvider(), llm=llm,
                 max_itinerary_retries=2, save_trip=lambda trip: None)

    full_answer = len(trip_llm(cities=["Kochi"] * 4).by_system_queues()[
        ITINERARY_AGENT_SYSTEM][0]) // 7 + 1
    assert llm.chunks_read == 3 < full_answer
    assert types(job)[-2:] == ["cancelled", "done"]


def test_stopping_after_the_trip_is_saved_says_so(maps, monkeypatch):
    job = jobs.Job(id="test")

    def search_then_stop(query):
        job.cancel()  # Stop pressed while the restaurants are being found
        return PlacesAnswer(query=query, summary="", places=[])

    monkeypatch.setattr(google_maps, "search_places", search_then_stop)
    saved = []
    jobs.run_ask(job, "2 days in Kerala from Delhi", choices(include_restaurants=True),
                 provider=MockTravelProvider(), llm=trip_llm(), max_itinerary_retries=2,
                 save_trip=saved.append)

    assert len(saved) == 1
    assert {"type": "cancelled", "trip_saved": True} in job.events


def test_a_finished_job_cannot_be_stopped(maps):
    job, _ = run(trip_llm())

    assert job.cancel() is False
    assert "cancelled" not in types(job)


def test_delete_stops_a_running_job_and_reports_a_finished_one():
    client = TestClient(app)
    running = jobs.registry.create()
    finished = jobs.registry.create()
    finished.add({"type": "done"})

    assert client.delete(f"/jobs/{running.id}").json() == {"status": "stopping"}
    assert running.stop.is_set()
    assert client.delete(f"/jobs/{finished.id}").json() == {"status": "finished"}
    assert client.delete("/jobs/nope").status_code == 404
