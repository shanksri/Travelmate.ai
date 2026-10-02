"""Background jobs that run the ask graph and record what happens, step by
step, for the page to stream.

`POST /jobs` starts one and returns its id at once; `GET /jobs/{id}/events`
streams its events (app/api/routes/jobs.py). The job runs in its own thread,
independent of any connection: closing the tab doesn't stop it, the trip is
saved when it's ready, and reconnecting — a reload with the job id in the
page URL — replays every event so far and carries on from there.

Events, in the order a trip produces them (each a dict with a `type`):

    status      {"stage", "message"}   what's happening now
    intent      {"kind"}               trip, places or route
    flights     {"outbound_options", "return_options",
                 "outbound_source", "return_source"}
    hotels      {"options", "source"}
    day         {"day"}                one day, as soon as the AI finishes it
    days_reset  {}                     an attempt was rejected; days restart
    trip        {"trip"}               the finished trip, already saved
    city        {"city"}               one city's places (CityPlaces)
    places      {"places"}             a places answer
    route       {"route"}              a route answer
    error       {"status", "detail"}   what went wrong, as the API would say
    done        {}                     always last

Jobs live in this process only: a server restart forgets them (a saved trip
stays saved). Finished jobs are forgotten after `JOB_TTL_SECONDS`.
"""

import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field

from pydantic_core import to_jsonable_python

from app.agent.graph import build_ask_graph
from app.agent.state import PageChoices, initial_ask_state
from app.api.errors import describe_error

logger = logging.getLogger(__name__)

JOB_TTL_SECONDS = 3600


@dataclass
class Job:
    id: str
    created: float = field(default_factory=time.time)
    events: list[dict] = field(default_factory=list)
    done: bool = False
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, event: dict) -> None:
        jsonable = to_jsonable_python(event)
        with self._lock:
            self.events.append(jsonable)
            if jsonable.get("type") == "done":
                self.done = True

    def since(self, index: int) -> tuple[list[dict], bool]:
        """Events from `index` on, and whether the job has finished."""
        with self._lock:
            return list(self.events[index:]), self.done


class JobRegistry:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create(self) -> Job:
        job = Job(id=uuid.uuid4().hex[:16])
        now = time.time()
        with self._lock:
            for old in [j for j in self._jobs.values() if now - j.created > JOB_TTL_SECONDS]:
                if old.done:
                    del self._jobs[old.id]
            self._jobs[job.id] = job
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)


registry = JobRegistry()


def _events_for(step: str, update: dict) -> list[dict]:
    """The events a finished graph step means for the page. Steps that report
    progress as they go (days, cities) emit their own events instead."""
    if step == "flight_agent":
        results = update.get("flight_results") or {}
        return [
            {
                "type": "flights",
                "outbound_options": results.get("outbound_options", []),
                "return_options": results.get("return_options", []),
                "outbound_source": results.get("outbound_source"),
                "return_source": results.get("return_source"),
            }
        ]
    if step == "hotel_agent":
        results = update.get("hotel_results") or {}
        return [
            {
                "type": "hotels",
                "options": results.get("options", []),
                "source": results.get("source"),
            }
        ]
    if step == "package_trip":
        return [{"type": "trip", "trip": update["trip"]}]
    if step == "places":
        return [{"type": "places", "places": update["places"]}]
    if step == "route":
        return [{"type": "route", "route": update["route"]}]
    return []


def run_ask(
    job: Job,
    prompt: str,
    choices: PageChoices,
    *,
    provider,
    llm,
    max_itinerary_retries: int,
    save_trip: Callable,
) -> None:
    """Stream the ask graph into `job`. Never raises: a failure becomes an
    `error` event, and the job always ends with `done`."""
    graph = build_ask_graph(provider, llm, max_itinerary_retries)
    try:
        for mode, chunk in graph.stream(
            initial_ask_state(prompt, choices), stream_mode=["updates", "custom"]
        ):
            if mode == "custom":
                job.add(chunk)
                continue
            for step, update in chunk.items():
                if step == "package_trip":
                    # Saved before it's announced, so a page that reloads on
                    # the `trip` event finds it in the store.
                    save_trip(update["trip"])
                for event in _events_for(step, update or {}):
                    job.add(event)
    except Exception as exc:
        described = describe_error(exc)
        if described is None:
            logger.exception("job %s failed", job.id)
            described = (500, f"Something went wrong: {exc}")
        job.add({"type": "error", "status": described[0], "detail": described[1]})
    finally:
        job.add({"type": "done"})


def start_ask_job(prompt: str, choices: PageChoices) -> Job:
    """Start the ask graph for `prompt` on a background thread."""
    from app.agent.planner import _build_llm
    from app.core.config import get_settings
    from app.providers import get_provider
    from app.store import get_store

    settings = get_settings()
    job = registry.create()
    thread = threading.Thread(
        target=run_ask,
        args=(job, prompt, choices),
        kwargs={
            "provider": get_provider(),
            "llm": _build_llm(settings),
            "max_itinerary_retries": settings.max_itinerary_retries,
            "save_trip": get_store().save,
        },
        name=f"job-{job.id}",
        daemon=True,
    )
    thread.start()
    return job
