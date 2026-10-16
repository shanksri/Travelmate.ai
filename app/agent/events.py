"""Progress events a graph step sends while it runs.

A step calls `emit({...})`; when the graph is being streamed
(`stream_mode="custom"`, see app/jobs.py) the event reaches the page at
once. Run any other way — `invoke`, or a node called directly in a test —
there's nobody listening, and `emit` does nothing.

Stopping a job (the page's Stop button, `DELETE /jobs/{id}`) works through
here too. The job puts a `threading.Event` in the graph's config as
`configurable["cancel"]`; once it's set, the next `emit` (or
`stop_if_cancelled`) raises `JobCancelled`, which ends the run. A Python
thread can't be stopped from outside, so the steps stop themselves at these
points: every status line, every day written, and every chunk of the
streamed itinerary.
"""

import logging

from langgraph.config import get_config, get_stream_writer

logger = logging.getLogger(__name__)


class JobCancelled(Exception):
    """The traveller pressed Stop; raised at the next safe point."""


def cancelled() -> bool:
    """Whether this graph run has been asked to stop. False outside a run."""
    try:
        event = get_config().get("configurable", {}).get("cancel")
    except RuntimeError:  # not inside a graph run at all
        return False
    return bool(event is not None and event.is_set())


def stop_if_cancelled() -> None:
    if cancelled():
        raise JobCancelled()


def emit(event: dict) -> None:
    stop_if_cancelled()
    try:
        writer = get_stream_writer()
    except RuntimeError:  # not inside a graph run at all
        return
    writer(event)


def status(stage: str, message: str, state: str = "active") -> None:
    """One step's progress, for the page's step list: `stage` names the step,
    `state` is active, done, skipped or warn, and `message` is the line shown
    under it ("Searching Kochi, Madurai…", "9 outbound, 5 return")."""
    emit({"type": "status", "stage": stage, "state": state, "message": message})
