"""Progress events a graph step sends while it runs.

A step calls `emit({...})`; when the graph is being streamed
(`stream_mode="custom"`, see app/jobs.py) the event reaches the page at
once. Run any other way — `invoke`, or a node called directly in a test —
there's nobody listening, and `emit` does nothing.
"""

import logging

from langgraph.config import get_stream_writer

logger = logging.getLogger(__name__)


def emit(event: dict) -> None:
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
