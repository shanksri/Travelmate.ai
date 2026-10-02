"""POST /jobs and GET /jobs/{id}/events — the prompt box, streamed.

The page starts a job and gets its id back at once, then listens to the
job's events over Server-Sent Events: what's being searched, flights and
hotels as they're found, each day of the plan as it's written, the saved
trip, and each city's places. See app/jobs.py for the events themselves.

The event stream can be dropped and reopened at any time. Each event's SSE
`id` is its position, so a browser reconnecting sends `Last-Event-ID` and
picks up after it; a fresh connection (a reload) replays from the start.
"""

import asyncio
import json

from fastapi import APIRouter, Header, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.agent.assistant import page_choices
from app.api.schemas import PlanFromPromptRequest
from app.jobs import registry, start_ask_job

router = APIRouter(prefix="/jobs", tags=["jobs"])

POLL_SECONDS = 0.1
KEEPALIVE_SECONDS = 15


class JobStarted(BaseModel):
    job_id: str


@router.post("", response_model=JobStarted, status_code=status.HTTP_202_ACCEPTED)
def start_job(payload: PlanFromPromptRequest) -> JobStarted:
    choices = page_choices(
        include_flights=payload.include_flights,
        include_hotels=payload.include_hotels,
        include_restaurants=payload.include_restaurants,
        start_date=payload.start_date,
        end_date=payload.end_date,
    )
    return JobStarted(job_id=start_ask_job(payload.prompt, choices).id)


@router.get("/{job_id}/events")
async def job_events(
    job_id: str,
    request: Request,
    last_event_id: str | None = Header(default=None),
) -> StreamingResponse:
    job = registry.get(job_id)
    if job is None:
        raise HTTPException(
            status.HTTP_404_NOT_FOUND,
            f"No job {job_id} — it may have finished over an hour ago, or the server restarted.",
        )
    start = int(last_event_id) + 1 if last_event_id and last_event_id.isdigit() else 0

    async def stream():
        index = start
        quiet = 0.0
        while True:
            events, done = job.since(index)
            for event in events:
                yield f"id: {index}\ndata: {json.dumps(event)}\n\n"
                index += 1
            if done and not events:
                return
            if events:
                quiet = 0.0
            elif quiet >= KEEPALIVE_SECONDS:
                yield ": keepalive\n\n"
                quiet = 0.0
            if await request.is_disconnected():
                return
            await asyncio.sleep(POLL_SECONDS)
            quiet += POLL_SECONDS

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
