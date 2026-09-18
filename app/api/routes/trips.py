import logging
from collections.abc import Callable

import openai
from fastapi import APIRouter, HTTPException, status

from app.agent.planner import PlanningError, plan_trip, plan_trip_from_prompt
from app.agent.prompt_parser import PromptParseError
from app.agent.reviser import RevisionError, revise_trip
from app.api.schemas import (
    PlanFromPromptRequest,
    PlanTripRequest,
    PlanTripResponse,
    ReviseTripRequest,
    TripHistoryResponse,
    TripListResponse,
)
from app.models.itinerary import PlannedTrip
from app.store import get_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trips", tags=["trips"])


def _run(planning_call: Callable[[], PlannedTrip]) -> PlanTripResponse:
    """Shared plumbing for both entry points below: run the call, translate
    the ways it can fail into HTTP errors, save what succeeds."""
    try:
        trip = planning_call()
    except openai.AuthenticationError as exc:
        logger.warning("openai auth failed: %s", exc)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "OpenAI credentials are missing or invalid — set OPENAI_API_KEY.",
        ) from exc
    except openai.RateLimitError as exc:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, "Rate limited by the OpenAI API."
        ) from exc
    except openai.APIStatusError as exc:
        logger.error("openai api error %s: %s", exc.status_code, exc.message)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"OpenAI API error: {exc.message}"
        ) from exc
    except openai.APIConnectionError as exc:
        raise HTTPException(
            status.HTTP_504_GATEWAY_TIMEOUT, "Could not reach the OpenAI API."
        ) from exc
    except PlanningError as exc:
        logger.error("planning failed: %s", exc)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"The planner did not finish: {exc}"
        ) from exc
    except RevisionError as exc:
        logger.error("revision failed: %s", exc)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"The change could not be applied: {exc}"
        ) from exc

    get_store().save(trip)
    return PlanTripResponse(trip=trip)


@router.post("/plan", response_model=PlanTripResponse)
def create_plan(payload: PlanTripRequest) -> PlanTripResponse:
    """Plan a trip from structured parameters.

    Defined `def`, not `async def`: planning runs the whole agent graph
    (several blocking LLM calls), so FastAPI runs it in a worker thread
    instead of stalling the event loop.
    """
    return _run(lambda: plan_trip(payload))


@router.post("/plan-from-prompt", response_model=PlanTripResponse)
def create_plan_from_prompt(payload: PlanFromPromptRequest) -> PlanTripResponse:
    """Plan a trip from one free-text sentence — parses it into the same
    structured parameters `create_plan` takes, then runs the same pipeline."""
    try:
        return _run(lambda: plan_trip_from_prompt(payload.prompt))
    except PromptParseError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"Couldn't understand that trip request: {exc}"
        ) from exc


@router.post("/{thread_id}/revise", response_model=PlanTripResponse)
def revise_plan(thread_id: str, payload: ReviseTripRequest) -> PlanTripResponse:
    """Apply one change to an already-planned trip.

    Builds on the newest version of `thread_id` and saves the result as the
    next version — the earlier ones stay exactly as they were, so the whole
    history is still there afterwards.
    """
    previous = get_store().get_latest(thread_id)
    if previous is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No trip {thread_id}")

    return _run(lambda: revise_trip(previous, payload.change_request))


@router.get("/{thread_id}/history", response_model=TripHistoryResponse)
def get_history(thread_id: str) -> TripHistoryResponse:
    """Every version of one trip, oldest first — version 1 is the original."""
    versions = get_store().list_versions(thread_id)
    if not versions:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No trip {thread_id}")

    return TripHistoryResponse(thread_id=thread_id, versions=versions)


@router.get("", response_model=TripListResponse)
def list_trips() -> TripListResponse:
    return TripListResponse(trips=get_store().list_all())


@router.get("/{trip_id}", response_model=PlannedTrip)
def get_trip(trip_id: str) -> PlannedTrip:
    trip = get_store().get(trip_id)
    if trip is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No trip {trip_id}")
    return trip
