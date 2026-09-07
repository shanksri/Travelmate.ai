import logging

import anthropic
from fastapi import APIRouter, HTTPException, status

from app.agent.planner import PlanningError, plan_trip
from app.api.schemas import PlanTripRequest, PlanTripResponse, TripListResponse
from app.models.itinerary import PlannedTrip
from app.store import get_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trips", tags=["trips"])


@router.post("/plan", response_model=PlanTripResponse)
def create_plan(payload: PlanTripRequest) -> PlanTripResponse:
    """Plan a trip.

    Defined `def`, not `async def`: planning is a blocking multi-call agent run,
    so FastAPI runs it in a worker thread instead of stalling the event loop.
    """
    try:
        trip = plan_trip(payload)
    except anthropic.AuthenticationError as exc:
        logger.warning("anthropic auth failed: %s", exc)
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "Claude credentials are missing or invalid — set ANTHROPIC_API_KEY.",
        ) from exc
    except anthropic.RateLimitError as exc:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS, "Rate limited by the Claude API."
        ) from exc
    except anthropic.APIStatusError as exc:
        logger.error("claude api error %s: %s", exc.status_code, exc.message)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"Claude API error: {exc.message}"
        ) from exc
    except anthropic.APIConnectionError as exc:
        raise HTTPException(
            status.HTTP_504_GATEWAY_TIMEOUT, "Could not reach the Claude API."
        ) from exc
    except PlanningError as exc:
        logger.error("planning failed: %s", exc)
        raise HTTPException(
            status.HTTP_502_BAD_GATEWAY, f"The planner did not finish: {exc}"
        ) from exc

    get_store().save(trip)
    return PlanTripResponse(trip=trip)


@router.get("", response_model=TripListResponse)
def list_trips() -> TripListResponse:
    return TripListResponse(trips=get_store().list_all())


@router.get("/{trip_id}", response_model=PlannedTrip)
def get_trip(trip_id: str) -> PlannedTrip:
    trip = get_store().get(trip_id)
    if trip is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No trip {trip_id}")
    return trip
