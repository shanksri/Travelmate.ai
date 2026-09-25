"""HTTP request and response bodies.

The domain models are the response shapes — there is nothing to translate yet,
so this module only adds what the wire needs on top of them.
"""

from pydantic import BaseModel, Field

from app.models.itinerary import PlannedTrip, TripRequest


class PlanTripRequest(TripRequest):
    """A trip request, straight off the wire."""


class PlanFromPromptRequest(BaseModel):
    """One free-text trip request, e.g. "Plan a 5 day Dubai trip from Dhaka"."""

    prompt: str = Field(min_length=1, max_length=2000)
    # The frontend's checkboxes. Default on, so a caller that doesn't send
    # them gets the full plan, as before these existed.
    include_flights: bool = True
    include_hotels: bool = True


class ReviseTripRequest(BaseModel):
    """One change to an already-planned trip, e.g. "more activities on day 3"."""

    change_request: str = Field(min_length=1, max_length=2000)


class PlanTripResponse(BaseModel):
    trip: PlannedTrip


class TripListResponse(BaseModel):
    trips: list[PlannedTrip]


class TripHistoryResponse(BaseModel):
    """Every version of one trip, oldest first."""

    thread_id: str
    versions: list[PlannedTrip]


class HealthResponse(BaseModel):
    status: str
    model: str
    provider: str
    store: str
