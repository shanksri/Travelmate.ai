"""HTTP request and response bodies.

The domain models are the response shapes — there is nothing to translate yet,
so this module only adds what the wire needs on top of them.
"""

from pydantic import BaseModel

from app.models.itinerary import PlannedTrip, TripRequest


class PlanTripRequest(TripRequest):
    """A trip request, straight off the wire."""


class PlanTripResponse(BaseModel):
    trip: PlannedTrip


class TripListResponse(BaseModel):
    trips: list[PlannedTrip]


class HealthResponse(BaseModel):
    status: str
    model: str
    provider: str
    store: str
