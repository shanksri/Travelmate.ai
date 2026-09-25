"""HTTP request and response bodies.

The domain models are the response shapes — there is nothing to translate yet,
so this module only adds what the wire needs on top of them.
"""

from datetime import date

from pydantic import BaseModel, Field, model_validator

from app.models.itinerary import MAX_TRIP_NIGHTS, PlannedTrip, TripRequest


class PlanTripRequest(TripRequest):
    """A trip request, straight off the wire."""


class PlanFromPromptRequest(BaseModel):
    """One free-text trip request, e.g. "Plan a 5 day Dubai trip from Dhaka"."""

    prompt: str = Field(min_length=1, max_length=2000)
    # The frontend's checkboxes. Default on, so a caller that doesn't send
    # them gets the full plan, as before these existed.
    include_flights: bool = True
    include_hotels: bool = True
    # The frontend's calendar pickers. Optional: when both are given they
    # replace whatever dates the sentence implied; when neither is, the
    # sentence (or the parser's defaults) decides, as before.
    start_date: date | None = None
    end_date: date | None = None

    @model_validator(mode="after")
    def _dates_come_as_a_valid_pair(self) -> "PlanFromPromptRequest":
        if (self.start_date is None) != (self.end_date is None):
            raise ValueError("give both start_date and end_date, or neither")
        if self.start_date is None or self.end_date is None:
            return self
        if self.start_date < date.today():
            raise ValueError("start_date is in the past")
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be before start_date")
        if (self.end_date - self.start_date).days > MAX_TRIP_NIGHTS:
            raise ValueError("trips longer than 60 days are not supported")
        return self


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
