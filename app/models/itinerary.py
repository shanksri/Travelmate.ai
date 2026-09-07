"""Domain models for a planned trip.

`Itinerary` is the contract between the agent and everything else: the agent
must produce one that validates, and the API returns exactly this shape.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, model_validator

ActivityCategory = Literal[
    "food", "sightseeing", "transit", "lodging", "activity", "rest"
]
Pace = Literal["relaxed", "balanced", "packed"]


class TripRequest(BaseModel):
    """What the traveller asks for."""

    start_date: date
    end_date: date
    travelers: int = Field(default=1, ge=1, le=20)
    destination: str | None = Field(
        default=None, description="Leave unset to let the agent suggest one."
    )
    origin: str | None = None
    budget_usd: float | None = Field(default=None, gt=0)
    interests: list[str] = Field(default_factory=list)
    pace: Pace = "balanced"
    notes: str | None = None

    @model_validator(mode="after")
    def _dates_are_ordered(self) -> "TripRequest":
        if self.end_date < self.start_date:
            raise ValueError("end_date must not be before start_date")
        if (self.end_date - self.start_date).days > 60:
            raise ValueError("trips longer than 60 days are not supported")
        return self

    @property
    def nights(self) -> int:
        return (self.end_date - self.start_date).days


class Activity(BaseModel):
    time: str = Field(description="Local start time, 24h 'HH:MM'.")
    title: str
    description: str = ""
    location: str | None = None
    category: ActivityCategory = "activity"
    estimated_cost_usd: float | None = Field(default=None, ge=0)


class DayPlan(BaseModel):
    day: int = Field(ge=1, description="1-indexed day of the trip.")
    date: date
    summary: str
    activities: list[Activity] = Field(default_factory=list)


class Itinerary(BaseModel):
    destination: str
    start_date: date
    end_date: date
    travelers: int
    days: list[DayPlan]
    currency: str = "USD"
    total_estimated_cost: float | None = Field(default=None, ge=0)
    notes: list[str] = Field(default_factory=list)


class PlannedTrip(BaseModel):
    """An itinerary plus how the agent arrived at it."""

    id: str
    request: TripRequest
    itinerary: Itinerary
    tool_calls: list[str] = Field(default_factory=list)
    summary: str = ""
