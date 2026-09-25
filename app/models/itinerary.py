"""Domain models for a planned trip.

`Itinerary` is the contract between the agent and everything else: the agent
must produce one that validates, and the API returns exactly this shape.

Money fields are deliberately currency-neutral (`total`, `nightly`, `budget`,
`estimated_cost`) rather than named after a currency. What they're
denominated in is `Itinerary.currency`, which is INR everywhere now. Trips
stored before that switch carry `currency: "USD"` and their old `*_usd`
field names, so each model below maps those old names onto the new ones —
the amounts are left exactly as they were, because an old trip really was
priced in dollars and relabelling it rupees would be a lie.
"""

from datetime import date, datetime, timedelta
from typing import Literal

from pydantic import BaseModel, Field, model_validator

ActivityCategory = Literal[
    "food", "sightseeing", "transit", "lodging", "activity", "rest"
]
Pace = Literal["relaxed", "balanced", "packed"]

DEFAULT_CURRENCY = "INR"


def _rename_legacy_usd_fields(data: object, renames: dict[str, str]) -> object:
    """Map pre-INR `*_usd` keys onto their currency-neutral replacements.

    Only fills a new key that isn't already present, so a current payload is
    untouched and a legacy one loads without its amounts being reinterpreted.
    """
    if not isinstance(data, dict):
        return data

    patched = dict(data)
    for old, new in renames.items():
        if old in patched and new not in patched:
            patched[new] = patched.pop(old)
    return patched


class TripRequest(BaseModel):
    """What the traveller asks for."""

    start_date: date
    end_date: date
    travelers: int = Field(default=1, ge=1, le=20)
    destination: str | None = Field(
        default=None, description="Leave unset to let the agent suggest one."
    )
    origin: str | None = None
    budget: float | None = Field(default=None, gt=0, description="Total for the party.")
    interests: list[str] = Field(default_factory=list)
    pace: Pace = "balanced"
    notes: str | None = None
    # Whether to search for flights / lodging at all. Default on, so every
    # entry point that doesn't ask (the CLI, POST /trips/plan, trips stored
    # before these existed) behaves as it always has; the frontend sends both
    # explicitly from its checkboxes.
    include_flights: bool = True
    include_hotels: bool = True

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_usd_names(cls, data: object) -> object:
        return _rename_legacy_usd_fields(data, {"budget_usd": "budget"})

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

    @property
    def dates(self) -> list[date]:
        """One entry per calendar day of the trip, start to end inclusive —
        the single source of truth for "which dates need a `days` entry",
        shared by the itinerary prompt and its own response validation so
        neither has to re-derive it (and risk disagreeing) from arithmetic."""
        return [self.start_date + timedelta(days=offset) for offset in range(self.nights + 1)]


class Activity(BaseModel):
    time: str = Field(description="Local start time, 24h 'HH:MM'.")
    title: str
    description: str = ""
    location: str | None = None
    category: ActivityCategory = "activity"
    estimated_cost: float | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_usd_names(cls, data: object) -> object:
        return _rename_legacy_usd_fields(data, {"estimated_cost_usd": "estimated_cost"})


class DayPlan(BaseModel):
    day: int = Field(ge=1, description="1-indexed day of the trip.")
    date: date
    summary: str
    activities: list[Activity] = Field(default_factory=list)


class FlightLeg(BaseModel):
    """One flight leg, built from a real search result — never LLM-authored,
    so the carrier, price and timing here can't be hallucinated."""

    carrier: str
    origin: str
    destination: str
    depart_date: date
    departure_at: datetime | None = Field(
        default=None,
        description="Local departure time, when the source knows it. Mock data doesn't.",
    )
    stops: int = 0
    duration_hours: float | None = None
    price_per_person: float | None = Field(default=None, ge=0)
    total: float | None = Field(default=None, ge=0)
    rationale: str | None = Field(
        default=None, description="The flight agent's reasoning for this pick."
    )

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_usd_names(cls, data: object) -> object:
        return _rename_legacy_usd_fields(
            data, {"price_usd_per_person": "price_per_person", "total_usd": "total"}
        )


class LodgingOption(BaseModel):
    """One candidate place to stay, built from a real search result."""

    name: str
    tier: str | None = None
    rating: float | None = None
    neighbourhood: str | None = None
    nightly: float | None = Field(default=None, ge=0)
    total: float | None = Field(default=None, ge=0)

    @model_validator(mode="before")
    @classmethod
    def _accept_legacy_usd_names(cls, data: object) -> object:
        return _rename_legacy_usd_fields(data, {"nightly_usd": "nightly", "total_usd": "total"})


class DraftItinerary(BaseModel):
    """What the itinerary agent itself is asked to produce — just the days.

    Flights and lodging are assembled separately from real search results
    (see app/agent/nodes.py) and merged in to make the full `Itinerary`; the
    agent never re-types facts it could get wrong.
    """

    days: list[DayPlan]
    notes: list[str] = Field(default_factory=list)


class Itinerary(BaseModel):
    destination: str
    start_date: date
    end_date: date
    travelers: int
    outbound_flight: FlightLeg | None = None
    return_flight: FlightLeg | None = None
    # Every option the search returned for each leg, cheapest first — what the
    # cheapest-vs-fastest comparison is built from. `outbound_flight` /
    # `return_flight` above remain the one actually selected and costed.
    outbound_options: list[FlightLeg] = Field(default_factory=list)
    return_options: list[FlightLeg] = Field(default_factory=list)
    lodging_options: list[LodgingOption] = Field(default_factory=list)
    days: list[DayPlan]
    currency: str = DEFAULT_CURRENCY
    total_estimated_cost: float | None = Field(default=None, ge=0)
    notes: list[str] = Field(default_factory=list)


class PlannedTrip(BaseModel):
    """One version of an itinerary, plus how the agent pipeline arrived at it.

    `id` identifies this version; `thread_id` groups every version of the same
    trip together and is what survives a revision. Planning a new trip starts
    a new thread at version 1; each accepted change appends version 2, 3, ...
    under the same thread, so the store keeps the whole history rather than
    overwriting (see app/store.py).
    """

    id: str
    thread_id: str
    version: int = Field(default=1, ge=1)
    change_note: str | None = Field(
        default=None,
        description="What was asked for to produce this version. None on the original.",
    )
    request: TripRequest
    itinerary: Itinerary
    agent_trace: list[str] = Field(default_factory=list)
    summary: str = ""

    @model_validator(mode="before")
    @classmethod
    def _legacy_rows_are_version_one_of_their_own_thread(cls, data: object) -> object:
        """Trips stored before versioning existed have no `thread_id`. Treating
        each one as version 1 of a thread named after its own id is exactly
        right — they have no history — and means old rows still load."""
        if isinstance(data, dict) and "thread_id" not in data and "id" in data:
            return {**data, "thread_id": data["id"]}
        return data
