"""Turns one free-text trip request into a `TripRequest`.

Powers the frontend's single-prompt input ("Plan a 5 days Dubai trip from
Dhaka..."). The LLM extracts what it can; anything left unstated resolves to
a documented default here — in code, not by the model — so date arithmetic
stays exact and testable without depending on model behaviour.
"""

import json
from datetime import date, timedelta

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.agent.llm import LLM
from app.agent.prompts import PARSE_REQUEST_SYSTEM
from app.models.itinerary import Pace, TripRequest

# If the traveller gave a length ("5 days") but no start date, assume the
# trip starts this far out. If they gave neither, assume this length too.
DEFAULT_LEAD_DAYS = 14
DEFAULT_DURATION_DAYS = 5


class PromptParseError(ValueError):
    """The model's JSON was well-formed but didn't describe a usable trip.

    `feedback` is meant to be fed straight back into the next prompt attempt.
    """

    def __init__(self, feedback: str) -> None:
        super().__init__(feedback)
        self.feedback = feedback


class ParsedPrompt(BaseModel):
    """What the LLM extracts — deliberately looser than `TripRequest`: dates
    may be partial (a duration instead of an end date) or entirely absent."""

    destination: str | None = None
    origin: str | None = None
    start_date: date | None = None
    end_date: date | None = None
    duration_days: int | None = Field(default=None, ge=1, le=60)
    travelers: int = Field(default=1, ge=1, le=20)
    budget: float | None = Field(default=None, gt=0)
    interests: list[str] = Field(default_factory=list)
    pace: Pace = "balanced"
    notes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _null_means_use_the_default(cls, data: object) -> object:
        """The model sometimes emits an explicit `null` for a field even where
        the prompt names a default for it (travelers, interests, pace) —
        Pydantic only applies a field's default when the key is *absent*, not
        when it's present with value null, so left alone this fails
        validation instead of quietly falling back. Treat null the same as
        omitted for exactly these three."""
        if not isinstance(data, dict):
            return data
        defaults: dict[str, object] = {"travelers": 1, "interests": [], "pace": "balanced"}
        return {key: (defaults.get(key) if key in defaults and value is None else value)
                for key, value in data.items()}


def _resolve_dates(parsed: ParsedPrompt, today: date) -> tuple[date, date]:
    """Fill in whichever of start/end/duration the model left out."""
    if parsed.start_date and parsed.end_date:
        return parsed.start_date, parsed.end_date
    if parsed.start_date and parsed.duration_days:
        return parsed.start_date, parsed.start_date + timedelta(days=parsed.duration_days - 1)
    if parsed.end_date and parsed.duration_days:
        return parsed.end_date - timedelta(days=parsed.duration_days - 1), parsed.end_date

    start = parsed.start_date or (today + timedelta(days=DEFAULT_LEAD_DAYS))
    duration = parsed.duration_days or DEFAULT_DURATION_DAYS
    if parsed.end_date:
        # Only an end date was given — count backward from it instead of
        # ignoring it.
        return parsed.end_date - timedelta(days=duration - 1), parsed.end_date
    return start, start + timedelta(days=duration - 1)


def parse_trip_prompt(
    prompt: str,
    llm: LLM,
    *,
    today: date | None = None,
    max_retries: int = 1,
) -> TripRequest:
    """Parse one free-text request into a `TripRequest`, retrying once on a
    malformed or unusable response — the same feedback-loop shape as the
    itinerary agent's JSON validation."""
    today = today or date.today()
    feedback = ""
    last_error = ""

    for _ in range(max_retries + 1):
        user = prompt
        if feedback:
            user += f"\n\nYour previous attempt was rejected: {feedback}\nFix it and try again."

        raw = llm.complete(system=PARSE_REQUEST_SYSTEM, user=user, json_mode=True)
        try:
            parsed = ParsedPrompt.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            feedback = last_error = f"could not parse that as trip parameters: {exc}"
            continue

        start_date, end_date = _resolve_dates(parsed, today)
        try:
            return TripRequest(
                start_date=start_date,
                end_date=end_date,
                travelers=parsed.travelers,
                destination=parsed.destination,
                origin=parsed.origin,
                budget=parsed.budget,
                interests=parsed.interests,
                pace=parsed.pace,
                notes=parsed.notes,
            )
        except ValidationError as exc:
            feedback = last_error = f"those trip parameters were not valid: {exc}"
            continue

    raise PromptParseError(f"couldn't understand that trip request: {last_error}")
