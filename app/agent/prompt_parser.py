"""Turns one free-text trip request into a `TripRequest`.

Powers the frontend's single-prompt input ("Plan a 5 days Dubai trip from
Dhaka..."). The LLM extracts what it can; anything left unstated resolves to
a documented default here — in code, not by the model — so date arithmetic
stays exact and testable without depending on model behaviour.
"""

import json
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Literal

from pydantic import BaseModel, Field, ValidationError, model_validator

from app.agent.llm import LLM
from app.agent.prompts import PARSE_REQUEST_SYSTEM
from app.models.itinerary import Pace, TripRequest
from app.models.maps import TravelMode

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


Intent = Literal["trip", "places", "route"]

# The input guardrail's verdict, read in the same call as everything else
# (PARSE_REQUEST_SYSTEM). The first three refuse the request; "caution"
# plans it and passes the concerns along.
Verdict = Literal["ok", "caution", "not_travel", "harmful", "impossible"]
REFUSED: dict[str, str] = {
    "not_travel": "This only plans trips and answers travel questions.",
    "harmful": "This can't help with that.",
    "impossible": "This trip can't be planned as asked.",
}


class RequestRefused(Exception):
    """The input guardrail turned the request down; `str()` is the message
    for the traveller. Not a parse error: the request was understood."""

    def __init__(self, verdict: str, message: str) -> None:
        super().__init__(message)
        self.verdict = verdict


class ParsedPrompt(BaseModel):
    """What the LLM extracts — deliberately looser than `TripRequest`: dates
    may be partial (a duration instead of an end date) or entirely absent."""

    # "trip" is the default whenever the model leaves it out: everything that
    # predates intents was a trip.
    intent: Intent = "trip"
    places_query: str | None = None
    travel_mode: TravelMode | None = None
    places_per_city: str | None = None

    destination: str | None = None
    origin: str | None = None
    start_date: date | None = None
    # "In July": a month with no date. Found by the evals: before this, the
    # month was dropped and the trip planned two weeks out, in October.
    travel_month: int | None = Field(default=None, ge=1, le=12)
    end_date: date | None = None
    duration_days: int | None = Field(default=None, ge=1, le=60)
    travelers: int = Field(default=1, ge=1, le=20)
    budget: float | None = Field(default=None, gt=0)
    interests: list[str] = Field(default_factory=list)
    pace: Pace = "balanced"
    notes: str | None = None

    # The guardrail: refuse, plan with concerns, or just plan. Last, so the
    # model writes it after reading the request: as the first field, it
    # judged "Kerala in July" before reading the month, and never saw the
    # monsoon (0 of 3 runs flagged it; the evals found it).
    verdict: Verdict = "ok"
    concerns: list[str] = Field(default_factory=list)

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
        defaults: dict[str, object] = {
            "travelers": 1,
            "interests": [],
            "pace": "balanced",
            "intent": "trip",
            "verdict": "ok",
            "concerns": [],
        }
        return {key: (defaults.get(key) if key in defaults and value is None else value)
                for key, value in data.items()}


def _month_start(month: int, today: date) -> date:
    """The first day of the next `month` at least DEFAULT_LEAD_DAYS away, or
    that earliest day itself when it already falls in `month`."""
    earliest = today + timedelta(days=DEFAULT_LEAD_DAYS)
    if earliest.month == month:
        return earliest
    start = date(earliest.year, month, 1)
    return start if start > earliest else date(earliest.year + 1, month, 1)


def _resolve_dates(parsed: ParsedPrompt, today: date) -> tuple[date, date]:
    """Fill in whichever of start/end/duration the model left out."""
    if parsed.travel_month and not (parsed.start_date or parsed.end_date):
        start = _month_start(parsed.travel_month, today)
        duration = parsed.duration_days or DEFAULT_DURATION_DAYS
        return start, start + timedelta(days=duration - 1)
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


@dataclass(frozen=True)
class Interpretation:
    """What one free-text request turned out to be. `trip` is set exactly
    when `intent` is "trip"; places and routes are answered from `parsed`."""

    intent: Intent
    parsed: ParsedPrompt
    trip: TripRequest | None = None


def _trip_request(parsed: ParsedPrompt, today: date) -> TripRequest:
    start_date, end_date = _resolve_dates(parsed, today)
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
        places_per_city=parsed.places_per_city,
        cautions=parsed.concerns if parsed.verdict == "caution" else [],
    )


def _refuse_if_asked(parsed: ParsedPrompt) -> None:
    """Raise RequestRefused when the guardrail's verdict refuses."""
    if parsed.verdict in REFUSED:
        reason = " ".join(parsed.concerns[:2])
        raise RequestRefused(parsed.verdict, f"{REFUSED[parsed.verdict]} {reason}".strip())


def _missing_for(parsed: ParsedPrompt) -> str | None:
    """Feedback when a places/route answer lacks what it needs to be run."""
    if parsed.intent == "places" and not parsed.places_query:
        return 'intent is "places" but places_query is null.'
    if parsed.intent == "route" and not (parsed.origin and parsed.destination):
        return 'intent is "route" but origin or destination is null.'
    return None


def _interpret(
    prompt: str, llm: LLM, *, today: date, max_retries: int, always_trip: bool
) -> Interpretation:
    feedback = ""
    last_error = ""

    for _ in range(max_retries + 1):
        user = prompt
        if feedback:
            user += f"\n\nYour previous attempt was rejected: {feedback}\nFix it and try again."

        raw = llm.complete(
            system=PARSE_REQUEST_SYSTEM, user=user, json_mode=True, schema=ParsedPrompt
        )
        try:
            parsed = ParsedPrompt.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            feedback = last_error = f"could not parse that as request parameters: {exc}"
            continue

        # Refused requests aren't retried: the answer was understood.
        _refuse_if_asked(parsed)

        if not always_trip and parsed.intent != "trip":
            missing = _missing_for(parsed)
            if missing:
                feedback = last_error = missing
                continue
            return Interpretation(intent=parsed.intent, parsed=parsed)

        try:
            return Interpretation(intent="trip", parsed=parsed, trip=_trip_request(parsed, today))
        except ValidationError as exc:
            feedback = last_error = f"those trip parameters were not valid: {exc}"
            continue

    raise PromptParseError(f"couldn't understand that request: {last_error}")


def interpret_prompt(
    prompt: str,
    llm: LLM,
    *,
    today: date | None = None,
    max_retries: int = 1,
) -> Interpretation:
    """Work out whether one free-text request is a trip to plan, places to
    find, or a route to measure — and extract what that needs. One LLM call,
    retried once on a malformed or unusable response."""
    return _interpret(
        prompt, llm, today=today or date.today(), max_retries=max_retries, always_trip=False
    )


def parse_trip_prompt(
    prompt: str,
    llm: LLM,
    *,
    today: date | None = None,
    max_retries: int = 1,
) -> TripRequest:
    """Parse one free-text request into a `TripRequest`, retrying once on a
    malformed or unusable response — the same feedback-loop shape as the
    itinerary agent's JSON validation.

    Always a trip, whatever the sentence reads as: this is for callers that
    only plan (POST /trips/plan-from-prompt, the CLI). `interpret_prompt` is
    the one that tells trips, places and routes apart.
    """
    interpretation = _interpret(
        prompt, llm, today=today or date.today(), max_retries=max_retries, always_trip=True
    )
    assert interpretation.trip is not None  # always_trip guarantees it
    return interpretation.trip


def apply_page_choices(
    parsed: TripRequest,
    *,
    include_flights: bool,
    include_hotels: bool,
    places_per_city: str | None,
    start_date: date | None,
    end_date: date | None,
) -> TripRequest:
    """The request to plan: what the sentence said, with the page's own
    choices (checkboxes, calendar dates) applied on top. Raises
    PromptParseError if no destination was named.

    `places_per_city` always replaces whatever the parser read, so a saved
    trip records what was actually looked up — nothing, unless the page's
    Restaurants box asked for it.
    """
    overrides: dict = {
        "include_flights": include_flights,
        "include_hotels": include_hotels,
        "places_per_city": places_per_city,
    }
    if start_date and end_date:
        overrides |= {"start_date": start_date, "end_date": end_date}
    # Re-validated rather than model_copy'd, so picked dates go through the
    # same ordering and length checks as parsed ones.
    request = TripRequest.model_validate(parsed.model_dump() | overrides)
    # Without this, a missing destination falls through to resolve_destination,
    # which picks from the provider's sample list — "Varanasi to Kerala and
    # Tamil Nadu" came back as five days in Chiang Mai. Asking is better than
    # planning a trip nobody requested.
    if not request.destination:
        raise PromptParseError(
            "no destination was named. Say where you want to go — a city, state, "
            "region or country."
        )
    return request
