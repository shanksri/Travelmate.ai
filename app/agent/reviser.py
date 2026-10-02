"""Apply one requested change to a trip that was already planned.

Deliberately *not* the five-node planning graph. A revision first routes the
change (ROUTE_CHANGE_SYSTEM) to the parts it's about — the outbound flight,
the return flight, the hotel, the day plan — and then touches only those:

- Flights and the hotel are re-picked from the options the original search
  already found, by index. Nothing is searched again (no SerpApi quota), and
  nothing is typed by a model, so a new pick can't be hallucinated.
- The day plan is re-generated only when the change is about it, with
  flights and lodging withheld from that prompt so it can't alter them.

Dates and destination never change. The result is appended as the next
version of the same thread rather than replacing it — see app/store.py.
"""

import json
import logging
import uuid

import openai
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agent.itinerary_json import ItineraryValidationError, parse_itinerary
from app.agent.llm import LLM, OpenAILLM
from app.agent.prompts import (
    FINAL_RESPONSE_AGENT_SYSTEM,
    REVISE_ITINERARY_SYSTEM,
    ROUTE_CHANGE_SYSTEM,
)
from app.core.config import Settings, get_settings
from app.models.itinerary import (
    DayPlan,
    DraftItinerary,
    FlightLeg,
    Itinerary,
    LodgingOption,
    PlannedTrip,
    TripRequest,
)

logger = logging.getLogger(__name__)


class RevisionError(RuntimeError):
    """The change was understood but no valid revised itinerary came back."""


class RevisionDeclined(RuntimeError):
    """The change wasn't made — nothing it asked for could be applied.

    Raised instead of saving a version identical to the one before it, so the
    traveller is told why rather than shown the same plan again.
    """


# REVISE_ITINERARY_SYSTEM tells the model to start its explanation with this
# when it declines, so the reason can be found among the other notes.
DECLINE_PREFIX = "Couldn't apply:"


def _decline_reason(draft_notes: list[str], previous_notes: list[str]) -> str:
    for note in draft_notes:
        if note.strip().startswith(DECLINE_PREFIX):
            return note.strip().removeprefix(DECLINE_PREFIX).strip()
    # The model may explain itself without the prefix; a note it just added
    # is the next best thing.
    new_notes = [note for note in draft_notes if note not in previous_notes]
    if new_notes:
        return " ".join(new_notes)
    return "the planner returned the plan unchanged."


def _build_llm(settings: Settings) -> LLM:
    return OpenAILLM(
        openai.OpenAI(),
        model=settings.revise_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


# --- routing -----------------------------------------------------------------


class ChangeRoute(BaseModel):
    """Which parts of the trip a change is about, as ROUTE_CHANGE_SYSTEM
    decides them. Indexes point into the trip's existing option lists."""

    model_config = ConfigDict(populate_by_name=True)

    outbound: int | None = None
    return_leg: int | None = Field(default=None, alias="return")
    hotel: int | None = None
    itinerary: str | None = None
    declined: str | None = None
    reasoning: str = ""


def _same_flight(a: FlightLeg | None, b: FlightLeg | None) -> bool:
    if a is None or b is None:
        return False
    return a.model_dump(exclude={"rationale"}) == b.model_dump(exclude={"rationale"})


def _flight_menu(options: list[FlightLeg], booked: FlightLeg | None) -> list[dict] | str:
    if not options:
        # Worded as the traveller's choice, not a gap: "none — this trip has
        # no hotel" read to the router as a reason to refuse day changes.
        return "none: the traveller planned this trip without flights"
    return [
        {
            "index": i,
            "carrier": f.carrier,
            "departs": (f.departure_at or f.depart_date).isoformat(),
            "stops": f.stops,
            "duration_hours": f.duration_hours,
            "total": f.total,
            "booked": _same_flight(f, booked),
        }
        for i, f in enumerate(options)
    ]


def _hotel_menu(options: list[LodgingOption]) -> list[dict] | str:
    if not options:
        return "none: the traveller planned this trip without a hotel"
    return [
        {
            "index": i,
            "name": h.name,
            "tier": h.tier,
            "rating": h.rating,
            "neighbourhood": h.neighbourhood,
            "nightly": h.nightly,
            "total": h.total,
            "selected": i == 0,
        }
        for i, h in enumerate(options)
    ]


def _check_indexes(route: ChangeRoute, itinerary: Itinerary) -> str | None:
    """Feedback for the router when it named an option that isn't there."""
    for name, index, options in (
        ("outbound", route.outbound, itinerary.outbound_options),
        ("return", route.return_leg, itinerary.return_options),
        ("hotel", route.hotel, itinerary.lodging_options),
    ):
        if index is not None and not 0 <= index < len(options):
            return f"`{name}` is {index}, but there are only {len(options)} {name} option(s)."
    return None


def _route_change(
    previous: PlannedTrip, change_request: str, llm: LLM, attempts: int
) -> ChangeRoute:
    itinerary = previous.itinerary
    base_prompt = json.dumps(
        {
            "change_request": change_request,
            "outbound_options": _flight_menu(
                itinerary.outbound_options, itinerary.outbound_flight
            ),
            "return_options": _flight_menu(itinerary.return_options, itinerary.return_flight),
            "hotel_options": _hotel_menu(itinerary.lodging_options),
            "days": [
                {"day": d.day, "date": d.date.isoformat(), "summary": d.summary}
                for d in itinerary.days
            ],
        },
        default=str,
    )

    feedback = ""
    for _ in range(attempts):
        user_prompt = base_prompt
        if feedback:
            user_prompt += f"\n\nYour previous answer was rejected: {feedback} Fix it."
        raw = llm.complete(
            system=ROUTE_CHANGE_SYSTEM, user=user_prompt, json_mode=True, schema=ChangeRoute
        )
        try:
            route = ChangeRoute.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            feedback = f"that wasn't the JSON asked for: {exc}"
            continue
        feedback = _check_indexes(route, itinerary) or ""
        if not feedback:
            return route

    raise RevisionError(f"couldn't work out what the change is about: {feedback}")


# --- applying each part ------------------------------------------------------


def _revise_days(
    itinerary: Itinerary,
    request: TripRequest,
    instruction: str,
    llm: LLM,
    attempts: int,
) -> tuple[list[DayPlan], list[str]]:
    """Re-plan the days for `instruction`. Raises RevisionDeclined if the
    model hands every day back unchanged."""
    # Only the days go to the model. Flights and lodging are withheld on
    # purpose — it cannot change what it cannot see. The previous notes are
    # withheld too: a note from an earlier refusal ("X is not included…")
    # anchored the model into refusing the same place again, and it writes
    # fresh notes every time anyway.
    current_days = json.dumps(
        [day.model_dump(mode="json") for day in itinerary.days], default=str
    )
    trip_dates = [d.isoformat() for d in request.dates]

    base_prompt = (
        f"Destination: {itinerary.destination}\n"
        f"Travellers: {itinerary.travelers}\n"
        f"Pace: {request.pace}\n"
        f"Current plan: {current_days}\n\n"
        f"The traveller asks: {instruction}\n\n"
        f"Return all {len(trip_dates)} days, using exactly these dates in this "
        f"order: {json.dumps(trip_dates)}"
    )

    feedback = ""
    last_error = ""
    for _ in range(attempts):
        user_prompt = base_prompt
        if feedback:
            user_prompt += (
                f"\n\nYour previous attempt was rejected: {feedback}\n"
                "Fix it and return the complete JSON again."
            )

        raw = llm.complete(
            system=REVISE_ITINERARY_SYSTEM,
            user=user_prompt,
            json_mode=True,
            schema=DraftItinerary,
        )
        try:
            draft = parse_itinerary(raw, request)
        except ItineraryValidationError as exc:
            feedback = last_error = exc.feedback
            continue

        unchanged = [day.model_dump(mode="json") for day in draft.days] == [
            day.model_dump(mode="json") for day in itinerary.days
        ]
        if unchanged:
            raise RevisionDeclined(_decline_reason(draft.notes, itinerary.notes))
        return draft.days, draft.notes

    raise RevisionError(f"gave up after {attempts} attempt(s): {last_error}")


def _rebooked(option: FlightLeg, change_request: str) -> FlightLeg:
    return option.model_copy(update={"rationale": f"Chosen on request: {change_request}"})


def revise_trip(
    previous: PlannedTrip,
    change_request: str,
    *,
    llm: LLM | None = None,
    settings: Settings | None = None,
) -> PlannedTrip:
    """Produce the next version of `previous` with `change_request` applied.

    Pure in the sense that matters: it reads `previous` and returns a new
    `PlannedTrip`, and never touches the store — the caller decides whether
    to keep the result. Raises `RevisionDeclined` when nothing the change
    asked for could be applied; a change is applied whole or not at all.
    """
    settings = settings or get_settings()
    llm = llm or _build_llm(settings)
    attempts = settings.max_itinerary_retries + 1

    request = previous.request
    itinerary = previous.itinerary

    if (
        not itinerary.outbound_options
        and not itinerary.return_options
        and not itinerary.lodging_options
    ):
        # Nothing to re-pick, so the only thing a change can touch is the
        # days. Routing it anyway once refused "adjust more days to tamil
        # nadu" (2 runs in 5) because "the trip has no hotel". This is every
        # trip planned with Flights and Hotels unticked, the page's default,
        # and skipping the router saves a call. A request that really is about
        # flights or a hotel is declined by the day planner instead.
        route = ChangeRoute(itinerary=change_request)
        trace = [
            "change_router: skipped (no flights or hotel to re-pick; the change goes to the days)"
        ]
    else:
        route = _route_change(previous, change_request, llm, attempts)
        if route.declined:
            raise RevisionDeclined(route.declined)
        trace = [f"change_router: {route.reasoning or 'routed the change'}"]

    outbound = itinerary.outbound_flight
    if route.outbound is not None:
        pick = itinerary.outbound_options[route.outbound]
        if not _same_flight(pick, outbound):
            outbound = _rebooked(pick, change_request)
            trace.append(f"flights: outbound rebooked to option {route.outbound}")

    return_flight = itinerary.return_flight
    if route.return_leg is not None:
        pick = itinerary.return_options[route.return_leg]
        if not _same_flight(pick, return_flight):
            return_flight = _rebooked(pick, change_request)
            trace.append(f"flights: return rebooked to option {route.return_leg}")

    # The first lodging option is the selected one everywhere (the itinerary
    # node's cost, the page's "Selected" tag), so switching means moving it
    # to the front.
    lodging = list(itinerary.lodging_options)
    if route.hotel:
        lodging.insert(0, lodging.pop(route.hotel))
        trace.append(f"hotel: switched to {lodging[0].name}")

    days, notes = itinerary.days, itinerary.notes
    if route.itinerary:
        days, notes = _revise_days(itinerary, request, route.itinerary, llm, attempts)
        trace.append("reviser: re-planned the days")

    if len(trace) == 1:
        # Routed, but everything it pointed at was already the case — e.g.
        # "book the cheapest flight" when the cheapest is already booked.
        raise RevisionDeclined(
            route.reasoning or "the plan already matches what was asked for."
        )

    flight_cost = (outbound.total or 0 if outbound else 0) + (
        return_flight.total or 0 if return_flight else 0
    )
    hotel_cost = lodging[0].total or 0 if lodging else 0
    activity_cost = sum(
        activity.estimated_cost or 0 for day in days for activity in day.activities
    )

    revised = Itinerary(
        destination=itinerary.destination,
        start_date=itinerary.start_date,
        end_date=itinerary.end_date,
        travelers=itinerary.travelers,
        outbound_flight=outbound,
        return_flight=return_flight,
        outbound_options=itinerary.outbound_options,
        return_options=itinerary.return_options,
        lodging_options=lodging,
        outbound_source=itinerary.outbound_source,
        return_source=itinerary.return_source,
        lodging_source=itinerary.lodging_source,
        days=days,
        currency=itinerary.currency,
        total_estimated_cost=flight_cost + hotel_cost + activity_cost,
        notes=notes,
    )

    summary = llm.complete(
        system=FINAL_RESPONSE_AGENT_SYSTEM,
        user=f"Itinerary: {revised.model_dump_json()}",
    )
    trace.append("final_response_agent: rewrote the trip summary")

    return PlannedTrip(
        id=uuid.uuid4().hex[:12],
        thread_id=previous.thread_id,
        version=previous.version + 1,
        change_note=change_request,
        request=request,
        itinerary=revised,
        agent_trace=trace,
        summary=summary.strip(),
    )
