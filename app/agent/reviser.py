"""Apply one requested change to a trip that was already planned.

Deliberately *not* the five-node planning graph: a revision re-runs only the
itinerary step. Flights, lodging, dates and destination carry over untouched
from the version being revised, so asking to reshuffle day 3 can't quietly
swap the hotel or re-price a flight. The result is appended as the next
version of the same thread rather than replacing it — see app/store.py.
"""

import json
import logging
import uuid

import openai

from app.agent.itinerary_json import ItineraryValidationError, parse_itinerary
from app.agent.llm import LLM, OpenAILLM
from app.agent.prompts import FINAL_RESPONSE_AGENT_SYSTEM, REVISE_ITINERARY_SYSTEM
from app.core.config import Settings, get_settings
from app.models.itinerary import Itinerary, PlannedTrip

logger = logging.getLogger(__name__)


class RevisionError(RuntimeError):
    """The change was understood but no valid revised itinerary came back."""


def _build_llm(settings: Settings) -> LLM:
    return OpenAILLM(
        openai.OpenAI(),
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


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
    to keep the result.
    """
    settings = settings or get_settings()
    llm = llm or _build_llm(settings)

    request = previous.request
    itinerary = previous.itinerary

    # Only the days go to the model. Flights and lodging are withheld on
    # purpose — it cannot change what it cannot see, and they're merged back
    # in untouched below.
    current_days = json.dumps(
        [day.model_dump(mode="json") for day in itinerary.days], default=str
    )
    trip_dates = [d.isoformat() for d in request.dates]

    base_prompt = (
        f"Destination: {itinerary.destination}\n"
        f"Travellers: {itinerary.travelers}\n"
        f"Pace: {request.pace}\n"
        f"Current plan: {current_days}\n\n"
        f"Existing notes: {json.dumps(itinerary.notes)}\n\n"
        f"The traveller asks: {change_request}\n\n"
        f"Return all {len(trip_dates)} days, using exactly these dates in this "
        f"order: {json.dumps(trip_dates)}"
    )

    feedback = ""
    last_error = ""
    attempts = settings.max_itinerary_retries + 1
    for attempt in range(1, attempts + 1):
        user_prompt = base_prompt
        if feedback:
            user_prompt += (
                f"\n\nYour previous attempt was rejected: {feedback}\n"
                "Fix it and return the complete JSON again."
            )

        raw = llm.complete(system=REVISE_ITINERARY_SYSTEM, user=user_prompt, json_mode=True)
        try:
            draft = parse_itinerary(raw, request)
        except ItineraryValidationError as exc:
            feedback = exc.feedback
            last_error = exc.feedback
            continue

        # Flights and lodging come straight off the previous version; only the
        # activity portion of the total is recomputed.
        outbound = itinerary.outbound_flight
        return_leg = itinerary.return_flight
        flight_cost = (outbound.total_usd or 0 if outbound else 0) + (
            return_leg.total_usd or 0 if return_leg else 0
        )
        hotel_cost = (
            itinerary.lodging_options[0].total_usd or 0 if itinerary.lodging_options else 0
        )
        activity_cost = sum(
            activity.estimated_cost_usd or 0 for day in draft.days for activity in day.activities
        )

        revised = Itinerary(
            destination=itinerary.destination,
            start_date=itinerary.start_date,
            end_date=itinerary.end_date,
            travelers=itinerary.travelers,
            outbound_flight=itinerary.outbound_flight,
            return_flight=itinerary.return_flight,
            lodging_options=itinerary.lodging_options,
            days=draft.days,
            currency=itinerary.currency,
            total_estimated_cost=flight_cost + hotel_cost + activity_cost,
            notes=draft.notes,
        )

        summary = llm.complete(
            system=FINAL_RESPONSE_AGENT_SYSTEM,
            user=f"Itinerary: {revised.model_dump_json()}",
        )

        return PlannedTrip(
            id=uuid.uuid4().hex[:12],
            thread_id=previous.thread_id,
            version=previous.version + 1,
            change_note=change_request,
            request=request,
            itinerary=revised,
            agent_trace=[
                f"reviser: applied a change to version {previous.version} "
                f"(attempt {attempt})",
                "final_response_agent: rewrote the trip summary",
            ],
            summary=summary.strip(),
        )

    raise RevisionError(f"gave up after {attempts} attempt(s): {last_error}")
