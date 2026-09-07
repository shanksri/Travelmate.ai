"""Turn the itinerary agent's raw JSON text into a validated `DraftItinerary`.

Pulled out of the node itself so the retry loop in `nodes.py` and its tests can
call one small, deterministic function instead of re-parsing inline.
"""

import json

from pydantic import ValidationError

from app.models.itinerary import DraftItinerary, TripRequest


class ItineraryValidationError(ValueError):
    """The model's JSON was well-formed but did not describe a valid trip.

    `feedback` is meant to be fed straight back into the next prompt attempt.
    """

    def __init__(self, feedback: str) -> None:
        super().__init__(feedback)
        self.feedback = feedback


def parse_itinerary(raw_json: str, request: TripRequest) -> DraftItinerary:
    try:
        payload = json.loads(raw_json)
    except json.JSONDecodeError as exc:
        raise ItineraryValidationError(f"Your last response was not valid JSON: {exc}") from exc

    try:
        draft = DraftItinerary.model_validate(payload)
    except ValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:12]
        )
        raise ItineraryValidationError(
            f"The JSON did not validate against the required schema: {problems}"
        ) from exc

    expected_days = request.nights + 1
    if len(draft.days) != expected_days:
        raise ItineraryValidationError(
            f"The trip runs {request.start_date} to {request.end_date}, which is "
            f"{expected_days} days, but you returned {len(draft.days)} entries in "
            "`days`. Return one entry per calendar day."
        )

    return draft
