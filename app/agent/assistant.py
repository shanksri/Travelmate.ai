"""One entry point for the prompt box: plan a trip, find places, or measure
a route — whichever the sentence asks for.

The parser's one LLM call decides which (`interpret_prompt`), so telling them
apart costs no extra call. Trips go through the planner as before; places and
routes go to Google Maps (app/providers/google_maps.py) and are answered
directly, with no agent graph and no further LLM calls.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel

from app.agent.llm import LLM
from app.agent.planner import _build_llm, plan_parsed_trip
from app.agent.prompt_parser import interpret_prompt
from app.core.config import Settings, get_settings
from app.models.itinerary import PlannedTrip
from app.models.maps import PlacesAnswer, RouteAnswer
from app.providers import google_maps
from app.providers.base import TravelProvider


class Answer(BaseModel):
    """Exactly one of `trip`, `places`, `route` is set, matching `kind`."""

    kind: Literal["trip", "places", "route"]
    trip: PlannedTrip | None = None
    places: PlacesAnswer | None = None
    route: RouteAnswer | None = None


def answer_prompt(
    prompt: str,
    *,
    include_flights: bool = True,
    include_hotels: bool = True,
    start_date: date | None = None,
    end_date: date | None = None,
    llm: LLM | None = None,
    provider: TravelProvider | None = None,
    settings: Settings | None = None,
) -> Answer:
    """The checkboxes and picked dates only apply when it turns out to be a
    trip; a places search or a route ignores them."""
    if (start_date is None) != (end_date is None):
        raise ValueError("give both start_date and end_date, or neither")

    settings = settings or get_settings()
    llm = llm or _build_llm(settings)

    interpretation = interpret_prompt(prompt, llm)
    parsed = interpretation.parsed

    if interpretation.intent == "places":
        assert parsed.places_query  # interpret_prompt guarantees it
        return Answer(kind="places", places=google_maps.search_places(parsed.places_query))

    if interpretation.intent == "route":
        assert parsed.origin and parsed.destination  # interpret_prompt guarantees it
        return Answer(
            kind="route",
            route=google_maps.compute_route(
                parsed.origin, parsed.destination, parsed.travel_mode or "DRIVE"
            ),
        )

    assert interpretation.trip is not None
    trip = plan_parsed_trip(
        interpretation.trip,
        include_flights=include_flights,
        include_hotels=include_hotels,
        start_date=start_date,
        end_date=end_date,
        llm=llm,
        provider=provider,
        settings=settings,
    )
    return Answer(kind="trip", trip=trip)
