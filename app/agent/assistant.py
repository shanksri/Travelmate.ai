"""One entry point for the prompt box: plan a trip, find places, or measure
a route — whichever the sentence asks for.

Runs the ask graph (app/agent/graph.py) to completion and returns its
answer. The page streams the same graph instead (app/jobs.py), to show each
step as it happens; this is the blocking version, behind `POST /ask`.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from app.agent.ask_nodes import MAX_CITIES, trip_cities
from app.agent.graph import build_ask_graph
from app.agent.llm import LLM
from app.agent.planner import _build_llm
from app.agent.state import PageChoices, initial_ask_state
from app.core.config import Settings, get_settings
from app.models.itinerary import PlannedTrip
from app.models.maps import CityPlaces, PlacesAnswer, RouteAnswer
from app.providers import get_provider
from app.providers.base import TravelProvider

__all__ = ["MAX_CITIES", "Answer", "answer_from_state", "answer_prompt", "trip_cities"]


class Answer(BaseModel):
    """Exactly one of `trip`, `places`, `route` is set, matching `kind`.

    `places_by_city` rides along with a trip whose request asked for places
    in each city. It's part of the answer, not the trip: Google's terms
    forbid storing Maps results, so it is never saved with the trip.
    """

    kind: Literal["trip", "places", "route"]
    trip: PlannedTrip | None = None
    places: PlacesAnswer | None = None
    route: RouteAnswer | None = None
    places_by_city: list[CityPlaces] = Field(default_factory=list)


def answer_from_state(state: dict) -> Answer:
    """The ask graph's final state as an Answer. Per-city places arrive in
    whatever order their parallel searches finished; they're put back in
    the order the trip visits the cities."""
    trip = state.get("trip")
    found = state.get("places_by_city") or []
    if trip is not None and found:
        order = {city.lower(): i for i, city in enumerate(trip_cities(trip))}
        found = sorted(found, key=lambda c: order.get(c.city.lower(), len(order)))
    return Answer(
        kind=state["intent"],
        trip=trip,
        places=state.get("places"),
        route=state.get("route"),
        places_by_city=found,
    )


def page_choices(
    *,
    include_flights: bool = True,
    include_hotels: bool = True,
    include_restaurants: bool = False,
    start_date: date | None = None,
    end_date: date | None = None,
) -> PageChoices:
    if (start_date is None) != (end_date is None):
        raise ValueError("give both start_date and end_date, or neither")
    return PageChoices(
        include_flights=include_flights,
        include_hotels=include_hotels,
        include_restaurants=include_restaurants,
        start_date=start_date,
        end_date=end_date,
    )


def answer_prompt(
    prompt: str,
    *,
    include_flights: bool = True,
    include_hotels: bool = True,
    include_restaurants: bool = False,
    start_date: date | None = None,
    end_date: date | None = None,
    llm: LLM | None = None,
    provider: TravelProvider | None = None,
    settings: Settings | None = None,
) -> Answer:
    """The checkboxes and picked dates only apply when it turns out to be a
    trip; a places search or a route ignores them.

    Places per city are looked up only when `include_restaurants` is on,
    like flights and hotels only when theirs are. The sentence can still
    pick the kind ("street food in each city"); without one it's restaurants.
    """
    choices = page_choices(
        include_flights=include_flights,
        include_hotels=include_hotels,
        include_restaurants=include_restaurants,
        start_date=start_date,
        end_date=end_date,
    )
    settings = settings or get_settings()
    graph = build_ask_graph(
        provider or get_provider(), llm or _build_llm(settings), settings.max_itinerary_retries
    )
    return answer_from_state(graph.invoke(initial_ask_state(prompt, choices)))
