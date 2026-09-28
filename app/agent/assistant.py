"""One entry point for the prompt box: plan a trip, find places, or measure
a route — whichever the sentence asks for.

The parser's one LLM call decides which (`interpret_prompt`), so telling them
apart costs no extra call. Trips go through the planner as before; places and
routes go to Google Maps (app/providers/google_maps.py) and are answered
directly, with no agent graph and no further LLM calls.

A trip can also ask for places in each of its cities ("…also tell me the best
places to eat in each city"). Then, once the plan exists and its days name
their cities, one Maps search per city runs in parallel.
"""

import logging
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from app.agent.llm import LLM
from app.agent.planner import _build_llm, plan_parsed_trip
from app.agent.prompt_parser import interpret_prompt
from app.core.config import Settings, get_settings
from app.models.itinerary import PlannedTrip
from app.models.maps import CityPlaces, PlacesAnswer, RouteAnswer
from app.providers import google_maps
from app.providers.base import TravelProvider

logger = logging.getLogger(__name__)


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


# A long multi-city trip shouldn't turn into a dozen Maps searches.
MAX_CITIES = 6


def trip_cities(trip: PlannedTrip) -> list[str]:
    """The trip's cities in the order it visits them, each once. Falls back
    to the destination for trips whose days carry no city."""
    cities: list[str] = []
    seen: set[str] = set()
    for day in trip.itinerary.days:
        city = (day.city or "").strip()
        if city and city.lower() not in seen:
            seen.add(city.lower())
            cities.append(city)
    return cities[:MAX_CITIES] or [trip.itinerary.destination]


def places_by_city(trip: PlannedTrip, kind: str) -> list[CityPlaces]:
    """One Maps search per city, all at once over the shared MCP session;
    results in trip order. One city failing doesn't cost the traveller the
    others, or the plan."""
    cities = trip_cities(trip)
    try:
        results = google_maps.search_places_many([f"best {kind} in {city}" for city in cities])
    except google_maps.GoogleMapsError as exc:  # e.g. no key: every city fails alike
        results = [exc] * len(cities)

    found = []
    for city, result in zip(cities, results, strict=True):
        if isinstance(result, google_maps.GoogleMapsError):
            logger.warning("places for %s failed: %s", city, result)
            found.append(CityPlaces(city=city, error=str(result)))
        else:
            found.append(CityPlaces(city=city, places=result))
    return found


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
        places_per_city=(parsed.places_per_city or "restaurants") if include_restaurants else None,
        start_date=start_date,
        end_date=end_date,
        llm=llm,
        provider=provider,
        settings=settings,
    )
    kind = trip.request.places_per_city
    return Answer(
        kind="trip", trip=trip, places_by_city=places_by_city(trip, kind) if kind else []
    )
