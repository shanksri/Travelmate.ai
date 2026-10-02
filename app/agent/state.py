"""The shared state LangGraph threads through every agent node.

Each node reads what it needs and returns only the keys it changes — LangGraph
merges those into this dict between steps. Nothing here is provider- or
LLM-specific, so a node's tests can build a `TravelState` by hand.

Two graphs share these (app/agent/graph.py):
- `TravelState` — planning one trip from a `TripRequest`.
- `AskState` — the prompt box: a `TravelState` plus what's needed to read the
  sentence, route it to a trip, places or a route, and look up places per
  city once the trip exists.
"""

import operator
from datetime import date
from typing import Annotated, Literal, TypedDict

from app.models.itinerary import Itinerary, PlannedTrip, TripRequest
from app.models.maps import CityPlaces, PlacesAnswer, RouteAnswer


class TravelState(TypedDict):
    request: TripRequest
    resolved_destination: str

    # Raw options plus each agent's short recommendation over them.
    flight_results: dict
    hotel_results: dict

    itinerary: Itinerary | None
    final_response: str | None
    # The finished, saveable trip — set by the last step, `package_trip`.
    trip: PlannedTrip | None

    # Annotated with operator.add so parallel branches (flight + hotel both
    # run off of START) accumulate into these lists instead of one branch's
    # update silently overwriting the other's — LangGraph's default merge
    # behavior for a plain list field is last-write-wins, not append.
    messages: Annotated[list[str], operator.add]
    errors: Annotated[list[str], operator.add]


def initial_state(request: TripRequest) -> TravelState:
    return TravelState(
        request=request,
        resolved_destination=request.destination or "",
        flight_results={},
        hotel_results={},
        itinerary=None,
        final_response=None,
        trip=None,
        messages=[],
        errors=[],
    )


class PageChoices(TypedDict):
    """What the page decided rather than the sentence: the checkboxes and
    the calendar dates. Applied only if the sentence turns out to be a trip."""

    include_flights: bool
    include_hotels: bool
    include_restaurants: bool
    start_date: date | None
    end_date: date | None


class AskState(TravelState, total=False):
    prompt: str
    choices: PageChoices

    intent: Literal["trip", "places", "route"]
    parsed: object  # app.agent.prompt_parser.ParsedPrompt
    # The parser's TripRequest, before the page's choices are applied.
    parsed_request: TripRequest | None

    places: PlacesAnswer | None
    route: RouteAnswer | None

    # One entry per city, from parallel steps (one `Send` per city, each
    # given its city and the kind of place) — appended, never overwritten,
    # so they arrive in whatever order the searches finish.
    places_by_city: Annotated[list[CityPlaces], operator.add]


def initial_ask_state(prompt: str, choices: PageChoices) -> AskState:
    return AskState(
        prompt=prompt,
        choices=choices,
        resolved_destination="",
        flight_results={},
        hotel_results={},
        itinerary=None,
        final_response=None,
        trip=None,
        places=None,
        route=None,
        places_by_city=[],
        messages=[],
        errors=[],
    )
