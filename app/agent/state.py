"""The shared state LangGraph threads through every agent node.

Each node reads what it needs and returns only the keys it changes — LangGraph
merges those into this dict between steps. Nothing here is provider- or
LLM-specific, so a node's tests can build a `TravelState` by hand.
"""

import operator
from typing import Annotated, TypedDict

from app.models.itinerary import Itinerary, TripRequest


class TravelState(TypedDict):
    request: TripRequest
    resolved_destination: str

    # Raw options plus each agent's short recommendation over them.
    flight_results: dict
    hotel_results: dict

    itinerary: Itinerary | None
    final_response: str | None

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
        messages=[],
        errors=[],
    )
