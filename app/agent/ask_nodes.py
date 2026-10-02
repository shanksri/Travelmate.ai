"""The steps of the ask graph (app/agent/graph.py) that the trip graph doesn't
have: read the sentence, answer a places or route request from Google Maps,
apply the page's choices to a trip, and look up places in each of the
trip's cities once it's planned — one parallel step per city.
"""

from langgraph.graph import END
from langgraph.types import Send

from app.agent.events import emit, status
from app.agent.llm import LLM
from app.agent.prompt_parser import apply_page_choices, interpret_prompt
from app.agent.state import AskState
from app.models.itinerary import PlannedTrip
from app.models.maps import CityPlaces
from app.providers import google_maps

# A long multi-city trip shouldn't turn into a dozen Maps searches.
MAX_CITIES = 6


def build_interpret_node(llm: LLM):
    """One LLM call: is this a trip, places or a route, and what does it
    need? (`interpret_prompt`.) The graph branches on `intent`."""

    def node(state: AskState) -> dict:
        status("interpret", "Reading your request…")
        interpretation = interpret_prompt(state["prompt"], llm)
        emit({"type": "intent", "kind": interpretation.intent})
        return {
            "intent": interpretation.intent,
            "parsed": interpretation.parsed,
            "parsed_request": interpretation.trip,
            "messages": [f"interpreter: read as a {interpretation.intent} request"],
        }

    return node


def next_after_interpret(state: AskState) -> str:
    return {"places": "places", "route": "route"}.get(state["intent"], "prepare_trip")


def build_places_node():
    def node(state: AskState) -> dict:
        query = state["parsed"].places_query
        status("places", f"Searching Google Maps for {query}…")
        return {"places": google_maps.search_places(query)}

    return node


def build_route_node():
    def node(state: AskState) -> dict:
        parsed = state["parsed"]
        status("route", f"Working out the route from {parsed.origin} to {parsed.destination}…")
        route = google_maps.compute_route(
            parsed.origin, parsed.destination, parsed.travel_mode or "DRIVE"
        )
        return {"route": route}

    return node


def build_prepare_trip_node():
    """The parsed request with the page's checkboxes and dates applied —
    what the trip steps that follow plan. Raises PromptParseError when no
    destination was named."""

    def node(state: AskState) -> dict:
        choices = state["choices"]
        parsed = state["parsed"]
        request = apply_page_choices(
            state["parsed_request"],
            include_flights=choices["include_flights"],
            include_hotels=choices["include_hotels"],
            # Only when the Restaurants box asks; the sentence picks the kind.
            places_per_city=(
                (parsed.places_per_city or "restaurants")
                if choices["include_restaurants"]
                else None
            ),
            start_date=choices["start_date"],
            end_date=choices["end_date"],
        )
        return {"request": request, "resolved_destination": request.destination or ""}

    return node


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


def places_for_each_city(state: AskState):
    """After the trip is packaged: one parallel `city_places` step per city
    when the trip asked for places, otherwise the end."""
    trip = state["trip"]
    kind = trip.request.places_per_city
    if not kind:
        return END
    return [Send("city_places", {"city": city, "kind": kind}) for city in trip_cities(trip)]


def build_city_places_node():
    """One city's places. Runs once per city, in parallel; a failure is kept
    as that city's error rather than raised, so it can't cost the others or
    the plan."""

    def node(payload: dict) -> dict:
        city, kind = payload["city"], payload["kind"]
        try:
            found = CityPlaces(
                city=city, places=google_maps.search_places(f"best {kind} in {city}")
            )
        except google_maps.GoogleMapsError as exc:
            found = CityPlaces(city=city, error=str(exc))
        emit({"type": "city", "city": found.model_dump(mode="json")})
        return {"places_by_city": [found]}

    return node
