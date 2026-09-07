"""The agent's research tools.

These are built per request by `build_research_tools` so each one closes over
the provider and the trip's own dates — the model never has to restate context
it already gave us, and it cannot query a provider we did not hand it.

Every tool returns a JSON string. Tool results are text on the wire, and JSON
keeps the shape unambiguous for the model.
"""

import json
from collections.abc import Callable
from datetime import date

from app.models.itinerary import TripRequest
from app.providers.base import TravelProvider

# The tool functions themselves; the planner decorates them for the SDK.
ToolFn = Callable[..., str]

_BUDGET_LEVELS = ("low", "medium", "high")


def _dump(payload: object) -> str:
    return json.dumps(payload, default=str)


def _parse_date(value: str) -> date | None:
    try:
        return date.fromisoformat(value)
    except ValueError:
        return None


def _error(message: str) -> str:
    """Errors go back to the model as data, so it can correct itself."""
    return _dump({"error": message})


def build_research_tools(
    provider: TravelProvider, request: TripRequest
) -> list[ToolFn]:
    """Bind the read-only travel lookups to one traveller's request."""

    def search_destinations(interests: str, budget_level: str = "medium") -> str:
        """Suggest destinations matching the traveller's interests and budget.

        Use this only when the traveller has not named a destination.

        Args:
            interests: Comma-separated interests, e.g. "food, history, hiking".
            budget_level: One of "low", "medium", "high". Rough daily spend.
        """
        if budget_level not in _BUDGET_LEVELS:
            return _error(f"budget_level must be one of {list(_BUDGET_LEVELS)}")
        parsed = [i.strip() for i in interests.split(",") if i.strip()]
        results = provider.search_destinations(
            interests=parsed, month=request.start_date.month, budget_level=budget_level
        )
        return _dump({"destinations": results})

    def get_weather_outlook(destination: str) -> str:
        """Typical weather for a destination during the trip's dates.

        Args:
            destination: City and country, e.g. "Lisbon, Portugal".
        """
        return _dump(provider.get_weather_outlook(destination, request.start_date))

    def search_flights(origin: str, destination: str, depart_date: str) -> str:
        """Priced flight options for one leg, for the whole party.

        Args:
            origin: Departure city or airport code.
            destination: Arrival city or airport code.
            depart_date: Departure date as YYYY-MM-DD.
        """
        depart = _parse_date(depart_date)
        if depart is None:
            return _error("depart_date must be an ISO date, e.g. 2026-04-12")
        options = provider.search_flights(
            origin=origin,
            destination=destination,
            depart=depart,
            travelers=request.travelers,
        )
        return _dump({"travelers": request.travelers, "options": options})

    def search_lodging(destination: str, check_in: str, nights: int) -> str:
        """Priced lodging options for the stay, sized to the party.

        Args:
            destination: City and country, e.g. "Lisbon, Portugal".
            check_in: Check-in date as YYYY-MM-DD.
            nights: Number of nights.
        """
        start = _parse_date(check_in)
        if start is None:
            return _error("check_in must be an ISO date, e.g. 2026-04-12")
        if nights < 1:
            return _error("nights must be at least 1")
        options = provider.search_lodging(
            destination=destination,
            check_in=start,
            nights=nights,
            travelers=request.travelers,
        )
        return _dump({"nights": nights, "options": options})

    def search_attractions(destination: str, interests: str, limit: int = 8) -> str:
        """Things to do at a destination, with rough cost and duration.

        Args:
            destination: City and country, e.g. "Lisbon, Portugal".
            interests: Comma-separated interests to bias the results.
            limit: Maximum number of results, 1-10.
        """
        parsed = [i.strip() for i in interests.split(",") if i.strip()]
        results = provider.search_attractions(
            destination=destination, interests=parsed, limit=limit
        )
        return _dump({"attractions": results})

    return [
        search_destinations,
        get_weather_outlook,
        search_flights,
        search_lodging,
        search_attractions,
    ]
