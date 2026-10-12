"""The route planner: where the trip sleeps, decided before anything is searched.

Flights and hotels used to be searched for the destination exactly as typed.
"Kerala and Tamil Nadu" isn't an airport, so a two-state trip got no flights,
and one hotel search across two whole states returned hostels in towns the
trip never visited. Now one short LLM call (`route_model`, gpt-4o) decides the
towns slept in, the nights in each, and the airports to fly into and out of:

- flights search those two airports (app/agent/nodes.py, `flight_agent`);
- each town gets its own hotel search, for its own dates (`hotel_agent`);
- the itinerary agent writes the days around the route instead of deciding
  it, which also fixes the uneven split between regions.

Its answer is checked in code (nights add up, at most `MAX_STOPS` towns, the
airports exist) and retried with the reason. If it never comes right, the
trip falls back to one stop at the destination, as before routes existed.
"""

import json
import logging

from pydantic import ValidationError

from app.agent.events import status
from app.agent.llm import LLM
from app.agent.prompts import ROUTE_PLAN_SYSTEM, describe_request
from app.agent.state import TravelState
from app.models.itinerary import RoutePlan, RouteStop, TripRequest
from app.providers.base import TravelProvider

logger = logging.getLogger(__name__)

# Each stop is one hotel search, against 250 searches a month.
MAX_STOPS = 4


def single_stop(request: TripRequest, destination: str, reasoning: str) -> RoutePlan:
    """The whole trip in one place: the route before routes existed."""
    return RoutePlan(
        stops=[RouteStop(city=destination, area="", nights=request.nights)],
        fly_into=destination,
        fly_out_of=destination,
        reasoning=reasoning,
    )


def describe_route(plan: RoutePlan) -> str:
    """'Thiruvananthapuram 2 nights → Kochi 3 → Madurai 4'."""
    parts = []
    for i, stop in enumerate(plan.stops):
        unit = (" night" if stop.nights == 1 else " nights") if i == 0 else ""
        parts.append(f"{stop.city} {stop.nights}{unit}")
    return " → ".join(parts)


def _needs_flights(request: TripRequest) -> bool:
    return request.include_flights and bool(request.origin)


def _problem(plan: RoutePlan, request: TripRequest, provider: TravelProvider) -> str | None:
    """What's wrong with the plan, worded for the model to fix; None if nothing."""
    if not plan.stops:
        return "`stops` is empty; give at least one town."
    if len(plan.stops) > MAX_STOPS:
        return f"There are {len(plan.stops)} stops; use at most {MAX_STOPS}."
    for stop in plan.stops:
        if stop.nights < 1:
            return f"{stop.city} has {stop.nights} nights; every stop needs at least 1."
    total = sum(stop.nights for stop in plan.stops)
    if total != request.nights:
        return f"The nights add up to {total}, but the trip has {request.nights} nights."
    if not _needs_flights(request):
        return None
    has_airport = getattr(provider, "has_airport", None)
    for key in ("fly_into", "fly_out_of"):
        city = getattr(plan, key)
        if not city:
            return f"`{key}` is missing; the traveller is flying from {request.origin}."
        if has_airport is not None and not has_airport(city):
            return (
                f"No airport found for {city!r} (`{key}`). Give the nearest city with a "
                "commercial airport, by its official name."
            )
    return None


def plan_route(
    request: TripRequest,
    destination: str,
    llm: LLM,
    provider: TravelProvider,
    attempts: int = 3,
) -> tuple[RoutePlan, str]:
    """The route, and a line for the trip's trace saying how it was decided."""
    if request.nights == 0:
        return single_stop(request, destination, "A day trip."), "route_planner: day trip"

    flights = (
        f"Flights: needed, to and from {request.origin}."
        if _needs_flights(request)
        else "Flights: not needed."
    )
    base_prompt = (
        f"{describe_request(request)}\n"
        f"Destination: {destination}\n"
        f"The trip has {request.nights} nights.\n"
        f"{flights}"
    )

    feedback = ""
    for attempt in range(1, attempts + 1):
        prompt = base_prompt
        if feedback:
            prompt += f"\n\nYour previous answer was rejected: {feedback} Fix it."
        raw = llm.complete(system=ROUTE_PLAN_SYSTEM, user=prompt, json_mode=True, schema=RoutePlan)
        try:
            plan = RoutePlan.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            feedback = f"that wasn't the JSON asked for: {exc}"
            continue
        feedback = _problem(plan, request, provider) or ""
        if not feedback:
            return plan, f"route_planner: {describe_route(plan)} (attempt {attempt})"
        logger.warning("route attempt %d rejected: %s", attempt, feedback)

    logger.warning("no usable route for %s; one stop instead: %s", destination, feedback)
    plan = single_stop(request, destination, "No usable route came back.")
    return plan, f"route_planner: fell back to one stop ({feedback})"


def build_route_plan_node(
    provider: TravelProvider, llm: LLM, route_model: str | None = None
):
    """The route step. Runs on `route_model` when the LLM can switch models
    (`OpenAILLM.with_model`); a test's fake answers as itself."""
    route_llm = llm.with_model(route_model) if route_model and hasattr(llm, "with_model") else llm

    def node(state: TravelState) -> dict:
        destination = state["resolved_destination"]
        status("route_plan", f"Deciding which towns to stay in for {destination}…")
        plan, note = plan_route(state["request"], destination, route_llm, provider)
        status("route_plan", describe_route(plan), state="done")
        return {"route_plan": plan, "messages": [note]}

    return node
