"""The graph's five nodes: a destination coordinator plus the four agents.

Each `build_*_node` closes over the (stateless, shareable) provider and LLM
and returns a plain `state -> dict` callable — the shape LangGraph nodes take.
Building them once and compiling the graph once means `plan_trip` can invoke
the same compiled graph for every request.
"""

import json
from datetime import date

from app.agent.itinerary_json import ItineraryValidationError, parse_itinerary
from app.agent.llm import LLM
from app.agent.prompts import (
    FINAL_RESPONSE_AGENT_SYSTEM,
    FLIGHT_AGENT_SYSTEM,
    HOTEL_AGENT_SYSTEM,
    ITINERARY_AGENT_SYSTEM,
    describe_request,
)
from app.agent.state import TravelState
from app.models.itinerary import FlightLeg, Itinerary, LodgingOption
from app.providers.base import TravelProvider

# Rough daily-spend-per-person bands in rupees, used only to steer destination
# search when the traveller didn't name one. Independent of any provider's
# internal thresholds — providers are free to interpret these labels as they
# like.
_LOW_BUDGET_DAILY = 10_000
_MEDIUM_BUDGET_DAILY = 15_000


def _budget_level(request) -> str:
    if request.budget is None:
        return "high"
    daily_per_person = request.budget / (request.nights + 1) / request.travelers
    if daily_per_person <= _LOW_BUDGET_DAILY:
        return "low"
    if daily_per_person <= _MEDIUM_BUDGET_DAILY:
        return "medium"
    return "high"


def build_resolve_destination_node(provider: TravelProvider):
    """Not one of the four agents in the diagram — a coordinator step that
    only does anything when the traveller left the destination open."""

    def node(state: TravelState) -> dict:
        request = state["request"]
        if request.destination:
            return {"resolved_destination": request.destination}

        candidates = provider.search_destinations(
            interests=request.interests,
            month=request.start_date.month,
            budget_level=_budget_level(request),
        )
        if not candidates:
            return {
                "resolved_destination": "Lisbon, Portugal",
                "messages": ["coordinator: no destination matched; defaulted to Lisbon, Portugal"],
            }

        top = candidates[0]
        return {
            "resolved_destination": top["name"],
            "messages": [f"coordinator: picked {top['name']} (matched {top['matched_interests']})"],
        }

    return node


def build_flight_node(provider: TravelProvider, llm: LLM):
    """Searches both legs — there and back — since a trip needs both to be
    actually bookable. The cheapest option in each list is what ends up on
    the final itinerary (see build_itinerary_node); the LLM's job here is
    just to explain that pick or flag a concern with it."""

    def node(state: TravelState) -> dict:
        request = state["request"]
        if not request.include_flights:
            return {
                "flight_results": {
                    "outbound_options": [],
                    "return_options": [],
                    "recommendation": "Flights weren't requested, so flight search was skipped.",
                },
                "messages": ["flight_agent: skipped (flights not requested)"],
            }
        if not request.origin:
            return {
                "flight_results": {
                    "outbound_options": [],
                    "return_options": [],
                    "recommendation": "No origin was given, so flight search was skipped.",
                },
                "messages": ["flight_agent: skipped (no origin given)"],
            }

        destination = state["resolved_destination"]
        outbound = provider.search_flights(
            origin=request.origin,
            destination=destination,
            depart=request.start_date,
            travelers=request.travelers,
        )
        return_leg = provider.search_flights(
            origin=destination,
            destination=request.origin,
            depart=request.end_date,
            travelers=request.travelers,
        )
        recommendation = llm.complete(
            system=FLIGHT_AGENT_SYSTEM,
            user=json.dumps(
                {
                    "outbound_date": request.start_date,
                    "return_date": request.end_date,
                    "party_size": request.travelers,
                    "budget": request.budget,
                    "booked_outbound": _booked_option(outbound, request.start_date),
                    "booked_return": _booked_option(return_leg, request.end_date),
                    "outbound_options": outbound,
                    "return_options": return_leg,
                },
                default=str,
            ),
        )
        return {
            "flight_results": {
                "outbound_options": outbound,
                "return_options": return_leg,
                "recommendation": recommendation,
            },
            "messages": [
                f"flight_agent: {len(outbound)} outbound, {len(return_leg)} return "
                "option(s) found, recommended a pick for each"
            ],
        }

    return node


def build_hotel_node(provider: TravelProvider, llm: LLM):
    def node(state: TravelState) -> dict:
        request = state["request"]
        if not request.include_hotels:
            return {
                "hotel_results": {
                    "options": [],
                    "recommendation": "Hotels weren't requested, so hotel search was skipped.",
                },
                "messages": ["hotel_agent: skipped (hotels not requested)"],
            }

        options = provider.search_lodging(
            destination=state["resolved_destination"],
            check_in=request.start_date,
            nights=request.nights,
            travelers=request.travelers,
        )
        recommendation = llm.complete(
            system=HOTEL_AGENT_SYSTEM,
            user=json.dumps(
                {
                    "party_size": request.travelers,
                    "interests": request.interests,
                    "budget": request.budget,
                    "options": options,
                },
                default=str,
            ),
        )
        return {
            "hotel_results": {"options": options, "recommendation": recommendation},
            "messages": [f"hotel_agent: {len(options)} option(s) found, recommended one"],
        }

    return node


def _booked_option(options: list[dict], on: date) -> dict | None:
    """The cheapest option departing on the trip's own date, or — only if
    there's none — the cheapest in the whole search window.

    Providers sort by total ascending (see mock.py), so the first match is the
    cheapest. The date preference matters because a live search covers a day
    either side of the travel date: without it, a flight a day early would be
    booked just for being ₹100 cheaper.

    Used by both the flight node (to tell the flight agent what *is* booked,
    rather than asking it to predict) and the itinerary node (to book it), so
    the explanation and the booking can't disagree.
    """
    if not options:
        return None
    same_day = [o for o in options if str(o.get("depart_date")) == on.isoformat()]
    return (same_day or options)[0]


def _booked_flight(options: list[dict], rationale: str, on: date) -> FlightLeg | None:
    option = _booked_option(options, on)
    return FlightLeg(**option, rationale=rationale) if option else None


def _lodging_options(options: list[dict]) -> list[LodgingOption]:
    return [LodgingOption(**option) for option in options]


def _flight_options(options: list[dict]) -> list[FlightLeg]:
    """Every option the search returned, for the cheapest-vs-fastest table —
    not just the one that gets booked."""
    return [FlightLeg(**option) for option in options]


def build_itinerary_node(provider: TravelProvider, llm: LLM, max_retries: int):
    def node(state: TravelState) -> dict:
        request = state["request"]
        destination = state["resolved_destination"]
        attractions = provider.search_attractions(destination, request.interests, limit=10)
        weather = provider.get_weather_outlook(destination, request.start_date)

        # Selected deterministically here — and reused below when the full
        # Itinerary is assembled — so the numbers shown to the traveller and
        # the numbers the itinerary agent budgets against are the same ones.
        flight_results = state["flight_results"]
        rationale = flight_results.get("recommendation", "")
        outbound_flight = _booked_flight(
            flight_results.get("outbound_options", []), rationale, on=request.start_date
        )
        return_flight = _booked_flight(
            flight_results.get("return_options", []), rationale, on=request.end_date
        )
        lodging = _lodging_options(state["hotel_results"].get("options", []))

        flight_cost = (outbound_flight.total or 0 if outbound_flight else 0) + (
            return_flight.total or 0 if return_flight else 0
        )
        hotel_cost = lodging[0].total or 0 if lodging else 0

        if not request.include_flights:
            flight_note = (
                "The traveller didn't ask for flights, so no flight is included in the cost "
                "below. Don't plan or cost one yourself."
            )
        elif not request.origin:
            flight_note = "No origin was given, so no flight is included in the cost below."
        elif outbound_flight is None and return_flight is None:
            flight_note = "No flight fares were found, so no flight is included in the cost below."
        else:
            flight_note = (
                f"Flights already booked: Rs {flight_cost:,.0f} total. "
                "Do not plan or re-cost the flight yourself."
            )
        if not request.include_hotels:
            hotel_note = (
                "The traveller didn't ask for a hotel, so no lodging is included in the cost "
                "below. Don't plan or cost one yourself."
            )
        elif lodging:
            hotel_note = (
                f"Hotel already booked: {lodging[0].name}, Rs {hotel_cost:,.0f} total. "
                "Do not plan or re-cost lodging yourself."
            )
        else:
            hotel_note = "No lodging options were found."

        # Handed over pre-computed, not left for the model to derive from
        # start/end dates — asking it to count calendar days itself invites
        # exactly the off-by-one mistake parse_itinerary's validation exists
        # to catch (seen live: a 5-day trip came back with 4 `days` entries,
        # three attempts running, feedback notwithstanding).
        trip_dates = [d.isoformat() for d in request.dates]

        base_prompt = (
            f"{describe_request(request)}\n"
            f"Destination: {destination}\n\n"
            f"Weather outlook: {json.dumps(weather, default=str)}\n\n"
            f"{flight_note}\n"
            f"{hotel_note}\n\n"
            f"Candidate attractions: {json.dumps(attractions, default=str)}\n\n"
            f"`days` must have exactly one entry for each of these {len(trip_dates)} dates, "
            f"in this order — do not compute the date range yourself: "
            f"{json.dumps(trip_dates)}"
        )

        feedback = ""
        last_error = ""
        attempts = max_retries + 1
        for attempt in range(1, attempts + 1):
            user_prompt = base_prompt
            if feedback:
                user_prompt += (
                    f"\n\nYour previous attempt was rejected: {feedback}\n"
                    "Fix it and return the complete JSON again."
                )

            raw = llm.complete(system=ITINERARY_AGENT_SYSTEM, user=user_prompt, json_mode=True)
            try:
                draft = parse_itinerary(raw, request)
            except ItineraryValidationError as exc:
                feedback = exc.feedback
                last_error = exc.feedback
                continue

            activity_cost = sum(
                activity.estimated_cost or 0
                for day in draft.days
                for activity in day.activities
            )
            itinerary = Itinerary(
                destination=destination,
                start_date=request.start_date,
                end_date=request.end_date,
                travelers=request.travelers,
                outbound_flight=outbound_flight,
                return_flight=return_flight,
                outbound_options=_flight_options(flight_results.get("outbound_options", [])),
                return_options=_flight_options(flight_results.get("return_options", [])),
                lodging_options=lodging,
                days=draft.days,
                total_estimated_cost=flight_cost + hotel_cost + activity_cost,
                notes=draft.notes,
            )
            return {
                "itinerary": itinerary,
                "messages": [
                    f"itinerary_agent: built a {len(itinerary.days)}-day plan (attempt {attempt})"
                ],
            }

        return {
            "itinerary": None,
            "errors": [f"itinerary_agent: gave up after {attempts} attempt(s): {last_error}"],
            "messages": [f"itinerary_agent: failed after {attempts} attempt(s)"],
        }

    return node


def build_final_response_node(llm: LLM):
    def node(state: TravelState) -> dict:
        itinerary = state["itinerary"]
        if itinerary is None:
            # Nothing to summarise — the itinerary agent already recorded why.
            return {"final_response": ""}

        # Deliberately not passing along flight_results/hotel_results'
        # free-text recommendations here: itinerary already carries the
        # actual selected flights and hotel (see build_itinerary_node), and
        # including both risked the model narrating the agent's opinion
        # instead of what was actually booked.
        summary = llm.complete(
            system=FINAL_RESPONSE_AGENT_SYSTEM,
            user=f"Itinerary: {itinerary.model_dump_json()}",
        )
        return {
            "final_response": summary.strip(),
            "messages": ["final_response_agent: wrote the trip summary"],
        }

    return node
