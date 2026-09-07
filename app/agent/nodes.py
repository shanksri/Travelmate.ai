"""The graph's five nodes: a destination coordinator plus the four agents.

Each `build_*_node` closes over the (stateless, shareable) provider and LLM
and returns a plain `state -> dict` callable — the shape LangGraph nodes take.
Building them once and compiling the graph once means `plan_trip` can invoke
the same compiled graph for every request.
"""

import json

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
from app.providers.base import TravelProvider

# Rough daily-spend-per-person bands, used only to steer destination search
# when the traveller didn't name one. Independent of any provider's internal
# thresholds — providers are free to interpret these labels as they like.
_LOW_BUDGET_DAILY_USD = 120
_MEDIUM_BUDGET_DAILY_USD = 180


def _budget_level(request) -> str:
    if request.budget_usd is None:
        return "high"
    daily_per_person = request.budget_usd / (request.nights + 1) / request.travelers
    if daily_per_person <= _LOW_BUDGET_DAILY_USD:
        return "low"
    if daily_per_person <= _MEDIUM_BUDGET_DAILY_USD:
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
    def node(state: TravelState) -> dict:
        request = state["request"]
        if not request.origin:
            return {
                "flight_results": {
                    "options": [],
                    "recommendation": "No origin was given, so flight search was skipped.",
                },
                "messages": ["flight_agent: skipped (no origin given)"],
            }

        options = provider.search_flights(
            origin=request.origin,
            destination=state["resolved_destination"],
            depart=request.start_date,
            travelers=request.travelers,
        )
        recommendation = llm.complete(
            system=FLIGHT_AGENT_SYSTEM,
            user=json.dumps(
                {
                    "party_size": request.travelers,
                    "budget_usd": request.budget_usd,
                    "options": options,
                },
                default=str,
            ),
        )
        return {
            "flight_results": {"options": options, "recommendation": recommendation},
            "messages": [f"flight_agent: {len(options)} option(s) found, recommended one"],
        }

    return node


def build_hotel_node(provider: TravelProvider, llm: LLM):
    def node(state: TravelState) -> dict:
        request = state["request"]
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
                    "budget_usd": request.budget_usd,
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


def build_itinerary_node(provider: TravelProvider, llm: LLM, max_retries: int):
    def node(state: TravelState) -> dict:
        request = state["request"]
        destination = state["resolved_destination"]
        attractions = provider.search_attractions(destination, request.interests, limit=10)
        weather = provider.get_weather_outlook(destination, request.start_date)

        base_prompt = (
            f"{describe_request(request)}\n"
            f"Destination: {destination}\n\n"
            f"Weather outlook: {json.dumps(weather, default=str)}\n\n"
            f"Flight pick: {state['flight_results'].get('recommendation', 'n/a')}\n"
            f"Flight options considered: "
            f"{json.dumps(state['flight_results'].get('options', []), default=str)}\n\n"
            f"Hotel pick: {state['hotel_results'].get('recommendation', 'n/a')}\n"
            f"Hotel options considered: "
            f"{json.dumps(state['hotel_results'].get('options', []), default=str)}\n\n"
            f"Candidate attractions: {json.dumps(attractions, default=str)}"
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
                itinerary = parse_itinerary(raw, request)
            except ItineraryValidationError as exc:
                feedback = exc.feedback
                last_error = exc.feedback
                continue

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

        summary = llm.complete(
            system=FINAL_RESPONSE_AGENT_SYSTEM,
            user=(
                f"Flight pick: {state['flight_results'].get('recommendation', 'n/a')}\n"
                f"Hotel pick: {state['hotel_results'].get('recommendation', 'n/a')}\n"
                f"Itinerary: {itinerary.model_dump_json()}"
            ),
        )
        return {
            "final_response": summary.strip(),
            "messages": ["final_response_agent: wrote the trip summary"],
        }

    return node
