"""The trip graph's steps: a destination coordinator, the four agents, and
`package_trip`, which turns their output into a saveable `PlannedTrip`.

Each `build_*_node` closes over the (stateless, shareable) provider and LLM
and returns a plain `state -> dict` callable — the shape LangGraph nodes take.

Steps report progress with `app.agent.events` (a status line as each starts,
and each day as the itinerary agent finishes writing it). That reaches the
page when the graph is streamed (app/jobs.py) and does nothing otherwise.
"""

import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date

from pydantic import ValidationError

from app.agent.errors import PlanningError
from app.agent.events import emit, status
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
from app.agent.stream_json import DayStreamer
from app.models.itinerary import (
    DataSource,
    DayPlan,
    DraftItinerary,
    FlightLeg,
    Itinerary,
    LodgingOption,
    PlannedTrip,
)
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
        status("flights", f"Searching flights from {request.origin} to {destination}…")
        # Both legs at once: they're independent, and a live search takes
        # ~4 s each, so searching them one after the other doubled the wait.
        with ThreadPoolExecutor(max_workers=2) as pool:
            outbound_search = pool.submit(
                _search_with_source,
                provider,
                "flights",
                origin=request.origin,
                destination=destination,
                depart=request.start_date,
                travelers=request.travelers,
            )
            return_search = pool.submit(
                _search_with_source,
                provider,
                "flights",
                origin=destination,
                destination=request.origin,
                depart=request.end_date,
                travelers=request.travelers,
            )
            (outbound, outbound_source), (return_leg, return_source) = (
                outbound_search.result(),
                return_search.result(),
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
                "outbound_source": outbound_source,
                "return_source": return_source,
                "recommendation": recommendation,
            },
            "messages": [
                f"flight_agent: {len(outbound)} outbound{_status(outbound_source)}, "
                f"{len(return_leg)} return{_status(return_source)} option(s) found, "
                "recommended a pick for each"
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

        status("hotels", f"Searching hotels in {state['resolved_destination']}…")
        options, source = _search_with_source(
            provider,
            "lodging",
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
            "hotel_results": {
                "options": options,
                "source": source,
                "recommendation": recommendation,
            },
            "messages": [
                f"hotel_agent: {len(options)} option(s) found{_status(source)}, recommended one"
            ],
        }

    return node


def _search_with_source(
    provider: TravelProvider, kind: str, **query
) -> tuple[list[dict], DataSource | None]:
    """`search_<kind>_with_source` where the provider has it — every real
    one does — and the plain search, with no source, for a bare test fake."""
    with_source = getattr(provider, f"search_{kind}_with_source", None)
    if with_source is not None:
        return with_source(**query)
    return getattr(provider, f"search_{kind}")(**query), None


def _status(source: DataSource | None) -> str:
    return f" ({source.status})" if source is not None else ""


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


def _write_itinerary(llm: LLM, user_prompt: str) -> str:
    """The itinerary agent's answer. Streamed where the LLM supports it, so
    each day can be shown the moment it's written; the full text is still
    returned and validated as a whole afterwards."""
    stream = getattr(llm, "stream", None)
    if stream is None:
        return llm.complete(
            system=ITINERARY_AGENT_SYSTEM, user=user_prompt, json_mode=True, schema=DraftItinerary
        )

    streamer = DayStreamer()
    parts: list[str] = []
    for chunk in stream(
        system=ITINERARY_AGENT_SYSTEM, user=user_prompt, json_mode=True, schema=DraftItinerary
    ):
        parts.append(chunk)
        for raw_day in streamer.feed(chunk):
            try:
                day = DayPlan.model_validate(raw_day)
            except ValidationError:
                continue  # left for the full validation to report
            emit({"type": "day", "day": day.model_dump(mode="json")})
    return "".join(parts)


def build_itinerary_node(provider: TravelProvider, llm: LLM, max_retries: int):
    def node(state: TravelState) -> dict:
        request = state["request"]
        destination = state["resolved_destination"]
        status("itinerary", "Writing your day-by-day plan…")
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

            if attempt > 1:
                # Days already shown came from an attempt that was rejected.
                emit({"type": "days_reset"})
            raw = _write_itinerary(llm, user_prompt)
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
                outbound_source=flight_results.get("outbound_source"),
                return_source=flight_results.get("return_source"),
                lodging_source=state["hotel_results"].get("source"),
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

        status("summary", "Writing the trip summary…")
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


def build_package_trip_node():
    """The last trip step: the itinerary and summary as a `PlannedTrip`, ready
    to save. A fresh plan starts its own thread at version 1."""

    def node(state: TravelState) -> dict:
        itinerary = state.get("itinerary")
        if itinerary is None:
            reason = "; ".join(state.get("errors") or []) or "no reason recorded"
            raise PlanningError(f"the agent pipeline did not produce an itinerary: {reason}")

        # `id` identifies this version; `thread_id` is what a later revision
        # is addressed to, and is the same value here only because nothing
        # has been revised yet.
        trip_id = uuid.uuid4().hex[:12]
        return {
            "trip": PlannedTrip(
                id=trip_id,
                thread_id=trip_id,
                version=1,
                request=state["request"],
                itinerary=itinerary,
                agent_trace=state["messages"],
                summary=state.get("final_response") or "",
            )
        }

    return node
