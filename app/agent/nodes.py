"""The trip graph's steps: a destination coordinator, the four agents, and
`package_trip`, which turns their output into a saveable `PlannedTrip`.

Each `build_*_node` closes over the (stateless, shareable) provider and LLM
and returns a plain `state -> dict` callable — the shape LangGraph nodes take.

Steps report progress with `app.agent.events` (a status line as each starts,
and each day as the itinerary agent finishes writing it). That reaches the
page when the graph is streamed (app/jobs.py) and does nothing otherwise.
"""

import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import ValidationError

from app.agent.errors import PlanningError
from app.agent.events import emit, status, stop_if_cancelled
from app.agent.itinerary_json import ItineraryValidationError, parse_itinerary
from app.agent.llm import LLM
from app.agent.prompts import (
    FINAL_RESPONSE_AGENT_SYSTEM,
    FLIGHT_AGENT_SYSTEM,
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
    HotelStay,
    Itinerary,
    LodgingOption,
    PlannedTrip,
    RouteStop,
    TripRequest,
    draft_itinerary_for,
)
from app.providers.base import TravelProvider

logger = logging.getLogger(__name__)

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


def _route_stops(state: TravelState) -> list[RouteStop]:
    """The towns the trip sleeps in, from the route step; one stop at the
    destination when there's no route (a state built by hand in a test)."""
    plan = state.get("route_plan")
    if plan and plan.stops:
        return plan.stops
    return [RouteStop(city=state["resolved_destination"], area="", nights=state["request"].nights)]


def _stays_for(request: TripRequest, stops: list[RouteStop]) -> list[HotelStay]:
    """One stay per stop, its check-in following on from the one before."""
    stays, check_in = [], request.start_date
    for stop in stops:
        stays.append(HotelStay(city=stop.city, check_in=check_in, nights=stop.nights))
        check_in += timedelta(days=stop.nights)
    return stays


def _flight_ends(state: TravelState) -> tuple[str, str]:
    """Where the outbound flight lands and the return flight leaves from."""
    plan = state.get("route_plan")
    destination = state["resolved_destination"]
    if plan is None:
        return destination, destination
    return plan.fly_into or destination, plan.fly_out_of or destination


def build_flight_node(provider: TravelProvider, llm: LLM):
    """Searches both legs, there and back, since a trip needs both to be
    actually bookable: into the route's first airport and out of its last
    (`route_plan`), which needn't be the same one. The cheapest option on
    each travel date is what gets booked (see build_itinerary_node); the
    LLM's job here is just to explain that pick or flag a concern with it."""

    def node(state: TravelState) -> dict:
        request = state["request"]
        if not request.include_flights:
            status("flights", "Not requested", state="skipped")
            return {
                "flight_results": {
                    "outbound_options": [],
                    "return_options": [],
                    "recommendation": "Flights weren't requested, so flight search was skipped.",
                },
                "messages": ["flight_agent: skipped (flights not requested)"],
            }
        if not request.origin:
            status("flights", "No starting city given", state="skipped")
            return {
                "flight_results": {
                    "outbound_options": [],
                    "return_options": [],
                    "recommendation": "No origin was given, so flight search was skipped.",
                },
                "messages": ["flight_agent: skipped (no origin given)"],
            }

        arrive, leave = _flight_ends(state)
        status("flights", f"{request.origin} → {arrive}, and {leave} → {request.origin}…")
        # Both legs at once: they're independent, and a live search takes
        # ~4 s each, so searching them one after the other doubled the wait.
        with ThreadPoolExecutor(max_workers=2) as pool:
            outbound_search = pool.submit(
                _search_with_source,
                provider,
                "flights",
                origin=request.origin,
                destination=arrive,
                depart=request.start_date,
                travelers=request.travelers,
            )
            return_search = pool.submit(
                _search_with_source,
                provider,
                "flights",
                origin=leave,
                destination=request.origin,
                depart=request.end_date,
                travelers=request.travelers,
            )
            (outbound, outbound_source), (return_leg, return_source) = (
                outbound_search.result(),
                return_search.result(),
            )

        found = (
            f"flight_agent: {len(outbound)} outbound{_status(outbound_source)}, "
            f"{len(return_leg)} return{_status(return_source)} option(s) found"
        )
        if outbound or return_leg:
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
            found += ", recommended a pick for each"
        else:
            # Nothing to explain, so no call.
            recommendation = ""

        summary = f"{len(outbound)} outbound, {len(return_leg)} return options"
        missing = [s.detail for s in (outbound_source, return_source) if s and s.detail]
        if not (outbound and return_leg) and missing:
            summary += f" ({missing[0]})"
        status("flights", summary, state="done" if outbound and return_leg else "warn")
        return {
            "flight_results": {
                "outbound_options": outbound,
                "return_options": return_leg,
                "outbound_source": outbound_source,
                "return_source": return_source,
                "recommendation": recommendation,
            },
            "messages": [found],
        }

    return node


def _hotel_place(stop: RouteStop) -> str:
    """What's searched: "Kochi, Kerala, India", not a bare "Kochi" that
    Google could take for Kochi in Japan."""
    return f"{stop.city}, {stop.area}" if stop.area else stop.city


def build_hotel_node(provider: TravelProvider):
    """One hotel search per town on the route, for that town's own nights,
    all at once. The cheapest of each town's options is booked. No LLM: an
    explanation of the pick was written here once, and nothing read it."""

    def node(state: TravelState) -> dict:
        request = state["request"]
        stops = _route_stops(state)
        stays = _stays_for(request, stops)
        if not request.include_hotels:
            status("hotels", "Not requested", state="skipped")
            return {
                "hotel_results": {"stays": stays},
                "messages": ["hotel_agent: skipped (hotels not requested)"],
            }

        searched = [i for i, stop in enumerate(stops) if stop.nights > 0]
        status("hotels", ", ".join(stops[i].city for i in searched) + "…")
        with ThreadPoolExecutor(max_workers=max(1, len(searched))) as pool:
            searches = {
                i: pool.submit(
                    _search_with_source,
                    provider,
                    "lodging",
                    destination=_hotel_place(stops[i]),
                    check_in=stays[i].check_in,
                    nights=stops[i].nights,
                    travelers=request.travelers,
                )
                for i in searched
            }
            for i, search in searches.items():
                options, source = search.result()
                stays[i] = stays[i].model_copy(
                    update={"options": _lodging_options(options), "source": source}
                )

        found = [
            f"{len(stays[i].options)} in {stays[i].city}{_status(stays[i].source)}"
            for i in searched
        ]
        everywhere = all(stays[i].options for i in searched)
        status(
            "hotels",
            " · ".join(f"{stays[i].city} {len(stays[i].options)}" for i in searched),
            state="done" if everywhere else "warn",
        )
        return {
            "hotel_results": {"stays": stays},
            "messages": [f"hotel_agent: option(s) found: {', '.join(found)}"],
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


def _write_itinerary(llm: LLM, user_prompt: str, day_count: int) -> str:
    """The itinerary agent's answer. Streamed where the LLM supports it, so
    each day can be shown the moment it's written; the full text is still
    returned and validated as a whole afterwards."""
    schema = draft_itinerary_for(day_count)
    stream = getattr(llm, "stream", None)
    if stream is None:
        return llm.complete(
            system=ITINERARY_AGENT_SYSTEM, user=user_prompt, json_mode=True, schema=schema
        )

    streamer = DayStreamer()
    parts: list[str] = []
    for chunk in stream(
        system=ITINERARY_AGENT_SYSTEM, user=user_prompt, json_mode=True, schema=schema
    ):
        # The longest call of a plan. Stopping here closes the stream, and
        # OpenAI stops writing (and billing) the rest of the answer.
        stop_if_cancelled()
        parts.append(chunk)
        for raw_day in streamer.feed(chunk):
            try:
                day = DayPlan.model_validate(raw_day)
            except ValidationError:
                continue  # left for the full validation to report
            emit({"type": "day", "day": day.model_dump(mode="json")})
    return "".join(parts)


# --- the itinerary agent ---------------------------------------------------------
#
# A subgraph with its own state, which the trip graph never sees: the trip
# graph hands it a prompt and gets back a draft. Its short-term memory is
# `rejections`, why each earlier attempt failed validation. Every retry is
# shown all of them, not only the last, so fixing one mistake doesn't bring
# back the one before it.
#
#   START -> write_days -> check_days -> END   (a valid draft, or out of attempts)
#                ^              |
#                +--------------+              (rejected: again, with every reason)
#
# Nothing here is checkpointed: the memory only matters while the trip is
# being written. The days it streams reach the page because app/jobs.py
# streams with `subgraphs=True`.


class ItineraryAgentState(TypedDict, total=False):
    request: TripRequest
    base_prompt: str
    day_count: int
    attempts: int
    raw: str
    rejections: list[str]
    draft: DraftItinerary | None


def _with_rejections(base_prompt: str, rejections: list[str]) -> str:
    if not rejections:
        return base_prompt
    if len(rejections) == 1:
        return (
            f"{base_prompt}\n\nYour previous attempt was rejected: {rejections[0]}\n"
            "Fix it and return the complete JSON again."
        )
    listed = "\n".join(f"{i}. {reason}" for i, reason in enumerate(rejections, 1))
    return (
        f"{base_prompt}\n\nYour earlier attempts were rejected, for these reasons:\n"
        f"{listed}\nFix all of them and return the complete JSON again."
    )


def build_itinerary_agent(llm: LLM) -> CompiledStateGraph:
    def write_days(state: ItineraryAgentState) -> dict:
        if state["rejections"]:
            # Days already shown came from an attempt that was rejected.
            emit({"type": "days_reset"})
            status("itinerary", "An attempt didn't pass the checks; writing it again…")
        prompt = _with_rejections(state["base_prompt"], state["rejections"])
        return {"raw": _write_itinerary(llm, prompt, state["day_count"])}

    def check_days(state: ItineraryAgentState) -> dict:
        try:
            return {"draft": parse_itinerary(state["raw"], state["request"])}
        except ItineraryValidationError as exc:
            attempt = len(state["rejections"]) + 1
            logger.warning("itinerary attempt %d rejected: %s", attempt, exc.feedback)
            return {"rejections": [*state["rejections"], exc.feedback]}

    def next_step(state: ItineraryAgentState) -> str:
        if state.get("draft") is not None or len(state["rejections"]) >= state["attempts"]:
            return END
        return "write_days"

    graph = StateGraph(ItineraryAgentState)
    graph.add_node("write_days", write_days)
    graph.add_node("check_days", check_days)
    graph.add_edge(START, "write_days")
    graph.add_edge("write_days", "check_days")
    graph.add_conditional_edges("check_days", next_step, ["write_days", END])
    return graph.compile()


def _leg_words(leg: FlightLeg) -> str:
    when = leg.departure_at.strftime("%Y-%m-%d %H:%M") if leg.departure_at else leg.depart_date
    length = f", about {leg.duration_hours:g} h" if leg.duration_hours else ""
    return f"{leg.carrier} {leg.origin} → {leg.destination}, departing {when}{length}"


def _base_by_date(request: TripRequest, stays: list[HotelStay]) -> dict[str, str]:
    """The town each day is based in: where that night is spent, and on the
    last day, the town the trip leaves from."""
    bases: dict[str, str] = {}
    for stay in stays:
        for offset in range(stay.nights):
            bases[(stay.check_in + timedelta(days=offset)).isoformat()] = stay.city
    last = stays[-1].city
    return {day.isoformat(): bases.get(day.isoformat(), last) for day in request.dates}


def build_itinerary_node(llm: LLM, max_retries: int):
    agent = build_itinerary_agent(llm)

    def node(state: TravelState) -> dict:
        request = state["request"]
        destination = state["resolved_destination"]
        trip_dates = [d.isoformat() for d in request.dates]
        status("itinerary", f"Writing {len(trip_dates)} days…")

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
        stays = state["hotel_results"].get("stays") or _stays_for(request, _route_stops(state))

        flight_cost = (outbound_flight.total or 0 if outbound_flight else 0) + (
            return_flight.total or 0 if return_flight else 0
        )
        hotel_cost = sum(stay.booked.total or 0 for stay in stays if stay.booked)

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
            legs = [
                f"{name}: {_leg_words(leg)}"
                for name, leg in (("Outbound", outbound_flight), ("Return", return_flight))
                if leg
            ]
            flight_note = (
                f"Flights already booked ({'; '.join(legs)}): Rs {flight_cost:,.0f} total. "
                "Do not plan or re-cost the flights yourself."
            )
        if not request.include_hotels:
            hotel_note = (
                "The traveller didn't ask for hotels, so no lodging is included in the cost "
                "below. Don't plan or cost any yourself."
            )
        elif any(stay.booked for stay in stays):
            hotel_note = (
                f"Hotels already booked, one per town (above): Rs {hotel_cost:,.0f} total. "
                "Do not plan or re-cost lodging yourself."
            )
        else:
            hotel_note = "No hotel options were found."

        route = "\n".join(
            f"- {stay.city}: {stay.nights} night(s), {stay.check_in.isoformat()} to "
            f"{stay.check_out.isoformat()}"
            + (f"; hotel: {stay.booked.name}" if stay.booked else "")
            for stay in stays
        )

        # Dates handed over pre-computed, not left for the model to derive
        # from start/end dates — asking it to count calendar days itself
        # invites exactly the off-by-one mistake parse_itinerary's validation
        # exists to catch (seen live: a 5-day trip came back with 4 `days`
        # entries, three attempts running, feedback notwithstanding). The
        # town for each date likewise.
        base_prompt = (
            f"{describe_request(request)}\n"
            f"Destination: {destination}\n\n"
            f"Route, fixed (plan the days around it):\n{route}\n"
            f"Each day's `city`, by date. Use exactly these: "
            f"{json.dumps(_base_by_date(request, stays))}\n\n"
            f"{flight_note}\n"
            f"{hotel_note}\n\n"
            f"`days` must have exactly one entry for each of these {len(trip_dates)} dates, "
            f"in this order — do not compute the date range yourself: "
            f"{json.dumps(trip_dates)}"
        )

        attempts = max_retries + 1
        result = agent.invoke(
            {
                "request": request,
                "base_prompt": base_prompt,
                "day_count": len(trip_dates),
                "attempts": attempts,
                "rejections": [],
                "draft": None,
            },
            # Two steps per attempt; LangGraph's default limit is 25.
            {"recursion_limit": 2 * attempts + 2},
        )
        draft, rejections = result.get("draft"), result["rejections"]
        if draft is None:
            return {
                "itinerary": None,
                "errors": [
                    f"itinerary_agent: gave up after {attempts} attempt(s): {rejections[-1]}"
                ],
                "messages": [f"itinerary_agent: failed after {attempts} attempt(s)"],
            }

        activity_cost = sum(
            activity.estimated_cost or 0 for day in draft.days for activity in day.activities
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
            outbound_source=flight_results.get("outbound_source"),
            return_source=flight_results.get("return_source"),
            stays=stays,
            days=draft.days,
            total_estimated_cost=flight_cost + hotel_cost + activity_cost,
            notes=draft.notes,
        )
        status("itinerary", f"{len(itinerary.days)} days", state="done")
        return {
            "itinerary": itinerary,
            "messages": [
                f"itinerary_agent: built a {len(itinerary.days)}-day plan "
                f"(attempt {len(rejections) + 1})"
            ],
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
        status("summary", "Written", state="done")
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

        # What comes next on the page: places in each city, or nothing.
        kind = state["request"].places_per_city
        if kind:
            status("restaurants", f"Finding {kind} in each city…")
        else:
            status("restaurants", "Not requested", state="skipped")

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
