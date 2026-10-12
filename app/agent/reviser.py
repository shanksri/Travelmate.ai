"""Apply one requested change to a trip that was already planned.

Deliberately *not* the five-node planning graph. A revision first routes the
change (ROUTE_CHANGE_SYSTEM) to the parts it's about — the outbound flight,
the return flight, the hotel, the day plan — and then touches only those:

- Flights and the hotel are re-picked from the options the original search
  already found, by index. Nothing is searched again (no SerpApi quota), and
  nothing is typed by a model, so a new pick can't be hallucinated.
- The day plan is re-generated only when the change is about it, with
  flights and lodging withheld from that prompt so it can't alter them.

Dates and destination never change. The result is appended as the next
version of the same thread rather than replacing it — see app/store.py.

The agent remembers the trip's earlier changes: what was asked for, whether
it was made, and what it changed. That's what lets "undo that" or "go back
to the other hotel" mean something. The memory is a channel of the graph's
state that a checkpointer carries from one change to the next, under the
trip's thread_id (app/agent/memory.py).
"""

import json
import logging
import uuid
from typing import Annotated, TypedDict

import openai
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.agent.events import status
from app.agent.itinerary_json import ItineraryValidationError, parse_itinerary
from app.agent.llm import LLM, OpenAILLM
from app.agent.memory import get_checkpointer
from app.agent.prompts import (
    FINAL_RESPONSE_AGENT_SYSTEM,
    REVISE_ITINERARY_SYSTEM,
    ROUTE_CHANGE_SYSTEM,
)
from app.core.config import Settings, get_settings
from app.models.itinerary import (
    DayPlan,
    FlightLeg,
    Itinerary,
    LodgingOption,
    PlannedTrip,
    TripRequest,
    draft_itinerary_for,
)

logger = logging.getLogger(__name__)


class RevisionError(RuntimeError):
    """The change was understood but no valid revised itinerary came back."""


class RevisionDeclined(RuntimeError):
    """The change wasn't made — nothing it asked for could be applied.

    Raised instead of saving a version identical to the one before it, so the
    traveller is told why rather than shown the same plan again.
    """


# REVISE_ITINERARY_SYSTEM tells the model to start its explanation with this
# when it declines, so the reason can be found among the other notes.
DECLINE_PREFIX = "Couldn't apply:"


def _decline_reason(draft_notes: list[str], previous_notes: list[str]) -> str:
    for note in draft_notes:
        if note.strip().startswith(DECLINE_PREFIX):
            return note.strip().removeprefix(DECLINE_PREFIX).strip()
    # The model may explain itself without the prefix; a note it just added
    # is the next best thing.
    new_notes = [note for note in draft_notes if note not in previous_notes]
    if new_notes:
        return " ".join(new_notes)
    return "the planner returned the plan unchanged."


def _build_llm(settings: Settings) -> LLM:
    return OpenAILLM(
        openai.OpenAI(),
        model=settings.revise_model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


# --- routing -----------------------------------------------------------------


class ChangeRoute(BaseModel):
    """Which parts of the trip a change is about, as ROUTE_CHANGE_SYSTEM
    decides them. Indexes point into the trip's existing option lists."""

    model_config = ConfigDict(populate_by_name=True)

    outbound: int | None = None
    return_leg: int | None = Field(default=None, alias="return")
    hotel: int | None = None
    itinerary: str | None = None
    declined: str | None = None
    reasoning: str = ""


def _same_flight(a: FlightLeg | None, b: FlightLeg | None) -> bool:
    if a is None or b is None:
        return False
    return a.model_dump(exclude={"rationale"}) == b.model_dump(exclude={"rationale"})


def _flight_menu(options: list[FlightLeg], booked: FlightLeg | None) -> list[dict] | str:
    if not options:
        # Worded as the traveller's choice, not a gap: "none — this trip has
        # no hotel" read to the router as a reason to refuse day changes.
        return "none: the traveller planned this trip without flights"
    return [
        {
            "index": i,
            "carrier": f.carrier,
            "departs": (f.departure_at or f.depart_date).isoformat(),
            "stops": f.stops,
            "duration_hours": f.duration_hours,
            "total": f.total,
            "booked": _same_flight(f, booked),
        }
        for i, f in enumerate(options)
    ]


def _hotel_menu(options: list[LodgingOption]) -> list[dict] | str:
    if not options:
        return "none: the traveller planned this trip without a hotel"
    return [
        {
            "index": i,
            "name": h.name,
            "tier": h.tier,
            "rating": h.rating,
            "neighbourhood": h.neighbourhood,
            "nightly": h.nightly,
            "total": h.total,
            "selected": i == 0,
        }
        for i, h in enumerate(options)
    ]


def _check_indexes(route: ChangeRoute, itinerary: Itinerary) -> str | None:
    """Feedback for the router when it named an option that isn't there."""
    for name, index, options in (
        ("outbound", route.outbound, itinerary.outbound_options),
        ("return", route.return_leg, itinerary.return_options),
        ("hotel", route.hotel, itinerary.lodging_options),
    ):
        if index is not None and not 0 <= index < len(options):
            return f"`{name}` is {index}, but there are only {len(options)} {name} option(s)."
    return None


def _route_change(
    previous: PlannedTrip,
    change_request: str,
    memory: list["EditTurn"],
    llm: LLM,
    attempts: int,
) -> ChangeRoute:
    itinerary = previous.itinerary
    question = {
        "change_request": change_request,
        "outbound_options": _flight_menu(itinerary.outbound_options, itinerary.outbound_flight),
        "return_options": _flight_menu(itinerary.return_options, itinerary.return_flight),
        "hotel_options": _hotel_menu(itinerary.lodging_options),
        "days": [
            {"day": d.day, "date": d.date.isoformat(), "summary": d.summary}
            for d in itinerary.days
        ],
    }
    if memory:
        question["earlier_changes"] = _recall(memory)
    base_prompt = json.dumps(question, default=str)

    feedback = ""
    for _ in range(attempts):
        user_prompt = base_prompt
        if feedback:
            user_prompt += f"\n\nYour previous answer was rejected: {feedback} Fix it."
        raw = llm.complete(
            system=ROUTE_CHANGE_SYSTEM, user=user_prompt, json_mode=True, schema=ChangeRoute
        )
        try:
            route = ChangeRoute.model_validate(json.loads(raw))
        except (json.JSONDecodeError, ValidationError) as exc:
            feedback = f"that wasn't the JSON asked for: {exc}"
            continue
        feedback = _check_indexes(route, itinerary) or ""
        if not feedback:
            return route

    raise RevisionError(f"couldn't work out what the change is about: {feedback}")


# --- applying each part ------------------------------------------------------


def _revise_days(
    itinerary: Itinerary,
    request: TripRequest,
    instruction: str,
    llm: LLM,
    attempts: int,
    recalled: str = "",
) -> tuple[list[DayPlan], list[str]]:
    """Re-plan the days for `instruction`. Raises RevisionDeclined if the
    model hands every day back unchanged. `recalled` is the agent's memory of
    earlier changes, as `_recall_for_days` words it."""
    # Only the days go to the model. Flights and lodging are withheld on
    # purpose — it cannot change what it cannot see. The previous notes are
    # withheld too: a note from an earlier refusal ("X is not included…")
    # anchored the model into refusing the same place again, and it writes
    # fresh notes every time anyway.
    current_days = json.dumps(
        [day.model_dump(mode="json") for day in itinerary.days], default=str
    )
    trip_dates = [d.isoformat() for d in request.dates]

    base_prompt = (
        f"Destination: {itinerary.destination}\n"
        f"Travellers: {itinerary.travelers}\n"
        f"Pace: {request.pace}\n"
        f"Current plan: {current_days}\n\n"
        f"{recalled}"
        f"The traveller asks: {instruction}\n\n"
        f"Return all {len(trip_dates)} days, using exactly these dates in this "
        f"order: {json.dumps(trip_dates)}"
    )

    feedback = ""
    last_error = ""
    for _ in range(attempts):
        user_prompt = base_prompt
        if feedback:
            user_prompt += (
                f"\n\nYour previous attempt was rejected: {feedback}\n"
                "Fix it and return the complete JSON again."
            )

        raw = llm.complete(
            system=REVISE_ITINERARY_SYSTEM,
            user=user_prompt,
            json_mode=True,
            schema=draft_itinerary_for(len(trip_dates)),
        )
        try:
            draft = parse_itinerary(raw, request)
        except ItineraryValidationError as exc:
            feedback = last_error = exc.feedback
            continue

        unchanged = [day.model_dump(mode="json") for day in draft.days] == [
            day.model_dump(mode="json") for day in itinerary.days
        ]
        if unchanged:
            raise RevisionDeclined(_decline_reason(draft.notes, itinerary.notes))
        return draft.days, draft.notes

    raise RevisionError(f"gave up after {attempts} attempt(s): {last_error}")


def _rebooked(option: FlightLeg, change_request: str) -> FlightLeg:
    return option.model_copy(update={"rationale": f"Chosen on request: {change_request}"})


# --- the agent's memory ------------------------------------------------------
#
# The last few changes asked for on this trip: whether each was made, and
# what it changed. It's RevisionState's `memory` channel, which no input
# sets, so the checkpointer brings it back from the trip's previous change
# while every other channel starts fresh (`_new_change`).

# How many earlier changes it remembers. Older ones drop off, so the prompts
# stay the same size however long a trip is edited.
MEMORY_TURNS = 6


class EditTurn(BaseModel):
    """One change the traveller asked for, as the revision agent remembers it."""

    request: str
    applied: bool
    version: int | None = Field(default=None, description="The version it produced.")
    changes: list[str] = Field(default_factory=list, description="What it changed, in words.")
    days_before: list[DayPlan] = Field(
        default_factory=list, description="The days it replaced, exactly as they were."
    )


def _remember(memory: list[EditTurn], new: list[EditTurn]) -> list[EditTurn]:
    """The `memory` channel's reducer: add the new turn, keep the latest few."""
    return (memory + new)[-MEMORY_TURNS:]


def _flight_words(leg: FlightLeg | None) -> str:
    if leg is None:
        return "no flight"
    when = leg.departure_at.strftime("%H:%M") if leg.departure_at else leg.depart_date.isoformat()
    return f"{leg.carrier} at {when}"


def _what_changed(before: Itinerary, after: Itinerary) -> tuple[list[str], list[DayPlan]]:
    """What a change did, in words, and the days it replaced as they were."""
    changes: list[str] = []
    days_before: list[DayPlan] = []
    for old, new in zip(before.days, after.days, strict=False):
        if old.model_dump(mode="json") == new.model_dump(mode="json"):
            continue
        days_before.append(old)
        if old.summary == new.summary:
            changes.append(f'Day {old.day} ("{old.summary}"): its activities changed')
        else:
            changes.append(f'Day {old.day}: "{old.summary}" became "{new.summary}"')

    for name, old_leg, new_leg in (
        ("Outbound flight", before.outbound_flight, after.outbound_flight),
        ("Return flight", before.return_flight, after.return_flight),
    ):
        if (old_leg or new_leg) and not _same_flight(old_leg, new_leg):
            changes.append(f"{name}: {_flight_words(old_leg)} became {_flight_words(new_leg)}")

    old_hotel = before.lodging_options[0].name if before.lodging_options else None
    new_hotel = after.lodging_options[0].name if after.lodging_options else None
    if old_hotel != new_hotel:
        changes.append(f"Hotel: {old_hotel} became {new_hotel}")
    return changes, days_before


def _recall(memory: list[EditTurn]) -> list[str]:
    """The earlier changes in words, oldest first."""
    lines = []
    for turn in memory:
        if turn.applied:
            lines.append(
                f'"{turn.request}": made (version {turn.version}). ' + "; ".join(turn.changes)
            )
        else:
            # Without the reason it was declined. A refusal shown to the model
            # anchors it into refusing again (see `_revise_days`), and the
            # traveller may be asking again because the refusal was wrong.
            lines.append(f'"{turn.request}": not made')
    return lines


def _recall_for_days(memory: list[EditTurn]) -> str:
    """The memory section of the day planner's prompt: the earlier changes in
    words, plus the days the latest one replaced in full, which is what
    "undo that" needs to put them back exactly."""
    if not memory:
        return ""
    numbered = [f"{i}. {line}" for i, line in enumerate(_recall(memory), 1)]
    text = "Earlier changes to this trip, oldest first:\n" + "\n".join(numbered)
    latest = next((t for t in reversed(memory) if t.applied and t.days_before), None)
    if latest:
        replaced = json.dumps([day.model_dump(mode="json") for day in latest.days_before])
        text += f'\n\nThe days "{latest.request}" replaced, exactly as they were before it: '
        text += replaced
    return text + "\n\n"


# --- the revision graph ----------------------------------------------------------
#
#   START -> route_change -> (declined) -------------------------------> remember -> END
#                         -> apply_picks -> revise_days -> package_revision -> remember -> END
#                                        -> package_revision -----------> remember -> END
#
# `revise_days` runs only when the change is about the days; any step can
# end it early with a reason in `declined`, and `revise_trip` raises that.
# Every path ends in `remember`, which writes the change into the memory,
# made or not.


class RevisionState(TypedDict, total=False):
    previous: PlannedTrip
    change_request: str
    route: ChangeRoute | None
    outbound: FlightLeg | None
    return_flight: FlightLeg | None
    lodging: list[LodgingOption]
    days: list[DayPlan]
    notes: list[str]
    trace: list[str]
    declined: str | None
    trip: PlannedTrip | None
    # Kept from one change to the next by the checkpointer; see above.
    memory: Annotated[list[EditTurn], _remember]


def _new_change(previous: PlannedTrip, change_request: str) -> RevisionState:
    """The input for one change. It sets every channel except `memory`, so
    nothing from the trip's previous change leaks into this one, while the
    memory, left out, is restored from the checkpoint."""
    return {
        "previous": previous,
        "change_request": change_request,
        "route": None,
        "outbound": None,
        "return_flight": None,
        "lodging": [],
        "days": [],
        "notes": [],
        "trace": [],
        "declined": None,
        "trip": None,
    }


def _has_nothing_to_repick(itinerary: Itinerary) -> bool:
    return (
        not itinerary.outbound_options
        and not itinerary.return_options
        and not itinerary.lodging_options
    )


def _build_route_change_node(llm: LLM, attempts: int):
    def node(state: RevisionState) -> dict:
        previous = state["previous"]
        itinerary = previous.itinerary
        memory = state.get("memory", [])
        status("route_change", "Working out what to change…")
        recalled = (
            [f"memory: recalled {len(memory)} earlier change(s) to this trip"] if memory else []
        )
        start = {
            "outbound": itinerary.outbound_flight,
            "return_flight": itinerary.return_flight,
            "lodging": list(itinerary.lodging_options),
            "days": itinerary.days,
            "notes": itinerary.notes,
        }
        if _has_nothing_to_repick(itinerary):
            # Nothing to re-pick, so the only thing a change can touch is the
            # days. Routing it anyway once refused "adjust more days to tamil
            # nadu" (2 runs in 5) because "the trip has no hotel". This is
            # every trip planned with Flights and Hotels unticked, the page's
            # default, and skipping the router saves a call. A request that
            # really is about flights or a hotel is declined by the day
            # planner instead.
            return start | {
                "route": ChangeRoute(itinerary=state["change_request"]),
                "trace": [
                    *recalled,
                    "change_router: skipped (no flights or hotel to re-pick; "
                    "the change goes to the days)",
                ],
            }

        route = _route_change(previous, state["change_request"], memory, llm, attempts)
        if route.declined:
            return {"route": route, "declined": route.declined}
        return start | {
            "route": route,
            "trace": [*recalled, f"change_router: {route.reasoning or 'routed the change'}"],
        }

    return node


def _build_apply_picks_node():
    def node(state: RevisionState) -> dict:
        route, itinerary = state["route"], state["previous"].itinerary
        change_request = state["change_request"]
        trace = list(state["trace"])
        update: dict = {"trace": trace}

        if route.outbound is not None:
            pick = itinerary.outbound_options[route.outbound]
            if not _same_flight(pick, state["outbound"]):
                update["outbound"] = _rebooked(pick, change_request)
                trace.append(f"flights: outbound rebooked to option {route.outbound}")

        if route.return_leg is not None:
            pick = itinerary.return_options[route.return_leg]
            if not _same_flight(pick, state["return_flight"]):
                update["return_flight"] = _rebooked(pick, change_request)
                trace.append(f"flights: return rebooked to option {route.return_leg}")

        # The first lodging option is the selected one everywhere (the
        # itinerary node's cost, the page's "Selected" tag), so switching
        # means moving it to the front.
        if route.hotel:
            lodging = list(state["lodging"])
            lodging.insert(0, lodging.pop(route.hotel))
            update["lodging"] = lodging
            trace.append(f"hotel: switched to {lodging[0].name}")
        return update

    return node


def _build_revise_days_node(llm: LLM, attempts: int):
    def node(state: RevisionState) -> dict:
        status("revise_days", "Re-planning the days…")
        previous = state["previous"]
        try:
            days, notes = _revise_days(
                previous.itinerary,
                previous.request,
                state["route"].itinerary,
                llm,
                attempts,
                recalled=_recall_for_days(state.get("memory", [])),
            )
        except RevisionDeclined as exc:
            return {"declined": str(exc)}
        return {
            "days": days,
            "notes": notes,
            "trace": [*state["trace"], "reviser: re-planned the days"],
        }

    return node


def _build_package_revision_node(llm: LLM):
    def node(state: RevisionState) -> dict:
        previous = state["previous"]
        itinerary = previous.itinerary
        outbound, return_flight = state["outbound"], state["return_flight"]
        lodging, days = state["lodging"], state["days"]

        flight_cost = (outbound.total or 0 if outbound else 0) + (
            return_flight.total or 0 if return_flight else 0
        )
        hotel_cost = lodging[0].total or 0 if lodging else 0
        activity_cost = sum(
            activity.estimated_cost or 0 for day in days for activity in day.activities
        )
        revised = Itinerary(
            destination=itinerary.destination,
            start_date=itinerary.start_date,
            end_date=itinerary.end_date,
            travelers=itinerary.travelers,
            outbound_flight=outbound,
            return_flight=return_flight,
            outbound_options=itinerary.outbound_options,
            return_options=itinerary.return_options,
            lodging_options=lodging,
            outbound_source=itinerary.outbound_source,
            return_source=itinerary.return_source,
            lodging_source=itinerary.lodging_source,
            days=days,
            currency=itinerary.currency,
            total_estimated_cost=flight_cost + hotel_cost + activity_cost,
            notes=state["notes"],
        )
        if not _what_changed(itinerary, revised)[0]:
            # Routed, but everything it pointed at was already the case —
            # e.g. "book the cheapest flight" when the cheapest is booked.
            return {
                "declined": state["route"].reasoning
                or "the plan already matches what was asked for."
            }

        status("summary", "Writing the trip summary…")
        summary = llm.complete(
            system=FINAL_RESPONSE_AGENT_SYSTEM,
            user=f"Itinerary: {revised.model_dump_json()}",
        )
        trace = [*state["trace"], "final_response_agent: rewrote the trip summary"]
        return {
            "trace": trace,
            "trip": PlannedTrip(
                id=uuid.uuid4().hex[:12],
                thread_id=previous.thread_id,
                version=previous.version + 1,
                change_note=state["change_request"],
                request=previous.request,
                itinerary=revised,
                agent_trace=trace,
                summary=summary.strip(),
            ),
        }

    return node


def _remember_node(state: RevisionState) -> dict:
    """Write this change into the agent's memory, made or not."""
    trip = state.get("trip")
    if state.get("declined") or trip is None:
        turn = EditTurn(request=state["change_request"], applied=False)
    else:
        changes, days_before = _what_changed(state["previous"].itinerary, trip.itinerary)
        turn = EditTurn(
            request=state["change_request"],
            applied=True,
            version=trip.version,
            changes=changes,
            days_before=days_before,
        )
    return {"memory": [turn]}


def _unless_declined(next_step: str):
    def choose(state: RevisionState) -> str:
        return "remember" if state.get("declined") else next_step

    return choose


def _after_picks(state: RevisionState) -> str:
    return "revise_days" if state["route"].itinerary else "package_revision"


def build_revision_graph(
    llm: LLM, attempts: int, checkpointer: BaseCheckpointSaver | None = None
) -> CompiledStateGraph:
    graph = StateGraph(RevisionState)
    graph.add_node("route_change", _build_route_change_node(llm, attempts))
    graph.add_node("apply_picks", _build_apply_picks_node())
    graph.add_node("revise_days", _build_revise_days_node(llm, attempts))
    graph.add_node("package_revision", _build_package_revision_node(llm))
    graph.add_node("remember", _remember_node)

    graph.add_edge(START, "route_change")
    graph.add_conditional_edges(
        "route_change", _unless_declined("apply_picks"), ["apply_picks", "remember"]
    )
    graph.add_conditional_edges(
        "apply_picks", _after_picks, ["revise_days", "package_revision"]
    )
    graph.add_conditional_edges(
        "revise_days", _unless_declined("package_revision"), ["package_revision", "remember"]
    )
    graph.add_edge("package_revision", "remember")
    graph.add_edge("remember", END)
    return graph.compile(checkpointer=checkpointer)


def _memory_of(thread_id: str) -> dict:
    """The checkpoint thread holding the revision agent's memory of a trip.
    Prefixed, so a graph checkpointed under the same trip later can't land
    in this one's state."""
    return {"configurable": {"thread_id": f"revision:{thread_id}"}}


def revise_trip(
    previous: PlannedTrip,
    change_request: str,
    *,
    llm: LLM | None = None,
    settings: Settings | None = None,
    checkpointer: BaseCheckpointSaver | None = None,
) -> PlannedTrip:
    """Produce the next version of `previous` with `change_request` applied,
    by running the revision graph.

    It never touches the trip store: it reads `previous` and returns a new
    `PlannedTrip`, and the caller decides whether to keep it. What it does
    keep is its own memory of the change, made or declined, in the
    checkpointer (by default the app's, app/agent/memory.py). Raises
    `RevisionDeclined` when nothing the change asked for could be applied; a
    change is applied whole or not at all.
    """
    settings = settings or get_settings()
    llm = llm or _build_llm(settings)
    graph = build_revision_graph(
        llm, settings.max_itinerary_retries + 1, checkpointer or get_checkpointer()
    )

    result = graph.invoke(_new_change(previous, change_request), _memory_of(previous.thread_id))
    if result.get("declined"):
        raise RevisionDeclined(result["declined"])
    return result["trip"]


def remembered_changes(
    thread_id: str, checkpointer: BaseCheckpointSaver | None = None
) -> list[EditTurn]:
    """What the revision agent remembers about a trip, oldest first."""
    graph = build_revision_graph(None, 1, checkpointer or get_checkpointer())
    return graph.get_state(_memory_of(thread_id)).values.get("memory", [])
