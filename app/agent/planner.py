"""Entry point: turn a `TripRequest` into a `PlannedTrip` via the agent graph."""

import logging
import uuid

import groq

from app.agent.graph import build_planner_graph
from app.agent.llm import LLM, GroqLLM
from app.agent.state import initial_state
from app.core.config import Settings, get_settings
from app.models.itinerary import PlannedTrip, TripRequest
from app.providers import get_provider
from app.providers.base import TravelProvider

logger = logging.getLogger(__name__)


class PlanningError(RuntimeError):
    """The agent pipeline ran but never produced a usable itinerary."""


def plan_trip(
    request: TripRequest,
    *,
    llm: LLM | None = None,
    provider: TravelProvider | None = None,
    settings: Settings | None = None,
) -> PlannedTrip:
    """Plan one trip by running it through the flight/hotel/itinerary/response
    agent graph. Blocking; expect several LLM calls and tens of seconds."""
    settings = settings or get_settings()
    provider = provider or get_provider()
    llm = llm or GroqLLM(
        groq.Groq(),
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )

    graph = build_planner_graph(provider, llm, settings.max_itinerary_retries)
    result = graph.invoke(initial_state(request))

    if result["itinerary"] is None:
        reason = "; ".join(result["errors"]) or "no reason recorded"
        raise PlanningError(f"the agent pipeline did not produce an itinerary: {reason}")

    return PlannedTrip(
        id=uuid.uuid4().hex[:12],
        request=request,
        itinerary=result["itinerary"],
        agent_trace=result["messages"],
        summary=result.get("final_response") or "",
    )
