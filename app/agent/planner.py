"""Entry points: turn a `TripRequest` — or free text — into a `PlannedTrip`."""

import logging
import uuid

import openai

from app.agent.graph import build_planner_graph
from app.agent.llm import LLM, OpenAILLM
from app.agent.prompt_parser import parse_trip_prompt
from app.agent.state import initial_state
from app.core.config import Settings, get_settings
from app.models.itinerary import PlannedTrip, TripRequest
from app.providers import get_provider
from app.providers.base import TravelProvider

logger = logging.getLogger(__name__)


class PlanningError(RuntimeError):
    """The agent pipeline ran but never produced a usable itinerary."""


def _build_llm(settings: Settings) -> LLM:
    return OpenAILLM(
        openai.OpenAI(),
        model=settings.model,
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )


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
    llm = llm or _build_llm(settings)

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


def plan_trip_from_prompt(
    prompt: str,
    *,
    llm: LLM | None = None,
    provider: TravelProvider | None = None,
    settings: Settings | None = None,
) -> PlannedTrip:
    """Parse one free-text request, then plan it — the frontend's single
    prompt box calls this. Builds the LLM once and reuses it for both the
    parsing step and the whole planning graph."""
    settings = settings or get_settings()
    llm = llm or _build_llm(settings)

    request = parse_trip_prompt(prompt, llm)
    return plan_trip(request, llm=llm, provider=provider, settings=settings)
