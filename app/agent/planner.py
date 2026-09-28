"""Entry points: turn a `TripRequest` — or free text — into a `PlannedTrip`."""

import logging
import uuid
from datetime import date

import openai

from app.agent.graph import build_planner_graph
from app.agent.llm import LLM, OpenAILLM
from app.agent.prompt_parser import PromptParseError, parse_trip_prompt
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

    # A fresh plan starts its own thread at version 1. `id` identifies this
    # version; `thread_id` is what a later revision is addressed to, and is
    # the same value here only because nothing has been revised yet.
    trip_id = uuid.uuid4().hex[:12]
    return PlannedTrip(
        id=trip_id,
        thread_id=trip_id,
        version=1,
        request=request,
        itinerary=result["itinerary"],
        agent_trace=result["messages"],
        summary=result.get("final_response") or "",
    )


def plan_trip_from_prompt(
    prompt: str,
    *,
    include_flights: bool = True,
    include_hotels: bool = True,
    start_date: date | None = None,
    end_date: date | None = None,
    llm: LLM | None = None,
    provider: TravelProvider | None = None,
    settings: Settings | None = None,
) -> PlannedTrip:
    """Parse one free-text request, then plan it — the frontend's single
    prompt box calls this. Builds the LLM once and reuses it for both the
    parsing step and the whole planning graph.

    `include_flights` / `include_hotels` come from the frontend's checkboxes,
    and `start_date` / `end_date` from its calendar pickers, not from the
    sentence, so they're applied after parsing rather than left for the model
    to infer. Picked dates replace any the sentence implied.
    """
    if (start_date is None) != (end_date is None):
        raise ValueError("give both start_date and end_date, or neither")

    settings = settings or get_settings()
    llm = llm or _build_llm(settings)

    return plan_parsed_trip(
        parse_trip_prompt(prompt, llm),
        include_flights=include_flights,
        include_hotels=include_hotels,
        start_date=start_date,
        end_date=end_date,
        llm=llm,
        provider=provider,
        settings=settings,
    )


def plan_parsed_trip(
    parsed: TripRequest,
    *,
    include_flights: bool,
    include_hotels: bool,
    start_date: date | None,
    end_date: date | None,
    llm: LLM,
    places_per_city: str | None = None,
    provider: TravelProvider | None = None,
    settings: Settings | None = None,
) -> PlannedTrip:
    """Apply the page's own choices to a request parsed from free text, then
    plan it. Shared by `plan_trip_from_prompt` and app/agent/assistant.py.

    `places_per_city` always replaces whatever the parser read, so a saved
    trip records what was actually looked up — nothing, unless the page's
    Restaurants box asked for it.
    """
    overrides: dict = {
        "include_flights": include_flights,
        "include_hotels": include_hotels,
        "places_per_city": places_per_city,
    }
    if start_date and end_date:
        overrides |= {"start_date": start_date, "end_date": end_date}
    # Re-validated rather than model_copy'd, so picked dates go through the
    # same ordering and length checks as parsed ones.
    request = TripRequest.model_validate(parsed.model_dump() | overrides)
    # Without this, a missing destination falls through to resolve_destination,
    # which picks from the provider's sample list — "Varanasi to Kerala and
    # Tamil Nadu" came back as five days in Chiang Mai. Asking is better than
    # planning a trip nobody requested.
    if not request.destination:
        raise PromptParseError(
            "no destination was named. Say where you want to go — a city, state, "
            "region or country."
        )
    return plan_trip(request, llm=llm, provider=provider, settings=settings)
