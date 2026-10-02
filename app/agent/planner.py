"""Entry points: turn a `TripRequest` — or free text — into a `PlannedTrip`.

Both run the trip graph (app/agent/graph.py). The prompt box itself goes
through the larger ask graph instead (app/agent/assistant.py, app/jobs.py),
which routes the sentence first; these remain for `POST /trips/plan`,
`POST /trips/plan-from-prompt` and the CLI, which only ever plan trips.
"""

import logging
from datetime import date

import openai

from app.agent.errors import PlanningError
from app.agent.graph import build_planner_graph
from app.agent.llm import LLM, OpenAILLM
from app.agent.prompt_parser import apply_page_choices, parse_trip_prompt
from app.agent.state import initial_state
from app.core.config import Settings, get_settings
from app.models.itinerary import PlannedTrip, TripRequest
from app.providers import get_provider
from app.providers.base import TravelProvider

logger = logging.getLogger(__name__)

__all__ = ["PlanningError", "plan_parsed_trip", "plan_trip", "plan_trip_from_prompt"]


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
    agent graph. Blocking; expect several LLM calls and tens of seconds.
    Raises PlanningError if no itinerary came out of it."""
    settings = settings or get_settings()
    provider = provider or get_provider()
    llm = llm or _build_llm(settings)

    graph = build_planner_graph(provider, llm, settings.max_itinerary_retries)
    return graph.invoke(initial_state(request))["trip"]


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
    """Parse one free-text request, then plan it. Builds the LLM once and
    reuses it for both the parsing step and the whole planning graph.

    `include_flights` / `include_hotels` come from the page's checkboxes,
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
    """Apply the page's own choices to a request parsed from free text
    (`apply_page_choices`), then plan it."""
    request = apply_page_choices(
        parsed,
        include_flights=include_flights,
        include_hotels=include_hotels,
        places_per_city=places_per_city,
        start_date=start_date,
        end_date=end_date,
    )
    return plan_trip(request, llm=llm, provider=provider, settings=settings)
