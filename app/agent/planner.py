"""The agentic loop that turns a `TripRequest` into an `Itinerary`.

Claude drives: it researches with the tools in `app.agent.tools`, then commits
by calling `submit_itinerary`. We own the stopping conditions — an iteration
cap, one retry if it forgets to submit, and a hard failure if it never does.
"""

import logging
import uuid

import anthropic
from anthropic import beta_tool

from app.agent.prompts import SYSTEM_PROMPT, build_briefing
from app.agent.tools import ItinerarySink, build_research_tools, build_submit_tool
from app.core.config import Settings, get_settings
from app.models.itinerary import PlannedTrip, TripRequest
from app.providers import get_provider
from app.providers.base import TravelProvider

logger = logging.getLogger(__name__)

_NUDGE = (
    "You have not called submit_itinerary yet, so nothing has been delivered. "
    "Submit the complete itinerary now using the research you already have."
)


class PlanningError(RuntimeError):
    """The agent finished without producing a usable itinerary."""


class _Run:
    """One pass of the tool loop, with the message history mirrored.

    The SDK runner keeps its own history and does not expose it, so we mirror
    it as we iterate — that is what lets us restart the loop (to nudge, or to
    resume a paused turn) instead of starting the whole plan over.
    """

    def __init__(
        self,
        client: anthropic.Anthropic,
        settings: Settings,
        tools: list,
        briefing: str,
    ) -> None:
        self.client = client
        self.settings = settings
        self.tools = tools
        self.messages: list = [{"role": "user", "content": briefing}]
        self.tool_calls: list[str] = []
        self.last_text = ""
        self.stop_reason: str | None = None

    def step(self) -> None:
        runner = self.client.beta.messages.tool_runner(
            model=self.settings.model,
            max_tokens=self.settings.max_tokens,
            system=SYSTEM_PROMPT,
            tools=self.tools,
            messages=self.messages,
            thinking={"type": "adaptive"},
            output_config={"effort": self.settings.effort},
            max_iterations=self.settings.max_tool_iterations,
        )

        for message in runner:
            self.stop_reason = message.stop_reason
            self.messages.append({"role": "assistant", "content": message.content})

            for block in message.content:
                if block.type == "tool_use":
                    self.tool_calls.append(block.name)
                elif block.type == "text" and block.text.strip():
                    self.last_text = block.text.strip()

            # Cached by the SDK — the tools themselves still run exactly once.
            tool_response = runner.generate_tool_call_response()
            if tool_response is not None:
                self.messages.append(tool_response)

    def nudge(self, text: str) -> None:
        self.messages.append({"role": "user", "content": text})


def plan_trip(
    request: TripRequest,
    *,
    client: anthropic.Anthropic | None = None,
    provider: TravelProvider | None = None,
    settings: Settings | None = None,
) -> PlannedTrip:
    """Plan one trip. Blocking; expect tens of seconds and several API calls."""
    settings = settings or get_settings()
    provider = provider or get_provider()
    client = client or anthropic.Anthropic()

    sink = ItinerarySink()
    tools = [beta_tool(fn) for fn in build_research_tools(provider, request)]
    tools.append(beta_tool(build_submit_tool(sink, request)))

    run = _Run(client, settings, tools, build_briefing(request))
    run.step()

    # A paused turn is resumable as-is; a finished turn with nothing submitted
    # needs to be told so. Either way we restart the loop at most once.
    if not sink.received:
        if run.stop_reason == "pause_turn":
            logger.info("turn paused; resuming")
        else:
            logger.info("agent stopped without submitting; nudging once")
            run.nudge(_NUDGE)
        run.step()

    if sink.itinerary is None:
        raise PlanningError(
            f"the agent made {len(run.tool_calls)} tool calls but never submitted "
            f"an itinerary (stop_reason={run.stop_reason})"
        )

    return PlannedTrip(
        id=uuid.uuid4().hex[:12],
        request=request,
        itinerary=sink.itinerary,
        tool_calls=run.tool_calls,
        summary=run.last_text,
    )
