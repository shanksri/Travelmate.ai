"""How each way a planning, revising or answering call can fail is reported.

One mapping for both delivery routes: an HTTP error response for the
blocking endpoints (`run_translating_errors`), and an `error` event for a
streamed job (app/jobs.py). The same failure reads the same either way.
"""

import logging
from collections.abc import Callable

import openai
from fastapi import HTTPException, status

from app.agent.errors import PlanningError
from app.agent.prompt_parser import PromptParseError
from app.agent.reviser import RevisionDeclined, RevisionError
from app.providers.google_maps import GoogleMapsError

logger = logging.getLogger(__name__)


def describe_error(exc: Exception) -> tuple[int, str] | None:
    """(HTTP status, message for the traveller) for a failure we know how to
    explain, or None for one we don't — a bug, which should surface as one."""
    if isinstance(exc, openai.AuthenticationError):
        logger.warning("openai auth failed: %s", exc)
        return (
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "OpenAI credentials are missing or invalid — set OPENAI_API_KEY.",
        )
    if isinstance(exc, openai.RateLimitError):
        return status.HTTP_429_TOO_MANY_REQUESTS, "Rate limited by the OpenAI API."
    if isinstance(exc, openai.APIStatusError):
        logger.error("openai api error %s: %s", exc.status_code, exc.message)
        return status.HTTP_502_BAD_GATEWAY, f"OpenAI API error: {exc.message}"
    if isinstance(exc, openai.APIConnectionError):
        return status.HTTP_504_GATEWAY_TIMEOUT, "Could not reach the OpenAI API."
    if isinstance(exc, PlanningError):
        logger.error("planning failed: %s", exc)
        return status.HTTP_502_BAD_GATEWAY, f"The planner did not finish: {exc}"
    if isinstance(exc, RevisionDeclined):
        # Nothing is saved: a version identical to the last one would only
        # pad the history. The reason is the planner's own, for the traveller.
        return status.HTTP_422_UNPROCESSABLE_CONTENT, f"Couldn't apply this change: {exc}"
    if isinstance(exc, RevisionError):
        logger.error("revision failed: %s", exc)
        return status.HTTP_502_BAD_GATEWAY, f"The change could not be applied: {exc}"
    if isinstance(exc, GoogleMapsError):
        logger.error("google maps failed: %s", exc)
        return status.HTTP_502_BAD_GATEWAY, f"Google Maps: {exc}"
    if isinstance(exc, PromptParseError):
        return status.HTTP_422_UNPROCESSABLE_CONTENT, f"Couldn't understand that request: {exc}"
    return None


def run_translating_errors[T](call: Callable[[], T], *, parse_errors: bool = False) -> T:
    """Run a call, turning the ways it can fail into HTTP errors.

    PromptParseError is left to the caller unless `parse_errors` is set: the
    trip endpoints word it as "that trip request", the ask endpoint as "that
    request".
    """
    try:
        return call()
    except Exception as exc:
        if isinstance(exc, PromptParseError) and not parse_errors:
            raise
        described = describe_error(exc)
        if described is None:
            raise
        raise HTTPException(*described) from exc
