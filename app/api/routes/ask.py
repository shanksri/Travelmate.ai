"""POST /ask — the prompt box's endpoint: a trip, places, or a route.

`POST /trips/plan-from-prompt` still plans a trip from any sentence, for
callers that only want trips. This one lets the sentence decide.
"""

from fastapi import APIRouter, HTTPException, status

from app.agent.assistant import Answer, answer_prompt
from app.agent.prompt_parser import PromptParseError
from app.api.errors import run_translating_errors
from app.api.schemas import PlanFromPromptRequest
from app.store import get_store

router = APIRouter(tags=["ask"])


@router.post("/ask", response_model=Answer)
def ask(payload: PlanFromPromptRequest) -> Answer:
    """Defined `def`, not `async def`: a trip runs the whole agent graph, and
    a Maps answer makes a blocking MCP call — both belong in a worker thread.

    Only trips are saved. Places and routes are shown live: Google's terms
    for Maps Grounding Lite forbid storing its results.
    """
    try:
        answer = run_translating_errors(
            lambda: answer_prompt(
                payload.prompt,
                include_flights=payload.include_flights,
                include_hotels=payload.include_hotels,
                include_restaurants=payload.include_restaurants,
                start_date=payload.start_date,
                end_date=payload.end_date,
            )
        )
    except PromptParseError as exc:
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_CONTENT, f"Couldn't understand that request: {exc}"
        ) from exc

    if answer.trip is not None:
        get_store().save(answer.trip)
    return answer
