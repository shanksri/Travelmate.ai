import json
from contextlib import asynccontextmanager
from pathlib import Path

import openai
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import health, trips
from app.core.logging import configure_logging
from app.providers.aviationstack import fetch_flights
from app.providers.tavily import fetch_search
from app.store import get_store

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build the store (and, for SqlTripStore, create its table) at boot
    # rather than on the first request — a bad TRAVELMATE_DATABASE_URL should
    # fail the container immediately, not the first trip a user plans.
    get_store()
    yield


app = FastAPI(
    title="travelmate.ai",
    version="0.1.0",
    description="TripMate AI: a LangGraph multi-agent travel planner backed by OpenAI.",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(trips.router)


# --- Frontend -----------------------------------------------------------
#
# Plain HTML/CSS/JS, no build step — served by this same process so the page
# and the API it calls are same-origin (no CORS needed). "/static" is a
# mount, matched before any route below it only for paths under that prefix,
# so it can't shadow /health or /trips.

app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.middleware("http")
async def no_heuristic_caching_for_static(request: Request, call_next):
    """Starlette's StaticFiles sends Last-Modified/ETag but no Cache-Control,
    so browsers fall back to heuristic freshness and can serve a stale
    index.html/app.js/style.css after an edit with no revalidation at all —
    caught live during development. `no-cache` still allows a cheap 304 via
    the ETag; it just stops the browser from skipping the check."""
    response = await call_next(request)
    if request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(FRONTEND_DIR / "index.html", headers={"Cache-Control": "no-cache"})


# --- Function calling for the four agents -----------------------------------
#
# One real OpenAI call per agent, each given exactly one tool bound to a real
# fetch client (app/providers/aviationstack.py, app/providers/tavily.py) — not
# the mock provider. Just the calling part, as asked: each function makes one
# tools-enabled chat completion, and if the model chooses to call its tool,
# executes the real fetch and attaches the result. It does not feed that
# result back for a second, natural-language turn, and none of this replaces
# app/agent/nodes.py — that graph is still what /trips/plan actually runs.
#
# Every call below costs real OpenAI tokens and, when the model calls its
# tool, a real AviationStack/Tavily request too — so this only runs when this
# file is executed directly (`python -m app.main`), never on `import app.main`
# (which every `uvicorn app.main:app` startup, and the test suite, does).

_FLIGHT_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_flights",
        "description": "Look up real-time flight data by departure/arrival airport or status.",
        "parameters": {
            "type": "object",
            "properties": {
                "dep_iata": {
                    "type": "string",
                    "description": "3-letter IATA departure airport code, e.g. LIS",
                },
                "arr_iata": {
                    "type": "string",
                    "description": "3-letter IATA arrival airport code, e.g. JFK",
                },
                "flight_status": {
                    "type": "string",
                    "enum": ["scheduled", "active", "landed", "cancelled", "incident", "diverted"],
                },
            },
            "required": [],
        },
    },
}

_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "fetch_search",
        "description": "Search the web for current, real information.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
}

_TOOL_IMPLS = {"fetch_flights": fetch_flights, "fetch_search": fetch_search}


def _call_agent(*, system: str, user: str, tool: dict, model: str = "gpt-4") -> dict:
    """One tools-enabled completion: ask, let the model decide, run the real
    tool if it picked one. Client is built here, not at module level, so
    importing this module never requires OPENAI_API_KEY to be set."""
    client = openai.OpenAI()
    response = client.chat.completions.create(
        model=model,
        tools=[tool],
        tool_choice="auto",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    message = response.choices[0].message
    calls = []
    for call in message.tool_calls or []:
        fn = _TOOL_IMPLS[call.function.name]
        args = json.loads(call.function.arguments)
        calls.append({"name": call.function.name, "arguments": args, "result": fn(**args)})
    return {"content": message.content, "tool_calls": calls}


def call_flight_agent(user: str) -> dict:
    return _call_agent(
        system="You are the flight-search agent. Use fetch_flights to look up real flights.",
        user=user,
        tool=_FLIGHT_TOOL,
    )


def call_hotel_agent(user: str) -> dict:
    return _call_agent(
        system="You are the hotel-research agent. Use fetch_search to research lodging.",
        user=user,
        tool=_SEARCH_TOOL,
    )


def call_itinerary_agent(user: str) -> dict:
    return _call_agent(
        system="You are the itinerary agent. Use fetch_search to research things to do.",
        user=user,
        tool=_SEARCH_TOOL,
    )


def call_final_response_agent(user: str) -> dict:
    return _call_agent(
        system=(
            "You are the final-response agent. Use fetch_search to confirm any detail "
            "you are unsure of before writing the trip summary."
        ),
        user=user,
        tool=_SEARCH_TOOL,
    )


if __name__ == "__main__":
    demo = [
        ("flight_agent", call_flight_agent, "Find active flights arriving at CDG."),
        ("hotel_agent", call_hotel_agent, "Find good boutique hotels in Lisbon, Portugal."),
        ("itinerary_agent", call_itinerary_agent, "Find top things to do in Lisbon, Portugal."),
        ("final_response_agent", call_final_response_agent, "Summarize a 3-day trip to Lisbon."),
    ]
    for name, fn, prompt in demo:
        print(f"=== {name} ===")
        print(json.dumps(fn(prompt), indent=2, default=str))
        print()
