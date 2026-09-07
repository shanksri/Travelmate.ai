# travelmate.ai — TripMate AI

A LangGraph multi-agent travel planner. You give it dates, a party size and
some constraints; four agents — flight, hotel, itinerary, and final response —
research and reason with OpenAI, coordinated through a shared `TravelState`,
and hand back a concrete day-by-day itinerary.

## Architecture

```
                         ┌──────────────────────┐
   TripRequest ────────▶ │ resolve_destination  │  (only runs if no
                         └──────────┬───────────┘   destination was given)
                                    │
                     ┌──────────────┴──────────────┐
                     ▼                              ▼
            ┌─────────────────┐           ┌─────────────────┐
            │  flight_agent    │           │  hotel_agent     │
            │  search_flights  │           │  search_lodging  │
            └────────┬────────┘           └────────┬────────┘
                     └──────────────┬──────────────┘
                                    ▼
                         ┌──────────────────────┐
                         │   itinerary_agent     │  search_attractions,
                         │  (JSON-mode, retries) │  get_weather_outlook
                         └──────────┬───────────┘
                                    ▼
                         ┌──────────────────────┐
                         │ final_response_agent  │
                         └──────────┬───────────┘
                                    ▼
                              PlannedTrip
```

Flight and hotel search run in parallel — neither depends on the other — then
both feed the itinerary agent, which needs both before it can plan a single
day. Every node is a plain `state -> dict` function (see
[app/agent/nodes.py](app/agent/nodes.py)); LangGraph merges each node's return
value into the shared `TravelState` between steps
([app/agent/state.py](app/agent/state.py)).

The itinerary agent is asked for JSON and nothing else
(`response_format: json_object`); its response is Pydantic-validated **while
the agent is still in the loop**
([app/agent/itinerary_json.py](app/agent/itinerary_json.py)) — a malformed or
day-count-mismatched plan is fed back to the model as a correction request
instead of surfacing later as a 500. It gets `TRAVELMATE_MAX_ITINERARY_RETRIES`
attempts (default 2) before the whole run fails with a `PlanningError`.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"
cp .env.example .env            # then put your OPENAI_API_KEY in it
```

> **Windows + PowerShell:** if activation is blocked with
> `running scripts is disabled on this system`, either skip activation and
> call `.venv\Scripts\python.exe` directly, or run
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned` once
> to allow local scripts (Microsoft's own recommended default).

Plan a trip from the command line:

```bash
python scripts/plan_trip.py --destination "Kyoto, Japan" --start 2026-04-10 --end 2026-04-15 --travelers 2 --budget 4000 --interests food,history
```

Or run the API:

```bash
uvicorn app.main:app --reload
```

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | Liveness, plus the configured model, provider and store |
| `POST` | `/trips/plan` | Plan a trip; returns the stored `PlannedTrip` |
| `GET` | `/trips` | Every trip planned since the process started |
| `GET` | `/trips/{id}` | One trip |

```bash
curl -X POST localhost:8000/trips/plan -H 'content-type: application/json' -d '{
  "start_date": "2026-04-10", "end_date": "2026-04-14",
  "travelers": 2, "origin": "Boston, MA",
  "budget_usd": 3500, "interests": ["food", "history"], "pace": "balanced"
}'
```

Leave `destination` out and `resolve_destination` will pick one and say why.

Interactive docs are at `/docs`.

### Layout

| Path | What lives there |
|---|---|
| `app/agent/graph.py` | Wires the five nodes into the LangGraph `StateGraph` |
| `app/agent/nodes.py` | The coordinator step and the four agents |
| `app/agent/state.py` | `TravelState` — the dict LangGraph threads through every node |
| `app/agent/llm.py` | `OpenAILLM` — the one call every node makes, narrowed to `.complete()` so it's fakeable in tests |
| `app/agent/prompts.py` | Each agent's system prompt and the itinerary JSON schema |
| `app/agent/itinerary_json.py` | Parses and Pydantic-validates the itinerary agent's JSON |
| `app/agent/planner.py` | `plan_trip()` — builds the graph, invokes it, maps the result to a `PlannedTrip` |
| `app/providers/` | The travel data seam (`TravelProvider` Protocol) |
| `app/models/itinerary.py` | `TripRequest`, `Itinerary`, `PlannedTrip` |
| `app/api/` | FastAPI routes and wire schemas |
| `app/store.py` | `TripStore` Protocol, `InMemoryTripStore`, and `SqlTripStore` (Postgres via Docker, or any SQLAlchemy engine) |

## Travel data

`TRAVELMATE_PROVIDER=mock` (the default) serves deterministic, seeded travel
data — plausible fiction, not quotes — standing in for AviationStack, Google
Places/Maps, and Tavily Search alike, so the whole system runs end to end with
no third-party keys and the tests never touch a network. Real integrations
implement the `TravelProvider` Protocol in `app/providers/base.py` and get
wired into `get_provider()`; `provider="live"` raises until one exists, so it
can never quietly serve invented prices as real ones.

## Configuration

Every setting is an environment variable prefixed `TRAVELMATE_`, or a line in
`.env`. See `.env.example`.

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | Required (no prefix; read by the OpenAI SDK) |
| `TRAVELMATE_MODEL` | `gpt-4` | Any OpenAI chat model |
| `TRAVELMATE_TEMPERATURE` | `0.3` | |
| `TRAVELMATE_MAX_TOKENS` | `4096` | Per LLM call |
| `TRAVELMATE_MAX_ITINERARY_RETRIES` | `2` | Extra attempts after a schema-invalid itinerary |
| `TRAVELMATE_PROVIDER` | `mock` | `mock` or `live` |
| `TRAVELMATE_STORE` | `memory` | `memory` or `postgres` |
| `TRAVELMATE_DATABASE_URL` | `postgresql+psycopg://travelmate:travelmate@localhost:5433/travelmate` | Only read when `TRAVELMATE_STORE=postgres` |

## Tests

```bash
pytest
ruff check .
```

The suite runs offline. `tests/conftest.py` has a `FakeLLM` that returns
scripted responses — either in call order, or routed by a substring of the
system prompt so a single fake can drive flight and hotel search running in
parallel without caring which one calls first. It calls the *real* nodes and
the *real* mock provider, so the graph wiring, the agents, and the itinerary
validation are all exercised without an API key.

`tests/test_store.py` exercises `SqlTripStore`'s actual SQL against an
in-memory SQLite engine — no Docker needed to test the query logic itself;
`docker-compose.yml` is what points that same class at a real Postgres.
`conftest.py` also forces `TRAVELMATE_STORE=memory` before any test can read a
developer's local `.env`, so a machine configured for Postgres never makes the
suite try to open a real database connection.

## Docker

```bash
docker compose up --build
```

Brings up two services: `db` (Postgres 16, host port **5433** — not 5432, in
case something else on your machine already uses it) and `api` (built from
the `Dockerfile`, `TRAVELMATE_STORE=postgres` inside the container talking to
`db:5432` on Docker's internal network). Requires `OPENAI_API_KEY` in your
shell or `.env`.

To run the API on your host instead while still using a dockerized database:
`docker compose up db` (or add `-d` to run it in the background), then set
`TRAVELMATE_STORE=postgres` and use `localhost:5433` in
`TRAVELMATE_DATABASE_URL` — that's what `.env.example` is set up for.

## Status

Early. The agent graph, nodes, validation, Postgres persistence, API and tests
are real; travel data is still mocked. Next up: real providers behind the
`TravelProvider` Protocol (AviationStack, Google Places/Maps, Tavily).

See [docs/ROADMAP.md](docs/ROADMAP.md) for the original diagram this was built
from and what's implemented vs. deferred.
