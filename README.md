# travelmate.ai

An agentic trip planner. You give it dates, a party size and some constraints;
Claude researches with travel tools and commits to a concrete day-by-day
itinerary.

The agent is not asked to write a plan in prose. It researches, then calls
`submit_itinerary` with a structured plan that is validated by Pydantic **while
the agent is still in the loop** — so a malformed plan comes back to the model
as a correctable error rather than surfacing later as a 500.

## Quick start

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows;  source .venv/bin/activate elsewhere
pip install -e ".[dev]"
cp .env.example .env            # then put your ANTHROPIC_API_KEY in it
```

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
| `GET` | `/health` | Liveness, plus the configured model and provider |
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

Leave `destination` out and the agent will pick one and justify it.

Interactive docs are at `/docs`.

## How it works

```
TripRequest
    │
    ▼
app/agent/planner.py ── Claude (tool_runner, adaptive thinking)
    │                       │
    │                       ├── search_destinations   ┐
    │                       ├── get_weather_outlook   │  read-only lookups,
    │                       ├── search_flights        ├─ bound to this request
    │                       ├── search_lodging        │  (app/agent/tools/)
    │                       ├── search_attractions    ┘
    │                       │
    │                       └── submit_itinerary ──▶ validated Itinerary
    ▼
PlannedTrip  →  app/store.py
```

The research tools are **built per request**, closing over the provider and the
trip's own dates. The model never restates context it already gave us, and it
cannot reach a data source we did not hand it.

The planner owns the stopping conditions: an iteration cap, a single retry if
the agent finishes without submitting, and a hard `PlanningError` if it still
has not. The message history is mirrored as the loop runs, which is what makes
that retry a continuation rather than a fresh start.

### Layout

| Path | What lives there |
|---|---|
| `app/agent/planner.py` | The agentic loop and its stopping conditions |
| `app/agent/prompts.py` | System prompt and the per-request briefing |
| `app/agent/tools/` | Research tools and the terminal `submit_itinerary` |
| `app/providers/` | The travel data seam (`TravelProvider` Protocol) |
| `app/models/itinerary.py` | `TripRequest`, `Itinerary` and friends |
| `app/api/` | FastAPI routes and wire schemas |
| `app/store.py` | Trip persistence (in-memory today) |

## Travel data

`TRAVELMATE_PROVIDER=mock` (the default) serves deterministic, seeded travel
data — plausible fiction, not quotes — so the whole system runs end to end with
no third-party keys and the tests never touch a network. Real integrations
implement the `TravelProvider` Protocol in `app/providers/base.py` and get wired
into `get_provider()`; `provider="live"` raises until one exists, so it can
never quietly serve invented prices as real ones.

## Configuration

Every setting is an environment variable prefixed `TRAVELMATE_`, or a line in
`.env`. See `.env.example`.

| Variable | Default | Notes |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Required (no prefix; read by the SDK) |
| `TRAVELMATE_MODEL` | `claude-opus-5` | |
| `TRAVELMATE_EFFORT` | `high` | `low` … `max`; trades cost against thoroughness |
| `TRAVELMATE_MAX_TOKENS` | `16000` | Per assistant turn |
| `TRAVELMATE_MAX_TOOL_ITERATIONS` | `24` | Loop safety rail |
| `TRAVELMATE_PROVIDER` | `mock` | `mock` or `live` |

## Tests

```bash
pytest
ruff check .
```

The suite runs offline. `tests/conftest.py` has a fake tool runner that replays
a scripted conversation while calling the *real* tools the planner built, so
the loop, the tools and the validation are all exercised without an API key.

## Docker

```bash
docker compose up --build
```

## Status

Early. The agent loop, tools, validation, API and tests are real; the travel
data is mocked and trips live in memory. Next up: a live provider behind the
Protocol, persistence, and streaming the plan as it is built.
