# travelmate.ai — TripMate AI

A LangGraph multi-agent travel planner. You give it dates, a party size and
some constraints; four agents — flight, hotel, itinerary, and final response —
research and reason with OpenAI, coordinated through a shared `TravelState`,
and hand back a concrete day-by-day itinerary.

## Architecture

```
 free text ──▶ prompt_parser ──┐
                                ▼
                          TripRequest ────────▶ ┌──────────────────────┐
                                                 │ resolve_destination  │  (only runs if no
                                                 └──────────┬───────────┘   destination was given)
                                                            │
                                             ┌──────────────┴──────────────┐
                                             ▼                              ▼
                                    ┌─────────────────┐           ┌─────────────────┐
                                    │  flight_agent    │           │  hotel_agent     │
                                    └────────┬────────┘           └────────┬────────┘
                                     search_flights,                search_lodging
                                     both legs (outbound+return)
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

**Flights and lodging on the final itinerary are real, structured data —
never LLM-retyped text.** `flight_agent` searches *both* legs (there and
back); the itinerary agent is only ever asked for `{days, notes}`
(`DraftItinerary`) and never sees or re-costs the flight/hotel numbers. The
full `Itinerary` — `outbound_flight`, `return_flight`, every `lodging_option`
considered, and a `total_estimated_cost` summed in code from real flight +
hotel + activity costs — is assembled deterministically in
`build_itinerary_node`. The cheapest option of each is what's structurally
booked; each agent's job is to explain that pick (or flag a concern with it),
not to change it — this is stated explicitly in `FLIGHT_AGENT_SYSTEM` and
`HOTEL_AGENT_SYSTEM` so the agent's narrative can't contradict what's actually
selected.

The itinerary agent is asked for JSON and nothing else
(`response_format: json_object`); its response is Pydantic-validated **while
the agent is still in the loop**
([app/agent/itinerary_json.py](app/agent/itinerary_json.py)) — a malformed,
day-count-wrong, or wrong-dates plan is fed back to the model as a correction
request instead of surfacing later as a 500. It gets
`TRAVELMATE_MAX_ITINERARY_RETRIES` attempts (default 2) before the whole run
fails with a `PlanningError`. The exact list of calendar dates the trip needs
(`TripRequest.dates`) is handed to the model up front rather than left for it
to compute from start/end dates — a live bug once had it miscount three
retries running.

Before planning where to go, the itinerary agent first decides whether the
destination is a **country/region** (splits days across 2-4 well-known
cities, in geographic order, naming the city each day) or a **single city**
(stays confined to that city's real neighbourhoods) — and is told to use
real, specific place names from its own knowledge rather than the mock
attraction list's generic ones. Worth knowing: this makes the plan read like
a knowledgeable travel agent's suggestions, not live-verified facts.

Two ways into the graph: `plan_trip(TripRequest)` for structured input (the
CLI, `POST /trips/plan`), or `plan_trip_from_prompt(str)` for one free-text
sentence (the frontend, `POST /trips/plan-from-prompt`) — which parses the
sentence into a `TripRequest` first
([app/agent/prompt_parser.py](app/agent/prompt_parser.py)) and then calls the
same `plan_trip()`. Missing dates resolve to a documented default computed in
code (a bare duration defaults to starting 2 weeks out; nothing at all
defaults to a 5-day trip) — never left for the model to invent.

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
python scripts/plan_trip.py --destination "Kyoto, Japan" --start 2026-04-10 --end 2026-04-15 --travelers 2 --budget 300000 --interests food,history
```

Or run the API — and open `http://localhost:8000` for the frontend (plain
HTML/CSS/JS, no build step, no Node — the same process serves both the page
and the API, so there's no CORS to configure): type a sentence like *"Plan a
5 day Dubai trip from Dhaka with flights, hotels and sightseeing"* and it
plans the whole trip:

```bash
uvicorn app.main:app --reload
```

| Method | Path | What it does |
|---|---|---|
| `GET` | `/` | The frontend — plan a trip from one sentence, see the result rendered as a page |
| `GET` | `/health` | Liveness, plus the configured model, provider and store |
| `POST` | `/trips/plan` | Plan a trip from structured parameters; returns the stored `PlannedTrip` |
| `POST` | `/trips/plan-from-prompt` | Plan a trip from one free-text sentence — what the frontend calls |
| `POST` | `/trips/{thread_id}/revise` | Apply one change to an existing trip; saves and returns the next version |
| `GET` | `/trips/{thread_id}/history` | Every version of one trip, oldest first |
| `GET` | `/trips` | Every trip planned since the process started |
| `GET` | `/trips/{id}` | One trip version |

```bash
curl -X POST localhost:8000/trips/plan -H 'content-type: application/json' -d '{
  "start_date": "2026-04-10", "end_date": "2026-04-14",
  "travelers": 2, "origin": "Delhi",
  "budget": 300000, "interests": ["food", "history"], "pace": "balanced"
}'

curl -X POST localhost:8000/trips/plan-from-prompt -H 'content-type: application/json' -d '{
  "prompt": "Plan a complete 7 day Japan trip from Delhi under 3 lakhs"
}'
```

Leave `destination` out (structured) or unmentioned (free text) and
`resolve_destination` will pick one and say why. A free-text prompt that
leaves out dates gets a default 5-day trip starting two weeks out — see
`app/agent/prompt_parser.py` for exactly what's inferred versus defaulted.

Interactive docs are at `/docs`.

### Revisions and trip history

A planned trip isn't final. Ask for a change against its `thread_id` and the
result is saved as the **next version**, leaving every earlier one exactly as
it was:

```bash
curl -X POST localhost:8000/trips/{thread_id}/revise -H 'content-type: application/json' -d '{
  "change_request": "I want to do more activities on day 3"
}'

curl localhost:8000/trips/{thread_id}/history
```

Every `PlannedTrip` carries `thread_id` (stable across the whole edit
history), `version` (1, 2, 3...) and `change_note` (what was asked for —
`null` on the original). `id` still identifies that one version, so
`GET /trips/{id}` keeps working and old versions stay retrievable forever.

A revision runs **only** the itinerary step (`app/agent/reviser.py`), not the
five-node planning graph: destination, dates, flights and lodging carry over
untouched from the version being revised, and are deliberately withheld from
the model's prompt so a request to reshuffle day 3 can't quietly swap the
hotel or re-price a flight. `REVISE_ITINERARY_SYSTEM` requires the complete
plan back — every day, with untouched days returned unchanged — and the
result goes through the same `parse_itinerary` validation (exact dates, in
order) and the same retry-with-feedback loop as the original. Only the
activity portion of the total cost is recomputed.

Verified end-to-end on a real 4-day Kyoto trip: "more activities on day 3"
took day 3 from 2 to 3 activities, left days 1, 2 and 4 byte-for-byte
identical, and moved the total from ₹2,953.69 to ₹3,003.69. A follow-up
"make day 1 more relaxed" built on *that* version, swapping day 1's walking
loop for a tea house while leaving the newly added day-3 activity in place.

### Layout

| Path | What lives there |
|---|---|
| `app/agent/graph.py` | Wires the five nodes into the LangGraph `StateGraph` |
| `app/agent/nodes.py` | The coordinator step and the four agents |
| `app/agent/state.py` | `TravelState` — the dict LangGraph threads through every node |
| `app/agent/llm.py` | `OpenAILLM` — the one call every node makes, narrowed to `.complete()` so it's fakeable in tests |
| `app/agent/prompts.py` | Each agent's system prompt and the itinerary JSON schema |
| `app/agent/itinerary_json.py` | Parses and Pydantic-validates the itinerary agent's JSON |
| `app/agent/prompt_parser.py` | Turns one free-text sentence into a `TripRequest` — what powers the frontend's single prompt box |
| `app/agent/planner.py` | `plan_trip()` / `plan_trip_from_prompt()` — build the graph, invoke it, map the result to a `PlannedTrip` |
| `app/agent/reviser.py` | `revise_trip()` — applies one requested change to an existing trip, re-running only the itinerary step |
| `app/providers/` | The travel data seam (`TravelProvider` Protocol) |
| `app/models/itinerary.py` | `TripRequest`, `Itinerary`, `PlannedTrip` |
| `app/api/` | FastAPI routes and wire schemas |
| `app/store.py` | `TripStore` Protocol, `InMemoryTripStore`, and `SqlTripStore` (Postgres via Docker, or any SQLAlchemy engine) — append-only version history per `thread_id` |
| `frontend/` | The plain HTML/CSS/JS frontend (dark theme, single prompt box) — served by `app/main.py`, no build step |
| `scripts/run_server.py` | Launches the API from an absolute path — a workaround if something ever runs `uvicorn` from the wrong working directory and silently imports a same-named `app` package from elsewhere |

## Currency

**Everything is priced in Indian rupees.** Not converted at display time —
rupee-native throughout: the mock provider generates rupee figures, the
prompts ask the model for rupee amounts (and read "2 lakhs" / "1.5 crore"
natively when parsing a budget), and no exchange rate exists anywhere in the
codebase.

Money fields are named for what they are rather than for a currency —
`total`, `nightly`, `budget`, `estimated_cost`, `price_per_person` — and
`Itinerary.currency` (default `INR`) says what they're denominated in.

Trips stored before this switch are kept exactly as they were: their amounts
are untouched and their `currency` stays `"USD"`, because relabelling dollars
as rupees would silently multiply every price by ~85. Each model has a
`model_validator(mode="before")` mapping the old `*_usd` field names onto the
new ones, and both the frontend and the CLI pick their symbol from the trip's
own `currency` — so an old trip still renders `$13,690` while a new one
renders `₹2,30,610` (with Indian digit grouping, via `en-IN`). See
`tests/test_legacy_usd_payloads.py`.

## Travel data

`TRAVELMATE_PROVIDER=mock` (the default) serves deterministic, seeded travel
data — plausible fiction, not quotes — so the whole system runs end to end with
no third-party keys and the tests never touch a network.

`TRAVELMATE_PROVIDER=live` serves **live flight fares from Google Flights**,
via SerpApi's hosted MCP server, and mock data for everything else — lodging,
attractions and weather still have no real source. Needs `SERPAPI_API_KEY`.

### Live flight fares (`TRAVELMATE_PROVIDER=live`)

`app/providers/google_flights.py` is an **MCP client** to SerpApi's official
hosted server (`https://mcp.serpapi.com/mcp`, streamable HTTP). It calls the
server's `search` tool with `engine=google_flights` — a one-way, one-adult,
rupee-priced search for each leg on the exact travel date. The key goes in an
`Authorization: Bearer` header rather than the URL path SerpApi also accepts,
because the MCP SDK logs every request URL.

Each result carries price, total duration, every segment's airline and flight
number, and departure time. Place names resolve through
`app/providers/iata.py`, which uses Travelpayouts' free public city and airport
directories (no account needed): "Delhi" → `DEL`; a shared name like Kochi
(Japan's `KCZ` vs India's `COK`) resolves to the match in the same country as
the other end of the route; and a multi-airport city expands to its airports,
since Google Flights wants airport codes (Tokyo `TYO` → `NRT,HND`).

Every option is kept on the itinerary (`outbound_options` / `return_options`).
The frontend shows one table per leg, each with a **Cheapest** and a
**Fastest** row of up to three flights, each showing duration, stops, date and
local departure time next to its price. The flight booked and costed is the
cheapest on the travel date.

Measured: Varanasi→Cochin on 8 Oct returned **9 outbound and 5 return
flights** on the exact dates, cheapest ₹9,131 (Cleartrip showed ~₹8k).

Limits:

- **100 searches/month** on SerpApi's free plan, and every *new* route and date
  spends one search per leg. Searches are **cached** for
  `TRAVELMATE_FLIGHT_CACHE_TTL_HOURS` (default 6) in
  `app/providers/flight_cache.py`, keyed by airports and date: re-planning the
  same trip within that window costs nothing (measured: 2 searches / 7.8s the
  first time, 0 / 1.1s the second). It's the raw per-adult result that's
  cached, so different party sizes share it. It lives in a
  `flight_search_cache` table under `TRAVELMATE_STORE=postgres`, so it survives
  `uvicorn --reload` restarts; failed searches are never cached. Tests never
  touch the real API — they mock the MCP round trip.
- Google occasionally lists a flight with no price; those are skipped.
- If a search finds nothing or fails, the trip plans **without flights** rather
  than falling back to mock ones, which would be indistinguishable from real
  fares on the page.

### Real fetch clients (built, tested, not wired into the pipeline yet)

`app/providers/aviationstack.py` (`fetch_flights` + `normalize_flights`) and
`app/providers/tavily.py` (`fetch_search` + `normalize_search`) are real HTTP
clients over `httpx`, verified against the live APIs during development.
Each fetch function returns the provider's raw JSON as a plain `dict`; the
matching `normalize_*` function turns that into typed Pydantic models
(`Flight`/`FlightEndpoint`, `SearchResult`/`SearchResults`) with the noise
dropped (timezones, codeshare data, favicons, raw HTML) and strings converted
to real `datetime`/`float` values. Normalization is pure — no network call —
so it's unit-tested against a saved sample response
(`tests/test_aviationstack.py`, `tests/test_tavily.py`) without spending API
quota.

They need `AVIATION_API_KEY` / `TAVILY_API_KEY` (loaded via `load_dotenv()` at
import time) and are runnable standalone:

```bash
python -m app.providers.aviationstack
python -m app.providers.tavily
```

**Not implementing `TravelProvider`** — the mock provider is still what
`/trips/plan` actually calls. Wiring one of these in as a `live` data source
is a separate next step.

### MCP client (also standalone, not wired into the pipeline yet)

`app/providers/tavily_mcp.py` (`fetch_search_mcp` + `search_via_mcp`) reaches
the same Tavily search results a second way: over
[MCP](https://modelcontextprotocol.io) instead of Tavily's REST API, using
Tavily's hosted remote server at `https://mcp.tavily.com/mcp/`
(streamable-HTTP transport, authenticated via `?tavilyApiKey=...` on the URL —
no OAuth flow needed). It calls the server's `tavily_search` tool, whose JSON
result matches the REST `/search` response shape exactly, so this module
normalizes through the same `tavily.normalize_search` rather than duplicating
models. The server also exposes `tavily_extract`, `tavily_crawl`,
`tavily_map`, `tavily_research`, and `tavily_feedback` tools — only
`tavily_search` is called so far.

Built with the official `mcp` Python SDK (`ClientSession` +
`streamable_http_client`). One real gotcha found while wiring this up: the SDK
vendors its own HTTP client under a genuinely separate package name,
`httpx2` — not an alias for `httpx` — so `tests/conftest.py`'s network-block
fixture (which patches `httpx.get`/`post`) needed a second patch for
`httpx2.get`/`post`, or a forgetful test here would silently make a real call.
This module's own tests instead mock `_call_tool` directly, matching how
`test_tavily.py` mocks `httpx.post`.

Runnable standalone (needs `TAVILY_API_KEY`, same as the REST client above):

```bash
python -m app.providers.tavily_mcp
```

**Not implementing `TravelProvider`** — same status as the fetch clients
above: built and verified against the live server, not yet a `live` data
source for `/trips/plan`.

### MCP server (our own, wrapping AviationStack)

The client above talks to someone else's MCP server; `app/mcp_server/aviationstack.py`
is the other direction — our own MCP server, exposing two tools that wrap
`app/providers/aviationstack.py`: `search_flights` (real-time flights, now
also filterable by `airline_name`/`airline_iata`) and `future_flight_schedule`
(scheduled routes for one airport by weekday, from `/v1/flightsFuture`). It
adds no HTTP logic of its own; the MCP surface is the only new thing here.

**Only these two are implemented, deliberately.** A reference project's tool
list (screenshot shared 2026-09-18) also had `list_airports`, `list_airlines`,
`list_routes`, `list_taxes`, historical flights-by-date, and several "random
X detailed info" tools (airplanes, aircraft types, cities, countries).
Probing AviationStack's real endpoints live with this project's key returned
`403 function_access_restricted` for every one of those — they're gated
behind a paid plan. `search_flights` and `future_flight_schedule` are the
only two endpoints (`/v1/flights`, `/v1/flightsFuture`) this key can actually
call, so those are the only two built; the rest are candidates for later if
the plan is ever upgraded, not before.

Built with the official `mcp` SDK's `MCPServer` (in `mcp==2.2.0`, this is
what `FastMCP` was renamed to — see the SDK's own migration-guide error
message if you hit the old name). Runnable standalone, needs `AVIATION_API_KEY`:

```bash
python -m app.mcp_server.aviationstack                     # stdio — what Claude
                                                             # Desktop/Code launch a
                                                             # local server with
python -m app.mcp_server.aviationstack --transport streamable-http --port 8001
```

One real gotcha found while wiring this up, confirmed by running the tool
both ways: `MCPServer.call_tool(...)` called **in-process** (no transport —
this project's own tests, or another module composing this server) only
turns a deliberately-raised `ToolError`/`ResourceError` into a result; any
other exception *raises* as `UnexpectedToolError` instead of returning
one. Over a **real transport** (stdio or streamable HTTP), the SDK's
JSON-RPC dispatch layer catches either kind and reports it to the client as
`CallToolResult(is_error=True)` — confirmed against both this server and
Tavily's remote one. Both tools deliberately raise `ToolError` (not a bare
exception) for `AviationStackError` — including the `function_access_restricted`
case above — so in-process callers get a clean result too, not a crash.

A second gotcha, found the same way: AviationStack doesn't always report a
bad key as its usual 200-with-an-`error`-body shape — a live test with a
bogus key came back as a genuine HTTP 401. `fetch_flights` only handled the
200 shape before this; it now also catches `httpx.HTTPStatusError` and
raises `AviationStackError` for that case too, so both failure shapes behave
the same way instead of one of them crashing.

**Not implementing `TravelProvider`** — same status as everything else here:
built and tested, not yet a `live` data source for `/trips/plan`.

### MCP server (our own, wrapping Open-Meteo weather)

`app/mcp_server/weather.py` exposes one tool, `get_weather_forecast(place, days)`,
wrapping `app/providers/weather.py`. Unlike every other integration in this
project, **Open-Meteo needs no API key at all** — it's a free, keyless
service — so there's no new `.env` entry for this one. Two chained real HTTP
calls: geocode the place name to coordinates
(`https://geocoding-api.open-meteo.com/v1/search`), then fetch its daily
forecast (`https://api.open-meteo.com/v1/forecast`). The forecast horizon
caps at 16 days out on the free tier; a longer `days` is clamped, not
rejected.

The raw forecast response is columnar (one array per field, aligned by index
to a shared `daily.time` array) rather than a list of per-day records, and
returns only numeric WMO weather codes with no description —
`normalize_forecast` pivots it into one `DailyForecast` per date and maps
each code to a human-readable condition (`WEATHER_CODES` in the same file).

Runnable standalone, no key needed:

```bash
python -m app.providers.weather                     # one real geocode + fetch
python -m app.mcp_server.weather                     # stdio
python -m app.mcp_server.weather --transport streamable-http --port 8002
```

Same `ToolError`-vs-bare-exception distinction as the AviationStack server
applies here too (see above) — `get_weather_forecast` raises `ToolError` for
`WeatherError` (a place that doesn't geocode, or an API error) so in-process
callers get a clean result instead of a crash.

**Not implementing `TravelProvider`** — same status as everything else here:
built and tested (this one against a genuinely free live API, so there's no
plan-tier gate to hit), not yet wired in as the mocked `get_weather_outlook`'s
real replacement.

### Real OpenAI function-calling demo (also standalone)

`app/main.py` has four functions (`call_flight_agent`, `call_hotel_agent`,
`call_itinerary_agent`, `call_final_response_agent`) demonstrating genuine
LLM-driven tool use: one OpenAI call each, `tools=[...]`, `tool_choice="auto"`
— the model itself decides whether to call `fetch_flights`/`fetch_search`,
bound to the real fetch clients above, not the mock provider. This is
**unrelated to the LangGraph pipeline** — a separate proof that the model can
drive these tools directly, not something `/trips/plan` runs. Every call
costs real OpenAI tokens and, when the model calls its tool, a real
AviationStack/Tavily request too, so it's guarded to only run when this file
is executed directly:

```bash
python -m app.main
```

Importing `app.main` (every `uvicorn` startup, and the test suite) never
triggers it — the OpenAI client is constructed inside the call, not at
module load time.

## Configuration

Every setting is an environment variable prefixed `TRAVELMATE_`, or a line in
`.env`. See `.env.example`.

| Variable | Default | Notes |
|---|---|---|
| `OPENAI_API_KEY` | — | Required (no prefix; read by the OpenAI SDK) |
| `TRAVELMATE_MODEL` | `gpt-4o-mini` | **Must support JSON mode** (`response_format: json_object`) — the itinerary agent and the prompt parser both depend on it. Plain `gpt-4` does not support it and fails every request with a 400; `gpt-4o`/`gpt-4o-mini`/`gpt-4-turbo` do. |
| `TRAVELMATE_TEMPERATURE` | `0.3` | |
| `TRAVELMATE_MAX_TOKENS` | `16000` | Per LLM call. Deliberately generous — OpenAI bills by tokens actually generated, not this ceiling, and a longer or packed-pace itinerary can genuinely need more than a few thousand tokens. Too low and JSON mode's one failure mode (a response cut off mid-generation) starts happening under real use; see Known issues below. |
| `TRAVELMATE_MAX_ITINERARY_RETRIES` | `2` | Extra attempts after a schema-invalid (or day-count-wrong) itinerary |
| `TRAVELMATE_PROVIDER` | `mock` | `mock` or `live` |
| `TRAVELMATE_STORE` | `memory` | `memory` or `postgres` |
| `TRAVELMATE_DATABASE_URL` | `postgresql+psycopg://travelmate:travelmate@localhost:5433/travelmate` | Only read when `TRAVELMATE_STORE=postgres` |

Also read directly by the SDKs/libraries that use them (not `TRAVELMATE_`-prefixed):
`AVIATION_API_KEY`, `TAVILY_API_KEY` (the standalone fetch clients — see below),
`LANGSMITH_TRACING`/`LANGSMITH_API_KEY`/etc. (**keep `LANGSMITH_TRACING=false`** —
see Known issues). None of these three are wired into the running app; they're
picked up automatically by `load_dotenv()` in the files that use them.

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

## Frontend

`frontend/` — plain HTML/CSS/JS, no build step, no Node, dark theme. One
input: a free-text prompt box ("Plan a 5 day Dubai trip from Dhaka with
flights, hotels and sightseeing"), four example chips that fill it
(Japan/Dubai/Thailand/Lisbon trips), and a result view showing flights, the
hotel list with the selected one marked, the day-by-day plan, and the summary
— in that order, summary first. The green/red "Online" badge is a real
`GET /health` check on page load, not decoration; the trip ID shown is the
real `PlannedTrip.id`, not a fabricated "thread" (there's no conversation
tracking to back one).

Served by FastAPI itself — `GET /` returns `index.html`, `/static` is
mounted for `style.css`/`app.js` — so the page and the API it calls are
same-origin and there's no CORS to configure.

## Known issues, and what fixed them

Found by actually running the app, not by inspection — kept here because
each one is easy to reintroduce by accident:

- **`gpt-4` fails every itinerary request with a 400.** JSON mode
  (`response_format: json_object`), which the itinerary agent and the prompt
  parser both require, isn't supported by that model. `TRAVELMATE_MODEL` must
  be one that supports it (`gpt-4o-mini` default, `gpt-4o`, `gpt-4-turbo`, …).
- **A long or packed-pace itinerary can come back with invalid JSON.** JSON
  mode guarantees valid JSON *except* when the response is cut off by hitting
  `TRAVELMATE_MAX_TOKENS` mid-generation — the model's output length varies
  per call, so this is probabilistic, not deterministic (some requests at a
  tight budget succeed, others don't). Keep the budget generous; it's free
  unless actually used.
- **The itinerary agent can miscount days** if left to derive the date range
  from start/end dates itself — seen live: a 5-day request came back with 4
  `days` entries, three attempts running. Fixed at the root:
  `TripRequest.dates` is the single source of truth for which calendar dates
  need an entry; the prompt hands that list over directly, and
  `parse_itinerary`'s validation checks the actual dates match
  position-by-position, not just the count.
- **An LLM can return explicit `null` for a field you told it to default.**
  Pydantic only applies a field's default when the key is *absent*, not when
  it's present as `null` — `ParsedPrompt` (`app/agent/prompt_parser.py`) has a
  `model_validator(mode="before")` that treats `null` the same as omitted for
  its non-`Optional` fields (`travelers`, `interests`, `pace`).
  Worth checking for the same gotcha in any future LLM-filled Pydantic model.
- **An agent's free-text narrative can contradict the structural pick it's
  describing.** `final_response_agent` once summarized "staying at Hotel
  Meridiana" while the itinerary's actual selection (cheapest-first) was
  "Casa Vista Hostel" — two different hotels in one output. Fixed by having
  it read only the itinerary's own JSON, never the agents' recommendation
  text, which can name a different option than what's actually booked.
- **Static files can go stale with zero revalidation.** Starlette's
  `StaticFiles` sends `Last-Modified`/`ETag` but no `Cache-Control`, so a
  browser can serve an old `index.html`/`app.js`/`style.css` after an edit via
  heuristic freshness — not even a 304 check happens. `app/main.py` sets
  `Cache-Control: no-cache` on `/` and `/static/*` (still allows a cheap 304
  via the existing ETag). If a change doesn't seem to be showing up in a
  browser, hard-refresh or open a different origin (`127.0.0.1` vs
  `localhost`) before assuming the code is wrong.
- **`LANGSMITH_TRACING=true` makes LangGraph try to phone home for real.**
  Nothing in this codebase uses LangSmith; a `.env` with tracing on and no
  valid/scoped key produces a `403 Forbidden` warning on every graph run.
  Keep it `false`.
- **`TRAVELMATE_STORE=postgres` with the container not running hangs the app
  at startup**, not with an error — `get_store()` in the FastAPI lifespan
  tries to actually connect. If `uvicorn` sits at "Waiting for application
  startup" forever, check `docker compose ps` before anything else.
- **Windows' legacy console codepage (cp1252) cannot encode `→`/`★`** — a real
  `UnicodeEncodeError` crash when printed, not just visual garbling.
  `scripts/plan_trip.py` forces `sys.stdout.reconfigure(encoding="utf-8")` and
  avoids non-ASCII decorative characters in its output. Hit again live via
  `app.providers.tavily_mcp`'s demo — this time from an emoji inside real
  search-result text, not a decorative character we chose — fixed the same
  way.
- **A vendored SDK dependency can bypass a network-blocking test fixture that
  looks complete.** The `mcp` Python SDK ships its own HTTP client under a
  genuinely separate package name, `httpx2` — not an alias for `httpx` — so
  the existing fixture patching `httpx.get`/`post` silently did not cover
  `app/providers/tavily_mcp.py`'s calls. Worth checking what HTTP library a
  new dependency actually uses before assuming an existing network-block
  fixture covers it.
- **Don't ask an agent to predict a choice that code makes.** The flight agent
  was told "the cheapest option is booked" and asked to explain it; once
  booking started preferring the travel date, it kept naming the cheaper
  flight a day early as "chosen" — even after its prompt was reworded to
  describe the new rule. Fixed by computing the booked flight *before* the LLM
  call (`_booked_option` in `app/agent/nodes.py`) and handing it over as
  `booked_outbound` / `booked_return`. Same shape as the earlier hotel-summary
  bug: an agent's text should describe decisions, never anticipate them.
- **A fare API can ignore your dates without telling you.** Travelpayouts'
  `/v2/prices/latest` returns the cheapest fares found across a *year*: a trip
  for 10–13 October got an outbound on 5 October and a return on 29 September.
  Mock data hid it, because it stamps whatever date you pass in. (Travelpayouts
  was later dropped entirely: even its exact-date endpoint had too few cached
  fares — two for Varanasi→Cochin in all of October. A cached-fare API can't
  stand in for a live search on quieter routes.)
- **SQLAlchemy's `create_all()` creates missing *tables*, never missing
  *columns*.** Adding `thread_id`/`version` to the `trips` table meant a
  database created before versioning existed would keep its old three columns
  and fail on every insert — `create_all()` sees the table exists and does
  nothing. `SqlTripStore._add_versioning_columns_if_missing()` inspects the
  live table and `ALTER`s in what's missing (plus the `thread_id` index, which
  `create_all` also skips on an existing table), backfilling each old row as
  version 1 of its own thread. Verified against the real Postgres with 7
  pre-existing trips, then re-run from scratch to confirm it's idempotent.
  Worth remembering for any future column on that table — this project has no
  Alembic.
- **A field the LLM already fills in can go completely unseen.**
  `Activity.description` ("what and why") was being generated by the
  itinerary agent on every request, but neither the frontend
  (`renderActivity` in `frontend/app.js`) nor the CLI (`render` in
  `scripts/plan_trip.py`) ever displayed it — days rendered as bare lists of
  activity titles with no depth, even though the richer text existed in the
  response the whole time. Fixed by rendering it in both places, and
  strengthened `ITINERARY_AGENT_SYSTEM` to require 2-3 concrete sentences per
  activity instead of accepting a token description. Worth checking, for any
  model field that looks unused, whether it's actually being silently dropped
  by a renderer rather than never populated.

## Status

The agent graph, nodes, validation, Postgres persistence, the frontend, the
API, and tests are all real and working. Flight fares are real under
`TRAVELMATE_PROVIDER=live` (Google Flights via SerpApi's MCP server); lodging, attractions and weather
are still mock data. The AviationStack, Tavily and Open-Meteo clients exist and
are tested but aren't used by the planner.

See [docs/ROADMAP.md](docs/ROADMAP.md) for the original diagram this was built
from and what's implemented vs. deferred, and
[docs/BUILD_LOG.md](docs/BUILD_LOG.md) for the step-by-step history of how it
got here — every change, the decisions behind them, and the bugs found along
the way.
