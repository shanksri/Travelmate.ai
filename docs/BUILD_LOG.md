# Build log

Every step of this project so far, in order, with what changed and why.
[docs/ROADMAP.md](ROADMAP.md) says what was *planned*; this says what was
actually built, including the detours and the things that turned out to be
wrong. New steps get appended as features land — see
[Adding to this log](#adding-to-this-log) at the bottom.

Commit hashes are the authoritative record; `git show <hash>` has the full
reasoning for any step.

---

## Phase 1 — Foundation

### Step 1 · Scaffold the project (`16a389f`, 2026-09-07)

First working version: a FastAPI app where **Claude** researched with travel
tools in a loop, then committed to a plan by calling a `submit_itinerary`
tool.

- Pydantic validated the plan **while the agent was still in the loop**, so a
  malformed itinerary came back as a correctable tool error rather than a 500
  later. *This pattern survived every rewrite since.*
- Travel data sat behind a `TravelProvider` Protocol from day one, with a
  seeded deterministic provider — so the system ran end-to-end with no
  third-party keys and tests never touched a network.
- `provider="live"` raised rather than quietly serving invented prices as real
  ones.

### Step 2 · Rebuild as a LangGraph multi-agent pipeline (`c9f0b4a`, 2026-09-07)

The single-loop agent was replaced with the shape from the roadmap diagram:

```
resolve_destination → [flight_agent ‖ hotel_agent] → itinerary_agent → final_response_agent
```

- LangGraph `StateGraph` with a shared `TravelState`; flight and hotel search
  run in parallel since neither depends on the other.
- **Deliberate simplification**: each node calls its provider function
  directly, then asks the LLM to reason over the results — rather than a full
  per-node tool-calling loop. Documented as a choice, not an oversight.
- LLM was **Groq / Llama 3** (`llama-3.3-70b-versatile`) at this point.
- Kept from the scaffold: domain models, the mock provider, the in-memory
  store, the FastAPI layer. `PlannedTrip.tool_calls` became `agent_trace`,
  since it now logged agent steps rather than tool names.

---

## Phase 2 — Real infrastructure

### Step 3 · Switch to OpenAI, add Postgres (`f56f928`, 2026-09-07)

Two changes at the user's request, in one commit:

- **Groq → OpenAI.** The `LLM` Protocol meant every node and every test needed
  zero changes — only the three places that *construct or catch* the concrete
  client moved.
- **`SqlTripStore`** behind the same `TripStore` Protocol as the in-memory
  default. Each trip is one JSON row, not a normalized schema, so there's
  nothing to migrate when `PlannedTrip`'s shape changes.
- `docker-compose.yml` runs Postgres 16 on **host port 5433** — 5432 was
  already taken by another local project's container.
- Default stayed `TRAVELMATE_STORE=memory`: no infra needed, tests stay
  offline.

**Verified for real**, not just against SQLite unit tests: brought up the `db`
container, ran `SqlTripStore` against it, then ran the `api` container against
that same `db` over Docker's network and confirmed `/health` reported
`store=postgres`.

**Bug this surfaced**: a developer's local `.env` leaked into the test session
and hung pytest trying to reach a database that wasn't running. Fixed by
forcing `TRAVELMATE_STORE=memory` / `TRAVELMATE_PROVIDER=mock` via
`os.environ` in `tests/conftest.py` *before* any app module can cache a
`Settings` instance.

> **Security note** (`28b00a7`): real API keys were briefly pasted into
> `.env.example` (a tracked file) instead of `.env`. Moved to `.env`
> (gitignored) and `.env.example` restored to placeholders. **No secrets were
> ever committed.** `.env.example` stays placeholder-only.

---

## Phase 3 — Real external data

### Step 4 · Fetch clients for AviationStack and Tavily (`b5d6702`, 2026-09-07)

Real `httpx` clients — `fetch_flights` and `fetch_search` — reading
`AVIATION_API_KEY` / `TAVILY_API_KEY`.

**Verified against the real APIs, not written from docs**: both run standalone
via `python -m app.providers.<name>` and returned live data.

Deliberately **fetch-only**: neither implements `TravelProvider`, and
`TRAVELMATE_PROVIDER` stayed `mock`. (Still true today.)

### Step 5 · Normalize the fetched data (`70c2e94`, 2026-09-07)

`normalize_flights` / `normalize_search` turn each API's deeply-nested raw
response into small Pydantic models holding only what a planner would use —
flattened, real `datetime`s instead of ISO strings, noise dropped (terminal,
gate, codeshare, aircraft registration; favicon, raw_content, images).

**Kept separate from the fetch functions on purpose**: normalization is pure
and needs no network, so it's unit-testable against a saved sample response
without spending API quota — which matters on AviationStack's free tier.

### Step 6 · Test both clients (`83a645a`, 2026-09-07)

Fetch tests mock `httpx` entirely (missing key, success, API-error payload,
HTTP error). Normalize tests run against a saved sample shaped exactly like a
real captured response.

### Step 7 · Real OpenAI function-calling demo (`8445a4f`, 2026-09-08)

Four functions in `app/main.py` (`call_flight_agent` etc.) demonstrating
genuine LLM-driven tool use — `tools=[...]`, `tool_choice="auto"`, bound to
the real fetch clients.

**Unrelated to the LangGraph pipeline** — a standalone proof, guarded to only
run under `python -m app.main` so importing the app never fires it.

---

## Phase 4 — Making the output actually useful

### Step 8 · Fix the model default: `gpt-4` can't do JSON mode (`34e66cd`, 2026-09-08)

**Found by actually running a trip.** `itinerary_agent` needs
`response_format={"type": "json_object"}`, but plain `gpt-4` rejects that
parameter with a 400 — the pipeline died at that exact step for *every*
request. Default changed to `gpt-4o-mini`.

Also fixed a stale `database_url` default still pointing at port 5432.

### Step 9 · Real flights and hotels on the itinerary (`ea15b2c`, 2026-09-08)

**The biggest single restructure.** Previously `flight_results` /
`hotel_results` only influenced the day plan, then were *discarded* — the
final `Itinerary` had no field for them, so `/trips/plan` could never tell you
what flight to take or where to stay.

- Three new models: `FlightLeg`, `LodgingOption`, `DraftItinerary` (just
  `{days, notes}` — all the LLM is now asked for).
- **Key design choice**: flights and lodging are built from real search
  results, never retyped by the LLM. The itinerary agent takes the cheapest of
  each and assembles the `Itinerary` **programmatically**, so
  `total_estimated_cost` is a real sum computed in code, not an LLM
  arithmetic guess.
- `flight_agent` now searches **both legs** — previously only outbound was
  ever searched, so a return flight was impossible even in principle.

**Bug found live**: `final_response_agent` narrated from `hotel_agent`'s
free-text recommendation ("Hotel Meridiana") while the itinerary's actual
structural pick was the cheapest option ("Casa Vista Hostel") — the summary
named a hotel that wasn't the one shown as selected. Fixed by having it read
**only the itinerary's own JSON**, and both agent prompts now say explicitly
that the cheapest option wins regardless of what they recommend.

**Two more live bugs**: a `UnicodeEncodeError` crash (`→`/`★` aren't in
Windows' legacy console codepage) and a LangSmith leak once `plan_trip.py`
started calling `load_dotenv()`.

---

## Phase 5 — The frontend

### Step 10 · Plain HTML/CSS/JS frontend (`c3f12a0`, 2026-09-08)

A form POSTing to `/trips/plan`, rendering flights, the hotel list with the
selection marked, the day-by-day plan, notes and summary.

**No build step, no Node, no framework** — the user's explicit choice over
React+Vite. Served by FastAPI itself (`GET /` + `/static` mount), so
same-origin with no CORS.

All LLM/provider-sourced text is escaped before `innerHTML` — it comes from an
LLM and mock data, not something to trust into the DOM.

Also added `scripts/run_server.py`, after a dev-server launcher's cached
working directory went stale mid-session and `uvicorn app.main:app` silently
imported *an unrelated project's* same-named `app.main` on port 8000, with no
error.

### Step 11 · Raise `TRAVELMATE_MAX_TOKENS` (`324f55a`, 2026-09-15)

Reported failure: `"gave up after 3 attempt(s): ... not valid JSON: Expecting
value: line 531 column 22"`. JSON mode guarantees valid output **except** when
cut off by `max_tokens` mid-generation — the only way it can produce invalid
JSON at all. Raised 4096 → 16000 (under gpt-4o-mini's 16384 ceiling, and free
unless actually used).

Probabilistic, not deterministic — which is why it hadn't shown up earlier.

### Step 12 · Country-vs-city awareness, summary first (`17f1aa9`, 2026-09-15)

Two real complaints:

1. A bare `"Japan"` produced a generic single-city-flavoured plan, because the
   agent had no instruction to notice the difference between a country and a
   city. Now it decides **country/region vs. single city first**: a country
   splits days across 2-4 well-known cities in geographic order with lighter
   transition days; a city stays in its real neighbourhoods. Either way it
   uses real place names from its own knowledge.
2. The summary rendered *after* the whole day-by-day plan. Moved to the top in
   both the CLI and the frontend.

**Honest caveat recorded at the time**: this makes plans read like a
knowledgeable travel agent's suggestions, not live-verified facts.

### Step 13 · Dark theme + free-text planning (`cfde3e2`, 2026-09-15)

- Dark theme matching a reference screenshot the user shared.
- The structured form became a **single free-text prompt box**, backed by a
  genuinely new capability: `app/agent/prompt_parser.py` parses one sentence
  into a `TripRequest` via an LLM call, with the same retry-on-invalid shape
  as the itinerary agent. New endpoint `POST /trips/plan-from-prompt`.
- Missing dates resolve to defaults **computed in code**, not invented by the
  model.

**Two bugs found live**: an LLM returning explicit `null` for a field with a
documented default (Pydantic only applies defaults to *absent* keys — fixed
with a `model_validator(mode="before")`), and Starlette's `StaticFiles`
sending no `Cache-Control`, so browsers served stale JS with no revalidation
at all.

### Step 14 · Documentation pass (`7aa0198`, 2026-09-15)

Fixed stale config defaults in the README (`gpt-4`/`4096` were still
documented), added the missing sections, and started the "Known issues, and
what fixed them" section.

---

## Phase 6 — MCP (and one visibility fix)

### Step 15 · MCP client for Tavily's remote server (`0a1b055`, 2026-09-17)

`app/providers/tavily_mcp.py` reaches Tavily a second way — over MCP
(`https://mcp.tavily.com/mcp/`, streamable HTTP, `?tavilyApiKey=` auth)
instead of REST. The tool's JSON matches the REST shape exactly, so it reuses
`normalize_search` unchanged.

**Gotcha found**: the `mcp` SDK vendors its own HTTP client under a genuinely
separate package name, **`httpx2`** — not an alias for `httpx` — so the test
suite's network-blocking fixture didn't cover it and a forgetful test would
have made real calls. Fixture extended.

### Step 16 · Show the activity descriptions (`146ef8e`, 2026-09-18)

**A field the LLM was already filling in was completely invisible.**
`Activity.description` ("what and why") was generated on every request, but
neither the frontend nor the CLI ever rendered it — days looked like bare
lists of titles while the richer text sat in the response the whole time.

Both renderers now show it, and `ITINERARY_AGENT_SYSTEM` was strengthened to
require 2-3 concrete sentences per activity, gate food/drink activities on
stated interests, and keep `notes` to logistics only.

### Step 17 · Custom MCP server for AviationStack (`b5266ac`, 2026-09-18)

The other direction: **our own** MCP server exposing `search_flights`, built
on the SDK's `MCPServer` (the `mcp` 2.x rename of `FastMCP`). Runs over stdio
or streamable HTTP.

**Two gotchas, both found by testing over both transports:**

1. `MCPServer.call_tool()` **in-process** only converts a deliberately-raised
   `ToolError`/`ResourceError` into a result — anything else *raises* as
   `UnexpectedToolError`. Over a **real transport**, the JSON-RPC layer
   catches either kind and reports `is_error=True`. The tool now raises
   `ToolError` explicitly.
2. AviationStack doesn't always report a bad key as its usual
   200-with-error-body — a bogus key came back as a genuine **HTTP 401**,
   which `fetch_flights` didn't handle and let crash.

### Step 18 · Expand the AviationStack server (`d261552`, 2026-09-18)

Added `future_flight_schedule` (`/v1/flightsFuture`) and `airline_name` /
`airline_iata` filters on `search_flights`.

**Scope was set by probing the live API first.** A reference project's tool
list included `list_airports`, `list_airlines`, `list_routes`, `list_taxes`,
historical flights and several "random X" lookups — **every one of those
endpoints returned `403 function_access_restricted`** on this project's free
plan. Only `/v1/flights` and `/v1/flightsFuture` are callable, so only those
two were built.

### Step 19 · Weather client + MCP server (`2ed347a`, 2026-09-18)

`app/providers/weather.py` + `app/mcp_server/weather.py`, on **Open-Meteo —
free and keyless**, the only integration here with no API key at all.

Two chained calls: geocode a place name → fetch its daily forecast. The raw
response is **columnar** (one array per field, aligned by index to a shared
date array) with bare numeric WMO codes, so `normalize_forecast` pivots it
into one record per day and maps codes to readable conditions.

---

## Phase 7 — Memory and revisions

### Step 20 · Versioned history + trip revisions (`afb7681`, 2026-09-19)

Trips became an **append-only history** rather than a latest-only snapshot.

- `PlannedTrip` gained `thread_id` (stable across a trip's whole edit
  history), `version` (1, 2, 3...) and `change_note` (what was asked for;
  `null` on the original). `id` still identifies one version, so
  `GET /trips/{id}` is unchanged and old versions stay retrievable.
- `TripStore` gained `list_versions()` and `get_latest()`; `SqlTripStore`
  denormalizes `thread_id`/`version` into real columns so those are SQL
  queries rather than deserializing every row.
- `app/agent/reviser.py` applies one requested change ("more activities on
  day 3") by re-running **only the itinerary step** — not the five-node graph.
  Destination, dates, flights and lodging carry over and are **deliberately
  withheld from the model's prompt**, so a request about day 3 can't quietly
  swap the hotel.
- New endpoints: `POST /trips/{thread_id}/revise`,
  `GET /trips/{thread_id}/history`.

**Design decision**: chosen over a LangGraph **checkpointer**. This needs
versioned snapshots of a *finished* itinerary, not mid-graph resumability, and
it reuses the persistence and validation already here.

**Gotcha**: SQLAlchemy's `create_all()` creates missing *tables* but never
missing *columns* (and skips indexes on an existing table), so the live
Postgres DB with 7 trips would have failed every insert. `SqlTripStore` now
inspects the live table and `ALTER`s in what's missing, backfilling old rows
as version 1 of their own thread. **This project has no Alembic** — copy that
pattern for any future column.

**Verified live** on a 4-day Kyoto trip: "more activities on day 3" took day 3
from 2 → 3 activities, left days 1/2/4 byte-for-byte identical, and moved the
total correctly. A follow-up "make day 1 more relaxed" built on *that*
version, keeping the new day-3 activity.

**Measured afterwards**: a revision is *slower* than a fresh plan (~29.5s vs
~16.8s over two samples each), because both regenerate the full itinerary JSON
and the revision also reads the existing plan as input. The two "extra" calls
in the plan path run in parallel and cost almost nothing. A day-level patch
(returning only changed days) is the fix if this ever matters — not yet built.

---

## Phase 8 — Currency

### Step 21 · Rupee-native pricing (`eb7d5a6`, 2026-09-19)

**Everything is priced in Indian rupees, natively** — not converted at display
time. The mock provider generates rupee figures, the prompts ask for rupee
amounts and read "2 lakhs" / "1.5 crore" when parsing a budget, and **no
exchange rate exists anywhere**.

- Money fields renamed to be currency-neutral: `total`, `nightly`, `budget`,
  `estimated_cost`, `price_per_person`. `Itinerary.currency` (default `INR`)
  says what they're denominated in.
- **The 16 already-stored trips keep their amounts and stay `"USD"`** —
  relabelling dollars as rupees would have silently multiplied every price by
  ~85. Each model has a `model_validator(mode="before")` mapping old `*_usd`
  names onto the new ones, filling only a key that isn't already present.
- Frontend and CLI take their symbol from the trip's own `currency`, and
  rupees use `en-IN` grouping — `₹2,30,610`, not `₹230,610`.

**Verified live**: all 16 legacy trips still load with amounts intact; a fresh
"4 day Jaipur trip from Delhi, budget 2 lakhs" parsed the budget as `200000`;
the rendered page contains 20 rupee symbols and zero dollar signs.

---

## Phase 9 — Real flight fares

### Step 22 · Remove the Notes card and per-activity prices (`876b2ee`, 2026-09-24)

Both removed from the frontend and the CLI, at the user's request. Neither had
been asked for in the first place — they were additions of mine, and the
feedback was direct: don't add things that weren't asked for.

### Step 23 · Real fares, cheapest vs. fastest per leg (`876b2ee`, 2026-09-24)

`TRAVELMATE_PROVIDER=live` now serves **real flight fares**, from
**Travelpayouts** (Aviasales) rather than AviationStack — AviationStack has no
fare data at all, only schedules and status. Lodging, attractions and weather
stay mock (`LiveTravelProvider` subclasses the mock provider and overrides
exactly one method).

**Choosing the API** took some digging, since the landscape shifted this year:
Amadeus shut its free self-service tier on 17 July 2026, and Kiwi's Tequila
went invite-only. On the Travelpayouts side, the tools page's "Data API" card
turned out to be affiliate *statistics*, not fares; the fare API comes with
connecting to the Aviasales program. The first token added was the 63-character
`travelpayouts-…` one, which gets `401` on every fare endpoint — so does
sending no token at all, which is how it became clear the account wasn't
entitled rather than the token being malformed. The 32-character hex token
from the Aviasales program's API section works. **INR is supported.**

**What was built:**

- `app/providers/travelpayouts.py` on `/aviasales/v3/prices_for_dates`:
  operating airline, flight number, departure time, per-leg duration, stops,
  price. Place names resolve to IATA codes through Travelpayouts' own city
  directory.
- Each leg is searched **±1 day** around the travel date (the user's choice,
  over exact-date-only or a wider window). Every option is kept on the
  itinerary as `outbound_options` / `return_options`.
- One table per leg, **Flight | Price** columns, **Cheapest | Fastest** rows,
  up to three flights per row, each with duration, stops, date and local
  departure time.
- The flight **booked** is the cheapest on the travel date itself; a cheaper
  one a day early is listed, not booked. It only falls back to a neighbouring
  day when nothing departs on the date.

**Bugs found by running it against the live API before committing:**

1. **The first endpoint ignored travel dates.** `/v2/prices/latest` returns
   the year's cheapest fares: a 10–13 October trip got an outbound on
   5 October and a return on 29 September. Mock data hid it completely,
   because it stamps whatever date it's given. Also, its live response turned
   out to include `duration` even though the published docs omit it — trust
   the live API over the docs, in both directions.
2. **The flight agent kept predicting the wrong booking.** Asked to explain
   "the booked flight", it named the cheaper day-early flight as "chosen" —
   and still did after its prompt was reworded to describe the date rule.
   Fixed structurally: the booked flight is computed *before* the LLM call
   and handed over as `booked_outbound` / `booked_return`. Same lesson as the
   Step 9 hotel bug — an agent's text should describe decisions, never
   anticipate them.
3. **"Delhi" resolved to the wrong city** in testing: an exact-name match on a
   city with no flightable airport beat New Delhi's partial match. Flightable
   matches now win.
4. **My own first version fell back to mock flights** when a route had no
   cached fares. Removed before it shipped: invented flights next to real ones
   are indistinguishable on the page. No fares now means no flights, and the
   itinerary agent is told "no fares were found" rather than the old, now-false
   "no origin was given".

**Measured limit worth knowing:** the cache holds roughly **one fare per route
per day**. The ±1-day window around 10 October on Delhi→Mumbai — one of India's
busiest routes — found a single outbound flight, so both of its rows showed the
same one. The return leg found three, and there Cheapest (12 Oct) and Fastest
(14 Oct, 3h 18m) genuinely differ. `DATE_WINDOW_DAYS` is the one line to change
if the tables are too thin.

---

## Where things stand

| Area | State |
|---|---|
| Agent pipeline | ✅ 5 nodes, parallel flight/hotel, JSON-validated itinerary with retries |
| Persistence | ✅ Postgres, append-only version history per `thread_id` |
| Revisions | ✅ Backend + API — **no frontend UI yet** |
| Frontend | ✅ Dark theme, free-text prompt, rupee rendering, cheapest/fastest flight table per leg — no history/revise UI |
| Travel data | ⚠️ **Flights are real** under `TRAVELMATE_PROVIDER=live` (Travelpayouts). Lodging, attractions and weather are **still mock**. The `.env` default is still `mock` |
| MCP | ✅ One client (Tavily remote), two servers (AviationStack, weather) — all standalone |
| Currency | ✅ Rupee-native, with legacy USD trips preserved |
| Tests | ✅ 159 passing, `ruff` clean |
| GitHub | ❌ Never pushed — `gh auth login` was never completed, no remote configured |

**Obvious next steps**, roughly in order of value:

1. Real lodging data — the next-biggest gap now that flights are real.
2. Frontend UI for trip history and revisions (backend is done).
3. Day-level patch revisions, if revision latency matters.
4. Push to GitHub.

---

## Adding to this log

When a feature lands, append a step under the right phase (or start a new
phase) with:

- **Heading**: `### Step N · <what it is> (<commit hash>, <date>)`
- **What changed** — the substance, not a diff summary.
- **Why**, when a real decision was made, especially one where the obvious
  alternative was rejected.
- **Bugs found live** — this log is more useful for the mistakes than the
  features. Keep them.
- Update the **Where things stand** table in the same pass, or it goes stale
  and becomes actively misleading.
