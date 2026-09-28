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

### Step 24 · Drop the Flights card; fix shared city names (`dae969b`, 2026-09-24)

- **Flights summary card removed** from the frontend, as asked. The per-leg
  cheapest/fastest tables remain.
- **"Kochi" was going to Japan.** The user saw ₹43,480 for Varanasi→Kochi
  where Cleartrip showed ₹8k. Two causes, both found by checking rather than
  guessing: the server was on `mock`, so "Northwind" and its price were
  invented; and under `live`, "Kochi" resolved to **Kochi, Japan (`KCZ`)**,
  since both Kochis are flightable and Japan's comes first. `resolve_route`
  now resolves both ends together and prefers a shared country when a name is
  ambiguous.

**What the check also showed:** even with the right airport (`COK`), the cache
held **two Varanasi→Cochin fares in all of October**, none within five days
of 8 Oct. Where a fare exists, it's in line with Cleartrip (₹9,157 on 3 Oct vs
₹8k). The pricing is sound; the coverage isn't — a cached-fare API can't match
a live search on a quieter route.

### Step 25 · Google Flights via SerpApi's MCP server (`5773bb7`, 2026-09-24)

**Travelpayouts' live-search API was out of reach**: it requires 50,000
confirmed monthly active users, with "no exceptions" — confirmed from their
help center article directly. So the live flight source moved to **Google
Flights via SerpApi** (free plan: 100 searches/month).

The user asked to use SerpApi's **MCP server** rather than its REST API. There
is an official hosted one (`https://mcp.serpapi.com/mcp`, one `search` tool for
every engine), so `app/providers/google_flights.py` is an **MCP client** — the
second in this project after the Tavily one. The key goes in a Bearer header,
not the URL path, because the MCP SDK logs every request URL.

- One search per leg, on the exact date — no date window needed, since one
  date returns many flights. The ±1-day window and the Travelpayouts fare
  client were deleted.
- Place-name resolution moved to `app/providers/iata.py`, still on
  Travelpayouts' **free public directories** (no key), and now expands
  multi-airport cities: Google Flights wants airport codes, and `TYO` isn't one.
- Google sometimes lists a flight with no price; those are skipped.

**Verified live**: the same Varanasi→Cochin trip that had zero cached fares
returned **9 outbound and 5 return flights** on the exact dates, cheapest
₹9,131 (Cleartrip: ~₹8k), and the Cheapest and Fastest rows now show genuinely
different flights. The flight agent's explanation matched the booked flight.

**Also changed by the user**: the Travelpayouts and AviationStack keys were
removed from `.env`. The AviationStack MCP server (Step 17) is still in the
code but now has no key; whether to keep or delete it is undecided.

### Step 26 · Cache flight searches (`8077f80`, 2026-09-24)

With 100 SerpApi searches a month and two per trip, re-planning the same trip
was burning quota for identical results. The user asked about using a second
SerpApi account instead; that was advised against — SerpApi's terms reserve
the right to limit use per person and household and to terminate for
circumvention — and caching was chosen as the legitimate way to stretch it.

- `app/providers/flight_cache.py` caches each **raw, per-adult** search for
  `TRAVELMATE_FLIGHT_CACHE_TTL_HOURS` (default 6; 0 disables), keyed by
  airports and date — so different party sizes on the same route share one
  search. Failed searches are never cached.
- It follows `TRAVELMATE_STORE`: in memory, or a `flight_search_cache` table
  in Postgres. Postgres matters because `uvicorn --reload` restarts the
  process on every code change, which would keep wiping an in-memory cache.
  Timestamps are stored as epoch seconds to avoid naive-vs-aware datetime
  differences between Postgres and SQLite. Expired rows are pruned on write.

**Verified against the real Postgres**, in two separate processes to stand in
for a restart: the same Varanasi↔Kochi searches took **2 real searches and
7.8s** the first time, **0 and 1.1s** the second.

**Also found this round:** the user reported "wrong" return flights — they
were mock data (the server is still on `TRAVELMATE_PROVIDER=mock`); the exact
figures reproduced from the mock provider. Second time mock data has passed
for real on the page.

### Step 27 · Flights and Hotels checkboxes (`f02e877`, 2026-09-25)

Two checkboxes under the prompt box, **unchecked by default**: flights and
hotels are only searched when ticked. They set `include_flights` /
`include_hotels` on `POST /trips/plan-from-prompt`, applied after parsing the
sentence rather than inferred from it.

- An unchecked agent **skips its search and its LLM call entirely**. Hiding
  results after searching would still have spent a SerpApi search per leg.
- The itinerary agent is told the traveller didn't ask for flights or a
  hotel. The previous wording, "No lodging options were found", invited it to
  improvise a hotel into the plan.
- Both default to `true` on `TripRequest` and the API, so `POST /trips/plan`,
  the CLI and stored trips are unchanged.

**Verified without running the pipeline**, as the user asked: the request was
intercepted in the browser, and all three combinations (none, flights only,
both) went out exactly as ticked. Tests prove a skipped agent never touches
its provider or the LLM.

### Step 28 · "Change this plan" box (`8785852`, 2026-09-25)

A second text box appears **above the generated plan** once there is one.
Typing a change ("add more activities on day 3") posts it to the existing
`POST /trips/{thread_id}/revise` endpoint from Step 20, and the page re-renders
with the result. Changes stack, and each one builds on the previous result.

- **The original is kept.** Each change is saved as the next version of the
  same `thread_id`, and earlier versions are never overwritten. This is the
  versioned store, not a cache. A cache can expire or evict, which is wrong
  for anything a user edits.
- Frontend only. The backend already existed.
- Generating a new plan hides the box until that plan arrives, so a change
  can't land on the wrong trip.

**Verified without spending an OpenAI call**: a stored trip was rendered in
the browser and the revise request was intercepted. It went to the right
thread with the typed change, and the returned trip replaced the one on
screen.

### Step 29 · Revisions that can reshape the route, and say when they can't (`4914537`, 2026-09-26)

**The bug:** "adjust rameshwaram in this itenary as well" saved a version 2
with all ten days identical to version 1. The model had declined in a note
("requires significant travel adjustments"), and the page no longer shows
notes, so nothing seemed to happen.

- **Two kinds of change in the revise prompt.** A small edit ("more on day 3")
  still changes only the days it names. A route change (add, drop or extend a
  place) may rework the days around it: shorten other stops, put the new
  place where it falls geographically, add realistic transit. The old rule,
  "every other day byte-for-byte", was what the model hid behind.
- **Hard limits on declining.** The trip must still start and end where it
  does, a new stop must fit its travel both ways, and a requested number of
  days is given in full or declined. A decline starts with
  `Couldn't apply:` so the code can find the reason.
- **No silent failures.** If every day comes back unchanged, `revise_trip`
  raises `RevisionDeclined` before the summary call. The API returns 422 with
  the model's reason, the revise box shows it, and **nothing is saved**.
- **Previous notes are no longer sent to the model.** Found live: revising
  version 2 kept failing because its old refusal note, "Rameshwaram is not
  included…", was fed back in as an existing note. Revising version 1 of the
  same trip worked.

**Tested live on the real Kerala trip, without saving:**

| Request | `gpt-4o-mini` | `gpt-4o` |
|---|---|---|
| Add Rameshwaram | Applied, but placement varies between runs (after Madurai on one run, just before the Kochi departure on another) | Applied after Madurai, then Kanyakumari, then Kochi |
| Add 3 days in Ladakh | Agreed: 1–2 days in Leh, once leaving the traveller in Leh on departure day | Declined, with a clear reason |

The instructions were right; `gpt-4o-mini` can't judge travel distances
reliably. Resolved in Step 30.

### Step 30 · `gpt-4o` for revisions only (`770e1ac`, 2026-09-26)

New setting `TRAVELMATE_REVISE_MODEL`, default `gpt-4o`, used by the reviser
alone. Planning stays on `TRAVELMATE_MODEL` (`gpt-4o-mini`), so the extra
cost only applies when someone changes a plan.

- Same API key. One OpenAI key covers every model the account can use.
- Revisions take about the same time (20–30 s). They cost more per call
  than `gpt-4o-mini`.
- **Verified through the running server:** "add 3 days in Ladakh" on the
  Kerala trip returned 422 with *"Adding 3 days in Ladakh is not feasible
  within the current itinerary…"*. The trip still had 2 versions afterwards.

### Step 31 · No more planning somewhere nobody asked for (`b6f6afd`, 2026-09-26)

**The bug:** "give me varanasi to kerala and tamil nadu itenary" produced a
five-day **Chiang Mai, Thailand** plan. The itinerary AI wasn't the cause:

1. The parser returned **no destination**, 4 times out of 4. It put "Kerala
   and Tamil Nadu" in `notes`, apparently not counting two states as *a*
   place.
2. An empty destination sends the graph to `resolve_destination`, which
   picks from the provider's **sample list**. The trace read
   `coordinator: picked Chiang Mai, Thailand (matched [])`, meaning it
   matched none of the interests.
3. Five days is the default length when none is given.

**Fixes:**

- The parser prompt says a destination can be a city, state, region or
  country, **or several together**, and is null only when no place was
  named. After the fix, the same sentence parsed as "Kerala and Tamil Nadu"
  7 times out of 7, and "Rajasthan and Gujarat" works the same way.
- `plan_trip_from_prompt` refuses a missing destination with a 422, *"no
  destination was named. Say where you want to go…"*, before any agent
  runs. Verified through the server: refused in 2.3 s, nothing saved.
- The structured `POST /trips/plan` keeps its documented pick-one behaviour.

**Tried and reverted:** a line telling the parser "a kind of place (a
beach, the mountains) is not a destination". `gpt-4o-mini` copied it
literally and returned `"a beach"` as the destination. So a vague request
like "suggest me a beach trip" still gets a vague destination.

The checkboxes were confirmed working as intended: the Chiang Mai trip's
trace shows both flight and hotel agents skipped, with no search.

### Step 32 · Calendar date pickers (`a50d976`, 2026-09-26)

Optional **Start** and **End** pickers sit next to the Flights/Hotels
checkboxes. They use the browser's built-in date input, with
`color-scheme: dark` so the calendar popup matches the theme.

- **Optional, and they win.** With both picked, `start_date` / `end_date`
  replace whatever dates the sentence implied. This follows the checkboxes'
  pattern: applied after parsing, not left for the model to infer. With
  neither picked, the sentence and the defaults decide. That is where the
  "5 days starting two weeks out" guess comes from, which is what a request
  like "Varanasi to Kerala" used to get.
- **Checked twice.** The pickers won't offer a past start, an end before the
  start, or more than 60 nights, and the page refuses half a pair. The API
  enforces the same limits with a 422 for anyone calling it directly. The
  60-night limit is now one constant, `MAX_TRIP_NIGHTS`, shared with
  `TripRequest`.
- The parsed request is re-validated with the picked dates rather than
  `model_copy`'d, so picked dates pass the same checks as parsed ones.
- Dates are formatted in the viewer's timezone. `toISOString()` would give
  the UTC date, which is yesterday in India before 05:30.

**Verified without running the pipeline:** in the browser, with the plan
request intercepted, only-start was blocked on the page, both-picked sent
the dates, and neither sent nulls. A real request with a past start got a
422 from the server. At phone width the pickers wrap onto their own rows
with no sideways scroll.

### Step 33 · Changes routed to the right agent (`067a6eb`, 2026-09-26)

Taken from a reference diagram's "request changes → loop back to relevant
agents". Before this, "Change this plan" could only re-plan the days;
"nonstop flights" or "a better hotel" had nowhere to go.

- **A router call first** (`ROUTE_CHANGE_SYSTEM`, on the revise model). It
  sees the flight options per leg with the booked one marked, the hotel
  options with the selected one marked, and a one-line outline of each day.
  It answers with the option index to switch to per leg and for the hotel,
  the instruction for the days, or a decline reason. An index that doesn't
  exist is sent back with feedback.
- **Flights and the hotel are re-picked, never re-searched.** Choosing by
  index from the saved options means no SerpApi quota is spent and no model
  writes a flight or hotel, so nothing can be hallucinated. The hotel switch
  moves the pick to the front of the list, because the first option is
  "selected" everywhere.
- **The days are re-planned only when the change concerns them.** A
  flight-only change skips the ~15–20 s itinerary call: "nonstop flights both
  ways" took 2.8 s live.
- **All or nothing.** If any part can't be done, the whole change is refused
  with a reason naming that part, and nothing is saved.
- **Booked flights are now visible.** The flight tables tag the booked option
  per leg. Before, the booked flight only showed up in the total cost, so a
  flight change would have looked like nothing happened.

**Bugs found live** (real `gpt-4o`, nothing saved):

- "Nonstop outbound, and a houseboat trip on day 3" put the houseboat on
  **day 2**. The router had passed on only "add a backwater houseboat trip".
  The prompt now requires the day instruction in the traveller's own words,
  keeping day numbers and places. On re-test only day 3 changed.
- "A cheaper return flight (already the cheapest) and a houseboat on day 3"
  applied only the houseboat and dropped the flight part silently. It now
  declines with *"A cheaper return flight cannot be booked as the current
  one is already the cheapest available."*

**Limits:** most saved trips predate option lists, so on those, flight
changes are refused ("no flight options"). A trip planned without flights
can't gain them through a change; plan it again with Flights ticked.

### Step 34 · Places and routes from Google Maps (`293a3b5`, 2026-09-28)

"Show me best places to eat in Bhubaneswar" and "how far is Rameshwaram from
Madurai" now get answers instead of being planned as trips.

- **Google's official hosted MCP server**, Maps Grounding Lite
  (`https://mapstools.googleapis.com/mcp`), over a custom server, for the
  same reason SerpApi's official one was used for flights. Its
  `search_places` returns a written summary citing places as `[0]`, `[1]`…
  with a Maps link each; `compute_routes` returns distance and duration by
  road or on foot. 10,000 free requests a month.
- **No extra LLM call to tell requests apart.** The parser's existing call
  now also returns `intent` (trip / places / route), `places_query` and
  `travel_mode`. On real `gpt-4o-mini`, 8 of 8 test sentences sorted
  correctly, including "5 days in Goa with good seafood restaurants", which
  stays a trip. Places and routes make no further LLM calls.
- **`POST /ask`** is what the prompt box calls now.
  `POST /trips/plan-from-prompt` still plans a trip from any sentence.
- **Google's terms shaped the rest.** Results are never saved or cached, so
  `/ask` stores only trips. Every answer shows "Results from Google Maps",
  and walking routes show Google's required beta warning. The summary's
  citations become numbered links to each place.

**Found live:**

- The tool description says `text_query`, but the server only accepts
  camelCase (`textQuery`, `travelMode`).
- The key worked for listing tools, but every call failed because **the Maps
  Grounding Lite API wasn't enabled** on its Google Cloud project. The MCP
  SDK reduced Google's 403 (which names the fix) to a bare JSON-RPC
  `-32603`, "Server returned an error response". A raw JSON-RPC probe
  showed the real message. The client now turns that case into a message
  saying what to check.

**Verified live** once the API was enabled (the first call after enabling it
was still refused, while the change took effect):

| Sentence | Result | Time |
|---|---|---|
| "show me best places to eat in bhuvneshwar" | 5 restaurants, each with ★ rating and review count in the summary | 5.0 s |
| "best street food in Kolkata" | 5 places, from Zakaria Street to a sweet shop open since 1844 | 3.0 s |
| "how far is rameshwaram from madurai" | 173 km, about 3 h 3 min by road | 2.9 s |

Each place's attribution title turned out to be its **name** ("The BOMBAI -
Google Maps"). The footer now reads "From Google Maps:" followed by every
place name linked (`0425071`), instead of repeating the full titles.


### Step 35 · Places in each city of a trip (`1fd47a3`, 2026-09-28)

**The bug:** "give me 6 day kerala itenary, also tell me best places to eat
in each city" returned a plan with no places to eat. The parser rightly
treated it as a trip, but reduced the second half to a `food` interest, so
Google Maps was never asked. The plan's own food items ("Lunch at a Local
Restaurant") were generic, because the itinerary AI only has the sample
attractions list.

- **The parser keeps the request.** A new `places_per_city` field
  ("restaurants", "cafes", "street food") goes on the trip request. The
  prompt says a food interest alone is not a request for places.
- **Every itinerary day names its `city`**, in both the planning and
  revising formats. Before, the city was only inside the free-text summary,
  where "Departure" could pass for a place.
- **One Maps search per city, in parallel,** once the plan exists: "best
  restaurants in Kochi", "…in Munnar". Capped at 6 cities, in visiting
  order. Results come back beside the trip (`places_by_city`), not inside
  it, so they're never saved (Google's terms). If one city fails, the plan
  is still returned with an error line for that city.

**Verified live** with the exact sentence through the page: 24 s in total.
The days came back based in Kochi and Munnar, and there were two cards,
"best restaurants in Kochi" and "…in Munnar", with 5 real places each
(Restaurant Chef Pillai 4.7★, Fort House Restaurant, Seagull and others).

**Limit:** a revised version of the plan is shown without the city cards.
They belong to the answer, not the saved trip.


### Step 36 · Restaurants checkbox (`b8fc858`, 2026-09-28)

A **Restaurants** box next to Flights and Hotels. Per-city places are looked
up only when it's ticked, the same rule as the other two, rather than
whenever the sentence mentions food. That was Step 35's trigger, and it made
every such trip wait for Maps searches.

- `include_restaurants` on `/ask`, **off by default**. Flights and hotels
  default on, for callers that predate their boxes; nothing predates this.
- The sentence still chooses the kind: "street food in each area" with
  the box ticked searches "best street food in Kolkata". Without a kind,
  it searches restaurants.
- `plan_parsed_trip` always sets `places_per_city` from the box, so a saved
  trip records what was actually looked up, not what the parser read.

Verified in the browser with the request intercepted: the box is sent
unticked by default and ticked when checked. The Maps lookup itself was
verified live in Step 35.

---

## Where things stand

| Area | State |
|---|---|
| Agent pipeline | ✅ 5 nodes, parallel flight/hotel, JSON-validated itinerary with retries |
| Persistence | ✅ Postgres, append-only version history per `thread_id` |
| Revisions | ✅ "Change this plan" box, routed to flights, hotel and/or days; refused whole with a reason when any part cannot be done; `gpt-4o` |
| Frontend | ✅ Dark theme, free-text prompt, rupee rendering, cheapest/fastest flight table per leg, Flights/Hotels/Restaurants checkboxes, optional date pickers, revise box — no history browser |
| Travel data | ⚠️ **Flights are real** under `TRAVELMATE_PROVIDER=live` (Google Flights via SerpApi's MCP server, 100 searches/month, cached 6h). Lodging, attractions and weather are **still mock**. The `.env` default is still `mock` |
| MCP | ✅ Three clients (SerpApi — flights; Google Maps Grounding Lite — places and routes; Tavily — standalone), two servers (AviationStack — now keyless, weather) |
| Currency | ✅ Rupee-native, with legacy USD trips preserved |
| Tests | ✅ 264 passing, `ruff` clean |
| GitHub | ✅ Pushed to `shanksri/Travelmate.ai` (public) over SSH |

---

## Future plans

Agreed on 2026-09-28 after an architecture review. Status is kept up to date
here as items land.

### Architecture

The system grew in layers. Only trip planning runs as a LangGraph graph;
routing /ask, revisions and per-city places are plain Python beside it.
Every MCP call opens a fresh connection. The page waits in silence for up
to a minute.

| # | Plan | Why | Status |
|---|---|---|---|
| 1 | **Stream progress and results.** A plan request returns a job id at once; the server streams events ("flights found", then each day as it's written, then restaurants) over SSE. | Total time barely changes, but the first content appears in seconds instead of ~25 s, and closing the tab no longer loses the work. | Planned, with #2 |
| 2 | **One orchestrator.** /ask becomes a graph with a routing step, then trip, places or route branches. Revisions become a small graph of their own. Per-city places become a LangGraph fan-out. | Parallelism, tracing and streaming in one place. Budget check and guardrails slot in as extra steps. | Planned |
| 3 | **Keep MCP connections open; run parallel work concurrently.** One long-lived session per MCP server, reused across calls, instead of a new connection and handshake every time. | Every Maps and flight call pays the connection cost today. | **In progress** |
| 4 | **Structured outputs.** Every JSON call passes its exact schema to OpenAI (`json_schema`, strict), instead of JSON mode, then hand checks, then retry. | Several fixed bugs were format failures, and each retry costs a full LLM call. | **In progress** |
| 5 | **Evaluation suite and tracing.** The live checks done by hand become a fixed prompt set with expected outcomes, run against the real model on demand. Examples: "Ladakh → declined", "Kerala and Tamil Nadu → one destination", "houseboat → only day 3 changed". LangSmith tracing is already configured in `.env` but not connected. | Prompt and model changes stop being guesswork; per-plan cost becomes visible. | Planned, before the next big prompt change |
| 6 | **One tools layer.** Flights, hotels, places, routes and weather each get an interface with sample and live implementations chosen by config. | Real hotels and real weather become small changes. The weather MCP server built in Step 19 is still unused by the planner. | Planned, with real hotels |

The single `app.js` file and the versioned-JSON store with a hand-written
migration are fine at this size and are not planned to change.

### Features

- **Input guardrails.** A "is this a travel request, and if not why" check
  inside the existing parser call. Deferred by the user.
- **Budget check.** Compare the estimated total with the stated budget in
  code, not with an LLM agent. Deferred by the user.
- **Real hotel data.** Google Hotels via SerpApi (`SERPAPI_HOTEL_API_KEY` is
  already in `.env`); lodging is still sample data.
- **Real weather in the itinerary.** Use the existing weather client or MCP
  server instead of sample weather. Forecasts only reach about two weeks
  ahead; later trips would use typical weather.
- **Browse a trip's earlier versions** on the page (the API already has
  `/trips/{id}/history`).
- **Restaurants inside the day plan.** Named places in the activities ("Lunch
  at Fort House"), not only in a separate card. Needs the cities decided
  before the itinerary is written.
- **City cards for revised versions.** Look up a revised plan's city places
  again, since they aren't saved (Google's terms).
- **Day-level patch revisions.** Regenerate only the days a change touches,
  if revision latency matters.

### Known limits carried forward

- `gpt-4o-mini` turns "suggest me a beach trip" into destination "a beach"
  (Step 31).
- Maps Grounding Lite routes are driving or walking only (Step 34).
- The Google Cloud free trial ends in December 2026; Maps stops unless the
  account is upgraded then.

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
