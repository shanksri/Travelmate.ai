# Original roadmap

The diagram this project was built from — a 4-agent LangGraph pipeline with a
shared `TravelState` and long-term memory in PostgreSQL.

![TripMate AI roadmap](roadmap.png)

> This is a Multi Agents travel planner that turns a natural-language trip
> request into a practical travel plan with flight suggestions, hotel ideas,
> and a day-by-day itinerary. The project uses a multi-agent workflow built
> with LangGraph, combining a flight-search agent, a hotel-research agent, an
> itinerary-planning agent, and a final response agent, all coordinated
> through a LangGraph workflow.

## What's implemented vs. deferred

| Diagram | This codebase |
|---|---|
| LangGraph, 4 agents + shared state | ✅ [app/agent/graph.py](../app/agent/graph.py), [state.py](../app/agent/state.py) — plus a `resolve_destination` coordinator step the diagram doesn't show, for when the traveller leaves the destination open |
| Groq LLM (Llama 3) | ✅ every agent reasons with Groq (`llama-3.3-70b-versatile` by default) — see [app/agent/llm.py](../app/agent/llm.py) |
| AviationStack, Google Places/Maps, Tavily Search | ⏸ stood in for by one deterministic mock provider ([app/providers/mock.py](../app/providers/mock.py)) behind the `TravelProvider` Protocol, so the system runs with zero third-party keys. Real integrations are a drop-in behind that Protocol. |
| PostgreSQL long-term memory | ⏸ trips live in an in-memory `TripStore` ([app/store.py](../app/store.py)) for now, behind its own Protocol |

The diagram's per-agent "Tools/APIs" boxes don't specify whether flight/hotel/
itinerary search itself is LLM-driven tool-calling or deterministic retrieval
with the LLM reasoning over the results. This codebase takes the latter,
simpler path — each node calls its provider function directly, then asks Groq
to pick/plan over what came back — documented in
[README.md](../README.md#architecture).
