"""Live flight fares from Google Flights, via SerpApi's hosted MCP server.

This is the planner's live flight source. It replaces Travelpayouts, whose
cached fares were too sparse to use: Varanasi -> Cochin had two fares in all
of October and none near the travel date, while this same search returns
several real flights on the exact day.

Talks MCP to `https://mcp.serpapi.com/mcp` (streamable HTTP), calling its
`search` tool with `engine=google_flights`. The API key goes in an
`Authorization: Bearer` header rather than the URL path SerpApi also accepts:
the MCP SDK logs every request URL, so a key in the path lands in the logs.

Each call is one search against SerpApi's quota (100/month on the free plan),
and a trip makes two — one per leg.
"""

import asyncio
import json
import os
from datetime import date
from typing import Any

import httpx2
from dotenv import load_dotenv
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import CallToolResult

from app.providers.iata import airports_for, resolve_route

load_dotenv()

MCP_URL = "https://mcp.serpapi.com/mcp"


class GoogleFlightsError(RuntimeError):
    """The search failed, or returned an error instead of flights."""


def _api_key(api_key: str | None = None) -> str:
    key = api_key or os.environ.get("SERPAPI_API_KEY")
    if not key:
        raise GoogleFlightsError("SERPAPI_API_KEY is not set")
    return key.strip()


async def _call_search(params: dict, api_key: str) -> CallToolResult:
    headers = {"Authorization": f"Bearer {api_key}"}
    async with httpx2.AsyncClient(headers=headers, timeout=90) as http_client:
        async with streamable_http_client(MCP_URL, http_client=http_client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                return await session.call_tool("search", {"params": params, "mode": "compact"})


def fetch_flights(
    *,
    departure_ids: str,
    arrival_ids: str,
    outbound_date: str,
    api_key: str | None = None,
) -> dict:
    """One-way Google Flights results for one day, as SerpApi returns them.

    `departure_ids` / `arrival_ids` are airport codes, comma-separated for a
    multi-airport city ("NRT,HND"). Prices are per adult, in rupees.
    """
    params = {
        "engine": "google_flights",
        "departure_id": departure_ids,
        "arrival_id": arrival_ids,
        "outbound_date": outbound_date,
        "type": "2",  # one way
        "adults": "1",
        "currency": "INR",
        "hl": "en",
        "gl": "in",
    }
    try:
        result = asyncio.run(_call_search(params, _api_key(api_key)))
    except GoogleFlightsError:
        raise
    except Exception as exc:  # network/transport failures surface uniformly
        raise GoogleFlightsError(f"SerpApi MCP call failed: {exc}") from exc

    text = next((block.text for block in result.content if hasattr(block, "text")), "")
    if result.is_error:
        raise GoogleFlightsError(text or "SerpApi search failed")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise GoogleFlightsError(f"SerpApi returned non-JSON output: {text[:200]}") from exc
    # Some responses wrap the JSON as a string under "result".
    if isinstance(payload.get("result"), str):
        payload = json.loads(payload["result"])
    if payload.get("error"):
        raise GoogleFlightsError(str(payload["error"]))
    return payload


def normalize_flights(raw: dict, *, travelers: int = 1) -> list[dict[str, Any]]:
    """Turn raw Google Flights results into the flight-option shape the rest
    of this app uses (the same keys `MockTravelProvider.search_flights`
    returns, so `FlightLeg(**option)` works unchanged). Cheapest first.

    Google occasionally lists a flight with no price; those are skipped,
    since an unpriced option can't sit in a cheapest-first list.
    """
    options: list[dict[str, Any]] = []
    for itinerary in (raw.get("best_flights") or []) + (raw.get("other_flights") or []):
        price = itinerary.get("price")
        segments = itinerary.get("flights") or []
        if price is None or not segments:
            continue

        departure = segments[0].get("departure_airport") or {}
        arrival = segments[-1].get("arrival_airport") or {}
        departs = departure.get("time") or ""  # "2026-10-08 11:55", local time
        if not departs:
            continue
        airlines = list(dict.fromkeys(s.get("airline") for s in segments if s.get("airline")))
        minutes = itinerary.get("total_duration")

        options.append(
            {
                "carrier": " + ".join(airlines) or "unknown",
                "origin": departure.get("id"),
                "destination": arrival.get("id"),
                "depart_date": departs[:10],
                "departure_at": departs.replace(" ", "T"),
                "stops": len(segments) - 1,
                "duration_hours": round(minutes / 60, 1) if minutes else None,
                "price_per_person": float(price),
                "total": round(float(price) * travelers, 2),
            }
        )
    options.sort(key=lambda o: o["total"])
    return options


def search_flights(
    origin: str, destination: str, depart: date, travelers: int
) -> list[dict[str, Any]]:
    """Flights for one leg on `depart`, taking place names, cheapest first.

    Returns `[]` rather than raising when either place can't be resolved —
    a trip should still plan without flights, as it does with no origin.
    """
    origin_code, destination_code = resolve_route(origin, destination)
    if not origin_code or not destination_code or origin_code == destination_code:
        return []

    raw = fetch_flights(
        departure_ids=",".join(airports_for(origin_code)),
        arrival_ids=",".join(airports_for(destination_code)),
        outbound_date=depart.isoformat(),
    )
    return normalize_flights(raw, travelers=travelers)


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    flights = search_flights("Varanasi", "Kochi", date(2026, 10, 8), travelers=1)
    print(f"{len(flights)} option(s) for Varanasi -> Kochi on 2026-10-08")
    print(json.dumps(flights[:3], indent=2))
