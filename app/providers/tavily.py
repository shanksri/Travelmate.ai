"""A thin client for the Tavily search API.

Fetch-only for now — this does not yet implement `TravelProvider`; wiring it
in as a "live" data source is a separate step once this is tested.
Standalone-runnable: `python -m app.providers.tavily` does one real fetch,
normalizes it, and prints both.
"""

import json
import os

import httpx
from dotenv import load_dotenv
from pydantic import BaseModel

# Idempotent, and doesn't override a var already set in the real environment
# — consistent with how the rest of this project treats .env as a fallback,
# not an authority.
load_dotenv()

BASE_URL = "https://api.tavily.com"


class TavilyError(RuntimeError):
    """The API responded with an error instead of search results."""


def fetch_search(
    query: str,
    *,
    search_depth: str = "basic",
    max_results: int = 5,
    include_answer: bool = False,
    api_key: str | None = None,
    timeout: float = 15.0,
) -> dict:
    key = api_key or os.environ.get("TAVILY_API_KEY")
    if not key:
        raise TavilyError("TAVILY_API_KEY is not set")

    response = httpx.post(
        f"{BASE_URL}/search",
        headers={"Authorization": f"Bearer {key}"},
        json={
            "query": query,
            "search_depth": search_depth,
            "max_results": max_results,
            "include_answer": include_answer,
        },
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise TavilyError(f"Tavily returned {response.status_code}: {response.text}")

    return response.json()


# --- Normalization ------------------------------------------------------
#
# Tavily's raw response carries fields a trip-planning agent has no use for
# (favicon, raw_content, images, response_time, request_id) alongside the
# ones it does (title, url, a snippet, a relevance score). `normalize_search`
# keeps only the latter.
#
# Kept separate from `fetch_search` on purpose: normalization is pure and
# needs no network access, so it can be unit-tested against a saved sample
# response without spending API calls.


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str
    relevance: float


class SearchResults(BaseModel):
    query: str
    answer: str | None = None
    results: list[SearchResult]


def normalize_search(raw: dict) -> SearchResults:
    """Turn a raw `fetch_search` response into `SearchResults`."""
    return SearchResults(
        query=raw.get("query", ""),
        answer=raw.get("answer"),
        results=[
            SearchResult(
                title=item.get("title", ""),
                url=item.get("url", ""),
                snippet=item.get("content", ""),
                relevance=item.get("score", 0.0),
            )
            for item in raw.get("results", [])
        ],
    )


if __name__ == "__main__":
    raw = fetch_search("best time of year to visit Lisbon, Portugal")
    print("=== raw ===")
    print(json.dumps(raw, indent=2))
    print("\n=== normalized ===")
    print(normalize_search(raw).model_dump_json(indent=2))
