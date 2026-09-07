"""A thin client for the Tavily search API.

Fetch-only for now — this does not yet implement `TravelProvider`; wiring it
in as a "live" data source is a separate step once this is tested.
Standalone-runnable: `python -m app.providers.tavily` does one real fetch and
prints it.
"""

import json
import os

import httpx
from dotenv import load_dotenv

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


if __name__ == "__main__":
    result = fetch_search("best time of year to visit Lisbon, Portugal")
    print(json.dumps(result, indent=2))
