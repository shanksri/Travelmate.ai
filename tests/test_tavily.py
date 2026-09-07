"""Fetch tests mock httpx entirely (see the autouse block in conftest.py) —
no real call, no API quota spent. Normalization tests need no network at all;
they run a saved sample shaped exactly like the real response fetched during
development."""

import pytest
from conftest import FakeHTTPResponse

from app.providers.tavily import TavilyError, fetch_search, normalize_search

# Trimmed version of a real response, captured from a live fetch during
# development — one result is enough to exercise the shape.
SAMPLE_RESPONSE = {
    "query": "best time of year to visit Lisbon, Portugal",
    "follow_up_questions": None,
    "answer": None,
    "images": [],
    "results": [
        {
            "url": "https://www.royalcaribbean.com/inspire/best-time-to-visit-lisbon",
            "title": "The Best Time to Visit Lisbon: A Seasonal Guide",
            "content": "The best time to visit Lisbon is during spring and fall...",
            "score": 0.93735445,
            "raw_content": None,
            "id": "3dbd3a-00",
        }
    ],
    "response_time": 1.73,
    "request_id": "636753a7-ed21-4fcb-87d3-489c3cdbb82c",
}


# --- fetch_search ----------------------------------------------------------


def test_fetch_search_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(TavilyError, match="not set"):
        fetch_search("Lisbon")


def test_fetch_search_returns_the_parsed_payload(monkeypatch):
    monkeypatch.setattr("httpx.post", lambda *a, **k: FakeHTTPResponse(SAMPLE_RESPONSE))

    result = fetch_search("Lisbon", api_key="fake-key")

    assert result == SAMPLE_RESPONSE


def test_fetch_search_raises_on_an_http_error(monkeypatch):
    monkeypatch.setattr(
        "httpx.post", lambda *a, **k: FakeHTTPResponse({"detail": "bad key"}, status_code=401)
    )

    with pytest.raises(TavilyError, match="401"):
        fetch_search("Lisbon", api_key="fake-key")


# --- normalize_search --------------------------------------------------------


def test_normalize_search_keeps_only_what_a_planner_needs():
    normalized = normalize_search(SAMPLE_RESPONSE)

    assert normalized.query == SAMPLE_RESPONSE["query"]
    assert normalized.answer is None
    assert len(normalized.results) == 1

    result = normalized.results[0]
    assert result.title == "The Best Time to Visit Lisbon: A Seasonal Guide"
    assert result.url == SAMPLE_RESPONSE["results"][0]["url"]
    assert result.snippet == SAMPLE_RESPONSE["results"][0]["content"]
    assert result.relevance == 0.93735445


def test_normalize_search_drops_fields_a_planner_does_not_need():
    dumped = normalize_search(SAMPLE_RESPONSE).model_dump()

    assert "response_time" not in dumped
    assert "request_id" not in dumped
    assert "images" not in dumped
    assert "raw_content" not in dumped["results"][0]
    assert "id" not in dumped["results"][0]


def test_normalize_search_keeps_a_real_answer_when_present():
    raw = SAMPLE_RESPONSE | {"answer": "Spring and fall are best."}

    assert normalize_search(raw).answer == "Spring and fall are best."


def test_normalize_search_on_no_results_returns_an_empty_list():
    normalized = normalize_search({"query": "x", "results": []})

    assert normalized.results == []
