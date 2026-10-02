"""Tests mock `_call_search` (the MCP round trip) — no real call, and none of
the 100-a-month SerpApi quota spent. The sample is trimmed from a real
Varanasi -> Cochin search captured during development."""

import json
from datetime import date

import pytest
from conftest import FakeMCPToolResult

from app.models.itinerary import FlightLeg
from app.providers import iata, serpapi
from app.providers.google_flights import (
    GoogleFlightsError,
    fetch_flights,
    normalize_flights,
    search_flights,
)


def _itinerary(price, departs, segments, total_duration):
    return {
        "flights": [
            {
                "departure_airport": {"id": frm, "time": dep},
                "arrival_airport": {"id": to, "time": arr},
                "airline": airline,
                "flight_number": number,
            }
            for frm, to, dep, arr, airline, number in segments
        ],
        "total_duration": total_duration,
        "price": price,
        "_departs": departs,
    }


VIA_BLR = [
    ("VNS", "BLR", "2026-10-08 11:55", "2026-10-08 14:25", "IndiGo", "6E 6559"),
    ("BLR", "COK", "2026-10-08 15:50", "2026-10-08 16:50", "IndiGo", "6E 2138"),
]
VIA_DEL = [
    ("VNS", "DEL", "2026-10-08 18:40", "2026-10-08 20:10", "IndiGo", "6E 2011"),
    ("DEL", "COK", "2026-10-08 21:30", "2026-10-08 23:25", "Air India", "AI 511"),
]

SAMPLE = {
    "other_flights": [
        _itinerary(12589, "18:40", VIA_DEL, 285),
        _itinerary(9131, "11:55", VIA_BLR, 295),
        _itinerary(None, "06:00", VIA_BLR, 300),  # real quirk: Google lists some unpriced
    ],
    "price_insights": {"lowest_price": 9131},
}


@pytest.fixture
def fake_search(monkeypatch):
    """Replaces the MCP round trip; records the params each search sent."""
    monkeypatch.setenv("SERPAPI_FLIGHTS_API_KEY", "fake")
    sent = []

    def install(result):
        async def fake_call_search(params, api_key):
            sent.append(params)
            return result

        monkeypatch.setattr(serpapi, "_call_search", fake_call_search)
        return sent

    return install


# --- fetch_flights ---------------------------------------------------------


def test_fetch_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("SERPAPI_FLIGHTS_API_KEY", raising=False)
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    with pytest.raises(GoogleFlightsError, match="SERPAPI_FLIGHTS_API_KEY is not set"):
        fetch_flights(departure_ids="VNS", arrival_ids="COK", outbound_date="2026-10-08")


def test_the_older_single_key_name_still_works(monkeypatch):
    from app.providers.google_flights import _api_key

    monkeypatch.delenv("SERPAPI_FLIGHTS_API_KEY", raising=False)
    monkeypatch.setenv("SERPAPI_API_KEY", "old-name")

    assert _api_key() == "old-name"


def test_fetch_sends_a_one_way_rupee_search(fake_search):
    sent = fake_search(FakeMCPToolResult(SAMPLE))

    fetch_flights(departure_ids="VNS", arrival_ids="COK", outbound_date="2026-10-08")

    params = sent[0]
    assert params["engine"] == "google_flights"
    assert params["outbound_date"] == "2026-10-08"
    assert params["type"] == "2"
    assert params["currency"] == "INR"
    assert params["adults"] == "1"


def test_fetch_returns_the_parsed_payload(fake_search):
    fake_search(FakeMCPToolResult(SAMPLE))

    raw = fetch_flights(departure_ids="VNS", arrival_ids="COK", outbound_date="2026-10-08")

    assert len(raw["other_flights"]) == 3


def test_fetch_unwraps_json_serialized_under_result(fake_search):
    fake_search(FakeMCPToolResult({"result": json.dumps(SAMPLE)}))

    raw = fetch_flights(departure_ids="VNS", arrival_ids="COK", outbound_date="2026-10-08")

    assert len(raw["other_flights"]) == 3


def test_a_tool_error_raises(fake_search):
    fake_search(FakeMCPToolResult(is_error=True))

    with pytest.raises(GoogleFlightsError):
        fetch_flights(departure_ids="VNS", arrival_ids="COK", outbound_date="2026-10-08")


def test_an_error_in_the_payload_raises(fake_search):
    fake_search(FakeMCPToolResult({"error": "Something unexpected went wrong"}))

    with pytest.raises(GoogleFlightsError, match="Something unexpected"):
        fetch_flights(departure_ids="VNS", arrival_ids="COK", outbound_date="2026-10-08")


def test_no_results_is_an_empty_answer_not_an_error(fake_search):
    """Google finding no flights on a route is an answer, not a failure: it
    mustn't be retried, or count against the source."""
    fake_search(FakeMCPToolResult({"error": "Google Flights hasn't returned any results"}))

    raw = fetch_flights(departure_ids="VNS", arrival_ids="COK", outbound_date="2026-10-08")

    assert normalize_flights(raw) == []


def test_a_transport_failure_becomes_a_google_flights_error(monkeypatch):
    monkeypatch.setenv("SERPAPI_FLIGHTS_API_KEY", "fake")

    async def boom(params, api_key):
        raise ConnectionError("network down")

    monkeypatch.setattr(serpapi, "_call_search", boom)

    with pytest.raises(GoogleFlightsError, match="network down"):
        fetch_flights(departure_ids="VNS", arrival_ids="COK", outbound_date="2026-10-08")


# --- normalize_flights -----------------------------------------------------


def test_normalize_skips_unpriced_flights_and_sorts_cheapest_first():
    options = normalize_flights(SAMPLE)

    assert [o["price_per_person"] for o in options] == [9131.0, 12589.0]


def test_normalize_reads_route_time_stops_and_duration():
    cheapest = normalize_flights(SAMPLE)[0]

    assert cheapest["origin"] == "VNS"
    assert cheapest["destination"] == "COK"  # the final arrival, not the layover
    assert cheapest["departure_at"] == "2026-10-08T11:55"
    assert cheapest["depart_date"] == "2026-10-08"
    assert cheapest["stops"] == 1
    assert cheapest["duration_hours"] == 4.9  # 295 minutes


def test_normalize_names_every_airline_on_a_mixed_itinerary():
    mixed = normalize_flights(SAMPLE)[1]

    assert mixed["carrier"] == "IndiGo + Air India"


def test_normalize_scales_total_by_party_size():
    cheapest = normalize_flights(SAMPLE, travelers=3)[0]

    assert cheapest["total"] == 27393.0


def test_normalize_output_builds_a_valid_FlightLeg():
    leg = FlightLeg(**normalize_flights(SAMPLE)[0])

    assert leg.departure_at.hour == 11
    assert leg.depart_date == date(2026, 10, 8)


def test_normalize_handles_best_and_other_flights_together():
    best, other = SAMPLE["other_flights"][1], SAMPLE["other_flights"][0]
    raw = {"best_flights": [best], "other_flights": [other]}

    assert len(normalize_flights(raw)) == 2


# --- search_flights --------------------------------------------------------


@pytest.fixture
def fake_directories(monkeypatch):
    iata._matches.cache_clear()
    iata.airports_for.cache_clear()
    monkeypatch.setattr(
        iata,
        "_cities",
        lambda *a, **k: [
            {"name": "Varanasi", "code": "VNS", "country_code": "IN",
             "has_flightable_airport": True},
            {"name": "Kochi", "code": "KCZ", "country_code": "JP", "has_flightable_airport": True},
            {"name": "Kochi", "code": "COK", "country_code": "IN", "has_flightable_airport": True},
            {"name": "Tokyo", "code": "TYO", "country_code": "JP", "has_flightable_airport": True},
        ],
    )
    monkeypatch.setattr(
        iata,
        "_airports",
        lambda *a, **k: [
            {"code": "NRT", "city_code": "TYO", "iata_type": "airport", "flightable": True},
            {"code": "HND", "city_code": "TYO", "iata_type": "airport", "flightable": True},
        ],
    )
    yield
    iata._matches.cache_clear()
    iata.airports_for.cache_clear()


def test_search_resolves_names_and_searches_the_exact_date(fake_search, fake_directories):
    sent = fake_search(FakeMCPToolResult(SAMPLE))

    options = search_flights("Varanasi", "Kochi", date(2026, 10, 8), 2)

    assert sent[0]["departure_id"] == "VNS"
    assert sent[0]["arrival_id"] == "COK"  # India's Kochi, not Japan's
    assert sent[0]["outbound_date"] == "2026-10-08"
    assert options[0]["total"] == 18262.0


def test_search_sends_every_airport_of_a_multi_airport_city(fake_search, fake_directories):
    sent = fake_search(FakeMCPToolResult(SAMPLE))

    search_flights("Tokyo", "Kochi", date(2026, 10, 8), 1)

    assert set(sent[0]["departure_id"].split(",")) == {"NRT", "HND"}


def test_search_returns_empty_without_searching_when_a_place_is_unknown(
    fake_search, fake_directories
):
    sent = fake_search(FakeMCPToolResult(SAMPLE))

    assert search_flights("Atlantis", "Kochi", date(2026, 10, 8), 1) == []
    assert sent == []  # no quota spent on an impossible search


# --- the retry and fallback chain ----------------------------------------------


@pytest.fixture
def searches(monkeypatch):
    """Scripts the MCP round trip: each call takes the next outcome — a
    payload dict, or an exception to raise. Records how many searches ran."""
    monkeypatch.setenv("SERPAPI_FLIGHTS_API_KEY", "fake")
    script: list = []
    made: list = []

    async def fake_call_search(params, api_key):
        made.append(params)
        outcome = script.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return FakeMCPToolResult(outcome)

    monkeypatch.setattr(serpapi, "_call_search", fake_call_search)
    return script, made


@pytest.fixture
def clock(monkeypatch):
    from app.providers import flight_cache

    now = {"t": 2_000_000.0}
    monkeypatch.setattr(flight_cache.time, "time", lambda: now["t"])
    return now


def _chain():
    from app.providers.google_flights import fetch_flights_with_fallback

    return fetch_flights_with_fallback(
        departure_ids="VNS", arrival_ids="COK", outbound_date="2026-10-08", link="L"
    )


def test_a_live_search_is_labelled_live_and_cached(searches, clock):
    script, made = searches
    script.append(SAMPLE)

    raw, source = _chain()

    assert raw == SAMPLE and source.status == "live" and source.link == "L"
    raw, source = _chain()  # the same search again: no second call
    assert source.status == "cached" and len(made) == 1


def test_a_transient_failure_is_retried(searches, clock):
    script, made = searches
    script.extend([ConnectionError("blip"), SAMPLE])

    raw, source = _chain()

    assert source.status == "live" and len(made) == 2


def test_an_exhausted_quota_is_not_retried_and_skips_later_searches(searches, clock):
    from app.providers import google_flights

    script, made = searches
    script.append({"error": "Your account has run out of searches."})

    _, first = _chain()
    _, second = _chain()  # the breaker is open: no search at all

    assert len(made) == 1
    assert first.status == "unavailable" and "run out of searches" in first.detail
    assert second.status == "unavailable" and "skipped after repeated failures" in second.detail
    assert not google_flights.breaker.allow()


def test_a_failure_falls_back_to_an_older_search_labelled_stale(searches, clock):
    script, made = searches
    script.append(SAMPLE)
    _chain()  # cached now
    clock["t"] += 20 * 3600  # past fresh (6 h), within kept (48 h)
    script.extend([ConnectionError("down")] * 3)

    raw, source = _chain()

    assert raw == SAMPLE
    assert source.status == "stale"
    assert "down" in source.detail
    assert source.fetched_at is not None


def test_with_nothing_kept_a_failure_is_unavailable_with_a_link(searches, clock):
    script, _ = searches
    script.extend([ConnectionError("down")] * 3)

    raw, source = _chain()

    assert raw is None
    assert source.status == "unavailable" and source.link == "L"


def test_search_flights_with_source_never_raises_and_links_google_flights(
    searches, clock, monkeypatch
):
    from app.providers import google_flights

    monkeypatch.setattr(google_flights, "resolve_route", lambda o, d: ("VNS", "COK"))
    monkeypatch.setattr(google_flights, "airports_for", lambda code: [code])
    script, _ = searches
    script.extend([ConnectionError("down")] * 3)

    flights, source = google_flights.search_flights_with_source(
        "Varanasi", "Kochi", date(2026, 10, 8), 1
    )

    assert flights == []
    assert source.status == "unavailable"
    assert "Varanasi" in source.link and "Kochi" in source.link and "2026-10-08" in source.link


def test_an_unmatched_place_is_unavailable_without_searching(searches, monkeypatch):
    from app.providers import google_flights

    monkeypatch.setattr(google_flights, "resolve_route", lambda o, d: ("VNS", None))
    _, made = searches

    flights, source = google_flights.search_flights_with_source(
        "Varanasi", "Kerala and Tamil Nadu", date(2026, 10, 8), 1
    )

    assert flights == [] and made == []
    assert "Kerala and Tamil Nadu" in source.detail


def test_serpapis_real_invalid_key_message_opens_the_breaker(searches, clock):
    """The exact wording seen live. Matching only "invalid api key" missed it,
    so the breaker stayed closed and every search tried the bad key again."""
    from app.providers import google_flights

    script, made = searches
    script.append(
        {
            "error": "Error: Invalid SerpApi API key. Check the key in the request path "
            "or Authorization header, or in SERPAPI_API_KEY for stdio hosts."
        }
    )

    _chain()
    _, second = _chain()

    assert len(made) == 1
    assert "skipped after repeated failures" in second.detail
    assert not google_flights.breaker.allow()
