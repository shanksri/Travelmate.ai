"""Tests mock `_call_search` (the MCP round trip) — no real call, and none of
the 100-a-month SerpApi quota spent. The sample is trimmed from a real
Varanasi -> Cochin search captured during development."""

import json
from datetime import date

import pytest
from conftest import FakeMCPToolResult

from app.models.itinerary import FlightLeg
from app.providers import google_flights, iata
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
    monkeypatch.setenv("SERPAPI_API_KEY", "fake")
    sent = []

    def install(result):
        async def fake_call_search(params, api_key):
            sent.append(params)
            return result

        monkeypatch.setattr(google_flights, "_call_search", fake_call_search)
        return sent

    return install


# --- fetch_flights ---------------------------------------------------------


def test_fetch_requires_an_api_key(monkeypatch):
    monkeypatch.delenv("SERPAPI_API_KEY", raising=False)

    with pytest.raises(GoogleFlightsError, match="not set"):
        fetch_flights(departure_ids="VNS", arrival_ids="COK", outbound_date="2026-10-08")


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
    fake_search(FakeMCPToolResult({"error": "Google Flights hasn't returned any results"}))

    with pytest.raises(GoogleFlightsError, match="hasn't returned any results"):
        fetch_flights(departure_ids="VNS", arrival_ids="COK", outbound_date="2026-10-08")


def test_a_transport_failure_becomes_a_google_flights_error(monkeypatch):
    monkeypatch.setenv("SERPAPI_API_KEY", "fake")

    async def boom(params, api_key):
        raise ConnectionError("network down")

    monkeypatch.setattr(google_flights, "_call_search", boom)

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
