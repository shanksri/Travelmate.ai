"""Google Hotels via SerpApi. Tests mock the MCP round trip — no real search
is made. The sample is trimmed from a real Kochi search (27–30 Oct, 2 adults)
captured during development."""

from datetime import date

import pytest
from conftest import FakeMCPToolResult

from app.providers import google_hotels, serpapi
from app.providers.google_hotels import normalize_hotels, search_hotels_with_source


def _prop(name, nightly, total, rating, hotel_class="4-star hotel", **extra):
    return {
        "type": "hotel",
        "name": name,
        "hotel_class": hotel_class,
        "extracted_hotel_class": int(hotel_class[0]) if hotel_class else None,
        "reviews": 500,
        "overall_rating": rating,
        "rate_per_night": {"lowest": f"₹{nightly:,}", "extracted_lowest": nightly},
        "total_rate": {"lowest": f"₹{total:,}", "extracted_lowest": total},
        **extra,
    }


SAMPLE = {
    "properties": [
        _prop("Grand Hyatt Kochi Bolgatty", 19024, 57071, 4.7, "5-star hotel"),
        _prop("Taj Malabar Resort & Spa, Cochin", 23152, 69455, 4.6, "5-star hotel"),
        _prop("Nihara Resort & Spa, Kadamakudy Island, Kochi", 5239, 15718, 4.4),
        _prop("Cheap And Poorly Rated Lodge", 900, 2700, 2.9, "2-star hotel"),
        {"type": "hotel", "name": "No Price Listed Inn", "overall_rating": 4.5},
        _prop("Unrated New Place", 1500, 4500, None),
    ]
}


def test_normalize_keeps_priced_well_rated_hotels_cheapest_first():
    hotels = normalize_hotels(SAMPLE)

    assert [h["name"] for h in hotels] == [
        "Nihara Resort & Spa, Kadamakudy Island, Kochi",
        "Grand Hyatt Kochi Bolgatty",
        "Taj Malabar Resort & Spa, Cochin",
    ]
    nihara = hotels[0]
    assert nihara["tier"] == "4-star hotel"
    assert nihara["rating"] == 4.4
    assert nihara["nightly"] == 5239.0
    assert nihara["total"] == 15718.0  # Google's total for the stay, as given


def test_hostels_homestays_and_two_star_places_are_left_out():
    """Seen live (Kerala, 2026-10-12): rated 3.5+ and cheapest first, the
    list was student hostels and homestay rooms. A hostel has no star class."""
    raw = {
        "properties": [
            _prop("Hope Hostels, Munnar", 305, 2741, 4.8, hotel_class=None),
            _prop("Zostel Ooty", 792, 7128, 4.7, hotel_class="2-star hotel"),
            _prop("Ladies Homestay", 485, 4361, 5.0, hotel_class=None, type="vacation rental"),
            _prop("Townhouse Oak Thiruvallam", 1622, 14598, 4.2, hotel_class="3-star hotel"),
        ]
    }

    assert [h["name"] for h in normalize_hotels(raw)] == ["Townhouse Oak Thiruvallam"]


def test_a_hostel_or_homestay_filed_as_a_three_star_hotel_is_left_out():
    """Seen live in Kochi with the star filter on: Google files these as
    3-star hotels; their names, and the price of a dorm bed, say otherwise."""
    raw = {
        "properties": [
            _prop("The Hosteller Fort Kochi, Ocean Edge", 565, 2259, 4.7, "3-star hotel"),
            _prop("Juliet's Homestay Fort kochi", 709, 2835, 3.9, "3-star hotel"),
            _prop("Calvin's Inn Home Stay", 883, 3532, 4.6, "3-star hotel"),
            _prop("Hala Residency", 781, 3124, 4.0, "3-star hotel"),
        ]
    }

    assert [h["name"] for h in normalize_hotels(raw)] == ["Hala Residency"]


def test_a_rating_needs_enough_reviews_behind_it():
    raw = {
        "properties": [
            _prop("FabExpress Ascot", 1298, 5192, 4.2, "3-star hotel", reviews=19),
            _prop("Hotel Prime Palace", 1040, 4160, 3.7, "3-star hotel", reviews=1287),
        ]
    }

    assert [h["name"] for h in normalize_hotels(raw)] == ["Hotel Prime Palace"]


def test_normalize_caps_the_list(monkeypatch):
    many = {"properties": [_prop(f"Hotel {i}", 1000 + i, 3000 + i, 4.0) for i in range(9)]}

    assert len(normalize_hotels(many)) == google_hotels.MAX_OPTIONS


def test_normalized_hotels_fit_the_lodging_model():
    from app.models.itinerary import LodgingOption

    for hotel in normalize_hotels(SAMPLE):
        LodgingOption(**hotel)


@pytest.fixture
def searches(monkeypatch):
    monkeypatch.setenv("SERPAPI_HOTEL_API_KEY", "fake-hotel-key")
    script: list = []
    made: list = []

    async def fake_call_search(params, api_key):
        made.append((params, api_key))
        outcome = script.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return FakeMCPToolResult(outcome)

    monkeypatch.setattr(serpapi, "_call_search", fake_call_search)
    return script, made


def _search():
    return search_hotels_with_source("Kochi", date(2026, 10, 27), 3, 2)


def test_a_live_search_sends_the_stay_and_party_on_the_hotel_account(searches):
    script, made = searches
    script.append(SAMPLE)

    hotels, source = _search()

    params, key = made[0]
    assert key == "fake-hotel-key"  # the hotel account, not the flights one
    assert params["engine"] == "google_hotels"
    assert params["q"] == "Kochi"
    assert (params["check_in_date"], params["check_out_date"]) == ("2026-10-27", "2026-10-30")
    assert params["adults"] == "2" and params["currency"] == "INR"
    # 3 stars and up, rated 3.5+, cheapest first: asked of Google, not just
    # filtered afterwards, so the cheapest real hotels come back at all.
    assert (params["hotel_class"], params["rating"], params["sort_by"]) == ("3,4,5", "7", "3")
    assert source.status == "live" and len(hotels) == 3


def test_a_repeat_search_comes_from_the_cache(searches):
    script, made = searches
    script.append(SAMPLE)

    _search()
    hotels, source = _search()

    assert source.status == "cached" and len(made) == 1 and len(hotels) == 3


def test_a_failure_is_unavailable_with_a_google_hotels_link_never_sample_data(searches):
    script, _ = searches
    script.extend([ConnectionError("down")] * 3)

    hotels, source = _search()

    assert hotels == []
    assert source.status == "unavailable"
    assert source.link.startswith("https://www.google.com/travel/hotels?")
    assert "Kochi" in source.link


def test_an_exhausted_hotel_quota_opens_only_the_hotel_breaker(searches):
    from app.providers import google_flights

    script, made = searches
    script.append({"error": "Your account has run out of searches."})

    _search()
    _, second = _search()

    assert len(made) == 1
    assert "skipped after repeated failures" in second.detail
    assert not google_hotels.breaker.allow()
    assert google_flights.breaker.allow()  # a separate account, a separate breaker


def test_a_missing_hotel_key_is_unavailable(monkeypatch):
    monkeypatch.delenv("SERPAPI_HOTEL_API_KEY", raising=False)

    hotels, source = _search()

    assert hotels == []
    assert source.status == "unavailable"
    assert "SERPAPI_HOTEL_API_KEY is not set" in source.detail
