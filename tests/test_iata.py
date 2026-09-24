"""Place-name resolution against small fake directories shaped like
Travelpayouts' real cities.json / airports.json — no network."""

import pytest

from app.providers import iata
from app.providers.iata import airports_for, resolve_iata, resolve_route

CITIES = [
    {"name": "New Delhi", "code": "DEL", "country_code": "IN", "has_flightable_airport": True},
    {"name": "Mumbai", "code": "BOM", "country_code": "IN", "has_flightable_airport": True},
    {"name": "Delhi", "code": "XXX", "country_code": "US", "has_flightable_airport": False},
    {"name": "Lisbon", "code": "LIS", "country_code": "PT", "has_flightable_airport": True},
    # Both real and flightable; Japan's comes first in the live directory too.
    {"name": "Kochi", "code": "KCZ", "country_code": "JP", "has_flightable_airport": True},
    {"name": "Kochi", "code": "COK", "country_code": "IN", "has_flightable_airport": True},
    {"name": "Varanasi", "code": "VNS", "country_code": "IN", "has_flightable_airport": True},
    {"name": "Tokyo", "code": "TYO", "country_code": "JP", "has_flightable_airport": True},
]

AIRPORTS = [
    {"code": "LMJ", "city_code": "TYO", "iata_type": "bus", "flightable": True},
    {"code": "NRT", "city_code": "TYO", "iata_type": "airport", "flightable": True},
    {"code": "HND", "city_code": "TYO", "iata_type": "airport", "flightable": True},
    {"code": "QAH", "city_code": "DEL", "iata_type": "airport", "flightable": True},
    {"code": "DEL", "city_code": "DEL", "iata_type": "airport", "flightable": True},
    {"code": "BQH", "city_code": "LON", "iata_type": "airport", "flightable": False},
    {"code": "LHR", "city_code": "LON", "iata_type": "airport", "flightable": True},
    {"code": "XQE", "city_code": "LON", "iata_type": "railway", "flightable": True},
]


@pytest.fixture(autouse=True)
def _fake_directories(monkeypatch):
    """Lookups are lru_cached, so caches are cleared around each test or the
    first one's data leaks into the rest."""
    iata._matches.cache_clear()
    airports_for.cache_clear()
    monkeypatch.setattr(iata, "_cities", lambda *a, **k: CITIES)
    monkeypatch.setattr(iata, "_airports", lambda *a, **k: AIRPORTS)
    yield
    iata._matches.cache_clear()
    airports_for.cache_clear()


# --- resolve_iata / resolve_route --------------------------------------------


def test_resolves_an_exact_city_name():
    assert resolve_iata("Mumbai") == "BOM"


def test_ignores_anything_after_a_comma():
    assert resolve_iata("Lisbon, Portugal") == "LIS"


def test_prefers_a_flightable_airport_over_an_exact_name_match():
    """"Delhi" matches a non-flightable city exactly and New Delhi partially;
    a city with no flightable airport can't be searched from, so it loses."""
    assert resolve_iata("Delhi") == "DEL"


def test_unknown_place_resolves_to_none():
    assert resolve_iata("Atlantis") is None


def test_a_shared_name_resolves_to_the_same_country_as_the_other_end():
    """A real Varanasi -> Kochi search went to Kochi, Japan and found nothing."""
    assert resolve_route("Varanasi", "Kochi") == ("VNS", "COK")


def test_the_same_country_rule_works_from_either_end():
    assert resolve_route("Kochi", "Varanasi") == ("COK", "VNS")


def test_a_shared_name_still_goes_abroad_when_the_other_end_is_abroad():
    assert resolve_route("Tokyo", "Kochi") == ("TYO", "KCZ")


def test_a_route_with_no_shared_country_keeps_each_best_match():
    assert resolve_route("Mumbai", "Lisbon") == ("BOM", "LIS")


def test_a_route_with_an_unknown_end_resolves_that_end_to_none():
    assert resolve_route("Atlantis", "Mumbai") == (None, "BOM")


# --- airports_for ----------------------------------------------------------


def test_a_multi_airport_city_expands_to_its_airports_only():
    """TYO isn't an airport; Google Flights needs NRT/HND. The bus terminal
    in the same city code is excluded."""
    assert set(airports_for("TYO")) == {"NRT", "HND"}


def test_the_citys_own_code_comes_first_when_it_is_an_airport():
    assert airports_for("DEL") == ("DEL", "QAH")


def test_railway_stations_and_non_flightable_airports_are_excluded():
    assert airports_for("LON") == ("LHR",)


def test_a_city_the_directory_does_not_list_falls_back_to_its_own_code():
    assert airports_for("VNS") == ("VNS",)
