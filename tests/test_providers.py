from datetime import date

from app.providers.mock import MockTravelProvider


def test_destination_search_ranks_by_interest_overlap(provider):
    results = provider.search_destinations(
        interests=["food", "history"], month=4, budget_level="medium"
    )
    assert results
    assert results[0]["match_score"] >= results[-1]["match_score"]
    assert "food" in results[0]["matched_interests"]


def test_budget_level_filters_expensive_destinations(provider):
    cheap = provider.search_destinations(["nature"], month=6, budget_level="low")
    assert all(d["estimated_daily_cost"] <= 10_000 for d in cheap)


def test_results_are_deterministic():
    a = MockTravelProvider().search_flights("BOS", "LIS", date(2026, 4, 10), 2)
    b = MockTravelProvider().search_flights("BOS", "LIS", date(2026, 4, 10), 2)
    assert a == b


def test_flight_total_scales_with_party_size(provider):
    solo = provider.search_flights("BOS", "LIS", date(2026, 4, 10), 1)[0]
    pair = provider.search_flights("BOS", "LIS", date(2026, 4, 10), 2)[0]
    assert pair["total"] == round(solo["price_per_person"] * 2, 2)


def test_lodging_is_sorted_by_total_cost(provider):
    options = provider.search_lodging("Lisbon, Portugal", date(2026, 4, 10), 3, 2)
    assert options == sorted(options, key=lambda o: o["total"])


def test_attraction_limit_is_clamped(provider):
    assert len(provider.search_attractions("Rome, Italy", ["food"], limit=3)) == 3
    assert len(provider.search_attractions("Rome, Italy", ["food"], limit=99)) <= 10


# --- LiveTravelProvider ------------------------------------------------------


def test_live_provider_serves_real_flights(monkeypatch):
    from app.models.itinerary import DataSource
    from app.providers.live import LiveTravelProvider

    real = [{"carrier": "IX", "total": 6899.0}]
    live_source = DataSource(status="live", provider="Google Flights")
    monkeypatch.setattr(
        "app.providers.live.search_flights_with_source", lambda *a, **k: (real, live_source)
    )
    provider = LiveTravelProvider()

    assert provider.search_flights("Delhi", "Mumbai", date(2026, 10, 10), 1) == real
    assert provider.search_flights_with_source("Delhi", "Mumbai", date(2026, 10, 10), 1) == (
        real,
        live_source,
    )


def test_live_provider_returns_no_flights_rather_than_mock_ones_on_failure(monkeypatch):
    """Invented flights shown alongside real ones would be indistinguishable
    from them, so a failed search means no flights, not fictional ones —
    labelled unavailable, with a link to search for them."""
    from app.providers import serpapi
    from app.providers.live import LiveTravelProvider

    monkeypatch.setenv("SERPAPI_FLIGHTS_API_KEY", "fake")
    monkeypatch.setattr(
        "app.providers.google_flights.resolve_route", lambda origin, dest: ("DEL", "BOM")
    )
    monkeypatch.setattr("app.providers.google_flights.airports_for", lambda code: [code])

    async def down(params, api_key):
        raise ConnectionError("network down")

    monkeypatch.setattr(serpapi, "_call_search", down)

    flights, source = LiveTravelProvider().search_flights_with_source(
        "Delhi", "Mumbai", date(2026, 10, 10), 1
    )

    assert flights == []
    assert source.status == "unavailable"
    assert "network down" in source.detail
    assert source.link.startswith("https://www.google.com/travel/flights?")


def test_live_provider_keeps_mock_data_for_everything_but_flights_and_hotels():
    from app.providers.live import LiveTravelProvider

    live, mock = LiveTravelProvider(), MockTravelProvider()

    assert live.search_attractions("Lisbon", ["food"], 5) == mock.search_attractions(
        "Lisbon", ["food"], 5
    )
    assert live.get_weather_outlook("Lisbon", date(2026, 4, 10)) == mock.get_weather_outlook(
        "Lisbon", date(2026, 4, 10)
    )


def test_live_provider_serves_real_hotels(monkeypatch):
    from app.models.itinerary import DataSource
    from app.providers.live import LiveTravelProvider

    real = [{"name": "Grand Hyatt Kochi Bolgatty", "total": 57071.0}]
    source = DataSource(status="live", provider="Google Hotels")
    monkeypatch.setattr(
        "app.providers.live.search_hotels_with_source", lambda *a, **k: (real, source)
    )

    assert LiveTravelProvider().search_lodging_with_source(
        "Kochi", date(2026, 10, 27), 3, 2
    ) == (real, source)


def test_the_mock_provider_labels_its_results_as_sample():
    flights, flight_source = MockTravelProvider().search_flights_with_source(
        "Delhi", "Goa", date(2026, 10, 10), 1
    )
    hotels, hotel_source = MockTravelProvider().search_lodging_with_source(
        "Goa", date(2026, 10, 10), 3, 1
    )

    assert flights and hotels
    assert flight_source.status == hotel_source.status == "sample"
