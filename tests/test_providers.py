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
    from app.providers.live import LiveTravelProvider

    real = [{"carrier": "IX", "total": 6899.0}]
    monkeypatch.setattr("app.providers.live.search_real_flights", lambda *a, **k: real)

    assert LiveTravelProvider().search_flights("Delhi", "Mumbai", date(2026, 10, 10), 1) == real


def test_live_provider_returns_no_flights_rather_than_mock_ones_on_failure(monkeypatch):
    """Invented flights shown alongside real ones would be indistinguishable
    from them, so an API failure means no flights, not fictional ones."""
    from app.providers.google_flights import GoogleFlightsError
    from app.providers.live import LiveTravelProvider

    def boom(*args, **kwargs):
        raise GoogleFlightsError("SERPAPI_API_KEY is not set")

    monkeypatch.setattr("app.providers.live.search_real_flights", boom)

    assert LiveTravelProvider().search_flights("Delhi", "Mumbai", date(2026, 10, 10), 1) == []


def test_live_provider_keeps_mock_data_for_everything_but_flights():
    from app.providers.live import LiveTravelProvider

    live, mock = LiveTravelProvider(), MockTravelProvider()

    assert live.search_lodging("Lisbon", date(2026, 4, 10), 3, 2) == mock.search_lodging(
        "Lisbon", date(2026, 4, 10), 3, 2
    )
