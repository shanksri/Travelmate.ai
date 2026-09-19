"""Trips stored before the switch to rupees used `*_usd` field names and
`currency: "USD"`. They must still load, with their amounts left exactly as
they were — relabelling dollars as rupees would silently multiply every price
by about 85. These tests pin that down."""

from app.models.itinerary import (
    Activity,
    FlightLeg,
    Itinerary,
    LodgingOption,
    PlannedTrip,
    TripRequest,
)

LEGACY_TRIP = {
    "id": "old1",
    "request": {
        "start_date": "2026-04-10",
        "end_date": "2026-04-13",
        "travelers": 2,
        "destination": "Lisbon, Portugal",
        "budget_usd": 3000,
    },
    "itinerary": {
        "destination": "Lisbon, Portugal",
        "start_date": "2026-04-10",
        "end_date": "2026-04-13",
        "travelers": 2,
        "currency": "USD",
        "outbound_flight": {
            "carrier": "Meridian Air",
            "origin": "BOS",
            "destination": "LIS",
            "depart_date": "2026-04-10",
            "price_usd_per_person": 250.0,
            "total_usd": 500.0,
        },
        "lodging_options": [
            {"name": "Casa Vista Hostel", "nightly_usd": 100.0, "total_usd": 300.0}
        ],
        "days": [
            {
                "day": 1,
                "date": "2026-04-10",
                "summary": "Day 1",
                "activities": [
                    {"time": "09:00", "title": "Walk", "estimated_cost_usd": 20.0}
                ],
            }
        ],
        "total_estimated_cost": 820.0,
    },
    "summary": "A short stay.",
}


def test_legacy_amounts_are_not_reinterpreted_as_rupees():
    trip = PlannedTrip.model_validate(LEGACY_TRIP)

    assert trip.itinerary.outbound_flight.total == 500.0
    assert trip.itinerary.lodging_options[0].nightly == 100.0
    assert trip.itinerary.days[0].activities[0].estimated_cost == 20.0
    assert trip.request.budget == 3000


def test_a_legacy_trip_still_says_it_is_in_dollars():
    trip = PlannedTrip.model_validate(LEGACY_TRIP)

    assert trip.itinerary.currency == "USD"


def test_a_new_trip_defaults_to_rupees():
    itinerary = Itinerary(
        destination="Jaipur, India",
        start_date="2026-04-10",
        end_date="2026-04-13",
        travelers=2,
        days=[],
    )

    assert itinerary.currency == "INR"


def test_each_model_maps_its_own_legacy_names():
    assert TripRequest(
        start_date="2026-04-10", end_date="2026-04-13", budget_usd=1000
    ).budget == 1000
    assert Activity(time="09:00", title="x", estimated_cost_usd=20).estimated_cost == 20
    assert LodgingOption(name="x", nightly_usd=100, total_usd=300).nightly == 100
    assert FlightLeg(
        carrier="x",
        origin="a",
        destination="b",
        depart_date="2026-04-10",
        price_usd_per_person=250,
        total_usd=500,
    ).price_per_person == 250


def test_a_current_payload_is_untouched_by_the_compat_shim():
    """The rename only fills a new key that isn't already there, so a payload
    already using the new names can't be clobbered by a stale old one."""
    lodging = LodgingOption.model_validate(
        {"name": "x", "nightly": 8000, "nightly_usd": 100, "total": 24000}
    )

    assert lodging.nightly == 8000
