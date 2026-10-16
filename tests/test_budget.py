"""The budget check (app/agent/budget.py): plain arithmetic on what the plan
costs, against the budget the traveller gave."""

from datetime import date

import pytest
from conftest import sample_planned_trip

from app.agent.budget import build_budget_check_node, check_budget, money
from app.agent.state import initial_state
from app.models.itinerary import Activity, DayPlan, FlightLeg, HotelStay, LodgingOption


@pytest.mark.parametrize(
    ("amount", "shown"),
    [(999, "₹999"), (51027, "₹51,027"), (100000, "₹1,00,000"), (5102700, "₹51,02,700")],
)
def test_rupees_are_grouped_the_indian_way(amount, shown):
    assert money(amount) == shown


def _itinerary(trip_request, *, flights=(10_000, 18_000), hotel=6_000, per_day=1_000):
    trip = sample_planned_trip(
        trip_request,
        days=[
            DayPlan(
                day=i + 1,
                date=d,
                summary=f"Day {i + 1}",
                activities=[Activity(time="09:00", title="Walk", estimated_cost=per_day)],
            )
            for i, d in enumerate(trip_request.dates)
        ],
    )
    it = trip.itinerary
    if flights:
        it.outbound_flight = FlightLeg(
            carrier="IndiGo", origin="VNS", destination="COK",
            depart_date=trip_request.start_date, total=flights[0],
        )
        it.return_flight = FlightLeg(
            carrier="IndiGo", origin="TRV", destination="VNS",
            depart_date=trip_request.end_date, total=flights[1],
        )
    if hotel:
        it.stays = [
            HotelStay(
                city="Kochi", check_in=trip_request.start_date, nights=trip_request.nights,
                options=[LodgingOption(name="Hala Residency", total=hotel)],
            )
        ]
    return it


def test_no_budget_no_check(trip_request):
    request = trip_request.model_copy(update={"budget": None})

    assert check_budget(request, _itinerary(request)) is None


def test_within_budget_says_what_is_left(trip_request):
    request = trip_request.model_copy(update={"budget": 60_000})

    check = check_budget(request, _itinerary(request))

    # 28,000 flights + 6,000 hotel + 4 days x 1,000 activities.
    assert (check.total, check.within) == (38_000, True)
    assert check.message.startswith("Within your ₹60,000 budget: about ₹38,000")
    assert "leaving ₹22,000" in check.message
    assert check.message.endswith("Activity costs are estimates.")


def test_over_budget_says_by_how_much_and_what_it_is_made_of(trip_request):
    request = trip_request.model_copy(update={"budget": 30_000})

    check = check_budget(request, _itinerary(request))

    assert check.within is False
    assert check.message.startswith("Over your ₹30,000 budget by ₹8,000: about ₹38,000")
    assert "flights ₹28,000, hotels ₹6,000, activities about ₹4,000" in check.message


def test_when_flights_and_hotels_alone_break_the_budget_it_says_so(trip_request):
    request = trip_request.model_copy(update={"budget": 20_000})

    check = check_budget(request, _itinerary(request))

    assert "Flights and hotels alone come to ₹34,000, more than the whole budget." in (
        check.message
    )


def test_a_total_without_flights_says_they_are_missing(trip_request):
    request = trip_request.model_copy(update={"budget": 60_000, "include_flights": False})

    check = check_budget(request, _itinerary(request, flights=None))

    assert check.missing == ["flights"]
    assert "Not included: flights, so the trip will cost more." in check.message


def test_hotels_searched_but_none_found_are_called_that(trip_request):
    request = trip_request.model_copy(update={"budget": 60_000})

    check = check_budget(request, _itinerary(request, hotel=None))

    assert check.missing == ["hotels (none found)"]


def test_the_step_puts_the_check_on_the_itinerary(trip_request):
    request = trip_request.model_copy(update={"budget": 30_000})
    node = build_budget_check_node()

    result = node(initial_state(request) | {"itinerary": _itinerary(request)})

    assert result["itinerary"].budget_check.within is False
    assert result["messages"] == ["budget_check: over (Over by ₹8,000)"]


def test_the_step_does_nothing_without_a_budget(trip_request):
    request = trip_request.model_copy(update={"budget": None})
    node = build_budget_check_node()

    assert node(initial_state(request) | {"itinerary": _itinerary(request)}) == {}


def test_dates_in_this_file_line_up_with_the_fixture(trip_request):
    """The arithmetic above assumes the shared fixture's 4-day trip."""
    assert trip_request.dates[0] == date(2026, 4, 10) and len(trip_request.dates) == 4
