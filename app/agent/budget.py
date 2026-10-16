"""The budget check: the plan's cost against the traveller's stated budget.

Plain arithmetic, deliberately not an LLM. Before this, the only check was a
line in the itinerary prompt asking the model to "note honestly" if costs
threatened the budget, and nothing compared the numbers: a plan could come
back ₹21,000 over budget with no mention of it.

The message says what the total is made of, because the parts aren't equally
certain: flights and hotels are prices from a search, activities are the
itinerary agent's estimates. It also says what the total leaves out (flights
not requested, or none found), since a total that silently omits the flight
looks cheaper than the trip will be.
"""

from app.agent.events import status
from app.agent.state import TravelState
from app.models.itinerary import BudgetCheck, Itinerary, TripRequest


def money(amount: float, currency: str = "INR") -> str:
    """₹51,02,700 in Indian digit grouping; other currencies plainly."""
    if currency != "INR":
        return f"{currency} {amount:,.0f}"
    digits = str(round(abs(amount)))
    if len(digits) > 3:
        head, groups = digits[:-3], [digits[-3:]]
        while len(head) > 2:
            groups.insert(0, head[-2:])
            head = head[:-2]
        digits = ",".join([head, *groups] if head else groups)
    return ("-" if amount < 0 else "") + "₹" + digits


def check_budget(request: TripRequest, itinerary: Itinerary) -> BudgetCheck | None:
    """None when no budget was given."""
    if request.budget is None:
        return None

    budget = request.budget
    flights = sum(
        leg.total or 0
        for leg in (itinerary.outbound_flight, itinerary.return_flight)
        if leg is not None
    )
    hotels = itinerary.hotel_cost
    activities = sum(
        activity.estimated_cost or 0 for day in itinerary.days for activity in day.activities
    )
    total = flights + hotels + activities
    within = total <= budget

    missing = []
    if not (itinerary.outbound_flight and itinerary.return_flight):
        missing.append("flights" if not request.include_flights else "some flights (none found)")
    if not any(stay.booked for stay in itinerary.stays):
        missing.append("hotels" if not request.include_hotels else "hotels (none found)")

    def m(amount: float) -> str:
        return money(amount, itinerary.currency)

    parts = f"flights {m(flights)}, hotels {m(hotels)}, activities about {m(activities)}"
    if within:
        message = (
            f"Within your {m(budget)} budget: about {m(total)} ({parts}), "
            f"leaving {m(budget - total)}."
        )
    else:
        message = (
            f"Over your {m(budget)} budget by {m(total - budget)}: "
            f"about {m(total)} ({parts})."
        )
        if flights + hotels > budget:
            message += (
                f" Flights and hotels alone come to {m(flights + hotels)}, "
                "more than the whole budget."
            )
    if missing:
        message += f" Not included: {' and '.join(missing)}, so the trip will cost more."
    message += " Activity costs are estimates."

    return BudgetCheck(
        budget=budget,
        total=total,
        flights=flights,
        hotels=hotels,
        activities=activities,
        within=within,
        missing=missing,
        message=message,
    )


def build_budget_check_node():
    """The trip graph's budget step, between the itinerary and the summary,
    so the summary is written knowing whether the plan fits."""

    def node(state: TravelState) -> dict:
        itinerary = state.get("itinerary")
        if itinerary is None:
            return {}
        check = check_budget(state["request"], itinerary)
        if check is None:
            status("budget", "No budget given", state="skipped")
            return {}

        currency = itinerary.currency
        if check.within:
            line = f"{money(check.total, currency)} of {money(check.budget, currency)}"
        else:
            line = f"Over by {money(check.total - check.budget, currency)}"
        status("budget", line, state="done" if check.within else "warn")
        return {
            "itinerary": itinerary.model_copy(update={"budget_check": check}),
            "messages": [f"budget_check: {'within' if check.within else 'over'} ({line})"],
        }

    return node
