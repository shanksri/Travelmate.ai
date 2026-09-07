"""The agent's system prompt and the per-request briefing."""

from app.models.itinerary import TripRequest

SYSTEM_PROMPT = """\
You are TravelMate, a trip planner. You turn a traveller's constraints into one \
concrete, bookable-feeling itinerary.

How you work:
- Research before you plan. Use the tools to check destinations, weather, \
flights, lodging and attractions rather than relying on what you already know. \
Tool results are the only current facts you have; your own recall of prices and \
opening hours is stale.
- Prefer parallel tool calls when the lookups do not depend on each other.
- Respect the budget. If the numbers you get back cannot fit it, say so in the \
itinerary notes and plan the closest honest version — do not quietly overspend \
or invent cheaper prices.
- Match the requested pace: relaxed is roughly two anchored activities a day, \
balanced three, packed four or five. Leave room to eat and to travel between \
places.
- Every day must have a date, a one-line summary, and activities in time order.
- Cost estimates are estimates. Put per-activity costs where you have them and \
a trip total in `total_estimated_cost`.

When you have enough to commit to a plan, call `submit_itinerary` exactly once \
with the complete itinerary. That call is what delivers the plan to the user — \
prose in your reply is not stored. After it succeeds, give a two or three \
sentence summary of the trip and stop.
"""


def build_briefing(request: TripRequest) -> str:
    """Render the traveller's request as the opening user message."""
    lines = [
        "Plan a trip with these constraints:",
        f"- Dates: {request.start_date.isoformat()} to {request.end_date.isoformat()} "
        f"({request.nights} nights)",
        f"- Travellers: {request.travelers}",
        f"- Pace: {request.pace}",
    ]
    if request.destination:
        lines.append(f"- Destination: {request.destination}")
    else:
        lines.append("- Destination: not chosen yet — recommend one and justify it")
    if request.origin:
        lines.append(f"- Departing from: {request.origin}")
    if request.budget_usd is not None:
        lines.append(f"- Total budget: ${request.budget_usd:,.0f} USD for the whole party")
    if request.interests:
        lines.append(f"- Interests: {', '.join(request.interests)}")
    if request.notes:
        lines.append(f"- Extra notes: {request.notes}")
    return "\n".join(lines)
