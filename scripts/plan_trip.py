"""Plan one trip from the command line and print it.

    python scripts/plan_trip.py --destination "Kyoto, Japan" \
        --start 2026-04-10 --end 2026-04-15 --travelers 2 \
        --budget 300000 --interests food,history

Money is in Indian rupees throughout.

Needs OPENAI_API_KEY. Travel data comes from the mock provider unless
TRAVELMATE_PROVIDER says otherwise.
"""

import argparse
import json
import sys
from datetime import date

import openai
from dotenv import load_dotenv

from app.agent.planner import PlanningError, plan_trip
from app.core.logging import configure_logging
from app.models.itinerary import FlightLeg, PlannedTrip, TripRequest

# So OPENAI_API_KEY (and friends) in .env reach this process without having
# to export them by hand — app/core/config.py only loads .env for TRAVELMATE_*
# settings, not for keys the SDKs read directly from the environment.
load_dotenv()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a trip with TravelMate.")
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--destination", help="Omit to let the agent choose one.")
    parser.add_argument("--origin")
    parser.add_argument("--travelers", type=int, default=1)
    parser.add_argument("--budget", type=float, help="Total INR for the whole party.")
    parser.add_argument("--interests", default="", help="Comma-separated.")
    parser.add_argument(
        "--pace", choices=["relaxed", "balanced", "packed"], default="balanced"
    )
    parser.add_argument("--notes")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead.")
    return parser.parse_args(argv)


def _money(amount: float | None, currency: str = "INR") -> str:
    """Rupees by default; a trip stored before the switch says so itself."""
    if amount is None:
        return ""
    symbol = "₹" if currency == "INR" else "$"
    return f"{symbol}{amount:,.0f}"


def _render_flight(flight: FlightLeg, currency: str = "INR") -> str:
    stops = "nonstop" if flight.stops == 0 else f"{flight.stops} stop(s)"
    price = f", {_money(flight.total, currency)} total" if flight.total is not None else ""
    return (
        f"{flight.carrier} — {flight.origin} -> {flight.destination}, "
        f"{flight.depart_date} ({stops}{price})"
    )


def render(trip: PlannedTrip) -> str:
    it = trip.itinerary
    lines = [
        f"{it.destination} — {it.start_date} to {it.end_date}, "
        f"{it.travelers} traveller(s)",
    ]
    if it.total_estimated_cost is not None:
        lines.append(f"Estimated total: {_money(it.total_estimated_cost, it.currency)}")
    lines.append("")

    if trip.summary:
        lines.append(trip.summary)
        lines.append("")

    lines.append("Flights:")
    if it.outbound_flight:
        lines.append(f"  Outbound: {_render_flight(it.outbound_flight, it.currency)}")
    if it.return_flight:
        lines.append(f"  Return:   {_render_flight(it.return_flight, it.currency)}")
    if not it.outbound_flight and not it.return_flight:
        lines.append("  (none — no origin was given, or no fares were found)")
    lines.append("")

    if it.lodging_options:
        lines.append("Where to stay (* = selected):")
        for i, hotel in enumerate(it.lodging_options):
            marker = "*" if i == 0 else " "
            bits = [hotel.tier] if hotel.tier else []
            if hotel.rating is not None:
                bits.append(f"{hotel.rating} rating")
            tag = f" ({', '.join(bits)})" if bits else ""
            cost = (
                f" — {_money(hotel.nightly, it.currency)}/night, "
                f"{_money(hotel.total, it.currency)} total"
                if hotel.nightly is not None and hotel.total is not None
                else ""
            )
            lines.append(f"  {marker} {hotel.name}{tag}{cost}")
        lines.append("")

    lines.append("Day by day:")
    for day in it.days:
        lines.append(f"Day {day.day} — {day.date}: {day.summary}")
        for act in day.activities:
            where = f" @ {act.location}" if act.location else ""
            lines.append(f"  {act.time}  {act.title}{where}")
            if act.description:
                lines.append(f"          {act.description}")
        lines.append("")

    lines.append(f"[trip {trip.id} · {len(trip.agent_trace)} agent step(s)]")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    # render() uses Unicode (em dashes, middle dots) that isn't in every
    # Windows console codepage — force UTF-8 stdout so printing it can't
    # crash with a UnicodeEncodeError regardless of the terminal's own
    # settings.
    sys.stdout.reconfigure(encoding="utf-8")
    configure_logging()
    args = parse_args(argv)

    request = TripRequest(
        start_date=args.start,
        end_date=args.end,
        destination=args.destination,
        origin=args.origin,
        travelers=args.travelers,
        budget=args.budget,
        interests=[i.strip() for i in args.interests.split(",") if i.strip()],
        pace=args.pace,
        notes=args.notes,
    )

    try:
        trip = plan_trip(request)
    except openai.AuthenticationError:
        print("OPENAI_API_KEY is missing or invalid.", file=sys.stderr)
        return 2
    except (PlanningError, openai.APIError) as exc:
        print(f"Planning failed: {exc}", file=sys.stderr)
        return 1

    print(json.dumps(trip.model_dump(mode="json"), indent=2) if args.json else render(trip))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
