"""Plan one trip from the command line and print it.

    python scripts/plan_trip.py --destination "Kyoto, Japan" \
        --start 2026-04-10 --end 2026-04-15 --travelers 2 \
        --budget 4000 --interests food,history

Needs OPENAI_API_KEY. Travel data comes from the mock provider unless
TRAVELMATE_PROVIDER says otherwise.
"""

import argparse
import json
import sys
from datetime import date

import openai

from app.agent.planner import PlanningError, plan_trip
from app.core.logging import configure_logging
from app.models.itinerary import PlannedTrip, TripRequest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plan a trip with TravelMate.")
    parser.add_argument("--start", required=True, type=date.fromisoformat)
    parser.add_argument("--end", required=True, type=date.fromisoformat)
    parser.add_argument("--destination", help="Omit to let the agent choose one.")
    parser.add_argument("--origin")
    parser.add_argument("--travelers", type=int, default=1)
    parser.add_argument("--budget", type=float, help="Total USD for the party.")
    parser.add_argument("--interests", default="", help="Comma-separated.")
    parser.add_argument(
        "--pace", choices=["relaxed", "balanced", "packed"], default="balanced"
    )
    parser.add_argument("--notes")
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead.")
    return parser.parse_args(argv)


def render(trip: PlannedTrip) -> str:
    it = trip.itinerary
    lines = [
        f"{it.destination} — {it.start_date} to {it.end_date}, "
        f"{it.travelers} traveller(s)",
    ]
    if it.total_estimated_cost is not None:
        lines.append(f"Estimated total: {it.currency} {it.total_estimated_cost:,.0f}")
    lines.append("")

    for day in it.days:
        lines.append(f"Day {day.day} — {day.date}: {day.summary}")
        for act in day.activities:
            cost = (
                f"  (~${act.estimated_cost_usd:,.0f})"
                if act.estimated_cost_usd
                else ""
            )
            where = f" @ {act.location}" if act.location else ""
            lines.append(f"  {act.time}  {act.title}{where}{cost}")
        lines.append("")

    if it.notes:
        lines.append("Notes:")
        lines.extend(f"  - {note}" for note in it.notes)
        lines.append("")
    if trip.summary:
        lines.append(trip.summary)
    lines.append("")
    lines.append(f"[trip {trip.id} · {len(trip.agent_trace)} agent step(s)]")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv)

    request = TripRequest(
        start_date=args.start,
        end_date=args.end,
        destination=args.destination,
        origin=args.origin,
        travelers=args.travelers,
        budget_usd=args.budget,
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
