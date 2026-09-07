"""The terminal tool: how the agent hands back a finished itinerary.

The agent's prose is not the deliverable — a validated `Itinerary` is. Routing
the plan through a tool means Pydantic validates it while the agent is still in
the loop, so a malformed plan comes back as a correctable error instead of a
500 after the fact.
"""

import json

from pydantic import ValidationError

from app.models.itinerary import Itinerary, TripRequest

_SCHEMA_HINT = """\
{
  "destination": "City, Country",
  "start_date": "YYYY-MM-DD",
  "end_date": "YYYY-MM-DD",
  "travelers": 2,
  "currency": "USD",
  "total_estimated_cost": 2400.0,
  "notes": ["anything the traveller should know"],
  "days": [
    {
      "day": 1,
      "date": "YYYY-MM-DD",
      "summary": "one line",
      "activities": [
        {
          "time": "09:00",
          "title": "short title",
          "description": "what and why",
          "location": "where",
          "category": "food|sightseeing|transit|lodging|activity|rest",
          "estimated_cost_usd": 25.0
        }
      ]
    }
  ]
}"""


class ItinerarySink:
    """Catches the itinerary the agent submits during a run."""

    def __init__(self) -> None:
        self.itinerary: Itinerary | None = None

    @property
    def received(self) -> bool:
        return self.itinerary is not None


def build_submit_tool(sink: ItinerarySink, request: TripRequest):
    """Build the submit tool bound to one run's sink."""

    def submit_itinerary(itinerary_json: str) -> str:
        """Submit the finished itinerary. Call this exactly once, at the end.

        Args:
            itinerary_json: The complete itinerary as a JSON object matching the
                required schema: destination (string), start_date and end_date
                (YYYY-MM-DD), travelers (integer), currency, total_estimated_cost
                (number), notes (array of strings), and days (array of objects
                with day, date, summary, and activities; each activity has time,
                title, description, location, category, estimated_cost_usd).
        """
        try:
            payload = json.loads(itinerary_json)
        except json.JSONDecodeError as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": f"itinerary_json is not valid JSON: {exc}",
                    "expected_shape": _SCHEMA_HINT,
                }
            )

        try:
            itinerary = Itinerary.model_validate(payload)
        except ValidationError as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": "The itinerary did not validate. Fix these and resubmit.",
                    "problems": [
                        {"field": ".".join(str(p) for p in e["loc"]), "issue": e["msg"]}
                        for e in exc.errors()[:12]
                    ],
                    "expected_shape": _SCHEMA_HINT,
                }
            )

        if len(itinerary.days) != request.nights + 1:
            return json.dumps(
                {
                    "ok": False,
                    "error": (
                        f"The trip runs {request.start_date} to {request.end_date}, "
                        f"which is {request.nights + 1} days, but the itinerary has "
                        f"{len(itinerary.days)}. Resubmit with one entry per day."
                    ),
                }
            )

        sink.itinerary = itinerary
        return json.dumps(
            {"ok": True, "message": "Itinerary accepted. Now summarise it briefly."}
        )

    return submit_itinerary
