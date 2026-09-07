"""Deterministic stand-in travel data.

Seeded from the query itself, so the same question always yields the same
numbers — which keeps agent behaviour reproducible in tests and demos. The
prices are plausible fiction, not quotes.
"""

import hashlib
from datetime import date
from typing import Any


def _destination(name: str, tags: list[str], base_daily_usd: int) -> dict[str, Any]:
    return {"name": name, "tags": tags, "base_daily_usd": base_daily_usd}


_DESTINATIONS: list[dict[str, Any]] = [
    _destination("Lisbon, Portugal", ["food", "history", "coast", "budget"], 110),
    _destination("Kyoto, Japan", ["culture", "history", "food", "temples"], 165),
    _destination("Reykjavik, Iceland", ["nature", "hiking", "adventure"], 230),
    _destination("Mexico City, Mexico", ["food", "art", "nightlife", "budget"], 95),
    _destination("Queenstown, New Zealand", ["adventure", "hiking", "nature"], 190),
    _destination("Rome, Italy", ["history", "food", "art"], 150),
    _destination("Chiang Mai, Thailand", ["food", "temples", "budget", "nature"], 70),
    _destination("Barcelona, Spain", ["beach", "art", "food", "nightlife"], 140),
]

_ATTRACTION_KINDS = [
    ("old town walking loop", "sightseeing", 0, 120),
    ("central food market", "food", 25, 90),
    ("national museum", "sightseeing", 18, 150),
    ("hilltop viewpoint at sunset", "sightseeing", 0, 90),
    ("half-day cooking class", "activity", 75, 240),
    ("coastal or river day trip", "activity", 60, 420),
    ("neighbourhood cafe crawl", "food", 30, 120),
    ("botanical gardens", "sightseeing", 12, 100),
    ("live music venue", "activity", 25, 150),
    ("guided history tour", "activity", 40, 180),
]

_CONDITIONS = [
    "mild and dry",
    "warm and humid",
    "cool with showers",
    "cold and clear",
    "hot and sunny",
]

_BUDGET_CAPS = {"low": 120, "medium": 180, "high": 10_000}


def _seed(*parts: object) -> int:
    """A stable integer derived from the query, so results never drift."""
    raw = "|".join(str(p) for p in parts).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def _jitter(seed: int, low: float, high: float) -> float:
    return round(low + (seed % 1000) / 1000 * (high - low), 2)


class MockTravelProvider:
    def search_destinations(
        self, interests: list[str], month: int, budget_level: str
    ) -> list[dict[str, Any]]:
        wanted = {i.strip().lower() for i in interests}
        cap = _BUDGET_CAPS.get(budget_level, 10_000)

        scored = []
        for dest in _DESTINATIONS:
            if dest["base_daily_usd"] > cap:
                continue
            tags = set(dest["tags"])
            matched = sorted(wanted & tags)
            # A shoulder-season nudge, so the month actually moves the ranking.
            seasonal = 1 if month in (4, 5, 9, 10) else 0
            scored.append(
                {
                    "name": dest["name"],
                    "match_score": len(matched) * 2 + seasonal,
                    "matched_interests": matched,
                    "estimated_daily_cost_usd": dest["base_daily_usd"],
                    "known_for": dest["tags"][:3],
                }
            )
        scored.sort(key=lambda d: (-d["match_score"], d["estimated_daily_cost_usd"]))
        return scored[:5]

    def search_flights(
        self, origin: str, destination: str, depart: date, travelers: int
    ) -> list[dict[str, Any]]:
        base = _seed(origin, destination, depart.isoformat())
        options = []
        for i, (carrier, stops) in enumerate(
            [("Meridian Air", 0), ("Northwind", 1), ("Cadence Airways", 1)]
        ):
            price = max(_jitter(base + i, 180, 950) - stops * 40, 90.0)
            options.append(
                {
                    "carrier": carrier,
                    "origin": origin,
                    "destination": destination,
                    "depart_date": depart.isoformat(),
                    "stops": stops,
                    "duration_hours": round(4 + stops * 3.5 + (base + i) % 7, 1),
                    "price_usd_per_person": round(price, 2),
                    "total_usd": round(price * travelers, 2),
                }
            )
        return sorted(options, key=lambda o: o["total_usd"])

    def search_lodging(
        self, destination: str, check_in: date, nights: int, travelers: int
    ) -> list[dict[str, Any]]:
        base = _seed(destination, check_in.isoformat(), nights)
        rooms = max(1, (travelers + 1) // 2)
        city = destination.split(",")[0].strip()
        options = []
        for i, (name, tier, rating) in enumerate(
            [
                ("Casa Vista Hostel", "budget", 8.4),
                ("Hotel Meridiana", "midrange", 8.9),
                ("The Ardent House", "boutique", 9.3),
            ]
        ):
            nightly = _jitter(base + i * 17, 35, 340) + i * 45
            options.append(
                {
                    "name": name,
                    "tier": tier,
                    "rating": rating,
                    "neighbourhood": f"central {city}",
                    "nightly_usd": round(nightly, 2),
                    "rooms_needed": rooms,
                    "total_usd": round(nightly * nights * rooms, 2),
                }
            )
        return sorted(options, key=lambda o: o["total_usd"])

    def search_attractions(
        self, destination: str, interests: list[str], limit: int
    ) -> list[dict[str, Any]]:
        base = _seed(destination, ",".join(sorted(interests)))
        city = destination.split(",")[0].strip()
        results = []
        for i, (kind, category, cost, minutes) in enumerate(_ATTRACTION_KINDS):
            results.append(
                {
                    "name": f"{city} {kind}",
                    "category": category,
                    "typical_cost_usd": cost,
                    "duration_minutes": minutes,
                    "rating": round(7.5 + ((base + i) % 25) / 10, 1),
                }
            )
        results.sort(key=lambda a: -a["rating"])
        return results[: max(1, min(limit, len(results)))]

    def get_weather_outlook(self, destination: str, when: date) -> dict[str, Any]:
        base = _seed(destination, when.month)
        high = round(8 + (base % 26), 1)
        return {
            "destination": destination,
            "month": when.strftime("%B"),
            "typical_high_c": high,
            "typical_low_c": round(high - 6 - (base % 5), 1),
            "conditions": _CONDITIONS[base % len(_CONDITIONS)],
            "rain_days_per_month": base % 15,
            "note": "Climate outlook for the time of year, not a forecast.",
        }
