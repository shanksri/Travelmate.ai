"""Deterministic stand-in travel data.

Seeded from the query itself, so the same question always yields the same
numbers — which keeps agent behaviour reproducible in tests and demos. The
prices are plausible fiction, not quotes, and they are **rupees** — not
dollars converted at some rate, just figures chosen to be realistic in INR.
"""

import hashlib
from datetime import date
from typing import Any


def _destination(name: str, tags: list[str], base_daily: int) -> dict[str, Any]:
    return {"name": name, "tags": tags, "base_daily": base_daily}


_DESTINATIONS: list[dict[str, Any]] = [
    _destination("Lisbon, Portugal", ["food", "history", "coast", "budget"], 9_000),
    _destination("Kyoto, Japan", ["culture", "history", "food", "temples"], 14_000),
    _destination("Reykjavik, Iceland", ["nature", "hiking", "adventure"], 19_500),
    _destination("Mexico City, Mexico", ["food", "art", "nightlife", "budget"], 8_000),
    _destination("Queenstown, New Zealand", ["adventure", "hiking", "nature"], 16_000),
    _destination("Rome, Italy", ["history", "food", "art"], 12_500),
    _destination("Chiang Mai, Thailand", ["food", "temples", "budget", "nature"], 6_000),
    _destination("Barcelona, Spain", ["beach", "art", "food", "nightlife"], 11_500),
]

_ATTRACTION_KINDS = [
    ("old town walking loop", "sightseeing", 0, 120),
    ("central food market", "food", 2_000, 90),
    ("national museum", "sightseeing", 1_500, 150),
    ("hilltop viewpoint at sunset", "sightseeing", 0, 90),
    ("half-day cooking class", "activity", 6_000, 240),
    ("coastal or river day trip", "activity", 5_000, 420),
    ("neighbourhood cafe crawl", "food", 2_500, 120),
    ("botanical gardens", "sightseeing", 1_000, 100),
    ("live music venue", "activity", 2_000, 150),
    ("guided history tour", "activity", 3_500, 180),
]

_CONDITIONS = [
    "mild and dry",
    "warm and humid",
    "cool with showers",
    "cold and clear",
    "hot and sunny",
]

_BUDGET_CAPS = {"low": 10_000, "medium": 15_000, "high": 1_000_000}


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
        cap = _BUDGET_CAPS.get(budget_level, 1_000_000)

        scored = []
        for dest in _DESTINATIONS:
            if dest["base_daily"] > cap:
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
                    "estimated_daily_cost": dest["base_daily"],
                    "known_for": dest["tags"][:3],
                }
            )
        scored.sort(key=lambda d: (-d["match_score"], d["estimated_daily_cost"]))
        return scored[:5]

    def search_flights(
        self, origin: str, destination: str, depart: date, travelers: int
    ) -> list[dict[str, Any]]:
        base = _seed(origin, destination, depart.isoformat())
        options = []
        for i, (carrier, stops) in enumerate(
            [("Meridian Air", 0), ("Northwind", 1), ("Cadence Airways", 1)]
        ):
            price = max(_jitter(base + i, 15_000, 80_000) - stops * 3_500, 7_500.0)
            options.append(
                {
                    "carrier": carrier,
                    "origin": origin,
                    "destination": destination,
                    "depart_date": depart.isoformat(),
                    "stops": stops,
                    "duration_hours": round(4 + stops * 3.5 + (base + i) % 7, 1),
                    "price_per_person": round(price, 2),
                    "total": round(price * travelers, 2),
                }
            )
        return sorted(options, key=lambda o: o["total"])

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
            nightly = _jitter(base + i * 17, 3_000, 28_000) + i * 3_800
            options.append(
                {
                    "name": name,
                    "tier": tier,
                    "rating": rating,
                    "neighbourhood": f"central {city}",
                    "nightly": round(nightly, 2),
                    "rooms_needed": rooms,
                    "total": round(nightly * nights * rooms, 2),
                }
            )
        return sorted(options, key=lambda o: o["total"])

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
                    "typical_cost": cost,
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
