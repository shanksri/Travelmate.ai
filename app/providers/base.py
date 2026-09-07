"""The data surface the agent's tools are allowed to reach.

Everything the agent learns about the world comes through this Protocol, so a
real integration is a drop-in replacement and the tests never touch a network.
"""

from datetime import date
from typing import Any, Protocol


class TravelProvider(Protocol):
    def search_destinations(
        self, interests: list[str], month: int, budget_level: str
    ) -> list[dict[str, Any]]:
        """Candidate destinations matching interests, season and budget."""
        ...

    def search_flights(
        self, origin: str, destination: str, depart: date, travelers: int
    ) -> list[dict[str, Any]]:
        """Priced flight options for one leg."""
        ...

    def search_lodging(
        self, destination: str, check_in: date, nights: int, travelers: int
    ) -> list[dict[str, Any]]:
        """Priced lodging options for the stay."""
        ...

    def search_attractions(
        self, destination: str, interests: list[str], limit: int
    ) -> list[dict[str, Any]]:
        """Things to do, with rough cost and duration."""
        ...

    def get_weather_outlook(self, destination: str, when: date) -> dict[str, Any]:
        """Typical conditions for that place at that time of year."""
        ...
