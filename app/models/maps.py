"""Answers from Google Maps, for requests that aren't a trip to plan.

Shown live and never saved: Google's terms for Maps Grounding Lite forbid
storing or caching its results, so unlike `PlannedTrip` nothing here goes
through app/store.py.
"""

from typing import Literal

from pydantic import BaseModel

TravelMode = Literal["DRIVE", "WALK"]


class Attribution(BaseModel):
    """Google requires this be shown with the content it came with."""

    title: str
    url: str | None = None


class PlaceLink(BaseModel):
    """One place a search summary cites. Maps Grounding Lite doesn't return
    names here — the names are in the summary text, cited as [0], [1]…"""

    index: int
    place_url: str | None = None
    directions_url: str | None = None
    reviews_url: str | None = None
    attribution: Attribution | None = None


class PlacesAnswer(BaseModel):
    query: str
    summary: str
    places: list[PlaceLink]


class CityPlaces(BaseModel):
    """One city's places for a trip that asked for them. `error` is set
    instead of `places` when that city's search failed — the trip itself is
    still returned."""

    city: str
    places: PlacesAnswer | None = None
    error: str | None = None


class RouteAnswer(BaseModel):
    origin: str
    destination: str
    travel_mode: TravelMode = "DRIVE"
    distance_meters: int | None = None
    duration_seconds: int | None = None
    maps_url: str
    attribution: Attribution | None = None
