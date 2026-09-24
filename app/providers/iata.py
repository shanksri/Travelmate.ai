"""Place names -> IATA codes, for flight searches.

Flight APIs take airport codes; this app deals in place names ("Delhi",
"Kochi", "Kyoto, Japan"). Resolution uses Travelpayouts' public city and
airport directories — keyless static JSON, fetched once per process and
cached — so it works without any Travelpayouts account.
"""

from functools import lru_cache

import httpx

CITIES_URL = "https://api.travelpayouts.com/data/en/cities.json"
AIRPORTS_URL = "https://api.travelpayouts.com/data/en/airports.json"


@lru_cache(maxsize=1)
def _cities(timeout: float = 60.0) -> list[dict]:
    response = httpx.get(CITIES_URL, timeout=timeout)
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=1)
def _airports(timeout: float = 60.0) -> list[dict]:
    response = httpx.get(AIRPORTS_URL, timeout=timeout)
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=256)
def _matches(place: str) -> tuple[dict, ...]:
    """The best tier of city matches for a place name, best first; possibly
    several when the name is shared (there's a Kochi in India and in Japan).

    Takes the part before any comma, so "Kyoto, Japan" resolves like "Kyoto".
    Prefers an exact name match, then a city whose name contains the query —
    which is what turns "Delhi" into New Delhi's DEL.
    """
    name = place.split(",")[0].strip().lower()
    if not name:
        return ()

    exact, partial = [], []
    for city in _cities():
        city_name = (city.get("name") or "").lower()
        if not city_name:
            continue
        if city_name == name:
            exact.append(city)
        elif name in city_name:
            partial.append(city)

    def flightable(cities: list[dict]) -> list[dict]:
        return [c for c in cities if c.get("has_flightable_airport")]

    # A flightable partial match beats a non-flightable exact one: searching
    # from a city with no flightable airport returns nothing at all, so an
    # exactly-named dead end is worse than a nearby usable airport.
    for candidates in (flightable(exact), flightable(partial), exact, partial):
        if candidates:
            return tuple(candidates)
    return ()


def resolve_iata(place: str) -> str | None:
    """Best-effort IATA city code for one place name on its own, or None."""
    matches = _matches(place)
    return matches[0].get("code") if matches else None


def resolve_route(origin: str, destination: str) -> tuple[str | None, str | None]:
    """IATA city codes for both ends of a route, resolving shared names together.

    "Kochi" alone is ambiguous — Kochi, Japan (KCZ) and Kochi/Cochin, India
    (COK) are both flightable, and Japan's comes first in the directory. A
    real "Varanasi -> Kochi" search went to Japan and found nothing. So when
    a name is shared, prefer the match in the same country as the other end
    of the route; if no pair shares a country, each end keeps its best match.
    """
    from_matches, to_matches = _matches(origin), _matches(destination)
    if not from_matches or not to_matches:
        return (
            from_matches[0].get("code") if from_matches else None,
            to_matches[0].get("code") if to_matches else None,
        )

    for a in from_matches:
        for b in to_matches:
            if a.get("country_code") and a.get("country_code") == b.get("country_code"):
                return a.get("code"), b.get("code")
    return from_matches[0].get("code"), to_matches[0].get("code")


@lru_cache(maxsize=256)
def airports_for(city_code: str) -> tuple[str, ...]:
    """The flightable airports serving a city code, the city's own code first.

    A city code isn't always an airport: Tokyo is TYO, served by NRT and HND,
    and London's LON covers six airports plus a list of railway stations.
    Stations and bus terminals are excluded. Falls back to the city code
    itself when the directory lists nothing, which is right for the many
    cities whose city and airport codes are the same.
    """
    codes = [
        a["code"]
        for a in _airports()
        if a.get("city_code") == city_code
        and a.get("iata_type") == "airport"
        and a.get("flightable")
        and a.get("code")
    ]
    if not codes:
        return (city_code,)
    codes.sort(key=lambda code: code != city_code)  # the city's own code first
    return tuple(codes)
