from app.core.config import get_settings
from app.providers.base import TravelProvider
from app.providers.live import LiveTravelProvider
from app.providers.mock import MockTravelProvider


def get_provider() -> TravelProvider:
    """Return the provider named by settings.

    "mock" is deterministic fiction, needs no keys, and is what the tests run
    against; the page labels it "Sample data". "live" serves real Google
    Flights and Google Hotels prices via SerpApi, each through a retry and
    fallback chain, and mock data for everything else: attractions and
    weather still have no real source here.
    """
    provider = get_settings().provider
    if provider == "mock":
        return MockTravelProvider()
    if provider == "live":
        return LiveTravelProvider()
    raise NotImplementedError(
        f"provider={provider!r} has no implementation yet — add one under "
        "app/providers/ and wire it up here."
    )


__all__ = ["LiveTravelProvider", "MockTravelProvider", "TravelProvider", "get_provider"]
