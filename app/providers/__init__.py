from app.core.config import get_settings
from app.providers.base import TravelProvider
from app.providers.mock import MockTravelProvider


def get_provider() -> TravelProvider:
    """Return the provider named by settings.

    Only "mock" ships today. "live" is the seam where real flight and lodging
    clients go; it fails loudly rather than quietly serving invented data.
    """
    provider = get_settings().provider
    if provider == "mock":
        return MockTravelProvider()
    raise NotImplementedError(
        f"provider={provider!r} has no implementation yet — add one under "
        "app/providers/ and wire it up here."
    )


__all__ = ["MockTravelProvider", "TravelProvider", "get_provider"]
