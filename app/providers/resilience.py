"""Retries and a circuit breaker for live data sources.

Shared by the live flight and hotel clients. The fallback order itself —
fresh cache, live source, stale cache, unavailable — lives in each client;
this module only answers two questions: should this failure be retried, and
should the live source be tried at all right now.

- `with_retries` retries only failures that tend to clear up by themselves
  (timeouts, dropped connections). A bad key, an exhausted quota or "no
  results" is not retried: it would only cost time, and with a quota, more
  searches.
- `CircuitBreaker` stops calling a source that keeps failing. After
  `failure_threshold` failures in a row — or at once, for a failure that
  won't clear up on its own, like an exhausted quota — it skips the source
  for `cooldown_seconds`, then lets one call through to test it again. While
  it's open, searches go straight to the fallback instead of each waiting
  out a timeout.
"""

import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime

from app.models.itinerary import DataSource
from app.providers.flight_cache import FlightCache

logger = logging.getLogger(__name__)

RETRY_DELAYS_SECONDS = (1.0, 3.0)


def with_retries[T](
    call: Callable[[], T],
    *,
    is_transient: Callable[[Exception], bool],
    delays: tuple[float, ...] = RETRY_DELAYS_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """`call()`, retried once per entry in `delays` (waiting that long first)
    while it fails with something `is_transient` accepts."""
    for attempt, delay in enumerate((*delays, None), start=1):
        try:
            return call()
        except Exception as exc:
            if delay is None or not is_transient(exc):
                raise
            logger.info("attempt %d failed (%s); retrying in %.0f s", attempt, exc, delay)
            sleep(delay)
    raise AssertionError("unreachable")


class CircuitBreaker:
    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 3,
        cooldown_seconds: float = 900,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self._threshold = failure_threshold
        self._cooldown = cooldown_seconds
        self._clock = clock
        self._failures = 0
        self._opened_at: float | None = None
        self._reason = ""
        # Flight legs are searched from two threads at once.
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        with self._lock:
            return self._opened_at is not None and not self._cooldown_over()

    @property
    def reason(self) -> str:
        return self._reason

    def _cooldown_over(self) -> bool:
        return self._opened_at is not None and self._clock() - self._opened_at >= self._cooldown

    def allow(self) -> bool:
        """Whether to try the live source now. Once the cooldown is over,
        calls are let through again; one more failure re-opens it."""
        with self._lock:
            return self._opened_at is None or self._cooldown_over()

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._reason = ""

    def record_failure(self, reason: str, *, open_now: bool = False) -> None:
        with self._lock:
            self._failures += 1
            if open_now or self._failures >= self._threshold or self._cooldown_over():
                self._opened_at = self._clock()
                self._reason = reason
                logger.warning(
                    "%s: skipping the live source for %.0f min (%s)",
                    self.name,
                    self._cooldown / 60,
                    reason,
                )


def _when(epoch_seconds: float) -> datetime:
    return datetime.fromtimestamp(epoch_seconds, tz=UTC)


def fetch_with_fallback(
    *,
    cache: FlightCache,
    cache_key: str,
    live: Callable[[], dict],
    breaker: CircuitBreaker,
    provider: str,
    link: str,
) -> tuple[dict | None, DataSource]:
    """The retry and fallback chain, shared by flights and hotels. In order:

    1. a fresh cached search;
    2. `live()` — which retries transient failures itself — unless `breaker`
       is open after repeated failures;
    3. an older cached search, still kept, labelled stale;
    4. nothing, with the reason and `link` to search for themselves.

    Never raises, and never falls back to sample data. A failure that has an
    `account_problem` attribute set (bad key, exhausted quota) opens the
    breaker at once.
    """
    kept = cache.lookup(cache_key)
    if kept is not None and cache.get(cache_key) is not None:
        return kept.payload, DataSource(
            status="cached", provider=provider, fetched_at=_when(kept.fetched_at), link=link
        )

    if breaker.allow():
        try:
            payload = live()
        except Exception as exc:
            failure = str(exc)
            breaker.record_failure(failure, open_now=getattr(exc, "account_problem", False))
            logger.warning("live %s search failed: %s", provider, failure)
        else:
            breaker.record_success()
            cache.put(cache_key, payload)
            return payload, DataSource(
                status="live", provider=provider, fetched_at=datetime.now(UTC), link=link
            )
    else:
        failure = f"skipped after repeated failures ({breaker.reason})"

    if kept is not None:
        return kept.payload, DataSource(
            status="stale",
            provider=provider,
            fetched_at=_when(kept.fetched_at),
            detail=f"the live search failed: {failure}",
            link=link,
        )
    return None, DataSource(status="unavailable", provider=provider, detail=failure, link=link)
