"""Retries and the circuit breaker, on their own — no network, no sleeping."""

import pytest

from app.providers.resilience import CircuitBreaker, with_retries


class Flaky(Exception):
    def __init__(self, transient: bool) -> None:
        super().__init__("transient" if transient else "permanent")
        self.transient = transient


def _calls(*outcomes):
    """A call that fails with each outcome in turn, then returns "ok"."""
    seen = []

    def call():
        seen.append(len(seen))
        if len(seen) <= len(outcomes):
            raise outcomes[len(seen) - 1]
        return "ok"

    return call, seen


def _is_transient(exc):
    return getattr(exc, "transient", False)


def test_a_transient_failure_is_retried_until_it_succeeds():
    call, seen = _calls(Flaky(True), Flaky(True))
    waits = []

    assert with_retries(call, is_transient=_is_transient, sleep=waits.append) == "ok"
    assert len(seen) == 3
    assert waits == [1.0, 3.0]


def test_retries_stop_after_the_last_delay():
    call, seen = _calls(Flaky(True), Flaky(True), Flaky(True))

    with pytest.raises(Flaky):
        with_retries(call, is_transient=_is_transient, sleep=lambda s: None)
    assert len(seen) == 3


def test_a_permanent_failure_is_not_retried():
    call, seen = _calls(Flaky(False))

    with pytest.raises(Flaky, match="permanent"):
        with_retries(call, is_transient=_is_transient, sleep=lambda s: None)
    assert len(seen) == 1


class Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_the_breaker_opens_after_repeated_failures_and_reopens_after_cooldown():
    clock = Clock()
    breaker = CircuitBreaker("test", failure_threshold=3, cooldown_seconds=900, clock=clock)

    breaker.record_failure("timeout")
    breaker.record_failure("timeout")
    assert breaker.allow()

    breaker.record_failure("timeout")
    assert not breaker.allow()
    assert breaker.reason == "timeout"

    clock.now = 899
    assert not breaker.allow()
    clock.now = 900
    assert breaker.allow()  # one call may test the source again


def test_an_account_problem_opens_the_breaker_at_once():
    breaker = CircuitBreaker("test", clock=Clock())

    breaker.record_failure("Your account has run out of searches.", open_now=True)

    assert not breaker.allow()


def test_a_success_closes_it_and_clears_the_count():
    clock = Clock()
    breaker = CircuitBreaker("test", failure_threshold=2, clock=clock)
    breaker.record_failure("x")
    breaker.record_success()
    breaker.record_failure("x")

    assert breaker.allow()


def test_a_failure_after_the_cooldown_reopens_it_straight_away():
    clock = Clock()
    breaker = CircuitBreaker("test", failure_threshold=3, cooldown_seconds=60, clock=clock)
    breaker.record_failure("down", open_now=True)
    clock.now = 60

    breaker.record_failure("still down")

    assert not breaker.allow()
