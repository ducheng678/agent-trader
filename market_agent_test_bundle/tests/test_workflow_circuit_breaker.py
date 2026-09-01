from __future__ import annotations

from market_agent.workflow_circuit_breaker import CircuitBreaker


def test_breaker_admits_only_one_half_open_probe():
    """Allowing a second cooldown probe can stampede an unhealthy dependency."""
    breaker = CircuitBreaker(failure_threshold=1, cooldown=10.0)
    breaker.record("luna", "extract", success=False, now=0.0)

    assert breaker.acquire("luna", "extract", now=11.0).kind == "probe"
    assert breaker.acquire("luna", "extract", now=11.0).kind == "reject"


def test_breaker_reopens_on_failed_probe_and_isolates_model_task_keys():
    """A failed probe must reopen only its own model/task circuit."""
    breaker = CircuitBreaker(failure_threshold=1, cooldown=10.0)
    breaker.record("luna", "extract", success=False, now=0.0)

    assert breaker.acquire("luna", "extract", now=10.0).kind == "probe"
    breaker.record("luna", "extract", success=False, now=10.0)

    assert breaker.acquire("luna", "extract", now=19.0).kind == "reject"
    assert breaker.acquire("terra", "extract", now=19.0).kind == "allow"
    assert breaker.acquire("luna", "analyze", now=19.0).kind == "allow"


def test_successful_probe_closes_the_circuit():
    """Leaving a recovered circuit half-open would reject normal subsequent calls."""
    breaker = CircuitBreaker(failure_threshold=1, cooldown=10.0)
    breaker.record("luna", "extract", success=False, now=0.0)
    assert breaker.acquire("luna", "extract", now=10.0).kind == "probe"

    breaker.record("luna", "extract", success=True, now=10.0)

    assert breaker.acquire("luna", "extract", now=10.0).kind == "allow"


def test_success_resets_consecutive_failure_count():
    """Counting non-consecutive failures would open a circuit after a recovered call."""
    breaker = CircuitBreaker(failure_threshold=2, cooldown=10.0)

    breaker.record("luna", "extract", success=False, now=0.0)
    breaker.record("luna", "extract", success=True, now=1.0)
    breaker.record("luna", "extract", success=False, now=2.0)

    assert breaker.acquire("luna", "extract", now=2.0).kind == "allow"
