from __future__ import annotations

import pytest

from market_agent.workflow_retry_policy import ProviderError, RetryPolicy


def test_retry_uses_full_jitter_but_stops_before_deadline():
    """Scheduling a retry that cannot start before the deadline must be terminal."""
    policy = RetryPolicy(base_delay=0.1, max_delay=1.0, max_attempts=3)

    decision = policy.decide(
        TimeoutError("provider timed out"),
        attempt=1,
        deadline=1.0,
        remaining_cost=1.0,
        now=0.99,
        random=lambda low, high: high,
    )

    assert decision.kind == "terminal"
    assert decision.reason == "deadline"


def test_retry_classifies_transient_http_errors_and_honors_retry_after():
    """Dropping a supported transient status or server wait must stop retries too early."""
    policy = RetryPolicy(base_delay=0.1, max_delay=1.0, max_attempts=3)

    decision = policy.decide(
        ProviderError(status_code=429, retry_after=0.75),
        attempt=1,
        deadline=2.0,
        remaining_cost=1.0,
        now=0.0,
        random=lambda low, high: 0.01,
    )

    assert decision.kind == "retry"
    assert decision.delay == 0.75


def test_retry_uses_the_capped_exponential_full_jitter_range():
    """Changing the exponent, cap, or uniform bounds must change the scheduled retry."""
    sampled_ranges: list[tuple[float, float]] = []

    def sample(low: float, high: float) -> float:
        sampled_ranges.append((low, high))
        return high

    policy = RetryPolicy(base_delay=0.5, max_delay=1.0, max_attempts=4)

    decision = policy.decide(TimeoutError(), 2, 5.0, 1.0, 0.0, sample)

    assert decision.delay == 1.0
    assert sampled_ranges == [(0.0, 1.0)]


def test_retry_classifies_all_supported_transient_http_statuses():
    """Omitting one listed transient provider status must fail closed rather than silently change policy."""
    policy = RetryPolicy(max_attempts=2)

    for status_code in (408, 409, 429, 500, 599):
        decision = policy.decide(ProviderError(status_code=status_code), 0, 2.0, 1.0, 0.0, lambda low, high: 0.0)
        assert decision.kind == "retry"


def test_retry_denies_auth_and_schema_errors_and_insufficient_cost():
    """Treating permanent failures or an unfunded attempt as retryable wastes authority."""
    policy = RetryPolicy(base_delay=0.1, max_delay=1.0, max_attempts=3, retry_cost=0.2)

    assert policy.decide(ProviderError(status_code=401), 0, 2.0, 1.0, 0.0, lambda low, high: high).reason == "non_retryable"
    assert policy.decide(ProviderError(code="schema"), 0, 2.0, 1.0, 0.0, lambda low, high: high).reason == "non_retryable"
    assert policy.decide(TimeoutError(), 0, 2.0, 0.19, 0.0, lambda low, high: high).reason == "cost"


@pytest.mark.parametrize("code", ["authentication", "authorization", "validation", "schema", "safety", "malformed_output"])
@pytest.mark.parametrize("status", [408, 409, 429, 503])
def test_permanent_provider_code_overrides_transient_status_in_public_policy(code, status):
    """Permanent facts must stop retries at the public policy boundary itself."""
    policy = RetryPolicy()
    error = ProviderError(status_code=status, code=code)
    assert policy.is_retryable(error) is False
    decision = policy.decide(error, 0, 2.0, 1.0, 0.0, lambda low, high: high)
    assert decision.terminal
    assert decision.reason == "non_retryable"
    assert decision.delay is None


def test_permanent_provider_code_overrides_transport_exception_type():
    error = TimeoutError()
    error.code = "safety"
    assert RetryPolicy.is_retryable(error) is False
