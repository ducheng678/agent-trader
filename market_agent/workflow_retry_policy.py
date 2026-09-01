"""Closed retry classification and bounded full-jitter scheduling for agent calls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Literal, Protocol


_PERMANENT_PROVIDER_CODES = frozenset({
    "authentication", "authorization", "validation", "schema", "safety", "malformed_output",
})


class UniformRandom(Protocol):
    def uniform(self, lower: float, upper: float, /) -> float: ...


@dataclass(frozen=True, slots=True)
class ProviderError(Exception):
    """Normalized provider failure facts; unknown facts remain non-retryable."""

    status_code: int | None = None
    code: str | None = None
    retry_after: float | None = None


@dataclass(frozen=True, slots=True)
class RetryDecision:
    kind: Literal["retry", "terminal"]
    delay: float | None
    reason: str

    @property
    def terminal(self) -> bool:
        return self.kind == "terminal"


class RetryPolicy:
    """Reject all unknown failures and schedule only funded, deadline-safe retries."""

    def __init__(
        self,
        *,
        base_delay: float = 0.25,
        max_delay: float = 10.0,
        max_attempts: int = 3,
        retry_cost: float = 0.0,
    ) -> None:
        if base_delay < 0 or max_delay < 0 or max_attempts < 1 or retry_cost < 0:
            raise ValueError("retry policy configuration must be non-negative and allow an attempt")
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._max_attempts = max_attempts
        self._retry_cost = retry_cost

    def decide(
        self,
        error: BaseException,
        attempt: int,
        deadline: float,
        remaining_cost: float,
        now: float,
        random: Callable[[float, float], float] | UniformRandom,
    ) -> RetryDecision:
        if not self.is_retryable(error):
            return RetryDecision("terminal", None, "non_retryable")
        if attempt < 0 or attempt >= self._max_attempts:
            return RetryDecision("terminal", None, "attempt_limit")
        if remaining_cost < self._retry_cost:
            return RetryDecision("terminal", None, "cost")

        ceiling = min(self._max_delay, self._base_delay * 2**attempt)
        delay = self._uniform(random, ceiling)
        retry_after = self._retry_after(error)
        if retry_after is not None:
            delay = max(delay, retry_after)
        if now + delay >= deadline:
            return RetryDecision("terminal", None, "deadline")
        return RetryDecision("retry", delay, "retryable")

    @staticmethod
    def is_retryable(error: BaseException) -> bool:
        if getattr(error, "code", None) in _PERMANENT_PROVIDER_CODES:
            return False
        if isinstance(error, (TimeoutError, ConnectionError)):
            return True
        status_code = getattr(error, "status_code", None)
        return isinstance(status_code, int) and (status_code in {408, 409, 429} or 500 <= status_code <= 599)

    @staticmethod
    def _retry_after(error: BaseException) -> float | None:
        value = getattr(error, "retry_after", None)
        return value if isinstance(value, (int, float)) and value >= 0 else None

    @staticmethod
    def _uniform(random: Callable[[float, float], float] | UniformRandom, ceiling: float) -> float:
        sample = random(0.0, ceiling) if callable(random) else random.uniform(0.0, ceiling)
        if not isinstance(sample, (int, float)) or not 0.0 <= sample <= ceiling:
            raise ValueError("random source returned a value outside the full-jitter range")
        return float(sample)
