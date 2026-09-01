"""Model/task-isolated circuit breaker for provider invocation admission."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class CircuitDecision:
    kind: Literal["allow", "probe", "reject"]


@dataclass(slots=True)
class _CircuitState:
    status: Literal["closed", "open", "half_open"] = "closed"
    failures: int = 0
    opened_at: float | None = None


class CircuitBreaker:
    """Open after consecutive failures, with one recovery probe per keyed circuit."""

    def __init__(self, *, failure_threshold: int = 3, cooldown: float = 30.0) -> None:
        if failure_threshold < 1 or cooldown < 0:
            raise ValueError("breaker threshold must be positive and cooldown non-negative")
        self._failure_threshold = failure_threshold
        self._cooldown = cooldown
        self._states: dict[tuple[str, str], _CircuitState] = {}

    def acquire(self, model: str, task_kind: str, now: float) -> CircuitDecision:
        state = self._state_for(model, task_kind)
        if state.status == "closed":
            return CircuitDecision("allow")
        if state.status == "half_open":
            return CircuitDecision("reject")
        assert state.opened_at is not None
        if now < state.opened_at + self._cooldown:
            return CircuitDecision("reject")
        state.status = "half_open"
        return CircuitDecision("probe")

    def record(self, model: str, task_kind: str, success: bool, now: float) -> None:
        state = self._state_for(model, task_kind)
        if success:
            state.status = "closed"
            state.failures = 0
            state.opened_at = None
            return
        if state.status == "half_open":
            state.status = "open"
            state.opened_at = now
            return
        state.failures += 1
        if state.failures >= self._failure_threshold:
            state.status = "open"
            state.opened_at = now

    def _state_for(self, model: str, task_kind: str) -> _CircuitState:
        if not model or not task_kind:
            raise ValueError("model and task_kind must be non-empty")
        return self._states.setdefault((model, task_kind), _CircuitState())
