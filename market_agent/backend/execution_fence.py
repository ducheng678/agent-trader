"""Cooperative, owner-scoped execution fencing; never a workflow cancellation."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from threading import Event

from market_agent.backend.errors import ExecutionLeaseLostError


class ExecutionFence:
    """A monotonic latch shared by one executor and its lease heartbeat.

    This prevents subsequent cooperative work. It cannot revoke a synchronous
    external request that was already sent before ownership became uncertain.
    """

    def __init__(self) -> None:
        self._lost = Event()

    def is_lost(self) -> bool:
        return self._lost.is_set()

    def raise_if_lost(self) -> None:
        if self.is_lost():
            raise ExecutionLeaseLostError("task execution lease was lost")

    def latch_lost(self) -> None:
        self._lost.set()


_current_fence: ContextVar[ExecutionFence | None] = ContextVar("execution_fence", default=None)


def current_execution_fence() -> ExecutionFence | None:
    return _current_fence.get()


@contextmanager
def execution_fence_context(fence: ExecutionFence | None) -> Iterator[None]:
    token = _current_fence.set(fence)
    try:
        yield
    finally:
        _current_fence.reset(token)
