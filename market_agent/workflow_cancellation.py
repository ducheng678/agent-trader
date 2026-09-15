"""Cooperative run-scoped cancellation shared by API, Harness, graph and agents."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from market_agent.backend.cancellation_store import CancellationStore


@dataclass(frozen=True, slots=True)
class CancellationSignal:
    _event: threading.Event
    _store: CancellationStore | None = None
    _run_id: str = ""

    def is_cancelled(self) -> bool:
        if self._event.is_set():
            return True
        if self._store is None:
            return False
        try:
            return self._store.is_cancelled(self._run_id)
        except Exception:
            # Stop work while shared authority is unavailable, but do not turn
            # an outage into a permanent local user-cancellation latch.
            return True


class WorkflowCancellationRegistry:
    def __init__(self, store: CancellationStore | None = None) -> None:
        self._lock = threading.RLock()
        self._events: dict[str, threading.Event] = {}
        self._store = store

    @property
    def store(self) -> CancellationStore | None:
        return self._store

    def signal(self, run_id: str) -> CancellationSignal:
        if not isinstance(run_id, str) or not run_id.strip() or len(run_id) > 500 or "\x00" in run_id:
            raise ValueError("run identifier must be a nonempty string of at most 500 characters without NUL")
        with self._lock:
            return CancellationSignal(
                self._events.setdefault(run_id, threading.Event()), self._store, run_id,
            )

    def cancel(self, run_id: str) -> None:
        signal = self.signal(run_id)
        # A successful response promises cross-replica cancellation, so the
        # durable write must complete before publishing the local fast latch.
        if self._store is not None:
            self._store.cancel(run_id)
        signal._event.set()
