"""Cooperative run-scoped cancellation shared by API, Harness, graph and agents."""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CancellationSignal:
    _event: threading.Event

    def is_cancelled(self) -> bool:
        return self._event.is_set()


class WorkflowCancellationRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._events: dict[str, threading.Event] = {}

    def signal(self, run_id: str) -> CancellationSignal:
        key = str(run_id).strip()
        if not key:
            raise ValueError("run identifier is required")
        with self._lock:
            return CancellationSignal(self._events.setdefault(key, threading.Event()))

    def cancel(self, run_id: str) -> None:
        self.signal(run_id)._event.set()
