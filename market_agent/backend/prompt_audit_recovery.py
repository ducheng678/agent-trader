"""Background replay for durable prompt-activation audit outboxes."""

from __future__ import annotations

import logging
import math
import threading
from collections.abc import Callable
from typing import Protocol


class PromptAuditReplayManager(Protocol):
    def replay_pending_audit(self, *, limit: int = 100) -> int:
        """Deliver pending prompt activation audit records."""


class PromptAuditRecoveryScheduler:
    """Periodically recover prompt audit events that missed their first delivery."""

    _BATCH_LIMIT = 100

    def __init__(
        self,
        manager: PromptAuditReplayManager,
        *,
        error_observer: Callable[[str, Exception], object] | None = None,
        interval_seconds: float = 30.0,
    ) -> None:
        if not callable(getattr(manager, "replay_pending_audit", None)):
            raise TypeError("prompt audit recovery requires a replay manager")
        if error_observer is not None and not callable(error_observer):
            raise TypeError("prompt audit recovery error observer must be callable")
        if (
            isinstance(interval_seconds, bool)
            or not isinstance(interval_seconds, (int, float))
            or not math.isfinite(interval_seconds)
            or interval_seconds <= 0
        ):
            raise ValueError("prompt audit recovery interval must be finite and positive")
        self._manager = manager
        self._error_observer = error_observer
        self._interval = float(interval_seconds)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._thread is not None:
                return
            self._thread = threading.Thread(
                target=self._run,
                name="prompt-audit-recovery",
                daemon=True,
            )
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=5.0)

    def run_once(self) -> int:
        return self._manager.replay_pending_audit(limit=self._BATCH_LIMIT)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception as error:
                self._observe_error(error)
            self._stop.wait(self._interval)

    def _observe_error(self, error: Exception) -> None:
        if self._error_observer is None:
            return
        try:
            self._error_observer("prompt_audit_recovery_failed", error)
        except Exception as observer_error:
            logging.getLogger("market_agent.prompt_audit_recovery").warning(
                "prompt audit recovery error observer failed: %s",
                type(observer_error).__name__,
            )
