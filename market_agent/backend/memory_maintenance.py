"""Background scheduling for governed memory forgetting and derivative cleanup."""

from __future__ import annotations

import threading
import time
import logging
from collections.abc import Callable, Iterable
from datetime import datetime, timezone
from hashlib import sha256

from market_agent.workflow_memory_lifecycle import LifecycleLimits, LifecycleWorker


class MemoryMaintenanceScheduler:
    def __init__(self, worker: LifecycleWorker, *, tenant_id: str, authority: object,
                 interval_seconds: float = 3600.0,
                 cleanup_callbacks: Iterable[Callable[[float], object]] = (),
                 error_observer: Callable[[str, Exception], object] | None = None) -> None:
        if type(worker) is not LifecycleWorker or not tenant_id.strip() or interval_seconds <= 0:
            raise ValueError("memory maintenance configuration is invalid")
        self._worker = worker
        self._tenant = tenant_id
        self._authority = authority
        self._interval = float(interval_seconds)
        self._cleanup_callbacks = tuple(cleanup_callbacks)
        if not all(callable(callback) for callback in self._cleanup_callbacks):
            raise ValueError("maintenance cleanup callbacks must be callable")
        self._error_observer = error_observer
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="memory-lifecycle", daemon=True)

    def start(self) -> None:
        if not self._thread.is_alive():
            self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def run_once(self) -> None:
        now = datetime.now(timezone.utc)
        plan = self._worker.plan(self._tenant, now)
        key = sha256((self._tenant + plan.plan_hash).encode("utf-8")).hexdigest()
        self._worker.apply(
            plan,
            LifecycleLimits(max_actions=100, max_cleanup=100),
            tenant_id=self._tenant,
            # Lifecycle writes participate in the same trace contract as
            # request work.  Derive a stable, non-zero W3C trace id for the
            # maintenance window instead of passing a prose identifier.
            trace_id=sha256(
                f"memory-lifecycle:{self._tenant}:{now.strftime('%Y%m%d%H')}".encode("utf-8")
            ).hexdigest()[:32],
            idempotency_key=key,
            authority=self._authority,
        )
        epoch = now.timestamp()
        for cleanup in self._cleanup_callbacks:
            cleanup(epoch)

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.run_once()
            except Exception as error:
                if self._error_observer is not None:
                    try:
                        self._error_observer("memory_maintenance_failed", error)
                    except Exception as observer_error:
                        logging.getLogger("market_agent.memory_maintenance").warning(
                            "memory maintenance error observer failed: %s",
                            type(observer_error).__name__,
                        )
            self._stop.wait(self._interval)
