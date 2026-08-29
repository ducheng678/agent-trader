from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from market_agent.backend.cache import CacheBackend
from market_agent.backend.database import EventRecord, JobRecord, JobRepository
from market_agent.backend.errors import (
    BackendError,
    DependencyUnavailableError,
    JobNotFoundError,
    TaskQueueFullError,
    UnknownTaskError,
)
from market_agent.backend.message_bus import MessageBus, MessageEnvelope
from market_agent.backend.observability import MetricsRegistry, request_context

TaskHandler = Callable[[dict[str, Any]], Any]


@dataclass(frozen=True)
class TaskSubmission:
    job: JobRecord
    reused: bool


class BackgroundTaskQueue:
    def __init__(
        self,
        repository: JobRepository,
        cache: CacheBackend,
        message_bus: MessageBus,
        metrics: MetricsRegistry,
        max_workers: int,
        queue_capacity: int,
        default_max_attempts: int,
        retry_delay_seconds: float,
    ) -> None:
        worker_count = int(max_workers)
        waiting_capacity = int(queue_capacity)
        attempt_limit = int(default_max_attempts)
        retry_delay = float(retry_delay_seconds)
        if worker_count < 1:
            raise ValueError("max_workers must be at least 1")
        if waiting_capacity < 0:
            raise ValueError("queue_capacity cannot be negative")
        if worker_count + waiting_capacity > 9999:
            raise ValueError("combined worker and queue capacity cannot exceed 9999")
        if attempt_limit < 1:
            raise ValueError("default_max_attempts must be at least 1")
        if not math.isfinite(retry_delay) or retry_delay < 0:
            raise ValueError("retry_delay_seconds must be finite and non-negative")
        self._repository = repository
        self._cache = cache
        self._message_bus = message_bus
        self._metrics = metrics
        self._default_max_attempts = attempt_limit
        self._retry_delay_seconds = retry_delay
        self._handlers: dict[str, TaskHandler] = {}
        self._handlers_lock = threading.RLock()
        self._submission_lock = threading.RLock()
        self._active_job_ids: set[str] = set()
        self._active_jobs_lock = threading.RLock()
        self._active_jobs_changed = threading.Condition(self._active_jobs_lock)
        self._active_jobs_version = 0
        self._futures: set[Future[Any]] = set()
        self._futures_lock = threading.RLock()
        self._recovery_threads: set[threading.Thread] = set()
        self._recovery_threads_lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="market-agent-task")
        self._capacity_limit = worker_count + waiting_capacity
        self._recovery_page_size = self._capacity_limit + 1
        self._capacity = threading.BoundedSemaphore(self._capacity_limit)
        self._logger = logging.getLogger("market_agent.backend.tasks")
        self._metrics.set_gauge("market_agent_task_worker_capacity", worker_count)
        self._metrics.set_gauge("market_agent_task_queue_capacity", waiting_capacity)
        self._metrics.set_gauge("market_agent_task_inflight", 0)

    def register(self, task_name: str, handler: TaskHandler) -> None:
        normalized_name = str(task_name or "").strip()
        if not normalized_name:
            raise ValueError("task_name is required")
        if not callable(handler):
            raise TypeError("handler must be callable")
        with self._handlers_lock:
            if normalized_name in self._handlers:
                raise ValueError(f"task handler already registered: {normalized_name}")
            self._handlers[normalized_name] = handler
        self._start_recovery(normalized_name, handler)

    def _start_recovery(self, task_name: str, handler: TaskHandler) -> None:
        if self._shutdown_event.is_set():
            return

        def run() -> None:
            try:
                self._recover_registered_tasks(task_name, handler)
            finally:
                current = threading.current_thread()
                with self._recovery_threads_lock:
                    self._recovery_threads.discard(current)

        thread = threading.Thread(target=run, name=f"market-agent-recovery-{task_name}", daemon=True)
        with self._recovery_threads_lock:
            self._recovery_threads.add(thread)
        thread.start()

    def _recover_registered_tasks(self, task_name: str, handler: TaskHandler) -> None:
        while not self._shutdown_event.is_set():
            try:
                with self._submission_lock:
                    probe_jobs = self._repository.list_recoverable_jobs(task_name, limit=self._recovery_page_size)
                    with self._active_jobs_changed:
                        active_jobs_version = self._active_jobs_version
                        has_unscheduled_job = any(job.job_id not in self._active_job_ids for job in probe_jobs)
                if not probe_jobs:
                    return
                if not has_unscheduled_job:
                    with self._active_jobs_changed:
                        if active_jobs_version == self._active_jobs_version and not self._shutdown_event.is_set():
                            self._active_jobs_changed.wait(timeout=0.5)
                    continue
            except Exception as exc:
                self._logger.error(
                    "durable task recovery probe failed",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                self._metrics.increment("market_agent_task_recovery_failed_total", labels={"task_name": task_name})
                self._shutdown_event.wait(0.5)
                continue
            while not self._shutdown_event.is_set():
                if self._capacity.acquire(timeout=0.25):
                    break
            else:
                return
            release_capacity = True
            try:
                with self._submission_lock:
                    jobs = self._repository.list_recoverable_jobs(task_name, limit=self._recovery_page_size)
                    with self._active_jobs_lock:
                        job = next((item for item in jobs if item.job_id not in self._active_job_ids), None)
                    if job is None:
                        return
                    if job.attempt_count >= job.max_attempts:
                        self._repository.mark_failed(
                            job.job_id,
                            {"type": "RecoveryExhausted", "message": "task exhausted attempts before recovery"},
                        )
                    else:
                        if job.status == "running":
                            job = self._repository.mark_recovery_queued(
                                job.job_id,
                                {"attempt_count": job.attempt_count, "reason": "worker_restart"},
                            )
                        else:
                            self._repository.append_event(
                                job.job_id,
                                "task_recovery_queued",
                                {"attempt_count": job.attempt_count, "reason": "worker_restart"},
                            )
                        self._submit_reserved(job, handler)
                        release_capacity = False
                        self._metrics.increment("market_agent_task_recovered_total", labels={"task_name": task_name})
            except Exception as exc:
                self._logger.error(
                    "durable task recovery failed",
                    exc_info=(type(exc), exc, exc.__traceback__),
                )
                self._metrics.increment("market_agent_task_recovery_failed_total", labels={"task_name": task_name})
                self._shutdown_event.wait(0.5)
            finally:
                if release_capacity:
                    self._capacity.release()

    def submit(
        self,
        task_name: str,
        payload: dict[str, Any],
        *,
        idempotency_key: str | None = None,
        request_id: str = "",
        max_attempts: int | None = None,
    ) -> TaskSubmission:
        normalized_name = str(task_name or "").strip()
        with self._handlers_lock:
            handler = self._handlers.get(normalized_name)
        if handler is None:
            raise UnknownTaskError(f"unknown task: {normalized_name}")
        if self._shutdown_event.is_set():
            raise DependencyUnavailableError("task queue has been shut down")
        attempts = self._default_max_attempts if max_attempts is None else int(max_attempts)
        if attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if not self._capacity.acquire(blocking=False):
            if idempotency_key is not None:
                try:
                    with self._submission_lock:
                        existing = self._repository.find_idempotent_job(
                            normalized_name,
                            dict(payload or {}),
                            idempotency_key,
                        )
                except BackendError:
                    raise
                except Exception as exc:
                    raise DependencyUnavailableError("idempotency lookup failed") from exc
                if existing is not None:
                    self._metrics.increment(
                        "market_agent_task_idempotency_reused_total",
                        labels={"task_name": normalized_name},
                    )
                    return TaskSubmission(job=existing, reused=True)
            self._metrics.increment("market_agent_task_rejected_total", labels={"task_name": normalized_name})
            raise TaskQueueFullError("task queue is at capacity")
        release_capacity = True
        try:
            with self._submission_lock:
                job, reused = self._repository.create_or_get_job(
                    normalized_name,
                    dict(payload or {}),
                    idempotency_key,
                    attempts,
                    str(request_id or ""),
                )
                if reused:
                    self._metrics.increment("market_agent_task_idempotency_reused_total", labels={"task_name": normalized_name})
                    return TaskSubmission(job=job, reused=True)
                self._submit_reserved(job, handler)
                release_capacity = False
            self._metrics.increment("market_agent_task_submitted_total", labels={"task_name": normalized_name})
            return TaskSubmission(job=job, reused=False)
        except BackendError:
            raise
        except Exception as exc:
            if "job" in locals() and not locals().get("reused", False):
                self._record_terminal_failure(job, exc, "TaskSchedulingError")
            raise DependencyUnavailableError("task could not be scheduled") from exc
        finally:
            if release_capacity:
                self._capacity.release()

    def _submit_reserved(self, job: JobRecord, handler: TaskHandler) -> None:
        with self._active_jobs_changed:
            if job.job_id in self._active_job_ids:
                raise RuntimeError(f"task is already scheduled: {job.job_id}")
            self._active_job_ids.add(job.job_id)
            self._active_jobs_version += 1
            self._metrics.set_gauge("market_agent_task_inflight", len(self._active_job_ids))
            self._active_jobs_changed.notify_all()
        try:
            future = self._executor.submit(self._run_task_guarded, job, handler)
        except Exception:
            with self._active_jobs_changed:
                if job.job_id in self._active_job_ids:
                    self._active_job_ids.remove(job.job_id)
                    self._active_jobs_version += 1
                self._metrics.set_gauge("market_agent_task_inflight", len(self._active_job_ids))
                self._active_jobs_changed.notify_all()
            raise
        with self._futures_lock:
            self._futures.add(future)
        future.add_done_callback(lambda completed, submitted_job=job: self._task_done(completed, submitted_job))

    def _task_done(self, future: Future[Any], job: JobRecord) -> None:
        unexpected = None
        try:
            unexpected = future.exception()
        except Exception as exc:
            unexpected = exc
        if unexpected is not None:
            self._logger.error(
                "background task future failed",
                exc_info=(type(unexpected), unexpected, unexpected.__traceback__),
            )
            self._metrics.increment("market_agent_task_infrastructure_failed_total", labels={"task_name": job.task_name})
        with self._futures_lock:
            self._futures.discard(future)
        with self._active_jobs_changed:
            if job.job_id in self._active_job_ids:
                self._active_job_ids.remove(job.job_id)
                self._active_jobs_version += 1
            self._metrics.set_gauge("market_agent_task_inflight", len(self._active_job_ids))
            self._active_jobs_changed.notify_all()
        self._capacity.release()

    def get_job(self, job_id: str) -> JobRecord:
        cache_key = f"job:{job_id}"
        cached = self._cache.get(cache_key)
        if isinstance(cached, JobRecord):
            return cached
        job = self._repository.get_job(job_id)
        if job is None:
            raise JobNotFoundError(f"job not found: {job_id}")
        if job.status in {"succeeded", "failed"}:
            self._cache.set(cache_key, job)
        return job

    def list_events(self, job_id: str, limit: int = 100) -> list[EventRecord]:
        self.get_job(job_id)
        return self._repository.list_events(job_id, limit=limit)

    def _publish(self, topic: str, job: JobRecord, payload: dict[str, Any]) -> None:
        try:
            self._message_bus.publish(
                MessageEnvelope(topic=topic, payload=payload, request_id=job.request_id, job_id=job.job_id)
            )
        except Exception as exc:
            self._logger.error(
                "message subscriber failed",
                exc_info=(type(exc), exc, exc.__traceback__),
                extra={"topic": topic},
            )
            self._metrics.increment("market_agent_message_publish_failures_total", labels={"topic": topic})

    @staticmethod
    def _error_payload(exc: Exception) -> dict[str, Any]:
        error: dict[str, Any] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "retryable": bool(getattr(exc, "retryable", False)),
        }
        if isinstance(exc, BackendError):
            error["code"] = exc.error_code
            if exc.details:
                error["details"] = exc.details
        return error

    def _record_terminal_failure(self, job: JobRecord, exc: Exception, error_type: str) -> None:
        error = self._error_payload(exc)
        error["type"] = error_type
        try:
            current = self._repository.get_job(job.job_id)
            if current is not None and current.status in {"accepted", "running"}:
                self._repository.mark_failed(job.job_id, error)
        except Exception as persistence_exc:
            self._logger.error(
                "task infrastructure failure could not be persisted",
                exc_info=(type(persistence_exc), persistence_exc, persistence_exc.__traceback__),
            )
        self._logger.error(
            "task infrastructure failure",
            exc_info=(type(exc), exc, exc.__traceback__),
        )
        self._metrics.increment("market_agent_task_infrastructure_failed_total", labels={"task_name": job.task_name})

    def _run_task_guarded(self, job: JobRecord, handler: TaskHandler) -> None:
        try:
            self._run_task(job, handler)
        except Exception as exc:
            self._record_terminal_failure(job, exc, "TaskInfrastructureError")

    def _run_task(self, job: JobRecord, handler: TaskHandler) -> None:
        with request_context(job.request_id, job.job_id):
            for attempt in range(max(1, job.attempt_count + 1), job.max_attempts + 1):
                running = self._repository.mark_running(job.job_id, attempt)
                self._publish("task.started", running, {"task_name": running.task_name, "attempt": attempt})
                started_at = time.perf_counter()
                try:
                    result = handler(dict(job.payload))
                except Exception as exc:
                    duration = time.perf_counter() - started_at
                    self._metrics.observe("market_agent_task_duration_seconds", duration, labels={"task_name": job.task_name})
                    error = self._error_payload(exc)
                    if error["retryable"] and attempt < job.max_attempts:
                        retrying = self._repository.mark_retry_scheduled(
                            job.job_id,
                            {"attempt": attempt, "next_attempt": attempt + 1, "error": error},
                        )
                        self._metrics.increment("market_agent_task_retry_total", labels={"task_name": job.task_name})
                        self._publish("task.retry_scheduled", retrying, {"attempt": attempt, "error": error})
                        delay = self._retry_delay_seconds * (2 ** (attempt - 1))
                        if delay > 0 and self._shutdown_event.wait(delay):
                            self._repository.mark_failed(
                                job.job_id,
                                {"type": "TaskShutdown", "message": "task queue shut down before retry", "retryable": False},
                            )
                            return
                        continue
                    failed = self._repository.mark_failed(job.job_id, error)
                    self._metrics.increment("market_agent_task_failed_total", labels={"task_name": job.task_name})
                    self._publish("task.failed", failed, error)
                    self._logger.error(
                        "background task failed",
                        exc_info=(type(exc), exc, exc.__traceback__),
                    )
                    return
                duration = time.perf_counter() - started_at
                succeeded = self._repository.mark_succeeded(job.job_id, result)
                self._metrics.observe("market_agent_task_duration_seconds", duration, labels={"task_name": job.task_name})
                self._metrics.increment("market_agent_task_succeeded_total", labels={"task_name": job.task_name})
                self._publish("task.succeeded", succeeded, {"result": result})
                return

    def is_healthy(self) -> bool:
        return not self._shutdown_event.is_set()

    def shutdown(self, wait: bool = True) -> None:
        self._shutdown_event.set()
        with self._active_jobs_changed:
            self._active_jobs_changed.notify_all()
        with self._recovery_threads_lock:
            recovery_threads = tuple(self._recovery_threads)
        for thread in recovery_threads:
            thread.join()
        self._executor.shutdown(wait=wait, cancel_futures=False)
