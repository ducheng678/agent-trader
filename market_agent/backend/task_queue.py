from __future__ import annotations

import logging
import math
import re
import threading
import time
import uuid
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from market_agent.backend.cache import CacheBackend
from market_agent.backend.database import EventRecord, JobRecord, JobRepository
from market_agent.backend.errors import (
    BackendError,
    DependencyUnavailableError,
    ExecutionLeaseLostError,
    JobNotFoundError,
    TaskQueueFullError,
    UnknownTaskError,
)
from market_agent.backend.execution_fence import ExecutionFence, execution_fence_context
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
        trace_observability: Any = None,
        lease_seconds: float = 30.0,
        recovery_poll_seconds: float = 0.5,
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
        if not math.isfinite(lease_seconds) or lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive and finite")
        if not math.isfinite(recovery_poll_seconds) or recovery_poll_seconds <= 0:
            raise ValueError("recovery_poll_seconds must be positive and finite")
        self._repository = repository
        self._cache = cache
        self._message_bus = message_bus
        self._durable_dispatch = bool(getattr(message_bus, "durable_dispatch", False))
        self._dispatch_unsubscribe: Callable[[], None] | None = None
        self._metrics = metrics
        self._trace_observability = trace_observability
        self._default_max_attempts = attempt_limit
        self._retry_delay_seconds = retry_delay
        self._handlers: dict[str, TaskHandler] = {}
        self._handlers_lock = threading.RLock()
        self._submission_lock = threading.RLock()
        self._active_job_ids: set[str] = set()
        self._execution_tokens: dict[str, str] = {}
        self._execution_fences: dict[str, ExecutionFence] = {}
        self._lease_seconds = float(lease_seconds)
        self._recovery_poll_seconds = float(recovery_poll_seconds)
        self._active_jobs_lock = threading.RLock()
        self._active_jobs_changed = threading.Condition(self._active_jobs_lock)
        self._active_jobs_version = 0
        self._futures: set[Future[Any]] = set()
        self._futures_lock = threading.RLock()
        self._recovery_threads: set[threading.Thread] = set()
        self._recovery_threads_lock = threading.RLock()
        self._shutdown_event = threading.Event()
        self._lease_stop_event = threading.Event()
        self._executor = ThreadPoolExecutor(max_workers=worker_count, thread_name_prefix="market-agent-task")
        self._capacity_limit = worker_count + waiting_capacity
        self._recovery_page_size = self._capacity_limit + 1
        self._capacity = threading.BoundedSemaphore(self._capacity_limit)
        self._logger = logging.getLogger("market_agent.backend.tasks")
        self._metrics.set_gauge("market_agent_task_worker_capacity", worker_count)
        self._metrics.set_gauge("market_agent_task_queue_capacity", waiting_capacity)
        self._metrics.set_gauge("market_agent_task_inflight", 0)
        self._lease_thread = threading.Thread(
            target=self._renew_leases, name="market-agent-job-leases", daemon=True,
        )
        self._lease_thread.start()

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
            if self._durable_dispatch and self._dispatch_unsubscribe is None:
                self._dispatch_unsubscribe = self._message_bus.subscribe(
                    "task.dispatch", self._handle_dispatch_message
                )
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
                    jobs = self._repository.list_recoverable_jobs(task_name, limit=self._recovery_page_size)
                    for job in jobs:
                        if self._shutdown_event.is_set():
                            break
                        with self._active_jobs_lock:
                            if job.job_id in self._active_job_ids:
                                continue
                        if not self._capacity.acquire(blocking=False):
                            break
                        scheduled = False
                        try:
                            # Claim directly: republishing unclaimed recovery
                            # candidates can flood the stream on every poll.
                            scheduled = self._submit_reserved(job, handler, recovery=True)
                            if scheduled:
                                self._metrics.increment("market_agent_task_recovered_total", labels={"task_name": task_name})
                        finally:
                            if not scheduled:
                                self._capacity.release()
            except Exception as exc:
                self._logger.error("durable task recovery failed", exc_info=(type(exc), exc, exc.__traceback__))
                self._metrics.increment("market_agent_task_recovery_failed_total", labels={"task_name": task_name})
            # Empty pages are expected while another process owns healthy jobs.
            # Keep checking so their later expiry can be recovered.
            self._shutdown_event.wait(self._recovery_poll_seconds)

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
        if not self._durable_dispatch and not self._capacity.acquire(blocking=False):
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
        release_capacity = not self._durable_dispatch
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
                if self._durable_dispatch:
                    self._publish_dispatch(job)
                elif self._submit_reserved(job, handler):
                    release_capacity = False
            self._metrics.increment("market_agent_task_submitted_total", labels={"task_name": normalized_name})
            return TaskSubmission(job=job, reused=False)
        except BackendError:
            raise
        except Exception as exc:
            # Accepted jobs remain recoverable if publishing or scheduling
            # fails; another worker may already own them.
            raise DependencyUnavailableError("task could not be scheduled") from exc
        finally:
            if release_capacity:
                self._capacity.release()

    def _submit_reserved(self, job: JobRecord, handler: TaskHandler, *, recovery: bool = False) -> bool:
        with self._active_jobs_changed:
            if job.job_id in self._active_job_ids:
                return False
            token = uuid.uuid4().hex
            claimed = self._repository.claim_job(job.job_id, token, self._lease_seconds, recovery=recovery)
            if claimed is None:
                return False
            job = claimed
            if job.attempt_count >= job.max_attempts:
                try:
                    self._repository.mark_failed(
                        job.job_id,
                        {"type": "RecoveryExhausted", "message": "task exhausted attempts before recovery"},
                        execution_token=token,
                    )
                finally:
                    self._repository.release_job_lease(job.job_id, token)
                return False
            self._active_job_ids.add(job.job_id)
            self._execution_tokens[job.job_id] = token
            self._execution_fences[job.job_id] = ExecutionFence()
            self._active_jobs_version += 1
            self._metrics.set_gauge("market_agent_task_inflight", len(self._active_job_ids))
            self._active_jobs_changed.notify_all()
        try:
            future = self._executor.submit(self._run_task_guarded, job, handler)
        except Exception:
            try:
                self._repository.release_job_lease(job.job_id, token)
            finally:
                with self._active_jobs_changed:
                    self._execution_tokens.pop(job.job_id, None)
                    self._execution_fences.pop(job.job_id, None)
                    self._active_job_ids.discard(job.job_id)
                    self._active_jobs_version += 1
                    self._metrics.set_gauge("market_agent_task_inflight", len(self._active_job_ids))
                    self._active_jobs_changed.notify_all()
            raise
        with self._futures_lock:
            self._futures.add(future)
        future.add_done_callback(lambda completed, submitted_job=job: self._task_done(completed, submitted_job))
        return True

    def _publish_dispatch(self, job: JobRecord) -> None:
        """Publish an immutable job envelope to the durable worker stream."""

        trace_id = str(job.payload.get("trace_id") or job.request_id or "")
        if not re.fullmatch(r"[0-9a-fA-F]{32}", trace_id) or not int(trace_id, 16):
            raise DependencyUnavailableError(
                "durable task dispatch requires a nonzero trace_id"
            )
        self._message_bus.publish(
            MessageEnvelope(
                topic="task.dispatch",
                payload={
                    "trace_id": trace_id,
                    "task_name": job.task_name,
                    "job_id": job.job_id,
                    "request_id": job.request_id,
                    "payload": dict(job.payload),
                    "attempt_count": job.attempt_count,
                    "max_attempts": job.max_attempts,
                },
                request_id=job.request_id or trace_id,
                job_id=job.job_id,
            )
        )
        self._metrics.increment(
            "market_agent_task_dispatch_published_total",
            labels={"task_name": job.task_name},
        )

    def _handle_dispatch_message(self, message: MessageEnvelope) -> None:
        """ACK only after local scheduling or a durable competing claim.

        A scheduled message can be ACKed: the shared lease/recovery loop owns
        subsequent crash recovery independently of Redis delivery ownership.
        """
        if message.topic != "task.dispatch" or not isinstance(message.payload, dict):
            return
        task_name = message.payload.get("task_name")
        job_id = message.payload.get("job_id")
        if not isinstance(task_name, str) or not isinstance(job_id, str):
            raise ValueError("task dispatch envelope is malformed")
        with self._handlers_lock:
            handler = self._handlers.get(task_name)
        if handler is None:
            raise UnknownTaskError(f"unknown task: {task_name}")
        if self._shutdown_event.is_set():
            raise DependencyUnavailableError("task worker is shutting down")
        with self._submission_lock:
            job = self._repository.get_job(job_id)
            if job is None or job.status in {"succeeded", "failed"}:
                return
            if job.task_name != task_name:
                raise ValueError("task dispatch does not match persisted task")
            with self._active_jobs_lock:
                if job_id in self._active_job_ids:
                    return
            if not self._capacity.acquire(blocking=False):
                raise TaskQueueFullError("task worker capacity is temporarily exhausted")
            scheduled = False
            try:
                scheduled = self._submit_reserved(job, handler, recovery=True)
            except Exception as exc:
                raise DependencyUnavailableError("task worker could not schedule delivery") from exc
            finally:
                if not scheduled:
                    self._capacity.release()

    def _task_done(self, future: Future[Any], job: JobRecord) -> None:
        try:
            unexpected = future.exception()
        except Exception as exc:
            unexpected = exc
        if unexpected is not None:
            self._logger.error("background task future failed", exc_info=(type(unexpected), unexpected, unexpected.__traceback__))
            self._metrics.increment("market_agent_task_infrastructure_failed_total", labels={"task_name": job.task_name})
        with self._futures_lock:
            self._futures.discard(future)
        with self._active_jobs_changed:
            self._execution_tokens.pop(job.job_id, None)
            self._execution_fences.pop(job.job_id, None)
            self._active_job_ids.discard(job.job_id)
            self._active_jobs_version += 1
            self._metrics.set_gauge("market_agent_task_inflight", len(self._active_job_ids))
            self._active_jobs_changed.notify_all()
            if self._shutdown_event.is_set() and not self._active_job_ids:
                self._lease_stop_event.set()
        try:
            self._repository.release_job_lease(job.job_id, job.execution_token)
        except Exception:
            self._metrics.increment("market_agent_task_lease_release_failed_total")
        self._capacity.release()

    def _renew_leases(self) -> None:
        interval = min(1.0, self._lease_seconds / 3)
        while not self._lease_stop_event.wait(interval):
            with self._active_jobs_lock:
                owned = tuple(
                    (job_id, token) for job_id, token in self._execution_tokens.items()
                    if not self._execution_fences[job_id].is_lost()
                )
            for job_id, token in owned:
                try:
                    renewed = self._repository.renew_job_lease(job_id, token, self._lease_seconds)
                    if not renewed:
                        self._latch_execution_lost(job_id, token)
                except Exception:
                    self._latch_execution_lost(job_id, token)
                    self._metrics.increment("market_agent_task_lease_renew_failed_total")

    def _latch_execution_lost(self, job_id: str, token: str) -> None:
        with self._active_jobs_lock:
            if self._execution_tokens.get(job_id) == token:
                fence = self._execution_fences.get(job_id)
                if fence is not None:
                    fence.latch_lost()

    def _execution_fence(self, job: JobRecord) -> ExecutionFence | None:
        with self._active_jobs_lock:
            if self._execution_tokens.get(job.job_id) == job.execution_token:
                return self._execution_fences.get(job.job_id)
        return None

    def _check_execution_fence(self, job: JobRecord) -> None:
        fence = self._execution_fence(job)
        if fence is not None:
            fence.raise_if_lost()

    def _ensure_execution_lease(self, job: JobRecord) -> None:
        self._check_execution_fence(job)
        try:
            renewed = self._repository.renew_job_lease(job.job_id, job.execution_token, self._lease_seconds)
        except Exception as exc:
            self._latch_execution_lost(job.job_id, job.execution_token)
            raise ExecutionLeaseLostError("task execution lease renewal is uncertain") from exc
        if not renewed:
            self._latch_execution_lost(job.job_id, job.execution_token)
            raise ExecutionLeaseLostError("task execution lease was lost")
        self._check_execution_fence(job)

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

    def get_job_by_idempotency_key(self, idempotency_key: str) -> JobRecord | None:
        return self._repository.get_job_by_idempotency_key(idempotency_key)

    def list_events(self, job_id: str, limit: int = 100) -> list[EventRecord]:
        self.get_job(job_id)
        return self._repository.list_events(job_id, limit=limit)

    def _publish(self, topic: str, job: JobRecord, payload: dict[str, Any]) -> None:
        try:
            message_payload = dict(payload)
            message_payload.setdefault("trace_id", job.payload.get("trace_id") or job.request_id)
            self._message_bus.publish(
                MessageEnvelope(topic=topic, payload=message_payload, request_id=job.request_id, job_id=job.job_id)
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
            self._check_execution_fence(job)
            current = self._repository.get_job(job.job_id)
            if current is not None and current.status in {"accepted", "running"}:
                self._check_execution_fence(job)
                self._repository.mark_failed(job.job_id, error, execution_token=job.execution_token)
        except ExecutionLeaseLostError:
            # A later owner, not this failed worker, controls durable status.
            return
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
        with execution_fence_context(self._execution_fence(job)):
            try:
                self._run_task(job, handler)
            except ExecutionLeaseLostError:
                self._latch_execution_lost(job.job_id, job.execution_token)
                self._metrics.increment("market_agent_task_lease_lost_total", labels={"task_name": job.task_name})
            except Exception as exc:
                self._record_terminal_failure(job, exc, "TaskInfrastructureError")

    def _run_task(self, job: JobRecord, handler: TaskHandler) -> None:
        with request_context(job.request_id, job.job_id):
            for attempt in range(max(1, job.attempt_count + 1), job.max_attempts + 1):
                self._ensure_execution_lease(job)
                running = self._repository.mark_running(job.job_id, attempt, execution_token=job.execution_token)
                self._record_trace(job, "queue_started", "started", attempt=attempt)
                self._publish("task.started", running, {"task_name": running.task_name, "attempt": attempt})
                started_at = time.perf_counter()
                self._check_execution_fence(job)
                try:
                    result = handler(dict(job.payload))
                except Exception as exc:
                    self._check_execution_fence(job)
                    if isinstance(exc, ExecutionLeaseLostError):
                        raise
                    duration = time.perf_counter() - started_at
                    self._metrics.observe("market_agent_task_duration_seconds", duration, labels={"task_name": job.task_name})
                    error = self._error_payload(exc)
                    if error["retryable"] and attempt < job.max_attempts:
                        self._check_execution_fence(job)
                        retrying = self._repository.mark_retry_scheduled(
                            job.job_id,
                            {"attempt": attempt, "next_attempt": attempt + 1, "error": error},
                            execution_token=job.execution_token,
                        )
                        self._metrics.increment("market_agent_task_retry_total", labels={"task_name": job.task_name})
                        self._publish("task.retry_scheduled", retrying, {"attempt": attempt, "error": error})
                        self._record_trace(job, "retry_scheduled", "started", attempt=attempt, reason="provider_error")
                        delay = self._retry_delay_seconds * (2 ** (attempt - 1))
                        stopped = delay > 0 and self._shutdown_event.wait(delay)
                        self._check_execution_fence(job)
                        if stopped:
                            self._repository.mark_failed(
                                job.job_id,
                                {"type": "TaskShutdown", "message": "task queue shut down before retry", "retryable": False},
                                execution_token=job.execution_token,
                            )
                            return
                        continue
                    self._check_execution_fence(job)
                    failed = self._repository.mark_failed(job.job_id, error, execution_token=job.execution_token)
                    self._metrics.increment("market_agent_task_failed_total", labels={"task_name": job.task_name})
                    self._publish("task.failed", failed, error)
                    self._record_trace(job, "queue_failed", "failed", attempt=attempt, reason="provider_error")
                    self._logger.error(
                        "background task failed",
                        exc_info=(type(exc), exc, exc.__traceback__),
                    )
                    return
                self._check_execution_fence(job)
                duration = time.perf_counter() - started_at
                self._check_execution_fence(job)
                succeeded = self._repository.mark_succeeded(job.job_id, result, execution_token=job.execution_token)
                self._metrics.observe("market_agent_task_duration_seconds", duration, labels={"task_name": job.task_name})
                self._metrics.increment("market_agent_task_succeeded_total", labels={"task_name": job.task_name})
                self._publish("task.succeeded", succeeded, {"result": result})
                self._record_trace(job, "queue_completed", "succeeded", attempt=attempt)
                return

    def _record_trace(self, job: JobRecord, event: str, status: str, *, attempt: int = 0, reason: str = "none") -> None:
        observer = self._trace_observability
        trace_id = str(job.payload.get("trace_id") or job.request_id or "")
        if observer is None or not re.fullmatch(r"[0-9a-fA-F]{32}", trace_id) or not int(trace_id, 16):
            return
        try:
            from market_agent.workflow_tracing import TraceContext
            from hashlib import sha256
            span = sha256(f"queue:{job.job_id}:{attempt}:{event}".encode("utf-8")).hexdigest()[:16]
            observer.record_component(
                TraceContext(trace_id=trace_id.lower(), span_id=span), event=event,
                status=status, component="queue", workflow_id=job.job_id,
                task_id=job.task_name, attempt=attempt, reason=reason,
            )
        except Exception:
            self._metrics.increment("market_agent_observability_failures_total", labels={"phase": "queue"})

    def is_healthy(self) -> bool:
        return not self._shutdown_event.is_set()

    def shutdown(self, wait: bool = True) -> None:
        self._shutdown_event.set()
        if self._dispatch_unsubscribe is not None:
            self._dispatch_unsubscribe()
            self._dispatch_unsubscribe = None
        with self._active_jobs_changed:
            self._active_jobs_changed.notify_all()
        with self._recovery_threads_lock:
            recovery_threads = tuple(self._recovery_threads)
        for thread in recovery_threads:
            thread.join()
        self._executor.shutdown(wait=wait, cancel_futures=False)
        with self._active_jobs_lock:
            if wait or not self._active_job_ids:
                self._lease_stop_event.set()
        if wait:
            self._lease_thread.join()
