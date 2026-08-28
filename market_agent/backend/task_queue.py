from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from market_agent.backend.database import EventRecord, JobRecord, JobRepository
from market_agent.backend.errors import JobNotFoundError, UnknownTaskError
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
        message_bus: MessageBus,
        metrics: MetricsRegistry,
        max_workers: int,
        default_max_attempts: int,
        retry_delay_seconds: float,
    ) -> None:
        self._repository = repository
        self._message_bus = message_bus
        self._metrics = metrics
        self._default_max_attempts = int(default_max_attempts)
        self._retry_delay_seconds = float(retry_delay_seconds)
        self._handlers: dict[str, TaskHandler] = {}
        self._handlers_lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=int(max_workers), thread_name_prefix="market-agent-task")
        self._shutdown = False
        self._logger = logging.getLogger("market_agent.backend.tasks")
        self._metrics.set_gauge("market_agent_task_worker_capacity", int(max_workers))

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
        self._recover_registered_tasks(normalized_name, handler)

    def _recover_registered_tasks(self, task_name: str, handler: TaskHandler) -> None:
        for job in self._repository.list_recoverable_jobs(task_name):
            if job.attempt_count >= job.max_attempts:
                self._repository.mark_failed(
                    job.job_id,
                    {"type": "RecoveryExhausted", "message": "task exhausted attempts before recovery"},
                )
                continue
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
            self._metrics.increment("market_agent_task_recovered_total", labels={"task_name": task_name})
            self._executor.submit(self._run_task, job, handler)

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
        if self._shutdown:
            raise RuntimeError("task queue has been shut down")
        attempts = self._default_max_attempts if max_attempts is None else int(max_attempts)
        if attempts < 1:
            raise ValueError("max_attempts must be at least 1")
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
        self._metrics.increment("market_agent_task_submitted_total", labels={"task_name": normalized_name})
        self._executor.submit(self._run_task, job, handler)
        return TaskSubmission(job=job, reused=False)

    def get_job(self, job_id: str) -> JobRecord:
        job = self._repository.get_job(job_id)
        if job is None:
            raise JobNotFoundError(f"job not found: {job_id}")
        return job

    def list_events(self, job_id: str, limit: int = 100) -> list[EventRecord]:
        self.get_job(job_id)
        return self._repository.list_events(job_id, limit=limit)

    def _publish(self, topic: str, job: JobRecord, payload: dict[str, Any]) -> None:
        try:
            self._message_bus.publish(
                MessageEnvelope(topic=topic, payload=payload, request_id=job.request_id, job_id=job.job_id)
            )
        except Exception:
            self._logger.exception("message subscriber failed", extra={"topic": topic})
            self._metrics.increment("market_agent_message_publish_failures_total", labels={"topic": topic})

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
                    error = {"type": type(exc).__name__, "message": str(exc)}
                    if attempt < job.max_attempts:
                        retrying = self._repository.mark_retry_scheduled(
                            job.job_id,
                            {"attempt": attempt, "next_attempt": attempt + 1, "error": error},
                        )
                        self._metrics.increment("market_agent_task_retry_total", labels={"task_name": job.task_name})
                        self._publish("task.retry_scheduled", retrying, {"attempt": attempt, "error": error})
                        delay = self._retry_delay_seconds * (2 ** (attempt - 1))
                        if delay > 0:
                            time.sleep(delay)
                        continue
                    failed = self._repository.mark_failed(job.job_id, error)
                    self._metrics.increment("market_agent_task_failed_total", labels={"task_name": job.task_name})
                    self._publish("task.failed", failed, error)
                    self._logger.exception("background task failed", exc_info=exc)
                    return
                duration = time.perf_counter() - started_at
                succeeded = self._repository.mark_succeeded(job.job_id, result)
                self._metrics.observe("market_agent_task_duration_seconds", duration, labels={"task_name": job.task_name})
                self._metrics.increment("market_agent_task_succeeded_total", labels={"task_name": job.task_name})
                self._publish("task.succeeded", succeeded, {"result": result})
                return

    def shutdown(self, wait: bool = True) -> None:
        self._shutdown = True
        self._executor.shutdown(wait=wait, cancel_futures=False)
