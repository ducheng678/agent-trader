from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from market_agent.backend.cache import TTLCache
from market_agent.backend.database import JobRepository
from market_agent.backend.message_bus import InMemoryMessageBus
from market_agent.backend.observability import MetricsRegistry, configure_structured_logging
from market_agent.backend.settings import BackendSettings
from market_agent.backend.task_queue import BackgroundTaskQueue


@dataclass
class BackendContainer:
    settings: BackendSettings
    repository: JobRepository
    cache: TTLCache
    message_bus: InMemoryMessageBus
    metrics: MetricsRegistry
    task_queue: BackgroundTaskQueue
    agent_service: Any = None

    @classmethod
    def create(cls, settings: BackendSettings | None = None) -> "BackendContainer":
        resolved_settings = (settings or BackendSettings.from_env()).validate()
        configure_structured_logging()
        repository = JobRepository(resolved_settings.database_path)
        cache = TTLCache(resolved_settings.cache_max_entries, resolved_settings.cache_default_ttl_seconds)
        metrics = MetricsRegistry()
        message_bus = InMemoryMessageBus()
        task_queue = BackgroundTaskQueue(
            repository=repository,
            message_bus=message_bus,
            metrics=metrics,
            max_workers=resolved_settings.task_workers,
            default_max_attempts=resolved_settings.task_max_attempts,
            retry_delay_seconds=resolved_settings.task_retry_delay_seconds,
        )
        container = cls(
            settings=resolved_settings,
            repository=repository,
            cache=cache,
            message_bus=message_bus,
            metrics=metrics,
            task_queue=task_queue,
        )
        from market_agent.backend.agent_service import register_agent_tasks

        container.agent_service = register_agent_tasks(task_queue)
        return container

    def readiness(self) -> dict[str, str]:
        database_status = "ok" if self.repository.healthcheck() else "failed"
        cache_stats = self.cache.stats()
        self.metrics.set_gauge("market_agent_cache_entries", cache_stats.size)
        self.metrics.set_gauge("market_agent_cache_hits_total", cache_stats.hits)
        self.metrics.set_gauge("market_agent_cache_misses_total", cache_stats.misses)
        return {"database": database_status, "task_queue": "ok", "cache": "ok"}

    def shutdown(self) -> None:
        self.task_queue.shutdown(wait=True)
        self.repository.close()
