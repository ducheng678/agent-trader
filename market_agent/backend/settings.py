from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path

from market_agent.backend.errors import ConfigurationError


def _environment_value(name: str, default: str = "") -> str:
    return str(os.getenv(name, default) or default).strip()


def _integer_value(name: str, default: int) -> int:
    value = _environment_value(name, str(default))
    try:
        return int(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be an integer") from exc


def _float_value(name: str, default: float) -> float:
    value = _environment_value(name, str(default))
    try:
        return float(value)
    except ValueError as exc:
        raise ConfigurationError(f"{name} must be a number") from exc


@dataclass(frozen=True)
class BackendSettings:
    database_path: Path = field(default_factory=lambda: Path(_environment_value("MARKET_AGENT_DATABASE_PATH", "runtime/market_agent_backend.sqlite3")))
    cache_max_entries: int = field(default_factory=lambda: _integer_value("MARKET_AGENT_CACHE_MAX_ENTRIES", 1024))
    cache_default_ttl_seconds: float = field(default_factory=lambda: _float_value("MARKET_AGENT_CACHE_TTL_SECONDS", 60.0))
    task_workers: int = field(default_factory=lambda: _integer_value("MARKET_AGENT_TASK_WORKERS", 4))
    task_queue_capacity: int = field(default_factory=lambda: _integer_value("MARKET_AGENT_TASK_QUEUE_CAPACITY", 128))
    task_max_attempts: int = field(default_factory=lambda: _integer_value("MARKET_AGENT_TASK_MAX_ATTEMPTS", 3))
    task_retry_delay_seconds: float = field(default_factory=lambda: _float_value("MARKET_AGENT_TASK_RETRY_DELAY_SECONDS", 1.0))
    api_token: str = field(default_factory=lambda: _environment_value("MARKET_AGENT_API_TOKEN"))
    api_host: str = field(default_factory=lambda: _environment_value("MARKET_AGENT_API_HOST", "127.0.0.1"))
    api_port: int = field(default_factory=lambda: _integer_value("MARKET_AGENT_API_PORT", 8080))
    environment: str = field(default_factory=lambda: _environment_value("MARKET_AGENT_ENVIRONMENT", "development").lower())
    trace_event_capacity: int = field(default_factory=lambda: _integer_value("MARKET_AGENT_TRACE_EVENT_CAPACITY", 10000))
    trace_query_limit: int = field(default_factory=lambda: _integer_value("MARKET_AGENT_TRACE_QUERY_LIMIT", 100))
    workflow_metric_series_limit: int = field(default_factory=lambda: _integer_value("MARKET_AGENT_METRIC_SERIES_LIMIT", 2048))

    @classmethod
    def from_env(cls) -> "BackendSettings":
        return cls()

    def validate(self) -> "BackendSettings":
        if type(self.trace_event_capacity) is not int or not 1 <= self.trace_event_capacity <= 100000:
            raise ConfigurationError("trace_event_capacity must be between 1 and 100000")
        if type(self.trace_query_limit) is not int or not 1 <= self.trace_query_limit <= min(500, self.trace_event_capacity):
            raise ConfigurationError("trace_query_limit must be within event capacity and at most 500")
        if type(self.workflow_metric_series_limit) is not int or not 16 <= self.workflow_metric_series_limit <= 4096:
            raise ConfigurationError("workflow_metric_series_limit must be between 16 and 4096")
        if self.cache_max_entries < 1:
            raise ConfigurationError("cache_max_entries must be at least 1")
        if not math.isfinite(self.cache_default_ttl_seconds) or self.cache_default_ttl_seconds <= 0:
            raise ConfigurationError("cache_default_ttl_seconds must be finite and positive")
        if self.task_workers < 1:
            raise ConfigurationError("task_workers must be at least 1")
        if self.task_queue_capacity < 0:
            raise ConfigurationError("task_queue_capacity cannot be negative")
        if self.task_workers + self.task_queue_capacity > 9999:
            raise ConfigurationError("combined task worker and queue capacity cannot exceed 9999")
        if self.task_max_attempts < 1:
            raise ConfigurationError("task_max_attempts must be at least 1")
        if not math.isfinite(self.task_retry_delay_seconds) or self.task_retry_delay_seconds < 0:
            raise ConfigurationError("task_retry_delay_seconds must be finite and non-negative")
        if not 1 <= self.api_port <= 65535:
            raise ConfigurationError("api_port must be between 1 and 65535")
        normalized_host = str(self.api_host).strip().lower()
        if not normalized_host or normalized_host != str(self.api_host).lower():
            raise ConfigurationError("api_host must be non-empty and cannot contain surrounding whitespace")
        local_hosts = {"127.0.0.1", "localhost", "::1"}
        environment = str(self.environment).strip().lower()
        if (environment in {"production", "prod", "staging"} or normalized_host not in local_hosts) and not str(self.api_token).strip():
            raise ConfigurationError("MARKET_AGENT_API_TOKEN is required outside local-only development")
        return self
