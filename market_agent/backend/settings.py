from __future__ import annotations

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
    task_max_attempts: int = field(default_factory=lambda: _integer_value("MARKET_AGENT_TASK_MAX_ATTEMPTS", 3))
    task_retry_delay_seconds: float = field(default_factory=lambda: _float_value("MARKET_AGENT_TASK_RETRY_DELAY_SECONDS", 1.0))
    api_token: str = field(default_factory=lambda: _environment_value("MARKET_AGENT_API_TOKEN"))
    environment: str = field(default_factory=lambda: _environment_value("MARKET_AGENT_ENVIRONMENT", "development").lower())

    @classmethod
    def from_env(cls) -> "BackendSettings":
        return cls()

    def validate(self) -> "BackendSettings":
        if self.cache_max_entries < 1:
            raise ConfigurationError("cache_max_entries must be at least 1")
        if self.cache_default_ttl_seconds <= 0:
            raise ConfigurationError("cache_default_ttl_seconds must be positive")
        if self.task_workers < 1:
            raise ConfigurationError("task_workers must be at least 1")
        if self.task_max_attempts < 1:
            raise ConfigurationError("task_max_attempts must be at least 1")
        if self.task_retry_delay_seconds < 0:
            raise ConfigurationError("task_retry_delay_seconds cannot be negative")
        if self.environment in {"production", "prod", "staging"} and not self.api_token:
            raise ConfigurationError("MARKET_AGENT_API_TOKEN is required outside local development")
        return self
