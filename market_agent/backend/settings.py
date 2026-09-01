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


def _boolean_value(name: str, default: bool) -> bool:
    value = _environment_value(name, "true" if default else "false").casefold()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ConfigurationError(f"{name} must be a boolean")


def _legacy_playbook_default() -> bool:
    environment = _environment_value("MARKET_AGENT_ENVIRONMENT", "development").casefold()
    return environment not in {"production", "prod", "staging"}


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
    legacy_playbook_api_enabled: bool = field(
        default_factory=lambda: _boolean_value(
            "MARKET_AGENT_LEGACY_PLAYBOOK_API_ENABLED", _legacy_playbook_default()
        )
    )
    harness_host_factory: str = field(default_factory=lambda: _environment_value("MARKET_AGENT_HARNESS_HOST_FACTORY"))
    trace_event_capacity: int = field(default_factory=lambda: _integer_value("MARKET_AGENT_TRACE_EVENT_CAPACITY", 10000))
    trace_query_limit: int = field(default_factory=lambda: _integer_value("MARKET_AGENT_TRACE_QUERY_LIMIT", 100))
    workflow_metric_series_limit: int = field(default_factory=lambda: _integer_value("MARKET_AGENT_METRIC_SERIES_LIMIT", 2048))
    tenant_id: str = field(default_factory=lambda: _environment_value("MARKET_AGENT_TENANT_ID", "default"))
    redis_url: str = field(default_factory=lambda: _environment_value("MARKET_AGENT_REDIS_URL"))
    postgres_dsn: str = field(default_factory=lambda: _environment_value("MARKET_AGENT_POSTGRES_DSN"))
    embedding_dimension: int = field(default_factory=lambda: _integer_value("MARKET_AGENT_EMBEDDING_DIMENSION", 1536))
    prompt_registry_path: Path = field(default_factory=lambda: Path(_environment_value("MARKET_AGENT_PROMPT_REGISTRY_PATH", "runtime/prompt_releases.sqlite3")))
    audit_database_path: Path = field(default_factory=lambda: Path(_environment_value("MARKET_AGENT_AUDIT_DATABASE_PATH", "runtime/workflow_audit.sqlite3")))
    memory_database_path: Path = field(default_factory=lambda: Path(_environment_value("MARKET_AGENT_MEMORY_DATABASE_PATH", "runtime/workflow_memory.sqlite3")))
    memory_maintenance_interval_seconds: float = field(default_factory=lambda: _float_value("MARKET_AGENT_MEMORY_MAINTENANCE_SECONDS", 3600.0))
    workflow_sol_model_id: str = field(default_factory=lambda: _environment_value("MARKET_AGENT_WORKFLOW_SOL_MODEL_ID", "gpt-5.6-sol"))
    workflow_terra_model_id: str = field(default_factory=lambda: _environment_value("MARKET_AGENT_WORKFLOW_TERRA_MODEL_ID", "gpt-5.6-terra"))
    workflow_luna_model_id: str = field(default_factory=lambda: _environment_value("MARKET_AGENT_WORKFLOW_LUNA_MODEL_ID", "gpt-5.6-luna"))
    workflow_sol_model_version: str = field(default_factory=lambda: _environment_value("MARKET_AGENT_WORKFLOW_SOL_MODEL_VERSION", "gpt-5.6-sol-v1"))
    workflow_terra_model_version: str = field(default_factory=lambda: _environment_value("MARKET_AGENT_WORKFLOW_TERRA_MODEL_VERSION", "gpt-5.6-terra-v1"))
    workflow_luna_model_version: str = field(default_factory=lambda: _environment_value("MARKET_AGENT_WORKFLOW_LUNA_MODEL_VERSION", "gpt-5.6-luna-v1"))
    embedding_model_id: str = field(default_factory=lambda: _environment_value("MARKET_AGENT_EMBEDDING_MODEL_ID", "text-embedding-3-small"))
    embedding_model_version: str = field(default_factory=lambda: _environment_value("MARKET_AGENT_EMBEDDING_MODEL_VERSION", "text-embedding-3-small-v1"))
    embedding_vector_version: str = field(default_factory=lambda: _environment_value("MARKET_AGENT_EMBEDDING_VECTOR_VERSION", "embedding-1536-v1"))
    prompt_cache_namespace: str = field(default_factory=lambda: _environment_value("MARKET_AGENT_PROMPT_CACHE_NAMESPACE", "market-agent-workflow-v1"))
    workflow_request_deadline_seconds: float = field(default_factory=lambda: _float_value("MARKET_AGENT_WORKFLOW_DEADLINE_SECONDS", 120.0))
    workflow_response_cache_ttl_seconds: float = field(default_factory=lambda: _float_value("MARKET_AGENT_WORKFLOW_CACHE_TTL_SECONDS", 300.0))

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
        if type(self.legacy_playbook_api_enabled) is not bool:
            raise ConfigurationError("legacy_playbook_api_enabled must be a boolean")
        if (environment in {"production", "prod", "staging"} or normalized_host not in local_hosts) and not str(self.api_token).strip():
            raise ConfigurationError("MARKET_AGENT_API_TOKEN is required outside local-only development")
        if not self.tenant_id or any(character.isspace() for character in self.tenant_id):
            raise ConfigurationError("MARKET_AGENT_TENANT_ID must be a non-empty compact identifier")
        if not 1 <= self.embedding_dimension <= 2000:
            raise ConfigurationError("MARKET_AGENT_EMBEDDING_DIMENSION must be between 1 and 2000")
        if environment in {"production", "prod", "staging"} and (not self.redis_url or not self.postgres_dsn):
            raise ConfigurationError("Redis and PostgreSQL are required in staging and production")
        if environment in {"production", "prod", "staging"}:
            module_name, separator, attribute = self.harness_host_factory.rpartition(":")
            if not separator or not module_name or not attribute:
                raise ConfigurationError("MARKET_AGENT_HARNESS_HOST_FACTORY must be module:callable in staging and production")
        if not math.isfinite(self.memory_maintenance_interval_seconds) or self.memory_maintenance_interval_seconds <= 0:
            raise ConfigurationError("memory maintenance interval must be finite and positive")
        expected_models = {
            "workflow_sol_model_id": "gpt-5.6-sol",
            "workflow_terra_model_id": "gpt-5.6-terra",
            "workflow_luna_model_id": "gpt-5.6-luna",
        }
        for field_name, expected in expected_models.items():
            if getattr(self, field_name) != expected:
                raise ConfigurationError(f"{field_name} must remain pinned to {expected}")
        version_fields = (
            self.workflow_sol_model_version,
            self.workflow_terra_model_version,
            self.workflow_luna_model_version,
            self.embedding_model_id,
            self.embedding_model_version,
            self.embedding_vector_version,
            self.prompt_cache_namespace,
        )
        if any(not value or len(value) > 128 or any(character.isspace() for character in value)
               for value in version_fields):
            raise ConfigurationError("workflow model, embedding, and cache versions must be compact identifiers")
        if not math.isfinite(self.workflow_request_deadline_seconds) or not 1.0 <= self.workflow_request_deadline_seconds <= 300.0:
            raise ConfigurationError("workflow request deadline must be finite and between 1 and 300 seconds")
        if not math.isfinite(self.workflow_response_cache_ttl_seconds) or self.workflow_response_cache_ttl_seconds <= 0:
            raise ConfigurationError("workflow response cache TTL must be finite and positive")
        return self
