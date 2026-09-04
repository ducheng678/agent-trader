from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any

from market_agent.backend.cache import TTLCache
from market_agent.backend.database import JobRepository, PostgresJobRepository
from market_agent.backend.message_bus import InMemoryMessageBus
from market_agent.backend.observability import MetricsRegistry, configure_structured_logging
from market_agent.backend.settings import BackendSettings
from market_agent.backend.task_queue import BackgroundTaskQueue
from market_agent.backend.trace_observability import BackendObservability
from market_agent.workflow_cancellation import WorkflowCancellationRegistry


@dataclass
class BackendContainer:
    settings: BackendSettings
    repository: JobRepository
    cache: Any
    message_bus: Any
    metrics: MetricsRegistry
    task_queue: BackgroundTaskQueue
    agent_service: Any = None
    observability: BackendObservability | None = None
    memory_repository: Any = None
    memory_maintenance: Any = None
    governed_memory_repository: Any = None
    semantic_response_cache: Any = None
    historical_answer_cache: Any = None
    memory_authority: object | None = None
    harness_kernel: Any = None
    harness_application: Any = None
    harness_completion_candidate_factory: Any = None
    prompt_release_manager: Any = None
    cancellation_registry: WorkflowCancellationRegistry | None = None
    admin_capability_verifier: Any = None
    local_knowledge_base: Any = None
    audit_writer: Any = None

    def __post_init__(self) -> None:
        if self.cancellation_registry is None:
            self.cancellation_registry = WorkflowCancellationRegistry()
        if self.harness_application is not None:
            application_kernel = getattr(self.harness_application, "kernel", None)
            if self.harness_kernel is None or application_kernel is not self.harness_kernel:
                raise ValueError(
                    "Harness application and API kernel must share one host authority"
                )
        if self.observability is None:
            self.observability = BackendObservability.create(
                event_capacity=self.settings.trace_event_capacity,
                maximum_query=self.settings.trace_query_limit,
                maximum_metric_series=self.settings.workflow_metric_series_limit,
            )

    @classmethod
    def create(
        cls,
        settings: BackendSettings | None = None,
        *,
        observability: BackendObservability | None = None,
        harness_kernel: Any = None,
        harness_application: Any = None,
        harness_completion_candidate_factory: Any = None,
        admin_capability_verifier: Any = None,
    ) -> "BackendContainer":
        resolved_settings = (settings or BackendSettings.from_env()).validate()
        configure_structured_logging()
        repository: Any = JobRepository(resolved_settings.database_path)
        if resolved_settings.postgres_dsn:
            try:
                import psycopg
                repository = PostgresJobRepository(
                    lambda: psycopg.connect(resolved_settings.postgres_dsn)
                )
                repository.migrate()
            except Exception as error:
                if resolved_settings.environment in {"production", "prod", "staging"}:
                    raise RuntimeError("configured PostgreSQL job backend is unavailable") from error
        cache: Any = TTLCache(resolved_settings.cache_max_entries, resolved_settings.cache_default_ttl_seconds)
        metrics = MetricsRegistry()
        trace_observability = observability or BackendObservability.create(
            event_capacity=resolved_settings.trace_event_capacity,
            maximum_query=resolved_settings.trace_query_limit,
            maximum_metric_series=resolved_settings.workflow_metric_series_limit,
        )
        message_bus: Any = InMemoryMessageBus()
        if resolved_settings.redis_url:
            try:
                import redis
                from market_agent.backend.redis_adapters import (
                    RedisMessageBusAdapter,
                    RedisJobCache,
                    RedisAuditWriter,
                    RedisStreamMessageBus,
                    RedisTenantCache,
                )
                redis_client = redis.Redis.from_url(resolved_settings.redis_url)
                cache = RedisJobCache(RedisTenantCache(
                    redis_client, tenant_id=resolved_settings.tenant_id,
                    default_ttl_seconds=max(1, int(resolved_settings.cache_default_ttl_seconds))))
                message_bus = RedisMessageBusAdapter(
                    RedisStreamMessageBus(redis_client, tenant_id=resolved_settings.tenant_id)
                )
            except Exception as error:
                if resolved_settings.environment in {"production", "prod", "staging"}:
                    raise RuntimeError("configured Redis backend is unavailable") from error
        task_queue = BackgroundTaskQueue(
            repository=repository,
            cache=cache,
            message_bus=message_bus,
            metrics=metrics,
            max_workers=resolved_settings.task_workers,
            queue_capacity=resolved_settings.task_queue_capacity,
            default_max_attempts=resolved_settings.task_max_attempts,
            retry_delay_seconds=resolved_settings.task_retry_delay_seconds,
            trace_observability=trace_observability,
        )
        if admin_capability_verifier is None and resolved_settings.admin_capability_secret:
            from market_agent.workflow_capabilities import SignedCapabilityVerifier
            admin_capability_verifier = SignedCapabilityVerifier(
                resolved_settings.admin_capability_secret
            )
        if admin_capability_verifier is not None and not callable(getattr(admin_capability_verifier, "authorize", None)):
            raise TypeError("admin capability verifier must expose authorize")
        container = cls(
            settings=resolved_settings,
            repository=repository,
            cache=cache,
            message_bus=message_bus,
            metrics=metrics,
            task_queue=task_queue,
            observability=trace_observability,
            harness_kernel=harness_kernel,
            harness_application=harness_application,
            harness_completion_candidate_factory=harness_completion_candidate_factory,
            cancellation_registry=WorkflowCancellationRegistry(),
            admin_capability_verifier=admin_capability_verifier,
            audit_writer=(
                RedisAuditWriter(getattr(message_bus, "_bus"))
                if resolved_settings.redis_url and hasattr(message_bus, "_bus") else None
            ),
        )
        try:
            from market_agent.backend.memory_maintenance import MemoryMaintenanceScheduler
            from market_agent.backend.agent_service import register_agent_tasks
            from market_agent.workflow_memory_lifecycle import LifecycleWorker
            from market_agent.workflow_memory_sqlite import SQLiteMemoryRepository
            from market_agent.workflow_prompt_config import default_prompt_manager
            from market_agent.local_knowledge_base import LocalKnowledgeBase

            memory_authority = object()
            container.memory_authority = memory_authority
            resolved_settings.memory_database_path.parent.mkdir(parents=True, exist_ok=True)
            container.memory_repository = SQLiteMemoryRepository(
                resolved_settings.memory_database_path, writer_authority=memory_authority)
            if resolved_settings.postgres_dsn:
                import psycopg
                from market_agent.workflow_historical_answer_cache import PostgresHistoricalAnswerCache
                from market_agent.workflow_memory_postgres import PostgresMemoryRepository
                from market_agent.workflow_semantic_cache_postgres import PostgresSemanticRequestCache

                connection_factory = lambda: psycopg.connect(resolved_settings.postgres_dsn)
                container.governed_memory_repository = PostgresMemoryRepository(
                    connection_factory, embedding_dimension=resolved_settings.embedding_dimension,
                    writer_authority=memory_authority)
                container.governed_memory_repository.migrate()
                container.semantic_response_cache = PostgresSemanticRequestCache(
                    connection_factory, embedding_dimension=resolved_settings.embedding_dimension)
                container.semantic_response_cache.migrate()
                container.historical_answer_cache = PostgresHistoricalAnswerCache(
                    connection_factory, embedding_dimension=resolved_settings.embedding_dimension)
                container.historical_answer_cache.migrate()

            cleanup_callbacks = []
            if container.semantic_response_cache is not None:
                cleanup_callbacks.append(lambda now: container.semantic_response_cache.cleanup(now=now))
            if container.historical_answer_cache is not None:
                cleanup_callbacks.append(lambda now: container.historical_answer_cache.cleanup(now=now))
            container.memory_maintenance = MemoryMaintenanceScheduler(
                LifecycleWorker(container.memory_repository), tenant_id=resolved_settings.tenant_id,
                authority=memory_authority,
                interval_seconds=resolved_settings.memory_maintenance_interval_seconds,
                cleanup_callbacks=cleanup_callbacks,
                error_observer=lambda event, error: container.metrics.increment(
                    "market_agent_maintenance_errors_total", labels={"event": event, "kind": type(error).__name__}),
            )
            container.memory_maintenance.start()
            container.prompt_release_manager = default_prompt_manager(
                registry_path=resolved_settings.prompt_registry_path,
                git_root=Path(__file__).resolve().parents[2],
                audit_hook=(
                    __import__("market_agent.backend.redis_adapters", fromlist=["RedisPromptActivationMirror"])
                    .RedisPromptActivationMirror(getattr(message_bus, "_bus"))
                    if resolved_settings.redis_url and hasattr(message_bus, "_bus") else None
                ),
                metric_hook=lambda activation, _pin: container.metrics.increment(
                    "market_agent_prompt_release_actions_total",
                    labels={"action": activation.action},
                ),
            )
            try:
                knowledge_path = resolved_settings.local_knowledge_path
                if not knowledge_path.is_absolute():
                    knowledge_path = Path(__file__).resolve().parents[2] / knowledge_path
                container.local_knowledge_base = LocalKnowledgeBase.from_jsonl(
                    knowledge_path
                )
            except FileNotFoundError:
                # The local provider is optional in development.  Production
                # readiness reports the missing configured provider explicitly.
                container.local_knowledge_base = LocalKnowledgeBase()

            def application_factory():
                from market_agent.workflow_production_application import ProductionWorkflowApplication
                from market_agent.workflow_memory_result_writer import MemoryResultWriter

                result_writer = MemoryResultWriter(
                    repository=(container.governed_memory_repository or container.memory_repository),
                    authority=memory_authority,
                    tenant_id=resolved_settings.tenant_id,
                )

                return ProductionWorkflowApplication.from_backend(
                    settings=resolved_settings,
                    memory_repository=(container.governed_memory_repository
                                       or container.memory_repository),
                    semantic_cache=container.semantic_response_cache,
                    historical_answer_cache=container.historical_answer_cache,
                    prompt_release_manager=container.prompt_release_manager,
                    completion_hook=result_writer.record,
                    local_knowledge_base=container.local_knowledge_base,
                    trace_observability=container.observability,
                    audit_writer=container.audit_writer,
                )

            if container.harness_kernel is not None and container.harness_application is None:
                from market_agent.workflow_harness_application import HarnessWorkflowApplication

                production_application = application_factory()
                container.harness_application = HarnessWorkflowApplication(
                    kernel=container.harness_kernel,
                    run_workflow=lambda request: production_application.run_workflow(
                        request,
                        cancellation_signal=container.cancellation_registry.signal(request.workflow_id),
                    ),
                    run_observed_workflow=lambda request, checkpoint_sink: production_application.execute_workflow(
                        request,
                        cancellation_signal=container.cancellation_registry.signal(request.workflow_id),
                        checkpoint_sink=checkpoint_sink,
                    ),
                    completion_candidate_factory=container.harness_completion_candidate_factory,
                    accepted_result_committer=production_application.commit_accepted_result,
                    cancellation_signal_factory=container.cancellation_registry.signal,
                )

            container.agent_service = register_agent_tasks(
                task_queue,
                application_factory=application_factory,
            )
            if container.harness_application is not None:
                from market_agent.backend.harness_service import HarnessWorkflowService

                container.task_queue.register(
                    "execute_harness_workflow",
                    HarnessWorkflowService(container.harness_application).execute,
                )
        except BaseException:
            container.shutdown()
            raise
        return container

    def readiness(self) -> dict[str, str]:
        try:
            database_status = "ok" if self.repository.healthcheck() else "failed"
        except Exception:
            database_status = "failed"
        cache_stats = getattr(self.cache, "stats", None)
        if callable(cache_stats):
            stats = cache_stats()
            self.metrics.set_gauge("market_agent_cache_entries", stats.size)
            self.metrics.set_gauge("market_agent_cache_hits_total", stats.hits)
            self.metrics.set_gauge("market_agent_cache_misses_total", stats.misses)
            cache_status = "ok"
        else:
            cache_health = getattr(self.cache, "health", lambda: None)()
            cache_status = getattr(cache_health, "status", "failed")
        task_queue_status = "ok" if self.task_queue.is_healthy() else "failed"
        bus_health = getattr(self.message_bus, "health", lambda: None)()
        bus_status = getattr(bus_health, "status", "ok" if isinstance(self.message_bus, InMemoryMessageBus) else "failed")
        components = {"database": database_status, "task_queue": task_queue_status,
                      "cache": cache_status, "message_bus": bus_status}
        if self.settings.postgres_dsn:
            components["postgres"] = self._probe_postgres()
        if self.settings.environment in {"production", "prod", "staging"}:
            components["redis"] = self._probe_redis()
            components["prompt_registry"] = self._probe_prompt_registry()
            components["prompt_activation_state"] = (
                "ok" if self._probe_prompt_registry() == "ok" and self._probe_redis() == "ok" else "failed"
            )
            components["model_configuration"] = self._probe_model_configuration()
            components["local_knowledge"] = (
                "ok" if bool(getattr(self.local_knowledge_base, "configured", False)) else "failed"
            )
            components["completion_evidence_issuer"] = (
                "ok" if callable(self.harness_completion_candidate_factory) else "failed"
            )
            audit_health = getattr(self.audit_writer, "healthy", False)
            components["audit_state"] = "ok" if audit_health else "failed"
            components["shared_job_state"] = "ok" if isinstance(self.repository, PostgresJobRepository) else "failed"
            components["admin_capability"] = (
                "ok" if self.admin_capability_verifier is not None else "failed"
            )
        if self.settings.environment in {"production", "prod", "staging"}:
            components["harness"] = (
                "ok"
                if self.harness_kernel is not None and self.harness_application is not None
                else "failed"
            )
        return components

    def _probe_redis(self) -> str:
        health = getattr(self.message_bus, "health", lambda: None)()
        return "ok" if getattr(health, "status", "failed") == "ok" else "failed"

    def _probe_postgres(self) -> str:
        repository = self.governed_memory_repository
        probe = getattr(repository, "healthcheck", None)
        if callable(probe):
            try:
                return "ok" if probe() else "failed"
            except Exception:
                return "failed"
        return "ok" if repository is not None else "failed"

    def _probe_prompt_registry(self) -> str:
        manager = self.prompt_release_manager
        try:
            return "ok" if manager is not None and manager.current() is not None else "failed"
        except Exception:
            return "failed"

    def _probe_model_configuration(self) -> str:
        api_key = str(os.getenv("OPENAI_API_KEY", "") or "").strip()
        model_ids = (
            self.settings.workflow_sol_model_id,
            self.settings.workflow_terra_model_id,
            self.settings.workflow_luna_model_id,
            self.settings.embedding_model_id,
        )
        versions = (
            self.settings.workflow_sol_model_version,
            self.settings.workflow_terra_model_version,
            self.settings.workflow_luna_model_version,
            self.settings.embedding_model_version,
            self.settings.embedding_vector_version,
            self.settings.prompt_cache_namespace,
        )
        return (
            "ok"
            if api_key
            and all(isinstance(value, str) and value.strip() for value in model_ids + versions)
            else "failed"
        )

    def shutdown(self) -> None:
        self.task_queue.shutdown(wait=True)
        close_bus = getattr(self.message_bus, "close", None)
        if callable(close_bus):
            close_bus()
        if self.memory_maintenance is not None:
            self.memory_maintenance.close()
        if self.memory_repository is not None:
            self.memory_repository.close()
        self.repository.close()
