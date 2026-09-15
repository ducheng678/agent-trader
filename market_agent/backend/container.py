from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
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
    memory_promotion_scheduler: Any = None
    memory_promotion_cursor_store: Any = None
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
    workflow_result_store: Any = None
    prompt_activation_store: Any = None
    prompt_audit_recovery: Any = None
    cancellation_store: Any = None
    trace_store: Any = None
    memory_candidate_pipeline: Any = None

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
        if observability is None:
            from market_agent.backend.shared_trace_store import (
                PostgresSharedTraceSink, SQLiteSharedTraceSink,
            )
            if isinstance(repository, PostgresJobRepository):
                import psycopg

                trace_store = PostgresSharedTraceSink(
                    lambda: psycopg.connect(resolved_settings.postgres_dsn),
                    namespace=resolved_settings.tenant_id,
                    capacity=resolved_settings.trace_event_capacity,
                    maximum_query=resolved_settings.trace_query_limit,
                )
                trace_store.migrate()
            else:
                job_path = Path(resolved_settings.database_path)
                trace_store = SQLiteSharedTraceSink(
                    job_path.with_name(job_path.stem + ".traces.sqlite3"),
                    namespace=resolved_settings.tenant_id,
                    capacity=resolved_settings.trace_event_capacity,
                    maximum_query=resolved_settings.trace_query_limit,
                )
            trace_observability.sink = trace_store
        else:
            trace_store = trace_observability.sink
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
            trace_store=trace_store,
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
            from market_agent.backend.cancellation_store import (
                PostgresCancellationStore, SQLiteCancellationStore,
            )
            if isinstance(repository, PostgresJobRepository):
                container.cancellation_store = PostgresCancellationStore(
                    lambda: psycopg.connect(resolved_settings.postgres_dsn),
                    namespace=resolved_settings.tenant_id,
                )
                container.cancellation_store.migrate()
            else:
                job_path = Path(resolved_settings.database_path)
                container.cancellation_store = SQLiteCancellationStore(
                    job_path.with_name(job_path.stem + ".cancellations.sqlite3"),
                    namespace=resolved_settings.tenant_id,
                )
            container.cancellation_registry = WorkflowCancellationRegistry(
                container.cancellation_store,
            )
            from market_agent.workflow_result_store import (
                PostgresWorkflowResultStore, SqliteWorkflowResultStore,
            )
            if isinstance(repository, PostgresJobRepository):
                from market_agent.backend.prompt_activation_store import PostgresPromptActivationStore

                container.workflow_result_store = PostgresWorkflowResultStore(
                    lambda: psycopg.connect(resolved_settings.postgres_dsn),
                    namespace=resolved_settings.tenant_id,
                )
                container.prompt_activation_store = PostgresPromptActivationStore(
                    lambda: psycopg.connect(resolved_settings.postgres_dsn),
                    namespace=resolved_settings.tenant_id,
                )
                container.prompt_activation_store.migrate()
            else:
                job_path = Path(resolved_settings.database_path)
                container.workflow_result_store = SqliteWorkflowResultStore(
                    job_path.with_name(job_path.stem + ".workflow_results.sqlite3"),
                    namespace=resolved_settings.tenant_id,
                )
            from market_agent.backend.memory_maintenance import MemoryMaintenanceScheduler
            from market_agent.backend.agent_service import register_agent_tasks
            from market_agent.workflow_memory_lifecycle import LifecycleWorker
            from market_agent.workflow_memory_promotion import PromotionScheduler
            from market_agent.workflow_long_term_memory import KnowledgeRevision, Lifecycle
            from market_agent.workflow_memory_sqlite import SQLiteMemoryRepository
            from market_agent.workflow_prompt_config import default_prompt_manager
            from market_agent.local_knowledge_base import LocalKnowledgeBase
            from market_agent.workflow_audit import AuditStore, AuditWriter

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

            # Keep a durable local audit projection whenever Redis is not the
            # configured audit transport.  The production application and
            # the maintenance worker then share the same append-only writer;
            # promotion decisions cannot disappear into an in-memory callback.
            if container.audit_writer is None:
                resolved_settings.audit_database_path.parent.mkdir(parents=True, exist_ok=True)
                container.audit_writer = AuditWriter(
                    AuditStore(resolved_settings.audit_database_path)
                )

            promotion_repository = (
                container.governed_memory_repository or container.memory_repository
            )
            from market_agent.backend.promotion_cursor_store import (
                PostgresPromotionCursorStore, SQLitePromotionCursorStore,
            )
            if container.governed_memory_repository is not None:
                container.memory_promotion_cursor_store = PostgresPromotionCursorStore(
                    lambda: psycopg.connect(resolved_settings.postgres_dsn),
                )
                container.memory_promotion_cursor_store.migrate()
            else:
                job_path = Path(resolved_settings.database_path)
                container.memory_promotion_cursor_store = SQLitePromotionCursorStore(
                    job_path.with_name(job_path.stem + ".promotion_cursor.sqlite3"),
                )
            from market_agent.workflow_memory_candidate_pipeline import AcceptedResultMemoryPipeline
            from market_agent.workflow_memory_result_writer import MemoryResultWriter

            container.memory_candidate_pipeline = AcceptedResultMemoryPipeline(
                writer=MemoryResultWriter(
                    repository=promotion_repository,
                    authority=memory_authority,
                    tenant_id=resolved_settings.tenant_id,
                ),
                repository=promotion_repository,
                authority=memory_authority,
                tenant_id=resolved_settings.tenant_id,
            )

            def observe_promotion(evaluation: object) -> None:
                # PromotionEvaluation is a closed contract.  Importing the
                # type here keeps backend composition one-way while making
                # malformed host callbacks fail closed.
                from market_agent.workflow_memory_promotion import PromotionEvaluation
                from market_agent.workflow_tracing import TraceContext
                from market_agent.workflow_structured_logging import summarize_payload

                if not isinstance(evaluation, PromotionEvaluation):
                    raise TypeError("promotion observer received an invalid evaluation")
                container.metrics.increment(
                    "market_agent_memory_promotion_evaluations_total",
                    labels={
                        "status": evaluation.status,
                        "reason": evaluation.reason_code,
                    },
                )
                container.observability.record_component(
                    TraceContext(
                        trace_id=evaluation.trace_id,
                        span_id=evaluation.evaluation_id[:16],
                    ),
                    event="memory_completed",
                    status=("succeeded" if evaluation.status == "promoted" else "rejected"),
                    component="memory",
                    workflow_id="memory-promotion",
                    task_id=evaluation.candidate_id,
                    reason={
                        "promoted": "none",
                        "verified_outcome_required": "insufficient_evidence",
                        "expired_or_forgotten": "unavailable",
                        "contradictory_evidence": "invalid_output",
                        "evidence_mismatch": "invalid_output",
                        "repository_rejected": "unavailable",
                        "cancelled": "none",
                    }[evaluation.reason_code],
                    payload=summarize_payload({"reason_code": evaluation.reason_code}),
                )
                # The configured audit writer is the host-owned append-only
                # projection for both request and maintenance traces.  Only
                # bounded identifiers and registry codes are persisted here.
                from market_agent.workflow_audit import (
                    AuditActor, AuditEvent, AuditEventType, AuditOutcome,
                    AuditPayload, AuditReason, AuditStatus,
                )
                reason_map = {
                    "promoted": None,
                    "verified_outcome_required": AuditReason.MISSING_EVIDENCE.value,
                    "expired_or_forgotten": AuditReason.MEMORY_CONTEXT_EXPIRED.value,
                    "contradictory_evidence": AuditReason.UNRESOLVED_CONFLICT.value,
                    "evidence_mismatch": AuditReason.EVALUATION_FAILURE.value,
                    "repository_rejected": AuditReason.EVALUATION_FAILURE.value,
                    "cancelled": AuditReason.CANCELLATION.value,
                }
                reason_code = reason_map[evaluation.reason_code]
                container.audit_writer.record(AuditEvent(
                    event_id="memory-promotion-" + evaluation.evaluation_id[:32],
                    trace_id=evaluation.trace_id,
                    workflow_id="memory-promotion",
                    task_id=evaluation.candidate_id,
                    occurred_at=evaluation.evaluated_at,
                    actor=AuditActor.SCHEDULER.value,
                    event_type=AuditEventType.MEMORY_PROMOTED.value,
                    status=(AuditStatus.PROMOTED.value
                            if evaluation.status == "promoted"
                            else AuditStatus.REJECTED.value),
                    input_hash=evaluation.evaluation_id,
                    source_references=(evaluation.candidate_id,),
                    payload=AuditPayload(
                        kind="selection",
                        subject_ids=(evaluation.candidate_id,),
                        outcome_code=(AuditOutcome.PROMOTED.value
                                      if evaluation.status == "promoted"
                                      else AuditOutcome.REJECTED.value),
                        reason_code=reason_code,
                        item_count=1,
                    ),
                ))

            container.memory_promotion_scheduler = PromotionScheduler(
                repository=promotion_repository,
                authority=memory_authority,
                tenant_id=resolved_settings.tenant_id,
                max_candidates_per_run=10,
                evaluation_observer=observe_promotion,
                cursor_store=container.memory_promotion_cursor_store,
            )

            def run_memory_promotion(epoch: float) -> tuple[object, ...]:
                now = datetime.fromtimestamp(epoch, timezone.utc)
                candidates = tuple(
                    record for record in promotion_repository.list_records(
                        tenant_id=resolved_settings.tenant_id
                    )
                    if isinstance(record, KnowledgeRevision)
                    and record.lifecycle is Lifecycle.PROPOSED
                )
                if not candidates:
                    return ()
                trace_id = sha256(
                    f"memory-promotion:{resolved_settings.tenant_id}:"
                    f"{now.strftime('%Y%m%d%H')}".encode("utf-8")
                ).hexdigest()[:32]
                return container.memory_promotion_scheduler.evaluate(
                    candidates,
                    now=now,
                    trace_id=trace_id,
                )

            cleanup_callbacks = []
            def retry_accepted_candidates(_epoch: float) -> tuple[object, ...]:
                reports = container.memory_candidate_pipeline.retry_pending(max_outcomes=100)
                for report in reports:
                    container.metrics.increment(
                        "market_agent_candidate_extraction_total",
                        labels={"status": report.status},
                    )
                return reports

            cleanup_callbacks.append(retry_accepted_candidates)
            if container.semantic_response_cache is not None:
                cleanup_callbacks.append(lambda now: container.semantic_response_cache.cleanup(now=now))
            if container.historical_answer_cache is not None:
                cleanup_callbacks.append(lambda now: container.historical_answer_cache.cleanup(now=now))
            container.memory_maintenance = MemoryMaintenanceScheduler(
                LifecycleWorker(promotion_repository), tenant_id=resolved_settings.tenant_id,
                authority=memory_authority,
                interval_seconds=resolved_settings.memory_maintenance_interval_seconds,
                cleanup_callbacks=cleanup_callbacks,
                promotion_callback=run_memory_promotion,
                error_observer=lambda event, error: container.metrics.increment(
                    "market_agent_maintenance_errors_total", labels={"event": event, "kind": type(error).__name__}),
            )
            container.memory_maintenance.start()
            container.prompt_release_manager = default_prompt_manager(
                registry_path=resolved_settings.prompt_registry_path,
                activation_store=container.prompt_activation_store,
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
            from market_agent.backend.prompt_audit_recovery import PromptAuditRecoveryScheduler

            container.prompt_audit_recovery = PromptAuditRecoveryScheduler(
                container.prompt_release_manager,
                error_observer=lambda event, error: container.metrics.increment(
                    "market_agent_maintenance_errors_total",
                    labels={"event": event, "kind": type(error).__name__},
                ),
            )
            container.prompt_audit_recovery.start()
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

                return ProductionWorkflowApplication.from_backend(
                    settings=resolved_settings,
                    memory_repository=(container.governed_memory_repository
                                       or container.memory_repository),
                    semantic_cache=container.semantic_response_cache,
                    historical_answer_cache=container.historical_answer_cache,
                    prompt_release_manager=container.prompt_release_manager,
                    completion_hook=container.memory_candidate_pipeline.record,
                    local_knowledge_base=container.local_knowledge_base,
                    trace_observability=container.observability,
                    audit_writer=container.audit_writer,
                    result_store=container.workflow_result_store,
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
                    result_store=container.workflow_result_store,
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
            components["prompt_activation_state"] = self._probe_prompt_activation()
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
            components["workflow_result_state"] = self._probe_workflow_results()
            components["shared_cancellation_state"] = self._probe_shared_cancellation()
            components["shared_trace_state"] = self._probe_shared_trace()
            components["governed_memory_lifecycle"] = self._probe_governed_memory_lifecycle()
            components["shared_promotion_cursor"] = self._probe_shared_promotion_cursor()
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

    def _probe_workflow_results(self) -> str:
        from market_agent.workflow_result_store import PostgresWorkflowResultStore

        try:
            return "ok" if (
                isinstance(self.workflow_result_store, PostgresWorkflowResultStore)
                and getattr(self.harness_application, "result_store", None) is self.workflow_result_store
                and self.workflow_result_store.healthcheck()
            ) else "failed"
        except Exception:
            return "failed"

    def _probe_shared_cancellation(self) -> str:
        from market_agent.backend.cancellation_store import PostgresCancellationStore

        try:
            return "ok" if (
                isinstance(self.cancellation_store, PostgresCancellationStore)
                and self.cancellation_registry.store is self.cancellation_store
                and self.cancellation_store.healthcheck()
            ) else "failed"
        except Exception:
            return "failed"

    def _probe_shared_trace(self) -> str:
        from market_agent.backend.shared_trace_store import PostgresSharedTraceSink

        try:
            return "ok" if (
                isinstance(self.trace_store, PostgresSharedTraceSink)
                and self.observability.sink is self.trace_store
                and self.trace_store.healthcheck()
            ) else "failed"
        except Exception:
            return "failed"

    def _probe_governed_memory_lifecycle(self) -> str:
        from market_agent.workflow_memory_postgres import PostgresMemoryRepository

        try:
            repository = self.governed_memory_repository
            scheduler = self.memory_maintenance
            return "ok" if (
                isinstance(repository, PostgresMemoryRepository)
                and scheduler is not None
                and scheduler._worker._repository is repository
                and scheduler._thread.is_alive()
                and repository.healthcheck()
            ) else "failed"
        except Exception:
            return "failed"

    def _probe_shared_promotion_cursor(self) -> str:
        from market_agent.backend.promotion_cursor_store import PostgresPromotionCursorStore

        try:
            return "ok" if (
                isinstance(self.memory_promotion_cursor_store, PostgresPromotionCursorStore)
                and self.memory_promotion_scheduler._cursor_store is self.memory_promotion_cursor_store
                and self.memory_promotion_cursor_store.healthcheck()
            ) else "failed"
        except Exception:
            return "failed"

    def _probe_prompt_activation(self) -> str:
        from market_agent.backend.prompt_activation_store import PostgresPromptActivationStore

        try:
            return "ok" if (
                isinstance(self.prompt_activation_store, PostgresPromptActivationStore)
                and self.prompt_release_manager.activation_store is self.prompt_activation_store
                and self.prompt_activation_store.healthcheck()
                and self._probe_prompt_registry() == "ok"
            ) else "failed"
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
        if self.prompt_audit_recovery is not None:
            self.prompt_audit_recovery.close()
        close_bus = getattr(self.message_bus, "close", None)
        if callable(close_bus):
            close_bus()
        if self.memory_maintenance is not None:
            self.memory_maintenance.close()
        if self.memory_repository is not None:
            self.memory_repository.close()
        self.repository.close()
