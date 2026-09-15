from types import SimpleNamespace

from market_agent.backend.cancellation_store import PostgresCancellationStore, SQLiteCancellationStore
from market_agent.backend.container import BackendContainer
from market_agent.backend.settings import BackendSettings
from market_agent.backend.shared_trace_store import PostgresSharedTraceSink, SQLiteSharedTraceSink
from market_agent.backend.promotion_cursor_store import PostgresPromotionCursorStore, SQLitePromotionCursorStore
from market_agent.workflow_tracing import TraceContext
from market_agent.workflow_memory_candidate_pipeline import AcceptedResultMemoryPipeline


def test_container_replicas_share_durable_cancellation(tmp_path):
    settings = BackendSettings(database_path=tmp_path / "jobs.sqlite3", environment="test")
    first = BackendContainer.create(settings)
    second = BackendContainer.create(settings)
    try:
        assert isinstance(first.cancellation_store, SQLiteCancellationStore)
        assert first.cancellation_registry.store is first.cancellation_store
        assert second.cancellation_registry.store is second.cancellation_store
        observed = second.cancellation_registry.signal("workflow-1")
        assert not observed.is_cancelled()
        first.cancellation_registry.cancel("workflow-1")
        assert observed.is_cancelled()
    finally:
        first.shutdown()
        second.shutdown()


def test_production_readiness_requires_healthy_shared_cancellation():
    container = object.__new__(BackendContainer)
    store = object.__new__(PostgresCancellationStore)
    store.healthcheck = lambda: True
    container.cancellation_store = store
    container.cancellation_registry = SimpleNamespace(store=None)
    assert container._probe_shared_cancellation() == "failed"
    container.cancellation_registry.store = store
    assert container._probe_shared_cancellation() == "ok"
    store.healthcheck = lambda: False
    assert container._probe_shared_cancellation() == "failed"


def test_container_replicas_share_durable_trace_history(tmp_path):
    settings = BackendSettings(database_path=tmp_path / "jobs.sqlite3", environment="test")
    first = BackendContainer.create(settings)
    second = BackendContainer.create(settings)
    trace = TraceContext(trace_id="7" * 32, span_id="8" * 16)
    try:
        assert isinstance(first.observability.sink, SQLiteSharedTraceSink)
        assert isinstance(second.observability.sink, SQLiteSharedTraceSink)
        first.observability.record_component(trace, event="request_started", status="started", component="ingress")
        assert [item.event.event for item in second.observability.query(trace.trace_id).items] == ["request_started"]
    finally:
        first.shutdown()
        second.shutdown()


def test_production_probe_rejects_unwired_or_unhealthy_shared_trace():
    container = object.__new__(BackendContainer)
    store = object.__new__(PostgresSharedTraceSink)
    store.healthcheck = lambda: True
    container.trace_store = store
    container.observability = SimpleNamespace(sink=None)
    assert container._probe_shared_trace() == "failed"
    container.observability.sink = store
    assert container._probe_shared_trace() == "ok"
    store.healthcheck = lambda: False
    assert container._probe_shared_trace() == "failed"


def test_container_candidate_pipeline_is_same_repository_as_production_hook(tmp_path, monkeypatch):
    from market_agent.workflow_production_application import ProductionWorkflowApplication

    observed = {}
    monkeypatch.setattr(ProductionWorkflowApplication, "from_backend", classmethod(
        lambda _cls, **kwargs: observed.update(kwargs) or object()
    ))
    settings = BackendSettings(database_path=tmp_path / "jobs.sqlite3", environment="test")
    container = BackendContainer.create(settings)
    try:
        assert isinstance(container.memory_candidate_pipeline, AcceptedResultMemoryPipeline)
        assert container.memory_candidate_pipeline._repository is container.memory_repository
        container.agent_service._engine_factory()
        assert observed["completion_hook"].__self__ is container.memory_candidate_pipeline
        assert observed["memory_repository"] is container.memory_candidate_pipeline._repository
        assert container.memory_maintenance._worker._repository is container.memory_candidate_pipeline._repository
        assert isinstance(container.memory_promotion_cursor_store, SQLitePromotionCursorStore)
        assert container.memory_promotion_scheduler._cursor_store is container.memory_promotion_cursor_store
    finally:
        container.shutdown()


def test_production_lifecycle_probe_requires_same_healthy_pg_memory_repository():
    from market_agent.workflow_memory_postgres import PostgresMemoryRepository

    container = object.__new__(BackendContainer)
    store = object.__new__(PostgresMemoryRepository)
    store.healthcheck = lambda: True
    container.governed_memory_repository = store
    container.memory_maintenance = SimpleNamespace(
        _worker=SimpleNamespace(_repository=None),
        _thread=SimpleNamespace(is_alive=lambda: True),
    )
    assert container._probe_governed_memory_lifecycle() == "failed"
    container.memory_maintenance._worker._repository = store
    assert container._probe_governed_memory_lifecycle() == "ok"
    store.healthcheck = lambda: False
    assert container._probe_governed_memory_lifecycle() == "failed"


def test_production_promotion_cursor_probe_requires_pg_store_and_scheduler_binding():
    container = object.__new__(BackendContainer)
    store = object.__new__(PostgresPromotionCursorStore)
    store.healthcheck = lambda: True
    container.memory_promotion_cursor_store = store
    container.memory_promotion_scheduler = SimpleNamespace(_cursor_store=None)
    assert container._probe_shared_promotion_cursor() == "failed"
    container.memory_promotion_scheduler._cursor_store = store
    assert container._probe_shared_promotion_cursor() == "ok"
    store.healthcheck = lambda: False
    assert container._probe_shared_promotion_cursor() == "failed"
