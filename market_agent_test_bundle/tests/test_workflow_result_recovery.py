from dataclasses import replace

import pytest

from market_agent.backend.errors import RetryableTaskError
from market_agent.workflow_contracts import WorkflowRequest
from market_agent.workflow_harness_contracts import RunState
from market_agent_test_bundle.tests.test_backend_harness_service import (
    _application_for_terminal, _known_result, _payload,
)
from market_agent_test_bundle.tests.test_workflow_production_application import _application


def _setup(tmp_path, monkeypatch):
    from market_agent.workflow_result_store import SqliteWorkflowResultStore

    monkeypatch.setattr(
        "market_agent.workflow_memory_result_writer.verify_committed_transition_receipt",
        lambda _receipt: True,
    )
    store = SqliteWorkflowResultStore(tmp_path / "results.sqlite3")
    application = _application_for_terminal(RunState.SUCCEEDED, [])
    application._result_store = store
    return application, store, WorkflowRequest.model_validate(_payload())


def test_result_is_durable_before_success_transition(tmp_path, monkeypatch):
    application, store, request = _setup(tmp_path, monkeypatch)
    original = application._kernel.advance

    def advance(*args, **kwargs):
        assert store.load(request).execution.result == _known_result()
        return original(*args, **kwargs)

    application._kernel.advance = advance
    execution = application.execute(request)
    assert execution.workflow_result == _known_result()
    assert store.load(request).committed


def test_terminal_redelivery_restores_result_without_model_replay(tmp_path, monkeypatch):
    application, store, request = _setup(tmp_path, monkeypatch)
    first = application.execute(request)
    from market_agent.workflow_result_store import SqliteWorkflowResultStore

    application._result_store = SqliteWorkflowResultStore(tmp_path / "results.sqlite3")
    application._run_observed_workflow = lambda *_: pytest.fail("model replay")
    application._accepted_result_committer = lambda *_: pytest.fail("committed replay")
    replay = application.execute(request)
    assert replay.workflow_result == first.workflow_result
    assert replay.workflow_usage == first.workflow_usage


def test_commit_failure_remains_retryable_and_reuses_proof(tmp_path, monkeypatch):
    application, store, request = _setup(tmp_path, monkeypatch)
    proofs = []

    def commit(_request, _result, proof):
        proofs.append(proof.proof_digest)
        if len(proofs) == 1:
            raise OSError("temporary writer failure")

    application._accepted_result_committer = commit
    with pytest.raises(RetryableTaskError):
        application.execute(request)
    assert application._kernel.finished
    assert not store.load(request).committed
    application._run_observed_workflow = lambda *_: pytest.fail("model replay")
    assert application.execute(request).workflow_result == _known_result()
    assert len(proofs) == 2 and proofs[0] == proofs[1]
    assert store.load(request).committed


def test_prepared_result_survives_crash_before_success(tmp_path, monkeypatch):
    application, store, request = _setup(tmp_path, monkeypatch)
    advance = application._kernel.advance
    application._kernel.advance = lambda *_args, **_kwargs: (_ for _ in ()).throw(
        SystemExit("host crash")
    )
    with pytest.raises(SystemExit):
        application.execute(request)
    assert not store.load(request).committed
    application._kernel.advance = advance
    application._run_observed_workflow = lambda *_: pytest.fail("model replay")
    assert application.execute(request).workflow_result == _known_result()


def test_store_rejects_reused_identity_with_changed_request(tmp_path, monkeypatch):
    application, store, request = _setup(tmp_path, monkeypatch)
    application.execute(request)
    with pytest.raises(ValueError, match="request"):
        store.load(request.model_copy(update={"user_query": "changed question"}))


def test_new_production_instance_recovers_validated_prompt_binding(tmp_path, monkeypatch):
    application, store, request = _setup(tmp_path, monkeypatch)
    application.execute(request)
    saved = store.load(request)
    committed = []
    production = _application(committed)
    dependencies = production._get_dependencies()
    production._dependencies = replace(dependencies, result_store=store)
    production.commit_accepted_result(request, saved.execution.result, saved.proof)
    assert committed == [_known_result()]


def test_storage_failure_cannot_transition_to_success(tmp_path, monkeypatch):
    application, store, request = _setup(tmp_path, monkeypatch)
    monkeypatch.setattr(store, "stage", lambda *_: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(RetryableTaskError):
        application.execute(request)
    assert not application._kernel.finished


def test_journal_is_checked_before_resuming_summarizing_state(tmp_path, monkeypatch):
    application, store, request = _setup(tmp_path, monkeypatch)
    snapshot = application._kernel.snapshot
    application._kernel.snapshot = lambda run_id: snapshot(run_id).model_copy(
        update={"run_state": RunState.SUMMARIZING}
    )
    monkeypatch.setattr(store, "load", lambda *_: (_ for _ in ()).throw(OSError("disk")))
    with pytest.raises(RetryableTaskError):
        application.execute(request)
    assert not application._kernel.finished


def test_container_shares_durable_journal_with_production_and_harness(tmp_path, monkeypatch):
    from market_agent.backend.container import BackendContainer
    from market_agent.backend.settings import BackendSettings
    from market_agent.workflow_harness import HarnessKernel
    from market_agent.workflow_production_application import ProductionWorkflowApplication

    captured = []
    original = ProductionWorkflowApplication.from_backend.__func__

    def from_backend(cls, **kwargs):
        captured.append(kwargs)
        return original(cls, **kwargs)

    monkeypatch.setattr(ProductionWorkflowApplication, "from_backend", classmethod(from_backend))
    container = BackendContainer.create(BackendSettings(
        environment="test", database_path=tmp_path / "jobs.sqlite3",
        memory_database_path=tmp_path / "memory.sqlite3",
        prompt_registry_path=tmp_path / "prompts.sqlite3",
        audit_database_path=tmp_path / "audit.sqlite3",
        redis_url="", postgres_dsn="",
    ), harness_kernel=object.__new__(HarnessKernel))
    try:
        assert captured[0]["result_store"] is container.workflow_result_store
        assert container.harness_application._result_store is container.workflow_result_store
        assert container.workflow_result_store.healthcheck()
    finally:
        container.shutdown()


def test_staged_execution_and_proof_cannot_be_replaced(tmp_path, monkeypatch):
    application, store, request = _setup(tmp_path, monkeypatch)
    application.execute(request)
    saved = store.load(request)
    changed = saved.execution.model_copy(update={"prompt_release_digest": "c" * 64})
    with pytest.raises(ValueError, match="replaced"):
        store.stage(request, changed)
    with pytest.raises(ValueError, match="proof"):
        store.mark_committed(request, saved.proof.model_copy(update={"prompt_release_digest": "d" * 64}))
    assert store.load(request) == saved


def test_partial_memory_commit_replays_without_duplicate_records(tmp_path, monkeypatch):
    from market_agent.workflow_memory_result_writer import MemoryResultWriter
    from market_agent.workflow_memory_sqlite import SQLiteMemoryRepository

    application, store, request = _setup(tmp_path, monkeypatch)
    authority = object()
    repository = SQLiteMemoryRepository(tmp_path / "memory.sqlite3", writer_authority=authority)
    writer = MemoryResultWriter(repository=repository, authority=authority, tenant_id="default")
    append_decision = repository.append_decision
    attempts = []

    def fail_once(*args, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("crash after event write")
        return append_decision(*args, **kwargs)

    monkeypatch.setattr(repository, "append_decision", fail_once)
    application._accepted_result_committer = writer.record
    try:
        with pytest.raises(RetryableTaskError):
            application.execute(request)
        assert len(repository.list_records(tenant_id="default")) == 1
        application._run_observed_workflow = lambda *_: pytest.fail("model replay")
        application.execute(request)
        assert len(repository.list_records(tenant_id="default")) == 3
        assert len(repository.list_audit(tenant_id="default")) == 3
        assert store.load(request).committed
    finally:
        repository.close()


def test_postgres_result_adapter_uses_parameterized_locked_journal(tmp_path, monkeypatch):
    """DB-API contract test only, not live PostgreSQL concurrency validation."""
    import sqlite3
    from market_agent.workflow_result_store import PostgresWorkflowResultStore

    queries = []

    class Cursor:
        def __init__(self, connection):
            self.cursor = connection.cursor()

        def execute(self, sql, values=()):
            queries.append(sql)
            if values:
                assert "%s" in sql and "?" not in sql
            if sql.startswith("SELECT trace_id"):
                assert sql.endswith(" FOR UPDATE")
            self.cursor.execute(sql.replace("%s", "?").removesuffix(" FOR UPDATE"), values)

        def fetchone(self):
            return self.cursor.fetchone()

        def close(self):
            self.cursor.close()

    class Connection:
        def __init__(self):
            self.connection = sqlite3.connect(tmp_path / "pg-contract.sqlite3")

        def cursor(self):
            return Cursor(self.connection)

        def commit(self):
            self.connection.commit()

        def rollback(self):
            self.connection.rollback()

        def close(self):
            self.connection.close()

    application, _, request = _setup(tmp_path, monkeypatch)
    application._result_store = PostgresWorkflowResultStore(Connection, namespace="tenant")
    first = application.execute(request)
    application._result_store = PostgresWorkflowResultStore(Connection, namespace="tenant")
    application._run_observed_workflow = lambda *_: pytest.fail("model replay")
    assert application.execute(request).workflow_result == first.workflow_result
    assert any(sql.startswith("UPDATE market_agent_workflow_results SET committed") for sql in queries)


def test_production_readiness_rejects_unwired_result_journal():
    from types import SimpleNamespace
    from market_agent.backend.container import BackendContainer
    from market_agent.workflow_result_store import PostgresWorkflowResultStore

    store = object.__new__(PostgresWorkflowResultStore)
    store.healthcheck = lambda: True
    container = object.__new__(BackendContainer)
    container.workflow_result_store = store
    container.harness_application = SimpleNamespace(result_store=None)
    assert container._probe_workflow_results() == "failed"
    container.harness_application.result_store = store
    assert container._probe_workflow_results() == "ok"
