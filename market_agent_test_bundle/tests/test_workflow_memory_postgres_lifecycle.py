"""Lifecycle parity through the PostgreSQL offline DB-API boundary."""
from __future__ import annotations

from datetime import timedelta
import sqlite3

import pytest

from market_agent.workflow_long_term_memory import (
    Lifecycle, MemoryAuthorityError, MemoryConflictError, MemoryIntegrityError,
    Provenance,
)
from market_agent.workflow_memory_lifecycle import LifecycleLimits, LifecyclePolicy, LifecycleScope, LifecycleWorker
from market_agent.workflow_memory_postgres import PostgresMemoryRepository, PostgresMemoryUnavailableError
from market_agent.workflow_memory_sqlite import SQLiteMemoryRepository
from market_agent.workflow_object_store import FileArtifactStore
from market_agent_test_bundle.tests import test_workflow_memory_lifecycle as lifecycle
from market_agent_test_bundle.tests import test_workflow_memory_storage as storage
from market_agent_test_bundle.tests.test_workflow_memory_postgres_integrity import OfflinePostgres, OfflineCursor


@pytest.fixture
def pg_repo(tmp_path):
    database = OfflinePostgres(tmp_path / "postgres-lifecycle.db")
    authority, clock = object(), lifecycle.Clock()
    repository = PostgresMemoryRepository(database.connect, embedding_dimension=3,
                                          writer_authority=authority, clock=clock)
    repository.test_authority, repository.test_clock = authority, clock
    repository.test_database = database
    repository.migrate()
    return repository


@pytest.fixture(params=["sqlite", "postgres"])
def repo(request, tmp_path, pg_repo):
    if request.param == "postgres":
        yield pg_repo
    else:
        authority, clock = object(), lifecycle.Clock()
        with SQLiteMemoryRepository(tmp_path / "sqlite-lifecycle.db", writer_authority=authority, clock=clock) as repository:
            repository.test_authority, repository.test_clock = authority, clock
            yield repository


_SHARED_CASES = [
    (lifecycle.test_plan_is_deterministic_dry_run_and_retention_scoped, False),
    (lifecycle.test_capacity_chooses_lowest_decayed_confidence_deterministically, False),
    (lifecycle.test_reference_and_hold_protection_and_stale_plan_recheck, False),
    (lifecycle.test_apply_requires_authority_tenant_trace_and_original_plan, False),
    (lifecycle.test_shared_artifact_remains_for_another_live_record, True),
    (lifecycle.test_expired_referenced_evidence_is_retained_but_yields_no_memory, False),
    (lifecycle.test_bounded_apply_resumes_remaining_actions_in_the_same_plan, False),
    (lifecycle.test_forged_valid_plan_cannot_skip_active_to_purge, False),
    (lifecycle.test_pending_artifact_cleanup_cannot_race_a_new_attachment, True),
    (lifecycle.test_delayed_apply_starts_each_quarantine_at_actual_transition, False),
]


@pytest.mark.parametrize("contract,uses_tmp_path", [
    pytest.param(contract, uses_tmp_path, id=contract.__name__)
    for contract, uses_tmp_path in _SHARED_CASES
])
def test_shared_lifecycle_contract(repo, tmp_path, contract, uses_tmp_path):
    contract(repo, **({"tmp_path": tmp_path} if uses_tmp_path else {}))


def snapshot(repository):
    with sqlite3.connect(repository.test_database.path) as connection:
        tables = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name LIKE 'governed_memory_%' ORDER BY name"
        ).fetchall()
        return {table: connection.execute(f"SELECT * FROM {table}").fetchall() for (table,) in tables}


def reopen(repository):
    other = PostgresMemoryRepository(repository.test_database.connect, embedding_dimension=3,
                                     writer_authority=repository.test_authority, clock=repository.test_clock)
    other.test_authority, other.test_clock = repository.test_authority, repository.test_clock
    other.test_database = repository.test_database
    return other


def cleanup_context(repository, task, **changes):
    return storage.write(repository, task.task_id, trace_id=task.trace_id, **changes)


def test_pg_purge_scrubs_replay_snapshots_and_blocks_identity_or_payload_resurrection(pg_repo):
    original = pg_repo.append_event(storage.event(payload={"secret": "purged-value"}), **storage.write(pg_repo, "event"))
    service = lifecycle.worker(pg_repo)
    purge = lifecycle.retire(pg_repo, service)
    result = lifecycle.apply(pg_repo, service, purge, "purge")
    assert result.applied_ids == ("event-1",)
    assert pg_repo.get_by_id("event-1", tenant_id="tenant-a") is None
    state = snapshot(pg_repo)
    assert not state["governed_memory_idempotency"]
    assert not state["governed_memory_lifecycle_state"]
    assert "purged-value" not in str(state)
    before = pg_repo.list_audit(tenant_id="tenant-a")
    assert lifecycle.apply(pg_repo, service, purge, "purge").applied_ids == ()
    assert pg_repo.list_audit(tenant_id="tenant-a") == before
    for record, key in ((original, "event"),
                        (original.model_copy(update={"record_id": "resurrection"}), "other")):
        with pytest.raises(MemoryConflictError):
            pg_repo.append_event(record, **storage.write(pg_repo, key))
    # A different tenant still owns an independent identity and payload namespace.
    pg_repo.append_event(original.model_copy(update={"tenant_id": "tenant-b"}),
                         **storage.write(pg_repo, "event", tenant="tenant-b"))


@pytest.mark.parametrize("phase", ["archive", "tombstone", "purge"])
def test_pg_lifecycle_audit_failure_rolls_back_all_tables(pg_repo, phase):
    pg_repo.append_event(storage.event(), **storage.write(pg_repo, "event"))
    service = lifecycle.worker(pg_repo)
    plan = service.plan("tenant-a", now=storage.NOW + timedelta(seconds=100))
    if phase in ("tombstone", "purge"):
        lifecycle.apply(pg_repo, service, plan, "archive")
        plan = service.plan("tenant-a", now=storage.NOW + timedelta(seconds=110))
    if phase == "purge":
        lifecycle.apply(pg_repo, service, plan, "tombstone")
        plan = service.plan("tenant-a", now=storage.NOW + timedelta(seconds=120))
    before = snapshot(pg_repo)
    pg_repo.test_database.fail_audit = "lifecycle_" + phase
    with pytest.raises(PostgresMemoryUnavailableError):
        lifecycle.apply(pg_repo, service, plan, phase)
    assert snapshot(pg_repo) == before
    pg_repo.test_database.fail_audit = None
    assert lifecycle.apply(pg_repo, service, plan, phase).applied_ids == ("event-1",)


def test_pg_outbox_insert_failure_rolls_back_tombstone_and_audit(pg_repo, monkeypatch):
    pg_repo.append_event(storage.event(), **storage.write(pg_repo, "event"))
    service = lifecycle.worker(pg_repo)
    lifecycle.apply(pg_repo, service, service.plan("tenant-a", now=storage.NOW + timedelta(seconds=100)), "archive")
    plan = service.plan("tenant-a", now=storage.NOW + timedelta(seconds=110))
    before = snapshot(pg_repo)
    original_execute = OfflineCursor.execute

    def fail(cursor, operation, parameters=None):
        if operation.startswith("INSERT INTO governed_memory_cleanup "):
            raise OSError("outbox unavailable")
        return original_execute(cursor, operation, parameters)

    monkeypatch.setattr(OfflineCursor, "execute", fail)
    with pytest.raises(PostgresMemoryUnavailableError):
        lifecycle.apply(pg_repo, service, plan, "tombstone")
    assert snapshot(pg_repo) == before


@pytest.mark.parametrize("protection", ["hold", "reference", "hash"])
def test_pg_purge_rechecks_stale_plan_guards(pg_repo, protection):
    original = pg_repo.append_event(storage.event(), **storage.write(pg_repo, "event"))
    service = lifecycle.worker(pg_repo)
    plan = lifecycle.retire(pg_repo, service)
    record = pg_repo.get_by_id(original.record_id, tenant_id="tenant-a")
    if protection in ("hold", "hash"):
        changes = {"legal_hold": True} if protection == "hold" else {"source": "changed-source"}
        pg_repo.test_database.replace_record(record.model_copy(update=changes))
    else:
        pg_repo.append_event(storage.event("late-reference", payload={"ref": 1}, scope="other",
            provenance=Provenance(source_id="system", source_kind="system", independent_group="system",
                                  derived_from=("event-1",))), **storage.write(pg_repo, "reference"))
    result = lifecycle.apply(pg_repo, service, plan, "purge")
    assert result.applied_ids == () and result.skipped_ids == ("event-1",)
    assert pg_repo.get_by_id("event-1", tenant_id="tenant-a") is not None


def test_pg_knowledge_head_is_never_purged_and_decay_is_persisted(pg_repo):
    storage.evidence(pg_repo)
    pg_repo.propose_knowledge(storage.candidate(confidence=0.8), **storage.write(pg_repo, "proposal"))
    policy = LifecyclePolicy(standard_retention_seconds=10000, standard_half_life_seconds=10,
                             min_confidence=0.5, archive_grace_seconds=10, tombstone_grace_seconds=10)
    service = LifecycleWorker(pg_repo, policy=policy)
    plan = service.plan("tenant-a", now=storage.NOW + timedelta(seconds=10))
    assert plan.archive_ids == ("knowledge-1",) and plan.actions[0].reason == "decay"
    lifecycle.apply(pg_repo, service, plan, "archive")
    record = pg_repo.get_by_id("knowledge-1", tenant_id="tenant-a")
    assert record.lifecycle is Lifecycle.ARCHIVED and record.confidence == 0.8
    lifecycle.apply(pg_repo, service, service.plan("tenant-a", now=storage.NOW + timedelta(seconds=20)), "tombstone")
    assert "knowledge-1" not in service.plan("tenant-a", now=storage.NOW + timedelta(seconds=100)).purge_ids


def test_pg_scope_snapshot_retains_references_from_other_scopes(pg_repo):
    pg_repo.append_event(storage.event(scope="target"), **storage.write(pg_repo, "event"))
    pg_repo.append_decision(storage.decision(scope="other"), **storage.write(pg_repo, "decision"))
    entries = pg_repo.lifecycle_snapshot(LifecycleScope(tenant_id="tenant-a", scope="target"))
    assert len(entries) == 1 and entries[0].referenced_by == ("decision-1",)
    plan = lifecycle.worker(pg_repo).plan(LifecycleScope(tenant_id="tenant-a", scope="target"),
                                          now=storage.NOW + timedelta(seconds=100))
    assert plan.actions == ()


def test_pg_stale_first_action_does_not_consume_success_budget(pg_repo):
    for identifier in ("a", "b"):
        pg_repo.append_event(storage.event(identifier, payload={"id": identifier}), **storage.write(pg_repo, identifier))
    service = lifecycle.worker(pg_repo)
    plan = service.plan("tenant-a", now=storage.NOW + timedelta(seconds=100))
    record = pg_repo.get_by_id("a", tenant_id="tenant-a")
    pg_repo.test_database.replace_record(record.model_copy(update={"legal_hold": True}))
    result = lifecycle.apply(pg_repo, service, plan, "batch", max_actions=1)
    assert result.applied_ids == ("b",) and result.skipped_ids == ("a",)


def test_pg_cleanup_reopens_retries_fairly_and_finishes_with_original_context(pg_repo):
    pg_repo.append_event(storage.event(), **storage.write(pg_repo, "event"))
    purge = lifecycle.retire(pg_repo, lifecycle.worker(pg_repo))
    tasks = pg_repo.list_cleanup(tenant_id="tenant-a")
    failed_id, healthy_id = tasks[0].task_id, tasks[1].task_id
    attempts = []

    def clean(task, **context):
        assert context["trace_id"] == task.trace_id and context["idempotency_key"] == task.task_id
        attempts.append(task.task_id)
        if task.task_id == failed_id:
            raise OSError("permanent adapter failure")

    service = lifecycle.worker(pg_repo, cleanup_adapters={"vector": clean, "cache": clean})
    lifecycle.apply(pg_repo, service, purge, "purge", max_cleanup=1)
    assert attempts == [failed_id]
    restored = reopen(pg_repo)
    result = lifecycle.apply(restored, lifecycle.worker(restored, cleanup_adapters={"vector": clean, "cache": clean}),
                             purge, "purge", max_cleanup=1)
    assert attempts == [failed_id, healthy_id]
    assert result.cleaned_ids == (healthy_id,) and result.pending_cleanup == 1
    audit = restored.list_audit(tenant_id="tenant-a")
    healthy = next(task for task in tasks if task.task_id == healthy_id)
    assert not restored.begin_cleanup(healthy, **cleanup_context(restored, healthy))
    restored.finish_cleanup(healthy, **cleanup_context(restored, healthy))
    assert restored.list_audit(tenant_id="tenant-a") == audit


@pytest.mark.parametrize("operation", ["lifecycle_cleanup_attempt", "cleanup_vector"])
def test_pg_cleanup_audit_rollback_preserves_pending_task(pg_repo, operation):
    pg_repo.append_event(storage.event(), **storage.write(pg_repo, "event"))
    lifecycle.retire(pg_repo, lifecycle.worker(pg_repo))
    task = next(task for task in pg_repo.list_cleanup(tenant_id="tenant-a") if task.kind == "vector")
    before = snapshot(pg_repo)
    pg_repo.test_database.fail_audit = operation
    call = pg_repo.begin_cleanup if operation == "lifecycle_cleanup_attempt" else pg_repo.finish_cleanup
    with pytest.raises(PostgresMemoryUnavailableError):
        call(task, **cleanup_context(pg_repo, task))
    assert snapshot(pg_repo) == before
    pg_repo.test_database.fail_audit = None
    call(task, **cleanup_context(pg_repo, task))


@pytest.mark.parametrize("field,value", [("tenant_id", "tenant-b"), ("trace_id", "other"), ("idempotency_key", "other")])
def test_pg_cleanup_requires_original_tenant_trace_and_task(pg_repo, field, value):
    pg_repo.append_event(storage.event(), **storage.write(pg_repo, "event"))
    lifecycle.retire(pg_repo, lifecycle.worker(pg_repo))
    task = pg_repo.list_cleanup(tenant_id="tenant-a")[0]
    context = dict(cleanup_context(pg_repo, task), **{field: value})
    for call in (pg_repo.begin_cleanup, pg_repo.finish_cleanup):
        with pytest.raises(MemoryAuthorityError):
            call(task, **context)


def test_pg_cleanup_checksum_corruption_fails_closed(pg_repo):
    pg_repo.append_event(storage.event(), **storage.write(pg_repo, "event"))
    lifecycle.retire(pg_repo, lifecycle.worker(pg_repo))
    with sqlite3.connect(pg_repo.test_database.path) as connection:
        connection.execute("UPDATE governed_memory_cleanup SET body_hash=?", ("f" * 64,))
    with pytest.raises(MemoryIntegrityError):
        pg_repo.list_cleanup(tenant_id="tenant-a")


def test_pg_supersession_uses_actual_transition_clock(pg_repo):
    storage.evidence(pg_repo)
    pg_repo.propose_knowledge(storage.candidate(), **storage.write(pg_repo, "proposal"))
    pg_repo.activate_knowledge("knowledge-1", expected_revision=1, now=storage.NOW,
                              **storage.write(pg_repo, "activate"))
    pg_repo.propose_knowledge(storage.candidate("knowledge-2", revision=2, lineage_ids=("knowledge-1",)),
                             **storage.write(pg_repo, "proposal-2"))
    pg_repo.test_clock.value = storage.NOW + timedelta(seconds=50)
    pg_repo.activate_knowledge("knowledge-2", expected_revision=2, now=storage.NOW,
                              **storage.write(pg_repo, "activate-2"))
    entries = {entry.record.record_id: entry for entry in pg_repo.lifecycle_snapshot(LifecycleScope(tenant_id="tenant-a"))}
    assert entries["knowledge-1"].changed_at == pg_repo.test_clock.value


def test_pg_each_lifecycle_mutation_uses_tenant_advisory_lock(pg_repo):
    pg_repo.append_event(storage.event(), **storage.write(pg_repo, "event"))
    service = lifecycle.worker(pg_repo)
    purge = lifecycle.retire(pg_repo, service)
    task = pg_repo.list_cleanup(tenant_id="tenant-a")[0]
    operations = [
        lambda: lifecycle.apply(pg_repo, service, purge, "purge", max_cleanup=0),
        lambda: pg_repo.begin_cleanup(task, **cleanup_context(pg_repo, task)),
        lambda: pg_repo.finish_cleanup(task, **cleanup_context(pg_repo, task)),
    ]
    for operation in operations:
        pg_repo.test_database.statements.clear()
        operation()
        sql, parameters = pg_repo.test_database.statements[0]
        assert "pg_advisory_xact_lock" in sql and parameters == ("governed-memory:tenant-a",)


@pytest.mark.parametrize("phase", ["archive", "purge"])
def test_pg_lifecycle_zero_row_cas_rolls_back_every_side_effect(pg_repo, monkeypatch, phase):
    pg_repo.append_event(storage.event(), **storage.write(pg_repo, "event"))
    service = lifecycle.worker(pg_repo)
    plan = (lifecycle.retire(pg_repo, service) if phase == "purge" else
            service.plan("tenant-a", now=storage.NOW + timedelta(seconds=100)))
    before = snapshot(pg_repo)
    original_execute = OfflineCursor.execute
    prefix = "DELETE FROM governed_memory_records" if phase == "purge" else "UPDATE governed_memory_records"

    def lose_cas(cursor, operation, parameters=None):
        if operation.startswith(prefix):
            assert "AND body_hash = %s" in operation
            cursor.forced_rowcount = 0
            return cursor
        return original_execute(cursor, operation, parameters)

    monkeypatch.setattr(OfflineCursor, "execute", lose_cas)
    with pytest.raises(MemoryConflictError):
        lifecycle.apply(pg_repo, service, plan, phase)
    assert snapshot(pg_repo) == before


def test_pg_artifact_cleanup_survives_reopen_and_respects_cleanup_limit(pg_repo, tmp_path):
    store = FileArtifactStore(tmp_path / "artifacts", writer_authority=pg_repo.test_authority)
    reference = store.put(b"retired artifact", **storage.write(pg_repo, "artifact"))
    pg_repo.append_event(storage.event(artifact=reference), **storage.write(pg_repo, "event"))
    service = lifecycle.worker(pg_repo)
    purge = lifecycle.retire(pg_repo, service)
    lifecycle.apply(pg_repo, service, purge, "purge", max_cleanup=0)
    assert {task.kind for task in pg_repo.list_cleanup(tenant_id="tenant-a")} == {"vector", "cache", "artifact"}
    assert store.get(reference, tenant_id="tenant-a") == b"retired artifact"
    restored = reopen(pg_repo)
    calls = []

    def clean(task, **context):
        calls.append(task.task_id)

    resumed = lifecycle.worker(restored, artifact_store=store,
                                cleanup_adapters={"vector": clean, "cache": clean})
    for remaining in (2, 1, 0):
        result = lifecycle.apply(restored, resumed, purge, "purge", max_cleanup=1)
        assert len(result.cleaned_ids) == 1 and result.pending_cleanup == remaining
    with pytest.raises(FileNotFoundError):
        store.get(reference, tenant_id="tenant-a")
    assert len(set(calls)) == 2
    assert sum(item.operation.startswith("cleanup_") for item in restored.list_audit(tenant_id="tenant-a")) == 3
    before = snapshot(restored)
    assert not lifecycle.apply(restored, resumed, purge, "purge", max_cleanup=1).cleaned_ids
    assert snapshot(restored) == before


def test_pg_lifecycle_replay_rejects_changed_trace_and_policy(pg_repo):
    pg_repo.append_event(storage.event(), **storage.write(pg_repo, "event"))
    service = lifecycle.worker(pg_repo)
    plan = service.plan("tenant-a", now=storage.NOW + timedelta(seconds=100))
    lifecycle.apply(pg_repo, service, plan, "archive")
    before = snapshot(pg_repo)
    with pytest.raises(MemoryConflictError):
        service.apply(plan, LifecycleLimits(), **storage.write(pg_repo, "archive", trace_id="other"))
    with pytest.raises(MemoryConflictError):
        pg_repo.apply_lifecycle(plan, LifecyclePolicy(), LifecycleLimits(), **storage.write(pg_repo, "archive"))
    assert snapshot(pg_repo) == before


def test_pg_legacy_archives_without_transition_time_remain_protected(pg_repo):
    original = pg_repo.append_event(storage.event(), **storage.write(pg_repo, "event"))
    pg_repo.test_database.replace_record(original.model_copy(update={"lifecycle": Lifecycle.ARCHIVED}))
    entries = pg_repo.lifecycle_snapshot(LifecycleScope(tenant_id="tenant-a"))
    assert entries[0].changed_at is None
    assert lifecycle.worker(pg_repo).plan("tenant-a", now=storage.NOW + timedelta(days=365)).actions == ()
