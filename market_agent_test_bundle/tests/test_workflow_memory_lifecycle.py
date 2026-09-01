from __future__ import annotations

from datetime import timedelta
import importlib
import json
import sqlite3

import pytest
from pydantic import ValidationError

from market_agent.workflow_long_term_memory import (
    DecisionRecord, Lifecycle, MemoryAuthorityError, MemoryConflictError,
)
from market_agent.workflow_memory_retrieval import MemoryQuery, build_core_experience_summary, retrieve_memory
from market_agent.workflow_object_store import FileArtifactStore
from market_agent.workflow_memory_sqlite import SQLiteMemoryRepository
from test_workflow_memory_storage import NOW, candidate, event, evidence, write


class Clock:
    value = NOW

    def __call__(self):
        return self.value


@pytest.fixture
def repo(tmp_path):
    clock, authority = Clock(), object()
    with SQLiteMemoryRepository(tmp_path / "memory.db", writer_authority=authority, clock=clock) as repository:
        repository.test_authority, repository.test_clock = authority, clock
        yield repository


def api():
    assert importlib.util.find_spec("market_agent.workflow_memory_lifecycle") is not None, "LifecycleWorker is not implemented"
    return importlib.import_module("market_agent.workflow_memory_lifecycle")


def worker(repository, **changes):
    return api().LifecycleWorker(repository, policy=api().LifecyclePolicy(
        standard_retention_seconds=100, short_retention_seconds=10,
        standard_half_life_seconds=100, short_half_life_seconds=10,
        archive_grace_seconds=10, tombstone_grace_seconds=10,
        min_confidence=0.0, max_live_records=100), **changes)


def apply(repository, service, plan, key, **changes):
    repository.test_clock.value = plan.now
    return service.apply(plan, api().LifecycleLimits(**changes), **write(repository, key))


def retire(repository, service, identifier="event-1"):
    plan = service.plan("tenant-a", now=NOW + timedelta(seconds=100))
    assert identifier in plan.archive_ids
    apply(repository, service, plan, "archive")
    plan = service.plan("tenant-a", now=NOW + timedelta(seconds=110))
    assert identifier in plan.tombstone_ids
    apply(repository, service, plan, "tombstone")
    return service.plan("tenant-a", now=NOW + timedelta(seconds=120))


def test_half_life_decay_preserves_original_confidence_and_permanent_records():
    policy = api().LifecyclePolicy(standard_half_life_seconds=100)
    record = candidate(confidence=0.8)
    assert api().effective_confidence(record, NOW, policy) == pytest.approx(0.8)
    assert api().effective_confidence(record, NOW + timedelta(seconds=100), policy) == pytest.approx(0.4)
    assert api().effective_confidence(record, NOW + timedelta(seconds=200), policy) == pytest.approx(0.2)
    assert api().effective_confidence(record.model_copy(update={"retention_class": "permanent"}),
                                      NOW + timedelta(days=500), policy) == pytest.approx(0.8)
    assert record.confidence == 0.8
    with pytest.raises(ValidationError):
        api().effective_confidence(record, NOW.replace(tzinfo=None), policy)


def test_plan_is_deterministic_dry_run_and_retention_scoped(repo):
    repo.append_event(event(retention_class="short"), **write(repo, "event"))
    repo.append_event(event("permanent", payload={"id": 2}, retention_class="permanent"), **write(repo, "permanent"))
    repo.append_event(event("foreign", tenant_id="tenant-b", payload={"id": 3}), **write(repo, "foreign", tenant="tenant-b"))
    before = repo.list_audit(tenant_id="tenant-a")
    service = worker(repo)
    plan = service.plan("tenant-a", now=NOW + timedelta(seconds=10))
    assert plan == service.plan("tenant-a", now=NOW + timedelta(seconds=10))
    assert plan.archive_ids == ("event-1",)
    assert not plan.tombstone_ids and not plan.purge_ids
    assert repo.get_by_id("event-1", tenant_id="tenant-a").lifecycle is Lifecycle.ACTIVE
    assert repo.list_audit(tenant_id="tenant-a") == before


def test_capacity_chooses_lowest_decayed_confidence_deterministically(repo):
    evidence(repo)
    for identifier, confidence in (("low", 0.2), ("high", 0.9)):
        repo.propose_knowledge(candidate(identifier, knowledge_id=identifier, confidence=confidence),
                               **write(repo, identifier))
    service = api().LifecycleWorker(repo, policy=api().LifecyclePolicy(max_live_records=3, min_confidence=0.0))
    plan = service.plan("tenant-a", now=NOW)
    assert plan.archive_ids == ("low",)
    assert plan.actions[0].reason == "capacity"


def test_retirement_advances_one_phase_with_grace_and_idempotent_audit(repo):
    original = repo.append_event(event(payload={"api_key": "secret-value"}), **write(repo, "event"))
    service = worker(repo)
    plan = service.plan("tenant-a", now=NOW + timedelta(seconds=100))
    first = apply(repo, service, plan, "archive")
    assert first.applied_ids == ("event-1",)
    assert repo.get_by_id("event-1", tenant_id="tenant-a").payload_hash == original.payload_hash
    assert service.plan("tenant-a", now=NOW + timedelta(seconds=109)).tombstone_ids == ()
    before = repo.list_audit(tenant_id="tenant-a")
    apply(repo, service, plan, "archive")
    assert repo.list_audit(tenant_id="tenant-a") == before
    apply(repo, service, service.plan("tenant-a", now=NOW + timedelta(seconds=110)), "tombstone")
    assert service.plan("tenant-a", now=NOW + timedelta(seconds=119)).purge_ids == ()
    final = service.plan("tenant-a", now=NOW + timedelta(seconds=120))
    assert final.purge_ids == ("event-1",)
    apply(repo, service, final, "purge")
    assert repo.get_by_id("event-1", tenant_id="tenant-a") is None
    audit = repo.list_audit(tenant_id="tenant-a")
    assert [entry.operation for entry in audit] == ["append_event", "lifecycle_archive", "lifecycle_tombstone", "lifecycle_purge"]
    assert {entry.trace_id for entry in audit} == {"trace-1"}
    assert "secret-value" not in json.dumps([entry.model_dump(mode="json") for entry in audit])
    # Purging only memory_records would leave replay snapshots containing raw material.
    with sqlite3.connect(repo.path) as db:
        assert not any("secret-value" in row[0] for row in db.execute("SELECT result FROM memory_idempotency"))
    apply(repo, service, final, "purge")
    assert repo.list_audit(tenant_id="tenant-a") == audit
    with pytest.raises(MemoryConflictError):
        repo.append_event(original, **write(repo, "event"))
    with pytest.raises(MemoryConflictError):
        repo.append_event(original.model_copy(update={"record_id": "resurrection"}), **write(repo, "resurrection"))


def test_reference_and_hold_protection_and_stale_plan_recheck(repo):
    repo.append_event(event(), **write(repo, "event"))
    repo.append_event(event("held", payload={"id": "held"}, legal_hold=True), **write(repo, "held"))
    service = worker(repo)
    stale = service.plan("tenant-a", now=NOW + timedelta(seconds=100))
    repo.append_decision(DecisionRecord(record_id="decision", tenant_id="tenant-a", observed_at=NOW,
                                        decision="no_trade", status="final", evidence_ids=("event-1",),
                                        retention_class="permanent"), **write(repo, "decision"))
    result = apply(repo, service, stale, "archive")
    assert result.applied_ids == () and result.skipped_ids == ("event-1",)
    for elapsed in (100, 1000, 10000):
        plan = service.plan("tenant-a", now=NOW + timedelta(seconds=elapsed))
        assert not plan.archive_ids and not plan.tombstone_ids and not plan.purge_ids
    assert repo.get_by_id("event-1", tenant_id="tenant-a").lifecycle is Lifecycle.ACTIVE


def test_apply_requires_authority_tenant_trace_and_original_plan(repo):
    repo.append_event(event(), **write(repo, "event"))
    service = worker(repo)
    plan = service.plan("tenant-a", now=NOW + timedelta(seconds=100))
    for changes in (dict(authority=object()), dict(tenant_id="tenant-b"), dict(trace_id="")):
        with pytest.raises((MemoryAuthorityError, ValidationError)):
            service.apply(plan, api().LifecycleLimits(), **write(repo, "apply", **changes))
    with pytest.raises(ValidationError):
        plan.model_copy(update={"now": NOW + timedelta(seconds=1000)})
    assert repo.get_by_id("event-1", tenant_id="tenant-a").lifecycle is Lifecycle.ACTIVE


def test_audit_failure_rolls_back_phase_and_outbox(repo):
    repo.append_event(event(), **write(repo, "event"))
    service = worker(repo)
    apply(repo, service, service.plan("tenant-a", now=NOW + timedelta(seconds=100)), "archive")
    plan = service.plan("tenant-a", now=NOW + timedelta(seconds=110))
    with sqlite3.connect(repo.path) as db:
        db.execute("CREATE TRIGGER fail_lifecycle BEFORE INSERT ON memory_audit WHEN NEW.operation='lifecycle_tombstone' BEGIN SELECT RAISE(ABORT, 'audit unavailable'); END")
    with pytest.raises(sqlite3.IntegrityError):
        apply(repo, service, plan, "tombstone")
    assert repo.get_by_id("event-1", tenant_id="tenant-a").lifecycle is Lifecycle.ARCHIVED
    assert repo.list_cleanup(tenant_id="tenant-a") == ()


def test_cleanup_outbox_retries_across_reopen_and_is_bounded(repo, tmp_path):
    from market_agent.workflow_memory_sqlite import SQLiteMemoryRepository
    authority = repo.test_authority
    store = FileArtifactStore(tmp_path / "objects", writer_authority=authority)
    ref = store.put(b"private evidence", **write(repo, "artifact"))
    repo.append_event(event(artifact=ref), **write(repo, "event"))
    service = worker(repo, artifact_store=store)
    purge = retire(repo, service)
    apply(repo, service, purge, "purge", max_cleanup=0)
    assert store.get(ref, tenant_id="tenant-a") == b"private evidence"
    pending = repo.list_cleanup(tenant_id="tenant-a")
    assert {item.kind for item in pending} == {"vector", "cache", "artifact"}
    # Local adapters exercise real deletion and idempotency without network services.
    derivative_files = {}
    for kind in ("vector", "cache"):
        path = tmp_path / kind
        path.write_text("event-1", encoding="utf-8")
        derivative_files[kind] = path
    def clean(item, **context):
        assert context["trace_id"] == "trace-1"
        derivative_files[item.kind].unlink(missing_ok=True)
    with SQLiteMemoryRepository(repo.path, writer_authority=authority, clock=repo.test_clock) as reopened:
        reopened.test_authority = authority
        reopened.test_clock = repo.test_clock
        restored = worker(reopened, artifact_store=store, cleanup_adapters={"vector": clean, "cache": clean})
        result = apply(reopened, restored, purge, "purge", max_cleanup=1)
        assert len(result.cleaned_ids) == 1
        for _ in range(3):
            apply(reopened, restored, purge, "purge", max_cleanup=1)
        assert reopened.list_cleanup(tenant_id="tenant-a") == ()
        audit = reopened.list_audit(tenant_id="tenant-a")
        apply(reopened, restored, purge, "purge", max_cleanup=1)
        assert reopened.list_audit(tenant_id="tenant-a") == audit
        assert len([entry for entry in audit if entry.operation.startswith("cleanup_")]) == 3
    assert not any(path.exists() for path in derivative_files.values())
    with pytest.raises(FileNotFoundError):
        store.get(ref, tenant_id="tenant-a")


def test_shared_artifact_remains_for_another_live_record(repo, tmp_path):
    store = FileArtifactStore(tmp_path / "objects", writer_authority=repo.test_authority)
    ref = store.put(b"shared", **write(repo, "artifact"))
    repo.append_event(event(artifact=ref), **write(repo, "event"))
    repo.append_event(event("held", payload={"id": "held"}, artifact=ref, legal_hold=True), **write(repo, "held"))
    service = worker(repo, artifact_store=store)
    apply(repo, service, retire(repo, service), "purge")
    assert store.get(ref, tenant_id="tenant-a") == b"shared"
    assert "artifact" not in {item.kind for item in repo.list_cleanup(tenant_id="tenant-a")}


def test_expired_referenced_evidence_is_retained_but_yields_no_memory(repo):
    for number in (1, 2):
        repo.append_event(event(f"event-{number}", source=str(number), payload={"id": number},
                                expires_at=NOW + timedelta(seconds=1)), **write(repo, f"event-{number}"))
    repo.propose_knowledge(candidate(), **write(repo, "proposal"))
    repo.activate_knowledge("knowledge-1", expected_revision=1, now=NOW, **write(repo, "activate"))
    service = worker(repo)
    now = NOW + timedelta(seconds=2)
    assert service.plan("tenant-a", now=now).purge_ids == ()
    result = retrieve_memory(MemoryQuery(tenant_id="tenant-a", task="Check funding before entry.",
                                         applicability=("BTC",), now=now), repo)
    assert result.status == "miss" and result.omissions == ("evidence_gap",)
    assert build_core_experience_summary(result, 5000).as_dynamic_context() == ""


def test_bounded_apply_resumes_remaining_actions_in_the_same_plan(repo):
    for identifier in ("a", "b", "c"):
        repo.append_event(event(identifier, payload={"id": identifier}), **write(repo, identifier))
    service = worker(repo)
    plan = service.plan("tenant-a", now=NOW + timedelta(seconds=100))
    for expected in ("a", "b", "c"):
        result = apply(repo, service, plan, "batch", max_actions=1)
        assert result.applied_ids == (expected,)
    assert all(record.lifecycle is Lifecycle.ARCHIVED for record in repo.list_records(tenant_id="tenant-a"))
    assert apply(repo, service, plan, "batch", max_actions=1).applied_ids == ()


def test_stale_first_action_does_not_starve_later_work_in_a_bounded_plan(repo):
    from test_workflow_memory_storage import replace_stored_record
    for identifier in ("a", "b"):
        repo.append_event(event(identifier, payload={"id": identifier}), **write(repo, identifier))
    service = worker(repo)
    plan = service.plan("tenant-a", now=NOW + timedelta(seconds=100))
    record = repo.get_by_id("a", tenant_id="tenant-a")
    replace_stored_record(repo, record.model_copy(update={"legal_hold": True}))
    result = apply(repo, service, plan, "batch", max_actions=1)
    assert result.applied_ids == ("b",) and result.skipped_ids == ("a",)


@pytest.mark.parametrize("protection", ["hold", "reference"])
def test_purge_rechecks_protection_added_after_dry_run(repo, protection):
    from test_workflow_memory_storage import replace_stored_record
    repo.append_event(event(), **write(repo, "event"))
    service = worker(repo)
    plan = retire(repo, service)
    if protection == "hold":
        record = repo.get_by_id("event-1", tenant_id="tenant-a")
        replace_stored_record(repo, record.model_copy(update={"legal_hold": True}))
    else:
        from market_agent.workflow_long_term_memory import Provenance
        repo.append_event(event("late-reference", payload={"ref": 1},
            provenance=Provenance(source_id="system", source_kind="system", independent_group="system",
                                  derived_from=("event-1",))), **write(repo, "reference"))
    result = apply(repo, service, plan, "purge")
    assert result.applied_ids == () and result.skipped_ids == ("event-1",)
    assert repo.get_by_id("event-1", tenant_id="tenant-a") is not None


def test_forged_valid_plan_cannot_skip_active_to_purge(repo):
    from market_agent.workflow_long_term_memory import content_hash
    record = repo.append_event(event(), **write(repo, "event"))
    service = worker(repo)
    plan = service.plan("tenant-a", now=NOW + timedelta(seconds=100))
    action = api().LifecycleAction(record_id="event-1", expected_hash=content_hash(record.model_dump(mode="json")),
                                    kind="purge", reason="tombstone_grace")
    forged = plan.model_copy(update={"actions": (action,), "plan_hash": None})
    result = apply(repo, service, forged, "forged")
    assert result.applied_ids == ()
    assert repo.get_by_id("event-1", tenant_id="tenant-a").lifecycle is Lifecycle.ACTIVE


def test_pending_artifact_cleanup_cannot_race_a_new_attachment(repo, tmp_path):
    store = FileArtifactStore(tmp_path / "objects", writer_authority=repo.test_authority)
    ref = store.put(b"race-sensitive", **write(repo, "artifact"))
    repo.append_event(event(artifact=ref), **write(repo, "event"))
    service = worker(repo)
    apply(repo, service, retire(repo, service), "purge")
    with pytest.raises(MemoryConflictError):
        repo.append_event(event("late", payload={"new": 1}, artifact=ref), **write(repo, "late"))
    assert store.get(ref, tenant_id="tenant-a") == b"race-sensitive"


def test_failed_cleanup_retries_and_cleanup_audit_failure_is_idempotent(repo, tmp_path):
    repo.append_event(event(), **write(repo, "event"))
    service = worker(repo)
    plan = retire(repo, service)
    path = tmp_path / "vector"
    path.write_text("derived evidence", encoding="utf-8")
    def unavailable(task, **context):
        raise OSError("adapter unavailable")
    failing = worker(repo, cleanup_adapters={"vector": unavailable})
    result = apply(repo, failing, plan, "purge")
    assert result.cleaned_ids == () and result.pending_cleanup == 2
    def delete(task, **context):
        path.unlink(missing_ok=True)
    restored = worker(repo, cleanup_adapters={"vector": delete})
    with sqlite3.connect(repo.path) as db:
        db.execute("CREATE TRIGGER fail_cleanup BEFORE INSERT ON memory_audit WHEN NEW.operation='cleanup_vector' BEGIN SELECT RAISE(ABORT, 'audit unavailable'); END")
    with pytest.raises(sqlite3.IntegrityError):
        apply(repo, restored, plan, "purge")
    assert not path.exists() and len(repo.list_cleanup(tenant_id="tenant-a")) == 2
    with sqlite3.connect(repo.path) as db:
        db.execute("DROP TRIGGER fail_cleanup")
    result = apply(repo, restored, plan, "purge")
    assert len(result.cleaned_ids) == 1 and result.pending_cleanup == 1


def test_delayed_apply_starts_each_quarantine_at_actual_transition(repo):
    repo.append_event(event(), **write(repo, "event"))
    service = worker(repo)
    actual = repo.test_clock.value = NOW + timedelta(seconds=1000)
    archive = service.plan("tenant-a", now=NOW + timedelta(seconds=100))
    service.apply(archive, api().LifecycleLimits(), **write(repo, "archive"))
    assert repo.lifecycle_snapshot(api().LifecycleScope(tenant_id="tenant-a"))[0].changed_at == actual
    assert not service.plan("tenant-a", now=actual + timedelta(seconds=9)).tombstone_ids
    tombstone = service.plan("tenant-a", now=actual + timedelta(seconds=10))
    actual += timedelta(seconds=1000)
    repo.test_clock.value = actual
    service.apply(tombstone, api().LifecycleLimits(), **write(repo, "tombstone"))
    assert repo.lifecycle_snapshot(api().LifecycleScope(tenant_id="tenant-a"))[0].changed_at == actual
    assert not service.plan("tenant-a", now=actual + timedelta(seconds=9)).purge_ids
    purge = service.plan("tenant-a", now=actual + timedelta(seconds=10))
    with pytest.raises(MemoryConflictError):
        service.apply(purge, api().LifecycleLimits(), **write(repo, "future-purge"))
    assert repo.get_by_id("event-1", tenant_id="tenant-a").lifecycle is Lifecycle.TOMBSTONED
    actual += timedelta(seconds=100)
    repo.test_clock.value = actual
    service.apply(purge, api().LifecycleLimits(), **write(repo, "purge"))
    assert repo.get_by_id("event-1", tenant_id="tenant-a") is None


def test_failed_first_cleanup_cannot_starve_healthy_tasks_across_reopen(repo, tmp_path):
    from market_agent.workflow_memory_sqlite import SQLiteMemoryRepository
    repo.append_event(event(), **write(repo, "event"))
    service = worker(repo)
    purge = retire(repo, service)
    tasks = repo.list_cleanup(tenant_id="tenant-a")
    failed_id, healthy_id = tasks[0].task_id, tasks[1].task_id
    derivative = tmp_path / "healthy-derivative"
    derivative.write_text("pending", encoding="utf-8")
    attempts = []
    def clean(task, **context):
        attempts.append(task.task_id)
        if task.task_id == failed_id:
            raise OSError("persistent backend failure")
        derivative.unlink(missing_ok=True)
    service = worker(repo, cleanup_adapters={"vector": clean, "cache": clean})
    apply(repo, service, purge, "purge", max_cleanup=1)
    assert attempts == [failed_id] and derivative.exists()
    with SQLiteMemoryRepository(repo.path, writer_authority=repo.test_authority, clock=repo.test_clock) as reopened:
        reopened.test_authority = repo.test_authority
        reopened.test_clock = repo.test_clock
        resumed = worker(reopened, cleanup_adapters={"vector": clean, "cache": clean})
        result = apply(reopened, resumed, purge, "purge", max_cleanup=1)
        assert result.cleaned_ids == (healthy_id,)
        assert result.pending_cleanup == 1 and not derivative.exists()
        assert attempts == [failed_id, healthy_id]
        apply(reopened, resumed, purge, "purge", max_cleanup=1)
        assert attempts == [failed_id, healthy_id, failed_id]
