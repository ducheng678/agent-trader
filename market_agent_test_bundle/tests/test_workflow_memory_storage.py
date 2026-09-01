from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import hashlib
import sqlite3

import pytest
from pydantic import ValidationError

from market_agent.workflow_long_term_memory import (
    ArtifactReference, DecisionLesson, DecisionRecord, EventRecord,
    KnowledgeRevision, Lifecycle, MemoryAuthorityError, MemoryConflictError,
    MemoryIntegrityError, MemoryPromotionError, OutcomeRecord, Provenance,
)
from market_agent.workflow_memory_sqlite import SQLiteMemoryRepository
from market_agent.workflow_object_store import FileArtifactStore


NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


def event(record_id="event-1", tenant_id="tenant-a", source="exchange", **changes):
    values = dict(record_id=record_id, tenant_id=tenant_id, observed_at=NOW,
                  source=source, payload={"price": 100},
                  provenance=Provenance(source_id=source, source_kind="external",
                                        independent_group=source))
    values.update(changes)
    return EventRecord(**values)


def candidate(record_id="knowledge-1", **changes):
    values = dict(record_id=record_id, tenant_id="tenant-a", observed_at=NOW,
                  knowledge_id="rule-1", revision=1, rule="Check funding before entry.",
                  applicability=("BTC",), confidence=0.8, effective_at=NOW,
                  evidence_ids=("event-1", "event-2"))
    values.update(changes)
    return KnowledgeRevision(**values)


@pytest.fixture
def repo(tmp_path):
    authority = object()
    repository = SQLiteMemoryRepository(tmp_path / "memory.db", writer_authority=authority)
    repository.test_authority = authority
    yield repository
    repository.close()


def write(repo, key, tenant="tenant-a", **changes):
    values = dict(tenant_id=tenant, trace_id="trace-1", idempotency_key=key,
                  authority=repo.test_authority)
    values.update(changes)
    return values


def evidence(repo):
    repo.append_event(event(), **write(repo, "event-1"))
    repo.append_event(event("event-2", source="independent", payload={"price": 101}),
                      **write(repo, "event-2"))


def test_event_idempotency_does_not_duplicate_audit_truth(repo):
    first = repo.append_event(event(), **write(repo, "one"))
    assert repo.append_event(event(), **write(repo, "one")) == first
    assert repo.get_by_id("event-1", tenant_id="tenant-a") == first
    assert len(repo.list_audit(tenant_id="tenant-a")) == 1


def test_knowledge_requires_existing_same_tenant_event_evidence(repo):
    with pytest.raises(MemoryPromotionError):
        repo.propose_knowledge(candidate(evidence_ids=("missing",)), **write(repo, "proposal"))
    repo.append_event(event(tenant_id="tenant-b"), **write(repo, "event", tenant="tenant-b"))
    with pytest.raises(MemoryPromotionError):
        repo.propose_knowledge(candidate(evidence_ids=("event-1",)), **write(repo, "proposal"))
    assert repo.get_by_id("knowledge-1", tenant_id="tenant-a") is None


def test_activation_is_compare_and_set_and_has_separate_audit(repo):
    evidence(repo)
    proposed = repo.propose_knowledge(candidate(), **write(repo, "proposal"))
    assert proposed.lifecycle is Lifecycle.PROPOSED
    active = repo.activate_knowledge("knowledge-1", expected_revision=1, now=NOW,
                                     **write(repo, "activation"))
    assert active.lifecycle is Lifecycle.ACTIVE
    assert repo.activate_knowledge("knowledge-1", expected_revision=1, now=NOW,
                                   **write(repo, "activation")) == active
    with pytest.raises(MemoryConflictError):
        repo.activate_knowledge("knowledge-1", expected_revision=1, now=NOW,
                                **write(repo, "other-activation"))
    assert [entry.operation for entry in repo.list_audit(tenant_id="tenant-a")][-2:] == [
        "propose_knowledge", "activate_knowledge"]


@pytest.mark.parametrize("changes", [dict(authority="service"), dict(authority=object()),
                                     dict(tenant_id="tenant-b"), dict(trace_id=""),
                                     dict(idempotency_key="")])
def test_mutations_require_service_authority_and_scope(repo, changes):
    with pytest.raises((MemoryAuthorityError, ValidationError)):
        repo.append_event(event(), **write(repo, "event", **changes))
    assert repo.list_records(tenant_id="tenant-a") == ()


def test_idempotency_key_cannot_be_rebound(repo):
    repo.append_event(event(), **write(repo, "one"))
    with pytest.raises(MemoryConflictError):
        repo.append_event(event(payload={"price": 200}), **write(repo, "one"))


def test_audit_failure_rolls_back_record_and_idempotency(repo):
    with sqlite3.connect(repo.path) as db:
        db.execute("CREATE TRIGGER fail_audit BEFORE INSERT ON memory_audit BEGIN SELECT RAISE(ABORT, 'audit unavailable'); END")
    with pytest.raises(sqlite3.IntegrityError):
        repo.append_event(event(), **write(repo, "one"))
    assert repo.get_by_id("event-1", tenant_id="tenant-a") is None
    with sqlite3.connect(repo.path) as db:
        db.execute("DROP TRIGGER fail_audit")
    assert repo.append_event(event(), **write(repo, "one")).record_id == "event-1"


def test_event_hash_is_canonical_immutable_and_unique_per_tenant(repo):
    payload = {"b": [2, {"price": 100}], "a": 1}
    first = event(payload=payload)
    payload["b"][1]["price"] = 999
    assert first.payload["b"][1]["price"] == 100
    with pytest.raises(TypeError):
        first.payload["b"][1]["price"] = 999
    with pytest.raises(TypeError):
        dict.__setitem__(first.payload, "a", 4)
    reordered = event("event-2", payload={"a": 1, "b": [2, {"price": 100}]})
    assert reordered.payload_hash == first.payload_hash
    repo.append_event(first, **write(repo, "one"))
    assert repo.append_event(reordered, **write(repo, "two")).record_id == "event-1"
    assert len(repo.list_audit(tenant_id="tenant-a")) == 1
    assert repo.get_by_id("event-1", tenant_id="tenant-b") is None
    repo.append_event(event(tenant_id="tenant-b", payload={"a": 1, "b": [2, {"price": 100}]}),
                      **write(repo, "one", tenant="tenant-b"))


@pytest.mark.parametrize("changes", [dict(payload={"price": float("nan")}),
                                     dict(payload={"price": float("inf")}),
                                     dict(observed_at=NOW.replace(tzinfo=None)),
                                     dict(extra="forbidden"), dict(payload_hash="0" * 64)])
def test_event_rejects_invalid_values(changes):
    with pytest.raises((ValidationError, ValueError)):
        event(**changes)


def test_copy_and_rehydration_cannot_bypass_validation(repo):
    original = event()
    assert EventRecord.model_validate_json(original.model_dump_json()) == original
    with pytest.raises(ValidationError):
        original.model_copy(update={"extra": "forbidden"})
    with pytest.raises(ValidationError):
        candidate().model_copy(update={"confidence": float("nan")})
    forged = EventRecord.model_construct(**dict(original.model_dump(), payload_hash="0" * 64))
    with pytest.raises((MemoryIntegrityError, ValidationError)):
        repo.append_event(forged, **write(repo, "forged"))


def test_rehydration_detects_database_corruption(repo):
    repo.append_event(event(), **write(repo, "one"))
    with sqlite3.connect(repo.path) as db:
        row = db.execute("SELECT body FROM memory_records").fetchone()
        body = json.loads(row[0])
        body["payload"]["price"] = 999
        db.execute("UPDATE memory_records SET body = ?", (json.dumps(body),))
    with pytest.raises(MemoryIntegrityError):
        repo.get_by_id("event-1", tenant_id="tenant-a")


def test_artifacts_are_immutable_checksum_addressed_and_tenant_scoped(tmp_path):
    authority = object()
    store = FileArtifactStore(tmp_path / "objects", writer_authority=authority)
    context = dict(authority=authority, tenant_id="tenant-a", trace_id="trace-1", idempotency_key="one")
    ref = store.put(b"abc", **context)
    assert ref.sha256 == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert store.put(b"abc", **context) == ref
    assert store.get(ref, tenant_id="tenant-a") == b"abc"
    with pytest.raises(MemoryAuthorityError):
        store.put(b"abc", **dict(context, authority="agent"))
    with pytest.raises(MemoryAuthorityError):
        store.get(ref, tenant_id="tenant-b")
    with pytest.raises(MemoryConflictError):
        store.put(b"different", **context)
    blob = next((tmp_path / "objects").rglob(ref.sha256))
    blob.write_bytes(b"tampered")
    with pytest.raises(MemoryIntegrityError):
        store.get(ref, tenant_id="tenant-a")


def test_artifact_reference_cannot_cross_tenant():
    ref = ArtifactReference(tenant_id="tenant-b", sha256="0" * 64, size_bytes=3)
    with pytest.raises(ValidationError):
        event(artifact=ref)


@pytest.mark.parametrize("mode", ["expired", "model", "single_source"])
def test_activation_requires_fresh_independent_or_verified_evidence(repo, mode):
    for number in (1, 2):
        source = "same" if mode == "single_source" else str(number)
        changes = {}
        if mode == "expired":
            changes["expires_at"] = NOW + timedelta(seconds=1)
        if mode == "model":
            changes["provenance"] = Provenance(source_id=source, source_kind="model", independent_group=source)
        repo.append_event(event(f"event-{number}", source=source, payload={"price": number}, **changes),
                          **write(repo, f"event-{number}"))
    repo.propose_knowledge(candidate(), **write(repo, "proposal"))
    with pytest.raises(MemoryPromotionError):
        repo.activate_knowledge("knowledge-1", expected_revision=1, now=NOW + timedelta(seconds=2),
                                **write(repo, "activation"))
    assert repo.get_by_id("knowledge-1", tenant_id="tenant-a").lifecycle is Lifecycle.PROPOSED


def test_circular_or_missing_provenance_is_denied(repo):
    with pytest.raises(ValidationError):
        candidate(evidence_ids=("knowledge-1",))
    with pytest.raises(ValidationError):
        event(provenance=Provenance(source_id="exchange", source_kind="external",
                                   independent_group="exchange", derived_from=("event-1",)))
    with pytest.raises(MemoryPromotionError):
        repo.append_event(event(provenance=Provenance(source_id="exchange", source_kind="external",
                                                      independent_group="exchange", derived_from=("missing",))),
                          **write(repo, "event"))


def test_revision_lineage_requires_current_head_and_cannot_replace_active_record(repo):
    evidence(repo)
    repo.propose_knowledge(candidate(), **write(repo, "proposal"))
    repo.activate_knowledge("knowledge-1", expected_revision=1, now=NOW, **write(repo, "activate"))
    newer = candidate("knowledge-2", revision=2, lineage_ids=("knowledge-1",))
    repo.propose_knowledge(newer, **write(repo, "proposal-2"))
    with pytest.raises(MemoryConflictError):
        repo.propose_knowledge(candidate("knowledge-3", revision=2, lineage_ids=("knowledge-1",)),
                               **write(repo, "proposal-3"))
    assert repo.get_by_id("knowledge-1", tenant_id="tenant-a").lifecycle is Lifecycle.ACTIVE
    repo.activate_knowledge("knowledge-2", expected_revision=2, now=NOW, **write(repo, "activate-2"))
    assert repo.get_by_id("knowledge-1", tenant_id="tenant-a").lifecycle is Lifecycle.ARCHIVED


def decision(record_id="decision-1", **changes):
    values = dict(record_id=record_id, tenant_id="tenant-a", observed_at=NOW,
                  decision="no_trade", status="final", evidence_ids=("event-1",))
    values.update(changes)
    return DecisionRecord(**values)


def outcome(record_id="outcome-1", **changes):
    values = dict(record_id=record_id, tenant_id="tenant-a", observed_at=NOW,
                  decision_id="decision-1", result="risk avoided", verified=True,
                  evidence_ids=("event-1",))
    values.update(changes)
    return OutcomeRecord(**values)


def test_decision_outcome_and_lesson_links_must_be_verified_and_same_tenant(repo):
    evidence(repo)
    repo.append_decision(decision(status="provisional"), **write(repo, "provisional"))
    with pytest.raises(MemoryPromotionError):
        repo.append_outcome(outcome(), **write(repo, "outcome"))
    repo.append_decision(decision("decision-final", supersedes_id="decision-1"), **write(repo, "final"))
    repo.append_outcome(outcome(decision_id="decision-final"), **write(repo, "outcome"))
    lesson = DecisionLesson(record_id="lesson-1", tenant_id="tenant-a", observed_at=NOW,
                            decision_id="decision-final", outcome_id="outcome-1",
                            lesson="Avoid funding exposure.", evidence_ids=("event-1",), confidence=0.8)
    assert repo.link_lesson(lesson, **write(repo, "lesson")) == lesson
    with pytest.raises(MemoryPromotionError):
        repo.link_lesson(lesson.model_copy(update={"record_id": "lesson-2", "decision_id": "decision-1"}),
                         **write(repo, "bad-lesson"))
    repo.propose_knowledge(candidate(evidence_ids=("event-1",), outcome_id="outcome-1"), **write(repo, "proposal"))
    assert repo.activate_knowledge("knowledge-1", expected_revision=1, now=NOW,
                                   **write(repo, "activate")).lifecycle is Lifecycle.ACTIVE


def test_audits_are_trace_bound_and_redacted_and_store_reopens(repo):
    repo.append_event(event(payload={"api_key": "secret-value"}), **write(repo, "one"))
    audit = repo.list_audit(tenant_id="tenant-a")[0]
    assert audit.trace_id == "trace-1" and audit.tenant_id == "tenant-a"
    assert "secret-value" not in audit.model_dump_json()
    assert "api_key" not in audit.model_dump_json()
    with sqlite3.connect(repo.path) as db:
        assert db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    with SQLiteMemoryRepository(repo.path) as reader:
        assert reader.get_by_id("event-1", tenant_id="tenant-a").payload["api_key"] == "secret-value"
        with pytest.raises(MemoryAuthorityError):
            reader.append_event(event("other"), **write(repo, "other"))


def test_derived_event_cannot_corroborate_its_own_knowledge_lineage(repo):
    evidence(repo)
    repo.propose_knowledge(candidate(), **write(repo, "proposal"))
    repo.activate_knowledge("knowledge-1", expected_revision=1, now=NOW, **write(repo, "activate"))
    repo.append_event(event("feedback", payload={"echo": "rule-1"},
                            provenance=Provenance(source_id="model", source_kind="model",
                                                  independent_group="model", derived_from=("knowledge-1",))),
                      **write(repo, "feedback"))
    with pytest.raises(MemoryPromotionError):
        repo.propose_knowledge(candidate("knowledge-2", revision=2, lineage_ids=("knowledge-1",),
                                         evidence_ids=("event-1", "event-2", "feedback")),
                               **write(repo, "proposal-2"))


def test_expired_event_cannot_verify_an_outcome(repo):
    repo.append_event(event(expires_at=NOW + timedelta(seconds=1)), **write(repo, "event"))
    repo.append_decision(decision(), **write(repo, "decision"))
    with pytest.raises(MemoryPromotionError):
        repo.append_outcome(outcome(observed_at=NOW + timedelta(seconds=2)), **write(repo, "outcome"))


def derived_event(repo, record_id, parents, *, kind="system", group=None):
    return repo.append_event(event(
        record_id, source=record_id, payload={"derived": record_id},
        provenance=Provenance(source_id=record_id, source_kind=kind,
                              independent_group=group or record_id, derived_from=parents)),
        **write(repo, record_id))


@pytest.mark.parametrize("depth", [1, 3])
def test_system_wrappers_cannot_verify_model_only_outcomes(repo, depth):
    derived_event(repo, "model-root", (), kind="model")
    parent = "model-root"
    for level in range(depth):
        wrapped = derived_event(repo, f"wrapper-{level}", (parent,))
        parent = wrapped.record_id
    repo.append_decision(decision(evidence_ids=(parent,)), **write(repo, "decision"))
    with pytest.raises(MemoryPromotionError):
        repo.append_outcome(outcome(evidence_ids=(parent,)), **write(repo, "outcome"))
    assert repo.get_by_id("outcome-1", tenant_id="tenant-a") is None
    assert repo.list_audit(tenant_id="tenant-a")[-1].operation == "append_decision"


def test_external_ancestry_does_not_turn_model_claim_into_verification(repo):
    repo.append_event(event(), **write(repo, "event-1"))
    derived_event(repo, "model-claim", ("event-1",), kind="model")
    derived_event(repo, "wrapper", ("model-claim",))
    repo.append_decision(decision(), **write(repo, "decision"))
    with pytest.raises(MemoryPromotionError):
        repo.append_outcome(outcome(evidence_ids=("wrapper",)), **write(repo, "outcome"))


def test_non_event_provenance_cannot_verify_outcome_from_decision_context(repo):
    evidence(repo)
    repo.append_decision(decision(), **write(repo, "decision"))
    derived_event(repo, "wrapper", ("decision-1",))
    with pytest.raises(MemoryPromotionError):
        repo.append_outcome(outcome(evidence_ids=("wrapper",)), **write(repo, "outcome"))


@pytest.mark.parametrize("source_kind", ["external", "system"])
def test_real_root_evidence_can_verify_through_multiple_system_derivations(repo, source_kind):
    derived_event(repo, "root", (), kind=source_kind)
    derived_event(repo, "normalized", ("root",))
    derived_event(repo, "parallel", ("root",))
    derived_event(repo, "verified-result", ("normalized", "parallel"))
    repo.append_decision(decision(evidence_ids=("root",)), **write(repo, "decision"))
    verified = repo.append_outcome(outcome(evidence_ids=("verified-result",)), **write(repo, "outcome"))
    assert verified.verified
    repo.propose_knowledge(candidate(evidence_ids=("verified-result",), outcome_id="outcome-1"),
                           **write(repo, "proposal"))
    assert repo.activate_knowledge("knowledge-1", expected_revision=1, now=NOW,
                                   **write(repo, "activation")).lifecycle is Lifecycle.ACTIVE


@pytest.mark.parametrize("independent", [True, False])
def test_promotion_counts_original_roots_not_system_wrapper_labels(repo, independent):
    derived_event(repo, "root-a", (), kind="external", group="source-a")
    derived_event(repo, "root-b", (), kind="external", group="source-b" if independent else "source-a")
    derived_event(repo, "wrapper-a", ("root-a",))
    derived_event(repo, "wrapper-b", ("root-b",))
    repo.propose_knowledge(candidate(evidence_ids=("wrapper-a", "wrapper-b")), **write(repo, "proposal"))
    if independent:
        assert repo.activate_knowledge("knowledge-1", expected_revision=1, now=NOW,
                                       **write(repo, "activation")).lifecycle is Lifecycle.ACTIVE
    else:
        with pytest.raises(MemoryPromotionError):
            repo.activate_knowledge("knowledge-1", expected_revision=1, now=NOW, **write(repo, "activation"))


def replace_stored_record(repo, record):
    # Simulate a pre-existing graph written by an older producer, with valid checksums.
    body = record.model_dump_json()
    with sqlite3.connect(repo.path) as db:
        db.execute("UPDATE memory_records SET body=?,body_hash=? WHERE tenant_id=? AND record_id=?",
                   (body, hashlib.sha256(body.encode()).hexdigest(), record.tenant_id, record.record_id))
        db.execute("DELETE FROM memory_links WHERE tenant_id=? AND record_id=?",
                   (record.tenant_id, record.record_id))
        for parent in record.provenance.derived_from:
            db.execute("INSERT INTO memory_links VALUES(?,?,?)", (record.tenant_id, record.record_id, parent))


def test_verified_outcome_rejects_cyclic_provenance(repo):
    root = derived_event(repo, "root", ())
    derived_event(repo, "wrapper", ("root",))
    replace_stored_record(repo, root.model_copy(update={
        "provenance": root.provenance.model_copy(update={"derived_from": ("wrapper",)})}))
    repo.append_decision(decision(evidence_ids=("wrapper",)), **write(repo, "decision"))
    with pytest.raises(MemoryPromotionError):
        repo.append_outcome(outcome(evidence_ids=("wrapper",)), **write(repo, "outcome"))


def test_promotion_revalidates_stored_outcome_provenance(repo):
    root = derived_event(repo, "root", ())
    derived_event(repo, "model-root", (), kind="model")
    derived_event(repo, "wrapper", ("root",))
    repo.append_decision(decision(evidence_ids=("wrapper",)), **write(repo, "decision"))
    repo.append_outcome(outcome(evidence_ids=("wrapper",)), **write(repo, "outcome"))
    repo.propose_knowledge(candidate(evidence_ids=("wrapper",), outcome_id="outcome-1"),
                           **write(repo, "proposal"))
    replace_stored_record(repo, root.model_copy(update={
        "provenance": root.provenance.model_copy(update={"derived_from": ("model-root",)})}))
    with pytest.raises(MemoryPromotionError):
        repo.activate_knowledge("knowledge-1", expected_revision=1, now=NOW, **write(repo, "activation"))
    assert repo.get_by_id("knowledge-1", tenant_id="tenant-a").lifecycle is Lifecycle.PROPOSED


@pytest.mark.parametrize("derived", [False, True])
def test_lesson_rechecks_all_outcome_evidence_at_link_time(repo, derived):
    repo.append_event(event(expires_at=NOW + timedelta(seconds=1)), **write(repo, "event-1"))
    repo.append_event(event("fresh", payload={"price": 200}), **write(repo, "fresh"))
    verification_id = "event-1"
    if derived:
        verification_id = derived_event(repo, "wrapper", (verification_id,)).record_id
    repo.append_decision(decision(), **write(repo, "decision"))
    repo.append_outcome(outcome(evidence_ids=(verification_id, "fresh")), **write(repo, "outcome"))
    lesson = DecisionLesson(record_id="lesson", tenant_id="tenant-a", observed_at=NOW + timedelta(seconds=2),
                            decision_id="decision-1", outcome_id="outcome-1", lesson="Do not use stale outcomes.",
                            evidence_ids=("fresh",), confidence=0.8)
    before_audit = repo.list_audit(tenant_id="tenant-a")
    with pytest.raises(MemoryPromotionError):
        repo.link_lesson(lesson, **write(repo, "lesson"))
    assert repo.get_by_id("lesson", tenant_id="tenant-a") is None
    assert repo.list_audit(tenant_id="tenant-a") == before_audit


def test_stale_connection_cannot_activate_superseded_proposal(repo):
    evidence(repo)
    repo.propose_knowledge(candidate(), **write(repo, "proposal"))
    with SQLiteMemoryRepository(repo.path, writer_authority=repo.test_authority) as other:
        repo.propose_knowledge(candidate("knowledge-2", revision=2, lineage_ids=("knowledge-1",)),
                               **write(repo, "proposal-2"))
        with pytest.raises(MemoryConflictError):
            other.activate_knowledge("knowledge-1", expected_revision=1, now=NOW, **write(repo, "activate"))


def test_outcome_for_another_decision_cannot_support_a_lesson(repo):
    evidence(repo)
    repo.append_decision(decision(), **write(repo, "decision"))
    repo.append_decision(decision("decision-2"), **write(repo, "decision-2"))
    repo.append_outcome(outcome(), **write(repo, "outcome"))
    with pytest.raises(MemoryPromotionError):
        repo.link_lesson(DecisionLesson(record_id="lesson", tenant_id="tenant-a", observed_at=NOW,
                                        decision_id="decision-2", outcome_id="outcome-1",
                                        lesson="Do not infer outcomes.", evidence_ids=("event-1",), confidence=0.8),
                         **write(repo, "lesson"))


def test_outcome_cannot_predate_its_decision(repo):
    evidence(repo)
    repo.append_decision(decision(observed_at=NOW + timedelta(seconds=1)), **write(repo, "decision"))
    with pytest.raises(MemoryPromotionError):
        repo.append_outcome(outcome(), **write(repo, "outcome"))


def test_failed_activation_preserves_both_revisions_and_retries(repo):
    evidence(repo)
    repo.propose_knowledge(candidate(), **write(repo, "proposal"))
    repo.activate_knowledge("knowledge-1", expected_revision=1, now=NOW, **write(repo, "activate"))
    repo.propose_knowledge(candidate("knowledge-2", revision=2, lineage_ids=("knowledge-1",)),
                           **write(repo, "proposal-2"))
    with sqlite3.connect(repo.path) as db:
        db.execute("CREATE TRIGGER fail_activation BEFORE INSERT ON memory_audit WHEN NEW.operation='activate_knowledge' BEGIN SELECT RAISE(ABORT, 'audit unavailable'); END")
    with pytest.raises(sqlite3.IntegrityError):
        repo.activate_knowledge("knowledge-2", expected_revision=2, now=NOW, **write(repo, "activate-2"))
    assert repo.get_by_id("knowledge-1", tenant_id="tenant-a").lifecycle is Lifecycle.ACTIVE
    assert repo.get_by_id("knowledge-2", tenant_id="tenant-a").lifecycle is Lifecycle.PROPOSED
    with sqlite3.connect(repo.path) as db:
        db.execute("DROP TRIGGER fail_activation")
    assert repo.activate_knowledge("knowledge-2", expected_revision=2, now=NOW,
                                   **write(repo, "activate-2")).lifecycle is Lifecycle.ACTIVE
