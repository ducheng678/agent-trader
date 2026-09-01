from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from market_agent.workflow_long_term_memory import (
    DecisionRecord, EventRecord, KnowledgeRevision, Lifecycle, MemoryAuthorityError,
    MemoryPromotionError, OutcomeRecord, Provenance,
)
from market_agent.workflow_memory_sqlite import SQLiteMemoryRepository
from market_agent.workflow_memory_promotion import promote_candidate


NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


@pytest.fixture
def repo(tmp_path):
    authority = object()
    with SQLiteMemoryRepository(tmp_path / "memory.db", writer_authority=authority) as repository:
        repository.test_authority = authority
        yield repository


def write(repo, key):
    return dict(tenant_id="tenant-a", trace_id="trace-1", idempotency_key=key,
                authority=repo.test_authority)


def seed_event(repo, identifier, group, *, kind="external", **changes):
    event = EventRecord(record_id=identifier, tenant_id="tenant-a", observed_at=NOW,
                        source=group, payload={"observation": identifier},
                        provenance=Provenance(source_id=group, source_kind=kind,
                                              independent_group=group), **changes)
    return repo.append_event(event, **write(repo, identifier))


def candidate(**changes):
    values = dict(record_id="rule-1", tenant_id="tenant-a", observed_at=NOW,
                  knowledge_id="funding", revision=1, rule="Check funding before entry.",
                  confidence=0.9, effective_at=NOW, evidence_ids=("event-a", "event-b"))
    values.update(changes)
    return KnowledgeRevision(**values)


def test_service_promotes_independent_evidence_and_retries_without_duplicate_audits(repo):
    seed_event(repo, "event-a", "exchange")
    seed_event(repo, "event-b", "independent")
    first = promote_candidate(candidate(), repo, now=NOW, **write(repo, "promote"))
    assert first.lifecycle is Lifecycle.ACTIVE
    assert promote_candidate(candidate(), repo, now=NOW, **write(repo, "promote")) == first
    audits = repo.list_audit(tenant_id="tenant-a")
    assert [audit.operation for audit in audits][-2:] == ["propose_knowledge", "activate_knowledge"]
    assert len(audits) == 4
    assert all(audit.trace_id == "trace-1" for audit in audits)
    assert "Check funding" not in "".join(audit.model_dump_json() for audit in audits)


@pytest.mark.parametrize("mode", ["same-source", "model", "expired", "missing", "conflict"])
def test_service_rejects_unqualified_evidence_without_activation(repo, mode):
    seed_event(repo, "event-a", "exchange",
               **({"expires_at": NOW + timedelta(seconds=1)} if mode == "expired" else {}))
    if mode != "missing":
        seed_event(repo, "event-b", "exchange" if mode == "same-source" else "independent",
                   kind="model" if mode == "model" else "external")
    changes = {"contradicting_ids": ("event-b",), "evidence_ids": ("event-a",)} if mode == "conflict" else {}
    with pytest.raises(MemoryPromotionError):
        promote_candidate(candidate(**changes), repo, now=NOW + timedelta(seconds=2), **write(repo, "promote"))
    stored = repo.get_by_id("rule-1", tenant_id="tenant-a")
    assert stored is None or stored.lifecycle is Lifecycle.PROPOSED
    assert "activate_knowledge" not in [audit.operation for audit in repo.list_audit(tenant_id="tenant-a")]


def test_verified_outcome_can_promote_one_independent_observation(repo):
    seed_event(repo, "event-a", "exchange")
    repo.append_decision(DecisionRecord(record_id="decision", tenant_id="tenant-a", observed_at=NOW,
                                        decision="no_trade", status="final", evidence_ids=("event-a",)),
                         **write(repo, "decision"))
    repo.append_outcome(OutcomeRecord(record_id="outcome", tenant_id="tenant-a", observed_at=NOW,
                                      decision_id="decision", result="risk avoided", verified=True,
                                      evidence_ids=("event-a",)), **write(repo, "outcome"))
    active = promote_candidate(candidate(evidence_ids=("event-a",), outcome_id="outcome"),
                               repo, now=NOW, **write(repo, "promote"))
    assert active.lifecycle is Lifecycle.ACTIVE


def test_circular_descendant_cannot_be_promoted(repo):
    seed_event(repo, "event-a", "exchange")
    seed_event(repo, "event-b", "independent")
    promote_candidate(candidate(), repo, now=NOW, **write(repo, "promote"))
    repo.append_event(EventRecord(record_id="feedback", tenant_id="tenant-a", observed_at=NOW,
                                  source="model", payload={"echo": "rule-1"},
                                  provenance=Provenance(source_id="model", source_kind="model",
                                                        independent_group="model", derived_from=("rule-1",))),
                       **write(repo, "feedback"))
    with pytest.raises(MemoryPromotionError):
        promote_candidate(candidate(record_id="rule-2", revision=2, lineage_ids=("rule-1",),
                                    evidence_ids=("event-a", "event-b", "feedback")),
                          repo, now=NOW, **write(repo, "promote-2"))
    assert repo.get_by_id("rule-2", tenant_id="tenant-a") is None


def test_agent_cannot_promote_or_cross_tenant(repo):
    seed_event(repo, "event-a", "exchange")
    seed_event(repo, "event-b", "independent")
    context = write(repo, "promote")
    context["authority"] = object()
    with pytest.raises(MemoryAuthorityError):
        promote_candidate(candidate(), repo, now=NOW, **context)
    context = write(repo, "promote")
    context["tenant_id"] = "tenant-b"
    with pytest.raises(MemoryAuthorityError):
        promote_candidate(candidate(), repo, now=NOW, **context)
    assert repo.get_by_id("rule-1", tenant_id="tenant-a") is None
