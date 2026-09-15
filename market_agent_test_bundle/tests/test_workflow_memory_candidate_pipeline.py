from __future__ import annotations

import pytest

from market_agent.workflow_contracts import (
    Action, InformationalAnswer, KnowledgeStatus, TerminalMode, WorkflowResult,
)
from market_agent.workflow_long_term_memory import EventRecord, KnowledgeRevision, Lifecycle, OutcomeRecord
from market_agent.workflow_memory_result_writer import MemoryResultWriter
from market_agent.workflow_memory_sqlite import SQLiteMemoryRepository
from market_agent.workflow_harness_contracts import RunState
from market_agent.workflow_memory_candidate_pipeline import (
    AcceptedResultMemoryPipeline, accepted_outcome_candidates,
)
from test_task7_workflow_closure import NOW, _workflow_request, _untrusted_proof


@pytest.fixture
def accepted(monkeypatch):
    monkeypatch.setattr(
        "market_agent.workflow_memory_result_writer.verify_committed_transition_receipt",
        lambda _receipt: True,
    )
    request = _workflow_request()
    result = WorkflowResult(
        workflow_id=request.workflow_id, trace_id=request.trace_id,
        terminal_mode=TerminalMode.INFORMATIONAL, final_action=Action.NO_TRADE,
        knowledge_status=KnowledgeStatus.KNOWN, uncertainty_reason=None,
        informational_answer=InformationalAnswer(
            knowledge_status=KnowledgeStatus.KNOWN, uncertainty_reason=None,
            answer="Untrusted model text must not become a verified factual rule.",
        ),
    )
    return request, result, _untrusted_proof(request, result, "e" * 64)


def pipeline(repository, authority, **kwargs):
    return AcceptedResultMemoryPipeline(
        writer=MemoryResultWriter(repository=repository, authority=authority, tenant_id="tenant-a"),
        repository=repository, authority=authority, tenant_id="tenant-a", **kwargs,
    )


def test_accepted_result_proposes_bound_candidate_and_replays_without_audit(tmp_path, accepted):
    authority = object()
    with SQLiteMemoryRepository(tmp_path / "memory.db", writer_authority=authority) as repository:
        service = pipeline(repository, authority)
        report = service.record(*accepted)
        assert report.status == "completed"
        records = repository.list_records(tenant_id="tenant-a")
        candidate, = [item for item in records if isinstance(item, KnowledgeRevision)]
        outcome, = [item for item in records if isinstance(item, OutcomeRecord)]
        evidence = repository.get_by_id(candidate.evidence_ids[0], tenant_id="tenant-a")
        assert candidate.lifecycle is Lifecycle.PROPOSED
        assert outcome.verified and candidate.outcome_id == outcome.record_id
        assert candidate.evidence_ids == outcome.evidence_ids
        assert isinstance(evidence, EventRecord) and evidence.provenance.source_kind == "system"
        assert "Untrusted model text" not in candidate.rule
        before = repository.list_audit(tenant_id="tenant-a")
        assert service.record(*accepted) == report
        assert repository.list_audit(tenant_id="tenant-a") == before


def test_invalid_proof_never_calls_factory_or_writes(tmp_path, accepted):
    authority = object()
    calls = []
    with SQLiteMemoryRepository(tmp_path / "memory.db", writer_authority=authority) as repository:
        service = pipeline(repository, authority, factory=lambda record: calls.append(record))
        request, result, proof = accepted
        with pytest.raises(ValueError, match="result digest"):
            service.record(request, result, proof.model_copy(update={"result_digest": "f" * 64}))
        assert calls == []
        assert repository.list_records(tenant_id="tenant-a") == ()


@pytest.mark.parametrize("error", [RuntimeError("extractor unavailable"), TimeoutError("deadline")])
def test_extraction_failure_survives_restart_and_retries_without_writer(tmp_path, accepted, error):
    authority = object()
    path = tmp_path / "memory.db"
    def failing_factory(_record):
        raise error
    with SQLiteMemoryRepository(path, writer_authority=authority) as repository:
        service = pipeline(repository, authority, factory=failing_factory)
        report = service.record(*accepted)
        assert report.status == "pending"
        assert any(isinstance(item, OutcomeRecord) and item.verified
                   for item in repository.list_records(tenant_id="tenant-a"))
    with SQLiteMemoryRepository(path, writer_authority=authority) as repository:
        service = pipeline(repository, authority)
        service._writer.record = lambda *_: pytest.fail("retry must not recommit accepted result")
        retried, = service.retry_pending()
        assert retried.status == "completed"
        assert service.retry_pending() == ()
        assert len([item for item in repository.list_records(tenant_id="tenant-a")
                    if isinstance(item, KnowledgeRevision)]) == 1


def test_partial_proposal_failure_retries_idempotently_and_rejects_changed_content(tmp_path, accepted, monkeypatch):
    authority = object()
    with SQLiteMemoryRepository(tmp_path / "memory.db", writer_authority=authority) as repository:
        append = repository.append_event
        def fail_completion(record, **kwargs):
            if record.source == "candidate_extraction_completed":
                raise RuntimeError("completion marker unavailable")
            return append(record, **kwargs)
        monkeypatch.setattr(repository, "append_event", fail_completion)
        service = pipeline(repository, authority)
        assert service.record(*accepted).status == "pending"
        def changed(record):
            candidate, = accepted_outcome_candidates(record)
            return (candidate.model_copy(update={"rule": "Different immutable content."}),)
        conflict, = pipeline(repository, authority, factory=changed).retry_pending()
        assert conflict.status == "pending"
        assert conflict.error_code == "MemoryConflictError"
        monkeypatch.setattr(repository, "append_event", append)
        recovered, = service.retry_pending()
        assert recovered.status == "completed"
        proposals = [item for item in repository.list_audit(tenant_id="tenant-a")
                     if item.operation == "propose_knowledge"]
        assert len(proposals) == 1


@pytest.mark.parametrize("change", [
    {"outcome_id": None}, {"evidence_ids": ("unrelated",)},
    {"tenant_id": "other"}, {"lifecycle": Lifecycle.ACTIVE},
])
def test_factory_cannot_escape_accepted_evidence_or_activate(tmp_path, accepted, change):
    authority = object()
    with SQLiteMemoryRepository(tmp_path / "memory.db", writer_authority=authority) as repository:
        def invalid_factory(record):
            candidate, = accepted_outcome_candidates(record)
            return (candidate.model_copy(update=change),)
        report = pipeline(repository, authority, factory=invalid_factory).record(*accepted)
        assert report.status == "pending"
        assert not any(isinstance(item, KnowledgeRevision)
                       for item in repository.list_records(tenant_id="tenant-a"))


@pytest.mark.parametrize("state", [RunState.FAILED, RunState.CANCELLED, RunState.DEGRADED, RunState.SUMMARIZING])
def test_unaccepted_terminal_states_do_not_extract(tmp_path, accepted, state):
    authority = object()
    calls = []
    request, result, proof = accepted
    receipt = proof.terminal_receipt
    view = receipt.post.folded_view.model_copy(update={"run_state": state})
    proof = proof.model_copy(update={"terminal_receipt": receipt.model_copy(update={
        "post": receipt.post.model_copy(update={"folded_view": view}),
    })})
    with SQLiteMemoryRepository(tmp_path / "memory.db", writer_authority=authority) as repository:
        service = pipeline(repository, authority, factory=lambda record: calls.append(record))
        with pytest.raises(ValueError, match="terminal succeeded"):
            service.record(request, result, proof)
        assert calls == []
        assert repository.list_records(tenant_id="tenant-a") == ()


@pytest.mark.parametrize("size", [0, 9])
def test_factory_cardinality_is_checked_before_proposal(tmp_path, accepted, size):
    authority = object()
    with SQLiteMemoryRepository(tmp_path / "memory.db", writer_authority=authority) as repository:
        service = pipeline(repository, authority, factory=lambda record: accepted_outcome_candidates(record) * size)
        assert service.record(*accepted).status == "pending"
        assert not any(isinstance(item, KnowledgeRevision)
                       for item in repository.list_records(tenant_id="tenant-a"))


def test_crash_after_writer_before_extraction_is_recoverable(tmp_path, accepted):
    authority = object()
    with SQLiteMemoryRepository(tmp_path / "memory.db", writer_authority=authority) as repository:
        writer = MemoryResultWriter(repository=repository, authority=authority, tenant_id="tenant-a")
        recorded = writer.record(*accepted)
        assert recorded.outcome.verified
        service = pipeline(repository, authority)
        assert service.retry_pending(cancellation_check=lambda: True) == ()
        recovered, = service.retry_pending()
        assert recovered.status == "completed"
        assert recovered.outcome_id == recorded.outcome.record_id
