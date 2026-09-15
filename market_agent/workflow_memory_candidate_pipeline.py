"""Host-only extraction of proposed facts; acceptance and promotion stay separate.

The verified outcome is the durable pending work item. A completion event records
the immutable output manifest only after every proposal succeeds. Missing markers
can therefore be retried after restart without invoking Harness or the result writer.
Factories are trusted, bounded, deterministic local functions, never model calls.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import Literal

from market_agent.workflow_contracts import ContractModel, ShortText, WorkflowRequest, WorkflowResult
from market_agent.workflow_long_term_memory import (
    DecisionRecord, EventRecord, KnowledgeRevision, Lifecycle, MemoryIntegrityError,
    MemoryPromotionError, MemoryRepository, OutcomeRecord, Provenance, content_hash,
)
from market_agent.workflow_memory_result_writer import AcceptedOutcomeProof, MemoryResultWriter, RecordedOutcome


KnowledgeCandidateFactory = Callable[[RecordedOutcome], tuple[KnowledgeRevision, ...]]


class CandidateExtractionResult(ContractModel):
    outcome_id: ShortText
    status: Literal["completed", "pending"]
    candidate_ids: tuple[ShortText, ...] = ()
    error_code: ShortText | None = None


def accepted_outcome_candidates(recorded: RecordedOutcome) -> tuple[KnowledgeRevision, ...]:
    """Summarize acceptance facts, not the truth/profitability of model assertions."""
    event, outcome = recorded.event, recorded.outcome
    identifier = content_hash({"service": "accepted_candidate_v1", "outcome_id": outcome.record_id})
    return (KnowledgeRevision(
        record_id="accepted-candidate-" + identifier,
        knowledge_id="accepted-knowledge-" + identifier,
        tenant_id=outcome.tenant_id, revision=1, observed_at=outcome.observed_at,
        effective_at=outcome.observed_at, confidence=1.0,
        rule=(f"Harness accepted workflow result with receipt {event.payload['harness_receipt_digest']}; "
              f"recorded action {recorded.decision.decision} and terminal mode {outcome.result}."),
        applicability=("workflow_acceptance",), evidence_ids=(event.record_id,),
        outcome_id=outcome.record_id,
    ),)


class AcceptedResultMemoryPipeline:
    def __init__(self, *, writer: MemoryResultWriter, repository: MemoryRepository,
                 authority: object, tenant_id: str,
                 factory: KnowledgeCandidateFactory = accepted_outcome_candidates,
                 max_candidates: int = 8) -> None:
        if writer is None or repository is None or authority is None or not tenant_id.strip():
            raise ValueError("candidate pipeline requires host-owned dependencies")
        if not callable(factory) or type(max_candidates) is not int or not 1 <= max_candidates <= 32:
            raise ValueError("candidate factory or limit is invalid")
        self._writer, self._repository = writer, repository
        self._authority, self._tenant_id = authority, tenant_id
        self._factory, self._max_candidates = factory, max_candidates

    def record(self, request: WorkflowRequest, result: WorkflowResult,
               proof: AcceptedOutcomeProof) -> CandidateExtractionResult:
        # Proof rejection and factual-persistence errors still fail acceptance commit.
        # Only subsequent extraction failures become independently retryable work.
        recorded = self._writer.record(request, result, proof)
        return self._extract(recorded)

    @staticmethod
    def _completion_id(outcome_id: str) -> str:
        return "candidate-completed-" + content_hash({"outcome_id": outcome_id, "version": 1})

    def _validate_recorded(self, recorded: RecordedOutcome) -> None:
        event, decision, outcome = recorded.event, recorded.decision, recorded.outcome
        if (any(item.tenant_id != self._tenant_id or item.lifecycle is not Lifecycle.ACTIVE
                for item in (event, decision, outcome))
                or not outcome.verified or decision.status != "final"
                or outcome.decision_id != decision.record_id
                or outcome.evidence_ids != (event.record_id,)
                or decision.evidence_ids != (event.record_id,)
                or event.source != "workflow_result"
                or event.provenance.source_kind != "system"
                or event.provenance.derived_from
                or event.provenance.source_id != "harness-receipt:" + str(event.payload.get("harness_receipt_digest", ""))
                or not event.provenance.independent_group.startswith("harness:")
                or not event.payload.get("acceptance_proof_digest")
                or event.observed_at != outcome.observed_at
                or decision.observed_at != outcome.observed_at):
            raise MemoryPromotionError("candidate extraction requires matching host-accepted records")

    def _extract(self, recorded: RecordedOutcome) -> CandidateExtractionResult:
        outcome_id = recorded.outcome.record_id
        try:
            self._validate_recorded(recorded)
            completion_id = self._completion_id(outcome_id)
            previous = self._repository.get_by_id(completion_id, tenant_id=self._tenant_id)
            if previous is not None:
                if (not isinstance(previous, EventRecord)
                        or previous.source != "candidate_extraction_completed"
                        or previous.payload.get("outcome_id") != outcome_id
                        or previous.provenance.derived_from != (recorded.event.record_id,)):
                    raise MemoryIntegrityError("invalid candidate completion marker")
                return CandidateExtractionResult(outcome_id=outcome_id, status="completed",
                                                 candidate_ids=previous.payload["candidate_ids"])
            candidates = self._factory(recorded)
            if type(candidates) is not tuple or not 1 <= len(candidates) <= self._max_candidates:
                raise MemoryPromotionError("factory must return a bounded nonempty candidate tuple")
            identifiers: set[str] = set()
            for candidate in candidates:
                if (not isinstance(candidate, KnowledgeRevision)
                        or candidate.lifecycle is not Lifecycle.PROPOSED
                        or candidate.tenant_id != self._tenant_id
                        or candidate.outcome_id != outcome_id
                        or candidate.evidence_ids != (recorded.event.record_id,)
                        or candidate.record_id in identifiers
                        or candidate.observed_at != recorded.outcome.observed_at):
                    raise MemoryPromotionError("candidate must bind the accepted outcome and event")
                identifiers.add(candidate.record_id)
            context = dict(tenant_id=self._tenant_id, authority=self._authority,
                           trace_id=recorded.event.provenance.independent_group.removeprefix("harness:"))
            for candidate in candidates:
                self._repository.propose_knowledge(candidate, idempotency_key=content_hash({
                    "service": "accepted_candidate_v1", "outcome_id": outcome_id,
                    "candidate_id": candidate.record_id,
                }), **context)
            completion = EventRecord(
                record_id=completion_id, tenant_id=self._tenant_id,
                observed_at=recorded.outcome.observed_at, source="candidate_extraction_completed",
                payload={"outcome_id": outcome_id,
                         "candidate_ids": [candidate.record_id for candidate in candidates]},
                provenance=Provenance(source_id="accepted_candidate_v1", source_kind="system",
                                      independent_group=recorded.event.provenance.independent_group,
                                      derived_from=(recorded.event.record_id,)),
            )
            self._repository.append_event(completion, idempotency_key=completion_id, **context)
            return CandidateExtractionResult(outcome_id=outcome_id, status="completed",
                                             candidate_ids=tuple(candidate.record_id for candidate in candidates))
        except Exception as error:
            # Do not leak exception payloads or propagate extraction failure into
            # the already durable Harness result commit. Pending outcomes survive.
            return CandidateExtractionResult(outcome_id=outcome_id, status="pending",
                                             error_code=type(error).__name__)

    def retry_pending(self, *, max_outcomes: int = 100,
                      cancellation_check: Callable[[], bool] = lambda: False,
                      ) -> tuple[CandidateExtractionResult, ...]:
        if type(max_outcomes) is not int or not 1 <= max_outcomes <= 1000:
            raise ValueError("candidate retry limit is invalid")
        reports = []
        for outcome in self._repository.list_records(tenant_id=self._tenant_id):
            if cancellation_check() or len(reports) >= max_outcomes:
                break
            if (not isinstance(outcome, OutcomeRecord) or not outcome.verified
                    or outcome.lifecycle is not Lifecycle.ACTIVE
                    or not outcome.record_id.startswith("workflow-outcome-")
                    or len(outcome.evidence_ids) != 1):
                continue
            if self._repository.get_by_id(self._completion_id(outcome.record_id), tenant_id=self._tenant_id) is not None:
                continue
            event = self._repository.get_by_id(outcome.evidence_ids[0], tenant_id=self._tenant_id)
            decision = self._repository.get_by_id(outcome.decision_id, tenant_id=self._tenant_id)
            if not isinstance(event, EventRecord) or not isinstance(decision, DecisionRecord):
                continue
            reports.append(self._extract(RecordedOutcome(event=event, decision=decision, outcome=outcome)))
        return tuple(reports)
