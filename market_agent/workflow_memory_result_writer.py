from __future__ import annotations

from datetime import datetime
from hashlib import sha256
from typing import Literal

from pydantic import AwareDatetime

from market_agent.workflow_contracts import ContractModel, Digest, ShortText, WorkflowResult
from market_agent.workflow_contracts import (
    WorkflowRequest,
    canonical_workflow_request_digest,
    canonical_workflow_result_digest,
)
from market_agent.workflow_execution_backend import (
    CommittedTransitionReceipt,
    verify_committed_transition_receipt,
)
from market_agent.workflow_harness_contracts import RunState
from market_agent.workflow_long_term_memory import (
    DecisionRecord,
    EventRecord,
    MemoryRepository,
    OutcomeRecord,
    Provenance,
)
from market_agent.workflow_prompt_release import canonical_json


class AcceptedOutcomeProof(ContractModel):
    """Host evidence that one exact result crossed the Harness commit boundary."""

    workflow_id: ShortText
    trace_id: ShortText
    request_digest: Digest
    result_digest: Digest
    harness_state: Literal["succeeded"]
    terminal_receipt: CommittedTransitionReceipt
    prompt_release_digest: Digest
    accepted_at: AwareDatetime

    @classmethod
    def bind(
        cls,
        request: WorkflowRequest,
        result: WorkflowResult,
        *,
        terminal_receipt: CommittedTransitionReceipt,
        prompt_release_digest: str,
        accepted_at: datetime,
    ) -> AcceptedOutcomeProof:
        request = WorkflowRequest.model_validate(request)
        result = WorkflowResult.model_validate(result)
        proof = cls(
            workflow_id=result.workflow_id,
            trace_id=result.trace_id,
            request_digest=canonical_workflow_request_digest(request),
            result_digest=canonical_workflow_result_digest(result),
            harness_state="succeeded",
            terminal_receipt=terminal_receipt,
            prompt_release_digest=prompt_release_digest,
            accepted_at=accepted_at,
        )
        proof.verify(request, result)
        return proof

    def verify(self, request: WorkflowRequest, result: WorkflowResult) -> None:
        request = WorkflowRequest.model_validate(request)
        result = WorkflowResult.model_validate(result)
        if (
            (self.workflow_id, self.trace_id)
            != (request.workflow_id, request.trace_id)
            or (self.workflow_id, self.trace_id)
            != (result.workflow_id, result.trace_id)
        ):
            raise ValueError("acceptance proof identity does not match result")
        if self.request_digest != canonical_workflow_request_digest(request):
            raise ValueError("acceptance proof request digest does not match request")
        digest = canonical_workflow_result_digest(result)
        if self.result_digest != digest:
            raise ValueError("acceptance proof result digest does not match result")
        receipt = self.terminal_receipt
        if not verify_committed_transition_receipt(receipt):
            raise ValueError("acceptance proof requires a host-signed terminal receipt")
        view = receipt.post.folded_view
        if (
            view is None
            or view.run_state is not RunState.SUCCEEDED
            or receipt.post.event_head_hash != view.last_event_hash
        ):
            raise ValueError("acceptance proof receipt is not terminal succeeded authority")
        if (
            receipt.post.run_id != self.workflow_id
            or receipt.post.trace_id != self.trace_id
            or view.run_id != self.workflow_id
            or view.trace_id != self.trace_id
        ):
            raise ValueError("acceptance proof receipt identity does not match result")
        if view.request_digest != self.request_digest:
            raise ValueError("acceptance proof receipt does not bind request")
        if view.prompt_release_digest != self.prompt_release_digest:
            raise ValueError("acceptance proof receipt does not bind prompt release")
        if view.accepted_result_digest != self.result_digest:
            raise ValueError("acceptance proof receipt does not bind result")

    @property
    def proof_digest(self) -> str:
        return sha256(canonical_json(self.model_dump(mode="json")).encode("utf-8")).hexdigest()

    @property
    def receipt_digest(self) -> str:
        return sha256(canonical_json(
            self.terminal_receipt.model_dump(mode="json")
        ).encode("utf-8")).hexdigest()


class MemoryResultWriter:
    def __init__(self, *, repository: MemoryRepository, authority: object, tenant_id: str) -> None:
        if repository is None or authority is None or not tenant_id.strip():
            raise ValueError("memory result writer requires host-owned dependencies")
        self._repository = repository
        self._authority = authority
        self._tenant_id = tenant_id

    def record(
        self,
        request: WorkflowRequest,
        result: WorkflowResult,
        proof: AcceptedOutcomeProof,
    ) -> None:
        request = WorkflowRequest.model_validate(request)
        result = WorkflowResult.model_validate(result)
        proof = AcceptedOutcomeProof.model_validate(proof)
        proof.verify(request, result)
        now = proof.accepted_at
        prefix = sha256((result.trace_id + result.workflow_id).encode()).hexdigest()[:32]
        event_id = "workflow-event-" + prefix
        decision_id = "workflow-decision-" + prefix
        outcome_id = "workflow-outcome-" + prefix
        context = dict(tenant_id=self._tenant_id, trace_id=result.trace_id, authority=self._authority)
        event = EventRecord(
            record_id=event_id,
            tenant_id=self._tenant_id,
            observed_at=now,
            source="workflow_result",
            payload={
                "workflow_id": result.workflow_id,
                "terminal_mode": result.terminal_mode.value,
                "final_action": result.final_action.value,
                "knowledge_status": result.knowledge_status.value,
                "evidence_references": list(result.evidence_references),
                "acceptance_proof_digest": proof.proof_digest,
                "harness_receipt_digest": proof.receipt_digest,
                "prompt_release_digest": proof.prompt_release_digest,
            },
            provenance=Provenance(
                source_id="harness-receipt:" + proof.receipt_digest,
                source_kind="system",
                independent_group="harness:" + result.trace_id,
            ),
        )
        self._repository.append_event(event, idempotency_key=prefix + ":event", **context)
        decision = DecisionRecord(
            record_id=decision_id,
            tenant_id=self._tenant_id,
            observed_at=now,
            decision=result.final_action.value,
            status="final",
            evidence_ids=(event_id,),
        )
        self._repository.append_decision(decision, idempotency_key=prefix + ":decision", **context)
        outcome = OutcomeRecord(
            record_id=outcome_id,
            tenant_id=self._tenant_id,
            observed_at=now,
            decision_id=decision_id,
            result=result.terminal_mode.value,
            # This writer is host-owned and is called only by the accepted
            # Harness commit callback.  Candidate results must never reach it.
            verified=True,
            evidence_ids=(event_id,),
        )
        self._repository.append_outcome(outcome, idempotency_key=prefix + ":outcome", **context)
