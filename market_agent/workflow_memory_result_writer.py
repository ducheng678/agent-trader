from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256

from market_agent.workflow_contracts import WorkflowResult
from market_agent.workflow_long_term_memory import (
    DecisionRecord,
    EventRecord,
    MemoryRepository,
    OutcomeRecord,
    Provenance,
)


class MemoryResultWriter:
    def __init__(self, *, repository: MemoryRepository, authority: object, tenant_id: str) -> None:
        if repository is None or authority is None or not tenant_id.strip():
            raise ValueError("memory result writer requires host-owned dependencies")
        self._repository = repository
        self._authority = authority
        self._tenant_id = tenant_id

    def record(self, result: WorkflowResult) -> None:
        result = WorkflowResult.model_validate(result)
        now = datetime.now(timezone.utc)
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
            },
            provenance=Provenance(
                source_id="workflow:" + result.workflow_id,
                source_kind="system",
                independent_group="workflow:" + result.trace_id,
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
            verified=False,
            evidence_ids=(event_id,),
        )
        self._repository.append_outcome(outcome, idempotency_key=prefix + ":outcome", **context)
