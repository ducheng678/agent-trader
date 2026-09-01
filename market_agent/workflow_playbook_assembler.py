from __future__ import annotations

from market_agent.workflow_contracts import (
    Action,
    DecisionDraft,
    KnowledgeStatus,
    RiskAssessment,
    TerminalMode,
    WorkflowResult,
)


def unknown_playbook(
    *, workflow_id: str, trace_id: str, reason: str, route_history: tuple[str, ...] = (),
) -> WorkflowResult:
    return WorkflowResult(
        workflow_id=workflow_id,
        trace_id=trace_id,
        knowledge_status=KnowledgeStatus.INSUFFICIENT,
        uncertainty_reason=reason,
        terminal_mode=TerminalMode.UNKNOWN,
        final_action=Action.NO_TRADE,
        route_history=route_history,
    )


def assemble_playbook(
    *,
    workflow_id: str,
    trace_id: str,
    decision: DecisionDraft | None,
    risk: RiskAssessment | None,
    evidence_references: tuple[str, ...] = (),
    route_history: tuple[str, ...] = (),
) -> WorkflowResult:
    if decision is None:
        return unknown_playbook(
            workflow_id=workflow_id,
            trace_id=trace_id,
            reason="decision draft is unavailable",
            route_history=route_history + ("safe_unknown",),
        )
    if decision.knowledge_status is KnowledgeStatus.INSUFFICIENT:
        return unknown_playbook(
            workflow_id=workflow_id,
            trace_id=trace_id,
            reason=decision.uncertainty_reason or "decision evidence is insufficient",
            route_history=route_history + ("safe_unknown",),
        )
    if decision.action is Action.NO_TRADE:
        return WorkflowResult(
            workflow_id=workflow_id,
            trace_id=trace_id,
            knowledge_status=KnowledgeStatus.KNOWN,
            uncertainty_reason=None,
            terminal_mode=TerminalMode.NO_TRADE,
            final_action=Action.NO_TRADE,
            evidence_references=evidence_references,
            route_history=route_history + ("no_trade",),
        )
    if risk is None or not risk.accepted or risk.knowledge_status is KnowledgeStatus.INSUFFICIENT:
        return unknown_playbook(
            workflow_id=workflow_id,
            trace_id=trace_id,
            reason="risk gate did not accept the decision",
            route_history=route_history + ("risk_rejected",),
        )
    return WorkflowResult(
        workflow_id=workflow_id,
        trace_id=trace_id,
        knowledge_status=KnowledgeStatus.KNOWN,
        uncertainty_reason=None,
        terminal_mode=TerminalMode.PLAYBOOK,
        final_action=decision.action,
        evidence_references=evidence_references,
        route_history=route_history + ("playbook_assembled",),
        playbook_payload=decision,
    )
