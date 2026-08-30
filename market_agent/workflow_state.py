from __future__ import annotations

import operator
from typing import Annotated, TypedDict

from market_agent.workflow_contracts import (
    Action,
    AgentReport,
    AgentTask,
    CachedAnswer,
    ContextSummary,
    CoordinatorPlan,
    DecisionDraft,
    EscalationReview,
    EventAssessment,
    FundamentalAnalysis,
    InformationalAnswer,
    KnowledgeStatus,
    MarketContextResult,
    ReportStatus,
    RiskAssessment,
    TechnicalAnalysis,
    WorkflowBudgetState,
    WorkflowError,
    WorkflowRequest,
)


def merge_reports(left: list[AgentReport], right: list[AgentReport]) -> list[AgentReport]:
    merged: dict[str, AgentReport] = {}
    for report in left + right:
        existing = merged.get(report.task_id)
        if existing is None:
            merged[report.task_id] = report
        elif existing != report:
            merged[report.task_id] = AgentReport(
                task_id=report.task_id,
                workflow_id=report.workflow_id,
                trace_id=report.trace_id,
                status=ReportStatus.CONFLICT,
                knowledge_status=KnowledgeStatus.INSUFFICIENT,
                uncertainty_reason="duplicate task reports disagree",
                summary="duplicate task reports require coordinator reconciliation",
                evidence_refs=tuple(dict.fromkeys(existing.evidence_refs + report.evidence_refs)),
                disputed_claims=(existing.summary, report.summary),
                missing_evidence=("coordinator reconciliation",),
                safe_fallback=Action.NO_TRADE,
            )
    return [merged[task_id] for task_id in sorted(merged)]


def merge_tasks(left: list[AgentTask], right: list[AgentTask]) -> list[AgentTask]:
    merged: dict[str, AgentTask] = {}
    for task in left + right:
        existing = merged.get(task.task_id)
        if existing is None:
            merged[task.task_id] = task
        elif existing != task:
            raise ValueError("duplicate task identifiers must be identical")
    return [merged[task_id] for task_id in sorted(merged)]


class TradingWorkflowState(TypedDict, total=False):
    request: WorkflowRequest
    context_summary: ContextSummary
    cached_answer: CachedAnswer
    market_context: MarketContextResult
    event_assessment: EventAssessment
    fundamental_analysis: FundamentalAnalysis
    technical_analysis: TechnicalAnalysis
    decision_draft: DecisionDraft
    risk_assessment: RiskAssessment
    escalation_review: EscalationReview
    final_playbook: dict[str, object]
    informational_answer: InformationalAnswer
    terminal_mode: str
    budget: WorkflowBudgetState
    coordinator_plan: CoordinatorPlan
    plan_revision: int
    pending_tasks: Annotated[list[AgentTask], merge_tasks]
    running_tasks: Annotated[list[AgentTask], merge_tasks]
    completed_tasks: Annotated[list[AgentReport], merge_reports]
    failed_tasks: Annotated[list[AgentReport], merge_reports]
    conflicted_tasks: Annotated[list[AgentReport], merge_reports]
    reports: Annotated[list[AgentReport], merge_reports]
    context_summaries: Annotated[list[ContextSummary], operator.add]
    usage_records: Annotated[list[dict[str, object]], operator.add]
    errors: Annotated[list[WorkflowError], operator.add]
    route_history: Annotated[list[str], operator.add]
    audit_sequence: int
    audit_healthy: bool
    conflict_set: Annotated[list[str], operator.add]
    reconciliation_history: Annotated[list[str], operator.add]
    final_knowledge_status: KnowledgeStatus
    unknown_reason: str | None
