from __future__ import annotations

import pytest
from pydantic import ValidationError

from market_agent.workflow_contracts import (
    Action,
    AgentReport,
    AgentTask,
    EventAlignment,
    FundamentalAnalysis,
    KnowledgeStatus,
    ModelTier,
    ReportStatus,
    TaskDifficulty,
    TaskType,
    WorkflowRequest,
)
from market_agent.workflow_state import merge_reports


def make_request() -> WorkflowRequest:
    return WorkflowRequest(
        workflow_id="workflow-1",
        trace_id="trace-1",
        user_query="Assess BTC",
        event_tape=(),
        trigger_reason="manual",
        active_symbol="BTC",
    )


def make_report(*, task_id: str = "task-1", summary: str = "completed") -> AgentReport:
    return AgentReport(
        task_id=task_id,
        workflow_id="workflow-1",
        trace_id="trace-1",
        status=ReportStatus.COMPLETED,
        knowledge_status=KnowledgeStatus.KNOWN,
        uncertainty_reason=None,
        summary=summary,
        evidence_refs=("event-1",),
    )


def test_workflow_request_forbids_extra_fields_and_is_immutable():
    with pytest.raises(ValidationError):
        WorkflowRequest(
            workflow_id="workflow-1",
            trace_id="trace-1",
            user_query="Assess BTC",
            event_tape=(),
            trigger_reason="manual",
            unexpected="value",
        )

    request = make_request()
    with pytest.raises(ValidationError):
        request.user_query = "Assess ETH"


def test_fundamental_analysis_rejects_nonfinite_confidence():
    with pytest.raises(ValidationError):
        FundamentalAnalysis(
            knowledge_status=KnowledgeStatus.KNOWN,
            uncertainty_reason=None,
            action=Action.LONG,
            direction_confidence=float("nan"),
            primary_driver="supportive event",
            supporting_factors=("event-1",),
            contradicting_factors=(),
            event_alignment=EventAlignment.REINFORCES,
        )


def test_insufficient_knowledge_cannot_report_confident_trade():
    with pytest.raises(ValidationError):
        FundamentalAnalysis(
            knowledge_status=KnowledgeStatus.INSUFFICIENT,
            uncertainty_reason="missing current evidence",
            action=Action.LONG,
            direction_confidence=0.8,
            primary_driver="unsupported",
            supporting_factors=(),
            contradicting_factors=(),
            event_alignment=EventAlignment.UNKNOWN,
        )


def test_task_step_bounds():
    common = {
        "task_id": "task-1",
        "workflow_id": "workflow-1",
        "trace_id": "trace-1",
        "task_type": TaskType.FUNDAMENTAL,
        "objective": "Assess event direction",
        "context_summary_id": "summary-1",
        "allowed_data": ("market_context",),
        "allowed_tools": (),
        "expected_output": "FundamentalAnalysis",
        "acceptance_criteria": ("cite evidence",),
        "difficulty": TaskDifficulty.NORMAL,
        "model_tier": ModelTier.TERRA,
        "prompt_version": "v1",
        "cache_key": None,
        "attempt_timeout_seconds": 35,
        "maximum_retries": 2,
        "reserved_cost": 0.08,
        "remaining_workflow_cost": 0.75,
        "escalation_rule": "return_conflict",
        "conflict_return_rule": "coordinator",
    }

    with pytest.raises(ValidationError):
        AgentTask(**common, analysis_steps=("read", "decide"))
    with pytest.raises(ValidationError):
        AgentTask(
            **common,
            analysis_steps=("one", "two", "three", "four", "five", "six"),
        )


def test_merge_reports_turns_duplicate_disagreement_into_conflict():
    merged = merge_reports([make_report()], [make_report(summary="different conclusion")])

    assert len(merged) == 1
    assert merged[0].status is ReportStatus.CONFLICT
    assert merged[0].task_id == "task-1"
    assert "duplicate" in merged[0].uncertainty_reason
