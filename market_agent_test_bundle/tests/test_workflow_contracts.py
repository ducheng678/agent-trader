from __future__ import annotations

from typing import get_args, get_type_hints

import pytest
from pydantic import ValidationError

from market_agent import workflow_state
from market_agent.workflow_contracts import (
    Action,
    AgentReport,
    AgentTask,
    ContextSummary,
    DataQuality,
    Digest,
    EventAlignment,
    EventAssessment,
    EventRelevance,
    ExtensionState,
    FundamentalAnalysis,
    KnowledgeStatus,
    MarketRegime,
    ModelTier,
    ReportStatus,
    SummaryCompleteness,
    TaskDifficulty,
    TaskType,
    TechnicalAnalysis,
    TerminalMode,
    TradeSetup,
    WorkflowBudgetState,
    WorkflowMode,
    WorkflowRequest,
    WorkflowResult,
)
from market_agent.workflow_state import TradingWorkflowState, merge_reports


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


def make_task(*, task_id: str = "task-1", objective: str = "Assess event direction") -> AgentTask:
    return AgentTask(
        task_id=task_id,
        workflow_id="workflow-1",
        trace_id="trace-1",
        task_type=TaskType.FUNDAMENTAL,
        objective=objective,
        context_summary_id="summary-1",
        allowed_data=("market_context",),
        allowed_tools=(),
        expected_output="FundamentalAnalysis",
        acceptance_criteria=("cite evidence",),
        difficulty=TaskDifficulty.NORMAL,
        model_tier=ModelTier.TERRA,
        prompt_version="v1",
        attempt_timeout_seconds=35,
        maximum_retries=2,
        reserved_cost=0.08,
        remaining_workflow_cost=0.75,
        analysis_steps=("read", "compare", "report"),
        escalation_rule="return_conflict",
        conflict_return_rule="coordinator",
    )


def nonviable_setup() -> TradeSetup:
    return TradeSetup(viable=False, confidence=0.0)


def viable_setup() -> TradeSetup:
    return TradeSetup(
        viable=True,
        confidence=0.5,
        entry_price=100.0,
        stop_price=90.0,
        observation_low=95.0,
        observation_high=105.0,
        candidate_condition="price confirms",
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


@pytest.mark.parametrize(
    ("left", "right"),
    [
        ([make_report(), make_report(summary="left duplicate")], []),
        ([], [make_report(), make_report(summary="right duplicate")]),
    ],
)
def test_merge_reports_detects_duplicates_within_each_update(left, right):
    merged = merge_reports(left, right)

    assert len(merged) == 1
    assert merged[0].status is ReportStatus.CONFLICT


def test_workflow_request_recursively_freezes_nested_json_values():
    event = {"levels": [{"price": 100.0}]}
    request = WorkflowRequest(
        workflow_id="workflow-1",
        trace_id="trace-1",
        user_query="Assess BTC",
        event_tape=(event,),
        trigger_reason="manual",
    )

    with pytest.raises(TypeError):
        request.event_tape[0]["levels"][0]["price"] = 101.0


def test_workflow_budget_applies_authoritative_mode_caps_and_rejects_invalid_state():
    active = WorkflowBudgetState(
        mode=WorkflowMode.ACTIVE,
        elapsed_seconds=0.0,
        remaining_cost=0.75,
        reserved_cost=0.0,
        settled_cost=0.0,
        remaining_attempts=10,
    )
    passive = WorkflowBudgetState(
        mode=WorkflowMode.PASSIVE,
        elapsed_seconds=0.0,
        remaining_cost=0.30,
        reserved_cost=0.0,
        settled_cost=0.0,
        remaining_attempts=10,
    )

    assert (active.time_cap_seconds, active.maximum_attempts, active.cost_cap) == (300.0, 10, 0.75)
    assert (passive.time_cap_seconds, passive.maximum_attempts, passive.cost_cap) == (130.0, 10, 0.30)
    with pytest.raises(ValidationError):
        WorkflowBudgetState(
            mode=WorkflowMode.PASSIVE,
            elapsed_seconds=131.0,
            remaining_cost=0.30,
            reserved_cost=0.0,
            settled_cost=0.0,
            remaining_attempts=10,
        )


def test_pending_and_running_task_state_use_deterministic_task_reducer():
    annotations = get_type_hints(TradingWorkflowState, include_extras=True)

    assert get_args(annotations["pending_tasks"])[0] == list[AgentTask]
    assert get_args(annotations["running_tasks"])[0] == list[AgentTask]
    assert workflow_state.merge_tasks([make_task()], [make_task()]) == [make_task()]
    with pytest.raises(ValueError):
        workflow_state.merge_tasks([make_task()], [make_task(objective="different task")])


def test_unknown_event_relevance_cannot_be_marked_known():
    with pytest.raises(ValidationError):
        EventAssessment(
            knowledge_status=KnowledgeStatus.KNOWN,
            uncertainty_reason=None,
            relevance=EventRelevance.UNKNOWN,
            impact_confidence=0.0,
            material_change=False,
        )


def test_insufficient_technical_data_requires_nonviable_setups_and_insufficient_status():
    with pytest.raises(ValidationError):
        TechnicalAnalysis(
            knowledge_status=KnowledgeStatus.KNOWN,
            uncertainty_reason=None,
            current_price=None,
            market_regime=MarketRegime.INSUFFICIENT_DATA,
            extension_state=ExtensionState.INSUFFICIENT_DATA,
            long_setup=nonviable_setup(),
            short_setup=nonviable_setup(),
            data_quality=DataQuality.INSUFFICIENT,
        )
    with pytest.raises(ValidationError):
        TechnicalAnalysis(
            knowledge_status=KnowledgeStatus.INSUFFICIENT,
            uncertainty_reason="chart data unavailable",
            current_price=None,
            market_regime=MarketRegime.INSUFFICIENT_DATA,
            extension_state=ExtensionState.INSUFFICIENT_DATA,
            long_setup=viable_setup(),
            short_setup=nonviable_setup(),
            data_quality=DataQuality.INSUFFICIENT,
        )


def test_unknown_workflow_result_cannot_be_marked_known():
    with pytest.raises(ValidationError):
        WorkflowResult(
            workflow_id="workflow-1",
            trace_id="trace-1",
            knowledge_status=KnowledgeStatus.KNOWN,
            uncertainty_reason=None,
            terminal_mode=TerminalMode.UNKNOWN,
            final_action=Action.NO_TRADE,
        )


def test_noncompleted_report_cannot_offer_directional_safe_fallback():
    with pytest.raises(ValidationError):
        AgentReport(
            task_id="task-1",
            workflow_id="workflow-1",
            trace_id="trace-1",
            status=ReportStatus.FAILED,
            knowledge_status=KnowledgeStatus.INSUFFICIENT,
            uncertainty_reason="runner unavailable",
            summary="no result",
            error_category="connection",
            safe_fallback=Action.LONG,
        )


def test_agent_outputs_include_a_fixed_schema_version():
    assessment = EventAssessment(
        knowledge_status=KnowledgeStatus.KNOWN,
        uncertainty_reason=None,
        relevance=EventRelevance.RELEVANT,
        impact_confidence=0.5,
        material_change=True,
    )

    assert assessment.schema_version == "v1"


def test_workflow_result_rejects_unstructured_playbook_payload():
    with pytest.raises(ValidationError):
        WorkflowResult(
            workflow_id="workflow-1",
            trace_id="trace-1",
            knowledge_status=KnowledgeStatus.KNOWN,
            uncertainty_reason=None,
            terminal_mode=TerminalMode.PLAYBOOK,
            final_action=Action.LONG,
            playbook_payload={"free_form": "payload"},
        )


@pytest.mark.parametrize("digest", ["A" * 64, "g" * 64, "short"])
def test_shared_digest_alias_rejects_noncanonical_nested_summary_hashes(digest):
    with pytest.raises(ValidationError):
        ContextSummary(
            summary_id="summary-1",
            task_id="task-1",
            workflow_id="workflow-1",
            trace_id="trace-1",
            user_objective="Assess BTC",
            token_estimate=0,
            completeness=SummaryCompleteness.COMPLETE,
            summary_version="v1",
            source_record_hash=digest,
        )


def test_contract_model_revalidates_constructed_and_copied_instances():
    valid = ContextSummary(
        summary_id="summary-1",
        task_id="task-1",
        workflow_id="workflow-1",
        trace_id="trace-1",
        user_objective="Assess BTC",
        token_estimate=0,
        completeness=SummaryCompleteness.COMPLETE,
        summary_version="v1",
        source_record_hash="a" * 64,
    )
    constructed = ContextSummary.model_construct(
        **{
            **valid.model_dump(mode="python"),
            "source_record_hash": "BAD",
            "immutable_constraints": [],
        },
    )
    copied = valid.model_copy(update={"source_record_hash": "BAD"})

    with pytest.raises(ValidationError):
        ContextSummary.model_validate(constructed)
    with pytest.raises(ValidationError):
        ContextSummary.model_validate(copied)
