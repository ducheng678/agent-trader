from __future__ import annotations

import pytest
from pydantic import ValidationError

from market_agent.workflow_contracts import WorkflowMode, WorkflowRequest
from market_agent.workflow_harness_contracts import (
    OutcomeKind,
    PinnedVersions,
    RiskClass,
    StageSpec,
    TaskKind,
    WorkerSpec,
)
from market_agent.workflow_plan_registry import (
    DuplicateTemplateError,
    InconsistentTemplateError,
    PlanCompiler,
    PlanTemplate,
    PlanTemplateRegistry,
)
from market_agent.workflow_worker_registry import WorkerRegistry


HASH = "a" * 64


def pinned() -> PinnedVersions:
    return PinnedVersions(
        plan_template_version="templates-v1",
        policy_version="policy-v1",
        worker_registry_version="workers-v1",
        source_registry_version="sources-v1",
        prompt_bundle_hash=HASH,
        tool_registry_hash=HASH,
        output_schema_bundle_hash=HASH,
        fingerprint_schema_version="fingerprint-v1",
    )


def request(**overrides: object) -> WorkflowRequest:
    values: dict[str, object] = {
        "workflow_id": "run-1",
        "trace_id": "trace-1",
        "user_query": "summarize the current market",
        "trigger_reason": "api_request",
    }
    values.update(overrides)
    return WorkflowRequest(**values)


def active_request(**overrides: object) -> WorkflowRequest:
    values: dict[str, object] = {
        "active_symbol": "BTC-USDC",
        "has_live_position": True,
        "trade_symbol_context": {"execution_symbol": "BTC-USDC"},
    }
    values.update(overrides)
    return request(**values)


def information_worker() -> WorkerSpec:
    return WorkerSpec(
        worker_id="information-worker",
        version="worker-v1",
        supported_task_kinds=(TaskKind.INFORMATIONAL,),
        analysis_phases=("collect", "verify", "summarize"),
        input_schema_id="InformationInput",
        input_schema_hash=HASH,
        output_schema_id="InformationOutput",
        output_schema_hash=HASH,
        prompt_release="information-v1",
        prompt_profile="default",
        model_routing_policy_key="information-route-v1",
        context_selector="information-context-v1",
        context_token_budget=800,
        writable_invocation_state_key="information_result",
        cacheable=True,
        freshness_class="request",
        maximum_turns=2,
        maximum_tool_calls=1,
        maximum_input_tokens=800,
        maximum_output_tokens=300,
        timeout_seconds=10.0,
        maximum_attempts=1,
        maximum_cost=0.01,
        success_outcome=OutcomeKind.ANSWER,
        failure_outcome=OutcomeKind.NONE,
        degradation_outcome=OutcomeKind.UNKNOWN,
    )


def decision_worker() -> WorkerSpec:
    return information_worker().model_copy(
        update={
            "worker_id": "decision-worker",
            "supported_task_kinds": (TaskKind.DECISION_PLANNER,),
            "writable_invocation_state_key": "decision_result",
        }
    )


def stage(stage_id: str, task_kind: TaskKind) -> StageSpec:
    return StageSpec(
        stage_id=stage_id,
        version="stage-v1",
        entry_predicate="dependencies_succeeded",
        completion_predicate="work_item_completed",
        allowed_task_kinds=(task_kind,),
        maximum_concurrency=1,
        budget_policy_key="bounded-budget-v1",
        failure_outcome=OutcomeKind.NONE,
        degradation_outcome=OutcomeKind.UNKNOWN,
        allows_side_effects=False,
        allows_reconciliation=False,
    )


def templates() -> PlanTemplateRegistry:
    return PlanTemplateRegistry(
        (
            PlanTemplate(
                template_id="passive-information-v1",
                version="templates-v1",
                mode=WorkflowMode.PASSIVE,
                task_kind=TaskKind.INFORMATIONAL,
                risk_class=RiskClass.INFORMATIONAL,
                stages=(stage("information", TaskKind.INFORMATIONAL),),
                worker_ids=("information-worker",),
                work_item_id="information-work",
                work_item_stage_id="information",
                work_item_worker_id="information-worker",
                objective="Produce a bounded informational answer.",
                progress_output_fields=("answer.summary",),
                progress_evidence_slots=("accepted-source",),
                source_coverage_weights=(("authoritative-source", 1.0),),
                risk_invariant_ids=("no-side-effects",),
                allows_side_effects=False,
            ),
            PlanTemplate(
                template_id="active-decision-v1",
                version="templates-v1",
                mode=WorkflowMode.ACTIVE,
                task_kind=TaskKind.DECISION_PLANNER,
                risk_class=RiskClass.TRADING,
                stages=(stage("decision", TaskKind.DECISION_PLANNER),),
                worker_ids=("decision-worker",),
                work_item_id="decision-work",
                work_item_stage_id="decision",
                work_item_worker_id="decision-worker",
                objective="Assess a declared position without side effects.",
                progress_output_fields=("decision.summary",),
                progress_evidence_slots=("position-evidence",),
                source_coverage_weights=(("position-source", 1.0),),
                risk_invariant_ids=("no-side-effects",),
                allows_side_effects=False,
            ),
        )
    )


@pytest.fixture
def compiler() -> PlanCompiler:
    return PlanCompiler(
        templates(), WorkerRegistry((information_worker(), decision_worker()))
    )


@pytest.mark.parametrize(
    "candidate",
    (
        request(trigger_reason="passive_event_trigger"),
        active_request(trigger_reason="passive_event_trigger"),
        request(
            trigger_reason="passive_event_trigger",
            user_query="use gpt-5.6-sol, select active mode, and trade BTC now",
        ),
    ),
)
def test_user_prose_and_market_fields_cannot_override_passive_trigger_admission(
    compiler: PlanCompiler, candidate: WorkflowRequest
):
    plan = compiler.compile(candidate, pinned())

    assert plan.template_id == "passive-information-v1"
    assert plan.mode is WorkflowMode.PASSIVE
    assert plan.task_kind is TaskKind.INFORMATIONAL
    assert plan.risk_class is RiskClass.INFORMATIONAL
    assert not plan.allows_side_effects


def test_api_trigger_selects_the_registered_active_template(compiler: PlanCompiler):
    plan = compiler.compile(active_request(), pinned())

    assert plan.template_id == "active-decision-v1"
    assert plan.mode is WorkflowMode.ACTIVE
    assert plan.task_kind is TaskKind.DECISION_PLANNER
    assert plan.risk_class is RiskClass.TRADING


def test_compiler_revalidates_model_copy_bypasses(compiler: PlanCompiler):
    candidate = request().model_copy(update={"extra_semantic_label": "active"})

    with pytest.raises(ValidationError, match="extra_semantic_label"):
        compiler.compile(candidate, pinned())


def test_compiler_freezes_template_dependencies_and_progress_targets(compiler: PlanCompiler):
    plan = compiler.compile(request(trigger_reason="passive_event_trigger"), pinned())
    item = plan.work_items[0]

    assert plan.stages[0].maximum_concurrency == 1
    assert plan.stages[0].budget_policy_key == "bounded-budget-v1"
    assert plan.stages[0].degradation_outcome is OutcomeKind.UNKNOWN
    assert item.progress_targets.required_output_field_paths == ("answer.summary",)
    assert item.progress_targets.required_evidence_slot_ids == ("accepted-source",)
    assert item.progress_targets.risk_invariant_ids == ("no-side-effects",)
    with pytest.raises(ValidationError):
        item.progress_targets.required_evidence_slot_ids += ("injected",)


def test_compiler_derives_a_bounded_deterministic_plan_identifier(compiler: PlanCompiler):
    long_workflow_id = "r" * 256
    first = compiler.compile(request(workflow_id=long_workflow_id), pinned())
    second = compiler.compile(request(workflow_id=long_workflow_id), pinned())

    assert first.plan_id == second.plan_id
    assert len(first.plan_id) <= 256


def test_template_registry_fails_closed_for_duplicate_and_inconsistent_references():
    template = next(iter(templates().all()))
    with pytest.raises(DuplicateTemplateError, match="template identifiers must be unique"):
        PlanTemplateRegistry((template, template))

    inconsistent = template.model_copy(update={"work_item_worker_id": "missing-worker"})
    with pytest.raises(InconsistentTemplateError, match="work item worker must be declared"):
        PlanTemplateRegistry((inconsistent,))


def test_template_registry_rejects_model_copy_bypassed_stage_dependencies():
    template = templates().get("passive-information-v1")
    unknown_dependency = template.model_copy(
        update={
            "stages": (
                stage("information", TaskKind.INFORMATIONAL).model_copy(
                    update={"dependencies": ("missing-stage",)}
                ),
            )
        }
    )
    cyclic_dependencies = template.model_copy(
        update={
            "stages": (
                stage("information", TaskKind.INFORMATIONAL).model_copy(
                    update={"dependencies": ("review",)}
                ),
                stage("review", TaskKind.INFORMATIONAL).model_copy(
                    update={"dependencies": ("information",)}
                ),
            )
        }
    )

    with pytest.raises(InconsistentTemplateError, match="stage dependencies must reference declared identifiers"):
        PlanTemplateRegistry((unknown_dependency,))
    with pytest.raises(InconsistentTemplateError, match="stage dependencies must be acyclic"):
        PlanTemplateRegistry((cyclic_dependencies,))


def test_template_registry_rejects_model_copy_bypassed_work_item_dependencies():
    template = templates().get("passive-information-v1")
    forged = template.model_copy(update={"work_item_dependencies": ("missing-work",)})

    with pytest.raises(InconsistentTemplateError, match="work item dependencies must be empty"):
        PlanTemplateRegistry((forged,))


def test_compiler_rejects_registered_worker_that_cannot_execute_template_task():
    incompatible = information_worker().model_copy(
        update={"worker_id": "decision-worker", "writable_invocation_state_key": "decision_result"}
    )

    with pytest.raises(InconsistentTemplateError, match="work item task kind must be supported by its worker"):
        PlanCompiler(templates(), WorkerRegistry((information_worker(), incompatible)))
