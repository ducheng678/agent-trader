from __future__ import annotations

import pytest
from pydantic import ValidationError

from market_agent.workflow_contracts import WorkflowMode
from market_agent.workflow_harness_contracts import (
    AttemptState,
    HarnessOutcome,
    HarnessPlan,
    HarnessSessionView,
    LeaseToken,
    OutcomeKind,
    PinnedVersions,
    ProgressTargetSet,
    ProgressVector,
    RiskClass,
    RunState,
    StageSpec,
    TaskKind,
    WorkItemSpec,
    WorkItemState,
    WorkerSpec,
)


HASH = "a" * 64


def target_set(**overrides: object) -> ProgressTargetSet:
    values: dict[str, object] = {
        "required_dependency_ids": (),
        "required_output_field_paths": ("result.summary",),
        "required_evidence_slot_ids": ("primary-source",),
        "required_source_coverage_weights": (("official-feed", 1.0),),
        "known_conflict_slot_ids": (),
        "risk_invariant_ids": ("no-unknown-side-effect",),
    }
    values.update(overrides)
    return ProgressTargetSet(**values)


def worker_spec(**overrides: object) -> WorkerSpec:
    values: dict[str, object] = {
        "worker_id": "fundamental-worker",
        "version": "v1",
        "supported_task_kinds": (TaskKind.FUNDAMENTAL,),
        "analysis_phases": ("collect", "compare", "conclude"),
        "input_schema_id": "FundamentalInput",
        "input_schema_hash": HASH,
        "output_schema_id": "FundamentalOutput",
        "output_schema_hash": HASH,
        "prompt_release": "fundamental-v1",
        "prompt_profile": "default",
        "model_routing_policy_key": "standard-analysis",
        "context_selector": "fundamental-context-v1",
        "context_token_budget": 8_000,
        "readable_state_keys": ("market_context",),
        "writable_invocation_state_key": "fundamental_result",
        "allowed_tool_capabilities": ("market_data.read",),
        "cacheable": True,
        "freshness_class": "intraday",
        "maximum_turns": 3,
        "maximum_tool_calls": 5,
        "maximum_input_tokens": 8_000,
        "maximum_output_tokens": 2_000,
        "timeout_seconds": 35.0,
        "maximum_attempts": 3,
        "maximum_cost": 0.25,
        "success_outcome": OutcomeKind.ANSWER,
        "failure_outcome": OutcomeKind.NONE,
        "degradation_outcome": OutcomeKind.UNKNOWN,
    }
    values.update(overrides)
    return WorkerSpec(**values)


def stage(stage_id: str, *, dependencies: tuple[str, ...] = ()) -> StageSpec:
    return StageSpec(
        stage_id=stage_id,
        version="v1",
        entry_predicate="dependencies_succeeded",
        completion_predicate="all_work_items_terminal",
        allowed_task_kinds=(TaskKind.FUNDAMENTAL,),
        dependencies=dependencies,
        maximum_concurrency=2,
        budget_policy_key="analysis-budget-v1",
        failure_outcome=OutcomeKind.NONE,
        degradation_outcome=OutcomeKind.UNKNOWN,
        allows_side_effects=False,
        allows_reconciliation=False,
    )


def work_item(work_item_id: str, *, dependencies: tuple[str, ...] = ()) -> WorkItemSpec:
    return WorkItemSpec(
        work_item_id=work_item_id,
        stage_id="analysis",
        worker_id="fundamental-worker",
        task_kind=TaskKind.FUNDAMENTAL,
        objective="Produce evidence-backed analysis",
        dependencies=dependencies,
        progress_targets=target_set(required_dependency_ids=dependencies),
    )


def pinned_versions() -> PinnedVersions:
    return PinnedVersions(
        plan_template_version="active-v1",
        policy_version="policy-v1",
        worker_registry_version="workers-v1",
        source_registry_version="sources-v1",
        prompt_bundle_hash=HASH,
        tool_registry_hash=HASH,
        output_schema_bundle_hash=HASH,
        fingerprint_schema_version="fingerprint-v1",
    )


def plan_values(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "plan_id": "plan-1",
        "run_id": "run-1",
        "trace_id": "trace-1",
        "template_id": "active-analysis",
        "revision": 1,
        "mode": WorkflowMode.ACTIVE,
        "task_kind": TaskKind.FUNDAMENTAL,
        "risk_class": RiskClass.TRADING,
        "pinned_versions": pinned_versions(),
        "stages": (stage("analysis"),),
        "workers": (worker_spec(),),
        "work_items": (work_item("a"),),
        "allows_side_effects": False,
    }
    values.update(overrides)
    return values


def outcome(
    terminal_state: RunState,
    outcome_kind: OutcomeKind,
    knowledge_status: str,
    terminal_reason: str,
) -> HarnessOutcome:
    return HarnessOutcome(
        terminal_state=terminal_state,
        outcome_kind=outcome_kind,
        knowledge_status=knowledge_status,
        terminal_reason=terminal_reason,
    )


def test_state_enums_cover_the_declared_global_state_machine():
    assert {state.value for state in WorkItemState} == {
        "pending",
        "ready",
        "leased",
        "running",
        "validating",
        "succeeded",
        "retry_wait",
        "blocked",
        "failed",
        "cancelled",
    }
    assert {state.value for state in AttemptState} == {
        "reserved",
        "dispatched",
        "streaming",
        "validating",
        "settling",
        "completed",
        "timed_out",
        "rejected",
        "failed",
        "stale",
        "cancelled",
    }


@pytest.mark.parametrize(
    "analysis_phases",
    [("one", "two"), ("one", "two", "three", "four", "five", "six")],
)
def test_worker_spec_requires_three_to_five_phases(analysis_phases):
    with pytest.raises(ValidationError):
        worker_spec(analysis_phases=analysis_phases)


def test_worker_spec_phases_are_strict_and_immutable():
    with pytest.raises(ValidationError):
        worker_spec(analysis_phases=["one", "two", "three"])

    spec = worker_spec()
    with pytest.raises(ValidationError):
        spec.analysis_phases = ("changed", "phase", "names")


def test_harness_plan_rejects_duplicate_and_unknown_identifiers():
    with pytest.raises(ValidationError):
        HarnessPlan(**plan_values(work_items=(work_item("a"), work_item("a"))))
    with pytest.raises(ValidationError):
        HarnessPlan(
            **plan_values(work_items=(work_item("a", dependencies=("missing",)),))
        )
    with pytest.raises(ValidationError):
        HarnessPlan(**plan_values(stages=(stage("analysis"), stage("analysis"))))


def test_harness_plan_rejects_dependency_cycles_at_both_levels():
    with pytest.raises(ValidationError):
        HarnessPlan(
            **plan_values(
                work_items=(
                    work_item("a", dependencies=("b",)),
                    work_item("b", dependencies=("a",)),
                )
            )
        )
    with pytest.raises(ValidationError):
        HarnessPlan(
            **plan_values(
                stages=(
                    stage("analysis", dependencies=("review",)),
                    stage("review", dependencies=("analysis",)),
                )
            )
        )


def test_harness_plan_rejects_unknown_worker_and_stage_references():
    with pytest.raises(ValidationError):
        HarnessPlan(
            **plan_values(
                work_items=(work_item("a").model_copy(update={"worker_id": "missing"}),)
            )
        )
    with pytest.raises(ValidationError):
        HarnessPlan(
            **plan_values(
                work_items=(work_item("a").model_copy(update={"stage_id": "missing"}),)
            )
        )


def test_progress_targets_are_bounded_and_canonical():
    with pytest.raises(ValidationError):
        target_set(required_evidence_slot_ids=tuple(f"slot-{i}" for i in range(65)))
    with pytest.raises(ValidationError):
        target_set(required_dependency_ids=("dependency", "dependency"))
    with pytest.raises(ValidationError):
        target_set(required_source_coverage_weights=(("source", 0.4), ("source", 0.6)))


@pytest.mark.parametrize("coverage", [-0.01, 1.01, float("nan")])
def test_progress_vector_requires_source_coverage_in_unit_interval(coverage):
    with pytest.raises(ValidationError):
        ProgressVector(fresh_authoritative_source_coverage=coverage)


@pytest.mark.parametrize(
    ("state", "kind", "knowledge"),
    [
        (RunState.SUCCEEDED, OutcomeKind.ANSWER, "known"),
        (RunState.SUCCEEDED, OutcomeKind.ANSWER, "partial"),
        (RunState.SUCCEEDED, OutcomeKind.NO_TRADE, "known"),
        (RunState.DEGRADED, OutcomeKind.ANSWER, "known"),
        (RunState.DEGRADED, OutcomeKind.ANSWER, "partial"),
        (RunState.DEGRADED, OutcomeKind.UNKNOWN, "unknown"),
        (RunState.DEGRADED, OutcomeKind.NO_TRADE, "unknown"),
        (RunState.DEGRADED, OutcomeKind.NO_TRADE, "partial"),
        (RunState.FAILED, OutcomeKind.NONE, "not_applicable"),
        (RunState.CANCELLED, OutcomeKind.NONE, "not_applicable"),
    ],
)
def test_terminal_outcome_accepts_only_declared_state_kind_knowledge_rows(
    state, kind, knowledge
):
    assert outcome(state, kind, knowledge, "declared_reason").terminal_state is state


@pytest.mark.parametrize(
    ("state", "kind", "knowledge"),
    [
        (RunState.RUNNING, OutcomeKind.ANSWER, "known"),
        (RunState.SUCCEEDED, OutcomeKind.NO_TRADE, "unknown"),
        (RunState.DEGRADED, OutcomeKind.UNKNOWN, "known"),
        (RunState.FAILED, OutcomeKind.ANSWER, "not_applicable"),
        (RunState.CANCELLED, OutcomeKind.NONE, "unknown"),
    ],
)
def test_terminal_outcome_rejects_undeclared_state_kind_knowledge_rows(
    state, kind, knowledge
):
    with pytest.raises(ValidationError):
        outcome(state, kind, knowledge, "invalid_combination")


def test_terminal_outcome_distinguishes_normal_and_degraded_no_trade():
    normal = outcome(
        RunState.SUCCEEDED, OutcomeKind.NO_TRADE, "known", "risk_gate_no_trade"
    )
    degraded = outcome(
        RunState.DEGRADED,
        OutcomeKind.NO_TRADE,
        "unknown",
        "safe_no_trade_due_to_degradation",
    )
    assert normal != degraded


def test_lease_token_rejects_nonpositive_epochs_and_is_frozen():
    with pytest.raises(ValidationError):
        LeaseToken(
            run_id="run-1",
            work_item_id="work-1",
            attempt_id="attempt-1",
            lease_epoch=0,
            fencing_token="fence-1",
            holder_id="worker-1",
            expires_at_monotonic=10.0,
        )

    lease = LeaseToken(
        run_id="run-1",
        work_item_id="work-1",
        attempt_id="attempt-1",
        lease_epoch=1,
        fencing_token="fence-1",
        holder_id="worker-1",
        expires_at_monotonic=10.0,
    )
    with pytest.raises(ValidationError):
        lease.lease_epoch = 2


def test_empty_session_view_has_replay_identity_and_no_run_state():
    assert HarnessSessionView.empty() == HarnessSessionView(
        sequence=0,
        state_revision=0,
        plan_revision=0,
        run_id=None,
        trace_id=None,
        run_state=None,
        outcome=None,
        work_item_states=(),
        attempt_states=(),
        dependency_versions=(),
        applied_idempotency_keys=(),
        external_side_effect_unknown=False,
        last_event_hash=None,
    )
