from __future__ import annotations

import pytest
from pydantic import ValidationError

from market_agent.workflow_contracts import WorkflowMode
from market_agent.workflow_harness_contracts import (
    AttemptWorkItemOwnershipRecord,
    AttemptState,
    HarnessOutcome,
    HarnessPlan,
    HarnessSessionView,
    HarnessTransition,
    LeaseToken,
    OutcomeKind,
    PinnedVersions,
    ProgressTargetSet,
    ProgressVector,
    ReconciliationResolutionRecord,
    RiskClass,
    RunState,
    StageSpec,
    TaskKind,
    TransitionAuthorityRecord,
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
    ("state", "kind", "knowledge", "reason"),
    [
        (RunState.SUCCEEDED, OutcomeKind.ANSWER, "known", "completed"),
        (
            RunState.SUCCEEDED,
            OutcomeKind.ANSWER,
            "partial",
            "fixed_seed_cache_hit",
        ),
        (
            RunState.SUCCEEDED,
            OutcomeKind.ANSWER,
            "known",
            "compatible_semantic_cache_hit",
        ),
        (
            RunState.SUCCEEDED,
            OutcomeKind.NO_TRADE,
            "known",
            "strategy_no_trade",
        ),
        (
            RunState.SUCCEEDED,
            OutcomeKind.NO_TRADE,
            "known",
            "risk_gate_no_trade",
        ),
        (
            RunState.DEGRADED,
            OutcomeKind.ANSWER,
            "known",
            "lower_model_fallback",
        ),
        (
            RunState.DEGRADED,
            OutcomeKind.ANSWER,
            "partial",
            "verified_local_knowledge_fallback",
        ),
        (
            RunState.DEGRADED,
            OutcomeKind.UNKNOWN,
            "unknown",
            "insufficient_evidence",
        ),
        (
            RunState.DEGRADED,
            OutcomeKind.UNKNOWN,
            "unknown",
            "confidence_recovery_exhausted",
        ),
        (
            RunState.DEGRADED,
            OutcomeKind.UNKNOWN,
            "unknown",
            "dependency_unavailable",
        ),
        (
            RunState.DEGRADED,
            OutcomeKind.NO_TRADE,
            "unknown",
            "safe_no_trade_due_to_degradation",
        ),
        (
            RunState.DEGRADED,
            OutcomeKind.NO_TRADE,
            "partial",
            "safe_no_trade_due_to_degradation",
        ),
        (
            RunState.FAILED,
            OutcomeKind.NONE,
            "not_applicable",
            "permanent_policy",
        ),
        (RunState.FAILED, OutcomeKind.NONE, "not_applicable", "integrity"),
        (RunState.FAILED, OutcomeKind.NONE, "not_applicable", "audit"),
        (
            RunState.FAILED,
            OutcomeKind.NONE,
            "not_applicable",
            "configuration_failure",
        ),
        (
            RunState.CANCELLED,
            OutcomeKind.NONE,
            "not_applicable",
            "cancellation_completed",
        ),
    ],
)
def test_terminal_outcome_accepts_only_declared_state_kind_knowledge_rows(
    state, kind, knowledge, reason
):
    assert outcome(state, kind, knowledge, reason).terminal_state is state


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


@pytest.mark.parametrize(
    ("state", "kind", "knowledge", "reason"),
    [
        (
            RunState.FAILED,
            OutcomeKind.NONE,
            "not_applicable",
            "cancellation_completed",
        ),
        (
            RunState.SUCCEEDED,
            OutcomeKind.ANSWER,
            "known",
            "safe_no_trade_due_to_degradation",
        ),
        (
            RunState.DEGRADED,
            OutcomeKind.NO_TRADE,
            "unknown",
            "completed",
        ),
        (
            RunState.SUCCEEDED,
            OutcomeKind.NO_TRADE,
            "known",
            "unregistered_reason",
        ),
    ],
)
def test_terminal_outcome_rejects_reason_from_another_mapping_row(
    state, kind, knowledge, reason
):
    with pytest.raises(ValidationError):
        outcome(state, kind, knowledge, reason)


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


def test_non_run_transition_requires_durable_lease_epoch_and_token_digest():
    values = {
        "run_id": "run-1",
        "trace_id": "trace-1",
        "entity_kind": "work_item",
        "entity_id": "work-1",
        "from_state": "ready",
        "to_state": "leased",
        "expected_state_revision": 1,
        "plan_revision": 0,
        "reason_code": "lease_acquired",
        "idempotency_key": "lease-1",
    }

    with pytest.raises(ValidationError):
        HarnessTransition(**values)

    transition = HarnessTransition(
        **values, lease_epoch=1, fencing_token_digest=HASH
    )

    assert (transition.lease_epoch, transition.fencing_token_digest) == (1, HASH)


def test_run_transition_rejects_lease_evidence_and_raw_fencing_token():
    values = {
        "run_id": "run-1",
        "trace_id": "trace-1",
        "entity_kind": "run",
        "entity_id": "run-1",
        "from_state": "none",
        "to_state": "created",
        "expected_state_revision": 0,
        "plan_revision": 0,
        "reason_code": "run_created",
        "idempotency_key": "create-1",
    }

    with pytest.raises(ValidationError):
        HarnessTransition(**values, lease_epoch=1, fencing_token_digest=HASH)
    with pytest.raises(ValidationError):
        HarnessTransition(**values, fencing_token="fence-live-secret")


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


@pytest.mark.parametrize(
    ("run_state", "sealed_outcome"),
    [
        (None, None),
        (
            RunState.FAILED,
            outcome(
                RunState.FAILED,
                OutcomeKind.NONE,
                "not_applicable",
                "permanent_policy",
            ),
        ),
        (
            RunState.CANCELLED,
            outcome(
                RunState.CANCELLED,
                OutcomeKind.NONE,
                "not_applicable",
                "cancellation_completed",
            ),
        ),
        (
            RunState.SUCCEEDED,
            outcome(RunState.SUCCEEDED, OutcomeKind.ANSWER, "known", "completed"),
        ),
        (
            RunState.DEGRADED,
            outcome(
                RunState.DEGRADED,
                OutcomeKind.UNKNOWN,
                "unknown",
                "dependency_unavailable",
            ),
        ),
    ],
)
def test_unknown_external_side_effect_requires_unsealed_reconciliation_state(
    run_state, sealed_outcome
):
    with pytest.raises(ValidationError):
        HarnessSessionView(
            run_id="run-1",
            trace_id="trace-1",
            run_state=run_state,
            outcome=sealed_outcome,
            external_side_effect_unknown=True,
        )


def test_unknown_external_side_effect_accepts_waiting_reconciliation_without_outcome():
    view = HarnessSessionView(
        run_id="run-1",
        trace_id="trace-1",
        run_state=RunState.WAITING_RECONCILIATION,
        external_side_effect_unknown=True,
    )

    assert view.outcome is None


def test_folded_authority_records_are_strict_and_never_contain_live_tokens():
    authority = TransitionAuthorityRecord(
        run_id="run-1",
        trace_id="trace-1",
        entity_kind="work_item",
        entity_id="work-1",
        from_state="ready",
        to_state="leased",
        expected_state_revision=3,
        plan_revision=2,
        reason_code="lease_acquired",
        idempotency_key="authority-1",
        dependency_versions=(("input", 7),),
        reservation_id="reservation-1",
        grant_id="grant-1",
        lease_epoch=4,
        fencing_token_digest=HASH,
    )
    ownership = AttemptWorkItemOwnershipRecord(
        run_id="run-1",
        trace_id="trace-1",
        attempt_id="attempt-1",
        work_item_id="work-1",
        plan_revision=2,
    )
    resolution = ReconciliationResolutionRecord(
        run_id="run-1",
        trace_id="trace-1",
        reconciliation_id="broker-observation-1",
        expected_state_revision=3,
        plan_revision=2,
        broker_observation_digest=HASH,
        side_effect_resolved=True,
    )

    assert authority.fencing_token_digest == HASH
    assert ownership.attempt_id == "attempt-1"
    assert resolution.side_effect_resolved is True
    with pytest.raises(ValidationError):
        TransitionAuthorityRecord(
            **{**authority.model_dump(), "fencing_token": "live"}
        )


@pytest.mark.parametrize("second_digest", [HASH, "b" * 64])
def test_reconciliation_resolution_scope_is_unique_across_caller_chosen_ids(
    second_digest,
):
    first = ReconciliationResolutionRecord(
        run_id="run-1",
        trace_id="trace-1",
        reconciliation_id="broker-observation-1",
        expected_state_revision=3,
        plan_revision=2,
        broker_observation_digest=HASH,
        side_effect_resolved=True,
    )
    second = first.model_copy(
        update={
            "reconciliation_id": "broker-observation-2",
            "broker_observation_digest": second_digest,
        }
    )

    with pytest.raises(ValidationError, match="reconciliation resolution"):
        HarnessSessionView(
            run_id="run-1",
            trace_id="trace-1",
            reconciliation_resolutions=(first, second),
        )
