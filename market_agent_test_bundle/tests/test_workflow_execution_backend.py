from __future__ import annotations

from typing import cast

import pytest
from pydantic import ValidationError

from market_agent.llm_workflow import LLMWorkflow
from market_agent.workflow_contracts import WorkflowMode
from market_agent.workflow_execution_backend import (
    CancelledExecutionError,
    DuplicateExecutionTransitionError,
    ExecutionBackend,
    ExecutionHandle,
    ExecutionHandleMismatchError,
    ExecutionIdentityError,
    ExecutionPlanMismatchError,
    InvalidExecutionInputError,
    LangGraphExecutionBackend,
    StaleExecutionTransitionError,
    UncommittedTransitionError,
    route_committed_transition,
)
from market_agent.workflow_harness_contracts import (
    HarnessPlan,
    HarnessSessionView,
    HarnessTransition,
    OutcomeKind,
    PinnedVersions,
    ProgressTargetSet,
    RiskClass,
    RunState,
    StageSpec,
    TaskKind,
    TransitionAuthorityRecord,
    WorkerSpec,
    WorkItemSpec,
)
from market_agent.workflow_state_machine import (
    GlobalTaskStateMachine,
    RunTransitionEvidence,
)


HASH = "a" * 64


def plan(**overrides: object) -> HarnessPlan:
    worker = WorkerSpec(
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
    stage = StageSpec(
        stage_id="information",
        version="stage-v1",
        entry_predicate="dependencies_succeeded",
        completion_predicate="work_item_completed",
        allowed_task_kinds=(TaskKind.INFORMATIONAL,),
        maximum_concurrency=1,
        budget_policy_key="bounded-budget-v1",
        failure_outcome=OutcomeKind.NONE,
        degradation_outcome=OutcomeKind.UNKNOWN,
        allows_side_effects=False,
        allows_reconciliation=False,
    )
    values: dict[str, object] = {
        "plan_id": "plan-1",
        "run_id": "run-1",
        "trace_id": "trace-1",
        "template_id": "passive-information-v1",
        "revision": 0,
        "mode": WorkflowMode.PASSIVE,
        "task_kind": TaskKind.INFORMATIONAL,
        "risk_class": RiskClass.INFORMATIONAL,
        "pinned_versions": PinnedVersions(
            plan_template_version="templates-v1",
            policy_version="policy-v1",
            worker_registry_version="workers-v1",
            source_registry_version="sources-v1",
            prompt_bundle_hash=HASH,
            tool_registry_hash=HASH,
            output_schema_bundle_hash=HASH,
            fingerprint_schema_version="fingerprint-v1",
        ),
        "stages": (stage,),
        "workers": (worker,),
        "work_items": (
            WorkItemSpec(
                work_item_id="information-work",
                stage_id="information",
                worker_id="information-worker",
                task_kind=TaskKind.INFORMATIONAL,
                objective="Produce a bounded informational answer.",
                progress_targets=ProgressTargetSet(
                    required_output_field_paths=("answer.summary",),
                    required_evidence_slot_ids=("accepted-source",),
                    required_source_coverage_weights=(("authoritative-source", 1.0),),
                    risk_invariant_ids=("no-side-effects",),
                ),
            ),
        ),
        "allows_side_effects": False,
    }
    values.update(overrides)
    return HarnessPlan(**values)


def view(**overrides: object) -> HarnessSessionView:
    values: dict[str, object] = {
        "sequence": 7,
        "state_revision": 4,
        "plan_revision": 0,
        "run_id": "run-1",
        "trace_id": "trace-1",
        "run_state": RunState.RUNNING,
    }
    values.update(overrides)
    return HarnessSessionView(**values)


def transition(**overrides: object) -> HarnessTransition:
    values: dict[str, object] = {
        "run_id": "run-1",
        "trace_id": "trace-1",
        "entity_kind": "run",
        "entity_id": "run-1",
        "from_state": RunState.RUNNING.value,
        "to_state": RunState.SUMMARIZING.value,
        "expected_state_revision": 4,
        "plan_revision": 0,
        "reason_code": "accepted_results_ready",
        "idempotency_key": "transition-1",
    }
    values.update(overrides)
    return HarnessTransition(**values)


@pytest.fixture
def backend() -> LangGraphExecutionBackend:
    return LangGraphExecutionBackend()


def test_backend_implements_runtime_protocol(backend: LangGraphExecutionBackend):
    assert isinstance(backend, ExecutionBackend)


def test_register_returns_frozen_strict_handle(
    backend: LangGraphExecutionBackend,
):
    handle = backend.register(plan(), view())

    assert handle == ExecutionHandle(
        run_id="run-1",
        trace_id="trace-1",
        plan_id="plan-1",
        plan_revision=0,
        state_revision=4,
        routed_state=RunState.RUNNING.value,
        cancelled=False,
    )
    with pytest.raises(ValidationError):
        handle.state_revision = 99  # type: ignore[misc]


def test_register_is_idempotent_for_the_same_plan_and_folded_view(
    backend: LangGraphExecutionBackend,
):
    first = backend.register(plan(), view())
    second = backend.register(plan(), view())
    assert second == first


@pytest.mark.parametrize("invalid", ({"run_id": "run-1"}, object()))
def test_register_rejects_non_contract_plan_values(
    backend: LangGraphExecutionBackend, invalid: object
):
    with pytest.raises(InvalidExecutionInputError):
        backend.register(cast(HarnessPlan, invalid), view())


def test_register_rejects_contract_subclasses(
    backend: LangGraphExecutionBackend,
):
    class PlanSubclass(HarnessPlan):
        pass

    subclass = PlanSubclass.model_validate(plan().model_dump(mode="python"))
    with pytest.raises(InvalidExecutionInputError):
        backend.register(subclass, view())


def test_register_rejects_model_copy_with_undeclared_fields(
    backend: LangGraphExecutionBackend,
):
    forged = plan().model_copy()
    object.__setattr__(forged, "raw_worker_candidate", {"goto": "succeeded"})
    with pytest.raises(InvalidExecutionInputError):
        backend.register(forged, view())


@pytest.mark.parametrize(
    "folded_view", (view(run_id="run-2"), view(trace_id="trace-2"))
)
def test_register_rejects_run_or_trace_mismatch(
    backend: LangGraphExecutionBackend, folded_view: HarnessSessionView
):
    with pytest.raises(ExecutionIdentityError):
        backend.register(plan(), folded_view)


def test_register_rejects_plan_revision_mismatch(
    backend: LangGraphExecutionBackend,
):
    with pytest.raises(ExecutionPlanMismatchError):
        backend.register(plan(), view(plan_revision=1))


def test_register_rejects_non_contract_or_subclass_views(
    backend: LangGraphExecutionBackend,
):
    class ViewSubclass(HarnessSessionView):
        pass

    subclass = ViewSubclass.model_validate(view().model_dump(mode="python"))
    for invalid in (view().model_dump(mode="python"), subclass):
        with pytest.raises(InvalidExecutionInputError):
            backend.register(plan(), cast(HarnessSessionView, invalid))


def test_raw_worker_candidate_cannot_select_edge(
    backend: LangGraphExecutionBackend,
):
    handle = backend.register(plan(), view())
    with pytest.raises(UncommittedTransitionError):
        backend.apply_committed_transition(
            handle,
            cast(HarnessTransition, {"goto": "succeeded", "retry": True}),
        )


def test_transition_subclass_cannot_select_edge(
    backend: LangGraphExecutionBackend,
):
    class TransitionSubclass(HarnessTransition):
        pass

    candidate = TransitionSubclass.model_validate(transition().model_dump(mode="python"))
    handle = backend.register(plan(), view())
    with pytest.raises(UncommittedTransitionError):
        backend.apply_committed_transition(handle, candidate)


def test_transition_with_undeclared_fields_cannot_select_edge(
    backend: LangGraphExecutionBackend,
):
    candidate = transition().model_copy()
    object.__setattr__(candidate, "model_selected_edge", "succeeded")
    handle = backend.register(plan(), view())
    with pytest.raises(UncommittedTransitionError):
        backend.apply_committed_transition(handle, candidate)


def test_route_committed_transition_accepts_only_exact_revalidated_transition():
    assert route_committed_transition(
        {"committed_transition": transition()}
    ) == RunState.SUMMARIZING.value
    with pytest.raises(UncommittedTransitionError):
        route_committed_transition(
            {"committed_transition": {"to_state": RunState.SUCCEEDED.value}}
        )


def test_apply_rejects_stale_or_forged_handle(
    backend: LangGraphExecutionBackend,
):
    handle = backend.register(plan(), view())
    forged = handle.model_copy(update={"state_revision": 3})
    with pytest.raises(ExecutionHandleMismatchError):
        backend.apply_committed_transition(forged, transition())


def test_apply_rejects_non_contract_or_subclass_handles(
    backend: LangGraphExecutionBackend,
):
    class HandleSubclass(ExecutionHandle):
        pass

    handle = backend.register(plan(), view())
    subclass = HandleSubclass.model_validate(handle.model_dump(mode="python"))
    for invalid in (handle.model_dump(mode="python"), subclass):
        with pytest.raises(ExecutionHandleMismatchError):
            backend.apply_committed_transition(
                cast(ExecutionHandle, invalid), transition()
            )


@pytest.mark.parametrize(
    ("candidate", "error_type"),
    (
        (transition(run_id="run-2", entity_id="run-2"), ExecutionIdentityError),
        (transition(trace_id="trace-2"), ExecutionIdentityError),
        (transition(plan_revision=1), ExecutionPlanMismatchError),
        (transition(expected_state_revision=3), StaleExecutionTransitionError),
    ),
)
def test_apply_rejects_identity_plan_and_revision_mismatch(
    backend: LangGraphExecutionBackend,
    candidate: HarnessTransition,
    error_type: type[Exception],
):
    handle = backend.register(plan(), view())
    with pytest.raises(error_type):
        backend.apply_committed_transition(handle, candidate)


def test_apply_projects_only_the_committed_transition_and_advances_one_revision(
    backend: LangGraphExecutionBackend,
):
    handle = backend.register(plan(), view())
    advanced = backend.apply_committed_transition(handle, transition())
    assert advanced.state_revision == 5
    assert advanced.routed_state == RunState.SUMMARIZING.value
    assert advanced.run_id == handle.run_id
    assert advanced.trace_id == handle.trace_id
    assert advanced.plan_revision == handle.plan_revision


def test_duplicate_and_stale_transitions_are_rejected(
    backend: LangGraphExecutionBackend,
):
    first = transition()
    handle = backend.register(plan(), view())
    advanced = backend.apply_committed_transition(handle, first)
    with pytest.raises(DuplicateExecutionTransitionError):
        backend.apply_committed_transition(advanced, first)
    with pytest.raises(StaleExecutionTransitionError):
        backend.apply_committed_transition(
            advanced, transition(idempotency_key="transition-2")
        )


def test_state_machine_committed_revision_and_backend_projection_agree(
    backend: LangGraphExecutionBackend,
):
    candidate = transition()
    evidence = RunTransitionEvidence(
        run_id="run-1",
        trace_id="trace-1",
        entity_id="run-1",
        expected_state_revision=4,
        plan_revision=0,
        dependency_versions=(),
    )
    authority = TransitionAuthorityRecord(
        run_id="run-1",
        trace_id="trace-1",
        entity_kind="run",
        entity_id="run-1",
        from_state=RunState.RUNNING.value,
        to_state=RunState.SUMMARIZING.value,
        expected_state_revision=4,
        plan_revision=0,
        reason_code="accepted_results_ready",
        idempotency_key="transition-1",
    )
    folded = view(transition_authorities=(authority,))
    committed = GlobalTaskStateMachine().apply(candidate, folded, authorization=evidence)
    handle = backend.register(plan(), folded)
    projected = backend.apply_committed_transition(handle, candidate)
    assert projected.state_revision == committed.state_revision
    assert projected.routed_state == committed.run_state.value


def test_resume_rebuilds_from_folded_view_not_disposable_checkpoint(
    backend: LangGraphExecutionBackend,
):
    folded = view(
        sequence=19,
        state_revision=9,
        run_state=RunState.RECONCILING,
        applied_idempotency_keys=("already-committed",),
    )
    stale_checkpoint = {
        "run_id": "attacker-run",
        "trace_id": "attacker-trace",
        "plan_revision": 99,
        "state_revision": 999,
        "routed_state": RunState.SUCCEEDED.value,
        "cancelled": True,
    }
    handle = backend.resume(plan(), folded, disposable_checkpoint=stale_checkpoint)
    assert handle.run_id == folded.run_id
    assert handle.trace_id == folded.trace_id
    assert handle.plan_revision == folded.plan_revision
    assert handle.state_revision == folded.state_revision
    assert handle.routed_state == folded.run_state.value
    assert not handle.cancelled


def test_resume_rejects_different_plan_for_an_existing_run(
    backend: LangGraphExecutionBackend,
):
    backend.register(plan(), view())

    with pytest.raises(ExecutionPlanMismatchError):
        backend.resume(plan(plan_id="different-plan"), view())


def test_resume_restores_duplicate_guard_from_folded_view(
    backend: LangGraphExecutionBackend,
):
    folded = view(applied_idempotency_keys=("transition-1",))
    handle = backend.resume(plan(), folded)
    with pytest.raises(DuplicateExecutionTransitionError):
        backend.apply_committed_transition(handle, transition())


def test_cancel_is_idempotent_by_run_id_and_blocks_further_projection(
    backend: LangGraphExecutionBackend,
):
    handle = backend.register(plan(), view())
    assert backend.cancel("run-1") is None
    assert backend.cancel("run-1") is None
    with pytest.raises(CancelledExecutionError):
        backend.apply_committed_transition(handle, transition())
    with pytest.raises(CancelledExecutionError):
        backend.resume(plan(), view())


def test_cancel_unknown_run_is_an_idempotent_no_op(
    backend: LangGraphExecutionBackend,
):
    assert backend.cancel("unknown-run") is None
    assert backend.cancel("unknown-run") is None


def test_cancel_rejects_non_string_and_blank_run_identifiers(
    backend: LangGraphExecutionBackend,
):
    for run_id in (1, True, "   "):
        with pytest.raises(InvalidExecutionInputError):
            backend.cancel(cast(str, run_id))


def test_legacy_llm_workflow_facade_remains_compatible():
    workflow = LLMWorkflow()
    assert workflow.run_single(lambda: "single-result") == "single-result"
    assert workflow.run_passive(
        judge=lambda: {"price_needed": False},
        should_price=lambda result: bool(result["price_needed"]),
        price=lambda result: {"price": 100, "judged": result},
        assemble=lambda result, pricing: (result, pricing),
    ) == ({"price_needed": False}, None)
