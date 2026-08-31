from __future__ import annotations

import pytest

from market_agent.workflow_harness_contracts import (
    AttemptState,
    HarnessSessionView,
    HarnessTransition,
    RunState,
    WorkItemState,
)
from market_agent.workflow_state_machine import (
    GlobalTaskStateMachine,
    InvalidTransitionError,
    PermanentFailureDecision,
)


HASH = "a" * 64


def run_view(state: RunState | None, **updates: object) -> HarnessSessionView:
    values: dict[str, object] = {
        "sequence": 4,
        "state_revision": 3,
        "plan_revision": 2,
        "run_id": "run-1",
        "trace_id": "trace-1",
        "run_state": state,
        "dependency_versions": (("input", 7),),
    }
    values.update(updates)
    return HarnessSessionView(**values)


def work_view(state: WorkItemState | None, **updates: object) -> HarnessSessionView:
    values: dict[str, object] = {
        "sequence": 4,
        "state_revision": 3,
        "plan_revision": 2,
        "run_id": "run-1",
        "trace_id": "trace-1",
        "run_state": RunState.RUNNING,
        "work_item_states": (("work-1", state),) if state is not None else (),
        "dependency_versions": (("input", 7),),
    }
    values.update(updates)
    return HarnessSessionView(**values)


def attempt_view(state: AttemptState | None, **updates: object) -> HarnessSessionView:
    values: dict[str, object] = {
        "sequence": 4,
        "state_revision": 3,
        "plan_revision": 2,
        "run_id": "run-1",
        "trace_id": "trace-1",
        "run_state": RunState.RUNNING,
        "attempt_states": (("attempt-1", state),) if state is not None else (),
        "dependency_versions": (("input", 7),),
    }
    values.update(updates)
    return HarnessSessionView(**values)


def transition(
    entity_kind: str,
    entity_id: str,
    source: str,
    target: str,
    **updates: object,
) -> HarnessTransition:
    values: dict[str, object] = {
        "run_id": "run-1",
        "trace_id": "trace-1",
        "entity_kind": entity_kind,
        "entity_id": entity_id,
        "from_state": source,
        "to_state": target,
        "expected_state_revision": 3,
        "plan_revision": 2,
        "reason_code": "test_transition",
        "idempotency_key": f"{entity_kind}-{source}-{target}",
    }
    if entity_kind != "run":
        values.update({"lease_epoch": 4, "fencing_token_digest": HASH})
    values.update(updates)
    return HarnessTransition(**values)


def run_transition(
    source: RunState | str, target: RunState | str, **updates: object
) -> HarnessTransition:
    return transition(
        "run",
        "run-1",
        source.value if isinstance(source, RunState) else source,
        target.value if isinstance(target, RunState) else target,
        **updates,
    )


def work_transition(
    source: WorkItemState | str, target: WorkItemState | str, **updates: object
) -> HarnessTransition:
    return transition(
        "work_item",
        "work-1",
        source.value if isinstance(source, WorkItemState) else source,
        target.value if isinstance(target, WorkItemState) else target,
        **updates,
    )


def attempt_transition(
    source: AttemptState | str, target: AttemptState | str, **updates: object
) -> HarnessTransition:
    return transition(
        "attempt",
        "attempt-1",
        source.value if isinstance(source, AttemptState) else source,
        target.value if isinstance(target, AttemptState) else target,
        **updates,
    )


@pytest.fixture
def machine() -> GlobalTaskStateMachine:
    return GlobalTaskStateMachine()


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (RunState.CREATED, RunState.ADMITTED),
        (RunState.ADMITTED, RunState.PLANNED),
        (RunState.PLANNED, RunState.READY),
        (RunState.READY, RunState.RUNNING),
        (RunState.RUNNING, RunState.RECONCILING),
        (RunState.RUNNING, RunState.WAITING_APPROVAL),
        (RunState.RUNNING, RunState.WAITING_RECONCILIATION),
        (RunState.RECONCILING, RunState.WAITING_RECONCILIATION),
        (RunState.DEGRADING, RunState.DEGRADED),
        (RunState.SUMMARIZING, RunState.SUCCEEDED),
    ],
)
def test_declared_run_edges_are_legal(machine, source, target):
    assert machine.validate(run_transition(source, target), run_view(source)).allowed


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (WorkItemState.PENDING, WorkItemState.READY),
        (WorkItemState.READY, WorkItemState.LEASED),
        (WorkItemState.LEASED, WorkItemState.RETRY_WAIT),
        (WorkItemState.RUNNING, WorkItemState.RETRY_WAIT),
        (WorkItemState.VALIDATING, WorkItemState.RETRY_WAIT),
        (WorkItemState.RETRY_WAIT, WorkItemState.READY),
        (WorkItemState.VALIDATING, WorkItemState.SUCCEEDED),
    ],
)
def test_declared_work_item_edges_are_legal(machine, source, target):
    assert machine.validate(work_transition(source, target), work_view(source)).allowed


@pytest.mark.parametrize(
    ("source", "target"),
    [
        (AttemptState.RESERVED, AttemptState.DISPATCHED),
        (AttemptState.DISPATCHED, AttemptState.STREAMING),
        (AttemptState.DISPATCHED, AttemptState.VALIDATING),
        (AttemptState.STREAMING, AttemptState.VALIDATING),
        (AttemptState.VALIDATING, AttemptState.SETTLING),
        (AttemptState.SETTLING, AttemptState.COMPLETED),
        (AttemptState.DISPATCHED, AttemptState.STALE),
    ],
)
def test_declared_attempt_edges_are_legal(machine, source, target):
    assert machine.validate(attempt_transition(source, target), attempt_view(source)).allowed


@pytest.mark.parametrize(
    ("entity", "view", "candidate"),
    [
        (
            "run",
            run_view(RunState.SUCCEEDED),
            run_transition(RunState.SUCCEEDED, RunState.FAILED),
        ),
        (
            "work",
            work_view(WorkItemState.BLOCKED),
            work_transition(WorkItemState.BLOCKED, WorkItemState.READY),
        ),
        (
            "attempt",
            attempt_view(AttemptState.STALE),
            attempt_transition(AttemptState.STALE, AttemptState.DISPATCHED),
        ),
    ],
)
def test_terminal_states_are_absorbing(machine, entity, view, candidate):
    decision = machine.validate(candidate, view)
    assert not decision.allowed
    assert "terminal" in decision.reason


def test_unknown_external_effect_forbids_failed_and_cancelled(machine):
    view = run_view(
        RunState.WAITING_RECONCILIATION, external_side_effect_unknown=True
    )
    for target in (RunState.FAILED, RunState.CANCELLED):
        assert not machine.validate(run_transition(view.run_state, target), view).allowed


def test_validation_checks_folded_revision_plan_identity_and_idempotency(machine):
    view = run_view(RunState.RUNNING, applied_idempotency_keys=("already-applied",))
    candidates = (
        run_transition(
            RunState.RUNNING, RunState.SUMMARIZING, expected_state_revision=2
        ),
        run_transition(RunState.RUNNING, RunState.SUMMARIZING, plan_revision=1),
        run_transition(RunState.RUNNING, RunState.SUMMARIZING, trace_id="trace-2"),
        run_transition(
            RunState.RUNNING,
            RunState.SUMMARIZING,
            idempotency_key="already-applied",
        ),
    )
    assert all(not machine.validate(candidate, view).allowed for candidate in candidates)


def test_validation_checks_dependency_versions_and_durable_lease_identity(machine):
    view = work_view(WorkItemState.READY)
    candidate = work_transition(WorkItemState.READY, WorkItemState.LEASED)
    assert machine.validate(
        candidate,
        view,
        dependency_versions={"input": 7},
        lease_epoch=4,
        fencing_token_digest=HASH,
    ).allowed
    assert not machine.validate(candidate, view, dependency_versions={"input": 8}).allowed
    assert not machine.validate(candidate, view, lease_epoch=5).allowed
    assert not machine.validate(candidate, view, fencing_token_digest="b" * 64).allowed


def test_stale_attempt_can_drive_nonterminal_work_item_to_retry_wait(machine):
    view = HarnessSessionView(
        run_id="run-1",
        trace_id="trace-1",
        run_state=RunState.RUNNING,
        work_item_states=(("work-1", WorkItemState.RUNNING),),
        attempt_states=(("attempt-1", AttemptState.STALE),),
        state_revision=3,
        plan_revision=2,
    )
    assert machine.validate(
        work_transition(WorkItemState.RUNNING, WorkItemState.RETRY_WAIT), view
    ).allowed
    assert not machine.validate(
        attempt_transition(AttemptState.STALE, AttemptState.DISPATCHED), view
    ).allowed


def test_apply_is_pure_and_advances_only_valid_transition(machine):
    view = run_view(RunState.RUNNING)
    candidate = run_transition(RunState.RUNNING, RunState.SUMMARIZING)
    applied = machine.apply(candidate, view)
    assert view.run_state is RunState.RUNNING
    assert (applied.run_state, applied.state_revision, applied.sequence) == (
        RunState.SUMMARIZING,
        4,
        5,
    )
    assert candidate.idempotency_key in applied.applied_idempotency_keys


def test_apply_rejects_invalid_transition_without_mutating_view(machine):
    view = run_view(RunState.SUCCEEDED)
    with pytest.raises(InvalidTransitionError):
        machine.apply(run_transition(RunState.SUCCEEDED, RunState.FAILED), view)
    assert view.run_state is RunState.SUCCEEDED


def test_permanent_failure_decision_emits_a_failed_run_transition(machine):
    view = run_view(RunState.ADMITTED)
    decision = PermanentFailureDecision(
        reason_code="configuration_failure", idempotency_key="failure-1"
    )
    transition = decision.transition(view)
    assert transition.to_state == RunState.FAILED.value
    assert machine.apply(transition, view).run_state is RunState.FAILED


def test_raw_fencing_token_is_never_an_input_to_state_machine(machine):
    view = work_view(WorkItemState.READY)
    candidate = work_transition(WorkItemState.READY, WorkItemState.LEASED)
    with pytest.raises(TypeError):
        machine.validate(candidate, view, fencing_token="live-secret")
