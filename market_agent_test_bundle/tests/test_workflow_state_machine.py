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
    AttemptTransitionAuthorization,
    GlobalTaskStateMachine,
    InvalidTransitionError,
    PermanentFailureDecision,
    ReconciliationResolution,
    RunTransitionEvidence,
    StaleAttemptRetryAuthorization,
    WorkItemTransitionAuthorization,
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


def authorization_for(
    candidate: HarnessTransition, view: HarnessSessionView
) -> (
    RunTransitionEvidence
    | WorkItemTransitionAuthorization
    | AttemptTransitionAuthorization
):
    common = {
        "run_id": candidate.run_id,
        "trace_id": candidate.trace_id,
        "entity_id": candidate.entity_id,
        "expected_state_revision": view.state_revision,
        "plan_revision": view.plan_revision,
        "dependency_versions": view.dependency_versions,
    }
    if candidate.entity_kind == "run":
        return RunTransitionEvidence(**common)
    evidence = {
        **common,
        "reservation_id": "reservation-1",
        "grant_id": "grant-1",
        "lease_epoch": candidate.lease_epoch,
        "fencing_token_digest": candidate.fencing_token_digest,
    }
    if candidate.entity_kind == "work_item":
        return WorkItemTransitionAuthorization(**evidence)
    return AttemptTransitionAuthorization(**evidence)


def validated(
    machine: GlobalTaskStateMachine,
    candidate: HarnessTransition | PermanentFailureDecision,
    view: HarnessSessionView,
    **kwargs: object,
):
    if isinstance(candidate, PermanentFailureDecision):
        return machine.validate(candidate, view, **kwargs)
    return machine.validate(
        candidate,
        view,
        authorization=authorization_for(candidate, view),
        **kwargs,
    )


def applied(
    machine: GlobalTaskStateMachine,
    candidate: HarnessTransition | PermanentFailureDecision,
    view: HarnessSessionView,
    **kwargs: object,
) -> HarnessSessionView:
    if isinstance(candidate, PermanentFailureDecision):
        return machine.apply(candidate, view, **kwargs)
    return machine.apply(
        candidate,
        view,
        authorization=authorization_for(candidate, view),
        **kwargs,
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
    candidate = run_transition(source, target)
    assert validated(machine, candidate, run_view(source)).allowed


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
    candidate = work_transition(source, target)
    if target is WorkItemState.RETRY_WAIT:
        view = work_view(source, attempt_states=(("attempt-1", AttemptState.STALE),))
        retry = StaleAttemptRetryAuthorization(
            run_id="run-1",
            trace_id="trace-1",
            work_item_id="work-1",
            attempt_id="attempt-1",
            expected_state_revision=3,
            plan_revision=2,
            lease_epoch=4,
            fencing_token_digest=HASH,
        )
        assert machine.validate(
            candidate,
            view,
            authorization=authorization_for(candidate, view),
            retry_authorization=retry,
        ).allowed
    else:
        assert validated(machine, candidate, work_view(source)).allowed


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
    candidate = attempt_transition(source, target)
    assert validated(machine, candidate, attempt_view(source)).allowed


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
    decision = validated(machine, candidate, view)
    assert not decision.allowed
    assert "terminal" in decision.reason


def test_unknown_external_effect_forbids_failed_and_cancelled(machine):
    view = run_view(
        RunState.WAITING_RECONCILIATION, external_side_effect_unknown=True
    )
    for target in (RunState.FAILED, RunState.CANCELLED):
        assert not validated(
            machine, run_transition(view.run_state, target), view
        ).allowed


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
    assert all(not validated(machine, candidate, view).allowed for candidate in candidates)


def test_validation_checks_dependency_versions_and_durable_lease_identity(machine):
    view = work_view(WorkItemState.READY)
    candidate = work_transition(WorkItemState.READY, WorkItemState.LEASED)
    evidence = authorization_for(candidate, view)
    assert machine.validate(candidate, view, authorization=evidence).allowed
    assert not machine.validate(
        candidate,
        view,
        authorization=evidence.model_copy(
            update={"dependency_versions": (("input", 8),)}
        ),
    ).allowed
    assert not machine.validate(
        candidate, view, authorization=evidence.model_copy(update={"lease_epoch": 5})
    ).allowed
    assert not machine.validate(
        candidate,
        view,
        authorization=evidence.model_copy(update={"fencing_token_digest": "b" * 64}),
    ).allowed


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
    candidate = work_transition(WorkItemState.RUNNING, WorkItemState.RETRY_WAIT)
    retry = StaleAttemptRetryAuthorization(
        run_id="run-1",
        trace_id="trace-1",
        work_item_id="work-1",
        attempt_id="attempt-1",
        expected_state_revision=3,
        plan_revision=2,
        lease_epoch=4,
        fencing_token_digest=HASH,
    )
    assert machine.validate(
        candidate,
        view,
        authorization=authorization_for(candidate, view),
        retry_authorization=retry,
    ).allowed
    assert not validated(
        machine, attempt_transition(AttemptState.STALE, AttemptState.DISPATCHED), view
    ).allowed


def test_apply_is_pure_and_advances_only_valid_transition(machine):
    view = run_view(RunState.RUNNING)
    candidate = run_transition(RunState.RUNNING, RunState.SUMMARIZING)
    next_view = applied(machine, candidate, view)
    assert view.run_state is RunState.RUNNING
    assert (next_view.run_state, next_view.state_revision, next_view.sequence) == (
        RunState.SUMMARIZING,
        4,
        5,
    )
    assert candidate.idempotency_key in next_view.applied_idempotency_keys


def test_apply_rejects_invalid_transition_without_mutating_view(machine):
    view = run_view(RunState.SUCCEEDED)
    with pytest.raises(InvalidTransitionError):
        applied(machine, run_transition(RunState.SUCCEEDED, RunState.FAILED), view)
    assert view.run_state is RunState.SUCCEEDED


def test_permanent_failure_decision_emits_a_failed_run_transition(machine):
    view = run_view(RunState.ADMITTED)
    decision = PermanentFailureDecision(
        run_id="run-1",
        trace_id="trace-1",
        expected_state_revision=3,
        plan_revision=2,
        from_state=RunState.ADMITTED,
        reason_code="configuration_failure",
        idempotency_key="failure-1",
    )
    assert machine.validate(decision, view).allowed
    assert machine.apply(decision, view).run_state is RunState.FAILED


def test_raw_fencing_token_is_never_an_input_to_state_machine(machine):
    view = work_view(WorkItemState.READY)
    candidate = work_transition(WorkItemState.READY, WorkItemState.LEASED)
    with pytest.raises(TypeError):
        machine.validate(candidate, view, fencing_token="live-secret")


def test_transition_validation_fails_closed_when_required_evidence_is_omitted(machine):
    run = run_transition(RunState.RUNNING, RunState.SUMMARIZING)
    work = work_transition(WorkItemState.READY, WorkItemState.LEASED)
    attempt = attempt_transition(AttemptState.RESERVED, AttemptState.DISPATCHED)

    assert not machine.validate(run, run_view(RunState.RUNNING)).allowed
    assert not machine.validate(work, work_view(WorkItemState.READY)).allowed
    assert not machine.validate(attempt, attempt_view(AttemptState.RESERVED)).allowed


@pytest.mark.parametrize("field", ["reservation_id", "grant_id", "lease_epoch", "fencing_token_digest"])
def test_non_run_evidence_fails_closed_when_a_required_value_is_missing(machine, field):
    view = work_view(WorkItemState.READY)
    candidate = work_transition(WorkItemState.READY, WorkItemState.LEASED)
    values = authorization_for(candidate, view).model_dump(mode="python")
    values.pop(field)

    with pytest.raises(Exception):
        WorkItemTransitionAuthorization(**values)


def test_evidence_must_bind_candidate_identity_and_folded_dependency_versions(machine):
    view = work_view(WorkItemState.READY)
    candidate = work_transition(WorkItemState.READY, WorkItemState.LEASED)
    evidence = authorization_for(candidate, view)

    assert not machine.validate(
        candidate,
        view,
        authorization=evidence.model_copy(update={"entity_id": "work-2"}),
    ).allowed
    assert not machine.validate(
        candidate,
        view,
        authorization=evidence.model_copy(
            update={"dependency_versions": (("input", 8),)}
        ),
    ).allowed


def test_reconciling_without_typed_resolution_preserves_unknown_effect_and_blocks_terminal(machine):
    view = run_view(
        RunState.WAITING_RECONCILIATION, external_side_effect_unknown=True
    )
    reconciling = run_transition(
        RunState.WAITING_RECONCILIATION, RunState.RECONCILING
    )
    assert not validated(machine, reconciling, view).allowed
    with pytest.raises(InvalidTransitionError):
        applied(machine, reconciling, view)
    assert view.external_side_effect_unknown is True
    failed = run_transition(RunState.WAITING_RECONCILIATION, RunState.FAILED)
    assert not validated(machine, failed, view).allowed


def test_typed_reconciliation_resolution_is_required_to_clear_unknown_effect(machine):
    view = run_view(
        RunState.WAITING_RECONCILIATION, external_side_effect_unknown=True
    )
    candidate = run_transition(
        RunState.WAITING_RECONCILIATION, RunState.RECONCILING
    )
    resolution = ReconciliationResolution(
        run_id="run-1",
        trace_id="trace-1",
        entity_id="run-1",
        expected_state_revision=3,
        plan_revision=2,
        reconciliation_id="broker-observation-1",
        broker_observation_digest=HASH,
        side_effect_resolved=True,
    )

    resolved = applied(machine, candidate, view, reconciliation_resolution=resolution)
    assert resolved.external_side_effect_unknown is False
    assert validated(
        machine,
        run_transition(
            RunState.RECONCILING, RunState.FAILED, expected_state_revision=4
        ),
        resolved,
    ).allowed


def test_generic_run_transition_cannot_use_broad_permanent_failure_escape_hatch(machine):
    view = run_view(RunState.ADMITTED)
    generic = run_transition(RunState.ADMITTED, RunState.FAILED)

    assert not validated(machine, generic, view).allowed
    decision = PermanentFailureDecision(
        run_id="run-1",
        trace_id="trace-1",
        expected_state_revision=3,
        plan_revision=2,
        from_state=RunState.ADMITTED,
        reason_code="configuration_failure",
        idempotency_key="permanent-failure-1",
    )
    assert machine.validate(decision, view).allowed
    assert machine.apply(decision, view).run_state is RunState.FAILED


def test_retry_wait_requires_a_stale_attempt_authorization_owned_by_work_item(machine):
    view = HarnessSessionView(
        run_id="run-1",
        trace_id="trace-1",
        run_state=RunState.RUNNING,
        work_item_states=(("work-1", WorkItemState.RUNNING),),
        attempt_states=(("attempt-1", AttemptState.STALE),),
        state_revision=3,
        plan_revision=2,
    )
    candidate = work_transition(WorkItemState.RUNNING, WorkItemState.RETRY_WAIT)
    evidence = authorization_for(candidate, view)
    proof = StaleAttemptRetryAuthorization(
        run_id="run-1",
        trace_id="trace-1",
        work_item_id="work-1",
        attempt_id="attempt-1",
        expected_state_revision=3,
        plan_revision=2,
        lease_epoch=4,
        fencing_token_digest=HASH,
    )

    assert not machine.validate(candidate, view, authorization=evidence).allowed
    assert machine.validate(
        candidate,
        view,
        authorization=evidence,
        retry_authorization=proof,
    ).allowed
    for change in (
        {"work_item_id": "work-2"},
        {"attempt_id": "attempt-2"},
        {"lease_epoch": 5},
    ):
        assert not machine.validate(
            candidate,
            view,
            authorization=evidence,
            retry_authorization=proof.model_copy(update=change),
        ).allowed


def test_retry_authorization_rejects_nonstale_and_reopened_terminal_attempt(machine):
    view = HarnessSessionView(
        run_id="run-1",
        trace_id="trace-1",
        run_state=RunState.RUNNING,
        work_item_states=(("work-1", WorkItemState.RUNNING),),
        attempt_states=(("attempt-1", AttemptState.COMPLETED),),
        state_revision=3,
        plan_revision=2,
    )
    candidate = work_transition(WorkItemState.RUNNING, WorkItemState.RETRY_WAIT)
    proof = StaleAttemptRetryAuthorization(
        run_id="run-1",
        trace_id="trace-1",
        work_item_id="work-1",
        attempt_id="attempt-1",
        expected_state_revision=3,
        plan_revision=2,
        lease_epoch=4,
        fencing_token_digest=HASH,
    )

    assert not machine.validate(
        candidate,
        view,
        authorization=authorization_for(candidate, view),
        retry_authorization=proof,
    ).allowed
    assert not validated(
        machine,
        attempt_transition(AttemptState.COMPLETED, AttemptState.DISPATCHED),
        view,
    ).allowed


def test_public_state_machine_payloads_are_strict_frozen_contract_models():
    decision = PermanentFailureDecision(
        run_id="run-1",
        trace_id="trace-1",
        expected_state_revision=3,
        plan_revision=2,
        from_state=RunState.ADMITTED,
        reason_code="configuration_failure",
        idempotency_key="permanent-failure-1",
    )
    with pytest.raises(Exception):
        PermanentFailureDecision(**{**decision.model_dump(), "unexpected": True})
    with pytest.raises(Exception):
        decision.reason_code = "integrity"
