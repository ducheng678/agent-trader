from __future__ import annotations

import pytest

from market_agent.workflow_harness_contracts import (
    AttemptWorkItemOwnershipRecord,
    AttemptState,
    HarnessSessionView,
    HarnessTransition,
    RunState,
    ReconciliationResolutionRecord,
    TransitionAuthorityRecord,
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
    authorization = authorization_for(candidate, view)
    view = folded_authority_view(view, candidate, authorization, **kwargs)
    return machine.validate(
        candidate,
        view,
        authorization=authorization,
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
    authorization = authorization_for(candidate, view)
    view = folded_authority_view(view, candidate, authorization, **kwargs)
    return machine.apply(
        candidate,
        view,
        authorization=authorization,
        **kwargs,
    )


def folded_authority_view(
    view: HarnessSessionView,
    candidate: HarnessTransition,
    authorization: RunTransitionEvidence
    | WorkItemTransitionAuthorization
    | AttemptTransitionAuthorization,
    **kwargs: object,
) -> HarnessSessionView:
    record_values: dict[str, object] = {
        "run_id": authorization.run_id,
        "trace_id": authorization.trace_id,
        "entity_kind": candidate.entity_kind,
        "entity_id": authorization.entity_id,
        "from_state": candidate.from_state,
        "to_state": candidate.to_state,
        "expected_state_revision": authorization.expected_state_revision,
        "plan_revision": authorization.plan_revision,
        "reason_code": candidate.reason_code,
        "idempotency_key": candidate.idempotency_key,
        "dependency_versions": authorization.dependency_versions,
    }
    if candidate.entity_kind != "run":
        record_values.update(
            {
                "reservation_id": authorization.reservation_id,
                "grant_id": authorization.grant_id,
                "lease_epoch": authorization.lease_epoch,
                "fencing_token_digest": authorization.fencing_token_digest,
            }
        )
    updates: dict[str, object] = {
        "transition_authorities": (
            *view.transition_authorities,
            TransitionAuthorityRecord(**record_values),
        )
    }
    retry = kwargs.get("retry_authorization")
    if isinstance(retry, StaleAttemptRetryAuthorization):
        updates["attempt_work_item_owners"] = (
            *view.attempt_work_item_owners,
            AttemptWorkItemOwnershipRecord(
                run_id=retry.run_id,
                trace_id=retry.trace_id,
                attempt_id=retry.attempt_id,
                work_item_id=retry.work_item_id,
                plan_revision=retry.plan_revision,
            ),
        )
    resolution = kwargs.get("reconciliation_resolution")
    if isinstance(resolution, ReconciliationResolution):
        updates["reconciliation_resolutions"] = (
            *view.reconciliation_resolutions,
            ReconciliationResolutionRecord(
                run_id=resolution.run_id,
                trace_id=resolution.trace_id,
                reconciliation_id=resolution.reconciliation_id,
                expected_state_revision=resolution.expected_state_revision,
                plan_revision=resolution.plan_revision,
                broker_observation_digest=resolution.broker_observation_digest,
                side_effect_resolved=resolution.side_effect_resolved,
            ),
        )
    return HarnessSessionView.model_validate(view.model_copy(update=updates).model_dump())


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
        authorization = authorization_for(candidate, view)
        view = folded_authority_view(
            view, candidate, authorization, retry_authorization=retry
        )
        assert machine.validate(
            candidate,
            view,
            authorization=authorization,
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
    view = folded_authority_view(view, candidate, evidence)
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
    authorization = authorization_for(candidate, view)
    view = folded_authority_view(
        view, candidate, authorization, retry_authorization=retry
    )
    assert machine.validate(
        candidate,
        view,
        authorization=authorization,
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
    authorized_view = folded_authority_view(
        view, candidate, evidence, retry_authorization=proof
    )
    assert machine.validate(
        candidate,
        authorized_view,
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
            authorized_view,
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


def test_mismatched_folded_authority_record_fails_even_when_candidate_proof_matches(
    machine,
):
    view = work_view(WorkItemState.READY)
    candidate = work_transition(WorkItemState.READY, WorkItemState.LEASED)
    evidence = authorization_for(candidate, view)
    authoritative = folded_authority_view(view, candidate, evidence)
    mismatched = authoritative.model_copy(
        update={
            "transition_authorities": (
                authoritative.transition_authorities[0].model_copy(
                    update={"grant_id": "another-grant"}
                ),
            )
        }
    )

    assert not machine.validate(
        candidate, mismatched, authorization=evidence
    ).allowed


def test_reconciliation_input_must_match_a_folded_resolution_record(machine):
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
    evidence = authorization_for(candidate, view)
    authoritative = folded_authority_view(view, candidate, evidence)

    assert not machine.validate(
        candidate,
        authoritative,
        authorization=evidence,
        reconciliation_resolution=resolution,
    ).allowed


def test_retry_requires_folded_attempt_ownership_not_only_retry_input(machine):
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
    authoritative = folded_authority_view(view, candidate, evidence)
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

    assert not machine.validate(
        candidate,
        authoritative,
        authorization=evidence,
        retry_authorization=retry,
    ).allowed


def test_orphan_nonrun_transition_fails_even_for_initial_none_state(machine):
    view = HarnessSessionView.empty()
    candidate = work_transition("none", WorkItemState.PENDING)
    evidence = WorkItemTransitionAuthorization(
        run_id="run-1",
        trace_id="trace-1",
        entity_id="work-1",
        expected_state_revision=0,
        plan_revision=0,
        dependency_versions=(),
        reservation_id="reservation-1",
        grant_id="grant-1",
        lease_epoch=4,
        fencing_token_digest=HASH,
    )

    assert not machine.validate(candidate, view, authorization=evidence).allowed


@pytest.mark.parametrize(
    "run_state",
    [RunState.SUCCEEDED, RunState.DEGRADED, RunState.FAILED, RunState.CANCELLED],
)
def test_terminal_run_rejects_every_nonrun_transition_including_initialization(
    machine, run_state
):
    view = HarnessSessionView(
        run_id="run-1", trace_id="trace-1", run_state=run_state
    )
    candidate = work_transition("none", WorkItemState.PENDING, expected_state_revision=0, plan_revision=0)
    evidence = WorkItemTransitionAuthorization(
        run_id="run-1",
        trace_id="trace-1",
        entity_id="work-1",
        expected_state_revision=0,
        plan_revision=0,
        dependency_versions=(),
        reservation_id="reservation-1",
        grant_id="grant-1",
        lease_epoch=4,
        fencing_token_digest=HASH,
    )
    authoritative = folded_authority_view(view, candidate, evidence)

    assert not machine.validate(
        candidate, authoritative, authorization=evidence
    ).allowed


def test_authority_cannot_authorize_another_branch_reason_or_idempotency(machine):
    view = run_view(RunState.RUNNING)
    authorized = run_transition(RunState.RUNNING, RunState.SUMMARIZING)
    authority = authorization_for(authorized, view)
    folded = folded_authority_view(view, authorized, authority)
    for candidate in (
        run_transition(RunState.RUNNING, RunState.RECONCILING),
        run_transition(
            RunState.RUNNING,
            RunState.SUMMARIZING,
            reason_code="another_reason",
        ),
        run_transition(
            RunState.RUNNING,
            RunState.SUMMARIZING,
            idempotency_key="another-idempotency-key",
        ),
    ):
        assert not machine.validate(
            candidate, folded, authorization=authorization_for(candidate, folded)
        ).allowed


def test_reconciliation_rejects_competing_record_in_the_same_semantic_scope(machine):
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
    evidence = authorization_for(candidate, view)
    authoritative = folded_authority_view(
        view,
        candidate,
        evidence,
        reconciliation_resolution=resolution,
    )
    competing = authoritative.reconciliation_resolutions[0].model_copy(
        update={
            "reconciliation_id": "broker-observation-2",
            "broker_observation_digest": "b" * 64,
        }
    )
    bypassed = authoritative.model_copy(
        update={
            "reconciliation_resolutions": (
                *authoritative.reconciliation_resolutions,
                competing,
            )
        }
    )

    assert not machine.validate(
        candidate,
        bypassed,
        authorization=evidence,
        reconciliation_resolution=resolution,
    ).allowed


def _assert_public_boundary_rejects(
    machine: GlobalTaskStateMachine,
    candidate: HarnessTransition | PermanentFailureDecision,
    view: HarnessSessionView,
    **kwargs: object,
) -> None:
    assert not machine.validate(candidate, view, **kwargs).allowed
    with pytest.raises(InvalidTransitionError):
        machine.apply(candidate, view, **kwargs)


def test_public_boundary_revalidates_conflicting_transition_authorities(machine):
    view = run_view(RunState.RUNNING)
    candidate = run_transition(RunState.RUNNING, RunState.SUMMARIZING)
    evidence = authorization_for(candidate, view)
    authoritative = folded_authority_view(view, candidate, evidence)
    competing = authoritative.transition_authorities[0].model_copy(
        update={"reason_code": "competing_authority"}
    )
    bypassed = authoritative.model_copy(
        update={
            "transition_authorities": (
                *authoritative.transition_authorities,
                competing,
            )
        }
    )

    _assert_public_boundary_rejects(
        machine, candidate, bypassed, authorization=evidence
    )


def test_public_boundary_revalidates_conflicting_attempt_ownership(machine):
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
    authoritative = folded_authority_view(
        view, candidate, evidence, retry_authorization=retry
    )
    competing = authoritative.attempt_work_item_owners[0].model_copy(
        update={"work_item_id": "work-2"}
    )
    bypassed = authoritative.model_copy(
        update={
            "attempt_work_item_owners": (
                *authoritative.attempt_work_item_owners,
                competing,
            )
        }
    )

    _assert_public_boundary_rejects(
        machine,
        candidate,
        bypassed,
        authorization=evidence,
        retry_authorization=retry,
    )


def test_public_boundary_revalidates_conflicting_reconciliation_resolutions(machine):
    resolution = ReconciliationResolutionRecord(
        run_id="run-1",
        trace_id="trace-1",
        reconciliation_id="broker-observation-1",
        expected_state_revision=3,
        plan_revision=2,
        broker_observation_digest=HASH,
        side_effect_resolved=True,
    )
    authoritative = run_view(
        RunState.ADMITTED, reconciliation_resolutions=(resolution,)
    )
    competing = resolution.model_copy(
        update={
            "reconciliation_id": "broker-observation-2",
            "broker_observation_digest": "b" * 64,
        }
    )
    bypassed = authoritative.model_copy(
        update={"reconciliation_resolutions": (resolution, competing)}
    )
    decision = PermanentFailureDecision(
        run_id="run-1",
        trace_id="trace-1",
        expected_state_revision=3,
        plan_revision=2,
        from_state=RunState.ADMITTED,
        reason_code="configuration_failure",
        idempotency_key="failure-invalid-resolution-view",
    )

    _assert_public_boundary_rejects(machine, decision, bypassed)


@pytest.mark.parametrize(
    "payload_kind",
    [
        "transition_cross_field",
        "transition_sensitive_value",
        "authorization_strict_value",
        "authorization_sensitive_value",
        "retry_strict_value",
        "resolution_cross_field",
        "decision_strict_value",
        "decision_sensitive_value",
        "decision_extra_field",
    ],
)
def test_public_boundary_revalidates_every_model_copy_crafted_payload(
    machine, payload_kind
):
    if payload_kind.startswith("transition"):
        view = run_view(RunState.RUNNING)
        candidate = run_transition(RunState.RUNNING, RunState.SUMMARIZING)
        if payload_kind == "transition_cross_field":
            forged_candidate = candidate.model_copy(
                update={"lease_epoch": 4, "fencing_token_digest": HASH}
            )
        else:
            forged_candidate = candidate.model_copy(
                update={"reason_code": "sk-live-secret"}
            )
        evidence = authorization_for(forged_candidate, view)
        authoritative = folded_authority_view(view, forged_candidate, evidence)
        inputs = (forged_candidate, authoritative, {"authorization": evidence})
    elif payload_kind.startswith("authorization"):
        view = work_view(WorkItemState.READY)
        candidate = work_transition(
            WorkItemState.READY, WorkItemState.LEASED, lease_epoch=1
        )
        evidence = authorization_for(candidate, view)
        update = (
            {"lease_epoch": True}
            if payload_kind == "authorization_strict_value"
            else {"grant_id": "sk-live-secret"}
        )
        forged_evidence = evidence.model_copy(update=update)
        authoritative = folded_authority_view(view, candidate, evidence)
        if payload_kind == "authorization_sensitive_value":
            sensitive_record = authoritative.transition_authorities[0].model_copy(
                update=update
            )
            authoritative = authoritative.model_copy(
                update={"transition_authorities": (sensitive_record,)}
            )
        inputs = (candidate, authoritative, {"authorization": forged_evidence})
    elif payload_kind == "retry_strict_value":
        view = work_view(
            WorkItemState.RUNNING,
            attempt_states=(("attempt-1", AttemptState.STALE),),
        )
        candidate = work_transition(
            WorkItemState.RUNNING, WorkItemState.RETRY_WAIT, lease_epoch=1
        )
        evidence = authorization_for(candidate, view)
        retry = StaleAttemptRetryAuthorization(
            run_id="run-1",
            trace_id="trace-1",
            work_item_id="work-1",
            attempt_id="attempt-1",
            expected_state_revision=3,
            plan_revision=2,
            lease_epoch=1,
            fencing_token_digest=HASH,
        )
        authoritative = folded_authority_view(
            view, candidate, evidence, retry_authorization=retry
        )
        inputs = (
            candidate,
            authoritative,
            {
                "authorization": evidence,
                "retry_authorization": retry.model_copy(
                    update={"lease_epoch": True}
                ),
            },
        )
    elif payload_kind == "resolution_cross_field":
        view = run_view(
            RunState.WAITING_RECONCILIATION,
            external_side_effect_unknown=True,
        )
        candidate = run_transition(
            RunState.WAITING_RECONCILIATION, RunState.RECONCILING
        )
        evidence = authorization_for(candidate, view)
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
        authoritative = folded_authority_view(
            view,
            candidate,
            evidence,
            reconciliation_resolution=resolution,
        )
        inputs = (
            candidate,
            authoritative,
            {
                "authorization": evidence,
                "reconciliation_resolution": resolution.model_copy(
                    update={"side_effect_resolved": False}
                ),
            },
        )
    else:
        view = run_view(RunState.ADMITTED, state_revision=1)
        decision = PermanentFailureDecision(
            run_id="run-1",
            trace_id="trace-1",
            expected_state_revision=1,
            plan_revision=2,
            from_state=RunState.ADMITTED,
            reason_code="configuration_failure",
            idempotency_key="failure-model-copy-boundary",
        )
        if payload_kind == "decision_strict_value":
            forged_decision = decision.model_copy(
                update={"expected_state_revision": True}
            )
        elif payload_kind == "decision_sensitive_value":
            forged_decision = decision.model_copy(
                update={"reason_code": "password=live-secret"}
            )
        else:
            forged_decision = decision.model_copy(update={"unexpected": True})
        inputs = (forged_decision, view, {})

    _assert_public_boundary_rejects(machine, inputs[0], inputs[1], **inputs[2])
