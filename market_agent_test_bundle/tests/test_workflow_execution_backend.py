from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import math
import secrets
from typing import cast

import pytest
from pydantic import ValidationError

from market_agent import workflow_execution_backend as execution_backend_module
from market_agent.llm_workflow import LLMWorkflow
from market_agent.workflow_contracts import WorkflowMode
from market_agent.workflow_execution_backend import (
    CancelledExecutionError,
    CommittedExecutionSnapshot,
    CommittedTransitionReceipt,
    DuplicateExecutionTransitionError,
    ExecutionBackend,
    ExecutionHandle,
    ExecutionHandleMismatchError,
    ExecutionIdentityError,
    ExecutionPlanMismatchError,
    ExecutionRegistrationError,
    ExecutionReceiptVerifier,
    IssuerTrustDescriptor,
    InvalidExecutionInputError,
    InvalidCommittedTransitionError,
    LangGraphExecutionBackend,
    RegistrationPreparation,
    RegistrationPreparationError,
    RegistrationTokenConsumedError,
    RegistrationTokenMismatchError,
    StaleExecutionSnapshotError,
    StaleExecutionTransitionError,
    UnverifiedExecutionReceiptError,
    UncommittedTransitionError,
    canonical_plan_digest,
    canonical_authority_signing_bytes,
    canonical_transition_digest,
    canonical_view_digest,
    route_committed_transition,
    verify_committed_execution_snapshot,
)
from market_agent.workflow_harness_contracts import (
    AttemptState,
    AttemptWorkItemOwnershipRecord,
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
    WorkItemState,
    WorkItemSpec,
)
from market_agent.workflow_session import HarnessEvent, SQLiteHarnessEventStore
from market_agent.workflow_state_machine import (
    GlobalTaskStateMachine,
    RunTransitionEvidence,
)


HASH = "a" * 64
VIEW_HASH = "b" * 64
POST_HASH = "c" * 64
NOW = datetime(2026, 8, 31, tzinfo=timezone.utc)


class TrustedReceiptVerifier:
    def __init__(
        self,
        *,
        modulus: int,
        private_exponent: int,
        key_id: str,
    ) -> None:
        self._modulus = modulus
        self._private_exponent = private_exponent
        self.key_id = key_id

    def approve(
        self, value: CommittedExecutionSnapshot | CommittedTransitionReceipt
    ) -> CommittedExecutionSnapshot | CommittedTransitionReceipt:
        payload = canonical_authority_signing_bytes(value)
        digest_info = (
            execution_backend_module._SHA256_DIGEST_INFO_PREFIX
            + hashlib.sha256(payload).digest()
        )
        size = 256
        encoded = b"\x00\x01" + b"\xff" * (size - len(digest_info) - 3) + b"\x00" + digest_info
        signature = pow(
            int.from_bytes(encoded, "big"),
            self._private_exponent,
            self._modulus,
        ).to_bytes(size, "big").hex()
        return type(value).model_validate(
            value.model_copy(update={"signature": signature}).model_dump(mode="python")
        )


def _is_probable_prime(candidate: int) -> bool:
    small_primes = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37)
    for prime in small_primes:
        if candidate % prime == 0:
            return candidate == prime
    odd = candidate - 1
    powers = 0
    while odd % 2 == 0:
        powers += 1
        odd //= 2
    for _ in range(24):
        base = secrets.randbelow(candidate - 3) + 2
        value = pow(base, odd, candidate)
        if value in (1, candidate - 1):
            continue
        for _ in range(powers - 1):
            value = pow(value, 2, candidate)
            if value == candidate - 1:
                break
        else:
            return False
    return True


def _generate_prime(bits: int) -> int:
    while True:
        candidate = secrets.randbits(bits) | 1 | (1 << (bits - 1))
        if _is_probable_prime(candidate):
            return candidate


def _generate_test_rsa_key() -> tuple[int, int, int]:
    exponent = 65537
    while True:
        left = _generate_prime(1024)
        right = _generate_prime(1024)
        if (
            left != right
            and (left * right).bit_length() == 2048
            and math.gcd(exponent, (left - 1) * (right - 1)) == 1
        ):
            modulus = left * right
            private = pow(exponent, -1, (left - 1) * (right - 1))
            return modulus, exponent, private


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
        "last_event_hash": VIEW_HASH,
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


@pytest.fixture(scope="session")
def test_rsa_key() -> tuple[int, int, int]:
    return _generate_test_rsa_key()


@pytest.fixture
def verifier(monkeypatch, test_rsa_key: tuple[int, int, int]) -> TrustedReceiptVerifier:
    modulus, exponent, private = test_rsa_key
    key_id = "test-host-rsa"
    keys = {key_id: (modulus, exponent)}
    monkeypatch.setattr(execution_backend_module, "_PINNED_PUBLIC_KEYS", keys)
    monkeypatch.setattr(
        execution_backend_module,
        "_EXPECTED_TRUST_CONFIG_DIGEST",
        execution_backend_module._trust_config_digest(keys),
    )
    return TrustedReceiptVerifier(
        modulus=modulus,
        private_exponent=private,
        key_id=key_id,
    )


_BACKEND_VERIFIERS: dict[LangGraphExecutionBackend, TrustedReceiptVerifier] = {}


@pytest.fixture
def backend(verifier: TrustedReceiptVerifier) -> LangGraphExecutionBackend:
    value = LangGraphExecutionBackend()
    _BACKEND_VERIFIERS[value] = verifier
    return value


def committed_snapshot(
    verifier: TrustedReceiptVerifier,
    plan_value: HarnessPlan | None = None,
    view_value: HarnessSessionView | None = None,
) -> CommittedExecutionSnapshot:
    plan_value = plan_value or plan()
    view_value = view_value or view()
    snapshot = CommittedExecutionSnapshot(
        run_id=plan_value.run_id,
        trace_id=plan_value.trace_id,
        plan_id=plan_value.plan_id,
        plan_digest=canonical_plan_digest(plan_value),
        plan_revision=plan_value.revision,
        sequence=view_value.sequence,
        state_revision=view_value.state_revision,
        view_digest=canonical_view_digest(view_value),
        event_head_hash=view_value.last_event_hash,
        folded_view=view_value,
        trust_key_id=verifier.key_id,
        signature="0" * 512,
    )
    return cast(CommittedExecutionSnapshot, verifier.approve(snapshot))


def test_public_snapshot_verifier_accepts_only_exact_pinned_authority(
    verifier: TrustedReceiptVerifier,
):
    snapshot = committed_snapshot(verifier)

    assert verify_committed_execution_snapshot(snapshot) is True
    assert verify_committed_execution_snapshot(
        snapshot.model_copy(update={"event_head_hash": "0" * 64})
    ) is False
    assert verify_committed_execution_snapshot(
        snapshot.model_copy(update={"trust_key_id": "untrusted-host"})
    ) is False
    forged_view = snapshot.folded_view.model_copy(
        update={"last_event_hash": "0" * 64}
    )
    assert verify_committed_execution_snapshot(
        snapshot.model_copy(update={"folded_view": forged_view})
    ) is False
    assert verify_committed_execution_snapshot(None) is False


def issuer_descriptor(verifier: TrustedReceiptVerifier) -> IssuerTrustDescriptor:
    return IssuerTrustDescriptor(
        trust_version=execution_backend_module._TRUST_VERSION,
        trust_config_digest=execution_backend_module._EXPECTED_TRUST_CONFIG_DIGEST,
        key_id=verifier.key_id,
    )


def provisional_view(plan_value: HarnessPlan) -> HarnessSessionView:
    return HarnessSessionView(
        plan_revision=plan_value.revision,
        run_id=plan_value.run_id,
        trace_id=plan_value.trace_id,
    )


def register_backend(
    backend: LangGraphExecutionBackend,
    plan_value: HarnessPlan | None = None,
    view_value: HarnessSessionView | None = None,
) -> ExecutionHandle:
    plan_value = plan_value or plan()
    view_value = view_value or view()
    return backend.register(
        plan_value,
        view_value,
        committed_snapshot(_BACKEND_VERIFIERS[backend], plan_value, view_value),
    )


def resume_backend(
    backend: LangGraphExecutionBackend,
    plan_value: HarnessPlan,
    view_value: HarnessSessionView,
    *,
    disposable_checkpoint: object | None = None,
) -> ExecutionHandle:
    return backend.resume(
        plan_value,
        view_value,
        committed_snapshot(_BACKEND_VERIFIERS[backend], plan_value, view_value),
        disposable_checkpoint=disposable_checkpoint,
    )


def committed_receipt(
    verifier: TrustedReceiptVerifier,
    transition_value: HarnessTransition,
    pre_view: HarnessSessionView,
    post_view: HarnessSessionView,
    *,
    plan_value: HarnessPlan | None = None,
) -> CommittedTransitionReceipt:
    plan_value = plan_value or plan()
    receipt = CommittedTransitionReceipt(
        pre=committed_snapshot(verifier, plan_value, pre_view),
        post=committed_snapshot(verifier, plan_value, post_view),
        transition_digest=canonical_transition_digest(transition_value),
        trust_key_id=verifier.key_id,
        signature="0" * 512,
    )
    return cast(CommittedTransitionReceipt, verifier.approve(receipt))


def authority_for(
    candidate: HarnessTransition,
    *,
    dependency_versions: tuple[tuple[str, int], ...] = (),
) -> TransitionAuthorityRecord:
    values: dict[str, object] = {
        "run_id": candidate.run_id,
        "trace_id": candidate.trace_id,
        "entity_kind": candidate.entity_kind,
        "entity_id": candidate.entity_id,
        "from_state": candidate.from_state,
        "to_state": candidate.to_state,
        "expected_state_revision": candidate.expected_state_revision,
        "plan_revision": candidate.plan_revision,
        "reason_code": candidate.reason_code,
        "idempotency_key": candidate.idempotency_key,
        "dependency_versions": dependency_versions,
    }
    if candidate.entity_kind != "run":
        values.update(
            {
                "reservation_id": "reservation-1",
                "grant_id": "grant-1",
                "lease_epoch": candidate.lease_epoch,
                "fencing_token_digest": candidate.fencing_token_digest,
            }
        )
    return TransitionAuthorityRecord(**values)


def post_view_for(
    pre_view: HarnessSessionView,
    candidate: HarnessTransition,
    **overrides: object,
) -> HarnessSessionView:
    changes: dict[str, object] = {
        "sequence": pre_view.sequence + 1,
        "state_revision": pre_view.state_revision + 1,
        "applied_idempotency_keys": (
            *pre_view.applied_idempotency_keys,
            candidate.idempotency_key,
        ),
        "last_event_hash": POST_HASH,
    }
    if candidate.entity_kind == "run":
        changes["run_state"] = RunState(candidate.to_state)
    elif candidate.entity_kind == "work_item":
        states = dict(pre_view.work_item_states)
        states[candidate.entity_id] = WorkItemState(candidate.to_state)
        changes["work_item_states"] = tuple(sorted(states.items()))
    else:
        states = dict(pre_view.attempt_states)
        states[candidate.entity_id] = AttemptState(candidate.to_state)
        changes["attempt_states"] = tuple(sorted(states.items()))
    changes.update(overrides)
    return HarnessSessionView.model_validate(
        pre_view.model_copy(update=changes).model_dump(mode="python")
    )


def authority_event(index: int, record: TransitionAuthorityRecord) -> HarnessEvent:
    return HarnessEvent(
        event_id=f"authority-{index}",
        trace_id=record.trace_id,
        span_id=f"span-authority-{index}",
        run_id=record.run_id,
        event_type="transition_authorized",
        occurred_at=NOW,
        monotonic_offset=float(index),
        actor="harness",
        payload={"authority": "committed"},
        transition_authority=record,
    )


def transition_event(index: int, candidate: HarnessTransition) -> HarnessEvent:
    return HarnessEvent(
        event_id=f"transition-{index}",
        trace_id=candidate.trace_id,
        span_id=f"span-transition-{index}",
        run_id=candidate.run_id,
        event_type="run_transitioned",
        occurred_at=NOW,
        monotonic_offset=float(index),
        actor="harness",
        payload={"transition": "committed"},
        transition=candidate,
    )


def append_run_transition(
    store: SQLiteHarnessEventStore,
    index: int,
    source: str,
    target: RunState,
) -> HarnessSessionView:
    before = store.snapshot("run-1")
    candidate = transition(
        from_state=source,
        to_state=target.value,
        expected_state_revision=before.state_revision,
        idempotency_key=f"transition-{index}",
        reason_code=f"advance-{index}",
    )
    store.append(
        authority_event(index, authority_for(candidate)),
        expected_sequence=before.sequence,
        expected_state_revision=before.state_revision,
    )
    authorized = store.snapshot("run-1")
    evidence = RunTransitionEvidence(
        run_id="run-1",
        trace_id="trace-1",
        entity_id="run-1",
        expected_state_revision=authorized.state_revision,
        plan_revision=0,
        dependency_versions=authorized.dependency_versions,
    )
    assert GlobalTaskStateMachine().validate(
        candidate, authorized, authorization=evidence
    ).allowed
    store.append(
        transition_event(index, candidate),
        expected_sequence=authorized.sequence,
        expected_state_revision=authorized.state_revision,
    )
    return store.snapshot("run-1")


def committed_running_to_summarizing(
    tmp_path,
) -> tuple[HarnessTransition, HarnessSessionView, HarnessSessionView]:
    store = SQLiteHarnessEventStore(tmp_path / "backend-authority.sqlite3")
    source = "none"
    for index, target in enumerate(
        (
            RunState.CREATED,
            RunState.ADMITTED,
            RunState.PLANNED,
            RunState.READY,
            RunState.RUNNING,
        ),
        start=1,
    ):
        append_run_transition(store, index, source, target)
        source = target.value

    before_authority = store.snapshot("run-1")
    candidate = transition(
        from_state=RunState.RUNNING.value,
        to_state=RunState.SUMMARIZING.value,
        expected_state_revision=before_authority.state_revision,
        idempotency_key="transition-6",
        reason_code="accepted-results-ready",
    )
    store.append(
        authority_event(6, authority_for(candidate)),
        expected_sequence=before_authority.sequence,
        expected_state_revision=before_authority.state_revision,
    )
    pre_view = store.snapshot("run-1")
    store.append(
        transition_event(6, candidate),
        expected_sequence=pre_view.sequence,
        expected_state_revision=pre_view.state_revision,
    )
    return candidate, pre_view, store.snapshot("run-1")


def test_backend_implements_runtime_protocol(backend: LangGraphExecutionBackend):
    assert isinstance(backend, ExecutionBackend)


def test_register_returns_frozen_strict_handle(
    backend: LangGraphExecutionBackend,
):
    handle = register_backend(backend)

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
    first = register_backend(backend)
    second = register_backend(backend)
    assert second == first


@pytest.mark.parametrize("invalid", ({"run_id": "run-1"}, object()))
def test_register_rejects_non_contract_plan_values(
    backend: LangGraphExecutionBackend, invalid: object
):
    with pytest.raises(InvalidExecutionInputError):
        backend.register(cast(HarnessPlan, invalid), view(), cast(CommittedExecutionSnapshot, None))


def test_register_rejects_contract_subclasses(
    backend: LangGraphExecutionBackend,
):
    class PlanSubclass(HarnessPlan):
        pass

    subclass = PlanSubclass.model_validate(plan().model_dump(mode="python"))
    with pytest.raises(InvalidExecutionInputError):
        backend.register(subclass, view(), cast(CommittedExecutionSnapshot, None))


def test_register_rejects_model_copy_with_undeclared_fields(
    backend: LangGraphExecutionBackend,
):
    forged = plan().model_copy()
    object.__setattr__(forged, "raw_worker_candidate", {"goto": "succeeded"})
    with pytest.raises(InvalidExecutionInputError):
        backend.register(forged, view(), cast(CommittedExecutionSnapshot, None))


@pytest.mark.parametrize(
    "folded_view", (view(run_id="run-2"), view(trace_id="trace-2"))
)
def test_register_rejects_run_or_trace_mismatch(
    backend: LangGraphExecutionBackend, folded_view: HarnessSessionView
):
    with pytest.raises(ExecutionIdentityError):
        backend.register(plan(), folded_view, cast(CommittedExecutionSnapshot, None))


def test_register_rejects_plan_revision_mismatch(
    backend: LangGraphExecutionBackend,
):
    with pytest.raises(ExecutionPlanMismatchError):
        backend.register(plan(), view(plan_revision=1), cast(CommittedExecutionSnapshot, None))


def test_register_rejects_non_contract_or_subclass_views(
    backend: LangGraphExecutionBackend,
):
    class ViewSubclass(HarnessSessionView):
        pass

    subclass = ViewSubclass.model_validate(view().model_dump(mode="python"))
    for invalid in (view().model_dump(mode="python"), subclass):
        with pytest.raises(InvalidExecutionInputError):
            backend.register(plan(), cast(HarnessSessionView, invalid), cast(CommittedExecutionSnapshot, None))


def test_raw_worker_candidate_cannot_select_edge(
    backend: LangGraphExecutionBackend,
):
    handle = register_backend(backend)
    with pytest.raises(UncommittedTransitionError):
        backend.apply_committed_transition(
            handle,
            cast(HarnessTransition, {"goto": "succeeded", "retry": True}),
            cast(HarnessSessionView, None),
            cast(HarnessSessionView, None),
            cast(CommittedTransitionReceipt, None),
        )


def test_transition_subclass_cannot_select_edge(
    backend: LangGraphExecutionBackend,
):
    class TransitionSubclass(HarnessTransition):
        pass

    candidate = TransitionSubclass.model_validate(transition().model_dump(mode="python"))
    handle = register_backend(backend)
    with pytest.raises(UncommittedTransitionError):
        backend.apply_committed_transition(handle, candidate, cast(HarnessSessionView, None), cast(HarnessSessionView, None), cast(CommittedTransitionReceipt, None))


def test_transition_with_undeclared_fields_cannot_select_edge(
    backend: LangGraphExecutionBackend,
):
    candidate = transition().model_copy()
    object.__setattr__(candidate, "model_selected_edge", "succeeded")
    handle = register_backend(backend)
    with pytest.raises(UncommittedTransitionError):
        backend.apply_committed_transition(handle, candidate, cast(HarnessSessionView, None), cast(HarnessSessionView, None), cast(CommittedTransitionReceipt, None))


def test_route_committed_transition_rejects_even_exact_transition():
    with pytest.raises(UncommittedTransitionError):
        route_committed_transition(cast(object, {"committed_transition": transition()}))


def test_apply_rejects_stale_or_forged_handle(
    backend: LangGraphExecutionBackend,
):
    handle = register_backend(backend)
    forged = handle.model_copy(update={"state_revision": 3})
    with pytest.raises(ExecutionHandleMismatchError):
        backend.apply_committed_transition(forged, transition(), cast(HarnessSessionView, None), cast(HarnessSessionView, None), cast(CommittedTransitionReceipt, None))


def test_apply_rejects_non_contract_or_subclass_handles(
    backend: LangGraphExecutionBackend,
):
    class HandleSubclass(ExecutionHandle):
        pass

    handle = register_backend(backend)
    subclass = HandleSubclass.model_validate(handle.model_dump(mode="python"))
    for invalid in (handle.model_dump(mode="python"), subclass):
        with pytest.raises(ExecutionHandleMismatchError):
            backend.apply_committed_transition(
                cast(ExecutionHandle, invalid), transition(), cast(HarnessSessionView, None), cast(HarnessSessionView, None), cast(CommittedTransitionReceipt, None)
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
    handle = register_backend(backend)
    with pytest.raises(error_type):
        backend.apply_committed_transition(handle, candidate, cast(HarnessSessionView, None), cast(HarnessSessionView, None), cast(CommittedTransitionReceipt, None))


def test_apply_projects_only_the_committed_transition_and_advances_one_revision(
    backend: LangGraphExecutionBackend,
):
    candidate = transition()
    pre_view = view(transition_authorities=(authority_for(candidate),))
    post_view = post_view_for(pre_view, candidate)
    verifier = _BACKEND_VERIFIERS[backend]
    handle = register_backend(backend, view_value=pre_view)
    advanced = backend.apply_committed_transition(
        handle,
        candidate,
        pre_view,
        post_view,
        committed_receipt(verifier, candidate, pre_view, post_view),
    )
    assert advanced.state_revision == 5
    assert advanced.routed_state == RunState.SUMMARIZING.value
    assert advanced.run_id == handle.run_id
    assert advanced.trace_id == handle.trace_id
    assert advanced.plan_revision == handle.plan_revision


def test_duplicate_and_stale_transitions_are_rejected(
    backend: LangGraphExecutionBackend,
):
    first = transition()
    pre_view = view(transition_authorities=(authority_for(first),))
    post_view = post_view_for(pre_view, first)
    verifier = _BACKEND_VERIFIERS[backend]
    handle = register_backend(backend, view_value=pre_view)
    advanced = backend.apply_committed_transition(
        handle,
        first,
        pre_view,
        post_view,
        committed_receipt(verifier, first, pre_view, post_view),
    )
    with pytest.raises(DuplicateExecutionTransitionError):
        backend.apply_committed_transition(advanced, first, pre_view, post_view, cast(CommittedTransitionReceipt, None))
    with pytest.raises(StaleExecutionTransitionError):
        backend.apply_committed_transition(
            advanced, transition(idempotency_key="transition-2"), pre_view, post_view, cast(CommittedTransitionReceipt, None)
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
    committed = committed.model_copy(update={"last_event_hash": POST_HASH})
    verifier = _BACKEND_VERIFIERS[backend]
    handle = register_backend(backend, view_value=folded)
    projected = backend.apply_committed_transition(
        handle,
        candidate,
        folded,
        committed,
        committed_receipt(verifier, candidate, folded, committed),
    )
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
    handle = resume_backend(backend, plan(), folded, disposable_checkpoint=stale_checkpoint)
    assert handle.run_id == folded.run_id
    assert handle.trace_id == folded.trace_id
    assert handle.plan_revision == folded.plan_revision
    assert handle.state_revision == folded.state_revision
    assert handle.routed_state == folded.run_state.value
    assert not handle.cancelled


def test_resume_rejects_different_plan_for_an_existing_run(
    backend: LangGraphExecutionBackend,
):
    register_backend(backend)

    with pytest.raises(ExecutionPlanMismatchError):
        resume_backend(backend, plan(plan_id="different-plan"), view())


def test_resume_restores_duplicate_guard_from_folded_view(
    backend: LangGraphExecutionBackend,
):
    folded = view(applied_idempotency_keys=("transition-1",))
    handle = resume_backend(backend, plan(), folded)
    with pytest.raises(DuplicateExecutionTransitionError):
        backend.apply_committed_transition(handle, transition(), cast(HarnessSessionView, None), cast(HarnessSessionView, None), cast(CommittedTransitionReceipt, None))


def test_cancel_is_idempotent_by_run_id_and_blocks_further_projection(
    backend: LangGraphExecutionBackend,
):
    handle = register_backend(backend)
    assert backend.cancel("run-1") is None
    assert backend.cancel("run-1") is None
    with pytest.raises(CancelledExecutionError):
        backend.apply_committed_transition(handle, transition(), cast(HarnessSessionView, None), cast(HarnessSessionView, None), cast(CommittedTransitionReceipt, None))
    with pytest.raises(CancelledExecutionError):
        resume_backend(backend, plan(), view())


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


def test_exact_transition_without_post_commit_receipt_is_rejected(
    backend: LangGraphExecutionBackend,
    verifier: TrustedReceiptVerifier,
):
    plan_value = plan()
    candidate = transition()
    pre_view = view(transition_authorities=(authority_for(candidate),))
    post_view = post_view_for(pre_view, candidate)
    handle = backend.register(
        plan_value,
        pre_view,
        committed_snapshot(verifier, plan_value, pre_view),
    )

    with pytest.raises(UnverifiedExecutionReceiptError):
        backend.apply_committed_transition(
            handle,
            candidate,
            pre_view,
            post_view,
            cast(CommittedTransitionReceipt, None),
        )


def test_verified_receipt_cannot_authorize_illegal_running_to_created_edge(
    backend: LangGraphExecutionBackend,
    verifier: TrustedReceiptVerifier,
):
    plan_value = plan()
    candidate = transition(to_state=RunState.CREATED.value)
    pre_view = view(transition_authorities=(authority_for(candidate),))
    post_view = post_view_for(pre_view, candidate)
    handle = backend.register(
        plan_value,
        pre_view,
        committed_snapshot(verifier, plan_value, pre_view),
    )
    receipt = committed_receipt(
        verifier, candidate, pre_view, post_view, plan_value=plan_value
    )

    with pytest.raises(InvalidCommittedTransitionError):
        backend.apply_committed_transition(
            handle, candidate, pre_view, post_view, receipt
        )


def test_verified_post_view_without_committed_authority_is_rejected(
    backend: LangGraphExecutionBackend,
    verifier: TrustedReceiptVerifier,
):
    plan_value = plan()
    candidate = transition()
    pre_view = view()
    post_view = post_view_for(pre_view, candidate)
    handle = backend.register(
        plan_value,
        pre_view,
        committed_snapshot(verifier, plan_value, pre_view),
    )
    receipt = committed_receipt(
        verifier, candidate, pre_view, post_view, plan_value=plan_value
    )

    with pytest.raises(InvalidCommittedTransitionError):
        backend.apply_committed_transition(
            handle, candidate, pre_view, post_view, receipt
        )


def test_verified_foreign_work_item_transition_is_rejected(
    backend: LangGraphExecutionBackend,
    verifier: TrustedReceiptVerifier,
):
    plan_value = plan()
    candidate = HarnessTransition(
        run_id="run-1",
        trace_id="trace-1",
        entity_kind="work_item",
        entity_id="foreign-work",
        from_state="none",
        to_state=WorkItemState.PENDING.value,
        expected_state_revision=4,
        plan_revision=0,
        reason_code="foreign-work-created",
        idempotency_key="foreign-transition",
        lease_epoch=1,
        fencing_token_digest=HASH,
    )
    pre_view = view(transition_authorities=(authority_for(candidate),))
    post_view = post_view_for(pre_view, candidate)
    handle = backend.register(
        plan_value,
        pre_view,
        committed_snapshot(verifier, plan_value, pre_view),
    )
    receipt = committed_receipt(
        verifier, candidate, pre_view, post_view, plan_value=plan_value
    )

    with pytest.raises(ExecutionIdentityError):
        backend.apply_committed_transition(
            handle, candidate, pre_view, post_view, receipt
        )


def test_append_and_fold_then_verified_legal_transition_projects(
    tmp_path,
    backend: LangGraphExecutionBackend,
    verifier: TrustedReceiptVerifier,
):
    plan_value = plan()
    candidate, pre_view, post_view = committed_running_to_summarizing(tmp_path)
    handle = backend.register(
        plan_value,
        pre_view,
        committed_snapshot(verifier, plan_value, pre_view),
    )
    receipt = committed_receipt(
        verifier, candidate, pre_view, post_view, plan_value=plan_value
    )

    advanced = backend.apply_committed_transition(
        handle, candidate, pre_view, post_view, receipt
    )

    assert advanced.state_revision == post_view.state_revision
    assert advanced.routed_state == post_view.run_state.value


def test_resume_rejects_older_authoritative_snapshot(
    backend: LangGraphExecutionBackend,
    verifier: TrustedReceiptVerifier,
):
    plan_value = plan()
    current_view = view(state_revision=5, sequence=9, last_event_hash=POST_HASH)
    backend.register(
        plan_value,
        current_view,
        committed_snapshot(verifier, plan_value, current_view),
    )
    older_view = view(state_revision=4, sequence=8, last_event_hash=VIEW_HASH)

    with pytest.raises(StaleExecutionSnapshotError):
        backend.resume(
            plan_value,
            older_view,
            committed_snapshot(verifier, plan_value, older_view),
        )


def test_same_revision_resume_cannot_clear_applied_keys(
    backend: LangGraphExecutionBackend,
    verifier: TrustedReceiptVerifier,
):
    plan_value = plan()
    current_view = view(applied_idempotency_keys=("transition-1",))
    handle = backend.register(
        plan_value,
        current_view,
        committed_snapshot(verifier, plan_value, current_view),
    )
    altered_view = view(applied_idempotency_keys=())

    with pytest.raises(StaleExecutionSnapshotError):
        backend.resume(
            plan_value,
            altered_view,
            committed_snapshot(verifier, plan_value, altered_view),
        )
    with pytest.raises(DuplicateExecutionTransitionError):
        backend.apply_committed_transition(
            handle,
            transition(),
            cast(HarnessSessionView, None),
            cast(HarnessSessionView, None),
            cast(CommittedTransitionReceipt, None),
        )


def test_newer_snapshot_cannot_remove_applied_keys(
    backend: LangGraphExecutionBackend,
    verifier: TrustedReceiptVerifier,
):
    plan_value = plan()
    current_view = view(applied_idempotency_keys=("transition-1",))
    backend.register(
        plan_value,
        current_view,
        committed_snapshot(verifier, plan_value, current_view),
    )
    newer_view = view(
        sequence=current_view.sequence + 2,
        state_revision=current_view.state_revision + 1,
        run_state=RunState.RECONCILING,
        applied_idempotency_keys=(),
        last_event_hash=POST_HASH,
    )

    with pytest.raises(StaleExecutionSnapshotError):
        backend.resume(
            plan_value,
            newer_view,
            committed_snapshot(verifier, plan_value, newer_view),
        )


def test_fresh_backend_rejects_self_asserted_unverified_snapshot(
    backend: LangGraphExecutionBackend,
):
    plan_value = plan()
    view_value = view()
    unverified = CommittedExecutionSnapshot(
        run_id=plan_value.run_id,
        trace_id=plan_value.trace_id,
        plan_id=plan_value.plan_id,
        plan_digest=canonical_plan_digest(plan_value),
        plan_revision=plan_value.revision,
        sequence=view_value.sequence,
        state_revision=view_value.state_revision,
        view_digest=canonical_view_digest(view_value),
        event_head_hash=view_value.last_event_hash,
        folded_view=view_value,
        trust_key_id="attacker-key",
        signature="0" * 512,
    )

    with pytest.raises(UnverifiedExecutionReceiptError):
        backend.register(plan_value, view_value, unverified)


def test_langgraph_router_rejects_transition_even_when_exact_type():
    with pytest.raises(UncommittedTransitionError):
        route_committed_transition(
            cast(object, {"committed_transition": transition()})
        )


def test_same_revision_authority_event_resume_then_legal_apply(
    backend: LangGraphExecutionBackend,
    verifier: TrustedReceiptVerifier,
):
    plan_value = plan()
    candidate = transition()
    initial = view()
    handle = backend.register(
        plan_value,
        initial,
        committed_snapshot(verifier, plan_value, initial),
    )
    authorized = view(
        sequence=initial.sequence + 1,
        transition_authorities=(authority_for(candidate),),
        last_event_hash="d" * 64,
    )
    handle = backend.resume(
        plan_value,
        authorized,
        committed_snapshot(verifier, plan_value, authorized),
    )
    committed = post_view_for(authorized, candidate)

    advanced = backend.apply_committed_transition(
        handle,
        candidate,
        authorized,
        committed,
        committed_receipt(
            verifier, candidate, authorized, committed, plan_value=plan_value
        ),
    )

    assert advanced.state_revision == initial.state_revision + 1
    assert advanced.routed_state == RunState.SUMMARIZING.value


@pytest.mark.parametrize("drift", ("delete", "modify", "sequence", "head", "state"))
def test_same_revision_authority_extension_rejects_non_append_drift(
    backend: LangGraphExecutionBackend,
    verifier: TrustedReceiptVerifier,
    drift: str,
):
    plan_value = plan()
    candidate = transition()
    old_authority = authority_for(candidate)
    additional = HarnessTransition(
        run_id="run-1",
        trace_id="trace-1",
        entity_kind="work_item",
        entity_id="information-work",
        from_state="none",
        to_state=WorkItemState.PENDING.value,
        expected_state_revision=4,
        plan_revision=0,
        reason_code="pending",
        idempotency_key="transition-2",
        lease_epoch=1,
        fencing_token_digest=HASH,
    )
    initial = view(transition_authorities=(old_authority,))
    backend.register(
        plan_value,
        initial,
        committed_snapshot(verifier, plan_value, initial),
    )
    changes: dict[str, object] = {
        "sequence": initial.sequence + 1,
        "last_event_hash": "d" * 64,
        "transition_authorities": (
            old_authority,
            authority_for(additional),
        ),
    }
    if drift == "delete":
        changes["transition_authorities"] = ()
    elif drift == "modify":
        changes["transition_authorities"] = (
            authority_for(transition(reason_code="modified")),
        )
    elif drift == "sequence":
        changes["sequence"] = initial.sequence
    elif drift == "head":
        changes["last_event_hash"] = initial.last_event_hash
    elif drift == "state":
        changes["run_state"] = RunState.RECONCILING
    altered = HarnessSessionView.model_validate(
        initial.model_copy(update=changes).model_dump(mode="python")
    )

    with pytest.raises(StaleExecutionSnapshotError):
        backend.resume(
            plan_value,
            altered,
            committed_snapshot(verifier, plan_value, altered),
        )


def test_route_set_is_complete_programmatic_enum_union():
    expected = {
        *(state.value for state in RunState),
        *(state.value for state in WorkItemState),
        *(state.value for state in AttemptState),
    }
    assert set(execution_backend_module._ALL_ROUTES) == expected


@pytest.mark.parametrize(
    ("candidate", "pre_view", "expected_route"),
    (
        (
            transition(to_state=RunState.WAITING_APPROVAL.value),
            view(),
            RunState.WAITING_APPROVAL.value,
        ),
        (
            HarnessTransition(
                run_id="run-1",
                trace_id="trace-1",
                entity_kind="work_item",
                entity_id="information-work",
                from_state=WorkItemState.READY.value,
                to_state=WorkItemState.LEASED.value,
                expected_state_revision=4,
                plan_revision=0,
                reason_code="leased",
                idempotency_key="work-leased",
                lease_epoch=1,
                fencing_token_digest=HASH,
            ),
            view(work_item_states=(("information-work", WorkItemState.READY),)),
            WorkItemState.LEASED.value,
        ),
        (
            HarnessTransition(
                run_id="run-1",
                trace_id="trace-1",
                entity_kind="attempt",
                entity_id="attempt-1",
                from_state=AttemptState.VALIDATING.value,
                to_state=AttemptState.SETTLING.value,
                expected_state_revision=4,
                plan_revision=0,
                reason_code="settling",
                idempotency_key="attempt-settling",
                lease_epoch=1,
                fencing_token_digest=HASH,
            ),
            view(
                attempt_states=(("attempt-1", AttemptState.VALIDATING),),
                attempt_work_item_owners=(
                    AttemptWorkItemOwnershipRecord(
                        run_id="run-1",
                        trace_id="trace-1",
                        attempt_id="attempt-1",
                        work_item_id="information-work",
                        plan_revision=0,
                    ),
                ),
            ),
            AttemptState.SETTLING.value,
        ),
    ),
)
def test_legal_routes_include_previously_omitted_enum_states(
    backend: LangGraphExecutionBackend,
    verifier: TrustedReceiptVerifier,
    candidate: HarnessTransition,
    pre_view: HarnessSessionView,
    expected_route: str,
):
    plan_value = plan()
    pre_view = HarnessSessionView.model_validate(
        pre_view.model_copy(
            update={"transition_authorities": (authority_for(candidate),)}
        ).model_dump(mode="python")
    )
    post_view = post_view_for(pre_view, candidate)
    handle = backend.register(
        plan_value,
        pre_view,
        committed_snapshot(verifier, plan_value, pre_view),
    )

    advanced = backend.apply_committed_transition(
        handle,
        candidate,
        pre_view,
        post_view,
        committed_receipt(
            verifier, candidate, pre_view, post_view, plan_value=plan_value
        ),
    )

    assert advanced.routed_state == expected_route


def test_backend_rejects_arbitrary_callable_as_receipt_verifier():
    with pytest.raises(InvalidExecutionInputError):
        LangGraphExecutionBackend(authority_verifier=lambda _: True)


def test_verifier_has_no_public_callback_or_constructor(
    backend: LangGraphExecutionBackend,
):
    with pytest.raises(TypeError):
        ExecutionReceiptVerifier(lambda _: True)  # type: ignore[arg-type]
    assert not hasattr(backend._authority_verifier, "_verify_bytes")


def test_backend_cannot_inject_forged_verifier_or_attacker_signed_snapshot(
    backend: LangGraphExecutionBackend,
    verifier: TrustedReceiptVerifier,
    test_rsa_key: tuple[int, int, int],
):
    modulus, exponent, private = test_rsa_key
    forged = object.__new__(ExecutionReceiptVerifier)
    object.__setattr__(
        forged,
        "_factory_capability",
        execution_backend_module._VERIFIER_FACTORY_CAPABILITY,
    )
    object.__setattr__(
        forged,
        "_config_digest",
        execution_backend_module._EXPECTED_TRUST_CONFIG_DIGEST,
    )
    object.__setattr__(forged, "_keys", {"attacker": (modulus, exponent)})
    with pytest.raises(InvalidExecutionInputError):
        LangGraphExecutionBackend(authority_verifier=forged)

    attacker = TrustedReceiptVerifier(
        modulus=modulus,
        private_exponent=private,
        key_id="attacker",
    )
    plan_value = plan()
    view_value = view()
    snapshot = CommittedExecutionSnapshot(
        run_id=plan_value.run_id,
        trace_id=plan_value.trace_id,
        plan_id=plan_value.plan_id,
        plan_digest=canonical_plan_digest(plan_value),
        plan_revision=plan_value.revision,
        sequence=view_value.sequence,
        state_revision=view_value.state_revision,
        view_digest=canonical_view_digest(view_value),
        event_head_hash=view_value.last_event_hash,
        folded_view=view_value,
        trust_key_id="attacker",
        signature="0" * 512,
    )
    snapshot = cast(CommittedExecutionSnapshot, attacker.approve(snapshot))
    with pytest.raises(UnverifiedExecutionReceiptError):
        backend.register(plan_value, view_value, snapshot)


def test_cancel_during_verification_cannot_publish_uncancelled_projection(
    monkeypatch,
    backend: LangGraphExecutionBackend,
    verifier: TrustedReceiptVerifier,
):
    plan_value = plan()
    candidate = transition()
    pre_view = view(transition_authorities=(authority_for(candidate),))
    post_view = post_view_for(pre_view, candidate)
    handle = backend.register(
        plan_value,
        pre_view,
        committed_snapshot(verifier, plan_value, pre_view),
    )
    receipt = committed_receipt(
        verifier, candidate, pre_view, post_view, plan_value=plan_value
    )
    original_verify = ExecutionReceiptVerifier.verify
    cancelled = False

    def cancel_then_verify(self, value):
        nonlocal cancelled
        if type(value) is CommittedTransitionReceipt and not cancelled:
            cancelled = True
            backend.cancel("run-1")
        return original_verify(self, value)

    monkeypatch.setattr(ExecutionReceiptVerifier, "verify", cancel_then_verify)

    with pytest.raises(CancelledExecutionError):
        backend.apply_committed_transition(
            handle, candidate, pre_view, post_view, receipt
        )
    with pytest.raises(CancelledExecutionError):
        backend.resume(
            plan_value,
            pre_view,
            committed_snapshot(verifier, plan_value, pre_view),
        )


@pytest.mark.parametrize("invalid_kind", ("mapping", "subclass", "forged"))
def test_receipt_rejects_non_exact_nested_snapshot_before_verification(
    backend: LangGraphExecutionBackend,
    verifier: TrustedReceiptVerifier,
    invalid_kind: str,
):
    class SnapshotSubclass(CommittedExecutionSnapshot):
        pass

    plan_value = plan()
    candidate = transition()
    pre_view = view(transition_authorities=(authority_for(candidate),))
    post_view = post_view_for(pre_view, candidate)
    handle = backend.register(
        plan_value,
        pre_view,
        committed_snapshot(verifier, plan_value, pre_view),
    )
    receipt = committed_receipt(
        verifier, candidate, pre_view, post_view, plan_value=plan_value
    )
    if invalid_kind == "mapping":
        invalid: object = receipt.pre.model_dump(mode="python")
    elif invalid_kind == "subclass":
        invalid = SnapshotSubclass.model_validate(receipt.pre.model_dump(mode="python"))
    else:
        invalid = receipt.pre.model_copy()
        object.__setattr__(invalid, "worker_candidate", {"goto": "succeeded"})
    forged = receipt.model_copy()
    object.__setattr__(forged, "pre", invalid)

    with pytest.raises(UnverifiedExecutionReceiptError):
        backend.apply_committed_transition(
            handle, candidate, pre_view, post_view, forged
        )


def test_plan_rejects_nested_contract_subclass_before_snapshot_verification(
    backend: LangGraphExecutionBackend,
):
    class WorkerSubclass(WorkerSpec):
        pass

    original = plan()
    nested = WorkerSubclass.model_validate(
        original.workers[0].model_dump(mode="python")
    )
    forged = original.model_copy(update={"workers": (nested,)})

    with pytest.raises(InvalidExecutionInputError):
        backend.register(
            forged,
            view(),
            cast(CommittedExecutionSnapshot, None),
        )


def test_pinned_verifier_factory_rejects_attacker_selected_key(monkeypatch):
    monkeypatch.setattr(
        execution_backend_module,
        "_PINNED_PUBLIC_KEYS",
        {"attacker": ((1 << 2047) + 1, 65537)},
    )
    with pytest.raises(InvalidExecutionInputError):
        execution_backend_module._load_pinned_execution_receipt_verifier()


@pytest.mark.parametrize(
    ("key_id", "key"),
    (
        ("weak-bits", ((1 << 2040) | 1, 65537)),
        ("even-modulus", ((1 << 2047) + 2, 65537)),
        ("weak-exponent", ((1 << 2047) + 1, 3)),
        ("bool-modulus", (True, 65537)),
        ("bool-exponent", ((1 << 2047) + 1, True)),
        (" Noncanonical ", ((1 << 2047) + 1, 65537)),
    ),
)
def test_pinned_factory_rejects_invalid_key_entries(
    monkeypatch,
    key_id: str,
    key: tuple[object, object],
):
    keys = {key_id: key}
    monkeypatch.setattr(execution_backend_module, "_PINNED_PUBLIC_KEYS", keys)
    monkeypatch.setattr(
        execution_backend_module,
        "_EXPECTED_TRUST_CONFIG_DIGEST",
        execution_backend_module._trust_config_digest(keys),
    )
    with pytest.raises(InvalidExecutionInputError):
        execution_backend_module._load_pinned_execution_receipt_verifier()


def test_pinned_factory_rejects_duplicate_key_ids(monkeypatch):
    key = ((1 << 2047) + 1, 65537)

    class DuplicatePinnedKeys(dict):
        def items(self):
            return (("duplicate", key), ("duplicate", key))

    keys = DuplicatePinnedKeys({"duplicate": key})
    monkeypatch.setattr(execution_backend_module, "_PINNED_PUBLIC_KEYS", keys)
    monkeypatch.setattr(
        execution_backend_module,
        "_EXPECTED_TRUST_CONFIG_DIGEST",
        execution_backend_module._trust_config_digest(keys),
    )
    with pytest.raises(InvalidExecutionInputError):
        execution_backend_module._load_pinned_execution_receipt_verifier()


def test_scalar_normalization_is_rejected_before_snapshot_verification(
    backend: LangGraphExecutionBackend,
):
    forged = plan().model_copy(update={"run_id": " run-1 "})
    with pytest.raises(InvalidExecutionInputError):
        backend.register(
            forged,
            view(),
            cast(CommittedExecutionSnapshot, None),
        )


def test_prepare_allocation_can_be_rolled_back_after_external_exception(
    backend: LangGraphExecutionBackend,
    verifier: TrustedReceiptVerifier,
):
    plan_value = plan()
    token = backend.prepare_registration(
        plan_value, provisional_view(plan_value), issuer_descriptor(verifier)
    )
    assert type(token) is RegistrationPreparation
    with pytest.raises(ValidationError):
        token.run_id = "other-run"  # type: ignore[misc]

    try:
        raise RuntimeError("event append failed")
    except RuntimeError:
        backend.rollback_registration(token)
    backend.rollback_registration(token)

    replacement = backend.prepare_registration(
        plan_value, provisional_view(plan_value), issuer_descriptor(verifier)
    )
    assert replacement.token_id != token.token_id
    backend.rollback_registration(replacement)


def test_prepare_does_not_create_handle_or_allow_register_bypass(
    backend: LangGraphExecutionBackend,
    verifier: TrustedReceiptVerifier,
):
    plan_value = plan()
    token = backend.prepare_registration(
        plan_value, provisional_view(plan_value), issuer_descriptor(verifier)
    )
    committed_view = view()
    snapshot = committed_snapshot(verifier, plan_value, committed_view)

    with pytest.raises(ExecutionRegistrationError):
        backend.register(plan_value, committed_view, snapshot)
    with pytest.raises(ExecutionHandleMismatchError):
        backend.apply_committed_transition(
            ExecutionHandle(
                run_id="run-1",
                trace_id="trace-1",
                plan_id="plan-1",
                plan_revision=0,
                state_revision=4,
            ),
            transition(),
            cast(HarnessSessionView, None),
            cast(HarnessSessionView, None),
            cast(CommittedTransitionReceipt, None),
        )
    backend.rollback_registration(token)


def test_prepare_rejects_wrong_issuer_without_pending_residue(
    backend: LangGraphExecutionBackend,
    verifier: TrustedReceiptVerifier,
):
    plan_value = plan()
    wrong = issuer_descriptor(verifier).model_copy(update={"key_id": "attacker"})
    with pytest.raises(RegistrationPreparationError):
        backend.prepare_registration(plan_value, provisional_view(plan_value), wrong)

    token = backend.prepare_registration(
        plan_value, provisional_view(plan_value), issuer_descriptor(verifier)
    )
    backend.rollback_registration(token)


def test_commit_registration_consumes_token_and_double_rollback_is_idempotent(
    backend: LangGraphExecutionBackend,
    verifier: TrustedReceiptVerifier,
):
    plan_value = plan()
    committed_view = view()
    token = backend.prepare_registration(
        plan_value, provisional_view(plan_value), issuer_descriptor(verifier)
    )
    snapshot = committed_snapshot(verifier, plan_value, committed_view)

    handle = backend.commit_registration(token, snapshot)

    assert handle.state_revision == committed_view.state_revision
    with pytest.raises(RegistrationTokenConsumedError):
        backend.commit_registration(token, snapshot)
    with pytest.raises(RegistrationTokenConsumedError):
        backend.rollback_registration(token)

    other_plan = plan(run_id="run-2", trace_id="trace-2", plan_id="plan-2")
    rolled_back = backend.prepare_registration(
        other_plan,
        provisional_view(other_plan),
        IssuerTrustDescriptor(
            trust_version=execution_backend_module._TRUST_VERSION,
            trust_config_digest=execution_backend_module._EXPECTED_TRUST_CONFIG_DIGEST,
            key_id=verifier.key_id,
        ),
    )
    backend.rollback_registration(rolled_back)
    assert backend.rollback_registration(rolled_back) is None


def test_failed_commit_remains_rollbackable_and_cross_run_token_is_rejected(
    backend: LangGraphExecutionBackend,
    verifier: TrustedReceiptVerifier,
):
    plan_value = plan()
    token = backend.prepare_registration(
        plan_value, provisional_view(plan_value), issuer_descriptor(verifier)
    )
    foreign = token.model_copy(update={"run_id": "run-2"})
    snapshot = committed_snapshot(verifier, plan_value, view())

    with pytest.raises(RegistrationTokenMismatchError):
        backend.commit_registration(foreign, snapshot)

    invalid_snapshot = snapshot.model_copy(update={"trust_key_id": "attacker"})
    with pytest.raises(RegistrationPreparationError):
        backend.commit_registration(token, invalid_snapshot)
    backend.rollback_registration(token)

    replacement = backend.prepare_registration(
        plan_value, provisional_view(plan_value), issuer_descriptor(verifier)
    )
    backend.rollback_registration(replacement)
