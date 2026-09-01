"""Pure, fail-closed validation and application of Harness transitions."""

from __future__ import annotations

import re
from types import MappingProxyType
from typing import Annotated, Literal, Mapping, cast

from pydantic import Field, model_validator

from market_agent.workflow_contracts import (
    ContractModel,
    Digest,
    NonNegativeInt,
    PositiveInt,
    ShortText,
)
from market_agent.workflow_harness_contracts import (
    AttemptWorkItemOwnershipRecord,
    AttemptState,
    HarnessSessionView,
    HarnessTransition,
    ReconciliationResolutionRecord,
    RunState,
    TransitionAuthorityRecord,
    WorkItemState,
)


DependencyVersions = Annotated[
    tuple[tuple[ShortText, NonNegativeInt], ...], Field(max_length=64)
]

RUN_TERMINAL_STATES = frozenset(
    {RunState.SUCCEEDED, RunState.DEGRADED, RunState.FAILED, RunState.CANCELLED}
)
WORK_ITEM_TERMINAL_STATES = frozenset(
    {
        WorkItemState.SUCCEEDED,
        WorkItemState.BLOCKED,
        WorkItemState.FAILED,
        WorkItemState.CANCELLED,
    }
)
ATTEMPT_TERMINAL_STATES = frozenset(
    {
        AttemptState.COMPLETED,
        AttemptState.TIMED_OUT,
        AttemptState.REJECTED,
        AttemptState.FAILED,
        AttemptState.STALE,
        AttemptState.CANCELLED,
    }
)

RUN_EDGES: Mapping[RunState | None, frozenset[RunState]] = MappingProxyType(
    {
        None: frozenset({RunState.CREATED}),
        RunState.CREATED: frozenset({RunState.ADMITTED, RunState.CANCELLED}),
        RunState.ADMITTED: frozenset({RunState.PLANNED, RunState.CANCELLED}),
        RunState.PLANNED: frozenset({RunState.READY, RunState.CANCELLED}),
        RunState.READY: frozenset({RunState.RUNNING, RunState.CANCELLED}),
        RunState.RUNNING: frozenset(
            {
                RunState.RECONCILING,
                RunState.WAITING_APPROVAL,
                RunState.WAITING_RECONCILIATION,
                RunState.DEGRADING,
                RunState.SUMMARIZING,
                RunState.CANCELLED,
            }
        ),
        RunState.RECONCILING: frozenset(
            {
                RunState.RUNNING,
                RunState.WAITING_RECONCILIATION,
                RunState.DEGRADING,
                RunState.SUMMARIZING,
                RunState.FAILED,
                RunState.CANCELLED,
            }
        ),
        RunState.WAITING_APPROVAL: frozenset(
            {RunState.RUNNING, RunState.FAILED, RunState.CANCELLED}
        ),
        RunState.WAITING_RECONCILIATION: frozenset(
            {RunState.RECONCILING, RunState.FAILED, RunState.CANCELLED}
        ),
        RunState.DEGRADING: frozenset(
            {
                RunState.RUNNING,
                RunState.SUMMARIZING,
                RunState.DEGRADED,
                RunState.FAILED,
                RunState.CANCELLED,
            }
        ),
        RunState.SUMMARIZING: frozenset(
            {RunState.SUCCEEDED, RunState.DEGRADED, RunState.FAILED, RunState.CANCELLED}
        ),
    }
)
WORK_ITEM_EDGES: Mapping[WorkItemState | None, frozenset[WorkItemState]] = (
    MappingProxyType(
        {
            None: frozenset({WorkItemState.PENDING}),
            WorkItemState.PENDING: frozenset({WorkItemState.READY}),
            WorkItemState.READY: frozenset({WorkItemState.LEASED}),
            WorkItemState.LEASED: frozenset(
                {WorkItemState.RUNNING, WorkItemState.RETRY_WAIT}
            ),
            WorkItemState.RUNNING: frozenset(
                {WorkItemState.VALIDATING, WorkItemState.RETRY_WAIT}
            ),
            WorkItemState.VALIDATING: frozenset(
                {WorkItemState.SUCCEEDED, WorkItemState.RETRY_WAIT}
            ),
            WorkItemState.RETRY_WAIT: frozenset({WorkItemState.READY}),
        }
    )
)
ATTEMPT_EDGES: Mapping[AttemptState | None, frozenset[AttemptState]] = (
    MappingProxyType(
        {
            None: frozenset({AttemptState.RESERVED}),
            AttemptState.RESERVED: frozenset({AttemptState.DISPATCHED}),
            AttemptState.DISPATCHED: frozenset(
                {AttemptState.STREAMING, AttemptState.VALIDATING}
            ),
            AttemptState.STREAMING: frozenset({AttemptState.VALIDATING}),
            AttemptState.VALIDATING: frozenset({AttemptState.SETTLING}),
            AttemptState.SETTLING: frozenset({AttemptState.COMPLETED}),
        }
    )
)


class StateMachineError(RuntimeError):
    """Base class for a rejected state-machine application."""


class InvalidTransitionError(StateMachineError):
    """A source, target, terminal, or required proof is not legal."""


class StaleTransitionError(StateMachineError):
    """A candidate was prepared against an older folded revision or plan."""


class IdentityMismatchError(StateMachineError):
    """A candidate or its evidence does not belong to the folded run."""


class IdempotencyConflictError(StateMachineError):
    """The candidate's idempotency key has already been applied."""


class DependencyVersionError(StateMachineError):
    """Pinned dependency versions no longer match the folded view."""


class LeaseEvidenceError(StateMachineError):
    """Durable lease epoch or fencing-token digest evidence is inconsistent."""


class SideEffectReconciliationRequiredError(InvalidTransitionError):
    """Unknown external effects require durable broker reconciliation."""


class TransitionValidation(ContractModel):
    allowed: bool
    reason: ShortText | None = None

    def __init__(
        self, allowed: bool, reason: str | None = None, **values: object
    ) -> None:
        super().__init__(allowed=allowed, reason=reason, **values)


class _TransitionEvidence(ContractModel):
    run_id: ShortText
    trace_id: ShortText
    entity_id: ShortText
    expected_state_revision: NonNegativeInt
    plan_revision: NonNegativeInt
    dependency_versions: DependencyVersions

    @model_validator(mode="after")
    def validate_dependency_identifiers(self) -> _TransitionEvidence:
        identifiers = tuple(identifier for identifier, _ in self.dependency_versions)
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("dependency versions must have unique identifiers")
        return self


class RunTransitionEvidence(_TransitionEvidence):
    """Strict folded evidence required for every ordinary run transition."""


class _LeasedTransitionAuthorization(_TransitionEvidence):
    reservation_id: ShortText
    grant_id: ShortText
    lease_epoch: PositiveInt
    fencing_token_digest: Digest


class WorkItemTransitionAuthorization(_LeasedTransitionAuthorization):
    """Strict reservation, grant, and durable lease evidence for work items."""


class AttemptTransitionAuthorization(_LeasedTransitionAuthorization):
    """Strict reservation, grant, and durable lease evidence for attempts."""


class ReconciliationResolution(ContractModel):
    """A durable broker observation that resolves an unknown side effect."""

    run_id: ShortText
    trace_id: ShortText
    entity_id: ShortText
    expected_state_revision: NonNegativeInt
    plan_revision: NonNegativeInt
    reconciliation_id: ShortText
    broker_observation_digest: Digest
    side_effect_resolved: Literal[True]


class StaleAttemptRetryAuthorization(ContractModel):
    """Binds a stale attempt to its nonterminal owner's retry transition."""

    run_id: ShortText
    trace_id: ShortText
    work_item_id: ShortText
    attempt_id: ShortText
    expected_state_revision: NonNegativeInt
    plan_revision: NonNegativeInt
    lease_epoch: PositiveInt
    fencing_token_digest: Digest


class PermanentFailureDecision(ContractModel):
    """Trusted policy authority for failure outside declared ordinary edges."""

    run_id: ShortText
    trace_id: ShortText
    expected_state_revision: NonNegativeInt
    plan_revision: NonNegativeInt
    from_state: RunState
    reason_code: ShortText
    idempotency_key: ShortText


Candidate = HarnessTransition | PermanentFailureDecision
Authorization = (
    RunTransitionEvidence
    | WorkItemTransitionAuthorization
    | AttemptTransitionAuthorization
)
_ValidatedInputs = tuple[
    Candidate,
    HarnessSessionView,
    Authorization | None,
    StaleAttemptRetryAuthorization | None,
    ReconciliationResolution | None,
]

_SENSITIVE_VALUE = re.compile(
    r"(?:\bbearer\s+|(?:^|[^a-z0-9])sk[-_]|fence-[0-9]+-|"
    r"(?:password|secret|credential|api[ _-]?key|authorization)\s*[:=]|"
    r"https?://\S+[?&](?:token|key|secret|signature)=|"
    r"-----BEGIN[^\n]*PRIVATE KEY-----)",
    re.IGNORECASE,
)
_INVALID_PUBLIC_INPUT = "invalid state-machine input"


def _reject_undeclared_model_fields(value: object) -> None:
    if isinstance(value, ContractModel):
        declared_fields = type(value).model_fields
        if set(value.__dict__).difference(declared_fields):
            raise ValueError("contract model contains undeclared fields")
        for item in value.__dict__.values():
            _reject_undeclared_model_fields(item)
    elif isinstance(value, Mapping):
        for item in value.values():
            _reject_undeclared_model_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_undeclared_model_fields(item)


def _reject_sensitive_values(value: object) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _reject_sensitive_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_sensitive_values(item)
    elif isinstance(value, str) and _SENSITIVE_VALUE.search(value):
        raise ValueError("state-machine input contains a sensitive value")


def _fresh_contract(
    value: object, allowed_types: tuple[type[ContractModel], ...]
) -> ContractModel:
    if not isinstance(value, ContractModel) or type(value) not in allowed_types:
        raise TypeError("state-machine input has an incompatible contract type")
    _reject_undeclared_model_fields(value)
    values = value.model_dump(mode="python", exclude_unset=True)
    _reject_sensitive_values(values)
    return type(value).model_validate(values)


def _fresh_optional_contract(
    value: object | None, allowed_types: tuple[type[ContractModel], ...]
) -> ContractModel | None:
    if value is None:
        return None
    return _fresh_contract(value, allowed_types)


def _revalidate_public_inputs(
    candidate: Candidate,
    view: HarnessSessionView,
    authorization: Authorization | None,
    retry_authorization: StaleAttemptRetryAuthorization | None,
    reconciliation_resolution: ReconciliationResolution | None,
) -> _ValidatedInputs:
    return cast(
        _ValidatedInputs,
        (
            _fresh_contract(
                candidate, (HarnessTransition, PermanentFailureDecision)
            ),
            _fresh_contract(view, (HarnessSessionView,)),
            _fresh_optional_contract(
                authorization,
                (
                    RunTransitionEvidence,
                    WorkItemTransitionAuthorization,
                    AttemptTransitionAuthorization,
                ),
            ),
            _fresh_optional_contract(
                retry_authorization, (StaleAttemptRetryAuthorization,)
            ),
            _fresh_optional_contract(
                reconciliation_resolution, (ReconciliationResolution,)
            ),
        ),
    )


class GlobalTaskStateMachine:
    """Applies only legal, evidence-bound transitions to a folded view."""

    def validate(
        self,
        candidate: Candidate,
        view: HarnessSessionView,
        *,
        authorization: Authorization | None = None,
        retry_authorization: StaleAttemptRetryAuthorization | None = None,
        reconciliation_resolution: ReconciliationResolution | None = None,
    ) -> TransitionValidation:
        try:
            validated = _revalidate_public_inputs(
                candidate,
                view,
                authorization,
                retry_authorization,
                reconciliation_resolution,
            )
        except Exception:
            return TransitionValidation(False, _INVALID_PUBLIC_INPUT)
        return self._validate_revalidated(*validated)

    def _validate_revalidated(
        self,
        candidate: Candidate,
        view: HarnessSessionView,
        authorization: Authorization | None,
        retry_authorization: StaleAttemptRetryAuthorization | None,
        reconciliation_resolution: ReconciliationResolution | None,
    ) -> TransitionValidation:
        if isinstance(candidate, PermanentFailureDecision):
            return self._validate_permanent_failure(candidate, view)
        rejected = self._validate_evidence(candidate, view, authorization)
        if rejected is not None:
            return rejected
        if candidate.idempotency_key in view.applied_idempotency_keys:
            return TransitionValidation(False, "duplicate idempotency key")
        return self._validate_edge(
            candidate, view, retry_authorization, reconciliation_resolution
        )

    def apply(
        self,
        candidate: Candidate,
        view: HarnessSessionView,
        *,
        authorization: Authorization | None = None,
        retry_authorization: StaleAttemptRetryAuthorization | None = None,
        reconciliation_resolution: ReconciliationResolution | None = None,
    ) -> HarnessSessionView:
        try:
            validated = _revalidate_public_inputs(
                candidate,
                view,
                authorization,
                retry_authorization,
                reconciliation_resolution,
            )
        except Exception as error:
            raise InvalidTransitionError(_INVALID_PUBLIC_INPUT) from error
        (
            candidate,
            view,
            authorization,
            retry_authorization,
            reconciliation_resolution,
        ) = validated
        decision = self._validate_revalidated(*validated)
        if not decision.allowed:
            raise _error_for(decision.reason)
        transition = self._transition(candidate)
        changes: dict[str, object] = {
            "sequence": view.sequence + 1,
            "state_revision": view.state_revision + 1,
            "plan_revision": transition.plan_revision,
            "run_id": transition.run_id if view.run_id is None else view.run_id,
            "trace_id": transition.trace_id if view.trace_id is None else view.trace_id,
            "applied_idempotency_keys": (
                *view.applied_idempotency_keys,
                transition.idempotency_key,
            ),
        }
        if transition.entity_kind == "run":
            changes["run_state"] = RunState(transition.to_state)
            if reconciliation_resolution is not None:
                changes["external_side_effect_unknown"] = False
        elif transition.entity_kind == "work_item":
            changes["work_item_states"] = _replace_state(
                view.work_item_states,
                transition.entity_id,
                WorkItemState(transition.to_state),
            )
        else:
            changes["attempt_states"] = _replace_state(
                view.attempt_states,
                transition.entity_id,
                AttemptState(transition.to_state),
            )
        return HarnessSessionView.model_validate(
            view.model_copy(update=changes).model_dump(mode="python")
        )

    def _validate_evidence(
        self,
        transition: HarnessTransition,
        view: HarnessSessionView,
        authorization: Authorization | None,
    ) -> TransitionValidation | None:
        expected_type: type[Authorization]
        if transition.entity_kind == "run":
            expected_type = RunTransitionEvidence
        elif transition.entity_kind == "work_item":
            expected_type = WorkItemTransitionAuthorization
        else:
            expected_type = AttemptTransitionAuthorization
        if authorization is None:
            return TransitionValidation(False, "required transition evidence is missing")
        if transition.entity_kind != "run" and (
            view.run_id is None or view.trace_id is None or view.run_state is None
        ):
            return TransitionValidation(False, "non-run transition requires active run")
        if transition.entity_kind != "run" and view.run_state in RUN_TERMINAL_STATES:
            return TransitionValidation(False, "non-run transition requires nonterminal run")
        if type(authorization) is not expected_type:
            return TransitionValidation(False, "transition evidence type is incompatible")
        if authorization.run_id != transition.run_id or (
            view.run_id is not None and authorization.run_id != view.run_id
        ):
            return TransitionValidation(False, "evidence run identity does not match")
        if (
            authorization.trace_id != transition.trace_id
            or (view.trace_id is not None and authorization.trace_id != view.trace_id)
            or authorization.entity_id != transition.entity_id
        ):
            return TransitionValidation(False, "evidence identity does not match")
        if (
            authorization.expected_state_revision != transition.expected_state_revision
            or authorization.expected_state_revision != view.state_revision
        ):
            return TransitionValidation(False, "evidence state revision is stale")
        if (
            authorization.plan_revision != transition.plan_revision
            or authorization.plan_revision != view.plan_revision
        ):
            return TransitionValidation(False, "evidence plan revision is stale")
        if authorization.dependency_versions != view.dependency_versions:
            return TransitionValidation(False, "evidence dependency versions do not match")
        if isinstance(authorization, _LeasedTransitionAuthorization) and (
            authorization.lease_epoch != transition.lease_epoch
            or authorization.fencing_token_digest != transition.fencing_token_digest
        ):
            return TransitionValidation(False, "durable lease evidence does not match")
        authority_record = _authority_record(transition, authorization)
        if authority_record not in view.transition_authorities:
            return TransitionValidation(False, "committed transition authority is missing")
        return None

    def _validate_edge(
        self,
        transition: HarnessTransition,
        view: HarnessSessionView,
        retry_authorization: StaleAttemptRetryAuthorization | None,
        reconciliation_resolution: ReconciliationResolution | None,
    ) -> TransitionValidation:
        if transition.entity_kind == "run":
            return self._validate_run(transition, view, reconciliation_resolution)
        if reconciliation_resolution is not None:
            return TransitionValidation(False, "reconciliation proof requires run transition")
        if transition.entity_kind == "work_item":
            return self._validate_work_item(transition, view, retry_authorization)
        if retry_authorization is not None:
            return TransitionValidation(False, "retry proof requires work-item transition")
        return self._validate_attempt(transition, view)

    @staticmethod
    def _validate_run(
        transition: HarnessTransition,
        view: HarnessSessionView,
        resolution: ReconciliationResolution | None,
    ) -> TransitionValidation:
        if transition.entity_id != transition.run_id:
            return TransitionValidation(False, "run entity identity does not match")
        try:
            target = RunState(transition.to_state)
        except ValueError:
            return TransitionValidation(False, "unknown run target state")
        source = view.run_state
        if transition.from_state != (source.value if source is not None else "none"):
            return TransitionValidation(False, "run source state does not match")
        if source in RUN_TERMINAL_STATES:
            return TransitionValidation(False, "terminal run state is absorbing")
        if view.external_side_effect_unknown:
            if not _resolution_matches(resolution, transition, view):
                return TransitionValidation(
                    False, "unknown external effects require reconciliation"
                )
            if source is not RunState.WAITING_RECONCILIATION or target is not RunState.RECONCILING:
                return TransitionValidation(False, "reconciliation proof has invalid edge")
        elif resolution is not None:
            return TransitionValidation(False, "reconciliation proof is not applicable")
        if target not in RUN_EDGES.get(source, frozenset()):
            return TransitionValidation(False, "illegal run transition")
        return TransitionValidation(allowed=True)

    @staticmethod
    def _validate_work_item(
        transition: HarnessTransition,
        view: HarnessSessionView,
        retry: StaleAttemptRetryAuthorization | None,
    ) -> TransitionValidation:
        try:
            target = WorkItemState(transition.to_state)
        except ValueError:
            return TransitionValidation(False, "unknown work-item target state")
        source = dict(view.work_item_states).get(transition.entity_id)
        if transition.from_state != (source.value if source is not None else "none"):
            return TransitionValidation(False, "work-item source state does not match")
        if source in WORK_ITEM_TERMINAL_STATES:
            return TransitionValidation(False, "terminal work-item state is absorbing")
        if target is WorkItemState.RETRY_WAIT:
            return _validate_retry_authorization(retry, transition, view, source)
        if retry is not None:
            return TransitionValidation(False, "retry proof is only valid for retry wait")
        allowed = set(WORK_ITEM_EDGES.get(source, frozenset()))
        if source is not None:
            allowed.update(
                {WorkItemState.BLOCKED, WorkItemState.FAILED, WorkItemState.CANCELLED}
            )
        if target not in allowed:
            return TransitionValidation(False, "illegal work-item transition")
        return TransitionValidation(allowed=True)

    @staticmethod
    def _validate_attempt(
        transition: HarnessTransition, view: HarnessSessionView
    ) -> TransitionValidation:
        try:
            target = AttemptState(transition.to_state)
        except ValueError:
            return TransitionValidation(False, "unknown attempt target state")
        source = dict(view.attempt_states).get(transition.entity_id)
        if transition.from_state != (source.value if source is not None else "none"):
            return TransitionValidation(False, "attempt source state does not match")
        if source in ATTEMPT_TERMINAL_STATES:
            return TransitionValidation(False, "terminal attempt state is absorbing")
        allowed = set(ATTEMPT_EDGES.get(source, frozenset()))
        if source is not None:
            allowed.update(
                {
                    AttemptState.TIMED_OUT,
                    AttemptState.REJECTED,
                    AttemptState.FAILED,
                    AttemptState.STALE,
                    AttemptState.CANCELLED,
                }
            )
        if target not in allowed:
            return TransitionValidation(False, "illegal attempt transition")
        return TransitionValidation(allowed=True)

    @staticmethod
    def _validate_permanent_failure(
        decision: PermanentFailureDecision, view: HarnessSessionView
    ) -> TransitionValidation:
        if view.run_id is None or view.trace_id is None or view.run_state is None:
            return TransitionValidation(False, "permanent failure requires active run")
        if view.run_state in RUN_TERMINAL_STATES:
            return TransitionValidation(False, "terminal run state is absorbing")
        if view.external_side_effect_unknown:
            return TransitionValidation(
                False, "unknown external effects require reconciliation"
            )
        if decision.run_id != view.run_id or decision.trace_id != view.trace_id:
            return TransitionValidation(False, "permanent failure identity does not match")
        if decision.from_state is not view.run_state:
            return TransitionValidation(False, "permanent failure source state does not match")
        if decision.expected_state_revision != view.state_revision:
            return TransitionValidation(False, "permanent failure state revision is stale")
        if decision.plan_revision != view.plan_revision:
            return TransitionValidation(False, "permanent failure plan revision is stale")
        if decision.idempotency_key in view.applied_idempotency_keys:
            return TransitionValidation(False, "duplicate idempotency key")
        return TransitionValidation(allowed=True)

    @staticmethod
    def _transition(candidate: Candidate) -> HarnessTransition:
        if isinstance(candidate, HarnessTransition):
            return candidate
        return HarnessTransition(
            run_id=candidate.run_id,
            trace_id=candidate.trace_id,
            entity_kind="run",
            entity_id=candidate.run_id,
            from_state=candidate.from_state.value,
            to_state=RunState.FAILED.value,
            expected_state_revision=candidate.expected_state_revision,
            plan_revision=candidate.plan_revision,
            reason_code=candidate.reason_code,
            idempotency_key=candidate.idempotency_key,
        )


def _resolution_matches(
    resolution: ReconciliationResolution | None,
    transition: HarnessTransition,
    view: HarnessSessionView,
) -> bool:
    if resolution is None or not (
        resolution.run_id == transition.run_id == view.run_id
        and resolution.trace_id == transition.trace_id == view.trace_id
        and resolution.entity_id == transition.entity_id
        and resolution.expected_state_revision
        == transition.expected_state_revision
        == view.state_revision
        and resolution.plan_revision == transition.plan_revision == view.plan_revision
    ):
        return False
    expected_record = ReconciliationResolutionRecord(
        run_id=resolution.run_id,
        trace_id=resolution.trace_id,
        reconciliation_id=resolution.reconciliation_id,
        expected_state_revision=resolution.expected_state_revision,
        plan_revision=resolution.plan_revision,
        broker_observation_digest=resolution.broker_observation_digest,
        side_effect_resolved=resolution.side_effect_resolved,
    )
    authoritative_records = tuple(
        record
        for record in view.reconciliation_resolutions
        if record.semantic_authority_key()
        == expected_record.semantic_authority_key()
    )
    return len(authoritative_records) == 1 and authoritative_records[0] == expected_record


def _validate_retry_authorization(
    retry: StaleAttemptRetryAuthorization | None,
    transition: HarnessTransition,
    view: HarnessSessionView,
    source: WorkItemState | None,
) -> TransitionValidation:
    if source not in {
        WorkItemState.LEASED,
        WorkItemState.RUNNING,
        WorkItemState.VALIDATING,
    }:
        return TransitionValidation(False, "retry wait source is not retryable")
    if retry is None:
        return TransitionValidation(False, "stale attempt retry proof is missing")
    if (
        retry.run_id != transition.run_id == view.run_id
        or retry.trace_id != transition.trace_id == view.trace_id
        or retry.work_item_id != transition.entity_id
        or retry.expected_state_revision != transition.expected_state_revision
        or retry.expected_state_revision != view.state_revision
        or retry.plan_revision != transition.plan_revision
        or retry.plan_revision != view.plan_revision
        or retry.lease_epoch != transition.lease_epoch
        or retry.fencing_token_digest != transition.fencing_token_digest
    ):
        return TransitionValidation(False, "stale attempt retry proof does not match")
    if dict(view.attempt_states).get(retry.attempt_id) is not AttemptState.STALE:
        return TransitionValidation(False, "retry attempt is not stale")
    ownership = AttemptWorkItemOwnershipRecord(
        run_id=retry.run_id,
        trace_id=retry.trace_id,
        attempt_id=retry.attempt_id,
        work_item_id=retry.work_item_id,
        plan_revision=retry.plan_revision,
    )
    if ownership not in view.attempt_work_item_owners:
        return TransitionValidation(False, "committed attempt ownership is missing")
    return TransitionValidation(allowed=True)


def _authority_record(
    transition: HarnessTransition, authorization: Authorization
) -> TransitionAuthorityRecord:
    values: dict[str, object] = {
        "run_id": authorization.run_id,
        "trace_id": authorization.trace_id,
        "entity_kind": transition.entity_kind,
        "entity_id": authorization.entity_id,
        "from_state": transition.from_state,
        "to_state": transition.to_state,
        "expected_state_revision": authorization.expected_state_revision,
        "plan_revision": authorization.plan_revision,
        "reason_code": transition.reason_code,
        "idempotency_key": transition.idempotency_key,
        "dependency_versions": authorization.dependency_versions,
    }
    if isinstance(authorization, _LeasedTransitionAuthorization):
        values.update(
            {
                "reservation_id": authorization.reservation_id,
                "grant_id": authorization.grant_id,
                "lease_epoch": authorization.lease_epoch,
                "fencing_token_digest": authorization.fencing_token_digest,
            }
        )
    return TransitionAuthorityRecord(**values)


def _replace_state(
    values: tuple[tuple[str, object], ...], identifier: str, state: object
) -> tuple[tuple[str, object], ...]:
    next_states = dict(values)
    next_states[identifier] = state
    return tuple(sorted(next_states.items()))


def _error_for(reason: str | None) -> StateMachineError:
    message = reason or "state-machine validation failed"
    if "stale" in message and "attempt" not in message:
        return StaleTransitionError(message)
    if "identity" in message:
        return IdentityMismatchError(message)
    if "idempotency" in message:
        return IdempotencyConflictError(message)
    if "dependency" in message:
        return DependencyVersionError(message)
    if "lease" in message:
        return LeaseEvidenceError(message)
    if "unknown external effects" in message:
        return SideEffectReconciliationRequiredError(message)
    return InvalidTransitionError(message)
