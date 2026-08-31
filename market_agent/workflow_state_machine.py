"""Pure, deterministic validation and application of Harness transitions.

This module deliberately has no event-store dependency.  It validates a
candidate against a folded :class:`HarnessSessionView` and returns the next
view only after the candidate is legal; persistence and live lease-token
authorization remain the event store's responsibility.
"""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from market_agent.workflow_harness_contracts import (
    AttemptState,
    HarnessSessionView,
    HarnessTransition,
    RunState,
    WorkItemState,
)


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
        RunState.CREATED: frozenset({RunState.ADMITTED}),
        RunState.ADMITTED: frozenset({RunState.PLANNED}),
        RunState.PLANNED: frozenset({RunState.READY}),
        RunState.READY: frozenset({RunState.RUNNING}),
        RunState.RUNNING: frozenset(
            {
                RunState.RECONCILING,
                RunState.WAITING_APPROVAL,
                RunState.WAITING_RECONCILIATION,
                RunState.DEGRADING,
                RunState.SUMMARIZING,
            }
        ),
        RunState.RECONCILING: frozenset(
            {
                RunState.RUNNING,
                RunState.WAITING_RECONCILIATION,
                RunState.DEGRADING,
                RunState.SUMMARIZING,
                RunState.FAILED,
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
            }
        ),
        RunState.SUMMARIZING: frozenset(
            {RunState.SUCCEEDED, RunState.DEGRADED, RunState.FAILED}
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
    """A source, target, or terminal-state guard is not legal."""


class StaleTransitionError(StateMachineError):
    """A candidate was prepared against an older folded revision or plan."""


class IdentityMismatchError(StateMachineError):
    """A candidate does not belong to the folded run and trace."""


class IdempotencyConflictError(StateMachineError):
    """The candidate's idempotency key has already been applied."""


class DependencyVersionError(StateMachineError):
    """A candidate's pinned dependency versions no longer match the fold."""


class LeaseEvidenceError(StateMachineError):
    """Durable lease epoch or fencing-token digest evidence is inconsistent."""


class SideEffectReconciliationRequiredError(InvalidTransitionError):
    """Unknown external effects must reconcile before failure or cancellation."""


@dataclass(frozen=True, slots=True)
class TransitionValidation:
    allowed: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class PermanentFailureDecision:
    """Trusted policy decision that seals a nonterminal run as failed.

    This is intentionally typed rather than an untrusted reason string passed
    through a generic transition API.  The event store still atomically appends
    the resulting transition and is the only persistence authority.
    """

    reason_code: str
    idempotency_key: str

    def transition(self, view: HarnessSessionView) -> HarnessTransition:
        if view.run_id is None or view.trace_id is None or view.run_state is None:
            raise InvalidTransitionError("permanent failure requires an active run")
        if view.run_state in RUN_TERMINAL_STATES:
            raise InvalidTransitionError("terminal run state is absorbing")
        return HarnessTransition(
            run_id=view.run_id,
            trace_id=view.trace_id,
            entity_kind="run",
            entity_id=view.run_id,
            from_state=view.run_state.value,
            to_state=RunState.FAILED.value,
            expected_state_revision=view.state_revision,
            plan_revision=view.plan_revision,
            reason_code=self.reason_code,
            idempotency_key=self.idempotency_key,
        )


class GlobalTaskStateMachine:
    """Validates and folds transitions without reading or persisting live secrets."""

    def validate(
        self,
        transition: HarnessTransition,
        view: HarnessSessionView,
        *,
        dependency_versions: Mapping[str, int] | None = None,
        reservation_granted: bool = True,
        grant_granted: bool = True,
        lease_epoch: int | None = None,
        fencing_token_digest: str | None = None,
    ) -> TransitionValidation:
        """Return a deterministic decision for a transition against ``view``.

        ``lease_epoch`` and ``fencing_token_digest`` are durable values only.
        The raw fencing token is deliberately absent from this API: it is an
        out-of-band credential checked solely by event-store append authority.
        """
        if not reservation_granted:
            return TransitionValidation(False, "reservation was not granted")
        if not grant_granted:
            return TransitionValidation(False, "grant was not granted")
        if transition.expected_state_revision != view.state_revision:
            return TransitionValidation(False, "stale state revision")
        if transition.plan_revision != view.plan_revision:
            return TransitionValidation(False, "stale plan revision")
        if transition.idempotency_key in view.applied_idempotency_keys:
            return TransitionValidation(False, "duplicate idempotency key")
        if (
            dependency_versions is not None
            and tuple(sorted(dependency_versions.items()))
            != view.dependency_versions
        ):
            return TransitionValidation(
                False, "dependency versions do not match folded view"
            )
        if view.run_id is not None and transition.run_id != view.run_id:
            return TransitionValidation(False, "run identity does not match folded view")
        if view.trace_id is not None and transition.trace_id != view.trace_id:
            return TransitionValidation(False, "trace identity does not match folded view")
        if view.run_id is None and transition.entity_kind != "run":
            return TransitionValidation(
                False, "work and attempt transitions require an active run"
            )
        if transition.entity_kind != "run":
            if lease_epoch is not None and transition.lease_epoch != lease_epoch:
                return TransitionValidation(
                    False, "lease epoch does not match durable evidence"
                )
            if (
                fencing_token_digest is not None
                and transition.fencing_token_digest != fencing_token_digest
            ):
                return TransitionValidation(
                    False, "fencing digest does not match durable evidence"
                )
        return self._validate_edge(transition, view)

    def apply(
        self,
        candidate: HarnessTransition | PermanentFailureDecision,
        view: HarnessSessionView,
        *,
        dependency_versions: Mapping[str, int] | None = None,
        reservation_granted: bool = True,
        grant_granted: bool = True,
        lease_epoch: int | None = None,
        fencing_token_digest: str | None = None,
        external_side_effect_unknown: bool | None = None,
    ) -> HarnessSessionView:
        """Return the next folded view, or raise a typed error without mutation."""
        transition = (
            candidate.transition(view)
            if isinstance(candidate, PermanentFailureDecision)
            else candidate
        )
        decision = self.validate(
            transition,
            view,
            dependency_versions=dependency_versions,
            reservation_granted=reservation_granted,
            grant_granted=grant_granted,
            lease_epoch=lease_epoch,
            fencing_token_digest=fencing_token_digest,
        )
        if not decision.allowed:
            raise _error_for(decision.reason)

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
            target = RunState(transition.to_state)
            changes["run_state"] = target
            if external_side_effect_unknown is None:
                # Leaving a folded unknown-effect wait is necessarily preceded
                # by a broker observation; model that resolved observation in
                # the resulting view without persisting a raw broker credential.
                changes["external_side_effect_unknown"] = (
                    view.external_side_effect_unknown
                    if target is RunState.WAITING_RECONCILIATION
                    else False
                )
            else:
                changes["external_side_effect_unknown"] = external_side_effect_unknown
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

    def _validate_edge(
        self, transition: HarnessTransition, view: HarnessSessionView
    ) -> TransitionValidation:
        if transition.entity_kind == "run":
            return self._validate_run(transition, view)
        if transition.entity_kind == "work_item":
            return self._validate_work_item(transition, view)
        return self._validate_attempt(transition, view)

    @staticmethod
    def _validate_run(
        transition: HarnessTransition, view: HarnessSessionView
    ) -> TransitionValidation:
        if transition.entity_id != transition.run_id:
            return TransitionValidation(
                False, "run entity identity does not match run"
            )
        try:
            target = RunState(transition.to_state)
        except ValueError:
            return TransitionValidation(False, "unknown run target state")
        source = view.run_state
        if transition.from_state != (source.value if source is not None else "none"):
            return TransitionValidation(
                False, "run source state does not match folded view"
            )
        if source in RUN_TERMINAL_STATES:
            return TransitionValidation(False, "terminal run state is absorbing")
        if view.external_side_effect_unknown and target in {
            RunState.FAILED,
            RunState.CANCELLED,
        }:
            return TransitionValidation(
                False, "unknown external effects require reconciliation"
            )
        allowed = set(RUN_EDGES.get(source, frozenset()))
        if source not in RUN_TERMINAL_STATES:
            allowed.update({RunState.FAILED, RunState.CANCELLED})
        if target not in allowed:
            return TransitionValidation(False, "illegal run transition")
        return TransitionValidation(True)

    @staticmethod
    def _validate_work_item(
        transition: HarnessTransition, view: HarnessSessionView
    ) -> TransitionValidation:
        try:
            target = WorkItemState(transition.to_state)
        except ValueError:
            return TransitionValidation(False, "unknown work-item target state")
        source = dict(view.work_item_states).get(transition.entity_id)
        if transition.from_state != (source.value if source is not None else "none"):
            return TransitionValidation(
                False, "work-item source state does not match folded view"
            )
        if source in WORK_ITEM_TERMINAL_STATES:
            return TransitionValidation(False, "terminal work-item state is absorbing")
        allowed = set(WORK_ITEM_EDGES.get(source, frozenset()))
        if source is not None:
            allowed.update(
                {
                    WorkItemState.BLOCKED,
                    WorkItemState.FAILED,
                    WorkItemState.CANCELLED,
                }
            )
        if target not in allowed:
            return TransitionValidation(False, "illegal work-item transition")
        return TransitionValidation(True)

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
            return TransitionValidation(
                False, "attempt source state does not match folded view"
            )
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
        return TransitionValidation(True)


def _replace_state(
    values: tuple[tuple[str, object], ...], identifier: str, state: object
) -> tuple[tuple[str, object], ...]:
    next_states = dict(values)
    next_states[identifier] = state
    return tuple(sorted(next_states.items()))


def _error_for(reason: str | None) -> StateMachineError:
    message = reason or "state-machine validation failed"
    if message.startswith("stale"):
        return StaleTransitionError(message)
    if "identity" in message:
        return IdentityMismatchError(message)
    if "idempotency" in message:
        return IdempotencyConflictError(message)
    if "dependency" in message:
        return DependencyVersionError(message)
    if "lease" in message or "fencing" in message:
        return LeaseEvidenceError(message)
    if "unknown external effects" in message:
        return SideEffectReconciliationRequiredError(message)
    return InvalidTransitionError(message)
