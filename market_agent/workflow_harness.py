"""Deterministic composition root for the Phase 1 Harness lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone
import math
from typing import Callable, Protocol, cast

from pydantic import StrictBool, model_validator

from market_agent.workflow_budget import BudgetSnapshot, WorkflowBudgetLedger
from market_agent.workflow_confidence_calibration import (
    ConfidenceCalibratorArtifact,
    ConfidenceGate,
    ConfidenceObservation,
)
from market_agent.workflow_contracts import (
    ContractModel,
    NonNegativeInt,
    ShortText,
    WorkflowMode,
    WorkflowRequest,
)
from market_agent.workflow_execution_backend import (
    CommittedExecutionSnapshot,
    CommittedTransitionReceipt,
    ExecutionBackend,
    ExecutionBackendError,
    ExecutionHandle,
    ExecutionRegistrationError,
    LangGraphExecutionBackend,
    canonical_plan_digest,
    canonical_transition_digest,
    canonical_view_digest,
)
from market_agent.workflow_harness_contracts import (
    HarnessPlan,
    HarnessSessionView,
    HarnessTransition,
    PinnedVersions,
    RunState,
    TransitionAuthorityRecord,
)
from market_agent.workflow_loop_guard import LoopGuard, SemanticCheckpoint
from market_agent.workflow_plan_registry import PlanCompiler
from market_agent.workflow_session import (
    HarnessEvent,
    HarnessEventStore,
    fold_events,
)
from market_agent.workflow_state_machine import (
    GlobalTaskStateMachine,
    RunTransitionEvidence,
)


class HarnessKernelError(RuntimeError):
    """Base class for deterministic Harness composition failures."""


class InvalidHarnessInputError(HarnessKernelError):
    """A public input was not an exact, freshly valid contract."""


class UnknownHarnessRunError(HarnessKernelError):
    """The requested run has no authoritative event stream."""


class HarnessDependencyError(HarnessKernelError):
    """A mandatory dependency failed readiness or returned invalid authority."""


class HarnessClock(Protocol):
    def utc_now(self) -> datetime: ...

    def monotonic(self) -> float: ...


class HarnessIdentifierSource(Protocol):
    def new(self, purpose: str) -> str: ...


class ExecutionCommitReceiptIssuer(Protocol):
    """Host event-store boundary; implementations own signing capability."""

    def ready(self) -> bool: ...

    def issue_snapshot(self, plan: HarnessPlan) -> CommittedExecutionSnapshot: ...

    def issue_transition_receipt(
        self,
        plan: HarnessPlan,
        transition: HarnessTransition,
        *,
        pre_sequence: int,
    ) -> CommittedTransitionReceipt: ...


class RunHandle(ContractModel):
    run_id: ShortText
    trace_id: ShortText
    plan_id: ShortText
    plan_revision: NonNegativeInt
    sequence: NonNegativeInt
    state_revision: NonNegativeInt
    run_state: RunState
    backend_synchronized: StrictBool


class HarnessDecision(ContractModel):
    run_id: ShortText
    trace_id: ShortText
    sequence: NonNegativeInt
    state_revision: NonNegativeInt
    run_state: RunState
    reason_code: ShortText
    transition: HarnessTransition | None = None
    retry_authorized: StrictBool = False
    no_trade: StrictBool = False
    reconciliation_required: StrictBool = False
    backend_synchronized: StrictBool = False

    @model_validator(mode="after")
    def validate_shape(self) -> HarnessDecision:
        if self.retry_authorized:
            raise ValueError("Phase 1 Harness decisions cannot authorize retries")
        if self.reconciliation_required and self.run_state is not RunState.WAITING_RECONCILIATION:
            raise ValueError("reconciliation is required only while waiting reconciliation")
        if self.transition is not None and (
            self.transition.run_id != self.run_id
            or self.transition.trace_id != self.trace_id
            or self.transition.to_state != self.run_state.value
            or self.transition.expected_state_revision + 1 != self.state_revision
        ):
            raise ValueError("decision and committed transition are inconsistent")
        return self


class _AdvanceCandidate(ContractModel):
    confidence_observation: ConfidenceObservation | None = None
    confidence_artifact: ConfidenceCalibratorArtifact | None = None
    loop_checkpoint: SemanticCheckpoint | None = None


class _PreparedRegistration:
    __slots__ = ("token", "rollback")

    def __init__(self, token: object, rollback: Callable[[object], None] | None) -> None:
        self.token = token
        self.rollback = rollback


_INITIAL_TARGETS = {
    RunState.CREATED: (RunState.ADMITTED, "request_admitted"),
    RunState.ADMITTED: (RunState.PLANNED, "plan_committed"),
    RunState.PLANNED: (RunState.READY, "dependencies_ready"),
    RunState.READY: (RunState.RUNNING, "execution_started"),
}
_TERMINAL_STATES = frozenset(
    {RunState.SUCCEEDED, RunState.DEGRADED, RunState.FAILED, RunState.CANCELLED}
)


def _fresh_contract(value: object, expected_type: type[ContractModel]) -> ContractModel:
    if type(value) is not expected_type:
        raise InvalidHarnessInputError(
            f"expected exact {expected_type.__name__} contract"
        )
    if set(value.__dict__).difference(expected_type.model_fields):
        raise InvalidHarnessInputError("contract contains undeclared fields")
    try:
        return expected_type.model_validate(value.__dict__)
    except Exception as error:
        raise InvalidHarnessInputError("contract failed strict revalidation") from error


def _strict_run_id(run_id: object) -> str:
    if type(run_id) is not str or not run_id or run_id != run_id.strip() or len(run_id) > 256:
        raise InvalidHarnessInputError("run identifier must be a canonical string")
    return run_id


class HarnessKernel:
    """Compose deterministic policies around the append-only event authority."""

    def __init__(
        self,
        *,
        event_store: HarnessEventStore,
        state_machine: GlobalTaskStateMachine,
        plan_compiler: PlanCompiler,
        pinned_versions: PinnedVersions,
        loop_guard_factory: Callable[[], LoopGuard],
        confidence_gate_factory: Callable[[HarnessPlan], ConfidenceGate],
        budget_factory: Callable[[WorkflowMode], WorkflowBudgetLedger],
        execution_backend: ExecutionBackend,
        receipt_issuer: ExecutionCommitReceiptIssuer,
        clock: HarnessClock,
        identifiers: HarnessIdentifierSource,
    ) -> None:
        if type(state_machine) is not GlobalTaskStateMachine:
            raise HarnessDependencyError("state machine must be the exact Phase 1 policy")
        if type(plan_compiler) is not PlanCompiler:
            raise HarnessDependencyError("plan compiler must be the exact Phase 1 compiler")
        self.event_store = event_store
        self._state_machine = state_machine
        self._plan_compiler = plan_compiler
        self._pinned_versions = cast(
            PinnedVersions, _fresh_contract(pinned_versions, PinnedVersions)
        )
        self._loop_guard_factory = loop_guard_factory
        self._confidence_gate_factory = confidence_gate_factory
        self._budget_factory = budget_factory
        self._execution = execution_backend
        self._receipt_issuer = receipt_issuer
        self._clock = clock
        self._identifiers = identifiers
        self._budgets: dict[str, WorkflowBudgetLedger] = {}
        self._confidence_gates: dict[str, ConfidenceGate] = {}

    def create(self, request: WorkflowRequest) -> RunHandle:
        request = cast(WorkflowRequest, _fresh_contract(request, WorkflowRequest))
        if self.event_store.load(request.workflow_id):
            raise ExecutionRegistrationError("run already exists")
        plan = self._plan_compiler.compile(request, self._pinned_versions)
        plan = cast(HarnessPlan, _fresh_contract(plan, HarnessPlan))
        if (
            plan.mode is not WorkflowMode.PASSIVE
            or plan.allows_side_effects
            or plan.run_id != request.workflow_id
            or plan.trace_id != request.trace_id
        ):
            raise HarnessDependencyError("current request did not compile to passive no-trade")

        provisional = HarnessSessionView(
            plan_revision=plan.revision,
            run_id=plan.run_id,
            trace_id=plan.trace_id,
        )
        prepared = self._prepare_dependencies(plan, provisional)
        published = False
        try:
            view, transition, _ = self._commit_run_transition(
                plan,
                HarnessSessionView.empty(),
                target=RunState.CREATED,
                reason_code="run_created",
                backend_handle=None,
                event_payload={"plan_json": plan.model_dump_json()},
            )
            published = True
        except BaseException:
            if not published and prepared.rollback is not None:
                prepared.rollback(prepared.token)
            raise

        backend_synchronized = True
        try:
            snapshot = self._issued_snapshot(plan, view)
            self._execution.register(plan, view, snapshot)
        except ExecutionBackendError:
            backend_synchronized = False
        return self._handle(plan, view, backend_synchronized)

    def resume(
        self, run_id: str, *, disposable_checkpoint: object | None = None
    ) -> RunHandle:
        run_id = _strict_run_id(run_id)
        events, plan, view = self._load(run_id)
        del events
        self._ensure_runtime_dependencies(plan)
        snapshot = self._issued_snapshot(plan, view)
        self._execution.resume(
            plan,
            view,
            snapshot,
            disposable_checkpoint=disposable_checkpoint,
        )
        return self._handle(plan, view, True)

    def snapshot(self, run_id: str) -> HarnessSessionView:
        run_id = _strict_run_id(run_id)
        events = self.event_store.load(run_id)
        if not events:
            raise UnknownHarnessRunError("unknown Harness run")
        return fold_events(events)

    def advance(
        self,
        run_id: str,
        *,
        candidate: object = None,
        expected_state_revision: int | None = None,
    ) -> HarnessDecision:
        run_id = _strict_run_id(run_id)
        events, plan, view = self._load(run_id)
        if expected_state_revision is not None and (
            type(expected_state_revision) is not int
            or expected_state_revision < 0
        ):
            raise InvalidHarnessInputError("expected revision must be a nonnegative integer")
        if expected_state_revision is not None and expected_state_revision != view.state_revision:
            return self._decision(plan, view, "stale_revision", backend_synchronized=False)

        parsed = self._candidate(candidate)
        if parsed is None:
            rejected = self._append_observation(
                plan, view, "candidate_rejected", {"policy": "strict_candidate_schema"}
            )
            return self._decision(
                plan, rejected, "candidate_rejected", backend_synchronized=False
            )
        if view.run_state in _TERMINAL_STATES:
            return self._decision(plan, view, "terminal_state", backend_synchronized=True)
        if view.run_state is RunState.WAITING_RECONCILIATION:
            return self._decision(
                plan,
                view,
                "reconciliation_required",
                reconciliation_required=True,
                backend_synchronized=True,
            )

        self._ensure_runtime_dependencies(plan)
        current_snapshot = self._issued_snapshot(plan, view)
        handle = self._execution.resume(plan, view, current_snapshot)
        target, reason, no_trade, payload = self._policy_decision(
            events, plan, view, parsed
        )
        post, transition, backend_synchronized = self._commit_run_transition(
            plan,
            view,
            target=target,
            reason_code=reason,
            backend_handle=handle,
            event_payload=payload,
        )
        return self._decision(
            plan,
            post,
            reason,
            transition=transition,
            no_trade=no_trade,
            backend_synchronized=backend_synchronized,
        )

    def cancel(self, run_id: str, reason: str) -> HarnessDecision:
        run_id = _strict_run_id(run_id)
        if type(reason) is not str or not reason or reason != reason.strip() or len(reason) > 256:
            raise InvalidHarnessInputError("cancellation reason must be canonical text")
        _, plan, view = self._load(run_id)
        recorded = self._append_observation(
            plan,
            view,
            "cancellation_requested",
            {"reason_code": reason, "policy": "task4_cancellation"},
        )
        if (
            recorded.run_state is RunState.WAITING_RECONCILIATION
            or recorded.external_side_effect_unknown
        ):
            return self._decision(
                plan,
                recorded,
                "cancellation_waits_for_reconciliation",
                reconciliation_required=True,
                backend_synchronized=False,
            )
        return self._decision(
            plan, recorded, "cancellation_intent_recorded", backend_synchronized=False
        )

    def _prepare_dependencies(
        self, plan: HarnessPlan, provisional: HarnessSessionView
    ) -> _PreparedRegistration:
        self._ensure_runtime_dependencies(plan)
        try:
            ready = self._receipt_issuer.ready()
        except Exception as error:
            raise HarnessDependencyError("receipt issuer readiness failed") from error
        if type(ready) is not bool or not ready:
            raise HarnessDependencyError("receipt issuer is not ready")

        prepare = getattr(self._execution, "prepare_registration", None)
        rollback = getattr(self._execution, "rollback_registration", None)
        if callable(prepare):
            token = prepare(plan, provisional)
            return _PreparedRegistration(token, rollback if callable(rollback) else None)
        if type(self._execution) is LangGraphExecutionBackend:
            return _PreparedRegistration(None, None)
        raise ExecutionRegistrationError(
            "backend requires explicit provisional registration readiness"
        )

    def _ensure_runtime_dependencies(self, plan: HarnessPlan) -> None:
        if plan.pinned_versions != self._pinned_versions:
            raise HarnessDependencyError("stored plan dependency pins changed")
        loop_guard = self._loop_guard_factory()
        confidence_gate = self._confidence_gate_factory(plan)
        budget = self._budgets.get(plan.run_id)
        if budget is None:
            budget = self._budget_factory(plan.mode)
        if type(loop_guard) is not LoopGuard:
            raise HarnessDependencyError("loop guard factory returned an invalid policy")
        if type(confidence_gate) is not ConfidenceGate:
            raise HarnessDependencyError("confidence gate factory returned an invalid policy")
        if type(budget) is not WorkflowBudgetLedger:
            raise HarnessDependencyError("budget factory returned an invalid ledger")
        snapshot = budget.snapshot()
        if type(snapshot) is not BudgetSnapshot or snapshot.mode is not plan.mode:
            raise HarnessDependencyError("budget snapshot does not match the run")
        self._budgets[plan.run_id] = budget
        self._confidence_gates[plan.run_id] = confidence_gate

    @staticmethod
    def _candidate(candidate: object) -> _AdvanceCandidate | None:
        if candidate is None:
            return _AdvanceCandidate()
        if type(candidate) is not dict:
            return None
        try:
            return _AdvanceCandidate.model_validate(candidate)
        except Exception:
            return None

    def _policy_decision(
        self,
        events: tuple[HarnessEvent, ...],
        plan: HarnessPlan,
        view: HarnessSessionView,
        candidate: _AdvanceCandidate,
    ) -> tuple[RunState, str, bool, dict[str, object]]:
        initial = _INITIAL_TARGETS.get(view.run_state)
        if initial is not None:
            return initial[0], initial[1], False, {"policy": "state_machine"}
        if view.run_state is RunState.DEGRADING:
            return RunState.SUMMARIZING, "safe_no_trade_summary", True, {
                "policy": "degradation"
            }
        if view.run_state is RunState.SUMMARIZING:
            return RunState.DEGRADED, "safe_no_trade_due_to_degradation", True, {
                "policy": "terminal_degradation"
            }
        if view.run_state is not RunState.RUNNING:
            raise HarnessKernelError("run has no deterministic advance policy")

        payload: dict[str, object] = {"policy": "confidence_gate"}
        budget = self._budgets[plan.run_id].snapshot()
        if type(budget) is not BudgetSnapshot or budget.mode is not plan.mode:
            raise HarnessDependencyError("budget snapshot does not match the run")
        if budget.exhausted or budget.overdrawn:
            return RunState.DEGRADING, "budget_exhausted", True, {
                "policy": "budget"
            }
        if candidate.loop_checkpoint is not None:
            guard = self._replayed_loop_guard(events)
            loop_decision = guard.observe_checkpoint(candidate.loop_checkpoint)
            payload["loop_checkpoint_json"] = candidate.loop_checkpoint.model_dump_json()
            payload["loop_reason"] = loop_decision.stop_reason or "allowed"
            if not loop_decision.allowed:
                return RunState.DEGRADING, "loop_guard_stopped", True, payload

        gate = self._confidence_gates[plan.run_id]
        confidence = gate.evaluate(
            candidate.confidence_observation, candidate.confidence_artifact
        )
        payload["confidence_action"] = confidence.next_action
        payload["confidence_reason"] = confidence.reason_code
        if confidence.may_succeed:
            return RunState.SUMMARIZING, "confidence_sufficient", False, payload
        return RunState.DEGRADING, "confidence_fail_closed", True, payload

    def _replayed_loop_guard(self, events: tuple[HarnessEvent, ...]) -> LoopGuard:
        guard = self._loop_guard_factory()
        if type(guard) is not LoopGuard:
            raise HarnessDependencyError("loop guard factory returned an invalid policy")
        for event in events:
            value = event.payload.get("loop_checkpoint_json")
            if value is None:
                continue
            if type(value) is not str:
                raise HarnessDependencyError("committed loop checkpoint is invalid")
            try:
                checkpoint = SemanticCheckpoint.model_validate_json(value)
            except Exception as error:
                raise HarnessDependencyError("committed loop checkpoint is invalid") from error
            guard.observe_checkpoint(checkpoint)
        return guard

    def _commit_run_transition(
        self,
        plan: HarnessPlan,
        view: HarnessSessionView,
        *,
        target: RunState,
        reason_code: str,
        backend_handle: ExecutionHandle | None,
        event_payload: dict[str, object],
    ) -> tuple[HarnessSessionView, HarnessTransition, bool]:
        source = view.run_state.value if view.run_state is not None else "none"
        transition = HarnessTransition(
            run_id=plan.run_id,
            trace_id=plan.trace_id,
            entity_kind="run",
            entity_id=plan.run_id,
            from_state=source,
            to_state=target.value,
            expected_state_revision=view.state_revision,
            plan_revision=plan.revision,
            reason_code=reason_code,
            idempotency_key=f"run-{view.state_revision}-{target.value}",
        )
        authority = TransitionAuthorityRecord(
            run_id=plan.run_id,
            trace_id=plan.trace_id,
            entity_kind="run",
            entity_id=plan.run_id,
            from_state=source,
            to_state=target.value,
            expected_state_revision=view.state_revision,
            plan_revision=plan.revision,
            reason_code=reason_code,
            idempotency_key=transition.idempotency_key,
            dependency_versions=view.dependency_versions,
        )
        authorization = RunTransitionEvidence(
            run_id=plan.run_id,
            trace_id=plan.trace_id,
            entity_id=plan.run_id,
            expected_state_revision=view.state_revision,
            plan_revision=plan.revision,
            dependency_versions=view.dependency_versions,
        )
        authority_already_committed = authority in view.transition_authorities
        projected = view
        if not authority_already_committed:
            projected = HarnessSessionView.model_validate(
                view.model_copy(
                    update={
                        "transition_authorities": (
                            *view.transition_authorities,
                            authority,
                        )
                    }
                ).model_dump(mode="python")
            )
        validation = self._state_machine.validate(
            transition, projected, authorization=authorization
        )
        if not validation.allowed:
            raise HarnessKernelError(f"deterministic transition rejected: {validation.reason}")

        # Creation is one event-store transaction.  There is no committed
        # snapshot before this append, so the initial policy evidence is
        # validated above and recorded in the transition event payload; later
        # transitions use a separately committed authority pre-fold.
        if view.run_id is None and backend_handle is None:
            create_event = self._event(
                plan,
                "transition_committed",
                payload={**event_payload, "reason_code": reason_code},
                transition=transition,
            )
            self.event_store.append(
                create_event,
                expected_sequence=view.sequence,
                expected_state_revision=view.state_revision,
            )
            return self.snapshot(plan.run_id), transition, True

        if authority_already_committed:
            pre_view = view
        else:
            authority_event = self._event(
                plan,
                "transition_authorized",
                payload={**event_payload, "reason_code": reason_code},
                transition_authority=authority,
            )
            self.event_store.append(
                authority_event,
                expected_sequence=view.sequence,
                expected_state_revision=view.state_revision,
            )
            pre_view = self.snapshot(plan.run_id)
        pre_snapshot = self._issued_snapshot(plan, pre_view)
        if backend_handle is not None:
            backend_handle = self._execution.resume(plan, pre_view, pre_snapshot)

        transition_event = self._event(
            plan,
            "transition_committed",
            payload={"reason_code": reason_code, "policy": event_payload.get("policy", "create")},
            transition=transition,
        )
        self.event_store.append(
            transition_event,
            expected_sequence=pre_view.sequence,
            expected_state_revision=pre_view.state_revision,
        )
        post_view = self.snapshot(plan.run_id)
        if backend_handle is None:
            return post_view, transition, True

        receipt = self._issued_receipt(
            plan, transition, pre_view, post_view, pre_snapshot
        )
        synchronized = True
        try:
            self._execution.apply_committed_transition(
                backend_handle,
                transition,
                pre_view,
                post_view,
                receipt,
            )
        except ExecutionBackendError:
            synchronized = False
        return post_view, transition, synchronized

    def _append_observation(
        self,
        plan: HarnessPlan,
        view: HarnessSessionView,
        event_type: str,
        payload: dict[str, object],
    ) -> HarnessSessionView:
        event = self._event(plan, event_type, payload=payload)
        self.event_store.append(
            event,
            expected_sequence=view.sequence,
            expected_state_revision=view.state_revision,
        )
        return self.snapshot(plan.run_id)

    def _event(
        self,
        plan: HarnessPlan,
        event_type: str,
        *,
        payload: dict[str, object],
        transition: HarnessTransition | None = None,
        transition_authority: TransitionAuthorityRecord | None = None,
    ) -> HarnessEvent:
        occurred_at = self._clock.utc_now()
        monotonic = self._clock.monotonic()
        if (
            type(occurred_at) is not datetime
            or occurred_at.tzinfo is None
            or occurred_at.utcoffset() != timezone.utc.utcoffset(occurred_at)
            or type(monotonic) is not float
            or not math.isfinite(monotonic)
            or monotonic < 0.0
        ):
            raise HarnessDependencyError("clock returned an invalid deterministic point")
        event_id = self._identifier("event")
        span_id = self._identifier("span")
        return HarnessEvent(
            event_id=event_id,
            trace_id=plan.trace_id,
            span_id=span_id,
            run_id=plan.run_id,
            event_type=event_type,
            occurred_at=occurred_at,
            monotonic_offset=monotonic,
            actor="harness-kernel",
            payload=payload,
            transition=transition,
            transition_authority=transition_authority,
        )

    def _identifier(self, purpose: str) -> str:
        value = self._identifiers.new(purpose)
        if type(value) is not str or not value or value != value.strip() or len(value) > 256:
            raise HarnessDependencyError("identifier source returned invalid text")
        return value

    def _load(
        self, run_id: str
    ) -> tuple[tuple[HarnessEvent, ...], HarnessPlan, HarnessSessionView]:
        events = self.event_store.load(run_id)
        if type(events) is not tuple or not events:
            raise UnknownHarnessRunError("unknown Harness run")
        view = fold_events(events)
        plan = self._plan_from_events(events)
        if (
            plan.run_id != run_id
            or plan.run_id != view.run_id
            or plan.trace_id != view.trace_id
            or plan.revision != view.plan_revision
        ):
            raise HarnessDependencyError("stored plan and folded stream disagree")
        return events, plan, view

    @staticmethod
    def _plan_from_events(events: tuple[HarnessEvent, ...]) -> HarnessPlan:
        candidates = [
            event.payload.get("plan_json")
            for event in events
            if "plan_json" in event.payload
        ]
        if len(candidates) != 1:
            raise HarnessDependencyError("run stream must contain one committed plan")
        if type(candidates[0]) is not str:
            raise HarnessDependencyError("committed plan is invalid")
        try:
            return HarnessPlan.model_validate_json(candidates[0])
        except Exception as error:
            raise HarnessDependencyError("committed plan is invalid") from error

    def _issued_snapshot(
        self, plan: HarnessPlan, view: HarnessSessionView
    ) -> CommittedExecutionSnapshot:
        try:
            value = self._receipt_issuer.issue_snapshot(plan)
            snapshot = cast(
                CommittedExecutionSnapshot,
                _fresh_contract(value, CommittedExecutionSnapshot),
            )
        except Exception as error:
            raise HarnessDependencyError("host snapshot issuer failed") from error
        if (
            snapshot.run_id != plan.run_id
            or snapshot.trace_id != plan.trace_id
            or snapshot.plan_id != plan.plan_id
            or snapshot.plan_digest != canonical_plan_digest(plan)
            or snapshot.plan_revision != plan.revision
            or snapshot.sequence != view.sequence
            or snapshot.state_revision != view.state_revision
            or snapshot.view_digest != canonical_view_digest(view)
            or snapshot.event_head_hash != view.last_event_hash
        ):
            raise HarnessDependencyError("host snapshot does not bind committed truth")
        return snapshot

    def _issued_receipt(
        self,
        plan: HarnessPlan,
        transition: HarnessTransition,
        pre_view: HarnessSessionView,
        post_view: HarnessSessionView,
        pre_snapshot: CommittedExecutionSnapshot,
    ) -> CommittedTransitionReceipt:
        try:
            value = self._receipt_issuer.issue_transition_receipt(
                plan, transition, pre_sequence=pre_view.sequence
            )
            receipt = cast(
                CommittedTransitionReceipt,
                _fresh_contract(value, CommittedTransitionReceipt),
            )
        except Exception as error:
            raise HarnessDependencyError("host transition receipt issuer failed") from error
        if (
            receipt.pre != pre_snapshot
            or receipt.transition_digest != canonical_transition_digest(transition)
        ):
            raise HarnessDependencyError("host receipt does not bind the committed transition")
        self._validate_snapshot_value(plan, post_view, receipt.post)
        return receipt

    @staticmethod
    def _validate_snapshot_value(
        plan: HarnessPlan,
        view: HarnessSessionView,
        snapshot: CommittedExecutionSnapshot,
    ) -> None:
        if (
            snapshot.run_id != plan.run_id
            or snapshot.trace_id != plan.trace_id
            or snapshot.plan_id != plan.plan_id
            or snapshot.plan_digest != canonical_plan_digest(plan)
            or snapshot.plan_revision != plan.revision
            or snapshot.sequence != view.sequence
            or snapshot.state_revision != view.state_revision
            or snapshot.view_digest != canonical_view_digest(view)
            or snapshot.event_head_hash != view.last_event_hash
        ):
            raise HarnessDependencyError("receipt endpoint does not bind committed truth")

    @staticmethod
    def _handle(
        plan: HarnessPlan, view: HarnessSessionView, synchronized: bool
    ) -> RunHandle:
        if view.run_state is None:
            raise HarnessDependencyError("published run has no state")
        return RunHandle(
            run_id=plan.run_id,
            trace_id=plan.trace_id,
            plan_id=plan.plan_id,
            plan_revision=plan.revision,
            sequence=view.sequence,
            state_revision=view.state_revision,
            run_state=view.run_state,
            backend_synchronized=synchronized,
        )

    @staticmethod
    def _decision(
        plan: HarnessPlan,
        view: HarnessSessionView,
        reason: str,
        *,
        transition: HarnessTransition | None = None,
        no_trade: bool = False,
        reconciliation_required: bool = False,
        backend_synchronized: bool,
    ) -> HarnessDecision:
        if view.run_state is None:
            raise HarnessDependencyError("decision requires an identified run state")
        return HarnessDecision(
            run_id=plan.run_id,
            trace_id=plan.trace_id,
            sequence=view.sequence,
            state_revision=view.state_revision,
            run_state=view.run_state,
            reason_code=reason,
            transition=transition,
            retry_authorized=False,
            no_trade=no_trade,
            reconciliation_required=reconciliation_required,
            backend_synchronized=backend_synchronized,
        )
