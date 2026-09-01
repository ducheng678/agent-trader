"""Deterministic composition root for the Phase 1 Harness lifecycle."""

from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
import json
import math
import os
from pathlib import Path
from functools import wraps
from threading import RLock
from typing import Any, Callable, Mapping, Protocol, cast
from weakref import WeakValueDictionary

from pydantic import StrictBool, model_validator

from market_agent.openai_usage import UsageTokens, estimate_workflow_usage_cost
from market_agent.workflow_budget import (
    BudgetExceededError,
    BudgetReservation,
    BudgetSnapshot,
    BudgetSettlement,
    WorkflowBudgetLedger,
)
from market_agent.workflow_confidence_calibration import (
    ConfidenceCalibratorArtifact,
    ConfidenceGate,
    ConfidenceObservation,
    TrustedConfidencePolicy,
    TrustedRequestContext,
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
    IssuerTrustDescriptor,
    RegistrationPreparation,
    canonical_plan_digest,
    canonical_transition_digest,
    canonical_view_digest,
    verify_committed_execution_snapshot,
)
from market_agent.workflow_harness_contracts import (
    HarnessPlan,
    HarnessSessionView,
    HarnessTransition,
    PinnedVersions,
    RunState,
    TransitionAuthorityRecord,
)
from market_agent.workflow_loop_guard import (
    ActionObservationFingerprint,
    LoopGuard,
    SemanticCheckpoint,
)
from market_agent.workflow_model_routing import policy_for
from market_agent.workflow_plan_registry import PlanCompiler
from market_agent.workflow_session import (
    HarnessEvent,
    HarnessEventStore,
    OptimisticConcurrencyError,
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

    def trust_descriptor(self) -> IssuerTrustDescriptor: ...

    def issue_snapshot(self, plan: HarnessPlan) -> CommittedExecutionSnapshot: ...

    def issue_transition_receipt(
        self,
        plan: HarnessPlan,
        transition: HarnessTransition,
        *,
        pre_sequence: int,
    ) -> CommittedTransitionReceipt: ...


class _StrictOutput(ContractModel):
    def model_copy(
        self, *, update: dict[str, Any] | None = None, deep: bool = False
    ) -> Any:
        values = self.model_dump(mode="python", round_trip=True)
        values.update(update or {})
        return type(self).model_validate(values)


class RunHandle(_StrictOutput):
    run_id: ShortText
    trace_id: ShortText
    plan_id: ShortText
    plan_revision: NonNegativeInt
    sequence: NonNegativeInt
    state_revision: NonNegativeInt
    run_state: RunState
    backend_synchronized: StrictBool


class HarnessDecision(_StrictOutput):
    run_id: ShortText
    trace_id: ShortText
    sequence: NonNegativeInt
    state_revision: NonNegativeInt
    run_state: RunState
    plan_revision: NonNegativeInt
    reason_code: ShortText
    previous_run_state: RunState | None = None
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
        if self.run_state is RunState.WAITING_RECONCILIATION and not self.reconciliation_required:
            raise ValueError("waiting reconciliation decisions must require reconciliation")
        must_be_no_trade = (
            self.run_state in {
                RunState.DEGRADING,
                RunState.DEGRADED,
                RunState.WAITING_RECONCILIATION,
            }
            or (
                self.run_state is RunState.SUMMARIZING
                and self.reason_code != "confidence_sufficient"
            )
        )
        if self.no_trade != must_be_no_trade:
            raise ValueError("decision no-trade flag and state/reason are inconsistent")
        if self.reason_code == "completed" and self.run_state is not RunState.SUCCEEDED:
            raise ValueError("completed reason requires succeeded state")
        allowed_states = _DECISION_REASON_STATES.get(self.reason_code)
        if allowed_states is None or self.run_state not in allowed_states:
            raise ValueError("decision reason and run state are inconsistent")
        if (
            self.transition is not None
            and self.reason_code not in _COMMITTED_DECISION_REASONS
        ):
            raise ValueError("non-committing decisions cannot carry a transition")
        if self.transition is None:
            if self.previous_run_state is not None:
                raise ValueError("non-transition decisions cannot claim a prior state")
            return self
        transition = self.transition
        if (
            self.previous_run_state is None
            or transition.entity_kind != "run"
            or transition.entity_id != self.run_id
            or transition.run_id != self.run_id
            or transition.trace_id != self.trace_id
            or transition.from_state != self.previous_run_state.value
            or transition.to_state != self.run_state.value
            or transition.expected_state_revision + 1 != self.state_revision
            or transition.plan_revision != self.plan_revision
            or transition.reason_code != self.reason_code
            or transition.idempotency_key
            != f"run-{transition.expected_state_revision}-{self.run_state.value}"
        ):
            raise ValueError("decision and committed transition are inconsistent")
        return self


class _AdvanceCandidate(ContractModel):
    confidence_observation: ConfidenceObservation | None = None
    confidence_artifact: ConfidenceCalibratorArtifact | None = None
    loop_checkpoint: SemanticCheckpoint | None = None
    action_observation: ActionObservationFingerprint | None = None


class _PreparedRegistration:
    __slots__ = ("token",)

    def __init__(self, token: RegistrationPreparation) -> None:
        self.token = token


_INITIAL_TARGETS = {
    RunState.CREATED: (RunState.ADMITTED, "request_admitted"),
    RunState.ADMITTED: (RunState.PLANNED, "plan_committed"),
    RunState.PLANNED: (RunState.READY, "dependencies_ready"),
    RunState.READY: (RunState.RUNNING, "execution_started"),
}
_TERMINAL_STATES = frozenset(
    {RunState.SUCCEEDED, RunState.DEGRADED, RunState.FAILED, RunState.CANCELLED}
)

_ALL_RUN_STATES = frozenset(RunState)
_DECISION_REASON_STATES: dict[str, frozenset[RunState]] = {
    "request_admitted": frozenset({RunState.ADMITTED}),
    "plan_committed": frozenset({RunState.PLANNED}),
    "dependencies_ready": frozenset({RunState.READY}),
    "execution_started": frozenset({RunState.RUNNING}),
    "safe_no_trade_summary": frozenset({RunState.SUMMARIZING}),
    "confidence_sufficient": frozenset({RunState.SUMMARIZING}),
    "completed": frozenset({RunState.SUCCEEDED}),
    "budget_exhausted": frozenset({RunState.DEGRADING}),
    "loop_guard_stopped": frozenset({RunState.DEGRADING}),
    "loop_checkpoint_binding_mismatch": frozenset({RunState.DEGRADING}),
    "confidence_context_mismatch": frozenset({RunState.DEGRADING}),
    "confidence_fail_closed": frozenset({RunState.DEGRADING}),
    "safe_no_trade_due_to_degradation": frozenset(
        {RunState.DEGRADING, RunState.DEGRADED}
    ),
    "reconciliation_required": frozenset({RunState.WAITING_RECONCILIATION}),
    "cancellation_waits_for_reconciliation": frozenset(
        {RunState.WAITING_RECONCILIATION}
    ),
    "cancellation_completed": frozenset({RunState.CANCELLED}),
    "terminal_state": _TERMINAL_STATES,
    "stale_revision": _ALL_RUN_STATES,
    "candidate_rejected": _ALL_RUN_STATES,
    "cancellation_intent_recorded": frozenset(
        {
            RunState.CREATED,
            RunState.ADMITTED,
            RunState.PLANNED,
            RunState.READY,
            RunState.RUNNING,
            RunState.RECONCILING,
            RunState.DEGRADING,
            RunState.SUMMARIZING,
            RunState.SUCCEEDED,
            RunState.DEGRADED,
            RunState.FAILED,
        }
    ),
}

_BUDGET_INTEGER_FIELDS = frozenset(
    {
        "budget_before_attempts",
        "budget_delta_attempts",
        "budget_remaining_attempts",
        "budget_before_tokens",
        "budget_delta_tokens",
        "budget_remaining_tokens",
    }
)
_BUDGET_DECIMAL_FIELDS = frozenset(
    {
        "budget_before_seconds",
        "budget_delta_seconds",
        "budget_remaining_seconds",
        "budget_before_cost",
        "budget_delta_cost",
        "budget_remaining_cost",
    }
)
_BUDGET_FIELDS = _BUDGET_INTEGER_FIELDS | _BUDGET_DECIMAL_FIELDS
_BUDGET_SETTLEMENT_EVIDENCE_KEY = "budget_settlement_evidence"
_BUDGET_SETTLEMENT_SCHEMA = "budget-settlement-evidence-v1"
_BUDGET_AUTHORITY_TARGETS = {
    "budget_exhausted": RunState.DEGRADING,
    "loop_guard_stopped": RunState.DEGRADING,
    "loop_checkpoint_binding_mismatch": RunState.DEGRADING,
    "confidence_context_mismatch": RunState.DEGRADING,
    "confidence_fail_closed": RunState.DEGRADING,
    "confidence_sufficient": RunState.SUMMARIZING,
}

# A decision is an observation unless it is one of these explicitly committed
# state-machine outcomes.  Keeping this closed makes it impossible for callers
# to manufacture a transition-shaped response for stale/terminal/rejected
# requests by using ``model_copy``.
_COMMITTED_DECISION_REASONS = frozenset(
    {
        "request_admitted",
        "plan_committed",
        "dependencies_ready",
        "execution_started",
        "safe_no_trade_summary",
        "confidence_sufficient",
        "completed",
        "budget_exhausted",
        "loop_guard_stopped",
        "loop_checkpoint_binding_mismatch",
        "confidence_context_mismatch",
        "confidence_fail_closed",
        "safe_no_trade_due_to_degradation",
        "cancellation_completed",
    }
)

_RUN_LOCKS_GUARD = RLock()
_RUN_LOCKS: WeakValueDictionary[tuple[object, str], RLock] = WeakValueDictionary()


def _store_coordination_identity(store: HarnessEventStore) -> object:
    database_path = getattr(store, "_database_path", None)
    if type(database_path) is not str or database_path == ":memory:":
        return (type(store), id(store))
    canonical_path = os.path.normcase(
        os.path.normpath(str(Path(database_path).resolve(strict=False)))
    )
    return (type(store), canonical_path)


def _run_lock(store: HarnessEventStore, run_id: str) -> RLock:
    key = (_store_coordination_identity(store), _strict_run_id(run_id))
    with _RUN_LOCKS_GUARD:
        existing = _RUN_LOCKS.get(key)
        if existing is not None:
            return existing
        created = RLock()
        _RUN_LOCKS[key] = created
        return created


def _serialized_run(method: Callable[..., Any]) -> Callable[..., Any]:
    @wraps(method)
    def wrapper(self: HarnessKernel, run_id: str, *args: Any, **kwargs: Any) -> Any:
        canonical = _strict_run_id(run_id)
        with _run_lock(self.event_store, canonical):
            for retry in range(2):
                try:
                    return method(self, canonical, *args, **kwargs)
                except OptimisticConcurrencyError:
                    self._budgets.pop(canonical, None)
                    self._confidence_gates.pop(canonical, None)
                    self._confidence_binding_failed.discard(canonical)
                    if retry:
                        raise
            raise AssertionError("unreachable concurrency retry")

    return wrapper


def _fresh_contract(value: object, expected_type: type[ContractModel]) -> ContractModel:
    if type(value) is not expected_type:
        raise InvalidHarnessInputError(
            f"expected exact {expected_type.__name__} contract"
        )
    if set(value.__dict__).difference(expected_type.model_fields):
        raise InvalidHarnessInputError("contract contains undeclared fields")
    try:
        return expected_type.model_validate(
            value.model_dump(mode="python", round_trip=True)
        )
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
        self._confidence_binding_failed: set[str] = set()
        self._backend_cancelled: set[str] = set()

    def create(self, request: WorkflowRequest) -> RunHandle:
        request = cast(WorkflowRequest, _fresh_contract(request, WorkflowRequest))
        with _run_lock(self.event_store, request.workflow_id):
            return self._create_locked(request)

    def _create_locked(self, request: WorkflowRequest) -> RunHandle:
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
            run_id=plan.run_id,
            trace_id=plan.trace_id,
            plan_revision=plan.revision,
        )
        prepared = self._prepare_dependencies(plan, provisional)
        committed_registration = False
        try:
            view, _, _ = self._commit_run_transition(
                plan,
                HarnessSessionView.empty(),
                target=RunState.CREATED,
                reason_code="run_created",
                backend_handle=None,
                event_payload={"plan_json": plan.model_dump_json()},
            )
            snapshot = self._issued_snapshot(plan, view)
            self._execution.commit_registration(prepared.token, snapshot)
            committed_registration = True
            return self._handle(plan, view, True)
        finally:
            if not committed_registration:
                self._execution.rollback_registration(prepared.token)

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

    @_serialized_run
    def advance(
        self,
        run_id: str,
        *,
        candidate: object = None,
        expected_state_revision: int | None = None,
    ) -> HarnessDecision:
        run_id = _strict_run_id(run_id)
        events, plan, view = self._load(run_id)

        pending = self._pending_authority(events, view)
        if pending is not None:
            authority, payload = pending
            self._ensure_runtime_dependencies(plan)
            snapshot = self._issued_snapshot(plan, view)
            handle = self._execution.resume(plan, view, snapshot)
            post, transition, synchronized = self._commit_run_transition(
                plan,
                view,
                target=RunState(authority.to_state),
                reason_code=authority.reason_code,
                backend_handle=handle,
                event_payload=payload,
            )
            return self._decision(
                plan,
                post,
                authority.reason_code,
                transition=transition,
                no_trade=self._strict_payload_bool(payload, "no_trade"),
                backend_synchronized=synchronized,
            )

        if expected_state_revision is not None and (
            type(expected_state_revision) is not int
            or expected_state_revision < 0
        ):
            raise InvalidHarnessInputError("expected revision must be a nonnegative integer")
        if expected_state_revision is not None and expected_state_revision != view.state_revision:
            return self._decision(plan, view, "stale_revision", backend_synchronized=False)

        parsed, candidate_digest = self._candidate(candidate)
        if parsed is None:
            rejected = self._append_observation(
                plan, view, "candidate_rejected", {"policy": "strict_candidate_schema"}
            )
            return self._decision(
                plan, rejected, "candidate_rejected", backend_synchronized=False
            )
        if candidate_digest is not None:
            replay = self._committed_candidate(events, candidate_digest)
            if replay is not None:
                transition, payload = replay
                self._ensure_runtime_dependencies(plan)
                snapshot = self._issued_snapshot(plan, view)
                self._execution.resume(plan, view, snapshot)
                return self._decision(
                    plan,
                    view,
                    transition.reason_code,
                    transition=transition,
                    no_trade=self._strict_payload_bool(payload, "no_trade"),
                    backend_synchronized=True,
                )
        if view.run_state in _TERMINAL_STATES:
            return self._decision(
                plan,
                view,
                "terminal_state",
                no_trade=view.run_state is RunState.DEGRADED,
                backend_synchronized=True,
            )
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
        target, reason, no_trade, payload = self._policy_decision(events, plan, view, parsed)
        payload["no_trade"] = no_trade
        if candidate_digest is not None:
            payload["candidate_digest"] = candidate_digest
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

    @_serialized_run
    def cancel(self, run_id: str, reason: str) -> HarnessDecision:
        run_id = _strict_run_id(run_id)
        if type(reason) is not str or not reason or reason != reason.strip() or len(reason) > 256:
            raise InvalidHarnessInputError("cancellation reason must be canonical text")
        events, plan, view = self._load(run_id)
        existing_intents = tuple(
            event for event in events if event.event_type == "cancellation_requested"
        )
        if len(existing_intents) > 1:
            raise HarnessDependencyError("run has duplicate cancellation intents")
        recorded = view
        if not existing_intents:
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
        if recorded.run_state is RunState.CANCELLED:
            if plan.run_id not in self._backend_cancelled:
                self._ensure_runtime_dependencies(plan)
                snapshot = self._issued_snapshot(plan, recorded)
                try:
                    self._execution.resume(plan, recorded, snapshot)
                except ExecutionBackendError:
                    return self._decision(
                        plan,
                        recorded,
                        "cancellation_completed",
                        backend_synchronized=False,
                    )
                else:
                    self._execution.cancel(plan.run_id)
                    self._backend_cancelled.add(plan.run_id)
            return self._decision(
                plan,
                recorded,
                "cancellation_completed",
                backend_synchronized=True,
            )
        if recorded.run_state is RunState.WAITING_APPROVAL:
            self._ensure_runtime_dependencies(plan)
            snapshot = self._issued_snapshot(plan, recorded)
            handle = self._execution.resume(plan, recorded, snapshot)
            post, transition, synchronized = self._commit_run_transition(
                plan,
                recorded,
                target=RunState.CANCELLED,
                reason_code="cancellation_completed",
                backend_handle=handle,
                event_payload={
                    "policy": "task4_cancellation",
                    "no_trade": False,
                },
            )
            if synchronized:
                self._execution.cancel(plan.run_id)
                self._backend_cancelled.add(plan.run_id)
            return self._decision(
                plan,
                post,
                "cancellation_completed",
                transition=transition,
                backend_synchronized=synchronized,
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

        try:
            descriptor = cast(
                IssuerTrustDescriptor,
                _fresh_contract(
                    self._receipt_issuer.trust_descriptor(),
                    IssuerTrustDescriptor,
                ),
            )
            token = self._execution.prepare_registration(
                plan, provisional, descriptor
            )
            token = cast(
                RegistrationPreparation,
                _fresh_contract(token, RegistrationPreparation),
            )
        except ExecutionBackendError:
            raise
        except Exception as error:
            raise HarnessDependencyError(
                "registration preparation failed"
            ) from error
        if token.issuer != descriptor:
            try:
                self._execution.rollback_registration(token)
            finally:
                raise HarnessDependencyError(
                    "registration token changed issuer trust binding"
                )
        return _PreparedRegistration(token)

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
        if plan.run_id not in self._budgets:
            self._restore_budget_from_events(plan, budget)
        snapshot = budget.snapshot()
        if type(snapshot) is not BudgetSnapshot or snapshot.mode is not plan.mode:
            raise HarnessDependencyError("budget snapshot does not match the run")
        self._budgets[plan.run_id] = budget
        self._confidence_gates[plan.run_id] = confidence_gate
        if not self._confidence_gate_matches_plan(plan, confidence_gate):
            self._confidence_binding_failed.add(plan.run_id)

    @staticmethod
    def _confidence_gate_matches_plan(
        plan: HarnessPlan, gate: ConfidenceGate
    ) -> bool:
        try:
            context = gate.trusted_context_snapshot()
            policy = gate.trusted_policy_snapshot()
            if type(context) is not TrustedRequestContext or type(policy) is not TrustedConfidencePolicy:
                return False
            context = cast(
                TrustedRequestContext,
                _fresh_contract(context, TrustedRequestContext),
            )
            policy = cast(
                TrustedConfidencePolicy,
                _fresh_contract(policy, TrustedConfidencePolicy),
            )
        except Exception:
            return False
        expected_policy_hash = sha256(
            plan.pinned_versions.policy_version.encode("utf-8")
        ).hexdigest()
        gates = context.hard_gates
        return (
            context.request_class == "informational"
            and gates.run_id == plan.run_id
            and gates.trace_hash
            == sha256(plan.trace_id.encode("utf-8")).hexdigest()
            and gates.plan_revision == plan.revision
            and gates.policy_hash == expected_policy_hash
            and policy.policy_hash == expected_policy_hash
            and policy.policy_hash == gates.policy_hash
        )

    def _restore_budget_from_events(
        self, plan: HarnessPlan, budget: WorkflowBudgetLedger
    ) -> None:
        # Durable remaining values, not a replay against a newly full ledger,
        # are the restart authority.  Parse them here so corrupt projections
        # fail before any runtime policy can mutate the fresh ledger.
        self._budget_state(self.event_store.load(plan.run_id), plan, budget.snapshot())

    @staticmethod
    def _candidate(candidate: object) -> tuple[_AdvanceCandidate | None, str | None]:
        if candidate is None:
            return _AdvanceCandidate(), None
        if type(candidate) is not dict:
            return None, None
        try:
            parsed = _AdvanceCandidate.model_validate(candidate)
        except Exception:
            return None, None
        if not candidate:
            return parsed, None
        canonical = json.dumps(
            parsed.model_dump(mode="json", exclude_none=True),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        )
        return parsed, sha256(canonical.encode("utf-8")).hexdigest()

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
                "policy": "degradation",
                "summary_source": "degradation",
            }
        if view.run_state is RunState.SUMMARIZING:
            source = self._summary_source(events, view)
            if source == "confidence_success":
                return RunState.SUCCEEDED, "completed", False, {
                    "policy": "terminal_success",
                    "summary_source": source,
                }
            return RunState.DEGRADED, "safe_no_trade_due_to_degradation", True, {
                "policy": "terminal_degradation",
                "summary_source": "degradation",
            }
        if view.run_state is not RunState.RUNNING:
            raise HarnessKernelError("run has no deterministic advance policy")

        payload: dict[str, object] = {"policy": "confidence_gate"}
        ledger = self._budgets[plan.run_id]
        budget = ledger.snapshot()
        if type(budget) is not BudgetSnapshot or budget.mode is not plan.mode:
            raise HarnessDependencyError("budget snapshot does not match the run")
        if self._budget_closed(events, plan, budget):
            return RunState.DEGRADING, "budget_exhausted", True, {
                "policy": "budget"
            }
        usage = UsageTokens(input_tokens=1, output_tokens=1)
        if not self._budget_can_reserve(events, plan, budget, usage):
            return RunState.DEGRADING, "budget_exhausted", True, {
                "policy": "budget"
            }
        before = budget
        try:
            reservation = ledger.reserve(
                node_name="event_filter",
                model="gpt-5.6-luna",
                band="short",
                usage=usage,
            )
            settlement = ledger.settle(reservation, usage)
        except BudgetExceededError:
            return RunState.DEGRADING, "budget_exhausted", True, {
                "policy": "budget"
            }
        budget = ledger.snapshot()
        payload.update(
            self._budget_payload(
                events,
                plan,
                before,
                budget,
                usage,
                reservation,
                settlement,
            )
        )

        guard: LoopGuard | None = None
        if candidate.action_observation is not None:
            guard = self._replayed_loop_guard(events, plan)
            loop_decision = guard.observe_action_result(candidate.action_observation)
            payload["loop_action_json"] = candidate.action_observation.model_dump_json()
            payload.update(self._loop_binding_payload(plan))
            payload["loop_reason"] = loop_decision.stop_reason or "allowed"
            if not loop_decision.allowed:
                return RunState.DEGRADING, "loop_guard_stopped", True, payload
        if candidate.loop_checkpoint is not None:
            if (
                candidate.loop_checkpoint.plan_revision != plan.revision
                or candidate.loop_checkpoint.fingerprint_schema_version
                != plan.pinned_versions.fingerprint_schema_version
            ):
                return RunState.DEGRADING, "loop_checkpoint_binding_mismatch", True, {
                    **payload,
                    "policy": "loop_guard",
                }
            guard = guard or self._replayed_loop_guard(events, plan)
            loop_decision = guard.observe_checkpoint(candidate.loop_checkpoint)
            payload["loop_checkpoint_json"] = candidate.loop_checkpoint.model_dump_json()
            payload.update(self._loop_binding_payload(plan))
            payload["loop_reason"] = loop_decision.stop_reason or "allowed"
            if not loop_decision.allowed:
                return RunState.DEGRADING, "loop_guard_stopped", True, payload

        if plan.run_id in self._confidence_binding_failed:
            return RunState.DEGRADING, "confidence_context_mismatch", True, {
                **payload,
                "policy": "confidence_binding",
            }
        gate = self._confidence_gates[plan.run_id]
        confidence = gate.evaluate(
            candidate.confidence_observation, candidate.confidence_artifact
        )
        payload["confidence_action"] = confidence.next_action
        payload["confidence_reason"] = confidence.reason_code
        if confidence.may_succeed:
            payload["summary_source"] = "confidence_success"
            return RunState.SUMMARIZING, "confidence_sufficient", False, payload
        return RunState.DEGRADING, "confidence_fail_closed", True, payload

    def _replayed_loop_guard(
        self, events: tuple[HarnessEvent, ...], plan: HarnessPlan
    ) -> LoopGuard:
        guard = self._loop_guard_factory()
        if type(guard) is not LoopGuard:
            raise HarnessDependencyError("loop guard factory returned an invalid policy")
        for event in events:
            action_value = event.payload.get("loop_action_json")
            checkpoint_value = event.payload.get("loop_checkpoint_json")
            if action_value is not None or checkpoint_value is not None:
                if (
                    event.payload.get("loop_plan_revision") != plan.revision
                    or event.payload.get("loop_fingerprint_schema_version")
                    != plan.pinned_versions.fingerprint_schema_version
                    or event.payload.get("loop_policy_version")
                    != plan.pinned_versions.policy_version
                ):
                    raise HarnessDependencyError(
                        "committed loop observation pins do not match the plan"
                    )
            if action_value is not None:
                if type(action_value) is not str:
                    raise HarnessDependencyError("committed loop action is invalid")
                try:
                    action = ActionObservationFingerprint.model_validate_json(
                        action_value
                    )
                except Exception as error:
                    raise HarnessDependencyError("committed loop action is invalid") from error
                guard.observe_action_result(action)
            if checkpoint_value is not None:
                if type(checkpoint_value) is not str:
                    raise HarnessDependencyError("committed loop checkpoint is invalid")
                try:
                    checkpoint = SemanticCheckpoint.model_validate_json(checkpoint_value)
                except Exception as error:
                    raise HarnessDependencyError("committed loop checkpoint is invalid") from error
                guard.observe_checkpoint(checkpoint)
        return guard

    @staticmethod
    def _loop_binding_payload(plan: HarnessPlan) -> dict[str, object]:
        return {
            "loop_plan_revision": plan.revision,
            "loop_fingerprint_schema_version": (
                plan.pinned_versions.fingerprint_schema_version
            ),
            "loop_policy_version": plan.pinned_versions.policy_version,
        }

    @staticmethod
    def _strict_payload_bool(payload: dict[str, object], name: str) -> bool:
        value = payload.get(name, False)
        if type(value) is not bool:
            raise HarnessDependencyError(f"committed {name} decision is invalid")
        return value

    @staticmethod
    def _pending_authority(
        events: tuple[HarnessEvent, ...], view: HarnessSessionView
    ) -> tuple[TransitionAuthorityRecord, dict[str, object]] | None:
        pending = tuple(
            authority
            for authority in view.transition_authorities
            if authority.idempotency_key not in view.applied_idempotency_keys
            and authority.expected_state_revision == view.state_revision
        )
        if not pending:
            return None
        if len(pending) != 1:
            raise HarnessDependencyError("run has ambiguous pending transition authority")
        authority = pending[0]
        matching = tuple(
            event
            for event in events
            if event.transition_authority == authority
        )
        if len(matching) != 1:
            raise HarnessDependencyError("pending authority has no unique policy event")
        current = view.run_state.value if view.run_state is not None else "none"
        try:
            RunState(authority.to_state)
        except ValueError as error:
            raise HarnessDependencyError("pending authority targets an invalid state") from error
        if (
            authority.entity_kind != "run"
            or authority.entity_id != view.run_id
            or authority.from_state != current
        ):
            raise HarnessDependencyError("pending authority does not bind current run")
        return authority, dict(matching[0].payload)

    @staticmethod
    def _committed_candidate(
        events: tuple[HarnessEvent, ...], candidate_digest: str
    ) -> tuple[HarnessTransition, dict[str, object]] | None:
        applied = {
            event.transition.idempotency_key: event.transition
            for event in events
            if event.transition is not None
        }
        matches: list[tuple[HarnessTransition, dict[str, object]]] = []
        for event in events:
            authority = event.transition_authority
            if (
                authority is None
                or event.payload.get("candidate_digest") != candidate_digest
                or authority.idempotency_key not in applied
            ):
                continue
            matches.append((applied[authority.idempotency_key], dict(event.payload)))
        if len(matches) > 1:
            raise HarnessDependencyError("candidate digest identifies multiple decisions")
        return matches[0] if matches else None

    @staticmethod
    def _summary_source(
        events: tuple[HarnessEvent, ...], view: HarnessSessionView
    ) -> str:
        for event in reversed(events):
            transition = event.transition
            if (
                transition is not None
                and transition.entity_kind == "run"
                and transition.to_state == RunState.SUMMARIZING.value
                and transition.expected_state_revision + 1 == view.state_revision
            ):
                source = event.payload.get("summary_source")
                if source in {"confidence_success", "degradation"}:
                    return cast(str, source)
                raise HarnessDependencyError("summarizing transition has no valid source")
        raise HarnessDependencyError("summarizing state has no source transition")

    @staticmethod
    def _budget_payload(
        events: tuple[HarnessEvent, ...],
        plan: HarnessPlan,
        before: BudgetSnapshot,
        after: BudgetSnapshot,
        usage: UsageTokens,
        reservation: BudgetReservation,
        settlement: BudgetSettlement,
    ) -> dict[str, object]:
        attempts, seconds, cost, tokens = HarnessKernel._budget_state(
            events, plan, before
        )
        attempt_delta = before.remaining_attempts - after.remaining_attempts
        seconds_delta = Decimal(str(before.remaining_seconds)) - Decimal(
            str(after.remaining_seconds)
        )
        cost_delta = before.remaining_cost - after.remaining_cost
        token_delta = usage.input_tokens + usage.cache_write_tokens + usage.output_tokens
        if attempt_delta < 0 or seconds_delta < 0 or cost_delta < 0 or token_delta < 0:
            raise HarnessDependencyError("runtime budget counters moved backwards")
        remaining_attempts = attempts - attempt_delta
        remaining_seconds = seconds - seconds_delta
        remaining_cost = cost - cost_delta
        remaining_tokens = tokens - token_delta
        if min(remaining_attempts, remaining_tokens) < 0 or min(
            remaining_seconds, remaining_cost
        ) < 0:
            raise HarnessDependencyError("runtime budget exceeded durable authority")
        projection: dict[str, object] = {
            "budget_before_attempts": attempts,
            "budget_delta_attempts": attempt_delta,
            "budget_remaining_attempts": remaining_attempts,
            "budget_before_seconds": format(seconds, "f"),
            "budget_delta_seconds": format(seconds_delta, "f"),
            "budget_remaining_seconds": format(remaining_seconds, "f"),
            "budget_before_cost": format(cost, "f"),
            "budget_delta_cost": format(cost_delta, "f"),
            "budget_remaining_cost": format(remaining_cost, "f"),
            "budget_before_tokens": tokens,
            "budget_delta_tokens": token_delta,
            "budget_remaining_tokens": remaining_tokens,
        }
        if (
            settlement.reservation_id != reservation.reservation_id
            or settlement.timeout
            or settlement.charged_cost != cost_delta
            or attempt_delta != 1
        ):
            raise HarnessDependencyError("budget settlement does not bind runtime usage")
        settlement_core: dict[str, object] = {
            "schema_version": _BUDGET_SETTLEMENT_SCHEMA,
            "reservation_id": reservation.reservation_id,
            "charged_cost": format(settlement.charged_cost, "f"),
            "timeout": settlement.timeout,
            "usage": {
                "input_tokens": usage.input_tokens,
                "cached_input_tokens": usage.cached_input_tokens,
                "cache_write_tokens": usage.cache_write_tokens,
                "output_tokens": usage.output_tokens,
                "web_search_tool_calls": usage.web_search_tool_calls,
            },
            "projection": projection,
        }
        settlement_evidence = {
            **settlement_core,
            "settlement_digest": HarnessKernel._canonical_digest(settlement_core),
        }
        return {**projection, _BUDGET_SETTLEMENT_EVIDENCE_KEY: settlement_evidence}

    @staticmethod
    def _canonical_digest(value: object) -> str:
        return sha256(
            json.dumps(
                value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _budget_decimal(value: object) -> Decimal:
        if type(value) is not str or not value or len(value) > 80:
            raise HarnessDependencyError("committed budget projection is invalid")
        try:
            parsed = Decimal(value)
        except Exception as error:
            raise HarnessDependencyError("committed budget projection is invalid") from error
        if (
            not parsed.is_finite()
            or parsed < 0
            or "e" in value.casefold()
            or value.startswith(("+", "-"))
            or format(parsed, "f") != value
        ):
            raise HarnessDependencyError("committed budget projection is invalid")
        return parsed

    @staticmethod
    def _validated_budget_projection(
        event: HarnessEvent,
        plan: HarnessPlan,
        current: tuple[int, Decimal, Decimal, int],
        prior_events: tuple[HarnessEvent, ...],
    ) -> tuple[int, Decimal, Decimal, int]:
        budget_keys = frozenset(key for key in event.payload if key in _BUDGET_FIELDS)
        unknown_budget_keys = frozenset(
            key
            for key in event.payload
            if str(key).startswith("budget_")
            and key not in _BUDGET_FIELDS
            and key != _BUDGET_SETTLEMENT_EVIDENCE_KEY
        )
        if (
            budget_keys != _BUDGET_FIELDS
            or unknown_budget_keys
            or event.event_type != "transition_authorized"
        ):
            raise HarnessDependencyError("committed budget projection is not canonical")
        authority = event.transition_authority
        expected_target = (
            _BUDGET_AUTHORITY_TARGETS.get(authority.reason_code)
            if authority is not None
            else None
        )
        if (
            authority is None
            or expected_target is None
            or event.run_id != plan.run_id
            or event.trace_id != plan.trace_id
            or event.sequence <= 0
            or event.state_revision != authority.expected_state_revision
            or authority.run_id != plan.run_id
            or authority.trace_id != plan.trace_id
            or authority.entity_kind != "run"
            or authority.entity_id != plan.run_id
            or authority.plan_revision != plan.revision
            or authority.from_state != RunState.RUNNING.value
            or authority.to_state != expected_target.value
            or authority.idempotency_key
            != f"run-{authority.expected_state_revision}-{expected_target.value}"
            or event.payload.get("reason_code") != authority.reason_code
        ):
            raise HarnessDependencyError("committed budget authority binding is invalid")
        integers: dict[str, int] = {}
        for field in _BUDGET_INTEGER_FIELDS:
            value = event.payload.get(field)
            if type(value) is not int or value < 0:
                raise HarnessDependencyError("committed budget projection is invalid")
            integers[field] = value
        decimals = {
            field: HarnessKernel._budget_decimal(event.payload.get(field))
            for field in _BUDGET_DECIMAL_FIELDS
        }
        before = (
            integers["budget_before_attempts"],
            decimals["budget_before_seconds"],
            decimals["budget_before_cost"],
            integers["budget_before_tokens"],
        )
        delta = (
            integers["budget_delta_attempts"],
            decimals["budget_delta_seconds"],
            decimals["budget_delta_cost"],
            integers["budget_delta_tokens"],
        )
        after = (
            integers["budget_remaining_attempts"],
            decimals["budget_remaining_seconds"],
            decimals["budget_remaining_cost"],
            integers["budget_remaining_tokens"],
        )
        if before != current or tuple(
            before[index] - delta[index] for index in range(4)
        ) != after:
            raise HarnessDependencyError("committed budget projection breaks monotonic chain")
        HarnessKernel._validated_budget_settlement_evidence(
            event, plan, prior_events=prior_events, projection={
                "budget_before_attempts": before[0],
                "budget_delta_attempts": delta[0],
                "budget_remaining_attempts": after[0],
                "budget_before_seconds": format(before[1], "f"),
                "budget_delta_seconds": format(delta[1], "f"),
                "budget_remaining_seconds": format(after[1], "f"),
                "budget_before_cost": format(before[2], "f"),
                "budget_delta_cost": format(delta[2], "f"),
                "budget_remaining_cost": format(after[2], "f"),
                "budget_before_tokens": before[3],
                "budget_delta_tokens": delta[3],
                "budget_remaining_tokens": after[3],
            },
        )
        return after

    @staticmethod
    def _validated_budget_settlement_evidence(
        event: HarnessEvent,
        plan: HarnessPlan,
        *,
        prior_events: tuple[HarnessEvent, ...],
        projection: dict[str, object],
    ) -> None:
        """Require a durable host snapshot and exact settlement facts.

        Budget numbers in an authority event are projections, not authority by
        themselves.  This companion evidence binds each projection to the
        immediately preceding issuer snapshot, an actual reservation, and the
        fixed Phase-1 usage contract.  It is intentionally checked during
        replay, before any backend registration/resume can occur.
        """
        evidence = event.payload.get(_BUDGET_SETTLEMENT_EVIDENCE_KEY)
        if not isinstance(evidence, Mapping):
            raise HarnessDependencyError("budget settlement evidence is missing")
        expected_keys = {
            "settlement",
            "settlement_event_hash",
            "settlement_event_sequence",
            "host_receipt",
            "binding_digest",
        }
        if set(evidence) != expected_keys:
            raise HarnessDependencyError("budget settlement evidence is not canonical")
        digest = evidence.get("binding_digest")
        unsigned = {key: value for key, value in evidence.items() if key != "binding_digest"}
        if (
            type(digest) is not str
            or len(digest) != 64
            or digest != HarnessKernel._canonical_digest(unsigned)
        ):
            raise HarnessDependencyError("budget settlement evidence digest is invalid")
        settlement = evidence.get("settlement")
        if not isinstance(settlement, Mapping):
            raise HarnessDependencyError("budget settlement evidence is invalid")
        expected_settlement_keys = {
            "schema_version",
            "reservation_id",
            "charged_cost",
            "timeout",
            "usage",
            "projection",
            "settlement_digest",
        }
        if set(settlement) != expected_settlement_keys:
            raise HarnessDependencyError("budget settlement evidence is not canonical")
        settlement_unsigned = {
            key: value for key, value in settlement.items() if key != "settlement_digest"
        }
        if (
            settlement.get("schema_version") != _BUDGET_SETTLEMENT_SCHEMA
            or settlement.get("projection") != projection
            or settlement.get("settlement_digest")
            != HarnessKernel._canonical_digest(settlement_unsigned)
        ):
            raise HarnessDependencyError("budget settlement evidence digest is invalid")
        reservation_id = settlement.get("reservation_id")
        if (
            type(reservation_id) is not str
            or not reservation_id
            or reservation_id != reservation_id.strip()
            or len(reservation_id) > 256
            or settlement.get("timeout") is not False
        ):
            raise HarnessDependencyError("budget settlement evidence is invalid")
        usage_value = settlement.get("usage")
        if not isinstance(usage_value, Mapping) or set(usage_value) != {
            "input_tokens",
            "cached_input_tokens",
            "cache_write_tokens",
            "output_tokens",
            "web_search_tool_calls",
        }:
            raise HarnessDependencyError("budget settlement usage is invalid")
        try:
            usage = UsageTokens(**usage_value)
        except Exception as error:
            raise HarnessDependencyError("budget settlement usage is invalid") from error
        # Phase 1 performs precisely one bounded event-filter attempt.  This
        # rejects self-consistent all-zero projections and any synthetic usage
        # that cannot have been settled by the runtime.
        expected_usage = UsageTokens(input_tokens=1, output_tokens=1)
        if usage != expected_usage or projection["budget_delta_attempts"] != 1:
            raise HarnessDependencyError("budget settlement has no real attempt")
        expected_tokens = usage.input_tokens + usage.cache_write_tokens + usage.output_tokens
        if projection["budget_delta_tokens"] != expected_tokens:
            raise HarnessDependencyError("budget settlement token total is inconsistent")
        charged_cost = HarnessKernel._budget_decimal(settlement.get("charged_cost"))
        expected_cost = estimate_workflow_usage_cost("gpt-5.6-luna", "short", usage)
        if charged_cost != expected_cost or charged_cost != Decimal(
            cast(str, projection["budget_delta_cost"])
        ):
            raise HarnessDependencyError("budget settlement cost is inconsistent")
        receipt_json = evidence.get("host_receipt")
        if (
            type(receipt_json) is not str
            or not receipt_json
            or len(receipt_json) > 262_144
        ):
            raise HarnessDependencyError("budget settlement receipt is invalid")
        try:
            receipt_value = json.loads(receipt_json)
            if (
                json.dumps(
                    receipt_value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
                )
                != receipt_json
            ):
                raise ValueError("receipt JSON is noncanonical")
            pre = CommittedExecutionSnapshot.model_validate_json(receipt_json)
        except Exception as error:
            raise HarnessDependencyError("budget settlement receipt is invalid") from error
        if not verify_committed_execution_snapshot(pre):
            raise HarnessDependencyError("budget settlement receipt is not host verified")
        try:
            receipt_view = fold_events(prior_events)
        except Exception as error:
            raise HarnessDependencyError("budget settlement receipt cannot fold authority") from error
        if (
            pre.run_id != plan.run_id
            or pre.trace_id != plan.trace_id
            or pre.plan_id != plan.plan_id
            or pre.plan_digest != canonical_plan_digest(plan)
            or pre.plan_revision != plan.revision
            or pre.sequence != event.sequence - 1
            or pre.state_revision != event.state_revision
            or pre.view_digest != canonical_view_digest(receipt_view)
            or pre.event_head_hash != event.previous_event_hash
            or evidence.get("settlement_event_hash") != event.previous_event_hash
            or evidence.get("settlement_event_sequence") != event.sequence - 1
        ):
            raise HarnessDependencyError("budget settlement receipt does not bind authority")
        if not prior_events:
            raise HarnessDependencyError("budget settlement record is missing")
        settlement_event = prior_events[-1]
        if (
            settlement_event.event_type != "budget_settlement_recorded"
            or settlement_event.sequence != event.sequence - 1
            or settlement_event.state_revision != event.state_revision
            or settlement_event.event_hash != event.previous_event_hash
            or settlement_event.payload != {"budget_settlement": settlement}
        ):
            raise HarnessDependencyError("budget settlement record does not bind authority")

    @staticmethod
    def _budget_state(
        events: tuple[HarnessEvent, ...],
        plan: HarnessPlan,
        snapshot: BudgetSnapshot,
    ) -> tuple[int, Decimal, Decimal, int]:
        if (
            type(snapshot.remaining_attempts) is not int
            or type(snapshot.remaining_seconds) is not float
            or type(snapshot.remaining_cost) is not Decimal
            or not math.isfinite(snapshot.remaining_seconds)
            or snapshot.remaining_attempts < 0
            or snapshot.remaining_seconds < 0
            or not snapshot.remaining_cost.is_finite()
            or snapshot.remaining_cost < 0
        ):
            raise HarnessDependencyError("initial budget caps are invalid")
        initial = WorkflowBudgetLedger(plan.mode, clock=lambda: 0.0).snapshot()
        if (
            snapshot.remaining_attempts > initial.remaining_attempts
            or snapshot.remaining_seconds > initial.remaining_seconds
            or snapshot.remaining_cost > initial.remaining_cost
        ):
            raise HarnessDependencyError("runtime budget exceeds initial caps")
        attempts = initial.remaining_attempts
        seconds = Decimal(str(initial.remaining_seconds))
        cost = initial.remaining_cost
        tokens = sum(worker.context_token_budget for worker in plan.workers)
        current = (attempts, seconds, cost, tokens)
        for index, event in enumerate(events):
            budget_keys = frozenset(key for key in event.payload if key in _BUDGET_FIELDS)
            if not budget_keys:
                continue
            current = HarnessKernel._validated_budget_projection(
                event, plan, current, events[:index]
            )
        return current

    @staticmethod
    def _budget_closed(
        events: tuple[HarnessEvent, ...], plan: HarnessPlan, snapshot: BudgetSnapshot
    ) -> bool:
        attempts, seconds, cost, tokens = HarnessKernel._budget_state(
            events, plan, snapshot
        )
        node_closed = any(
            node.exhausted
            or node.overdrawn
            or node.remaining_attempts <= 0
            or node.remaining_seconds <= 0
            or node.remaining_cost <= 0
            for node in snapshot.nodes
        )
        return (
            snapshot.exhausted
            or snapshot.overdrawn
            or attempts <= 0
            or seconds <= Decimal("0")
            or cost <= 0
            or tokens <= 0
            or node_closed
        )

    @staticmethod
    def _budget_can_reserve(
        events: tuple[HarnessEvent, ...],
        plan: HarnessPlan,
        snapshot: BudgetSnapshot,
        usage: UsageTokens,
    ) -> bool:
        attempts, seconds, cost, tokens = HarnessKernel._budget_state(
            events, plan, snapshot
        )
        policy = policy_for("event_filter")
        required_cost = estimate_workflow_usage_cost(
            "gpt-5.6-luna", "short", usage
        )
        required_tokens = (
            usage.input_tokens + usage.cache_write_tokens + usage.output_tokens
        )
        return (
            attempts >= 1
            and seconds >= Decimal(str(policy.attempt_timeout_seconds))
            and cost >= required_cost
            and tokens >= required_tokens
        )

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

        authority_view = view
        if (
            not authority_already_committed
            and _BUDGET_SETTLEMENT_EVIDENCE_KEY in event_payload
        ):
            settlement = event_payload[_BUDGET_SETTLEMENT_EVIDENCE_KEY]
            settlement_event = self._event(
                plan,
                "budget_settlement_recorded",
                payload={"budget_settlement": settlement},
            )
            self.event_store.append(
                settlement_event,
                expected_sequence=view.sequence,
                expected_state_revision=view.state_revision,
            )
            authority_view = self.snapshot(plan.run_id)
            host_receipt = self._issued_snapshot(plan, authority_view)
            evidence_core = {
                "settlement": settlement,
                "settlement_event_hash": authority_view.last_event_hash,
                "settlement_event_sequence": authority_view.sequence,
                # Keep the full signed snapshot as canonical JSON text. Event
                # payloads freeze nested sequences as tuples, so a mapping
                # would no longer satisfy the JSON-only payload contract.
                "host_receipt": json.dumps(
                    host_receipt.model_dump(mode="json"),
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                ),
            }
            event_payload = {
                **event_payload,
                _BUDGET_SETTLEMENT_EVIDENCE_KEY: {
                    **evidence_core,
                    "binding_digest": self._canonical_digest(evidence_core),
                },
            }
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
                expected_sequence=authority_view.sequence,
                expected_state_revision=authority_view.state_revision,
            )
            pre_view = self.snapshot(plan.run_id)
        pre_snapshot = self._issued_snapshot(plan, pre_view)
        if backend_handle is not None:
            backend_handle = self._execution.resume(plan, pre_view, pre_snapshot)

        transition_event = self._event(
            plan,
            "transition_committed",
            payload={
                **{
                    key: value
                    for key, value in event_payload.items()
                    if key not in _BUDGET_FIELDS
                    and key != _BUDGET_SETTLEMENT_EVIDENCE_KEY
                },
                "reason_code": reason_code,
            },
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
            or snapshot.folded_view != view
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
            or snapshot.folded_view != view
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
        no_trade: bool | None = None,
        reconciliation_required: bool | None = None,
        backend_synchronized: bool,
    ) -> HarnessDecision:
        if view.run_state is None:
            raise HarnessDependencyError("decision requires an identified run state")
        if no_trade is None:
            no_trade = view.run_state in {
                RunState.DEGRADING,
                RunState.DEGRADED,
                RunState.WAITING_RECONCILIATION,
            } or (
                view.run_state is RunState.SUMMARIZING
                and reason != "confidence_sufficient"
            )
        if reconciliation_required is None:
            reconciliation_required = view.run_state is RunState.WAITING_RECONCILIATION
        return HarnessDecision(
            run_id=plan.run_id,
            trace_id=plan.trace_id,
            sequence=view.sequence,
            state_revision=view.state_revision,
            run_state=view.run_state,
            plan_revision=plan.revision,
            reason_code=reason,
            previous_run_state=(
                RunState(transition.from_state) if transition is not None else None
            ),
            transition=transition,
            retry_authorized=False,
            no_trade=no_trade,
            reconciliation_required=reconciliation_required,
            backend_synchronized=backend_synchronized,
        )
