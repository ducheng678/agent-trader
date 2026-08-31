from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal

import pytest

from market_agent.workflow_budget import BudgetSnapshot, WorkflowBudgetLedger
from market_agent.workflow_confidence_calibration import ConfidenceGate
from market_agent.workflow_contracts import WorkflowMode, WorkflowRequest
from market_agent.workflow_execution_backend import (
    CommittedExecutionSnapshot,
    CommittedTransitionReceipt,
    ExecutionHandle,
    ExecutionProjectionError,
    ExecutionRegistrationError,
    canonical_plan_digest,
    canonical_transition_digest,
    canonical_view_digest,
)
from market_agent.workflow_harness import HarnessDecision, HarnessKernel, RunHandle
from market_agent.workflow_harness_contracts import (
    HarnessPlan,
    HarnessTransition,
    OutcomeKind,
    PinnedVersions,
    RiskClass,
    RunState,
    StageSpec,
    TaskKind,
    TransitionAuthorityRecord,
    WorkerSpec,
)
from market_agent.workflow_loop_guard import LoopGuard, SeverityPolicy
from market_agent.workflow_plan_registry import (
    PlanCompiler,
    PlanTemplate,
    PlanTemplateRegistry,
)
from market_agent.workflow_session import HarnessEvent, SQLiteHarnessEventStore
from market_agent.workflow_state_machine import GlobalTaskStateMachine
from market_agent.workflow_worker_registry import WorkerRegistry


HASH = "a" * 64
SIGNATURE = "0" * 512
NOW = datetime(2026, 8, 31, tzinfo=timezone.utc)


def _pinned() -> PinnedVersions:
    return PinnedVersions(
        plan_template_version="templates-v1",
        policy_version="policy-v1",
        worker_registry_version="workers-v1",
        source_registry_version="sources-v1",
        prompt_bundle_hash=HASH,
        tool_registry_hash=HASH,
        output_schema_bundle_hash=HASH,
        fingerprint_schema_version="v1",
    )


def _request(**updates: object) -> WorkflowRequest:
    values: dict[str, object] = {
        "workflow_id": "run-1",
        "trace_id": "trace-1",
        "user_query": "summarize the current market",
        "trigger_reason": "api_request",
    }
    values.update(updates)
    return WorkflowRequest(**values)


def _compiler() -> PlanCompiler:
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
    template = PlanTemplate(
        template_id="passive-information-v1",
        version="templates-v1",
        mode=WorkflowMode.PASSIVE,
        task_kind=TaskKind.INFORMATIONAL,
        risk_class=RiskClass.INFORMATIONAL,
        stages=(stage,),
        worker_ids=(worker.worker_id,),
        work_item_id="information-work",
        work_item_stage_id=stage.stage_id,
        work_item_worker_id=worker.worker_id,
        objective="Produce a bounded informational answer.",
        progress_output_fields=("answer.summary",),
        progress_evidence_slots=("accepted-source",),
        source_coverage_weights=(("authoritative-source", 1.0),),
        risk_invariant_ids=("no-side-effects",),
        allows_side_effects=False,
    )
    return PlanCompiler(PlanTemplateRegistry((template,)), WorkerRegistry((worker,)))


class DeterministicClock:
    def utc_now(self) -> datetime:
        return NOW

    def monotonic(self) -> float:
        return 100.0


class DeterministicIds:
    def __init__(self) -> None:
        self._next = 0

    def new(self, purpose: str) -> str:
        self._next += 1
        return f"{purpose}-{self._next}"


class StoreBackedIssuer:
    def __init__(self, store: SQLiteHarnessEventStore) -> None:
        self.store = store
        self.snapshots: list[CommittedExecutionSnapshot] = []
        self.receipts: list[CommittedTransitionReceipt] = []

    def ready(self) -> bool:
        return True

    def _snapshot(self, plan: HarnessPlan, sequence: int | None = None) -> CommittedExecutionSnapshot:
        events = self.store.load(plan.run_id)
        if sequence is not None:
            events = events[:sequence]
        from market_agent.workflow_session import fold_events

        view = fold_events(events)
        snapshot = CommittedExecutionSnapshot(
            run_id=plan.run_id,
            trace_id=plan.trace_id,
            plan_id=plan.plan_id,
            plan_digest=canonical_plan_digest(plan),
            plan_revision=plan.revision,
            sequence=view.sequence,
            state_revision=view.state_revision,
            view_digest=canonical_view_digest(view),
            event_head_hash=view.last_event_hash,
            trust_key_id="test-host",
            signature=SIGNATURE,
        )
        self.snapshots.append(snapshot)
        return snapshot

    def issue_snapshot(self, plan: HarnessPlan) -> CommittedExecutionSnapshot:
        return self._snapshot(plan)

    def issue_transition_receipt(
        self, plan: HarnessPlan, transition: HarnessTransition, *, pre_sequence: int
    ) -> CommittedTransitionReceipt:
        receipt = CommittedTransitionReceipt(
            pre=self._snapshot(plan, pre_sequence),
            post=self._snapshot(plan),
            transition_digest=canonical_transition_digest(transition),
            trust_key_id="test-host",
            signature=SIGNATURE,
        )
        self.receipts.append(receipt)
        return receipt


class RecordingBackend:
    def __init__(
        self,
        *,
        fail_prepare: bool = False,
        fail_apply: bool = False,
        fail_resume_number: int | None = None,
    ) -> None:
        self.fail_prepare = fail_prepare
        self.fail_apply = fail_apply
        self.fail_resume_number = fail_resume_number
        self.resume_count = 0
        self.operations: list[str] = []
        self.last_receipt: CommittedTransitionReceipt | None = None

    def prepare_registration(self, plan: HarnessPlan, provisional_view: object) -> object:
        self.operations.append("prepare")
        if self.fail_prepare:
            raise ExecutionRegistrationError("backend unavailable")
        return (plan.run_id, "provisional")

    def rollback_registration(self, token: object) -> None:
        self.operations.append("rollback")

    @staticmethod
    def _handle(plan: HarnessPlan, view: object) -> ExecutionHandle:
        return ExecutionHandle(
            run_id=plan.run_id,
            trace_id=plan.trace_id,
            plan_id=plan.plan_id,
            plan_revision=plan.revision,
            state_revision=view.state_revision,
            routed_state=view.run_state.value if view.run_state is not None else None,
            cancelled=False,
        )

    def register(self, plan: HarnessPlan, view: object, snapshot: object) -> ExecutionHandle:
        self.operations.append("register")
        return self._handle(plan, view)

    def resume(
        self,
        plan: HarnessPlan,
        folded_view: object,
        committed_snapshot: object,
        *,
        disposable_checkpoint: object | None = None,
    ) -> ExecutionHandle:
        self.operations.append("resume")
        self.resume_count += 1
        if self.resume_count == self.fail_resume_number:
            raise ExecutionProjectionError("authority projection unavailable")
        return self._handle(plan, folded_view)

    def apply_committed_transition(
        self,
        handle: ExecutionHandle,
        transition: HarnessTransition,
        pre_view: object,
        post_view: object,
        receipt: CommittedTransitionReceipt,
    ) -> ExecutionHandle:
        self.operations.append("apply")
        self.last_receipt = receipt
        if self.fail_apply:
            raise ExecutionProjectionError("projection unavailable")
        return ExecutionHandle(
            run_id=handle.run_id,
            trace_id=handle.trace_id,
            plan_id=handle.plan_id,
            plan_revision=handle.plan_revision,
            state_revision=post_view.state_revision,
            routed_state=(
                post_view.run_state.value if post_view.run_state is not None else None
            ),
            cancelled=False,
        )

    def cancel(self, run_id: str) -> None:
        self.operations.append("cancel")


def _kernel(
    tmp_path,
    *,
    backend: RecordingBackend | None = None,
    budget: WorkflowBudgetLedger | None = None,
):
    store = SQLiteHarnessEventStore(tmp_path / "harness.sqlite", monotonic=lambda: 100.0)
    backend = backend or RecordingBackend()
    issuer = StoreBackedIssuer(store)
    kernel = HarnessKernel(
        event_store=store,
        state_machine=GlobalTaskStateMachine(),
        plan_compiler=_compiler(),
        pinned_versions=_pinned(),
        loop_guard_factory=lambda: LoopGuard(
            severity_policy=SeverityPolicy(policy_version="policy-v1")
        ),
        confidence_gate_factory=lambda plan: ConfidenceGate(),
        budget_factory=lambda mode: budget
        or WorkflowBudgetLedger(mode, clock=lambda: 100.0),
        execution_backend=backend,
        receipt_issuer=issuer,
        clock=DeterministicClock(),
        identifiers=DeterministicIds(),
    )
    return kernel, store, backend, issuer


def _advance_to_running(kernel: HarnessKernel, run_id: str) -> None:
    for _ in range(4):
        kernel.advance(run_id, candidate={})


def _append_waiting_reconciliation(store: SQLiteHarnessEventStore, run_id: str) -> None:
    view = store.snapshot(run_id)
    transition = HarnessTransition(
        run_id=run_id,
        trace_id=view.trace_id,
        entity_kind="run",
        entity_id=run_id,
        from_state=RunState.RUNNING.value,
        to_state=RunState.WAITING_RECONCILIATION.value,
        expected_state_revision=view.state_revision,
        plan_revision=view.plan_revision,
        reason_code="unknown_external_effect",
        idempotency_key=f"unknown-{view.state_revision}",
    )
    authority = TransitionAuthorityRecord(
        **transition.model_dump(
            mode="python", exclude={"schema_version", "lease_epoch", "fencing_token_digest"}
        ),
        dependency_versions=view.dependency_versions,
    )
    store.append(
        HarnessEvent(
            event_id="unknown-authority",
            trace_id=view.trace_id,
            span_id="unknown-span-1",
            run_id=run_id,
            event_type="transition_authorized",
            occurred_at=NOW,
            monotonic_offset=100.0,
            actor="host-policy",
            payload={"reason_code": "unknown_external_effect"},
            transition_authority=authority,
        ),
        expected_sequence=view.sequence,
        expected_state_revision=view.state_revision,
    )
    view = store.snapshot(run_id)
    store.append(
        HarnessEvent(
            event_id="unknown-transition",
            trace_id=view.trace_id,
            span_id="unknown-span-2",
            run_id=run_id,
            event_type="transition_committed",
            occurred_at=NOW,
            monotonic_offset=100.0,
            actor="harness-kernel",
            payload={"reason_code": "unknown_external_effect"},
            transition=transition,
        ),
        expected_sequence=view.sequence,
        expected_state_revision=view.state_revision,
    )


def test_create_publishes_only_after_all_dependencies_are_ready(tmp_path):
    backend = RecordingBackend(fail_prepare=True)
    kernel, store, _, _ = _kernel(tmp_path, backend=backend)

    with pytest.raises(ExecutionRegistrationError):
        kernel.create(_request())

    assert store.load("run-1") == ()


def test_create_is_passive_and_returns_frozen_strict_handle(tmp_path):
    kernel, store, backend, _ = _kernel(tmp_path)

    handle = kernel.create(_request(has_live_position=True, active_symbol="BTC"))

    assert type(handle) is RunHandle
    assert handle.run_state is RunState.CREATED
    assert handle.backend_synchronized is True
    assert store.snapshot(handle.run_id).run_state is RunState.CREATED
    assert len(store.load(handle.run_id)) == 1
    assert backend.operations == ["prepare", "register"]
    with pytest.raises(Exception):
        handle.run_id = "changed"


def test_model_payload_cannot_change_control_state(tmp_path):
    kernel, _, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())

    decision = kernel.advance(
        handle.run_id,
        candidate={
            "goto": "succeeded",
            "retry": True,
            "permission": True,
            "plan": {"allows_side_effects": True},
            "risk": "approved",
            "terminal": True,
        },
    )

    assert type(decision) is HarnessDecision
    assert decision.run_state is RunState.CREATED
    assert decision.retry_authorized is False
    assert decision.reason_code == "candidate_rejected"


def test_duplicate_or_stale_advance_does_not_commit_another_transition(tmp_path):
    kernel, store, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    first = kernel.advance(
        handle.run_id, candidate={}, expected_state_revision=handle.state_revision
    )
    sequence = store.snapshot(handle.run_id).sequence

    duplicate = kernel.advance(
        handle.run_id, candidate={}, expected_state_revision=handle.state_revision
    )

    assert first.run_state is RunState.ADMITTED
    assert duplicate.run_state is RunState.ADMITTED
    assert duplicate.reason_code == "stale_revision"
    assert duplicate.retry_authorized is False
    assert store.snapshot(handle.run_id).sequence == sequence


def test_advance_without_worker_candidate_uses_only_deterministic_policy(tmp_path):
    kernel, _, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())

    decision = kernel.advance(handle.run_id)

    assert decision.run_state is RunState.ADMITTED
    assert decision.retry_authorized is False


def test_backend_failure_after_commit_keeps_durable_truth_for_resume(tmp_path):
    backend = RecordingBackend(fail_apply=True)
    kernel, store, _, _ = _kernel(tmp_path, backend=backend)
    handle = kernel.create(_request())

    decision = kernel.advance(handle.run_id, candidate={})

    assert decision.run_state is RunState.ADMITTED
    assert decision.backend_synchronized is False
    assert store.snapshot(handle.run_id).run_state is RunState.ADMITTED
    backend.fail_apply = False
    resumed = kernel.resume(handle.run_id, disposable_checkpoint={"state": "succeeded"})
    assert resumed.run_state is RunState.ADMITTED
    assert resumed.backend_synchronized is True


def test_authority_append_survives_backend_failure_and_is_reused_on_replay(tmp_path):
    backend = RecordingBackend(fail_resume_number=2)
    kernel, store, _, _ = _kernel(tmp_path, backend=backend)
    handle = kernel.create(_request())

    with pytest.raises(ExecutionProjectionError):
        kernel.advance(handle.run_id, candidate={})
    authority_view = store.snapshot(handle.run_id)
    assert authority_view.run_state is RunState.CREATED
    assert len(authority_view.transition_authorities) == 1

    backend.fail_resume_number = None
    kernel.resume(handle.run_id)
    decision = kernel.advance(handle.run_id, candidate={})

    assert decision.run_state is RunState.ADMITTED
    assert len(store.snapshot(handle.run_id).transition_authorities) == 1


def test_resume_and_snapshot_replay_only_the_authoritative_stream(tmp_path):
    kernel, store, backend, issuer = _kernel(tmp_path)
    handle = kernel.create(_request())
    kernel.advance(handle.run_id, candidate={})
    expected = kernel.snapshot(handle.run_id)
    restarted = HarnessKernel(
        event_store=store,
        state_machine=GlobalTaskStateMachine(),
        plan_compiler=_compiler(),
        pinned_versions=_pinned(),
        loop_guard_factory=lambda: LoopGuard(
            severity_policy=SeverityPolicy(policy_version="policy-v1")
        ),
        confidence_gate_factory=lambda plan: ConfidenceGate(),
        budget_factory=lambda mode: WorkflowBudgetLedger(mode, clock=lambda: 100.0),
        execution_backend=backend,
        receipt_issuer=issuer,
        clock=DeterministicClock(),
        identifiers=DeterministicIds(),
    )

    resumed = restarted.resume(
        handle.run_id, disposable_checkpoint={"run_state": "succeeded"}
    )

    assert restarted.snapshot(handle.run_id) == expected
    assert resumed.run_state is expected.run_state
    assert resumed.sequence == expected.sequence


def test_cancel_unknown_order_records_intent_and_waits_for_reconciliation(tmp_path):
    kernel, store, backend, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    _append_waiting_reconciliation(store, handle.run_id)

    decision = kernel.cancel(handle.run_id, "user_requested")

    assert decision.run_state is RunState.WAITING_RECONCILIATION
    assert decision.reconciliation_required is True
    assert decision.retry_authorized is False
    assert store.snapshot(handle.run_id).run_state is RunState.WAITING_RECONCILIATION
    assert backend.operations[-1] != "cancel"


def test_unpinned_confidence_fails_closed_to_no_trade_degradation(tmp_path):
    kernel, _, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)

    decision = kernel.advance(handle.run_id, candidate={})

    assert decision.run_state is RunState.DEGRADING
    assert decision.retry_authorized is False
    assert decision.no_trade is True


def test_exhausted_budget_is_a_hard_no_trade_gate(tmp_path):
    budget = WorkflowBudgetLedger(WorkflowMode.PASSIVE, clock=lambda: 100.0)
    budget.snapshot = lambda: BudgetSnapshot(
        mode=WorkflowMode.PASSIVE,
        remaining_cost=Decimal("0"),
        reserved_cost=Decimal("0"),
        settled_cost=Decimal("0.30"),
        remaining_attempts=0,
        remaining_seconds=0.0,
        deadline_monotonic=100.0,
        nodes=(),
        exhausted=True,
        overdrawn=False,
    )
    kernel, _, _, _ = _kernel(tmp_path, budget=budget)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)

    decision = kernel.advance(handle.run_id)

    assert decision.run_state is RunState.DEGRADING
    assert decision.reason_code == "budget_exhausted"
    assert decision.no_trade is True


def test_receipts_come_from_injected_host_issuer_and_are_forwarded(tmp_path):
    kernel, _, backend, issuer = _kernel(tmp_path)
    handle = kernel.create(_request())

    kernel.advance(handle.run_id, candidate={})

    assert len(issuer.receipts) == 1
    assert backend.last_receipt == issuer.receipts[0]
    assert not any("private" in name or "sign" in name for name in vars(kernel))


def test_run_and_trace_identity_propagate_through_every_event(tmp_path):
    kernel, store, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    kernel.advance(handle.run_id, candidate={})

    events = store.load(handle.run_id)

    assert {event.run_id for event in events} == {handle.run_id}
    assert {event.trace_id for event in events} == {handle.trace_id}
    assert all(
        event.transition is None or event.transition.trace_id == handle.trace_id
        for event in events
    )
