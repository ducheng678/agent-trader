from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from hashlib import sha256
from concurrent.futures import ThreadPoolExecutor
import gc
import json

import pytest
from pydantic import ValidationError

from market_agent.workflow_budget import BudgetSnapshot, WorkflowBudgetLedger
from market_agent.workflow_confidence_calibration import (
    FEATURE_ORDER,
    AcceptedEvidenceRecord,
    ConfidenceCalibratorArtifact,
    ConfidenceFeatureSpec,
    ConfidenceFoldedState,
    ConfidenceGate,
    ConfidenceObservation,
    ConfidenceTargetSnapshot,
    ConflictRecord,
    HardGateSnapshot,
    SourceRegistryRecord,
    TrustedConfidencePolicy,
    TrustedRequestContext,
    artifact_payload,
    confidence_snapshot_hashes,
)
from market_agent.workflow_contracts import (
    Action,
    AgentReport,
    AgentTask,
    CoordinatorPlan,
    DecisionDraft,
    KnowledgeStatus,
    ModelTier,
    ReportStatus,
    SummaryCompleteness,
    TaskDifficulty,
    TaskType,
    WorkflowMode,
    WorkflowRequest,
)
from market_agent.workflow_agent_contracts import AgentUsage, ModelTier as DriverModelTier
from market_agent.workflow_agent_driver import AgentDriver, ModelResponse
from market_agent.workflow_agents.common import profile_for
from market_agent.workflow_capabilities import CapabilityIssuer
from market_agent.workflow_circuit_breaker import CircuitBreaker
from market_agent.workflow_coordinator_services import AgentCoordinatorServices
from market_agent.workflow_fallback import FallbackPolicy
from market_agent.workflow_graph import CoordinatedWorkflow, WorkflowServices
from market_agent.workflow_prompt_release import PromptReleaseRegistry, canonical_json
from market_agent.workflow_reflection_agent import (
    ObjectiveCheck,
    reflection_output_schema,
    reflection_release,
)
from market_agent.workflow_retry_policy import RetryPolicy
from market_agent.openai_usage import UsageTokens as PricedUsageTokens, estimate_workflow_usage_cost
from market_agent.workflow_execution_backend import (
    CommittedExecutionSnapshot,
    CommittedTransitionReceipt,
    ExecutionHandle,
    ExecutionProjectionError,
    ExecutionRegistrationError,
    IssuerTrustDescriptor,
    RegistrationPreparation,
    LangGraphExecutionBackend,
    canonical_plan_digest,
    canonical_transition_digest,
    canonical_view_digest,
)
import market_agent.workflow_execution_backend as execution_backend_module
import market_agent.workflow_harness as harness_module
from market_agent.workflow_harness import (
    HarnessDecision,
    HarnessDependencyError,
    HarnessKernel,
    RunHandle,
)
from market_agent.workflow_harness_contracts import (
    HarnessPlan,
    HarnessSessionView,
    HarnessTransition,
    OutcomeKind,
    PinnedVersions,
    ProgressVector,
    RiskClass,
    RunState,
    StageSpec,
    TaskKind,
    TransitionAuthorityRecord,
    WorkerSpec,
)
from market_agent.workflow_loop_guard import (
    ActionObservationFingerprint,
    LoopGuard,
    LoopScope,
    ObservationKind,
    SemanticCheckpoint,
    SeverityPolicy,
    build_action_fingerprint,
    build_result_fingerprint,
    build_state_fingerprint,
)
from market_agent.workflow_observation import (
    AttemptUsage,
    CheckpointDecision,
    CoreNodeName,
    ExecutionObservationCollector,
    NodeOutcome,
    ObservedWorkItem,
    TaskRetryState,
    TokenUsage,
)
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


_HARNESS_RECEIPT_SIGNER = None
_HARNESS_RECEIPT_KEYS = None


@pytest.fixture(autouse=True)
def _pin_signed_harness_receipts(monkeypatch):
    """Give the test-only host issuer a key matching the public backend pin."""
    global _HARNESS_RECEIPT_SIGNER, _HARNESS_RECEIPT_KEYS
    if _HARNESS_RECEIPT_SIGNER is None:
        from market_agent_test_bundle.tests.test_workflow_execution_backend import (
            TrustedReceiptVerifier,
            _generate_test_rsa_key,
        )

        modulus, exponent, private = _generate_test_rsa_key()
        _HARNESS_RECEIPT_KEYS = {"test-host-rsa": (modulus, exponent)}
        _HARNESS_RECEIPT_SIGNER = TrustedReceiptVerifier(
            modulus=modulus, private_exponent=private, key_id="test-host-rsa"
        )
    monkeypatch.setattr(
        execution_backend_module, "_PINNED_PUBLIC_KEYS", _HARNESS_RECEIPT_KEYS
    )
    monkeypatch.setattr(
        execution_backend_module,
        "_EXPECTED_TRUST_CONFIG_DIGEST",
        execution_backend_module._trust_config_digest(_HARNESS_RECEIPT_KEYS),
    )


class ConfidenceVerifier:
    def verify(self, key_id: str, payload: bytes, signature: str) -> bool:
        return key_id == "host-key" and signature == sha256(
            b"host-secret" + payload
        ).hexdigest()


def _confidence_material(
    plan: HarnessPlan,
) -> tuple[ConfidenceGate, ConfidenceObservation, ConfidenceCalibratorArtifact]:
    targets = ConfidenceTargetSnapshot(
        required_dependency_ids=("collect",),
        required_output_field_paths=("result.summary",),
        required_evidence_slot_ids=("primary",),
        required_source_ids=("official",),
        known_conflict_slot_ids=("claim",),
        risk_invariant_ids=("safe",),
    )
    values = {
        "applicability_domain": "market",
        "targets": targets,
        "accepted_evidence": (
            AcceptedEvidenceRecord(
                evidence_id="evidence",
                source_id="official",
                required_slot_id="primary",
                provenance_hash="b" * 64,
                accepted_by_host=True,
            ),
        ),
        "conflicts": (
            ConflictRecord(
                conflict_id="claim",
                evidence_ids=("evidence",),
                resolved=True,
                provenance_hash="c" * 64,
            ),
        ),
        "source_registry": (
            SourceRegistryRecord(
                source_id="official", registry_hash="d" * 64, enabled=True
            ),
        ),
        "folded_state": ConfidenceFoldedState(
            completed_dependency_ids=("collect",),
            valid_output_field_paths=("result.summary",),
            satisfied_risk_invariant_ids=("safe",),
            event_fold_hash="e" * 64,
        ),
        "accepted_record_snapshot_hash": "f" * 64,
        "provenance_snapshot_hash": "1" * 64,
    }
    raw = ConfidenceObservation.model_construct(**values)
    accepted_hash, provenance_hash = confidence_snapshot_hashes(raw)
    values.update(
        accepted_record_snapshot_hash=accepted_hash,
        provenance_snapshot_hash=provenance_hash,
    )
    observation = ConfidenceObservation(**values)
    policy_hash = sha256(plan.pinned_versions.policy_version.encode()).hexdigest()
    artifact_values = {
        "artifact_id": "cal-v1",
        "artifact_version": "v1",
        "schema_hash": HASH,
        "policy_hash": policy_hash,
        "dataset_hash": "c" * 64,
        "applicability_domains": ("market",),
        "feature_specs": tuple(
            ConfidenceFeatureSpec(feature_name=name, coefficient=Decimal(value))
            for name, value in zip(FEATURE_ORDER, (".4", ".3", ".15"), strict=True)
        ),
        "intercept": Decimal("0"),
        "issued_epoch": 10,
        "expires_epoch": 20,
        "key_id": "host-key",
        "artifact_hash": "d" * 64,
        "signature": "0" * 64,
    }
    unsigned = ConfidenceCalibratorArtifact.model_construct(**artifact_values)
    artifact_values["signature"] = sha256(
        b"host-secret" + artifact_payload(unsigned)
    ).hexdigest()
    artifact = ConfidenceCalibratorArtifact(**artifact_values)
    policy = TrustedConfidencePolicy(
        artifact_id=artifact.artifact_id,
        artifact_version=artifact.artifact_version,
        artifact_hash=artifact.artifact_hash,
        schema_hash=artifact.schema_hash,
        policy_hash=artifact.policy_hash,
        dataset_hash=artifact.dataset_hash,
        key_id=artifact.key_id,
        applicability_domain="market",
        accepted_record_snapshot_hash=accepted_hash,
        provenance_snapshot_hash=provenance_hash,
        issued_epoch=artifact.issued_epoch,
        expires_epoch=artifact.expires_epoch,
    )
    context = TrustedRequestContext(
        request_class="informational",
        evaluation_epoch=15,
        recovery_used=False,
        hard_gates=HardGateSnapshot(
            permission=True,
            risk=True,
            budget=True,
            loop=True,
            evidence=True,
            audit_integrity=True,
            run_id=plan.run_id,
            trace_hash=sha256(plan.trace_id.encode()).hexdigest(),
            plan_revision=plan.revision,
            policy_hash=policy_hash,
        ),
    )
    return (
        ConfidenceGate(
            trusted_policy=policy,
            signature_verifier=ConfidenceVerifier(),
            request_context=context,
        ),
        observation,
        artifact,
    )


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
        # This unit-test compiler intentionally declares only the passive
        # informational template. Active admission is covered by the plan
        # registry tests with a complete active registry.
        "trigger_reason": "passive_event_trigger",
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
    def __init__(self, start: int = 0) -> None:
        self._next = start

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

    def trust_descriptor(self) -> IssuerTrustDescriptor:
        return IssuerTrustDescriptor(
            trust_version=execution_backend_module._TRUST_VERSION,
            trust_config_digest=execution_backend_module._EXPECTED_TRUST_CONFIG_DIGEST,
            key_id=_HARNESS_RECEIPT_SIGNER.key_id,
        )

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
            folded_view=view,
            trust_key_id=_HARNESS_RECEIPT_SIGNER.key_id,
            signature=SIGNATURE,
        )
        snapshot = _HARNESS_RECEIPT_SIGNER.approve(snapshot)
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
            trust_key_id=_HARNESS_RECEIPT_SIGNER.key_id,
            signature=SIGNATURE,
        )
        receipt = _HARNESS_RECEIPT_SIGNER.approve(receipt)
        self.receipts.append(receipt)
        return receipt


class RecordingBackend:
    def __init__(
        self,
        *,
        fail_prepare: bool = False,
        fail_commit: bool = False,
        fail_apply: bool = False,
        fail_resume_number: int | None = None,
    ) -> None:
        self.fail_prepare = fail_prepare
        self.fail_commit = fail_commit
        self.fail_apply = fail_apply
        self.fail_resume_number = fail_resume_number
        self.resume_count = 0
        self.operations: list[str] = []
        self.last_receipt: CommittedTransitionReceipt | None = None

    def prepare_registration(
        self,
        plan: HarnessPlan,
        provisional_view: object,
        issuer_trust_descriptor: IssuerTrustDescriptor,
    ) -> RegistrationPreparation:
        self.operations.append("prepare")
        if self.fail_prepare:
            raise ExecutionRegistrationError("backend unavailable")
        return RegistrationPreparation(
            token_id=HASH,
            run_id=plan.run_id,
            trace_id=plan.trace_id,
            plan_id=plan.plan_id,
            plan_digest=canonical_plan_digest(plan),
            plan_revision=plan.revision,
            provisional_view_digest=canonical_view_digest(provisional_view),
            provisional_sequence=provisional_view.sequence,
            provisional_state_revision=provisional_view.state_revision,
            issuer=issuer_trust_descriptor,
        )

    def rollback_registration(self, token: RegistrationPreparation) -> None:
        self.operations.append("rollback")

    def commit_registration(
        self,
        token: RegistrationPreparation,
        signed_committed_snapshot: CommittedExecutionSnapshot,
    ) -> ExecutionHandle:
        self.operations.append("commit")
        if self.fail_commit:
            raise ExecutionRegistrationError("commit unavailable")
        assert signed_committed_snapshot.trust_key_id == token.issuer.key_id
        return ExecutionHandle(
            run_id=token.run_id,
            trace_id=token.trace_id,
            plan_id=token.plan_id,
            plan_revision=token.plan_revision,
            state_revision=signed_committed_snapshot.state_revision,
            routed_state=signed_committed_snapshot.folded_view.run_state.value,
            cancelled=False,
        )

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
    confidence_gate_factory=None,
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
        confidence_gate_factory=confidence_gate_factory or (lambda plan: ConfidenceGate()),
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


def _append_waiting_approval(store: SQLiteHarnessEventStore, run_id: str) -> None:
    view = store.snapshot(run_id)
    transition = HarnessTransition(
        run_id=run_id,
        trace_id=view.trace_id,
        entity_kind="run",
        entity_id=run_id,
        from_state=RunState.RUNNING.value,
        to_state=RunState.WAITING_APPROVAL.value,
        expected_state_revision=view.state_revision,
        plan_revision=view.plan_revision,
        reason_code="approval_required",
        idempotency_key=f"approval-{view.state_revision}",
    )
    authority = TransitionAuthorityRecord(
        **transition.model_dump(
            mode="python", exclude={"schema_version", "lease_epoch", "fencing_token_digest"}
        ),
        dependency_versions=view.dependency_versions,
    )
    store.append(
        HarnessEvent(
            event_id="approval-authority",
            trace_id=view.trace_id,
            span_id="approval-span-1",
            run_id=run_id,
            event_type="transition_authorized",
            occurred_at=NOW,
            monotonic_offset=100.0,
            actor="host-policy",
            payload={"reason_code": "approval_required"},
            transition_authority=authority,
        ),
        expected_sequence=view.sequence,
        expected_state_revision=view.state_revision,
    )
    view = store.snapshot(run_id)
    store.append(
        HarnessEvent(
            event_id="approval-transition",
            trace_id=view.trace_id,
            span_id="approval-span-2",
            run_id=run_id,
            event_type="transition_committed",
            occurred_at=NOW,
            monotonic_offset=100.0,
            actor="harness-kernel",
            payload={"reason_code": "approval_required"},
            transition=transition,
        ),
        expected_sequence=view.sequence,
        expected_state_revision=view.state_revision,
    )


def _action_observation(plan_revision: int) -> ActionObservationFingerprint:
    action = build_action_fingerprint(
        worker_id="information-worker",
        worker_version="worker-v1",
        action_kind="summarize",
        canonical_arguments={"operation": "summarize"},
        context_hash="1" * 64,
        dependency_hash="2" * 64,
        plan_revision=plan_revision,
        prompt_hash="3" * 64,
        tool_hash="4" * 64,
        output_schema_hash="5" * 64,
        model_route="luna",
        correction_ordinal=0,
    )
    result = build_result_fingerprint(
        outcome_kind="answer",
        validated_output_hash="6" * 64,
        normalized_error_class=None,
        normalized_error_code=None,
        result_schema_version="v1",
    )
    return ActionObservationFingerprint.from_parts(action, result, scope=LoopScope.RUN)


def test_create_publishes_only_after_all_dependencies_are_ready(tmp_path):
    backend = RecordingBackend(fail_prepare=True)
    kernel, store, _, _ = _kernel(tmp_path, backend=backend)

    with pytest.raises(ExecutionRegistrationError):
        kernel.create(_request())

    assert store.load("run-1") == ()


def test_bad_issuer_is_rejected_before_create_publishes_any_event(tmp_path):
    class TrustCheckingBackend(RecordingBackend):
        def prepare_registration(self, plan, provisional_view, issuer_trust_descriptor):
            if issuer_trust_descriptor.key_id != _HARNESS_RECEIPT_SIGNER.key_id:
                raise ExecutionRegistrationError("issuer not pinned")
            return super().prepare_registration(
                plan, provisional_view, issuer_trust_descriptor
            )

    class BadIssuer(StoreBackedIssuer):
        def trust_descriptor(self):
            return IssuerTrustDescriptor(
                trust_version="test-trust-v1",
                trust_config_digest=HASH,
                key_id="attacker-host",
            )

    kernel, store, _, _ = _kernel(tmp_path, backend=TrustCheckingBackend())
    kernel._receipt_issuer = BadIssuer(store)
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
    assert backend.operations == ["prepare", "commit"]
    with pytest.raises(Exception):
        handle.run_id = "changed"


def test_create_uses_exact_identified_provisional_fold(tmp_path):
    class ExactProvisionalBackend(RecordingBackend):
        def prepare_registration(self, plan, provisional_view, issuer_trust_descriptor):
            assert provisional_view == HarnessSessionView(
                run_id=plan.run_id,
                trace_id=plan.trace_id,
                plan_revision=plan.revision,
            )
            return super().prepare_registration(
                plan, provisional_view, issuer_trust_descriptor
            )

    kernel, store, _, _ = _kernel(tmp_path, backend=ExactProvisionalBackend())
    handle = kernel.create(_request())
    assert handle.run_state is RunState.CREATED
    assert len(store.load(handle.run_id)) == 1


def test_real_langgraph_backend_accepts_valid_two_phase_create(tmp_path, monkeypatch):
    from market_agent_test_bundle.tests.test_workflow_execution_backend import (
        TrustedReceiptVerifier,
        _generate_test_rsa_key,
    )

    modulus, exponent, private = _generate_test_rsa_key()
    key_id = "test-host-rsa"
    keys = {key_id: (modulus, exponent)}
    digest = execution_backend_module._trust_config_digest(keys)
    monkeypatch.setattr(execution_backend_module, "_PINNED_PUBLIC_KEYS", keys)
    monkeypatch.setattr(
        execution_backend_module, "_EXPECTED_TRUST_CONFIG_DIGEST", digest
    )
    signer = TrustedReceiptVerifier(
        modulus=modulus, private_exponent=private, key_id=key_id
    )
    store = SQLiteHarnessEventStore(tmp_path / "real.sqlite", monotonic=lambda: 100.0)

    class SignedIssuer(StoreBackedIssuer):
        def trust_descriptor(self):
            return IssuerTrustDescriptor(
                trust_version=execution_backend_module._TRUST_VERSION,
                trust_config_digest=digest,
                key_id=key_id,
            )

        def _snapshot(self, plan, sequence=None):
            unsigned = super()._snapshot(plan, sequence).model_copy(
                update={"trust_key_id": key_id}
            )
            signed = signer.approve(unsigned)
            self.snapshots[-1] = signed
            return signed

        def issue_transition_receipt(self, plan, transition, *, pre_sequence):
            unsigned = CommittedTransitionReceipt(
                pre=self._snapshot(plan, pre_sequence),
                post=self._snapshot(plan),
                transition_digest=canonical_transition_digest(transition),
                trust_key_id=key_id,
                signature=SIGNATURE,
            )
            receipt = signer.approve(unsigned)
            self.receipts.append(receipt)
            return receipt

    kernel = HarnessKernel(
        event_store=store,
        state_machine=GlobalTaskStateMachine(),
        plan_compiler=_compiler(),
        pinned_versions=_pinned(),
        loop_guard_factory=lambda: LoopGuard(
            severity_policy=SeverityPolicy(policy_version="policy-v1")
        ),
        confidence_gate_factory=lambda plan: ConfidenceGate(),
        budget_factory=lambda mode: WorkflowBudgetLedger(mode, clock=lambda: 100.0),
        execution_backend=LangGraphExecutionBackend(),
        receipt_issuer=SignedIssuer(store),
        clock=DeterministicClock(),
        identifiers=DeterministicIds(),
    )
    handle = kernel.create(_request())
    assert handle.run_state is RunState.CREATED
    assert handle.backend_synchronized is True
    assert len(store.load(handle.run_id)) == 1


def test_create_commit_failure_keeps_truth_but_rolls_back_and_does_not_return(tmp_path):
    backend = RecordingBackend(fail_commit=True)
    kernel, store, _, _ = _kernel(tmp_path, backend=backend)

    with pytest.raises(ExecutionRegistrationError):
        kernel.create(_request())

    assert store.snapshot("run-1").run_state is RunState.CREATED
    assert backend.operations == ["prepare", "commit", "rollback"]


def test_public_output_copy_revalidates_all_scalars_and_cross_fields(tmp_path):
    kernel, _, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    decision = kernel.advance(handle.run_id, candidate={})

    with pytest.raises(ValidationError):
        handle.model_copy(update={"sequence": -1})
    with pytest.raises(ValidationError):
        handle.model_copy(update={"backend_synchronized": "false"})
    with pytest.raises(ValidationError):
        decision.model_copy(update={"retry_authorized": True, "run_state": RunState.SUCCEEDED})
    with pytest.raises(ValidationError):
        decision.model_copy(update={"backend_synchronized": "true"})


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
    decision = kernel.advance(handle.run_id, candidate={"goto": "failed"})

    assert decision.run_state is RunState.ADMITTED
    assert decision.reason_code == "request_admitted"
    assert len(store.snapshot(handle.run_id).transition_authorities) == 1


def test_backend_apply_failure_replays_same_candidate_without_double_append(tmp_path):
    backend = RecordingBackend(fail_apply=True)
    kernel, store, _, _ = _kernel(tmp_path, backend=backend)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    plan = HarnessPlan.model_validate_json(store.load(handle.run_id)[0].payload["plan_json"])
    observation = _action_observation(plan.revision)

    first = kernel.advance(
        handle.run_id, candidate={"action_observation": observation}
    )
    committed_sequence = store.snapshot(handle.run_id).sequence
    backend.fail_apply = False
    replayed = kernel.advance(
        handle.run_id, candidate={"action_observation": observation}
    )

    assert first.backend_synchronized is False
    assert replayed.run_state is first.run_state
    assert replayed.reason_code == first.reason_code
    assert replayed.backend_synchronized is True
    assert store.snapshot(handle.run_id).sequence == committed_sequence


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

    sequence = store.snapshot(handle.run_id).sequence
    duplicate = kernel.cancel(handle.run_id, "user_requested")
    assert duplicate.run_state is RunState.WAITING_RECONCILIATION
    assert store.snapshot(handle.run_id).sequence == sequence
    assert sum(event.event_type == "cancellation_requested" for event in store.load(handle.run_id)) == 1


def test_concurrent_cancellation_records_one_durable_intent(tmp_path):
    kernel, store, backend, issuer = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    _append_waiting_reconciliation(store, handle.run_id)
    second = HarnessKernel(
        event_store=SQLiteHarnessEventStore(tmp_path / "harness.sqlite", monotonic=lambda: 100.0),
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
        identifiers=DeterministicIds(100),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        decisions = tuple(
            pool.map(
                lambda item: item.cancel(handle.run_id, "user_requested"),
                (kernel, second),
            )
        )
    assert all(item.run_state is RunState.WAITING_RECONCILIATION for item in decisions)
    assert sum(
        event.event_type == "cancellation_requested"
        for event in store.load(handle.run_id)
    ) == 1
    assert backend.operations.count("cancel") == 0


def test_cancel_waiting_approval_commits_legal_terminal_then_cancels_backend(tmp_path):
    kernel, store, backend, issuer = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    _append_waiting_approval(store, handle.run_id)

    decision = kernel.cancel(handle.run_id, "user_requested")

    assert decision.run_state is RunState.CANCELLED
    assert decision.reason_code == "cancellation_completed"
    assert decision.transition is not None
    assert issuer.receipts[-1].transition_digest == canonical_transition_digest(
        decision.transition
    )
    assert backend.operations[-2:] == ["apply", "cancel"]
    sequence = store.snapshot(handle.run_id).sequence
    duplicate = kernel.cancel(handle.run_id, "user_requested")
    assert duplicate.run_state is RunState.CANCELLED
    assert store.snapshot(handle.run_id).sequence == sequence
    assert backend.operations.count("cancel") == 1


def test_cancelled_restart_does_not_cancel_backend_when_resume_fails(tmp_path):
    kernel, store, backend, issuer = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    _append_waiting_approval(store, handle.run_id)
    kernel.cancel(handle.run_id, "user_requested")
    prior_cancels = backend.operations.count("cancel")
    backend.fail_resume_number = backend.resume_count + 1
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
    decision = restarted.cancel(handle.run_id, "user_requested")
    assert decision.backend_synchronized is False
    assert backend.operations.count("cancel") == prior_cancels


def test_unpinned_confidence_fails_closed_to_no_trade_degradation(tmp_path):
    kernel, _, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)

    decision = kernel.advance(handle.run_id, candidate={})

    assert decision.run_state is RunState.DEGRADING
    assert decision.retry_authorized is False
    assert decision.no_trade is True


def test_bound_confidence_success_finishes_succeeded_not_degraded(tmp_path):
    material: dict[str, object] = {}

    def gate_factory(plan: HarnessPlan) -> ConfidenceGate:
        gate, observation, artifact = _confidence_material(plan)
        material.update(observation=observation, artifact=artifact)
        return gate

    kernel, _, _, _ = _kernel(tmp_path, confidence_gate_factory=gate_factory)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)

    summarizing = kernel.advance(
        handle.run_id,
        candidate={
            "confidence_observation": material["observation"],
            "confidence_artifact": material["artifact"],
        },
    )
    terminal = kernel.advance(handle.run_id, candidate={})

    assert summarizing.run_state is RunState.SUMMARIZING
    assert summarizing.no_trade is False
    assert terminal.run_state is RunState.SUCCEEDED
    assert terminal.reason_code == "completed"
    assert terminal.no_trade is False


def test_succeeded_run_exposes_a_host_verified_terminal_receipt(tmp_path):
    import market_agent.workflow_execution_backend as backend_module

    material: dict[str, object] = {}

    def gate_factory(plan: HarnessPlan) -> ConfidenceGate:
        gate, observation, artifact = _confidence_material(plan)
        material.update(observation=observation, artifact=artifact)
        return gate

    kernel, _, _, _ = _kernel(tmp_path, confidence_gate_factory=gate_factory)
    request = _request()
    handle = kernel.create(request)
    _advance_to_running(kernel, handle.run_id)
    kernel.advance(
        handle.run_id,
        candidate={
            "confidence_observation": material["observation"],
            "confidence_artifact": material["artifact"],
        },
    )
    kernel.advance(
        handle.run_id,
        candidate={},
        observed_execution=True,
        prompt_release_digest="e" * 64,
        accepted_result_digest="f" * 64,
    )

    assert hasattr(backend_module, "verify_committed_transition_receipt")
    receipt = kernel.terminal_receipt(handle.run_id)
    assert backend_module.verify_committed_transition_receipt(receipt)
    assert receipt.post.folded_view.run_state is RunState.SUCCEEDED
    assert receipt.post.folded_view.request_digest is not None
    assert receipt.post.folded_view.prompt_release_digest == "e" * 64
    assert receipt.post.folded_view.accepted_result_digest == "f" * 64


def test_gate_snapshot_mismatch_is_permanent_no_trade_for_run(tmp_path):
    def wrong_gate(plan: HarnessPlan) -> ConfidenceGate:
        gate, _, _ = _confidence_material(plan)
        context = gate.trusted_context_snapshot()
        policy = gate.trusted_policy_snapshot()
        assert context is not None and policy is not None
        wrong = context.model_copy(
            update={
                "hard_gates": context.hard_gates.model_copy(
                    update={"trace_hash": "9" * 64}
                )
            }
        )
        return ConfidenceGate(
            trusted_policy=policy,
            signature_verifier=ConfidenceVerifier(),
            request_context=wrong,
        )

    kernel, _, _, _ = _kernel(tmp_path, confidence_gate_factory=wrong_gate)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)

    decision = kernel.advance(handle.run_id, candidate={})

    assert decision.run_state is RunState.DEGRADING
    assert decision.reason_code == "confidence_context_mismatch"
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


def test_budget_reservation_is_settled_and_durably_projected(tmp_path):
    budget = WorkflowBudgetLedger(WorkflowMode.PASSIVE, clock=lambda: 100.0)
    calls: list[str] = []
    reserve = budget.reserve
    settle = budget.settle

    def recording_reserve(**kwargs):
        calls.append("reserve")
        return reserve(**kwargs)

    def recording_settle(reservation, usage):
        calls.append("settle")
        return settle(reservation, usage)

    budget.reserve = recording_reserve
    budget.settle = recording_settle
    kernel, store, _, _ = _kernel(tmp_path, budget=budget)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)

    kernel.advance(handle.run_id, candidate={})

    assert calls == ["reserve", "settle"]
    authority_events = [
        event for event in store.load(handle.run_id) if event.event_type == "transition_authorized"
    ]
    assert authority_events[-1].payload["budget_remaining_attempts"] == 9
    assert authority_events[-1].payload["budget_remaining_tokens"] == 798


def test_untrusted_self_reported_budget_projection_fails_closed_before_backend(tmp_path):
    kernel, store, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    plan = HarnessPlan.model_validate_json(store.load(handle.run_id)[0].payload["plan_json"])
    view = store.snapshot(handle.run_id)
    store.append(
        HarnessEvent(
            event_id="budget-baseline",
            trace_id=plan.trace_id,
            span_id="budget-span",
            run_id=plan.run_id,
            event_type="budget_projected",
            occurred_at=NOW,
            monotonic_offset=100.0,
            actor="harness-kernel",
            payload={
                "budget_remaining_attempts": 1,
                "budget_remaining_seconds": 100.0,
                "budget_remaining_cost": "0.01",
                "budget_remaining_tokens": 2,
            },
        ),
        expected_sequence=view.sequence,
        expected_state_revision=view.state_revision,
    )
    restarted, _, backend, _ = _kernel(tmp_path)
    restarted._identifiers = DeterministicIds(100)
    before = tuple(backend.operations)
    with pytest.raises(HarnessDependencyError, match="canonical"):
        restarted.advance(handle.run_id)
    assert tuple(backend.operations) == before


@pytest.mark.parametrize(
    ("dimension", "value"),
    (
        ("budget_remaining_attempts", 0),
        ("budget_remaining_seconds", 0.0),
        ("budget_remaining_cost", "0"),
        ("budget_remaining_tokens", 0),
    ),
)
def test_partial_or_wrongly_typed_budget_projection_is_rejected(
    tmp_path, dimension, value
):
    kernel, store, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    plan = HarnessPlan.model_validate_json(store.load(handle.run_id)[0].payload["plan_json"])
    view = store.snapshot(handle.run_id)
    payload = {
        "budget_remaining_attempts": 1,
        "budget_remaining_seconds": 100.0,
        "budget_remaining_cost": "0.01",
        "budget_remaining_tokens": 2,
    }
    payload[dimension] = value
    store.append(
        HarnessEvent(
            event_id="budget-hard-gate",
            trace_id=plan.trace_id,
            span_id="budget-hard-gate-span",
            run_id=plan.run_id,
            event_type="budget_projected",
            occurred_at=NOW,
            monotonic_offset=100.0,
            actor="harness-kernel",
            payload=payload,
        ),
        expected_sequence=view.sequence,
        expected_state_revision=view.state_revision,
    )
    restarted, _, backend, _ = _kernel(tmp_path)
    restarted._identifiers = DeterministicIds(100)
    before = tuple(backend.operations)
    with pytest.raises(HarnessDependencyError):
        restarted.advance(handle.run_id)
    assert tuple(backend.operations) == before


def _canonical_budget_payload(**updates: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "policy": "confidence_gate",
        "reason_code": "confidence_fail_closed",
        "no_trade": True,
        "budget_before_attempts": 10,
        "budget_delta_attempts": 1,
        "budget_remaining_attempts": 9,
        "budget_before_seconds": "130.0",
        "budget_delta_seconds": "0.0",
        "budget_remaining_seconds": "130.0",
        "budget_before_cost": "0.30",
        "budget_delta_cost": "0.01",
        "budget_remaining_cost": "0.29",
        "budget_before_tokens": 800,
        "budget_delta_tokens": 2,
        "budget_remaining_tokens": 798,
    }
    payload.update(updates)
    return payload


def _budget_authority_event(
    plan: HarnessPlan,
    view: HarnessSessionView,
    payload: dict[str, object],
    *,
    event_type: str = "transition_authorized",
    authority_updates: dict[str, object] | None = None,
) -> HarnessEvent:
    authority_values: dict[str, object] = {
        "run_id": plan.run_id,
        "trace_id": plan.trace_id,
        "entity_kind": "run",
        "entity_id": plan.run_id,
        "from_state": RunState.RUNNING.value,
        "to_state": RunState.DEGRADING.value,
        "expected_state_revision": view.state_revision,
        "plan_revision": plan.revision,
        "reason_code": "confidence_fail_closed",
        "idempotency_key": f"run-{view.state_revision}-{RunState.DEGRADING.value}",
        "dependency_versions": view.dependency_versions,
    }
    authority_values.update(authority_updates or {})
    authority = TransitionAuthorityRecord(**authority_values)
    return HarnessEvent(
        event_id=f"budget-event-{view.sequence}",
        trace_id=plan.trace_id,
        span_id=f"budget-span-{view.sequence}",
        run_id=plan.run_id,
        sequence=view.sequence + 1,
        state_revision=view.state_revision,
        event_type=event_type,
        occurred_at=NOW,
        monotonic_offset=100.0,
        actor="harness-kernel",
        payload=payload,
        transition_authority=authority if event_type == "transition_authorized" else None,
    )


def test_real_budget_authority_is_complete_and_survives_restart(tmp_path):
    kernel, store, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    kernel.advance(handle.run_id, candidate={})
    events = tuple(store.load(handle.run_id))
    plan = HarnessPlan.model_validate_json(events[0].payload["plan_json"])
    authority = next(
        event
        for event in reversed(events)
        if event.event_type == "transition_authorized"
        and "budget_before_attempts" in event.payload
    )
    assert set(harness_module._BUDGET_FIELDS).issubset(authority.payload)
    assert "budget_settlement_evidence" in authority.payload
    restarted, _, _, _ = _kernel(tmp_path)
    restarted.resume(handle.run_id)
    assert restarted._budget_state(
        tuple(store.load(handle.run_id)), plan, restarted._budgets[handle.run_id].snapshot()
    ) == (
        authority.payload["budget_remaining_attempts"],
        Decimal(authority.payload["budget_remaining_seconds"]),
        Decimal(authority.payload["budget_remaining_cost"]),
        authority.payload["budget_remaining_tokens"],
    )


def _tampered_budget_authority(
    authority: HarnessEvent, mutate: callable
) -> HarnessEvent:
    payload = dict(authority.payload)
    evidence = dict(payload["budget_settlement_evidence"])
    settlement = dict(evidence["settlement"])
    settlement["usage"] = dict(settlement["usage"])
    settlement["projection"] = dict(settlement["projection"])
    mutate(payload, evidence, settlement)
    settlement_unsigned = {
        key: value for key, value in settlement.items() if key != "settlement_digest"
    }
    settlement["settlement_digest"] = HarnessKernel._canonical_digest(settlement_unsigned)
    evidence["settlement"] = settlement
    evidence_unsigned = {
        key: value for key, value in evidence.items() if key != "binding_digest"
    }
    evidence["binding_digest"] = HarnessKernel._canonical_digest(evidence_unsigned)
    payload["budget_settlement_evidence"] = evidence
    return authority.model_copy(update={"payload": payload})


@pytest.mark.parametrize(
    "mutate",
    (
        lambda payload, evidence, settlement: (
            payload.update(
                budget_delta_attempts=0,
                budget_remaining_attempts=10,
                budget_delta_seconds="0.0",
                budget_remaining_seconds="130.0",
                budget_delta_cost="0.0",
                budget_remaining_cost="0.30",
                budget_delta_tokens=0,
                budget_remaining_tokens=800,
            ),
            settlement["projection"].update(
                budget_delta_attempts=0,
                budget_remaining_attempts=10,
                budget_delta_seconds="0.0",
                budget_remaining_seconds="130.0",
                budget_delta_cost="0.0",
                budget_remaining_cost="0.30",
                budget_delta_tokens=0,
                budget_remaining_tokens=800,
            ),
        ),
        lambda payload, evidence, settlement: settlement["usage"].update(output_tokens=0),
        lambda payload, evidence, settlement: settlement.update(charged_cost="0.0"),
        lambda payload, evidence, settlement: settlement.update(reservation_id="forged-reservation"),
        lambda payload, evidence, settlement: evidence.update(settlement_event_hash="0" * 64),
    ),
)
def test_budget_settlement_evidence_tampering_fails_before_restart_or_backend(
    tmp_path, mutate
):
    kernel, store, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    kernel.advance(handle.run_id, candidate={})
    events = tuple(store.load(handle.run_id))
    plan = HarnessPlan.model_validate_json(events[0].payload["plan_json"])
    index, authority = next(
        (index, event)
        for index, event in enumerate(events)
        if event.event_type == "transition_authorized"
        and "budget_before_attempts" in event.payload
    )
    tampered = _tampered_budget_authority(authority, mutate)
    with pytest.raises(HarnessDependencyError):
        HarnessKernel._budget_state(
            (*events[:index], tampered, *events[index + 1 :]),
            plan,
            WorkflowBudgetLedger(WorkflowMode.PASSIVE, clock=lambda: 100.0).snapshot(),
        )


@pytest.mark.parametrize(
    "mutate_receipt",
    (
        lambda receipt: receipt.update(signature="0" * 512),
        lambda receipt: receipt.update(trust_key_id="untrusted-host"),
        lambda receipt: receipt.update(plan_digest="0" * 64),
        lambda receipt: receipt.update(event_head_hash="0" * 64),
    ),
)
def test_budget_settlement_receipt_tampering_fails_before_backend(
    tmp_path, mutate_receipt
):
    kernel, store, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    kernel.advance(handle.run_id, candidate={})
    events = tuple(store.load(handle.run_id))
    plan = HarnessPlan.model_validate_json(events[0].payload["plan_json"])
    index, authority = next(
        (index, event)
        for index, event in enumerate(events)
        if event.event_type == "transition_authorized"
        and "budget_before_attempts" in event.payload
    )
    payload = dict(authority.payload)
    evidence = dict(payload["budget_settlement_evidence"])
    receipt = json.loads(evidence["host_receipt"])
    mutate_receipt(receipt)
    evidence["host_receipt"] = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    unsigned = {key: value for key, value in evidence.items() if key != "binding_digest"}
    evidence["binding_digest"] = HarnessKernel._canonical_digest(unsigned)
    payload["budget_settlement_evidence"] = evidence
    tampered = authority.model_copy(update={"payload": payload})

    with pytest.raises(HarnessDependencyError, match="receipt"):
        HarnessKernel._budget_state(
            (*events[:index], tampered, *events[index + 1 :]),
            plan,
            WorkflowBudgetLedger(WorkflowMode.PASSIVE, clock=lambda: 100.0).snapshot(),
        )


def test_budget_settlement_outer_evidence_digest_tampering_fails_before_backend(tmp_path):
    kernel, store, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    kernel.advance(handle.run_id, candidate={})
    events = tuple(store.load(handle.run_id))
    plan = HarnessPlan.model_validate_json(events[0].payload["plan_json"])
    index, authority = next(
        (index, event)
        for index, event in enumerate(events)
        if event.event_type == "transition_authorized"
        and "budget_before_attempts" in event.payload
    )
    payload = dict(authority.payload)
    evidence = dict(payload["budget_settlement_evidence"])
    evidence["binding_digest"] = "0" * 64
    payload["budget_settlement_evidence"] = evidence
    tampered = authority.model_copy(update={"payload": payload})

    with pytest.raises(HarnessDependencyError, match="digest"):
        HarnessKernel._budget_state(
            (*events[:index], tampered, *events[index + 1 :]),
            plan,
            WorkflowBudgetLedger(WorkflowMode.PASSIVE, clock=lambda: 100.0).snapshot(),
        )


@pytest.mark.parametrize(
    "updates",
    (
        {"budget_before_attempts": 11},
        {"budget_remaining_attempts": 10},
        {"budget_delta_attempts": -1},
        {"budget_before_seconds": 130.0},
        {"budget_remaining_cost": "2.9E-1"},
        {"budget_untrusted_override": 1},
    ),
)
def test_budget_projection_rejects_raise_nonmonotonic_type_and_extra_fields(
    tmp_path, updates
):
    kernel, store, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    events = tuple(store.load(handle.run_id))
    plan = HarnessPlan.model_validate_json(events[0].payload["plan_json"])
    view = store.snapshot(handle.run_id)
    event = _budget_authority_event(plan, view, _canonical_budget_payload(**updates))
    with pytest.raises(HarnessDependencyError):
        HarnessKernel._budget_state(
            (*events, event),
            plan,
            WorkflowBudgetLedger(WorkflowMode.PASSIVE, clock=lambda: 100.0).snapshot(),
        )


def test_budget_projection_rejects_wrong_authority_and_zero_cannot_recover(tmp_path):
    kernel, store, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    events = tuple(store.load(handle.run_id))
    plan = HarnessPlan.model_validate_json(events[0].payload["plan_json"])
    view = store.snapshot(handle.run_id)
    wrong = _budget_authority_event(
        plan,
        view,
        _canonical_budget_payload(),
        authority_updates={"plan_revision": plan.revision + 1},
    )
    with pytest.raises(HarnessDependencyError, match="authority"):
        HarnessKernel._budget_state(
            (*events, wrong),
            plan,
            WorkflowBudgetLedger(WorkflowMode.PASSIVE, clock=lambda: 100.0).snapshot(),
        )

    zero = _budget_authority_event(
        plan,
        view,
        _canonical_budget_payload(
            budget_delta_attempts=10,
            budget_remaining_attempts=0,
        ),
    )
    recovered = _budget_authority_event(
        plan,
        view.model_copy(update={"sequence": view.sequence + 1}),
        _canonical_budget_payload(
            budget_before_attempts=0,
            budget_delta_attempts=0,
            budget_remaining_attempts=1,
            budget_before_seconds="130.0",
            budget_before_cost="0.29",
            budget_before_tokens=798,
        ),
    )
    with pytest.raises(HarnessDependencyError, match="monotonic|evidence"):
        HarnessKernel._budget_state(
            (*events, zero, recovered),
            plan,
            WorkflowBudgetLedger(WorkflowMode.PASSIVE, clock=lambda: 100.0).snapshot(),
        )


def test_same_revision_concurrent_advances_commit_only_one_policy_delta(tmp_path):
    kernel, store, backend, issuer = _kernel(tmp_path)
    handle = kernel.create(_request())
    second = HarnessKernel(
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
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(
            pool.map(
                lambda item: item.advance(
                    handle.run_id,
                    candidate={},
                    expected_state_revision=handle.state_revision,
                ),
                (kernel, second),
            )
        )
    assert sorted(result.reason_code for result in results) == [
        "request_admitted",
        "stale_revision",
    ]
    assert store.snapshot(handle.run_id).state_revision == handle.state_revision + 1
    assert len(store.load(handle.run_id)) == 3


def test_concurrent_running_advances_consume_one_durable_budget_delta(tmp_path):
    kernel, store, backend, issuer = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    running = store.snapshot(handle.run_id)
    second = HarnessKernel(
        event_store=SQLiteHarnessEventStore(tmp_path / "harness.sqlite", monotonic=lambda: 100.0),
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
        identifiers=DeterministicIds(100),
    )
    with ThreadPoolExecutor(max_workers=2) as pool:
        decisions = tuple(
            pool.map(
                lambda item: item.advance(
                    handle.run_id,
                    candidate={},
                    expected_state_revision=running.state_revision,
                ),
                (kernel, second),
            )
        )
    assert sum(item.reason_code == "stale_revision" for item in decisions) == 1
    budget_authorities = [
        event for event in store.load(handle.run_id)
        if event.event_type == "transition_authorized"
        and "budget_remaining_attempts" in event.payload
    ]
    assert len(budget_authorities) == 1
    assert budget_authorities[0].payload["budget_remaining_attempts"] == 9


def test_concurrent_kernels_using_database_path_aliases_share_one_lock(tmp_path):
    database = tmp_path / "harness.sqlite"
    alias_directory = tmp_path / "alias"
    alias_directory.mkdir()
    store = SQLiteHarnessEventStore(database, monotonic=lambda: 100.0)
    alias_store = SQLiteHarnessEventStore(
        alias_directory / ".." / "harness.sqlite", monotonic=lambda: 100.0
    )
    backend = RecordingBackend()
    issuer = StoreBackedIssuer(store)

    def build(event_store, start=0):
        return HarnessKernel(
            event_store=event_store,
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
            identifiers=DeterministicIds(start),
        )

    first = build(store)
    handle = first.create(_request())
    _advance_to_running(first, handle.run_id)
    running = store.snapshot(handle.run_id)
    second = build(alias_store, 100)
    with ThreadPoolExecutor(max_workers=2) as pool:
        decisions = tuple(
            pool.map(
                lambda kernel: kernel.advance(
                    handle.run_id,
                    candidate={},
                    expected_state_revision=running.state_revision,
                ),
                (first, second),
            )
        )
    assert sum(decision.reason_code == "stale_revision" for decision in decisions) == 1
    projections = [
        event
        for event in store.load(handle.run_id)
        if event.event_type == "transition_authorized"
        and "budget_before_attempts" in event.payload
    ]
    assert len(projections) == 1


@pytest.mark.parametrize(
    ("update", "message"),
    (
        ({"run_state": RunState.DEGRADING, "no_trade": False}, "no-trade"),
        ({"run_state": RunState.DEGRADED, "no_trade": False}, "no-trade"),
        ({"run_state": RunState.WAITING_RECONCILIATION, "reconciliation_required": False}, "reconciliation"),
        ({"reason_code": "completed", "run_state": RunState.DEGRADING, "no_trade": True}, "completed"),
    ),
)
def test_decision_copy_enforces_bidirectional_state_invariants(tmp_path, update, message):
    kernel, _, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    decision = kernel.advance(handle.run_id, candidate={})
    with pytest.raises(ValidationError, match=message):
        decision.model_copy(update=update)


@pytest.mark.parametrize(
    ("reason_code", "allowed_states"),
    (
        ("request_admitted", {RunState.ADMITTED}),
        ("plan_committed", {RunState.PLANNED}),
        ("dependencies_ready", {RunState.READY}),
        ("execution_started", {RunState.RUNNING}),
        ("safe_no_trade_summary", {RunState.SUMMARIZING}),
        ("confidence_sufficient", {RunState.SUMMARIZING}),
        ("completed", {RunState.SUCCEEDED}),
        ("budget_exhausted", {RunState.DEGRADING}),
        ("confidence_fail_closed", {RunState.DEGRADING}),
        ("safe_no_trade_due_to_degradation", {RunState.DEGRADING, RunState.DEGRADED}),
        ("reconciliation_required", {RunState.WAITING_RECONCILIATION}),
        ("cancellation_completed", {RunState.CANCELLED}),
    ),
)
def test_decision_reason_codes_have_closed_state_mapping(reason_code, allowed_states):
    for state in RunState:
        values = {
            "run_id": "run-1",
            "trace_id": "trace-1",
            "sequence": 1,
            "state_revision": 1,
            "run_state": state,
            "plan_revision": 0,
            "reason_code": reason_code,
            "no_trade": state in {
                RunState.DEGRADING,
                RunState.DEGRADED,
                RunState.WAITING_RECONCILIATION,
            } or reason_code == "safe_no_trade_summary",
            "reconciliation_required": state is RunState.WAITING_RECONCILIATION,
        }
        if state in allowed_states:
            assert HarnessDecision(**values).run_state is state
        else:
            with pytest.raises(ValidationError):
                HarnessDecision(**values)


def test_decision_rejects_unknown_reason_and_running_no_trade_even_on_copy():
    with pytest.raises(ValidationError, match="reason"):
        HarnessDecision(
            run_id="run-1",
            trace_id="trace-1",
            sequence=1,
            state_revision=1,
            run_state=RunState.RUNNING,
            plan_revision=0,
            reason_code="invented_reason",
        )
    running = HarnessDecision(
        run_id="run-1",
        trace_id="trace-1",
        sequence=1,
        state_revision=1,
        run_state=RunState.RUNNING,
        plan_revision=0,
        reason_code="execution_started",
    )
    with pytest.raises(ValidationError, match="no-trade"):
        running.model_copy(update={"no_trade": True})


@pytest.mark.parametrize("reason_code", ("stale_revision", "candidate_rejected", "terminal_state"))
def test_observation_decisions_cannot_be_upgraded_to_transitions(reason_code):
    state = RunState.SUCCEEDED if reason_code == "terminal_state" else RunState.RUNNING
    transition = HarnessTransition(
        run_id="run-1",
        trace_id="trace-1",
        entity_kind="run",
        entity_id="run-1",
        from_state=state.value,
        to_state=state.value,
        expected_state_revision=0,
        plan_revision=0,
        reason_code=reason_code,
        idempotency_key=f"run-0-{state.value}",
    )
    with pytest.raises(ValidationError, match="non-committing"):
        HarnessDecision(
            run_id="run-1",
            trace_id="trace-1",
            sequence=1,
            state_revision=1,
            run_state=state,
            plan_revision=0,
            reason_code=reason_code,
            no_trade=False,
            transition=transition,
        )


def test_decision_transition_is_bound_to_the_exact_committed_run_transition():
    transition = HarnessTransition(
        run_id="run-1",
        trace_id="trace-1",
        entity_kind="run",
        entity_id="run-1",
        from_state=RunState.RUNNING.value,
        to_state=RunState.SUMMARIZING.value,
        expected_state_revision=5,
        plan_revision=3,
        reason_code="safe_no_trade_summary",
        idempotency_key="run-5-summarizing",
    )
    decision = HarnessDecision(
        run_id="run-1",
        trace_id="trace-1",
        sequence=12,
        state_revision=6,
        run_state=RunState.SUMMARIZING,
        previous_run_state=RunState.RUNNING,
        plan_revision=3,
        reason_code="safe_no_trade_summary",
        no_trade=True,
        transition=transition,
    )

    for forged in (
        {"previous_run_state": RunState.CREATED},
        {"plan_revision": 4},
        {"transition": transition.model_copy(update={"entity_id": "other-run"})},
        {"transition": transition.model_copy(update={"from_state": "created"})},
        {"transition": transition.model_copy(update={"plan_revision": 4})},
        {"transition": transition.model_copy(update={"idempotency_key": "forged"})},
    ):
        with pytest.raises(ValidationError, match="committed transition"):
            decision.model_copy(update=forged)

    observation = HarnessDecision(
        run_id="run-1",
        trace_id="trace-1",
        sequence=12,
        state_revision=6,
        run_state=RunState.RUNNING,
        plan_revision=3,
        reason_code="candidate_rejected",
    )
    with pytest.raises(ValidationError, match="prior state"):
        observation.model_copy(update={"previous_run_state": RunState.RUNNING})


def test_store_aliases_share_coordination_lock_and_lock_table_is_reclaimed(tmp_path, monkeypatch):
    database = tmp_path / "locks" / "harness.sqlite"
    database.parent.mkdir()
    store = SQLiteHarnessEventStore(database)
    alias = SQLiteHarnessEventStore(database.parent / "." / "harness.sqlite")
    monkeypatch.chdir(database.parent)
    relative_store = SQLiteHarnessEventStore("harness.sqlite")

    assert harness_module._run_lock(store, "run-1") is harness_module._run_lock(alias, "run-1")
    assert harness_module._run_lock(store, "run-1") is harness_module._run_lock(relative_store, "run-1")
    memory_one = SQLiteHarnessEventStore(":memory:")
    memory_two = SQLiteHarnessEventStore(":memory:")
    assert harness_module._run_lock(memory_one, "run-1") is not harness_module._run_lock(memory_two, "run-1")

    for index in range(500):
        harness_module._run_lock(store, f"run-{index}")
    gc.collect()
    assert len(harness_module._RUN_LOCKS) <= 4


def test_loop_action_result_history_is_replayed_before_policy_decision(tmp_path):
    kernel, store, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    plan = HarnessPlan.model_validate_json(store.load(handle.run_id)[0].payload["plan_json"])
    observation = _action_observation(plan.revision)
    for index in range(2):
        view = store.snapshot(handle.run_id)
        store.append(
            HarnessEvent(
                event_id=f"prior-action-{index}",
                trace_id=plan.trace_id,
                span_id=f"prior-action-span-{index}",
                run_id=plan.run_id,
                event_type="loop_observed",
                occurred_at=NOW,
                monotonic_offset=100.0,
                actor="harness-kernel",
                payload={
                    "loop_action_json": observation.model_dump_json(),
                    "loop_plan_revision": plan.revision,
                    "loop_fingerprint_schema_version": (
                        plan.pinned_versions.fingerprint_schema_version
                    ),
                    "loop_policy_version": plan.pinned_versions.policy_version,
                },
            ),
            expected_sequence=view.sequence,
            expected_state_revision=view.state_revision,
        )

    decision = kernel.advance(
        handle.run_id, candidate={"action_observation": observation}
    )

    assert decision.run_state is RunState.DEGRADING
    assert decision.reason_code == "loop_guard_stopped"
    assert decision.no_trade is True


def test_loop_checkpoint_must_bind_current_plan_and_schema(tmp_path):
    kernel, store, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    plan = HarnessPlan.model_validate_json(store.load(handle.run_id)[0].payload["plan_json"])
    progress = ProgressVector()
    checkpoint = SemanticCheckpoint(
        scope=LoopScope.RUN,
        observation_kind=ObservationKind.SEMANTIC_CHECKPOINT,
        state_fingerprint=build_state_fingerprint(
            run_state="running",
            work_item_state="running",
            attempt_state="streaming",
            stage_id="information",
            plan_revision=plan.revision + 1,
            unresolved_work_ids=(),
            dependency_versions=(),
            progress=progress,
            normalized_error_class=None,
        ),
        progress=progress,
        plan_revision=plan.revision + 1,
        fingerprint_schema_version=plan.pinned_versions.fingerprint_schema_version,
    )

    decision = kernel.advance(
        handle.run_id, candidate={"loop_checkpoint": checkpoint}
    )

    assert decision.run_state is RunState.DEGRADING
    assert decision.reason_code == "loop_checkpoint_binding_mismatch"
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


def test_observed_checkpoint_settles_exact_provider_usage_and_replays_v2(tmp_path):
    kernel, store, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    observations = ExecutionObservationCollector(handle.run_id, handle.trace_id)
    work_item = ObservedWorkItem(
        task_id="coordinator-task-1", task_kind="technical",
        worker_id="technical-agent", owner_node=CoreNodeName.DISPATCH,
        maximum_retries=1, execution_state="pending",
    )
    observations.checkpoint(
        plan_revision=0, node=CoreNodeName.PLAN,
        outcome=NodeOutcome.COMPLETED, task_ids=("coordinator-task-1",),
        completed_task_ids=(), failed_task_ids=(),
        retry_state=(TaskRetryState(task_id="coordinator-task-1",
                                    attempts_consumed=0, retries_consumed=0,
                                    retries_remaining=1),),
        work_items=(work_item,), action_fingerprint="a" * 64,
    )
    plan_permit = kernel.record_checkpoint(handle.run_id, observations.checkpoints()[-1])
    assert plan_permit.decision is CheckpointDecision.CONTINUE
    observations.record_attempt(AttemptUsage(
        workflow_id=handle.run_id,
        trace_id=handle.trace_id,
        task_id="coordinator-task-1",
        attempt=0,
        node=CoreNodeName.DISPATCH,
        provider="openai",
        provider_request_id="response-actual-1",
        model_id="gpt-5.6-terra-2026-08-15",
        model_tier="terra",
        pricing_version="openai-standard-2026-08-01",
        pricing_model_id="gpt-5.6-terra",
        pricing_band="short",
        tokens=TokenUsage(
            input_tokens=100,
            cached_input_tokens=25,
            cache_write_tokens=10,
            output_tokens=20,
        ),
        estimated_cost_usd=0.00042,
        latency_ms=125,
        source="provider_response",
    ))
    observations.checkpoint(
        plan_revision=0,
        node=CoreNodeName.DISPATCH,
        outcome=NodeOutcome.COMPLETED,
        task_ids=("coordinator-task-1",),
        completed_task_ids=("coordinator-task-1",),
        failed_task_ids=(),
        retry_state=(TaskRetryState(task_id="coordinator-task-1",
                                    attempts_consumed=1, retries_consumed=0,
                                    retries_remaining=1),),
        work_items=(work_item.model_copy(update={
            "execution_state": "succeeded", "attempt_ids": ("response-actual-1",)
        }),),
        action_fingerprint="b" * 64,
    )

    dispatch_permit = kernel.record_checkpoint(handle.run_id, observations.checkpoints()[-1])
    assert dispatch_permit.decision is CheckpointDecision.CONTINUE
    assert kernel.coordinator_task_inventory(handle.run_id) == ("coordinator-task-1",)
    kernel.advance(handle.run_id, candidate={}, workflow_usage=observations.usage())

    authority = next(
        event for event in reversed(store.load(handle.run_id))
        if event.event_type == "transition_authorized"
        and "budget_settlement_evidence" in event.payload
    )
    evidence = authority.payload["budget_settlement_evidence"]
    assert evidence["settlement"]["schema_version"] == "budget-settlement-evidence-v2"
    settled_usage = json.loads(evidence["settlement"]["workflow_usage_json"])
    assert settled_usage["aggregate"] == {
        "schema_version": "v1",
        "input_tokens": 100,
        "cached_input_tokens": 25,
        "cache_write_tokens": 10,
        "output_tokens": 20,
        "web_search_tool_calls": 0,
    }
    assert authority.payload["budget_delta_tokens"] == 130
    assert authority.payload["budget_remaining_tokens"] == 670
    assert authority.payload["budget_delta_attempts"] == 1
    assert authority.payload["budget_remaining_attempts"] == 9
    assert authority.payload["budget_delta_cost"] == "0.00042"

    restarted, _, _, _ = _kernel(tmp_path)
    restarted.resume(handle.run_id)
    plan = HarnessPlan.model_validate_json(store.load(handle.run_id)[0].payload["plan_json"])
    assert restarted._budget_state(
        tuple(store.load(handle.run_id)),
        plan,
        restarted._budgets[handle.run_id].snapshot(),
    ) == (9, Decimal("129.875"), Decimal("0.29958"), 670)


def test_harness_rejects_terminal_usage_not_bound_by_latest_checkpoint(tmp_path):
    kernel, _, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    observations = ExecutionObservationCollector(handle.run_id, handle.trace_id)
    work_item = ObservedWorkItem(
        task_id="coordinator-task-1", task_kind="technical",
        worker_id="technical-agent", owner_node=CoreNodeName.DISPATCH,
        maximum_retries=1, execution_state="pending",
    )
    observations.checkpoint(
        plan_revision=0,
        node=CoreNodeName.PLAN,
        outcome=NodeOutcome.COMPLETED,
        task_ids=("coordinator-task-1",),
        completed_task_ids=(),
        failed_task_ids=(),
        retry_state=(TaskRetryState(task_id="coordinator-task-1",
                                    attempts_consumed=0, retries_consumed=0,
                                    retries_remaining=1),),
        work_items=(work_item,),
        action_fingerprint="c" * 64,
    )
    kernel.record_checkpoint(handle.run_id, observations.checkpoints()[-1])
    observations.record_attempt(AttemptUsage(
        workflow_id=handle.run_id,
        trace_id=handle.trace_id,
        task_id="coordinator-task-1",
        attempt=0,
        node=CoreNodeName.DISPATCH,
        provider="openai",
        provider_request_id="response-after-checkpoint",
        model_id="gpt-5.6-terra",
        model_tier="terra",
        pricing_version="openai-standard-2026-08-01",
        pricing_model_id="gpt-5.6-terra",
        pricing_band="short",
        tokens=TokenUsage(input_tokens=3, output_tokens=2),
        estimated_cost_usd=0.00003,
        latency_ms=1,
        source="provider_response",
    ))

    with pytest.raises(HarnessDependencyError, match="checkpoint"):
        kernel.advance(handle.run_id, candidate={}, workflow_usage=observations.usage())


def test_explicit_historical_cache_usage_settles_without_provider_attempt(tmp_path):
    kernel, store, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    observations = ExecutionObservationCollector(handle.run_id, handle.trace_id)
    observations.record_attempt(AttemptUsage(
        workflow_id=handle.run_id,
        trace_id=handle.trace_id,
        task_id="historical-cache",
        attempt=0,
        node=CoreNodeName.PLAN,
        provider="host",
        provider_request_id="historical-cache-entry-1",
        model_id="historical-cache",
        model_tier=None,
        pricing_version="host-zero-v1",
        tokens=TokenUsage.zero(),
        estimated_cost_usd=0.0,
        latency_ms=0,
        source="historical_cache",
    ))

    kernel.advance(handle.run_id, candidate={}, workflow_usage=observations.usage())

    authority = next(
        event for event in reversed(store.load(handle.run_id))
        if event.event_type == "transition_authorized"
        and "budget_settlement_evidence" in event.payload
    )
    assert authority.payload["budget_delta_attempts"] == 0
    assert authority.payload["budget_delta_tokens"] == 0
    assert authority.payload["budget_delta_cost"] == "0.0"


def test_recovery_checkpoint_preserves_survivors_and_advances_revision(tmp_path):
    kernel, _, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    observations = ExecutionObservationCollector(handle.run_id, handle.trace_id)
    original = ObservedWorkItem(
        task_id="task-original", task_kind="technical",
        worker_id="technical-agent", owner_node=CoreNodeName.DISPATCH,
        maximum_retries=1, execution_state="pending",
    )
    observations.checkpoint(
        plan_revision=0, node=CoreNodeName.PLAN,
        outcome=NodeOutcome.COMPLETED, task_ids=("task-original",),
        completed_task_ids=(), failed_task_ids=(),
        retry_state=(TaskRetryState(task_id="task-original", attempts_consumed=0,
                                    retries_consumed=0, retries_remaining=1),),
        work_items=(original,), action_fingerprint="0" * 64,
    )
    kernel.record_checkpoint(handle.run_id, observations.checkpoints()[-1])
    observations.checkpoint(
        plan_revision=0,
        node=CoreNodeName.DISPATCH,
        outcome=NodeOutcome.COMPLETED,
        task_ids=("task-original",),
        completed_task_ids=("task-original",),
        failed_task_ids=(),
        retry_state=(TaskRetryState(task_id="task-original", attempts_consumed=0,
                                    retries_consumed=0, retries_remaining=1),),
        work_items=(original.model_copy(update={"execution_state": "succeeded"}),),
        action_fingerprint="1" * 64,
    )
    kernel.record_checkpoint(handle.run_id, observations.checkpoints()[-1])
    recovery = ObservedWorkItem(
        task_id="task-recovery", task_kind="technical",
        worker_id="technical-agent", owner_node=CoreNodeName.RECOVER,
        dependency_ids=("task-original",), maximum_retries=1,
        execution_state="pending",
    )
    observations.checkpoint(
        plan_revision=1,
        node=CoreNodeName.RECOVER,
        outcome=NodeOutcome.COMPLETED,
        task_ids=("task-original", "task-recovery"),
        completed_task_ids=("task-original",),
        failed_task_ids=(),
        retry_state=(
            TaskRetryState(task_id="task-original", attempts_consumed=0,
                           retries_consumed=0, retries_remaining=1),
            TaskRetryState(task_id="task-recovery", attempts_consumed=0,
                           retries_consumed=0, retries_remaining=1),
        ),
        work_items=(original.model_copy(update={"execution_state": "succeeded"}), recovery),
        action_fingerprint="2" * 64,
    )

    kernel.record_checkpoint(handle.run_id, observations.checkpoints()[-1])

    assert kernel.coordinator_task_inventory(handle.run_id) == (
        "task-original", "task-recovery"
    )
    recovery_items = {
        item.task_id: item for item in kernel.checkpoint_history(handle.run_id)[-1].work_items
    }
    assert recovery_items["task-original"].owner_node is CoreNodeName.DISPATCH
    assert recovery_items["task-recovery"].owner_node is CoreNodeName.RECOVER
    assert recovery_items["task-recovery"].dependency_ids == ("task-original",)


def test_graph_recovery_checkpoint_preserves_replaced_task_history(tmp_path):
    request = _request(trace_id="1" * 32)

    def task(identifier: str, *, parent: str | None = None) -> AgentTask:
        return AgentTask(
            task_id=identifier,
            parent_task_id=parent,
            workflow_id=request.workflow_id,
            trace_id=request.trace_id,
            task_type=TaskType.TECHNICAL,
            objective="Analyze bounded technical evidence.",
            context_summary_id=f"summary-{identifier}",
            allowed_data=("context_summary",),
            allowed_tools=(),
            expected_output="technical-v1",
            acceptance_criteria=("Return one report.",),
            difficulty=TaskDifficulty.NORMAL,
            model_tier=ModelTier.TERRA,
            prompt_version="technical-v1",
            attempt_timeout_seconds=30,
            maximum_retries=1,
            reserved_cost=0.05,
            remaining_workflow_cost=0.10,
            analysis_steps=("Inspect evidence.", "Assess setup.", "Return conclusion."),
            escalation_rule="return_to_coordinator",
            conflict_return_rule="return_typed_conflict",
        )

    survivor = task("task-survivor")
    replaced = task("task-replaced")
    replacement = task("task-replacement", parent=replaced.task_id)
    initial = CoordinatorPlan(
        workflow_id=request.workflow_id,
        trace_id=request.trace_id,
        revision=0,
        mode=WorkflowMode.ACTIVE,
        tasks=(survivor, replaced),
    )
    recovered = CoordinatorPlan(
        workflow_id=request.workflow_id,
        trace_id=request.trace_id,
        revision=1,
        mode=WorkflowMode.ACTIVE,
        tasks=(survivor, replacement),
    )

    def report(task_value: AgentTask, status: ReportStatus) -> AgentReport:
        return AgentReport(
            task_id=task_value.task_id,
            workflow_id=request.workflow_id,
            trace_id=request.trace_id,
            status=status,
            knowledge_status=(
                KnowledgeStatus.KNOWN
                if status is ReportStatus.COMPLETED
                else KnowledgeStatus.INSUFFICIENT
            ),
            uncertainty_reason=(None if status is ReportStatus.COMPLETED else "provider_error"),
            error_category=(None if status is ReportStatus.COMPLETED else "provider_error"),
            summary="Bounded task outcome.",
        )

    initial_reports = (
        report(survivor, ReportStatus.COMPLETED),
        report(replaced, ReportStatus.FAILED),
    )
    recovered_reports = (
        report(survivor, ReportStatus.COMPLETED),
        report(replacement, ReportStatus.COMPLETED),
    )
    kernel, _, _, _ = _kernel(tmp_path)
    handle = kernel.create(request)
    _advance_to_running(kernel, handle.run_id)
    observations = ExecutionObservationCollector(
        request.workflow_id,
        request.trace_id,
        checkpoint_sink=lambda checkpoint: kernel.record_checkpoint(handle.run_id, checkpoint),
    )

    CoordinatedWorkflow().invoke(request, WorkflowServices(
        plan=lambda _request: initial,
        dispatch=lambda _plan: initial_reports,
        recover=lambda _plan, _reports: (recovered, recovered_reports),
        decide=lambda *_args: None,
        technical=lambda _reports: None,
        execution_observer=observations,
    ))

    recovery = kernel.checkpoint_history(handle.run_id)[2]
    assert recovery.node is CoreNodeName.RECOVER
    assert recovery.task_ids == (
        survivor.task_id,
        replaced.task_id,
        replacement.task_id,
    )
    items = {item.task_id: item for item in recovery.work_items}
    assert items[replaced.task_id].owner_node is CoreNodeName.DISPATCH
    assert items[replaced.task_id].execution_state == "failed"
    assert items[replacement.task_id].owner_node is CoreNodeName.RECOVER
    assert items[replacement.task_id].dependency_ids == (replaced.task_id,)


def test_harness_rejects_attempt_for_unknown_durable_task(tmp_path):
    kernel, _, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    observations = ExecutionObservationCollector(handle.run_id, handle.trace_id)
    observations.record_attempt(AttemptUsage(
        workflow_id=handle.run_id, trace_id=handle.trace_id,
        task_id="unknown-task", attempt=0, node=CoreNodeName.PLAN,
        provider="openai", provider_request_id="unknown-attempt-0",
        model_id="gpt-5.6-luna", model_tier="luna",
        pricing_version="openai-standard-2026-08-01",
        pricing_model_id="gpt-5.6-luna", pricing_band="short",
        tokens=TokenUsage(input_tokens=1, output_tokens=1),
        estimated_cost_usd=0.0000014, latency_ms=1,
        source="provider_response",
    ))
    item = ObservedWorkItem(
        task_id="known-task", task_kind="technical",
        worker_id="technical-agent", owner_node=CoreNodeName.DISPATCH,
        maximum_retries=0, execution_state="pending",
    )
    observations.checkpoint(
        plan_revision=0, node=CoreNodeName.PLAN,
        outcome=NodeOutcome.COMPLETED, task_ids=("known-task",),
        completed_task_ids=(), failed_task_ids=(),
        retry_state=(TaskRetryState(
            task_id="known-task", attempts_consumed=0,
            retries_consumed=0, retries_remaining=0,
        ),),
        work_items=(item,), action_fingerprint="3" * 64,
    )

    with pytest.raises(HarnessDependencyError, match="unknown durable task"):
        kernel.record_checkpoint(handle.run_id, observations.checkpoints()[-1])


def test_harness_rejects_non_contiguous_attempt_ordinals(tmp_path):
    kernel, _, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    observations = ExecutionObservationCollector(handle.run_id, handle.trace_id)
    for ordinal in (0, 2):
        observations.record_attempt(AttemptUsage(
            workflow_id=handle.run_id, trace_id=handle.trace_id,
            task_id="task-1", attempt=ordinal, node=CoreNodeName.PLAN,
            provider="openai", provider_request_id=f"attempt-{ordinal}",
            model_id="gpt-5.6-luna", model_tier="luna",
            pricing_version="openai-standard-2026-08-01",
            pricing_model_id="gpt-5.6-luna", pricing_band="short",
            tokens=TokenUsage(input_tokens=1, output_tokens=1),
            estimated_cost_usd=0.0000014, latency_ms=1,
            source="provider_response",
        ))
    item = ObservedWorkItem(
        task_id="task-1", task_kind="technical",
        worker_id="technical-agent", owner_node=CoreNodeName.DISPATCH,
        maximum_retries=2, execution_state="running",
        attempt_ids=("attempt-0", "attempt-2"),
    )
    observations.checkpoint(
        plan_revision=0, node=CoreNodeName.PLAN,
        outcome=NodeOutcome.COMPLETED, task_ids=("task-1",),
        completed_task_ids=(), failed_task_ids=(),
        retry_state=(TaskRetryState(
            task_id="task-1", attempts_consumed=2,
            retries_consumed=1, retries_remaining=1,
        ),),
        work_items=(item,), action_fingerprint="4" * 64,
    )

    with pytest.raises(HarnessDependencyError, match="attempt ordinals"):
        kernel.record_checkpoint(handle.run_id, observations.checkpoints()[-1])


def test_harness_rejects_recovery_that_drops_work_or_changes_retry_limit(tmp_path):
    kernel, _, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    observations = ExecutionObservationCollector(handle.run_id, handle.trace_id)
    original = ObservedWorkItem(
        task_id="task-original", task_kind="technical",
        worker_id="technical-agent", owner_node=CoreNodeName.DISPATCH,
        maximum_retries=1, execution_state="pending",
    )
    retry = TaskRetryState(
        task_id="task-original", attempts_consumed=0,
        retries_consumed=0, retries_remaining=1,
    )
    for node in (CoreNodeName.PLAN, CoreNodeName.DISPATCH):
        observations.checkpoint(
            plan_revision=0, node=node, outcome=NodeOutcome.COMPLETED,
            task_ids=("task-original",), completed_task_ids=(),
            failed_task_ids=(), retry_state=(retry,), work_items=(original,),
            action_fingerprint=("5" if node is CoreNodeName.PLAN else "6") * 64,
        )
        kernel.record_checkpoint(handle.run_id, observations.checkpoints()[-1])
    replacement = ObservedWorkItem(
        task_id="task-recovery", task_kind="technical",
        worker_id="technical-agent", owner_node=CoreNodeName.RECOVER,
        maximum_retries=2, execution_state="pending",
    )
    observations.checkpoint(
        plan_revision=1, node=CoreNodeName.RECOVER,
        outcome=NodeOutcome.COMPLETED, task_ids=("task-recovery",),
        completed_task_ids=(), failed_task_ids=(),
        retry_state=(TaskRetryState(
            task_id="task-recovery", attempts_consumed=0,
            retries_consumed=0, retries_remaining=2,
        ),),
        work_items=(replacement,), action_fingerprint="7" * 64,
    )

    with pytest.raises(HarnessDependencyError, match="recovery inventory"):
        kernel.record_checkpoint(handle.run_id, observations.checkpoints()[-1])


def test_harness_rejects_recovery_with_unknown_parent(tmp_path):
    kernel, _, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    observations = ExecutionObservationCollector(handle.run_id, handle.trace_id)
    original = ObservedWorkItem(
        task_id="task-original", task_kind="technical",
        worker_id="technical-agent", owner_node=CoreNodeName.DISPATCH,
        maximum_retries=0, execution_state="pending",
    )
    retry = TaskRetryState(
        task_id="task-original", attempts_consumed=0,
        retries_consumed=0, retries_remaining=0,
    )
    for node in (CoreNodeName.PLAN, CoreNodeName.DISPATCH):
        observations.checkpoint(
            plan_revision=0, node=node, outcome=NodeOutcome.COMPLETED,
            task_ids=("task-original",), completed_task_ids=(),
            failed_task_ids=(), retry_state=(retry,), work_items=(original,),
            action_fingerprint=("8" if node is CoreNodeName.PLAN else "9") * 64,
        )
        kernel.record_checkpoint(handle.run_id, observations.checkpoints()[-1])
    recovery = ObservedWorkItem(
        task_id="task-recovery", task_kind="technical",
        worker_id="technical-agent", owner_node=CoreNodeName.RECOVER,
        dependency_ids=("missing-parent",), maximum_retries=0,
        execution_state="pending",
    )
    observations.checkpoint(
        plan_revision=1, node=CoreNodeName.RECOVER,
        outcome=NodeOutcome.COMPLETED,
        task_ids=("task-original", "task-recovery"),
        completed_task_ids=(), failed_task_ids=(), retry_state=(
            retry,
            TaskRetryState(task_id="task-recovery", attempts_consumed=0,
                           retries_consumed=0, retries_remaining=0),
        ), work_items=(original, recovery), action_fingerprint="a" * 64,
    )

    with pytest.raises(HarnessDependencyError, match="dependency"):
        kernel.record_checkpoint(handle.run_id, observations.checkpoints()[-1])


def test_terminal_checkpoint_permit_forbids_later_checkpoint(tmp_path):
    kernel, _, _, _ = _kernel(tmp_path)
    handle = kernel.create(_request())
    _advance_to_running(kernel, handle.run_id)
    observations = ExecutionObservationCollector(handle.run_id, handle.trace_id)
    observations.record_attempt(AttemptUsage(
        workflow_id=handle.run_id, trace_id=handle.trace_id, task_id="task-1",
        attempt=0, node=CoreNodeName.PLAN, provider="openai",
        provider_request_id="response-over-budget", model_id="gpt-5.6-sol",
        model_tier="sol", pricing_version="openai-standard-2026-08-01",
        pricing_model_id="gpt-5.6-sol", pricing_band="long",
        tokens=TokenUsage(input_tokens=100_000, output_tokens=100_000),
        estimated_cost_usd=3.8, latency_ms=1, source="provider_response",
    ))
    item = ObservedWorkItem(
        task_id="task-1", task_kind="technical", worker_id="technical-agent",
        owner_node=CoreNodeName.DISPATCH, maximum_retries=0,
        execution_state="running", attempt_ids=("response-over-budget",),
    )
    retry = TaskRetryState(
        task_id="task-1", attempts_consumed=1,
        retries_consumed=0, retries_remaining=0,
    )
    observations.checkpoint(
        plan_revision=0, node=CoreNodeName.PLAN,
        outcome=NodeOutcome.COMPLETED, task_ids=("task-1",),
        completed_task_ids=(), failed_task_ids=(), retry_state=(retry,),
        work_items=(item,), action_fingerprint="a" * 64,
    )
    permit = kernel.record_checkpoint(handle.run_id, observations.checkpoints()[-1])
    assert permit.decision is CheckpointDecision.DEGRADE
    observations.checkpoint(
        plan_revision=0, node=CoreNodeName.DISPATCH,
        outcome=NodeOutcome.COMPLETED, task_ids=("task-1",),
        completed_task_ids=(), failed_task_ids=(), retry_state=(retry,),
        work_items=(item,), action_fingerprint="b" * 64,
    )

    with pytest.raises(HarnessDependencyError, match="terminal checkpoint permit"):
        kernel.record_checkpoint(handle.run_id, observations.checkpoints()[-1])


def test_composed_decision_and_reflection_usage_is_node_bound_durable_and_settled(tmp_path):
    """Exercise the real driver/coordinator/graph/Harness composition.

    This is deliberately not a fabricated observation: each attempt is emitted
    by ``AgentDriver`` while the graph checkpoint sink persists it through the
    real Harness.  The test guards the production seam where decision and
    reflection used to inherit the generic ``dispatch`` node and therefore
    could not be admitted to the durable inventory.
    """

    class Clock:
        def now(self):
            return 1.0

        def sleep(self, _seconds):
            raise AssertionError("this bounded happy path must not retry")

    class Audit:
        def record(self, _event):
            return None

    request = _request(trace_id="1" * 32)
    task = AgentTask(
        task_id="technical-1", workflow_id=request.workflow_id,
        trace_id=request.trace_id, task_type=TaskType.TECHNICAL,
        objective="Analyze bounded technical evidence.", context_summary_id="summary-1",
        allowed_data=("context_summary",), allowed_tools=(),
        expected_output="technical-v1", acceptance_criteria=("Return one report.",),
        difficulty=TaskDifficulty.NORMAL, model_tier=ModelTier.TERRA,
        prompt_version="technical-v1", attempt_timeout_seconds=30,
        maximum_retries=0, reserved_cost=0.05, remaining_workflow_cost=0.10,
        analysis_steps=("Read evidence.", "Check setup.", "Return result."),
        escalation_rule="return_to_coordinator", conflict_return_rule="return_typed_conflict",
    )
    plan = CoordinatorPlan(
        workflow_id=request.workflow_id, trace_id=request.trace_id,
        revision=0, mode=WorkflowMode.PASSIVE, tasks=(task,),
    )
    report = AgentReport(
        task_id=task.task_id, workflow_id=request.workflow_id,
        trace_id=request.trace_id, status=ReportStatus.COMPLETED,
        knowledge_status=KnowledgeStatus.KNOWN, uncertainty_reason=None,
        summary="Technical evidence is bounded and current.", evidence_refs=("source-1",),
    )
    decision = DecisionDraft(
        knowledge_status=KnowledgeStatus.KNOWN, uncertainty_reason=None,
        action=Action.NO_TRADE, execute_now=False, decision_confidence=0.8,
    )
    decision_payload = decision.model_dump(mode="json")
    decision_hash = sha256(canonical_json(decision_payload).encode("utf-8")).hexdigest()
    decision_schema_hash = sha256(
        canonical_json(DecisionDraft.model_json_schema()).encode("utf-8")
    ).hexdigest()
    checks = [ObjectiveCheck(code=code, status="pass").model_dump(mode="json") for code in (
        "schema_valid", "numeric_consistency", "direction_consistency",
        "evidence_support", "uncertainty_consistency", "risk_invariants",
    )]
    decision_content = canonical_json({
        "conclusion": "supported",
        "result": decision_payload,
        "evidence_refs": ["source-1"],
    })
    reflection_content = canonical_json({
        "conclusion": "supported",
        "review": {
            "target_hash": decision_hash,
            "output_schema_hash": decision_schema_hash,
            "checks": checks,
        },
    })

    def response(content, tier, request_id):
        tokens = PricedUsageTokens(input_tokens=10, output_tokens=5)
        return ModelResponse(
            content=content,
            usage=AgentUsage(
                input_tokens=10, output_tokens=5,
                cost_usd=float(estimate_workflow_usage_cost(
                    f"gpt-5.6-{tier.value}", "short", tokens,
                )),
                model_tier=tier, provider="openai", provider_request_id=request_id,
                model_id=f"gpt-5.6-{tier.value}",
                pricing_version="openai-standard-2026-08-01",
                pricing_model_id=f"gpt-5.6-{tier.value}", pricing_band="short",
            ),
        )

    class Client:
        def __init__(self):
            self.requests = []
            self.outcomes = [
                response(decision_content, DriverModelTier.TERRA, "decision-response-1"),
                response(reflection_content, DriverModelTier.LUNA, "reflection-response-1"),
            ]

        def invoke(self, value):
            self.requests.append(value)
            return self.outcomes.pop(0)

    kernel, store, _, _ = _kernel(tmp_path)
    handle = kernel.create(request)
    _advance_to_running(kernel, handle.run_id)
    observations = ExecutionObservationCollector(
        request.workflow_id,
        request.trace_id,
        checkpoint_sink=lambda checkpoint: kernel.record_checkpoint(handle.run_id, checkpoint),
    )
    client = Client()
    clock = Clock()
    driver = AgentDriver(
        model_client=client, audit_observer=Audit(), clock=clock,
        random=lambda _low, high: high,
        prompt_releases=PromptReleaseRegistry(releases=(
            profile_for(TaskType.DECISION_PLANNER).release(), reflection_release(),
        )),
        output_schemas=(
            profile_for(TaskType.DECISION_PLANNER).output_schema(),
            reflection_output_schema(),
        ),
        retry_policy=RetryPolicy(max_attempts=2, base_delay=0.01),
        circuit_breaker=CircuitBreaker(failure_threshold=3, cooldown=1.0),
        fallback_policy=FallbackPolicy((
            DriverModelTier.SOL, DriverModelTier.TERRA, DriverModelTier.LUNA,
        )),
        model_costs={
            DriverModelTier.SOL: 0.2,
            DriverModelTier.TERRA: 0.05,
            DriverModelTier.LUNA: 0.01,
        },
        attempt_observer=observations.record_attempt,
    )
    coordinator = AgentCoordinatorServices(
        driver=driver, issuer=CapabilityIssuer(clock=clock.now), tenant_id="tenant-a",
        deadline_epoch=100.0, clock=clock.now,
    )
    result = CoordinatedWorkflow().invoke(request, WorkflowServices(
        plan=lambda _request: plan,
        dispatch=lambda _plan: (report,),
        recover=lambda current, reports: (current, reports),
        decide=coordinator.decide,
        technical=coordinator.technical,
        verify=coordinator.verifier,
        execution_observer=observations,
    ))

    assert result.knowledge_status is KnowledgeStatus.KNOWN
    assert tuple(attempt.node for attempt in observations.usage().attempts) == (
        CoreNodeName.DECIDE, CoreNodeName.REFLECT,
    )
    inventory = kernel.coordinator_task_inventory(handle.run_id)
    attempt_task_ids = tuple(attempt.task_id for attempt in observations.usage().attempts)
    assert attempt_task_ids[0] in inventory
    assert attempt_task_ids[1] in inventory
    assert attempt_task_ids[0] != attempt_task_ids[1]
    assert tuple(item.owner_node for item in observations.checkpoints()[-1].work_items[-2:]) == (
        CoreNodeName.DECIDE, CoreNodeName.REFLECT,
    )

    kernel.advance(
        handle.run_id, candidate={}, workflow_usage=observations.usage(),
        observed_execution=True,
    )
    authority = next(
        event for event in reversed(store.load(handle.run_id))
        if event.event_type == "transition_authorized"
        and "budget_settlement_evidence" in event.payload
    )
    settled = json.loads(authority.payload["budget_settlement_evidence"]["settlement"]["workflow_usage_json"])
    assert [item["node"] for item in settled["attempts"]] == ["decide", "reflect"]
    assert settled["provider_attempt_count"] == 2
