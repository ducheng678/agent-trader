from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, StrictBool, StrictFloat, model_validator

from market_agent.workflow_contracts import (
    ContractModel,
    Digest,
    NonNegativeInt,
    PositiveInt,
    ShortText,
    Text,
    WorkflowMode,
)


PositiveFinite = Annotated[StrictFloat, Field(gt=0.0)]
NonNegativeFinite = Annotated[StrictFloat, Field(ge=0.0)]
FiniteUnit = Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
TargetIds = Annotated[tuple[ShortText, ...], Field(max_length=64)]


class RunState(str, Enum):
    CREATED = "created"
    ADMITTED = "admitted"
    PLANNED = "planned"
    READY = "ready"
    RUNNING = "running"
    RECONCILING = "reconciling"
    WAITING_APPROVAL = "waiting_approval"
    WAITING_RECONCILIATION = "waiting_reconciliation"
    DEGRADING = "degrading"
    SUMMARIZING = "summarizing"
    SUCCEEDED = "succeeded"
    DEGRADED = "degraded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class WorkItemState(str, Enum):
    PENDING = "pending"
    READY = "ready"
    LEASED = "leased"
    RUNNING = "running"
    VALIDATING = "validating"
    SUCCEEDED = "succeeded"
    RETRY_WAIT = "retry_wait"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AttemptState(str, Enum):
    RESERVED = "reserved"
    DISPATCHED = "dispatched"
    STREAMING = "streaming"
    VALIDATING = "validating"
    SETTLING = "settling"
    COMPLETED = "completed"
    TIMED_OUT = "timed_out"
    REJECTED = "rejected"
    FAILED = "failed"
    STALE = "stale"
    CANCELLED = "cancelled"


class OutcomeKind(str, Enum):
    ANSWER = "answer"
    NO_TRADE = "no_trade"
    UNKNOWN = "unknown"
    NONE = "none"


class TaskKind(str, Enum):
    MARKET_CONTEXT = "market_context"
    EVENT_FILTER = "event_filter"
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    DECISION_PLANNER = "decision_planner"
    ESCALATION = "escalation"
    RECONCILIATION = "reconciliation"
    RISK_GATE = "risk_gate"
    PLAYBOOK_ASSEMBLY = "playbook_assembly"
    PRESENTATION_SYNTHESIS = "presentation_synthesis"
    LOCALE_RENDERING = "locale_rendering"
    AUDIT_FINALIZATION = "audit_finalization"
    INFORMATIONAL = "informational"


class RiskClass(str, Enum):
    INFORMATIONAL = "informational"
    TRADING = "trading"
    MANDATORY_EVIDENCE = "mandatory_evidence"
    SIDE_EFFECTING = "side_effecting"


class HarnessOutcome(ContractModel):
    terminal_state: RunState
    outcome_kind: OutcomeKind
    knowledge_status: Literal["known", "partial", "unknown", "not_applicable"]
    terminal_reason: ShortText

    @model_validator(mode="after")
    def validate_terminal_mapping(self) -> HarnessOutcome:
        allowed = {
            (RunState.SUCCEEDED, OutcomeKind.ANSWER, "known"): frozenset(
                {
                    "completed",
                    "fixed_seed_cache_hit",
                    "compatible_semantic_cache_hit",
                }
            ),
            (RunState.SUCCEEDED, OutcomeKind.ANSWER, "partial"): frozenset(
                {
                    "completed",
                    "fixed_seed_cache_hit",
                    "compatible_semantic_cache_hit",
                }
            ),
            (RunState.SUCCEEDED, OutcomeKind.NO_TRADE, "known"): frozenset(
                {"strategy_no_trade", "risk_gate_no_trade"}
            ),
            (RunState.DEGRADED, OutcomeKind.ANSWER, "known"): frozenset(
                {"lower_model_fallback", "verified_local_knowledge_fallback"}
            ),
            (RunState.DEGRADED, OutcomeKind.ANSWER, "partial"): frozenset(
                {"lower_model_fallback", "verified_local_knowledge_fallback"}
            ),
            (RunState.DEGRADED, OutcomeKind.UNKNOWN, "unknown"): frozenset(
                {
                    "insufficient_evidence",
                    "confidence_recovery_exhausted",
                    "dependency_unavailable",
                }
            ),
            (RunState.DEGRADED, OutcomeKind.NO_TRADE, "unknown"): frozenset(
                {"safe_no_trade_due_to_degradation"}
            ),
            (RunState.DEGRADED, OutcomeKind.NO_TRADE, "partial"): frozenset(
                {"safe_no_trade_due_to_degradation"}
            ),
            (RunState.FAILED, OutcomeKind.NONE, "not_applicable"): frozenset(
                {"permanent_policy", "integrity", "audit", "configuration_failure"}
            ),
            (RunState.CANCELLED, OutcomeKind.NONE, "not_applicable"): frozenset(
                {"cancellation_completed"}
            ),
        }
        reasons = allowed.get(
            (self.terminal_state, self.outcome_kind, self.knowledge_status)
        )
        if reasons is None or self.terminal_reason not in reasons:
            raise ValueError("terminal state, outcome, and knowledge status are incompatible")
        return self


class PinnedVersions(ContractModel):
    plan_template_version: ShortText
    policy_version: ShortText
    worker_registry_version: ShortText
    source_registry_version: ShortText
    prompt_bundle_hash: Digest
    tool_registry_hash: Digest
    output_schema_bundle_hash: Digest
    fingerprint_schema_version: ShortText


class StageSpec(ContractModel):
    stage_id: ShortText
    version: ShortText
    entry_predicate: ShortText
    completion_predicate: ShortText
    allowed_task_kinds: Annotated[tuple[TaskKind, ...], Field(min_length=1, max_length=64)]
    dependencies: TargetIds = ()
    maximum_concurrency: PositiveInt
    budget_policy_key: ShortText
    failure_outcome: OutcomeKind
    degradation_outcome: OutcomeKind
    allows_side_effects: StrictBool
    allows_reconciliation: StrictBool

    @model_validator(mode="after")
    def validate_canonical_identifiers(self) -> StageSpec:
        _require_unique(self.allowed_task_kinds, "stage task kinds")
        _require_unique(self.dependencies, "stage dependencies")
        if self.stage_id in self.dependencies:
            raise ValueError("a stage cannot depend on itself")
        return self


class WorkerSpec(ContractModel):
    worker_id: ShortText
    version: ShortText
    supported_task_kinds: Annotated[tuple[TaskKind, ...], Field(min_length=1, max_length=64)]
    analysis_phases: Annotated[tuple[ShortText, ...], Field(min_length=3, max_length=5)]
    input_schema_id: ShortText
    input_schema_hash: Digest
    output_schema_id: ShortText
    output_schema_hash: Digest
    prompt_release: ShortText
    prompt_profile: ShortText
    model_routing_policy_key: ShortText
    context_selector: ShortText
    context_token_budget: PositiveInt
    readable_state_keys: TargetIds = ()
    writable_invocation_state_key: ShortText
    allowed_tool_capabilities: TargetIds = ()
    cacheable: StrictBool
    freshness_class: ShortText
    maximum_turns: PositiveInt
    maximum_tool_calls: NonNegativeInt
    maximum_input_tokens: PositiveInt
    maximum_output_tokens: PositiveInt
    timeout_seconds: PositiveFinite
    maximum_attempts: PositiveInt
    maximum_cost: NonNegativeFinite
    success_outcome: OutcomeKind
    failure_outcome: OutcomeKind
    degradation_outcome: OutcomeKind

    @model_validator(mode="after")
    def validate_canonical_identifiers(self) -> WorkerSpec:
        _require_unique(self.supported_task_kinds, "worker task kinds")
        _require_unique(self.analysis_phases, "analysis phases")
        _require_unique(self.readable_state_keys, "readable state keys")
        _require_unique(self.allowed_tool_capabilities, "tool capabilities")
        if self.writable_invocation_state_key in self.readable_state_keys:
            raise ValueError("the writable invocation-state key cannot also be readable")
        return self


SourceCoverageWeight = tuple[ShortText, Annotated[StrictFloat, Field(gt=0.0, le=1.0)]]


class ProgressTargetSet(ContractModel):
    required_dependency_ids: TargetIds = ()
    required_output_field_paths: TargetIds = ()
    required_evidence_slot_ids: TargetIds = ()
    required_source_coverage_weights: Annotated[
        tuple[SourceCoverageWeight, ...], Field(max_length=64)
    ] = ()
    known_conflict_slot_ids: TargetIds = ()
    risk_invariant_ids: TargetIds = ()

    @model_validator(mode="after")
    def validate_canonical_targets(self) -> ProgressTargetSet:
        for name in (
            "required_dependency_ids",
            "required_output_field_paths",
            "required_evidence_slot_ids",
            "known_conflict_slot_ids",
            "risk_invariant_ids",
        ):
            _require_unique(getattr(self, name), name)
        _require_unique(
            tuple(source_id for source_id, _ in self.required_source_coverage_weights),
            "required source identifiers",
        )
        return self


class WorkItemSpec(ContractModel):
    work_item_id: ShortText
    stage_id: ShortText
    worker_id: ShortText
    task_kind: TaskKind
    objective: Text
    dependencies: TargetIds = ()
    progress_targets: ProgressTargetSet

    @model_validator(mode="after")
    def validate_dependencies(self) -> WorkItemSpec:
        _require_unique(self.dependencies, "work-item dependencies")
        if self.work_item_id in self.dependencies:
            raise ValueError("a work item cannot depend on itself")
        if frozenset(self.dependencies) != frozenset(
            self.progress_targets.required_dependency_ids
        ):
            raise ValueError("progress dependency targets must match work-item dependencies")
        return self


class HarnessPlan(ContractModel):
    plan_id: ShortText
    run_id: ShortText
    trace_id: ShortText
    template_id: ShortText
    revision: NonNegativeInt
    mode: WorkflowMode
    task_kind: TaskKind
    risk_class: RiskClass
    pinned_versions: PinnedVersions
    stages: Annotated[tuple[StageSpec, ...], Field(min_length=1, max_length=64)]
    workers: Annotated[tuple[WorkerSpec, ...], Field(min_length=1, max_length=64)]
    work_items: Annotated[tuple[WorkItemSpec, ...], Field(min_length=1, max_length=256)]
    allows_side_effects: StrictBool

    @model_validator(mode="after")
    def validate_graph(self) -> HarnessPlan:
        stage_ids = tuple(stage.stage_id for stage in self.stages)
        worker_ids = tuple(worker.worker_id for worker in self.workers)
        work_item_ids = tuple(item.work_item_id for item in self.work_items)
        _require_unique(stage_ids, "stage identifiers")
        _require_unique(worker_ids, "worker identifiers")
        _require_unique(work_item_ids, "work-item identifiers")

        _validate_dependency_graph(
            {stage.stage_id: stage.dependencies for stage in self.stages}, "stage"
        )
        _validate_dependency_graph(
            {item.work_item_id: item.dependencies for item in self.work_items},
            "work-item",
        )

        stages = {stage.stage_id: stage for stage in self.stages}
        workers = {worker.worker_id: worker for worker in self.workers}
        for item in self.work_items:
            if item.stage_id not in stages:
                raise ValueError("work items must reference a declared stage")
            if item.worker_id not in workers:
                raise ValueError("work items must reference a declared worker")
            if item.task_kind not in stages[item.stage_id].allowed_task_kinds:
                raise ValueError("work-item kind is not allowed by its stage")
            if item.task_kind not in workers[item.worker_id].supported_task_kinds:
                raise ValueError("work-item kind is not supported by its worker")

        if any(stage.allows_side_effects for stage in self.stages) and not self.allows_side_effects:
            raise ValueError("side-effecting stages require a side-effecting plan")
        if self.mode is WorkflowMode.PASSIVE and self.allows_side_effects:
            raise ValueError("passive plans cannot allow side effects")
        return self


class HarnessTransition(ContractModel):
    run_id: ShortText
    trace_id: ShortText
    entity_kind: Literal["run", "work_item", "attempt"]
    entity_id: ShortText
    from_state: ShortText
    to_state: ShortText
    expected_state_revision: NonNegativeInt
    plan_revision: NonNegativeInt
    reason_code: ShortText
    idempotency_key: ShortText
    lease_epoch: PositiveInt | None = None
    fencing_token_digest: Digest | None = None

    @model_validator(mode="after")
    def validate_lease_evidence(self) -> HarnessTransition:
        has_evidence = (
            self.lease_epoch is not None and self.fencing_token_digest is not None
        )
        if self.entity_kind == "run" and (
            self.lease_epoch is not None or self.fencing_token_digest is not None
        ):
            raise ValueError("run transitions cannot carry lease evidence")
        if self.entity_kind != "run" and not has_evidence:
            raise ValueError("work-item and attempt transitions require lease evidence")
        return self


class ProgressVector(ContractModel):
    completed_dependency_count: NonNegativeInt = 0
    valid_required_field_count: NonNegativeInt = 0
    filled_required_evidence_slot_count: NonNegativeInt = 0
    fresh_authoritative_source_coverage: FiniteUnit = 1.0
    missing_evidence_count: NonNegativeInt = 0
    validation_error_count: NonNegativeInt = 0
    unresolved_conflict_count: NonNegativeInt = 0
    risk_invariant_failure_count: NonNegativeInt = 0


class LeaseToken(ContractModel):
    run_id: ShortText
    work_item_id: ShortText
    attempt_id: ShortText
    lease_epoch: PositiveInt
    fencing_token: ShortText
    holder_id: ShortText
    expires_at_monotonic: PositiveFinite


class TransitionAuthorityRecord(ContractModel):
    """Committed authority for one transition; it contains no live secret."""

    run_id: ShortText
    trace_id: ShortText
    entity_kind: Literal["run", "work_item", "attempt"]
    entity_id: ShortText
    from_state: ShortText
    to_state: ShortText
    expected_state_revision: NonNegativeInt
    plan_revision: NonNegativeInt
    reason_code: ShortText
    idempotency_key: ShortText
    dependency_versions: tuple[tuple[ShortText, NonNegativeInt], ...] = ()
    reservation_id: ShortText | None = None
    grant_id: ShortText | None = None
    lease_epoch: PositiveInt | None = None
    fencing_token_digest: Digest | None = None

    @model_validator(mode="after")
    def validate_authority_shape(self) -> TransitionAuthorityRecord:
        dependency_ids = tuple(identifier for identifier, _ in self.dependency_versions)
        _require_unique(dependency_ids, "authority dependency versions")
        leased = (self.reservation_id, self.grant_id, self.lease_epoch, self.fencing_token_digest)
        if self.entity_kind == "run" and any(value is not None for value in leased):
            raise ValueError("run authority cannot carry reservation or lease evidence")
        if self.entity_kind != "run" and any(value is None for value in leased):
            raise ValueError("non-run authority requires reservation, grant, and lease evidence")
        return self


class AttemptWorkItemOwnershipRecord(ContractModel):
    """Immutable committed relationship from an attempt to its owning work item."""

    run_id: ShortText
    trace_id: ShortText
    attempt_id: ShortText
    work_item_id: ShortText
    plan_revision: NonNegativeInt


class ReconciliationResolutionRecord(ContractModel):
    """Durable broker observation proving an unknown external effect resolved."""

    run_id: ShortText
    trace_id: ShortText
    reconciliation_id: ShortText
    expected_state_revision: NonNegativeInt
    plan_revision: NonNegativeInt
    broker_observation_digest: Digest
    side_effect_resolved: Literal[True]

    def semantic_authority_key(self) -> tuple[str, str, int, int]:
        """Return the run-scoped authority key, excluding the caller-chosen ID."""

        return (
            self.run_id,
            self.trace_id,
            self.plan_revision,
            self.expected_state_revision,
        )


class HarnessSessionView(ContractModel):
    sequence: NonNegativeInt = 0
    state_revision: NonNegativeInt = 0
    plan_revision: NonNegativeInt = 0
    run_id: ShortText | None = None
    trace_id: ShortText | None = None
    request_digest: Digest | None = None
    prompt_release_digest: Digest | None = None
    accepted_result_digest: Digest | None = None
    run_state: RunState | None = None
    outcome: HarnessOutcome | None = None
    work_item_states: tuple[tuple[ShortText, WorkItemState], ...] = ()
    attempt_states: tuple[tuple[ShortText, AttemptState], ...] = ()
    dependency_versions: tuple[tuple[ShortText, NonNegativeInt], ...] = ()
    transition_authorities: tuple[TransitionAuthorityRecord, ...] = ()
    attempt_work_item_owners: tuple[AttemptWorkItemOwnershipRecord, ...] = ()
    reconciliation_resolutions: tuple[ReconciliationResolutionRecord, ...] = ()
    applied_idempotency_keys: TargetIds = ()
    external_side_effect_unknown: StrictBool = False
    last_event_hash: Digest | None = None

    @classmethod
    def empty(cls) -> HarnessSessionView:
        return cls()

    @model_validator(mode="after")
    def validate_identity_and_terminal_outcome(self) -> HarnessSessionView:
        if (self.run_id is None) != (self.trace_id is None):
            raise ValueError("run and trace identity must be present together")
        if self.external_side_effect_unknown and (
            self.run_state is not RunState.WAITING_RECONCILIATION
            or self.outcome is not None
        ):
            raise ValueError(
                "unknown external effects require unsealed waiting reconciliation state"
            )
        if self.outcome is not None and self.run_state is not self.outcome.terminal_state:
            raise ValueError("the sealed outcome must match the current run state")
        _require_unique(tuple(key for key, _ in self.work_item_states), "work-item states")
        _require_unique(tuple(key for key, _ in self.attempt_states), "attempt states")
        _require_unique(
            tuple(key for key, _ in self.dependency_versions), "dependency versions"
        )
        _require_unique(
            tuple(
                (record.entity_kind, record.entity_id, record.expected_state_revision)
                for record in self.transition_authorities
            ),
            "transition authorities",
        )
        _require_unique(
            tuple(record.attempt_id for record in self.attempt_work_item_owners),
            "attempt ownership records",
        )
        _require_unique(
            tuple(
                record.semantic_authority_key()
                for record in self.reconciliation_resolutions
            ),
            "reconciliation resolution records",
        )
        _require_unique(self.applied_idempotency_keys, "idempotency keys")
        return self


def _require_unique(values: tuple[object, ...], description: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"{description} must be unique")


def _validate_dependency_graph(
    dependencies_by_id: dict[str, tuple[str, ...]], description: str
) -> None:
    known_ids = frozenset(dependencies_by_id)
    if any(
        dependency not in known_ids
        for dependencies in dependencies_by_id.values()
        for dependency in dependencies
    ):
        raise ValueError(f"{description} dependencies must reference declared identifiers")

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(identifier: str) -> None:
        if identifier in visiting:
            raise ValueError(f"{description} dependencies must be acyclic")
        if identifier in visited:
            return
        visiting.add(identifier)
        for dependency in dependencies_by_id[identifier]:
            visit(dependency)
        visiting.remove(identifier)
        visited.add(identifier)

    for identifier in dependencies_by_id:
        visit(identifier)
