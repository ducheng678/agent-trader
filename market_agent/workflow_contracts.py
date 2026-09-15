from __future__ import annotations

# Shared workflow primitives and legacy agent contracts live in this module.
# Deterministic harness contracts depend on them one-way from
# ``workflow_harness_contracts`` and are intentionally not re-exported here.

from enum import Enum
from hashlib import sha256
import json
from typing import Annotated, Literal

from pydantic import AfterValidator, BaseModel, BeforeValidator, ConfigDict, Field, JsonValue, PlainSerializer, StrictBool, StrictFloat, StrictInt, StringConstraints, field_validator, model_validator


Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4000)]
ShortText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
FiniteUnit = Annotated[StrictFloat, Field(ge=0.0, le=1.0)]
PositiveFinite = Annotated[StrictFloat, Field(gt=0.0)]
NonNegativeFinite = Annotated[StrictFloat, Field(ge=0.0)]
PositiveInt = Annotated[StrictInt, Field(gt=0)]
NonNegativeInt = Annotated[StrictInt, Field(ge=0)]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class FrozenDict(dict[str, JsonValue]):
    def __readonly(self, *args: object, **kwargs: object) -> None:
        raise TypeError("workflow request values are immutable")

    __setitem__ = __readonly
    __delitem__ = __readonly
    __ior__ = __readonly
    clear = __readonly
    pop = __readonly
    popitem = __readonly
    setdefault = __readonly
    update = __readonly

    def copy(self) -> FrozenDict:
        return FrozenDict(self)


def thaw_json(value: object) -> object:
    """Restore frozen JSON values before Pydantic validates a model again."""

    if isinstance(value, dict):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json(item) for item in value]
    return value


def freeze_json(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return FrozenDict({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    return value


FrozenJsonMapping = Annotated[
    dict[str, JsonValue],
    BeforeValidator(thaw_json),
    AfterValidator(freeze_json),
    PlainSerializer(thaw_json, return_type=dict, when_used="always"),
]


class ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False, str_strip_whitespace=True, revalidate_instances="always")
    schema_version: Literal["v1"] = "v1"


class KnowledgeStatus(str, Enum):
    KNOWN = "known"
    INSUFFICIENT = "insufficient"


class TaskType(str, Enum):
    MARKET_CONTEXT = "market_context"
    EVENT_FILTER = "event_filter"
    FUNDAMENTAL = "fundamental"
    TECHNICAL = "technical"
    DECISION_PLANNER = "decision_planner"
    ESCALATION = "escalation"
    RECONCILIATION = "reconciliation"


class TaskDifficulty(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class ModelTier(str, Enum):
    LUNA = "gpt-5.6-luna"
    TERRA = "gpt-5.6-terra"
    SOL = "gpt-5.6-sol"


class ReportStatus(str, Enum):
    COMPLETED = "completed"
    UNCERTAIN = "uncertain"
    CONFLICT = "conflict"
    FAILED = "failed"


class Action(str, Enum):
    LONG = "long"
    SHORT = "short"
    NO_TRADE = "no_trade"


class EventRelevance(str, Enum):
    RELEVANT = "relevant"
    UNRELATED = "unrelated"
    DUPLICATE = "duplicate"
    UNKNOWN = "unknown"


class EventAlignment(str, Enum):
    REINFORCES = "reinforces"
    WEAKENS = "weakens"
    CHANGES = "changes"
    NOT_APPLICABLE = "not_applicable"
    UNKNOWN = "unknown"


class MarketRegime(str, Enum):
    UPTREND = "uptrend"
    DOWNTREND = "downtrend"
    RANGE = "range"
    TRANSITION = "transition"
    INSUFFICIENT_DATA = "insufficient_data"


class ExtensionState(str, Enum):
    UPSIDE_EXTENDED = "upside_extended"
    DOWNSIDE_EXTENDED = "downside_extended"
    BALANCED = "balanced"
    INSUFFICIENT_DATA = "insufficient_data"


class DataQuality(str, Enum):
    GOOD = "good"
    DEGRADED = "degraded"
    INSUFFICIENT = "insufficient"


class ReviewDecision(str, Enum):
    APPROVE = "approve"
    REVISE = "revise"
    REJECT = "reject"


class WorkflowMode(str, Enum):
    ACTIVE = "active"
    PASSIVE = "passive"


class TerminalMode(str, Enum):
    PLAYBOOK = "playbook"
    NO_TRADE = "no_trade"
    UNKNOWN = "unknown"
    INFORMATIONAL = "informational"


class SummaryCompleteness(str, Enum):
    COMPLETE = "complete"
    INCOMPLETE = "incomplete"


class KnowledgeBoundContract(ContractModel):
    knowledge_status: KnowledgeStatus
    uncertainty_reason: Text | None

    @model_validator(mode="after")
    def validate_knowledge_status(self) -> KnowledgeBoundContract:
        if self.knowledge_status is KnowledgeStatus.INSUFFICIENT and self.uncertainty_reason is None:
            raise ValueError("insufficient knowledge requires an uncertainty reason")
        if self.knowledge_status is KnowledgeStatus.KNOWN and self.uncertainty_reason is not None:
            raise ValueError("known knowledge cannot include an uncertainty reason")
        return self


class WorkflowRequest(ContractModel):
    workflow_id: ShortText
    trace_id: ShortText
    user_query: Text
    event_tape: tuple[FrozenJsonMapping, ...] = Field(default_factory=tuple, max_length=200)
    trigger_reason: Text
    trigger_event: FrozenJsonMapping | None = None
    recent_events: tuple[FrozenJsonMapping, ...] = Field(default_factory=tuple, max_length=50)
    trade_symbol_context: FrozenJsonMapping | None = None
    active_symbol: ShortText | None = None
    has_live_position: StrictBool = False
    prefetched_passive_event_judge: FrozenJsonMapping | None = None

    @field_validator("event_tape", "recent_events", mode="before")
    @classmethod
    def freeze_event_sequences(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(value)
        return value


def canonical_workflow_request_digest(request: WorkflowRequest) -> str:
    """Bind the complete validated ingress request to canonical JSON bytes."""

    request = WorkflowRequest.model_validate(request)
    payload = json.dumps(
        request.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def canonical_workflow_result_digest(result: WorkflowResult) -> str:
    """Bind the complete validated workflow result to canonical JSON bytes."""

    result = WorkflowResult.model_validate(result)
    payload = json.dumps(
        result.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return sha256(payload).hexdigest()


class SourceFact(ContractModel):
    source_id: ShortText
    observed_at: ShortText
    fact: Text


class OmittedSection(ContractModel):
    section: ShortText
    count: NonNegativeInt


class ContextSummary(ContractModel):
    summary_id: ShortText
    task_id: ShortText
    workflow_id: ShortText
    trace_id: ShortText
    user_objective: Text
    immutable_constraints: tuple[Text, ...] = Field(default_factory=tuple, max_length=20)
    market_facts: tuple[SourceFact, ...] = Field(default_factory=tuple, max_length=30)
    prior_conclusions: tuple[Text, ...] = Field(default_factory=tuple, max_length=20)
    unresolved_questions: tuple[Text, ...] = Field(default_factory=tuple, max_length=20)
    conflicts: tuple[Text, ...] = Field(default_factory=tuple, max_length=20)
    omitted_sections: tuple[OmittedSection, ...] = Field(default_factory=tuple, max_length=20)
    token_estimate: NonNegativeInt
    completeness: SummaryCompleteness
    summary_version: ShortText
    summarizer_model: ModelTier | None = None
    source_record_hash: Digest
    source_references: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=50)

    @model_validator(mode="after")
    def validate_completeness(self) -> ContextSummary:
        if self.completeness is SummaryCompleteness.COMPLETE and self.omitted_sections:
            raise ValueError("complete summaries cannot omit sections")
        if self.completeness is SummaryCompleteness.INCOMPLETE and not self.omitted_sections:
            raise ValueError("incomplete summaries must identify omitted sections")
        return self


class AgentTask(ContractModel):
    task_id: ShortText
    parent_task_id: ShortText | None = None
    workflow_id: ShortText
    trace_id: ShortText
    task_type: TaskType
    objective: Text
    context_summary_id: ShortText
    allowed_data: tuple[ShortText, ...] = Field(max_length=20)
    allowed_tools: tuple[ShortText, ...] = Field(max_length=5)
    expected_output: ShortText
    acceptance_criteria: tuple[Text, ...] = Field(min_length=1, max_length=10)
    difficulty: TaskDifficulty
    model_tier: ModelTier
    prompt_version: ShortText
    cache_key: ShortText | None = None
    attempt_timeout_seconds: PositiveInt
    maximum_retries: Annotated[NonNegativeInt, Field(le=3)]
    reserved_cost: NonNegativeFinite
    remaining_workflow_cost: NonNegativeFinite
    analysis_steps: tuple[Text, ...] = Field(min_length=3, max_length=5)
    escalation_rule: ShortText
    conflict_return_rule: ShortText

    @model_validator(mode="after")
    def validate_reserved_cost(self) -> AgentTask:
        if self.reserved_cost > self.remaining_workflow_cost:
            raise ValueError("reserved cost cannot exceed remaining workflow cost")
        return self


class AgentReport(KnowledgeBoundContract):
    task_id: ShortText
    workflow_id: ShortText
    trace_id: ShortText
    status: ReportStatus
    summary: Text
    evidence_refs: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=20)
    disputed_claims: tuple[Text, ...] = Field(default_factory=tuple, max_length=10)
    missing_evidence: tuple[Text, ...] = Field(default_factory=tuple, max_length=10)
    error_category: ShortText | None = None
    retryable: StrictBool = False
    consumed_cost: NonNegativeFinite = 0.0
    safe_fallback: Action | None = None

    @model_validator(mode="after")
    def validate_report_status(self) -> AgentReport:
        if self.status is ReportStatus.COMPLETED and self.knowledge_status is not KnowledgeStatus.KNOWN:
            raise ValueError("completed reports require known knowledge")
        if self.status in {ReportStatus.UNCERTAIN, ReportStatus.CONFLICT, ReportStatus.FAILED} and self.knowledge_status is not KnowledgeStatus.INSUFFICIENT:
            raise ValueError("non-completed reports require insufficient knowledge")
        if self.status is ReportStatus.CONFLICT and not self.disputed_claims:
            raise ValueError("conflict reports require disputed claims")
        if self.status is ReportStatus.FAILED and self.error_category is None:
            raise ValueError("failed reports require an error category")
        if (self.status is not ReportStatus.COMPLETED or self.knowledge_status is KnowledgeStatus.INSUFFICIENT) and self.safe_fallback not in {None, Action.NO_TRADE}:
            raise ValueError("uncertain reports can only use no_trade as a safe fallback")
        return self


class EventAssessment(KnowledgeBoundContract):
    relevance: EventRelevance
    impact_confidence: FiniteUnit
    material_change: StrictBool
    reason_codes: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=5)

    @model_validator(mode="after")
    def validate_event_assessment(self) -> EventAssessment:
        if self.knowledge_status is KnowledgeStatus.INSUFFICIENT and self.relevance is not EventRelevance.UNKNOWN:
            raise ValueError("insufficient event knowledge requires unknown relevance")
        if self.relevance is EventRelevance.UNKNOWN and self.knowledge_status is not KnowledgeStatus.INSUFFICIENT:
            raise ValueError("unknown relevance requires insufficient knowledge")
        return self


class FundamentalAnalysis(KnowledgeBoundContract):
    action: Action
    direction_confidence: FiniteUnit
    primary_driver: Text
    supporting_factors: tuple[Text, ...] = Field(default_factory=tuple, max_length=5)
    contradicting_factors: tuple[Text, ...] = Field(default_factory=tuple, max_length=5)
    event_alignment: EventAlignment

    @model_validator(mode="after")
    def validate_fundamental_analysis(self) -> FundamentalAnalysis:
        if self.knowledge_status is KnowledgeStatus.INSUFFICIENT and self.action is not Action.NO_TRADE:
            raise ValueError("insufficient fundamental knowledge requires no_trade")
        return self


class TradeSetup(ContractModel):
    viable: StrictBool
    confidence: FiniteUnit
    entry_price: PositiveFinite | None = None
    stop_price: PositiveFinite | None = None
    observation_low: PositiveFinite | None = None
    observation_high: PositiveFinite | None = None
    candidate_condition: Text | None = None

    @model_validator(mode="after")
    def validate_trade_setup(self) -> TradeSetup:
        values = (self.entry_price, self.stop_price, self.observation_low, self.observation_high, self.candidate_condition)
        if self.viable and any(value is None for value in values):
            raise ValueError("viable setups require complete entry, stop, observation, and condition values")
        if not self.viable and any(value is not None for value in values):
            raise ValueError("nonviable setups cannot carry execution values")
        if self.observation_low is not None and self.observation_high is not None and self.observation_low >= self.observation_high:
            raise ValueError("observation range must be ordered")
        return self


class TechnicalAnalysis(KnowledgeBoundContract):
    current_price: PositiveFinite | None = None
    market_regime: MarketRegime
    extension_state: ExtensionState
    long_setup: TradeSetup
    short_setup: TradeSetup
    data_quality: DataQuality

    @model_validator(mode="after")
    def validate_technical_analysis(self) -> TechnicalAnalysis:
        if self.knowledge_status is KnowledgeStatus.INSUFFICIENT and self.data_quality is not DataQuality.INSUFFICIENT:
            raise ValueError("insufficient technical knowledge requires insufficient data quality")
        if self.data_quality is DataQuality.INSUFFICIENT and self.knowledge_status is not KnowledgeStatus.INSUFFICIENT:
            raise ValueError("insufficient technical data requires insufficient knowledge")
        if self.market_regime is MarketRegime.INSUFFICIENT_DATA and self.knowledge_status is not KnowledgeStatus.INSUFFICIENT:
            raise ValueError("insufficient market regime requires insufficient knowledge")
        if self.extension_state is ExtensionState.INSUFFICIENT_DATA and self.knowledge_status is not KnowledgeStatus.INSUFFICIENT:
            raise ValueError("insufficient extension state requires insufficient knowledge")
        if self.knowledge_status is KnowledgeStatus.INSUFFICIENT and (self.long_setup.viable or self.short_setup.viable):
            raise ValueError("insufficient technical data requires nonviable setups")
        return self


class ObservationScenario(ContractModel):
    lower_bound: PositiveFinite
    upper_bound: PositiveFinite
    condition: Text
    timeout_seconds: PositiveInt

    @model_validator(mode="after")
    def validate_bounds(self) -> ObservationScenario:
        if self.lower_bound >= self.upper_bound:
            raise ValueError("observation bounds must be ordered")
        return self


class DecisionDraft(KnowledgeBoundContract):
    action: Action
    execute_now: StrictBool
    entry_price: PositiveFinite | None = None
    stop_price: PositiveFinite | None = None
    observation_scenario: ObservationScenario | None = None
    decision_confidence: FiniteUnit
    selected_setup: ShortText | None = None
    conflict_codes: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=5)

    @model_validator(mode="after")
    def validate_decision_draft(self) -> DecisionDraft:
        if self.knowledge_status is KnowledgeStatus.INSUFFICIENT and self.action is not Action.NO_TRADE:
            raise ValueError("insufficient decision knowledge requires no_trade")
        if self.action is Action.NO_TRADE:
            if self.execute_now or self.entry_price is not None or self.stop_price is not None or self.observation_scenario is not None:
                raise ValueError("no_trade decisions cannot include execution values")
            return self
        if self.entry_price is None or self.stop_price is None or self.selected_setup is None:
            raise ValueError("trade decisions require entry, stop, and selected setup")
        if self.execute_now and self.observation_scenario is not None:
            raise ValueError("immediate execution cannot include an observation scenario")
        if not self.execute_now and self.observation_scenario is None:
            raise ValueError("deferred execution requires an observation scenario")
        return self


class EscalationReview(KnowledgeBoundContract):
    decision: ReviewDecision
    revision: DecisionDraft | None = None
    resolved_conflict_codes: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=5)
    confidence: FiniteUnit
    reason: Text

    @model_validator(mode="after")
    def validate_escalation_review(self) -> EscalationReview:
        if self.decision is ReviewDecision.REVISE and self.revision is None:
            raise ValueError("revision decisions require a complete revision")
        if self.decision is not ReviewDecision.REVISE and self.revision is not None:
            raise ValueError("only revision decisions can include a revision")
        if self.knowledge_status is KnowledgeStatus.INSUFFICIENT and self.decision is not ReviewDecision.REJECT:
            raise ValueError("insufficient escalation knowledge requires rejection")
        return self


class MarketContextResult(KnowledgeBoundContract):
    facts: tuple[SourceFact, ...] = Field(default_factory=tuple, max_length=30)
    source_references: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=50)


class RiskAssessment(KnowledgeBoundContract):
    accepted: StrictBool
    reason_codes: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=10)
    requires_escalation: StrictBool = False

    @model_validator(mode="after")
    def validate_risk_assessment(self) -> RiskAssessment:
        if self.knowledge_status is KnowledgeStatus.INSUFFICIENT and self.accepted:
            raise ValueError("insufficient risk knowledge cannot accept a decision")
        return self


class CoordinatorPlan(ContractModel):
    workflow_id: ShortText
    trace_id: ShortText
    revision: NonNegativeInt
    mode: WorkflowMode
    tasks: tuple[AgentTask, ...] = Field(min_length=1, max_length=10)
    unresolved_conflicts: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=10)

    @model_validator(mode="after")
    def validate_task_identity(self) -> CoordinatorPlan:
        task_ids = tuple(task.task_id for task in self.tasks)
        if len(task_ids) != len(set(task_ids)):
            raise ValueError("coordinator plans cannot contain duplicate task identifiers")
        if any(task.workflow_id != self.workflow_id or task.trace_id != self.trace_id for task in self.tasks):
            raise ValueError("planned tasks must share the plan workflow and trace")
        return self


class InformationalAnswer(KnowledgeBoundContract):
    answer: Text
    source_references: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=20)


class CachedAnswer(ContractModel):
    cache_key: ShortText
    answer: InformationalAnswer
    source_version: ShortText


class WorkflowBudgetState(ContractModel):
    mode: WorkflowMode
    elapsed_seconds: NonNegativeFinite
    time_cap_seconds: PositiveFinite | None = None
    maximum_attempts: PositiveInt | None = None
    cost_cap: NonNegativeFinite | None = None
    remaining_cost: NonNegativeFinite
    reserved_cost: NonNegativeFinite
    settled_cost: NonNegativeFinite
    remaining_attempts: NonNegativeInt

    @model_validator(mode="before")
    @classmethod
    def set_mode_caps(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        values = dict(value)
        caps = {
            WorkflowMode.ACTIVE: (300.0, 10, 0.75),
            WorkflowMode.PASSIVE: (130.0, 10, 0.30),
        }
        mode = values.get("mode")
        if isinstance(mode, str):
            try:
                mode = WorkflowMode(mode)
            except ValueError:
                return values
        if mode in caps:
            time_cap, maximum_attempts, cost_cap = caps[mode]
            values.setdefault("time_cap_seconds", time_cap)
            values.setdefault("maximum_attempts", maximum_attempts)
            values.setdefault("cost_cap", cost_cap)
        return values

    @model_validator(mode="after")
    def validate_workflow_caps(self) -> WorkflowBudgetState:
        caps = {
            WorkflowMode.ACTIVE: (300.0, 10, 0.75),
            WorkflowMode.PASSIVE: (130.0, 10, 0.30),
        }
        expected_time, expected_attempts, expected_cost = caps[self.mode]
        if (self.time_cap_seconds, self.maximum_attempts, self.cost_cap) != (expected_time, expected_attempts, expected_cost):
            raise ValueError("workflow budget caps must match the selected mode")
        if self.elapsed_seconds > expected_time:
            raise ValueError("workflow elapsed time exceeds its cap")
        if self.remaining_attempts > expected_attempts:
            raise ValueError("remaining attempts exceed the workflow cap")
        if self.remaining_cost > expected_cost or self.reserved_cost + self.settled_cost > expected_cost:
            raise ValueError("workflow cost exceeds its cap")
        if self.remaining_cost + self.reserved_cost + self.settled_cost > expected_cost:
            raise ValueError("workflow cost transition exceeds its cap")
        return self


class WorkflowError(ContractModel):
    category: ShortText
    message: Text
    retryable: StrictBool


class WorkflowResult(KnowledgeBoundContract):
    workflow_id: ShortText
    trace_id: ShortText
    terminal_mode: TerminalMode
    final_action: Action
    evidence_references: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=50)
    route_history: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=50)
    playbook_payload: DecisionDraft | None = None
    informational_answer: InformationalAnswer | None = None

    @model_validator(mode="after")
    def validate_workflow_result(self) -> WorkflowResult:
        if self.knowledge_status is KnowledgeStatus.INSUFFICIENT and self.final_action is not Action.NO_TRADE:
            raise ValueError("insufficient workflow knowledge requires no_trade")
        if self.terminal_mode is TerminalMode.PLAYBOOK and self.playbook_payload is None:
            raise ValueError("playbook results require a playbook payload")
        if self.terminal_mode is TerminalMode.INFORMATIONAL and self.informational_answer is None:
            raise ValueError("informational results require an informational answer")
        if self.terminal_mode in {TerminalMode.NO_TRADE, TerminalMode.UNKNOWN} and self.final_action is not Action.NO_TRADE:
            raise ValueError("safe terminal results require no_trade")
        if self.terminal_mode is TerminalMode.UNKNOWN and self.knowledge_status is not KnowledgeStatus.INSUFFICIENT:
            raise ValueError("unknown results require insufficient knowledge")
        if self.knowledge_status is KnowledgeStatus.INSUFFICIENT and self.terminal_mode not in {TerminalMode.NO_TRADE, TerminalMode.UNKNOWN}:
            raise ValueError("insufficient workflow knowledge requires a safe terminal mode")
        return self
