from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterable, Literal, Protocol

from pydantic import AfterValidator, Field, StringConstraints, field_validator, model_validator
from typing import Annotated

from market_agent.workflow_contracts import ContractModel, Digest, NonNegativeFinite, NonNegativeInt, PositiveInt, ShortText


_MAX_PAGE_SIZE = 100
_MAX_PAYLOAD_BYTES = 4096
_UNSAFE_VALUE = re.compile(r"(?:authorization|bearer|cookie|api[ _-]?key|credential|secret|token|password|private[ _-]?key|raw[ _-]?prompt|system[ _-]?prompt|reasoning|instruction|ignore.*previous|chain[ _-]?of[ _-]?thought|(?:^|[^a-z0-9])sk-[a-z0-9]|-----BEGIN|eyJ[a-zA-Z0-9_-]*\.|https?://\S+[?&](?:token|key|secret|signature)=)", re.IGNORECASE)
_OPAQUE_ID = re.compile(r"^(?!sk-)(?!eyJ)[a-z][a-z0-9_-]{0,63}$")
_CODE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_PROMPT_VERSION = re.compile(r"^(?:(?:prompt|release)-v[0-9]+(?:\.[0-9]+){0,2}(?:-[a-z][a-z0-9]*)?|legacy_identifier)$")
_SCHEMA_NAME = re.compile(r"^(?:[A-Z][A-Za-z0-9]{0,63}|[a-z][a-z0-9]*(?:_[a-z0-9]+){1,15}|legacy_identifier)$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_LEGACY_HASH_POLICY = "null_noncanonical_v1"
_LEGACY_SEMANTIC_NAMESPACES = ("c96_", "old_")


class AuditUnavailableError(RuntimeError):
    pass


def _require_safe_text(value: str) -> str:
    compact = re.sub(r"[^a-z0-9]+", "", value.casefold())
    compact_markers = ("rawprompt", "systemprompt", "privatereasoning", "chainofthought", "apikey", "privatekey")
    secret_prefix = re.search(r"(?:^|[^a-z0-9])sk[._/:-]+(?:live(?:[._/:-]+)?)?[a-z0-9]", value, re.IGNORECASE)
    pem_variant = compact.startswith("begin") and "privatekey" in compact
    compact_secret_prefix = compact.startswith(("sklive", "skprod", "skproj", "sktest"))
    if _UNSAFE_VALUE.search(value) or any(marker in compact for marker in compact_markers) or secret_prefix or compact_secret_prefix or pem_variant:
        raise ValueError("audit values cannot contain credentials, authorization data, or URL secrets")
    return value


def _require_id(value: str) -> str:
    _require_safe_text(value)
    if not _OPAQUE_ID.fullmatch(value):
        raise ValueError("audit opaque identifiers must be bounded non-secret identifiers")
    return value


def _require_code(value: str) -> str:
    _require_safe_text(value)
    if not _CODE.fullmatch(value):
        raise ValueError("audit codes must be compact identifiers, never prose or URLs")
    return value


def _require_prompt_version(value: str) -> str:
    _require_safe_text(value)
    if not _PROMPT_VERSION.fullmatch(value):
        raise ValueError("audit prompt versions must use the release identifier grammar")
    return value


def _require_schema_name(value: str) -> str:
    _require_safe_text(value)
    if not _SCHEMA_NAME.fullmatch(value):
        raise ValueError("audit schema names must use the schema identifier grammar")
    return value


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("audit timestamps must be UTC")
    return value


AuditEventId = Annotated[str, StringConstraints(strip_whitespace=True), AfterValidator(_require_id)]
AuditTraceId = Annotated[str, StringConstraints(strip_whitespace=True), AfterValidator(_require_id)]
AuditWorkflowId = Annotated[str, StringConstraints(strip_whitespace=True), AfterValidator(_require_id)]
AuditTaskId = Annotated[str, StringConstraints(strip_whitespace=True), AfterValidator(_require_id)]
AuditAttemptId = Annotated[str, StringConstraints(strip_whitespace=True), AfterValidator(_require_id)]
AuditSourceReference = Annotated[str, StringConstraints(strip_whitespace=True), AfterValidator(_require_id)]
AuditSubjectId = Annotated[str, StringConstraints(strip_whitespace=True), AfterValidator(_require_id)]
AuditCode = Annotated[str, StringConstraints(strip_whitespace=True), AfterValidator(_require_code)]
AuditPromptVersion = Annotated[str, StringConstraints(strip_whitespace=True), AfterValidator(_require_prompt_version)]
AuditSchemaName = Annotated[str, StringConstraints(strip_whitespace=True), AfterValidator(_require_schema_name)]


class AuditActor(str, Enum):
    COORDINATOR = "coordinator"
    AUDIT_STORE = "audit_store"
    CONTEXT_SELECTOR = "context_selector"
    CONTEXT_SUMMARIZER = "context_summarizer"
    SPECIALIST = "specialist"
    MODEL = "model"
    TOOL = "tool"
    QUEUE = "queue"
    MEMORY = "memory"
    EXCHANGE = "exchange"
    INGRESS = "ingress"
    NORMALIZER = "normalizer"
    CLASSIFIER = "classifier"
    TASK_PLANNER = "task_planner"
    TASK_DISPATCHER = "task_dispatcher"
    SCHEDULER = "scheduler"
    MODEL_ROUTER = "model_router"
    PROMPT_BUILDER = "prompt_builder"
    SCHEMA_VALIDATOR = "schema_validator"
    CACHE = "cache"
    RETRY_CONTROLLER = "retry_controller"
    CIRCUIT_BREAKER = "circuit_breaker"
    BUDGET_CONTROLLER = "budget_controller"
    KNOWLEDGE_STORE = "knowledge_store"
    FALLBACK = "fallback"
    CONFLICT_RESOLVER = "conflict_resolver"
    REFLECTOR = "reflector"
    CORRECTOR = "corrector"
    RISK_MANAGER = "risk_manager"
    FINALIZER = "finalizer"
    TRACER = "tracer"
    PROMPT_RELEASE_MANAGER = "prompt_release_manager"
    EVALUATOR = "evaluator"
    LEGACY = "legacy_actor"


class AuditEventType(str, Enum):
    CREATED = "created"
    TASK_DISPATCHED = "task_dispatched"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    VALIDATION_COMPLETED = "validation_completed"
    CONTEXT_SELECTED = "context_selected"
    CONTEXT_SUMMARIZED = "context_summarized"
    LEGACY_MIGRATED = "legacy_migrated"
    DISPATCH_BLOCKED = "dispatch_blocked"
    EXTERNAL_DISPATCH = "external_dispatch"
    INGRESS_RECEIVED = "ingress_received"
    REQUEST_NORMALIZED = "request_normalized"
    EVENT_CLASSIFIED = "event_classified"
    TASK_PLAN_CREATED = "task_plan_created"
    TASK_DECOMPOSED = "task_decomposed"
    TASK_RESCHEDULED = "task_rescheduled"
    MODEL_ROUTED = "model_routed"
    PROMPT_COMPOSED = "prompt_composed"
    SCHEMA_VALIDATED = "schema_validated"
    FIXED_CACHE_HIT = "fixed_cache_hit"
    FIXED_CACHE_MISS = "fixed_cache_miss"
    FIXED_CACHE_WRITE = "fixed_cache_write"
    SEMANTIC_CACHE_HIT = "semantic_cache_hit"
    SEMANTIC_CACHE_MISS = "semantic_cache_miss"
    SEMANTIC_CACHE_WRITE = "semantic_cache_write"
    PROMPT_CACHE_HIT = "prompt_cache_hit"
    PROMPT_CACHE_MISS = "prompt_cache_miss"
    PROMPT_CACHE_WRITE = "prompt_cache_write"
    TOOL_DISPATCHED = "tool_dispatched"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"
    RETRY_SCHEDULED = "retry_scheduled"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"
    BACKOFF_SCHEDULED = "backoff_scheduled"
    CIRCUIT_OPENED = "circuit_opened"
    CIRCUIT_CLOSED = "circuit_closed"
    CIRCUIT_PROBE = "circuit_probe"
    CIRCUIT_RECORDED = "circuit_recorded"
    MODEL_DOWNGRADED = "model_downgraded"
    ABSTAINED = "abstained"
    CORE_RESULT_READY = "core_result_ready"
    BUDGET_EXHAUSTED = "budget_exhausted"
    LOCAL_KNOWLEDGE_RETRIEVED = "local_knowledge_retrieved"
    FALLBACK_SELECTED = "fallback_selected"
    CONFLICT_DETECTED = "conflict_detected"
    REFLECTION_COMPLETED = "reflection_completed"
    CORRECTION_APPLIED = "correction_applied"
    RISK_EVALUATED = "risk_evaluated"
    FINAL_DECISION = "final_decision"
    MEMORY_CREATED = "memory_created"
    MEMORY_UPDATED = "memory_updated"
    MEMORY_EXPIRED = "memory_expired"
    MEMORY_DELETED = "memory_deleted"
    MEMORY_RETRIEVED = "memory_retrieved"
    MEMORY_PROMOTED = "memory_promoted"
    TRACE_STARTED = "trace_started"
    TRACE_COMPLETED = "trace_completed"
    SPAN_STARTED = "span_started"
    SPAN_COMPLETED = "span_completed"
    PROMPT_RELEASED = "prompt_released"
    PROMPT_ROLLED_BACK = "prompt_rolled_back"
    EVALUATION_STARTED = "evaluation_started"
    EVALUATION_COMPLETED = "evaluation_completed"
    EVALUATION_FAILED = "evaluation_failed"
    LEGACY = "legacy_event"


class AuditStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    OMITTED = "omitted"
    RECEIVED = "received"
    NORMALIZED = "normalized"
    CLASSIFIED = "classified"
    PLANNED = "planned"
    DISPATCHED = "dispatched"
    RESCHEDULED = "rescheduled"
    HIT = "hit"
    MISS = "miss"
    TIMED_OUT = "timed_out"
    CANCELLED = "cancelled"
    OPEN = "open"
    PROMOTED = "promoted"
    RUNNING = "running"
    ROLLED_BACK = "rolled_back"
    PASSED = "passed"
    LEGACY = "legacy_status"


class AuditModel(str, Enum):
    LUNA = "gpt-5.6-luna"
    TERRA = "gpt-5.6-terra"
    SOL = "gpt-5.6-sol"
    LEGACY = "legacy_model"


class AuditOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COMPLETED = "completed"
    OMITTED = "omitted"
    LEGACY_PAYLOAD = "legacy_payload"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    HIT = "hit"
    MISS = "miss"
    ROUTED = "routed"
    SELECTED = "selected"
    RESCHEDULED = "rescheduled"
    CANCELLED = "cancelled"
    TIMED_OUT = "timed_out"
    OPENED = "opened"
    CLOSED = "closed"
    EXHAUSTED = "exhausted"
    RETRIEVED = "retrieved"
    PROMOTED = "promoted"
    RESOLVED = "resolved"
    CORRECTED = "corrected"
    APPROVED = "approved"
    ROLLED_BACK = "rolled_back"
    PASSED = "passed"


class AuditReason(str, Enum):
    VALIDATION_ERROR = "validation_error"
    BUDGET_LIMIT = "budget_limit"
    SOURCE_LIMIT = "source_limit"
    UNRESOLVED_CONFLICT = "unresolved_conflict"
    MISSING_EVIDENCE = "missing_evidence"
    LEGACY_SCHEMA = "legacy_schema"
    AUDIT_FAILURE = "audit_failure"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    RETRYABLE_ERROR = "retryable_error"
    TIMEOUT = "timeout"
    CANCELLATION = "cancellation"
    BACKOFF = "backoff"
    CIRCUIT_OPEN = "circuit_open"
    BUDGET_EXHAUSTED = "budget_exhausted"
    LOCAL_KNOWLEDGE = "local_knowledge"
    FALLBACK = "fallback"
    REFLECTION_FAILURE = "reflection_failure"
    REFLECTION_REQUIRED = "reflection_required"
    RISK_REJECTED = "risk_rejected"
    PROMPT_ROLLBACK = "prompt_rollback"
    EVALUATION_FAILURE = "evaluation_failure"
    MEMORY_CONTEXT_EXPIRED = "memory_context_expired"


_ACTORS = frozenset(item.value for item in AuditActor)
_EVENT_TYPES = frozenset(item.value for item in AuditEventType)
_STATUSES = frozenset(item.value for item in AuditStatus)
_MODELS = frozenset(item.value for item in AuditModel)
_OUTCOMES = frozenset(item.value for item in AuditOutcome)
_REASONS = frozenset(item.value for item in AuditReason)


class AuditPayload(ContractModel):
    kind: Literal["transition", "validation", "usage", "selection", "summary", "legacy_migration"]
    subject_ids: tuple[AuditSubjectId, ...] = Field(default_factory=tuple, max_length=50)
    outcome_code: AuditCode | None = None
    reason_code: AuditCode | None = None
    item_count: NonNegativeInt | None = None
    legacy_payload_digest: Digest | None = None
    legacy_schema_lineage: Literal["v0", "v1"] | None = None
    legacy_hash_policy: Literal["null_noncanonical_v1"] | None = None

    @field_validator("outcome_code")
    @classmethod
    def validate_outcome(cls, value: str | None) -> str | None:
        if value is not None and value not in _OUTCOMES:
            raise ValueError("audit outcomes must use the semantic registry")
        return value

    @field_validator("reason_code")
    @classmethod
    def validate_reason(cls, value: str | None) -> str | None:
        if value is not None and value not in _REASONS:
            raise ValueError("audit reasons must use the semantic registry")
        return value

    @model_validator(mode="after")
    def validate_kind_contract(self) -> AuditPayload:
        legacy_fields = (self.legacy_payload_digest, self.legacy_schema_lineage, self.legacy_hash_policy)
        if self.kind == "transition":
            if not self.subject_ids or any(value is not None for value in (self.outcome_code, self.reason_code, self.item_count, *legacy_fields)):
                raise ValueError("transition payloads require subjects and forbid result fields")
        elif self.kind == "validation":
            if not self.subject_ids or self.outcome_code is None or self.item_count is not None or any(value is not None for value in legacy_fields):
                raise ValueError("validation payload fields are inconsistent")
        elif self.kind in {"usage", "selection", "summary"}:
            if self.item_count is None or any(value is not None for value in legacy_fields):
                raise ValueError("aggregate payloads require item_count and forbid legacy fields")
        elif self.kind == "legacy_migration":
            if self.subject_ids or self.outcome_code != AuditOutcome.LEGACY_PAYLOAD.value or self.reason_code is not None or self.item_count is None or any(value is None for value in legacy_fields):
                raise ValueError("legacy migration payloads require complete lineage and hash policy")
        return self


class AuditEvent(ContractModel):
    event_id: AuditEventId
    trace_id: AuditTraceId
    workflow_id: AuditWorkflowId
    task_id: AuditTaskId | None = None
    attempt_id: AuditAttemptId | None = None
    sequence: PositiveInt | None = None
    occurred_at: datetime
    actor: AuditCode
    event_type: AuditCode
    status: AuditCode
    input_hash: Digest | None = None
    output_hash: Digest | None = None
    latency_ms: NonNegativeInt = 0
    token_usage: NonNegativeInt = 0
    cached_token_usage: NonNegativeInt = 0
    estimated_cost: NonNegativeFinite = 0.0
    cumulative_cost: NonNegativeFinite = 0.0
    model: AuditCode | None = None
    prompt_version: AuditPromptVersion | None = None
    schema_name: AuditSchemaName | None = None
    schema_hash: Digest | None = None
    source_references: tuple[AuditSourceReference, ...] = Field(default_factory=tuple, max_length=50)
    payload: AuditPayload
    source_schema_lineage: Literal["v0", "generic_v1", "current_v1"] = "current_v1"
    hash_policy: Literal["null_noncanonical_v1", "strict_canonical_v1"] = "strict_canonical_v1"
    legacy_semantic_digest: Digest | None = None

    @field_validator("occurred_at")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        return _require_utc(value)

    @field_validator("actor")
    @classmethod
    def validate_actor(cls, value: str) -> str:
        if value not in _ACTORS:
            raise ValueError("audit actors must use the semantic registry")
        return value

    @field_validator("event_type")
    @classmethod
    def validate_event_type(cls, value: str) -> str:
        if value not in _EVENT_TYPES:
            raise ValueError("audit event types must use the semantic registry")
        return value

    @field_validator("status")
    @classmethod
    def validate_status(cls, value: str) -> str:
        if value not in _STATUSES:
            raise ValueError("audit statuses must use the semantic registry")
        return value

    @field_validator("model")
    @classmethod
    def validate_model(cls, value: str | None) -> str | None:
        if value is not None and value not in _MODELS:
            raise ValueError("audit models must use the semantic registry")
        return value

    @model_validator(mode="after")
    def reject_sensitive_event_values(self) -> AuditEvent:
        for value in (
            self.event_id, self.trace_id, self.workflow_id, self.task_id, self.attempt_id, self.actor,
            self.event_type, self.status, self.input_hash, self.output_hash, self.model, self.prompt_version,
            self.schema_name, self.schema_hash, *self.source_references,
        ):
            if value is not None:
                _require_safe_text(value)
        encoded_payload = json.dumps(self.payload.model_dump(mode="json"), sort_keys=True, separators=(",", ":")).encode("utf-8")
        if len(encoded_payload) > _MAX_PAYLOAD_BYTES:
            raise ValueError("audit payload exceeds encoded-byte limit")
        if self.source_schema_lineage == "current_v1":
            if self.hash_policy != "strict_canonical_v1" or self.legacy_semantic_digest is not None:
                raise ValueError("current audit rows require strict canonical metadata")
        elif self.hash_policy != _LEGACY_HASH_POLICY or self.legacy_semantic_digest is None:
            raise ValueError("transformed legacy audit rows require lineage and semantic digest metadata")
        return self


class AuditObserver(Protocol):
    """Capability-scoped event sink; drivers receive no audit storage authority."""

    def record(self, event: AuditEvent) -> object: ...


class AuditPage(list[AuditEvent]):
    def __init__(self, items: Iterable[AuditEvent] = (), next_cursor: str | None = None) -> None:
        super().__init__(items)
        self.next_cursor = next_cursor


_STORAGE_COLUMNS = (
    "event_id", "trace_id", "workflow_id", "task_id", "attempt_id", "sequence", "occurred_at", "actor",
    "event_type", "status", "input_hash", "output_hash", "latency_ms", "token_usage", "cached_token_usage",
    "estimated_cost", "cumulative_cost", "model", "prompt_version", "schema_name", "schema_hash",
    "source_references", "payload", "schema_version", "source_schema_lineage", "hash_policy",
    "legacy_semantic_digest",
)


def _event_from_storage_row(row: tuple[object, ...]) -> AuditEvent:
    values = dict(zip(_STORAGE_COLUMNS, row, strict=True))
    payload = json.loads(str(values["payload"]))
    if isinstance(payload, dict) and isinstance(payload.get("subject_ids"), list):
        payload["subject_ids"] = tuple(payload["subject_ids"])
    return AuditEvent.model_validate({
        **values,
        "occurred_at": datetime.fromisoformat(str(values["occurred_at"])),
        "source_references": tuple(json.loads(str(values["source_references"]))),
        "payload": payload,
    })


def _has_current_semantics(row: list[object]) -> bool:
    if row[7] not in _ACTORS or row[8] not in _EVENT_TYPES or row[9] not in _STATUSES:
        return False
    if row[17] is not None and row[17] not in _MODELS:
        return False
    try:
        if row[18] is not None:
            _require_prompt_version(str(row[18]))
        if row[19] is not None:
            _require_schema_name(str(row[19]))
    except ValueError:
        return False
    return True


def _has_legacy_semantic_signature(row: list[object]) -> bool:
    values = (row[7], row[8], row[9], row[17], row[18], row[19])
    if any(value is not None and (not isinstance(value, str) or len(value) > 256) for value in values):
        return False
    try:
        for value in values:
            if value is not None:
                _require_code(value)
    except ValueError:
        return False
    return any(
        all(value is None or str(value).startswith(namespace) for value in values)
        for namespace in _LEGACY_SEMANTIC_NAMESPACES
    )


def _has_known_generic_payload_shape(value: object) -> bool:
    if not isinstance(value, dict):
        return False
    if "kind" not in value:
        return bool(value) and all(isinstance(key, str) for key in value)
    allowed = {"kind", "subject_ids", "outcome_code", "reason_code", "item_count", "schema_version"}
    if set(value) - allowed or value.get("kind") not in {"transition", "validation", "usage", "selection", "summary"}:
        return False
    if value.get("schema_version", "v1") != "v1":
        return False
    subjects = value.get("subject_ids", ())
    if not isinstance(subjects, (list, tuple)) or len(subjects) > 50 or any(not isinstance(item, str) for item in subjects):
        return False
    if any(value.get(field) is not None and not isinstance(value.get(field), str) for field in ("outcome_code", "reason_code")):
        return False
    item_count = value.get("item_count")
    return item_count is None or isinstance(item_count, int) and not isinstance(item_count, bool) and item_count >= 0


def _classify_storage_row(row: list[object], *, had_schema_version: bool, had_row_metadata: bool, parsed_payload: object, validated_payload: AuditPayload | None) -> Literal["stored", "current_v1", "v0", "generic_v1"]:
    if had_row_metadata:
        return "stored"
    if not had_schema_version:
        return "v0"
    if row[23] != "v1":
        return "current_v1"
    current_semantics = _has_current_semantics(row)
    if current_semantics and validated_payload is not None:
        return "current_v1"
    if current_semantics:
        if isinstance(parsed_payload, dict) and "kind" not in parsed_payload and _has_known_generic_payload_shape(parsed_payload):
            return "generic_v1"
        return "current_v1"
    if _has_legacy_semantic_signature(row) and (validated_payload is not None or _has_known_generic_payload_shape(parsed_payload)):
        return "generic_v1"
    return "current_v1"


class AuditStore:
    def __init__(self, database_path: str | Path) -> None:
        self._database_path = str(database_path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=5.0, isolation_level=None)
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA recursive_triggers = ON")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute("CREATE TABLE IF NOT EXISTS audit_events (event_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL, workflow_id TEXT NOT NULL, task_id TEXT, attempt_id TEXT, sequence INTEGER NOT NULL, occurred_at TEXT NOT NULL, actor TEXT NOT NULL, event_type TEXT NOT NULL, status TEXT NOT NULL, input_hash TEXT, output_hash TEXT, latency_ms INTEGER NOT NULL, token_usage INTEGER NOT NULL, cached_token_usage INTEGER NOT NULL, estimated_cost REAL NOT NULL, cumulative_cost REAL NOT NULL, model TEXT, prompt_version TEXT, schema_name TEXT, schema_hash TEXT, source_references TEXT NOT NULL, payload TEXT NOT NULL, schema_version TEXT NOT NULL DEFAULT 'v1', source_schema_lineage TEXT NOT NULL DEFAULT 'current_v1', hash_policy TEXT NOT NULL DEFAULT 'strict_canonical_v1', legacy_semantic_digest TEXT, UNIQUE(trace_id, sequence))")
                self._migrate_legacy_payloads(connection)
                self._rebuild_indexes(connection)
                self._create_triggers(connection)
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        finally:
            connection.close()

    @staticmethod
    def _migrate_legacy_payloads(connection: sqlite3.Connection) -> None:
        columns = {str(row[1]) for row in connection.execute("PRAGMA table_info(audit_events)")}
        had_schema_version = "schema_version" in columns
        metadata_columns = {"source_schema_lineage", "hash_policy", "legacy_semantic_digest"}
        present_metadata_columns = metadata_columns & columns
        if present_metadata_columns and present_metadata_columns != metadata_columns:
            raise ValueError("audit row metadata schema is incomplete")
        had_row_metadata = metadata_columns.issubset(columns)
        schema_expression = "schema_version" if had_schema_version else "'v0'"
        lineage_expression = "source_schema_lineage" if had_row_metadata else "'current_v1'"
        policy_expression = "hash_policy" if had_row_metadata else "'strict_canonical_v1'"
        semantic_expression = "legacy_semantic_digest" if had_row_metadata else "NULL"
        base_columns = ", ".join(_STORAGE_COLUMNS[:23])
        rows = connection.execute(f"SELECT {base_columns}, {schema_expression}, {lineage_expression}, {policy_expression}, {semantic_expression} FROM audit_events ORDER BY event_id").fetchall()
        migrations: list[tuple[object, ...]] = []
        for raw_row in rows:
            row = list(raw_row)
            payload_text = row[22]
            try:
                legacy_value: object = json.loads(str(payload_text))
            except (TypeError, ValueError):
                legacy_value = str(payload_text)
            payload_value = dict(legacy_value) if isinstance(legacy_value, dict) else legacy_value
            if isinstance(payload_value, dict) and isinstance(payload_value.get("subject_ids"), list):
                payload_value["subject_ids"] = tuple(payload_value["subject_ids"])
            try:
                validated_payload = AuditPayload.model_validate(payload_value)
            except Exception:
                validated_payload = None
            classification = _classify_storage_row(row, had_schema_version=had_schema_version, had_row_metadata=had_row_metadata, parsed_payload=legacy_value, validated_payload=validated_payload)
            if classification == "stored":
                _event_from_storage_row(tuple(row))
                continue
            if classification == "current_v1":
                row[24] = "current_v1"
                row[25] = "strict_canonical_v1"
                row[26] = None
                _event_from_storage_row(tuple(row))
                migrations.append(tuple(row))
                continue
            source_lineage = classification
            payload_lineage = "v0" if classification == "v0" else "v1"
            original_semantics = {
                "actor": row[7],
                "event_type": row[8],
                "model": row[17],
                "prompt_version": row[18],
                "schema_name": row[19],
                "status": row[9],
            }
            canonical_semantics = json.dumps(original_semantics, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            row[7] = AuditActor.LEGACY.value
            row[8] = AuditEventType.LEGACY.value
            row[9] = AuditStatus.LEGACY.value
            row[17] = AuditModel.LEGACY.value if row[17] is not None else None
            row[18] = "legacy_identifier" if row[18] is not None else None
            row[19] = "legacy_identifier" if row[19] is not None else None
            row[10], row[11], row[20] = tuple(value if value is None or _DIGEST.fullmatch(str(value)) else None for value in (row[10], row[11], row[20]))
            if validated_payload is None:
                canonical_payload = json.dumps(legacy_value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                item_count = len(legacy_value) if isinstance(legacy_value, (dict, list)) else 1
                validated_payload = AuditPayload(kind="legacy_migration", outcome_code="legacy_payload", item_count=item_count, legacy_payload_digest=sha256(canonical_payload.encode("utf-8")).hexdigest(), legacy_schema_lineage=payload_lineage, legacy_hash_policy=_LEGACY_HASH_POLICY)
            row[22] = json.dumps(validated_payload.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
            row[23] = "v1"
            row[24] = source_lineage
            row[25] = _LEGACY_HASH_POLICY
            row[26] = sha256(canonical_semantics.encode("utf-8")).hexdigest()
            _event_from_storage_row(tuple(row))
            migrations.append(tuple(row))
        schema_changes = not had_schema_version or not had_row_metadata
        if schema_changes or migrations:
            connection.execute("DROP TRIGGER IF EXISTS audit_events_no_update")
            connection.execute("DROP TRIGGER IF EXISTS audit_events_no_delete")
            connection.execute("DROP TRIGGER IF EXISTS audit_events_no_replace")
            if not had_schema_version:
                connection.execute("ALTER TABLE audit_events ADD COLUMN schema_version TEXT NOT NULL DEFAULT 'v1'")
            if "source_schema_lineage" not in columns:
                connection.execute("ALTER TABLE audit_events ADD COLUMN source_schema_lineage TEXT NOT NULL DEFAULT 'current_v1'")
            if "hash_policy" not in columns:
                connection.execute("ALTER TABLE audit_events ADD COLUMN hash_policy TEXT NOT NULL DEFAULT 'strict_canonical_v1'")
            if "legacy_semantic_digest" not in columns:
                connection.execute("ALTER TABLE audit_events ADD COLUMN legacy_semantic_digest TEXT")
            for row in migrations:
                connection.execute("UPDATE audit_events SET actor = ?, event_type = ?, status = ?, input_hash = ?, output_hash = ?, model = ?, prompt_version = ?, schema_name = ?, schema_hash = ?, payload = ?, schema_version = ?, source_schema_lineage = ?, hash_policy = ?, legacy_semantic_digest = ? WHERE event_id = ?", (row[7], row[8], row[9], row[10], row[11], row[17], row[18], row[19], row[20], row[22], row[23], row[24], row[25], row[26], row[0]))

    @staticmethod
    def _rebuild_indexes(connection: sqlite3.Connection) -> None:
        definitions = {
            "audit_events_trace_sequence_idx": "trace_id, sequence, event_id",
            "audit_events_workflow_idx": "workflow_id, trace_id, sequence, event_id",
            "audit_events_task_idx": "task_id, trace_id, sequence, event_id",
            "audit_events_attempt_idx": "attempt_id, trace_id, sequence, event_id",
            "audit_events_occurred_at_idx": "occurred_at, trace_id, sequence, event_id",
            "audit_events_type_time_idx": "event_type, occurred_at, trace_id, sequence, event_id",
        }
        for name, fields in definitions.items():
            connection.execute(f"DROP INDEX IF EXISTS {name}")
            connection.execute(f"CREATE INDEX {name} ON audit_events({fields})")

    @staticmethod
    def _create_triggers(connection: sqlite3.Connection) -> None:
        connection.execute("CREATE TRIGGER IF NOT EXISTS audit_events_no_update BEFORE UPDATE ON audit_events BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END")
        connection.execute("CREATE TRIGGER IF NOT EXISTS audit_events_no_delete BEFORE DELETE ON audit_events BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END")
        connection.execute("CREATE TRIGGER IF NOT EXISTS audit_events_no_replace BEFORE INSERT ON audit_events WHEN EXISTS (SELECT 1 FROM audit_events WHERE event_id = NEW.event_id) OR EXISTS (SELECT 1 FROM audit_events WHERE trace_id = NEW.trace_id AND sequence = NEW.sequence) BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END")

    def append(self, event: AuditEvent) -> AuditEvent:
        event = AuditEvent.model_validate(event.model_dump(mode="python"))
        if event.sequence is not None:
            raise ValueError("audit sequence is assigned by the store")
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            next_sequence = connection.execute("SELECT COALESCE(MAX(sequence), 0) + 1 FROM audit_events WHERE trace_id = ?", (event.trace_id,)).fetchone()[0]
            connection.execute(
                "INSERT INTO audit_events (event_id, trace_id, workflow_id, task_id, attempt_id, sequence, occurred_at, actor, event_type, status, input_hash, output_hash, latency_ms, token_usage, cached_token_usage, estimated_cost, cumulative_cost, model, prompt_version, schema_name, schema_hash, source_references, payload, schema_version, source_schema_lineage, hash_policy, legacy_semantic_digest) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (event.event_id, event.trace_id, event.workflow_id, event.task_id, event.attempt_id, next_sequence, event.occurred_at.isoformat(), event.actor, event.event_type, event.status, event.input_hash, event.output_hash, event.latency_ms, event.token_usage, event.cached_token_usage, event.estimated_cost, event.cumulative_cost, event.model, event.prompt_version, event.schema_name, event.schema_hash, json.dumps(event.source_references, separators=(",", ":")), json.dumps(event.payload.model_dump(mode="json"), sort_keys=True, separators=(",", ":")), event.schema_version, event.source_schema_lineage, event.hash_policy, event.legacy_semantic_digest),
            )
            connection.execute("COMMIT")
        except BaseException:
            if connection is not None and connection.in_transaction:
                try:
                    connection.execute("ROLLBACK")
                except BaseException:
                    pass
            raise
        finally:
            if connection is not None:
                try:
                    connection.close()
                except BaseException:
                    pass
        return AuditEvent.model_validate({**event.model_dump(mode="python"), "sequence": next_sequence})

    def list(self, *, trace_id: str | None = None, workflow_id: str | None = None, task_id: str | None = None, attempt_id: str | None = None, event_type: str | None = None, start_time: datetime | None = None, end_time: datetime | None = None, page_size: int = _MAX_PAGE_SIZE, cursor: str | None = None) -> AuditPage:
        if isinstance(page_size, bool) or not 1 <= page_size <= _MAX_PAGE_SIZE:
            raise ValueError(f"page_size must be between 1 and {_MAX_PAGE_SIZE}")
        clauses: list[str] = []
        values: list[object] = []
        for field_name, field_value in (("trace_id", trace_id), ("workflow_id", workflow_id), ("task_id", task_id), ("attempt_id", attempt_id), ("event_type", event_type)):
            if field_value is not None:
                clauses.append(f"{field_name} = ?")
                values.append(field_value)
        for operator, timestamp in ((">=", start_time), ("<=", end_time)):
            if timestamp is not None:
                clauses.append(f"occurred_at {operator} ?")
                values.append(_require_utc(timestamp).isoformat())
        filter_hash = self._filter_hash(trace_id, workflow_id, task_id, attempt_id, event_type, start_time, end_time)
        if cursor is not None:
            cursor_trace, cursor_sequence, cursor_event, cursor_filter_hash = self._decode_cursor(cursor)
            if cursor_filter_hash != filter_hash:
                raise ValueError("audit cursor does not match active filters")
            clauses.append("(trace_id > ? OR (trace_id = ? AND (sequence > ? OR (sequence = ? AND event_id > ?))))")
            values.extend((cursor_trace, cursor_trace, cursor_sequence, cursor_sequence, cursor_event))
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        connection = self._connect()
        try:
            rows = connection.execute("SELECT * FROM audit_events" + where + " ORDER BY trace_id ASC, sequence ASC, event_id ASC LIMIT ?", (*values, page_size + 1)).fetchall()
        finally:
            connection.close()
        events = [self._row_to_event(row) for row in rows[:page_size]]
        next_cursor = self._encode_cursor(events[-1], filter_hash) if len(rows) > page_size and events else None
        return AuditPage(events, next_cursor)

    @staticmethod
    def _encode_cursor(event: AuditEvent, filter_hash: str) -> str:
        rendered = json.dumps((event.trace_id, event.sequence, event.event_id, filter_hash), separators=(",", ":")).encode("utf-8")
        return base64.urlsafe_b64encode(rendered).decode("ascii")

    @staticmethod
    def _decode_cursor(cursor: str) -> tuple[str, int, str, str]:
        if len(cursor) > 512:
            raise ValueError("invalid audit cursor")
        try:
            trace_id, sequence, event_id, filter_hash = json.loads(base64.urlsafe_b64decode(cursor.encode("ascii")))
        except Exception as error:
            raise ValueError("invalid audit cursor") from error
        if not isinstance(trace_id, str) or isinstance(sequence, bool) or not isinstance(sequence, int) or not isinstance(event_id, str) or not isinstance(filter_hash, str) or sequence < 1:
            raise ValueError("invalid audit cursor")
        if not _DIGEST.fullmatch(filter_hash):
            raise ValueError("invalid audit cursor")
        return _require_id(trace_id), sequence, _require_id(event_id), filter_hash

    @staticmethod
    def _filter_hash(trace_id: str | None, workflow_id: str | None, task_id: str | None, attempt_id: str | None, event_type: str | None, start_time: datetime | None, end_time: datetime | None) -> str:
        return __import__("hashlib").sha256(json.dumps((trace_id, workflow_id, task_id, attempt_id, event_type, start_time.isoformat() if start_time else None, end_time.isoformat() if end_time else None), separators=(",", ":")).encode("utf-8")).hexdigest()

    @staticmethod
    def _row_to_event(row: tuple[object, ...]) -> AuditEvent:
        return _event_from_storage_row(row)


class AuditWriter:
    """Compatibility writer for the audit projection, not orchestration state."""

    def __init__(self, store: AuditStore) -> None:
        self._store = store
        self._healthy = True

    @property
    def healthy(self) -> bool:
        return self._healthy

    def record(self, event: AuditEvent) -> AuditEvent:
        if not self._healthy:
            raise AuditUnavailableError("audit writer is unavailable")
        try:
            return self._store.append(event)
        except Exception as error:
            self._healthy = False
            raise AuditUnavailableError("required audit write failed") from error
