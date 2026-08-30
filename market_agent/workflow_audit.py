from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
from enum import Enum
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterable, Literal

from pydantic import AfterValidator, Field, StringConstraints, field_validator, model_validator
from typing import Annotated

from market_agent.workflow_contracts import ContractModel, Digest, NonNegativeFinite, NonNegativeInt, PositiveInt, ShortText


_MAX_PAGE_SIZE = 100
_MAX_PAYLOAD_BYTES = 4096
_UNSAFE_VALUE = re.compile(r"(?:authorization|bearer|cookie|api[ _-]?key|credential|secret|token|password|private[ _-]?key|raw[ _-]?prompt|system[ _-]?prompt|reasoning|chain[ _-]?of[ _-]?thought|-----BEGIN|eyJ[a-zA-Z0-9_-]*\.|https?://\S+[?&](?:token|key|secret|signature)=)", re.IGNORECASE)
_OPAQUE_ID = re.compile(r"^(?!sk-)(?!eyJ)[a-z][a-z0-9_-]{0,63}$")
_CODE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
_LEGACY_HASH_POLICY = "null_noncanonical_v1"


class AuditUnavailableError(RuntimeError):
    pass


def _require_safe_text(value: str) -> str:
    if _UNSAFE_VALUE.search(value):
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


class AuditStatus(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COMPLETED = "completed"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"
    OMITTED = "omitted"


class AuditModel(str, Enum):
    LUNA = "gpt-5.6-luna"
    TERRA = "gpt-5.6-terra"
    SOL = "gpt-5.6-sol"


class AuditOutcome(str, Enum):
    ACCEPTED = "accepted"
    REJECTED = "rejected"
    COMPLETED = "completed"
    OMITTED = "omitted"
    LEGACY_PAYLOAD = "legacy_payload"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    UNAVAILABLE = "unavailable"


class AuditReason(str, Enum):
    VALIDATION_ERROR = "validation_error"
    BUDGET_LIMIT = "budget_limit"
    SOURCE_LIMIT = "source_limit"
    UNRESOLVED_CONFLICT = "unresolved_conflict"
    MISSING_EVIDENCE = "missing_evidence"
    LEGACY_SCHEMA = "legacy_schema"
    AUDIT_FAILURE = "audit_failure"


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
    prompt_version: AuditCode | None = None
    schema_name: AuditCode | None = None
    schema_hash: Digest | None = None
    source_references: tuple[AuditSourceReference, ...] = Field(default_factory=tuple, max_length=50)
    payload: AuditPayload

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
        return self


class AuditPage(list[AuditEvent]):
    def __init__(self, items: Iterable[AuditEvent] = (), next_cursor: str | None = None) -> None:
        super().__init__(items)
        self.next_cursor = next_cursor


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
                connection.execute("CREATE TABLE IF NOT EXISTS audit_events (event_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL, workflow_id TEXT NOT NULL, task_id TEXT, attempt_id TEXT, sequence INTEGER NOT NULL, occurred_at TEXT NOT NULL, actor TEXT NOT NULL, event_type TEXT NOT NULL, status TEXT NOT NULL, input_hash TEXT, output_hash TEXT, latency_ms INTEGER NOT NULL, token_usage INTEGER NOT NULL, cached_token_usage INTEGER NOT NULL, estimated_cost REAL NOT NULL, cumulative_cost REAL NOT NULL, model TEXT, prompt_version TEXT, schema_name TEXT, schema_hash TEXT, source_references TEXT NOT NULL, payload TEXT NOT NULL, schema_version TEXT NOT NULL DEFAULT 'v1', UNIQUE(trace_id, sequence))")
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
        schema_expression = "schema_version" if had_schema_version else "'v0'"
        rows = connection.execute(f"SELECT event_id, payload, input_hash, output_hash, schema_hash, {schema_expression} FROM audit_events ORDER BY event_id").fetchall()
        migrations: list[tuple[str | None, str | None, str | None, str | None, str]] = []
        for event_id, payload_text, input_hash, output_hash, schema_hash, schema_version in rows:
            legacy_value: object
            try:
                legacy_value = json.loads(str(payload_text))
            except (TypeError, ValueError):
                legacy_value = str(payload_text)
            payload_value = dict(legacy_value) if isinstance(legacy_value, dict) else legacy_value
            if isinstance(payload_value, dict) and isinstance(payload_value.get("subject_ids"), list):
                payload_value["subject_ids"] = tuple(payload_value["subject_ids"])
            try:
                validated_payload = AuditPayload.model_validate(payload_value)
                migrated_payload = None
            except Exception:
                canonical = json.dumps(legacy_value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                item_count = len(legacy_value) if isinstance(legacy_value, (dict, list)) else 1
                migrated_payload = json.dumps(AuditPayload(kind="legacy_migration", outcome_code="legacy_payload", item_count=item_count, legacy_payload_digest=sha256(canonical.encode("utf-8")).hexdigest(), legacy_schema_lineage="v1" if schema_version == "v1" else "v0", legacy_hash_policy=_LEGACY_HASH_POLICY).model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
                validated_payload = None
            cleaned_hashes = tuple(value if value is None or _DIGEST.fullmatch(str(value)) else None for value in (input_hash, output_hash, schema_hash))
            if migrated_payload is not None or cleaned_hashes != (input_hash, output_hash, schema_hash) or schema_version != "v1" or not had_schema_version:
                payload_update = migrated_payload if migrated_payload is not None else json.dumps(validated_payload.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
                migrations.append((payload_update, *cleaned_hashes, str(event_id)))
        if not had_schema_version or migrations:
            connection.execute("DROP TRIGGER IF EXISTS audit_events_no_update")
            connection.execute("DROP TRIGGER IF EXISTS audit_events_no_delete")
            connection.execute("DROP TRIGGER IF EXISTS audit_events_no_replace")
            if not had_schema_version:
                connection.execute("ALTER TABLE audit_events ADD COLUMN schema_version TEXT NOT NULL DEFAULT 'v1'")
            for payload_update, input_hash, output_hash, schema_hash, event_id in migrations:
                connection.execute("UPDATE audit_events SET payload = ?, input_hash = ?, output_hash = ?, schema_hash = ?, schema_version = 'v1' WHERE event_id = ?", (payload_update, input_hash, output_hash, schema_hash, event_id))

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
        if event.sequence is not None:
            raise ValueError("audit sequence is assigned by the store")
        connection: sqlite3.Connection | None = None
        try:
            connection = self._connect()
            connection.execute("BEGIN IMMEDIATE")
            next_sequence = connection.execute("SELECT COALESCE(MAX(sequence), 0) + 1 FROM audit_events WHERE trace_id = ?", (event.trace_id,)).fetchone()[0]
            connection.execute(
                "INSERT INTO audit_events (event_id, trace_id, workflow_id, task_id, attempt_id, sequence, occurred_at, actor, event_type, status, input_hash, output_hash, latency_ms, token_usage, cached_token_usage, estimated_cost, cumulative_cost, model, prompt_version, schema_name, schema_hash, source_references, payload, schema_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (event.event_id, event.trace_id, event.workflow_id, event.task_id, event.attempt_id, next_sequence, event.occurred_at.isoformat(), event.actor, event.event_type, event.status, event.input_hash, event.output_hash, event.latency_ms, event.token_usage, event.cached_token_usage, event.estimated_cost, event.cumulative_cost, event.model, event.prompt_version, event.schema_name, event.schema_hash, json.dumps(event.source_references, separators=(",", ":")), json.dumps(event.payload.model_dump(mode="json"), sort_keys=True, separators=(",", ":")), event.schema_version),
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
        return event.model_copy(update={"sequence": next_sequence})

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
        columns = ("event_id", "trace_id", "workflow_id", "task_id", "attempt_id", "sequence", "occurred_at", "actor", "event_type", "status", "input_hash", "output_hash", "latency_ms", "token_usage", "cached_token_usage", "estimated_cost", "cumulative_cost", "model", "prompt_version", "schema_name", "schema_hash", "source_references", "payload", "schema_version")
        values = dict(zip(columns, row, strict=True))
        payload = json.loads(str(row[22]))
        if isinstance(payload, dict) and "subject_ids" in payload:
            payload["subject_ids"] = tuple(payload["subject_ids"])
        return AuditEvent(**{**values, "occurred_at": datetime.fromisoformat(str(row[6])), "source_references": tuple(json.loads(str(row[21])),), "payload": payload})


class AuditWriter:
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
