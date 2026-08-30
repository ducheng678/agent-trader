from __future__ import annotations

import base64
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
import sqlite3
from typing import Iterable, Literal

from pydantic import Field, StringConstraints, field_validator, model_validator
from typing import Annotated

from market_agent.workflow_contracts import ContractModel, NonNegativeFinite, NonNegativeInt, PositiveInt, ShortText


_MAX_PAGE_SIZE = 100
_MAX_PAYLOAD_BYTES = 4096
_UNSAFE_VALUE = re.compile(r"(?:authorization\s*[:=]|bearer\s+\S+|cookie\s*[:=]|(?:api[ _-]?key|credential|secret|token|private[ _-]?key)\s*[:=]|https?://\S+[?&](?:token|key|secret|signature)=)", re.IGNORECASE)
_OPAQUE_ID = re.compile(r"^(?!sk-)(?!eyJ)[a-z][a-z0-9_-]{0,63}$")
_CODE = re.compile(r"^[A-Za-z][A-Za-z0-9_.-]{0,127}$")
_DIGEST = re.compile(r"^[0-9a-f]{64}$")
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class AuditUnavailableError(RuntimeError):
    pass


def _require_safe_text(value: str) -> str:
    if _UNSAFE_VALUE.search(value):
        raise ValueError("audit values cannot contain credentials, authorization data, or URL secrets")
    return value


def _require_id(value: str) -> str:
    if not _OPAQUE_ID.fullmatch(value):
        raise ValueError("audit opaque identifiers must be bounded non-secret identifiers")
    return value


def _require_code(value: str) -> str:
    if not _CODE.fullmatch(value):
        raise ValueError("audit codes must be compact identifiers, never prose or URLs")
    return value


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("audit timestamps must be UTC")
    return value


class AuditPayload(ContractModel):
    kind: Literal["transition", "validation", "usage", "selection", "summary", "legacy_migration"]
    subject_ids: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=50)
    outcome_code: ShortText | None = None
    reason_code: ShortText | None = None
    item_count: NonNegativeInt | None = None
    legacy_payload_digest: Digest | None = None
    legacy_schema_lineage: Literal["v0", "v1"] | None = None

    @field_validator("subject_ids", "outcome_code", "reason_code")
    @classmethod
    def reject_sensitive_payload_values(cls, value: tuple[str, ...] | str | None) -> tuple[str, ...] | str | None:
        if isinstance(value, tuple):
            return tuple(_require_safe_text(item) for item in value)
        return _require_safe_text(value) if value is not None else value


class AuditEvent(ContractModel):
    event_id: ShortText
    trace_id: ShortText
    workflow_id: ShortText
    task_id: ShortText | None = None
    attempt_id: ShortText | None = None
    sequence: PositiveInt | None = None
    occurred_at: datetime
    actor: ShortText
    event_type: ShortText
    status: ShortText
    input_hash: Digest | None = None
    output_hash: Digest | None = None
    latency_ms: NonNegativeInt = 0
    token_usage: NonNegativeInt = 0
    cached_token_usage: NonNegativeInt = 0
    estimated_cost: NonNegativeFinite = 0.0
    cumulative_cost: NonNegativeFinite = 0.0
    model: ShortText | None = None
    prompt_version: ShortText | None = None
    schema_name: ShortText | None = None
    schema_hash: Digest | None = None
    source_references: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=50)
    payload: AuditPayload

    @field_validator("occurred_at")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        return _require_utc(value)

    @field_validator("event_id", "trace_id", "workflow_id", "task_id", "attempt_id", "source_references")
    @classmethod
    def validate_identifiers(cls, value: tuple[str, ...] | str | None) -> tuple[str, ...] | str | None:
        if isinstance(value, tuple):
            return tuple(_require_id(item) for item in value)
        return _require_id(value) if value is not None else value

    @field_validator("actor", "event_type", "status", "model", "prompt_version", "schema_name")
    @classmethod
    def validate_codes(cls, value: str | None) -> str | None:
        return _require_code(value) if value is not None else value

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
            connection.execute(
                "CREATE TABLE IF NOT EXISTS audit_events (event_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL, workflow_id TEXT NOT NULL, task_id TEXT, attempt_id TEXT, sequence INTEGER NOT NULL, occurred_at TEXT NOT NULL, actor TEXT NOT NULL, event_type TEXT NOT NULL, status TEXT NOT NULL, input_hash TEXT, output_hash TEXT, latency_ms INTEGER NOT NULL, token_usage INTEGER NOT NULL, cached_token_usage INTEGER NOT NULL, estimated_cost REAL NOT NULL, cumulative_cost REAL NOT NULL, model TEXT, prompt_version TEXT, schema_name TEXT, schema_hash TEXT, source_references TEXT NOT NULL, payload TEXT NOT NULL, schema_version TEXT NOT NULL DEFAULT 'v1', UNIQUE(trace_id, sequence))"
            )
            columns = {row[1] for row in connection.execute("PRAGMA table_info(audit_events)")}
            if "schema_version" not in columns:
                connection.execute("ALTER TABLE audit_events ADD COLUMN schema_version TEXT NOT NULL DEFAULT 'v1'")
            self._migrate_legacy_payloads(connection)
            connection.execute("CREATE INDEX IF NOT EXISTS audit_events_trace_sequence_idx ON audit_events(trace_id, sequence)")
            connection.execute("CREATE INDEX IF NOT EXISTS audit_events_workflow_idx ON audit_events(workflow_id, sequence)")
            connection.execute("CREATE INDEX IF NOT EXISTS audit_events_attempt_idx ON audit_events(attempt_id, sequence)")
            connection.execute("CREATE INDEX IF NOT EXISTS audit_events_occurred_at_idx ON audit_events(occurred_at)")
            connection.execute("CREATE INDEX IF NOT EXISTS audit_events_type_time_idx ON audit_events(event_type, occurred_at)")
            connection.execute("CREATE TRIGGER IF NOT EXISTS audit_events_no_update BEFORE UPDATE ON audit_events BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END")
            connection.execute("CREATE TRIGGER IF NOT EXISTS audit_events_no_delete BEFORE DELETE ON audit_events BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END")
            connection.execute("CREATE TRIGGER IF NOT EXISTS audit_events_no_replace BEFORE INSERT ON audit_events WHEN EXISTS (SELECT 1 FROM audit_events WHERE event_id = NEW.event_id) OR EXISTS (SELECT 1 FROM audit_events WHERE trace_id = NEW.trace_id AND sequence = NEW.sequence) BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END")
        finally:
            connection.close()

    @staticmethod
    def _migrate_legacy_payloads(connection: sqlite3.Connection) -> None:
        connection.execute("DROP TRIGGER IF EXISTS audit_events_no_update")
        connection.execute("DROP TRIGGER IF EXISTS audit_events_no_delete")
        connection.execute("DROP TRIGGER IF EXISTS audit_events_no_replace")
        connection.execute("BEGIN IMMEDIATE")
        try:
            rows = connection.execute("SELECT event_id, payload, schema_version FROM audit_events").fetchall()
            for event_id, payload_text, schema_version in rows:
                try:
                    parsed = json.loads(payload_text)
                    AuditPayload.model_validate(parsed)
                    continue
                except Exception:
                    pass
                canonical = json.dumps(parsed if 'parsed' in locals() else str(payload_text), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
                migrated = {"kind": "legacy_migration", "subject_ids": (), "outcome_code": "legacy_payload", "item_count": len(parsed) if isinstance(parsed, (dict, list)) else 1, "legacy_payload_digest": __import__("hashlib").sha256(canonical.encode("utf-8")).hexdigest(), "legacy_schema_lineage": "v0" if schema_version != "v1" else "v1"}
                connection.execute("UPDATE audit_events SET payload = ? WHERE event_id = ?", (json.dumps(migrated, sort_keys=True, separators=(",", ":")), event_id))
            connection.execute("COMMIT")
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise

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
        if not payload:
            payload = {"kind": "transition", "subject_ids": ()}
        elif "subject_ids" in payload:
            payload["subject_ids"] = tuple(payload["subject_ids"])
        for field_name in ("input_hash", "output_hash", "schema_hash"):
            if values[field_name] is not None and not _DIGEST.fullmatch(str(values[field_name])):
                values[field_name] = None
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
