from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Mapping

from pydantic import Field, JsonValue, field_validator

from market_agent.workflow_contracts import ContractModel, NonNegativeFinite, NonNegativeInt, PositiveInt, ShortText


_REDACTED = "[REDACTED]"
_SENSITIVE_FIELD_FRAGMENTS = ("authorization", "credential", "secret", "token", "password", "api_key", "apikey", "private_key", "cookie", "reasoning", "analysis")


class AuditUnavailableError(RuntimeError):
    pass


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
    input_hash: ShortText | None = None
    output_hash: ShortText | None = None
    latency_ms: NonNegativeInt = 0
    token_usage: NonNegativeInt = 0
    cached_token_usage: NonNegativeInt = 0
    estimated_cost: NonNegativeFinite = 0.0
    cumulative_cost: NonNegativeFinite = 0.0
    model: ShortText | None = None
    prompt_version: ShortText | None = None
    schema_name: ShortText | None = None
    schema_hash: ShortText | None = None
    source_references: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=50)
    payload: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("occurred_at")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("audit timestamps must be UTC")
        return value


def _redact(value: Any, field_name: str | None = None) -> Any:
    if field_name is not None and any(fragment in field_name.lower() for fragment in _SENSITIVE_FIELD_FRAGMENTS):
        return _REDACTED
    if isinstance(value, Mapping):
        return {str(key): _redact(item, str(key)) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, tuple):
        return [_redact(item) for item in value]
    return value


class AuditStore:
    def __init__(self, database_path: str | Path) -> None:
        self._database_path = str(database_path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self._database_path, timeout=5.0, isolation_level=None)
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    trace_id TEXT NOT NULL,
                    workflow_id TEXT NOT NULL,
                    task_id TEXT,
                    attempt_id TEXT,
                    sequence INTEGER NOT NULL,
                    occurred_at TEXT NOT NULL,
                    actor TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    status TEXT NOT NULL,
                    input_hash TEXT,
                    output_hash TEXT,
                    latency_ms INTEGER NOT NULL,
                    token_usage INTEGER NOT NULL,
                    cached_token_usage INTEGER NOT NULL,
                    estimated_cost REAL NOT NULL,
                    cumulative_cost REAL NOT NULL,
                    model TEXT,
                    prompt_version TEXT,
                    schema_name TEXT,
                    schema_hash TEXT,
                    source_references TEXT NOT NULL,
                    payload TEXT NOT NULL,
                    UNIQUE(trace_id, sequence)
                )
                """
            )
            connection.execute("CREATE INDEX IF NOT EXISTS audit_events_trace_sequence_idx ON audit_events(trace_id, sequence)")
            connection.execute("CREATE INDEX IF NOT EXISTS audit_events_workflow_idx ON audit_events(workflow_id, sequence)")
            connection.execute("CREATE INDEX IF NOT EXISTS audit_events_task_attempt_idx ON audit_events(task_id, attempt_id, sequence)")
            connection.execute("CREATE INDEX IF NOT EXISTS audit_events_type_time_idx ON audit_events(event_type, occurred_at)")
            connection.execute("CREATE TRIGGER IF NOT EXISTS audit_events_no_update BEFORE UPDATE ON audit_events BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END")
            connection.execute("CREATE TRIGGER IF NOT EXISTS audit_events_no_delete BEFORE DELETE ON audit_events BEGIN SELECT RAISE(ABORT, 'audit events are append-only'); END")
        finally:
            connection.close()

    def append(self, event: AuditEvent) -> AuditEvent:
        if event.sequence is not None:
            raise ValueError("audit sequence is assigned by the store")
        redacted_event = event.model_copy(update={"payload": _redact(event.payload)})
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            next_sequence = connection.execute(
                "SELECT COALESCE(MAX(sequence), 0) + 1 FROM audit_events WHERE trace_id = ?",
                (redacted_event.trace_id,),
            ).fetchone()[0]
            connection.execute(
                """
                INSERT INTO audit_events (
                    event_id, trace_id, workflow_id, task_id, attempt_id, sequence, occurred_at, actor,
                    event_type, status, input_hash, output_hash, latency_ms, token_usage, cached_token_usage,
                    estimated_cost, cumulative_cost, model, prompt_version, schema_name, schema_hash,
                    source_references, payload
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    redacted_event.event_id,
                    redacted_event.trace_id,
                    redacted_event.workflow_id,
                    redacted_event.task_id,
                    redacted_event.attempt_id,
                    next_sequence,
                    redacted_event.occurred_at.astimezone(timezone.utc).isoformat(),
                    redacted_event.actor,
                    redacted_event.event_type,
                    redacted_event.status,
                    redacted_event.input_hash,
                    redacted_event.output_hash,
                    redacted_event.latency_ms,
                    redacted_event.token_usage,
                    redacted_event.cached_token_usage,
                    redacted_event.estimated_cost,
                    redacted_event.cumulative_cost,
                    redacted_event.model,
                    redacted_event.prompt_version,
                    redacted_event.schema_name,
                    redacted_event.schema_hash,
                    json.dumps(redacted_event.source_references, separators=(",", ":")),
                    json.dumps(redacted_event.payload, sort_keys=True, separators=(",", ":")),
                ),
            )
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
        return redacted_event.model_copy(update={"sequence": next_sequence})

    def list(
        self,
        *,
        trace_id: str | None = None,
        workflow_id: str | None = None,
        task_id: str | None = None,
        attempt_id: str | None = None,
        event_type: str | None = None,
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> list[AuditEvent]:
        clauses: list[str] = []
        values: list[object] = []
        for field_name, field_value in (("trace_id", trace_id), ("workflow_id", workflow_id), ("task_id", task_id), ("attempt_id", attempt_id), ("event_type", event_type)):
            if field_value is not None:
                clauses.append(f"{field_name} = ?")
                values.append(field_value)
        if start_time is not None:
            clauses.append("occurred_at >= ?")
            values.append(start_time.astimezone(timezone.utc).isoformat())
        if end_time is not None:
            clauses.append("occurred_at <= ?")
            values.append(end_time.astimezone(timezone.utc).isoformat())
        where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM audit_events" + where + " ORDER BY trace_id ASC, sequence ASC, event_id ASC",
                values,
            ).fetchall()
        finally:
            connection.close()
        columns = (
            "event_id", "trace_id", "workflow_id", "task_id", "attempt_id", "sequence", "occurred_at", "actor",
            "event_type", "status", "input_hash", "output_hash", "latency_ms", "token_usage", "cached_token_usage",
            "estimated_cost", "cumulative_cost", "model", "prompt_version", "schema_name", "schema_hash",
            "source_references", "payload",
        )
        return [
            AuditEvent(
                **{
                    **dict(zip(columns, row, strict=True)),
                    "occurred_at": datetime.fromisoformat(row[6]),
                    "source_references": tuple(json.loads(row[21])),
                    "payload": json.loads(row[22]),
                }
            )
            for row in rows
        ]


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
