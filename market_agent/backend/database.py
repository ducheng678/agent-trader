from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypeVar

from market_agent.backend.errors import IdempotencyConflictError, ValidationError

T = TypeVar("T")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_dump(value: Any, *, sort_keys: bool = False) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=sort_keys, default=str)


def _json_load(value: str | None) -> Any:
    return None if value is None else json.loads(value)


def _payload_fingerprint(payload: dict[str, Any]) -> str:
    return hashlib.sha256(_json_dump(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _normalize_idempotency_key(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        raise ValidationError("idempotency key cannot be blank")
    if len(normalized) > 256:
        raise ValidationError("idempotency key cannot exceed 256 characters")
    return normalized


@dataclass(frozen=True)
class JobRecord:
    job_id: str
    task_name: str
    status: str
    payload: dict[str, Any]
    idempotency_key: str | None
    payload_fingerprint: str
    result: Any
    error: dict[str, Any] | None
    attempt_count: int
    max_attempts: int
    request_id: str
    created_at: str
    updated_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "task_name": self.task_name,
            "status": self.status,
            "payload": self.payload,
            "idempotency_key": self.idempotency_key,
            "result": self.result,
            "error": self.error,
            "attempt_count": self.attempt_count,
            "max_attempts": self.max_attempts,
            "request_id": self.request_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }


@dataclass(frozen=True)
class EventRecord:
    event_id: int
    job_id: str
    event_type: str
    payload: dict[str, Any]
    created_at: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "job_id": self.job_id,
            "event_type": self.event_type,
            "payload": self.payload,
            "created_at": self.created_at,
        }


class JobRepository:
    _allowed_transitions = {
        "accepted": {"running", "failed"},
        "running": {"accepted", "succeeded", "failed"},
        "succeeded": set(),
        "failed": set(),
    }

    def __init__(self, database_path: str | Path) -> None:
        raw_path = str(database_path)
        self._lock = threading.RLock()
        self._uses_uri = raw_path == ":memory:"
        self._anchor_connection: sqlite3.Connection | None = None
        if self._uses_uri:
            self._database_target = f"file:market_agent_backend_{uuid.uuid4().hex}?mode=memory&cache=shared"
            self._anchor_connection = self._connect()
        else:
            database_file = Path(raw_path)
            database_file.parent.mkdir(parents=True, exist_ok=True)
            self._database_target = str(database_file)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._database_target,
            timeout=10.0,
            isolation_level=None,
            check_same_thread=False,
            uri=self._uses_uri,
        )
        try:
            connection.row_factory = sqlite3.Row
            connection.execute("PRAGMA foreign_keys = ON")
            connection.execute("PRAGMA busy_timeout = 10000")
            if not self._uses_uri:
                connection.execute("PRAGMA journal_mode = WAL")
                connection.execute("PRAGMA synchronous = NORMAL")
            return connection
        except BaseException:
            connection.close()
            raise

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    task_name TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    idempotency_key TEXT UNIQUE,
                    payload_fingerprint TEXT NOT NULL,
                    result_json TEXT,
                    error_json TEXT,
                    attempt_count INTEGER NOT NULL DEFAULT 0,
                    max_attempts INTEGER NOT NULL,
                    request_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_jobs_status_updated_at ON jobs(status, updated_at);
                CREATE TABLE IF NOT EXISTS job_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(job_id) REFERENCES jobs(job_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_job_events_job_id_event_id ON job_events(job_id, event_id);
                """
            )
        finally:
            connection.close()

    @staticmethod
    def _to_job(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            job_id=str(row["job_id"]),
            task_name=str(row["task_name"]),
            status=str(row["status"]),
            payload=dict(_json_load(row["payload_json"]) or {}),
            idempotency_key=row["idempotency_key"],
            payload_fingerprint=str(row["payload_fingerprint"]),
            result=_json_load(row["result_json"]),
            error=_json_load(row["error_json"]),
            attempt_count=int(row["attempt_count"]),
            max_attempts=int(row["max_attempts"]),
            request_id=str(row["request_id"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    @staticmethod
    def _to_event(row: sqlite3.Row) -> EventRecord:
        return EventRecord(
            event_id=int(row["event_id"]),
            job_id=str(row["job_id"]),
            event_type=str(row["event_type"]),
            payload=dict(_json_load(row["payload_json"]) or {}),
            created_at=str(row["created_at"]),
        )

    def _transaction(self, operation: Callable[[sqlite3.Connection], T]) -> T:
        with self._lock:
            connection = self._connect()
            transaction_started = False
            try:
                connection.execute("BEGIN IMMEDIATE")
                transaction_started = True
                result = operation(connection)
                connection.execute("COMMIT")
                transaction_started = False
                return result
            except BaseException:
                if transaction_started:
                    with suppress(sqlite3.Error):
                        connection.execute("ROLLBACK")
                raise
            finally:
                connection.close()

    @staticmethod
    def _append_event(connection: sqlite3.Connection, job_id: str, event_type: str, payload: dict[str, Any]) -> None:
        connection.execute(
            "INSERT INTO job_events(job_id, event_type, payload_json, created_at) VALUES (?, ?, ?, ?)",
            (job_id, event_type, _json_dump(payload), _utc_now()),
        )

    @staticmethod
    def _validate_existing_job(existing: JobRecord, task_name: str, fingerprint: str) -> None:
        if existing.task_name != task_name or existing.payload_fingerprint != fingerprint:
            raise IdempotencyConflictError(
                "idempotency key was already used with a different task or payload",
                {"job_id": existing.job_id},
            )

    def find_idempotent_job(
        self,
        task_name: str,
        payload: dict[str, Any],
        idempotency_key: str | None,
    ) -> JobRecord | None:
        normalized_key = _normalize_idempotency_key(idempotency_key)
        if normalized_key is None:
            return None
        fingerprint = _payload_fingerprint(dict(payload or {}))
        connection = self._connect()
        try:
            row = connection.execute("SELECT * FROM jobs WHERE idempotency_key = ?", (normalized_key,)).fetchone()
            if row is None:
                return None
            existing = self._to_job(row)
            self._validate_existing_job(existing, str(task_name), fingerprint)
            return existing
        finally:
            connection.close()

    def create_or_get_job(
        self,
        task_name: str,
        payload: dict[str, Any],
        idempotency_key: str | None,
        max_attempts: int,
        request_id: str,
    ) -> tuple[JobRecord, bool]:
        normalized_payload = dict(payload or {})
        normalized_key = _normalize_idempotency_key(idempotency_key)
        fingerprint = _payload_fingerprint(normalized_payload)

        def operation(connection: sqlite3.Connection) -> tuple[JobRecord, bool]:
            if normalized_key:
                row = connection.execute("SELECT * FROM jobs WHERE idempotency_key = ?", (normalized_key,)).fetchone()
                if row is not None:
                    existing = self._to_job(row)
                    self._validate_existing_job(existing, task_name, fingerprint)
                    return existing, True
            now = _utc_now()
            job_id = uuid.uuid4().hex
            connection.execute(
                """
                INSERT INTO jobs(
                    job_id, task_name, status, payload_json, idempotency_key, payload_fingerprint,
                    result_json, error_json, attempt_count, max_attempts, request_id, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, NULL, NULL, 0, ?, ?, ?, ?)
                """,
                (job_id, task_name, "accepted", _json_dump(normalized_payload), normalized_key, fingerprint, int(max_attempts), request_id, now, now),
            )
            self._append_event(connection, job_id, "task_accepted", {"task_name": task_name, "request_id": request_id})
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            return self._to_job(row), False

        return self._transaction(operation)

    def list_recoverable_jobs(self, task_name: str, limit: int = 1000) -> list[JobRecord]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """
                SELECT * FROM jobs
                WHERE task_name = ? AND status IN (?, ?)
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (str(task_name), "accepted", "running", max(1, min(int(limit), 10000))),
            ).fetchall()
            return [self._to_job(row) for row in rows]
        finally:
            connection.close()

    def get_job(self, job_id: str) -> JobRecord | None:
        connection = self._connect()
        try:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (str(job_id),)).fetchone()
            return None if row is None else self._to_job(row)
        finally:
            connection.close()

    def _transition(
        self,
        job_id: str,
        target_status: str,
        event_type: str,
        event_payload: dict[str, Any],
        *,
        result: Any = None,
        error: dict[str, Any] | None = None,
        attempt_count: int | None = None,
    ) -> JobRecord:
        def operation(connection: sqlite3.Connection) -> JobRecord:
            row = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            current = self._to_job(row)
            if target_status not in self._allowed_transitions.get(current.status, set()):
                raise RuntimeError(f"invalid job state transition: {current.status} -> {target_status}")
            now = _utc_now()
            connection.execute(
                """
                UPDATE jobs
                SET status = ?, result_json = ?, error_json = ?, attempt_count = ?, updated_at = ?
                WHERE job_id = ?
                """,
                (
                    target_status,
                    _json_dump(result) if target_status == "succeeded" else None,
                    _json_dump(error) if target_status == "failed" else None,
                    current.attempt_count if attempt_count is None else int(attempt_count),
                    now,
                    job_id,
                ),
            )
            self._append_event(connection, job_id, event_type, event_payload)
            updated = connection.execute("SELECT * FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
            return self._to_job(updated)

        return self._transaction(operation)

    def mark_running(self, job_id: str, attempt_count: int) -> JobRecord:
        return self._transition(
            str(job_id),
            "running",
            "task_started",
            {"attempt": int(attempt_count)},
            attempt_count=attempt_count,
        )

    def mark_retry_scheduled(self, job_id: str, payload: dict[str, Any]) -> JobRecord:
        return self._transition(str(job_id), "accepted", "task_retry_scheduled", dict(payload or {}))

    def mark_recovery_queued(self, job_id: str, payload: dict[str, Any]) -> JobRecord:
        return self._transition(str(job_id), "accepted", "task_recovery_queued", dict(payload or {}))

    def mark_succeeded(self, job_id: str, result: Any) -> JobRecord:
        return self._transition(str(job_id), "succeeded", "task_succeeded", {"result": result}, result=result)

    def mark_failed(self, job_id: str, error: dict[str, Any]) -> JobRecord:
        return self._transition(str(job_id), "failed", "task_failed", {"error": error}, error=error)

    def append_event(self, job_id: str, event_type: str, payload: dict[str, Any]) -> None:
        def operation(connection: sqlite3.Connection) -> None:
            row = connection.execute("SELECT job_id FROM jobs WHERE job_id = ?", (str(job_id),)).fetchone()
            if row is None:
                raise KeyError(job_id)
            self._append_event(connection, str(job_id), event_type, dict(payload or {}))

        self._transaction(operation)

    def list_events(self, job_id: str, limit: int = 100) -> list[EventRecord]:
        connection = self._connect()
        try:
            rows = connection.execute(
                "SELECT * FROM job_events WHERE job_id = ? ORDER BY event_id ASC LIMIT ?",
                (str(job_id), max(1, min(int(limit), 1000))),
            ).fetchall()
            return [self._to_event(row) for row in rows]
        finally:
            connection.close()

    def healthcheck(self) -> bool:
        connection = self._connect()
        try:
            return connection.execute("SELECT 1").fetchone()[0] == 1
        finally:
            connection.close()

    def close(self) -> None:
        if self._anchor_connection is not None:
            self._anchor_connection.close()
            self._anchor_connection = None
