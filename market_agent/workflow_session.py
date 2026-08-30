from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256
import json
from pathlib import Path
import secrets
import sqlite3
from typing import Iterable, Mapping, Protocol

from pydantic import field_validator, model_validator

from market_agent.workflow_contracts import (
    ContractModel,
    Digest,
    FrozenJsonMapping,
    NonNegativeFinite,
    NonNegativeInt,
    ShortText,
)
from market_agent.workflow_harness_contracts import (
    AttemptState,
    HarnessSessionView,
    HarnessTransition,
    LeaseToken,
    RunState,
    WorkItemState,
)


class EventIntegrityError(RuntimeError):
    """The canonical run stream cannot be trusted or replayed."""


class OptimisticConcurrencyError(RuntimeError):
    """The caller's expected run counters are stale."""


class LeaseConflictError(RuntimeError):
    """A lease epoch or fencing token is stale."""


class HarnessEvent(ContractModel):
    event_id: ShortText
    trace_id: ShortText
    span_id: ShortText
    parent_span_id: ShortText | None = None
    run_id: ShortText
    work_item_id: ShortText | None = None
    attempt_id: ShortText | None = None
    sequence: NonNegativeInt = 0
    state_revision: NonNegativeInt = 0
    event_type: ShortText
    occurred_at: datetime
    monotonic_offset: NonNegativeFinite
    actor: ShortText
    payload: FrozenJsonMapping
    transition: HarnessTransition | None = None
    previous_event_hash: Digest | None = None
    event_hash: Digest | None = None

    @field_validator("occurred_at")
    @classmethod
    def require_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() != timedelta(0):
            raise ValueError("harness event timestamps must be UTC")
        return value

    @model_validator(mode="after")
    def validate_transition_identity(self) -> HarnessEvent:
        if self.transition is None:
            return self
        if (
            self.transition.run_id != self.run_id
            or self.transition.trace_id != self.trace_id
        ):
            raise ValueError("event and transition identity must match")
        if self.transition.entity_kind == "run":
            if self.transition.entity_id != self.run_id:
                raise ValueError("run transition entity must match the event run")
        elif self.transition.entity_kind == "work_item":
            if self.work_item_id != self.transition.entity_id:
                raise ValueError("work-item transition entity must match the event")
        elif self.attempt_id != self.transition.entity_id:
            raise ValueError("attempt transition entity must match the event")
        return self


class HarnessEventStore(Protocol):
    def append(
        self,
        event: HarnessEvent,
        *,
        expected_sequence: int,
        expected_state_revision: int,
    ) -> HarnessEvent: ...

    def load(self, run_id: str) -> tuple[HarnessEvent, ...]: ...

    def snapshot(self, run_id: str) -> HarnessSessionView: ...

    def acquire_lease(
        self,
        run_id: str,
        work_item_id: str,
        attempt_id: str,
        holder_id: str,
        *,
        expires_at_monotonic: float,
        expected_lease_epoch: int,
    ) -> LeaseToken: ...


def canonical_event_hash(values: Mapping[str, object]) -> str:
    payload = json.dumps(
        values, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def _event_hash_values(event: HarnessEvent) -> dict[str, object]:
    return event.model_dump(mode="json", exclude={"event_hash"})


def _validated_event(event: HarnessEvent) -> HarnessEvent:
    try:
        return HarnessEvent.model_validate(event.model_dump(mode="python"))
    except Exception as error:
        raise EventIntegrityError("schema mismatch or invalid harness event") from error


def _replace_state(
    values: tuple[tuple[str, object], ...], identifier: str, state: object
) -> tuple[tuple[str, object], ...]:
    by_id = dict(values)
    by_id[identifier] = state
    return tuple(sorted(by_id.items()))


def _apply_committed_event(
    view: HarnessSessionView, event: HarnessEvent
) -> HarnessSessionView:
    if view.run_id is None:
        run_id, trace_id = event.run_id, event.trace_id
    else:
        if event.run_id != view.run_id:
            raise EventIntegrityError("run identity changed during replay")
        if event.trace_id != view.trace_id:
            raise EventIntegrityError("trace identity changed during replay")
        run_id, trace_id = view.run_id, view.trace_id

    transition = event.transition
    if transition is None:
        if event.state_revision != view.state_revision:
            raise EventIntegrityError("non-transition changed state revision")
        return view.model_copy(
            update={
                "sequence": event.sequence,
                "run_id": run_id,
                "trace_id": trace_id,
                "last_event_hash": event.event_hash,
            }
        )

    if transition.expected_state_revision != view.state_revision:
        raise EventIntegrityError("transition expected state revision is stale")
    if event.state_revision != view.state_revision + 1:
        raise EventIntegrityError("transition did not advance state revision")
    if transition.run_id != run_id or transition.trace_id != trace_id:
        raise EventIntegrityError("transition identity changed during replay")

    changes: dict[str, object] = {
        "sequence": event.sequence,
        "state_revision": event.state_revision,
        "plan_revision": transition.plan_revision,
        "run_id": run_id,
        "trace_id": trace_id,
        "applied_idempotency_keys": (
            *view.applied_idempotency_keys,
            transition.idempotency_key,
        ),
        "last_event_hash": event.event_hash,
    }
    if transition.idempotency_key in view.applied_idempotency_keys:
        raise EventIntegrityError("duplicate transition idempotency key")

    if transition.entity_kind == "run":
        current = view.run_state.value if view.run_state is not None else "none"
        if transition.from_state != current:
            raise EventIntegrityError("run transition source does not match replay state")
        try:
            changes["run_state"] = RunState(transition.to_state)
        except ValueError as error:
            raise EventIntegrityError("unknown run state in transition") from error
    elif transition.entity_kind == "work_item":
        current_states = dict(view.work_item_states)
        current = current_states.get(transition.entity_id)
        current_value = current.value if current is not None else "none"
        if transition.from_state != current_value:
            raise EventIntegrityError(
                "work-item transition source does not match replay state"
            )
        try:
            target = WorkItemState(transition.to_state)
        except ValueError as error:
            raise EventIntegrityError("unknown work-item state in transition") from error
        changes["work_item_states"] = _replace_state(
            view.work_item_states, transition.entity_id, target
        )
    else:
        current_states = dict(view.attempt_states)
        current = current_states.get(transition.entity_id)
        current_value = current.value if current is not None else "none"
        if transition.from_state != current_value:
            raise EventIntegrityError(
                "attempt transition source does not match replay state"
            )
        try:
            target = AttemptState(transition.to_state)
        except ValueError as error:
            raise EventIntegrityError("unknown attempt state in transition") from error
        changes["attempt_states"] = _replace_state(
            view.attempt_states, transition.entity_id, target
        )
    try:
        return HarnessSessionView.model_validate(
            view.model_copy(update=changes).model_dump(mode="python")
        )
    except Exception as error:
        raise EventIntegrityError("committed event produces an invalid session view") from error


def fold_events(events: Iterable[HarnessEvent]) -> HarnessSessionView:
    view = HarnessSessionView.empty()
    for expected_sequence, candidate in enumerate(events, start=1):
        if candidate.schema_version != "v1":
            raise EventIntegrityError("schema version mismatch")
        if candidate.sequence != expected_sequence:
            raise EventIntegrityError("non-contiguous run sequence")
        if view.run_id is not None and candidate.run_id != view.run_id:
            raise EventIntegrityError("run identity changed during replay")
        if view.trace_id is not None and candidate.trace_id != view.trace_id:
            raise EventIntegrityError("trace identity changed during replay")
        event = _validated_event(candidate)
        if event.previous_event_hash != view.last_event_hash:
            raise EventIntegrityError("invalid previous event hash link")
        if event.event_hash is None:
            raise EventIntegrityError("committed event hash is missing")
        if canonical_event_hash(_event_hash_values(event)) != event.event_hash:
            raise EventIntegrityError("event hash does not match canonical content")
        view = _apply_committed_event(view, event)
    return view


class SQLiteHarnessEventStore:
    def __init__(self, database_path: str | Path) -> None:
        self._database_path = str(database_path)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self._database_path, timeout=5.0, isolation_level=None
        )
        connection.execute("PRAGMA busy_timeout = 5000")
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA recursive_triggers = ON")
        return connection

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS harness_runs ("
                    "run_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL, "
                    "sequence INTEGER NOT NULL, state_revision INTEGER NOT NULL, "
                    "last_event_hash TEXT)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS harness_events ("
                    "event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, "
                    "trace_id TEXT NOT NULL, sequence INTEGER NOT NULL, "
                    "state_revision INTEGER NOT NULL, event_hash TEXT NOT NULL, "
                    "previous_event_hash TEXT, event_json TEXT NOT NULL, "
                    "UNIQUE(run_id, sequence), "
                    "FOREIGN KEY(run_id) REFERENCES harness_runs(run_id))"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS harness_outbox ("
                    "event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, "
                    "sequence INTEGER NOT NULL, event_json TEXT NOT NULL, "
                    "published INTEGER NOT NULL DEFAULT 0, "
                    "FOREIGN KEY(event_id) REFERENCES harness_events(event_id))"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS harness_leases ("
                    "run_id TEXT NOT NULL, work_item_id TEXT NOT NULL, "
                    "attempt_id TEXT NOT NULL, lease_epoch INTEGER NOT NULL, "
                    "fencing_token TEXT NOT NULL, holder_id TEXT NOT NULL, "
                    "expires_at_monotonic REAL NOT NULL, "
                    "PRIMARY KEY(run_id, work_item_id), "
                    "FOREIGN KEY(run_id) REFERENCES harness_runs(run_id))"
                )
                connection.execute(
                    "CREATE INDEX IF NOT EXISTS harness_events_run_sequence_idx "
                    "ON harness_events(run_id, sequence)"
                )
                self._create_event_triggers(connection)
                connection.execute("COMMIT")
            except BaseException:
                if connection.in_transaction:
                    connection.execute("ROLLBACK")
                raise
        finally:
            connection.close()

    @staticmethod
    def _create_event_triggers(connection: sqlite3.Connection) -> None:
        connection.execute(
            "CREATE TRIGGER IF NOT EXISTS harness_events_no_update "
            "BEFORE UPDATE ON harness_events BEGIN "
            "SELECT RAISE(ABORT, 'harness events are append-only'); END"
        )
        connection.execute(
            "CREATE TRIGGER IF NOT EXISTS harness_events_no_delete "
            "BEFORE DELETE ON harness_events BEGIN "
            "SELECT RAISE(ABORT, 'harness events are append-only'); END"
        )
        connection.execute(
            "CREATE TRIGGER IF NOT EXISTS harness_events_no_replace "
            "BEFORE INSERT ON harness_events WHEN "
            "EXISTS (SELECT 1 FROM harness_events WHERE event_id = NEW.event_id) "
            "OR EXISTS (SELECT 1 FROM harness_events "
            "WHERE run_id = NEW.run_id AND sequence = NEW.sequence) BEGIN "
            "SELECT RAISE(ABORT, 'harness events are append-only'); END"
        )

    @staticmethod
    def _load_in_transaction(
        connection: sqlite3.Connection, run_id: str
    ) -> tuple[HarnessEvent, ...]:
        rows = connection.execute(
            "SELECT event_json FROM harness_events "
            "WHERE run_id = ? ORDER BY sequence", (run_id,)
        ).fetchall()
        try:
            return tuple(HarnessEvent.model_validate_json(row[0]) for row in rows)
        except Exception as error:
            raise EventIntegrityError("persisted harness event is invalid") from error

    def append(
        self,
        event: HarnessEvent,
        *,
        expected_sequence: int,
        expected_state_revision: int,
    ) -> HarnessEvent:
        event = _validated_event(event)
        if (
            event.sequence != 0
            or event.state_revision != 0
            or event.previous_event_hash is not None
            or event.event_hash is not None
        ):
            raise ValueError("event counters and hashes are assigned by the store")
        if (
            isinstance(expected_sequence, bool)
            or not isinstance(expected_sequence, int)
            or expected_sequence < 0
            or isinstance(expected_state_revision, bool)
            or not isinstance(expected_state_revision, int)
            or expected_state_revision < 0
        ):
            raise ValueError("expected counters must be nonnegative integers")

        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT trace_id, sequence, state_revision, last_event_hash "
                "FROM harness_runs WHERE run_id = ?", (event.run_id,)
            ).fetchone()
            if row is None:
                trace_id, sequence, state_revision, previous_hash = (
                    event.trace_id,
                    0,
                    0,
                    None,
                )
                connection.execute(
                    "INSERT INTO harness_runs "
                    "(run_id, trace_id, sequence, state_revision, last_event_hash) "
                    "VALUES (?, ?, 0, 0, NULL)",
                    (event.run_id, event.trace_id),
                )
            else:
                trace_id, sequence, state_revision, previous_hash = row
            if trace_id != event.trace_id:
                raise EventIntegrityError("trace identity changed within run")
            if (
                sequence != expected_sequence
                or state_revision != expected_state_revision
            ):
                raise OptimisticConcurrencyError(
                    "run sequence or state revision changed"
                )

            transition = event.transition
            if transition is not None:
                if transition.expected_state_revision != state_revision:
                    raise OptimisticConcurrencyError(
                        "transition expected state revision changed"
                    )
                next_revision = state_revision + 1
                self._validate_fencing(connection, event)
            else:
                next_revision = state_revision
            next_sequence = sequence + 1
            committed = event.model_copy(
                update={
                    "sequence": next_sequence,
                    "state_revision": next_revision,
                    "previous_event_hash": previous_hash,
                }
            )
            committed = committed.model_copy(
                update={"event_hash": canonical_event_hash(_event_hash_values(committed))}
            )
            committed = _validated_event(committed)

            prior_events = self._load_in_transaction(connection, event.run_id)
            fold_events((*prior_events, committed))
            rendered = json.dumps(
                committed.model_dump(mode="json"),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            connection.execute(
                "INSERT INTO harness_events "
                "(event_id, run_id, trace_id, sequence, state_revision, event_hash, "
                "previous_event_hash, event_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    committed.event_id,
                    committed.run_id,
                    committed.trace_id,
                    committed.sequence,
                    committed.state_revision,
                    committed.event_hash,
                    committed.previous_event_hash,
                    rendered,
                ),
            )
            connection.execute(
                "INSERT INTO harness_outbox "
                "(event_id, run_id, sequence, event_json) VALUES (?, ?, ?, ?)",
                (
                    committed.event_id,
                    committed.run_id,
                    committed.sequence,
                    rendered,
                ),
            )
            connection.execute(
                "UPDATE harness_runs SET sequence = ?, state_revision = ?, "
                "last_event_hash = ? WHERE run_id = ?",
                (
                    committed.sequence,
                    committed.state_revision,
                    committed.event_hash,
                    committed.run_id,
                ),
            )
            connection.execute("COMMIT")
            return committed
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    @staticmethod
    def _validate_fencing(
        connection: sqlite3.Connection, event: HarnessEvent
    ) -> None:
        transition = event.transition
        if transition is None or transition.fencing_token is None:
            return
        if event.work_item_id is None:
            raise LeaseConflictError("fencing token requires a work item")
        row = connection.execute(
            "SELECT attempt_id, fencing_token FROM harness_leases "
            "WHERE run_id = ? AND work_item_id = ?",
            (event.run_id, event.work_item_id),
        ).fetchone()
        if (
            row is None
            or row[1] != transition.fencing_token
            or event.attempt_id != row[0]
        ):
            raise LeaseConflictError("stale lease fencing token")

    def load(self, run_id: str) -> tuple[HarnessEvent, ...]:
        connection = self._connect()
        try:
            return self._load_in_transaction(connection, run_id)
        finally:
            connection.close()

    def snapshot(self, run_id: str) -> HarnessSessionView:
        return fold_events(self.load(run_id))

    def acquire_lease(
        self,
        run_id: str,
        work_item_id: str,
        attempt_id: str,
        holder_id: str,
        *,
        expires_at_monotonic: float,
        expected_lease_epoch: int,
    ) -> LeaseToken:
        if (
            isinstance(expected_lease_epoch, bool)
            or not isinstance(expected_lease_epoch, int)
            or expected_lease_epoch < 0
        ):
            raise ValueError("expected lease epoch must be a nonnegative integer")
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            if connection.execute(
                "SELECT 1 FROM harness_runs WHERE run_id = ?", (run_id,)
            ).fetchone() is None:
                raise EventIntegrityError("cannot lease work for an unknown run")
            row = connection.execute(
                "SELECT lease_epoch FROM harness_leases "
                "WHERE run_id = ? AND work_item_id = ?",
                (run_id, work_item_id),
            ).fetchone()
            current_epoch = 0 if row is None else int(row[0])
            if current_epoch != expected_lease_epoch:
                raise LeaseConflictError("lease epoch changed")
            next_epoch = current_epoch + 1
            lease = LeaseToken(
                run_id=run_id,
                work_item_id=work_item_id,
                attempt_id=attempt_id,
                lease_epoch=next_epoch,
                fencing_token=f"fence-{next_epoch}-{secrets.token_hex(16)}",
                holder_id=holder_id,
                expires_at_monotonic=expires_at_monotonic,
            )
            connection.execute(
                "INSERT INTO harness_leases "
                "(run_id, work_item_id, attempt_id, lease_epoch, fencing_token, "
                "holder_id, expires_at_monotonic) VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(run_id, work_item_id) DO UPDATE SET "
                "attempt_id = excluded.attempt_id, "
                "lease_epoch = excluded.lease_epoch, "
                "fencing_token = excluded.fencing_token, "
                "holder_id = excluded.holder_id, "
                "expires_at_monotonic = excluded.expires_at_monotonic",
                (
                    lease.run_id,
                    lease.work_item_id,
                    lease.attempt_id,
                    lease.lease_epoch,
                    lease.fencing_token,
                    lease.holder_id,
                    lease.expires_at_monotonic,
                ),
            )
            connection.execute("COMMIT")
            return lease
        except BaseException:
            if connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()
