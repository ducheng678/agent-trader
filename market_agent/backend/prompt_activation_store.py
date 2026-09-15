"""Shared, digest-bound prompt activation authority.

The active pointer and its append-only audit record are committed in one SQL
transaction.  Callers use the revision as an optimistic concurrency fence.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import re
import sqlite3
from typing import Any, Callable, Iterator, Protocol


_DIGEST = re.compile(r"[0-9a-f]{64}")


class PromptActivationStoreError(RuntimeError):
    """The shared prompt authority returned malformed state."""


@dataclass(frozen=True, slots=True)
class PromptReleasePointer:
    release_id: str
    release_digest: str
    manifest_hash: str

    def __post_init__(self) -> None:
        if type(self.release_id) is not str or not self.release_id.strip():
            raise ValueError("prompt release pointer requires a release ID")
        if type(self.release_digest) is not str or not _DIGEST.fullmatch(self.release_digest):
            raise ValueError("prompt release pointer digest is invalid")
        if type(self.manifest_hash) is not str or not _DIGEST.fullmatch(self.manifest_hash):
            raise ValueError("prompt release pointer manifest hash is invalid")


@dataclass(frozen=True, slots=True)
class PromptActivationState:
    active: PromptReleasePointer
    previous: PromptReleasePointer | None
    revision: int

    def __post_init__(self) -> None:
        if type(self.active) is not PromptReleasePointer:
            raise TypeError("active prompt release pointer is required")
        if self.previous is not None and type(self.previous) is not PromptReleasePointer:
            raise TypeError("previous prompt release pointer is invalid")
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("prompt activation revision must be positive")


@dataclass(frozen=True, slots=True)
class PendingPromptActivationAudit:
    """One immutable shared transition awaiting at-least-once audit delivery."""

    revision: int
    action: str
    active: PromptReleasePointer
    previous: PromptReleasePointer | None

    def __post_init__(self) -> None:
        if type(self.revision) is not int or self.revision < 1:
            raise ValueError("pending prompt audit revision must be positive")
        if self.action not in {"bootstrap", "activate", "rollback"}:
            raise ValueError("pending prompt audit action is invalid")
        if type(self.active) is not PromptReleasePointer:
            raise TypeError("pending prompt audit active pointer is invalid")
        if self.previous is not None and type(self.previous) is not PromptReleasePointer:
            raise TypeError("pending prompt audit previous pointer is invalid")


class PromptActivationStore(Protocol):
    def read(self) -> PromptActivationState | None: ...

    def bootstrap(self, active: PromptReleasePointer) -> PromptActivationState: ...

    def compare_and_swap(
        self,
        expected_revision: int,
        active: PromptReleasePointer,
        previous: PromptReleasePointer | None,
        action: str,
    ) -> PromptActivationState | None: ...

    def pending_audits(self, *, limit: int = 100) -> tuple[PendingPromptActivationAudit, ...]: ...

    def acknowledge_audit(self, revision: int) -> bool: ...


class _SqlPromptActivationStore:
    _state_columns = (
        "active_release_id",
        "active_release_digest",
        "active_manifest_hash",
        "previous_release_id",
        "previous_release_digest",
        "previous_manifest_hash",
        "revision",
    )

    def __init__(
        self,
        connection_factory: Callable[[], Any],
        *,
        namespace: str,
        postgres: bool,
    ) -> None:
        if not callable(connection_factory):
            raise TypeError("a prompt activation connection factory is required")
        if type(namespace) is not str or not namespace.strip():
            raise ValueError("prompt activation namespace is required")
        self._factory = connection_factory
        self._namespace = namespace
        self._postgres = postgres

    @contextmanager
    def _transaction(self) -> Iterator[Any]:
        connection = self._factory()
        cursor = connection.cursor()
        try:
            if not self._postgres:
                cursor.execute("BEGIN IMMEDIATE")
            yield cursor
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def _execute(self, cursor: Any, sql: str, values: tuple[Any, ...] = ()) -> None:
        cursor.execute(sql.replace("?", "%s") if self._postgres else sql, values)

    def migrate(self) -> None:
        with self._transaction() as cursor:
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS market_agent_prompt_activation_state ("
                "namespace TEXT PRIMARY KEY, active_release_id TEXT NOT NULL, "
                "active_release_digest TEXT NOT NULL, active_manifest_hash TEXT NOT NULL, "
                "previous_release_id TEXT, previous_release_digest TEXT, previous_manifest_hash TEXT, "
                "revision BIGINT NOT NULL CHECK (revision >= 1), "
                "CHECK ((previous_release_id IS NULL AND previous_release_digest IS NULL AND previous_manifest_hash IS NULL) "
                "OR (previous_release_id IS NOT NULL AND previous_release_digest IS NOT NULL AND previous_manifest_hash IS NOT NULL)))"
            )
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS market_agent_prompt_activation_audit ("
                "namespace TEXT NOT NULL, revision BIGINT NOT NULL, action TEXT NOT NULL, "
                "active_release_id TEXT NOT NULL, active_release_digest TEXT NOT NULL, "
                "active_manifest_hash TEXT NOT NULL, previous_release_id TEXT, "
                "previous_release_digest TEXT, previous_manifest_hash TEXT, "
                "PRIMARY KEY(namespace, revision), "
                "CHECK ((previous_release_id IS NULL AND previous_release_digest IS NULL AND previous_manifest_hash IS NULL) "
                "OR (previous_release_id IS NOT NULL AND previous_release_digest IS NOT NULL AND previous_manifest_hash IS NOT NULL)))"
            )
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS market_agent_prompt_activation_delivery ("
                "namespace TEXT NOT NULL, revision BIGINT NOT NULL, "
                "PRIMARY KEY(namespace, revision))"
            )
            if self._postgres:
                cursor.execute(
                    "CREATE OR REPLACE FUNCTION market_agent_prompt_activation_audit_reject_mutation() "
                    "RETURNS trigger AS $$ BEGIN "
                    "RAISE EXCEPTION 'prompt activation audit is append-only'; "
                    "END; $$ LANGUAGE plpgsql"
                )
                cursor.execute(
                    "DROP TRIGGER IF EXISTS market_agent_prompt_activation_audit_no_mutation "
                    "ON market_agent_prompt_activation_audit"
                )
                cursor.execute(
                    "CREATE TRIGGER market_agent_prompt_activation_audit_no_mutation "
                    "BEFORE UPDATE OR DELETE ON market_agent_prompt_activation_audit "
                    "FOR EACH ROW EXECUTE FUNCTION market_agent_prompt_activation_audit_reject_mutation()"
                )
            else:
                cursor.execute(
                    "CREATE TRIGGER IF NOT EXISTS market_agent_prompt_activation_audit_no_update "
                    "BEFORE UPDATE ON market_agent_prompt_activation_audit BEGIN "
                    "SELECT RAISE(ABORT, 'prompt activation audit is append-only'); END"
                )
                cursor.execute(
                    "CREATE TRIGGER IF NOT EXISTS market_agent_prompt_activation_audit_no_delete "
                    "BEFORE DELETE ON market_agent_prompt_activation_audit BEGIN "
                    "SELECT RAISE(ABORT, 'prompt activation audit is append-only'); END"
                )

    def read(self) -> PromptActivationState | None:
        with self._transaction() as cursor:
            return self._select(cursor, for_update=False)

    def bootstrap(self, active: PromptReleasePointer) -> PromptActivationState:
        active = self._require_pointer(active)
        with self._transaction() as cursor:
            self._execute(
                cursor,
                "INSERT INTO market_agent_prompt_activation_state "
                "(namespace,active_release_id,active_release_digest,active_manifest_hash,revision) "
                "VALUES (?,?,?,?,1) ON CONFLICT(namespace) DO NOTHING",
                (self._namespace, active.release_id, active.release_digest, active.manifest_hash),
            )
            if cursor.rowcount == 1:
                # Bootstrap is the first externally visible activation; recording
                # it as such keeps replayed events identical to the caller's hook.
                self._insert_audit(cursor, 1, "activate", active, None)
            state = self._select(cursor, for_update=self._postgres)
            if state is None:
                raise PromptActivationStoreError("prompt activation bootstrap did not create state")
            return state

    def compare_and_swap(
        self,
        expected_revision: int,
        active: PromptReleasePointer,
        previous: PromptReleasePointer | None,
        action: str,
    ) -> PromptActivationState | None:
        if type(expected_revision) is not int or expected_revision < 1:
            raise ValueError("expected prompt activation revision must be positive")
        active = self._require_pointer(active)
        if previous is not None:
            previous = self._require_pointer(previous)
        if type(action) is not str or action not in {"activate", "rollback"}:
            raise ValueError("prompt activation action is invalid")
        revision = expected_revision + 1
        previous_values = self._pointer_values(previous)
        with self._transaction() as cursor:
            state_before = self._select(cursor, for_update=self._postgres)
            if state_before is None or state_before.revision != expected_revision:
                return None
            if action == "activate":
                if previous != state_before.active:
                    raise ValueError("activation must retain the former active prompt release")
                audit_previous = previous
            else:
                if state_before.previous is None or active != state_before.previous or previous is not None:
                    raise ValueError("rollback must restore the previous prompt release")
                # The state clears its rollback pointer, but the immutable event
                # records the release that was active immediately before rollback.
                audit_previous = state_before.active
            self._execute(
                cursor,
                "UPDATE market_agent_prompt_activation_state SET "
                "active_release_id=?,active_release_digest=?,active_manifest_hash=?,"
                "previous_release_id=?,previous_release_digest=?,previous_manifest_hash=?,revision=? "
                "WHERE namespace=? AND revision=?",
                (
                    active.release_id,
                    active.release_digest,
                    active.manifest_hash,
                    *previous_values,
                    revision,
                    self._namespace,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                return None
            self._insert_audit(cursor, revision, action, active, audit_previous)
            return PromptActivationState(active=active, previous=previous, revision=revision)

    def healthcheck(self) -> bool:
        try:
            with self._transaction() as cursor:
                self._execute(
                    cursor,
                    "SELECT revision FROM market_agent_prompt_activation_state WHERE namespace=?",
                    (self._namespace,),
                )
                cursor.fetchone()
            return True
        except Exception:
            return False

    def pending_audits(self, *, limit: int = 100) -> tuple[PendingPromptActivationAudit, ...]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("pending prompt audit limit is invalid")
        with self._transaction() as cursor:
            self._execute(
                cursor,
                "SELECT audit.revision,audit.action,audit.active_release_id,audit.active_release_digest,"
                "audit.active_manifest_hash,audit.previous_release_id,audit.previous_release_digest,"
                "audit.previous_manifest_hash "
                "FROM market_agent_prompt_activation_audit AS audit "
                "LEFT JOIN market_agent_prompt_activation_delivery AS delivery "
                "ON delivery.namespace=audit.namespace AND delivery.revision=audit.revision "
                "WHERE audit.namespace=? AND delivery.revision IS NULL "
                "ORDER BY audit.revision LIMIT ?",
                (self._namespace, limit),
            )
            rows = cursor.fetchall()
        records: list[PendingPromptActivationAudit] = []
        try:
            for row in rows:
                values = tuple(row)
                if len(values) != 8:
                    raise ValueError("wrong pending prompt audit column count")
                previous_values = values[5:8]
                if all(value is None for value in previous_values):
                    previous = None
                elif any(value is None for value in previous_values):
                    raise ValueError("partial previous prompt pointer")
                else:
                    previous = PromptReleasePointer(
                        str(previous_values[0]), str(previous_values[1]), str(previous_values[2])
                    )
                records.append(PendingPromptActivationAudit(
                    revision=values[0],
                    action=str(values[1]),
                    active=PromptReleasePointer(str(values[2]), str(values[3]), str(values[4])),
                    previous=previous,
                ))
        except (TypeError, ValueError) as error:
            raise PromptActivationStoreError("shared prompt activation audit is malformed") from error
        return tuple(records)

    def acknowledge_audit(self, revision: int) -> bool:
        if type(revision) is not int or revision < 1:
            raise ValueError("prompt audit revision must be positive")
        with self._transaction() as cursor:
            self._execute(
                cursor,
                "INSERT INTO market_agent_prompt_activation_delivery (namespace,revision) "
                "SELECT namespace,revision FROM market_agent_prompt_activation_audit "
                "WHERE namespace=? AND revision=? "
                "ON CONFLICT(namespace,revision) DO NOTHING",
                (self._namespace, revision),
            )
            return cursor.rowcount == 1

    def _select(self, cursor: Any, *, for_update: bool) -> PromptActivationState | None:
        query = (
            "SELECT " + ",".join(self._state_columns)
            + " FROM market_agent_prompt_activation_state WHERE namespace=?"
        )
        if for_update and self._postgres:
            query += " FOR UPDATE"
        self._execute(cursor, query, (self._namespace,))
        row = cursor.fetchone()
        if row is None:
            return None
        try:
            values = tuple(row)
            if len(values) != len(self._state_columns):
                raise ValueError("wrong prompt activation column count")
            active = PromptReleasePointer(str(values[0]), str(values[1]), str(values[2]))
            previous_values = values[3:6]
            if all(value is None for value in previous_values):
                previous = None
            elif any(value is None for value in previous_values):
                raise ValueError("partial previous prompt pointer")
            else:
                previous = PromptReleasePointer(
                    str(previous_values[0]), str(previous_values[1]), str(previous_values[2])
                )
            return PromptActivationState(active=active, previous=previous, revision=values[6])
        except (TypeError, ValueError) as error:
            raise PromptActivationStoreError("shared prompt activation state is malformed") from error

    def _insert_audit(
        self,
        cursor: Any,
        revision: int,
        action: str,
        active: PromptReleasePointer,
        previous: PromptReleasePointer | None,
    ) -> None:
        self._execute(
            cursor,
            "INSERT INTO market_agent_prompt_activation_audit "
            "(namespace,revision,action,active_release_id,active_release_digest,active_manifest_hash,"
            "previous_release_id,previous_release_digest,previous_manifest_hash) VALUES (?,?,?,?,?,?,?,?,?)",
            (
                self._namespace,
                revision,
                action,
                active.release_id,
                active.release_digest,
                active.manifest_hash,
                *self._pointer_values(previous),
            ),
        )

    @staticmethod
    def _pointer_values(pointer: PromptReleasePointer | None) -> tuple[str | None, str | None, str | None]:
        if pointer is None:
            return None, None, None
        return pointer.release_id, pointer.release_digest, pointer.manifest_hash

    @staticmethod
    def _require_pointer(pointer: PromptReleasePointer) -> PromptReleasePointer:
        if type(pointer) is not PromptReleasePointer:
            raise TypeError("a concrete prompt release pointer is required")
        return pointer


class SQLitePromptActivationStore(_SqlPromptActivationStore):
    """SQLite shared-authority adapter for offline and multi-manager tests."""

    def __init__(self, path: str | Path, *, namespace: str = "default") -> None:
        path = Path(path)
        if str(path) == ":memory:":
            raise ValueError("prompt activation store requires a durable SQLite file")
        path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(
            lambda: sqlite3.connect(path, timeout=30),
            namespace=namespace,
            postgres=False,
        )
        self.migrate()


class PostgresPromptActivationStore(_SqlPromptActivationStore):
    """PostgreSQL prompt authority for horizontally scaled production managers."""

    def __init__(
        self,
        connection_factory: Callable[[], Any],
        *,
        namespace: str = "default",
    ) -> None:
        super().__init__(connection_factory, namespace=namespace, postgres=True)
