"""Tenant-scoped promotion progress with short, generation-fenced transactions.

Only completed and durably observed evaluations may be checkpointed. A generation
fences entire batches, including wraparound; lexical ID comparisons cannot fence
an older worker once a newer worker has wrapped. No evaluation holds a SQL lock.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Any, Callable, Iterator, Protocol


@dataclass(frozen=True, slots=True)
class PromotionCursorState:
    record_id: str | None
    generation: int

    def __post_init__(self) -> None:
        if self.record_id is not None:
            _require_text(self.record_id, "record ID")
        if type(self.generation) is not int or self.generation < 0:
            raise ValueError("promotion cursor generation must be nonnegative")


class PromotionCursorStore(Protocol):
    def read(self, *, tenant_id: str) -> PromotionCursorState: ...

    def checkpoint(self, *, tenant_id: str, expected_generation: int,
                   record_id: str) -> PromotionCursorState | None: ...


def _require_text(value: str, field: str) -> None:
    if type(value) is not str or not value.strip():
        raise ValueError(f"promotion cursor {field} is required")


class _SqlPromotionCursorStore:
    def __init__(self, connection_factory: Callable[[], Any], *, namespace: str,
                 postgres: bool) -> None:
        if not callable(connection_factory):
            raise TypeError("promotion cursor requires a connection factory")
        _require_text(namespace, "namespace")
        self._factory = connection_factory
        self._namespace = namespace
        self._postgres = postgres

    @contextmanager
    def _transaction(self) -> Iterator[Any]:
        connection = self._factory()
        cursor = None
        try:
            cursor = connection.cursor()
            if not self._postgres:
                cursor.execute("BEGIN IMMEDIATE")
            yield cursor
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            if cursor is not None:
                cursor.close()
            connection.close()

    def _execute(self, cursor: Any, sql: str, values: tuple[Any, ...] = ()) -> None:
        cursor.execute(sql.replace("?", "%s") if self._postgres else sql, values)

    def migrate(self) -> None:
        with self._transaction() as cursor:
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS market_agent_promotion_cursor ("
                "namespace TEXT NOT NULL, tenant_id TEXT NOT NULL, record_id TEXT, "
                "generation BIGINT NOT NULL CHECK (generation >= 0), "
                "PRIMARY KEY(namespace, tenant_id))"
            )

    def healthcheck(self) -> bool:
        with self._transaction() as cursor:
            cursor.execute(
                "SELECT namespace, tenant_id, record_id, generation "
                "FROM market_agent_promotion_cursor WHERE 1=0"
            )
        return True

    def read(self, *, tenant_id: str) -> PromotionCursorState:
        _require_text(tenant_id, "tenant ID")
        with self._transaction() as cursor:
            self._execute(cursor,
                "INSERT INTO market_agent_promotion_cursor "
                "(namespace,tenant_id,record_id,generation) VALUES (?,?,NULL,0) "
                "ON CONFLICT(namespace,tenant_id) DO NOTHING",
                (self._namespace, tenant_id))
            self._execute(cursor,
                "SELECT record_id,generation FROM market_agent_promotion_cursor "
                "WHERE namespace=? AND tenant_id=?", (self._namespace, tenant_id))
            row = cursor.fetchone()
            if row is None:
                raise RuntimeError("promotion cursor initialization failed")
            return PromotionCursorState(record_id=row[0], generation=row[1])

    def checkpoint(self, *, tenant_id: str, expected_generation: int,
                   record_id: str) -> PromotionCursorState | None:
        _require_text(tenant_id, "tenant ID")
        _require_text(record_id, "record ID")
        state = PromotionCursorState(record_id=record_id, generation=expected_generation)
        with self._transaction() as cursor:
            # UPDATE serializes contenders on this namespace/tenant row. An old
            # batch must stop on a failed CAS instead of reading and overwriting.
            self._execute(cursor,
                "UPDATE market_agent_promotion_cursor SET record_id=?, generation=? "
                "WHERE namespace=? AND tenant_id=? AND generation=?",
                (record_id, state.generation + 1, self._namespace, tenant_id, state.generation))
            if cursor.rowcount != 1:
                return None
            return PromotionCursorState(record_id=record_id, generation=state.generation + 1)


class SQLitePromotionCursorStore(_SqlPromotionCursorStore):
    def __init__(self, path: str | Path, *, namespace: str = "memory-promotion") -> None:
        path = Path(path)
        if str(path) == ":memory:":
            raise ValueError("promotion cursor requires a durable SQLite file")
        path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(lambda: sqlite3.connect(path, timeout=30),
                         namespace=namespace, postgres=False)
        self.migrate()


class PostgresPromotionCursorStore(_SqlPromotionCursorStore):
    """Migration is explicit; constructing the store does not contact PostgreSQL."""

    def __init__(self, connection_factory: Callable[[], Any], *,
                 namespace: str = "memory-promotion") -> None:
        super().__init__(connection_factory, namespace=namespace, postgres=True)
