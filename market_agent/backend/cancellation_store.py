"""Persistent, namespaced cancellation authority for cooperating runtime replicas."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Any, Callable, Iterator, Protocol


class CancellationStore(Protocol):
    def cancel(self, run_id: str) -> None: ...

    def is_cancelled(self, run_id: str) -> bool: ...


def _validate_identifier(value: str, name: str) -> str:
    # IDs are opaque: validate without trimming or interpolating them into SQL.
    if not isinstance(value, str) or not value.strip() or len(value) > 500 or "\x00" in value:
        raise ValueError(f"{name} must be a nonempty string of at most 500 characters without NUL")
    return value


class _SqlCancellationStore:
    def __init__(
        self,
        connection_factory: Callable[[], Any],
        *,
        namespace: str,
        postgres: bool,
    ) -> None:
        if not callable(connection_factory):
            raise TypeError("a cancellation connection factory is required")
        self._namespace = _validate_identifier(namespace, "cancellation namespace")
        self._factory = connection_factory
        self._postgres = postgres

    @contextmanager
    def _transaction(self) -> Iterator[Any]:
        connection = self._factory()
        try:
            cursor = connection.cursor()
            try:
                yield cursor
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                cursor.close()
        finally:
            connection.close()

    def _execute(self, cursor: Any, sql: str, values: tuple[str, ...]) -> None:
        cursor.execute(sql.replace("?", "%s") if self._postgres else sql, values)

    def migrate(self) -> None:
        with self._transaction() as cursor:
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS market_agent_workflow_cancellations ("
                "namespace TEXT NOT NULL, run_id TEXT NOT NULL, "
                "PRIMARY KEY(namespace, run_id))"
            )

    def cancel(self, run_id: str) -> None:
        run_id = _validate_identifier(run_id, "run identifier")
        with self._transaction() as cursor:
            self._execute(
                cursor,
                "INSERT INTO market_agent_workflow_cancellations (namespace, run_id) "
                "VALUES (?, ?) ON CONFLICT(namespace, run_id) DO NOTHING",
                (self._namespace, run_id),
            )

    def is_cancelled(self, run_id: str) -> bool:
        run_id = _validate_identifier(run_id, "run identifier")
        with self._transaction() as cursor:
            self._execute(
                cursor,
                "SELECT 1 FROM market_agent_workflow_cancellations WHERE namespace=? AND run_id=?",
                (self._namespace, run_id),
            )
            return cursor.fetchone() is not None

    def healthcheck(self) -> bool:
        try:
            with self._transaction() as cursor:
                self._execute(
                    cursor,
                    "SELECT 1 FROM market_agent_workflow_cancellations WHERE namespace=? LIMIT 1",
                    (self._namespace,),
                )
                cursor.fetchone()
            return True
        except Exception:
            return False


class SQLiteCancellationStore(_SqlCancellationStore):
    """Durable local/shared-file adapter; migrate when constructing an instance."""

    def __init__(self, path: str | Path, namespace: str = "default") -> None:
        path = Path(path)
        if str(path) == ":memory:":
            raise ValueError("cancellation store requires a durable SQLite file")
        super().__init__(lambda: sqlite3.connect(path, timeout=30), namespace=namespace, postgres=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()


class PostgresCancellationStore(_SqlCancellationStore):
    """Production shared authority with explicit migration and readiness checks."""

    def __init__(
        self,
        connection_factory: Callable[[], Any],
        namespace: str = "default",
    ) -> None:
        super().__init__(connection_factory, namespace=namespace, postgres=True)
