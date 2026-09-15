"""Bounded trace history shared by backend replicas in the same namespace."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import sqlite3
from typing import Any, Callable, Iterator

from pydantic import TypeAdapter

from market_agent.backend.trace_observability import StoredTraceEvent, TracePage
from market_agent.workflow_structured_logging import StructuredEvent
from market_agent.workflow_tracing import TraceId


class _SQLSharedTraceSink:
    def __init__(self, connection_factory: Callable[[], Any], *, namespace: str,
                 capacity: int, maximum_query: int, postgres: bool) -> None:
        if type(capacity) is not int or not 1 <= capacity <= 100000:
            raise ValueError("trace event capacity must be between 1 and 100000")
        if type(maximum_query) is not int or not 1 <= maximum_query <= min(500, capacity):
            raise ValueError("trace query limit exceeds its bounded capacity")
        if not isinstance(namespace, str) or not namespace.strip() or len(namespace) > 500 or "\x00" in namespace:
            raise ValueError("trace namespace must be a nonempty string of at most 500 characters without NUL")
        if not callable(connection_factory):
            raise TypeError("a trace connection factory is required")
        self._factory = connection_factory
        self._namespace = namespace
        self._capacity, self._maximum_query = capacity, maximum_query
        self._postgres = postgres

    @contextmanager
    def _transaction(self, *, write: bool = False) -> Iterator[Any]:
        connection = self._factory()
        try:
            cursor = connection.cursor()
            try:
                # Explicit BEGIN also protects factories using autocommit. PostgreSQL
                # writers additionally serialize on the namespace counter row.
                cursor.execute("BEGIN IMMEDIATE" if write and not self._postgres else "BEGIN")
                yield cursor
                connection.commit()
            except BaseException:
                connection.rollback()
                raise
            finally:
                cursor.close()
        finally:
            connection.close()

    def _execute(self, cursor: Any, sql: str, parameters: tuple[Any, ...]) -> None:
        cursor.execute(sql.replace("?", "%s") if self._postgres else sql, parameters)

    def migrate(self) -> None:
        with self._transaction(write=True) as cursor:
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS market_agent_trace_sequences ("
                "namespace TEXT PRIMARY KEY, sequence BIGINT NOT NULL CHECK(sequence>=0))"
            )
            cursor.execute(
                "CREATE TABLE IF NOT EXISTS market_agent_trace_events ("
                "namespace TEXT NOT NULL, sequence BIGINT NOT NULL CHECK(sequence>0), "
                "trace_id TEXT NOT NULL, event_json TEXT NOT NULL, PRIMARY KEY(namespace, sequence))"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS market_agent_trace_events_lookup "
                "ON market_agent_trace_events(namespace, trace_id, sequence)"
            )

    def record(self, event: StructuredEvent) -> None:
        event = StructuredEvent.model_validate(event)
        body = event.model_dump_json()
        # Exercise the persisted representation's validation before writing too.
        StructuredEvent.model_validate_json(body)
        with self._transaction(write=True) as cursor:
            self._execute(cursor,
                "INSERT INTO market_agent_trace_sequences(namespace, sequence) VALUES (?, 0) "
                "ON CONFLICT(namespace) DO NOTHING", (self._namespace,))
            self._execute(cursor,
                "SELECT sequence FROM market_agent_trace_sequences WHERE namespace=?" +
                (" FOR UPDATE" if self._postgres else ""), (self._namespace,))
            sequence = cursor.fetchone()[0] + 1
            self._execute(cursor,
                "UPDATE market_agent_trace_sequences SET sequence=? WHERE namespace=?",
                (sequence, self._namespace))
            self._execute(cursor,
                "INSERT INTO market_agent_trace_events(namespace, sequence, trace_id, event_json) "
                "VALUES (?, ?, ?, ?)", (self._namespace, sequence, event.trace.trace_id, body))
            self._execute(cursor,
                "DELETE FROM market_agent_trace_events WHERE namespace=? AND sequence<=?",
                (self._namespace, sequence - self._capacity))

    def query(self, trace_id: str, *, after_sequence: int = 0, limit: int = 100) -> TracePage:
        trace_id = TypeAdapter(TraceId).validate_python(trace_id, strict=True)
        if type(after_sequence) is not int or after_sequence < 0 or type(limit) is not int or limit < 1:
            raise ValueError("trace query cursor and limit are invalid")
        size = min(limit, self._maximum_query)
        with self._transaction() as cursor:
            # One statement gives oldest and matches the same snapshot even under
            # PostgreSQL READ COMMITTED. LEFT JOIN preserves metadata for no matches.
            self._execute(cursor,
                "SELECT matches.sequence, matches.event_json, bounds.oldest FROM "
                "(SELECT COALESCE(MIN(sequence), 0) AS oldest FROM market_agent_trace_events "
                "WHERE namespace=?) bounds LEFT JOIN "
                "(SELECT sequence, event_json FROM market_agent_trace_events "
                "WHERE namespace=? AND trace_id=? AND sequence>? ORDER BY sequence LIMIT ?) matches "
                "ON 1=1 ORDER BY matches.sequence",
                (self._namespace, self._namespace, trace_id, min(after_sequence, 2**63 - 1), size + 1))
            rows = cursor.fetchall()
        oldest = rows[0][2]
        items = []
        for sequence, body, _ in rows:
            if sequence is None:
                continue
            event = StructuredEvent.model_validate_json(body)
            event.trace.assert_same_trace(trace_id)
            items.append(StoredTraceEvent(sequence, event))
        has_more = len(items) > size
        selected = tuple(items[:size])
        return TracePage(items=selected, next_cursor=selected[-1].sequence if selected else after_sequence,
                         oldest_available_sequence=oldest, has_more=has_more,
                         truncated=oldest > 1 and after_sequence < oldest - 1)

    def healthcheck(self) -> bool:
        try:
            with self._transaction() as cursor:
                for table in ("market_agent_trace_sequences", "market_agent_trace_events"):
                    self._execute(cursor, f"SELECT sequence FROM {table} WHERE namespace=? LIMIT 1",
                                  (self._namespace,))
                    cursor.fetchone()
            return True
        except Exception:
            return False


class SQLiteSharedTraceSink(_SQLSharedTraceSink):
    """File-backed trace history; schema migration occurs at construction."""

    def __init__(self, path: str | Path, namespace: str = "default", capacity: int = 10000,
                 maximum_query: int = 100) -> None:
        path = Path(path)
        if str(path) == ":memory:":
            raise ValueError("shared trace storage requires a durable SQLite file")
        super().__init__(lambda: sqlite3.connect(path, timeout=30), namespace=namespace,
                         capacity=capacity, maximum_query=maximum_query, postgres=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.migrate()


class PostgresSharedTraceSink(_SQLSharedTraceSink):
    """Production history with explicit migration and namespace-serialized writes."""

    def __init__(self, connection_factory: Callable[[], Any], namespace: str,
                 capacity: int = 10000, maximum_query: int = 100) -> None:
        super().__init__(connection_factory, namespace=namespace, capacity=capacity,
                         maximum_query=maximum_query, postgres=True)
