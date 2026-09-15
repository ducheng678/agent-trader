from __future__ import annotations

import sqlite3

import pytest

from market_agent.backend.cancellation_store import (
    PostgresCancellationStore,
    SQLiteCancellationStore,
)
from market_agent.workflow_cancellation import WorkflowCancellationRegistry


def test_existing_signal_observes_other_instance_and_restart(tmp_path):
    path = tmp_path / "cancellations.sqlite3"
    first = WorkflowCancellationRegistry(SQLiteCancellationStore(path, namespace="tenant-a"))
    second = WorkflowCancellationRegistry(SQLiteCancellationStore(path, namespace="tenant-a"))
    other = WorkflowCancellationRegistry(SQLiteCancellationStore(path, namespace="tenant-b"))
    signal = second.signal("run-1")
    assert not signal.is_cancelled()

    first.cancel("run-1")
    first.cancel("run-1")

    assert signal.is_cancelled()
    assert not second.signal("run-2").is_cancelled()
    assert not other.signal("run-1").is_cancelled()
    restarted = WorkflowCancellationRegistry(SQLiteCancellationStore(path, namespace="tenant-a"))
    assert restarted.signal("run-1").is_cancelled()


class _IntermittentStore:
    def __init__(self):
        self.unavailable = False
        self.cancelled = set()
        self.before_write = lambda: None

    def cancel(self, run_id):
        self.before_write()
        if self.unavailable:
            raise OSError("shared storage unavailable")
        self.cancelled.add(run_id)

    def is_cancelled(self, run_id):
        if self.unavailable:
            raise OSError("shared storage unavailable")
        return run_id in self.cancelled


def test_outage_fails_closed_without_becoming_explicit_cancellation():
    store = _IntermittentStore()
    registry = WorkflowCancellationRegistry(store)
    signal = registry.signal("run-1")
    assert registry.store is store
    assert not signal.is_cancelled()
    store.unavailable = True
    assert signal.is_cancelled()
    with pytest.raises(OSError, match="unavailable"):
        registry.cancel("run-1")
    store.unavailable = False
    assert not signal.is_cancelled()
    assert store.cancelled == set()


def test_local_latch_is_set_only_after_shared_write_and_survives_outage():
    store = _IntermittentStore()
    registry = WorkflowCancellationRegistry(store)
    signal = registry.signal("run-1")
    store.before_write = lambda: pytest.fail("local cancellation preceded durable write") if signal.is_cancelled() else None
    registry.cancel("run-1")
    assert store.cancelled == {"run-1"}
    store.is_cancelled = lambda _: pytest.fail("local latch did not short circuit shared read")
    assert signal.is_cancelled()
    with pytest.raises(AttributeError):
        registry.store = None


def test_run_ids_are_opaque_and_sql_values(tmp_path):
    store = SQLiteCancellationStore(tmp_path / "cancellations.sqlite3", namespace="tenant';--")
    registry = WorkflowCancellationRegistry(store)
    opaque = " run'); DROP TABLE market_agent_workflow_cancellations;-- "
    registry.cancel(opaque)
    registry.cancel("x" * 500)
    assert registry.signal(opaque).is_cancelled()
    assert not registry.signal(opaque.strip()).is_cancelled()
    assert store.is_cancelled("x" * 500)
    assert store.healthcheck()


@pytest.mark.parametrize("invalid", [None, 42, "", "   ", "x" * 501, "bad\x00id"])
def test_invalid_ids_never_open_sql_connection(invalid):
    def unexpected_connection():
        pytest.fail("invalid ID reached SQL")

    store = PostgresCancellationStore(unexpected_connection)
    registry = WorkflowCancellationRegistry(store)
    for action in (store.cancel, store.is_cancelled, registry.cancel, registry.signal):
        with pytest.raises(ValueError):
            action(invalid)


@pytest.mark.parametrize("invalid", [None, 42, "", "   ", "x" * 501, "bad\x00namespace"])
def test_invalid_namespace_never_opens_sql_connection(tmp_path, invalid):
    with pytest.raises(ValueError):
        PostgresCancellationStore(lambda: pytest.fail("invalid namespace reached SQL"), namespace=invalid)
    with pytest.raises(ValueError):
        SQLiteCancellationStore(tmp_path / "cancellations.sqlite3", namespace=invalid)


def test_sqlite_requires_durable_file():
    with pytest.raises(ValueError):
        SQLiteCancellationStore(":memory:")


class _PgCursor:
    def __init__(self, connection):
        self.connection = connection
        self.cursor = connection.sqlite.cursor()

    def execute(self, sql, parameters=()):
        assert "?" not in sql
        assert "%s" in sql or not parameters
        self.connection.calls.append((sql, parameters))
        self.cursor.execute(sql.replace("%s", "?"), parameters)

    def fetchone(self):
        return self.cursor.fetchone()

    def close(self):
        self.cursor.close()
        self.connection.cursor_closed = True


class _PgConnection:
    def __init__(self, path, calls, *, fail_commit=False):
        self.sqlite = sqlite3.connect(path)
        self.calls = calls
        self.fail_commit = fail_commit
        self.committed = self.rolled_back = self.closed = self.cursor_closed = False

    def cursor(self):
        return _PgCursor(self)

    def commit(self):
        if self.fail_commit:
            raise OSError("commit unavailable")
        self.sqlite.commit()
        self.committed = True

    def rollback(self):
        self.sqlite.rollback()
        self.rolled_back = True

    def close(self):
        self.sqlite.close()
        self.closed = True


def test_postgres_explicit_migration_parameterization_and_transactions(tmp_path):
    calls, connections = [], []

    def connect():
        connection = _PgConnection(tmp_path / "pg-contract.sqlite3", calls)
        connections.append(connection)
        return connection

    namespace, run_id = "tenant';--", " run');-- "
    store = PostgresCancellationStore(connect, namespace=namespace)
    assert calls == []
    assert not store.healthcheck()
    store.migrate()
    assert store.healthcheck()
    assert not store.is_cancelled(run_id)
    store.cancel(run_id)
    store.cancel(run_id)
    assert store.is_cancelled(run_id)
    assert not PostgresCancellationStore(connect, namespace="other").is_cancelled(run_id)
    inserts = [(sql, params) for sql, params in calls if sql.startswith("INSERT")]
    assert len(inserts) == 2
    assert all("ON CONFLICT" in sql and "DO NOTHING" in sql for sql, _ in inserts)
    assert all(params == (namespace, run_id) for _, params in inserts)
    assert all(namespace not in sql and run_id not in sql for sql, _ in calls)
    assert connections[0].rolled_back
    assert all(c.committed for c in connections[1:])
    assert all(c.closed and c.cursor_closed for c in connections)


def test_postgres_commit_failure_does_not_latch_or_persist(tmp_path):
    path = tmp_path / "pg-contract.sqlite3"
    store = PostgresCancellationStore(lambda: _PgConnection(path, []))
    store.migrate()
    failed = _PgConnection(path, [], fail_commit=True)
    registry = WorkflowCancellationRegistry(PostgresCancellationStore(lambda: failed))
    signal = registry.signal("run-1")
    with pytest.raises(OSError, match="commit unavailable"):
        registry.cancel("run-1")
    assert not signal._event.is_set()
    assert not store.is_cancelled("run-1")
    assert failed.rolled_back and failed.closed and failed.cursor_closed


def test_postgres_cursor_creation_failure_closes_connection():
    class Connection:
        closed = False

        def cursor(self):
            raise OSError("cursor unavailable")

        def close(self):
            self.closed = True

    connection = Connection()
    store = PostgresCancellationStore(lambda: connection)
    assert not store.healthcheck()
    assert connection.closed
