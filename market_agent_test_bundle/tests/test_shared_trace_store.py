from concurrent.futures import ThreadPoolExecutor
import sqlite3
from threading import Event

import pytest
from pydantic import ValidationError

from market_agent.backend.shared_trace_store import PostgresSharedTraceSink, SQLiteSharedTraceSink
from market_agent.backend.trace_observability import BoundedTraceSink
from market_agent.workflow_structured_logging import StructuredEvent, summarize_payload
from market_agent.workflow_tracing import TraceContext


def event(trace=None):
    return StructuredEvent.create(trace or TraceContext.new_request(), event="request_started",
                                  actor="ingress", status="started", workflow_id="secret-id",
                                  payload=summarize_payload({"secret": "never store this"}))


def test_cross_instance_restart_privacy_and_namespace(tmp_path):
    path = tmp_path / "traces.db"
    first = SQLiteSharedTraceSink(path, namespace="tenant';--")
    second = SQLiteSharedTraceSink(path, namespace="tenant';--")
    other = SQLiteSharedTraceSink(path, namespace="other")
    original = event()
    first.record(original)
    assert second.query(original.trace.trace_id).items[0].event == original
    assert other.query(original.trace.trace_id).items == ()
    other.record(original)
    assert other.query(original.trace.trace_id).items[0].sequence == 1
    restarted = SQLiteSharedTraceSink(path, namespace="tenant';--")
    restarted.record(original)
    assert [x.sequence for x in restarted.query(original.trace.trace_id).items] == [1, 2]
    with sqlite3.connect(path) as connection:
        raw = connection.execute("SELECT event_json FROM market_agent_trace_events").fetchone()[0]
    assert "secret-id" not in raw and "never store this" not in raw
    assert restarted.healthcheck()


def test_capacity_pagination_matches_local_contract(tmp_path):
    shared = SQLiteSharedTraceSink(tmp_path / "traces.db", capacity=3, maximum_query=2)
    local = BoundedTraceSink(capacity=3, maximum_query=2)
    trace, other = TraceContext.new_request(), TraceContext.new_request()
    for context in (trace, other, trace, trace, other):
        item = event(context)
        shared.record(item)
        local.record(item)
    for context in (trace, other, TraceContext.new_request()):
        for cursor in (0, 1, 2, 3, 5, 100, 2**80):
            for limit in (1, 2, 100):
                assert shared.query(context.trace_id, after_sequence=cursor, limit=limit) == local.query(
                    context.trace_id, after_sequence=cursor, limit=limit)


def test_serialized_commit_and_concurrent_cursor_never_skip(tmp_path):
    path = tmp_path / "traces.db"
    first, second = SQLiteSharedTraceSink(path), SQLiteSharedTraceSink(path)
    trace = TraceContext.new_request()
    pending, release, second_started = Event(), Event(), Event()
    factory = first._factory

    class PausedCommit:
        def __init__(self):
            self.connection = factory()

        def __getattr__(self, name):
            return getattr(self.connection, name)

        def commit(self):
            pending.set()
            assert release.wait(5)
            self.connection.commit()

    first._factory = PausedCommit

    def next_writer():
        second_started.set()
        second.record(event(trace))

    with ThreadPoolExecutor(max_workers=2) as pool:
        slow = pool.submit(first.record, event(trace))
        try:
            assert pending.wait(5)
            fast = pool.submit(next_writer)
            assert second_started.wait(5)
            assert second.query(trace.trace_id).items == ()
        finally:
            release.set()
        slow.result(timeout=5)
        fast.result(timeout=5)
    first._factory = factory
    seen = []
    cursor = 0
    with ThreadPoolExecutor(max_workers=2) as pool:
        def append(sink):
            for _ in range(20):
                sink.record(event(trace))
        writers = [pool.submit(append, sink) for sink in (first, second)]
        while True:
            page = second.query(trace.trace_id, after_sequence=cursor, limit=3)
            seen.extend(item.sequence for item in page.items)
            cursor = page.next_cursor
            if all(writer.done() for writer in writers) and not page.items:
                break
        for writer in writers:
            writer.result()
    assert seen == list(range(1, 43))


@pytest.mark.parametrize("invalid", [None, 42, "", "   ", "x" * 501, "bad\x00tenant"])
def test_invalid_namespace(tmp_path, invalid):
    with pytest.raises(ValueError):
        SQLiteSharedTraceSink(tmp_path / "traces.db", namespace=invalid)
    with pytest.raises(ValueError):
        PostgresSharedTraceSink(lambda: pytest.fail("unexpected SQL"), namespace=invalid)


@pytest.mark.parametrize("kwargs", [{"capacity": True}, {"capacity": 0}, {"capacity": 100001},
                                       {"maximum_query": 0}, {"maximum_query": True},
                                       {"maximum_query": 501}, {"capacity": 2}])
def test_invalid_bounds(tmp_path, kwargs):
    with pytest.raises(ValueError):
        SQLiteSharedTraceSink(tmp_path / "traces.db", **kwargs)
    with pytest.raises(ValueError):
        PostgresSharedTraceSink(lambda: pytest.fail("unexpected SQL"), namespace="default", **kwargs)


@pytest.mark.parametrize("kwargs", [{"after_sequence": -1}, {"after_sequence": True},
                                       {"limit": 0}, {"limit": True}, {"limit": "2"}])
def test_invalid_query_does_not_connect(kwargs):
    sink = PostgresSharedTraceSink(lambda: pytest.fail("unexpected SQL"), namespace="default")
    with pytest.raises(ValueError):
        sink.query(TraceContext.new_request().trace_id, **kwargs)


def test_malformed_event_query_and_stored_json_rejected(tmp_path):
    path = tmp_path / "traces.db"
    sink = SQLiteSharedTraceSink(path)
    original = event()
    with pytest.raises(ValidationError):
        sink.record({"raw_secret": "forbidden"})
    with pytest.raises(ValidationError):
        sink.query("invalid")
    sink.record(original)
    assert sink.query(original.trace.trace_id).items[0].sequence == 1
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE market_agent_trace_events SET event_json='{}'")
    with pytest.raises(ValidationError):
        sink.query(original.trace.trace_id)


def test_stored_event_trace_mismatch_rejected(tmp_path):
    path = tmp_path / "traces.db"
    sink = SQLiteSharedTraceSink(path)
    original = event()
    sink.record(original)
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE market_agent_trace_events SET event_json=?", (event().model_dump_json(),))
    with pytest.raises(ValueError, match="trace"):
        sink.query(original.trace.trace_id)


def test_outage_raises_without_local_fallback(tmp_path):
    path = tmp_path / "traces.db"
    sink = SQLiteSharedTraceSink(path)
    original = event()
    sink.record(original)
    with sqlite3.connect(path) as connection:
        connection.execute("DROP TABLE market_agent_trace_events")
    assert not sink.healthcheck()
    with pytest.raises(sqlite3.OperationalError):
        sink.query(original.trace.trace_id)
    with pytest.raises(sqlite3.OperationalError):
        sink.record(original)


class PgConnection:
    def __init__(self, path, calls, fail_commit=False):
        self.sql = sqlite3.connect(path)
        self.calls = calls
        self.fail_commit = fail_commit
        self.rolled_back = self.closed = self.cursor_closed = False

    def cursor(self):
        connection = self
        delegate = self.sql.cursor()

        class Cursor:
            def execute(self, sql, params=()):
                assert "?" not in sql
                connection.calls.append((sql, params))
                delegate.execute(sql.replace("%s", "?").replace(" FOR UPDATE", ""), params)

            def fetchone(self):
                return delegate.fetchone()

            def fetchall(self):
                return delegate.fetchall()

            def close(self):
                delegate.close()
                connection.cursor_closed = True

        return Cursor()

    def commit(self):
        if self.fail_commit:
            raise OSError("commit unavailable")
        self.sql.commit()

    def rollback(self):
        self.rolled_back = True
        self.sql.rollback()

    def close(self):
        self.closed = True
        self.sql.close()


def test_postgres_explicit_migration_sql_contract_and_rollback(tmp_path):
    path, calls, connections = tmp_path / "pg.db", [], []

    def connect():
        connection = PgConnection(path, calls)
        connections.append(connection)
        return connection

    namespace = "tenant';--"
    sink = PostgresSharedTraceSink(connect, namespace=namespace, capacity=2, maximum_query=1)
    assert calls == []
    assert not sink.healthcheck()
    sink.migrate()
    assert sink.healthcheck()
    original = event()
    for _ in range(3):
        before = len(calls)
        sink.record(original)
        transaction = calls[before:]
        assert [sql.split()[0] for sql, _ in transaction] == ["BEGIN", "INSERT", "SELECT", "UPDATE", "INSERT", "DELETE"]
        assert "FOR UPDATE" in transaction[2][0]
        assert "market_agent_trace_events" in transaction[4][0]
        assert transaction[5][1][0] == namespace
    page = sink.query(original.trace.trace_id)
    assert [item.sequence for item in page.items] == [2]
    assert page.truncated and page.has_more and page.oldest_available_sequence == 2
    assert PostgresSharedTraceSink(connect, namespace="other").query(original.trace.trace_id).items == ()
    locks = [(sql, params) for sql, params in calls if "FOR UPDATE" in sql]
    assert len(locks) == 3 and all(params == (namespace,) for _, params in locks)
    assert all(namespace not in sql and "BIGSERIAL" not in sql for sql, _ in calls)
    query = next((sql, params) for sql, params in calls if "event_json" in sql and sql.startswith("SELECT"))
    assert "trace_id=%s" in query[0] and query[1][:3] == (namespace, namespace, original.trace.trace_id)
    assert all(c.closed and c.cursor_closed for c in connections)
    failed = PgConnection(path, calls, fail_commit=True)
    broken = PostgresSharedTraceSink(lambda: failed, namespace=namespace)
    with pytest.raises(OSError, match="commit unavailable"):
        broken.record(original)
    assert failed.rolled_back and failed.closed and failed.cursor_closed
    sink.record(original)
    assert sink.query(original.trace.trace_id, after_sequence=3).items[0].sequence == 4


def test_sqlite_requires_file():
    with pytest.raises(ValueError):
        SQLiteSharedTraceSink(":memory:")


def test_api_reads_other_replica_trace_with_existing_auth_contract(tmp_path):
    from fastapi.testclient import TestClient

    from market_agent.backend.api import create_app
    from market_agent.backend.container import BackendContainer
    from market_agent.backend.settings import BackendSettings
    from market_agent.backend.trace_observability import BackendObservability

    path = tmp_path / "traces.db"
    original = event()
    SQLiteSharedTraceSink(path, namespace="tenant").record(original)
    observability = BackendObservability.create()
    observability.sink = SQLiteSharedTraceSink(path, namespace="tenant")
    container = BackendContainer.create(
        BackendSettings(database_path=tmp_path / "jobs.db", environment="test", api_token="test-token"),
        observability=observability,
    )
    try:
        client = TestClient(create_app(container))
        url = f"/v1/traces/{original.trace.trace_id}"
        assert client.get(url).status_code == 401
        response = client.get(url, headers={"Authorization": "Bearer test-token"})
        assert response.status_code == 200
        body = response.json()
        assert body["items"] == [{"sequence": 1, "event": original.model_dump(mode="json")}]
        assert body["next_cursor"] == body["oldest_available_sequence"] == 1
        assert not body["has_more"] and not body["truncated"]
        with sqlite3.connect(path) as connection:
            connection.execute("DROP TABLE market_agent_trace_events")
        assert client.get(url, headers={"Authorization": "Bearer test-token"}).status_code == 503
    finally:
        container.shutdown()
