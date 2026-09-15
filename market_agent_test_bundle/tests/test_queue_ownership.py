from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import pytest

from market_agent.backend.cache import TTLCache
from market_agent.backend.database import JobRepository, PostgresJobRepository
from market_agent.backend.errors import DependencyUnavailableError, RetryableTaskError, TaskQueueFullError
from market_agent.backend.message_bus import InMemoryMessageBus, MessageEnvelope
from market_agent.backend.observability import MetricsRegistry
from market_agent.backend.redis_adapters import RedisMessageBusAdapter, RedisStreamMessageBus, RedisUnavailableError
from market_agent.backend.task_queue import BackgroundTaskQueue


def eventually(predicate, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


def create(repo):
    return repo.create_or_get_job("echo", {"trace_id": "1" * 32}, None, 3, "1" * 32)[0]


def queue(repo, bus=None, **kwargs):
    return BackgroundTaskQueue(repo, TTLCache(32, 60), bus or InMemoryMessageBus(),
                               MetricsRegistry(), 1, kwargs.pop("capacity", 0), 3,
                               kwargs.pop("retry", 0), **kwargs)


def test_two_connections_have_one_claim_winner(tmp_path):
    path = tmp_path / "jobs.db"
    first, second = JobRepository(path), JobRepository(path)
    job = create(first)
    barrier = threading.Barrier(2)
    def claim(repo, token):
        barrier.wait()
        return repo.claim_job(job.job_id, token, 30)
    with ThreadPoolExecutor(2) as executor:
        futures = [executor.submit(claim, repo, token) for repo, token in ((first, "a"), (second, "b"))]
        assert sum(f.result() is not None for f in futures) == 1
    assert first.list_recoverable_jobs("echo") == []
    assert "execution_token" not in first.get_job(job.job_id).as_dict()


@pytest.mark.parametrize("transition,args", [("mark_succeeded", ({"bad": True},)), ("mark_failed", ({"type": "stale"},)), ("mark_retry_scheduled", ({},)), ("mark_recovery_queued", ({},))])
def test_expired_owner_cannot_write_transition(tmp_path, monkeypatch, transition, args):
    from market_agent.backend import database
    repo = JobRepository(tmp_path / "jobs.db")
    job = create(repo)
    monkeypatch.setattr(database, "_utc_now", lambda: "2026-09-08T00:00:00+00:00")
    repo.claim_job(job.job_id, "old", 1)
    repo.mark_running(job.job_id, 1, execution_token="old")
    monkeypatch.setattr(database, "_utc_now", lambda: "2026-09-08T00:00:02+00:00")
    before = repo.list_events(job.job_id)
    with pytest.raises(RuntimeError, match="lease"):
        getattr(repo, transition)(job.job_id, *args, execution_token="old")
    assert repo.get_job(job.job_id).status == "running"
    assert repo.list_events(job.job_id) == before
    assert not repo.renew_job_lease(job.job_id, "old", 10)
    assert repo.claim_job(job.job_id, "new", 10, recovery=True)
    with pytest.raises(RuntimeError, match="lease"):
        repo.mark_failed(job.job_id, {"type": "stale"}, execution_token="old")
    with pytest.raises(RuntimeError, match="lease"):
        repo.mark_failed(job.job_id, {"type": "legacy"})
    repo.mark_failed(job.job_id, {"type": "valid"}, execution_token="new")
    assert repo.get_job(job.job_id).execution_token is None


class RemoteBus:
    durable_dispatch = True
    def __init__(self):
        self.messages = []
    def subscribe(self, topic, handler):
        return lambda: None
    def publish(self, message):
        self.messages.append(message)


def test_durable_publisher_reuses_capacity_after_remote_completion(tmp_path):
    repo = JobRepository(tmp_path / "jobs.db")
    bus = RemoteBus()
    publisher = queue(repo, bus)
    consumer = queue(JobRepository(tmp_path / "jobs.db"), RemoteBus())
    # Isolate delivery transport from recovery polling for this capacity test.
    publisher._start_recovery = consumer._start_recovery = lambda *args: None
    publisher.register("echo", lambda p: p)
    consumer.register("echo", lambda p: "remote")
    try:
        first = publisher.submit("echo", {"trace_id": "1" * 32})
        consumer._handle_dispatch_message(bus.messages[-1])
        eventually(lambda: repo.get_job(first.job.job_id).status == "succeeded")
        second = publisher.submit("echo", {"trace_id": "1" * 32})
        assert second.job.job_id != first.job.job_id
    finally:
        publisher.shutdown()
        consumer.shutdown()


def test_recovery_continues_after_empty_page(tmp_path):
    repo = JobRepository(tmp_path / "jobs.db")
    worker = queue(repo)
    worker.register("echo", lambda p: "recovered")
    try:
        time.sleep(0.1)
        job = create(JobRepository(tmp_path / "jobs.db"))
        eventually(lambda: repo.get_job(job.job_id).status == "succeeded")
    finally:
        worker.shutdown()


def test_healthy_owner_is_not_recovered(tmp_path):
    repo = JobRepository(tmp_path / "jobs.db")
    job = create(repo)
    assert repo.claim_job(job.job_id, "healthy-owner", 10) is not None
    worker = queue(JobRepository(tmp_path / "jobs.db"), recovery_poll_seconds=0.03)
    calls = []
    worker.register("echo", lambda payload: calls.append(payload) or payload)
    try:
        time.sleep(0.15)
        assert calls == []
        assert repo.get_job(job.job_id).status == "accepted"
        assert repo.release_job_lease(job.job_id, "healthy-owner")
        eventually(lambda: repo.get_job(job.job_id).status == "succeeded")
        assert len(calls) == 1
    finally:
        worker.shutdown()


def test_queued_and_retry_wait_leases_are_renewed(tmp_path):
    repo = JobRepository(tmp_path / "jobs.db")
    worker = queue(repo, capacity=1, retry=0.6, lease_seconds=0.15, recovery_poll_seconds=0.03)
    started, unblock = threading.Event(), threading.Event()
    calls = []
    def handler(payload):
        calls.append(payload["n"])
        if payload["n"] == 1 and calls.count(1) == 1:
            started.set()
            unblock.wait(2)
            raise RetryableTaskError("temporary")
        return payload["n"]
    worker.register("echo", handler)
    try:
        first = worker.submit("echo", {"n": 1}).job
        assert started.wait(1)
        second = worker.submit("echo", {"n": 2}).job
        time.sleep(0.35)
        other = JobRepository(tmp_path / "jobs.db")
        assert other.claim_job(second.job_id, "steal-queued", 1, recovery=True) is None
        assert other.claim_job(first.job_id, "steal-running", 1, recovery=True) is None
        unblock.set()
        eventually(lambda: repo.get_job(first.job_id).status == "accepted")
        time.sleep(0.3)
        assert other.claim_job(first.job_id, "steal-retry", 1, recovery=True) is None
        eventually(lambda: repo.get_job(second.job_id).status == "succeeded")
        assert calls == [1, 1, 2]
    finally:
        unblock.set()
        worker.shutdown()


def test_executor_submit_failure_releases_claim(tmp_path, monkeypatch):
    repo = JobRepository(tmp_path / "jobs.db")
    worker = queue(repo, RemoteBus())
    worker._start_recovery = lambda *args: None
    worker.register("echo", lambda p: p)
    job = create(repo)
    def broken(*args):
        raise RuntimeError("executor unavailable")
    monkeypatch.setattr(worker._executor, "submit", broken)
    try:
        with pytest.raises(DependencyUnavailableError):
            worker._handle_dispatch_message(MessageEnvelope(topic="task.dispatch", payload={"task_name": "echo", "job_id": job.job_id}))
        assert repo.get_job(job.job_id).execution_token is None
        assert repo.claim_job(job.job_id, "replacement", 1)
    finally:
        worker.shutdown()


@pytest.mark.parametrize("old_fails", [False, True])
def test_old_worker_cannot_overwrite_replacement_result(tmp_path, old_fails):
    repo = JobRepository(tmp_path / "jobs.db")
    worker = queue(repo, RemoteBus())
    worker._start_recovery = lambda *args: None
    started, unblock = threading.Event(), threading.Event()

    def handler(payload):
        started.set()
        assert unblock.wait(3)
        if old_fails:
            raise RuntimeError("stale worker failure")
        return "stale result"

    worker.register("echo", handler)
    job = create(repo)
    try:
        worker._handle_dispatch_message(MessageEnvelope(topic="task.dispatch", payload={"task_name": "echo", "job_id": job.job_id}))
        assert started.wait(2)
        old_token = repo.get_job(job.job_id).execution_token
        assert repo.release_job_lease(job.job_id, old_token)
        replacement = JobRepository(tmp_path / "jobs.db")
        assert replacement.claim_job(job.job_id, "replacement", 30, recovery=True)
        replacement.mark_running(job.job_id, 2, execution_token="replacement")
        replacement.mark_succeeded(job.job_id, "replacement result", execution_token="replacement")
        before = replacement.list_events(job.job_id)
        unblock.set()
        worker.shutdown()
        assert replacement.get_job(job.job_id).result == "replacement result"
        assert replacement.list_events(job.job_id) == before
    finally:
        unblock.set()
        worker.shutdown()


class PendingRedis:
    def __init__(self):
        self.fields = None
        self.delivered = False
        self.acks = []
        self.dead = []
    def xgroup_create(self, *args, **kwargs):
        pass
    def xadd(self, stream, fields):
        if stream.endswith(":dead"):
            self.dead.append(fields)
        else:
            self.fields = fields
        return "1-0"
    def xreadgroup(self, group, consumer, streams, **kwargs):
        if self.delivered or self.fields is None:
            time.sleep(0.005)
            return []
        self.delivered = True
        return [(next(iter(streams)), [("1-0", self.fields)])]
    def xautoclaim(self, stream, *args, **kwargs):
        return ("0-0", [("1-0", self.fields)] if self.delivered and not self.acks else [])
    def xack(self, stream, group, identifier):
        self.acks.append(identifier)


@pytest.mark.parametrize("error", [TaskQueueFullError, DependencyUnavailableError, RedisUnavailableError, RetryableTaskError])
def test_transient_delivery_remains_pending_then_succeeds(error):
    redis = PendingRedis()
    bus = RedisStreamMessageBus(redis, tenant_id="test")
    bus.publish(MessageEnvelope(topic="task.dispatch", payload={"trace_id": "1" * 32}, request_id="1" * 32))
    adapter = RedisMessageBusAdapter(bus)
    adapter._RETRY_BASE_SECONDS = 0.01
    attempts = []
    def handle(message):
        attempts.append(message)
        if len(attempts) == 1:
            raise error("transient")
    adapter.subscribe("task.dispatch", handle)
    try:
        eventually(lambda: len(attempts) >= 2 or bool(redis.dead))
        assert redis.dead == []
        eventually(lambda: redis.acks == ["1-0"])
        assert len(attempts) == 2
    finally:
        adapter.close()


def test_postgres_ownership_sql_is_migrated_and_claimed_atomically():
    statements = []

    class Cursor:
        rowcount = 0

        def execute(self, statement, parameters=None):
            statements.append((" ".join(statement.split()), parameters))

        def fetchone(self):
            return None

        def close(self):
            pass

    class Connection:
        def cursor(self):
            return Cursor()

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    repo = PostgresJobRepository(Connection)
    repo.migrate()
    assert repo.claim_job("job-1", "owner-1", 30) is None

    sql = "\n".join(statement for statement, _ in statements)
    assert "ADD COLUMN IF NOT EXISTS execution_token" in sql
    assert "ADD COLUMN IF NOT EXISTS lease_expires_at" in sql
    claim_sql, parameters = next((statement, parameters) for statement, parameters in statements if "execution_token = %s" in statement)
    assert "status IN ('accepted')" in claim_sql
    assert "execution_token IS NULL OR lease_expires_at IS NULL OR lease_expires_at <= clock_timestamp()" in claim_sql
    assert "clock_timestamp() + (%s * INTERVAL '1 second')" in claim_sql
    assert parameters == ("owner-1", 30.0, "job-1")
    assert "RETURNING" in claim_sql
