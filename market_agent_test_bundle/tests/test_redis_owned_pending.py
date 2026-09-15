from __future__ import annotations

import threading
import time

import pytest

from market_agent.backend.errors import (
    DependencyUnavailableError,
    RetryableTaskError,
    TaskQueueFullError,
)
from market_agent.backend.message_bus import MessageEnvelope
from market_agent.backend.redis_adapters import (
    RedisMessageBusAdapter,
    RedisStreamMessageBus,
    RedisUnavailableError,
)


class IdleAwareRedis:
    """Frozen Redis time: only own PEL reads can retry a fresh delivery."""

    def __init__(self):
        self.now_ms = 0
        self.rows = {}
        self.pending = {}
        self.delivered = set()
        self.acks = []
        self.dead = []
        self.reads = []
        self.claims = []
        self.ack_failures = 0
        self.hidden = set()

    def ping(self):
        return True

    def xgroup_create(self, *args, **kwargs):
        pass

    def xadd(self, stream, fields, id="*"):
        if stream.endswith(":dead"):
            self.dead.append(fields)
            return "1-0"
        identifier = f"{len(self.rows) + 1}-0"
        self.rows[identifier] = (stream, fields)
        return identifier

    def xreadgroup(self, group, consumer, streams, **kwargs):
        stream, cursor = next(iter(streams.items()))
        self.reads.append((consumer, cursor, kwargs))
        if cursor == "0":
            assert "block" not in kwargs
            ids = [identifier for identifier, (owner, _) in self.pending.items()
                   if owner == consumer and identifier not in self.hidden
                   and self.rows[identifier][0] == stream]
        else:
            assert cursor == ">"
            ids = [identifier for identifier, (name, _) in self.rows.items()
                   if identifier not in self.delivered and name == stream]
        ids = ids[:kwargs.get("count", 1)]
        for identifier in ids:
            self.pending[identifier] = (consumer, self.now_ms)
            self.delivered.add(identifier)
        if not ids:
            time.sleep(0.001)
        return [(stream, [(identifier, self.rows[identifier][1]) for identifier in ids])]

    def xautoclaim(self, stream, group, consumer, min_idle_time, **kwargs):
        self.claims.append(min_idle_time)
        ids = [identifier for identifier, (_, since) in self.pending.items()
               if self.now_ms - since >= min_idle_time and self.rows[identifier][0] == stream]
        ids = ids[:kwargs.get("count", 1)]
        for identifier in ids:
            self.pending[identifier] = (consumer, self.now_ms)
        return "0-0", [(identifier, self.rows[identifier][1]) for identifier in ids]

    def xack(self, stream, group, identifier):
        if self.ack_failures:
            self.ack_failures -= 1
            raise ConnectionError("ACK temporarily unavailable")
        self.acks.append(identifier)
        self.pending.pop(identifier, None)


def eventually(predicate, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


def publish(bus, number):
    return bus.publish(MessageEnvelope(
        topic="task.dispatch", payload={"trace_id": "1" * 32, "number": number},
        request_id=f"request-{number}",
    ))


def setup_bus(count=1):
    client = IdleAwareRedis()
    bus = RedisStreamMessageBus(client, tenant_id="test")
    for number in range(1, count + 1):
        publish(bus, number)
    adapter = RedisMessageBusAdapter(bus)
    adapter._RETRY_BASE_SECONDS = 0.01
    return client, bus, adapter


def test_owned_pending_reads_only_current_consumer_without_blocking():
    client, bus, _ = setup_bus(3)
    first = bus.consume(topic="task.dispatch", group="workers", consumer="owner", count=1)
    bus.consume(topic="task.dispatch", group="workers", consumer="other", count=1)
    owned = bus.recover_owned_pending(topic="task.dispatch", group="workers", consumer="owner", count=10)
    assert owned == first
    assert client.reads[-1] == ("owner", "0", {"count": 10})
    assert client.pending["2-0"][0] == "other"
    assert "3-0" not in client.pending


@pytest.mark.parametrize("overrides", [
    {"topic": ""}, {"topic": "bad topic"}, {"group": " "},
    {"consumer": None}, {"count": 0}, {"count": -1},
])
def test_owned_pending_validates_consume_arguments(overrides):
    _, bus, _ = setup_bus()
    arguments = dict(topic="task.dispatch", group="workers", consumer="owner", count=1)
    arguments.update(overrides)
    with pytest.raises(ValueError):
        bus.recover_owned_pending(**arguments)


@pytest.mark.parametrize("transport_failure", [False, True])
def test_owned_pending_wraps_transport_and_decode_errors(monkeypatch, transport_failure):
    client, bus, _ = setup_bus()

    def broken(group, consumer, streams, **kwargs):
        if transport_failure:
            raise ConnectionError("offline")
        return [(next(iter(streams)), [("1-0", {"envelope": "not-json"})])]

    monkeypatch.setattr(client, "xreadgroup", broken)
    with pytest.raises(RedisUnavailableError):
        bus.recover_owned_pending(topic="task.dispatch", group="workers", consumer="owner")


@pytest.mark.parametrize("error", [
    TaskQueueFullError, DependencyUnavailableError, RedisUnavailableError, RetryableTaskError,
])
def test_fresh_transient_delivery_retries_before_cross_owner_idle(error):
    client, _, adapter = setup_bus()
    attempts = []
    retry_entered, release = threading.Event(), threading.Event()

    def handle(message):
        attempts.append(message.payload["number"])
        if len(attempts) == 1:
            raise error("temporary")
        retry_entered.set()
        assert release.wait(2)

    adapter.subscribe("task.dispatch", handle)
    try:
        assert retry_entered.wait(1)
        assert adapter.health().status == "degraded"
        assert client.acks == []
        release.set()
        eventually(lambda: client.acks == ["1-0"] and adapter.health().status == "ok")
        assert attempts == [1, 1]
        assert client.now_ms == 0
        assert set(client.claims) == {60_000}
        assert client.dead == []
        assert client.reads[0][1] == "0"
    finally:
        release.set()
        adapter.close()


@pytest.mark.parametrize("permanent", [False, True])
def test_ack_transport_failure_redelivers_owned_entry(permanent):
    client, _, adapter = setup_bus()
    client.ack_failures = 1
    attempts = []

    def handle(message):
        attempts.append(message)
        if permanent:
            raise ValueError("invalid job")

    adapter.subscribe("task.dispatch", handle)
    try:
        eventually(lambda: client.acks == ["1-0"] and adapter.health().status == "ok")
        assert len(attempts) == 2  # Execution and DLQ effects are at-least-once.
        assert len(client.dead) == (2 if permanent else 0)
        assert client.now_ms == 0
    finally:
        adapter.close()


@pytest.mark.parametrize("reclaimed", [False, True])
def test_partial_batch_tail_recovers_without_waiting_sixty_seconds(reclaimed):
    client, bus, adapter = setup_bus(13)
    if reclaimed:
        bus.consume(topic="task.dispatch", group=adapter._group, consumer="old", count=13)
        client.now_ms = 60_000
    attempts = []

    def handle(message):
        number = message.payload["number"]
        attempts.append(number)
        if number == 2 and attempts.count(2) == 1:
            raise TaskQueueFullError("temporary")

    adapter.subscribe("task.dispatch", handle)
    try:
        eventually(lambda: len(client.acks) == 13)
        assert attempts == [1, 2, *range(2, 14)]
        assert len(set(client.acks)) == 13
        assert client.pending == {}
        assert client.now_ms == (60_000 if reclaimed else 0)
        assert set(client.claims) == {60_000}
    finally:
        adapter.close()


@pytest.mark.parametrize("resolve_to_dlq", [False, True])
def test_empty_poll_and_unrelated_ack_do_not_clear_unresolved_failure(resolve_to_dlq):
    client, bus, adapter = setup_bus()
    failed, unrelated = threading.Event(), threading.Event()
    attempts = []

    def handle(message):
        number = message.payload["number"]
        attempts.append(number)
        if number == 1 and attempts.count(1) == 1:
            client.hidden.add("1-0")  # Simulate an owned read temporarily missing this ID.
            failed.set()
            raise TaskQueueFullError("temporary")
        if number == 2:
            unrelated.set()
        elif resolve_to_dlq:
            raise ValueError("invalid job")

    adapter.subscribe("task.dispatch", handle)
    try:
        assert failed.wait(1)
        eventually(lambda: adapter.health().status == "degraded")
        polls = len(client.reads)
        eventually(lambda: len(client.reads) >= polls + 6)
        assert adapter.health().status == "degraded"
        publish(bus, 2)
        assert unrelated.wait(1)
        eventually(lambda: "2-0" in client.acks)
        polls = len(client.reads)
        eventually(lambda: len(client.reads) >= polls + 6)
        assert adapter.health().status == "degraded"
        client.hidden.clear()
        eventually(lambda: "1-0" in client.acks and adapter.health().status == "ok")
        assert len(client.dead) == int(resolve_to_dlq)
        assert attempts == [1, 2, 1]
    finally:
        adapter.close()
