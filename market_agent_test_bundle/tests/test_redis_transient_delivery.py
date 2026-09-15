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


class PendingStreamClient:
    """Minimal Redis Streams fake preserving one unacknowledged entry."""

    def __init__(self) -> None:
        self.fields = None
        self.delivered = False
        self.acks: list[str] = []
        self.dead: list[dict[str, str]] = []

    def xgroup_create(self, *args, **kwargs) -> None:
        return None

    def xadd(self, stream, fields, id="*") -> str:
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
        if self.delivered and not self.acks:
            return ("0-0", [("1-0", self.fields)])
        return ("0-0", [])

    def xack(self, stream, group, identifier) -> None:
        self.acks.append(identifier)


def eventually(predicate, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    assert predicate()


@pytest.mark.parametrize(
    "error",
    [TaskQueueFullError, DependencyUnavailableError, RedisUnavailableError, RetryableTaskError],
)
def test_transient_handler_failure_stays_pending_without_ack_or_dead_letter(error) -> None:
    client = PendingStreamClient()
    bus = RedisStreamMessageBus(client, tenant_id="test")
    bus.publish(MessageEnvelope(
        topic="task.dispatch",
        payload={"trace_id": "1" * 32},
        request_id="1" * 32,
    ))
    adapter = RedisMessageBusAdapter(bus)
    adapter._RETRY_BASE_SECONDS = 0.1
    first_attempt = threading.Event()
    attempts = []

    def handler(message) -> None:
        attempts.append(message)
        if len(attempts) == 1:
            first_attempt.set()
            raise error("temporary")

    adapter.subscribe("task.dispatch", handler)
    try:
        assert first_attempt.wait(1)
        assert client.acks == []
        assert client.dead == []
        eventually(lambda: len(attempts) == 2)
        eventually(lambda: client.acks == ["1-0"])
        assert client.dead == []
    finally:
        adapter.close()
