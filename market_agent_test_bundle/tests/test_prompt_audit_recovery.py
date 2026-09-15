from __future__ import annotations

import threading
import time
import json
from hashlib import sha256
from types import SimpleNamespace

import pytest

from market_agent.backend.prompt_audit_recovery import PromptAuditRecoveryScheduler
from market_agent.backend.redis_adapters import RedisPromptActivationMirror, RedisStreamMessageBus
from market_agent.workflow_prompt_config import PromptActivation


class _ReplayManager:
    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = iter(outcomes)
        self.calls: list[int] = []
        self.called = threading.Event()

    def replay_pending_audit(self, *, limit: int) -> int:
        self.calls.append(limit)
        self.called.set()
        outcome = next(self._outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return int(outcome)


def test_run_once_replays_the_bounded_shared_outbox_batch():
    """Changing the recovery batch size or skipping the manager replay would fail this test."""
    manager = _ReplayManager([3])

    assert PromptAuditRecoveryScheduler(manager).run_once() == 3
    assert manager.calls == [100]


@pytest.mark.parametrize("interval", [0, -1, float("inf"), float("nan"), True])
def test_recovery_scheduler_rejects_non_positive_or_non_finite_intervals(interval):
    """Accepting an unbounded wait duration would make a stopped recovery worker unreliable."""
    with pytest.raises(ValueError, match="interval"):
        PromptAuditRecoveryScheduler(_ReplayManager([0]), interval_seconds=interval)


def test_background_recovery_reports_a_failure_then_keeps_replaying():
    """Letting one replay outage kill the daemon would strand later shared outbox records."""
    manager = _ReplayManager([OSError("temporary outbox outage"), 2, 2, 2])
    observed: list[tuple[str, Exception]] = []
    scheduler = PromptAuditRecoveryScheduler(
        manager,
        interval_seconds=0.01,
        error_observer=lambda event, error: observed.append((event, error)),
    )
    scheduler.start()
    try:
        deadline = time.monotonic() + 1.0
        while len(manager.calls) < 2 and time.monotonic() < deadline:
            time.sleep(0.005)
    finally:
        scheduler.close()

    assert manager.calls[:2] == [100, 100]
    assert [event for event, _ in observed] == ["prompt_audit_recovery_failed"]
    assert isinstance(observed[0][1], OSError)


class _CaptureStreamClient:
    def __init__(self) -> None:
        self.fields: dict[str, str] | None = None

    def xadd(self, _stream: str, fields: dict[str, str], id: str = "*") -> str:
        self.fields = fields
        return "1-0"


@pytest.mark.parametrize("revision", [None, 7])
def test_redis_prompt_activation_mirror_emits_revision_in_payload_and_trace_identity(revision):
    """Dropping a shared revision would collapse distinct activation audit deliveries."""
    client = _CaptureStreamClient()
    mirror = RedisPromptActivationMirror(RedisStreamMessageBus(client, tenant_id="test"))
    activation = PromptActivation(
        active_release_id="release-b",
        previous_release_id="release-a",
        action="activate",
        revision=revision,
    )

    mirror(activation, SimpleNamespace(release_digest="d" * 64, manifest_hash="m" * 64))

    assert client.fields is not None
    envelope = json.loads(client.fields["envelope"])
    expected_trace = sha256(f"prompt:release-b:activate:{revision}".encode()).hexdigest()[:32]
    assert envelope["payload"]["revision"] == revision
    assert envelope["payload"]["trace_id"] == expected_trace
    assert envelope["request_id"] == expected_trace
