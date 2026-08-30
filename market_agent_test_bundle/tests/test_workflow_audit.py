from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import sqlite3

import pytest
from pydantic import ValidationError

from market_agent.workflow_audit import AuditEvent, AuditStore, AuditUnavailableError, AuditWriter


def event(event_id: str, trace_id: str = "trace-1", **overrides: object) -> AuditEvent:
    values: dict[str, object] = {
        "event_id": event_id,
        "trace_id": trace_id,
        "workflow_id": "workflow-1",
        "occurred_at": datetime(2026, 8, 29, tzinfo=timezone.utc),
        "actor": "coordinator",
        "event_type": "task_dispatched",
        "status": "accepted",
        "input_hash": "input-hash",
        "output_hash": "output-hash",
        "latency_ms": 12,
        "token_usage": 4,
        "cached_token_usage": 0,
        "estimated_cost": 0.01,
        "cumulative_cost": 0.01,
        "model": "gpt-5.6-terra",
        "prompt_version": "prompt-v1",
        "schema_name": "AgentTask",
        "schema_hash": "schema-hash",
        "source_references": ("source-1",),
        "payload": {"safe": "value"},
    }
    values.update(overrides)
    return AuditEvent(**values)


def test_append_assigns_monotonic_per_trace_sequences_and_lists_deterministically(tmp_path):
    store = AuditStore(tmp_path / "audit.sqlite3")

    first = store.append(event("event-1"))
    other_trace = store.append(event("event-2", trace_id="trace-2"))
    second = store.append(event("event-3"))

    assert (first.sequence, second.sequence, other_trace.sequence) == (1, 2, 1)
    assert [item.event_id for item in store.list()] == ["event-1", "event-3", "event-2"]
    assert [item.sequence for item in store.list(trace_id="trace-1")] == [1, 2]


def test_append_is_safe_under_concurrent_writers(tmp_path):
    store = AuditStore(tmp_path / "audit.sqlite3")

    with ThreadPoolExecutor(max_workers=8) as executor:
        stored = list(executor.map(lambda index: store.append(event(f"event-{index}")), range(24)))

    assert sorted(item.sequence for item in stored) == list(range(1, 25))
    assert [item.sequence for item in store.list(trace_id="trace-1")] == list(range(1, 25))


def test_audit_is_append_only_and_recursively_redacts_sensitive_payload_values(tmp_path):
    store = AuditStore(tmp_path / "audit.sqlite3")
    written = store.append(
        event(
            "event-1",
            payload={
                "token": "secret-token",
                "nested": {"authorization": "Bearer private", "analysis": "private chain"},
                "items": [{"api_key": "secret-key"}],
            },
        )
    )

    assert written.payload == {
        "token": "[REDACTED]",
        "nested": {"authorization": "[REDACTED]", "analysis": "[REDACTED]"},
        "items": [{"api_key": "[REDACTED]"}],
    }
    with sqlite3.connect(tmp_path / "audit.sqlite3") as connection:
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute("UPDATE audit_events SET status = 'changed'")
        with pytest.raises(sqlite3.DatabaseError):
            connection.execute("DELETE FROM audit_events")
    assert store.list(trace_id="trace-1") == [written]


def test_audit_event_is_a_strict_versioned_contract():
    with pytest.raises(ValidationError):
        event("event-1", unexpected="value")
    with pytest.raises(ValidationError):
        event("event-2", occurred_at=datetime(2026, 8, 29))
    with pytest.raises(ValidationError):
        event("event-3", payload={"not_json": object()})


def test_failed_required_audit_write_marks_writer_unhealthy_and_blocks_dispatch():
    class FailingStore:
        def append(self, _: AuditEvent) -> AuditEvent:
            raise OSError("database unavailable")

    writer = AuditWriter(FailingStore())

    with pytest.raises(AuditUnavailableError):
        writer.record(event("event-1"))
    assert writer.healthy is False
    with pytest.raises(AuditUnavailableError):
        writer.record(event("event-2"))
