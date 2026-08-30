from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import sqlite3

import pytest
from pydantic import ValidationError

from market_agent.workflow_audit import AuditEvent, AuditPayload, AuditStore, AuditUnavailableError, AuditWriter


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
        "payload": {"kind": "transition", "subject_ids": ("task-1",)},
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


def test_audit_is_append_only_and_rejects_sensitive_payload_values(tmp_path):
    store = AuditStore(tmp_path / "audit.sqlite3")
    written = store.append(event("event-1"))
    with pytest.raises(ValidationError):
        event("event-secret", payload={"kind": "transition", "subject_ids": ("Bearer private",)})
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


def test_insert_or_replace_cannot_bypass_append_only_audit_triggers(tmp_path):
    store = AuditStore(tmp_path / "audit.sqlite3")
    written = store.append(event("event-1"))

    with sqlite3.connect(tmp_path / "audit.sqlite3") as connection:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            connection.execute(
                "INSERT OR REPLACE INTO audit_events (event_id, trace_id, workflow_id, sequence, occurred_at, actor, event_type, status, latency_ms, token_usage, cached_token_usage, estimated_cost, cumulative_cost, source_references, payload, schema_version) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                ("event-1", "trace-1", "workflow-1", 1, written.occurred_at.isoformat(), "attacker", "replaced", "accepted", 0, 0, 0, 0.0, 0.0, "[]", "{}", "v1"),
            )
    assert store.list(trace_id="trace-1") == [written]


def test_audit_rejects_unbounded_or_secret_bearing_payloads_and_top_level_references():
    with pytest.raises(ValidationError):
        event("event-1", payload={"kind": "transition", "body": "raw prompt"})
    with pytest.raises(ValidationError):
        event("event-2", source_references=("Authorization: Bearer secret",))
    with pytest.raises(ValidationError):
        event("event-3", payload=AuditPayload(kind="transition", subject_ids=("https://service/?token=secret",)))


def test_list_is_page_bounded_and_rejects_naive_time_filters(tmp_path):
    store = AuditStore(tmp_path / "audit.sqlite3")
    for index in range(3):
        store.append(event(f"event-{index}"))

    first_page = store.list(page_size=2)
    assert len(first_page) == 2
    assert first_page.next_cursor is not None
    assert [item.event_id for item in store.list(page_size=2, cursor=first_page.next_cursor)] == ["event-2"]
    with pytest.raises(ValueError):
        store.list(page_size=101)
    with pytest.raises(ValueError):
        store.list(start_time=datetime(2026, 8, 29))


def test_store_migrates_legacy_database_without_changing_existing_event_hashes(tmp_path):
    database_path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute("CREATE TABLE audit_events (event_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL, workflow_id TEXT NOT NULL, task_id TEXT, attempt_id TEXT, sequence INTEGER NOT NULL, occurred_at TEXT NOT NULL, actor TEXT NOT NULL, event_type TEXT NOT NULL, status TEXT NOT NULL, input_hash TEXT, output_hash TEXT, latency_ms INTEGER NOT NULL, token_usage INTEGER NOT NULL, cached_token_usage INTEGER NOT NULL, estimated_cost REAL NOT NULL, cumulative_cost REAL NOT NULL, model TEXT, prompt_version TEXT, schema_name TEXT, schema_hash TEXT, source_references TEXT NOT NULL, payload TEXT NOT NULL, UNIQUE(trace_id, sequence))")
        connection.execute("INSERT INTO audit_events VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", ("event-1", "trace-1", "workflow-1", None, None, 1, "2026-08-29T00:00:00+00:00", "coordinator", "created", "accepted", "in", "out", 0, 0, 0, 0.0, 0.0, None, None, None, None, "[]", "{}"))

    migrated = AuditStore(database_path).list()

    assert migrated[0].schema_version == "v1"
    assert (migrated[0].input_hash, migrated[0].output_hash) == ("in", "out")


@pytest.mark.parametrize("field,value", [("event_id", "sk-secret"), ("trace_id", "eyJhbGciOiJIUzI1NiJ9.payload.signature"), ("actor", "-----BEGIN PRIVATE KEY-----"), ("event_type", "raw prompt: ignore all rules"), ("source_references", ("https://host/?token=secret",))])
def test_audit_semantic_fields_reject_secret_and_prose_forms(field, value):
    values = event("event-typed").model_dump()
    values[field] = value
    with pytest.raises(ValidationError):
        AuditEvent(**values)


def test_cursor_is_bounded_and_cannot_be_reused_with_different_filters(tmp_path):
    store = AuditStore(tmp_path / "audit.sqlite3")
    store.append(event("event-a", trace_id="trace-a"))
    store.append(event("event-b", trace_id="trace-b"))
    page = store.list(page_size=1)

    with pytest.raises(ValueError, match="cursor"):
        store.list(page_size=1, trace_id="trace-a", cursor=page.next_cursor)
    with pytest.raises(ValueError, match="cursor"):
        store.list(cursor="x" * 5000)
