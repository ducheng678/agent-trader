from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import sqlite3

import pytest
from pydantic import TypeAdapter, ValidationError

from market_agent import workflow_audit
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
        "input_hash": "a" * 64,
        "output_hash": "b" * 64,
        "latency_ms": 12,
        "token_usage": 4,
        "cached_token_usage": 0,
        "estimated_cost": 0.01,
        "cumulative_cost": 0.01,
        "model": "gpt-5.6-terra",
        "prompt_version": "prompt-v1",
        "schema_name": "AgentTask",
        "schema_hash": "c" * 64,
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
    assert (migrated[0].input_hash, migrated[0].output_hash) == (None, None)
    assert migrated[0].payload.kind == "legacy_migration"
    assert "safe" not in migrated[0].payload.model_dump_json()


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


@pytest.mark.parametrize("field,value", [("input_hash", "a" * 63), ("output_hash", "A" * 64), ("schema_hash", "g" * 64), ("legacy_payload_digest", "short")])
def test_audit_digest_fields_require_canonical_lowercase_sha256(field, value):
    values = event("event-digest").model_dump()
    if field == "legacy_payload_digest":
        values["payload"] = {"kind": "legacy_migration", "legacy_payload_digest": value}
    else:
        values[field] = value
    with pytest.raises(ValidationError):
        AuditEvent(**values)


def _create_legacy_database(database_path, rows, *, schema_version=False, triggers=False):
    schema_column = ", schema_version TEXT NOT NULL DEFAULT 'v1'" if schema_version else ""
    with sqlite3.connect(database_path) as connection:
        connection.execute(f"CREATE TABLE audit_events (event_id TEXT PRIMARY KEY, trace_id TEXT NOT NULL, workflow_id TEXT NOT NULL, task_id TEXT, attempt_id TEXT, sequence INTEGER NOT NULL, occurred_at TEXT NOT NULL, actor TEXT NOT NULL, event_type TEXT NOT NULL, status TEXT NOT NULL, input_hash TEXT, output_hash TEXT, latency_ms INTEGER NOT NULL, token_usage INTEGER NOT NULL, cached_token_usage INTEGER NOT NULL, estimated_cost REAL NOT NULL, cumulative_cost REAL NOT NULL, model TEXT, prompt_version TEXT, schema_name TEXT, schema_hash TEXT, source_references TEXT NOT NULL, payload TEXT NOT NULL{schema_column}, UNIQUE(trace_id, sequence))")
        columns = 24 if schema_version else 23
        placeholders = ",".join("?" for _ in range(columns))
        for row in rows:
            connection.execute(f"INSERT INTO audit_events VALUES ({placeholders})", row)
        if triggers:
            connection.execute("CREATE TRIGGER audit_events_no_update BEFORE UPDATE ON audit_events BEGIN SELECT RAISE(ABORT, 'legacy append-only'); END")
            connection.execute("CREATE TRIGGER audit_events_no_delete BEFORE DELETE ON audit_events BEGIN SELECT RAISE(ABORT, 'legacy append-only'); END")
            connection.execute("CREATE TRIGGER audit_events_no_replace BEFORE INSERT ON audit_events WHEN EXISTS (SELECT 1 FROM audit_events WHERE event_id = NEW.event_id) BEGIN SELECT RAISE(ABORT, 'legacy append-only'); END")


def _legacy_row(event_id, sequence, payload, *, input_hash="invalid", output_hash="b" * 64, schema_version=None):
    values = (event_id, "trace-1", "workflow-1", "task-1", "attempt-1", sequence, "2026-08-29T00:00:00+00:00", "coordinator", "task_dispatched", "accepted", input_hash, output_hash, 0, 0, 0, 0.0, 0.0, None, None, None, None, "[]", payload)
    return values + ((schema_version,) if schema_version is not None else ())


def test_migration_rollback_restores_old_triggers_and_schema(monkeypatch, tmp_path):
    database_path = tmp_path / "legacy-rollback.sqlite3"
    _create_legacy_database(database_path, [_legacy_row("event-1", 1, "not-json")], triggers=True)
    original_connect = AuditStore._connect

    def deny_trigger_creation(store):
        connection = original_connect(store)
        connection.set_authorizer(lambda action, *_: sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_CREATE_TRIGGER else sqlite3.SQLITE_OK)
        return connection

    monkeypatch.setattr(AuditStore, "_connect", deny_trigger_creation)
    with pytest.raises(sqlite3.DatabaseError):
        AuditStore(database_path)

    with sqlite3.connect(database_path) as connection:
        trigger_names = {row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'trigger'")}
        columns = {row[1] for row in connection.execute("PRAGMA table_info(audit_events)")}
        payload = connection.execute("SELECT payload FROM audit_events").fetchone()[0]
    assert trigger_names == {"audit_events_no_update", "audit_events_no_delete", "audit_events_no_replace"}
    assert "schema_version" not in columns
    assert payload == "not-json"


def test_reopening_canonical_database_does_not_drop_protection(monkeypatch, tmp_path):
    database_path = tmp_path / "canonical.sqlite3"
    AuditStore(database_path)
    original_connect = AuditStore._connect

    def deny_trigger_removal(store):
        connection = original_connect(store)
        connection.set_authorizer(lambda action, *_: sqlite3.SQLITE_DENY if action == sqlite3.SQLITE_DROP_TRIGGER else sqlite3.SQLITE_OK)
        return connection

    monkeypatch.setattr(AuditStore, "_connect", deny_trigger_removal)
    AuditStore(database_path)


def test_legacy_conversion_is_deterministic_for_invalid_valid_scalar_list_and_dict_payloads(tmp_path):
    database_path = tmp_path / "legacy-matrix.sqlite3"
    valid_payload = json.dumps({"kind": "transition", "subject_ids": ["task-1"]})
    payloads = ("not-json", valid_payload, json.dumps("scalar"), json.dumps(["one", "two"]), json.dumps({"one": 1, "two": 2}))
    rows = [_legacy_row(f"event-{index}", index, payload) for index, payload in enumerate(payloads, 1)]
    _create_legacy_database(database_path, rows, triggers=True)

    first = AuditStore(database_path).list()
    first_payloads = tuple(item.payload.model_dump(mode="json") for item in first)
    second_payloads = tuple(item.payload.model_dump(mode="json") for item in AuditStore(database_path).list())

    assert first_payloads == second_payloads
    assert first[1].payload.kind == "transition"
    expected_values = ("not-json", "scalar", ["one", "two"], {"one": 1, "two": 2})
    migrated = (first[0], first[2], first[3], first[4])
    assert tuple(item.payload.item_count for item in migrated) == (1, 1, 2, 2)
    assert tuple(item.payload.legacy_payload_digest for item in migrated) == tuple(sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest() for value in expected_values)
    assert all(item.payload.legacy_schema_lineage == "v0" for item in migrated)
    assert all(item.payload.legacy_hash_policy == "null_noncanonical_v1" for item in migrated)
    assert first[0].input_hash is None
    assert first[0].output_hash == "b" * 64


def test_reads_reject_malformed_persisted_digests(tmp_path):
    database_path = tmp_path / "corrupt.sqlite3"
    store = AuditStore(database_path)
    store.append(event("event-1"))
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TRIGGER audit_events_no_update")
        connection.execute("UPDATE audit_events SET input_hash = ?", ("A" * 64,))

    with pytest.raises(ValidationError):
        store.list()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("actor", "coordinator_v2"),
        ("event_type", "custom_event"),
        ("status", "maybe"),
        ("model", "gpt-9"),
        ("prompt_version", "raw_prompt"),
        ("schema_name", "reasoning_trace"),
    ],
)
def test_audit_semantic_registries_reject_code_shaped_unknown_categories(field, value):
    with pytest.raises(ValidationError):
        event("event-registry", **{field: value})


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "transition", "subject_ids": ()},
        {"kind": "transition", "subject_ids": ("task-1",), "item_count": 1},
        {"kind": "validation", "subject_ids": ("task-1",)},
        {"kind": "legacy_migration", "legacy_payload_digest": "a" * 64},
    ],
)
def test_audit_payload_kinds_enforce_required_and_forbidden_fields(payload):
    with pytest.raises(ValidationError):
        event("event-payload", payload=payload)


@pytest.mark.parametrize("type_name", ["AuditEventId", "AuditTraceId", "AuditWorkflowId", "AuditTaskId", "AuditAttemptId", "AuditSourceReference"])
@pytest.mark.parametrize("unsafe", ["raw prompt text", "reasoning_trace", "password=secret", "eyJhbGciOiJIUzI1NiJ9.payload.signature", "-----BEGIN PRIVATE KEY-----", "https://host/?token=secret"])
def test_dedicated_audit_identifier_contracts_reject_sensitive_categories(type_name, unsafe):
    identifier_type = getattr(workflow_audit, type_name)
    with pytest.raises(ValidationError):
        TypeAdapter(identifier_type).validate_python(unsafe)


def test_audit_indexes_match_filter_and_deterministic_ordering(tmp_path):
    database_path = tmp_path / "indexes.sqlite3"
    AuditStore(database_path)
    expected = {
        "audit_events_trace_sequence_idx": ("trace_id", "sequence", "event_id"),
        "audit_events_workflow_idx": ("workflow_id", "trace_id", "sequence", "event_id"),
        "audit_events_task_idx": ("task_id", "trace_id", "sequence", "event_id"),
        "audit_events_attempt_idx": ("attempt_id", "trace_id", "sequence", "event_id"),
        "audit_events_occurred_at_idx": ("occurred_at", "trace_id", "sequence", "event_id"),
        "audit_events_type_time_idx": ("event_type", "occurred_at", "trace_id", "sequence", "event_id"),
    }
    with sqlite3.connect(database_path) as connection:
        actual = {name: tuple(row[2] for row in connection.execute(f"PRAGMA index_info('{name}')")) for name in expected}
    assert actual == expected
