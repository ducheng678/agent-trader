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


def test_current_storage_corruption_fails_reopen_without_repair(tmp_path):
    database_path = tmp_path / "current-corrupt.sqlite3"
    store = AuditStore(database_path)
    store.append(event("event-current"))
    with sqlite3.connect(database_path) as connection:
        connection.execute("DROP TRIGGER audit_events_no_update")
        connection.execute("UPDATE audit_events SET input_hash = ?", ("BAD",))

    with pytest.raises(ValidationError):
        AuditStore(database_path)
    with sqlite3.connect(database_path) as connection:
        assert connection.execute("SELECT input_hash FROM audit_events").fetchone()[0] == "BAD"


def test_v0_and_generic_v1_rows_persist_row_lineage_and_hash_policy(tmp_path):
    v0_path = tmp_path / "v0.sqlite3"
    valid_payload = json.dumps({"kind": "transition", "subject_ids": ["task-1"]})
    _create_legacy_database(v0_path, [_legacy_row("event-v0", 1, valid_payload)])
    v0_event = AuditStore(v0_path).list()[0]

    generic_path = tmp_path / "generic-v1.sqlite3"
    generic_row = list(_legacy_row("event-generic", 1, json.dumps({"old": "payload"}), schema_version="v1"))
    original_semantics = {
        "actor": "old_actor_code",
        "event_type": "old_event_code",
        "model": "old_model_code",
        "prompt_version": "old_prompt_code",
        "schema_name": "old_schema_code",
        "status": "old_status_code",
    }
    generic_row[7] = original_semantics["actor"]
    generic_row[8] = original_semantics["event_type"]
    generic_row[9] = original_semantics["status"]
    generic_row[17] = original_semantics["model"]
    generic_row[18] = original_semantics["prompt_version"]
    generic_row[19] = original_semantics["schema_name"]
    _create_legacy_database(generic_path, [tuple(generic_row)], schema_version=True)
    generic_event = AuditStore(generic_path).list()[0]
    reopened = AuditStore(generic_path).list()[0]

    assert (v0_event.source_schema_lineage, v0_event.hash_policy) == ("v0", "null_noncanonical_v1")
    assert (generic_event.source_schema_lineage, generic_event.hash_policy) == ("generic_v1", "null_noncanonical_v1")
    assert (generic_event.actor, generic_event.event_type, generic_event.status, generic_event.model, generic_event.prompt_version, generic_event.schema_name) == ("legacy_actor", "legacy_event", "legacy_status", "legacy_model", "legacy_identifier", "legacy_identifier")
    assert generic_event.legacy_semantic_digest == sha256(json.dumps(original_semantics, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")).hexdigest()
    assert reopened == generic_event


def test_all_transformed_v0_rows_keep_explicit_row_metadata(tmp_path):
    database_path = tmp_path / "v0-matrix.sqlite3"
    valid_payload = json.dumps({"kind": "transition", "subject_ids": ["task-1"]})
    payloads = ("not-json", valid_payload, json.dumps("scalar"), json.dumps(["one"]), json.dumps({"one": 1}))
    _create_legacy_database(database_path, [_legacy_row(f"event-{index}", index, payload) for index, payload in enumerate(payloads, 1)])

    migrated = AuditStore(database_path).list()

    assert {item.source_schema_lineage for item in migrated} == {"v0"}
    assert {item.hash_policy for item in migrated} == {"null_noncanonical_v1"}
    assert all(item.legacy_semantic_digest is not None for item in migrated)


@pytest.mark.parametrize("field,value", [("prompt_version", "sk-liveabcdef"), ("schema_name", "sk-liveabcdef"), ("prompt_version", "ignore_all_previous_instructions"), ("schema_name", "ignore_all_previous_instructions")])
def test_prompt_and_schema_identifiers_reject_secret_and_instruction_categories(field, value):
    with pytest.raises(ValidationError):
        event("event-metadata", **{field: value})


@pytest.mark.parametrize(("type_name", "valid", "cross_type"), [("AuditPromptVersion", "release-v2.1", "AgentTask"), ("AuditSchemaName", "AgentTask", "release-v2.1")])
def test_prompt_and_schema_use_distinct_positive_identifier_grammars(type_name, valid, cross_type):
    identifier_type = getattr(workflow_audit, type_name)
    assert TypeAdapter(identifier_type).validate_python(valid) == valid
    with pytest.raises(ValidationError):
        TypeAdapter(identifier_type).validate_python(cross_type)
    with pytest.raises(ValidationError):
        TypeAdapter(identifier_type).validate_python("ignore_all_previous_instructions")


def test_planned_audit_taxonomy_is_complete_and_closed():
    taxonomy = (
        ("ingress", "ingress_received", "received"),
        ("normalizer", "request_normalized", "normalized"),
        ("classifier", "event_classified", "classified"),
        ("task_planner", "task_plan_created", "planned"),
        ("task_planner", "task_decomposed", "planned"),
        ("task_dispatcher", "task_dispatched", "dispatched"),
        ("scheduler", "task_rescheduled", "rescheduled"),
        ("specialist", "task_completed", "completed"),
        ("context_selector", "context_selected", "completed"),
        ("context_summarizer", "context_summarized", "completed"),
        ("model_router", "model_routed", "completed"),
        ("prompt_builder", "prompt_composed", "completed"),
        ("schema_validator", "schema_validated", "completed"),
        ("cache", "fixed_cache_hit", "hit"),
        ("cache", "semantic_cache_miss", "miss"),
        ("cache", "prompt_cache_write", "completed"),
        ("tool", "tool_dispatched", "dispatched"),
        ("retry_controller", "retry_scheduled", "rescheduled"),
        ("retry_controller", "timeout", "timed_out"),
        ("retry_controller", "cancelled", "cancelled"),
        ("retry_controller", "backoff_scheduled", "rescheduled"),
        ("circuit_breaker", "circuit_opened", "open"),
        ("budget_controller", "budget_exhausted", "rejected"),
        ("knowledge_store", "local_knowledge_retrieved", "completed"),
        ("fallback", "fallback_selected", "completed"),
        ("conflict_resolver", "conflict_detected", "received"),
        ("reflector", "reflection_completed", "completed"),
        ("corrector", "correction_applied", "completed"),
        ("risk_manager", "risk_evaluated", "completed"),
        ("finalizer", "final_decision", "completed"),
        ("memory", "memory_created", "completed"),
        ("memory", "memory_retrieved", "completed"),
        ("memory", "memory_promoted", "promoted"),
        ("tracer", "trace_started", "running"),
        ("tracer", "span_completed", "completed"),
        ("prompt_release_manager", "prompt_released", "completed"),
        ("prompt_release_manager", "prompt_rolled_back", "rolled_back"),
        ("evaluator", "evaluation_started", "running"),
        ("evaluator", "evaluation_completed", "passed"),
    )
    for index, (actor, event_type, status) in enumerate(taxonomy):
        event(f"taxonomy-{index}", actor=actor, event_type=event_type, status=status)
    for outcome in ("accepted", "rejected", "completed", "omitted", "succeeded", "failed", "unavailable", "hit", "miss", "routed", "selected", "rescheduled", "cancelled", "timed_out", "opened", "closed", "exhausted", "retrieved", "promoted", "resolved", "corrected", "approved", "rolled_back", "passed"):
        AuditPayload(kind="validation", subject_ids=("task-1",), outcome_code=outcome)
    for reason in ("validation_error", "budget_limit", "source_limit", "unresolved_conflict", "missing_evidence", "legacy_schema", "audit_failure", "cache_hit", "cache_miss", "retryable_error", "timeout", "cancellation", "backoff", "circuit_open", "budget_exhausted", "local_knowledge", "fallback", "reflection_failure", "risk_rejected", "prompt_rollback", "evaluation_failure"):
        AuditPayload(kind="validation", subject_ids=("task-1",), outcome_code="rejected", reason_code=reason)


def test_append_revalidates_constructed_and_copied_events_before_persistence(tmp_path):
    store = AuditStore(tmp_path / "boundary.sqlite3")
    bad_actor = event("event-copy").model_copy(update={"actor": "arbitrary_actor"})
    bad_payload = AuditPayload.model_construct(kind="transition", subject_ids=["sk-liveabcdef"])
    bad_nested = event("event-nested").model_copy(update={"payload": bad_payload})
    bad_list = event("event-list").model_copy(update={"source_references": ["source-1"]})

    for invalid in (bad_actor, bad_nested, bad_list):
        with pytest.raises(ValidationError):
            store.append(invalid)
    assert store.list() == []


def test_complete_row_classifier_migrates_only_positive_pre_metadata_signatures(tmp_path):
    database_path = tmp_path / "mixed-pre-metadata.sqlite3"
    strict_payload = json.dumps({"kind": "transition", "subject_ids": ["task-1"]})
    current_row = _legacy_row("event-current", 1, strict_payload, input_hash="a" * 64, schema_version="v1")
    semantic_legacy = list(_legacy_row("event-semantic-legacy", 2, strict_payload, input_hash="a" * 64, schema_version="v1"))
    semantic_legacy[7:10] = ["old_actor", "old_event", "old_status"]
    semantic_legacy[17:20] = ["old_model", "old_prompt", "old_schema"]
    generic_payload = list(_legacy_row("event-generic", 3, json.dumps({"kind": "transition", "subject_ids": []}), schema_version="v1"))
    generic_payload[7:10] = ["c96_actor", "c96_event", "c96_status"]
    _create_legacy_database(database_path, [current_row, tuple(semantic_legacy), tuple(generic_payload)], schema_version=True)

    first = AuditStore(database_path).list()
    second = AuditStore(database_path).list()

    assert first == second
    by_id = {item.event_id: item for item in first}
    assert (by_id["event-current"].source_schema_lineage, by_id["event-current"].hash_policy, by_id["event-current"].legacy_semantic_digest) == ("current_v1", "strict_canonical_v1", None)
    for event_id in ("event-semantic-legacy", "event-generic"):
        migrated = by_id[event_id]
        assert (migrated.source_schema_lineage, migrated.hash_policy) == ("generic_v1", "null_noncanonical_v1")
        assert migrated.legacy_semantic_digest is not None
        assert (migrated.actor, migrated.event_type, migrated.status) == ("legacy_actor", "legacy_event", "legacy_status")
        assert migrated.output_hash == "b" * 64
    assert by_id["event-semantic-legacy"].input_hash == "a" * 64
    assert by_id["event-generic"].input_hash is None


@pytest.mark.parametrize(("payload", "input_hash"), [(json.dumps({"kind": "transition", "subject_ids": []}), "a" * 64), (json.dumps({"kind": "transition", "subject_ids": ["task-1"]}), "BAD")])
def test_pre_metadata_current_shaped_corruption_fails_instead_of_migrating(tmp_path, payload, input_hash):
    database_path = tmp_path / f"current-corrupt-{len(payload)}-{len(input_hash)}.sqlite3"
    row = _legacy_row("event-current", 1, payload, input_hash=input_hash, schema_version="v1")
    _create_legacy_database(database_path, [row], schema_version=True)

    with pytest.raises(ValidationError):
        AuditStore(database_path)


@pytest.mark.parametrize(
    ("field_index", "corrupt_value"),
    [
        (7, "coordinator_typo"),
        (8, "task_dispatched_typo"),
        (9, "accepted_typo"),
        (17, "gpt-5.6-terra-typo"),
    ],
)
def test_pre_metadata_current_row_with_one_semantic_corruption_is_not_legacy(
    tmp_path, field_index, corrupt_value
):
    database_path = tmp_path / f"isolated-semantic-corruption-{field_index}.sqlite3"
    payload = json.dumps({"kind": "transition", "subject_ids": ["task-1"]})
    row = list(
        _legacy_row(
            "event-current",
            1,
            payload,
            input_hash="a" * 64,
            schema_version="v1",
        )
    )
    row[17] = "gpt-5.6-terra"
    row[18] = "prompt-v1"
    row[19] = "AgentTask"
    row[field_index] = corrupt_value
    _create_legacy_database(database_path, [tuple(row)], schema_version=True)

    with pytest.raises(ValidationError):
        AuditStore(database_path)


def test_three_current_semantic_typos_do_not_form_a_legacy_signature(tmp_path):
    database_path = tmp_path / "three-current-semantic-typos.sqlite3"
    payload = json.dumps({"kind": "transition", "subject_ids": ["task-1"]})
    row = list(
        _legacy_row(
            "event-current",
            1,
            payload,
            input_hash="a" * 64,
            schema_version="v1",
        )
    )
    row[7:10] = ["coordinator_typo", "task_dispatched_typo", "accepted_typo"]
    row[17:20] = ["gpt-5.6-terra", "prompt-v1", "AgentTask"]
    _create_legacy_database(database_path, [tuple(row)], schema_version=True)

    with pytest.raises(ValidationError):
        AuditStore(database_path)


def test_unknown_storage_schema_version_fails_without_transformation(tmp_path):
    database_path = tmp_path / "unknown-version.sqlite3"
    payload = json.dumps({"kind": "transition", "subject_ids": ["task-1"]})
    _create_legacy_database(database_path, [_legacy_row("event-v2", 1, payload, input_hash="a" * 64, schema_version="v2")], schema_version=True)

    with pytest.raises(ValidationError):
        AuditStore(database_path)


def test_partial_row_metadata_schema_is_rejected_as_ambiguous(tmp_path):
    database_path = tmp_path / "partial-metadata.sqlite3"
    payload = json.dumps({"kind": "transition", "subject_ids": ["task-1"]})
    _create_legacy_database(database_path, [_legacy_row("event-partial", 1, payload, input_hash="a" * 64, schema_version="v1")], schema_version=True)
    with sqlite3.connect(database_path) as connection:
        connection.execute("ALTER TABLE audit_events ADD COLUMN source_schema_lineage TEXT NOT NULL DEFAULT 'current_v1'")

    with pytest.raises(ValueError, match="metadata schema"):
        AuditStore(database_path)


@pytest.mark.parametrize(("type_name", "marker"), [("AuditEventId", "sk_live_abcdef"), ("AuditTraceId", "sk-live-abcdef"), ("AuditWorkflowId", "raw.prompt"), ("AuditTaskId", "system_prompt"), ("AuditAttemptId", "private-reasoning"), ("AuditSourceReference", "api_key"), ("AuditSubjectId", "eyJhbGciOiJIUzI1NiJ9.payload.signature"), ("AuditCode", "-----BEGIN_PRIVATE_KEY-----"), ("AuditCode", "skliveabcdef"), ("AuditPromptVersion", "system.prompt"), ("AuditSchemaName", "private.reasoning")])
def test_every_audit_string_contract_rejects_compact_sensitive_markers(type_name, marker):
    with pytest.raises(ValidationError):
        TypeAdapter(getattr(workflow_audit, type_name)).validate_python(marker)


@pytest.mark.parametrize(("field", "marker"), [("event_id", "sk_live_abcdef"), ("trace_id", "sk-live-abcdef"), ("workflow_id", "raw_prompt"), ("task_id", "system-prompt"), ("attempt_id", "private_reasoning"), ("actor", "api_key"), ("event_type", "raw/prompt"), ("status", "system:prompt"), ("model", "private.reasoning"), ("prompt_version", "-----BEGIN/PRIVATE/KEY-----"), ("schema_name", "eyJhbGciOiJIUzI1NiJ9.payload.signature"), ("source_references", ("sk_live_abcdef",))])
def test_audit_event_positions_reject_separator_and_compact_marker_variants(field, marker):
    values = event("event-sensitive").model_dump(mode="python")
    values[field] = marker
    with pytest.raises(ValidationError):
        AuditEvent.model_validate(values)


@pytest.mark.parametrize("payload", [{"kind": "transition", "subject_ids": ("sk_live_abcdef",)}, {"kind": "validation", "subject_ids": ("task-1",), "outcome_code": "api_key"}, {"kind": "validation", "subject_ids": ("task-1",), "outcome_code": "rejected", "reason_code": "private.reasoning"}])
def test_audit_payload_string_positions_reject_compact_sensitive_markers(payload):
    with pytest.raises(ValidationError):
        AuditPayload(**payload)
