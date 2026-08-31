from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
import json
import sqlite3

import pytest

from market_agent.workflow_harness_contracts import (
    AttemptWorkItemOwnershipRecord,
    HarnessTransition,
    ReconciliationResolutionRecord,
    RunState,
    TransitionAuthorityRecord,
)
from market_agent.workflow_session import (
    EventIntegrityError,
    HarnessEvent,
    LegacyHarnessDatabaseError,
    LeaseConflictError,
    OptimisticConcurrencyError,
    SQLiteHarnessEventStore,
    canonical_event_hash,
    fold_events,
)


NOW = datetime(2026, 8, 30, tzinfo=timezone.utc)


def run_event(
    event_id: str,
    *,
    run_id: str = "run-1",
    trace_id: str = "trace-1",
    from_state: str = "none",
    to_state: str = "created",
    expected_state_revision: int = 0,
    plan_revision: int = 0,
) -> HarnessEvent:
    return HarnessEvent(
        event_id=event_id,
        trace_id=trace_id,
        span_id=f"span-{event_id}",
        run_id=run_id,
        event_type="run_transitioned",
        occurred_at=NOW,
        monotonic_offset=1.0,
        actor="harness",
        payload={"reason": "test"},
        transition=HarnessTransition(
            run_id=run_id,
            trace_id=trace_id,
            entity_kind="run",
            entity_id=run_id,
            from_state=from_state,
            to_state=to_state,
            expected_state_revision=expected_state_revision,
            plan_revision=plan_revision,
            reason_code="run_created",
            idempotency_key=f"idempotency-{event_id}",
        ),
    )


def audit_event(
    event_id: str,
    *,
    run_id: str = "run-1",
    trace_id: str = "trace-1",
) -> HarnessEvent:
    return HarnessEvent(
        event_id=event_id,
        trace_id=trace_id,
        span_id=f"span-{event_id}",
        run_id=run_id,
        event_type="model_observed",
        occurred_at=NOW,
        monotonic_offset=2.0,
        actor="model_gateway",
        payload={"observation": "accepted"},
    )


def lease_digest(fencing_token: str) -> str:
    return sha256(fencing_token.encode("utf-8")).hexdigest()


def work_event(event_id: str, lease, *, payload=None) -> HarnessEvent:
    return HarnessEvent(
        event_id=event_id,
        trace_id="trace-1",
        span_id=f"span-{event_id}",
        run_id="run-1",
        work_item_id=lease.work_item_id,
        attempt_id=lease.attempt_id,
        event_type="work_item_transitioned",
        occurred_at=NOW,
        monotonic_offset=3.0,
        actor="harness",
        payload=payload or {"lease_epoch": lease.lease_epoch},
        transition=HarnessTransition(
            run_id="run-1",
            trace_id="trace-1",
            entity_kind="work_item",
            entity_id=lease.work_item_id,
            from_state="none",
            to_state="leased",
            expected_state_revision=1,
            plan_revision=0,
            reason_code="lease_acquired",
            idempotency_key=f"idempotency-{event_id}",
            lease_epoch=lease.lease_epoch,
            fencing_token_digest=lease_digest(lease.fencing_token),
        ),
    )


def authority_event(
    event_id: str,
    *,
    transition_authority=None,
    attempt_ownership=None,
    reconciliation_resolution=None,
) -> HarnessEvent:
    event_type = (
        "transition_authorized"
        if transition_authority is not None
        else "attempt_ownership_recorded"
        if attempt_ownership is not None
        else "reconciliation_resolved"
    )
    return HarnessEvent(
        event_id=event_id,
        trace_id="trace-1",
        span_id=f"span-{event_id}",
        run_id="run-1",
        event_type=event_type,
        occurred_at=NOW,
        monotonic_offset=4.0,
        actor="harness",
        payload={"authority_record": event_type},
        transition_authority=transition_authority,
        attempt_ownership=attempt_ownership,
        reconciliation_resolution=reconciliation_resolution,
    )


def resolution_record(
    reconciliation_id: str, *, broker_observation_digest: str = "a" * 64
) -> ReconciliationResolutionRecord:
    return ReconciliationResolutionRecord(
        run_id="run-1",
        trace_id="trace-1",
        reconciliation_id=reconciliation_id,
        expected_state_revision=1,
        plan_revision=0,
        broker_observation_digest=broker_observation_digest,
        side_effect_resolved=True,
    )


def rehash(event: HarnessEvent) -> HarnessEvent:
    unsigned = event.model_copy(update={"event_hash": None})
    return unsigned.model_copy(
        update={
            "event_hash": canonical_event_hash(
                {
                    key: value
                    for key, value in unsigned.model_dump(
                        mode="json", exclude={"event_hash"}
                    ).items()
                    if key
                    not in {
                        "transition_authority",
                        "attempt_ownership",
                        "reconciliation_resolution",
                    }
                    or key in unsigned.model_fields_set
                }
            )
        }
    )


def legacy_non_run_event_json() -> str:
    return json.dumps(
        {
            "schema_version": "v1",
            "event_id": "legacy-work-leased",
            "trace_id": "trace-1",
            "span_id": "span-legacy",
            "parent_span_id": None,
            "run_id": "run-1",
            "work_item_id": "work-1",
            "attempt_id": "attempt-1",
            "sequence": 1,
            "state_revision": 1,
            "event_type": "work_item_transitioned",
            "occurred_at": NOW.isoformat(),
            "monotonic_offset": 1.0,
            "actor": "harness",
            "payload": {"lease_epoch": 1},
            "transition": {
                "schema_version": "v1",
                "run_id": "run-1",
                "trace_id": "trace-1",
                "entity_kind": "work_item",
                "entity_id": "work-1",
                "from_state": "none",
                "to_state": "leased",
                "expected_state_revision": 0,
                "plan_revision": 0,
                "reason_code": "lease_acquired",
                "idempotency_key": "legacy-lease",
                "fencing_token": "fence-1-legacy-live-secret",
            },
            "previous_event_hash": None,
            "event_hash": "a" * 64,
        },
        sort_keys=True,
        separators=(",", ":"),
    )


@pytest.fixture
def store(tmp_path):
    return SQLiteHarnessEventStore(
        tmp_path / "session.sqlite3", monotonic=lambda: 5.0
    )


def test_sequence_advances_for_every_event_but_revision_only_for_transitions(store):
    created = store.append(
        run_event("run-created"), expected_sequence=0, expected_state_revision=0
    )
    observed = store.append(
        audit_event("model-observed"),
        expected_sequence=1,
        expected_state_revision=1,
    )

    assert (created.sequence, created.state_revision) == (1, 1)
    assert (observed.sequence, observed.state_revision) == (2, 1)


def test_hash_chain_links_each_event_to_the_committed_predecessor(store):
    created = store.append(
        run_event("run-created"), expected_sequence=0, expected_state_revision=0
    )
    observed = store.append(
        audit_event("model-observed"),
        expected_sequence=1,
        expected_state_revision=1,
    )

    assert created.previous_event_hash is None
    assert created.event_hash is not None
    assert observed.previous_event_hash == created.event_hash
    assert observed.event_hash != created.event_hash


@pytest.mark.parametrize(
    ("expected_sequence", "expected_revision"),
    [(1, 0), (0, 1)],
)
def test_append_rejects_optimistic_sequence_and_revision_conflicts(
    store, expected_sequence, expected_revision
):
    with pytest.raises(OptimisticConcurrencyError):
        store.append(
            run_event("run-created"),
            expected_sequence=expected_sequence,
            expected_state_revision=expected_revision,
        )

    assert store.load("run-1") == ()


def test_concurrent_writers_commit_one_contiguous_run_sequence(store):
    store.append(run_event("run-created"), expected_sequence=0, expected_state_revision=0)

    def append_with_retry(index: int) -> HarnessEvent:
        while True:
            view = store.snapshot("run-1")
            try:
                return store.append(
                    audit_event(f"observed-{index}"),
                    expected_sequence=view.sequence,
                    expected_state_revision=view.state_revision,
                )
            except OptimisticConcurrencyError:
                continue

    with ThreadPoolExecutor(max_workers=8) as executor:
        committed = list(executor.map(append_with_retry, range(24)))

    assert sorted(event.sequence for event in committed) == list(range(2, 26))
    assert [event.sequence for event in store.load("run-1")] == list(range(1, 26))


def test_append_rejects_trace_change_within_a_run(store):
    store.append(run_event("run-created"), expected_sequence=0, expected_state_revision=0)

    with pytest.raises(EventIntegrityError, match="trace"):
        store.append(
            audit_event("wrong-trace", trace_id="trace-2"),
            expected_sequence=1,
            expected_state_revision=1,
        )


def test_replay_rejects_run_identity_change(store):
    first = store.append(
        run_event("run-created"), expected_sequence=0, expected_state_revision=0
    )
    second = store.append(
        audit_event("model-observed"),
        expected_sequence=1,
        expected_state_revision=1,
    )
    corrupted = second.model_copy(update={"run_id": "run-2"})

    with pytest.raises(EventIntegrityError, match="run"):
        fold_events((first, corrupted))


def test_replay_rejects_schema_mismatch(store):
    committed = store.append(
        run_event("run-created"), expected_sequence=0, expected_state_revision=0
    )
    corrupted = committed.model_copy(update={"schema_version": "v2"})

    with pytest.raises(EventIntegrityError, match="schema"):
        fold_events((corrupted,))


def test_replay_rejects_hash_chain_corruption(store):
    committed = store.append(
        run_event("run-created"), expected_sequence=0, expected_state_revision=0
    )
    corrupted = committed.model_copy(update={"event_hash": "0" * 64})

    with pytest.raises(EventIntegrityError, match="hash"):
        fold_events((corrupted,))


def test_transition_plan_revision_must_equal_fixed_active_revision(store):
    store.append(run_event("run-created"), expected_sequence=0, expected_state_revision=0)
    committed = store.append(
        run_event(
            "run-admitted",
            from_state="created",
            to_state="admitted",
            expected_state_revision=1,
            plan_revision=0,
        ),
        expected_sequence=1,
        expected_state_revision=1,
    )

    assert committed.transition.plan_revision == 0
    assert store.snapshot("run-1").plan_revision == 0


def test_append_rejects_jumping_plan_revision(store):
    store.append(run_event("run-created"), expected_sequence=0, expected_state_revision=0)

    with pytest.raises(EventIntegrityError, match="plan revision"):
        store.append(
            run_event(
                "run-admitted",
                from_state="created",
                to_state="admitted",
                expected_state_revision=1,
                plan_revision=1,
            ),
            expected_sequence=1,
            expected_state_revision=1,
        )


def test_replay_rejects_changed_plan_revision_with_valid_event_hash(store):
    created = store.append(
        run_event("run-created"), expected_sequence=0, expected_state_revision=0
    )
    admitted = store.append(
        run_event(
            "run-admitted",
            from_state="created",
            to_state="admitted",
            expected_state_revision=1,
            plan_revision=0,
        ),
        expected_sequence=1,
        expected_state_revision=1,
    )
    corrupted = rehash(
        admitted.model_copy(
            update={
                "transition": admitted.transition.model_copy(
                    update={"plan_revision": 1}
                )
            }
        )
    )

    with pytest.raises(EventIntegrityError, match="plan revision"):
        fold_events((created, corrupted))


def test_reopen_replays_the_same_snapshot(tmp_path):
    database_path = tmp_path / "session.sqlite3"
    first_store = SQLiteHarnessEventStore(database_path)
    first_store.append(
        run_event("run-created"), expected_sequence=0, expected_state_revision=0
    )
    first_store.append(
        audit_event("model-observed"),
        expected_sequence=1,
        expected_state_revision=1,
    )
    before = first_store.snapshot("run-1")

    reopened = SQLiteHarnessEventStore(database_path)

    assert reopened.snapshot("run-1") == before
    assert reopened.snapshot("run-1").run_state is RunState.CREATED


def test_b9f82ee_raw_lease_schema_is_rejected_before_any_mutation(tmp_path):
    database_path = tmp_path / "legacy-raw-lease.sqlite3"
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE harness_leases ("
            "run_id TEXT NOT NULL, work_item_id TEXT NOT NULL, "
            "attempt_id TEXT NOT NULL, lease_epoch INTEGER NOT NULL, "
            "fencing_token TEXT NOT NULL, holder_id TEXT NOT NULL, "
            "expires_at_monotonic REAL NOT NULL, "
            "PRIMARY KEY(run_id, work_item_id))"
        )
        connection.execute(
            "INSERT INTO harness_leases VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                "run-1",
                "work-1",
                "attempt-1",
                1,
                "fence-1-legacy-live-secret",
                "worker-1",
                10.0,
            ),
        )
    before = database_path.read_bytes()
    with pytest.raises(
        LegacyHarnessDatabaseError, match="revoke.*securely.*fresh"
    ):
        SQLiteHarnessEventStore(database_path)

    assert database_path.read_bytes() == before
    with sqlite3.connect(database_path) as connection:
        columns = {
            row[1] for row in connection.execute("PRAGMA table_info(harness_leases)")
        }
        token = connection.execute(
            "SELECT fencing_token FROM harness_leases"
        ).fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
    assert columns == {
        "run_id",
        "work_item_id",
        "attempt_id",
        "lease_epoch",
        "fencing_token",
        "holder_id",
        "expires_at_monotonic",
    }
    assert token == "fence-1-legacy-live-secret"
    assert tables == {"harness_leases"}


def test_b9f82ee_v1_non_run_event_is_rejected_before_schema_mutation(tmp_path):
    database_path = tmp_path / "legacy-v1-event.sqlite3"
    rendered = legacy_non_run_event_json()
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "CREATE TABLE harness_events ("
            "event_id TEXT PRIMARY KEY, run_id TEXT NOT NULL, "
            "trace_id TEXT NOT NULL, sequence INTEGER NOT NULL, "
            "state_revision INTEGER NOT NULL, event_hash TEXT NOT NULL, "
            "previous_event_hash TEXT, event_json TEXT NOT NULL, "
            "UNIQUE(run_id, sequence))"
        )
        connection.execute(
            "INSERT INTO harness_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy-work-leased",
                "run-1",
                "trace-1",
                1,
                1,
                "a" * 64,
                None,
                rendered,
            ),
        )
        connection.execute(
            "CREATE TABLE harness_leases ("
            "run_id TEXT NOT NULL, work_item_id TEXT NOT NULL, "
            "attempt_id TEXT NOT NULL, lease_epoch INTEGER NOT NULL, "
            "fencing_token_digest TEXT NOT NULL, holder_id TEXT NOT NULL, "
            "expires_at_monotonic REAL NOT NULL, "
            "PRIMARY KEY(run_id, work_item_id))"
        )
    before = database_path.read_bytes()
    with pytest.raises(
        LegacyHarnessDatabaseError, match="incompatible v1 non-run event"
    ):
        SQLiteHarnessEventStore(database_path)

    assert database_path.read_bytes() == before


def test_current_digest_only_non_run_stream_reopens_unchanged(tmp_path):
    database_path = tmp_path / "current-digest-only.sqlite3"
    store = SQLiteHarnessEventStore(database_path, monotonic=lambda: 5.0)
    store.append(run_event("run-created"), expected_sequence=0, expected_state_revision=0)
    lease = store.acquire_lease(
        "run-1",
        "work-1",
        "attempt-1",
        "worker-a",
        expires_at_monotonic=10.0,
        expected_lease_epoch=0,
    )
    store.append(
        work_event("work-leased", lease),
        expected_sequence=1,
        expected_state_revision=1,
        lease=lease,
    )
    before = store.snapshot("run-1")

    reopened = SQLiteHarnessEventStore(database_path, monotonic=lambda: 5.0)

    assert reopened.snapshot("run-1") == before


def test_load_rejects_canonical_json_for_another_run_stored_under_requested_run(
    store, tmp_path
):
    committed = store.append(
        run_event("run-created"), expected_sequence=0, expected_state_revision=0
    )
    other_run = rehash(
        committed.model_copy(
            update={
                "run_id": "run-2",
                "transition": committed.transition.model_copy(
                    update={"run_id": "run-2", "entity_id": "run-2"}
                ),
            }
        )
    )
    rendered = json.dumps(
        other_run.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    with sqlite3.connect(tmp_path / "session.sqlite3") as connection:
        connection.execute("DROP TRIGGER harness_events_no_update")
        connection.execute(
            "UPDATE harness_events SET event_json = ? WHERE event_id = ?",
            (rendered, committed.event_id),
        )

    with pytest.raises(EventIntegrityError, match="row membership"):
        store.load("run-1")


def test_event_rows_are_append_only_even_via_replace(store, tmp_path):
    committed = store.append(
        run_event("run-created"), expected_sequence=0, expected_state_revision=0
    )
    database_path = tmp_path / "session.sqlite3"

    with sqlite3.connect(database_path) as connection:
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            connection.execute(
                "UPDATE harness_events SET event_json = '{}' WHERE event_id = ?",
                (committed.event_id,),
            )
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            connection.execute(
                "DELETE FROM harness_events WHERE event_id = ?", (committed.event_id,)
            )
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            connection.execute(
                "INSERT OR REPLACE INTO harness_events "
                "SELECT * FROM harness_events WHERE event_id = ?",
                (committed.event_id,),
            )


def test_conflict_rolls_back_event_counter_and_outbox_together(store, tmp_path):
    store.append(run_event("run-created"), expected_sequence=0, expected_state_revision=0)

    with pytest.raises(OptimisticConcurrencyError):
        store.append(
            audit_event("conflicted"),
            expected_sequence=0,
            expected_state_revision=1,
        )

    with sqlite3.connect(tmp_path / "session.sqlite3") as connection:
        event_count = connection.execute("SELECT COUNT(*) FROM harness_events").fetchone()[0]
        outbox_count = connection.execute("SELECT COUNT(*) FROM harness_outbox").fetchone()[0]
        counters = connection.execute(
            "SELECT sequence, state_revision FROM harness_runs WHERE run_id = 'run-1'"
        ).fetchone()
    assert (event_count, outbox_count, counters) == (1, 1, (1, 1))


def test_newer_lease_fences_a_stale_work_item_transition(store):
    store.append(run_event("run-created"), expected_sequence=0, expected_state_revision=0)
    stale = store.acquire_lease(
        "run-1",
        "work-1",
        "attempt-1",
        "worker-a",
        expires_at_monotonic=10.0,
        expected_lease_epoch=0,
    )
    current = store.acquire_lease(
        "run-1",
        "work-1",
        "attempt-2",
        "worker-b",
        expires_at_monotonic=20.0,
        expected_lease_epoch=1,
    )
    stale_event = work_event("work-leased-stale", stale)

    with pytest.raises(LeaseConflictError, match="fencing"):
        store.append(
            stale_event,
            expected_sequence=1,
            expected_state_revision=1,
            lease=stale,
        )

    fresh_event = work_event("work-leased-current", current)
    committed = store.append(
        fresh_event,
        expected_sequence=1,
        expected_state_revision=1,
        lease=current,
    )

    assert (committed.sequence, committed.state_revision) == (2, 2)
    assert store.snapshot("run-1").work_item_states == (("work-1", "leased"),)


def test_lease_epoch_is_optimistically_fenced(store):
    store.append(run_event("run-created"), expected_sequence=0, expected_state_revision=0)
    store.acquire_lease(
        "run-1",
        "work-1",
        "attempt-1",
        "worker-a",
        expires_at_monotonic=10.0,
        expected_lease_epoch=0,
    )

    with pytest.raises(LeaseConflictError, match="epoch"):
        store.acquire_lease(
            "run-1",
            "work-1",
            "attempt-2",
            "worker-b",
            expires_at_monotonic=20.0,
            expected_lease_epoch=0,
        )


def test_work_item_transition_cannot_omit_out_of_band_lease_proof(store):
    store.append(run_event("run-created"), expected_sequence=0, expected_state_revision=0)
    lease = store.acquire_lease(
        "run-1",
        "work-1",
        "attempt-1",
        "worker-a",
        expires_at_monotonic=10.0,
        expected_lease_epoch=0,
    )

    with pytest.raises(LeaseConflictError, match="lease proof"):
        store.append(
            work_event("work-no-proof", lease),
            expected_sequence=1,
            expected_state_revision=1,
        )


def test_expired_lease_cannot_authorize_transition(tmp_path):
    store = SQLiteHarnessEventStore(
        tmp_path / "expired.sqlite3", monotonic=lambda: 11.0
    )
    store.append(run_event("run-created"), expected_sequence=0, expected_state_revision=0)
    lease = store.acquire_lease(
        "run-1",
        "work-1",
        "attempt-1",
        "worker-a",
        expires_at_monotonic=10.0,
        expected_lease_epoch=0,
    )

    with pytest.raises(LeaseConflictError, match="expired"):
        store.append(
            work_event("work-expired", lease),
            expected_sequence=1,
            expected_state_revision=1,
            lease=lease,
        )


@pytest.mark.parametrize(
    "proof_update",
    [
        {"run_id": "run-2"},
        {"work_item_id": "work-2"},
        {"attempt_id": "attempt-2"},
        {"lease_epoch": 2},
        {"lease_epoch": True},
        {"fencing_token": "fence-1-wrong-live-secret"},
    ],
)
def test_lease_proof_must_match_current_identity_epoch_and_token(
    store, proof_update
):
    store.append(run_event("run-created"), expected_sequence=0, expected_state_revision=0)
    lease = store.acquire_lease(
        "run-1",
        "work-1",
        "attempt-1",
        "worker-a",
        expires_at_monotonic=10.0,
        expected_lease_epoch=0,
    )
    proof = lease.model_copy(update=proof_update)

    with pytest.raises(LeaseConflictError):
        store.append(
            work_event("work-bad-proof", lease),
            expected_sequence=1,
            expected_state_revision=1,
            lease=proof,
        )


@pytest.mark.parametrize(
    "payload",
    [
        {"nested": {"fencing_token": "not-even-a-live-token"}},
        {"nested": {"clientSecret": "opaque"}},
        {"nested": [{"proof": "fence-1-live-secret"}]},
        {"nested": [{"proof": "password=opaque"}]},
        {"authorization": "opaque"},
    ],
)
def test_event_payload_recursively_rejects_sensitive_keys_and_values(payload):
    with pytest.raises(ValueError, match="sensitive"):
        audit_event("sensitive-payload").model_copy(update={"payload": payload})
        HarnessEvent.model_validate(
            audit_event("sensitive-payload").model_copy(
                update={"payload": payload}
            ).model_dump(mode="python")
        )


def test_transition_metadata_cannot_duplicate_live_fencing_credential():
    candidate = run_event("sensitive-transition").model_copy(
        update={
            "transition": run_event("sensitive-transition").transition.model_copy(
                update={"reason_code": "fence-1-live-secret"}
            )
        }
    )

    with pytest.raises(ValueError, match="sensitive"):
        HarnessEvent.model_validate(candidate.model_dump(mode="python"))


def test_live_fencing_token_is_absent_from_event_and_outbox_json(store, tmp_path):
    store.append(run_event("run-created"), expected_sequence=0, expected_state_revision=0)
    lease = store.acquire_lease(
        "run-1",
        "work-1",
        "attempt-1",
        "worker-a",
        expires_at_monotonic=10.0,
        expected_lease_epoch=0,
    )
    committed = store.append(
        work_event("work-leased", lease),
        expected_sequence=1,
        expected_state_revision=1,
        lease=lease,
    )

    with sqlite3.connect(tmp_path / "session.sqlite3") as connection:
        event_json = connection.execute(
            "SELECT event_json FROM harness_events WHERE event_id = ?",
            (committed.event_id,),
        ).fetchone()[0]
        outbox_json = connection.execute(
            "SELECT event_json FROM harness_outbox WHERE event_id = ?",
            (committed.event_id,),
        ).fetchone()[0]
        lease_row = connection.execute(
            "SELECT * FROM harness_leases WHERE run_id = 'run-1'"
        ).fetchone()
    assert lease.fencing_token not in event_json
    assert lease.fencing_token not in outbox_json
    assert lease.fencing_token not in repr(lease_row)
    assert lease_digest(lease.fencing_token) in event_json
    assert lease_digest(lease.fencing_token) in lease_row
    assert json.loads(event_json)["transition"]["lease_epoch"] == lease.lease_epoch


def test_allowlisted_authority_events_fold_durable_transition_relationships(store):
    store.append(run_event("run-created"), expected_sequence=0, expected_state_revision=0)
    authority = TransitionAuthorityRecord(
        run_id="run-1",
        trace_id="trace-1",
        entity_kind="work_item",
        entity_id="work-1",
        from_state="none",
        to_state="leased",
        expected_state_revision=1,
        plan_revision=0,
        reason_code="lease_acquired",
        idempotency_key="authority-1",
        dependency_versions=(),
        reservation_id="reservation-1",
        grant_id="grant-1",
        lease_epoch=1,
        fencing_token_digest="a" * 64,
    )
    ownership = AttemptWorkItemOwnershipRecord(
        run_id="run-1",
        trace_id="trace-1",
        attempt_id="attempt-1",
        work_item_id="work-1",
        plan_revision=0,
    )
    resolution = ReconciliationResolutionRecord(
        run_id="run-1",
        trace_id="trace-1",
        reconciliation_id="broker-observation-1",
        expected_state_revision=1,
        plan_revision=0,
        broker_observation_digest="a" * 64,
        side_effect_resolved=True,
    )

    for event in (
        authority_event("authority", transition_authority=authority),
        authority_event("ownership", attempt_ownership=ownership),
        authority_event("resolution", reconciliation_resolution=resolution),
    ):
        store.append(
            event,
            expected_sequence=store.snapshot("run-1").sequence,
            expected_state_revision=store.snapshot("run-1").state_revision,
        )

    view = store.snapshot("run-1")
    assert view.transition_authorities == (authority,)
    assert view.attempt_work_item_owners == (ownership,)
    assert view.reconciliation_resolutions == (resolution,)


def test_generic_audit_payload_cannot_be_interpreted_as_authority(store):
    store.append(run_event("run-created"), expected_sequence=0, expected_state_revision=0)
    store.append(
        audit_event("looks-authoritative").model_copy(
            update={"payload": {"reservation_id": "reservation-1"}}
        ),
        expected_sequence=1,
        expected_state_revision=1,
    )

    assert store.snapshot("run-1").transition_authorities == ()


def test_pre_authority_v1_event_hash_replays_without_rewriting_its_bytes():
    legacy = run_event("pre-authority")
    unsigned = legacy.model_copy(
        update={"sequence": 1, "state_revision": 1, "event_hash": None}
    )
    old_values = unsigned.model_dump(
        mode="json",
        exclude={
            "event_hash",
            "transition_authority",
            "attempt_ownership",
            "reconciliation_resolution",
        },
    )
    historical = unsigned.model_copy(
        update={"event_hash": canonical_event_hash(old_values)}
    )

    assert fold_events((historical,)).run_state is RunState.CREATED


def test_pre_authority_v1_event_reopens_from_omitted_authority_json(tmp_path):
    database_path = tmp_path / "pre-authority.sqlite3"
    store = SQLiteHarnessEventStore(database_path)
    legacy = run_event("pre-authority-reopen")
    unsigned = legacy.model_copy(
        update={"sequence": 1, "state_revision": 1, "event_hash": None}
    )
    old_values = unsigned.model_dump(
        mode="json",
        exclude={
            "event_hash",
            "transition_authority",
            "attempt_ownership",
            "reconciliation_resolution",
        },
    )
    event_hash = canonical_event_hash(old_values)
    rendered = json.dumps(
        {**old_values, "event_hash": event_hash}, sort_keys=True, separators=(",", ":")
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO harness_runs VALUES (?, ?, ?, ?, ?)",
            ("run-1", "trace-1", 1, 1, event_hash),
        )
        connection.execute(
            "INSERT INTO harness_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "pre-authority-reopen",
                "run-1",
                "trace-1",
                1,
                1,
                event_hash,
                None,
                rendered,
            ),
        )

    reopened = SQLiteHarnessEventStore(database_path)
    assert reopened.snapshot("run-1").run_state is RunState.CREATED


def test_authority_records_reject_sensitive_values_even_after_model_copy(store):
    store.append(run_event("run-created"), expected_sequence=0, expected_state_revision=0)
    authority = TransitionAuthorityRecord(
        run_id="run-1",
        trace_id="trace-1",
        entity_kind="work_item",
        entity_id="work-1",
        from_state="none",
        to_state="leased",
        expected_state_revision=1,
        plan_revision=0,
        reason_code="lease_acquired",
        idempotency_key="authority-1",
        reservation_id="reservation-1",
        grant_id="grant-1",
        lease_epoch=1,
        fencing_token_digest="a" * 64,
    )
    bypassed = authority_event("sensitive-authority", transition_authority=authority).model_copy(
        update={"transition_authority": authority.model_copy(update={"grant_id": "sk-live-secret"})}
    )

    with pytest.raises(EventIntegrityError, match="sensitive"):
        store.append(bypassed, expected_sequence=1, expected_state_revision=1)


def test_fold_rejects_conflicting_authority_records_at_one_revision(store):
    store.append(run_event("run-created"), expected_sequence=0, expected_state_revision=0)
    first = TransitionAuthorityRecord(
        run_id="run-1",
        trace_id="trace-1",
        entity_kind="work_item",
        entity_id="work-1",
        from_state="none",
        to_state="leased",
        expected_state_revision=1,
        plan_revision=0,
        reason_code="lease_acquired",
        idempotency_key="authority-1",
        reservation_id="reservation-1",
        grant_id="grant-1",
        lease_epoch=1,
        fencing_token_digest="a" * 64,
    )
    conflicting = first.model_copy(update={"grant_id": "grant-2"})
    store.append(
        authority_event("first-authority", transition_authority=first),
        expected_sequence=1,
        expected_state_revision=1,
    )

    with pytest.raises(EventIntegrityError, match="authority"):
        store.append(
            authority_event("conflicting-authority", transition_authority=conflicting),
            expected_sequence=2,
            expected_state_revision=1,
        )


@pytest.mark.parametrize("second_digest", ["a" * 64, "b" * 64])
def test_append_rejects_duplicate_reconciliation_scope_across_record_ids(
    store, second_digest
):
    store.append(run_event("run-created"), expected_sequence=0, expected_state_revision=0)
    first = resolution_record("broker-observation-1")
    second = resolution_record(
        "broker-observation-2", broker_observation_digest=second_digest
    )
    store.append(
        authority_event("first-resolution", reconciliation_resolution=first),
        expected_sequence=1,
        expected_state_revision=1,
    )

    with pytest.raises(EventIntegrityError, match="authority"):
        store.append(
            authority_event("second-resolution", reconciliation_resolution=second),
            expected_sequence=2,
            expected_state_revision=1,
        )


@pytest.mark.parametrize("second_digest", ["a" * 64, "b" * 64])
def test_independent_fold_rejects_duplicate_reconciliation_scope_across_record_ids(
    second_digest,
):
    created = rehash(
        run_event("run-created").model_copy(
            update={"sequence": 1, "state_revision": 1}
        )
    )
    first = rehash(
        authority_event(
            "first-resolution",
            reconciliation_resolution=resolution_record("broker-observation-1"),
        ).model_copy(
            update={
                "sequence": 2,
                "state_revision": 1,
                "previous_event_hash": created.event_hash,
            }
        )
    )
    second = rehash(
        authority_event(
            "second-resolution",
            reconciliation_resolution=resolution_record(
                "broker-observation-2",
                broker_observation_digest=second_digest,
            ),
        ).model_copy(
            update={
                "sequence": 3,
                "state_revision": 1,
                "previous_event_hash": first.event_hash,
            }
        )
    )

    with pytest.raises(EventIntegrityError, match="authority"):
        fold_events((created, first, second))


@pytest.mark.parametrize(
    "event_type",
    [
        "transition_authorized",
        "attempt_ownership_recorded",
        "reconciliation_resolved",
    ],
)
@pytest.mark.parametrize("with_transition", [False, True])
def test_reserved_authority_event_requires_its_record_and_forbids_transitions(
    event_type, with_transition
):
    values = audit_event(f"missing-{event_type}").model_dump(mode="python")
    values["event_type"] = event_type
    if with_transition:
        values["transition"] = run_event("incompatible-transition").transition

    with pytest.raises(ValueError, match="authority"):
        HarnessEvent(**values)


def authority_records_for_cross_wiring():
    transition_authority = TransitionAuthorityRecord(
        run_id="run-1",
        trace_id="trace-1",
        entity_kind="run",
        entity_id="run-1",
        from_state="created",
        to_state="admitted",
        expected_state_revision=1,
        plan_revision=0,
        reason_code="admitted",
        idempotency_key="authority-cross-wire",
    )
    attempt_ownership = AttemptWorkItemOwnershipRecord(
        run_id="run-1",
        trace_id="trace-1",
        attempt_id="attempt-1",
        work_item_id="work-1",
        plan_revision=0,
    )
    return transition_authority, attempt_ownership, resolution_record("resolution-cross-wire")


@pytest.mark.parametrize(
    ("event_type", "record_field", "record_index"),
    [
        ("transition_authorized", "attempt_ownership", 1),
        ("transition_authorized", "reconciliation_resolution", 2),
        ("attempt_ownership_recorded", "transition_authority", 0),
        ("attempt_ownership_recorded", "reconciliation_resolution", 2),
        ("reconciliation_resolved", "transition_authority", 0),
        ("reconciliation_resolved", "attempt_ownership", 1),
    ],
)
def test_reserved_authority_event_rejects_cross_wired_records(
    event_type, record_field, record_index
):
    records = authority_records_for_cross_wiring()
    values = audit_event(f"cross-wired-{event_type}-{record_field}").model_dump(
        mode="python"
    )
    values.update({"event_type": event_type, record_field: records[record_index]})

    with pytest.raises(ValueError, match="authority"):
        HarnessEvent(**values)


@pytest.mark.parametrize(
    "event_type",
    [
        "transition_authorized",
        "attempt_ownership_recorded",
        "reconciliation_resolved",
    ],
)
def test_model_copy_cannot_bypass_reserved_authority_record_requirement(
    store, event_type
):
    bypassed = audit_event(f"bypassed-{event_type}").model_copy(
        update={"event_type": event_type}
    )

    with pytest.raises(EventIntegrityError, match="authority"):
        store.append(bypassed, expected_sequence=0, expected_state_revision=0)


@pytest.mark.parametrize(
    "event_type",
    [
        "transition_authorized",
        "attempt_ownership_recorded",
        "reconciliation_resolved",
    ],
)
def test_persisted_reserved_authority_event_without_record_fails_replay(
    tmp_path, event_type
):
    database_path = tmp_path / f"invalid-{event_type}.sqlite3"
    SQLiteHarnessEventStore(database_path)
    committed = rehash(
        audit_event(f"persisted-{event_type}").model_copy(
            update={"event_type": event_type, "sequence": 1}
        )
    )
    rendered = json.dumps(
        committed.model_dump(mode="json", exclude_unset=True),
        sort_keys=True,
        separators=(",", ":"),
    )
    with sqlite3.connect(database_path) as connection:
        connection.execute(
            "INSERT INTO harness_runs VALUES (?, ?, ?, ?, ?)",
            ("run-1", "trace-1", 1, 0, committed.event_hash),
        )
        connection.execute(
            "INSERT INTO harness_events VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                committed.event_id,
                committed.run_id,
                committed.trace_id,
                committed.sequence,
                committed.state_revision,
                committed.event_hash,
                committed.previous_event_hash,
                rendered,
            ),
        )

    with pytest.raises(EventIntegrityError, match="persisted harness event"):
        SQLiteHarnessEventStore(database_path).snapshot("run-1")
