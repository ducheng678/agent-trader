"""Persisted promotion fairness and short-transaction cursor contracts."""
from datetime import datetime, timezone
import sqlite3

import pytest

from market_agent.backend.promotion_cursor_store import (
    PostgresPromotionCursorStore, SQLitePromotionCursorStore,
)
from market_agent.workflow_long_term_memory import (
    DecisionRecord, EventRecord, KnowledgeRevision, Lifecycle, MemoryPromotionError,
    OutcomeRecord, Provenance,
)
from market_agent.workflow_memory_promotion import PromotionScheduler
from market_agent.workflow_memory_sqlite import SQLiteMemoryRepository


NOW = datetime(2026, 9, 1, tzinfo=timezone.utc)


def candidate(identifier, *, tenant="tenant-a", eligible=False):
    return KnowledgeRevision(
        record_id=identifier, tenant_id=tenant, observed_at=NOW,
        knowledge_id=identifier, revision=1, rule="Check independent observations.",
        confidence=0.9, effective_at=NOW, evidence_ids=("event-a", "event-b"),
        outcome_id="outcome" if eligible else None,
    )


def seed(repository, authority, *, tenant="tenant-a"):
    def context(key):
        return dict(tenant_id=tenant, trace_id="seed", idempotency_key=key, authority=authority)
    for identifier in ("event-a", "event-b"):
        repository.append_event(EventRecord(
            record_id=identifier, tenant_id=tenant, observed_at=NOW, source=identifier,
            payload={"fact": identifier}, provenance=Provenance(
                source_id=identifier, source_kind="external", independent_group=identifier,
            ),
        ), **context(identifier))
    repository.append_decision(DecisionRecord(
        record_id="decision", tenant_id=tenant, observed_at=NOW, decision="no_trade",
        status="final", evidence_ids=("event-a",),
    ), **context("decision"))
    repository.append_outcome(OutcomeRecord(
        record_id="outcome", tenant_id=tenant, observed_at=NOW, decision_id="decision",
        result="risk avoided", verified=True, evidence_ids=("event-a",),
    ), **context("outcome"))
    values = (candidate("a", tenant=tenant), candidate("b", tenant=tenant),
              candidate("z", tenant=tenant, eligible=True))
    for value in values:
        repository.propose_knowledge(value, **context(value.record_id))
    return values


def scheduler(repository, authority, store, observer, *, limit=2, tenant="tenant-a"):
    return PromotionScheduler(
        repository=repository, authority=authority, tenant_id=tenant,
        cursor_store=store, evaluation_observer=observer, max_candidates_per_run=limit,
    )


def test_real_repository_restart_reaches_eligible_candidate_and_wraps(tmp_path):
    authority = object()
    path = tmp_path / "cursor.db"
    observed = []
    with SQLiteMemoryRepository(tmp_path / "memory.db", writer_authority=authority) as repo:
        values = seed(repo, authority)
        first = scheduler(repo, authority, SQLitePromotionCursorStore(path), observed.append)
        assert first.evaluate(reversed(values), now=NOW, trace_id="first") == ()
        assert [item.candidate_id for item in observed] == ["a", "b"]
    with SQLiteMemoryRepository(tmp_path / "memory.db", writer_authority=authority) as repo:
        store = SQLitePromotionCursorStore(path)
        assert store.read(tenant_id="tenant-a").record_id == "b"
        second = scheduler(repo, authority, store, observed.append)
        active = second.evaluate(values, now=NOW, trace_id="second")
        assert [item.record_id for item in active] == ["z"]
        assert active[0].lifecycle is Lifecycle.ACTIVE
        assert [item.candidate_id for item in observed] == ["a", "b", "z", "a"]
        assert store.read(tenant_id="tenant-a").record_id == "a"


def test_cursor_tenant_namespace_isolation_and_missing_boundary(tmp_path):
    path = tmp_path / "cursor.db"
    store = SQLitePromotionCursorStore(path)
    store.read(tenant_id="tenant-a")
    store.checkpoint(tenant_id="tenant-a", expected_generation=0, record_id="m")
    assert store.read(tenant_id="tenant-b").record_id is None
    assert SQLitePromotionCursorStore(path, namespace="other").read(tenant_id="tenant-a").record_id is None
    observed = []
    run = scheduler(object(), object(), store, observed.append, limit=1)
    run.evaluate((candidate("a"), candidate("z"), candidate("n", tenant="tenant-b")),
                 now=NOW, trace_id="run")
    assert [item.candidate_id for item in observed] == ["z"]


@pytest.mark.parametrize("mode", ["repository", "observer", "cancel-before", "cancel-during"])
def test_incomplete_evaluation_does_not_acknowledge_cursor(tmp_path, mode):
    store = SQLitePromotionCursorStore(tmp_path / "cursor.db")
    cancelled = [mode == "cancel-before"]
    class Repository:
        def get_by_id(self, *args, **kwargs):
            if mode == "repository":
                raise RuntimeError("storage unavailable")
            cancelled[0] = True
            return None
    def observe(item):
        if mode == "observer":
            raise RuntimeError("audit unavailable")
    run = scheduler(Repository(), object(), store, observe)
    values = (candidate("a", eligible=mode in {"repository", "cancel-during"}),)
    if mode in {"repository", "observer"}:
        with pytest.raises(RuntimeError, match="unavailable"):
            run.evaluate(values, now=NOW, trace_id="run", cancellation_check=lambda: cancelled[0])
    else:
        assert run.evaluate(values, now=NOW, trace_id="run", cancellation_check=lambda: cancelled[0]) == ()
    state = store.read(tenant_id="tenant-a")
    assert state.record_id is None and state.generation == 0


def test_expected_repository_rejection_checkpoints_after_observer(tmp_path, monkeypatch):
    authority = object()
    store = SQLitePromotionCursorStore(tmp_path / "cursor.db")
    observed = []
    def reject(*args, **kwargs):
        raise MemoryPromotionError("expected evidence refusal")
    monkeypatch.setattr("market_agent.workflow_memory_promotion.promote_candidate", reject)
    with SQLiteMemoryRepository(tmp_path / "memory.db", writer_authority=authority) as repo:
        value = seed(repo, authority)[-1]
        def observe(item):
            assert store.read(tenant_id="tenant-a").record_id is None
            observed.append(item)
        scheduler(repo, authority, store, observe).evaluate((value,), now=NOW, trace_id="run")
    assert observed[-1].reason_code == "repository_rejected"
    assert store.read(tenant_id="tenant-a").record_id == "z"


def test_stale_concurrent_scheduler_cannot_regress_cursor_or_continue_batch(tmp_path):
    path = tmp_path / "cursor.db"
    store = SQLitePromotionCursorStore(path)
    other_store = SQLitePromotionCursorStore(path)
    values = (candidate("a"), candidate("b"), candidate("c"))
    observed = []
    def interleave(item):
        observed.append(item)
        # This succeeds while the older scheduler evaluates: no SQL lock spans work.
        scheduler(object(), object(), other_store, lambda _: None, limit=3).evaluate(
            values, now=NOW, trace_id="concurrent")
    scheduler(object(), object(), store, interleave).evaluate(values, now=NOW, trace_id="older")
    assert [item.candidate_id for item in observed] == ["a"]
    state = store.read(tenant_id="tenant-a")
    assert state.record_id == "c" and state.generation == 3
    assert store.checkpoint(tenant_id="tenant-a", expected_generation=0, record_id="a") is None


def test_cursor_wrap_generation_fences_aba_and_requires_durable_sqlite(tmp_path):
    with pytest.raises(ValueError, match="durable"):
        SQLitePromotionCursorStore(":memory:")
    store = SQLitePromotionCursorStore(tmp_path / "cursor.db")
    store.read(tenant_id="tenant-a")
    store.checkpoint(tenant_id="tenant-a", expected_generation=0, record_id="a")
    store.checkpoint(tenant_id="tenant-a", expected_generation=1, record_id="z")
    store.checkpoint(tenant_id="tenant-a", expected_generation=2, record_id="a")
    assert store.checkpoint(tenant_id="tenant-a", expected_generation=1, record_id="b") is None
    assert store.read(tenant_id="tenant-a").record_id == "a"


def test_checkpoint_write_failure_surfaces_and_rolls_back(tmp_path):
    path = tmp_path / "cursor.db"
    store = SQLitePromotionCursorStore(path)
    store.read(tenant_id="tenant-a")
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TRIGGER reject_checkpoint BEFORE UPDATE ON market_agent_promotion_cursor "
            "BEGIN SELECT RAISE(ABORT, 'checkpoint unavailable'); END"
        )
    observed = []
    with pytest.raises(sqlite3.IntegrityError, match="checkpoint unavailable"):
        scheduler(object(), object(), store, observed.append).evaluate(
            (candidate("a"), candidate("b")), now=NOW, trace_id="run")
    assert [item.candidate_id for item in observed] == ["a"]
    state = store.read(tenant_id="tenant-a")
    assert state.record_id is None and state.generation == 0


def test_successful_activation_audit_failure_does_not_checkpoint(tmp_path):
    authority = object()
    store = SQLitePromotionCursorStore(tmp_path / "cursor.db")
    with SQLiteMemoryRepository(tmp_path / "memory.db", writer_authority=authority) as repo:
        value = seed(repo, authority)[-1]
        def broken_observer(item):
            assert item.status == "promoted"
            raise RuntimeError("audit unavailable")
        with pytest.raises(RuntimeError, match="audit unavailable"):
            scheduler(repo, authority, store, broken_observer).evaluate((value,), now=NOW, trace_id="run")
        assert repo.get_by_id("z", tenant_id="tenant-a").lifecycle is Lifecycle.ACTIVE
        assert store.read(tenant_id="tenant-a").record_id is None


class OfflinePostgresConnection:
    """Execute the portable SQL subset locally while recording PG wire statements."""
    def __init__(self, path, statements):
        self.connection = sqlite3.connect(path)
        self.statements = statements
    def cursor(self):
        connection = self
        class Cursor:
            def __init__(self):
                self.inner = connection.connection.cursor()
            def execute(self, sql, values=()):
                connection.statements.append((sql, values))
                self.inner.execute(sql.replace("%s", "?"), values)
                return self
            def fetchone(self):
                return self.inner.fetchone()
            @property
            def rowcount(self):
                return self.inner.rowcount
            def close(self):
                self.inner.close()
        return Cursor()
    def commit(self):
        self.connection.commit()
    def rollback(self):
        self.connection.rollback()
    def close(self):
        self.connection.close()


def test_postgres_explicit_migration_healthcheck_parameterized_cas_contract(tmp_path):
    statements = []
    store = PostgresPromotionCursorStore(
        lambda: OfflinePostgresConnection(tmp_path / "pg-contract.db", statements),
        namespace="promotion",
    )
    assert statements == []
    with pytest.raises(sqlite3.OperationalError):
        store.healthcheck()
    store.migrate()
    assert store.healthcheck() is True
    assert store.read(tenant_id="tenant-a").generation == 0
    assert store.checkpoint(tenant_id="tenant-a", expected_generation=0, record_id="z").generation == 1
    assert store.checkpoint(tenant_id="tenant-a", expected_generation=0, record_id="a") is None
    assert store.read(tenant_id="tenant-b").record_id is None
    updates = [(sql, values) for sql, values in statements if sql.startswith("UPDATE")]
    assert updates and all("namespace=%s AND tenant_id=%s AND generation=%s" in sql for sql, _ in updates)
    assert all("?" not in sql for sql, _ in statements)
    assert all("BEGIN IMMEDIATE" not in sql for sql, _ in statements)
