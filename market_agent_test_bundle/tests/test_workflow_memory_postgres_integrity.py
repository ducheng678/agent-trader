"""Shared integrity contracts plus offline PostgreSQL SQL boundary checks.

The DB-API harness executes translated SQL on SQLite, preserving transactions,
constraints and row counts. It does not establish live PostgreSQL locking behavior.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
import re
import sqlite3

import pytest

from market_agent.workflow_long_term_memory import (
    Lifecycle, MemoryConflictError, MemoryIntegrityError, MemoryPromotionError,
)
from market_agent.workflow_memory_postgres import (
    PostgresMemoryRepository, PostgresMemoryUnavailableError,
)
from market_agent.workflow_memory_sqlite import SQLiteMemoryRepository
from market_agent_test_bundle.tests import test_workflow_memory_storage as storage


class OfflinePostgres:
    def __init__(self, path):
        self.path = path
        self.statements = []
        self.fail_audit = None
        self.fail_head_cas = False
        self.rollbacks = 0

    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("BEGIN IMMEDIATE")
        return OfflineConnection(self, connection)

    def replace_record(self, record, *, tenant_id=None, record_id=None):
        body = PostgresMemoryRepository._body(record)
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "UPDATE governed_memory_records SET body=?,body_hash=?,lifecycle=? "
                "WHERE tenant_id=? AND record_id=?",
                (body, PostgresMemoryRepository._hash(body), record.lifecycle.value,
                 tenant_id or record.tenant_id, record_id or record.record_id),
            )

    def snapshot(self):
        with sqlite3.connect(self.path) as connection:
            return {
                table: connection.execute(f"SELECT * FROM {table}").fetchall()
                for table in ("governed_memory_records", "governed_memory_heads",
                              "governed_memory_idempotency", "governed_memory_audit")
            }


class OfflineConnection:
    def __init__(self, database, connection):
        self.database = database
        self.connection = connection

    def cursor(self):
        return OfflineCursor(self.database, self.connection.cursor())

    def commit(self):
        self.connection.commit()

    def rollback(self):
        self.database.rollbacks += 1
        self.connection.rollback()

    def close(self):
        self.connection.close()


class OfflineCursor:
    def __init__(self, database, cursor):
        self.database = database
        self.cursor = cursor
        self.forced_rowcount = None

    @property
    def rowcount(self):
        return self.cursor.rowcount if self.forced_rowcount is None else self.forced_rowcount

    def execute(self, operation, parameters=None):
        self.database.statements.append((operation, parameters))
        self.forced_rowcount = None
        if operation.startswith("INSERT INTO governed_memory_audit"):
            if self.database.fail_audit in ("all", parameters[2]):
                raise OSError("audit unavailable")
        if operation.startswith("UPDATE governed_memory_heads") and self.database.fail_head_cas:
            self.forced_rowcount = 0
            return self
        if operation.startswith("CREATE EXTENSION") or "USING ivfflat" in operation:
            return self
        if "pg_extension" in operation or "pg_advisory_xact_lock" in operation:
            self.cursor.execute("SELECT 1")
            return self
        if operation.startswith("ALTER TABLE") and "ADD COLUMN IF NOT EXISTS" in operation:
            match = re.search(r"ALTER TABLE (\w+) ADD COLUMN IF NOT EXISTS (\w+)", operation)
            columns = self.cursor.execute(f"PRAGMA table_info({match[1]})").fetchall()
            if any(column[1] == match[2] for column in columns):
                return self
            operation = operation.replace("ADD COLUMN IF NOT EXISTS", "ADD COLUMN")
        translated = operation.replace("%s", "?")
        translated = re.sub(r"::(?:jsonb|vector)", "", translated)
        translated = re.sub(r"\s+FOR (?:UPDATE|SHARE)\b", "", translated)
        translated = re.sub(r"vector\(\d+\)", "TEXT", translated)
        translated = translated.replace("BIGSERIAL PRIMARY KEY", "INTEGER PRIMARY KEY AUTOINCREMENT")
        adapted = tuple(value.isoformat() if isinstance(value, datetime) else value
                        for value in (parameters or ()))
        self.cursor.execute(translated, adapted)
        return self

    def fetchone(self):
        return self.cursor.fetchone()

    def fetchall(self):
        return self.cursor.fetchall()

    def close(self):
        self.cursor.close()


@pytest.fixture
def pg_repo(tmp_path):
    database = OfflinePostgres(tmp_path / "postgres-contract.db")
    authority = object()
    repository = PostgresMemoryRepository(database.connect, embedding_dimension=3,
                                          writer_authority=authority)
    repository.test_authority = authority
    repository.test_database = database
    repository.migrate()
    return repository


@pytest.fixture(params=["sqlite", "postgres"])
def repo(request, tmp_path, pg_repo):
    if request.param == "postgres":
        yield pg_repo
    else:
        authority = object()
        with SQLiteMemoryRepository(tmp_path / "sqlite-contract.db", writer_authority=authority) as repository:
            repository.test_authority = authority
            yield repository


# Reuse the exact existing API contracts; SQLite-specific corruption/trigger cases
# have their PostgreSQL boundary equivalents below.
_SHARED_CASES = [
    (storage.test_event_idempotency_does_not_duplicate_audit_truth, {}),
    (storage.test_knowledge_requires_existing_same_tenant_event_evidence, {}),
    (storage.test_activation_is_compare_and_set_and_has_separate_audit, {}),
    (storage.test_idempotency_key_cannot_be_rebound, {}),
    (storage.test_event_hash_is_canonical_immutable_and_unique_per_tenant, {}),
    (storage.test_copy_and_rehydration_cannot_bypass_validation, {}),
    (storage.test_circular_or_missing_provenance_is_denied, {}),
    (storage.test_revision_lineage_requires_current_head_and_cannot_replace_active_record, {}),
    (storage.test_decision_outcome_and_lesson_links_must_be_verified_and_same_tenant, {}),
    (storage.test_derived_event_cannot_corroborate_its_own_knowledge_lineage, {}),
    (storage.test_expired_event_cannot_verify_an_outcome, {}),
    (storage.test_external_ancestry_does_not_turn_model_claim_into_verification, {}),
    (storage.test_non_event_provenance_cannot_verify_outcome_from_decision_context, {}),
    (storage.test_outcome_for_another_decision_cannot_support_a_lesson, {}),
    (storage.test_outcome_cannot_predate_its_decision, {}),
    *[(storage.test_activation_requires_fresh_independent_or_verified_evidence, {"mode": mode})
      for mode in ("expired", "model", "single_source")],
    *[(storage.test_system_wrappers_cannot_verify_model_only_outcomes, {"depth": depth})
      for depth in (1, 3)],
    *[(storage.test_real_root_evidence_can_verify_through_multiple_system_derivations, {"source_kind": kind})
      for kind in ("external", "system")],
    *[(storage.test_promotion_counts_original_roots_not_system_wrapper_labels, {"independent": value})
      for value in (True, False)],
    *[(storage.test_lesson_rechecks_all_outcome_evidence_at_link_time, {"derived": value})
      for value in (True, False)],
]


@pytest.mark.parametrize("contract,arguments", [
    pytest.param(contract, arguments, id=contract.__name__ + str(arguments))
    for contract, arguments in _SHARED_CASES
])
def test_shared_storage_integrity_contract(repo, contract, arguments):
    contract(repo, **arguments)


def test_pg_second_revision_updates_head_and_stale_activation_conflicts(pg_repo):
    storage.evidence(pg_repo)
    pg_repo.propose_knowledge(storage.candidate(), **storage.write(pg_repo, "proposal"))
    newer = storage.candidate("knowledge-2", revision=2, lineage_ids=("knowledge-1",))
    pg_repo.propose_knowledge(newer, **storage.write(pg_repo, "proposal-2"))
    assert pg_repo.test_database.snapshot()["governed_memory_heads"] == [
        ("tenant-a", "rule-1", 2, "knowledge-2")]
    with pytest.raises(MemoryConflictError):
        pg_repo.activate_knowledge("knowledge-1", expected_revision=1, now=storage.NOW,
                                  **storage.write(pg_repo, "stale"))


def test_pg_competing_second_revisions_have_one_domain_conflict(pg_repo):
    storage.evidence(pg_repo)
    pg_repo.propose_knowledge(storage.candidate(), **storage.write(pg_repo, "proposal"))

    def propose(identifier):
        repository = PostgresMemoryRepository(pg_repo.test_database.connect, embedding_dimension=3,
                                              writer_authority=pg_repo.test_authority)
        try:
            return repository.propose_knowledge(
                storage.candidate(identifier, revision=2, lineage_ids=("knowledge-1",)),
                **storage.write(pg_repo, identifier))
        except MemoryConflictError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(propose, ("revision-2-a", "revision-2-b")))
    assert sum(isinstance(result, MemoryConflictError) for result in results) == 1
    assert len(pg_repo.test_database.snapshot()["governed_memory_heads"]) == 1


def test_pg_zero_row_head_cas_rolls_back_every_write(pg_repo):
    storage.evidence(pg_repo)
    pg_repo.propose_knowledge(storage.candidate(), **storage.write(pg_repo, "proposal"))
    before = pg_repo.test_database.snapshot()
    pg_repo.test_database.fail_head_cas = True
    with pytest.raises(MemoryConflictError):
        pg_repo.propose_knowledge(storage.candidate("knowledge-2", revision=2, lineage_ids=("knowledge-1",)),
                                 **storage.write(pg_repo, "proposal-2"))
    assert pg_repo.test_database.snapshot() == before
    sql = [sql for sql, _ in pg_repo.test_database.statements if sql.startswith("UPDATE governed_memory_heads")][-1]
    assert "revision = %s AND record_id = %s" in sql


@pytest.mark.parametrize("operation", ["propose_knowledge", "activate_knowledge", "supersede_knowledge"])
def test_pg_audit_failure_rolls_back_revision_and_all_side_effects(pg_repo, operation):
    storage.evidence(pg_repo)
    pg_repo.propose_knowledge(storage.candidate(), **storage.write(pg_repo, "proposal"))
    pg_repo.activate_knowledge("knowledge-1", expected_revision=1, now=storage.NOW,
                              **storage.write(pg_repo, "activate"))
    newer = storage.candidate("knowledge-2", revision=2, lineage_ids=("knowledge-1",))
    if operation != "propose_knowledge":
        pg_repo.propose_knowledge(newer, **storage.write(pg_repo, "proposal-2"))
    before = pg_repo.test_database.snapshot()
    pg_repo.test_database.fail_audit = operation

    def mutation():
        if operation == "propose_knowledge":
            return pg_repo.propose_knowledge(newer, **storage.write(pg_repo, "proposal-2"))
        return pg_repo.activate_knowledge("knowledge-2", expected_revision=2, now=storage.NOW,
                                         **storage.write(pg_repo, "activate-2"))

    with pytest.raises(PostgresMemoryUnavailableError):
        mutation()
    assert pg_repo.test_database.snapshot() == before
    pg_repo.test_database.fail_audit = None
    assert mutation().record_id == "knowledge-2"


@pytest.mark.parametrize("corruption", ["cycle", "model_root", "missing", "cross_tenant", "wrong_kind", "future"])
def test_pg_activation_revalidates_legacy_evidence_graph(pg_repo, corruption):
    root = storage.derived_event(pg_repo, "root", ())
    storage.derived_event(pg_repo, "wrapper", ("root",))
    storage.derived_event(pg_repo, "model-root", (), kind="model")
    pg_repo.append_decision(storage.decision(evidence_ids=("wrapper",)), **storage.write(pg_repo, "decision"))
    pg_repo.append_outcome(storage.outcome(evidence_ids=("wrapper",)), **storage.write(pg_repo, "outcome"))
    pg_repo.propose_knowledge(storage.candidate(evidence_ids=("wrapper",), outcome_id="outcome-1"),
                             **storage.write(pg_repo, "proposal"))
    if corruption in ("cycle", "model_root", "missing", "wrong_kind"):
        parent = {"cycle": "wrapper", "model_root": "model-root", "missing": "missing", "wrong_kind": "decision-1"}[corruption]
        changed = root.model_copy(update={"provenance": root.provenance.model_copy(update={"derived_from": (parent,)})})
    elif corruption == "cross_tenant":
        changed = root.model_copy(update={"tenant_id": "tenant-b"})
    else:
        changed = root.model_copy(update={"observed_at": storage.NOW + timedelta(seconds=1)})
    pg_repo.test_database.replace_record(changed, tenant_id="tenant-a")
    before = pg_repo.test_database.snapshot()
    with pytest.raises((MemoryPromotionError, MemoryIntegrityError)):
        pg_repo.activate_knowledge("knowledge-1", expected_revision=1, now=storage.NOW,
                                  **storage.write(pg_repo, "activate"))
    assert pg_repo.test_database.snapshot() == before


def test_pg_migration_upgrades_legacy_replays_and_is_repeatable(pg_repo):
    original = pg_repo.append_event(storage.event(), **storage.write(pg_repo, "event"))
    with sqlite3.connect(pg_repo.test_database.path) as connection:
        for column in ("kind", "result", "result_hash"):
            connection.execute(f"ALTER TABLE governed_memory_idempotency DROP COLUMN {column}")
    pg_repo.migrate()
    pg_repo.migrate()
    assert pg_repo.append_event(storage.event(), **storage.write(pg_repo, "event")) == original
    assert len(pg_repo.list_audit(tenant_id="tenant-a")) == 1


def test_pg_stored_checksum_corruption_is_a_domain_integrity_error(pg_repo):
    pg_repo.append_event(storage.event(), **storage.write(pg_repo, "event"))
    with sqlite3.connect(pg_repo.test_database.path) as connection:
        connection.execute("UPDATE governed_memory_records SET body_hash=?", ("f" * 64,))
    with pytest.raises(MemoryIntegrityError):
        pg_repo.get_by_id("event-1", tenant_id="tenant-a")
    with pytest.raises(MemoryIntegrityError):
        pg_repo.propose_knowledge(storage.candidate(evidence_ids=("event-1",)),
                                 **storage.write(pg_repo, "proposal"))


def test_pg_wrapped_unavailability_still_rolls_back(pg_repo, monkeypatch):
    before = pg_repo.test_database.snapshot()
    rollbacks = pg_repo.test_database.rollbacks

    def fail(*args):
        raise PostgresMemoryUnavailableError("audit unavailable")

    monkeypatch.setattr(pg_repo, "_audit", fail)
    with pytest.raises(PostgresMemoryUnavailableError):
        pg_repo.append_event(storage.event(), **storage.write(pg_repo, "event"))
    assert pg_repo.test_database.rollbacks == rollbacks + 1
    assert pg_repo.test_database.snapshot() == before


@pytest.mark.parametrize("defect", ["expired", "future_effective", "contradiction", "stale_revision"])
def test_activation_checks_candidate_state_before_writing(repo, defect):
    storage.evidence(repo)
    changes = {
        "expired": {"expires_at": storage.NOW + timedelta(seconds=1)},
        "future_effective": {"effective_at": storage.NOW + timedelta(seconds=3)},
        "contradiction": {"evidence_ids": ("event-1",), "contradicting_ids": ("event-2",)},
        "stale_revision": {},
    }[defect]
    candidate = storage.candidate(**changes)
    repo.propose_knowledge(candidate, **storage.write(repo, "proposal"))
    before = repo.list_audit(tenant_id="tenant-a")
    error = MemoryConflictError if defect == "stale_revision" else MemoryPromotionError
    with pytest.raises(error):
        repo.activate_knowledge("knowledge-1", expected_revision=2 if defect == "stale_revision" else 1,
                                now=storage.NOW + timedelta(seconds=2), **storage.write(repo, "activate"))
    assert repo.get_by_id("knowledge-1", tenant_id="tenant-a") == candidate
    assert repo.list_audit(tenant_id="tenant-a") == before


@pytest.mark.parametrize("defect", ["wrong_evidence_kind", "unverified_outcome", "unrelated_outcome", "wrong_lineage", "duplicate_finalization"])
def test_pg_rejects_invalid_links_without_partial_writes(pg_repo, defect):
    storage.evidence(pg_repo)
    pg_repo.append_decision(storage.decision(), **storage.write(pg_repo, "decision"))
    pg_repo.append_outcome(storage.outcome(verified=defect != "unverified_outcome"), **storage.write(pg_repo, "outcome"))
    candidate = storage.candidate()
    if defect == "wrong_evidence_kind":
        candidate = candidate.model_copy(update={"evidence_ids": ("decision-1",)})
    elif defect in ("unverified_outcome", "unrelated_outcome"):
        candidate = candidate.model_copy(update={"evidence_ids": ("event-2",), "outcome_id": "outcome-1"})
    elif defect == "wrong_lineage":
        pg_repo.propose_knowledge(storage.candidate("other", knowledge_id="other-rule"), **storage.write(pg_repo, "other"))
        candidate = candidate.model_copy(update={"revision": 2, "lineage_ids": ("other",)})
    elif defect == "duplicate_finalization":
        pg_repo.append_decision(storage.decision("provisional", status="provisional"), **storage.write(pg_repo, "provisional"))
        pg_repo.append_decision(storage.decision("final", supersedes_id="provisional"), **storage.write(pg_repo, "final"))
    if defect == "unrelated_outcome":
        pg_repo.propose_knowledge(candidate, **storage.write(pg_repo, "proposal"))
    before = pg_repo.test_database.snapshot()
    with pytest.raises((MemoryPromotionError, MemoryConflictError)):
        if defect == "unrelated_outcome":
            pg_repo.activate_knowledge("knowledge-1", expected_revision=1, now=storage.NOW,
                                      **storage.write(pg_repo, "activate"))
        elif defect == "duplicate_finalization":
            pg_repo.append_decision(storage.decision("duplicate", supersedes_id="provisional"), **storage.write(pg_repo, "duplicate"))
        else:
            pg_repo.propose_knowledge(candidate, **storage.write(pg_repo, "proposal"))
    assert pg_repo.test_database.snapshot() == before


def test_pg_replay_preserves_original_result_after_supersession(pg_repo):
    storage.evidence(pg_repo)
    proposed = pg_repo.propose_knowledge(storage.candidate(), **storage.write(pg_repo, "proposal"))
    active = pg_repo.activate_knowledge("knowledge-1", expected_revision=1, now=storage.NOW,
                                      **storage.write(pg_repo, "activate"))
    pg_repo.propose_knowledge(storage.candidate("knowledge-2", revision=2, lineage_ids=("knowledge-1",)),
                             **storage.write(pg_repo, "proposal-2"))
    pg_repo.activate_knowledge("knowledge-2", expected_revision=2, now=storage.NOW,
                              **storage.write(pg_repo, "activate-2"))
    before = pg_repo.test_database.snapshot()
    assert pg_repo.propose_knowledge(storage.candidate(), **storage.write(pg_repo, "proposal")) == proposed
    assert pg_repo.activate_knowledge("knowledge-1", expected_revision=1, now=storage.NOW,
                                      **storage.write(pg_repo, "activate")) == active
    assert pg_repo.test_database.snapshot() == before
