"""Exhaustive vector-candidate paging, using fake pages and offline PostgreSQL SQL."""
from __future__ import annotations

from datetime import datetime, timedelta
import json
import math
import re

import pytest

from market_agent.workflow_long_term_memory import Lifecycle, MemoryIntegrityError
from market_agent.workflow_memory_postgres import PostgresMemoryRepository
from market_agent import workflow_memory_retrieval as memory
from market_agent.workflow_memory_sqlite import SQLiteMemoryRepository
from market_agent_test_bundle.tests import test_workflow_memory_retrieval as existing
from market_agent_test_bundle.tests import test_workflow_memory_storage as storage
from market_agent_test_bundle.tests.test_workflow_memory_postgres_integrity import (
    OfflinePostgres, OfflineConnection, OfflineCursor,
)


def query(**changes):
    values = dict(embedding=(1.0, 0.0, 0.0), model_version="m1", vector_version="v1", top_k=1)
    values.update(changes)
    return existing.query(**values)


def candidate(identifier, **changes):
    values = dict(knowledge_id=identifier, lifecycle=Lifecycle.ACTIVE,
                  embedding=(1.0, 0.0, 0.0), model_version="m1", vector_version="v1")
    values.update(changes)
    return storage.candidate(identifier, **values)


class PagedRepository:
    def __init__(self, candidates, *, page_size=1):
        self.candidates = tuple(candidates)
        self.page_size = page_size
        evidence = (storage.event(), storage.event("event-2", source="independent", payload={"price": 101}),
                    storage.event("counter", source="counter", payload={"price": 99}))
        self.records = {record.record_id: record for record in (*evidence, *self.candidates)}
        self.page_calls = []
        self.legacy_calls = 0

    def vector_candidates(self, request):
        # Model the former API during RED; repaired retrieval must use pages.
        self.legacy_calls += 1
        return self.candidates[:request.top_k]

    def vector_candidate_page(self, request, *, cursor=None, limit):
        offset = int(cursor or 0)
        self.page_calls.append((cursor, limit))
        selected = self.candidates[offset:offset + min(limit, self.page_size)]
        end = offset + len(selected)
        exhausted = end >= len(self.candidates)
        return memory.VectorCandidatePage(records=selected, exhausted=exhausted,
                                           next_cursor=None if exhausted else str(end))

    def get_by_id(self, record_id, *, tenant_id):
        record = self.records.get(record_id)
        return record if record is not None and record.tenant_id == tenant_id else None

    def list_records(self, **kwargs):
        raise AssertionError("vector paging must not fall back to an unrestricted snapshot")


@pytest.mark.parametrize("defect", ["private", "scope", "applicability", "effective", "confidence", "similarity", "evidence"])
def test_later_valid_page_survives_earlier_rejection(defect):
    changes = {
        "private": {"visibility": "private"},
        "scope": {"scope": "other"},
        "applicability": {"applicability": ("ETH",)},
        "effective": {"effective_at": storage.NOW + timedelta(seconds=1)},
        "confidence": {"confidence": 0.1},
        "similarity": {"embedding": (0.0, 1.0, 0.0)},
        "evidence": {"evidence_ids": ("missing",)},
    }[defect]
    repository = PagedRepository((candidate("first", **changes), candidate("later")))
    result = memory.retrieve_memory(query(), repository)
    assert result.status == "hit" and [match.record_id for match in result.matches] == ["later"]
    assert len(repository.page_calls) == 2 and repository.legacy_calls == 0


def test_later_contradiction_blocks_earlier_higher_ranked_advice():
    repository = PagedRepository((candidate("advice", confidence=1.0),
                                  candidate("conflict", contradicting_ids=("counter",))))
    result = memory.retrieve_memory(query(), repository)
    assert result.status == "conflict" and result.matches[0].record_id == "conflict"
    assert memory.build_core_experience_summary(result, 3000).as_dynamic_context() == ""
    assert len(repository.page_calls) == 2


def test_unexhausted_scan_cap_never_returns_clear_advice():
    repository = PagedRepository((candidate("advice"), candidate("unseen-conflict", contradicting_ids=("counter",))))
    result = memory.retrieve_memory(query(max_candidates_scanned=1), repository)
    assert result.status == "failed" and result.matches == ()
    assert result.omissions == ("candidate_scan_limit",)
    assert len(repository.page_calls) == 1
    assert memory.build_core_experience_summary(result, 3000).as_dynamic_context() == ""


def test_source_exhausted_exactly_at_scan_cap_is_complete():
    repository = PagedRepository((candidate("advice"),))
    result = memory.retrieve_memory(query(max_candidates_scanned=1), repository)
    assert result.status == "hit" and result.matches[0].record_id == "advice"


def test_page_failure_after_an_eligible_candidate_discards_partial_advice():
    class Failing(PagedRepository):
        def vector_candidate_page(self, request, *, cursor=None, limit):
            if cursor is not None:
                raise OSError("later page unavailable")
            return super().vector_candidate_page(request, cursor=cursor, limit=limit)

    result = memory.retrieve_memory(query(), Failing((candidate("advice"), candidate("later"))))
    assert result.status == "failed" and result.omissions == ("retrieval_failed",)
    assert result.matches == ()


def test_duplicate_candidate_across_pages_fails_closed():
    repeated = candidate("duplicate")
    result = memory.retrieve_memory(query(), PagedRepository((repeated, repeated)))
    assert result.status == "failed" and result.matches == ()


def test_incomplete_evidence_hydration_cannot_hide_a_conflict(monkeypatch):
    monkeypatch.setattr(memory, "_MAX_EVIDENCE_RECORDS", 2, raising=False)
    result = memory.retrieve_memory(query(), PagedRepository((candidate("advice"),)))
    assert result.status == "failed" and result.omissions == ("evidence_scan_limit",)
    assert memory.build_core_experience_summary(result, 3000).as_dynamic_context() == ""


class VectorCursor(OfflineCursor):
    def execute(self, operation, parameters=None):
        self.database.raw_statements.append((operation, parameters))
        operation = re.sub(r"(\w+\.)?embedding <=> %s::vector", r"cosine_distance(\1embedding, %s)", operation)
        operation = re.sub(r"\((body|\w+\.body) ->> 'confidence'\)::double precision",
                           r"CAST(\1 ->> 'confidence' AS REAL)", operation)
        operation = re.sub(r"\((body|\w+\.body) ->> 'effective_at'\)::timestamptz",
                           r"normalized_timestamp(\1 ->> 'effective_at')", operation)
        return super().execute(operation, parameters)


class VectorConnection(OfflineConnection):
    def cursor(self):
        return VectorCursor(self.database, self.connection.cursor())


class VectorDatabase(OfflinePostgres):
    def __init__(self, path):
        super().__init__(path)
        self.raw_statements = []

    def connect(self):
        wrapped = super().connect()

        def distance(left, right):
            a, b = json.loads(left), json.loads(right)
            norms = math.hypot(*a) * math.hypot(*b)
            return 1 - sum(x * y for x, y in zip(a, b)) / norms if norms else None

        wrapped.connection.create_function("cosine_distance", 2, distance)
        wrapped.connection.create_function("normalized_timestamp", 1,
            lambda value: datetime.fromisoformat(value).isoformat() if value is not None else None)
        return VectorConnection(self, wrapped.connection)


@pytest.fixture
def pg_repo(tmp_path):
    database = VectorDatabase(tmp_path / "pg-vector.db")
    authority = object()
    repository = PostgresMemoryRepository(database.connect, embedding_dimension=3, writer_authority=authority)
    repository.test_authority, repository.test_database = authority, database
    repository.migrate()
    return repository


def seed(repository, identifier, **changes):
    values = dict(embedding=(1.0, 0.0, 0.0), model_version="m1", vector_version="v1")
    values.update(changes)
    return existing.seed_rule(repository, identifier, **values)


def test_pg_equal_distance_cursor_has_no_duplicates_or_top_k_truncation(pg_repo):
    for identifier in ("z", "a", "b"):
        seed(pg_repo, identifier)
    seed(pg_repo, "private", visibility="private")
    seed(pg_repo, "foreign", tenant="tenant-b")
    seed(pg_repo, "wrong-scope", scope="other")
    seed(pg_repo, "wrong-model", model_version="m2")
    seed(pg_repo, "expired", expires_at=storage.NOW + timedelta(seconds=1))
    request = query(now=storage.NOW + timedelta(seconds=2), top_k=1)
    first = pg_repo.vector_candidate_page(request, cursor=None, limit=1)
    second = pg_repo.vector_candidate_page(request, cursor=first.next_cursor, limit=1)
    third = pg_repo.vector_candidate_page(request, cursor=second.next_cursor, limit=1)
    assert [page.records[0].record_id for page in (first, second, third)] == ["a", "b", "z"]
    assert not first.exhausted and not second.exhausted
    assert third.exhausted and third.next_cursor is None
    sql = next(sql for sql, _ in pg_repo.test_database.raw_statements if "AS MATERIALIZED" in sql)
    assert "ORDER BY distance, record_id" in sql
    assert "visibility" in sql and "schema_version" in sql and "confidence" in sql
    assert "effective_at" in sql and "observed_at >=" in sql


def test_pg_cursor_is_bound_to_query_and_detects_midscan_mutation(pg_repo):
    seed(pg_repo, "a")
    seed(pg_repo, "b")
    first = pg_repo.vector_candidate_page(query(), cursor=None, limit=1)
    with pytest.raises((MemoryIntegrityError, ValueError)):
        pg_repo.vector_candidate_page(query(tenant_id="tenant-b"), cursor=first.next_cursor, limit=1)
    seed(pg_repo, "0-before-cursor")
    with pytest.raises((MemoryIntegrityError, ValueError)):
        pg_repo.vector_candidate_page(query(), cursor=first.next_cursor, limit=1)


def test_pg_later_conflict_is_found_and_scan_cap_fails_closed(pg_repo, monkeypatch):
    monkeypatch.setattr(memory, "_VECTOR_PAGE_SIZE", 1)
    seed(pg_repo, "advice")
    conflict = seed(pg_repo, "later")
    pg_repo.test_database.replace_record(conflict.model_copy(update={"contradicting_ids": ("advice-exchange",)}))
    complete = memory.retrieve_memory(query(), pg_repo)
    assert complete.status == "conflict"
    incomplete = memory.retrieve_memory(query(max_candidates_scanned=1), pg_repo)
    assert incomplete.status == "failed" and incomplete.omissions == ("candidate_scan_limit",)
    assert memory.build_core_experience_summary(incomplete, 3000).as_dynamic_context() == ""


def test_pg_ineligible_evidence_on_first_page_does_not_hide_later_match(pg_repo, monkeypatch):
    monkeypatch.setattr(memory, "_VECTOR_PAGE_SIZE", 1)
    invalid = seed(pg_repo, "a-invalid")
    seed(pg_repo, "z-valid")
    pg_repo.test_database.replace_record(invalid.model_copy(update={"evidence_ids": ("missing",)}))
    pg_repo.test_database.raw_statements.clear()
    result = memory.retrieve_memory(query(), pg_repo)
    assert result.status == "hit" and result.matches[0].record_id == "z-valid"
    assert "evidence_gap" in result.omissions
    assert sum("AS MATERIALIZED" in sql for sql, _ in pg_repo.test_database.raw_statements) == 2


@pytest.mark.parametrize("changes", [
    {"confidence": 0.1}, {"effective_at": storage.NOW + timedelta(seconds=1)},
    {"observed_at": storage.NOW - timedelta(days=2)},
])
def test_pg_cheap_eligibility_gates_run_before_page_limit(pg_repo, changes):
    # Seed through the public gates, then emulate valid legacy metadata changes.
    invalid = seed(pg_repo, "a-ineligible")
    changed = invalid.model_copy(update=changes)
    pg_repo.test_database.replace_record(changed)
    if "observed_at" in changes:
        with pg_repo.test_database.connect().connection as connection:
            connection.execute("UPDATE governed_memory_records SET observed_at=? WHERE record_id=?",
                               (changed.observed_at.isoformat(), changed.record_id))
    seed(pg_repo, "z-valid")
    page = pg_repo.vector_candidate_page(query(), cursor=None, limit=1)
    assert page.exhausted and [record.record_id for record in page.records] == ["z-valid"]


@pytest.mark.parametrize("limit", [0, -1, True, 1025])
def test_pg_page_limit_is_bounded(pg_repo, limit):
    with pytest.raises(ValueError):
        pg_repo.vector_candidate_page(query(), cursor=None, limit=limit)


def test_pg_database_outage_returns_failed_retrieval():
    def unavailable():
        raise OSError("database unavailable")

    repository = PostgresMemoryRepository(unavailable, embedding_dimension=3)
    result = memory.retrieve_memory(query(), repository)
    assert result.status == "failed" and result.omissions == ("retrieval_failed",)


def test_sqlite_keeps_full_snapshot_behavior_with_small_vector_scan_cap(tmp_path):
    authority = object()
    with SQLiteMemoryRepository(tmp_path / "snapshot.db", writer_authority=authority) as repository:
        repository.test_authority = authority
        existing.seed_rule(repository, "ineligible", applicability=("ETH",))
        existing.seed_rule(repository, "valid")
        result = memory.retrieve_memory(existing.query(top_k=1, max_candidates_scanned=1), repository)
        assert result.status == "hit" and result.matches[0].record_id == "valid"
