"""PostgreSQL/pgvector boundary for governed memory; database handles stay host-owned."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from typing import Any, Callable, Iterator, Protocol, Unpack

from pydantic import AwareDatetime, Field, TypeAdapter, ValidationError

from market_agent.workflow_contracts import Digest, NonNegativeInt, ShortText
from market_agent.workflow_long_term_memory import (
    RECORD_TYPES, DecisionLesson, DecisionRecord, EventRecord, KnowledgeRevision,
    Lifecycle, MemoryAudit, MemoryAuthorityError, MemoryConflictError,
    MemoryIntegrityError, MemoryPromotionError, MutationContext, OutcomeRecord,
    MemoryContract, Record, WriteArguments, canonical_json, content_hash, validate_authority,
)
from market_agent.workflow_memory_retrieval import MemoryQuery, VectorCandidatePage
from market_agent.workflow_memory_lifecycle import (
    CleanupTask, LifecycleEntry, LifecycleLimits, LifecyclePlan, LifecyclePolicy,
    LifecycleResult, LifecycleScope, build_lifecycle_plan,
)


class PostgresMemoryUnavailableError(RuntimeError):
    pass


class _VectorCursor(MemoryContract):
    query_hash: Digest
    audit_revision: NonNegativeInt
    distance: float = Field(allow_inf_nan=False)
    record_id: ShortText


class DBAPICursor(Protocol):
    rowcount: int

    def execute(self, operation: str, parameters: object = None) -> object: ...
    def fetchone(self) -> object: ...
    def fetchall(self) -> object: ...
    def close(self) -> object: ...


class DBAPIConnection(Protocol):
    def cursor(self) -> DBAPICursor: ...
    def commit(self) -> object: ...
    def rollback(self) -> object: ...
    def close(self) -> object: ...


ConnectionFactory = Callable[[], DBAPIConnection]


def postgres_memory_ddl(embedding_dimension: int) -> tuple[str, ...]:
    if type(embedding_dimension) is not int or not 1 <= embedding_dimension <= 2_000:
        raise ValueError("ivfflat pgvector embedding dimension must be between 1 and 2000")
    return (
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"CREATE TABLE IF NOT EXISTS governed_memory_records (tenant_id TEXT NOT NULL, record_id TEXT NOT NULL, kind TEXT NOT NULL, body JSONB NOT NULL, body_hash TEXT NOT NULL, event_hash TEXT, embedding vector({embedding_dimension}), model_version TEXT NOT NULL, vector_version TEXT NOT NULL, scope TEXT NOT NULL, lifecycle TEXT NOT NULL, observed_at TIMESTAMPTZ NOT NULL, expires_at TIMESTAMPTZ, PRIMARY KEY (tenant_id, record_id), UNIQUE (tenant_id, event_hash))",
        "CREATE INDEX IF NOT EXISTS governed_memory_records_scope_idx ON governed_memory_records (tenant_id, scope, lifecycle, observed_at DESC)",
        "CREATE INDEX IF NOT EXISTS governed_memory_records_vector_idx ON governed_memory_records USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
        "CREATE TABLE IF NOT EXISTS governed_memory_heads (tenant_id TEXT NOT NULL, knowledge_id TEXT NOT NULL, revision INTEGER NOT NULL, record_id TEXT NOT NULL, PRIMARY KEY (tenant_id, knowledge_id))",
        "CREATE TABLE IF NOT EXISTS governed_memory_idempotency (tenant_id TEXT NOT NULL, idempotency_key TEXT NOT NULL, request_hash TEXT NOT NULL, record_id TEXT NOT NULL, kind TEXT, result JSONB, result_hash TEXT, PRIMARY KEY (tenant_id, idempotency_key))",
        "ALTER TABLE governed_memory_idempotency ADD COLUMN IF NOT EXISTS kind TEXT",
        "ALTER TABLE governed_memory_idempotency ADD COLUMN IF NOT EXISTS result JSONB",
        "ALTER TABLE governed_memory_idempotency ADD COLUMN IF NOT EXISTS result_hash TEXT",
        "CREATE TABLE IF NOT EXISTS governed_memory_audit (sequence BIGSERIAL PRIMARY KEY, tenant_id TEXT NOT NULL, trace_id TEXT NOT NULL, operation TEXT NOT NULL, record_id TEXT NOT NULL, idempotency_digest TEXT NOT NULL, record_hash TEXT NOT NULL)",
        "CREATE TABLE IF NOT EXISTS governed_memory_lifecycle_state (tenant_id TEXT NOT NULL, record_id TEXT NOT NULL, changed_at TIMESTAMPTZ NOT NULL, PRIMARY KEY (tenant_id, record_id), FOREIGN KEY (tenant_id, record_id) REFERENCES governed_memory_records (tenant_id, record_id))",
        "CREATE TABLE IF NOT EXISTS governed_memory_lifecycle_replay (tenant_id TEXT NOT NULL, idempotency_key TEXT NOT NULL, request_hash TEXT NOT NULL, PRIMARY KEY (tenant_id, idempotency_key))",
        "CREATE TABLE IF NOT EXISTS governed_memory_purged (tenant_id TEXT NOT NULL, record_id TEXT NOT NULL, event_hash TEXT, PRIMARY KEY (tenant_id, record_id), UNIQUE (tenant_id, event_hash))",
        "CREATE TABLE IF NOT EXISTS governed_memory_cleanup (tenant_id TEXT NOT NULL, task_id TEXT NOT NULL, body JSONB NOT NULL, body_hash TEXT NOT NULL, done INTEGER NOT NULL DEFAULT 0 CHECK (done IN (0,1)), PRIMARY KEY (tenant_id, task_id))",
        "CREATE TABLE IF NOT EXISTS governed_memory_cleanup_attempts (sequence BIGSERIAL PRIMARY KEY, tenant_id TEXT NOT NULL, task_id TEXT NOT NULL, FOREIGN KEY (tenant_id, task_id) REFERENCES governed_memory_cleanup (tenant_id, task_id))",
        "CREATE INDEX IF NOT EXISTS governed_memory_cleanup_last_attempt ON governed_memory_cleanup_attempts (tenant_id, task_id, sequence)",
    )


def pgvector_literal(values: tuple[float, ...], *, dimension: int) -> str:
    if len(values) != dimension:
        raise ValueError("pgvector embedding does not match the configured dimension")
    return "[" + ",".join(repr(value) for value in values) + "]"


class PostgresMemoryRepository:
    """Transactional repository with tenant predicates on every data operation."""

    def __init__(self, connection_factory: ConnectionFactory, *, embedding_dimension: int,
                 writer_authority: object | None = None,
                 clock: Callable[[], datetime] | None = None) -> None:
        if not callable(connection_factory):
            raise TypeError("a DB-API connection factory is required")
        postgres_memory_ddl(embedding_dimension)
        self._factory = connection_factory
        self._embedding_dimension = embedding_dimension
        self._authority = writer_authority
        self._clock = clock or (lambda: datetime.now(timezone.utc))

    def migrate(self) -> None:
        with self._transaction() as cursor:
            for statement in postgres_memory_ddl(self._embedding_dimension):
                cursor.execute(statement)
            cursor.execute("SELECT 1 FROM pg_extension WHERE extname = %s", ("vector",))
            if cursor.fetchone() is None:
                raise PostgresMemoryUnavailableError("pgvector extension is unavailable")

    def healthcheck(self) -> bool:
        """Run a real connection probe without mutating shared state."""

        with self._cursor() as cursor:
            cursor.execute("SELECT 1")
            row = cursor.fetchone()
        return bool(row and row[0] == 1)

    def validate_mutation(self, **context: Unpack[WriteArguments]) -> MutationContext:
        return validate_authority(self._authority, context.pop("authority"), **context)

    def append_event(self, record: EventRecord, **context: Unpack[WriteArguments]) -> EventRecord:
        return self._append(record, EventRecord, "append_event", **context)

    def propose_knowledge(self, record: KnowledgeRevision, **context: Unpack[WriteArguments]) -> KnowledgeRevision:
        return self._append(record, KnowledgeRevision, "propose_knowledge", **context)

    def append_decision(self, record: DecisionRecord, **context: Unpack[WriteArguments]) -> DecisionRecord:
        return self._append(record, DecisionRecord, "append_decision", **context)

    def append_outcome(self, record: OutcomeRecord, **context: Unpack[WriteArguments]) -> OutcomeRecord:
        return self._append(record, OutcomeRecord, "append_outcome", **context)

    def link_lesson(self, record: DecisionLesson, **context: Unpack[WriteArguments]) -> DecisionLesson:
        return self._append(record, DecisionLesson, "link_lesson", **context)

    def get_by_id(self, record_id: str, *, tenant_id: str) -> Record | None:
        with self._cursor() as cursor:
            return self._read(cursor, record_id, tenant_id)

    def list_records(self, *, tenant_id: str) -> tuple[Record, ...]:
        with self._cursor() as cursor:
            cursor.execute("SELECT tenant_id, record_id, kind, body, body_hash FROM governed_memory_records WHERE tenant_id = %s ORDER BY record_id", (tenant_id,))
            rows = cursor.fetchall()
        return tuple(self._row_record(row) for row in rows)

    def list_audit(self, *, tenant_id: str) -> tuple[MemoryAudit, ...]:
        with self._cursor() as cursor:
            cursor.execute("SELECT sequence, tenant_id, trace_id, operation, record_id, idempotency_digest, record_hash FROM governed_memory_audit WHERE tenant_id = %s ORDER BY sequence", (tenant_id,))
            rows = cursor.fetchall()
        return tuple(MemoryAudit.model_validate(dict(zip(("sequence", "tenant_id", "trace_id", "operation", "record_id", "idempotency_digest", "record_hash"), row))) for row in rows)

    def activate_knowledge(self, record_id: str, *, expected_revision: int, now: datetime, **context: Unpack[WriteArguments]) -> KnowledgeRevision:
        ctx = self.validate_mutation(**context)
        now = TypeAdapter(AwareDatetime).validate_python(now, strict=True)
        if type(expected_revision) is not int or expected_revision < 1:
            raise MemoryConflictError("activation requires a positive revision")
        request_hash = content_hash({"operation": "activate_knowledge", "trace_id": ctx.trace_id,
                                     "record_id": record_id, "revision": expected_revision, "now": now.isoformat()})
        with self._transaction() as cursor:
            self._lock_tenant(cursor, ctx.tenant_id)
            self._check_not_purged(cursor, ctx.tenant_id, record_id)
            replay = self._replay(cursor, ctx, request_hash)
            if replay is not None:
                return replay
            record = self._require(cursor, record_id, ctx.tenant_id, KnowledgeRevision)
            if record.revision != expected_revision or record.lifecycle is not Lifecycle.PROPOSED:
                raise MemoryConflictError("stale knowledge activation revision or state")
            cursor.execute("SELECT revision, record_id FROM governed_memory_heads WHERE tenant_id = %s AND knowledge_id = %s FOR UPDATE", (ctx.tenant_id, record.knowledge_id))
            head = cursor.fetchone()
            if head is None or tuple(head) != (expected_revision, record_id):
                raise MemoryConflictError("knowledge head revision is stale")
            self._check_links(cursor, record)
            if record.effective_at > now or (record.expires_at is not None and record.expires_at <= now):
                raise MemoryPromotionError("knowledge is not effective at activation time")
            if record.contradicting_ids:
                raise MemoryPromotionError("conflicting evidence prevents activation")
            sources = self._evidence_roots(cursor, record.evidence_ids, ctx.tenant_id, now)
            if record.outcome_id is not None:
                outcome = self._verified_outcome(cursor, record.outcome_id, ctx.tenant_id, now=now)
                if not set(outcome.evidence_ids) & set(record.evidence_ids):
                    raise MemoryPromotionError("verified outcome must support the candidate evidence")
            elif len(sources) < 2:
                raise MemoryPromotionError("activation needs independent corroboration or a verified outcome")
            cursor.execute("SELECT tenant_id, record_id, kind, body, body_hash FROM governed_memory_records WHERE tenant_id = %s AND kind = 'KnowledgeRevision' AND lifecycle = %s AND body ->> 'knowledge_id' = %s FOR UPDATE", (ctx.tenant_id, Lifecycle.ACTIVE.value, record.knowledge_id))
            transition_at = TypeAdapter(AwareDatetime).validate_python(self._clock(), strict=True)
            for row in cursor.fetchall():
                prior = self._row_record(row)
                archived = prior.model_copy(update={"lifecycle": Lifecycle.ARCHIVED})
                self._update_record(cursor, archived)
                self._set_lifecycle_time(cursor, archived, transition_at)
                self._audit(cursor, archived, "supersede_knowledge", ctx)
            active = record.model_copy(update={"lifecycle": Lifecycle.ACTIVE})
            self._update_record(cursor, active)
            self._audit(cursor, active, "activate_knowledge", ctx)
            self._remember(cursor, active, ctx, request_hash)
        return active

    def vector_candidate_page(self, query: MemoryQuery, *, cursor: str | None = None,
                              limit: int) -> VectorCandidatePage:
        query = MemoryQuery.model_validate(query)
        if type(limit) is not int or not 1 <= limit <= 1024:
            raise ValueError("candidate page limit must be between 1 and 1024")
        query_hash = content_hash(query.model_dump(mode="json"))
        continuation = None
        if cursor is not None:
            if not isinstance(cursor, str) or not 1 <= len(cursor) <= 2048:
                raise ValueError("invalid vector candidate cursor")
            continuation = _VectorCursor.model_validate_json(cursor)
            if continuation.query_hash != query_hash:
                raise MemoryIntegrityError("vector candidate cursor belongs to another query")
        if not query.embedding:
            return VectorCandidatePage(exhausted=True)
        vector = pgvector_literal(query.embedding, dimension=self._embedding_dimension)
        norm = math.hypot(*query.embedding)
        if not norm or not math.isfinite(norm):
            return VectorCandidatePage(exhausted=True)
        with self._transaction() as db_cursor:
            self._lock_tenant(db_cursor, query.tenant_id)
            db_cursor.execute("SELECT COALESCE(MAX(sequence), 0) FROM governed_memory_audit WHERE tenant_id = %s", (query.tenant_id,))
            revision = db_cursor.fetchone()[0]
            if continuation is not None and continuation.audit_revision != revision:
                raise MemoryIntegrityError("vector candidate source changed during paging")
            # Materialize the exact eligible relation before ordering. An ANN
            # index scan cannot establish exhaustion of all possible conflicts.
            # Applicability and similarity stay in Python to preserve Unicode
            # casefold and floating-point threshold semantics.
            statement = """WITH candidates AS MATERIALIZED (
                SELECT tenant_id, record_id, kind, body, body_hash,
                       embedding <=> %s::vector AS distance
                FROM governed_memory_records
                WHERE tenant_id = %s AND scope = %s AND lifecycle = %s
                  AND kind IN ('KnowledgeRevision', 'DecisionLesson')
                  AND model_version = %s AND vector_version = %s
                  AND COALESCE(body ->> 'visibility', 'tenant') = 'tenant'
                  AND COALESCE(body ->> 'schema_version', 'v1') = %s
                  AND observed_at >= %s AND observed_at <= %s
                  AND (expires_at IS NULL OR expires_at > %s)
                  AND (kind = 'DecisionLesson' OR (body ->> 'effective_at')::timestamptz <= %s)
                  AND (body ->> 'confidence')::double precision >= %s
                  AND embedding IS NOT NULL
            ) SELECT tenant_id, record_id, kind, body, body_hash, distance
              FROM candidates WHERE distance < %s"""
            parameters: tuple = (vector, query.tenant_id, query.scope, Lifecycle.ACTIVE.value,
                query.model_version, query.vector_version, query.memory_schema_version,
                query.now - timedelta(seconds=query.max_age_seconds), query.now, query.now,
                query.now, query.min_confidence, float("inf"))
            if continuation is not None:
                statement += " AND (distance, record_id) > (%s, %s)"
                parameters += (continuation.distance, continuation.record_id)
            statement += " ORDER BY distance, record_id LIMIT %s"
            db_cursor.execute(statement, (*parameters, limit + 1))
            rows = db_cursor.fetchall()
        exhausted = len(rows) <= limit
        selected = rows[:limit]
        records = tuple(self._row_record(row[:5]) for row in selected)
        next_cursor = None
        if not exhausted:
            last = selected[-1]
            next_cursor = _VectorCursor(query_hash=query_hash, audit_revision=revision,
                                        distance=float(last[5]), record_id=last[1]).model_dump_json()
        return VectorCandidatePage(records=records, next_cursor=next_cursor, exhausted=exhausted)

    def _append(self, record: Record, cls: type[Record], operation: str, **context: Unpack[WriteArguments]) -> Record:
        ctx = self.validate_mutation(**context)
        if type(record) is not cls:
            raise MemoryIntegrityError("record has the wrong type")
        record = cls.model_validate(record)
        if record.tenant_id != ctx.tenant_id:
            raise MemoryAuthorityError("mutation tenant does not match record")
        initial = Lifecycle.PROPOSED if cls is KnowledgeRevision else Lifecycle.ACTIVE
        if record.lifecycle is not initial:
            raise MemoryPromotionError("new records must use their initial lifecycle")
        body = self._body(record)
        request_hash = content_hash({"operation": operation, "trace_id": ctx.trace_id, "record": record.model_dump(mode="json")})
        with self._transaction() as cursor:
            self._lock_tenant(cursor, ctx.tenant_id)
            self._check_not_purged(cursor, ctx.tenant_id, record.record_id,
                                   record.payload_hash if isinstance(record, EventRecord) else None)
            if isinstance(record, EventRecord) and record.artifact is not None:
                cursor.execute("SELECT tenant_id, task_id, body, body_hash, done FROM governed_memory_cleanup WHERE tenant_id = %s", (ctx.tenant_id,))
                for row in cursor.fetchall():
                    task = self._cleanup_task(row)
                    if task.kind == "artifact" and task.artifact.sha256 == record.artifact.sha256:
                        raise MemoryConflictError("artifact address has entered cleanup")
            replay = self._replay(cursor, ctx, request_hash)
            if replay is not None:
                return replay
            if self._read(cursor, record.record_id, ctx.tenant_id, lock=True) is not None:
                raise MemoryConflictError("record identity is append-only")
            self._check_links(cursor, record)
            if isinstance(record, EventRecord):
                cursor.execute("SELECT record_id FROM governed_memory_records WHERE tenant_id = %s AND event_hash = %s FOR SHARE", (ctx.tenant_id, record.payload_hash))
                duplicate = cursor.fetchone()
                if duplicate is not None:
                    original = self._require(cursor, duplicate[0], ctx.tenant_id, EventRecord)
                    self._remember(cursor, original, ctx, request_hash)
                    return original
            if isinstance(record, KnowledgeRevision):
                cursor.execute("SELECT revision, record_id FROM governed_memory_heads WHERE tenant_id = %s AND knowledge_id = %s FOR UPDATE", (ctx.tenant_id, record.knowledge_id))
                head = cursor.fetchone()
                if (head is None and (record.revision != 1 or record.lineage_ids)) or (head is not None and (record.revision != head[0] + 1 or head[1] not in record.lineage_ids)):
                    raise MemoryConflictError("knowledge revision does not extend its tenant head")
            embedding = pgvector_literal(record.embedding, dimension=self._embedding_dimension) if record.embedding else None
            cursor.execute("INSERT INTO governed_memory_records (tenant_id, record_id, kind, body, body_hash, event_hash, embedding, model_version, vector_version, scope, lifecycle, observed_at, expires_at) VALUES (%s,%s,%s,%s::jsonb,%s,%s,%s::vector,%s,%s,%s,%s,%s,%s)", (record.tenant_id, record.record_id, type(record).__name__, body, self._hash(body), record.payload_hash if isinstance(record, EventRecord) else None, embedding, record.model_version, record.vector_version, record.scope, record.lifecycle.value, record.observed_at, record.expires_at))
            if isinstance(record, KnowledgeRevision):
                if head is None:
                    cursor.execute("INSERT INTO governed_memory_heads (tenant_id, knowledge_id, revision, record_id) VALUES (%s,%s,%s,%s) ON CONFLICT (tenant_id, knowledge_id) DO NOTHING", (record.tenant_id, record.knowledge_id, record.revision, record.record_id))
                else:
                    cursor.execute("UPDATE governed_memory_heads SET revision = %s, record_id = %s WHERE tenant_id = %s AND knowledge_id = %s AND revision = %s AND record_id = %s", (record.revision, record.record_id, record.tenant_id, record.knowledge_id, head[0], head[1]))
                if cursor.rowcount != 1:
                    raise MemoryConflictError("knowledge head changed during revision append")
            self._remember(cursor, record, ctx, request_hash)
            self._audit(cursor, record, operation, ctx)
        return record

    @staticmethod
    def _lock_tenant(cursor: DBAPICursor, tenant_id: str) -> None:
        # Covers absent rows as well as existing heads, replay keys and decision
        # finalizations. Every memory writer takes this lock before row locks.
        cursor.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                       ("governed-memory:" + tenant_id,))

    def _read(self, cursor: DBAPICursor, record_id: str, tenant_id: str, *, lock: bool = False) -> Record | None:
        cursor.execute("SELECT tenant_id, record_id, kind, body, body_hash FROM governed_memory_records WHERE tenant_id = %s AND record_id = %s" + (" FOR SHARE" if lock else ""), (tenant_id, record_id))
        row = cursor.fetchone()
        return None if row is None else self._row_record(row)

    def _row_record(self, row: object) -> Record:
        record = self._rehydrate(*row[2:])
        if (record.tenant_id, record.record_id) != tuple(row[:2]):
            raise MemoryIntegrityError("stored record identity mismatch")
        return record

    def _require(self, cursor: DBAPICursor, record_id: str, tenant_id: str, cls: Any) -> Record:
        record = self._read(cursor, record_id, tenant_id, lock=True)
        if not isinstance(record, cls):
            raise MemoryPromotionError("evidence must exist in the same tenant with the required type")
        return record

    @staticmethod
    def _references(record: Record) -> tuple[str, ...]:
        links = list(getattr(record, "evidence_ids", ()))
        links.extend(getattr(record, "lineage_ids", ()))
        links.extend(getattr(record, "contradicting_ids", ()))
        if isinstance(record, EventRecord):
            links.extend(record.provenance.derived_from)
        links.extend(value for name in ("decision_id", "outcome_id", "supersedes_id")
                     if (value := getattr(record, name, None)) is not None)
        return tuple(sorted(set(links)))

    def _check_links(self, cursor: DBAPICursor, record: Record) -> None:
        # These gates mirror SQLiteMemoryRepository; the shared storage contract
        # suite exercises both adapters to prevent drift.
        for ref in self._references(record):
            self._require(cursor, ref, record.tenant_id, tuple(RECORD_TYPES.values()))
        if isinstance(record, (KnowledgeRevision, OutcomeRecord, DecisionLesson)):
            for ref in record.evidence_ids:
                self._require(cursor, ref, record.tenant_id, EventRecord)
        if isinstance(record, KnowledgeRevision):
            self._check_evidence_ancestry(cursor, record)
            for ref in record.lineage_ids:
                prior = self._require(cursor, ref, record.tenant_id, KnowledgeRevision)
                if prior.knowledge_id != record.knowledge_id or prior.revision >= record.revision:
                    raise MemoryPromotionError("knowledge lineage must precede this rule revision")
            if record.outcome_id is not None:
                self._verified_outcome(cursor, record.outcome_id, record.tenant_id)
        if isinstance(record, DecisionRecord):
            for ref in record.evidence_ids:
                evidence = self._require(cursor, ref, record.tenant_id, (EventRecord, KnowledgeRevision))
                if evidence.lifecycle is not Lifecycle.ACTIVE:
                    raise MemoryPromotionError("decision evidence must be active")
            if record.supersedes_id is not None:
                prior = self._require(cursor, record.supersedes_id, record.tenant_id, DecisionRecord)
                if prior.status != "provisional" or record.status != "final":
                    raise MemoryPromotionError("only provisional decisions can be finalized")
                cursor.execute("SELECT record_id FROM governed_memory_records WHERE tenant_id = %s AND kind = 'DecisionRecord' AND body ->> 'supersedes_id' = %s FOR SHARE", (record.tenant_id, prior.record_id))
                if cursor.fetchone() is not None:
                    raise MemoryConflictError("decision already finalized")
        if isinstance(record, (OutcomeRecord, DecisionLesson)):
            decision = self._require(cursor, record.decision_id, record.tenant_id, DecisionRecord)
            if decision.status != "final":
                raise MemoryPromotionError("outcomes and lessons require a final decision")
            if decision.observed_at > record.observed_at:
                raise MemoryPromotionError("outcomes and lessons cannot predate their decision")
        if isinstance(record, OutcomeRecord) and record.verified:
            self._validate_outcome_evidence(cursor, record, record.observed_at)
        if isinstance(record, DecisionLesson):
            outcome = self._verified_outcome(cursor, record.outcome_id, record.tenant_id, now=record.observed_at)
            if outcome.decision_id != record.decision_id:
                raise MemoryPromotionError("lesson outcome must belong to its decision")
            self._evidence_roots(cursor, record.evidence_ids, record.tenant_id, record.observed_at)

    def _walk_ancestry(self, cursor: DBAPICursor, identifiers: tuple[str, ...], tenant_id: str, *,
                       events_only: bool = False) -> Iterator[Record]:
        """Visit parents before children and reject cycles without recursion."""
        visiting: set[str] = set()
        visited: set[str] = set()
        records: dict[str, Record] = {}
        pending = [(identifier, False) for identifier in reversed(identifiers)]
        while pending:
            identifier, expanded = pending.pop()
            if expanded:
                visiting.remove(identifier)
                visited.add(identifier)
                yield records[identifier]
                continue
            if identifier in visiting:
                raise MemoryPromotionError("circular evidence is forbidden")
            if identifier in visited:
                continue
            evidence = self._require(cursor, identifier, tenant_id,
                                     EventRecord if events_only else tuple(RECORD_TYPES.values()))
            records[identifier] = evidence
            visiting.add(identifier)
            pending.append((identifier, True))
            parents = evidence.provenance.derived_from if events_only else self._references(evidence)
            pending.extend((parent, False) for parent in reversed(parents))

    def _check_evidence_ancestry(self, cursor: DBAPICursor, candidate: KnowledgeRevision) -> None:
        identifiers = (*candidate.evidence_ids, *((candidate.outcome_id,) if candidate.outcome_id else ()))
        for evidence in self._walk_ancestry(cursor, identifiers, candidate.tenant_id):
            if isinstance(evidence, KnowledgeRevision) and evidence.knowledge_id == candidate.knowledge_id:
                raise MemoryPromotionError("a rule's descendants cannot corroborate that rule")

    def _evidence_roots(self, cursor: DBAPICursor, identifiers: tuple[str, ...], tenant_id: str, now: datetime) -> set[str]:
        roots: dict[str, set[str]] = {}
        for evidence in self._walk_ancestry(cursor, identifiers, tenant_id, events_only=True):
            self._fresh_evidence(evidence, now)
            provenance = evidence.provenance
            groups: set[str] = set()
            if provenance.source_kind != "model":
                if provenance.derived_from:
                    for parent in provenance.derived_from:
                        groups.update(roots[parent])
                else:
                    groups.add(provenance.independent_group)
            roots[evidence.record_id] = groups
        return set().union(*(roots[identifier] for identifier in identifiers))

    @staticmethod
    def _fresh_evidence(record: Record, now: datetime) -> None:
        if record.lifecycle is not Lifecycle.ACTIVE or record.observed_at > now or (record.expires_at is not None and record.expires_at <= now):
            raise MemoryPromotionError("evidence must be active, observed, and unexpired")

    def _validate_outcome_evidence(self, cursor: DBAPICursor, record: OutcomeRecord, now: datetime) -> None:
        self._fresh_evidence(record, now)
        if not self._evidence_roots(cursor, record.evidence_ids, record.tenant_id, now):
            raise MemoryPromotionError("independent non-model root evidence is required to verify an outcome")

    def _verified_outcome(self, cursor: DBAPICursor, record_id: str, tenant_id: str, *, now: datetime | None = None) -> OutcomeRecord:
        record = self._require(cursor, record_id, tenant_id, OutcomeRecord)
        if not record.verified:
            raise MemoryPromotionError("a verified outcome is required")
        decision = self._require(cursor, record.decision_id, tenant_id, DecisionRecord)
        if decision.status != "final" or decision.observed_at > record.observed_at:
            raise MemoryPromotionError("verified outcome requires an earlier final decision")
        self._validate_outcome_evidence(cursor, record, record.observed_at if now is None else now)
        return record

    def _replay(self, cursor: DBAPICursor, context: MutationContext, request_hash: str) -> Record | None:
        cursor.execute("SELECT request_hash, record_id, kind, result, result_hash FROM governed_memory_idempotency WHERE tenant_id = %s AND idempotency_key = %s FOR UPDATE", (context.tenant_id, context.idempotency_key))
        replay = cursor.fetchone()
        if replay is None:
            return None
        if replay[0] != request_hash:
            raise MemoryConflictError("idempotency key already binds a different request")
        if all(value is None for value in replay[2:]):
            # Legacy rows predate result snapshots. Keep their existing replay
            # behavior; all new writes preserve the original operation result.
            return self._require(cursor, replay[1], context.tenant_id, tuple(RECORD_TYPES.values()))
        record = self._rehydrate(*replay[2:])
        if record.tenant_id != context.tenant_id or record.record_id != replay[1]:
            raise MemoryIntegrityError("stored replay identity mismatch")
        return record

    def _remember(self, cursor: DBAPICursor, record: Record, context: MutationContext, request_hash: str) -> None:
        body = self._body(record)
        cursor.execute("INSERT INTO governed_memory_idempotency (tenant_id, idempotency_key, request_hash, record_id, kind, result, result_hash) VALUES (%s,%s,%s,%s,%s,%s::jsonb,%s)", (context.tenant_id, context.idempotency_key, request_hash, record.record_id, type(record).__name__, body, self._hash(body)))

    def _update_record(self, cursor: DBAPICursor, record: Record, *, expected_hash: str | None = None) -> None:
        body = self._body(record)
        parameters = (body, self._hash(body), record.lifecycle.value, record.tenant_id, record.record_id)
        predicate = "" if expected_hash is None else " AND body_hash = %s"
        cursor.execute("UPDATE governed_memory_records SET body = %s::jsonb, body_hash = %s, lifecycle = %s WHERE tenant_id = %s AND record_id = %s" + predicate,
                       parameters if expected_hash is None else (*parameters, expected_hash))
        if cursor.rowcount != 1:
            raise MemoryConflictError("stored memory changed during update")

    @staticmethod
    def _check_not_purged(cursor: DBAPICursor, tenant_id: str, record_id: str, event_hash: str | None = None) -> None:
        cursor.execute("SELECT 1 FROM governed_memory_purged WHERE tenant_id = %s AND (record_id = %s OR event_hash = %s)",
                       (tenant_id, record_id, event_hash))
        if cursor.fetchone() is not None:
            raise MemoryConflictError("purged memory cannot be resurrected by replay or reappend")

    @staticmethod
    def _set_lifecycle_time(cursor: DBAPICursor, record: Record, now: datetime) -> None:
        cursor.execute("INSERT INTO governed_memory_lifecycle_state (tenant_id, record_id, changed_at) VALUES (%s,%s,%s) ON CONFLICT (tenant_id, record_id) DO UPDATE SET changed_at = EXCLUDED.changed_at",
                       (record.tenant_id, record.record_id, now))

    def lifecycle_snapshot(self, scope: LifecycleScope) -> tuple[LifecycleEntry, ...]:
        scope = LifecycleScope.model_validate(scope)
        with self._cursor() as cursor:
            return self._lifecycle_snapshot(cursor, scope)

    def _lifecycle_snapshot(self, cursor: DBAPICursor, scope: LifecycleScope) -> tuple[LifecycleEntry, ...]:
        # A single statement reads records, transition times and head protection.
        # Derive references from every tenant record before applying the scope,
        # including old records that predate lifecycle support.
        cursor.execute("SELECT r.tenant_id, r.record_id, r.kind, r.body, r.body_hash, s.changed_at, EXISTS (SELECT 1 FROM governed_memory_heads h WHERE h.tenant_id = r.tenant_id AND h.record_id = r.record_id) FROM governed_memory_records r LEFT JOIN governed_memory_lifecycle_state s ON s.tenant_id = r.tenant_id AND s.record_id = r.record_id WHERE r.tenant_id = %s ORDER BY r.record_id", (scope.tenant_id,))
        rows = cursor.fetchall()
        records = [(self._row_record(row[:5]), row[5], bool(row[6])) for row in rows]
        references: dict[str, set[str]] = {}
        for record, _, _ in records:
            for target in self._references(record):
                references.setdefault(target, set()).add(record.record_id)
        return tuple(LifecycleEntry(record=record,
            changed_at=datetime.fromisoformat(changed_at) if isinstance(changed_at, str) else changed_at,
            referenced_by=tuple(sorted(references.get(record.record_id, ()))), is_knowledge_head=head)
            for record, changed_at, head in records
            if scope.scope is None or record.scope == scope.scope)

    def apply_lifecycle(self, plan: LifecyclePlan, policy: LifecyclePolicy, limits: LifecycleLimits,
                        **context: Unpack[WriteArguments]) -> LifecycleResult:
        ctx = self.validate_mutation(**context)
        plan, policy, limits = (LifecyclePlan.model_validate(plan), LifecyclePolicy.model_validate(policy),
                                LifecycleLimits.model_validate(limits))
        if plan.scope.tenant_id != ctx.tenant_id:
            raise MemoryAuthorityError("lifecycle tenant does not match mutation scope")
        if plan.policy_hash != content_hash(policy.model_dump(mode="json")):
            raise MemoryConflictError("lifecycle policy does not match plan")
        applied, skipped = [], []
        with self._transaction() as cursor:
            self._lock_tenant(cursor, ctx.tenant_id)
            applied_at = TypeAdapter(AwareDatetime).validate_python(self._clock(), strict=True)
            if plan.now > applied_at:
                raise MemoryConflictError("lifecycle plan is ahead of the trusted apply clock")
            entries = self._lifecycle_snapshot(cursor, LifecycleScope(tenant_id=ctx.tenant_id))
            current = build_lifecycle_plan(entries, plan.scope, applied_at, policy)
            eligible = {action.record_id: action for action in current.actions}
            by_id = {entry.record.record_id: entry for entry in entries}
            records = {entry.record.record_id: entry.record for entry in entries}
            examined = 0
            for action in plan.actions:
                if examined >= limits.max_actions:
                    break
                key = content_hash({"key": ctx.idempotency_key, "plan": plan.plan_hash, "record": action.record_id})
                request_hash = content_hash({"trace": ctx.trace_id, "plan": plan.plan_hash, "action": action.model_dump(mode="json")})
                cursor.execute("SELECT request_hash FROM governed_memory_lifecycle_replay WHERE tenant_id = %s AND idempotency_key = %s FOR UPDATE", (ctx.tenant_id, key))
                replay = cursor.fetchone()
                if replay is not None:
                    if replay[0] != request_hash:
                        raise MemoryConflictError("lifecycle replay context changed")
                    continue
                if eligible.get(action.record_id) != action:
                    skipped.append(action.record_id)
                    continue
                examined += 1
                record = records[action.record_id]
                action_ctx = ctx.model_copy(update={"idempotency_key": key})
                if action.kind == "purge":
                    entry = by_id[record.record_id]
                    if (record.lifecycle is not Lifecycle.TOMBSTONED or record.legal_hold
                            or entry.referenced_by or entry.is_knowledge_head):
                        skipped.append(action.record_id)
                        continue
                    if isinstance(record, EventRecord) and record.artifact is not None:
                        shared = any(isinstance(other, EventRecord) and other.record_id != record.record_id
                                     and other.artifact is not None and other.artifact.sha256 == record.artifact.sha256
                                     for other in records.values())
                        if not shared:
                            self._queue_cleanup(cursor, record, "artifact", action_ctx)
                    self._audit(cursor, record, "lifecycle_purge", action_ctx)
                    cursor.execute("INSERT INTO governed_memory_purged (tenant_id, record_id, event_hash) VALUES (%s,%s,%s)", (ctx.tenant_id, record.record_id, record.payload_hash if isinstance(record, EventRecord) else None))
                    cursor.execute("DELETE FROM governed_memory_lifecycle_state WHERE tenant_id = %s AND record_id = %s", (ctx.tenant_id, record.record_id))
                    cursor.execute("DELETE FROM governed_memory_idempotency WHERE tenant_id = %s AND record_id = %s", (ctx.tenant_id, record.record_id))
                    cursor.execute("DELETE FROM governed_memory_records WHERE tenant_id = %s AND record_id = %s AND body_hash = %s", (ctx.tenant_id, record.record_id, action.expected_hash))
                    if cursor.rowcount != 1:
                        raise MemoryConflictError("stored memory changed during purge")
                    del records[record.record_id]
                else:
                    target = Lifecycle.ARCHIVED if action.kind == "archive" else Lifecycle.TOMBSTONED
                    record = record.model_copy(update={"lifecycle": target})
                    self._update_record(cursor, record, expected_hash=action.expected_hash)
                    self._set_lifecycle_time(cursor, record, applied_at)
                    self._audit(cursor, record, "lifecycle_" + action.kind, action_ctx)
                    records[record.record_id] = record
                    if action.kind == "tombstone":
                        for kind in ("vector", "cache"):
                            self._queue_cleanup(cursor, record, kind, action_ctx)
                cursor.execute("INSERT INTO governed_memory_lifecycle_replay (tenant_id, idempotency_key, request_hash) VALUES (%s,%s,%s)", (ctx.tenant_id, key, request_hash))
                applied.append(action.record_id)
        return LifecycleResult(applied_ids=tuple(applied), skipped_ids=tuple(skipped))

    def _queue_cleanup(self, cursor: DBAPICursor, record: Record, kind: str, ctx: MutationContext) -> None:
        task = CleanupTask(task_id=content_hash({"tenant": record.tenant_id, "record": record.record_id, "kind": kind}),
            tenant_id=record.tenant_id, scope=record.scope, trace_id=ctx.trace_id, record_id=record.record_id,
            record_hash=content_hash(record.model_dump(mode="json")), kind=kind,
            artifact=record.artifact if kind == "artifact" else None)
        body = canonical_json(task.model_dump(mode="json"))
        cursor.execute("INSERT INTO governed_memory_cleanup (tenant_id, task_id, body, body_hash) VALUES (%s,%s,%s::jsonb,%s) ON CONFLICT (tenant_id, task_id) DO NOTHING",
                       (task.tenant_id, task.task_id, body, self._hash(body)))

    @classmethod
    def _cleanup_task(cls, row: object) -> CleanupTask:
        try:
            body = row[2] if isinstance(row[2], str) else canonical_json(row[2])
            if cls._hash(body) != row[3]:
                raise MemoryIntegrityError("cleanup task checksum mismatch")
            task = CleanupTask.model_validate_json(body)
            if (task.tenant_id, task.task_id) != tuple(row[:2]):
                raise MemoryIntegrityError("cleanup task identity mismatch")
            return task
        except (ValidationError, ValueError, TypeError) as error:
            raise MemoryIntegrityError("invalid cleanup task") from error

    def list_cleanup(self, *, tenant_id: str, scope: str | None = None) -> tuple[CleanupTask, ...]:
        with self._cursor() as cursor:
            cursor.execute("SELECT c.tenant_id, c.task_id, c.body, c.body_hash, c.done FROM governed_memory_cleanup c WHERE c.tenant_id = %s AND c.done = 0 ORDER BY COALESCE((SELECT MAX(a.sequence) FROM governed_memory_cleanup_attempts a WHERE a.tenant_id = c.tenant_id AND a.task_id = c.task_id), 0), c.task_id", (tenant_id,))
            tasks = tuple(self._cleanup_task(row) for row in cursor.fetchall())
        return tuple(task for task in tasks if scope is None or task.scope == scope)

    def _registered_cleanup(self, cursor: DBAPICursor, task: CleanupTask, ctx: MutationContext) -> bool:
        cursor.execute("SELECT tenant_id, task_id, body, body_hash, done FROM governed_memory_cleanup WHERE tenant_id = %s AND task_id = %s FOR UPDATE", (ctx.tenant_id, task.task_id))
        row = cursor.fetchone()
        if row is None or self._cleanup_task(row) != task:
            raise MemoryIntegrityError("cleanup task is not registered")
        return bool(row[4])

    def begin_cleanup(self, task: CleanupTask, **context: Unpack[WriteArguments]) -> bool:
        """Persist fair retry progress before calling an external cleanup adapter."""
        ctx = self.validate_mutation(**context)
        task = CleanupTask.model_validate(task)
        if task.tenant_id != ctx.tenant_id or task.trace_id != ctx.trace_id or task.task_id != ctx.idempotency_key:
            raise MemoryAuthorityError("cleanup attempt must match its original mutation context")
        with self._transaction() as cursor:
            self._lock_tenant(cursor, ctx.tenant_id)
            if self._registered_cleanup(cursor, task, ctx):
                return False
            cursor.execute("INSERT INTO governed_memory_cleanup_attempts (tenant_id, task_id) VALUES (%s,%s) RETURNING sequence", (ctx.tenant_id, task.task_id))
            attempt = cursor.fetchone()[0]
            cursor.execute("INSERT INTO governed_memory_audit (tenant_id, trace_id, operation, record_id, idempotency_digest, record_hash) VALUES (%s,%s,%s,%s,%s,%s)",
                           (ctx.tenant_id, ctx.trace_id, "lifecycle_cleanup_attempt", task.record_id,
                            content_hash({"task": task.task_id, "attempt": attempt}), task.record_hash))
            return True

    def finish_cleanup(self, task: CleanupTask, **context: Unpack[WriteArguments]) -> None:
        ctx = self.validate_mutation(**context)
        task = CleanupTask.model_validate(task)
        if task.tenant_id != ctx.tenant_id or task.trace_id != ctx.trace_id or task.task_id != ctx.idempotency_key:
            raise MemoryAuthorityError("cleanup acknowledgement must match its original mutation context")
        with self._transaction() as cursor:
            self._lock_tenant(cursor, ctx.tenant_id)
            if self._registered_cleanup(cursor, task, ctx):
                return
            cursor.execute("UPDATE governed_memory_cleanup SET done = 1 WHERE tenant_id = %s AND task_id = %s AND done = 0", (ctx.tenant_id, task.task_id))
            if cursor.rowcount != 1:
                raise MemoryConflictError("cleanup task changed during acknowledgement")
            cursor.execute("INSERT INTO governed_memory_audit (tenant_id, trace_id, operation, record_id, idempotency_digest, record_hash) VALUES (%s,%s,%s,%s,%s,%s)",
                           (ctx.tenant_id, ctx.trace_id, "cleanup_" + task.kind, task.record_id,
                            self._hash(ctx.idempotency_key), task.record_hash))

    @staticmethod
    def _body(record: Record) -> str:
        return canonical_json(record.model_dump(mode="json"))

    @staticmethod
    def _hash(body: str) -> str:
        return hashlib.sha256(body.encode()).hexdigest()

    @classmethod
    def _rehydrate(cls, kind: str, body: object, digest: str) -> Record:
        try:
            rendered = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            if cls._hash(rendered) != digest:
                raise MemoryIntegrityError("stored memory hash mismatch")
            return RECORD_TYPES[kind].model_validate_json(rendered)
        except (KeyError, ValidationError, ValueError, TypeError) as error:
            raise MemoryIntegrityError("invalid stored memory") from error

    def _audit(self, cursor: DBAPICursor, record: Record, operation: str, context: MutationContext) -> None:
        cursor.execute("INSERT INTO governed_memory_audit (tenant_id, trace_id, operation, record_id, idempotency_digest, record_hash) VALUES (%s,%s,%s,%s,%s,%s)", (context.tenant_id, context.trace_id, operation, record.record_id, self._hash(context.idempotency_key), self._hash(self._body(record))))

    @contextmanager
    def _cursor(self) -> Iterator[DBAPICursor]:
        connection: DBAPIConnection | None = None
        cursor: DBAPICursor | None = None
        try:
            connection = self._factory()
            cursor = connection.cursor()
            yield cursor
        except Exception as error:
            if isinstance(error, (MemoryAuthorityError, MemoryConflictError, MemoryIntegrityError, MemoryPromotionError)):
                raise
            raise PostgresMemoryUnavailableError("PostgreSQL memory connection is unavailable") from error
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[DBAPICursor]:
        connection: DBAPIConnection | None = None
        cursor: DBAPICursor | None = None
        try:
            connection = self._factory()
            cursor = connection.cursor()
            yield cursor
            connection.commit()
        except Exception as error:
            if connection is not None:
                connection.rollback()
            if isinstance(error, (PostgresMemoryUnavailableError, MemoryAuthorityError, MemoryConflictError, MemoryIntegrityError, MemoryPromotionError)):
                raise
            raise PostgresMemoryUnavailableError("PostgreSQL memory transaction failed") from error
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()
