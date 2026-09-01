"""PostgreSQL/pgvector boundary for governed memory; database handles stay host-owned."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime
import hashlib
import json
from typing import Any, Callable, Iterator, Protocol, Unpack

from pydantic import AwareDatetime, TypeAdapter, ValidationError

from market_agent.workflow_long_term_memory import (
    RECORD_TYPES, DecisionLesson, DecisionRecord, EventRecord, KnowledgeRevision,
    Lifecycle, MemoryAudit, MemoryAuthorityError, MemoryConflictError,
    MemoryIntegrityError, MemoryPromotionError, MutationContext, OutcomeRecord,
    Record, WriteArguments, canonical_json, content_hash, validate_authority,
)
from market_agent.workflow_memory_retrieval import MemoryQuery


class PostgresMemoryUnavailableError(RuntimeError):
    pass


class DBAPICursor(Protocol):
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
    if type(embedding_dimension) is not int or not 1 <= embedding_dimension <= 16_000:
        raise ValueError("pgvector embedding dimension must be between 1 and 16000")
    return (
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"CREATE TABLE IF NOT EXISTS governed_memory_records (tenant_id TEXT NOT NULL, record_id TEXT NOT NULL, kind TEXT NOT NULL, body JSONB NOT NULL, body_hash TEXT NOT NULL, event_hash TEXT, embedding vector({embedding_dimension}), model_version TEXT NOT NULL, vector_version TEXT NOT NULL, scope TEXT NOT NULL, lifecycle TEXT NOT NULL, observed_at TIMESTAMPTZ NOT NULL, expires_at TIMESTAMPTZ, PRIMARY KEY (tenant_id, record_id), UNIQUE (tenant_id, event_hash))",
        "CREATE INDEX IF NOT EXISTS governed_memory_records_scope_idx ON governed_memory_records (tenant_id, scope, lifecycle, observed_at DESC)",
        "CREATE INDEX IF NOT EXISTS governed_memory_records_vector_idx ON governed_memory_records USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100)",
        "CREATE TABLE IF NOT EXISTS governed_memory_heads (tenant_id TEXT NOT NULL, knowledge_id TEXT NOT NULL, revision INTEGER NOT NULL, record_id TEXT NOT NULL, PRIMARY KEY (tenant_id, knowledge_id))",
        "CREATE TABLE IF NOT EXISTS governed_memory_idempotency (tenant_id TEXT NOT NULL, idempotency_key TEXT NOT NULL, request_hash TEXT NOT NULL, record_id TEXT NOT NULL, PRIMARY KEY (tenant_id, idempotency_key))",
        "CREATE TABLE IF NOT EXISTS governed_memory_audit (sequence BIGSERIAL PRIMARY KEY, tenant_id TEXT NOT NULL, trace_id TEXT NOT NULL, operation TEXT NOT NULL, record_id TEXT NOT NULL, idempotency_digest TEXT NOT NULL, record_hash TEXT NOT NULL)",
    )


def pgvector_literal(values: tuple[float, ...], *, dimension: int) -> str:
    if len(values) != dimension:
        raise ValueError("pgvector embedding does not match the configured dimension")
    return "[" + ",".join(repr(value) for value in values) + "]"


class PostgresMemoryRepository:
    """Transactional repository with tenant predicates on every data operation."""

    def __init__(self, connection_factory: ConnectionFactory, *, embedding_dimension: int,
                 writer_authority: object | None = None) -> None:
        if not callable(connection_factory):
            raise TypeError("a DB-API connection factory is required")
        postgres_memory_ddl(embedding_dimension)
        self._factory = connection_factory
        self._embedding_dimension = embedding_dimension
        self._authority = writer_authority

    def migrate(self) -> None:
        with self._transaction() as cursor:
            for statement in postgres_memory_ddl(self._embedding_dimension):
                cursor.execute(statement)
            cursor.execute("SELECT 1 FROM pg_extension WHERE extname = %s", ("vector",))
            if cursor.fetchone() is None:
                raise PostgresMemoryUnavailableError("pgvector extension is unavailable")

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
            cursor.execute("SELECT kind, body, body_hash FROM governed_memory_records WHERE tenant_id = %s AND record_id = %s", (tenant_id, record_id))
            row = cursor.fetchone()
        return None if row is None else self._rehydrate(*row)

    def list_records(self, *, tenant_id: str) -> tuple[Record, ...]:
        with self._cursor() as cursor:
            cursor.execute("SELECT kind, body, body_hash FROM governed_memory_records WHERE tenant_id = %s ORDER BY record_id", (tenant_id,))
            rows = cursor.fetchall()
        return tuple(self._rehydrate(*row) for row in rows)

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
        with self._transaction() as cursor:
            cursor.execute("SELECT kind, body, body_hash FROM governed_memory_records WHERE tenant_id = %s AND record_id = %s FOR UPDATE", (ctx.tenant_id, record_id))
            row = cursor.fetchone()
            if row is None:
                raise MemoryPromotionError("knowledge must exist in the same tenant")
            record = self._rehydrate(*row)
            if not isinstance(record, KnowledgeRevision) or record.revision != expected_revision or record.lifecycle is not Lifecycle.PROPOSED:
                raise MemoryConflictError("stale knowledge activation revision or state")
            if record.effective_at > now or (record.expires_at is not None and record.expires_at <= now) or record.contradicting_ids:
                raise MemoryPromotionError("knowledge is not eligible for activation")
            cursor.execute("SELECT revision, record_id FROM governed_memory_heads WHERE tenant_id = %s AND knowledge_id = %s FOR UPDATE", (ctx.tenant_id, record.knowledge_id))
            head = cursor.fetchone()
            if head is None or tuple(head) != (expected_revision, record_id):
                raise MemoryConflictError("knowledge head revision is stale")
            active = record.model_copy(update={"lifecycle": Lifecycle.ACTIVE})
            body = self._body(active)
            cursor.execute("UPDATE governed_memory_records SET body = %s::jsonb, body_hash = %s, lifecycle = %s WHERE tenant_id = %s AND record_id = %s", (body, self._hash(body), active.lifecycle.value, ctx.tenant_id, record_id))
            self._audit(cursor, active, "activate_knowledge", ctx)
        return active

    def vector_candidates(self, query: MemoryQuery) -> tuple[Record, ...]:
        query = MemoryQuery.model_validate(query)
        if not query.embedding:
            return ()
        try:
            vector = pgvector_literal(query.embedding, dimension=self._embedding_dimension)
            with self._cursor() as cursor:
                cursor.execute("SELECT kind, body, body_hash FROM governed_memory_records WHERE tenant_id = %s AND scope = %s AND lifecycle = %s AND model_version = %s AND vector_version = %s AND (expires_at IS NULL OR expires_at > %s) AND embedding IS NOT NULL ORDER BY embedding <=> %s::vector LIMIT %s", (query.tenant_id, query.scope, Lifecycle.ACTIVE.value, query.model_version, query.vector_version, query.now, vector, query.top_k))
                rows = cursor.fetchall()
            return tuple(self._rehydrate(*row) for row in rows)
        except Exception as error:
            raise PostgresMemoryUnavailableError("pgvector candidate retrieval is unavailable") from error

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
            cursor.execute("SELECT request_hash, record_id FROM governed_memory_idempotency WHERE tenant_id = %s AND idempotency_key = %s FOR UPDATE", (ctx.tenant_id, ctx.idempotency_key))
            replay = cursor.fetchone()
            if replay is not None:
                if replay[0] != request_hash:
                    raise MemoryConflictError("idempotency key already binds a different request")
                cursor.execute("SELECT kind, body, body_hash FROM governed_memory_records WHERE tenant_id = %s AND record_id = %s", (ctx.tenant_id, replay[1]))
                return self._rehydrate(*cursor.fetchone())
            if isinstance(record, KnowledgeRevision):
                cursor.execute("SELECT revision, record_id FROM governed_memory_heads WHERE tenant_id = %s AND knowledge_id = %s FOR UPDATE", (ctx.tenant_id, record.knowledge_id))
                head = cursor.fetchone()
                if (head is None and (record.revision != 1 or record.lineage_ids)) or (head is not None and (record.revision != head[0] + 1 or head[1] not in record.lineage_ids)):
                    raise MemoryConflictError("knowledge revision does not extend its tenant head")
            embedding = pgvector_literal(record.embedding, dimension=self._embedding_dimension) if record.embedding else None
            cursor.execute("INSERT INTO governed_memory_records (tenant_id, record_id, kind, body, body_hash, event_hash, embedding, model_version, vector_version, scope, lifecycle, observed_at, expires_at) VALUES (%s,%s,%s,%s::jsonb,%s,%s,%s::vector,%s,%s,%s,%s,%s,%s)", (record.tenant_id, record.record_id, type(record).__name__, body, self._hash(body), record.payload_hash if isinstance(record, EventRecord) else None, embedding, record.model_version, record.vector_version, record.scope, record.lifecycle.value, record.observed_at, record.expires_at))
            if isinstance(record, KnowledgeRevision):
                cursor.execute("INSERT INTO governed_memory_heads (tenant_id, knowledge_id, revision, record_id) VALUES (%s,%s,%s,%s)", (record.tenant_id, record.knowledge_id, record.revision, record.record_id))
            cursor.execute("INSERT INTO governed_memory_idempotency (tenant_id, idempotency_key, request_hash, record_id) VALUES (%s,%s,%s,%s)", (ctx.tenant_id, ctx.idempotency_key, request_hash, record.record_id))
            self._audit(cursor, record, operation, ctx)
        return record

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
        except PostgresMemoryUnavailableError:
            raise
        except Exception as error:
            if connection is not None:
                connection.rollback()
            if isinstance(error, (MemoryAuthorityError, MemoryConflictError, MemoryIntegrityError, MemoryPromotionError)):
                raise
            raise PostgresMemoryUnavailableError("PostgreSQL memory transaction failed") from error
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()
