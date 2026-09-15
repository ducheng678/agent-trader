"""Persistent PostgreSQL/pgvector semantic response cache."""

from __future__ import annotations

import json
import math
from contextlib import contextmanager
from dataclasses import asdict
from typing import Iterable, Iterator, Mapping

from market_agent.workflow_memory_postgres import (
    ConnectionFactory,
    DBAPIConnection,
    DBAPICursor,
    pgvector_literal,
)
from market_agent.workflow_response_cache import (
    CacheMetadata,
    require_cache_safe,
    snapshot_safe_answers,
)
from market_agent.workflow_semantic_request_cache import SemanticCacheEntry


class PostgresSemanticCacheUnavailable(RuntimeError):
    pass


def postgres_semantic_cache_ddl(embedding_dimension: int) -> tuple[str, ...]:
    if type(embedding_dimension) is not int or not 1 <= embedding_dimension <= 2_000:
        raise ValueError("pgvector cache dimension must be between 1 and 2000")
    return (
        "CREATE EXTENSION IF NOT EXISTS vector",
        f"CREATE TABLE IF NOT EXISTS semantic_response_cache (entry_id TEXT PRIMARY KEY, tenant_scope TEXT NOT NULL, request_embedding vector({embedding_dimension}) NOT NULL, response JSONB NOT NULL, metadata JSONB NOT NULL, created_at DOUBLE PRECISION NOT NULL, expires_at DOUBLE PRECISION NOT NULL, vector_version TEXT NOT NULL, model_version TEXT NOT NULL)",
        "CREATE INDEX IF NOT EXISTS semantic_response_cache_scope_idx ON semantic_response_cache (tenant_scope, expires_at, vector_version, model_version)",
        "CREATE INDEX IF NOT EXISTS semantic_response_cache_vector_idx ON semantic_response_cache USING ivfflat (request_embedding vector_cosine_ops) WITH (lists = 100)",
    )


class PostgresSemanticRequestCache:
    """Apply strict >0.95 similarity and immutable compatibility gates in SQL and Python."""

    def __init__(
        self,
        connection_factory: ConnectionFactory,
        *,
        embedding_dimension: int,
        safe_answers: Mapping[str, Iterable[str]] | None = None,
    ) -> None:
        if not callable(connection_factory):
            raise TypeError("a DB-API connection factory is required")
        postgres_semantic_cache_ddl(embedding_dimension)
        self._factory = connection_factory
        self._dimension = embedding_dimension
        self._safe_answers = snapshot_safe_answers(safe_answers)

    def migrate(self) -> None:
        with self._transaction() as cursor:
            for statement in postgres_semantic_cache_ddl(self._dimension):
                cursor.execute(statement)

    def put(self, entry: SemanticCacheEntry) -> SemanticCacheEntry:
        entry = SemanticCacheEntry(
            entry_id=entry.entry_id,
            request_vector=entry.request_vector,
            response=dict(entry.response),
            metadata=entry.metadata,
            created_at=entry.created_at,
            vector_version=entry.vector_version,
            model_version=entry.model_version,
        )
        require_cache_safe(entry.metadata, entry.response, self._safe_answers)
        if entry.metadata.expires_at <= entry.created_at:
            raise ValueError("semantic cache entries must expire after creation")
        vector = pgvector_literal(entry.request_vector, dimension=self._dimension)
        metadata = json.dumps(asdict(entry.metadata), ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False)
        response = json.dumps(dict(entry.response), ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False)
        with self._transaction() as cursor:
            cursor.execute(
                "INSERT INTO semantic_response_cache (entry_id,tenant_scope,request_embedding,response,metadata,created_at,expires_at,vector_version,model_version) VALUES (%s,%s,%s::vector,%s::jsonb,%s::jsonb,%s,%s,%s,%s) ON CONFLICT (entry_id) DO UPDATE SET request_embedding=EXCLUDED.request_embedding,response=EXCLUDED.response,metadata=EXCLUDED.metadata,created_at=EXCLUDED.created_at,expires_at=EXCLUDED.expires_at,vector_version=EXCLUDED.vector_version,model_version=EXCLUDED.model_version WHERE semantic_response_cache.tenant_scope=EXCLUDED.tenant_scope",
                (entry.entry_id, entry.metadata.tenant_scope, vector, response, metadata,
                 entry.created_at, entry.metadata.expires_at, entry.vector_version, entry.model_version),
            )
        return entry

    store = put

    def lookup(self, query: tuple[float, ...], metadata: CacheMetadata, now: float) -> SemanticCacheEntry | None:
        if not math.isfinite(now) or metadata.vector_version is None or metadata.model_version is None:
            raise ValueError("semantic lookup requires finite time and version metadata")
        values = tuple(float(value) for value in query)
        if not values or not all(math.isfinite(value) for value in values) or not any(values):
            raise ValueError("semantic query vector must contain finite nonzero values")
        vector = pgvector_literal(values, dimension=self._dimension)
        expected = json.dumps(asdict(metadata), ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"), allow_nan=False)
        with self._cursor() as cursor:
            cursor.execute(
                "SELECT entry_id,request_embedding::text,response,metadata,created_at,vector_version,model_version FROM semantic_response_cache WHERE tenant_scope=%s AND metadata=%s::jsonb AND expires_at>%s AND vector_version=%s AND model_version=%s AND 1-(request_embedding <=> %s::vector)>0.95 ORDER BY request_embedding <=> %s::vector,created_at,entry_id LIMIT 1",
                (metadata.tenant_scope, expected, now, metadata.vector_version,
                 metadata.model_version, vector, vector),
            )
            row = cursor.fetchone()
        if row is None:
            return None
        stored_metadata = CacheMetadata(**(row[3] if isinstance(row[3], dict) else json.loads(row[3])))
        response = row[2] if isinstance(row[2], dict) else json.loads(row[2])
        require_cache_safe(stored_metadata, response, self._safe_answers)
        stored_vector = tuple(float(value) for value in str(row[1]).strip("[]").split(","))
        return SemanticCacheEntry(
            entry_id=str(row[0]), request_vector=stored_vector, response=response,
            metadata=stored_metadata, created_at=float(row[4]),
            vector_version=str(row[5]), model_version=str(row[6]),
        )

    def cleanup(self, *, now: float) -> int:
        if not math.isfinite(now):
            raise ValueError("cache cleanup time must be finite")
        with self._transaction() as cursor:
            cursor.execute("DELETE FROM semantic_response_cache WHERE expires_at<=%s", (now,))
            return max(0, int(getattr(cursor, "rowcount", 0)))

    @contextmanager
    def _cursor(self) -> Iterator[DBAPICursor]:
        connection: DBAPIConnection | None = None
        cursor: DBAPICursor | None = None
        try:
            connection = self._factory()
            cursor = connection.cursor()
            yield cursor
        except Exception as error:
            raise PostgresSemanticCacheUnavailable("semantic cache query failed") from error
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
            raise PostgresSemanticCacheUnavailable("semantic cache transaction failed") from error
        finally:
            if cursor is not None:
                cursor.close()
            if connection is not None:
                connection.close()
