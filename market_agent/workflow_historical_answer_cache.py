from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
import json
import math
import time
from typing import Any, Protocol


class HistoricalCacheError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class HistoricalAnswerMetadata:
    tenant_scope: str
    model_id: str
    model_version: str
    embedding_model: str
    embedding_version: str
    prompt_release_digest: str
    output_schema_digest: str
    safety_policy_version: str
    locale: str
    context_fingerprint: str
    knowledge_fingerprint: str
    evidence_references: tuple[str, ...]
    expires_at: float
    category: str = "informational"
    invalidation_reason: str | None = None

    def __post_init__(self) -> None:
        fields = (
            self.tenant_scope, self.model_id, self.model_version,
            self.embedding_model, self.embedding_version,
            self.prompt_release_digest, self.output_schema_digest,
            self.safety_policy_version, self.locale, self.context_fingerprint,
            self.knowledge_fingerprint, self.category,
        )
        if not all(type(item) is str and item.strip() for item in fields):
            raise HistoricalCacheError("historical cache metadata requires compact strings")
        if not math.isfinite(self.expires_at):
            raise HistoricalCacheError("historical cache expiry must be finite")
        if self.category != "informational":
            raise HistoricalCacheError("only safe informational answers are cacheable")
        if not all(type(item) is str and item.strip() for item in self.evidence_references):
            raise HistoricalCacheError("evidence references must be compact strings")

    def compatible_with(self, other: HistoricalAnswerMetadata) -> bool:
        return (
            self.tenant_scope, self.model_id, self.model_version,
            self.embedding_model, self.embedding_version,
            self.prompt_release_digest, self.output_schema_digest,
            self.safety_policy_version, self.locale,
            self.context_fingerprint, self.knowledge_fingerprint, self.category,
        ) == (
            other.tenant_scope, other.model_id, other.model_version,
            other.embedding_model, other.embedding_version,
            other.prompt_release_digest, other.output_schema_digest,
            other.safety_policy_version, other.locale,
            other.context_fingerprint, other.knowledge_fingerprint, other.category,
        )


@dataclass(frozen=True, slots=True)
class HistoricalAnswerRecord:
    entry_id: str
    request_text: str
    request_vector: tuple[float, ...]
    response: Mapping[str, object]
    request_timestamp: float
    response_timestamp: float
    metadata: HistoricalAnswerMetadata

    def __post_init__(self) -> None:
        if not self.entry_id.strip() or not self.request_text.strip():
            raise HistoricalCacheError("historical cache identity is invalid")
        if not self.request_vector or not all(math.isfinite(value) for value in self.request_vector):
            raise HistoricalCacheError("historical cache vector is invalid")
        if set(self.response) != {"answer"} or type(self.response["answer"]) is not str:
            raise HistoricalCacheError("historical cache response must be a plain answer")
        if not all(math.isfinite(value) for value in (self.request_timestamp, self.response_timestamp)):
            raise HistoricalCacheError("historical cache timestamps must be finite")


class HistoricalAnswerCache(Protocol):
    def lookup(self, vector: tuple[float, ...], metadata: HistoricalAnswerMetadata, *, now: float) -> HistoricalAnswerRecord | None: ...
    def put(self, record: HistoricalAnswerRecord) -> None: ...
    def cleanup(self, *, now: float) -> int: ...


def _similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    if len(left) != len(right):
        return -1.0
    left_norm = math.sqrt(math.fsum(value * value for value in left))
    right_norm = math.sqrt(math.fsum(value * value for value in right))
    if not left_norm or not right_norm:
        return -1.0
    return math.fsum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


class InMemoryHistoricalAnswerCache:
    def __init__(self) -> None:
        self._entries: dict[str, HistoricalAnswerRecord] = {}

    def lookup(self, vector: tuple[float, ...], metadata: HistoricalAnswerMetadata, *, now: float) -> HistoricalAnswerRecord | None:
        if not math.isfinite(now):
            raise HistoricalCacheError("lookup time must be finite")
        candidates = (
            (record, _similarity(vector, record.request_vector))
            for record in self._entries.values()
            if record.metadata.expires_at > now and record.metadata.invalidation_reason is None
            and record.metadata.compatible_with(metadata)
        )
        matched = [(record, score) for record, score in candidates if score > 0.95]
        if not matched:
            return None
        return min(matched, key=lambda item: (-item[1], item[0].response_timestamp, item[0].entry_id))[0]

    def put(self, record: HistoricalAnswerRecord) -> None:
        if not isinstance(record, HistoricalAnswerRecord):
            raise HistoricalCacheError("historical cache requires a concrete record")
        self._entries[record.entry_id] = record

    def cleanup(self, *, now: float) -> int:
        expired = [key for key, value in self._entries.items() if value.metadata.expires_at <= now]
        for key in expired:
            del self._entries[key]
        return len(expired)


class PostgresHistoricalAnswerCache:
    def __init__(self, connection_factory: Callable[[], Any], *, embedding_dimension: int) -> None:
        if not callable(connection_factory) or not 1 <= embedding_dimension <= 2000:
            raise HistoricalCacheError("historical PostgreSQL cache configuration is invalid")
        self._connect = connection_factory
        self._dimension = embedding_dimension

    def migrate(self) -> None:
        statements = (
            "CREATE EXTENSION IF NOT EXISTS vector",
            f"CREATE TABLE IF NOT EXISTS historical_answer_cache (entry_id TEXT PRIMARY KEY, tenant_scope TEXT NOT NULL, request_text TEXT NOT NULL, request_embedding vector({self._dimension}) NOT NULL, response JSONB NOT NULL, metadata JSONB NOT NULL, request_timestamp DOUBLE PRECISION NOT NULL, response_timestamp DOUBLE PRECISION NOT NULL, expires_at DOUBLE PRECISION NOT NULL, invalidation_reason TEXT NULL)",
            "CREATE INDEX IF NOT EXISTS historical_answer_cache_scope_idx ON historical_answer_cache (tenant_scope, expires_at)",
            "CREATE INDEX IF NOT EXISTS historical_answer_cache_vector_idx ON historical_answer_cache USING ivfflat (request_embedding vector_cosine_ops) WITH (lists = 100)",
        )
        with self._connect() as connection:
            with connection.cursor() as cursor:
                for statement in statements:
                    cursor.execute(statement)
            connection.commit()

    def put(self, record: HistoricalAnswerRecord) -> None:
        if not isinstance(record, HistoricalAnswerRecord):
            raise HistoricalCacheError("historical cache requires a concrete record")
        if len(record.request_vector) != self._dimension:
            raise HistoricalCacheError("embedding dimension does not match historical cache")
        metadata = asdict(record.metadata)
        vector = "[" + ",".join(repr(value) for value in record.request_vector) + "]"
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "INSERT INTO historical_answer_cache (entry_id,tenant_scope,request_text,request_embedding,response,metadata,request_timestamp,response_timestamp,expires_at,invalidation_reason) VALUES (%s,%s,%s,%s::vector,%s::jsonb,%s::jsonb,%s,%s,%s,%s) ON CONFLICT (entry_id) DO UPDATE SET request_text=EXCLUDED.request_text,request_embedding=EXCLUDED.request_embedding,response=EXCLUDED.response,metadata=EXCLUDED.metadata,request_timestamp=EXCLUDED.request_timestamp,response_timestamp=EXCLUDED.response_timestamp,expires_at=EXCLUDED.expires_at,invalidation_reason=EXCLUDED.invalidation_reason",
                    (record.entry_id, record.metadata.tenant_scope, record.request_text, vector,
                     json.dumps(dict(record.response), sort_keys=True), json.dumps(metadata, sort_keys=True),
                     record.request_timestamp, record.response_timestamp, record.metadata.expires_at,
                     record.metadata.invalidation_reason),
                )
            connection.commit()

    def lookup(self, vector: tuple[float, ...], metadata: HistoricalAnswerMetadata, *, now: float) -> HistoricalAnswerRecord | None:
        if len(vector) != self._dimension or not math.isfinite(now):
            raise HistoricalCacheError("historical cache lookup is invalid")
        rendered_vector = "[" + ",".join(repr(value) for value in vector) + "]"
        expected = json.dumps(asdict(metadata), sort_keys=True)
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    "SELECT entry_id,request_text,request_embedding::text,response,metadata,request_timestamp,response_timestamp FROM historical_answer_cache WHERE tenant_scope=%s AND metadata=%s::jsonb AND expires_at>%s AND invalidation_reason IS NULL AND 1-(request_embedding <=> %s::vector)>0.95 ORDER BY request_embedding <=> %s::vector,response_timestamp,entry_id LIMIT 1",
                    (metadata.tenant_scope, expected, now, rendered_vector, rendered_vector),
                )
                row = cursor.fetchone()
        if row is None:
            return None
        entry_id, request_text, encoded_vector, response, saved_metadata, request_timestamp, response_timestamp = row
        parsed_vector = tuple(float(value) for value in str(encoded_vector).strip("[]").split(",") if value)
        if isinstance(response, str):
            response = json.loads(response)
        if isinstance(saved_metadata, str):
            saved_metadata = json.loads(saved_metadata)
        return HistoricalAnswerRecord(str(entry_id), str(request_text), parsed_vector, dict(response),
                                      float(request_timestamp), float(response_timestamp),
                                      HistoricalAnswerMetadata(**dict(saved_metadata)))

    def cleanup(self, *, now: float) -> int:
        with self._connect() as connection:
            with connection.cursor() as cursor:
                cursor.execute("DELETE FROM historical_answer_cache WHERE expires_at<=%s", (now,))
                deleted = cursor.rowcount
            connection.commit()
        return max(0, int(deleted))


@dataclass(frozen=True, slots=True)
class FixedAnswerSeed:
    aliases: tuple[str, ...]
    answer: str
    category: str
    ttl_seconds: int


FIXED_ANSWER_SEEDS: tuple[FixedAnswerSeed, ...] = (
    FixedAnswerSeed(("这个系统会自动下单吗", "会自动交易吗"), "不会。系统只生成受限的分析与计划，不具备下单权限。", "policy", 86_400),
    FixedAnswerSeed(("不确定时怎么办", "不知道时怎么办"), "证据不足时会明确回答不知道，并返回不交易的安全结果。", "policy", 86_400),
    FixedAnswerSeed(("系统如何使用记忆", "长期记忆怎么用"), "系统只检索经过证据校验的摘要；代理不能直接读写长期记忆。", "explanation", 86_400),
    FixedAnswerSeed(("提示词缓存怎么用", "prompt cache怎么用"), "稳定系统提示词位于缓存前缀，动态市场数据和用户内容放在后续上下文。", "explanation", 86_400),
    FixedAnswerSeed(("系统如何保护权限", "agent有什么权限"), "每个代理只获得短期最小权限，只能读取分配给自己的上下文。", "policy", 86_400),
)


def lookup_fixed_seed(query: str, *, now: float, metadata: HistoricalAnswerMetadata) -> HistoricalAnswerRecord | None:
    normalized = " ".join(query.casefold().split())
    for index, seed in enumerate(FIXED_ANSWER_SEEDS):
        if normalized not in {" ".join(alias.casefold().split()) for alias in seed.aliases}:
            continue
        seed_metadata = HistoricalAnswerMetadata(
            **{**asdict(metadata), "category": "informational", "expires_at": now + seed.ttl_seconds}
        )
        return HistoricalAnswerRecord(
            entry_id=f"fixed-seed-{index}", request_text=normalized, request_vector=(1.0,),
            response={"answer": seed.answer}, request_timestamp=now, response_timestamp=now,
            metadata=seed_metadata,
        )
    return None
