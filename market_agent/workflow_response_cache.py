"""Safe, in-process exact-response cache for read-only fixed answers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math
from typing import Iterable, Mapping


class CacheSafetyError(ValueError):
    """Raised when a response is not explicitly safe to retain or replay."""


_SAFE_CATEGORIES = frozenset(
    {
        "documentation",
        "explanation",
        "extraction",
        "fixed_seed",
        "policy",
        "read_only",
        "reference",
        "safe_answer",
        "summary",
        "validation",
    }
)


@dataclass(frozen=True, slots=True)
class CacheMetadata:
    """Immutable compatibility and expiry gates carried by every cache entry."""

    tenant_scope: str
    prompt_release_digest: str
    output_schema_digest: str
    model_compatibility_key: str
    category: str
    expires_at: float
    vector_version: str | None = None
    model_version: str | None = None

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                self.tenant_scope,
                self.prompt_release_digest,
                self.output_schema_digest,
                self.model_compatibility_key,
                self.category,
            )
        ):
            raise ValueError("cache metadata strings must be non-empty")
        if not math.isfinite(self.expires_at):
            raise ValueError("cache expiry must be finite")
        for version in (self.vector_version, self.model_version):
            if version is not None and (not isinstance(version, str) or not version.strip()):
                raise ValueError("cache versions must be non-empty when present")

    def with_category(self, category: str) -> CacheMetadata:
        return replace(self, category=category)


@dataclass(frozen=True, slots=True)
class ExactCacheKey:
    """All identity fields for deterministic, metadata-scoped exact lookup."""

    tenant_scope: str
    canonical_request_hash: str
    prompt_release_digest: str
    output_schema_digest: str
    model_compatibility_key: str
    category: str

    def __post_init__(self) -> None:
        if not all(
            isinstance(value, str) and value.strip()
            for value in (
                self.tenant_scope,
                self.canonical_request_hash,
                self.prompt_release_digest,
                self.output_schema_digest,
                self.model_compatibility_key,
                self.category,
            )
        ):
            raise ValueError("exact cache key strings must be non-empty")

    @classmethod
    def from_metadata(cls, canonical_request_hash: str, metadata: CacheMetadata) -> ExactCacheKey:
        return cls(
            tenant_scope=metadata.tenant_scope,
            canonical_request_hash=canonical_request_hash,
            prompt_release_digest=metadata.prompt_release_digest,
            output_schema_digest=metadata.output_schema_digest,
            model_compatibility_key=metadata.model_compatibility_key,
            category=metadata.category,
        )


@dataclass(frozen=True, slots=True)
class CachedResponse:
    """A response only becomes cacheable after safety and metadata validation."""

    key: ExactCacheKey
    response: Mapping[str, object]
    metadata: CacheMetadata
    created_at: float


def require_cache_safe(
    metadata: CacheMetadata,
    response: Mapping[str, object],
    safe_answers: Mapping[str, frozenset[str]] | None = None,
) -> None:
    """Admit only the closed answer schema and a reviewed, schema-scoped literal.

    Categories alone cannot establish semantic safety. The only default answer
    is the fixed abstention; other literals must come from trusted static policy,
    never from provider output, tool results, or local-knowledge promotion.
    """
    if metadata.category not in _SAFE_CATEGORIES:
        raise CacheSafetyError("cache category is not explicitly safe and read-only")
    if not isinstance(response, Mapping) or set(response) != {"answer"}:
        raise CacheSafetyError("cache response must contain only the approved answer field")
    answer = response["answer"]
    if type(answer) is not str:
        raise CacheSafetyError("cache answer must be plain text, without nested content")
    approved = (safe_answers or {}).get(metadata.output_schema_digest, frozenset())
    if answer != "不知道" and answer not in approved:
        raise CacheSafetyError("cache answer is not an explicitly reviewed safe literal")


def snapshot_safe_answers(
    safe_answers: Mapping[str, Iterable[str]] | None,
) -> dict[str, frozenset[str]]:
    """Snapshot trusted schema/literal policy so caller mutation cannot widen admission."""
    approved: dict[str, frozenset[str]] = {}
    for schema, answers in (safe_answers or {}).items():
        if type(schema) is not str or not schema.strip() or isinstance(answers, (str, bytes)):
            raise ValueError("safe answers require a schema digest and a collection of literals")
        literals = tuple(answers)
        if not all(type(answer) is str and answer.strip() for answer in literals):
            raise ValueError("reviewed safe answers must be non-empty plain text literals")
        approved[schema] = frozenset(literals)
    return approved


class ExactResponseCache:
    """A process-local exact cache with hard metadata and expiry checks."""

    def __init__(self, *, safe_answers: Mapping[str, Iterable[str]] | None = None) -> None:
        self._safe_answers = snapshot_safe_answers(safe_answers)
        self._entries: dict[ExactCacheKey, CachedResponse] = {}

    def put(
        self,
        key: ExactCacheKey,
        response: Mapping[str, object],
        metadata: CacheMetadata,
        *,
        now: float,
    ) -> CachedResponse:
        self._validate_key_matches_metadata(key, metadata)
        require_cache_safe(metadata, response, self._safe_answers)
        if not math.isfinite(now):
            raise ValueError("cache creation time must be finite")
        entry = CachedResponse(key, deepcopy(dict(response)), metadata, now)
        self._entries[key] = entry
        return self._copy(entry)

    def get(self, key: ExactCacheKey, metadata: CacheMetadata, *, now: float) -> CachedResponse | None:
        if not math.isfinite(now):
            raise ValueError("cache lookup time must be finite")
        entry = self._entries.get(key)
        if entry is None or entry.metadata != metadata or entry.metadata.expires_at <= now:
            return None
        return self._copy(entry)

    def cleanup(self, *, now: float) -> int:
        if not math.isfinite(now):
            raise ValueError("cache cleanup time must be finite")
        expired = [key for key, entry in self._entries.items() if entry.metadata.expires_at <= now]
        for key in expired:
            del self._entries[key]
        return len(expired)

    @staticmethod
    def _validate_key_matches_metadata(key: ExactCacheKey, metadata: CacheMetadata) -> None:
        if (
            key.tenant_scope,
            key.prompt_release_digest,
            key.output_schema_digest,
            key.model_compatibility_key,
            key.category,
        ) != (
            metadata.tenant_scope,
            metadata.prompt_release_digest,
            metadata.output_schema_digest,
            metadata.model_compatibility_key,
            metadata.category,
        ):
            raise CacheSafetyError("exact cache key must match immutable metadata")

    @staticmethod
    def _copy(entry: CachedResponse) -> CachedResponse:
        return replace(entry, response=deepcopy(dict(entry.response)))
