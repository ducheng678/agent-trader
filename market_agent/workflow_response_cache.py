"""Safe, in-process exact-response cache for read-only fixed answers."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math
from typing import Mapping


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
_SENSITIVE_FIELD_TOKENS = frozenset(
    {"api_key", "authorization", "credential", "password", "secret", "token"}
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


def require_cache_safe(metadata: CacheMetadata, response: Mapping[str, object]) -> None:
    """Fail closed unless a response has an explicitly safe category and no secret field."""
    if metadata.category not in _SAFE_CATEGORIES:
        raise CacheSafetyError("cache category is not explicitly safe and read-only")
    if not isinstance(response, Mapping):
        raise CacheSafetyError("cache response must be a JSON object")
    _reject_sensitive_fields(response)


def _reject_sensitive_fields(value: object) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            normalized_key = str(key).strip().lower().replace("-", "_")
            if any(token in normalized_key for token in _SENSITIVE_FIELD_TOKENS):
                raise CacheSafetyError("cache response contains a sensitive field")
            _reject_sensitive_fields(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _reject_sensitive_fields(child)


class ExactResponseCache:
    """A process-local exact cache with hard metadata and expiry checks."""

    def __init__(self) -> None:
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
        require_cache_safe(metadata, response)
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
