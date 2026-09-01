"""Deterministic, in-process semantic request cache with strict reuse gates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math
from typing import Mapping, Sequence

from market_agent.workflow_response_cache import CacheMetadata, require_cache_safe


_SIMILARITY_THRESHOLD = 0.95


def _validated_vector(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(vector)
    if not values or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
        raise ValueError("semantic cache vectors must be non-empty finite numbers")
    if not any(value != 0 for value in values):
        raise ValueError("semantic cache vectors must not be zero")
    return tuple(float(value) for value in values)


@dataclass(frozen=True, slots=True)
class SemanticCacheEntry:
    """A safe response plus the versioned vector that may retrieve it."""

    entry_id: str
    request_vector: tuple[float, ...]
    response: Mapping[str, object]
    metadata: CacheMetadata
    created_at: float
    vector_version: str
    model_version: str

    def __post_init__(self) -> None:
        if not isinstance(self.entry_id, str) or not self.entry_id.strip():
            raise ValueError("semantic cache entry ID must be non-empty")
        _validated_vector(self.request_vector)
        if not math.isfinite(self.created_at):
            raise ValueError("semantic cache creation time must be finite")
        if not isinstance(self.vector_version, str) or not self.vector_version.strip():
            raise ValueError("semantic cache vector version must be non-empty")
        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise ValueError("semantic cache model version must be non-empty")


class SemanticRequestCache:
    """In-memory cosine matching; no external vector database is used in this phase."""

    def __init__(self) -> None:
        self._entries: dict[str, SemanticCacheEntry] = {}

    def put(self, entry: SemanticCacheEntry) -> SemanticCacheEntry:
        require_cache_safe(entry.metadata, entry.response)
        stored = self._copy(entry)
        self._entries[stored.entry_id] = stored
        return self._copy(stored)

    store = put

    def lookup(
        self, query: Sequence[float], metadata: CacheMetadata, now: float
    ) -> SemanticCacheEntry | None:
        if not math.isfinite(now):
            raise ValueError("semantic cache lookup time must be finite")
        request_vector = _validated_vector(query)
        eligible: list[tuple[float, SemanticCacheEntry]] = []
        for entry in self._entries.values():
            if entry.metadata != metadata or entry.metadata.expires_at <= now:
                continue
            if len(entry.request_vector) != len(request_vector):
                continue
            similarity = _cosine_similarity(request_vector, entry.request_vector)
            if similarity > _SIMILARITY_THRESHOLD:
                eligible.append((similarity, entry))
        if not eligible:
            return None
        _, selected = sorted(
            eligible, key=lambda candidate: (-candidate[0], candidate[1].created_at, candidate[1].entry_id)
        )[0]
        return self._copy(selected)

    def cleanup(self, *, now: float) -> int:
        if not math.isfinite(now):
            raise ValueError("semantic cache cleanup time must be finite")
        expired = [entry_id for entry_id, entry in self._entries.items() if entry.metadata.expires_at <= now]
        for entry_id in expired:
            del self._entries[entry_id]
        return len(expired)

    @staticmethod
    def _copy(entry: SemanticCacheEntry) -> SemanticCacheEntry:
        return replace(entry, response=deepcopy(dict(entry.response)))


def _cosine_similarity(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    dot_product = sum(left_value * right_value for left_value, right_value in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    return dot_product / (left_norm * right_norm)
