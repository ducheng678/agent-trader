"""Deterministic, in-process semantic request cache with strict reuse gates."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import math
from typing import Iterable, Mapping, Sequence

from market_agent.workflow_response_cache import (
    CacheMetadata,
    CacheSafetyError,
    require_cache_safe,
    snapshot_safe_answers,
)


_SIMILARITY_THRESHOLD = 0.95


def _validated_vector(vector: Sequence[float]) -> tuple[float, ...]:
    values = tuple(vector)
    if not values or not all(type(value) in (int, float) for value in values):
        raise ValueError("semantic cache vectors must be non-empty finite numbers")
    try:
        canonical = tuple(float(value) for value in values)
    except OverflowError as exc:
        raise ValueError("semantic cache vectors must fit finite floats") from exc
    if not all(math.isfinite(value) for value in canonical):
        raise ValueError("semantic cache vectors must be non-empty finite numbers")
    if not any(value != 0 for value in canonical):
        raise ValueError("semantic cache vectors must not be zero")
    return canonical


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
        object.__setattr__(self, "request_vector", _validated_vector(self.request_vector))
        if not math.isfinite(self.created_at):
            raise ValueError("semantic cache creation time must be finite")
        if not isinstance(self.vector_version, str) or not self.vector_version.strip():
            raise ValueError("semantic cache vector version must be non-empty")
        if not isinstance(self.model_version, str) or not self.model_version.strip():
            raise ValueError("semantic cache model version must be non-empty")


class SemanticRequestCache:
    """In-memory cosine matching; no external vector database is used in this phase."""

    def __init__(self, *, safe_answers: Mapping[str, Iterable[str]] | None = None) -> None:
        self._safe_answers = snapshot_safe_answers(safe_answers)
        self._entries: dict[str, SemanticCacheEntry] = {}

    def put(self, entry: SemanticCacheEntry) -> SemanticCacheEntry:
        require_cache_safe(entry.metadata, entry.response, self._safe_answers)
        if (
            entry.vector_version != entry.metadata.vector_version
            or entry.model_version != entry.metadata.model_version
        ):
            raise CacheSafetyError("semantic entry versions must match immutable metadata")
        stored = self._copy(entry)
        self._entries[stored.entry_id] = stored
        return self._copy(stored)

    store = put

    def lookup(
        self, query: Sequence[float], metadata: CacheMetadata, now: float
    ) -> SemanticCacheEntry | None:
        if not math.isfinite(now):
            raise ValueError("semantic cache lookup time must be finite")
        if metadata.vector_version is None or metadata.model_version is None:
            raise CacheSafetyError("semantic lookup requires vector and model versions")
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
    # Scale before multiplication: finite subnormals and near-maximum floats
    # otherwise underflow or overflow when squared. Powers of two preserve
    # representable significands (including the strict threshold boundary).
    # Each scaled vector has a component in [0.5, 1), ensuring nonzero norms.
    left_exponent = math.frexp(max(abs(value) for value in left))[1]
    right_exponent = math.frexp(max(abs(value) for value in right))[1]
    left_scaled = tuple(math.ldexp(value, -left_exponent) for value in left)
    right_scaled = tuple(math.ldexp(value, -right_exponent) for value in right)
    dot_product = math.fsum(
        a * b for a, b in zip(left_scaled, right_scaled, strict=True)
    )
    similarity = dot_product / (math.hypot(*left_scaled) * math.hypot(*right_scaled))
    return max(-1.0, min(1.0, similarity))
