from __future__ import annotations

import pytest
from dataclasses import replace

from market_agent.workflow_response_cache import CacheMetadata, CacheSafetyError
from market_agent.workflow_semantic_request_cache import SemanticCacheEntry, SemanticRequestCache


def metadata(*, expires_at: float = 20.0, tenant_scope: str = "tenant-a") -> CacheMetadata:
    return CacheMetadata(
        tenant_scope=tenant_scope,
        prompt_release_digest="release-1",
        output_schema_digest="schema-1",
        model_compatibility_key="model-v1",
        category="reference",
        expires_at=expires_at,
    )


def entry(
    vector: tuple[float, ...],
    *,
    entry_id: str = "entry-1",
    created_at: float = 1.0,
    cache_metadata: CacheMetadata | None = None,
) -> SemanticCacheEntry:
    return SemanticCacheEntry(
        entry_id=entry_id,
        request_vector=vector,
        response={"answer": entry_id},
        metadata=cache_metadata or metadata(),
        created_at=created_at,
        vector_version="fixed-v1",
        model_version="model-v1",
    )


def test_similarity_at_the_threshold_is_a_miss():
    """Changing strict >0.95 reuse to >= could return an insufficiently similar answer."""
    cache = SemanticRequestCache()
    cache.put(entry((1.0, 0.0)))

    assert cache.lookup((0.95, (1.0 - 0.95**2) ** 0.5), metadata(), now=1.0) is None


def test_lookup_requires_matching_metadata_and_rejects_expired_entries():
    """Ignoring tenant, release, schema, model, or TTL can cross a safety boundary."""
    cache = SemanticRequestCache()
    cached = entry((1.0, 0.0), cache_metadata=metadata(expires_at=5.0))
    cache.put(cached)

    assert cache.lookup((1.0, 0.0), metadata(expires_at=5.0, tenant_scope="tenant-b"), now=1.0) is None
    assert cache.lookup((1.0, 0.0), metadata(expires_at=5.0), now=5.0) is None


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenant_scope", "tenant-b"),
        ("prompt_release_digest", "release-2"),
        ("output_schema_digest", "schema-2"),
        ("model_compatibility_key", "model-v2"),
    ],
)
def test_semantic_cache_requires_every_contract_metadata_gate(field: str, value: str):
    """A single omitted metadata comparison can cross a tenant or schema boundary."""
    cache = SemanticRequestCache()
    cache.put(entry((1.0, 0.0)))

    assert cache.lookup((1.0, 0.0), replace(metadata(), **{field: value}), now=1.0) is None


def test_lookup_breaks_equal_similarity_ties_by_creation_time_then_entry_id():
    """Unstable nearest-neighbour ties would make a fixed-seed workflow nondeterministic."""
    cache = SemanticRequestCache()
    later = entry((1.0, 0.0), entry_id="z", created_at=2.0)
    earlier = entry((1.0, 0.0), entry_id="b", created_at=1.0)
    same_time_lower_id = entry((1.0, 0.0), entry_id="a", created_at=1.0)
    cache.put(later)
    cache.put(earlier)
    cache.put(same_time_lower_id)

    assert cache.lookup((1.0, 0.0), metadata(), now=1.0) == same_time_lower_id


def test_cleanup_removes_expired_entries_idempotently():
    """Keeping expired vectors after cleanup can expose stale answers later."""
    cache = SemanticRequestCache()
    cache.put(entry((1.0, 0.0), cache_metadata=metadata(expires_at=2.0)))

    assert cache.cleanup(now=2.0) == 1
    assert cache.cleanup(now=2.0) == 0


def test_unsafe_category_cannot_enter_semantic_cache():
    """Semantic storage must enforce the same no-trade/no-tool/no-secret boundary."""
    cache = SemanticRequestCache()

    with pytest.raises(CacheSafetyError):
        cache.put(entry((1.0, 0.0), cache_metadata=metadata().with_category("trade_decision")))
