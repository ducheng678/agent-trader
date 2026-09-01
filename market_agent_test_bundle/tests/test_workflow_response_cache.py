from __future__ import annotations

import pytest
from dataclasses import replace

from market_agent.workflow_response_cache import (
    CacheMetadata,
    CacheSafetyError,
    ExactCacheKey,
    ExactResponseCache,
)


def metadata(
    *, expires_at: float = 20.0, category: str = "reference", tenant_scope: str = "tenant-a"
) -> CacheMetadata:
    return CacheMetadata(
        tenant_scope=tenant_scope,
        prompt_release_digest="release-1",
        output_schema_digest="schema-1",
        model_compatibility_key="model-v1",
        category=category,
        expires_at=expires_at,
    )


def key() -> ExactCacheKey:
    return ExactCacheKey(
        tenant_scope="tenant-a",
        canonical_request_hash="request-1",
        prompt_release_digest="release-1",
        output_schema_digest="schema-1",
        model_compatibility_key="model-v1",
        category="reference",
    )


def test_exact_cache_returns_only_an_unexpired_metadata_compatible_response():
    """Dropping a release or expiry gate could replay an invalid answer."""
    cache = ExactResponseCache()
    entry = cache.put(key(), {"answer": "stable"}, metadata(), now=1.0)

    assert cache.get(key(), metadata(), now=19.0) == entry
    assert cache.get(key(), metadata(), now=20.0) is None
    assert cache.get(key(), metadata(tenant_scope="tenant-b"), now=19.0) is None


@pytest.mark.parametrize(
    "category",
    [
        "trade_decision",
        "order_instruction",
        "tool_result",
        "secret",
        "volatile_market_assertion",
        "personally_sensitive",
    ],
)
def test_unsafe_categories_cannot_enter_the_exact_cache(category: str):
    """Accepting an unsafe category could replay an order, tool output, or secret."""
    cache = ExactResponseCache()

    with pytest.raises(CacheSafetyError):
        cache.put(key(), {"answer": "unsafe"}, metadata(category=category), now=1.0)


@pytest.mark.parametrize(
    "field,value",
    [
        ("tenant_scope", "tenant-b"),
        ("prompt_release_digest", "release-2"),
        ("output_schema_digest", "schema-2"),
        ("model_compatibility_key", "model-v2"),
    ],
)
def test_exact_cache_metadata_gate_rejects_each_compatibility_mismatch(field: str, value: str):
    """Skipping any compatibility field can replay a response under a different contract."""
    cache = ExactResponseCache()
    cache.put(key(), {"answer": "stable"}, metadata(), now=1.0)

    assert cache.get(key(), replace(metadata(), **{field: value}), now=1.0) is None
