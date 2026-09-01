from __future__ import annotations

import pytest
from dataclasses import replace

from market_agent.workflow_response_cache import (
    CacheMetadata,
    CacheSafetyError,
    ExactCacheKey,
    ExactResponseCache,
)
from market_agent.workflow_semantic_request_cache import SemanticCacheEntry, SemanticRequestCache


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
    cache = ExactResponseCache(safe_answers={"schema-1": {"stable"}})
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
        unsafe_metadata = metadata(category=category)
        cache.put(
            ExactCacheKey.from_metadata("request-1", unsafe_metadata),
            {"answer": "不知道"}, unsafe_metadata, now=1.0,
        )


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
    cache = ExactResponseCache(safe_answers={"schema-1": {"stable"}})
    cache.put(key(), {"answer": "stable"}, metadata(), now=1.0)

    assert cache.get(key(), replace(metadata(), **{field: value}), now=1.0) is None


@pytest.mark.parametrize("cache_kind", ["exact", "semantic"])
@pytest.mark.parametrize(
    "response",
    [
        {"action": "BUY"},
        {"answer": "BUY"},
        {"answer": "sell"},
        {"answer": "Place a market order for 10 BTC."},
        {"answer": "stable", "tool_calls": [{"name": "place_order"}]},
        {"answer": {"nested": {"action": "sell"}}},
        {"answer": [{"order": {"side": "BUY"}}]},
        {"apiKey": "sensitive-value"},
        {"answer": {"nested": [{"apiKey": "sensitive-value"}]}},
        {"answer": "The API key is sensitive-value."},
        {"answer": "BTC is trading at 100000 right now."},
        {"answer": "Alice's private account balance is 2000."},
        {"answer": "unreviewed provider prose"},
    ],
)
def test_safe_category_cannot_authorize_untrusted_payloads(cache_kind, response):
    """A safe category must not admit actions, tools, nested objects, or arbitrary prose."""
    if cache_kind == "exact":
        with pytest.raises(CacheSafetyError):
            ExactResponseCache(safe_answers={"schema-1": {"stable"}}).put(
                key(), response, metadata(), now=1.0
            )
    else:
        candidate = SemanticCacheEntry(
            "unsafe", (1.0, 0.0), response,
            replace(metadata(), vector_version="fixed-v1", model_version="model-v1"),
            1.0, "fixed-v1", "model-v1",
        )
        with pytest.raises(CacheSafetyError):
            SemanticRequestCache(safe_answers={"schema-1": {"stable"}}).put(candidate)


def test_exact_cache_only_admits_reviewed_literals_for_the_declared_schema():
    """Accepting unlisted text or another schema's literals would bypass explicit safety review."""
    approved = {"schema-1": {"stable"}, "schema-2": {"other answer"}}
    cache = ExactResponseCache(safe_answers=approved)
    approved["schema-1"].add("late unreviewed text")
    cached = cache.put(key(), {"answer": "stable"}, metadata(), now=1.0)
    cached.response["answer"] = "tampered"
    assert cache.get(key(), metadata(), now=1.0).response == {"answer": "stable"}
    for answer in ("other answer", "late unreviewed text"):
        with pytest.raises(CacheSafetyError):
            cache.put(key(), {"answer": answer}, metadata(), now=1.0)


def test_default_exact_cache_admits_only_the_fixed_abstention():
    cache = ExactResponseCache()
    cache.put(key(), {"answer": "不知道"}, metadata(), now=1.0)
    assert cache.get(key(), metadata(), now=1.0).response == {"answer": "不知道"}
