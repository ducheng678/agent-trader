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
        vector_version="fixed-v1",
        model_version="model-v1",
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
        response={"answer": "stable"},
        metadata=cache_metadata or metadata(),
        created_at=created_at,
        vector_version="fixed-v1",
        model_version="model-v1",
    )


def test_similarity_at_the_threshold_is_a_miss():
    """Changing strict >0.95 reuse to >= could return an insufficiently similar answer."""
    cache = SemanticRequestCache(safe_answers={"schema-1": {"stable"}})
    cache.put(entry((1.0, 0.0)))

    assert cache.lookup((0.95, (1.0 - 0.95**2) ** 0.5), metadata(), now=1.0) is None


def test_lookup_requires_matching_metadata_and_rejects_expired_entries():
    """Ignoring tenant, release, schema, model, or TTL can cross a safety boundary."""
    cache = SemanticRequestCache(safe_answers={"schema-1": {"stable"}})
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
    cache = SemanticRequestCache(safe_answers={"schema-1": {"stable"}})
    cache.put(entry((1.0, 0.0)))

    assert cache.lookup((1.0, 0.0), replace(metadata(), **{field: value}), now=1.0) is None


def test_lookup_breaks_equal_similarity_ties_by_creation_time_then_entry_id():
    """Unstable nearest-neighbour ties would make a fixed-seed workflow nondeterministic."""
    cache = SemanticRequestCache(safe_answers={"schema-1": {"stable"}})
    later = entry((1.0, 0.0), entry_id="z", created_at=2.0)
    earlier = entry((1.0, 0.0), entry_id="b", created_at=1.0)
    same_time_lower_id = entry((1.0, 0.0), entry_id="a", created_at=1.0)
    cache.put(later)
    cache.put(earlier)
    cache.put(same_time_lower_id)

    assert cache.lookup((1.0, 0.0), metadata(), now=1.0) == same_time_lower_id


def test_cleanup_removes_expired_entries_idempotently():
    """Keeping expired vectors after cleanup can expose stale answers later."""
    cache = SemanticRequestCache(safe_answers={"schema-1": {"stable"}})
    cache.put(entry((1.0, 0.0), cache_metadata=metadata(expires_at=2.0)))

    assert cache.cleanup(now=2.0) == 1
    assert cache.cleanup(now=2.0) == 0


def test_unsafe_category_cannot_enter_semantic_cache():
    """Semantic storage must enforce the same no-trade/no-tool/no-secret boundary."""
    cache = SemanticRequestCache(safe_answers={"schema-1": {"stable"}})

    with pytest.raises(CacheSafetyError):
        cache.put(entry((1.0, 0.0), cache_metadata=metadata().with_category("trade_decision")))


@pytest.mark.parametrize("field,value", [("vector_version", "fixed-v2"), ("model_version", "model-v2")])
def test_lookup_rejects_incompatible_vector_and_model_versions(field, value):
    """Matching dimensions and compatibility keys cannot authorize a different version."""
    cache = SemanticRequestCache(safe_answers={"schema-1": {"stable"}})
    cache.put(entry((1.0, 0.0)))
    incompatible = replace(metadata(), **{field: value})
    assert cache.lookup((1.0, 0.0), incompatible, now=1.0) is None


@pytest.mark.parametrize("field", ["vector_version", "model_version"])
def test_semantic_admission_and_lookup_require_explicit_versions(field):
    """A caller omitting a version must not retrieve or store unversioned semantic content."""
    cache = SemanticRequestCache(safe_answers={"schema-1": {"stable"}})
    cache.put(entry((1.0, 0.0)))
    incomplete = replace(metadata(), **{field: None})
    with pytest.raises(CacheSafetyError):
        cache.lookup((1.0, 0.0), incomplete, now=1.0)
    with pytest.raises(CacheSafetyError):
        cache.put(entry((1.0, 0.0), cache_metadata=incomplete))


def test_semantic_safe_answer_policy_is_schema_scoped_and_snapshotted():
    """Another schema's answer and later policy mutations must not authorize cached content."""
    approved = {"schema-1": {"stable"}, "schema-2": {"other answer"}}
    cache = SemanticRequestCache(safe_answers=approved)
    approved["schema-1"].add("unreviewed")
    candidate = entry((1.0, 0.0))
    cache.put(candidate)
    candidate.response["answer"] = "tampered"
    for answer in ("other answer", "unreviewed"):
        with pytest.raises(CacheSafetyError):
            cache.put(replace(candidate, response={"answer": answer}))
    hit = cache.lookup((1.0, 0.0), metadata(), now=1.0)
    assert hit.response == {"answer": "stable"}
    hit.response["answer"] = "tampered"
    assert cache.lookup((1.0, 0.0), metadata(), now=1.0).response == {"answer": "stable"}


def test_default_semantic_cache_admits_only_the_fixed_abstention():
    cache = SemanticRequestCache()
    cache.put(replace(entry((1.0, 0.0)), response={"answer": "不知道"}))
    assert cache.lookup((1.0, 0.0), metadata(), now=1.0).response == {"answer": "不知道"}
    with pytest.raises(CacheSafetyError):
        cache.put(entry((1.0, 0.0)))


@pytest.mark.parametrize("field,value", [("vector_version", "fixed-v2"), ("model_version", "model-v2")])
def test_admission_rejects_entry_versions_inconsistent_with_metadata(field, value):
    """An entry must not claim compatibility metadata for a different vector or model."""
    cache = SemanticRequestCache(safe_answers={"schema-1": {"stable"}})
    candidate = replace(entry((1.0, 0.0)), **{field: value})
    with pytest.raises(CacheSafetyError):
        cache.put(candidate)


def test_entry_and_returned_vectors_cannot_alias_the_callers_mutable_list():
    """Mutating input or output must not change which request retrieves a stored answer."""
    vector = [1.0, 0.0]
    candidate = entry(vector)
    vector[:] = [0.0, 1.0]
    assert candidate.request_vector == (1.0, 0.0)
    cache = SemanticRequestCache(safe_answers={"schema-1": {"stable"}})
    stored = cache.put(candidate)
    assert isinstance(stored.request_vector, tuple)
    with pytest.raises(TypeError):
        stored.request_vector[0] = 0.0
    hit = cache.lookup((1.0, 0.0), metadata(), now=1.0)
    assert hit is not None
    assert isinstance(hit.request_vector, tuple)
    assert cache.lookup((0.0, 1.0), metadata(), now=1.0) is None


@pytest.mark.parametrize("scale", [1e-300, 5e-324, 1e300, 1.7e308])
def test_extreme_finite_vectors_match_without_underflow_or_overflow(scale):
    """Squaring raw finite components can divide by zero or turn exact matches into NaN misses."""
    cache = SemanticRequestCache(safe_answers={"schema-1": {"stable"}})
    cache.put(entry((scale, scale)))
    hit = cache.lookup((scale, scale), metadata(), now=1.0)
    assert hit is not None
    assert hit.entry_id == "entry-1"
    assert cache.lookup((-scale, -scale), metadata(), now=1.0) is None


@pytest.mark.parametrize(
    "vector", [(0.0, 0.0), (float("nan"), 1.0), (float("inf"), 1.0), (True, 0.0), (10**400, 1.0)]
)
def test_unusable_vectors_are_rejected_at_entry_and_lookup(vector):
    with pytest.raises(ValueError):
        entry(vector)
    with pytest.raises(ValueError):
        SemanticRequestCache().lookup(vector, metadata(), now=1.0)
