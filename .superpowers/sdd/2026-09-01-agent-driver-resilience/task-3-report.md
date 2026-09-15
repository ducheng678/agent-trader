# Task 3 Report

## Status

Implemented safe in-process exact and semantic response caches, cited local-knowledge fallback, and one-way model downgrade decisions.

## Commits

`feat: add safe response caches and fallback policy`

## Tests

- `python -m pytest -q market_agent_test_bundle/tests/test_workflow_response_cache.py market_agent_test_bundle/tests/test_workflow_semantic_request_cache.py market_agent_test_bundle/tests/test_workflow_fallback.py` — 23 passed.
- `python -m compileall -q market_agent/workflow_response_cache.py market_agent/workflow_semantic_request_cache.py market_agent/workflow_fallback.py market_agent/local_knowledge_base.py` — passed.
- `git diff --check` — passed.

## Concerns

- Semantic matching is deliberately deterministic in-process cosine similarity; this phase has no external vector-database dependency.
- Cache admission is fail-closed to an explicit read-only category allowlist and rejects sensitive response fields.

## Review closure — 2026-09-01

Closed the four Task 3 review findings:

- Both caches now enforce a closed `{ "answer": <plain text> }` contract. The default only permits the literal `不知道`. Additional safe fixed answers require an explicit `safe_answers={schema_digest: {reviewed_literal, ...}}` constructor policy, snapshotted into immutable sets. Categories, field-name heuristics, and untrusted prose cannot establish safety. Unknown fields, nested objects/lists, BUY/sell/order content, tool calls, camelCase secrets, private content, and volatile assertions are denied unless an operator incorrectly places unsafe plain text in the trusted static policy. The policy must never be populated from provider/tool output or local-knowledge promotion.
- `CacheMetadata` now carries optional `vector_version` and `model_version` fields for exact-cache compatibility. Both are mandatory for semantic admission and lookup. Entry versions must equal those metadata fields, and lookup requires full metadata equality, independently of the broad model compatibility key.
- Semantic entries replace caller-owned vector sequences with canonical immutable float tuples at construction. Stored/returned vectors cannot alias mutable input; response copies and constructor policy snapshots also retain isolation.
- Cosine calculation scales each vector by a power of two before computing products and norms. Subnormal and near-maximum finite vectors match without division by zero or NaN misses, while the strict `0.95` equality miss remains intact. Booleans, zero/non-finite vectors, and integers that overflow float conversion are rejected.

Regression evidence: the initial test-first run reproduced 37 failures (27 passing), including unsafe admission, version gaps, list aliasing, underflow exceptions, and overflow misses. Final focused verification:

- `python -m pytest -q market_agent_test_bundle/tests/test_workflow_response_cache.py market_agent_test_bundle/tests/test_workflow_semantic_request_cache.py market_agent_test_bundle/tests/test_workflow_fallback.py --tb=short` — 69 passed.
- `python -m compileall -q market_agent/workflow_response_cache.py market_agent/workflow_semantic_request_cache.py market_agent/workflow_fallback.py market_agent/local_knowledge_base.py` — passed.
- `git diff --check` — passed (Git emitted only the repository's LF/CRLF conversion notices).

Integration note: callers must provide reviewed schema-scoped fixed answers when caching content beyond abstention, and must specify both semantic versions in metadata. Local-knowledge fallback and its citations remain unchanged and usable outside cache admission. No `.tmpbudget` files were changed.

Closure commit: `fix: close cache safety and semantic compatibility gaps`.
