# Task 4 Report: Compose and Audit AgentDriver

## Status

Implemented the single bounded `AgentDriver.execute(invocation) -> AgentResult` path in `workflow_agent_driver.py`, extended the existing audit taxonomy and added the capability-scoped `AuditObserver` protocol. Task 1–3 public modules and cache configuration remain unchanged.

## Delivered behavior

- Ordered fixed-cache, semantic-cache, routed model/retry/circuit, downward fallback, cited local knowledge, and exact `不知道` abstention.
- Injected model client, clock/sleep, randomness, observer, release/schema bindings, cache context, caches and policies; no provider SDK, Harness persistence, execution backend, exchange, queue, or memory imports.
- Output bindings hash closed Pydantic JSON schemas. Every result is validated strictly; prose/markdown wrappers, duplicate keys, non-JSON values, extra fields, coercions, invalid provider usage and wrong model tiers fail with redacted typed failures.
- Stable released system prefix precedes fixed internal-reasoning/JSON/unknown instructions and canonical dynamic JSON user content. Requests carry the schema digest, approved temperature, deadline, attempt and per-call cost ceiling.
- Retry keeps its tier, honors Retry-After/full jitter and enforces invocation-wide attempts, deadline and conservative reserved cost. Explicit permanent error codes override transient HTTP statuses. Fallback never raises the model tier.
- Cache replay revalidates the schema, immutable metadata, TTL, safe literal policy, versions and strict semantic similarity. The Task 3 schema-scoped `safe_answers` policy is snapshotted. Generated model output and local knowledge are never promoted to the injected caches.
- Cache/retry/circuit/downgrade/knowledge/abstention/final events retain the original trace and contain only hashes, registry codes and numeric metadata. Existing `AuditWriter` accepts the events. Required audit failure stops execution.
- Core schema bindings can declare `reflection_required`; a `core_result_ready` event marks the validated result and an optional observation-only verification hook receives it. No reflection correction or authority transition is implemented.

## TDD and verification

- Initial RED: all 45 initial driver cases failed with the explicit assertion that AgentDriver did not exist.
- Initial GREEN: all 45 cases passed.
- Boundary RED: 12 failures reproduced contradictory permanent-provider codes, forged semantic versions/similarity, an audit-time deadline race, and misleading extra-field schema hints. Each failure class was fixed and its focused cases rerun.
- Final driver suite: `python -m pytest market_agent_test_bundle/tests/test_workflow_agent_driver.py -q --tb=short` — 69 passed.
- All eight Task 1–4 test files plus the existing audit suite — 275 passed. The two warnings originate from existing audit tests deliberately constructing invalid Pydantic values.
- `python -m pytest -q market_agent_test_bundle/tests -k workflow --tb=short` — 815 passed, 1044 deselected, 4 existing Pydantic serialization warnings in audit/context-summary tests.
- Compileall for driver, audit module and driver tests — passed.
- `git diff --check` — passed; only the repository's LF/CRLF conversion notice was emitted.

## Integration notes and concerns

- `ModelClient.invoke` must enforce the supplied deadline and cost reservation at its transport boundary. The synchronous driver prevents late dispatch/retry and discards late results; it cannot interrupt an uncooperative blocking adapter.
- `model_costs` are trusted upper bounds. Failed calls conservatively consume the full reservation; valid returned usage must fit it.
- Cache tenant/category/vector/model context comes from the trusted injected context adapter, never inferred from provider text. Default cache policy still admits only the fixed unknown literal.
- Local knowledge requires a declared schema that accepts both answer and citations; otherwise the driver returns that schema's prevalidated abstention.
- Trace identifiers must satisfy the existing audit identifier contract. Other caller identifiers are hashed before audit emission.
- No `.tmpbudget` files were read or changed.

## Commit

`feat: compose auditable resilient agent driver`

## Task 4 P2 review closure

- Revalidated the injected clock after both exact and semantic cache adapters return, so the driver's TTL and semantic reuse gates use retrieval-completion time. An entry expiring at or before that time produces a rejected miss and continues to the model path.
- Wrapped the pre-dispatch probe and prompt audit emissions so an audit failure records an acquired half-open probe as failed before propagating `audit_unavailable`. This reopens the circuit for its normal cooldown without calling the provider, retrying the failed observer, or admitting an extra probe.
- Added slow real-cache adapter regressions for both cache types immediately before, exactly at, and after expiry. Added failures at both pre-dispatch audit events, asserting the typed audit failure, no provider dispatch, cooldown exclusion, single-probe admission, and successful recovery after cooldown.
- RED: the focused regression run produced 6 expected failures and 2 valid-before-expiry passes; the failures reproduced expired cache replay and permanently acquired probes.
- GREEN: the complete driver suite passed with 77 tests. The driver, circuit-breaker, exact-cache, and semantic-cache suites together passed with 147 tests.
- `python -m compileall -q market_agent/workflow_agent_driver.py market_agent_test_bundle/tests/test_workflow_agent_driver.py` and `git diff --check` passed. Git emitted only the existing LF/CRLF conversion notices.
- Scope remained the driver, its regression tests, and this report; no `.tmpbudget` files were read or changed.

Closure commit: `fix: revalidate cache expiry and settle abandoned probes`
