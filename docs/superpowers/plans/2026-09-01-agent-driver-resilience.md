# AgentDriver and Resilience Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a deterministic, auditable LLM driver with safe caches, retry/circuit controls, and ordered fallback.

**Architecture:** Strict contracts isolate Harness from provider calls. The AgentDriver composes retry, circuit, cache, and fallback policies through injected adapters; it has no persistence, exchange, or memory-write authority.

**Tech Stack:** Python 3.13, Pydantic v2, pytest, existing workflow audit/contracts.

**Spec:** `docs/superpowers/specs/2026-09-01-agent-driver-resilience-design.md`

## Global Constraints

- Preserve `trace_id` in every result and audit event.
- New driver modules must not import Harness persistence, execution, exchange, queue, or memory repositories.
- Cache only safe read-only categories; semantic reuse requires similarity strictly greater than `0.95`.
- Stable prompt release system prefix comes first; dynamic context is canonical JSON user content.
- Malformed output, unsafe metadata, and exhausted limits fail closed.

## Prompt Release Activation Note

`workflow_prompt_config.py` loads only Git-tracked, hash-validated release
manifests. The local SQLite pointer supports atomic activation and a one-action
rollback to the immediately preceding release. Calls pin an immutable release
before dispatch, so an in-flight request is unaffected by later activation.
Optional gate, audit, and metric hooks receive immutable metadata only.

## Redis Adapter Note

`backend/redis_adapters.py` keeps Redis optional and injected. Tenant-qualified
cache keys use bounded, redacted JSON plus TTL and idempotent `SET NX`; stream
messages preserve trace identity and support explicit consume, acknowledgement,
and dead-letter paths. Backend failures are surfaced as unavailable rather than
silently falling back to cross-tenant or in-memory data.

---

### Task 1: Contracts and Prompt Releases

**Files:**
- Create: `market_agent/workflow_agent_contracts.py`
- Create: `market_agent/workflow_prompt_release.py`
- Test: `market_agent_test_bundle/tests/test_workflow_agent_contracts.py`
- Test: `market_agent_test_bundle/tests/test_workflow_prompt_release.py`

**Interfaces:** `AgentInvocation`, `AgentResult`, `AgentFailure`, `ModelTier`, `PromptRelease`, and `PromptReleaseRegistry.render(invocation) -> tuple[str, str]`.

- [x] **Step 1: Write failing strict-model tests.**

```python
def test_invocation_rejects_missing_trace_and_unbounded_limits():
    with pytest.raises(ValidationError):
        AgentInvocation(trace_id="", max_attempts=0, cost_limit_usd=-1)

def test_result_rejects_raw_text_and_extra_fields():
    with pytest.raises(ValidationError):
        AgentResult.model_validate({"raw_text": "```json {} ```"})
```

- [ ] **Step 2: Run `python -m pytest market_agent_test_bundle/tests/test_workflow_agent_contracts.py -q`; confirm it fails before implementation.**

- [x] **Step 3: Implement immutable strict contracts and canonical rendering.**

```python
class PromptRelease(StrictModel):
    release_id: ShortText
    digest: Digest
    stable_system_prefix: ShortText

def render(self, invocation: AgentInvocation) -> tuple[str, str]:
    return self.stable_system_prefix, canonical_json(invocation.user_payload)
```

- [ ] **Step 4: Add prefix-order, dynamic-system-value denial, and unsupported-tier tests; run both contract tests.**

- [x] **Step 5: Commit `feat: add agent driver contracts and prompt releases`.**

### Task 2: Retry and Circuit Policies

**Files:**
- Create: `market_agent/workflow_retry_policy.py`
- Create: `market_agent/workflow_circuit_breaker.py`
- Test: `market_agent_test_bundle/tests/test_workflow_retry_policy.py`
- Test: `market_agent_test_bundle/tests/test_workflow_circuit_breaker.py`

**Interfaces:** `RetryPolicy.decide(error, attempt, deadline, remaining_cost, now, random)` and `CircuitBreaker.acquire(model, task_kind, now)` / `record(..., success, now)`.

- [x] **Step 1: Write failing retry and breaker tests.**

```python
def test_retry_uses_full_jitter_but_stops_before_deadline():
    assert policy.decide(timeout(), 1, deadline=1.0, now=0.99, remaining_cost=1).terminal

def test_breaker_admits_only_one_half_open_probe():
    assert breaker.acquire("luna", "extract", now=11).kind == "probe"
    assert breaker.acquire("luna", "extract", now=11).kind == "reject"
```

- [ ] **Step 2: Run both test files; confirm failure.**

- [x] **Step 3: Implement closed error classification, `uniform(0, min(cap, base * 2**attempt))`, and model/task-keyed states.**

- [ ] **Step 4: Add 408/409/429/5xx retry, auth/schema denial, Retry-After, cost stop, cooldown/reopen, and isolation tests; rerun.**

- [x] **Step 5: Commit `feat: add retry and circuit policies`.**

### Task 3: Safe Cache and Fallback

**Files:**
- Create: `market_agent/workflow_response_cache.py`
- Create: `market_agent/workflow_semantic_request_cache.py`
- Create: `market_agent/workflow_fallback.py`
- Create: `market_agent/local_knowledge_base.py`
- Test: `market_agent_test_bundle/tests/test_workflow_response_cache.py`
- Test: `market_agent_test_bundle/tests/test_workflow_semantic_request_cache.py`
- Test: `market_agent_test_bundle/tests/test_workflow_fallback.py`

**Interfaces:** `ExactResponseCache.get/put`, `SemanticRequestCache.lookup(query, metadata, now)`, and `FallbackPolicy.next(current_tier, failure)`.

- [x] **Step 1: Write failing cache and fallback tests.**

```python
def test_similarity_at_threshold_is_a_miss():
    assert cache.lookup(vector(0.95), metadata(), now=1) is None

def test_trade_decision_cannot_enter_any_cache():
    with pytest.raises(CacheSafetyError): cache.put(trade_decision())

def test_fallback_ends_with_exact_abstention():
    assert fallback.next(None, unavailable()) == Abstain("不知道")
```

- [ ] **Step 2: Run the three cache/fallback test files; confirm failure.**

- [x] **Step 3: Implement TTL metadata gates, deterministic ranking, local knowledge citations, and one-way downgrade.**

```python
eligible = similarity > 0.95 and entry.expires_at > now and entry.metadata == metadata
return sorted(eligible, key=lambda e: (-e.similarity, e.created_at, e.entry_id))[0]
```

- [ ] **Step 4: Add expiry cleanup, tenant/release/schema/model mismatch, fixed-seed, citation, and no-upgrade tests; rerun.**

- [x] **Step 5: Commit `feat: add safe response caches and fallback policy`.**

### Task 4: Compose and Audit AgentDriver

**Files:**
- Create: `market_agent/workflow_agent_driver.py`
- Modify: `market_agent/workflow_audit.py`
- Test: `market_agent_test_bundle/tests/test_workflow_agent_driver.py`

**Interfaces:** `AgentDriver.execute(invocation) -> AgentResult`; injected `ModelClient`, `AuditObserver`, clock, and randomness only.

- [x] **Step 1: Write failing ordered-flow tests.**

```python
def test_driver_prefers_safe_cache_and_preserves_trace(observer):
    assert driver.execute(invocation()).origin == "semantic_cache"
    assert observer.events[-1].trace_id == invocation().trace_id

def test_driver_rejects_malformed_provider_output_before_fallback():
    assert driver.execute(invocation()).failure.code == "malformed_output"
```

- [ ] **Step 2: Run `python -m pytest market_agent_test_bundle/tests/test_workflow_agent_driver.py -q`; confirm failure.**

- [x] **Step 3: Implement the ordered cache → model/retry/circuit → downgrade → knowledge → abstain path, emitting redacted trace-bound audit events.**

- [ ] **Step 4: Add strict-schema, circuit/downgrade, unsafe-cache, trace, and forbidden-import tests; run all seven new test files.**

- [x] **Step 5: Commit `feat: compose auditable resilient agent driver`, then run compileall, workflow-targeted pytest, and `git diff --check`.**
