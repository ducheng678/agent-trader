# AgentDriver and Resilience Runtime Design

## Scope

This phase adds the bounded runtime path between a Harness work item and an LLM
result. It does not change Harness authority, execution permissions, long-term
memory persistence, RAG ingestion, or public API/database contracts. Those are
later phases.

The phase delivers six independently testable modules:

1. `workflow_agent_driver.py` owns every LLM invocation. It accepts a pinned
   request contract, builds stable prompt prefixes from a prompt release,
   chooses a permitted model tier, validates one strict structured response,
   and returns normalized usage or a typed failure.
2. `workflow_retry_policy.py` classifies transport/provider failures and
   produces capped exponential full-jitter delays. It enforces deadline,
   attempt, and reserved-cost limits before scheduling a retry.
3. `workflow_circuit_breaker.py` keeps isolated closed/open/half-open state by
   `(model, task_kind)`. It permits at most one probe after cooldown and emits
   typed decisions for audit.
4. `workflow_response_cache.py` owns exact fixed-seed and response-cache
   entries. Only explicitly safe, read-only answer categories are cacheable;
   trade decisions, orders, credentials, and tool side effects are never
   cacheable.
5. `workflow_semantic_request_cache.py` stores a request vector, response, and
   immutable metadata. A reusable response requires strictly greater than
   `0.95` similarity plus equal tenant, prompt release, output schema, model
   compatibility policy, and unexpired TTL.
6. `workflow_fallback.py` makes the sole ordered fallback decision: lower model
   tier, then local knowledge base, then a schema-valid abstention with the
   exact user-facing conclusion `"不知道"`.

Reflection is defined in the public contracts in this phase but implemented in
the following phase. The driver exposes one core-result verification hook; it
cannot alter an authoritative Harness transition or write memory.

## Contracts and Authority

`AgentInvocation` is immutable and contains `trace_id`, `run_id`, `task_id`,
task kind, prompt release ID and digest, allowed model tier, deadline, attempt
limit, cost ceiling, output schema ID/digest, and canonical user payload.

`AgentResult` is immutable and contains either a validated JSON object and
usage, or one typed failure. Raw provider text never leaves the driver. The
driver rejects prose wrappers, markdown fences, extra keys, non-JSON values,
and outputs that fail the declared schema.

Only the Harness may create an invocation. The driver has no event-store,
execution-backend, memory-repository, exchange, queue, or database write
dependency. It receives capability-scoped adapters for model invocation,
clock, randomness, audit emission, exact cache, semantic cache, circuit
breaker, and local knowledge lookup. The adapters have no ambient globals.

Every module receives and emits the caller's `trace_id`. Cache hit, cache miss,
retry, circuit state change, model downgrade, local-knowledge fallback, and
abstention each produce an audit record through the injected observer. Audit
payloads contain hashes and redacted metadata only; no prompt body, provider
secret, or tool credential is recorded.

## Prompt and Model Policy

Prompt releases are Git-tracked immutable files. A release supplies the stable
system prefix, schema version, supported task kinds, model-temperature profile,
and release digest. The driver sends the stable system prefix first and appends
dynamic task/context as JSON user content so provider prompt-cache prefixes are
stable. Dynamic values never enter the system prefix.

Routing is deterministic from task difficulty and allowed policy:
`luna` for simple extraction/validation, `terra` for normal analysis, and
`sol` for coordinator conflict resolution. A retry keeps its current model.
Fallback may only move downward one tier at a time. An unavailable or open
model never causes an upward escalation.

Every generated prompt tells the model to reason step by step internally,
return only the declared JSON, and answer `"不知道"` when evidence is
insufficient. The chain-of-thought is neither requested nor retained.

## Retry, Circuit, and Cost Rules

Retryable failures are timeout, connection failure, HTTP 408, 409, 429, and
5xx. Authentication, authorization, validation, schema, safety, and malformed
response failures are non-retryable. Retry delay is
`uniform(0, min(cap, base * 2**attempt))`, respecting provider `Retry-After`
when it is longer. No retry is scheduled when it would exceed deadline, attempt
limit, or remaining cost reservation.

The circuit breaker opens after the configured consecutive retryable failures,
remains open until cooldown, then admits one half-open probe. A successful
probe closes it; a failed probe reopens it. Breaker state is isolated by model
and task kind, and failed cache/knowledge operations never change its state.

## Cache and Expiry Rules

Fixed seeds are versioned, safe, read-only responses with explicit categories
and TTL. Exact cache keys include tenant scope, canonical request hash, prompt
release digest, output schema digest, model compatibility key, and answer
category.

Semantic entries additionally retain vector dimension/version, request time,
expiry time, model version, prompt release digest, schema digest, tenant scope,
category, and response digest. Lookup uses deterministic descending similarity
then creation-time/id tie breaking. It returns only entries with similarity
strictly above `0.95`; equality is a miss. Expired entries are unavailable and
removable by an idempotent cleanup pass.

Neither cache can contain trading decisions, order instructions, volatile
market assertions, personally sensitive content, secrets, or tool results.

## Fallback and Reflection Boundary

For an eligible invocation, the ordered flow is: fixed cache, semantic cache,
permitted model call with retries/circuit checks, lower model tier, local
knowledge answer, then `"不知道"`. A cache hit remains subject to schema and
metadata validation. A local-knowledge answer must cite local document IDs and
is never promoted into the semantic cache in this phase.

Core task results are marked `reflection_required`. The later reflection module
may only return an objective disposition: accept, targeted-correction with a
bounded error list, or reject. Correction retries include the errors in dynamic
user context and stop on a configured count or non-improving result. One final
full rewrite is permitted only after targeted correction fails. Reflection
cannot write state, memory, cache, or audit history except its own audit event.

## Verification

Tests must prove strict output parsing, prompt-prefix ordering, release/model
compatibility, retry classification and jitter bounds, deadline/attempt/cost
stops, circuit isolation and half-open probes, unsafe-cache denial, fixed seed
expiry, semantic threshold equality miss, metadata gate, deterministic tie
breaking, expiry cleanup, downgrade order, local-knowledge citation, and exact
`"不知道"` fallback. Tests must also prove every generated record preserves one
trace ID and that no driver module imports Harness persistence, execution, or
exchange services.
