# Production Workflow Closure Design

## Scope

This design closes the production gaps found in the 2026-09-02 static review. Work is split into independently releasable phases so every phase leaves a coherent, testable system:

1. result delivery and historical-cache correctness;
2. trusted result commit and cancellation propagation;
3. Harness-visible node execution and authoritative usage settlement;
4. prompt pinning, memory promotion, correction, distributed execution, and unified observability.

The first implementation phase must not weaken the existing no-trade and fail-closed boundaries.

## Result contract

`HarnessWorkflowExecution` remains the internal execution envelope. The queue adapter must preserve a public `workflow_result` instead of discarding it. A candidate may become public only after the Harness reaches `SUCCEEDED`. For `DEGRADED`, `FAILED`, or `CANCELLED`, the adapter returns a deterministic `UNKNOWN`/`NO_TRADE` result created without copying model output.

The workflow status resource will expose the durable result separately from transport/job status. Queue completion means the handler finished; workflow state remains the business outcome. Clients must not infer business success from the queue status.

## Historical answer cache

Cache lookup is a staged gate:

1. normalized exact fixed-seed match;
2. request-class admission for static informational queries;
3. vector recall with cosine similarity strictly greater than `0.95`;
4. deterministic intent, keyword, negation, number, and key-entity compatibility;
5. immutable compatibility fields: tenant, model, embedding, prompt, schema, safety policy, locale, context, and knowledge fingerprints;
6. expiry and invalidation checks.

Dynamic fields such as `expires_at`, evidence references, and invalidation reason are never part of PostgreSQL metadata equality. PostgreSQL and in-memory adapters must apply the same compatibility contract. Static informational admission is explicit and independent of passive trading-event mode.

## Trusted commit and cancellation

Model execution produces a candidate only. Memory writes and historical-cache writes are moved behind a host-owned commit operation invoked after final Harness acceptance. Degraded, failed, and cancelled runs cannot commit candidate-derived memory or cache entries.

Cancellation uses a run-scoped signal shared by the API, queue handler, Harness application, LangGraph runtime, and model adapter. It is checked before every graph node, retry, provider call, reflection, and commit. Cancellation cannot terminate an already-sent provider request synchronously, but it prevents further nodes, retries, and durable candidate side effects.

## Harness integration

The Harness must observe the real workflow rather than one opaque function call. Each core graph node reports a typed checkpoint containing run/trace/plan revision, task identity, progress, action fingerprint, retry state, and cumulative usage. Harness policy remains the sole authority for stop/degrade/continue decisions.

The final completion candidate includes authoritative aggregate input/output/cached tokens, model versions, attempt counts, latency, and settled cost. The fixed `1 + 1` token settlement is removed. Harness plan work items must correspond to the actual coordinator and specialist task inventory.

## Prompt, memory, and correction

One immutable prompt pin is captured at workflow ingress and passed to every coordinator, specialist, retry, and reflection call. Agent-specific stable prefixes are components of the pinned release instead of being replaced by one generic prompt.

Accepted outcomes are initially recorded as verified final outcomes. Knowledge promotion remains a separate host process requiring corroboration and provenance. Objective correction first attempts a bounded field patch using the verifier error context; one full rewrite is allowed only if the patch fails or does not objectively improve the error tuple.

## Distributed production and observability

Redis Streams becomes the durable dispatch transport rather than notification-only infrastructure. Job state, prompt activation state, audit events, and workflow result state use shared production storage. Local SQLite remains development-only.

One trace query aggregates HTTP ingress, queue, Harness, coordinator, Agent, model, cache, memory, and commit events by the same trace ID. Readiness performs real dependency probes, including PostgreSQL, Redis, prompt registry, Harness host bindings, completion evidence issuer, and model configuration.

## Verification

Every behavior change follows red-green-refactor. The first phase adds regression tests for PostgreSQL cache compatibility, hybrid query gating, result preservation, safe terminal result exposure, and prevention of pre-acceptance writes. Later phases add cancellation, per-node Harness checkpoint, actual usage settlement, prompt pin stability, memory promotion, and distributed recovery tests.
