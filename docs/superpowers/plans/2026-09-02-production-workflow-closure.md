# Production Workflow Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close the reviewed production gaps so Harness policy governs real Agent execution and only trusted, retrievable, correctly cached results leave the workflow.

**Architecture:** Deliver the repair in four independently testable phases. First fix public result and cache contracts, then move side effects behind Harness acceptance, expose real graph checkpoints and usage to Harness, and finally complete prompt/memory/distributed-observability wiring.

**Tech Stack:** Python 3.11+, Pydantic v2, FastAPI, LangGraph, PostgreSQL/pgvector, Redis Streams, pytest.

**Spec:** `docs/superpowers/specs/2026-09-02-production-workflow-closure-design.md`

## Global Constraints

- Preserve fail-closed `UNKNOWN`/`NO_TRADE` behavior.
- Similarity must be strictly greater than `0.95`.
- Dynamic metadata fields must not participate in cache compatibility equality.
- Candidate-derived cache and memory writes require final Harness acceptance.
- One trace ID and one prompt pin must remain stable for the whole workflow.
- Provider calls, retries, and graph nodes remain bounded by deadline, attempts, and cost.

---

### Task 1: Correct PostgreSQL historical-cache compatibility

**Files:**
- Modify: `market_agent/workflow_historical_answer_cache.py`
- Test: `market_agent_test_bundle/tests/test_workflow_historical_answer_cache.py`

**Interfaces:**
- Consumes: `HistoricalAnswerMetadata.compatible_with(other)`.
- Produces: PostgreSQL lookup using only immutable compatibility fields plus explicit TTL/invalidation predicates.

- [x] **Step 1: Add a failing adapter test** proving a stored row is returned when expiry and evidence references differ but immutable metadata and similarity match.
- [x] **Step 2: Run the targeted test** and confirm the current `metadata=%s::jsonb` predicate causes a miss.
- [x] **Step 3: Replace whole-JSON equality** with explicit immutable metadata predicates while retaining tenant, TTL, invalidation, and `> 0.95` vector gates.
- [x] **Step 4: Run the historical-cache tests** and confirm PostgreSQL and in-memory compatibility semantics agree.

### Task 2: Add hybrid informational-cache admission

**Files:**
- Modify: `market_agent/workflow_historical_answer_cache.py`
- Modify: `market_agent/workflow_production_application.py`
- Test: `market_agent_test_bundle/tests/test_workflow_historical_answer_cache.py`
- Test: `market_agent_test_bundle/tests/test_workflow_production_application.py`

**Interfaces:**
- Produces: `HistoricalQueryFingerprint` and `compatible_query(candidate, request)` deterministic gates.
- Produces: explicit static informational admission independent of passive event mode.

- [x] **Step 1: Add failing tests** for paraphrase hit, negation mismatch, different symbols, different numbers/timeframes, and ordinary active informational requests.
- [x] **Step 2: Run the tests** and verify current semantic-only/passive-only admission fails them.
- [x] **Step 3: Implement normalized intent, keyword, entity, number, and negation fingerprints** and derive them from the stored and incoming request text.
- [x] **Step 4: Apply hybrid gating after vector recall** in both adapters and update the production admission classifier.
- [x] **Step 5: Run targeted cache and production tests.**

### Task 3: Preserve and expose trusted workflow results

**Files:**
- Modify: `market_agent/backend/api_contracts.py`
- Modify: `market_agent/backend/harness_service.py`
- Modify: `market_agent/backend/api.py`
- Modify: `market_agent/workflow_harness_application.py`
- Test: `market_agent_test_bundle/tests/test_backend_api.py`
- Test: `market_agent_test_bundle/tests/test_workflow_harness_application.py`

**Interfaces:**
- Produces: `WorkflowExecutionResult` public contract containing Harness state and trusted `WorkflowResult`.
- Produces: deterministic unknown/no-trade result for non-success terminal states.

- [x] **Step 1: Add failing tests** showing a successful result survives the queue adapter and a degraded result never exposes candidate model output.
- [x] **Step 2: Run the targeted tests** and confirm the result is currently discarded.
- [x] **Step 3: Add the typed result field** to task and workflow status projections.
- [x] **Step 4: Preserve successful results and synthesize safe terminal results** for degraded/failed/cancelled workflows.
- [x] **Step 5: Run API, queue-adapter, and Harness-application tests.**

### Task 4: Commit side effects only after Harness acceptance

**Files:**
- Modify: `market_agent/workflow_production_application.py`
- Modify: `market_agent/workflow_harness_application.py`
- Modify: `market_agent/backend/container.py`
- Test: `market_agent_test_bundle/tests/test_workflow_production_application.py`
- Test: `market_agent_test_bundle/tests/test_workflow_harness_application.py`

**Interfaces:**
- Produces: candidate execution separated from idempotent `commit_accepted_result(request, result)`.

- [x] **Step 1: Add failing tests** proving memory/cache hooks are not called before acceptance or for degraded/cancelled runs.
- [x] **Step 2: Run tests** and confirm current eager writes fail.
- [x] **Step 3: Move completion and historical-cache writes** into an idempotent host-owned commit callback.
- [x] **Step 4: Invoke commit only after `RunState.SUCCEEDED`.**
- [x] **Step 5: Run result-writer, cache, and Harness integration tests.**

### Task 5: Propagate cancellation through queue, graph, and provider boundaries

**Files:**
- Modify: `market_agent/backend/task_queue.py`
- Modify: `market_agent/backend/api.py`
- Modify: `market_agent/workflow_harness_application.py`
- Modify: `market_agent/workflow_graph.py`
- Modify: `market_agent/workflow_agent_driver.py`
- Test: corresponding backend, graph, Harness, and driver test modules.

**Interfaces:**
- Produces: run-scoped `CancellationSignal.is_cancelled() -> bool`.

- [ ] **Step 1: Add failing cancellation tests** at queued, between-node, before-retry, and before-commit boundaries.
- [ ] **Step 2: Implement the run-scoped cancellation registry and signal.**
- [ ] **Step 3: Check cancellation at every bounded execution boundary.**
- [ ] **Step 4: Verify cancellation prevents later calls and durable candidate side effects.**

### Task 6: Bind Harness checkpoints and actual usage to LangGraph execution

**Files:**
- Modify: `market_agent/workflow_graph.py`
- Modify: `market_agent/workflow_harness_application.py`
- Modify: `market_agent/workflow_harness.py`
- Modify: `market_agent/workflow_contracts.py`
- Modify: `market_agent/workflow_coordinator_agent.py`
- Test: graph, Harness, loop-guard, budget, and coordinator tests.

**Interfaces:**
- Produces: typed node checkpoint observer and aggregate `WorkflowUsage`.

- [ ] **Step 1: Add failing tests** that require one Harness checkpoint per core node and exact aggregate usage settlement.
- [ ] **Step 2: Add typed usage/checkpoint contracts.**
- [ ] **Step 3: Emit checkpoints from plan, dispatch, recovery, decision, reflection, risk, and assembly nodes.**
- [ ] **Step 4: Replace fixed Harness usage with aggregate provider usage.**
- [ ] **Step 5: Make Harness work-item inventory correspond to actual coordinator tasks.**
- [ ] **Step 6: Run replay, loop, budget, graph, and integration tests.**

### Task 7: Pin prompts, complete correction and memory promotion

**Files:**
- Modify: `market_agent/workflow_production_application.py`
- Modify: `market_agent/workflow_agent_driver.py`
- Modify: `market_agent/workflow_coordinator_services.py`
- Modify: `market_agent/workflow_memory_result_writer.py`
- Modify: `market_agent/workflow_memory_promotion.py`
- Test: prompt, reflection, correction, memory lifecycle, and production tests.

**Interfaces:**
- Produces: ingress `PromptPin`, deterministic patch generator, verified outcome commit, and bounded promotion scheduler.

- [ ] **Step 1: Add failing tests** for mid-workflow prompt activation, successful objective patch, rewrite fallback, verified outcome, and promotion eligibility.
- [ ] **Step 2: Pass one prompt pin through all Agent invocations.**
- [ ] **Step 3: Implement verifier-error-driven field patching with objective improvement checks.**
- [ ] **Step 4: Record verified accepted outcomes and trigger host-controlled promotion evaluation.**
- [ ] **Step 5: Run prompt, reflection, and memory tests.**

### Task 8: Complete distributed execution, unified tracing, tools, knowledge, and evaluation gate

**Files:**
- Modify: backend container, queue, Redis adapters, readiness, observability, and API modules.
- Modify: production capabilities, OpenAI client, and local knowledge composition.
- Create: deployment configuration and evaluation CLI/CI workflow.
- Test: distributed recovery, trace aggregation, authorization, tool, knowledge fallback, and evaluation modules.

**Interfaces:**
- Produces: Redis-backed durable dispatch worker, shared production authorities, unified trace repository, explicit tool capabilities, populated knowledge provider, and executable evaluation quality gate.

- [ ] **Step 1: Add failing integration tests** for cross-process recovery and unified trace retrieval.
- [ ] **Step 2: Implement Redis durable dispatch and shared production state.**
- [ ] **Step 3: Add real readiness probes and admin authorization scope.**
- [ ] **Step 4: Wire bounded tools and configured local knowledge through capability grants.**
- [ ] **Step 5: Add evaluation CLI and CI quality gate using the versioned dataset.**
- [ ] **Step 6: Run the complete verification and security suites, synchronize both repositories, and commit.**
