# Production Harness Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the production wiring for the deterministic Harness, coordinated LangGraph agents, safe caching, governed memory, observability, and both-repository delivery.

**Architecture:** Keep `HarnessKernel` as the sole lifecycle authority and place the coordinated LangGraph workflow behind a production application adapter. Build all dynamic context from bounded source facts and governed memory summaries, run objective reflection only at core decision/finalization boundaries, and keep cache/memory writes in host-owned services. The existing synchronous playbook contract remains an adapter over the new application.

**Tech Stack:** Python 3.13, Pydantic v2, LangGraph, OpenAI/LangChain, PostgreSQL/pgvector, Redis Streams, SQLite WAL.

**Spec:** `docs/superpowers/specs/2026-08-30-deterministic-harness-control-design.md`

## Global Constraints

- `HarnessKernel` remains the only authority for plans, transitions, retry/degradation authorization, and terminal state.
- LLM output is validated content only and never selects graph edges, permissions, durable writes, budgets, or terminal status.
- Every request uses one nonzero 128-bit W3C trace ID across API, queue, Harness, workers, model calls, cache, memory, audit, and response.
- Trade decisions are never read from or written to response caches; direct cache hits are limited to compatible safe informational answers with cosine similarity strictly greater than `0.95`.
- Core reflection is objective, Luna-only, bounded to patch then one rewrite, and fails closed.
- Agents receive bounded summaries and capability grants; durable cache, memory, queue, audit, and database writes remain host-owned.
- Prompt/model/schema/embedding/safety versions and expiry metadata gate cache and memory reuse.
- The user requested no test execution in this phase; verification is limited to `compileall`, import smoke checks, and `git diff --check`.

---

### Task 1: Close Current Adapter Correctness and Production Dependencies

**Files:**
- Modify: `market_agent/workflow_service_factory.py`
- Modify: `market_agent/workflow_audit.py`
- Modify: `market_agent/backend/redis_adapters.py`
- Modify: `requirements.txt`

**Interfaces:** Preserve existing public types; ensure recovery budget copies from `TradingWorkflowState`, audit cursors accept W3C trace IDs, Redis consumers recover from transient read failures, and production drivers are installable.

- [x] Fix recovery budget state copying and W3C audit cursor validation.
- [x] Make the Redis consumer loop reconnect/back off without losing group semantics and expose health failure state.
- [x] Add bounded compatible `redis` and `psycopg[binary]` requirements.
- [x] Run `python -m compileall market_agent` and `git diff --check`; do not run tests.
- [x] Commit with author `Du Du <v6hit7cd@gmail.com>`.

### Task 2: Compose the Production Coordinated Application

**Files:**
- Create: `market_agent/workflow_production_application.py`
- Modify: `market_agent/workflow_coordinator_services.py`
- Modify: `market_agent/workflow_agent_driver.py`
- Modify: `market_agent/backend/container.py`
- Modify: `market_agent/backend/agent_service.py`

**Interfaces:** Produce `ProductionWorkflowApplication.get_playbook(...) -> tuple[GenericPlaybook, str]`; consume `AgentDriver`, `AgentCoordinatorServices`, `CoordinatedWorkflow`, prompt releases, semantic cache, memory repositories, and audit services.

- [x] Build stable source facts, bounded per-agent context summaries, capability grants, model routing, budget/retry/circuit policies, and mandatory decision reflection.
- [x] Query governed memory before dispatch and inject only `CoreExperienceSummary`; promote eligible outcomes through host authority after completion.
- [x] Query safe exact/semantic answer cache before informational execution and persist only cache-safe informational results with complete version/expiry metadata.
- [x] Preserve the legacy synchronous playbook adapter while making the new application the default container service.
- [x] Run static compilation/import/diff checks only and commit.

### Task 3: Extend Harness Control and API Wiring

**Files:**
- Modify: `market_agent/workflow_plan_registry.py`
- Modify: `market_agent/workflow_harness.py`
- Create: `market_agent/workflow_harness_application.py`
- Modify: `market_agent/backend/api.py`
- Modify: `market_agent/backend/api_contracts.py`
- Modify: `market_agent/backend/container.py`

**Interfaces:** Produce a host-owned `HarnessWorkflowApplication` that creates/adopts a run, advances committed Harness transitions, invokes the production workflow only while RUNNING, records signed observations, and seals a deterministic terminal result.

- [x] Compile active and passive plans only from trusted request fields, with immutable worker/model/prompt versions and no agent side effects.
- [x] Remove Phase-1 passive-only assumptions while retaining event-fold replay, budget, LoopGuard, confidence, and safe degradation invariants.
- [x] Route `POST /v1/workflows`, status, cancellation, and event reads through Harness-owned run state while preserving the existing generate-playbook endpoint as a compatibility adapter.
- [x] Ensure coordinator errors/conflicts preserve successful reports, carry bounded failure context, and trigger only one authorized replan before safe degradation.
- [x] Run static compilation/import/diff checks only and commit.

### Task 4: Finish Memory/Cache Operations, Prompt Rollback, Observability, and Delivery

**Files:**
- Modify: `market_agent/backend/memory_maintenance.py`
- Modify: `market_agent/backend/container.py`
- Modify: `market_agent/workflow_prompt_config.py`
- Modify: `market_agent/workflow_observability.py`
- Create: `market_agent/workflow_evaluation_dataset.py`
- Create: `evaluation/workflow_acceptance.jsonl`
- Modify: `README.md`

**Interfaces:** Provide recurring lifecycle cleanup, cache expiry cleanup, auditable prompt activation/rollback with pending-delivery recovery, trace-linked structured telemetry, and a versioned evaluation corpus loader without executing tests.

- [x] Schedule memory decay/archive/tombstone/purge and semantic-cache expiry cleanup with observable failures and safe shutdown.
- [x] Persist prompt release audit/outbox state so failed telemetry can be replayed; expose atomic rollback for new runs.
- [x] Ensure structured logs/metrics/spans cover API, queue, Harness, agent/model/tool, cache, memory, retry/circuit, and final outcome using trace exemplars rather than high-cardinality labels.
- [x] Add a schema-validated, versioned JSONL acceptance corpus with active/passive, cache, memory, reflection, loop, retry/circuit, trace, permissions, prompt rollback, and degradation cases; do not run it.
- [ ] Perform static verification and review, synchronize equivalent tracked content to `multi-agent-trader`, commit both repositories with `Du Du <v6hit7cd@gmail.com>`, and push both remotes as already authorized by the user.
