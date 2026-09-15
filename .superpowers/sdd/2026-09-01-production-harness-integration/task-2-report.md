# Task 2 — Production Coordinated Application Report

## Outcome

Implemented a host-owned, lazy `ProductionWorkflowApplication` and made it the production container default while preserving the explicitly injected legacy engine path. Model execution is no longer serialized behind the service construction lock.

Commit hash: recorded in the task status contract because a commit cannot contain its own final hash.

## Architecture

- `ProductionWorkflowApplication` admits or creates one nonzero 32-hex trace ID, binds the configured tenant separately, builds a strict `WorkflowRequest`, derives active/passive mode only from `trigger_reason == "passive_event_trigger"`, constructs a per-request driver/coordinator runtime, invokes `CoordinatedWorkflow`, and converts the result to a non-executing `GenericPlaybook` plus deterministic JSON report.
- Lazy construction owns prompt releases, audit storage, OpenAI model/embedding adapters, exact cache, injected semantic cache, retry/circuit/fallback policy, and bounded model reservations. Imports and backend creation do not call OpenAI.
- Each request gets its own `AgentDriver`, capability issuer, coordinator service, deadline, contexts, grants, and memory binding. Shared construction/caches/circuit state are created once; the only application/service lock covers lazy construction.
- The driver uses the Git-tracked `default_prompt_manager`, every specialist output schema plus the reflection schema, fixed Sol/Terra/Luna model IDs with separate version identifiers, exponential full-jitter retry, circuit breaker, ordered Sol→Terra→Luna→verified local knowledge→`不知道` fallback, exact safe cache, and injected PostgreSQL semantic cache.

## Exact Files

- Created `market_agent/workflow_production_application.py`.
- Created `market_agent/workflow_embedding_client.py`.
- Modified `market_agent/workflow_agents/common.py`, `market_agent/workflow_coordinator_agent.py`, and `market_agent/workflow_service_factory.py` to pass only a validated `CoreExperienceSummary` and its trusted tenant/scope through the existing driver boundary.
- Modified `market_agent/workflow_coordinator_services.py` to bind decision synthesis to Terra, coordinator memory context, coordinator-only context access, and Luna objective reflection.
- Modified `market_agent/workflow_agent_driver.py` to retain safe exact/semantic cache admission already present in the coordinated worktree.
- Modified `market_agent/workflow_openai_client.py` and `market_agent/langchain_runtime.py` for versioned per-tier model selection and configured temperature forwarding.
- Modified `market_agent/workflow_graph.py` and added `market_agent/workflow_decision_verifier.py` for the mandatory fail-closed objective reflection boundary.
- Modified `market_agent/backend/agent_service.py`, `market_agent/backend/container.py`, and `market_agent/backend/settings.py` for lazy production default wiring and bounded versioned configuration.
- Included `market_agent/workflow_semantic_cache_postgres.py`, the injected pgvector semantic cache implementation required by the production composition.

## Trace, Permissions, Memory, and Reflection

- A valid caller-supplied trace ID is preserved; otherwise one is generated exactly once before runtime work. Workflow request, tasks, summaries, grants, driver calls, audit events, result, completion hook, and report retain that ID. Tenant identity is validated against the configured host scope and is never conflated with trace identity.
- Context is normalized into `ContextRecord`, bounded with `select_context`, and projected with `summarize_context`. Runtime timestamps and request/market content remain in dynamic user context, not the stable prompt prefix. Each task receives only its own summary.
- Recovery summaries preserve bounded successful conclusions and add only exact directive/error/conflict codes. They do not include exceptions, credentials, repositories, connections, or service handles; successful reports remain in the coordinator recovery set.
- Specialist/coordinator grants can read only `context_summary`; reflection grants can read only `reflection_target`. No grant contains tools, service calls, state/durable writes, database/cache/memory/audit/queue access, exchange/order authority, or another task's context.
- Before dispatch, the host builds a versioned `MemoryQuery`, calls `retrieve_memory`, bounds the result with `build_core_experience_summary`, and injects only a clear, unexpired, non-conflicting summary whose tenant/scope matches. Miss, conflict, embedding/repository failure, or invalid/expired memory yields no injection and no confidence increase. Agents receive no repository or writer handle.
- The trace-bound completion hook is present. The container currently supplies a host no-op projection; Task 4 can replace it with governed outcome/promotion persistence.
- Decision synthesis runs on Terra. Objective reflection runs on Luna, checks only the declared objective invariants, permits at most one allowlisted patch followed by one full rewrite, stops on repeated hash/non-improvement, carries exact correction error codes, and degrades to safe unknown/no-trade if verification is unavailable or fails.
- Unknown and no-trade workflow results render exactly `不知道` and use the empty entry/position defaults. Valid playbook results carry a candidate decision but always set `execute_now=False`, zero notional/margin/loss/leverage, and an empty target/management plan.
- Construction/configuration and required-audit failures propagate rather than masquerading as a successful playbook. Runtime evidence/model/memory unavailability may produce the deterministic safe unknown result.

## Static Verification

No tests or reviews were run, per instruction.

1. `python -m compileall market_agent/workflow_production_application.py market_agent/workflow_embedding_client.py market_agent/workflow_coordinator_services.py market_agent/workflow_service_factory.py market_agent/workflow_coordinator_agent.py market_agent/workflow_agents/common.py market_agent/workflow_openai_client.py market_agent/backend/agent_service.py market_agent/backend/container.py market_agent/backend/settings.py`
   - Output: every named module compiled successfully.
2. `python -c "import market_agent.workflow_production_application as app; import market_agent.workflow_embedding_client; import market_agent.backend.agent_service; import market_agent.backend.container; import market_agent.backend.settings; assert app.ProductionWorkflowApplication"`
   - Output: success with no stdout; no network or database call was made.
3. `git diff --check` on all task files, followed by `git diff --cached --check` on the exact staged task set.
   - Output: no whitespace errors; Git emitted only the worktree's existing LF/CRLF conversion notices.

## Concerns / Deferred Boundaries

- Tests and code review were explicitly deferred, so validation is static only.
- Task 3 owns Harness lifecycle/API-route integration. The production application accepts queue `trace_id`/`tenant_id` now, but Harness authority and route behavior remain Task 3 work.
- Task 4 owns governed outcome/promotion persistence, operational cleanup/telemetry completion, and replacement of the current trace-bound no-op completion callback.
