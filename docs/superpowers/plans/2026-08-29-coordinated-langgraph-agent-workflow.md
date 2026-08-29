# Coordinated LangGraph Agent Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a coordinator-controlled LangGraph with bounded specialist agents, summarized handoffs, full-chain audit, three-layer long-term memory, safe caching, and fail-closed degradation.

**Architecture:** The existing engine adapts requests into one typed graph. The coordinator issues only catalogued three-to-five-step tasks, reconciles errors/conflicts, and summarizes. One runner owns model calls, retries, budgets, cache and fallback; deterministic gates own risk, memory promotion, forgetting, and final playbook assembly.

**Tech Stack:** Python, Pydantic v2, LangGraph 1.2.11, existing OpenAI/LangChain runtime, PostgreSQL/pgvector production interfaces, SQLite WAL/FTS5 local runtime, Redis/object-store adapters, Decimal, pytest 9.

**Spec:** `docs/superpowers/specs/2026-08-29-langgraph-multi-agent-trading-workflow-design.md` and `docs/superpowers/specs/2026-08-29-coordinator-audit-context-addendum.md`

## Global Constraints

- Preserve `get_playbook(...) -> tuple[GenericPlaybook, str]` and modes.
- Luna classifies, Terra reasons, and Sol reviews difficult conflicts.
- Stable system prompts precede dynamic summarized context.
- Agent prompts include the approved stepwise-analysis and uncertainty/unknown instruction.
- Enforce active 300s/10 attempts/$0.75 and passive 130s/10/$0.30 caps.
- Degrade through allowed lower models, local knowledge, then `不知道`; trading unknown is `no_trade`.
- Cache no live market or trading result; audit every transition before external calls.
- Agents cannot mutate exchanges or durable memory directly.
- Keep both repositories content-equivalent, histories separate, Python comment-free.

---

### Task 1: Workflow Contracts and State

**Files:** Create `market_agent/workflow_contracts.py`, `market_agent/workflow_state.py`; test `market_agent_test_bundle/tests/test_workflow_contracts.py`.

**Interfaces:** Produce `WorkflowRequest`, `AgentTask`, `AgentReport`, `ContextSummary`, specialist outputs, `CoordinatorPlan`, `WorkflowResult`, `TradingWorkflowState`, and `merge_reports`.

- [ ] **Step 1:** Add failing tests for extra fields, nonfinite values, knowledge/status consistency, three-to-five steps, and duplicate conflicts.
```python
def test_task_step_bounds():
    with pytest.raises(ValidationError):
        AgentTask(task_id=t, task_type=fundamental, objective=x, analysis_steps=[a, b])
```
- [ ] **Step 2:** Run `python -m pytest market_agent_test_bundle/tests/test_workflow_contracts.py -q`; expect import failure.
- [ ] **Step 3:** Implement frozen strict Pydantic models, enums, bounds, cross-field validators, immutable request, and reducers.
- [ ] **Step 4:** Rerun; expect pass.
- [ ] **Step 5:** Commit `feat: add strict workflow contracts`.

### Task 2: Full-Chain Audit and Summary Handoffs

**Files:** Create `workflow_audit.py`, `workflow_context_summary.py`; test `test_workflow_audit.py`, `test_workflow_context_summary.py`.

**Interfaces:** `AuditStore.append/list`, `AuditWriter.record/healthy`, `select_context`, `summarize_context`.

- [ ] **Step 1:** Test monotonic audit sequence, redaction, concurrency, fail-closed dispatch, and summaries preserving numbers, negation, timestamps, sources, uncertainty, hashes, and omissions.
```python
def test_audit_failure_blocks_dispatch():
    writer = AuditWriter(FailingStore())
    with pytest.raises(AuditUnavailableError):
        writer.record(model_dispatch, trace_id=trace)
    assert not writer.healthy
```
- [ ] **Step 2:** Run both focused files; expect missing modules.
- [ ] **Step 3:** Implement WAL/indexes, transactional sequences, recursive redaction, deterministic selection, hashing, bounded truncation, and completeness checks.
- [ ] **Step 4:** Run focused and backend architecture tests; expect pass.
- [ ] **Step 5:** Commit `feat: add audited context handoffs`.

### Task 3: Model Routing and Budget Ledger

**Files:** Create `workflow_model_routing.py`, `workflow_budget.py`; modify `openai_usage.py`; test `test_workflow_budget_routing.py`.

**Interfaces:** `policy_for(node_name)` and thread-safe `WorkflowBudgetLedger.reserve/settle/consume_timeout/snapshot`.

- [ ] **Step 1:** Test exact tier chains, node caps, parallel reservations, timeout charges, attempts, deadlines, and pre-overspend rejection.
```python
def test_terra_downgrades_to_luna():
    assert [x.model for x in policy_for("fundamental").tiers] == ["gpt-5.6-terra", "gpt-5.6-luna"]
```
- [ ] **Step 2:** Run `python -m pytest market_agent_test_bundle/tests/test_workflow_budget_routing.py -q`; expect failure.
- [ ] **Step 3:** Add 5.6 prices, conservative estimates, immutable policies, Decimal locks, monotonic deadlines, reservations, and settlement.
- [ ] **Step 4:** Run focused and `-k usage` unit tests; expect pass.
- [ ] **Step 5:** Commit `feat: enforce workflow model and cost budgets`.

### Task 4: Three-Layer Memory Storage

**Files:** Create `workflow_long_term_memory.py`, `workflow_memory_postgres.py`, `workflow_memory_sqlite.py`, `workflow_object_store.py`; test `test_workflow_memory_storage.py`.

**Interfaces:** `MemoryRepository.append_event/propose_knowledge/activate_knowledge/append_decision/append_outcome/link_lesson/get_by_id`; `ArtifactStore.put/get` returns immutable checksum-addressed references.

- [ ] **Step 1:** Test append-only events, knowledge evidence links, provisional/final decisions, idempotency, checksums, transactions, tenant isolation, and agents lacking write credentials.
```python
def test_knowledge_requires_existing_event_evidence(repo):
    with pytest.raises(MemoryPromotionError):
        repo.activate_knowledge(candidate(evidence_ids=["missing"]))
```
- [ ] **Step 2:** Run `python -m pytest market_agent_test_bundle/tests/test_workflow_memory_storage.py -q`; expect missing modules.
- [ ] **Step 3:** Implement repository protocols; SQLite WAL typed local tables; PostgreSQL DDL for partitioned events, versioned knowledge/pgvector, transactional decisions; local/S3 artifact boundaries.
- [ ] **Step 4:** Run focused tests; keep PostgreSQL/object-store integration opt-in and verify SQLite pass.
- [ ] **Step 5:** Commit `feat: add three-layer long-term memory storage`.

### Task 5: Memory Retrieval and Closed-Loop Update

**Files:** Create `workflow_memory_retrieval.py`, `workflow_memory_promotion.py`; modify `workflow_context_summary.py`; test `test_workflow_memory_retrieval.py`, `test_workflow_memory_promotion.py`.

**Interfaces:** `retrieve_memory`, `build_core_experience_summary`, and `promote_candidate`.

- [ ] **Step 1:** Test seed-first lookup, filtered Top-K retrieval, evidence expansion, hybrid ranking, conflicts, token limits, injection isolation, circular provenance, and outcome-gated promotion.
```python
def test_core_experience_has_citations(memory):
    result = retrieve_memory(query("new task"), policy(decision_k=5, knowledge_k=8))
    summary = build_core_experience_summary(result, token_budget=800)
    assert summary.memory_ids and summary.evidence_ids
```
- [ ] **Step 2:** Run both focused files; expect failure.
- [ ] **Step 3:** Implement task normalization, vector candidates, exact/full-text fallback, reranking, bounded summaries, and verified-outcome promotion.
- [ ] **Step 4:** Rerun and verify retrieval failure returns no memory.
- [ ] **Step 5:** Commit `feat: add governed memory retrieval loop`.

### Task 6: Memory Forgetting Lifecycle

**Files:** Create `workflow_memory_lifecycle.py`; modify memory repositories and `backend/task_queue.py`; test `test_workflow_memory_lifecycle.py`.

**Interfaces:** `LifecycleWorker.plan(scope, now)`, `apply(plan, limits)`, and `effective_confidence(record, now, policy)`.

- [ ] **Step 1:** Test layer decay, protected references, legal holds, capacity eviction, archive verification, tombstone grace, object/embedding/cache cleanup, idempotent batches, retry/dead-letter, and dry-run.
```python
def test_referenced_event_cannot_be_purged(worker, repo):
    plan = worker.plan("scope", now=expired_time())
    assert repo.referenced_event_id not in plan.purge_ids
```
- [ ] **Step 2:** Run `python -m pytest market_agent_test_bundle/tests/test_workflow_memory_lifecycle.py -q`; expect failure.
- [ ] **Step 3:** Implement retention classes, half-life decay, eviction score, archive/tombstone/purge guards, outbox checkpoints, metrics, and audited transitions.
- [ ] **Step 4:** Rerun and verify expired evidence creates an explicit gap and `不知道`/`no_trade`.
- [ ] **Step 5:** Commit `feat: add governed memory forgetting`.

### Task 7: Fixed Answers and Resilient Agent Runner

**Files:** Create `workflow_response_cache.py`, `local_knowledge_base.py`, `workflow_agent_runner.py`, two knowledge JSON files; modify `langchain_runtime.py`; test `test_workflow_cache_knowledge.py`, `test_workflow_agent_runner.py`.

**Interfaces:** `ResponseCache.get/put/lookup_seed`, `LocalKnowledgeBase.lookup`, and `AgentRunner.run`.

- [ ] **Step 1:** Test fixed aliases/TTL/unsafe categories plus 408/409/429/5xx retry, auth nonretry, timeout cost, model downgrade, knowledge fallback, unknown, and unhealthy-audit denial.
```python
def test_trade_result_is_not_cacheable(cache):
    with pytest.raises(UnsafeCacheEntryError):
        cache.put(key(), CachedAnswer(category="trade_decision", answer="long"), policy())
```
- [ ] **Step 2:** Run both focused files; expect failure.
- [ ] **Step 3:** Implement safe seeds, LRU/SQLite, exact retrieval, audit/reserve-before-call, bounded retries/deadlines, tiers, knowledge, unknown, and usage.
- [ ] **Step 4:** Run focused and `-k "prompt_cache or request_timeout"` tests.
- [ ] **Step 5:** Commit `feat: add resilient cached agent runner`.

### Task 8: Capability Enforcement

**Files:** Create `workflow_capabilities.py`; modify runner, coordinator, graph reducers, audit, and memory services; test `test_workflow_capabilities.py`.

**Interfaces:** `CapabilityContext`, `CapabilityPolicy.issue`, `authorize_read`, `authorize_tool`, `authorize_state_write`, and `authorize_service_request`.

- [ ] **Step 1:** Parameterize the design matrix and test every actor's allowed reads/state keys/tools plus denied database, tenant, memory, audit, queue, web, and exchange actions.
```python
def test_technical_agent_cannot_use_web(policy):
    grant = policy.issue(actor="technical", task_id="t")
    with pytest.raises(CapabilityDeniedError):
        authorize_tool(grant, "web_search")
```
- [ ] **Step 2:** Run the focused file; expect failure.
- [ ] **Step 3:** Implement expiring scoped grants, fixed tool/state allowlists, tenant checks, service identities, reducer validation, and denial audit without logging tokens.
- [ ] **Step 4:** Rerun and verify rescheduling issues a new grant and permission denial is not blindly retried.
- [ ] **Step 5:** Commit `feat: enforce agent capability boundaries`.

### Task 9: Coordinator and Specialist Agents

**Files:** Create `workflow_coordinator_agent.py` and six focused `*_agent.py` modules; test `test_workflow_agent_prompts.py`, `test_workflow_coordinator.py`.

**Interfaces:** Each specialist exports `SYSTEM_PROMPT`, `PROMPT_VERSION`, `build_messages`, `run_node`; coordinator exports `plan_request`, `dispatch_tasks`, `reconcile_reports`, `reschedule`, `summarize_result`.

- [ ] **Step 1:** Test stable abstention/stepwise prompts, dynamic user data, web-tool isolation, task bounds, model selection, errors returning to coordinator, conflict reconciliation, and final summary.
```python
def test_conflict_returns_to_coordinator():
    result = reconcile_reports(plan(), conflicting_reports(), budget())
    assert result.action == "schedule_reconciliation"
```
- [ ] **Step 2:** Run both focused files; expect failure.
- [ ] **Step 3:** Implement stable injection-resistant prefixes, JSON user content, bounded catalog, difficulty routing, report validation, retry/reschedule, Sol reconciliation, and fail-closed summary.
- [ ] **Step 4:** Rerun and verify each dispatched task contains three to five steps and a scoped capability.
- [ ] **Step 5:** Commit `feat: add coordinator and focused trading agents`.

### Task 10: Deterministic Risk, Assembly, and LangGraph

**Files:** Create `workflow_risk_gate.py`, `workflow_playbook_assembler.py`; replace `llm_workflow.py`; test `test_workflow_risk_assembly.py`, `test_workflow_graph.py`.

**Interfaces:** `evaluate_risk`, `assemble_playbook`, `unknown_playbook`, and `LLMWorkflow.invoke(request, services) -> WorkflowResult`.

- [ ] **Step 1:** Test invalid values/stops/scenarios, insufficiency, conflict/Sol, unknown/no_trade, active/passive routes, fan-out/join, memory summary, reschedule cycles, budgets, permissions, and audit-finalize.
```python
def test_specialists_receive_summaries(graph, services):
    graph.invoke(request(), services)
    assert all(isinstance(call.context, ContextSummary) for call in services.runner.calls)
```
- [ ] **Step 2:** Run both focused files; expect failure.
- [ ] **Step 3:** Implement deterministic gate/assembly and typed LangGraph with reducers, bounded cycles, guarded edges, coordinator feedback, and terminal nodes.
- [ ] **Step 4:** Rerun and verify unhealthy audit, denied capability, or exhausted budget cannot dispatch.
- [ ] **Step 5:** Commit `feat: build coordinated langgraph workflow`.

### Task 11: Engine and Production Backend Integration

**Files:** Modify `agent_runtime.py`, prompt/passive/RAG/routing modules, and `backend/{database,observability,container,api_contracts,api}.py`; test `test_workflow_engine_compatibility.py`, `test_workflow_backend_audit.py`.

**Interfaces:** Preserve `get_playbook`; extend debug; expose redacted read-only `GET /v1/audit/traces/{trace_id}` and bounded memory administration/status endpoints; publish low-cardinality metrics.

- [ ] **Step 1:** Test existing modes, prefetch, charts, symbols, caps, unknown, audit/memory queries, redaction, trace IDs, dependency injection, and metrics.
```python
def test_unknown_keeps_no_trade_contract(engine):
    playbook, _ = engine.get_playbook(**unknown_request())
    assert playbook.entry_plan.action_decision.action == "no_trade"
```
- [ ] **Step 2:** Run new compatibility/backend files plus existing unit/backend files; expect targeted failures.
- [ ] **Step 3:** Wire services, request/result adapters, debug, PostgreSQL/SQLite configuration, audit/memory repositories, strict endpoints, lifecycle queue job, and metrics.
- [ ] **Step 4:** Rerun all affected files; expect pass without duplicate orchestration.
- [ ] **Step 5:** Commit `refactor: integrate coordinated agent workflow`.

### Task 12: Security and Legacy Removal

**Files:** Modify migrated runtime/prompt modules; test `test_workflow_security.py`.

**Interfaces:** Only runner/runtime dispatches models; only service identities access durable stores; retrieved content remains user data.

- [ ] **Step 1:** Add AST/runtime tests for direct model calls, exchange/repository imports, raw-context handoff, dynamic system fragments, secret audit fields, unsafe caches, and unauthorized state keys/tools.
```python
def test_agent_modules_do_not_import_exchange():
    assert forbidden_imports(agent_paths()) == []
```
- [ ] **Step 2:** Run security tests; expect legacy violations.
- [ ] **Step 3:** Remove callback-only graph and migrated monolithic prompts while preserving externally imported compatibility helpers.
- [ ] **Step 4:** Run workflow, replay, state-machine, prompt-cache, and security tests.
- [ ] **Step 5:** Commit `refactor: remove legacy llm orchestration`.

### Task 13: Full Verification and Repository Synchronization

**Files:** Apply all changed source, migration, JSON, test, spec, and plan files to `C:\Users\chengdu\Documents\Codex\2026-08-24\b\work\multi-agent-trader`.

**Interfaces:** Both repositories contain identical implementation content with independent commits.

- [ ] **Step 1:** In repo one run `python -m compileall -q market_agent unified_market_agent.py`, all workflow tests, then `python -m pytest -q`.
- [ ] **Step 2:** Run `git diff --check`, inspect status, scan audit/memory payload fields and prompts for secrets or dynamic system values.
- [ ] **Step 3:** Apply explicit patches to repo two, excluding `.git`, runtime databases, objects, logs, caches, and credentials.
- [ ] **Step 4:** Compare corresponding files with `git diff --no-index --ignore-space-at-eol`; compile and run full pytest in repo two.
- [ ] **Step 5:** Commit independently `feat: add coordinated agent workflow`; verify both worktrees clean.

### Task 14: Final Architecture and Safety Review

**Files:** Review every workflow, memory, permission, prompt, backend, and specification path.

**Interfaces:** Final evidence includes commit IDs, exact test totals, content comparison, routing/cost/retention defaults, and opt-in integration coverage.

- [ ] **Step 1:** Trace active, passive-unrelated, conflict, timeout/downgrade, knowledge, unknown, memory promotion, and forgetting paths through audit finalize.
- [ ] **Step 2:** Verify summarized handoffs, audit/reservation/capability before calls, and bounded exits on every cycle.
- [ ] **Step 3:** Verify abstention prompts, dynamic-free system prefixes, actor permission matrix, and no direct durable writes by agents.
- [ ] **Step 4:** Verify vector retrieval citations, anti-circular promotion, protected evidence, legal holds, tombstones, and no live claims in fixed cache.
- [ ] **Step 5:** For each confirmed finding, add a failing test, patch it, rerun affected/full suites, and record final evidence.
