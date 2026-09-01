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
- Every agent/tool output uses versioned strict JSON Schema plus local Pydantic validation; malformed or extra text never enters state or memory.
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

- [ ] **Step 1:** Test exact tier chains including Luna-only reflection, node caps, parallel reservations, timeout charges, attempts, deadlines, and pre-overspend rejection.
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
- [ ] **Step 2:** Run all three focused files; expect failure.
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

### Task 7: Fixed/Semantic Answers and Resilient Agent Runner

**Files:** Create `workflow_response_cache.py`, `workflow_semantic_request_cache.py`, `workflow_circuit_breaker.py`, `local_knowledge_base.py`, `workflow_agent_runner.py`, two knowledge JSON files; modify `langchain_runtime.py`; test `test_workflow_cache_knowledge.py`, `test_workflow_semantic_request_cache.py`, `test_workflow_circuit_breaker.py`, `test_workflow_agent_runner.py`.

**Interfaces:** `ResponseCache.get/put/lookup_seed`, `SemanticRequestCache.lookup/store/expire`, `CircuitBreakerRegistry.before_call/record_success/record_failure/snapshot`, `LocalKnowledgeBase.lookup`, and `AgentRunner.run`.

- [ ] **Step 1:** Test fixed aliases/TTL/unsafe categories; semantic vector storage, strict `>0.95` threshold, metadata, tenant/version/context gates, deterministic ties, hard expiry and cleanup; capped exponential full jitter, `Retry-After`, deadline/cost checks; closed/open/half-open circuit transitions, isolated keys, bounded probes and audited fast fallback; plus strict schema parsing, rejection of prose/fences/extra fields, 408/409/429/5xx retry, auth nonretry, timeout cost, downgrade, knowledge fallback, unknown, and unhealthy-audit denial.
```python
def test_trade_result_is_not_cacheable(cache):
    with pytest.raises(UnsafeCacheEntryError):
        cache.put(key(), CachedAnswer(category="trade_decision", answer="long"), policy())
```
- [ ] **Step 2:** Run both focused files; expect failure.
- [ ] **Step 3:** Implement safe seeds, LRU/SQLite exact retrieval, PostgreSQL/pgvector semantic retrieval with local repository fallback, request/response/model/schema metadata, category TTLs and invalidation, audit/reserve-before-call, capped exponential full jitter, Redis-coordinated/process-local circuit breakers, bounded retries/deadlines, tiers, knowledge, unknown, and usage.
- [ ] **Step 4:** Run focused and `-k "prompt_cache or semantic_cache or request_timeout or circuit_breaker or retry"` tests.
- [ ] **Step 5:** Commit `feat: add resilient cached agent runner`.

### Task 8: Capability Enforcement

**Files:** Create `workflow_capabilities.py`; modify runner, coordinator, graph reducers, audit, and memory services; test `test_workflow_capabilities.py`.

**Interfaces:** `CapabilityContext`, `CapabilityPolicy.issue`, `authorize_read`, `authorize_tool`, `authorize_state_write`, and `authorize_service_request`.

- [ ] **Step 1:** Parameterize the design matrix and test every actor's allowed reads/state keys/tools plus denied database, tenant, memory, audit, queue, web, and exchange actions; reflection receives one target hash/evidence summary, writes only `ReflectionResult`, and cannot mutate the target or reflect itself.
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

**Files:** Create `workflow_prompt_config.py`, Git-tracked prompt release manifests/system text/profiles, `workflow_coordinator_agent.py`, `workflow_reflection_agent.py`, and six focused `*_agent.py` modules; modify `workflow_contracts.py`; test `test_workflow_prompt_config.py`, `test_workflow_agent_prompts.py`, `test_workflow_reflection.py`, `test_workflow_coordinator.py`.

**Interfaces:** `PromptConfigRepository.load_release/profile`; each specialist exports `PROMPT_PROFILE_ID`, `build_messages`, `run_node`; reflection exports `reflect_output`, deterministic `build_correction_context`, and `apply_correction_patch`; coordinator exports `plan_request`, `dispatch_tasks`, `reconcile_reports`, `reschedule`, `summarize_result`.

- [ ] **Step 1:** Test Git-tracked immutable prompt/profile manifests, canonical hashes, dynamic-free stable prefixes, versioned temperature/reasoning/model/schema/tool settings, unsupported-temperature omission, stable abstention/stepwise prompts, strict schemas, dynamic user data, web isolation, task bounds, routing, coordinator/error/conflict/final summary behavior; test objective core-only Luna reflection, deterministic disposition/correction guard, bounded `CorrectionContext`, strict allowlisted patch, and one fallback rewrite.
```python
def test_conflict_returns_to_coordinator():
    result = reconcile_reports(plan(), conflicting_reports(), budget())
    assert result.action == "schedule_reconciliation"
```
- [ ] **Step 2:** Run all four focused files; expect failure.
- [ ] **Step 3:** Implement strict prompt-release loading, stable injection-resistant prefixes from config, capability-checked temperature/settings, JSON user content, bounded catalog, difficulty routing, report validation, objective Luna checks plus deterministic disposition/correction guard, error-carrying correction contexts, targeted patch application, one fallback full rewrite, retry/reschedule, Sol reconciliation, and fail-closed summary.
- [ ] **Step 4:** Rerun and verify each dispatched task contains three to five steps and a scoped capability, each configured core result is reflected exactly once, non-core results are not reflected, and no rejected core target reaches downstream state/cache/memory.
- [ ] **Step 5:** Commit `feat: add coordinator and focused trading agents`.

### Task 10: Deterministic Risk, Assembly, and LangGraph

**Files:** Create `workflow_risk_gate.py`, `workflow_playbook_assembler.py`; replace `llm_workflow.py`; test `test_workflow_risk_assembly.py`, `test_workflow_graph.py`.

**Interfaces:** `evaluate_risk`, `assemble_playbook`, `unknown_playbook`, and `LLMWorkflow.invoke(request, services) -> WorkflowResult`.

- [ ] **Step 1:** Test invalid values/stops/scenarios, insufficiency, conflict/Sol, unknown/no_trade, active/passive routes, fan-out/join, memory summary, objective reflection accept/targeted-patch/fallback-full-rewrite/coordinator/safe-reject routes, strict-improvement/no-regression checks, hash cycles, hard correction limits, reflection outage, trace mismatch, reschedule cycles, budgets, permissions, and audit-finalize.
```python
def test_specialists_receive_summaries(graph, services):
    graph.invoke(request(), services)
    assert all(isinstance(call.context, ContextSummary) for call in services.runner.calls)
```
- [ ] **Step 2:** Run both focused files; expect failure.
- [ ] **Step 3:** Implement deterministic gate/assembly and typed LangGraph with reducers, Luna reflection gates only after decision planning, conditional Sol escalation, and coordinator final summary, bounded cycles, guarded edges, coordinator feedback, and terminal nodes.
- [ ] **Step 4:** Rerun and verify unhealthy audit, denied capability, or exhausted budget cannot dispatch.
- [ ] **Step 5:** Commit `feat: build coordinated langgraph workflow`.

### Task 11: Engine and Production Backend Integration

**Files:** Create `workflow_tracing.py`, `workflow_structured_logging.py`, `workflow_metrics.py`, `workflow_tool_observability.py`, `workflow_prompt_release.py`; modify contracts/state, `agent_runtime.py`, prompt/passive/RAG/routing modules, and `backend/{database,observability,container,api_contracts,api}.py`; test `test_workflow_engine_compatibility.py`, `test_workflow_trace_propagation.py`, `test_workflow_observability.py`, `test_workflow_prompt_release.py`, `test_workflow_backend_audit.py`.

**Interfaces:** `TraceContext.new_request/child/inject/extract/assert_same_trace`; typed log/metric/tool observers; `PromptReleaseManager.pin/current/activate/rollback_previous`; preserve `get_playbook`; extend debug; expose redacted trace queries, guarded prompt-release activate/rollback, and bounded memory endpoints; publish low-cardinality metrics.

- [ ] **Step 1:** Test existing modes and contracts; fresh unique traces/parented spans; structured JSON fields/redaction/searchability; business/interface/Agent/token/cost metrics and bounded labels; detailed redacted tool call/result records; cache-origin links; mismatch denial; prompt release pinning, hash/schema/capability/eval gates, atomic activation, one-action rollback, in-flight stability, audit, and metrics.
```python
def test_unknown_keeps_no_trade_contract(engine):
    playbook, _ = engine.get_playbook(**unknown_request())
    assert playbook.entry_plan.action_decision.action == "no_trade"
```
- [ ] **Step 2:** Run new compatibility/backend files plus existing unit/backend files; expect targeted failures.
- [ ] **Step 3:** Wire trace/span generation and propagation, structured logging, metrics, schema-aware tool observation, prompt-release registry/rollback, services, adapters/debug, PostgreSQL/SQLite, audit/memory, strict endpoints, lifecycle queue job, telemetry exporters, exemplars, and response trace headers.
- [ ] **Step 4:** Rerun all affected files; expect pass without duplicate orchestration.
- [ ] **Step 5:** Commit `refactor: integrate coordinated agent workflow`.

### Task 12: Security and Legacy Removal

**Files:** Modify migrated runtime/prompt modules; test `test_workflow_security.py`.

**Interfaces:** Only runner/runtime dispatches models; only service identities access durable stores; retrieved content remains user data.

- [ ] **Step 1:** Add AST/runtime tests for direct model calls, exchange/repository imports, raw-context/dynamic system fragments, unversioned inline system prompts, runtime prompt mutation, unsupported temperature, unaudited release activation, rollback bypass, secret audit/log/tool fields, unsafe caches, unauthorized state/tools, invalid reflection/correction, missing traces, and cross-trace writes.
```python
def test_agent_modules_do_not_import_exchange():
    assert forbidden_imports(agent_paths()) == []
```
- [ ] **Step 2:** Run security tests; expect legacy violations.
- [ ] **Step 3:** Remove callback-only graph and migrated monolithic prompts while preserving externally imported compatibility helpers.
- [ ] **Step 4:** Run workflow, replay, state-machine, prompt-cache, and security tests.
- [ ] **Step 5:** Commit `refactor: remove legacy llm orchestration`.

### Task 13: Evaluation Corpus and Release Gates

**Files:** Create `workflow_eval_dataset.py`, `workflow_evaluation.py`, `workflow_eval_metrics.py`, versioned `evals/schema` and `evals/datasets` JSON/JSONL files; test `test_workflow_eval_dataset.py`, `test_workflow_evaluation.py`.

**Interfaces:** `EvaluationDataset.load/validate`, `EvaluationRunner.run/compare`, and `ReleaseGate.evaluate`.

- [ ] **Step 1:** Test dataset schemas/hashes/provenance, duplicate and train/holdout leakage detection, immutable snapshots, expected/forbidden facts, tolerances, deterministic success scoring, safety hard gates, paired candidate/baseline comparison, confidence bounds, latency/token/cost budgets, and result reproducibility.
```python
def test_safety_failure_blocks_high_average(release_gate):
    result = release_gate.evaluate(candidate(high_success=True, risk_violation=True), baseline())
    assert not result.allowed
```
- [ ] **Step 2:** Run both focused files; expect failure.
- [ ] **Step 3:** Implement sanitized regression/security/resilience/cache/RAG/memory/reflection/permission/trace seeds, dataset lifecycle validation, offline recorded runner, opt-in live shadow runner with no orders, metric aggregation, immutable result artifacts, and release comparison.
- [ ] **Step 4:** Run the versioned offline corpus for the current release; record overall success plus schema, abstention, hallucination, evidence, risk, cache, retrieval, reflection, trace, latency, token, and cost results.
- [ ] **Step 5:** Commit `feat: add versioned workflow evaluation gates`.

### Task 14: Full Verification and Repository Synchronization

**Files:** Apply all changed source, migration, JSON, test, spec, and plan files to `C:\Users\chengdu\Documents\Codex\2026-08-24\b\work\multi-agent-trader`.

**Interfaces:** Both repositories contain identical implementation content with independent commits.

- [ ] **Step 1:** In repo one run `python -m compileall -q market_agent unified_market_agent.py`, all workflow tests, then `python -m pytest -q`.
- [ ] **Step 2:** Run `git diff --check`, inspect status, scan audit/memory payload fields and prompts for secrets or dynamic system values.
- [ ] **Step 3:** Apply explicit patches to repo two, excluding `.git`, runtime databases, objects, logs, caches, and credentials.
- [ ] **Step 4:** Compare corresponding files with `git diff --no-index --ignore-space-at-eol`; compile and run full pytest in repo two.
- [ ] **Step 5:** Commit independently `feat: add coordinated agent workflow`; verify both worktrees clean.

### Task 15: Final Architecture and Safety Review

**Files:** Review every workflow, memory, permission, prompt, backend, and specification path.

**Interfaces:** Final evidence includes commit IDs, exact test totals, content comparison, routing/cost/retention defaults, and opt-in integration coverage.

- [ ] **Step 1:** Trace active, passive-unrelated, objective reflection accept/patch/rewrite/regression-stop/coordinator/safe-reject, conflict, timeout/downgrade, knowledge, unknown, memory promotion, and forgetting paths through audit finalize under one immutable request trace ID with parented spans.
- [ ] **Step 2:** Verify summarized handoffs, audit/reservation/capability before calls, exactly one Luna reflection for each configured core output and none for non-core outputs, target immutability, and bounded exits on every cycle.
- [ ] **Step 3:** Verify Git-tracked prompt/temperature releases, pinned versions, one-action rollback, evaluation gates, abstention prompts, dynamic-free system prefixes, actor permission matrix, and no direct durable writes by agents.
- [ ] **Step 4:** Verify vector retrieval citations, anti-circular promotion, protected evidence, legal holds, tombstones, strict semantic-cache threshold/expiry/version metadata, and no live claims in fixed or semantic caches.
- [ ] **Step 5:** For each confirmed finding, add a failing test, patch it, rerun affected/full suites, and record final evidence.
