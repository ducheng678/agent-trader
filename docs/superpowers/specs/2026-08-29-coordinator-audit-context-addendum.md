# Coordinator, Audit, and Context Handoff Addendum

This addendum extends `2026-08-29-langgraph-multi-agent-trading-workflow-design.md` and is normative. If the two documents differ, this addendum governs coordinator behavior, context handoff, auditability, and conflict recovery.

## Coordinator Agent

The workflow has one coordinator agent responsible for orchestration. It does not perform specialist market analysis itself and has no authority to place orders.

Its responsibilities are:

- classify the request and select the active or passive workflow template;
- decompose work into bounded specialist tasks that normally require three to five reasoning steps;
- select `gpt-5.6-luna`, `gpt-5.6-terra`, or `gpt-5.6-sol` from task difficulty and risk policy;
- dispatch tasks subject to workflow timeout, attempt, and cost budgets;
- receive structured results, errors, timeouts, uncertainty, and conflicts;
- reschedule, retry, downgrade, or escalate tasks within the remaining budget;
- reconcile compatible specialist results and identify unresolved contradictions;
- produce the final structured playbook or informational answer;
- fail closed as `no_trade` or `不知道` when evidence is insufficient.

The coordinator uses a fixed LangGraph topology and a bounded task catalog. It may choose and repeat permitted nodes, but it may not invent arbitrary tools, bypass the risk gate, expand the user's request, or exceed configured budgets.

## Task Contract

Every dispatched task uses a strict structured contract containing:

- `task_id`, `parent_task_id`, `workflow_id`, and `trace_id`;
- `task_type`, `objective`, and allowed data or tools;
- a summarized context envelope and source references;
- expected output schema and acceptance criteria;
- difficulty, selected model tier, prompt version, and cache key;
- per-attempt timeout, maximum retries, reserved cost, and remaining workflow budget;
- a bounded analysis outline of three to five steps;
- escalation and conflict-return rules.

Tasks that cannot fit this contract must be decomposed again before dispatch.

## Context Summarization Before Handoff

Raw accumulated conversation and graph state are not forwarded directly to specialist agents. A deterministic context selector first chooses relevant source records. A context summarizer then emits a typed `ContextSummary` containing:

- user objective and immutable constraints;
- relevant market facts with timestamps and source identifiers;
- prior conclusions that the next agent is allowed to rely on;
- unresolved questions, conflicts, and uncertainty markers;
- omitted-section counts, token estimate, and completeness status;
- summary version, summarizer model, and hash of the selected source records.

Stable system instructions remain the first prompt prefix for prompt caching. The task-specific summary follows the stable prefix. Summaries must preserve numeric values, timestamps, symbols, units, negation, provenance, and uncertainty. A specialist must reject an incomplete summary when a required field is missing instead of guessing.

The full source records remain in shared workflow state and the audit store. Summary references allow the coordinator and final assembler to retrieve evidence without repeatedly placing the entire context into prompts.

## Reasoning Prompt Policy

Reasoning-capable agent prompts include the instruction:

> 请一步一步分析，但只输出符合 schema 的结论、关键依据和不确定性，不输出隐藏的详细推理过程。不确定时必须回答不知道。

The task contract supplies a three-to-five-step analysis outline. Outputs expose concise evidence and decision factors, not private chain-of-thought. Schema validation rejects unsupported certainty and missing uncertainty fields.

## Full-Chain Audit

Every workflow transition writes an append-only audit event. Required event categories include:

- request accepted, normalized, and classified;
- task created, decomposed, dispatched, rescheduled, and completed;
- context sources selected and summary created;
- model route selected and changed;
- prompt version and prompt-cache key used;
- high-frequency answer cache lookup and result;
- tool call requested, allowed, completed, or denied;
- structured output validation result;
- retry, timeout, cancellation, and cost reservation or settlement;
- local knowledge lookup and evidence selected;
- fallback tier entered and unknown returned;
- specialist conflict detected and coordinator resolution selected;
- risk gate decision and final response assembled.

Each event contains `event_id`, `trace_id`, `workflow_id`, optional `task_id` and `attempt_id`, monotonic sequence number, UTC timestamp, actor, event type, input and output hashes, status, latency, token usage, cached-token usage, estimated cost, cumulative cost, model, prompt version, source references, and a redacted structured payload.

Audit writes are transactional and append-only. Sensitive values, credentials, raw authorization headers, unrestricted prompt bodies, and private reasoning are never recorded. Payload schemas define field-level redaction. Audit persistence failure blocks new LLM or tool dispatches and causes a safe terminal result, because an unaudited workflow is not allowed to continue.

The primary durable implementation uses the configured production database. SQLite is permitted for local development with WAL enabled. Audit events are queryable by trace, workflow, task, attempt, event type, and time range. Logs and metrics include trace identifiers but do not replace the durable audit trail.

## Conflict and Error Feedback

Specialist agents never silently resolve cross-agent conflict. They return a typed `AgentReport` with status `completed`, `uncertain`, `conflict`, or `failed`.

A conflict report contains the disputed claims, evidence references, confidence or uncertainty, and the missing evidence needed for resolution. An error report contains a normalized error category, retryability, consumed budget, and safe fallback recommendation.

Both return to the coordinator. The coordinator applies this bounded policy:

1. validate reports and compare their cited evidence;
2. if evidence is missing or stale, reschedule a narrower retrieval or analysis task;
3. if the issue is transient, retry the same tier within per-task and global limits;
4. if a model tier is unavailable or exhausted, downgrade one tier;
5. if a high-risk contradiction remains and budget allows, escalate reconciliation to `gpt-5.6-sol`;
6. otherwise query the local knowledge base for an evidence-backed fixed answer;
7. if still unresolved, terminate as `no_trade` for trading decisions or `不知道` for informational answers.

Authentication, authorization, invalid configuration, audit persistence failure, and hard budget exhaustion are not retried blindly. They terminate safely or use only a permitted non-LLM fallback.

## LangGraph Control Flow

The revised graph is:

`request_normalizer -> seed_cache_lookup -> coordinator_plan -> context_select -> context_summarize -> dispatch_specialists -> collect_reports -> coordinator_reconcile`

From `coordinator_reconcile`:

- accepted reports continue to `risk_gate -> coordinator_summarize -> response_assembler -> audit_finalize`;
- missing evidence returns to `context_select` with a narrower task;
- retryable failure returns to `dispatch_specialists` with a new attempt;
- conflict routes to a bounded reconciliation task and then returns to `collect_reports`;
- exhausted budgets route to `local_knowledge_fallback -> unknown_guard -> response_assembler -> audit_finalize`.

All cycles have explicit counters and deadline and cost guards in shared state. No graph edge can dispatch work without a successful audit event and budget reservation.

## Shared State Additions

The global state adds:

- coordinator plan and plan revision;
- pending, running, completed, failed, and conflicted task collections;
- source records and typed context summaries;
- audit sequence and audit health;
- per-task and workflow deadlines;
- attempt counters and maximum attempts;
- reserved, settled, and remaining cost;
- model routing history and fallback level;
- conflict set and reconciliation history;
- final knowledge status and unknown reason.

State reducers are deterministic. Parallel specialist reports merge by task identifier. Conflicting writes produce a conflict record instead of last-write-wins behavior.

## File Boundary Additions

- `workflow_coordinator_agent.py`: bounded decomposition, scheduling, conflict resolution, rescheduling, and final summaries.
- `workflow_context_summary.py`: source selection, typed summaries, completeness checks, and handoff hashes.
- `workflow_audit.py`: append-only audit events, redaction, persistence, and trace queries.

## Acceptance Criteria

- Every specialist invocation is created and observed by the coordinator.
- Every specialist receives a typed summary rather than unrestricted raw workflow context.
- Ordinary specialist tasks contain a bounded three-to-five-step analysis outline.
- Every dispatch, model call, retry, fallback, conflict, and terminal answer is represented in the durable audit trail.
- An audit write failure prevents further external calls.
- Conflicts and errors return to the coordinator and cannot bypass reconciliation.
- Retry, timeout, attempt, and cost limits are enforced at both task and workflow scope.
- Budget exhaustion reaches local knowledge and then `不知道` or `no_trade` without hallucinated content.
- Final output includes evidence references, uncertainty, routing summary, and trace identifier without exposing private reasoning.
