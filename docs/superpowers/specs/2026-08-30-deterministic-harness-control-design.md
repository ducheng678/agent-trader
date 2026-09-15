# Deterministic Harness Control Design

## Status and precedence

This document is the normative control-plane specification for the trading-agent system. It supersedes the coordinator-agent ownership, free-form replanning, and coordinator-summary portions of:

- 2026-08-29-langgraph-multi-agent-trading-workflow-design.md
- 2026-08-29-coordinator-audit-context-addendum.md

Precedence is explicit rather than inferred:

| Legacy area | Status under this specification |
|---|---|
| coordinator ownership of decomposition, scheduling, replanning, retry, conflict routing, and terminal summary | replaced by HarnessKernel, versioned templates, deterministic policies, and sealed-result rendering |
| model-selected LangGraph edges or model-created tasks | replaced; LangGraph executes only committed Harness transitions |
| model self-confidence thresholds, including the former 0.60 rule | replaced by the versioned Harness confidence contract in this document |
| audit chain, context summaries, retry/cost bounds, and Task 1-3 contracts | retained where they do not grant control authority to an LLM |
| prompt-cache stability, strict output, cache/RAG/memory/forgetting, core-only reflection, trace, observability, prompt releases, evaluation, and production backend requirements | retained and refined by the corresponding sections below |

If a legacy sentence conflicts with this table or any normative rule below, this specification wins. No legacy coordinator, graph, confidence, or summary authority survives by implication.

> An LLM may propose content. Only the deterministic Harness may commit a plan, transition state, authorize a tool, accept a result, retry work, mutate durable state, or determine the terminal outcome.

## Context and decision

The earlier design assigned decomposition, scheduling, conflict recovery, rescheduling, and final summarization to a coordinator LLM constrained by a fixed LangGraph. That leaves critical control decisions dependent on probabilistic output and makes replay and safety enforcement unnecessarily difficult.

The approved design replaces the coordinator LLM with a deterministic Harness control plane. LangGraph remains the initial execution backend for fixed DAG execution, bounded parallelism, interrupts, and progress streaming. It does not own plans, transitions, budgets, permissions, retries, or terminal status and is replaceable behind an execution-backend interface.

The design selectively adopts ideas from DeepSeek Harness:

- a stable Agent/Worker interface separated from the concrete loop driver;
- an append-only typed session event log as the run source of truth;
- scoped prompt and tool visibility;
- guarded tool dispatch through a registry;
- explicit lifecycle events and rollback-covered creation/resume;
- replaceable model, persistence, tool, and presentation providers behind stable service definitions.

It does not adopt Cordis, the TypeScript package topology, runtime model-written plugins, unrestricted coding tools, a general autonomous ReAct loop, or an optional safety core. Production adapters are selected from trusted configuration at boot and runtime self-modification is forbidden.

Reference documentation:

- https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md
- https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/core.md
- https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/session.md
- https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/subsystems/tools.md
- https://github.com/deepseek-ai/deepseek-harness/blob/master/packages/core/system-prompt/README.md

## Goals

- Make workflow control deterministic, replayable, bounded, and auditable.
- Keep specialist LLM work narrow enough for three to five declared analysis phases.
- Prevent worker loops, retry loops, cross-worker oscillation, and false-progress loops.
- Enforce model, token, tool, time, attempt, concurrency, and dollar budgets before dispatch.
- Reconstruct every model request from pinned versions and immutable references without logging secrets or private reasoning.
- Keep Agent outputs strict, typed, evidence-linked, and unable to mutate durable state directly.
- Preserve the existing public trading-engine interface while replacing internal orchestration.

## Non-goals

- General-purpose autonomous planning or model-controlled graph construction.
- Arbitrary routing, worker spawning, tool discovery, or runtime plugin loading.
- Exposing chain-of-thought, credentials, unrestricted memory, or broker clients.
- Claiming exactly-once external trading effects; use idempotent intents, fencing, and reconciliation.
- Keeping the legacy coordinator and Harness as permanent competing implementations.

## Architecture

~~~text
API / Engine Adapter
        |
        v
HarnessKernel -----------------------------------------------------+
  +-- AdmissionPolicy                                             |
  +-- GlobalTaskStateMachine                                      |
  +-- PlanTemplateRegistry / PlanCompiler                         |
  +-- WorkerRegistry                                              |
  +-- TransitionPolicy / Reconciler                               |
  +-- LoopGuard / ConfidenceGate                                  |
  +-- Budget / Retry / Circuit / Degradation Policies             |
  +--> HarnessEventStore --> projections / audit / observability  |
  +--> ContextService --> cache / RAG / memory                    |
  +--> PromptAssembler --> pinned prompt release                  |
  +--> ScopedToolBroker --> approved services                    |
  +--> AgentDriver --> ModelGateway                               |
  +--> ExecutionBackend --> LangGraph initially                   |
  +--> RiskGate --> PlaybookAssembler --> WorkflowResult --------+
~~~

HarnessKernel is the only component allowed to create a run, commit a plan, dispatch work, transition state, authorize retries, invoke degradation, accept a result, or finalize a run. Its command handlers are deterministic over an event-folded projection, a command, and injected clock/random inputs.

Public operations are:

~~~text
create(request) -> RunHandle
resume(run_id) -> RunHandle
advance(run_id) -> HarnessDecision
cancel(run_id, reason) -> HarnessDecision
snapshot(run_id) -> HarnessSessionView
~~~

Creation and resume are rollback-covered. Session validation, release pinning, policy pinning, lease acquisition, budget restoration, and executor registration all succeed before publication.

Trusted providers implement narrow protocols for events, execution, Agent driving, model access, prompt releases, context, scoped tools, cache, memory, artifacts, circuit breakers, telemetry, clock, and randomness. Consumers depend on protocols rather than implementations. Transition validation, permissions, budgets, risk, output validation, and LoopGuard are mandatory kernel components and cannot be unloaded during a run.

LangGraph executes only committed HarnessPlan work. Conditional edges consume only a HarnessTransition emitted by the kernel; raw model output cannot select an edge. Checkpoints are disposable projections. Resume folds the authoritative event stream before reconstructing executor state.

## Control ownership

| Concern | Owner | LLM authority |
|---|---|---|
| intent classification | deterministic admission parser over validated explicit request fields | none |
| plan and dependencies | versioned plan compiler | none |
| scheduling and concurrency | Harness scheduler | none |
| model selection | routing policy | none |
| retry and reschedule | transition/retry policy | diagnostics only |
| tool execution | scoped broker | typed request only |
| cache use | cache safety policy | none |
| context and memory selection | retrieval pipeline | no raw-store access |
| conflict resolution | reconciler; optional adjudicator | recommendation only |
| durable memory writes | promotion/lifecycle services | candidate only |
| risk and trading action | deterministic services | analysis only |
| final status | global state machine | none |
| final wording | sealed report plus synthesis worker | expression only |

There is no coordinator Agent. Optional ConflictAdjudicationWorker and ResultSynthesisWorker have zero orchestration authority. Semantic labels emitted by any worker are evidence fields only and cannot select a template, raise risk authority, expand capabilities, or unlock a trading path.

Harness first constructs and seals the authoritative WorkflowResult and Playbook from validated evidence and deterministic risk decisions. ResultSynthesisWorker receives a read-only projection and may fill only allowlisted presentation fields. PlaybookAssembler never reads factual, trading, status, risk, or evidence fields from synthesis output. A synthesis timeout, refusal, schema mismatch, or inconsistency discards the entire synthesis payload and uses the deterministic locale renderer; it never changes the sealed result.

## Plans and workers

A HarnessPlan is compiled from a versioned template selected only from validated explicit request fields and deterministic admission rules over workflow mode, task kind, risk class, symbols, horizon, and required evidence. Active/trading templates, risk class, and capability scope are never derived from model labels. Ambiguity selects a passive informational template or no_trade. A model cannot add a stage or edge.

The default active path is:

~~~text
admission -> safe cache lookup -> context and memory retrieval
  -> event filter -> optional market context
  -> fundamental and technical workers in parallel
  -> decision draft -> objective reflection
  -> optional Sol escalation -> objective reflection
  -> deterministic risk gate
  -> playbook assembly and WorkflowResult sealing
  -> optional presentation synthesis -> objective reflection
  -> deterministic locale rendering -> audit finalization
~~~

The three-to-five constraint applies to each worker's declared logical analysis phases, not to deterministic cache, retrieval, safety, audit, and lifecycle stages. Passive and unrelated templates cannot enter live-trading stages.

Every HarnessPlan contains immutable StageSpec records. Each StageSpec declares stage_id and version, deterministic entry and completion predicates, allowed WorkItem kinds, frozen dependencies, bounded concurrency, stage budget policy, failure/degradation mapping, and whether side effects or reconciliation are possible. A stage is a first-class budget and state-machine scope, not an Agent-generated label.

Every immutable WorkerSpec declares:

~~~text
worker id and version
supported task kinds
three-to-five analysis phase identifiers
input/output schema identities and hashes
prompt release/profile
model-routing policy key
context selector and token budget
readable state keys and one writable invocation-state key
allowed tool capabilities
cacheability and freshness class
turn, tool, token, time, attempt, and cost limits
typed success, failure, and degradation outcomes
~~~

Analysis phases are observable work goals, not private chain-of-thought. Prompts may require internal step-by-step analysis, but output contains only the strict schema, conclusions, evidence references, uncertainty, and allowed diagnostics.

AgentDriver.invoke accepts a sealed WorkerInvocation containing run/work/attempt/plan identities, typed objective, immutable context and dependency references, deadline, budget reservation, capability grant, idempotency and fencing tokens, trace/span identities, and pinned configuration hashes.

On success it returns WorkerCandidateOutput; on execution failure it returns a WorkerFailureObservation containing only trusted transport/tool observations and an optional untrusted model diagnostic. A host-owned, versioned ErrorClassifier is the only component allowed to map an invocation observation to RetryableWorkerFailure, PermanentWorkerFailure, PermissionRequired, or conflict/reconciliation and to request RetryAuthorization. Cancellation comes only from a kernel cancellation command/token. Model fields never determine retryability, permission escalation, cancellation, or retry count. Workers cannot spawn workers, schedule tasks, finish runs, write durable memory, or access raw database/cache/queue/broker clients.

## Append-only event source

HarnessEventStore is the orchestration source of truth and canonical local audit chain. Every event includes:

~~~text
event_id, trace_id, span_id, parent_span_id
run_id, work_item_id, attempt_id when applicable
sequence and state_revision
event_type and schema_version
UTC occurrence time and monotonic offset metadata
actor/service identity
plan, prompt, model-policy, tool-schema, output-schema versions/hashes
redacted typed payload or content-addressed artifact reference
previous_event_hash and event_hash
~~~

sequence increases by exactly one for every committed event in a run. state_revision increases only for a committed state transition; audit, heartbeat, model, tool, and telemetry events that do not transition state carry the current revision. The store enforces both counters with optimistic concurrency.

Events cover run, plan, task, attempt, model, tool, budget, retry, circuit, cache, context, memory, reflection, correction, risk, summary, and terminal transitions. Large content goes to encrypted checksum-addressed storage; events retain hash, size, classification, and authorization-bound reference.

The fold operation is deterministic and rejects invalid sequence, hash chain, trace, transition, schema, or version relationships. Logs, metrics, queue messages, and remote telemetry derive from committed events through an outbox. Remote telemetry may be sampled; the canonical local event chain may not.

Requests are reconstructable from pinned prompt/model policies, ordered tool schemas, immutable context/dependency references, and request headers. Secrets and private reasoning are never logged.

## Global task state machine

~~~text
Run:
CREATED -> ADMITTED -> PLANNED -> READY -> RUNNING
RUNNING -> RECONCILING | WAITING_APPROVAL | WAITING_RECONCILIATION | DEGRADING | SUMMARIZING
RECONCILING -> RUNNING | WAITING_RECONCILIATION | DEGRADING | SUMMARIZING | FAILED
WAITING_APPROVAL -> RUNNING | FAILED | CANCELLED
WAITING_RECONCILIATION -> RECONCILING | FAILED | CANCELLED
DEGRADING -> RUNNING | SUMMARIZING | DEGRADED | FAILED
SUMMARIZING -> SUCCEEDED | DEGRADED | FAILED
any nonterminal -> FAILED | CANCELLED, subject to the side-effect rule below

Work item:
PENDING -> READY -> LEASED -> RUNNING -> VALIDATING -> SUCCEEDED
LEASED | RUNNING | VALIDATING -> RETRY_WAIT -> READY
nonterminal -> BLOCKED | FAILED | CANCELLED

Attempt:
RESERVED -> DISPATCHED
DISPATCHED -> STREAMING | VALIDATING
STREAMING -> VALIDATING
VALIDATING -> SETTLING -> COMPLETED
nonterminal -> TIMED_OUT | REJECTED | FAILED | STALE | CANCELLED
~~~

Only kernel transition handlers change state. Terminal Run and Attempt states are absorbing; completed/failed/blocked/cancelled WorkItems are absorbing. A stale Attempt can authorize its owning nonterminal WorkItem to enter RETRY_WAIT, but the stale Attempt itself never reopens. A trusted policy may emit a typed PermanentFailureDecision from any nonterminal Run state; the same atomic event append records the reason and FAILED transition, so early admission/configuration failures have no audit-finalization gap. If an external side effect may have occurred but its status is unknown, FAILED and CANCELLED are forbidden: the run enters or remains WAITING_RECONCILIATION until a broker observation resolves the effect. A cancellation request is then recorded as intent and takes effect only after reconciliation.

Transitions validate the current revision, plan and dependency versions, reservation, grant, trace, lease epoch, fencing token, and idempotency key. Stale results are recorded without mutating current state.

Run state and business outcome are orthogonal sealed fields:

| terminal_state | outcome_kind | knowledge_status | representative terminal_reason |
|---|---|---|---|
| SUCCEEDED | answer | known or partial | completed, fixed_seed_cache_hit, or compatible_semantic_cache_hit |
| SUCCEEDED | no_trade | known | strategy_no_trade or risk_gate_no_trade |
| DEGRADED | answer | known or partial | lower_model_fallback or verified_local_knowledge_fallback |
| DEGRADED | unknown | unknown | insufficient_evidence, confidence_recovery_exhausted, or dependency_unavailable |
| DEGRADED | no_trade | unknown or partial | safe_no_trade_due_to_degradation |
| FAILED | none | not_applicable | permanent_policy, integrity, audit, or configuration_failure |
| CANCELLED | none | not_applicable | cancellation_completed |

A normal, evidence-backed no_trade is therefore successful; a safety no_trade caused by missing infrastructure is degraded. A direct compatible seed/semantic-cache hit on the normal path is SUCCEEDED/answer. Once the degradation chain has started, an accepted lower-model or local-knowledge result is DEGRADED/answer. A local result is accepted only when its immutable version, scope, tenant, permission, applicability, freshness, provenance, evidence, safety policy, and schema all match and validation passes; it can answer approved informational classes but can never supply a live trading decision. A miss or rejection continues to DEGRADED/unknown for informational work or DEGRADED/no_trade for active trading.

Protocol clients branch on the structured fields, never localized prose. The locale renderer maps outcome_kind=unknown and knowledge_status=unknown to the approved abstention message; the zh-CN catalog value is `\u4e0d\u77e5\u9053`. Unknown and no_trade are never interchangeable in storage or metrics.

Wall-clock passage alone never mutates state. Deadline, lease expiry, cancellation, and staleness enter as explicit events produced from an injected monotonic clock.

## Loop prevention and termination

LoopGuard is a mandatory kernel component operating at attempt, work-item, stage, and run scope. It combines hard limits, repeated-action detection, objective progress checks, Harness confidence, state-cycle detection, and cost-to-go admission.

### Hard multidimensional limits

Budget scopes are hierarchical: run -> StageSpec/stage_id -> work item -> attempt. Provider, model, model tier, tool, data source, token category, and currency are orthogonal quota dimensions applied across those scopes; they are not hierarchy levels. Budgets bound:

- state transitions;
- worker turns and attempts;
- one correction patch and one full rewrite;
- input, cached-input, cache-write, reasoning, and output tokens;
- model and tool calls;
- parallel work;
- monotonic elapsed time and attempt timeout;
- dollar cost;
- no-progress transitions;
- conflict adjudications and reschedules.

Before a model or tool call, one serializable transaction or equivalently fenced lock atomically reserves conservative worst-case cost, attempts, tokens, calls, and time against every applicable scope and quota dimension. The strictest effective remaining cap wins. Either every ledger row and reservation event commits or none does; partial reservations and check-then-write admission are forbidden. Insufficient cost-to-go stops dispatch.

Settlement records actual usage and charges atomically for every terminal invocation outcome, even when actuals exceed the reservation. An overflow exception carries the already committed settlement. The same transaction emits budget/overdrawn, marks every violated scope/dimension exhausted, and prevents new dispatches that consume them. Queued work and undispatched leases are cancelled or degraded when safe; already-dispatched calls may only finish and settle, not open further work. Deterministic terminal policy is:

- informational work with an approved local fallback enters DEGRADING and ends DEGRADED/answer when that result passes every acceptance gate, otherwise DEGRADED/unknown;
- active trading or a mandatory evidence/risk stage enters DEGRADING and ends DEGRADED/no_trade;
- integrity, audit, permission, or settlement-persistence failure ends FAILED;
- an ambiguous external order effect enters WAITING_RECONCILIATION regardless of budget state.

### Fingerprints

Ephemeral IDs are excluded from loop fingerprints.

ActionFingerprint hashes worker/version, action kind, canonical arguments, context and dependency hashes, plan revision, prompt/tool/output-schema hashes, model route, and correction ordinal.

ResultFingerprint hashes outcome kind, validated output hash, normalized error class/code, sorted accepted evidence IDs, sorted tool/artifact result hashes, and result schema/version. Unvalidated prose and ephemeral provider IDs are excluded.

ActionObservationFingerprint hashes ActionFingerprint plus ResultFingerprint. It represents repeating the same logical action and obtaining the same logical result.

StateFingerprint hashes run, work-item, and attempt states; current stage_id; plan revision; unresolved work set; dependency versions; objective ProgressVector; and normalized error class. Attempt identity is excluded, but Attempt state is included.

CycleSignature is computed only at semantic checkpoints. For each attempt, work-item, stage, and run scope, LoopGuard retains the latest twelve StateFingerprint values. It examines candidate periods from one through min(6, floor(observation_count / 2)); a period is repeating only when the latest two consecutive subsequences of that length are equal. The smallest matching period wins, so nested cycles resolve to the shortest period. Its period sequence is rotation-normalized to the lexicographically smallest rotation, then hashed with scope, plan revision, and fingerprint-schema version. Infrastructure-only transitions are excluded. Window drift, new attempt IDs, heartbeats, prose changes, or additional tokens therefore cannot manufacture new signatures for the same logical cycle.

### Repetition and cycle rules

- The same ActionObservationFingerprint may complete at most twice, including the original; a third identical action/result observation stops the work item.
- The same ActionFingerprint appearing three times in the latest five semantic actions stops the work item even when its results differ.
- An identical state fingerprint repeated twice without progress stops the attempt.
- A shortest-period cycle such as A-B-A-B or A-B-C-A-B-C without progress stops the branch and records its CycleSignature.
- The same normalized failure passed between workers twice without changed context, dependency, correction, or route stops rescheduling.
- Heartbeats do not advance progress_epoch and cannot hide a stalled task.
- Infrastructure retries require RetryAuthorization, count against all budgets, and do not count as progress.

At plan commit, PlanCompiler freezes a bounded ProgressTargetSet for every scope: required dependency IDs, required output field paths, required evidence slots with relevance/provenance/authority/freshness predicates, required source-coverage weights, known conflict slots, and risk invariant IDs. Workers cannot add targets. Progress state stores canonical sets or bitsets keyed by those IDs and derives these bounded measures:

~~~text
completed_dependency_count
valid_required_field_count
filled_required_evidence_slot_count
fresh_authoritative_source_coverage
missing_evidence_count
validation_error_count
unresolved_conflict_count
risk_invariant_failure_count
~~~

Dependency, field, evidence, conflict, and invariant IDs are canonical and de-duplicated. Evidence increments progress only when an authorized accepted record fills an unfilled required evidence slot and passes that slot's relevance, provenance, authority, and freshness predicates. Extra evidence for an already filled slot and evidence unrelated to a frozen slot do not increase progress. Every count is capped by its frozen target cardinality. fresh_authoritative_source_coverage is a deterministic satisfied-weight / total-required-weight value in [0, 1], or 1 only when the plan declares no source requirement.

The versioned comparator treats dependency, field, filled-evidence, and source-coverage dimensions as positive and missing-evidence, validation-error, unresolved-conflict, and risk-failure dimensions as negative. Progress advances only when no oriented dimension worsens and at least one strictly improves. A versioned severity table defines critical validation, provenance, direction, and risk regressions; any critical regression fails comparison regardless of improvements elsewhere. Different wording, claimed progress, repeated retrieval, more tokens, a model change, or unrelated evidence is not progress.

No-progress counters advance only at semantic checkpoints after a validated worker result, context refresh, correction, adjudication, or recovery. Admission, queueing, lease, dispatch, heartbeat, settlement, and other infrastructure-only transitions do not affect the counter. Two consecutive semantic checkpoints without progress stop the work item. A new critical error, weaker risk posture, unsupported direction flip, lost evidence, repeated output hash, or hash cycle stops correction immediately.

### Harness confidence

Model self-reported confidence is advisory untrusted data and is never a scoring feature. ConfidenceGate computes features only from the frozen ProgressTargetSet, accepted evidence records, deterministic source registry, validators, conflict records, and event-folded state. Authority comes from the versioned source registry; agreement comes from deterministic comparison of normalized accepted facts; completeness comes from frozen required slots. A model-provided authority, agreement, freshness, or completeness claim has no weight.

Every score pins an immutable, checksum-addressed ConfidenceCalibratorArtifact containing artifact/schema/policy versions, evaluation dataset ID and hash, applicability domain (task kind, worker, model version, prompt release, output schema, horizon, and data regime), ordered feature definitions, normalization ranges and missing-value policy, monotonic directions, calibration method and parameters, thresholds, creation/review metadata, and signature. Feature normalization clamps only as declared by the artifact and records out-of-range values. The score and reason vector are reproducible from event references.

If the artifact is missing, invalid, incompatible, outside its applicability domain, or requires a missing feature, ConfidenceGate fails closed: it cannot authorize success or active trading. It may authorize the one predeclared safe evidence-retrieval recovery when all other gates and budgets permit; otherwise it enters the request-class-specific DEGRADED/unknown or DEGRADED/no_trade path.

- The initial versioned policy uses a success threshold of 0.85 and an abstention threshold of 0.45; changes require the evaluation release gate.
- Confidence at or above 0.85 permits success only when every mandatory schema, evidence, permission, and risk gate passes.
- Confidence at least 0.45 but below 0.85 permits exactly one declared retrieval, correction, adjudication, or model escalation when budgets allow.
- After that recovery, a score still below 0.85, or inability to reserve the recovery, enters the degradation chain and terminates as DEGRADED/unknown for informational work or DEGRADED/no_trade for active trading, with terminal_reason=confidence_recovery_exhausted.
- Confidence below 0.45 permits no further model work and immediately enters the compatible local-knowledge path. An accepted informational result ends DEGRADED/answer; a miss or rejection ends DEGRADED/unknown, while active trading ends DEGRADED/no_trade.
- Confidence never overrides a hard gate.

One deterministic recovery is allowed per unique cycle signature. It must change an authorized logical input: correction context, model route, evidence/context snapshot, or adjudication result. Returning to the same cycle signature terminates the branch and applies degradation. Ambiguous order side effects enter reconciliation and are never blindly retried.

Loop events include task/state_transitioned, task/progress_advanced, task/heartbeat, task/stalled, task/cycle_detected, task/recovery_started, task/recovery_failed, and task/terminated.

## Model routing and cost accounting

Routes are fixed by WorkerSpec and versioned policy. The initial policy uses:

- Luna for simple filtering, classification, and all reflection;
- Terra then Luna for bounded specialist analysis and result synthesis;
- Sol then Terra then Luna only for declared high-difficulty escalation.

The three deny-by-default reflection targets are reflect_decision, reflect_escalation_if_used, and reflect_final_result. No other node invokes reflection unless an evaluated policy release adds it.

Short and long context bands are explicit. Pricing separately accounts for input, cached input, cache writes, output, and tool calls with Decimal. Price, model version, prompt/schema/tool versions, and supported generation parameters are pinned per reservation. Unsupported temperature is omitted.

## Retry, timeout, circuit breaking, and degradation

Errors are classified only by the trusted versioned ErrorClassifier from observed transport/tool facts. Authentication, permission, trace, schema-policy, risk, audit, and deterministic business conflicts do not retry under the same grant. Retryable timeout, 408, 429, provider 5xx, and temporary source failures use capped exponential backoff with full jitter and valid Retry-After, subject to deadlines, attempts, cost, and LoopGuard. HTTP 409 is retryable only when the provider adapter explicitly identifies a transient transport conflict and the operation's idempotency contract proves repetition safe. Idempotency-key mismatch, workflow-state conflict, broker/order conflict, and ambiguous side effects enter deterministic conflict handling or reconciliation and never automatic retry.

Circuit breakers are isolated by provider/model/tool/data source and implement closed, open, and bounded half-open states. Redis coordinates production state with a process-local safe fallback. An open circuit follows only:

~~~text
next lower model tier -> compatible verified local knowledge -> structured terminal fallback
~~~

An accepted lower-tier or local informational result ends DEGRADED/answer; only a miss or rejection proceeds to DEGRADED/unknown or DEGRADED/no_trade by request class. Trade decisions are never recovered from an answer cache or local-knowledge fallback. Required evidence unavailability fails closed.

## Prompt and context engineering

System prompts, generation parameters, tool manifests, and output schemas live in immutable Git-tracked releases. A run pins one release. Atomic activation and one-click rollback affect only new runs, require evaluation gates, and are audited. Runtime Git checkout is forbidden.

Requests are ordered for provider prefix caching:

1. stable Harness identity and worker instructions;
2. stable abstention, safety, and structured-output rules;
3. stable ordered tool schemas visible to that worker;
4. append-only prior validated history when applicable;
5. dynamic task, context summary, memory summary, correction context, and current data in user/context content.

Dynamic timestamps, trace IDs, user data, market data, memory text, and correction text never enter the stable system prefix. Tool visibility and order remain stable within a request epoch. Scope, schema, release, or route changes open a new epoch and record the expected cache impact.

Every worker system prefix contains the stable abstention invariant: when accepted evidence does not support a required conclusion, emit knowledge_status=unknown and do not guess. Locale-specific wording is added only by the renderer, so the safety rule remains cache-stable.

Before each worker handoff, context selection emits a bounded deterministic ContextSummary with input/selection hashes, evidence IDs, conflicts, omissions, staleness, and token/byte limits. Raw long-term memory and unrestricted sources are never forwarded wholesale.

## Tool calling and permissions

Permissions are deny-by-default. A short-lived CapabilityGrant binds trace, run, work item, attempt, plan revision, tenant/scope, expiry, readable artifact classes, one writable invocation-state key, allowed tools/actions, schemas, rate limits, and fencing token.

ScopedToolBroker.call is the only Agent tool path. It validates the grant, typed arguments, scope, risk/approval prerequisites, circuit state, call budget, deadline, result schema, redaction, and trace. Tools expose only model-facing schemas; host-only executor metadata never enters prompts.

Specialists have no durable-write credentials. Audit, cache, queue, object, database, memory, and broker operations use separate service identities. Exchange/order tools are never visible to analysis workers.

## Structured outputs and reflection

Each worker uses one versioned strict JSON Schema/Pydantic contract with additional properties forbidden. Prose wrappers, Markdown fences, multiple JSON values, NaN/Infinity, unknown enums, missing fields, excessive lengths, trace/evidence mismatch, and cross-field contradictions are invalid.

Only the three configured core targets receive Luna reflection. The validator performs objective checks and emits allowlisted PASS, FAIL, or NOT_VERIFIABLE findings with check IDs, field locations, and evidence locations. It does not score style, strategy quality, profitability, or private reasoning and cannot mutate its target.

A deterministic policy chooses acceptance, targeted correction, full rewrite fallback, adjudication, or safe rejection. Correction receives a bounded typed CorrectionContext with errors and evidence references. It permits one allowlisted patch followed only when necessary by one complete rewrite. Every corrected candidate passes full validation and the applicable reflection target again. Strict improvement is mandatory; regression or a repeated hash stops correction.

Reflection has a dedicated fail-closed policy and is exempt from generic model/local-knowledge degradation. Only Luna may execute reflection; timeout, circuit-open, invalid schema, or unavailable Luna never switches reflection to another model and never substitutes local knowledge as a validator. Failure of reflect_decision or reflect_escalation_if_used rejects the target and enters the deterministic DEGRADED/unknown or DEGRADED/no_trade safe path. reflect_final_result runs only when optional synthesis produced a candidate; the authoritative WorkflowResult is already sealed, so failure discards that presentation and uses the deterministic renderer. If synthesis produced no candidate, Harness skips reflect_final_result and renders deterministically. Reflection output is never itself reflected, cached as an answer, or treated as evidence.

## Fixed and semantic response caching

A versioned seed cache contains at least five safe frequent informational answers with aliases and category TTLs. Seeds are reviewed artifacts and are not generated at runtime.

Historical safe requests are embedded in PostgreSQL/pgvector with request/response timestamps, model ID/version, embedding version, prompt release/hash, schema name/hash, safety policy, locale, tenant/scope, context/knowledge fingerprint, evidence references, expiry, and invalidation reason.

A semantic hit returns directly only when cosine similarity is strictly greater than 0.95 and every compatibility, scope, freshness, safety, version, and fingerprint gate matches. Ties are deterministic. Live trade decisions, order actions, personalized risk decisions, and unresolved-conflict answers are never cacheable.

Expiry is hard: lookup rejects expired records before ranking; cleanup is idempotent; prompt/model/schema/embedding/safety/knowledge changes invalidate incompatible records.

## RAG and long-term memory

| Layer | Production storage | Local storage | Content |
|---|---|---|---|
| event | partitioned PostgreSQL JSONB and checksum-addressed S3 | SQLite WAL and local artifacts | immutable observations/source material |
| knowledge | versioned PostgreSQL, pgvector, full text; Redis hot cache | SQLite FTS/vector adapter | evidence-linked reusable knowledge |
| decision | versioned PostgreSQL decisions/outcomes/lessons | SQLite WAL | provisional/final decisions and outcomes |

Retrieval follows receive task -> vector Top-K candidates -> hybrid reranking -> bounded context injection -> execution/update. Ranking combines similarity, exact/full-text match, authority, confidence, freshness decay, outcome validation, applicability, contradiction penalty, diversity, and source coverage. Results retain memory/evidence IDs and provenance.

Workers read selected summaries only and may propose MemoryCandidate. Deterministic promotion validates source existence, evidence, duplication, contradiction, circular provenance, outcomes, tenant scope, and retention before writing. Decision lessons remain provisional until outcomes close them.

The long-term event-memory layer is distinct from HarnessEventStore. HarnessEventStore records execution and audit truth for a run; event memory records domain observations used by future analysis. Cross-links use immutable event IDs and hashes, never shared mutable rows.

Forgetting uses retention classes, confidence half-life, capacity eviction, archive verification, legal holds, protected references, tombstones, grace periods, and idempotent purge batches. Artifact, embedding, and cache cleanup follows verified tombstones. Missing required evidence yields the structured unknown or no_trade outcome for the request class rather than increased confidence.

## Risk and side effects

Deterministic risk validation owns action, direction, sizing boundaries, stops, scenarios, evidence sufficiency, and terminal no_trade. Models cannot choose leverage, credentials, order mutation, or bypass flags.

External order effects use immutable intents, idempotency keys, run/work/attempt and fencing identities, broker reconciliation, and append-only outcomes. Timeout or unknown broker response enters WAITING_RECONCILIATION and cannot retry until broker state is reconciled.

## Observability and production backend

Every request receives a fresh immutable nonzero 128-bit trace ID propagated through API, Harness, worker, model, tool, retry, cache, queue, database, memory, risk, and final response. Each operation receives a unique parented span.

Structured one-line JSON logs use bounded searchable fields and redacted artifact references. Metrics cover success/abstention/no-trade rates, API/model/tool/database latency and errors, token categories, cost, cache behavior, retries, degradation, circuits, task cycles, progress plateaus, confidence gates, queue lag, memory lifecycle, and prompt releases. High-cardinality IDs use trace exemplars rather than metric labels.

Production uses PostgreSQL/pgvector, Redis for cache/locks/circuit coordination, checksum-addressed S3-compatible storage, and Redis Streams as the initial at-least-once queue fed by a transactional outbox. The queue remains behind a provider protocol so RabbitMQ or Kafka can replace it without changing Harness contracts. Consumers use idempotency, leases, fencing, bounded retries, and dead-letter queues. Local development uses SQLite WAL/FTS and process-local adapters.

The initial HTTP API is:

~~~text
POST /v1/workflows
GET  /v1/workflows/{run_id}
POST /v1/workflows/{run_id}:cancel
GET  /v1/workflows/{run_id}/events
POST /v1/admin/prompt-releases/{version}:activate
POST /v1/admin/prompt-releases:rollback
~~~

Creation accepts Idempotency-Key and returns 202 with run_id, trace_id, status, and status URL for asynchronous execution. Clients may supply a correlation ID, but the service always creates its own trace ID and records the correlation as a link. Cancellation is idempotent. Event reads are authorized, filtered, redacted, cursor-paginated, and bounded. Administrative release operations require a separate role and evaluated known-good release. The synchronous DiscretionaryLLMEngine.get_playbook contract remains compatible through an adapter.

Performance controls include bounded concurrency, parallel independent workers, immutable snapshots, indexed/paginated queries, batching, prepared statements, connection pools, artifact spilling, context/token limits, backpressure, and load shedding. Mandatory audit failure is fail-closed; optional telemetry export failure is isolated.

## Testing and evaluation corpus

A versioned JSONL evaluation corpus is stored in the repository with schema, fixtures, expected invariants, metadata, dataset version, review history, and train/development/holdout partitions. It covers:

- active and passive workflows;
- cache hit/miss/false-hit/expiry/invalidation;
- RAG retrieval, conflicts, stale sources, and citations;
- memory promotion, outcomes, decay, archive, tombstone, and purge;
- strict output, injection, permissions, trace propagation, and replay;
- timeout, backoff, jitter, circuit, model downgrade, local knowledge, and unknown/no-trade;
- reflection acceptance, patch, rewrite, regression stop, and outage;
- repeated actions/results, A-B-A and longer cycles, cross-worker oscillation, fake heartbeat, no-progress plateau, confidence termination, and every budget dimension;
- crash/resume, duplicate delivery, stale results, fencing, event corruption, and broker ambiguity;
- prompt activation/rollback, prefix stability, and request reconstruction.

Release metrics include schema success, end-to-end success, safe abstention, hallucination/evidence rates, risk violations, cache false hits, retrieval quality, reflection/correction effectiveness, loop-detection precision/recall, trace completeness, latency, tokens, and cost. Releases use paired baselines with confidence bounds. Any hard safety, permission, trace, audit, or trading-risk regression blocks activation.

## Module boundaries

- workflow_harness.py: kernel commands and run lifecycle.
- workflow_session.py: events, fold, replay, snapshots, and leases.
- workflow_state_machine.py: legal transitions and terminal rules.
- workflow_plan_registry.py: templates and deterministic compiler.
- workflow_worker_registry.py: immutable worker specifications.
- workflow_loop_guard.py: fingerprints, cycles, and progress enforcement.
- workflow_confidence_calibration.py: calibrator artifacts, feature computation, and confidence gates.
- workflow_execution_backend.py: backend protocol and LangGraph adapter.
- workflow_agent_driver.py: bounded worker loop and model integration.
- workflow_prompt_config.py: releases, assembly, activation, and rollback.
- workflow_capabilities.py: grants and scoped tool broker.
- workflow_model_routing.py: immutable model policies.
- workflow_budget.py: hierarchical reservations and settlements.
- workflow_retry.py: error taxonomy and backoff/jitter.
- workflow_error_classifier.py: trusted observation-to-control error classification.
- workflow_circuit_breaker.py: isolated breaker registry.
- workflow_response_cache.py: fixed safe seed/exact cache.
- workflow_semantic_request_cache.py: vector cache and expiry.
- workflow_context_summary.py: bounded evidence-linked handoffs.
- workflow_long_term_memory.py: memory contracts.
- workflow_memory_postgres.py and workflow_memory_sqlite.py: repositories.
- workflow_memory_retrieval.py, workflow_memory_promotion.py, workflow_memory_lifecycle.py: retrieval, update, forgetting.
- workflow_reflection_agent.py: Luna objective checker.
- workflow_correction.py: typed patch/rewrite guard.
- workflow_risk_gate.py: deterministic trading risk.
- workflow_result_contracts.py: sealed outcome, knowledge status, and terminal-reason contracts.
- workflow_playbook_assembler.py: deterministic final contract.
- workflow_locale_renderer.py: allowlisted presentation fields and deterministic fallback rendering.
- workflow_observability.py: spans, logs, metrics, and event linkage.
- workflow_evaluation.py: corpus runner and release gates.

Each file owns one coherent module. Service protocols and providers are not combined into a large runtime file.

## Migration from the current feature branch

Task 1 contracts/state, Task 2 audit/context, and Task 3 routing/budget are retained rather than restarted.

1. Add run/work/attempt/plan-version, event, transition, fencing, progress, and LoopGuard contracts. Keep deprecated CoordinatorPlan/AgentTask aliases only where compatibility requires them.
2. Resolve the known Task 2 legacy-classifier blocker with a positive legacy signature and current-corruption rejection.
3. Complete the two still-pending Task 3 invariants on HEAD 8e70799: BudgetOverflowError must expose its already committed settlement (the current exception does not), and each node remaining_attempts value must also respect the workflow-global remaining-attempt cap (the current snapshot does not).
4. Adapt audit into the canonical event source and make workflow state a deterministic fold.
5. Implement the global state machine, registries, LoopGuard, service seams, and LangGraph backend adapter.
6. Implement resilient AgentDriver, cache, capabilities, prompt releases, scoped tools, and degradation before specialist workers.
7. Add focused workers and reflection without creating workflow_coordinator_agent.py.
8. Add memory/RAG/forgetting, deterministic risk/assembly, backend integration, observability, and evaluation gates.
9. Run Harness shadow mode with side effects disabled and compare status, routes, retries, costs, and results.
10. Remove callback-only and monolithic orchestration after every entry point uses Harness.
11. Run verification, security scan, replay/fault injection, and independent review; then copy equivalent tracked content to multi-agent-trader while preserving separate histories.

## Acceptance criteria

- No LLM payload is interpreted as a control instruction. Validated content may enter only predeclared deterministic policy inputs and cannot directly create or alter a plan, graph edge, transition, retry, permission, budget, durable write, risk decision, or terminal status.
- Every run is replayable from a verified append-only event stream.
- Every worker has an immutable three-to-five-phase specification and bounded resources.
- Repeated actions/results, state cycles, cross-worker oscillation, false heartbeat progress, and confidence plateaus terminate under tested limits.
- Run state, outcome_kind, knowledge_status, and terminal_reason distinguish successful no_trade, degraded unknown, degraded safety no_trade, failure, and cancellation.
- Multi-scope, multi-quota reservation is atomic; overdrawn settlement blocks new dispatch and follows the declared terminal policy.
- One authorized recovery cannot repeat the same cycle.
- Prompt prefixes and tool schemas remain stable within an epoch and invalidation causes are explicit.
- Cache, memory, RAG, reflection, correction, degradation, and forgetting enforce safety and provenance gates.
- Model/tool calls reserve before dispatch and settle actual usage after every outcome.
- Trace, spans, audit events, logs, metrics, and artifact hashes cover the full path.
- Prompt releases and generation parameters are versioned, evaluated, atomically activated, and rollbackable for new runs.
- The evaluation corpus measures success and safety and blocks hard safety regressions.
- External API behavior remains compatible and both repositories end with equivalent tracked application code and tests.
