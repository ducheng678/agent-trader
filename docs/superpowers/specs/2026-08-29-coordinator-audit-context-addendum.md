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
- semantic request-cache embedding, filtered candidates, similarity, compatibility, expiry, model/schema metadata, and result;
- tool call requested, allowed, completed, or denied;
- structured output validation result;
- reflection target/schema hashes, Luna attempt, format/field/conclusion-data checks, contradictions, disposition, and coordinator response;
- correction-context/error codes, field paths, evidence references, targeted patch/application, fallback rewrite linkage, replacement output hash, and resolution;
- correction guard before/after objective error tuples, output-hash cycle checks, regression stop reason, and safe terminal;
- retry, timeout, cancellation, and cost reservation or settlement;
- exponential-backoff ceiling, full-jitter/server delay, scheduled/actual wait, and circuit-breaker state/probe/rejection/recovery;
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

`request_normalizer -> seed_cache_lookup -> coordinator_plan -> context_select -> context_summarize -> dispatch_specialists -> collect_reports -> coordinator_reconcile -> decision_planner -> reflect_decision -> risk_or_escalation -> reflect_escalation_if_used -> coordinator_summary -> reflect_coordinator_summary`

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
- `workflow_long_term_memory.py`: three-layer event, knowledge, and decision memory with governed promotion and retrieval.

## Three-Layer Long-Term Memory

Long-term memory follows a governed three-layer hierarchy:

1. The event layer stores immutable raw material: normalized external events, market snapshots, source records, specialist reports, tool results, timestamps, provenance, hashes, and uncertainty. Corrections append a superseding record rather than rewriting history.
2. The knowledge layer stores distilled reusable experience: validated patterns, operational rules, counterexamples, applicability constraints, confidence, evidence links, version, effective time, expiry, and invalidation reason. It contains no unsupported live market facts.
3. The decision layer stores trading lessons: the accepted or rejected playbook, risk-gate reasons, evidence and knowledge versions used, execution outcome when available, post-decision evaluation, and the lesson learned from the buy, sell, wait, or no-trade result.

Memory flows upward only through a governed promotion pipeline. Agents may propose an event normalization, knowledge candidate, or decision lesson, but they cannot mutate durable memory directly. The coordinator validates the proposal, deterministic policy checks provenance, minimum evidence, schema, duplication, contradictions, and retention rules, and the memory repository performs the append-only write. Knowledge promotion normally requires multiple supporting event records or one explicitly authoritative source plus outcome validation. Decision lessons remain provisional until an outcome or review closes them.

Retrieval is task-scoped. The coordinator searches decision lessons first for closely matching completed situations, then knowledge for reusable rules, then event records for supporting raw evidence. Results retain memory IDs, evidence IDs, timestamps, versions, confidence, and staleness. Retrieved records pass through the context selector and `ContextSummary`; raw long-term memory is never forwarded wholesale to a specialist or inserted into the system prompt.

Contradictory knowledge is not overwritten. The repository creates a conflict set, lowers effective confidence, and returns it to the coordinator for a bounded reconciliation task. Superseded or expired knowledge remains auditable but is excluded from normal retrieval. A memory result that is stale, weak, contradictory, or insufficient cannot justify a live trade and resolves to `不知道` or `no_trade`.

Every memory read, candidate proposal, duplicate decision, promotion, rejection, conflict, supersession, expiry, and decision-outcome update is recorded in the full-chain audit trail. Memory payloads use the same secret redaction and tenant/scope boundaries as audit data. Retention, maximum rows, compaction, and archival are deterministic background policies; compaction never removes evidence required by an active knowledge or decision record.

### Storage Architecture by Layer

PostgreSQL is the production system of record for all three layers so promotion, evidence links, decision outcomes, and audit references can share transactions and enforced foreign keys. Layers use separate tables, indexes, retention rules, repository interfaces, and database roles; they do not share an untyped document bucket.

The event layer uses append-only, time-partitioned PostgreSQL tables keyed by tenant/scope, event type, observed time, source ID, and content hash. JSONB stores bounded normalized attributes, while frequently filtered fields remain typed columns. Large source documents, chart images, response bodies, and binary tool artifacts live in S3-compatible object storage with immutable object version, URI, media type, byte length, checksum, encryption key reference, and retention class stored in the event row. Monthly partitions, BRIN indexes on time, B-tree indexes on source/symbol/hash, and duplicate uniqueness constraints support ingestion and replay. Hot partitions retain full normalized payloads; archival moves closed partitions and unreferenced objects according to policy without deleting evidence referenced by active knowledge or decisions.

The knowledge layer uses versioned PostgreSQL relational tables for knowledge items, revisions, evidence links, counterexamples, conflicts, applicability constraints, approvals, supersession, and expiry. Search combines PostgreSQL full-text search with `pgvector` embeddings; exact aliases and deterministic token overlap remain available and take precedence for curated operational answers. Embeddings are derived data tied to the knowledge revision and embedding-model version, never the canonical text. HNSW or IVFFlat vector indexes are rebuilt by versioned background jobs, while B-tree indexes cover status, category, scope, version, effective time, and expiry. A knowledge revision cannot become active unless its cited event rows exist and promotion policy passes.

The decision layer uses strongly typed PostgreSQL transaction tables for decision requests, accepted/rejected playbooks, risk-gate records, selected knowledge revisions, cited event records, execution references, outcome snapshots, review status, and final lessons. Foreign keys freeze the exact knowledge revisions and event IDs used at decision time. Unique workflow/decision IDs provide idempotency, and indexes cover symbol, action, decision time, outcome status, strategy version, and review state. Decisions are append-only by revision: outcome and lesson updates create linked records instead of overwriting the original rationale.

Redis is an optional acceleration layer for exact seed answers, hot knowledge-query results, recent decision summaries, duplicate-suppression keys, and short-lived distributed locks. Every cache entry contains source revision/version and TTL; a cache miss or Redis outage falls back to the system of record. Redis never owns canonical memory and cannot authorize a trade.

The message queue transports asynchronous memory work such as event normalization, knowledge-candidate generation, embedding creation, decision-outcome reconciliation, expiry review, and archival. Messages use an outbox table, idempotency key, schema version, retry/dead-letter policy, and trace ID. A queued message is not evidence of a completed memory write.

Local development and tests use SQLite WAL behind the same repository interfaces. Event and decision layers use typed relational tables; knowledge uses versioned tables plus FTS5 when available and deterministic exact/token retrieval otherwise. Large artifacts use a configured local object directory containing content-addressed immutable files. SQLite mode does not pretend to provide production distributed locking or vector similarity; tests that require PostgreSQL, pgvector, Redis, or object storage are opt-in integration tests.

Database roles are least-privilege: ingestion appends events, the promotion worker proposes and activates knowledge through stored repository operations, the decision service appends decisions/outcomes, and ordinary specialist agents receive no database credentials. Encryption in transit and at rest, tenant/scope predicates, bounded query limits, backup/restore drills, partition maintenance, object lifecycle, and migration versioning are mandatory production controls.

### Governed Forgetting

Forgetting is a policy-driven lifecycle, not an untracked delete. Every record has tenant/scope, retention class, created/effective/last-used times, reference count, legal-hold flag, lifecycle state, and policy version. The lifecycle is `active -> decayed -> superseded_or_expired -> archived -> tombstoned -> purged`. Each transition is idempotent and audited with the selected policy and affected evidence IDs.

Event memory forgets primarily by age, importance, duplication, and reference reachability. Recommended defaults are configurable: duplicate/transient normalized events remain hot for 30 days; ordinary market events remain hot for 180 days and cold-archived for two years; source records required by active knowledge, an open decision, an unresolved outcome, audit policy, or legal hold are protected regardless of age. Large objects follow the event retention class and use object-store lifecycle rules. Content-addressed objects are deleted only when no protected row references the checksum.

Knowledge memory forgets through time decay before deletion. Effective retrieval confidence is computed from base confidence, evidence freshness, contradiction state, observed outcome quality, and category-specific half-life. Suggested starting half-lives are 30 days for regime-specific knowledge, 180 days for market-behavior heuristics, and no automatic decay for curated operational/safety policy until a new version supersedes it. A record falling below the retrieval threshold becomes inactive, remains available for audit and conflict analysis, and is archived after a configurable grace period. Repeated validated use can refresh last-used time but cannot erase contradictory evidence or raise confidence without a new reviewed revision.

Decision memory preserves the decision, evidence versions, risk result, execution/outcome, and lesson as one reachable chain. Open positions, incomplete outcomes, disputes, and lessons referenced by active knowledge are protected. Completed decisions default to a seven-year retention class for production deployments, configurable to jurisdiction and tenant policy; after expiry they may be anonymized into aggregate evaluation statistics before the identifiable chain is tombstoned. Anonymization is a governed transformation with its own source hash and audit event, not silent mutation.

Capacity forgetting complements time retention. Each tenant/scope has byte and row budgets per layer. When a budget is approached, deterministic scoring prefers eviction of duplicate, low-confidence, stale, unreferenced, low-utility records while protecting authoritative evidence, counterexamples, rare failure modes, active conflicts, and records required for reproducibility. Cache eviction never changes durable-memory lifecycle state.

A scheduled memory-lifecycle worker evaluates policies in bounded batches using an outbox/idempotency key, distributed lease, checkpoint cursor, maximum runtime, retry limit, and dead-letter queue. The default run is dry-run capable and emits counts/bytes by proposed transition before mutation. Purge requires expiry plus grace period, zero protected references, no legal hold, successful archive verification when archival is required, and a transactional tombstone. Physical purge covers PostgreSQL rows eligible for deletion, derived embeddings/search indexes, Redis keys, local files, object-store versions, and expired backup generations according to their independent retention schedules.

Retrieval filters inactive/tombstoned records by default and reports when relevant memory was excluded as stale or conflicted. Forgetting cannot turn missing evidence into confidence: if required memory has expired or was purged, the coordinator records the gap and returns `不知道` or `no_trade`. Metrics cover active/decayed/archived/tombstoned rows, protected references, lifecycle lag, purge failures, dead-letter jobs, reclaimed bytes, and retrieval misses caused by expiry.

Administrative restore, legal hold, tenant erasure, and policy change operations use authenticated service interfaces, two-person approval where configured, and full audit. A policy change never retroactively purges data in the same transaction; it first produces an impact report and then runs through the normal lifecycle worker.

### Retrieval and Application Closed Loop

Long-term memory retrieval and application follows four mandatory stages.

Stage 1, receive the new task: the coordinator normalizes the user objective, immutable constraints, tenant/scope, locale, symbols, time horizon, task type, risk class, required evidence types, and maximum memory token budget. It creates a traceable `MemoryQuery` and checks the exact high-frequency seed cache first for safe informational intents. It then creates a `SemanticCacheQuery`, embeds the normalized historical-request projection, and asks the vector cache for the highest compatible unexpired result. A result is returned directly only when cosine similarity is strictly greater than `0.95` and tenant/scope, intent, locale, prompt, model policy, schema/hash, safety policy, knowledge/context fingerprint, and freshness gates all match. Live trading requests never terminate from a cached trading answer.

The semantic-cache audit records the query hash, embedding model/version, threshold, filtered candidate IDs and scores, compatibility rejections, selected cache ID, original request/response timestamps, model ID/version, prompt and schema versions/hashes, TTL class, expiry, and hit/miss reason. Expired entries are removed from eligibility before similarity ranking and are never used by local-knowledge fallback. Vector cache failure is a cache miss, not a workflow failure.

Stage 2, vector retrieval: the memory service embeds the normalized task with the configured embedding model/version and performs filtered Top-K retrieval. It first searches completed decision lessons for comparable situations and active knowledge revisions for reusable experience; it follows their evidence links into the event layer when raw support is required. Filters enforce tenant/scope, symbol/market applicability, lifecycle state, effective/expiry time, knowledge version, legal visibility, and maximum staleness before similarity search. Recommended defaults are configurable `decision_k=5`, `knowledge_k=8`, and `event_evidence_k=12`.

Vector similarity is a candidate generator rather than the acceptance rule. Hybrid ranking combines vector similarity, full-text or exact-alias match, evidence authority, confidence, freshness decay, outcome validation, applicability, contradiction penalty, diversity, and source coverage. Exact curated aliases outrank approximate matches. Near-duplicate candidates are collapsed by canonical hash. Results below the minimum final score are excluded, and conflicting high-ranked memories create a coordinator conflict instead of being averaged into false certainty.

Stage 3, context injection: the selector builds a bounded `CoreExperienceSummary` rather than injecting raw retrieved rows. It contains the task-relevant lesson or rule, applicability and non-applicability conditions, supporting and contradicting evidence IDs, decision/knowledge revision IDs, confidence, freshness, conflict status, omitted-result counts, retrieval scores, and a deterministic hash. The summary is treated as untrusted data, appears only after the stable system prompt in dynamic user content, and is wrapped with an instruction that retrieved memory cannot override system, tool, schema, risk, or user constraints. Numeric facts, units, negation, timestamps, and provenance must survive summarization.

The memory token budget is allocated deterministically across layers, with decision lessons and active knowledge receiving priority and event evidence retained for verification. When the budget is exceeded, diversity-aware compression keeps the strongest supporting item, strongest counterexample, and authoritative evidence before additional similar items. The audit store records the query hash, embedding version, filters, candidate IDs and scores, reranking factors, selected IDs, omissions, and injected-summary hash without storing secrets or private reasoning.

Stage 4, execute and update: specialists execute only from the typed task plus `CoreExperienceSummary`. Their reports cite the memory/evidence IDs actually used and mark ignored or contradicted memory. The coordinator and deterministic risk gate produce the response. The workflow immediately appends new raw observations and reports to event memory and appends a provisional decision record. Execution results or later evaluation update the decision through linked outcome records. Only the governed promotion pipeline may convert validated outcomes into a knowledge revision or finalized decision lesson.

The closed loop prevents self-reinforcing errors. Model output cannot be written back as validated knowledge merely because it retrieved and repeated an earlier model output. Promotion requires independent evidence or a verified outcome, excludes the candidate's own descendants from support counts, records negative outcomes and `no_trade` lessons, and checks for circular provenance. Retrieval effectiveness is measured using later outcomes, conflict rate, abstention accuracy, citation coverage, and stale-memory rejection; these metrics may propose ranking changes but cannot silently rewrite historical scores or memory content.

If embedding generation or vector search fails, retrieval degrades to exact aliases, PostgreSQL full-text search, deterministic token overlap, then no memory. If memory is missing, stale, weak, or contradictory, the task continues only with independently supplied current evidence; otherwise the terminal answer is `不知道` or `no_trade`. No retrieval failure triggers invention.

### Agent Capability and Permission Model

Permissions are deny-by-default and enforced outside prompts. Each node receives a short-lived `CapabilityContext` containing workflow/task/tenant scope, permitted read views, permitted tools, allowed graph-state output keys, call/tool/cost limits, and expiry. Repository credentials, exchange credentials, unrestricted database handles, environment variables, and general tool registries are never placed in an agent context.

An agent `write` means returning its strict result for one allowed invocation-state key. Specialists do not write durable storage. LangGraph reducers validate actor, task ID, schema, and destination key before accepting a result. Durable audit, cache, queue, object, database, and memory operations are performed only by deterministic services with separate service identities.

| Actor | Read permissions | Allowed actions / graph writes | Explicit denials |
| --- | --- | --- | --- |
| coordinator agent | normalized request, policy catalog, budget snapshot, typed summaries, specialist reports, conflict bundle, audit health | create/revise bounded tasks, select permitted model policy, write coordinator plan/route/final summary, request approved services | no raw credentials, raw unrestricted memory, direct database/cache/queue writes, web/exchange calls, risk bypass, order mutation |
| context selector/summarizer | task-scoped source records, permitted retrieved-memory records and provenance | write one `ContextSummary`/`CoreExperienceSummary` for the assigned task | no model tools by default, no memory promotion, no source mutation, no arbitrary conversation dump |
| market-context agent | current task summary, bounded event inputs, allowed market scope | read-only `web_search` within query/tool caps; write `MarketContextResult` | no order/exchange tools, no memory/database/cache writes, no filesystem/network client outside registered search |
| event-filter agent | trigger-event and recent-event summary, duplicate fingerprints | write `EventAssessment` | no tools, chart reads, direction/prices, durable writes |
| fundamental agent | event/market/core-experience summaries and evidence references | write `FundamentalAnalysis` | no chart pixels, execution values, tools, durable writes |
| technical agent | bounded chart text/images, market snapshot, core-experience summary | write direction-neutral `TechnicalAnalysis` | no web tools, event reinterpretation, final direction, durable writes |
| decision-planner agent | validated fundamental/technical reports and bounded summaries | write `DecisionDraft` | no tools, size/leverage, execution, durable writes |
| reflection agent | one validated redacted target output, its schema identity/hash, bounded evidence summary, and deterministic validation result | Luna-only no-tool consistency review; write one `ReflectionResult` for the assigned target hash | no target mutation/repair, no coordinator planning, no web/exchange/filesystem/database/cache/queue/memory access, no durable writes, no reflection of reflection output |
| escalation-reviewer agent | conflict bundle, cited summaries, deterministic rule result | write `EscalationReview` | no new facts/tools, symbol changes, risk bypass, execution, durable writes |
| deterministic risk gate | validated graph state and budget/audit health | write `RiskAssessment` only | no model/tool calls, no durable writes, no policy override |
| playbook assembler | accepted state and exact normalization/cap policy | write final invocation result | no model/tools, no execution or durable memory writes |
| memory promotion service | audited candidates, source events, outcomes, promotion policy | append event/knowledge/decision revisions through scoped repository operations | no arbitrary SQL, no model decision authority, no deletion or lifecycle override |
| memory lifecycle worker | lifecycle metadata, references, legal holds, retention policy | archive/tombstone/purge only records selected by validated plan | no model calls, no active/protected record purge, no knowledge promotion, no decision changes |
| audit writer | redacted typed audit event | append only | no update/delete and no unredacted secrets |

Tool dispatch validates the capability at call time, not only task creation. Capability IDs are included in audit events; tokens themselves are never logged. A task reschedule receives a new capability and cannot reuse an expired or broader prior grant. Parallel agents receive isolated contexts.

Static tests reject forbidden imports and direct client/repository calls. Runtime tests attempt unauthorized state keys, tools, tenant IDs, memory writes, SQL handles, and exchange operations and require a typed permission denial plus audit event. Permission failure is non-retryable at the same grant; the coordinator may only reschedule after deterministic policy issues a corrected narrower capability.

### Structured Agent Output Enforcement

Every coordinator and specialist invocation uses a versioned Pydantic contract rendered as an OpenAI strict JSON Schema. The request sets strict structured output, forbids additional properties, and supplies exactly one schema for the node. Free-form prose, Markdown fences, leading or trailing text, multiple JSON values, NaN/Infinity, unknown enums, missing required fields, excessive list/string lengths, and cross-field contradictions are invalid outputs.

Every output includes `schema_version`, knowledge status, uncertainty reason, bounded evidence references, and only the concise conclusion/reason fields defined by its contract. It never includes hidden chain-of-thought. Prompt instructions to reason step by step affect internal analysis only; the response contains the validated result, key evidence, and uncertainty.

`AgentRunner` parses the complete response once, validates it locally against the same Pydantic model, and rejects partial recovery or best-effort field dropping. A schema failure is an audited retryable attempt within the node's existing attempt/time/cost budget. Exhaustion follows the configured lower-model, local-knowledge, and explicit-unknown path. A malformed output can never reach a reducer, memory proposal, risk gate, cache, API response, or final playbook.

After deterministic parsing, only three configured core outputs are reviewed by the Luna-only reflection agent: the decision planner's draft, the conditional Sol escalation review, and the coordinator's final summary. It checks the declared schema/required-field result and semantic consistency between conclusion, numbers, direction, cited evidence, uncertainty, and the bounded source summary. Its strict disposition is `accept`, `retry_original`, `return_to_coordinator`, or `safe_reject`. It cannot change the target payload. Non-accepted core targets remain outside their downstream state, cache, memory, risk, and API boundaries. Non-core Agent outputs receive deterministic validation without an LLM reflection call. Reflection unavailability for a core trading output fails closed; the reflection output itself receives deterministic validation only, preventing recursive reflection.

For `retry_original`, the coordinator receives a deterministic `CorrectionContext` rather than raw reflection text. It carries only target and reflection hashes, typed error/invariant codes, affected field paths, contradiction/missing-evidence references, retry ordinal, and bounded summaries of the prior output and original task. The first retry receives this context in dynamic user content and returns a strict targeted `CorrectionPatch`. A deterministic service applies only allowlisted field replacements to a copy and fully revalidates the result. Only patch-generation/application failure, full-schema failure, or another rejected reflection may trigger one fallback full rewrite with bounded correction history. The rewritten result must be complete and strict. One patch plus one rewrite is the hard maximum; all attempts share existing time/retry/cost limits, and exhaustion fails closed.

Reflection is objective-only. Luna emits allowlisted checks with `pass`, `fail`, or `not_verifiable`, field paths, evidence IDs, observed values/hashes, and expected constraints. It cannot judge strategy quality, market opinion, profitability, prose, or confidence and cannot choose the workflow disposition. Deterministic policy maps required checks to a disposition. A deterministic correction guard permits at most one targeted patch and one fallback rewrite, requires strict improvement of the objective error tuple, detects repeated/cycling hashes, and stops immediately on new critical errors, weakened risk rules, unsupported direction changes, or lost evidence.

### Trace and Span Propagation

Each ingress creates one fresh internal 128-bit trace ID and an initial span before normalization. The ID is immutable and is passed explicitly in every request, task, report, summary, capability, audit event, cache lookup, model/tool call, retry, queue message, memory operation, log, metric exemplar, and final response. Every operation creates a child span; trace mismatches fail closed. Upstream trace context and cached-result origin traces are links, never replacements for the current request trace. Trace IDs are unique correlation identifiers and grant no permission.

### Logs, Metrics, Tool Traces, and Releases

Structured logs are typed one-line JSON and always include UTC time, trace/span hierarchy, workflow/task/attempt, actor/event/status, latency, model/prompt/schema/release identity, token classes, cost, and retry/cache/fallback/circuit outcomes. Metrics cover business success, abstention, interfaces, queues, Agent/LLM/tools, caches, memory, tokens, and cost with bounded labels and trace exemplars. Tool spans record capability-checked typed arguments/results as redacted summaries plus hashes, sizes, schemas, status, duration, and artifact references. Canonical audit remains complete even when remote telemetry export is sampled.

System prompts and generation parameters are immutable Git-tracked release files. Requests pin one release. Activation and rollback are atomic, authorized, hash/schema/model-capability/evaluation-gated, and fully audited. Rollback affects only new requests and returns to the immediately previous known-good release; dynamic values stay outside stable prefixes and unsupported temperature values are never sent.

The versioned evaluation corpus measures end-to-end success and safety across normal, unknown, adversarial, cache, routing, retry/circuit, reflection/correction, memory/RAG, permission, and trace paths. Release results bind code, corpus, prompts, schemas, model versions, latency, tokens, and cost. Hard safety failures block release regardless of average success; candidate/baseline comparison uses paired cases and confidence bounds.

Schema name/version and canonical schema hash are included in prompt-cache keys, response-cache keys, audit events, usage records, task contracts, and memory records derived from an output. Cache and memory reads require compatible versions; migrations are explicit deterministic transformations that retain the original payload/hash and audit linkage.

Tool arguments and tool results use separate strict schemas and capability validation. The coordinator cannot reinterpret an invalid specialist payload as a valid result. Reducers accept only the output model assigned to that actor and state key. API serialization uses the validated final contract rather than raw model text.

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
- Long-term memory is separated into immutable event material, distilled knowledge, and outcome-linked decision lessons.
- No agent writes durable memory directly; every promotion is coordinator-reviewed, policy-validated, source-linked, and audited.
- Retrieval returns versioned, freshness-aware records that are summarized before specialist handoff.
- Every agent response is strict, versioned, fully parsed structured output; malformed or extra text is retried or degraded and never enters state or memory.
- Decision drafts, conditional Sol escalation reviews, and coordinator final summaries pass exactly one Luna reflection gate; non-core outputs do not invoke reflection, reflection cannot mutate the target, and a reflection result is never recursively reflected.
- Reflection emits only objective falsifiable checks; deterministic policy owns disposition and stops correction on regression, cycles, or the one-patch/one-rewrite limit.
- Every ingress receives one fresh immutable internal trace ID that reaches every synchronous/asynchronous operation and final response; each operation has a unique parented span and cross-trace mutation is rejected.
- JSON logs, bounded-cardinality metrics, and Agent/LLM/tool spans expose success, latency, token, cost, arguments/results metadata, and failure paths without secrets, raw prompts, private reasoning, or unrestricted content.
- System prompts and supported generation parameters are immutable Git-tracked releases pinned per request; guarded atomic rollback restores the previous evaluated known-good release for new requests in one action.
- A versioned leak-checked evaluation corpus reports reproducible end-to-end success and safety metrics; hard safety failures or configured confidence-bound regressions block release regardless of average score.
