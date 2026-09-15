# Task 3 implementation report

Status: complete.

Commit: `58c236e00906a5d39e479fa3ca9847d467539b59` (`feat: add governed memory forgetting lifecycle`).

## Delivered

- `workflow_memory_lifecycle.py` adds frozen policy/scope/entry/action/plan/limits/result/outbox contracts, deterministic dry-run planning, a canonical plan hash, retention classes, immutable half-life confidence calculation, deterministic capacity selection, and bounded resumable application.
- `LifecycleWorker.plan("tenant-a", now=aware_datetime)` plans the whole tenant; `LifecycleScope(tenant_id="tenant-a", scope="default")` narrows it. The plan exposes `archive_ids`, `tombstone_ids`, and `purge_ids`. Every record advances only one phase per plan; default quarantine periods are one day archived and seven days tombstoned.
- `worker.apply(plan, LifecycleLimits(max_actions=100, max_cleanup=100), tenant_id=..., trace_id=..., idempotency_key=..., authority=...)` uses the existing service authority boundary. SQLite recomputes eligibility inside `BEGIN IMMEDIATE`, checks current record hashes, and rejects stale or forged actions. Legal holds, incoming references, permanent retention, and future observations protect records; current knowledge heads also cannot purge because they retain revision/CAS identity.
- Phase writes, timestamps, replay markers, redacted audits, and cleanup outbox tasks commit atomically. The test suite exercises stale references and legal holds added after dry-run planning, active-to-purge forgery, action limits, replay resumption, and audit rollback.
- Purge erases raw record snapshots from both memory records and idempotency replay storage. Hash-only purge markers prevent reappend/replay resurrection of the same event identity or content. Existing immutable event payload/provenance hashes remain intact through archival and tombstoning.
- Tombstoning queues vector/cache deletion; purging the last same-tenant artifact reference queues artifact deletion. Other live, archived, or held users of a shared artifact retain it. New attachments of an address already queued for deletion are denied to prevent a race with deferred cleanup.
- Cleanup tasks persist across repository reopening, carry original trace/context, retry idempotently, and audit acknowledgement without raw material. `FileArtifactStore.delete` validates authority/scope, persists deletion intent before unlink, and prevents put replay from resurrecting removed content. A crash after unlink but before acknowledgement safely retries.
- AgentDriver accepts only `CoreExperienceSummary` via `execute(invocation, memory_context=summary, memory_tenant_id="tenant-a", memory_scope="default")`. It revalidates hash and explicit tenant/scope binding; strings, mappings, arbitrary handles, subclasses, and forged instances do not reach the provider. It imports only the summary contract from the memory retrieval module and never imports memory storage, lifecycle, promotion, or object-store modules.
- Eligible summary JSON appears as a separate dynamic user message after the stable system prefix. Static instructions label memory untrusted. Weak, stale, conflicting, empty, failed, and oversized summaries inject no memory. Driver limits are confidence >= 0.6, reported age <= 86400 seconds, and <= 8192 UTF-8 bytes including metadata. Provider/fallback behavior remains bounded; insufficient evidence with unavailable model/local knowledge produces the existing exact `不知道` abstention.
- Memory-bound invocations bypass response caches because the existing exact/semantic cache contracts do not bind memory hashes or evidence versions. Audit input hashes include the summary hash, without logging summary content.

## Validation

- Initial lifecycle RED: 10 assertion failures because the worker module was absent; GREEN: 10 passed.
- Driver boundary RED: 11 failures for the missing memory keyword arguments and one existing import-isolation check passed; the existing AST boundary was narrowed to permit only the summary contract.
- Additional RED tests reproduced stale/weak/conflicting/oversized injection and replay starvation. Production fixes made these cases green; a later stale-first-action case verified protected actions do not starve the rest of a bounded plan.
- Final targeted command: `python -m pytest market_agent_test_bundle/tests/test_workflow_memory_lifecycle.py market_agent_test_bundle/tests/test_workflow_memory_storage.py market_agent_test_bundle/tests/test_workflow_memory_retrieval.py market_agent_test_bundle/tests/test_workflow_memory_promotion.py market_agent_test_bundle/tests/test_workflow_agent_driver.py -q` — **237 passed in 11.33s**.
- Full workflow gate: `python -m pytest market_agent_test_bundle/tests -q -k workflow` — **1024 passed, 1044 deselected, 4 warnings in 106.76s**. Four pre-existing Pydantic serializer warnings came from deliberate forged-instance tests in workflow audit/context summary. The final additional stale-action regression and its fix were verified in the targeted run after full-suite collection.
- `python -m compileall -q market_agent market_agent_test_bundle/tests`: passed.
- `git diff --check`: passed; only repository LF-to-CRLF normalization notices were emitted.

## Operational boundaries

- Construct the worker with `artifact_store=...` for local artifact cleanup and `cleanup_adapters={"vector": callable, "cache": callable}` for external derivatives. Callables receive `(CleanupTask, **WriteArguments)` and must honor task-id idempotency. Missing or failed adapters leave durable tasks pending, never falsely acknowledged.
- `LifecycleRepository` extends the base storage protocol separately, so non-SQLite implementations must explicitly provide lifecycle snapshot/application/outbox operations before they can host the worker. The base `ArtifactStore` protocol now includes its service-only delete operation.
- Retention never overrides legal holds or references, so protected data can keep capacity above its target. Unknown legacy archived/tombstoned transition timestamps are retained conservatively because the required grace period cannot be established. New archive transitions and knowledge supersession persist timestamps.
- The summary caller must retrieve/build a fresh summary for each invocation; the current summary contract reports evidence age but does not carry an issuance timestamp or repository capability. Driver screening uses the summary's validated reported freshness.
- No agents or `.tmpbudget` files were modified.

## Review fix 1: actual transition time, fair cleanup, summary expiry

Status: complete; supersedes the reported-age-only boundary described above.

Commit: `8a2d646` (`fix: enforce memory lifecycle time and summary expiry`).

- `SQLiteMemoryRepository(..., clock=callable)` defaults to actual aware UTC
  time. Lifecycle application samples that trusted clock inside its write
  transaction, rejects plans dated in the future, rechecks eligibility at apply
  time, and writes actual archive/tombstone transition timestamps. A delayed
  plan keeps its deterministic hash but cannot consume quarantine before the
  transition occurs. Knowledge supersession uses the same trusted clock.
- Cleanup persists an audited attempt sequence before external deletion and
  orders pending tasks by their last attempt. A failing first task rotates
  behind healthy tasks across worker/repository restarts. The existing
  per-invocation attempt cap, original trace, stable task idempotency key, and
  idempotent acknowledgement remain intact. An additive SQLite table/index
  supports existing databases without rewriting records. Other lifecycle
  adapters must implement the new `begin_cleanup` protocol operation.
- Summaries now carry hash-bound aware issuance/expiry timestamps from trusted
  retrieval. Expiry includes selected rules/lessons, event ancestry, outcomes,
  final decisions, explicit expiry dates, and the query's freshness cap. The
  driver validates timestamps against its clock and enforces its own age cap
  using elapsed time plus age at issuance. It checks again for retries and
  immediately before provider invocation after potentially slow audit work.
  Expired summaries inject no memory. Direct summary constructors and stored
  summary caches must include timestamps or be rebuilt through retrieval.
- RED: seven cases reproduced delayed-transition backdating, cleanup
  starvation after reopening, three deadline sources, summary reuse after
  expiry, and expiry during retry. One additional RED case reproduced expiry
  while the prompt audit callback ran. All eight now pass.
- Final focused gate: `python -m pytest
  market_agent_test_bundle/tests/test_workflow_memory_lifecycle.py
  market_agent_test_bundle/tests/test_workflow_memory_storage.py
  market_agent_test_bundle/tests/test_workflow_memory_retrieval.py
  market_agent_test_bundle/tests/test_workflow_memory_promotion.py
  market_agent_test_bundle/tests/test_workflow_agent_driver.py -q`:
  **245 passed in 11.64s**, exit 0.
- `python -m compileall -q market_agent market_agent_test_bundle/tests` and
  `git diff --check`: passed. Diff output includes only normal LF-to-CRLF
  normalization warnings. No agents or `.tmpbudget` files were modified.

## Review fix 2: completion-boundary summary expiry

Status: complete; supersedes the prior before-dispatch-only expiry claim.

- A provider response is now marked memory-bound only when its actual request
  included the dynamic summary. The driver rechecks that summary immediately
  after response validation, immediately after `schema_validated` audit, and
  immediately after `task_completed` audit (including reflection work before
  completion). If it expires at any of those acceptance boundaries, the output
  is discarded and the terminal `AgentResult` is the trace-preserving,
  non-retryable `memory_context_expired` failure with no output or usage.
- The trace records a redacted `memory_expired` rejection followed by
  `task_failed`; a successful provider/circuit record is retained as transport
  truth, never as an accepted answer. The audit reason registry now includes
  `memory_context_expired` instead of using an unregistered free-form reason.
- RED regressions advanced the injected clock during provider execution,
  `schema_validated` observer work, and `task_completed` observer work. All
  three previously returned the model result; all now discard it.
- Final focused gate: `python -m pytest
  market_agent_test_bundle/tests/test_workflow_memory_lifecycle.py
  market_agent_test_bundle/tests/test_workflow_memory_storage.py
  market_agent_test_bundle/tests/test_workflow_memory_retrieval.py
  market_agent_test_bundle/tests/test_workflow_memory_promotion.py
  market_agent_test_bundle/tests/test_workflow_agent_driver.py -q`:
  **248 passed in 11.57s**, exit 0.
- `python -m compileall -q market_agent market_agent_test_bundle/tests` and
  `git diff --check`: passed. Output had only normal LF-to-CRLF normalization
  notices. `.tmpbudget` was not touched.

## Review fix 3: completion audit terminal consistency

- Memory-bound model results now record `task_completed` as an output-free,
  `accepted`/`selected` candidate while its synchronous audit callback runs.
  Only after the callback returns and the summary remains valid does the driver
  append the successful `final_decision` event. If expiry occurs during that
  callback, immutable audit history instead ends in the candidate followed by
  redacted `memory_expired` and failed `task_failed` terminal events; no event
  in that trace represents an accepted answer. Non-memory invocations retain
  their existing single successful `task_completed` event.
- Regression coverage verifies both expiry during completion audit and the
  successful two-stage memory-bound completion path.
- Final focused gate: `python -m pytest -q
  market_agent_test_bundle/tests/test_workflow_memory_lifecycle.py
  market_agent_test_bundle/tests/test_workflow_memory_storage.py
  market_agent_test_bundle/tests/test_workflow_memory_retrieval.py
  market_agent_test_bundle/tests/test_workflow_memory_promotion.py
  market_agent_test_bundle/tests/test_workflow_agent_driver.py`:
  **249 passed in 11.68s**, exit 0. `python -m compileall -q market_agent
  market_agent_test_bundle/tests` and `git diff --check` also passed; only
  normal LF-to-CRLF notices were emitted. `.tmpbudget` was not touched.

## Review fix 4: final-decision audit expiry boundary

- `final_decision` is now also an output-free `accepted`/`selected` candidate
  for memory-bound model responses. The driver checks the trusted summary
  deadline after its synchronous observer returns before it returns the model
  output. This avoids a false immutable success if the observer consumes the
  final valid instant. Memory-bound audit trails intentionally contain only
  candidate selection records on success; the returned `AgentResult` is the
  acceptance boundary. Non-memory calls retain their existing successful
  `task_completed` audit record.
- A regression observer advances the trusted clock during `final_decision`.
  The trace ends with the two output-free candidates, `memory_expired`, and
  `task_failed`; no successful output or success outcome is recorded.

## Review fix 5: reflection candidate expiry boundary

- For a memory-bound response, `core_result_ready` is now an output-free
  candidate audit. The driver rechecks summary expiry immediately after that
  synchronous callback and before invoking the verification hook, so an audit
  callback that consumes the last valid instant cannot cause the hook to
  receive the response. It checks again when the hook returns before any
  completion candidate is emitted. Non-memory reflection keeps its existing
  output-bearing audit and hook behavior.
- The same candidate rule now applies to memory-bound `schema_validated`, so
  no synchronous observer receives a hash of an answer that may be discarded
  at the following expiry boundary.
- Regressions cover expiry during `core_result_ready` (the hook is never
  called and no output hash is emitted), stable reflection behavior, and a
  hook that itself consumes the remaining authority (the result is discarded
  before completion).
