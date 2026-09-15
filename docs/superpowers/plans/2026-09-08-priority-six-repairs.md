# Priority Six Repairs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans task-by-task.

**Goal:** Repair the six confirmed P1 production integration defects.
**Architecture:** Preserve current workflow/security contracts; introduce shared ownership and durable recovery at existing boundaries.
**Tech Stack:** Python, SQLite, PostgreSQL, Redis, LangGraph.
**Spec:** docs/superpowers/specs/2026-09-08-priority-six-repairs-design.md

## Global Constraints

Preserve all existing dirty changes. No commits, push, live models, live trading or cloud mutations. Implement in this isolated feature worktree; controller synchronizes only scoped changed files into the second repository. Every production change has a focused regression test. Do not claim live distributed validation from fake-adapter tests. Keep modules focused and avoid unrelated refactoring.

## Task 1: Queue ownership and retry (R01-R03)

Files: backend/task_queue.py, backend/database.py, backend/redis_adapters.py, backend/errors.py if necessary, and market_agent_test_bundle/tests/test_queue_ownership.py; existing focused tests may be adjusted only for changed intended semantics.

Interfaces:
- claim_job(job_id, execution_token, lease_seconds, *, recovery=False) -> JobRecord | None
- renew_job_lease(job_id, execution_token, lease_seconds) -> bool
- release_job_lease(job_id, execution_token) -> bool
- Existing mark_running/mark_retry_scheduled/mark_recovery_queued/mark_succeeded/mark_failed acquire optional keyword execution_token, checked transactionally; None works only on unleased legacy rows.
- JobRecord gains internal execution_token and lease_expires_at; do not expose ownership secrets through as_dict.
- SQLite serialized migration and PostgreSQL ADD COLUMN IF NOT EXISTS. Claim only absent/expired ownership, renew live exact token, fence every transition and terminal failure, release ownership on terminal.
- Durable publishers do not retain execution semaphore. Consumers acquire capacity, claim and schedule. Recovery claims directly, continuously polls, renews executor-queued/running/retry-delay ownership. Lost owners stop local retries and cannot write status/results.
- Redis transient TaskQueueFullError/DependencyUnavailableError/RedisUnavailableError/retryable errors remain pending with bounded backoff, never dead-letter or ACK.

- [x] Add failing tests: cross-process publisher capacity reuse; transient pending then success; two-connection claim winner; healthy owner not recovered; queued/retry jobs renewed; expired stale success/failure fenced; recovery continues after empty page; submit failure releases ownership; Postgres SQL contract.
- [x] Run focused red tests before production changes.
- [x] Implement interfaces above, preserving local in-memory queue backpressure.
- [x] Run focused queue and existing backend tests, record exact results and adapter limitations.
- [x] Independent spec/quality review; no commit.

## Task 2: Durable terminal results (R04)

Files: new workflow_result_store.py, workflow_harness_application.py, workflow_production_application.py, backend/harness_service.py, focused tests. Container wiring is controller-owned to avoid overlap.

Interfaces: host-owned result store stage/load/mark_committed bound by canonical request/result digests, usage, checkpoint metadata and immutable prompt release digest. SQLite development and shared PostgreSQL production adapters use their own table/connection factory, not JobRepository modifications.
Persist before success; resume prepared result without model reexecution; restore accepted result on terminal redelivery, bind proof to existing terminal receipt. Commit errors must be retryable and not rendered as successful unknown. Stable accepted timestamp/proof and idempotent commit required. Production commit may restore a binding only from validated durable host record, never from arbitrary model fields. Same ID/different request or digest fails closed.

- [x] Write failing crash-window, terminal redelivery, commit retry, binding mismatch and duplicate commit tests with real host boundaries where feasible.
- [x] Run focused red tests.
- [x] Implement durable stage before advance, terminal restore, idempotent commit and validation.
- [x] Run Harness/production focused regression tests and report.
- [x] Independent review; no commit.

## Task 3: Cost admission (R05)

Files: workflow_openai_client.py, new focused model budget module if needed, openai_usage.py/langchain_runtime.py only if required, corresponding tests. Do not edit AgentDriver behavior outside spend admission.
Use existing versioned pricing with Decimal. Conservatively account for serialized input/schema framing; bound max_output_tokens to remaining reservation; reject unknown model mappings, nonfinite/insufficient budget before provider call. Disable hidden provider retries for bounded calls. No cache discount assumed before actual usage. Preserve real usage settlement.

- [x] Write failing insufficient-budget/no-provider-call, output-cap, unknown-model, long-context pricing and no-hidden-retry tests.
- [x] Run focused red tests.
- [x] Implement admission and wire explicit cap into provider call.
- [x] Run model client, usage and driver regressions.
- [x] Independent review; no commit.

## Task 4: Shared Prompt activation (R06)

Files: workflow_prompt_config.py, new backend/prompt_activation_store.py, focused tests. Do not edit redis_adapters.py; container wiring is controller-owned.
Shared authoritative store holds active/previous immutable release ID plus manifest/release digests and revision. Bootstrap only if absent, never overwrite existing activation. Atomically compare-and-swap activation/rollback; production current reads shared authority and validates local manifests. Shared store outage or unavailable digest fails closed. Preserve local SQLite mode, release gate, immutable in-flight pins and audit recovery.

- [x] Write failing two-manager active/rollback visibility, concurrent change, unavailable manifest/digest, store outage and unchanged in-flight pin tests.
- [x] Run focused red tests.
- [x] Implement shared adapter and manager integration.
- [x] Run prompt/release focused tests and report production wiring required.
- [x] Independent review; no commit.

Review follow-up: shared outbox scheduler and revision-bearing Redis notification passed independent review and the 74-test incremental regression suite.

## Task 5: Integration and verification

Controller connects container dependencies after Task 2/4 interfaces settle; preserve four existing dirty files. Review each task plus integrated result. Run focused combined suite; inspect diffs and synchronize changed source files only after checking target matches original source baseline. Keep remote state untouched.

- [x] Wire durable result store and shared Prompt store with readiness checks.
- [x] Validate combined regression suite and inspect no conflict/secret regressions.
- [x] Synchronize scoped files to second repository and verify text identity.
- [x] Record any remaining unverified infrastructure conditions.
