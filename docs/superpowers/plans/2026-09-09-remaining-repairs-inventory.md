# Remaining repairs inventory

Scope recovered from the prior system review. Each finding must be checked against
the current dirty worktree before implementing. Previous six-P1 changes are preserved.

| ID | Remaining finding | Status |
|---|---|---|
| B01 | Per-attempt timeout does not bound model request | synced to both repositories; offline suites passed; live provider validation pending |
| B02 | Local knowledge fallback input/output incompatible with agent | synced to both repositories; offline suites passed; live provider validation pending |
| B03 | General question cache miss lacks generation/writeback | synced to both repositories; offline suites passed; live provider validation pending |
| B04 | Agent semantic cache lacks query vector | synced to both repositories; offline suites passed; live provider validation pending |
| B05 | Embedding cost and cancellation not integrated | synced to both repositories; offline suites passed; live provider validation pending |
| B06 | Accepted results do not create candidate knowledge | synced to both repositories; offline suites passed; live database validation pending |
| B07 | Production PostgreSQL memory bypassed by SQLite maintenance | synced to both repositories; offline suites passed; live PostgreSQL validation pending |
| B08 | PostgreSQL knowledge version two conflicts with primary key | synced to both repositories; offline suites passed; live PostgreSQL validation pending |
| B09 | PostgreSQL evidence integrity weaker than SQLite | synced to both repositories; offline suites passed; live PostgreSQL validation pending |
| B10 | Ineligible candidates starve later promotion candidates | synced to both repositories; offline suites passed; live concurrency validation pending |
| B11 | Top K truncates before qualification | synced to both repositories; offline suites passed; live PostgreSQL validation pending |
| B12 | Cancellation signals are process-local | synced to both repositories; offline suites passed; live shared-store validation pending |
| B13 | Trace query uses process-local memory | synced to both repositories; offline suites passed; live shared-store validation pending |
| B14 | Evaluation scores recordings without running current workflow | synced to both repositories; offline suites passed; trusted build metadata and live executed gate pending |
| F01 | Redis transient retry delayed by cross-consumer idle threshold | synced to both repositories; offline suites passed; live Redis validation pending |
| F02 | Lease renewal transport failure execution boundary | synced to both repositories; offline suites passed; live Redis/concurrency validation pending |
| F03 | Prompt audit acknowledgement can suppress a future revision | synced to both repositories; offline suites passed; live shared-store validation pending |

Execution constraints: no commits/push, no live model/trading/cloud/service mutations.
Use the existing isolated feature worktree. Scope-specific offline regression tests
and independent review are required. Synchronize only verified changes to the second
repository, preserving all preexisting uncommitted work. Use difficulty-matched models
and one implementation owner per file; parallel read-only assessment is allowed.
No claim of exactly-once external effects or real distributed validation from fakes.

Design approach: preserve existing contracts and repair production wiring, use shared
authoritative adapters for cross-process state, and distinguish recorded evaluations
from executed current-workflow evaluations. Do not replace the overall architecture.
Each subsystem received its own implementation brief. These statuses refer to
the dirty SOURCE and TARGET checkouts after both offline suites passed
(2613 passed, 20 skipped in each); they do not assert live distributed or provider
correctness. The independent review found four follow-up defects in usage,
cache accounting, executed-evaluation binding and rollback audit replay; their
repairs are synchronized and offline-validated. Follow-up review exposed a missing
production evaluation-binding interface; a host-revision input plus current
prompt/model-policy derivation was then added with an offline production-class test.
After this edit both full offline suites passed (2613 passed, 20 skipped in each).
This does not provide trusted deployment metadata or live executed-gate results.
