# Governed Memory, Retrieval, and Forgetting Design

## Scope

This phase adds durable three-layer memory to the Harness and AgentDriver:
immutable event material, evidence-backed knowledge revisions, and
outcome-linked decision lessons. SQLite is the local implementation; PostgreSQL,
pgvector, and object storage are explicit protocol adapters rather than local
runtime dependencies.

## Authority and Contracts

Agents receive only bounded read-only summaries, never repository handles. A
deterministic service is the only writer and every mutation requires a trace ID,
tenant scope, idempotency key, and redacted audit event. Memory is untrusted
data and cannot override system, risk, capability, schema, or user constraints.

`EventRecord` is append-only raw material with a canonical payload hash, source,
tenant, observed time, immutable provenance, and optional checksum-addressed
artifact. `KnowledgeRevision` holds one normalized rule, applicability,
confidence, effective/expiry time, lifecycle, evidence IDs, and lineage.
`DecisionLesson` links a provisional/final decision, evidence, and verified
outcome. Cross-tenant links, mutable fields, extra fields, non-finite values,
and circular evidence are rejected on construction and rehydration.

## Storage and Retrieval

`MemoryRepository` exposes transaction-scoped append, promote, finalize, and
read operations. `SQLiteMemoryRepository` uses WAL, foreign keys, tenant-scoped
indexes, canonical JSON, event hash uniqueness, revision compare-and-set, and
tenant/idempotency-key deduplication. `ArtifactStore` stores immutable bytes by
SHA-256 and verifies checksums on read.

`MemoryQuery` includes tenant/scope, normalized task, vector/model version,
freshness cap, and Top-K limits. Retrieval filters tenant, visibility, lifecycle,
effective/expiry time, applicability, and schema/vector/model compatibility
before deterministic vector or exact/text ranking. Ranking combines similarity,
exact match, authority, confidence, freshness, outcome verification,
applicability, contradiction penalty, and diversity. Conflicting strong records
produce conflict, not averaged advice.

`CoreExperienceSummary` is the sole agent-facing form: selected IDs, supporting
and contradicting evidence IDs, applicable rules/lessons, confidence, freshness,
conflict state, omissions, and deterministic hash. It is dynamic user content
after a stable system prefix. Weak, stale, contradictory, or failed retrieval
injects no memory.

Summaries include hash-bound aware `issued_at` and `expires_at` timestamps from
trusted retrieval. Expiry is the earliest freshness or explicit expiry deadline
of the selected records and their supporting ancestry, outcomes, and final
decisions. The driver compares them with its injected clock before every
provider dispatch, including retries; a cached reported age cannot extend reuse.

## Promotion and Forgetting

Promotion requires valid same-tenant evidence and independent corroboration or a
verified outcome. It rejects self-descendant/circular provenance, expired
evidence, conflicts, and unverified model-only claims. Proposal and activation
are separately audited.

`LifecycleWorker` first returns a deterministic dry-run plan. Retention class,
half-life confidence decay, capacity score, references, and legal holds select
records. Referenced, active, or legally held records cannot purge. Actions are
ordered archive, tombstone, then purge; idempotent outbox cleanup removes vector,
artifact, and cache derivatives. Missing evidence yields a retrieval gap and
safe `不知道`/`no_trade`, never invention.

Application rechecks plan eligibility and records transition time using a trusted
repository clock inside the write transaction. Future plans are rejected;
delayed plans never backdate archive or tombstone quarantine. Cleanup records an
audited durable attempt position before calling each adapter. Pending tasks are
ordered by their last attempt, so failed tasks cannot monopolize a bounded run,
including after restart. Adapter idempotency keys remain stable across retries.

## Verification

Tests cover append-only hashes, tenant isolation, idempotency, rollback,
evidence/decision links, agent-write denial, versioned retrieval filtering,
conflicts, cited bounded summaries, circular-promotion denial, verified outcomes,
half-life decay, reference/legal-hold protection, archive/tombstone/purge order,
idempotent cleanup, and trace-bound redacted audits. Harness and AgentDriver
regressions remain green.
