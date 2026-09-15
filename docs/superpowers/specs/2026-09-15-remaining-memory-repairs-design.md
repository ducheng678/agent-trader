# Remaining memory repairs design (B06–B11)

The six findings were reconfirmed against the existing dirty worktree in
`scratch/remaining-memory-assessment.md`. Retain three layers: factual events,
retrievable knowledge and governed decisions. An accepted result creates a proposed
candidate only; promotion remains a separately checked stage.

## Candidate and promotion path

The host completion pipeline first durably records accepted event/decision/outcome,
then deterministically proposes evidence-linked KnowledgeRevision candidates with
separate idempotency and retry state. Do not rerun Harness acceptance after candidate
generation failure. A candidate remains PROPOSED until normal verified promotion.
Promotion scans a fair persisted cursor. Rejections and successful evaluations advance
the cursor; outages and cancellation do not acknowledge unevaluated work. Wrap at end
and resume after restart. An eligible candidate behind permanently rejected candidates
must eventually be reached.

## PostgreSQL memory parity

PostgresMemoryRepository must enforce all SQLite append/activate evidence, ancestry,
tenant, lineage, decision/outcome and provenance gates inside the same write transaction.
Version 2 head update is conditional on expected old revision and record ID, and prior
active revision is archived in the activation transaction. Domain conflicts remain
MemoryConflictError, not transport errors. Implement LifecycleRepository contracts in
PostgreSQL using the same pure lifecycle policy and a durable cleanup outbox. Container
uses exactly one active repository for retrieval, writer, candidate, promotion and
forgetting in production. No shadow SQLite maintenance as a substitute.

## Retrieval

PostgreSQL vector candidate retrieval pages by stable `(distance,record_id)` cursor.
The retrieval engine validates each page before selecting Top K, including eventual
contradiction checks. The page protocol signals exhaustion. A finite scan cap is
explicit; if not exhausted at the cap, return a safe failure/omission and do not
inject supposedly clear advice. Oversampling alone is insufficient. SQLite full
snapshot behavior stays valid.

Offline validation includes SQLite real repository behavior and PostgreSQL SQL
boundary contracts. Production multi-process PostgreSQL behavior requires later live
infrastructure validation. Preserve user changes, do not commit/push or call live models.
