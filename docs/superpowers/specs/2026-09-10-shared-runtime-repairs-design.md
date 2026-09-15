# Shared runtime repair design (B12, B13, F02)

Repair existing cross-process boundaries without replacing Harness or LangGraph.
Continue using PostgreSQL as production authority; development retains local mode.

## Cancellation

Introduce a namespaced cancellation store with cancel(run_id) and is_cancelled(run_id).
Production uses PostgreSQL; real two-instance SQLite adapter tests establish the
shared contract offline. Cancellation is monotonic and must survive process restart.
Do not expire a cancellation while a delayed job may still resume. Store run IDs as
opaque bounded strings; values never become SQL identifiers. Registry signals query
the store on each cooperative check and fail closed on unavailable shared storage.
API cancellation acknowledges success only after the shared write succeeds; existing
local events remain fast latches. Container must inject shared authority and expose
readiness failure when production falls back to a local-only registry.

## Execution ownership loss

A lease-loss event is scoped to one owner, not the workflow cancellation record.
An uncertain renewal immediately latches that owner as lost. Task execution carries
a cooperative checker through a ContextVar execution context; the worker establishes
and resets it around handler invocation. Production/Harness checks combine that
checker with API cancellation, without persisting lease loss as user cancellation.
Check before new model/tool calls and before post-handler durable transitions.
Database token/expiry fencing remains authoritative. Already-running synchronous
external work cannot be killed; idempotency/fencing remains required downstream.

## Trace query

Introduce a TraceSink protocol matching record and query. PostgreSQL production sink
stores the same redacted StructuredEvent, tenant namespace and shared sequence.
Allocate sequence and append in one serialized namespace transaction, so a reader
cannot advance past an earlier uncommitted record. Retain a bounded count using the
existing trace_event_capacity and maximum_query settings; return the same TracePage
cursor/truncation contract. Multiple backend replicas use the same namespace store.
Never silently substitute process-local history when shared storage is unavailable.
Local BoundedTraceSink remains the development default. Preserve current event privacy
constraints and API authentication.

## Validation and boundaries

Real SQLite multi-instance tests plus offline PostgreSQL parameter/transaction
contracts; no claim of live PostgreSQL concurrency verification. Test cancellation
written on A observed on already-created signal B; namespace isolation, restart,
storage failure, and owner-loss distinct from global cancellation. Test cross-instance
trace visibility, pagination, bounded eviction and malformed/cross-namespace records.
No live services, commits or pushes. Preserve the six-P1 dirty changes and sync only
verified patches after independent review.
