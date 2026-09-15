# Priority six production repairs

Repair P1 queue capacity, retry/dead-letter classification, shared execution leases, durable accepted-result recovery, pre-dispatch cost admission and shared Prompt rollback.

Preserve existing dirty changes. Use this existing isolated worktree, then synchronize scoped changes to multi-agent-trader. No commits or pushes this phase.

Interfaces: release publisher admission after durable publication; worker-owned capacity; expiring atomic claim/renew/fenced completion; retryable transient delivery errors; durable host-owned result plus immutable prompt binding before success; idempotent terminal commit recovery without model replay; conservative input/output reservation with explicit provider output cap and no hidden retries; atomic shared digest-bound active/previous Prompt pointer with gates and audit.

Fail closed on ownership loss, insufficient/unknown pricing and unavailable authoritative production Prompt state. Preserve in-flight Prompt pins and security validation.

Verification covers two-worker handoff, backpressure, healthy/expired leases and stale owners, crashes around terminal result commit, insufficient budget/output cap, and two-manager activation/rollback. No live model/trading/cloud operations; deployment validation remains separate.
