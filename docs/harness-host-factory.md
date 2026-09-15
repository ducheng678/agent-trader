# Harness Host Factory Contract

Production sets `MARKET_AGENT_HARNESS_HOST_FACTORY=package.module:create_bindings`.
The callable returns `HarnessHostBindings` from
`market_agent.backend.governed_bootstrap`.

The host constructs the exact `HarnessKernel`. Its receipt issuer must use the
private key corresponding to the execution backend's pinned public key; private
key bytes, KMS credentials, and signing RPC credentials must never be supplied
to the HTTP application or environment-based application configuration.

The optional completion candidate factory receives an immutable
`WorkflowRequest`, validated `WorkflowResult`, and folded `HarnessSessionView`.
It may return a candidate only after independent evidence collection,
deterministic field validation, conflict checks, and signature verification. The
candidate is passed to the Harness confidence gate; it cannot choose graph
edges, permissions, cache writes, or durable state directly.

If the factory is unavailable or cannot issue valid evidence, omit the
completion candidate factory. The workflow then terminates through the existing
fail-closed no-trade degradation path.
