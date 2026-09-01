# Production Backend Architecture

## Module Boundaries

| Concern | Module | Responsibility |
| --- | --- | --- |
| Agent runtime | `market_agent/agent_runtime.py` | Orchestrates the discretionary LLM workflow. |
| Model routing | `market_agent/model_routing.py` | Selects model and reasoning effort without changing the model contract. |
| Prompt and context | `market_agent/prompt_context.py` | Builds stable prompt prefixes and dynamic market context. |
| Tool calling | `market_agent/tool_calling.py` | Defines tool schemas and handles model tool-call execution. |
| Memory and state | `market_agent/memory_state.py` | Owns file-backed helper context and state hydration. |
| Retrieval and RAG | `market_agent/retrieval_rag.py` | Produces local/news retrieval context for the model. |
| Structured outputs | `market_agent/structured_outputs.py` | Validates playbooks, decisions, and pricing-shaped model output. |
| Agent context | `market_agent/agent_context.py` | Normalizes market, position, and trigger context. |
| Passive workflow | `market_agent/passive_workflow.py` | Separates passive-event scoring and follow-up logic. |
| Compatibility facade | `market_agent/llm_engine.py` | Preserves existing imports while re-exporting the decomposed runtime. |
| HTTP API | `market_agent/backend/api.py` | FastAPI routes, authentication, request IDs, error mapping, health, and metrics. |
| API contracts | `market_agent/backend/api_contracts.py` | Pydantic request and response schemas. |
| Dependency wiring | `market_agent/backend/container.py` | Composes concrete runtime dependencies at application startup. |
| Configuration | `market_agent/backend/settings.py` | Validates environment-driven runtime settings. |
| Durable state | `market_agent/backend/database.py` | SQLite WAL job/event repository with idempotency protection. |
| Cache | `market_agent/backend/cache.py` | Thread-safe bounded TTL/LRU cache behind a cache interface. |
| Async work | `market_agent/backend/task_queue.py` | Bounded admission queue and worker pool, classified retries, task state transitions, and event publication. |
| Message queue boundary | `market_agent/backend/message_bus.py` | Typed event envelope and swappable message-bus interface. |
| Logging and monitoring | `market_agent/backend/observability.py` | JSON logs, request correlation, Prometheus text metrics, and timings. |
| Error boundary | `market_agent/backend/errors.py` | Typed domain errors mapped to consistent API responses. |
| LLM task adapter | `market_agent/backend/agent_service.py` | Lazily adapts `DiscretionaryLLMEngine.get_playbook` to an async task. |
| Harness lifecycle | `market_agent/workflow_harness.py` | Owns the append-only run state machine, budgets, loop protection, trusted receipts, and fail-closed terminal transitions. |
| Harness application | `market_agent/workflow_harness_application.py` | Bridges a committed run to the coordinated workflow without allowing model output to choose control flow. |
| Coordinated workflow | `market_agent/workflow_production_application.py` | Composes model routing, stable prompt releases, bounded context, capability grants, reflection, memory retrieval, cache policy, and final host writes. |
| Agent driver | `market_agent/workflow_agent_driver.py` | Enforces structured outputs, retry/backoff, circuit breaking, model fallback, and per-agent budgets. |
| Memory and RAG | `market_agent/workflow_long_term_memory.py`, `market_agent/workflow_memory_*` | Separates event, knowledge, and decision records; host-only writes; lifecycle forgetting; bounded retrieval summaries. |
| Safe answer cache | `market_agent/workflow_historical_answer_cache.py`, `market_agent/workflow_semantic_*` | Gates fixed and semantic informational reuse by tenant, versions, metadata, expiry, and strict similarity. |
| Workflow queue adapter | `market_agent/backend/harness_service.py` | Validates queued `WorkflowRequest` values and resumes the same Harness run after worker recovery. |

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health/live` | Process liveness probe. |
| `GET` | `/health/ready` | Dependency/readiness probe. |
| `POST` | `/v1/tasks/{task_name}` | Submit an idempotent asynchronous task. |
| `GET` | `/v1/tasks/{job_id}` | Fetch durable task state and result. |
| `GET` | `/v1/tasks/{job_id}/events` | Fetch the task audit/event stream. |
| `POST` | `/v1/workflows` | Admit and asynchronously dispatch a Harness-governed workflow. |
| `GET` | `/v1/workflows/{run_id}` | Fetch the authoritative Harness projection. |
| `POST` | `/v1/workflows/{run_id}:cancel` | Record a controlled cancellation intent. |
| `GET` | `/v1/workflows/{run_id}/events` | Read the append-only Harness event stream. |
| `GET` | `/metrics` | Prometheus-compatible operational metrics. |

Authenticated endpoints use `Authorization: Bearer <MARKET_AGENT_API_TOKEN>`. The token is required in production, staging, and whenever the API binds to a non-loopback host. Send `Idempotency-Key` or `idempotency_key` in the JSON request body when a caller may retry a submission. If both locations are present, their normalized values must match. A matching replay returns the original job even while new submissions are being rejected by backpressure.

The initial registered task is `generate_playbook`. It accepts the existing LLM runtime inputs (`user_query`, `event_tape`, `trigger_reason`, and the optional context fields), then returns the existing validated playbook and report. It does not place an order or enable live trading.

## Governed Workflow Control Plane

`/v1/workflows` is separate from the legacy `generate_playbook` compatibility
task. It creates a deterministic Harness run, then submits `execute_harness_workflow`
to the durable queue. The worker resumes the same `workflow_id` on retries or
process recovery, so no duplicate model run is created by queue redelivery.

The HTTP application never manufactures a receipt signer, a confidence policy,
or a completion candidate. A trusted execution host supplies a `HarnessKernel`
when building `BackendContainer`; the container then pairs it with the production
workflow runner. To permit a successful terminal transition, that host also
supplies `harness_completion_candidate_factory`, which derives the signed,
independently verified confidence/evidence candidate from the validated workflow
result. In its absence the state machine intentionally degrades to no-trade.

The legacy `generate_playbook` queue route is development-compatible only. It
is disabled by default in staging and production through
`MARKET_AGENT_LEGACY_PLAYBOOK_API_ENABLED=false`, preventing an unaudited
compatibility request from bypassing Harness control.

Deployment imports `market_agent.backend.governed_app.create_governed_app` and
passes the trusted kernel and optional host completion-evidence factory. The
default command-line backend remains a development compatibility launcher; it
never manufactures a signing authority from environment text.

Every workflow request carries the middleware-generated W3C trace through the
queue payload, Harness stream, coordinated agents, audit events, cache and memory
operations. Idempotent API replays return the original run trace; requests without
an idempotency key are independent runs. Tenant scope is checked at the workflow
API boundary before a run is admitted.

## Running Locally

```bash
pip install -r requirements.txt
MARKET_AGENT_API_TOKEN=local-token python -m market_agent.backend
```

Use `MARKET_AGENT_ENVIRONMENT=production` together with a non-empty `MARKET_AGENT_API_TOKEN` for production settings validation. `MARKET_AGENT_API_HOST` and `MARKET_AGENT_API_PORT` control the listener. `MARKET_AGENT_TASK_WORKERS` controls worker concurrency and `MARKET_AGENT_TASK_QUEUE_CAPACITY` bounds waiting work; their combined in-memory capacity cannot exceed 9999. The default durable store is `runtime/market_agent_backend.sqlite3`; SQLite WAL, a busy timeout, and short write transactions make it a sensible single-process, single-node baseline.

## Scaling Boundaries

The default implementation intentionally has no mandatory external service and must run with one application process. Do not start multiple Uvicorn workers against the built-in in-memory executor: recovery ownership is process-local. `JobRepository`, `CacheBackend`, and `MessageBus` form replacement seams for PostgreSQL, Redis, and a durable external broker/worker system when horizontal scale or cross-process delivery is required. Keep the HTTP/API contracts and task names stable while replacing those adapters.

Operational signals include structured request/job-correlated logs, task and HTTP counters, duration summaries, cache gauges, durable task events, health probes, and `/metrics`. Task handlers are admitted through bounded backpressure. Only errors explicitly marked retryable, such as `RetryableTaskError` and dependency-unavailable errors, receive exponential retry backoff; validation and unclassified failures terminate after one attempt. Errors are persisted and exposed as task state rather than being silently discarded. On startup/handler registration, unfinished durable tasks are recovered with at-least-once delivery semantics, so task handlers should remain idempotent. Terminal task retention is operator-managed; the runtime does not delete task history automatically.
