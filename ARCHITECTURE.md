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
| Async work | `market_agent/backend/task_queue.py` | Bounded worker pool, retries, task state transitions, and event publication. |
| Message queue boundary | `market_agent/backend/message_bus.py` | Typed event envelope and swappable message-bus interface. |
| Logging and monitoring | `market_agent/backend/observability.py` | JSON logs, request correlation, Prometheus text metrics, and timings. |
| Error boundary | `market_agent/backend/errors.py` | Typed domain errors mapped to consistent API responses. |
| LLM task adapter | `market_agent/backend/agent_service.py` | Lazily adapts `DiscretionaryLLMEngine.get_playbook` to an async task. |

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/health/live` | Process liveness probe. |
| `GET` | `/health/ready` | Dependency/readiness probe. |
| `POST` | `/v1/tasks/{task_name}` | Submit an idempotent asynchronous task. |
| `GET` | `/v1/tasks/{job_id}` | Fetch durable task state and result. |
| `GET` | `/v1/tasks/{job_id}/events` | Fetch the task audit/event stream. |
| `GET` | `/metrics` | Prometheus-compatible operational metrics. |

Authenticated endpoints use `Authorization: Bearer <MARKET_AGENT_API_TOKEN>`. The token is required in production and staging. Send `Idempotency-Key` or `idempotency_key` in the JSON request body when a caller may retry a submission.

The initial registered task is `generate_playbook`. It accepts the existing LLM runtime inputs (`user_query`, `event_tape`, `trigger_reason`, and the optional context fields), then returns the existing validated playbook and report. It does not place an order or enable live trading.

## Running Locally

```bash
pip install -r requirements.txt
MARKET_AGENT_API_TOKEN=local-token python -m market_agent.backend
```

Use `MARKET_AGENT_ENVIRONMENT=production` together with a non-empty `MARKET_AGENT_API_TOKEN` for production settings validation. The default durable store is `runtime/market_agent_backend.sqlite3`; SQLite WAL, a busy timeout, and short write transactions make it a sensible single-node baseline.

## Scaling Boundaries

The default implementation intentionally has no mandatory external service. `JobRepository`, `CacheBackend`, and `MessageBus` form replacement seams for PostgreSQL, Redis, and a durable external broker/worker system when horizontal scale or cross-process delivery is required. Keep the HTTP/API contracts and task names stable while replacing those adapters.

Operational signals include structured request/job-correlated logs, task and HTTP counters, duration summaries, cache gauges, durable task events, health probes, and `/metrics`. Task handlers are executed by a bounded worker pool with exponential retry backoff; errors are persisted and exposed as task state rather than being silently discarded. On startup/handler registration, unfinished durable tasks are recovered with at-least-once delivery semantics, so task handlers should remain idempotent.
