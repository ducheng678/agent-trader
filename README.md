# Auto Trade Runtime

This repository runs the market-news watcher stack, the Mihomo proxy layer, and the event-driven trading agent.

## Main Entry Points

Proxy:

```bash
scripts/proxy/start_mihomo.sh
```

Watcher:

```bash
nohup .venv/bin/python -u watch_free_sources_modular.py >> logs/watch_free_sources_console.log 2>&1 &
```

Trading agent:

```bash
nohup .venv/bin/python -u unified_market_agent.py >> logs/unified_market_agent_console.log 2>&1 &
```

## LLM Architecture

The trading agent remains OpenAI-only. Internally, LangChain `ChatOpenAI`
uses the OpenAI Responses API for structured output, Web Search, reasoning,
and chart-image input. LangGraph orchestrates individual model calls and the
passive event-judgment -> conditional technical-pricing flow.

Existing `OPENAI_*` environment variables, startup commands, JSON Schemas,
and `DiscretionaryLLMEngine` callers remain compatible.

## Install

Use Python 3.11+.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

System tools expected by the proxy scripts:

```text
bash
curl
tar
cron / crontab
flock
```

`scripts/proxy/start_mihomo.sh` downloads Mihomo automatically if `runtime/proxy/mihomo/mihomo` is missing.

## Required Configuration

Runtime configuration is read from `.env`.

Prompt caching is enabled by default. The stable system prompt remains the first input item, dynamic request context stays after it, and every Responses request receives a deterministic cache key derived from the model and request phase.

```env
OPENAI_PROMPT_CACHE_ENABLED=true
OPENAI_PROMPT_CACHE_KEY_PREFIX=market-agent
```

Important proxy keys:

```env
TROJANFLARE_SUBSCRIPTION_URL_1=...
TROJANFLARE_SUBSCRIPTION_URL_2=...
MIHOMO_MIXED_PORT=7897
MIHOMO_EXTERNAL_CONTROLLER=127.0.0.1:9097
MIHOMO_HEALTHCHECK_INTERVAL_SECONDS=3600
MIHOMO_PROVIDER_INTERVAL_SECONDS=43200
```

Truth Social watcher keys:

```env
TRUTHSOCIAL_TOKEN=...
TRUTHSOCIAL_HANDLES=realDonaldTrump,rapidresponse47
TRUTHSOCIAL_HTTP_PROXY=http://127.0.0.1:7897
TRUTHSOCIAL_HTTPS_PROXY=http://127.0.0.1:7897
PROXY_WATCHER_SOURCES=aaa,conference_board
```

The TruthSocial proxy group, healthcheck URL, and domain rules are auto-derived from watcher source URLs. You do not need to set `TRUTHSOCIAL_PROXY_SELECTOR_GROUP`, `TRUTHSOCIAL_PROXY_HEALTHCHECK_URL`, or `TRUTHSOCIAL_PROXY_RULE_DOMAINS` unless you intentionally want to override defaults.

`PROXY_WATCHER_SOURCES` enables the local Mihomo proxy for regular watcher sources that do not have their own source-specific proxy env. The source list is comma-separated and uses watcher source bases such as `aaa` and `conference_board`.
For these sources, repeated HTTP/proxy/TLS failures trigger the matching Mihomo selector to rotate nodes before retrying.

## Startup Order

Start Mihomo first:

```bash
scripts/proxy/start_mihomo.sh
```

Install the provider refresh cron:

```bash
scripts/proxy/install_mihomo_refresh_cron.sh
```

Start watchers:

```bash
nohup .venv/bin/python -u watch_free_sources_modular.py >> logs/watch_free_sources_console.log 2>&1 &
```

Start the trading agent in a separate process:

```bash
nohup .venv/bin/python -u unified_market_agent.py >> logs/unified_market_agent_console.log 2>&1 &
```

Do not leave the watcher or trading agent stdout/stderr attached to an interactive PTY. If terminal output is not consumed and the PTY buffer fills, writes to stdout/stderr can block the process main loop. For the watcher this can stop source polling and `events.jsonl` writes; for the agent this can stop reading new `events.jsonl` entries.

## Proxy Refresh Behavior

`scripts/proxy/render_mihomo_config.py` builds watcher instances without polling them, extracts source URLs such as `url`, `feed_url`, `target_url`, `base_url`, and `feed_urls`, then generates source-specific Mihomo groups and domain rules.

For TruthSocial this generates:

```text
TruthSocial Auto  -> url-test against https://truthsocial.com/
TruthSocial Proxy -> select group used by TruthSocial watchers
DOMAIN-SUFFIX,truthsocial.com,TruthSocial Proxy
```

`MIHOMO_HEALTHCHECK_INTERVAL_SECONDS` controls Mihomo url-test interval.

`MIHOMO_PROVIDER_INTERVAL_SECONDS` controls subscription-node refresh. The cron calls:

```bash
scripts/proxy/refresh_mihomo_if_due.sh
```

The cron runs hourly, but the script exits unless the provider interval has elapsed. When due, it:

```text
1. Re-renders Mihomo config from subscription URLs.
2. Tries to reload Mihomo through /configs?force=true.
3. Falls back to restarting Mihomo only if reload fails.
```

To remove the cron:

```bash
scripts/proxy/uninstall_mihomo_refresh_cron.sh
```

## Migration To A New Machine

1. Clone the repository.
2. Install Python dependencies with `pip install -r requirements.txt`.
3. Copy or recreate `.env`.
4. Start Mihomo with `scripts/proxy/start_mihomo.sh`.
5. Install refresh cron with `scripts/proxy/install_mihomo_refresh_cron.sh`.
6. Start `watch_free_sources_modular.py`.
7. Start `unified_market_agent.py`.

To preserve watcher state, also migrate:

```text
data/free_sources_watch/state.json
data/free_sources_watch/events.jsonl
```

If you do not migrate those files, the watcher stack can still run, but warmup/seen-state behavior will be reset.

## Quick Checks

Check Mihomo is running:

```bash
cat runtime/proxy/mihomo/mihomo.pid
tail -50 runtime/proxy/mihomo/mihomo.log
```

Check the refresh cron:

```bash
crontab -l | grep AUTO_TROJANFLARE_MIHOMO
```

Run the focused proxy/watcher tests:

```bash
pytest -q market_agent_test_bundle/tests/test_truth_social_watcher_failover.py
```

Run the focused market-agent regression suite:

```bash
pytest -q \
  market_agent_test_bundle/tests/test_unified_market_agent_unit.py \
  market_agent_test_bundle/tests/test_unified_market_agent_adapters.py \
  market_agent_test_bundle/tests/test_unified_market_agent_state_machine.py \
  market_agent_test_bundle/tests/test_unified_market_agent_replay.py \
  market_agent_test_bundle/tests/test_unified_market_agent_passive_irrelevant_openai.py
```

## Production Backend

The agent runtime is decomposed into named modules for runtime, model routing, tool calling, prompt/context engineering, memory/state, RAG, and structured output. A FastAPI production backend adds authenticated asynchronous task submission, idempotency, SQLite WAL task/event persistence, a bounded TTL cache, bounded task admission, classified retries, a message-bus boundary, structured logs, health checks, and Prometheus metrics. The built-in executor is a single-process baseline; horizontal deployment requires the documented external database, cache, and broker adapters.

See [ARCHITECTURE.md](ARCHITECTURE.md) for module ownership, API contracts, configuration, and deployment replacement points.

Run locally after installing the root requirements:

```bash
MARKET_AGENT_API_TOKEN=local-token python -m market_agent.backend
```

### Governed Workflow API

`POST /v1/workflows` creates an asynchronous, Harness-governed workflow; the
status, cancellation, and event endpoints are `/v1/workflows/{run_id}`,
`/v1/workflows/{run_id}:cancel`, and `/v1/workflows/{run_id}/events`.
Supply the same `Idempotency-Key` to safely replay a submission. The response
always returns the original run trace on a replay; requests without that key
are intentionally independent runs.
It also returns `job_id` and `job_status_url`, which expose the asynchronous
result payload through the existing durable task API.

The API deliberately requires a deployment-owned `HarnessKernel`. Construct
the container with `BackendContainer.create(harness_kernel=kernel)`; it builds
the paired `HarnessWorkflowApplication` around the production runner. The
kernel's receipt issuer and signing capability must be provisioned by the
trusted execution host; the HTTP service returns `503` rather than create an
untrusted local replacement when that authority is absent.

For a workflow to reach a successful terminal state, the same trusted host
also supplies `harness_completion_candidate_factory`. It receives the immutable
request, validated workflow result, and current Harness view, and must return
the independently signed confidence/evidence candidate expected by the kernel.
Without it, the default is intentional fail-closed degradation to no-trade.

In staging and production, the legacy `POST /v1/tasks/generate_playbook`
compatibility route is disabled by default so it cannot bypass Harness control.
Set `MARKET_AGENT_LEGACY_PLAYBOOK_API_ENABLED=true` only for a deliberately
isolated compatibility deployment.

Deployment code should construct the trusted `HarnessKernel` and then call
`market_agent.backend.governed_app.create_governed_app(...)`. The normal
`python -m market_agent.backend` launcher is a development/legacy launcher and
does not fabricate the production signing authority.
