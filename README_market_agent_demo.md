# Event-driven OpenAI + Hyperliquid demo

This demo reads the `events.jsonl` file produced by your existing watcher stack, sends each new event to the OpenAI Responses API, gets back a structured trading decision, and optionally places an order on Hyperliquid.

## What it does

- Watches `data/free_sources_watch/events.jsonl` by default.
- Uses `gpt-5.4` by default.
- Supports three modes:
  - `off`: never uses web search.
  - `hybrid`: fast pass first, then uses web search only when needed.
  - `always`: every decision is verified with web search.
- Defaults to `ENABLE_LIVE_TRADING=false`, so it only prints decisions until you are comfortable.

## Install

```bash
pip install openai python-dotenv pydantic hyperliquid-python-sdk eth-account
```

## Run order

First terminal:

```bash
python watch_free_sources_modular_v2.py
```

Second terminal:

```bash
cp .env.market_agent.example .env
python event_driven_hyperliquid_demo.py
```

## Recommended production choice

For trading, use `OPENAI_SEARCH_MODE=hybrid`.

Why:
- `off` is fastest, but it cannot cross-check ambiguous or second-hand headlines.
- `always` is safer than `off`, but it adds latency to every event.
- `hybrid` gives you a fast first reaction and only spends extra time when the event is ambiguous, low-confidence, or from a less-trusted source.

## Important

- Keep `ENABLE_LIVE_TRADING=false` until you have verified the whole pipeline.
- Start with Hyperliquid `testnet` first.
- The model should decide direction and confidence; hard risk caps should still live in code.
- This demo uses market orders via the Hyperliquid SDK convenience method.
