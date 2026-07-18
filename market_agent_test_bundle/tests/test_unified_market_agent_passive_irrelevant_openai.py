from __future__ import annotations

import os
from types import SimpleNamespace

import pytest


RUN_REAL_OPENAI_TESTS = os.environ.get("RUN_REAL_OPENAI_TESTS", "").strip().lower() in {"1", "true", "yes", "on"}

TRADE_CANDIDATES = [
    {
        "candidate_key": "BTC_USDC",
        "display_name": "BTC-USDC",
        "configured_execution_symbol": "BTC",
        "execution_symbol": "BTC",
        "tradable_on_hyperliquid": True,
    },
    {
        "candidate_key": "BRENTOIL_USDC",
        "display_name": "BRENTOIL-USDC",
        "configured_execution_symbol": "xyz:BRENTOIL",
        "execution_symbol": "xyz:BRENTOIL",
        "tradable_on_hyperliquid": True,
    },
    {
        "candidate_key": "SILVER_USDC",
        "display_name": "SILVER-USDC",
        "configured_execution_symbol": "xyz:SILVER",
        "execution_symbol": "xyz:SILVER",
        "tradable_on_hyperliquid": True,
    },
]

SYMBOL_META = {
    "BTC": {
        "display_name": "BTC-USDC",
        "price": 67200.0,
        "dex": "",
        "market_name": "BTC",
        "side": "long",
        "tech": {
            "current_price": 67200.0,
            "change_1h_pct": 0.3,
            "change_12h_pct": 1.1,
            "change_24h_pct": 1.6,
            "high_12h": 67500.0,
            "low_12h": 66800.0,
            "high_24h": 67650.0,
            "low_24h": 66250.0,
            "range_position_12h_pct": 57.0,
            "range_position_24h_pct": 67.0,
            "avg_5m_range_pct": 0.05,
            "avg_15m_range_pct": 0.12,
            "candles_5m_count": 145,
            "candles_15m_count": 97,
            "candles_1h_count": 73,
        },
    },
    "xyz:BRENTOIL": {
        "display_name": "BRENTOIL-USDC",
        "price": 109.8,
        "dex": "xyz",
        "market_name": "BRENTOIL",
        "side": "short",
        "tech": {
            "current_price": 109.8,
            "change_1h_pct": -0.1,
            "change_12h_pct": -0.2,
            "change_24h_pct": 0.1,
            "high_12h": 110.6,
            "low_12h": 109.3,
            "high_24h": 110.8,
            "low_24h": 109.1,
            "range_position_12h_pct": 38.0,
            "range_position_24h_pct": 41.0,
            "avg_5m_range_pct": 0.06,
            "avg_15m_range_pct": 0.19,
            "candles_5m_count": 145,
            "candles_15m_count": 97,
            "candles_1h_count": 73,
        },
    },
    "xyz:SILVER": {
        "display_name": "SILVER-USDC",
        "price": 72.9,
        "dex": "xyz",
        "market_name": "SILVER",
        "side": "long",
        "tech": {
            "current_price": 72.9,
            "change_1h_pct": 0.05,
            "change_12h_pct": 0.15,
            "change_24h_pct": 0.4,
            "high_12h": 73.2,
            "low_12h": 72.4,
            "high_24h": 73.4,
            "low_24h": 72.1,
            "range_position_12h_pct": 62.0,
            "range_position_24h_pct": 61.0,
            "avg_5m_range_pct": 0.02,
            "avg_15m_range_pct": 0.05,
            "candles_5m_count": 145,
            "candles_15m_count": 97,
            "candles_1h_count": 73,
        },
    },
}

BTC_EVENT = {
    "source": "coindesk",
    "title": "US spot Bitcoin ETFs post strong inflows as Bitcoin leads risk rally",
    "summary": "Bitcoin ETF inflows accelerated as BTC outperformed broader risk assets.",
    "published_at": "2026-04-04T18:58:52Z",
}
BRENTOIL_EVENT = {
    "source": "bloomberg",
    "title": "Iran Says Iraqi Ships Are Allowed to Use Strait of Hormuz",
    "summary": "Oil traders remain focused on Hormuz shipping risk and crude flow implications.",
    "published_at": "2026-04-04T17:48:02Z",
}
SILVER_EVENT = {
    "source": "reuters",
    "title": "Silver rallies as tariff fears boost precious and industrial metals",
    "summary": "Silver prices rose as tariff fears and industrial-demand expectations lifted metals.",
    "published_at": "2026-04-04T17:12:00Z",
}


def _build_reader(held_symbol: str):
    meta = SYMBOL_META[held_symbol]
    return SimpleNamespace(
        account_address="0xpassive",
        network="mainnet",
        get_all_positions=lambda: {
            "known": True,
            "account_address": "0xpassive",
            "network": "mainnet",
            "account_equity_usd": 500.0,
            "available_margin_usd": 120.0,
            "withdrawable_usd": 80.0,
            "remaining_capital_usd": 80.0,
            "positions": [
                {
                    "symbol": held_symbol,
                    "side": meta["side"],
                    "size": 1.0 if meta["side"] == "long" else -2.0,
                    "entry_price": meta["price"],
                    "notional_usd": abs(meta["price"] * (1.0 if meta["side"] == "long" else 2.0)),
                }
            ],
        },
        get_mid_price=lambda symbol: SYMBOL_META[symbol]["price"],
        get_position_snapshot=lambda symbol, **kwargs: {
            "known": True,
            "account_address": "0xpassive",
            "network": "mainnet",
            "symbol": symbol,
            "side": SYMBOL_META[symbol]["side"],
            "size": 1.0 if SYMBOL_META[symbol]["side"] == "long" else -2.0,
            "entry_price": SYMBOL_META[symbol]["price"],
            "mid_price": SYMBOL_META[symbol]["price"],
            "notional_usd": abs(SYMBOL_META[symbol]["price"] * (1.0 if SYMBOL_META[symbol]["side"] == "long" else 2.0)),
        },
        get_market_spec=lambda symbol: {
            "execution_symbol": symbol,
            "dex": SYMBOL_META[symbol]["dex"],
            "market_name": SYMBOL_META[symbol]["market_name"],
            "display_name": SYMBOL_META[symbol]["display_name"],
            "symbol": symbol,
            "sz_decimals": 2 if symbol.startswith("xyz:") else 5,
        },
        get_market_technical_context=lambda symbol: dict(SYMBOL_META[symbol]["tech"]),
        format_all_positions=lambda payload: "",
        format_symbol_position=lambda payload: "",
    )


def _build_agent(uma, engine, held_symbol: str):
    agent = object.__new__(uma.UnifiedMarketAgent)
    agent.symbol = held_symbol
    agent.reader = _build_reader(held_symbol)
    agent.trade_candidates = [dict(item) for item in TRADE_CANDIDATES]
    agent.max_planned_loss_usd = 100.0
    agent.local_risk_tolerance_usd = 1.0
    agent.local_size_from_stop = True
    agent.user_query_template = ""
    agent.current_playbook_reason = "active_periodic_refresh"
    existing_playbook = uma.GenericPlaybook(
        display_answer="existing",
        current_bias="neutral",
        selected_symbol=SYMBOL_META[held_symbol]["display_name"],
        selection_reason="existing selection",
        entry_plan=uma.EntryPlan(
            summary="existing-entry",
            execute_now=False,
            now_action=uma.build_empty_strategy_decision(),
            scenarios=[],
        ),
    )
    agent.current_mode = "raw_context_only"
    agent.current_playbook = existing_playbook
    agent.current_baseline_positions_signature = None
    agent.risk_session = None
    agent.last_playbook_query_at = None
    audited = []
    agent._audit_event = lambda name, payload=None: audited.append((name, payload))
    agent._augment_engine_debug_with_cost_metrics = lambda payload: payload
    agent._print_json_block = lambda *args, **kwargs: None
    agent.print_playbook = lambda *args, **kwargs: None
    agent._execute_immediate_playbook_action = lambda *args, **kwargs: None
    agent._arm_follow_up_plan_for_current_state = lambda *args, **kwargs: None
    agent._schedule_next_active_query = lambda *args, **kwargs: None
    agent.executor = SimpleNamespace(
        resolve_exit_levels=lambda decision, ref_price: {
            "reference_price": ref_price,
            "take_profit_price": decision.take_profit_price,
            "stop_loss_price": decision.stop_loss_price,
        }
    )
    agent.engine = engine
    return agent, existing_playbook, audited


@pytest.mark.skipif(not RUN_REAL_OPENAI_TESTS, reason="Set RUN_REAL_OPENAI_TESTS=true to run live OpenAI integration tests.")
@pytest.mark.parametrize(
    "held_symbol,trigger_event,expected_relevance",
    [
        pytest.param("BTC", BTC_EVENT, "relevant", id="btc-vs-btc"),
        pytest.param("BTC", BRENTOIL_EVENT, "unrelated", id="btc-vs-brent"),
        pytest.param("BTC", SILVER_EVENT, "unrelated", id="btc-vs-silver"),
        pytest.param("xyz:BRENTOIL", BTC_EVENT, "unrelated", id="brent-vs-btc"),
        pytest.param("xyz:BRENTOIL", BRENTOIL_EVENT, "relevant", id="brent-vs-brent"),
        pytest.param("xyz:BRENTOIL", SILVER_EVENT, "unrelated", id="brent-vs-silver"),
        pytest.param("xyz:SILVER", BTC_EVENT, "unrelated", id="silver-vs-btc"),
        pytest.param("xyz:SILVER", BRENTOIL_EVENT, "unrelated", id="silver-vs-brent"),
        pytest.param("xyz:SILVER", SILVER_EVENT, "relevant", id="silver-vs-silver"),
    ],
)
def test_real_openai_passive_relevance_matrix(uma, monkeypatch, held_symbol, trigger_event, expected_relevance):
    monkeypatch.setenv("OPENAI_ACTIVE_SEARCH_MODE", "off")
    monkeypatch.setenv("OPENAI_PASSIVE_SEARCH_MODE", "off")
    monkeypatch.setenv("OPENAI_SEARCH_MODE", "off")
    monkeypatch.setenv("OPENAI_ACTIVE_MODEL", os.environ.get("OPENAI_ACTIVE_MODEL", "gpt-5-mini"))
    monkeypatch.setenv("OPENAI_PASSIVE_MODEL", os.environ.get("OPENAI_PASSIVE_MODEL", "gpt-5-mini"))
    monkeypatch.setenv("OPENAI_ACTIVE_REASONING_EFFORT", os.environ.get("OPENAI_ACTIVE_REASONING_EFFORT", "high"))
    monkeypatch.setenv("OPENAI_PASSIVE_REASONING_EFFORT", os.environ.get("OPENAI_PASSIVE_REASONING_EFFORT", "medium"))

    engine = uma.DiscretionaryLLMEngine()
    agent, existing_playbook, audited = _build_agent(uma, engine, held_symbol)
    agent.events = SimpleNamespace(recent=lambda: [trigger_event])

    agent.query_new_playbook("passive_event_trigger", trigger_event)

    assert engine.last_call_debug["parsed_output"]["trigger_event_relevance"] == expected_relevance
    if expected_relevance == "unrelated":
        assert agent.current_playbook is existing_playbook
        assert any(name == "passive_query_irrelevant" for name, _payload in audited)
    else:
        assert agent.current_playbook is not existing_playbook
        assert not any(name == "passive_query_irrelevant" for name, _payload in audited)
