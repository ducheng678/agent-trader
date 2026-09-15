from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Iterable

import pytest


RUN_REAL_OPENAI_TESTS = os.environ.get("RUN_REAL_OPENAI_TESTS", "").strip().lower() in {"1", "true", "yes", "on"}


def _build_context():
    all_positions = {
        "known": True,
        "account_address": "0xintegration",
        "network": "mainnet",
        "margin_summary": {
            "account_value_usd": 500.0,
            "total_margin_used_usd": 0.0,
            "available_margin_usd": 300.0,
            "total_notional_usd": 0.0,
            "total_raw_usd": 0.0,
        },
        "cross_margin_summary": {
            "account_value_usd": 500.0,
            "total_margin_used_usd": 0.0,
            "available_margin_usd": 300.0,
            "total_notional_usd": 0.0,
            "total_raw_usd": 0.0,
        },
        "account_equity_usd": 500.0,
        "total_margin_used_usd": 0.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "remaining_capital_usd": 300.0,
        "remaining_capital_source": "withdrawable",
        "positions": [],
        "positions_count": 0,
        "total_notional_usd": 0.0,
    }
    symbol_position = {
        "known": True,
        "account_address": "0xintegration",
        "network": "mainnet",
        "symbol": "BTC",
        "side": "flat",
        "size": 0.0,
        "entry_price": 0.0,
        "mid_price": 100000.0,
        "notional_usd": 0.0,
        "leverage": 0.0,
        "max_leverage": 40,
        "only_isolated": False,
        "account_equity_usd": 500.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "remaining_capital_usd": 300.0,
        "margin_used": 0.0,
    }
    risk_constraints = {
        "max_planned_loss_usd": 100.0,
        "remaining_capital_usd": 300.0,
        "remaining_capital_source": "withdrawable",
        "account_equity_usd": 500.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "derive_notional_from_stop_distance": True,
    }
    return all_positions, symbol_position, risk_constraints


def _build_candidate_contexts():
    return [
        {
            "candidate_key": "BTC_USDC",
            "display_name": "BTC-USDC",
            "configured_execution_symbol": "BTC",
            "execution_symbol": "BTC",
            "tradable_on_hyperliquid": True,
            "current_price": 100000.0,
            "symbol_position": {
                "known": True,
                "account_address": "0xintegration",
                "network": "mainnet",
                "symbol": "BTC",
                "side": "flat",
                "size": 0.0,
                "entry_price": 0.0,
                "mid_price": 100000.0,
                "notional_usd": 0.0,
                "leverage": 0.0,
                "max_leverage": 40,
                "only_isolated": False,
                "account_equity_usd": 500.0,
                "available_margin_usd": 300.0,
                "withdrawable_usd": 300.0,
                "remaining_capital_usd": 300.0,
            },
            "market_spec": {
                "execution_symbol": "BTC",
                "dex": "",
                "market_name": "BTC",
                "display_name": "BTC-USDC",
                "symbol": "BTC",
                "sz_decimals": 5,
                "max_leverage": 40,
                "only_isolated": False,
            },
        },
        {
            "candidate_key": "BRENTOIL_USDC",
            "display_name": "BRENTOIL-USDC",
            "configured_execution_symbol": "xyz:BRENTOIL",
            "execution_symbol": "xyz:BRENTOIL",
            "tradable_on_hyperliquid": True,
            "current_price": 104.2,
            "symbol_position": {
                "known": True,
                "account_address": "0xintegration",
                "network": "mainnet",
                "symbol": "xyz:BRENTOIL",
                "side": "flat",
                "size": 0.0,
                "entry_price": 0.0,
                "mid_price": 104.2,
                "notional_usd": 0.0,
                "leverage": 0.0,
                "max_leverage": 20,
                "only_isolated": True,
                "account_equity_usd": 500.0,
                "available_margin_usd": 300.0,
                "withdrawable_usd": 300.0,
                "remaining_capital_usd": 300.0,
            },
            "market_spec": {
                "execution_symbol": "xyz:BRENTOIL",
                "dex": "xyz",
                "market_name": "BRENTOIL",
                "display_name": "BRENTOIL-USDC",
                "symbol": "xyz:BRENTOIL",
                "sz_decimals": 2,
                "max_leverage": 20,
                "only_isolated": True,
            },
        },
        {
            "candidate_key": "SILVER_USDC",
            "display_name": "SILVER-USDC",
            "configured_execution_symbol": "xyz:SILVER",
            "execution_symbol": "xyz:SILVER",
            "tradable_on_hyperliquid": True,
            "current_price": 75.0,
            "symbol_position": {
                "known": True,
                "account_address": "0xintegration",
                "network": "mainnet",
                "symbol": "xyz:SILVER",
                "side": "flat",
                "size": 0.0,
                "entry_price": 0.0,
                "mid_price": 75.0,
                "notional_usd": 0.0,
                "leverage": 0.0,
                "max_leverage": 25,
                "only_isolated": False,
                "account_equity_usd": 500.0,
                "available_margin_usd": 300.0,
                "withdrawable_usd": 300.0,
                "remaining_capital_usd": 300.0,
            },
            "market_spec": {
                "execution_symbol": "xyz:SILVER",
                "dex": "xyz",
                "market_name": "SILVER",
                "display_name": "SILVER-USDC",
                "symbol": "xyz:SILVER",
                "sz_decimals": 2,
                "max_leverage": 25,
                "only_isolated": False,
            },
        },
    ]


def _collect_parsed_coverage(serialized: dict) -> dict:
    actionable_candidates = []
    observation_texts = []
    confirmation_rules = 0
    tp_leg_count = len(serialized["position_management"]["take_profit_legs"])
    sl_leg_count = len(serialized["position_management"]["stop_loss_legs"])
    entry_tp_prices = []
    entry_sl_prices = []

    entry_now = serialized["entry_plan"]["now_action"]
    if entry_now["action"] in {"long", "short"}:
        actionable_candidates.append(entry_now)
        if float(entry_now.get("take_profit_price", 0.0) or 0.0) > 0:
            entry_tp_prices.append(float(entry_now["take_profit_price"]))
        if float(entry_now.get("stop_loss_price", 0.0) or 0.0) > 0:
            entry_sl_prices.append(float(entry_now["stop_loss_price"]))

    for scenario in serialized["entry_plan"]["scenarios"]:
        if scenario["observation_starts_when"].strip():
            observation_texts.append(scenario["observation_starts_when"])
        for rule in scenario["decision_rules"]:
            if rule["when_all"]:
                confirmation_rules += 1
            if rule["action_decision"]["action"] in {"long", "short"}:
                actionable_candidates.append(rule["action_decision"])
                if float(rule["action_decision"].get("take_profit_price", 0.0) or 0.0) > 0:
                    entry_tp_prices.append(float(rule["action_decision"]["take_profit_price"]))
                if float(rule["action_decision"].get("stop_loss_price", 0.0) or 0.0) > 0:
                    entry_sl_prices.append(float(rule["action_decision"]["stop_loss_price"]))

    management_now = serialized["position_management"]["now_action"]
    if management_now["action"] in {"reverse_to_long", "reverse_to_short"}:
        actionable_candidates.append(management_now)

    for scenario in serialized["position_management"]["scenarios"]:
        if scenario["observation_starts_when"].strip():
            observation_texts.append(scenario["observation_starts_when"])
        for rule in scenario["decision_rules"]:
            if rule["when_all"]:
                confirmation_rules += 1
            if rule["action_decision"]["action"] in {"reverse_to_long", "reverse_to_short"}:
                actionable_candidates.append(rule["action_decision"])

    return {
        "actionable_candidates": actionable_candidates,
        "observation_texts": observation_texts,
        "confirmation_rules": confirmation_rules,
        "tp_leg_count": tp_leg_count,
        "sl_leg_count": sl_leg_count,
        "entry_tp_prices": entry_tp_prices,
        "entry_sl_prices": entry_sl_prices,
    }


def _extract_request_user_payload(engine) -> dict:
    request_messages = list(engine.last_call_debug.get("request_messages") or [])
    assert len(request_messages) >= 2
    user_message = request_messages[1]
    payload_text = ""
    for item in user_message.get("content", []) or []:
        if item.get("type") == "input_text":
            payload_text = str(item.get("text", "") or "")
            break
    assert payload_text
    return __import__("json").loads(payload_text)


def _assert_no_pct_exit_keys(value):
    if isinstance(value, dict):
        assert "take_profit_pct" not in value
        assert "stop_loss_pct" not in value
        for nested in value.values():
            _assert_no_pct_exit_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            _assert_no_pct_exit_keys(nested)



def _build_real_brent_candidate():
    return {
        "candidate_key": "BRENTOIL_USDC",
        "display_name": "BRENTOIL-USDC",
        "configured_execution_symbol": "xyz:BRENTOIL",
        "execution_symbol": "xyz:BRENTOIL",
    }


def _build_real_sizing_agent(uma, reader):
    agent = object.__new__(uma.UnifiedMarketAgent)
    agent.symbol = "xyz:BRENTOIL"
    agent.reader = reader
    agent.max_planned_loss_usd = 100.0
    agent.local_risk_tolerance_usd = 1.0
    agent.local_size_from_stop = True
    agent.executor = SimpleNamespace(
        resolve_exit_levels=lambda decision, ref_price: {
            "reference_price": ref_price,
            "take_profit_price": decision.take_profit_price,
            "stop_loss_price": decision.stop_loss_price,
        }
    )
    return agent


def _build_real_brent_flat_playbook(uma):
    long_decision = uma.StrategyDecision(
        action="long",
        suggested_notional_usd=0.0,
        entry_price=109.8,
        entry_price_low=109.6,
        entry_price_high=110.0,
        take_profit_price=110.6,
        stop_loss_price=108.75,
        planned_margin_used_usd=0.0,
        planned_max_loss_usd=0.0,
        need_web_confirmation=False,
        requested_leverage=0,
    )
    scenario = uma.Scenario(
        name="Long Reclaim",
        note="",
        observe_when_all=[uma.Condition(type="price_between", low=109.6, high=110.0)],
        arm_when_all=[uma.Condition(type="sustained_ge", level=109.6, seconds=60)],
        cancel_when_any=[],
        timeout_seconds_after_arm=900,
        decision_rules=[
            uma.DecisionRule(
                name="Enter Long",
                when_all=[uma.Condition(type="cross_above", level=109.6)],
                action_decision=long_decision,
            )
        ],
        observation_starts_when="When price first trades inside the 109.60–110.00 zone.",
    )
    return uma.GenericPlaybook(
        display_answer="",
        current_bias="",
        trigger_event_relevance="not_applicable",
        selected_symbol="BRENTOIL-USDC",
        selection_reason="integration_test",
        entry_plan=uma.EntryPlan(
            summary="",
            execute_now=False,
            now_action=uma.build_empty_strategy_decision(),
            scenarios=[scenario],
        ),
        position_management=uma.build_empty_position_management_plan(),
        post_fill_risk_template=uma.build_empty_position_management_plan(),
    )


def _iter_entry_action_decisions(playbook) -> Iterable:
    if playbook.entry_plan.now_action.action in {"long", "short"}:
        yield ("entry_plan.now_action", playbook.entry_plan.now_action)
    for scenario in playbook.entry_plan.scenarios:
        for rule in scenario.decision_rules:
            if rule.action_decision.action in {"long", "short"}:
                yield (f"entry_plan.scenario:{scenario.name}.rule:{rule.name}", rule.action_decision)


@pytest.mark.skipif(not RUN_REAL_OPENAI_TESTS, reason="Set RUN_REAL_OPENAI_TESTS=true to run live OpenAI integration tests.")
def test_real_hyperliquid_flat_isolated_brent_local_sizing_populates_scenario_fields(uma):
    if not os.environ.get("HL_ACCOUNT_ADDRESS", "").strip():
        pytest.skip("HL_ACCOUNT_ADDRESS is required for real Hyperliquid snapshot tests.")

    reader = uma.HyperliquidRestReader()
    agent = _build_real_sizing_agent(uma, reader)
    all_positions = reader.get_all_positions()
    candidate_context = agent._build_candidate_contexts(all_positions, [_build_real_brent_candidate()])[0]
    playbook = _build_real_brent_flat_playbook(uma)

    sizing_debug = agent._apply_local_sizing_to_playbook(playbook, candidate_context, all_positions)
    snapshot = reader.get_position_snapshot(
        "xyz:BRENTOIL",
        all_positions=all_positions,
        current_price=candidate_context.get("current_price"),
    )
    _, decision = next(_iter_entry_action_decisions(playbook))

    assert snapshot.get("only_isolated") is True
    assert decision.suggested_notional_usd > 0.0, {
        "snapshot": snapshot,
        "all_positions": {
            "margin_summary": all_positions.get("margin_summary"),
            "cross_margin_summary": all_positions.get("cross_margin_summary"),
            "isolated_available_margin_usd": all_positions.get("isolated_available_margin_usd"),
            "cross_available_margin_usd": all_positions.get("cross_available_margin_usd"),
            "available_margin_usd": all_positions.get("available_margin_usd"),
            "withdrawable_usd": all_positions.get("withdrawable_usd"),
            "remaining_capital_usd": all_positions.get("remaining_capital_usd"),
            "spot_summary": all_positions.get("spot_summary"),
        },
        "candidate_context": candidate_context,
        "local_sizing_debug": sizing_debug,
        "playbook": playbook.to_dict(),
    }
    assert decision.planned_margin_used_usd > 0.0
    assert decision.planned_max_loss_usd > 0.0


@pytest.mark.skipif(not RUN_REAL_OPENAI_TESTS, reason="Set RUN_REAL_OPENAI_TESTS=true to run live OpenAI integration tests.")
def test_real_openai_and_real_hyperliquid_brent_startup_populates_nonzero_sizing_when_actionable(uma, monkeypatch):
    if not os.environ.get("HL_ACCOUNT_ADDRESS", "").strip():
        pytest.skip("HL_ACCOUNT_ADDRESS is required for real Hyperliquid snapshot tests.")

    monkeypatch.setenv("OPENAI_ACTIVE_SEARCH_MODE", "off")
    monkeypatch.setenv("OPENAI_PASSIVE_SEARCH_MODE", "off")
    monkeypatch.setenv("OPENAI_INCLUDE_CHART_IMAGES", "false")
    monkeypatch.setenv("OPENAI_INCLUDE_PASSIVE_CHART_IMAGES", "false")

    reader = uma.HyperliquidRestReader()
    agent = _build_real_sizing_agent(uma, reader)
    all_positions = reader.get_all_positions()
    candidate_context = agent._build_candidate_contexts(all_positions, [_build_real_brent_candidate()])[0]
    engine = uma.DiscretionaryLLMEngine()
    query = uma.build_default_query(symbol="BRENTOIL-USDC", candidate_labels=["BRENTOIL-USDC"])

    playbook, _ = engine.get_playbook(
        user_query=query,
        recent_events=[],
        trigger_reason="startup",
        trigger_event=None,
        candidate_contexts=[candidate_context],
        active_symbol="",
        has_live_position=False,
    )
    sizing_debug = agent._apply_local_sizing_to_playbook(playbook, candidate_context, all_positions)
    actionable = list(_iter_entry_action_decisions(playbook))
    if not actionable:
        pytest.skip("The live model returned no actionable entry decisions for the real Brent startup query.")

    assert any(
        decision.suggested_notional_usd > 0.0
        and decision.planned_margin_used_usd > 0.0
        and decision.planned_max_loss_usd > 0.0
        for _, decision in actionable
    ), {
        "snapshot": reader.get_position_snapshot(
            "xyz:BRENTOIL",
            all_positions=all_positions,
            current_price=candidate_context.get("current_price"),
        ),
        "candidate_context": candidate_context,
        "local_sizing_debug": sizing_debug,
        "playbook": playbook.to_dict(),
        "last_call_debug": getattr(engine, "last_call_debug", {}),
    }



@pytest.mark.skipif(not RUN_REAL_OPENAI_TESTS, reason="Set RUN_REAL_OPENAI_TESTS=true to run live OpenAI integration tests.")
def test_real_openai_get_playbook_returns_schema_valid_response(uma, monkeypatch):
    monkeypatch.setenv("OPENAI_ACTIVE_SEARCH_MODE", "off")
    monkeypatch.setenv("OPENAI_PASSIVE_SEARCH_MODE", "off")
    monkeypatch.setenv("OPENAI_SEARCH_MODE", "off")
    monkeypatch.setenv("OPENAI_ACTIVE_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("OPENAI_PASSIVE_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("OPENAI_ACTIVE_REASONING_EFFORT", "low")

    engine = uma.DiscretionaryLLMEngine()
    candidate_contexts = _build_candidate_contexts()
    query = uma.render_query_template(
        "",
        "BTC",
        {
            "remaining_capital_usd": 300.0,
            "max_planned_loss_usd": 100.0,
            "candidate_labels": [item["display_name"] for item in candidate_contexts],
        },
    )
    assert "Do not write separate plans based on whether the account is already holding a position" not in query
    assert "market-only entry_plan" in query
    assert "当前已有仓位时的管理方案" not in query
    assert "This round's candidate instruments are: BTC-USDC, BRENTOIL-USDC, SILVER-USDC." in query

    playbook, mode = engine.get_playbook(
        user_query=query,
        recent_events=[],
        trigger_reason="manual_once",
        trigger_event=None,
        candidate_contexts=candidate_contexts,
        active_symbol="BTC",
    )

    assert mode == "raw_context_only"
    assert playbook.entry_plan is not None
    assert playbook.position_management is not None
    assert playbook.to_dict()["target_position"]["position_state"] in {"open", "flat", "unknown"}
    assert playbook.to_dict()["entry_plan"]["now_action"]["action"] in {"long", "short", "hold", "no_trade"}
    assert hasattr(engine, "last_call_debug")
    assert engine.last_call_debug["raw_output_text"].strip()
    assert "capped_playbook" in engine.last_call_debug
    _assert_no_pct_exit_keys(playbook.to_dict())


@pytest.mark.skipif(not RUN_REAL_OPENAI_TESTS, reason="Set RUN_REAL_OPENAI_TESTS=true to run live OpenAI integration tests.")
def test_real_openai_query_mentions_budget_and_response_is_usable(uma, monkeypatch):
    monkeypatch.setenv("OPENAI_ACTIVE_SEARCH_MODE", "off")
    monkeypatch.setenv("OPENAI_PASSIVE_SEARCH_MODE", "off")
    monkeypatch.setenv("OPENAI_SEARCH_MODE", "off")
    monkeypatch.setenv("OPENAI_ACTIVE_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("OPENAI_PASSIVE_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("OPENAI_ACTIVE_REASONING_EFFORT", "low")

    engine = uma.DiscretionaryLLMEngine()
    candidate_contexts = _build_candidate_contexts()
    query = uma.render_query_template(
        "",
        "BTC",
        {
            "remaining_capital_usd": 300.0,
            "max_planned_loss_usd": 100.0,
            "candidate_labels": [item["display_name"] for item in candidate_contexts],
        },
    )

    playbook, _ = engine.get_playbook(
        user_query=query,
        recent_events=[],
        trigger_reason="manual_once",
        trigger_event=None,
        candidate_contexts=candidate_contexts,
        active_symbol="BTC",
    )

    serialized = playbook.to_dict()

    _assert_no_pct_exit_keys(serialized)
    assert "target_position" in serialized
    assert "entry_plan" in serialized
    assert "position_management" in serialized
    execution_view = uma.build_playbook_execution_view(playbook)
    assert "target_position" in execution_view
    assert "entry_plan" in execution_view
    coverage = _collect_parsed_coverage(serialized)
    structured_candidates = coverage["actionable_candidates"]

    assert structured_candidates or coverage["observation_texts"] or coverage["confirmation_rules"] >= 1, (
        "Expected the live model to return either an actionable setup or a structured wait-and-observe setup."
    )
    if structured_candidates:
        assert any(
            candidate["entry_price"] > 0 or (candidate["entry_price_low"] > 0 and candidate["entry_price_high"] > 0)
            for candidate in structured_candidates
        )
        assert all(candidate["planned_margin_used_usd"] >= 0 for candidate in structured_candidates)
        assert all(candidate["planned_max_loss_usd"] >= 0 for candidate in structured_candidates)
    if coverage["observation_texts"]:
        assert coverage["confirmation_rules"] >= 1, "Expected at least one structured confirmation rule."
    assert coverage["sl_leg_count"] >= 1 or coverage["entry_sl_prices"], "Expected structured stop-loss information."
    assert coverage["tp_leg_count"] >= 1 or coverage["entry_tp_prices"], "Expected structured take-profit information."


@pytest.mark.skipif(not RUN_REAL_OPENAI_TESTS, reason="Set RUN_REAL_OPENAI_TESTS=true to run live OpenAI integration tests.")
def test_real_openai_mini_with_search_enabled_records_web_search_calls(uma, monkeypatch):
    monkeypatch.setenv("OPENAI_ACTIVE_SEARCH_MODE", "always")
    monkeypatch.setenv("OPENAI_PASSIVE_SEARCH_MODE", "context_only")
    monkeypatch.setenv("OPENAI_ACTIVE_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("OPENAI_PASSIVE_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("OPENAI_ACTIVE_REASONING_EFFORT", "medium")

    engine = uma.DiscretionaryLLMEngine()
    candidate_contexts = _build_candidate_contexts()
    query = (
        "This round's candidate instruments are: BTC-USDC, BRENTOIL-USDC, SILVER-USDC. "
        "Compare these candidates first, then select exactly one instrument with the best opportunity right now. "
        "Use web search at least once to verify the latest public market background as of today before deciding which instrument is best. "
        "Reflect the verified conclusion in the returned selected_symbol and entry_plan. "
        "Do not use web search for price, candlestick, or technical-indicator lookup; rely on the supplied Hyperliquid context for price and market structure, and use web search only for news, macro, geopolitics, inventory, regulation, and supply-demand background. "
        "Return a structured market-only trading plan for the best current opportunity; if no trade is worth taking, explicitly output no_trade."
    )

    playbook, mode = engine.get_playbook(
        user_query=query,
        recent_events=[],
        trigger_reason="manual_once",
        trigger_event=None,
        candidate_contexts=candidate_contexts,
        active_symbol="BTC",
    )

    assert mode in {"context_enriched_with_web", "verified_with_web"}
    _assert_no_pct_exit_keys(playbook.to_dict())
    assert engine.last_call_debug["response_model"]
    assert engine.last_call_debug["web_search_tool_calls"] >= 1
    assert isinstance(engine.last_call_debug.get("web_search_calls"), list)
    assert engine.last_call_debug["web_search_calls"], "Expected at least one recorded web_search call detail."


@pytest.mark.skipif(not RUN_REAL_OPENAI_TESTS, reason="Set RUN_REAL_OPENAI_TESTS=true to run live OpenAI integration tests.")
def test_real_openai_passive_trigger_omits_recent_events_by_default_and_keeps_search_off(uma, monkeypatch):
    monkeypatch.setenv("OPENAI_ACTIVE_SEARCH_MODE", "off")
    monkeypatch.setenv("OPENAI_PASSIVE_SEARCH_MODE", "off")
    monkeypatch.setenv("OPENAI_SEARCH_MODE", "off")
    monkeypatch.setenv("OPENAI_ACTIVE_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("OPENAI_PASSIVE_MODEL", "gpt-5.4-mini")
    monkeypatch.setenv("OPENAI_PASSIVE_REASONING_EFFORT", "low")
    monkeypatch.setenv("PASSIVE_RECENT_EVENT_LIMIT", "0")

    engine = uma.DiscretionaryLLMEngine()
    candidate_contexts = _build_candidate_contexts()
    trigger_event = {
        "source": "integration",
        "title": "Hormuz shipping risk escalates after new official warning",
        "summary": "Officials warned that shipping conditions in Hormuz may deteriorate quickly.",
        "published_at": "2026-04-03T00:00:00Z",
    }
    recent_events = [trigger_event] + [
        {
            "source": "integration",
            "title": f"Background headline {idx}",
            "summary": "Older background context.",
            "published_at": "2026-04-02T23:30:00Z",
        }
        for idx in range(11)
    ]
    query = uma.render_query_template(
        "",
        "BTC",
        {
            "remaining_capital_usd": 300.0,
            "max_planned_loss_usd": 100.0,
            "candidate_labels": [item["display_name"] for item in candidate_contexts],
        },
    )

    playbook, mode = engine.get_playbook(
        user_query=query,
        recent_events=recent_events,
        trigger_reason="passive_event_trigger",
        trigger_event=trigger_event,
        candidate_contexts=candidate_contexts,
        active_symbol="BTC",
    )

    assert mode == "raw_context_only"
    _assert_no_pct_exit_keys(playbook.to_dict())
    assert engine.last_call_debug["web_search_tool_calls"] == 0
    payload = _extract_request_user_payload(engine)
    assert payload["trigger_reason"] == "passive_event_trigger"
    assert payload["trigger_event"]["title"] == trigger_event["title"]
    assert "recent_events" not in payload
    assert "recent_event_context" not in payload
