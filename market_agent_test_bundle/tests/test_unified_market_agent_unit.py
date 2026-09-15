from __future__ import annotations

from collections import deque
import json
from types import SimpleNamespace

import pytest


def install_test_helper_reset_profile(uma, agent, *, symbol="BTC"):
    profile = uma.InstrumentMarketProfile(
        name=symbol,
        timezone_name="UTC",
        helper_reset_time=(2, 0, 0),
        low_liquidity_windows=(),
        normal_liquidity_windows=(),
        reset_skip_weekdays=(6, 0),
    )
    agent.symbol = symbol
    agent.trade_candidates = [{"execution_symbol": symbol}]
    agent.instrument_market_profiles = {
        key: profile
        for key in agent._profile_lookup_keys(symbol)
    }
    return profile


def test_instrument_market_profile_parses_r_clip_multiples(uma):
    profile = uma.HelperResetMixin._coerce_instrument_market_profile(
        "BRENTOIL",
        {
            "name": "brent",
            "timezone": "Europe/London",
            "helper_reset_timezone": "America/New_York",
            "pre_disabled_weekday_reset_time": "06:00",
            "low_liquidity_windows": ["01:00-08:00"],
            "low_liquidity_weekdays": ["sat", "sun"],
            "low_liquidity_trade_disabled_weekdays": ["sun"],
            "normal_liquidity_windows": ["08:00-22:00"],
            "normal_liquidity_r_min_atr_multiple": 1.5,
            "normal_liquidity_r_max_atr_multiple": 2.5,
            "low_liquidity_r_min_atr_multiple": 2.5,
            "low_liquidity_r_max_atr_multiple": 3.0,
        },
    )

    assert profile is not None
    assert profile.normal_liquidity_r_min_atr_multiple == pytest.approx(1.5)
    assert profile.normal_liquidity_r_max_atr_multiple == pytest.approx(2.5)
    assert profile.low_liquidity_r_min_atr_multiple == pytest.approx(2.5)
    assert profile.low_liquidity_r_max_atr_multiple == pytest.approx(3.0)
    assert profile.low_liquidity_weekdays == (5, 6)
    assert profile.low_liquidity_trade_disabled_weekdays == (6,)
    assert profile.pre_disabled_weekday_reset_time == (6, 0, 0)
    assert profile.helper_reset_timezone_name == "America/New_York"


def test_low_liquidity_trade_disabled_profile_uses_configured_weekdays(uma):
    agent = object.__new__(uma.UnifiedMarketAgent)
    profile = uma.InstrumentMarketProfile(
        name="brent",
        timezone_name="UTC",
        helper_reset_time=None,
        low_liquidity_windows=(uma.LocalTimeWindow((1, 0, 0), (8, 0, 0)),),
        normal_liquidity_windows=(uma.LocalTimeWindow((8, 0, 0), (22, 0, 0)),),
        low_liquidity_weekdays=(5, 6),
        low_liquidity_trade_disabled_weekdays=(6,),
    )
    agent.instrument_market_profiles = {key: profile for key in agent._profile_lookup_keys("BRENTOIL")}

    weekday_low_window = uma.datetime(2026, 4, 29, 2, 0, tzinfo=uma.timezone.utc)
    saturday_normal_window = uma.datetime(2026, 5, 2, 12, 0, tzinfo=uma.timezone.utc)
    sunday_normal_window = uma.datetime(2026, 5, 3, 12, 0, tzinfo=uma.timezone.utc)

    assert agent._low_liquidity_profile_for_symbol("BRENTOIL", weekday_low_window) is profile
    assert agent._low_liquidity_trade_disabled_profile_for_symbol("BRENTOIL", weekday_low_window) is None
    assert agent._low_liquidity_profile_for_symbol("BRENTOIL", saturday_normal_window) is profile
    assert agent._low_liquidity_trade_disabled_profile_for_symbol("BRENTOIL", saturday_normal_window) is None
    assert agent._low_liquidity_trade_disabled_profile_for_symbol("BRENTOIL", sunday_normal_window) is profile


def make_entry_decision(uma, *, action="no_trade", notional=1000.0, stop_loss_price=0.0, leverage=10):
    return uma.StrategyDecision(
        action=action,
        suggested_notional_usd=notional,
        entry_price=100000.0,
        stop_loss_price=stop_loss_price,
        planned_margin_used_usd=0.0,
        planned_max_loss_usd=0.0,
        requested_leverage=leverage,
    )


def make_management_decision(uma, *, action="add_to_long", notional=1000.0, leverage=10, stop_loss_price=0.0):
    return uma.ManagementDecision(
        action=action,
        close_fraction=0.0,
        new_notional_usd=notional,
        entry_price=100000.0,
        stop_loss_price=stop_loss_price,
        planned_max_loss_usd=0.0,
        leverage=leverage,
        margin_basis_usd=0.0,
    )




def make_management_scenario(uma, *, observe_low=99500.0, observe_high=100000.0, trigger_level=100200.0, timeout_seconds=1800):
    return uma.Scenario(
        observe_when_all={"low": observe_low, "high": observe_high},
        execute_when_all={
            "condition": uma.Condition(type="price_between", low=observe_low, high=observe_high, note=""),
            "timeout_seconds": timeout_seconds,
        },
    )


def make_management_plan(uma, *, action="add_to_long", notional=1000.0, leverage=10, stop_loss_price=0.0, scenario=None):
    return uma.PositionManagementPlan(
        execute_now=False,
        action_decision=make_management_decision(
            uma,
            action=action,
            notional=notional,
            leverage=leverage,
            stop_loss_price=stop_loss_price,
        ),
        scenario=scenario,
    )


def test_llm_entry_decision_schema_only_uses_entry_and_stop(uma):
    entry_schema = uma.PLAYBOOK_SCHEMA["schema"]["$defs"]["entry_decision"]

    assert set(entry_schema["properties"]) == {"action", "entry_price", "stop_loss_price"}
    assert set(entry_schema["required"]) == {"action", "entry_price", "stop_loss_price"}


def test_helper_market_context_schema_uses_diagnostic_instruments(uma):
    context_schema = uma.HELPER_MARKET_NEWS_CONTEXT_SCHEMA["schema"]["properties"]["market_mainline_context"]

    assert "diagnostic_instruments" in context_schema["properties"]
    assert "diagnostic_instruments" in context_schema["required"]
    assert "influencing_instruments" not in context_schema["properties"]
    assert "influencing_instruments" not in context_schema["required"]


def test_default_diagnostic_universe_uses_vixy_proxy_not_vix_spot(uma):
    assert "VIXY" in uma.DEFAULT_DIAGNOSTIC_INSTRUMENT_UNIVERSE
    assert "VIX" not in uma.DEFAULT_DIAGNOSTIC_INSTRUMENT_UNIVERSE


def test_market_mainline_context_normalizes_diagnostic_instruments(uma):
    context = uma.DiscretionaryLLMEngine._normalize_market_mainline_context(
        {
            "current_move_logic_mainline": "mainline",
            "diagnostic_instruments": ["Brent-WTI spread", "DXY", "dxy"],
        },
        diagnostic_universe=["DXY", "GLD"],
    )

    assert context == {
        "current_move_logic_mainline": "mainline",
        "diagnostic_instruments": ["DXY"],
    }


def make_agent_stub(uma, *, max_loss=100.0, max_leverage=40, taker_fee_rate=0.0):
    agent = object.__new__(uma.UnifiedMarketAgent)
    agent.symbol = "BTC"
    agent.max_planned_loss_usd = max_loss
    agent.risk_tp1_r_multiple = 1.0
    agent.risk_tp2_r_multiple = 2.0
    agent.risk_tp1_close_fraction = 0.30
    agent.risk_tp2_close_fraction = 0.40
    agent.risk_post_tp1_stop_r_multiple = -0.40
    agent.risk_post_tp2_locked_r_multiple = 1.0
    agent.risk_trailing_timeframe = "15m"
    agent.risk_trailing_atr_period = 14
    agent.risk_trailing_atr_lookback_bars = 200
    agent.risk_trailing_soft_atr_multiple = 2.5
    agent.risk_trailing_hard_atr_multiple = 3.5
    agent.local_risk_tolerance_usd = 1.0
    agent.local_no_change_close_fraction_tolerance = 0.01
    agent.local_size_from_stop = True
    agent.position_size_change_tol = 0.0
    agent.price_history_seconds = 1800
    agent.loop_exception_sleep_seconds = 5.0
    agent.hyperliquid_transient_error_sleep_seconds = 30.0
    agent.risk_session = None
    agent.current_playbook = None
    agent.reader = SimpleNamespace(
        get_market_spec=lambda symbol: {"max_leverage": max_leverage},
        get_user_fee_rates=lambda: {
            "known": False,
            "taker_fee_rate": taker_fee_rate,
            "maker_fee_rate": 0.0,
            "source": "test_stub",
        },
    )
    agent.executor = SimpleNamespace()
    return agent


def test_normalize_selected_symbol_does_not_fallback_to_first_candidate(uma):
    agent = object.__new__(uma.UnifiedMarketAgent)
    playbook = uma.GenericPlaybook(
        display_answer="",
        current_bias="neutral",
        selected_symbol="UNKNOWN-USDC",
    )
    trade_symbol_context = {"trade_symbol_key": "BTC-USDC", "display_name": "BTC-USDC", "execution_symbol": "BTC"}

    selected = agent._normalize_selected_symbol(playbook, trade_symbol_context)

    assert selected == ""
    assert playbook.selected_symbol == ""


def test_passive_event_judge_batch_prefetch_selects_highest_confidence(uma):
    agent = object.__new__(uma.UnifiedMarketAgent)

    def call_passive_event_judge(**kwargs):
        title = str((kwargs.get("trigger_event") or {}).get("title", "") or "")
        confidence = 0.72 if "strong" in title else 0.18
        action = "long" if confidence >= 0.35 else "no_trade"
        relevance = "relevant" if confidence >= 0.35 else "unrelated"
        return (
            {
                "trigger_event_relevance": relevance,
                "trigger_confidence": confidence,
                "action": action,
            },
            {
                "response_id": title,
                "usage": None,
                "usage_cost": None,
                "web_search_tool_calls": 0,
                "web_search_calls": [],
            },
        )

    agent.engine = SimpleNamespace(_call_passive_event_judge=call_passive_event_judge)
    request = {
        "started_at": 100.0,
        "event_requests": [
            {
                "event_key": "weak",
                "trigger_event": {"source": "mni", "title": "weak cpi note"},
                "recent_events": [],
                "market_mainline_context": {},
                "phase": "fast",
                "reasoning_effort": "low",
                "trade_symbol": "BRENTOIL-USDC",
                "started_at": 100.0,
            },
            {
                "event_key": "strong",
                "trigger_event": {"source": "mni", "title": "strong hormuz shock"},
                "recent_events": [],
                "market_mainline_context": {},
                "phase": "fast",
                "reasoning_effort": "low",
                "trade_symbol": "BRENTOIL-USDC",
                "started_at": 100.0,
            },
        ],
    }

    result = agent._run_passive_event_judge_batch_prefetch(request)

    assert result["event_key"] == "strong"
    assert result["trigger_event"]["title"] == "strong hormuz shock"
    assert result["judge_output"]["trigger_confidence"] == pytest.approx(0.72)
    assert result["judge_output"]["action"] == "long"
    assert result["batch_event_count"] == 2
    assert result["batch_selection_rule"] == "realtime_relevant_max_then_relevant_tape_only_then_unrelated_then_duplicate"
    assert len(result["batch_candidates"]) == 2
    assert result["judge_debug"]["passive_event_judge_batch_event_count"] == 2


def test_select_visual_trade_symbol_context_requires_winner_or_active_symbol(uma):
    context = {"display_name": "BRENTOIL-USDC", "execution_symbol": "BRENTOIL"}

    assert uma.DiscretionaryLLMEngine._select_visual_trade_symbol_context({}, active_symbol="", market_news_debug={}) is None

    assert uma.DiscretionaryLLMEngine._select_visual_trade_symbol_context(
        context,
        active_symbol="BTC",
        market_news_debug={},
    ) is None

    selected = uma.DiscretionaryLLMEngine._select_visual_trade_symbol_context(
        context,
        active_symbol="BRENTOIL",
        market_news_debug={},
    )

    assert selected["display_name"] == "BRENTOIL-USDC"


def make_query_agent_stub(uma):
    agent = object.__new__(uma.UnifiedMarketAgent)
    helper_calls = []
    playbook_calls = []
    audit_events = []
    flat_snapshot = {
        "symbol": "BRENTOIL-USDC",
        "side": "flat",
        "size": 0.0,
        "entry_price": 0.0,
        "mid_price": 100.0,
        "notional_usd": 0.0,
    }

    class FakeEngine:
        default_search_mode = "on"
        active_search_mode = "on"
        passive_search_mode = "on"
        force_active_news_context = True
        force_passive_news_context = True
        last_call_debug = {}

        def _build_market_news_context(self, **kwargs):
            helper_calls.append(kwargs)
            return {"selected_instrument": "BRENTOIL-USDC"}, {"winner_display_name": "BRENTOIL-USDC"}

        def get_playbook(self, **kwargs):
            playbook_calls.append(kwargs)
            raise AssertionError("active playbook should not be called in this test")

    agent.symbol = "BRENTOIL-USDC"
    agent.pending_entry_order_session = None
    agent.event_context_max_items = 10
    agent.enable_active_playbook = False
    agent.current_playbook = None
    agent.current_mode = None
    agent.current_playbook_reason = ""
    agent.last_playbook_query_at = 0.0
    agent.reader = SimpleNamespace(
        get_all_positions=lambda: {"positions": [], "positions_count": 0},
        get_mid_price=lambda symbol: 100.0,
        get_position_snapshot=lambda symbol, all_positions=None, current_price=None: dict(flat_snapshot),
    )
    agent.events = SimpleNamespace(recent=lambda: [{"source": "reuters", "title": "A"}])
    agent.engine = FakeEngine()
    agent.step_pending_entry_order_session = lambda now: None
    agent._resolve_query_candidates = lambda all_positions, reason: (["BRENTOIL-USDC"], {})
    agent._find_management_symbol = lambda all_positions: "BRENTOIL-USDC"
    agent._build_candidate_contexts = lambda all_positions, candidates: [
        {"symbol": "BRENTOIL-USDC", "display_name": "BRENTOIL-USDC", "price": 100.0}
    ]
    agent._resolve_recent_passive_event_symbol = lambda **kwargs: "BRENTOIL-USDC"
    agent._recent_relevant_passive_events = lambda **kwargs: []
    agent._filter_recent_events_for_active_helper = lambda events: list(events or [])
    agent.render_user_query = lambda all_positions, query_candidates: "query"
    agent._empty_runtime_snapshot = lambda all_positions=None: dict(flat_snapshot)
    agent._print_json_block = lambda *args, **kwargs: None
    agent._audit_event = lambda event_type, payload=None: audit_events.append((event_type, payload or {}))
    agent._schedule_next_active_query = lambda position_snapshot=None: None
    agent._refresh_passive_llm_recent_events_from_helper = lambda: None
    return agent, helper_calls, playbook_calls, audit_events


def test_active_playbook_disabled_position_flat_does_not_run_helper_only(uma):
    agent, helper_calls, playbook_calls, audit_events = make_query_agent_stub(uma)

    agent.query_new_playbook("position_flat", None)

    assert helper_calls == []
    assert playbook_calls == []
    assert audit_events[-1][0] == "active_playbook_skipped"
    assert audit_events[-1][1]["helper_query_enabled"] is True
    assert audit_events[-1][1]["helper_query_invoked"] is False


def test_helper_reset_refresh_runs_helper_only_when_active_playbook_disabled(uma):
    agent, helper_calls, playbook_calls, audit_events = make_query_agent_stub(uma)

    agent.query_new_playbook("helper_reset_refresh", None)

    assert len(helper_calls) == 1
    assert playbook_calls == []
    assert audit_events[-1][0] == "active_playbook_skipped"
    assert audit_events[-1][1]["helper_query_enabled"] is True
    assert audit_events[-1][1]["helper_query_invoked"] is True


def test_startup_skips_helper_and_active_playbook(uma):
    agent, helper_calls, playbook_calls, audit_events = make_query_agent_stub(uma)

    agent.query_new_playbook("startup", None)

    assert helper_calls == []
    assert playbook_calls == []
    assert audit_events[-1][0] == "active_playbook_skipped"
    assert audit_events[-1][1]["helper_query_enabled"] is True
    assert audit_events[-1][1]["helper_query_invoked"] is False


def test_hydrate_llm_relevant_passive_event_buffer_from_log_seeds_recent_limit(tmp_path, uma):
    agent = object.__new__(uma.UnifiedMarketAgent)
    agent.event_context_max_items = 2
    agent.passive_event_context_max_items = 2
    agent.passive_llm_relevant_event_buffer_size = 20
    agent.passive_relevant_events_log_base_path = tmp_path / "passive_relevant_events.jsonl"
    agent.passive_relevant_events_log_path = agent.passive_relevant_events_log_base_path
    agent.llm_relevant_passive_events_by_symbol = {}
    rows = [
        {
            "recorded_at": "2026-04-14T00:00:10Z",
            "symbol": "BTC-USDC",
            "event": {"source": "reuters", "title": "A", "event_timestamp": "2026-04-14T00:00:00Z"},
        },
        {
            "recorded_at": "2026-04-14T00:01:10Z",
            "symbol": "BTC-USDC",
            "event": {"source": "reuters", "title": "B", "event_timestamp": "2026-04-14T00:01:00Z"},
        },
        {
            "recorded_at": "2026-04-14T00:02:10Z",
            "symbol": "BTC-USDC",
            "event": {"source": "reuters", "title": "C", "event_timestamp": "2026-04-14T00:02:00Z"},
        },
    ]
    with agent._passive_relevant_events_log_path_for_symbol("BTC-USDC").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    with agent._passive_relevant_events_log_path_for_symbol("ETH-USDC").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({
            "recorded_at": "2026-04-14T00:03:10Z",
            "symbol": "ETH-USDC",
            "event": {"source": "reuters", "title": "ETH-A", "event_timestamp": "2026-04-14T00:03:00Z"},
        }) + "\n")

    agent._hydrate_llm_relevant_passive_event_buffer_from_log()

    assert [item["title"] for item in agent.llm_relevant_passive_events_by_symbol["BTC-USDC"]] == ["B", "C"]
    assert [item["title"] for item in agent.llm_relevant_passive_events_by_symbol["ETH-USDC"]] == ["ETH-A"]


def test_build_risk_session_stage_exit_legs_uses_staged_tp1_tp2_and_initial_stop(uma):
    agent = make_agent_stub(uma)
    decision = uma.StrategyDecision(
        action="long",
        suggested_notional_usd=1000.0,
        entry_price=100000.0,
        stop_loss_price=99000.0,
        planned_margin_used_usd=0.0,
        planned_max_loss_usd=0.0,
        requested_leverage=10,
    )

    take_profit_legs, stop_loss_legs = agent._build_risk_session_stage_exit_legs(
        side="long",
        entry_price=decision.entry_price,
        stop_loss_price=decision.stop_loss_price,
    )

    assert [(leg.name, leg.close_fraction, leg.when_all[0].level) for leg in take_profit_legs] == [
        ("stage_tp1", pytest.approx(0.30), pytest.approx(101000.0)),
        ("stage_tp2", pytest.approx(0.40), pytest.approx(102000.0)),
    ]
    assert [(leg.name, leg.close_fraction, leg.when_all[0].level) for leg in stop_loss_legs] == [
        ("stage_initial_stop", pytest.approx(1.0), pytest.approx(99000.0)),
    ]


def test_atr_from_completed_candles_uses_wilder_rma_not_sma(uma):
    candles = [
        {"close_ms": 1, "high": 110.0, "low": 100.0, "close": 105.0},
        {"close_ms": 2, "high": 125.0, "low": 105.0, "close": 120.0},
        {"close_ms": 3, "high": 150.0, "low": 120.0, "close": 145.0},
        {"close_ms": 4, "high": 185.0, "low": 145.0, "close": 180.0},
    ]

    atr = uma.UnifiedMarketAgent._atr_from_completed_candles(candles, 3)

    assert atr == pytest.approx((20.0 * 2.0 + 40.0) / 3.0)
    assert atr != pytest.approx((20.0 + 30.0 + 40.0) / 3.0)


def test_latest_completed_candles_for_risk_session_returns_recent_lookback_only(uma):
    agent = make_agent_stub(uma)
    interval_ms = 15 * 60 * 1000
    rows = [
        {"t": index * interval_ms, "h": 100.0 + index, "l": 99.0 + index, "c": 99.5 + index}
        for index in range(25)
    ]

    def fake_candles(symbol, interval, start_ms, end_ms):
        return [
            dict(row)
            for row in rows
            if int(row["t"]) >= int(start_ms) and int(row["t"]) <= int(end_ms)
        ]

    agent.reader = SimpleNamespace(get_candles_snapshot=fake_candles)
    session = uma.RiskSession(
        trailing_timeframe="15m",
        trailing_atr_period=2,
        trailing_atr_lookback_bars=12,
    )

    candles = agent._latest_completed_candles_for_risk_session(session, now=(25 * 15 * 60) + 1)

    assert len(candles) == 12
    assert [item["open_ms"] for item in candles] == [index * interval_ms for index in range(13, 25)]


def test_runtime_error_sleep_uses_hyperliquid_transient_backoff(uma):
    agent = make_agent_stub(uma)
    response = uma.requests.Response()
    response.status_code = 502
    response.url = "https://api.hyperliquid.xyz/info"
    exc = uma.requests.exceptions.HTTPError(
        "502 Server Error: Bad Gateway for url: https://api.hyperliquid.xyz/info",
        response=response,
    )

    assert agent._runtime_error_sleep_seconds(exc) == pytest.approx(30.0)
    assert agent._runtime_error_sleep_seconds(RuntimeError("local bug")) == pytest.approx(5.0)


def test_engine_loads_passive_recent_events_from_helper_materiality_tail(tmp_path, uma):
    engine = object.__new__(uma.DiscretionaryLLMEngine)
    engine.helper_materially_new_first_events_path = tmp_path / "helper_materially_new_first_events.jsonl"
    rows = [
        {
            "materially_new_first_events": [
                {
                    "event_timestamp": "2026-04-20T10:00:00Z",
                    "item_id": "brent-1",
                    "source": "reuters",
                    "title": "Brent one",
                    "url": "https://example.com/brent-1",
                },
                {
                    "event_timestamp": "2026-04-20T10:05:00Z",
                    "item_id": "brent-2",
                    "source": "reuters",
                    "title": "Brent two",
                    "url": "https://example.com/brent-2",
                },
            ]
        },
        {
            "materially_new_first_events": [
                {
                    "event_timestamp": "2026-04-20T10:10:00Z",
                    "item_id": "btc-1",
                    "source": "reuters",
                    "title": "BTC one",
                    "url": "https://example.com/btc-1",
                }
            ]
        },
        {
            "materially_new_first_events": [
                {
                    "event_timestamp": "2026-04-20T10:20:00Z",
                    "item_id": "brent-3",
                    "source": "reuters",
                    "title": "Brent three",
                    "url": "https://example.com/brent-3",
                },
                {
                    "event_timestamp": "2026-04-20T10:25:00Z",
                    "item_id": "brent-4",
                    "source": "reuters",
                    "title": "Brent four",
                    "url": "https://example.com/brent-4",
                },
                {
                    "event_timestamp": "2026-04-20T10:30:00Z",
                    "item_id": "brent-5",
                    "source": "reuters",
                    "title": "Brent five",
                    "url": "https://example.com/brent-5",
                },
            ]
        },
        {
            "materially_new_first_events": [
                {
                    "event_timestamp": "2026-04-20T10:35:00Z",
                    "item_id": "brent-6",
                    "source": "reuters",
                    "title": "Brent six",
                    "url": "https://example.com/brent-6",
                }
            ]
        },
    ]
    with engine.helper_materially_new_first_events_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    recent = engine._load_passive_recent_events_from_helper_materiality("BRENTOIL-USDC", max_items=5)

    assert [item["title"] for item in recent] == [
        "BTC one",
        "Brent three",
        "Brent four",
        "Brent five",
        "Brent six",
    ]
    assert all("source" not in item and "event_timestamp" not in item and "item_id" not in item for item in recent)


def test_compute_next_helper_reset_at_skips_sunday_and_monday(uma):
    agent = object.__new__(uma.UnifiedMarketAgent)
    install_test_helper_reset_profile(uma, agent)

    same_day = agent._compute_next_helper_reset_at(uma.datetime(2026, 4, 24, 1, 30, tzinfo=uma.timezone.utc))
    next_allowed_day = agent._compute_next_helper_reset_at(uma.datetime(2026, 4, 24, 2, 30, tzinfo=uma.timezone.utc))
    saturday = agent._compute_next_helper_reset_at(uma.datetime(2026, 4, 25, 1, 30, tzinfo=uma.timezone.utc))
    sunday = agent._compute_next_helper_reset_at(uma.datetime(2026, 4, 26, 1, 30, tzinfo=uma.timezone.utc))
    monday = agent._compute_next_helper_reset_at(uma.datetime(2026, 4, 27, 1, 30, tzinfo=uma.timezone.utc))

    assert same_day == uma.datetime(2026, 4, 24, 2, 0, 0, tzinfo=uma.timezone.utc)
    assert next_allowed_day == uma.datetime(2026, 4, 25, 2, 0, 0, tzinfo=uma.timezone.utc)
    assert saturday == uma.datetime(2026, 4, 25, 2, 0, 0, tzinfo=uma.timezone.utc)
    assert sunday == uma.datetime(2026, 4, 28, 2, 0, 0, tzinfo=uma.timezone.utc)
    assert monday == uma.datetime(2026, 4, 28, 2, 0, 0, tzinfo=uma.timezone.utc)


def test_compute_next_helper_reset_at_includes_pre_disabled_weekday_reset(uma):
    agent = object.__new__(uma.UnifiedMarketAgent)
    profile = uma.InstrumentMarketProfile(
        name="brent",
        timezone_name="UTC",
        helper_reset_time=(22, 50, 0),
        low_liquidity_windows=(),
        normal_liquidity_windows=(),
        pre_disabled_weekday_reset_time=(6, 0, 0),
        low_liquidity_trade_disabled_weekdays=(6,),
        reset_skip_weekdays=(6,),
    )
    agent.symbol = "BRENTOIL"
    agent.trade_candidates = [{"execution_symbol": "BRENTOIL"}]
    agent.instrument_market_profiles = {
        key: profile
        for key in agent._profile_lookup_keys("BRENTOIL")
    }

    saturday_before_main = agent._compute_next_helper_reset_at(
        uma.datetime(2026, 4, 25, 21, 0, tzinfo=uma.timezone.utc)
    )
    saturday_after_main = agent._compute_next_helper_reset_at(
        uma.datetime(2026, 4, 25, 23, 0, tzinfo=uma.timezone.utc)
    )
    sunday_before_extra = agent._compute_next_helper_reset_at(
        uma.datetime(2026, 4, 26, 5, 30, tzinfo=uma.timezone.utc)
    )
    sunday_after_extra = agent._compute_next_helper_reset_at(
        uma.datetime(2026, 4, 26, 6, 1, tzinfo=uma.timezone.utc)
    )

    assert saturday_before_main == uma.datetime(2026, 4, 25, 22, 50, 0, tzinfo=uma.timezone.utc)
    assert saturday_after_main == uma.datetime(2026, 4, 26, 6, 0, 0, tzinfo=uma.timezone.utc)
    assert sunday_before_extra == uma.datetime(2026, 4, 26, 6, 0, 0, tzinfo=uma.timezone.utc)
    assert sunday_after_extra == uma.datetime(2026, 4, 27, 22, 50, 0, tzinfo=uma.timezone.utc)


def test_compute_next_helper_reset_at_uses_helper_reset_timezone(uma):
    agent = object.__new__(uma.UnifiedMarketAgent)
    profile = uma.InstrumentMarketProfile(
        name="brent",
        timezone_name="Europe/London",
        helper_reset_timezone_name="America/New_York",
        helper_reset_time=(16, 0, 0),
        low_liquidity_windows=(),
        normal_liquidity_windows=(),
        reset_skip_weekdays=(6,),
    )
    agent.symbol = "BRENTOIL"
    agent.trade_candidates = [{"execution_symbol": "BRENTOIL"}]
    agent.instrument_market_profiles = {
        key: profile
        for key in agent._profile_lookup_keys("BRENTOIL")
    }

    summer_before_close = agent._compute_next_helper_reset_at(
        uma.datetime(2026, 5, 19, 19, 30, tzinfo=uma.timezone.utc)
    )
    winter_before_close = agent._compute_next_helper_reset_at(
        uma.datetime(2026, 1, 6, 20, 30, tzinfo=uma.timezone.utc)
    )

    assert summer_before_close == uma.datetime(2026, 5, 19, 20, 0, 0, tzinfo=uma.timezone.utc)
    assert winter_before_close == uma.datetime(2026, 1, 6, 21, 0, 0, tzinfo=uma.timezone.utc)


def test_perform_helper_reset_clears_state_and_advances_schedule_when_flat(uma):
    agent = object.__new__(uma.UnifiedMarketAgent)
    install_test_helper_reset_profile(uma, agent)
    agent.next_helper_reset_at = uma.datetime(2026, 4, 24, 2, 0, 0, tzinfo=uma.timezone.utc)
    agent.position_management_session = object()
    agent.pending_entry_order_session = object()
    agent.risk_session = object()
    agent.current_playbook = object()
    agent.current_mode = "context_only"
    agent.current_playbook_reason = "passive_event_trigger"
    agent._audit_event = lambda *args, **kwargs: None
    agent._flatten_unselected_positions = lambda *args, **kwargs: {"results": [], "all_accepted": True}
    agent._cancel_pending_entry_order = lambda reason: setattr(agent, "pending_entry_order_session", None)
    agent._replace_risk_session = lambda session: setattr(agent, "risk_session", session)
    agent.reader = SimpleNamespace(get_all_positions=lambda: {"positions": []})

    should_run_helper = agent._perform_helper_reset(uma.datetime(2026, 4, 24, 2, 0, 1, tzinfo=uma.timezone.utc))

    assert should_run_helper is True
    assert agent.position_management_session is None
    assert agent.pending_entry_order_session is None
    assert agent.risk_session is None
    assert agent.current_playbook is None
    assert agent.current_mode is None
    assert agent.current_playbook_reason == ""
    assert agent.next_helper_reset_at == uma.datetime(2026, 4, 25, 2, 0, 0, tzinfo=uma.timezone.utc)


def test_perform_helper_reset_keeps_state_when_positions_remain_open(uma):
    agent = object.__new__(uma.UnifiedMarketAgent)
    install_test_helper_reset_profile(uma, agent)
    agent.next_helper_reset_at = uma.datetime(2026, 4, 24, 2, 0, 0, tzinfo=uma.timezone.utc)
    original_pm = object()
    original_pending = object()
    original_risk = object()
    original_playbook = object()
    agent.position_management_session = original_pm
    agent.pending_entry_order_session = original_pending
    agent.risk_session = original_risk
    agent.current_playbook = original_playbook
    agent.current_mode = "context_only"
    agent.current_playbook_reason = "passive_event_trigger"
    agent._audit_event = lambda *args, **kwargs: None
    agent._flatten_unselected_positions = lambda *args, **kwargs: {"results": [{"accepted": False}], "all_accepted": False}
    agent._cancel_pending_entry_order = lambda reason: setattr(agent, "pending_entry_order_session", None)
    agent._replace_risk_session = lambda session: setattr(agent, "risk_session", session)
    agent.reader = SimpleNamespace(
        get_all_positions=lambda: {
            "positions": [
                {
                    "size": 1.0,
                }
            ]
        }
    )

    should_run_helper = agent._perform_helper_reset(uma.datetime(2026, 4, 24, 2, 0, 1, tzinfo=uma.timezone.utc))

    assert should_run_helper is False
    assert agent.position_management_session is original_pm
    assert agent.pending_entry_order_session is original_pending
    assert agent.risk_session is original_risk
    assert agent.current_playbook is original_playbook
    assert agent.current_mode == "context_only"
    assert agent.current_playbook_reason == "passive_event_trigger"
    assert agent.next_helper_reset_at == uma.datetime(2026, 4, 24, 2, 0, 0, tzinfo=uma.timezone.utc)


def test_engine_loads_prior_materially_new_events_from_helper_materiality_tail(tmp_path, uma):
    engine = object.__new__(uma.DiscretionaryLLMEngine)
    engine.helper_materially_new_first_events_path = tmp_path / "helper_materially_new_first_events.jsonl"
    rows = [
        {
            "materially_new_first_events": [
                {
                    "event_timestamp": "2026-04-20T10:00:00Z",
                    "item_id": "brent-1",
                    "source": "reuters",
                    "title": "Brent one",
                    "url": "https://example.com/brent-1",
                },
                {
                    "event_timestamp": "2026-04-20T10:05:00Z",
                    "item_id": "brent-2",
                    "source": "reuters",
                    "title": "Brent two",
                    "url": "https://example.com/brent-2",
                },
                {
                    "event_timestamp": "2026-04-20T10:07:00Z",
                    "item_id": "btc-1",
                    "source": "reuters",
                    "title": "BTC one",
                    "url": "https://example.com/btc-1",
                },
            ]
        },
        {
            "materially_new_first_events": [
                {
                    "event_timestamp": "2026-04-20T10:20:00Z",
                    "item_id": "silver-1",
                    "source": "reuters",
                    "title": "Silver one",
                    "url": "https://example.com/silver-1",
                },
                {
                    "event_timestamp": "2026-04-20T10:25:00Z",
                    "item_id": "brent-3",
                    "source": "reuters",
                    "title": "Brent three",
                    "url": "https://example.com/brent-3",
                },
                {
                    "event_timestamp": "2026-04-20T10:30:00Z",
                    "item_id": "brent-4",
                    "source": "reuters",
                    "title": "Brent four",
                    "url": "https://example.com/brent-4",
                },
            ]
        },
        {
            "materially_new_first_events": [
                {
                    "event_timestamp": "2026-04-20T10:35:00Z",
                    "item_id": "brent-5",
                    "source": "reuters",
                    "title": "Brent five",
                    "url": "https://example.com/brent-5",
                }
            ]
        },
    ]
    with engine.helper_materially_new_first_events_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    recent = engine._load_helper_prior_materially_new_events(max_items=5)

    assert recent == [
        {
            "event_timestamp": "2026-04-20T10:07:00Z",
            "source": "reuters",
            "title": "BTC one",
        },
        {
            "event_timestamp": "2026-04-20T10:20:00Z",
            "source": "reuters",
            "title": "Silver one",
        },
        {
            "event_timestamp": "2026-04-20T10:25:00Z",
            "source": "reuters",
            "title": "Brent three",
        },
        {
            "event_timestamp": "2026-04-20T10:30:00Z",
            "source": "reuters",
            "title": "Brent four",
        },
        {
            "event_timestamp": "2026-04-20T10:35:00Z",
            "source": "reuters",
            "title": "Brent five",
        },
    ]


def test_build_system_prompt_passive_includes_duplicate_trigger_null_rule(uma):
    engine = object.__new__(uma.DiscretionaryLLMEngine)

    prompt = engine._build_passive_event_judge_prompt("context_only")

    assert "use it as the canonical materially-new fact tape for trade_symbol" in prompt
    assert "If trigger_event has no direct effect on trade_symbol" in prompt
    assert "set trigger_event_relevance as duplicate, and set action as no_trade" in prompt


def test_normalize_event_record_uses_seen_at_as_fallback_timestamp(uma):
    normalized = uma.normalize_event_record(
        {
            "source": "coindesk",
            "title": "No source timestamp",
            "seen_at": "2026-04-21T22:46:50.186777Z",
        }
    )

    assert normalized["event_timestamp"] == "2026-04-21T22:46:50Z"
    assert normalized["event_time_source"] == "seen_at"
    assert "ingested_at" not in normalized


def test_refresh_passive_llm_recent_events_from_helper_uses_latest_helper_context(tmp_path, uma):
    engine = object.__new__(uma.DiscretionaryLLMEngine)
    engine.helper_materially_new_first_events_path = tmp_path / "helper_materially_new_first_events.jsonl"
    rows = [
        {
            "materially_new_first_events": [
                {
                    "event_timestamp": "2026-04-20T10:00:00Z",
                    "item_id": "brent-1",
                    "source": "reuters",
                    "title": "Brent one",
                    "url": "https://example.com/brent-1",
                },
                {
                    "event_timestamp": "2026-04-20T10:05:00Z",
                    "item_id": "brent-2",
                    "source": "reuters",
                    "title": "Brent two",
                    "url": "https://example.com/brent-2",
                },
            ]
        }
    ]
    with engine.helper_materially_new_first_events_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    agent = object.__new__(uma.UnifiedMarketAgent)
    agent.engine = engine
    agent.symbol = "BRENTOIL"
    agent.trade_symbol_context = {"display_name": "BRENTOIL-USDC", "execution_symbol": "BRENTOIL"}
    agent.passive_recent_materially_new_event_limit = 2
    agent.passive_llm_recent_events_state_path = tmp_path / "passive_llm_recent_events_state.json"
    agent.passive_llm_recent_events = []
    agent.passive_llm_recent_events_symbol = ""
    agent.passive_llm_recent_events_source = ""

    agent._refresh_passive_llm_recent_events_from_helper()

    assert agent.passive_llm_recent_events_symbol == "BRENTOIL-USDC"
    assert [item["title"] for item in agent.passive_llm_recent_events] == ["Brent one", "Brent two"]
    assert all("source" not in item and "event_timestamp" not in item and "item_id" not in item for item in agent.passive_llm_recent_events)


def test_passive_llm_recent_events_state_round_trips_snapshot(tmp_path, uma):
    state_path = tmp_path / "passive_llm_recent_events_state.json"
    agent = object.__new__(uma.UnifiedMarketAgent)
    agent.passive_recent_materially_new_event_limit = 2
    agent.passive_llm_recent_events_state_path = state_path
    agent.passive_llm_recent_events_symbol = "BRENTOIL-USDC"
    agent.passive_llm_recent_events_source = "runtime_state"
    agent.passive_llm_recent_events = [
        {
            "event_timestamp": "2026-04-20T10:00:00Z",
            "item_id": "brent-1",
            "source": "reuters",
            "title": "Brent one",
            "url": "https://example.com/brent-1",
        },
        {
            "event_timestamp": "2026-04-20T10:05:00Z",
            "item_id": "brent-2",
            "source": "reuters",
            "title": "Brent two",
            "url": "https://example.com/brent-2",
        },
    ]

    agent._persist_passive_llm_recent_events_state()

    restored = object.__new__(uma.UnifiedMarketAgent)
    restored.passive_recent_materially_new_event_limit = 2
    restored.passive_llm_recent_events_state_path = state_path
    restored.passive_llm_recent_events = []
    restored.passive_llm_recent_events_symbol = ""
    restored.passive_llm_recent_events_source = ""

    assert restored._hydrate_passive_llm_recent_events_state() is True
    assert restored.passive_llm_recent_events_symbol == "BRENTOIL-USDC"
    assert restored.passive_llm_recent_events_source == "runtime_state"
    assert [item["title"] for item in restored.passive_llm_recent_events] == ["Brent one", "Brent two"]


def test_append_passive_llm_recent_event_keeps_latest_double_limit(uma):
    agent = object.__new__(uma.UnifiedMarketAgent)
    agent.passive_recent_materially_new_event_limit = 2
    agent.passive_llm_recent_events_state_path = None
    agent.passive_llm_recent_events_symbol = "BRENTOIL-USDC"
    agent.passive_llm_recent_events = [
        {
            "event_timestamp": "2026-04-20T10:00:00Z",
            "item_id": "brent-1",
            "source": "reuters",
            "title": "Brent one",
            "url": "https://example.com/brent-1",
        },
        {
            "event_timestamp": "2026-04-20T10:05:00Z",
            "item_id": "brent-2",
            "source": "reuters",
            "title": "Brent two",
            "url": "https://example.com/brent-2",
        },
        {
            "event_timestamp": "2026-04-20T10:10:00Z",
            "item_id": "brent-3",
            "source": "reuters",
            "title": "Brent three",
            "url": "https://example.com/brent-3",
        },
        {
            "event_timestamp": "2026-04-20T10:15:00Z",
            "item_id": "brent-4",
            "source": "reuters",
            "title": "Brent four",
            "url": "https://example.com/brent-4",
        },
    ]
    agent._refresh_passive_llm_recent_events_from_helper = lambda: None

    agent._append_passive_llm_recent_event(
        {
            "event_timestamp": "2026-04-20T10:20:00Z",
            "item_id": "brent-5",
            "source": "reuters",
            "title": "Brent five",
            "url": "https://example.com/brent-5",
        },
        "BRENTOIL-USDC",
    )

    assert [item["title"] for item in agent.passive_llm_recent_events] == [
        "Brent two",
        "Brent three",
        "Brent four",
        "Brent five",
    ]


def test_recent_relevant_passive_events_reads_symbol_bucket(uma):
    agent = object.__new__(uma.UnifiedMarketAgent)
    agent.llm_relevant_passive_events_by_symbol = {
        "BTC-USDC": deque([
            {"title": "A", "source": "reuters", "event_timestamp": "2026-04-14T00:00:00Z"},
            {"title": "B", "source": "reuters", "event_timestamp": "2026-04-14T00:01:00Z"},
        ], maxlen=10),
        "ETH-USDC": deque([
            {"title": "ETH-A", "source": "reuters", "event_timestamp": "2026-04-14T00:02:00Z"},
        ], maxlen=10),
    }

    recent = agent._recent_relevant_passive_events(symbol="BTC-USDC", max_items=5)

    assert [item["title"] for item in recent] == ["A", "B"]


def test_resolve_recent_passive_event_symbol_does_not_fallback_to_first_candidate(uma):
    agent = object.__new__(uma.UnifiedMarketAgent)
    agent.symbol = ""

    selected = agent._resolve_recent_passive_event_symbol(
        query_symbol="",
        query_trade_symbol_context=None,
        trade_symbol_context=None,
    )

    assert selected == ""


def test_resolve_recent_passive_event_symbol_uses_active_symbol_when_flat(uma):
    agent = object.__new__(uma.UnifiedMarketAgent)
    agent.symbol = "BRENTOIL"
    trade_symbol_context = {"display_name": "BRENTOIL-USDC", "execution_symbol": "BRENTOIL"}

    selected = agent._resolve_recent_passive_event_symbol(
        query_symbol="",
        query_trade_symbol_context=trade_symbol_context,
        trade_symbol_context=trade_symbol_context,
    )

    assert selected == "BRENTOIL-USDC"


def test_resolve_decision_entry_reference_uses_llm_entry_for_add_to_long(uma):
    agent = make_agent_stub(uma)
    decision = uma.ManagementDecision(
        action="add_to_long",
        close_fraction=0.0,
        new_notional_usd=1200.0,
        entry_price=105.0,
        stop_loss_price=104.0,
        planned_max_loss_usd=100.0,
        leverage=10,
        margin_basis_usd=0.0,
    )

    price, source = agent._resolve_decision_entry_reference(
        decision,
        {"entry_price": 100.0, "mid_price": 103.0},
    )

    assert price == pytest.approx(105.0)
    assert source == "llm_entry_price"


def test_resolve_decision_entry_reference_uses_llm_entry_for_add_to_short(uma):
    agent = make_agent_stub(uma)
    decision = uma.ManagementDecision(
        action="add_to_short",
        close_fraction=0.0,
        new_notional_usd=1200.0,
        entry_price=95.0,
        stop_loss_price=96.0,
        planned_max_loss_usd=100.0,
        leverage=10,
        margin_basis_usd=0.0,
    )

    price, source = agent._resolve_decision_entry_reference(
        decision,
        {"entry_price": 100.0, "mid_price": 97.0},
    )

    assert price == pytest.approx(95.0)
    assert source == "llm_entry_price"


def test_convert_entry_decision_same_side_long_uses_adjusted_target_for_add_to(uma):
    agent = make_agent_stub(uma, max_loss=100.0, max_leverage=20)
    decision = uma.StrategyDecision(
        action="long",
        suggested_notional_usd=0.0,
        entry_price=90.0,
        stop_loss_price=85.0,
        planned_margin_used_usd=0.0,
        planned_max_loss_usd=100.0,
        requested_leverage=10,
    )
    position_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 10.0,
        "entry_price": 100.0,
        "mid_price": 100.0,
        "notional_usd": 1000.0,
        "margin_used": 200.0,
        "leverage": 5.0,
    }
    agent._estimate_live_position_target_notional_from_entry = lambda *args, **kwargs: (
        800.0,
        {
            "allowed": True,
            "suggested_notional_usd": 800.0,
            "margin_basis_usd": 100.0,
            "planned_max_loss_usd": 100.0,
        },
    )

    materialized = agent._convert_entry_decision_to_management_decision(decision, position_snapshot, {})

    assert materialized.action == "add_to_long"
    assert materialized.new_notional_usd == pytest.approx(1440.0)


def test_low_liquidity_trading_gate_allows_time_window_only_match(uma):
    agent = make_agent_stub(uma, max_loss=100.0, max_leverage=20)
    profile = uma.InstrumentMarketProfile(
        name="btc-test",
        timezone_name="UTC",
        helper_reset_time=None,
        low_liquidity_windows=(uma.LocalTimeWindow((1, 0, 0), (8, 0, 0)),),
        normal_liquidity_windows=(),
        low_liquidity_weekdays=(5, 6),
        low_liquidity_trade_disabled_weekdays=(6,),
    )
    audit_events = []
    agent._audit_event = lambda event_type, payload=None: audit_events.append((event_type, payload or {}))
    agent._low_liquidity_profile_for_symbol = lambda symbol: profile
    agent._low_liquidity_trade_disabled_profile_for_symbol = lambda symbol: None
    agent._estimate_live_position_target_notional_from_entry = lambda *args, **kwargs: (
        1000.0,
        {
            "allowed": True,
            "suggested_notional_usd": 1000.0,
            "margin_basis_usd": 100.0,
            "max_planned_loss_usd": 100.0,
            "max_leverage": 20,
        },
    )
    decision = make_entry_decision(uma, action="long", stop_loss_price=99000.0)
    position_snapshot = {
        "symbol": "BTC",
        "side": "flat",
        "size": 0.0,
        "entry_price": 0.0,
        "mid_price": 100000.0,
        "notional_usd": 0.0,
        "margin_used": 0.0,
    }

    materialized = agent._convert_entry_decision_to_management_decision(decision, position_snapshot, {})

    assert materialized.action == "long"
    assert materialized.new_notional_usd == pytest.approx(1000.0)
    assert audit_events == []


def test_low_liquidity_trading_gate_blocks_weekday_match(uma):
    agent = make_agent_stub(uma, max_loss=100.0, max_leverage=20)
    profile = uma.InstrumentMarketProfile(
        name="btc-test",
        timezone_name="UTC",
        helper_reset_time=None,
        low_liquidity_windows=(),
        normal_liquidity_windows=(),
        low_liquidity_weekdays=(5, 6),
        low_liquidity_trade_disabled_weekdays=(6,),
    )
    audit_events = []
    agent._audit_event = lambda event_type, payload=None: audit_events.append((event_type, payload or {}))
    agent._low_liquidity_trade_disabled_profile_for_symbol = lambda symbol: profile
    agent._estimate_live_position_target_notional_from_entry = lambda *args, **kwargs: (
        1000.0,
        {
            "allowed": True,
            "suggested_notional_usd": 1000.0,
            "margin_basis_usd": 100.0,
            "max_planned_loss_usd": 100.0,
            "max_leverage": 20,
        },
    )
    debug_context = {}
    decision = make_entry_decision(uma, action="long", stop_loss_price=99000.0)
    position_snapshot = {
        "symbol": "BTC",
        "side": "flat",
        "size": 0.0,
        "entry_price": 0.0,
        "mid_price": 100000.0,
        "notional_usd": 0.0,
        "margin_used": 0.0,
    }

    materialized = agent._convert_entry_decision_to_management_decision(
        decision,
        position_snapshot,
        {},
        debug_context=debug_context,
    )

    assert materialized.action == "no_change"
    assert materialized.new_notional_usd == pytest.approx(0.0)
    assert debug_context["no_change_reason"] == "low_liquidity_trading_disabled"
    assert debug_context["low_liquidity_trade_block"]["low_liquidity_source"] == "trade_disabled_weekday"
    assert debug_context["low_liquidity_trade_block"]["low_liquidity_trade_disabled_weekdays"] == [6]
    assert audit_events[0][0] == "low_liquidity_trade_blocked"


def test_convert_entry_decision_same_side_long_caps_adjusted_target_by_original_margin_cap(uma):
    agent = make_agent_stub(uma, max_loss=100.0, max_leverage=20)
    decision = uma.StrategyDecision(
        action="long",
        suggested_notional_usd=0.0,
        entry_price=90.0,
        stop_loss_price=85.0,
        planned_margin_used_usd=0.0,
        planned_max_loss_usd=100.0,
        requested_leverage=10,
    )
    position_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 10.0,
        "entry_price": 100.0,
        "mid_price": 100.0,
        "notional_usd": 1000.0,
        "margin_used": 200.0,
        "leverage": 5.0,
    }
    agent._estimate_live_position_target_notional_from_entry = lambda *args, **kwargs: (
        800.0,
        {
            "allowed": True,
            "suggested_notional_usd": 800.0,
            "margin_basis_usd": 100.0,
            "max_notional_by_margin_usd": 1200.0,
            "planned_max_loss_usd": 100.0,
        },
    )

    materialized = agent._convert_entry_decision_to_management_decision(decision, position_snapshot, {})

    assert materialized.action == "add_to_long"
    assert materialized.new_notional_usd == pytest.approx(1200.0)


def test_convert_entry_decision_same_side_long_uses_adjusted_target_for_no_change(uma):
    agent = make_agent_stub(uma, max_loss=100.0, max_leverage=20)
    decision = uma.StrategyDecision(
        action="long",
        suggested_notional_usd=0.0,
        entry_price=90.0,
        stop_loss_price=85.0,
        planned_margin_used_usd=0.0,
        planned_max_loss_usd=100.0,
        requested_leverage=10,
    )
    position_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 10.0,
        "entry_price": 100.0,
        "mid_price": 100.0,
        "notional_usd": 1000.0,
        "margin_used": 200.0,
        "leverage": 5.0,
    }
    agent._estimate_live_position_target_notional_from_entry = lambda *args, **kwargs: (
        555.0,
        {
            "allowed": True,
            "suggested_notional_usd": 555.0,
            "margin_basis_usd": 100.0,
            "planned_max_loss_usd": 100.0,
        },
    )

    materialized = agent._convert_entry_decision_to_management_decision(decision, position_snapshot, {})

    assert materialized.action == "no_change"
    assert materialized.new_notional_usd == pytest.approx(1000.0)


def test_convert_entry_decision_same_side_long_adjustment_avoids_close_and_trims(uma):
    agent = make_agent_stub(uma, max_loss=100.0, max_leverage=20)
    decision = uma.StrategyDecision(
        action="long",
        suggested_notional_usd=0.0,
        entry_price=90.0,
        stop_loss_price=85.0,
        planned_margin_used_usd=0.0,
        planned_max_loss_usd=100.0,
        requested_leverage=10,
    )
    position_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 10.0,
        "entry_price": 100.0,
        "mid_price": 100.0,
        "notional_usd": 1000.0,
        "margin_used": 200.0,
        "leverage": 5.0,
    }
    agent._estimate_live_position_target_notional_from_entry = lambda *args, **kwargs: (
        0.8,
        {
            "allowed": True,
            "suggested_notional_usd": 0.8,
            "margin_basis_usd": 100.0,
            "planned_max_loss_usd": 100.0,
        },
    )

    materialized = agent._convert_entry_decision_to_management_decision(decision, position_snapshot, {})

    assert materialized.action == "trim"
    assert materialized.new_notional_usd == pytest.approx(1.44)


def test_convert_entry_decision_same_side_trim_within_fraction_tolerance_becomes_no_change(uma):
    agent = make_agent_stub(uma, max_loss=100.0, max_leverage=20)
    agent.local_no_change_close_fraction_tolerance = 0.04
    decision = uma.StrategyDecision(
        action="long",
        suggested_notional_usd=0.0,
        entry_price=100.0,
        stop_loss_price=95.0,
        planned_margin_used_usd=0.0,
        planned_max_loss_usd=100.0,
        requested_leverage=10,
    )
    position_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 10.0,
        "entry_price": 100.0,
        "mid_price": 100.0,
        "notional_usd": 1000.0,
        "margin_used": 200.0,
        "leverage": 5.0,
    }
    agent._estimate_live_position_target_notional_from_entry = lambda *args, **kwargs: (
        970.0,
        {
            "allowed": True,
            "suggested_notional_usd": 970.0,
            "margin_basis_usd": 100.0,
            "planned_max_loss_usd": 100.0,
        },
    )

    materialized = agent._convert_entry_decision_to_management_decision(decision, position_snapshot, {})

    assert materialized.action == "no_change"
    assert materialized.close_fraction == pytest.approx(0.0)
    assert materialized.new_notional_usd == pytest.approx(1000.0)


def test_materialize_live_position_management_non_refresh_no_change_uses_clean_noop_view(monkeypatch, uma):
    monkeypatch.setenv("TRIGGER_CONFIDENCE_RELEVANCE_THRESHOLD_BTC_USDC", "0.46")
    monkeypatch.setenv("TRIGGER_CONFIDENCE_FULL_SCALE_BTC_USDC", "0.52")
    agent = make_agent_stub(uma, max_loss=100.0, max_leverage=20)
    agent.symbol = "BTC-USDC"
    current_plan = uma.PositionManagementPlan(
        execute_now=False,
        action_decision=uma.ManagementDecision(
            action="no_change",
            close_fraction=0.0,
            new_notional_usd=1000.0,
            entry_price=100.0,
            planned_max_loss_usd=0.0,
            leverage=7,
            stop_loss_price=95.0,
        ),
        scenario=None,
    )
    agent.risk_session = SimpleNamespace(position_management=current_plan)
    agent._audit_event = lambda *args, **kwargs: None
    playbook = uma.GenericPlaybook(
        display_answer="display",
        current_bias="neutral",
        selected_symbol="BTC-USDC",
        selection_reason="btc",
        entry_plan=uma.EntryPlan(
            execute_now=True,
            action_decision=uma.StrategyDecision(
                action="long",
                suggested_notional_usd=0.0,
                entry_price=101.0,
                stop_loss_price=90.0,
                planned_margin_used_usd=0.0,
                planned_max_loss_usd=100.0,
                requested_leverage=16,
            ),
            scenario=None,
        ),
    )
    position_snapshot = {
        "symbol": "BTC-USDC",
        "side": "long",
        "size": 10.0,
        "entry_price": 100.0,
        "mid_price": 100.0,
        "notional_usd": 1000.0,
        "margin_used": 200.0,
        "leverage": 7.0,
    }
    playbook.trigger_confidence_raw = 0.31
    agent._estimate_live_position_target_notional_from_entry = lambda *args, **kwargs: (
        1200.0,
        {
            "allowed": True,
            "suggested_notional_usd": 1200.0,
            "margin_basis_usd": 100.0,
            "planned_max_loss_usd": 100.0,
        },
    )

    materialized = agent._materialize_live_position_management_from_entry_plan(playbook, position_snapshot, {})

    assert materialized.position_management.action_decision.action == "no_change"
    assert materialized.position_management.action_decision.leverage == 0
    assert materialized.position_management.action_decision.stop_loss_price == pytest.approx(0.0)
    assert materialized.position_management.action_decision.planned_max_loss_usd == pytest.approx(0.0)


def test_materialize_live_position_management_non_refresh_no_change_does_not_fallback_to_new_plan_tpsl(monkeypatch, uma):
    monkeypatch.setenv("TRIGGER_CONFIDENCE_RELEVANCE_THRESHOLD_BTC_USDC", "0.46")
    monkeypatch.setenv("TRIGGER_CONFIDENCE_FULL_SCALE_BTC_USDC", "0.52")
    agent = make_agent_stub(uma, max_loss=100.0, max_leverage=20)
    agent.symbol = "BTC-USDC"
    agent._audit_event = lambda *args, **kwargs: None
    playbook = uma.GenericPlaybook(
        display_answer="display",
        current_bias="neutral",
        selected_symbol="BTC-USDC",
        selection_reason="btc",
        entry_plan=uma.EntryPlan(
            execute_now=True,
            action_decision=uma.StrategyDecision(
                action="long",
                suggested_notional_usd=0.0,
                entry_price=101.0,
                stop_loss_price=90.0,
                planned_margin_used_usd=0.0,
                planned_max_loss_usd=100.0,
                requested_leverage=16,
            ),
            scenario=None,
        ),
    )
    playbook.position_management = uma.PositionManagementPlan(
        execute_now=False,
        action_decision=uma.ManagementDecision(
            action="no_change",
            close_fraction=0.0,
            new_notional_usd=1000.0,
            entry_price=101.0,
            stop_loss_price=90.0,
            planned_max_loss_usd=0.0,
            leverage=16,
        ),
        scenario=None,
    )
    position_snapshot = {
        "symbol": "BTC-USDC",
        "side": "long",
        "size": 10.0,
        "entry_price": 100.0,
        "mid_price": 100.0,
        "notional_usd": 1000.0,
        "margin_used": 200.0,
        "leverage": 7.0,
    }
    playbook.trigger_confidence_raw = 0.31
    agent._estimate_live_position_target_notional_from_entry = lambda *args, **kwargs: (
        1200.0,
        {
            "allowed": True,
            "suggested_notional_usd": 1200.0,
            "margin_basis_usd": 100.0,
            "planned_max_loss_usd": 100.0,
        },
    )

    materialized = agent._materialize_live_position_management_from_entry_plan(playbook, position_snapshot, {})

    assert materialized.position_management.action_decision.action == "no_change"
    assert materialized.position_management.action_decision.leverage == 0
    assert materialized.position_management.action_decision.stop_loss_price == pytest.approx(0.0)


def test_extract_open_order_tpsl_uses_exact_exchange_order_type_strings(uma):
    assert uma.UnifiedMarketAgent._extract_open_order_tpsl({"orderType": "Take Profit Market"}) == "tp"
    assert uma.UnifiedMarketAgent._extract_open_order_tpsl({"orderType": "Stop Market"}) == "sl"


def test_convert_entry_decision_same_side_short_uses_adjusted_target_for_add_to(uma):
    agent = make_agent_stub(uma, max_loss=100.0, max_leverage=20)
    decision = uma.StrategyDecision(
        action="short",
        suggested_notional_usd=0.0,
        entry_price=110.0,
        stop_loss_price=115.0,
        planned_margin_used_usd=0.0,
        planned_max_loss_usd=100.0,
        requested_leverage=10,
    )
    position_snapshot = {
        "symbol": "BTC",
        "side": "short",
        "size": -10.0,
        "entry_price": 100.0,
        "mid_price": 100.0,
        "notional_usd": 1000.0,
        "margin_used": 200.0,
        "leverage": 5.0,
    }
    agent._estimate_live_position_target_notional_from_entry = lambda *args, **kwargs: (
        800.0,
        {
            "allowed": True,
            "suggested_notional_usd": 800.0,
            "margin_basis_usd": 100.0,
            "planned_max_loss_usd": 100.0,
        },
    )

    materialized = agent._convert_entry_decision_to_management_decision(decision, position_snapshot, {})

    assert materialized.action == "add_to_short"
    assert materialized.new_notional_usd == pytest.approx(1440.0)


def test_convert_entry_decision_same_side_target_is_confidence_scaled_before_materialization(monkeypatch, uma):
    monkeypatch.setenv("TRIGGER_CONFIDENCE_RELEVANCE_THRESHOLD_BTC_USDC", "0.0")
    monkeypatch.setenv("TRIGGER_CONFIDENCE_FULL_SCALE_BTC_USDC", "1.0")
    agent = make_agent_stub(uma, max_loss=100.0, max_leverage=20)
    agent.symbol = "BTC-USDC"
    agent._adjust_same_side_target_notional_for_comparison = lambda **kwargs: 1600.0
    decision = uma.StrategyDecision(
        action="long",
        suggested_notional_usd=0.0,
        entry_price=90.0,
        stop_loss_price=85.0,
        planned_margin_used_usd=0.0,
        planned_max_loss_usd=100.0,
        requested_leverage=10,
    )
    position_snapshot = {
        "symbol": "BTC-USDC",
        "side": "long",
        "size": 10.0,
        "entry_price": 100.0,
        "mid_price": 100.0,
        "notional_usd": 1000.0,
        "margin_used": 200.0,
        "leverage": 5.0,
    }
    agent._estimate_live_position_target_notional_from_entry = lambda *args, **kwargs: (
        1600.0,
        {
            "allowed": True,
            "suggested_notional_usd": 1600.0,
            "margin_basis_usd": 100.0,
            "planned_max_loss_usd": 100.0,
        },
    )

    materialized = agent._convert_entry_decision_to_management_decision(
        decision,
        position_snapshot,
        {},
        trigger_confidence_raw=0.25,
    )

    assert materialized.action == "add_to_long"
    assert materialized.new_notional_usd == pytest.approx(1150.0)


def test_convert_entry_decision_opposite_signal_can_trim_with_zero_normalized_confidence(monkeypatch, uma):
    monkeypatch.setenv("TRIGGER_CONFIDENCE_RELEVANCE_THRESHOLD_BTC_USDC", "0.46")
    monkeypatch.setenv("TRIGGER_CONFIDENCE_FULL_SCALE_BTC_USDC", "0.52")
    monkeypatch.setenv("OPPOSITE_EVENT_TRIM_THRESHOLD_BTC_USDC", "0.40")
    monkeypatch.setenv("OPPOSITE_EVENT_FULL_CONFIDENCE_BTC_USDC", "0.85")
    agent = make_agent_stub(uma, max_loss=100.0, max_leverage=20)
    agent.symbol = "BTC-USDC"
    agent.position_basis_side = "short"
    agent.position_basis_confidence_raw = 0.90
    agent.position_basis_validity = 1.0
    decision = uma.StrategyDecision(
        action="long",
        suggested_notional_usd=0.0,
        entry_price=90.0,
        stop_loss_price=85.0,
        planned_margin_used_usd=0.0,
        planned_max_loss_usd=100.0,
        requested_leverage=10,
    )
    position_snapshot = {
        "symbol": "BTC-USDC",
        "side": "short",
        "size": -10.0,
        "entry_price": 100.0,
        "mid_price": 100.0,
        "notional_usd": 1000.0,
        "margin_used": 200.0,
        "leverage": 5.0,
    }
    agent._estimate_live_position_target_notional_from_entry = lambda *args, **kwargs: (
        1200.0,
        {
            "allowed": True,
            "suggested_notional_usd": 1200.0,
            "margin_basis_usd": 100.0,
            "planned_max_loss_usd": 100.0,
        },
    )

    materialized = agent._convert_entry_decision_to_management_decision(
        decision,
        position_snapshot,
        {},
        allow_immediate_reverse=True,
        trigger_confidence_raw=0.44,
    )

    assert materialized.action == "trim"
    assert materialized.close_fraction == pytest.approx(0.1355555556)
    assert materialized.new_notional_usd == pytest.approx(864.4444444)
    assert getattr(materialized, "opposite_event_decision")["decision"] == "trim"


def test_convert_entry_decision_opposite_signal_below_trim_threshold_is_no_change(monkeypatch, uma):
    monkeypatch.setenv("TRIGGER_CONFIDENCE_RELEVANCE_THRESHOLD_BTC_USDC", "0.0")
    monkeypatch.setenv("TRIGGER_CONFIDENCE_FULL_SCALE_BTC_USDC", "1.0")
    monkeypatch.setenv("OPPOSITE_EVENT_TRIM_THRESHOLD_BTC_USDC", "0.40")
    agent = make_agent_stub(uma, max_loss=100.0, max_leverage=20)
    agent.symbol = "BTC-USDC"
    agent.position_basis_side = "short"
    agent.position_basis_confidence_raw = 0.90
    agent.position_basis_validity = 1.0
    decision = uma.StrategyDecision(
        action="long",
        suggested_notional_usd=0.0,
        entry_price=90.0,
        stop_loss_price=85.0,
        planned_margin_used_usd=0.0,
        planned_max_loss_usd=100.0,
        requested_leverage=10,
    )
    position_snapshot = {
        "symbol": "BTC-USDC",
        "side": "short",
        "size": -10.0,
        "entry_price": 100.0,
        "mid_price": 100.0,
        "notional_usd": 1000.0,
        "margin_used": 200.0,
        "leverage": 5.0,
    }
    agent._estimate_live_position_target_notional_from_entry = lambda *args, **kwargs: (
        1200.0,
        {
            "allowed": True,
            "suggested_notional_usd": 1200.0,
            "margin_basis_usd": 100.0,
            "planned_max_loss_usd": 100.0,
        },
    )

    materialized = agent._convert_entry_decision_to_management_decision(
        decision,
        position_snapshot,
        {},
        allow_immediate_reverse=True,
        trigger_confidence_raw=0.25,
    )

    assert materialized.action == "no_change"
    assert materialized.new_notional_usd == pytest.approx(1000.0)


def test_same_side_add_establishes_missing_position_basis(uma):
    agent = make_agent_stub(uma, max_loss=100.0, max_leverage=20)
    agent.symbol = "BTC-USDC"
    decision = uma.ManagementDecision(
        action="add_to_long",
        close_fraction=0.0,
        new_notional_usd=1500.0,
        entry_price=100.0,
        stop_loss_price=95.0,
        planned_max_loss_usd=100.0,
        leverage=10,
        margin_basis_usd=150.0,
    )
    update = agent._update_position_basis_after_management_execution(
        decision=decision,
        trigger_confidence_raw=0.62,
        position_before={"symbol": "BTC-USDC", "side": "long", "size": 10.0, "notional_usd": 1000.0},
        position_after={"symbol": "BTC-USDC", "side": "long", "size": 15.0, "notional_usd": 1500.0},
        accepted=True,
    )

    assert update["position_basis_confidence_raw"] == pytest.approx(0.62)
    assert update["position_basis_validity"] == pytest.approx(1.0)
    assert agent.position_basis_confidence_raw == pytest.approx(0.62)
    assert agent.position_basis_validity == pytest.approx(1.0)


def test_execute_position_target_does_not_rescale_same_side_target_by_confidence(uma):
    executor = object.__new__(uma.HyperliquidExecutor)
    executor.symbol = "BTC"
    executor.enabled = False
    executor.slippage = 0.01
    executor.reader = SimpleNamespace(
        get_position_snapshot=lambda symbol: {
            "symbol": "BTC",
            "side": "long",
            "size": 0.1,
            "entry_price": 70000.0,
            "mid_price": 75000.0,
            "notional_usd": 1000.0,
        },
        get_mid_price=lambda symbol: 75000.0,
    )
    executor.apply_requested_leverage = lambda leverage: {"requested": leverage}
    executor._position_result = lambda snapshot, plan_name, reason: {
        "mode": "dry_run",
        "symbol": "BTC",
        "plan_name": plan_name,
        "reason": reason,
        "position_before": snapshot,
        "actions": [],
    }
    executor._round_size_to_precision = lambda qty: qty
    executor.usd_to_size = lambda usd, mid: usd / mid

    result = executor.execute_position_target(
        target_side="long",
        target_notional_usd=1600.0,
        requested_leverage=8,
        reason="test_same_side_confidence",
    )

    assert result["requested_target_notional_usd"] == pytest.approx(1600.0)
    assert result["target_notional_usd"] == pytest.approx(1600.0)
    assert result["open_notional_usd"] == pytest.approx(600.0)


def test_execute_position_target_does_not_rescale_reverse_target_by_confidence(uma):
    executor = object.__new__(uma.HyperliquidExecutor)
    executor.symbol = "BTC"
    executor.enabled = False
    executor.slippage = 0.01
    executor.reader = SimpleNamespace(
        get_position_snapshot=lambda symbol: {
            "symbol": "BTC",
            "side": "short",
            "size": -0.1,
            "entry_price": 70000.0,
            "mid_price": 75000.0,
            "notional_usd": 1000.0,
        },
        get_mid_price=lambda symbol: 75000.0,
    )
    executor.apply_requested_leverage = lambda leverage: {"requested": leverage}
    executor._position_result = lambda snapshot, plan_name, reason: {
        "mode": "dry_run",
        "symbol": "BTC",
        "plan_name": plan_name,
        "reason": reason,
        "position_before": snapshot,
        "actions": [],
    }
    executor._round_size_to_precision = lambda qty: qty
    executor.usd_to_size = lambda usd, mid: usd / mid

    result = executor.execute_position_target(
        target_side="long",
        target_notional_usd=2400.0,
        requested_leverage=8,
        reason="test_reverse_confidence",
    )

    assert result["requested_target_notional_usd"] == pytest.approx(2400.0)
    assert result["target_notional_usd"] == pytest.approx(2400.0)
    assert result["reverse_order_notional_usd"] == pytest.approx(3400.0)


def test_execute_management_trim_passes_requested_leverage_into_target_adjustment(uma):
    executor = object.__new__(uma.HyperliquidExecutor)
    executor.symbol = "BTC"
    executor.enabled = False
    executor.slippage = 0.01
    executor.reader = SimpleNamespace(
        get_position_snapshot=lambda symbol: {
            "symbol": "BTC",
            "side": "long",
            "size": 0.1,
            "entry_price": 70000.0,
            "mid_price": 75000.0,
            "notional_usd": 400.0,
        },
        get_mid_price=lambda symbol: 75000.0,
    )
    seen = {}

    def fake_execute_position_target(**kwargs):
        seen.update(kwargs)
        return {
            "mode": "dry_run",
            "symbol": "BTC",
            "plan_name": kwargs.get("plan_name"),
            "reason": kwargs.get("reason"),
            "position_before": executor.reader.get_position_snapshot("BTC"),
            "actions": [],
        }

    executor.execute_position_target = fake_execute_position_target
    decision = uma.ManagementDecision(
        action="trim",
        close_fraction=0.25,
        new_notional_usd=300.0,
        entry_price=75000.0,
        stop_loss_price=74000.0,
        planned_max_loss_usd=50.0,
        leverage=15,
        margin_basis_usd=20.0,
        continue_entry_plan_after_close=False,
    )

    result = executor.execute_management(decision, plan_name="position_management", trigger_confidence_raw=0.64)

    assert result["reason"] == "management_trim"
    assert seen["target_side"] == "long"
    assert seen["target_notional_usd"] == pytest.approx(300.0)
    assert seen["requested_leverage"] == 15
    assert result["trim_close_fraction"] == pytest.approx(0.25)


def test_reconcile_requested_leverage_after_execution_adds_isolated_margin_before_lowering_leverage(uma):
    executor = object.__new__(uma.HyperliquidExecutor)
    executor.symbol = "xyz:BRENTOIL"
    executor.enabled = True
    executor.slippage = 0.01
    snapshots = [
        {
            "symbol": "xyz:BRENTOIL",
            "side": "long",
            "size": 2.78,
            "entry_price": 90.77,
            "mid_price": 90.31,
            "notional_usd": 251.05763,
            "leverage": 20.0,
            "max_leverage": 20,
            "only_isolated": True,
            "margin_used": 15.90,
        },
        {
            "symbol": "xyz:BRENTOIL",
            "side": "long",
            "size": 2.78,
            "entry_price": 90.77,
            "mid_price": 90.31,
            "notional_usd": 251.05763,
            "leverage": 16.0,
            "max_leverage": 20,
            "only_isolated": True,
            "margin_used": 15.90,
        },
    ]

    def next_snapshot(symbol):
        if len(snapshots) > 1:
            return dict(snapshots.pop(0))
        return dict(snapshots[0])

    exchange_calls = {"margin": [], "leverage": []}

    def fake_update_isolated_margin(amount, name):
        exchange_calls["margin"].append((amount, name))
        return {"status": "ok"}

    def fake_update_leverage(leverage, name, is_cross=True):
        exchange_calls["leverage"].append((leverage, name, is_cross))
        return {"status": "ok"}

    executor.reader = SimpleNamespace(
        get_position_snapshot=next_snapshot,
        get_market_spec=lambda symbol: {"max_leverage": 20, "only_isolated": True},
    )
    executor._exchange = SimpleNamespace(
        update_isolated_margin=fake_update_isolated_margin,
        update_leverage=fake_update_leverage,
    )
    executor._ensure_exchange = lambda: None

    initial_snapshot = {
        "symbol": "xyz:BRENTOIL",
        "side": "long",
        "size": 2.78,
        "entry_price": 90.77,
        "mid_price": 90.31,
        "notional_usd": 251.05763,
        "leverage": 20.0,
        "max_leverage": 20,
        "only_isolated": True,
        "margin_used": 11.362321,
    }

    result = executor.reconcile_requested_leverage_after_execution(initial_snapshot, 16)

    assert "isolated_margin_update" in result
    assert result["isolated_margin_update"]["requested_amount_usd"] > 4.0
    assert exchange_calls["margin"] and exchange_calls["margin"][0][1] == "xyz:BRENTOIL"
    assert exchange_calls["leverage"] == [(16, "xyz:BRENTOIL", False)]
    assert result["position_after"]["leverage"] == pytest.approx(16.0)


def test_add_isolated_margin_if_needed_quantizes_exchange_amount_to_6dp(uma):
    executor = object.__new__(uma.HyperliquidExecutor)
    executor.symbol = "xyz:BRENTOIL"
    executor.enabled = True

    exchange_calls = []

    def fake_update_isolated_margin(amount, name):
        exchange_calls.append((amount, name))
        return {"status": "ok"}

    executor._exchange = SimpleNamespace(update_isolated_margin=fake_update_isolated_margin)
    executor._ensure_exchange = lambda: None
    executor._estimate_isolated_margin_top_up_usd = lambda snapshot, requested_leverage: 3.224202399999999

    result = executor._add_isolated_margin_if_needed({"only_isolated": True, "side": "long", "size": 1.0}, 12)

    assert result["raw_requested_amount_usd"] == pytest.approx(3.224202399999999)
    assert result["requested_amount_usd"] == pytest.approx(3.224203)
    assert exchange_calls == [(pytest.approx(3.224203), "xyz:BRENTOIL")]


def test_result_has_exchange_error_detects_status_err(uma):
    executor = object.__new__(uma.HyperliquidExecutor)
    assert executor._result_has_exchange_error({"status": "err", "response": "failure"}) is True


def test_normalize_confidence_value_uses_symbol_specific_env_calibration(monkeypatch, uma):
    monkeypatch.setenv("TRIGGER_CONFIDENCE_RELEVANCE_THRESHOLD_BTC_USDC", "0.30")
    monkeypatch.setenv("TRIGGER_CONFIDENCE_FULL_SCALE_BTC_USDC", "0.80")

    assert uma.normalize_confidence_value(0.25, symbol="BTC-USDC") == pytest.approx(0.0)
    assert uma.normalize_confidence_value(0.55, symbol="BTC-USDC") == pytest.approx(0.5)
    assert uma.normalize_confidence_value(0.90, symbol="BTC-USDC") == pytest.approx(1.0)


def test_convert_entry_decision_same_side_short_uses_adjusted_target_for_no_change(uma):
    agent = make_agent_stub(uma, max_loss=100.0, max_leverage=20)
    decision = uma.StrategyDecision(
        action="short",
        suggested_notional_usd=0.0,
        entry_price=110.0,
        stop_loss_price=115.0,
        planned_margin_used_usd=0.0,
        planned_max_loss_usd=100.0,
        requested_leverage=10,
    )
    position_snapshot = {
        "symbol": "BTC",
        "side": "short",
        "size": -10.0,
        "entry_price": 100.0,
        "mid_price": 100.0,
        "notional_usd": 1000.0,
        "margin_used": 200.0,
        "leverage": 5.0,
    }
    agent._estimate_live_position_target_notional_from_entry = lambda *args, **kwargs: (
        555.0,
        {
            "allowed": True,
            "suggested_notional_usd": 555.0,
            "margin_basis_usd": 100.0,
            "planned_max_loss_usd": 100.0,
        },
    )

    materialized = agent._convert_entry_decision_to_management_decision(decision, position_snapshot, {})

    assert materialized.action == "no_change"
    assert materialized.new_notional_usd == pytest.approx(1000.0)


def test_build_position_management_session_captures_current_playbook_reason(uma):
    agent = make_agent_stub(uma)
    agent.current_playbook_reason = "passive_event_trigger"
    agent.price_history_seconds = 1800
    plan = make_management_plan(
        uma,
        action="short",
        notional=1000.0,
        leverage=10,
        scenario=make_management_scenario(uma, observe_low=99500.0, observe_high=100000.0),
    )
    snapshot = {"side": "flat", "size": 0.0, "entry_price": 0.0, "mid_price": 99900.0, "notional_usd": 0.0}

    session = agent._build_position_management_session(plan, snapshot, "position_management")

    assert session is not None
    assert session.playbook_reason == "passive_event_trigger"


def test_execute_management_decision_uses_market_open_for_passive_flat_entry(uma):
    agent = object.__new__(uma.UnifiedMarketAgent)
    agent.symbol = "BTC"
    agent.current_playbook_reason = "active_periodic_refresh"
    decision = make_management_decision(uma, action="long", notional=1250.0, leverage=8)
    management_plan = make_management_plan(uma, action="long", notional=1250.0, leverage=8)
    agent.position_management_session = uma.PositionManagementSession(
        plan_name="position_management",
        playbook_reason="passive_event_trigger",
        position_management=management_plan,
        runtimes={},
    )
    flat_snapshot = {"symbol": "BTC", "side": "flat", "size": 0.0, "entry_price": 0.0, "mid_price": 100000.0, "notional_usd": 0.0}
    calls = []

    def fake_execute_position_target(**kwargs):
        calls.append(kwargs)
        return {
            "mode": "dry_run",
            "symbol": "BTC",
            "plan_name": kwargs.get("plan_name"),
            "position_before": dict(flat_snapshot),
            "actions": [],
            "accepted": True,
            "message": "market path",
        }

    def fake_execute_management(*args, **kwargs):
        raise AssertionError("execute_management should not be used for passive flat PM entries")

    agent.executor = SimpleNamespace(
        enabled=False,
        execute_position_target=fake_execute_position_target,
        execute_management=fake_execute_management,
        _result_has_exchange_error=lambda result: False,
    )
    agent.reader = SimpleNamespace(
        get_position_snapshot=lambda symbol, **kwargs: dict(flat_snapshot),
        get_all_positions=lambda: {"positions": [], "positions_count": 0},
        get_mid_price=lambda symbol: flat_snapshot["mid_price"],
    )
    agent._print_json_block = lambda *args, **kwargs: None
    agent._audit_event = lambda *args, **kwargs: None

    result = agent.execute_management_decision(decision, "position_management", management_plan)

    assert result["accepted"] is True
    assert calls and calls[0]["reason"] == "management_market_open_from_flat_passive"
    assert calls[0]["target_side"] == "long"
    assert calls[0]["target_notional_usd"] == pytest.approx(1250.0)


def test_fee_adjusted_exit_target_price_includes_entry_and_exit_fees(uma):
    agent = make_agent_stub(uma, taker_fee_rate=0.01)

    adjusted = agent._fee_adjusted_exit_target_price(
        side="long",
        entry_price=100.0,
        target_price=110.0,
        include_entry_fee=True,
    )

    assert adjusted == pytest.approx(112.0)


def test_fee_adjusted_exit_target_price_can_include_exit_fee_only(uma):
    agent = make_agent_stub(uma, taker_fee_rate=0.01)

    adjusted = agent._fee_adjusted_exit_target_price(
        side="short",
        entry_price=100.0,
        target_price=90.0,
        include_entry_fee=False,
    )

    assert adjusted == pytest.approx(89.0)


def test_validate_playbook_accepts_singular_scenario(uma):
    playbook = uma.validate_playbook({
        "trigger_event_relevance": "not_applicable",
        "trigger_confidence": None,
        "playbook": {
            "entry_plan": {
                "execute_now": False,
                "action_decision": make_entry_decision(uma, action="long", notional=1200.0, stop_loss_price=99500.0, leverage=12).to_dict(),
                "scenario": {
                    "observe_when_all": {"low": 99500.0, "high": 100000.0},
                    "execute_when_all": {
                        "condition": {"type": "price_between", "level": 0.0, "low": 99500.0, "high": 100000.0, "timer_seconds": 0, "tolerance_bps": 0.0, "min_ratio": 0.0},
                        "timeout_seconds": 30,
                    },
                },
            }
        },
    })

    assert playbook.entry_plan.scenario is not None
    assert playbook.entry_plan.scenario.observe_when_all.high == pytest.approx(100000.0)


def test_playbook_to_dict_emits_singular_scenario(uma):
    playbook = uma.GenericPlaybook(
        display_answer="display",
        current_bias="neutral",
        entry_plan=uma.EntryPlan(
            execute_now=False,
            action_decision=uma.build_empty_strategy_decision(),
            scenario=uma.EntryScenario(
                observe_when_all={"low": 100.0, "high": 100.0},
                execute_when_all={"condition": {"type": "price_ge", "level": 100.0, "low": 0.0, "high": 0.0, "timer_seconds": 0, "tolerance_bps": 0.0, "min_ratio": 0.0}, "timeout_seconds": 30},
            ),
        ),
    )

    serialized = playbook.to_dict()
    assert "scenario" in serialized["entry_plan"]
    assert "when_all" not in serialized["entry_plan"]["scenario"]
    assert "scenarios" not in serialized["entry_plan"]
    assert "decision_rules" not in serialized["entry_plan"]["scenario"]


def test_build_status_summary_for_playbook_selected_uses_singular_scenario(uma):
    payload = {
        "had_open_position": False,
        "execution_view": {
            "entry_plan": {
                "execute_now": False,
                "action_decision": {"action": "long", "entry_price": 68150.0, "stop_loss_price": 67150.0},
                "scenario": {
                    "observe_when_all": {"low": 67950.0, "high": 68250.0},
                    "execute_when_all": {
                        "condition": {"type": "sustained_ge", "level": 68120.0, "timer_seconds": 45, "min_ratio": 0.8},
                        "timeout_seconds": 900,
                    },
                },
            },
        },
    }

    summary = uma.build_status_summary("playbook_selected", payload)
    assert summary is not None
    assert "67950-68250" in summary["summary"]
    assert "执行 >=68120 (45s, ratio>=0.80)" in summary["summary"]
    assert "动作 long" in summary["summary"]


def test_status_execution_result_brief_does_not_invent_flat_position(uma):
    assert uma._status_execution_result_brief({"mode": "live", "close_size": 4.57}) == "live"
    assert uma._status_execution_result_brief({"mode": "live", "position_after": {"side": "flat", "size": 0.0}}) == "live | position flat"


def test_build_status_summary_for_management_scenario_stop_loss_uses_singular_rule(uma):
    payload = {
        "had_open_position": True,
        "execution_view": {
            "selected_symbol": "BRENTOIL-USDC",
            "position_management": {
                "execute_now": False,
                "action_decision": make_management_decision(uma, action="add_to_long", notional=7224.29, leverage=20, stop_loss_price=109.9).to_dict(),
                "take_profit_legs": [],
                "stop_loss_legs": [],
                "scenario": {
                    "observe_when_all": {"low": 110.4, "high": 110.6},
                    "execute_when_all": {
                        "condition": {"type": "sustained_ge", "level": 110.6, "low": 0.0, "high": 0.0, "timer_seconds": 60, "tolerance_bps": 0.0, "min_ratio": 0.0},
                        "timeout_seconds": 30,
                    },
                },
            },
        },
    }

    summary = uma.build_status_summary("playbook_selected", payload)
    assert summary is not None
    assert "sl 109.9" in summary["summary"]


def test_materialize_live_position_management_promotes_singular_close_scenario_to_immediate(uma):
    agent = make_agent_stub(uma, max_loss=100.0, max_leverage=20)
    playbook = uma.GenericPlaybook(
        display_answer="display",
        current_bias="bearish",
        selected_symbol="BTC",
        selection_reason="btc",
        entry_plan=uma.EntryPlan(
            execute_now=False,
            action_decision=make_entry_decision(uma, action="short", notional=800.0, stop_loss_price=101000.0, leverage=0),
            scenario=uma.EntryScenario(
                observe_when_all={"low": 100000.0, "high": 100000.0},
                execute_when_all={"condition": {"type": "price_le", "level": 100000.0, "low": 0.0, "high": 0.0, "timer_seconds": 0, "tolerance_bps": 0.0, "min_ratio": 0.0}, "timeout_seconds": 30},
            ),
        ),
    )
    position_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 1.0,
        "entry_price": 100000.0,
        "mid_price": 100500.0,
        "notional_usd": 1000.0,
        "margin_used": 200.0,
    }
    agent._estimate_live_position_target_notional_from_entry = lambda *args, **kwargs: (
        800.0,
        {
            "allowed": True,
            "suggested_notional_usd": 800.0,
            "requested_leverage": 4,
            "planned_margin_used_usd": 200.0,
            "planned_max_loss_usd": 100.0,
        },
    )

    agent._estimate_live_position_target_notional_from_entry = lambda *args, **kwargs: (
        0.0,
        {
            "allowed": True,
            "suggested_notional_usd": 0.0,
            "requested_leverage": 4,
            "planned_margin_used_usd": 0.0,
            "planned_max_loss_usd": 0.0,
        },
    )

    materialized = agent._materialize_live_position_management_from_entry_plan(playbook, position_snapshot, {})

    assert materialized.position_management.execute_now is True
    assert materialized.position_management.action_decision.action == "close"
    assert materialized.position_management.action_decision.continue_entry_plan_after_close is True
    assert materialized.position_management.scenario is None


def test_materialize_live_position_management_immediate_same_side_short_close_does_not_continue(uma):
    agent = make_agent_stub(uma, max_loss=100.0, max_leverage=20)
    playbook = uma.GenericPlaybook(
        display_answer="display",
        current_bias="bearish",
        selected_symbol="BTC",
        selection_reason="btc",
        entry_plan=uma.EntryPlan(
            execute_now=True,
            action_decision=make_entry_decision(uma, action="short", notional=0.0, stop_loss_price=101000.0, leverage=0),
            scenario=None,
        ),
    )
    position_snapshot = {
        "symbol": "BTC",
        "side": "short",
        "size": -1.0,
        "entry_price": 100000.0,
        "mid_price": 100500.0,
        "notional_usd": 1000.0,
        "margin_used": 200.0,
    }
    agent._estimate_live_position_target_notional_from_entry = lambda *args, **kwargs: (
        0.0,
        {
            "allowed": True,
            "suggested_notional_usd": 0.0,
            "requested_leverage": 4,
            "planned_margin_used_usd": 0.0,
            "planned_max_loss_usd": 0.0,
        },
    )

    materialized = agent._materialize_live_position_management_from_entry_plan(playbook, position_snapshot, {})

    assert materialized.position_management.execute_now is True
    assert materialized.position_management.action_decision.action == "close"
    assert materialized.position_management.action_decision.continue_entry_plan_after_close is False
    assert materialized.position_management.scenario is None


def test_execute_immediate_playbook_action_refreshing_no_change_sets_risk_session_only_after_leverage_matches(uma):
    agent = make_agent_stub(uma)
    agent.symbol = "BTC"
    agent._audit_event = lambda *args, **kwargs: None
    agent._schedule_next_active_query = lambda *args, **kwargs: None
    agent._arm_follow_up_plan_for_current_state = lambda *args, **kwargs: None
    risk_calls = []
    agent._set_risk_session_after_management_decision = lambda *args, **kwargs: risk_calls.append((args, kwargs))
    tp_leg = uma.ExitLeg(
        name="tp",
        note="",
        when_all=[uma.Condition(type="price_ge", level=110.0)],
        close_fraction=1.0,
    )
    sl_leg = uma.ExitLeg(
        name="sl",
        note="",
        when_all=[uma.Condition(type="cross_below", level=95.0)],
        close_fraction=1.0,
    )
    playbook = uma.GenericPlaybook(
        display_answer="display",
        current_bias="neutral",
        selected_symbol="BTC",
        selection_reason="btc",
        position_management=uma.PositionManagementPlan(
            execute_now=True,
            action_decision=uma.ManagementDecision(
                action="no_change",
                close_fraction=0.0,
                new_notional_usd=1000.0,
                entry_price=100.0,
                planned_max_loss_usd=0.0,
                leverage=16,
                stop_loss_price=95.0,
            ),
            scenario=None,
        ),
    )
    playbook.post_fill_risk_template = uma.build_empty_position_management_plan()
    playbook.trigger_confidence_raw = 0.74
    agent.execute_management_decision = lambda *args, **kwargs: {
        "decision": playbook.position_management.action_decision.to_dict(),
        "position_before": {"symbol": "BTC", "side": "long", "size": 3.0, "leverage": 20.0},
        "position_after": {"symbol": "BTC", "side": "long", "size": 3.0, "leverage": 16.0},
        "accepted": True,
    }

    agent._execute_immediate_playbook_action(playbook, {"symbol": "BTC", "side": "long", "size": 3.0, "leverage": 20.0})

    assert len(risk_calls) == 1

    risk_calls.clear()
    agent.execute_management_decision = lambda *args, **kwargs: {
        "decision": playbook.position_management.action_decision.to_dict(),
        "position_before": {"symbol": "BTC", "side": "long", "size": 3.0, "leverage": 20.0},
        "position_after": {"symbol": "BTC", "side": "long", "size": 3.0, "leverage": 20.0},
        "accepted": True,
    }

    agent._execute_immediate_playbook_action(playbook, {"symbol": "BTC", "side": "long", "size": 3.0, "leverage": 20.0})

    assert risk_calls == []


def test_execute_immediate_playbook_action_trim_requires_real_size_reduction_before_refreshing_risk_session(uma):
    agent = make_agent_stub(uma)
    agent.symbol = "BTC"
    agent._audit_event = lambda *args, **kwargs: None
    agent._schedule_next_active_query = lambda *args, **kwargs: None
    agent._arm_follow_up_plan_for_current_state = lambda *args, **kwargs: None
    risk_calls = []
    agent._set_risk_session_after_management_decision = lambda *args, **kwargs: risk_calls.append((args, kwargs))
    playbook = uma.GenericPlaybook(
        display_answer="display",
        current_bias="neutral",
        selected_symbol="BTC",
        selection_reason="btc",
        position_management=uma.PositionManagementPlan(
            execute_now=True,
            action_decision=uma.ManagementDecision(
                action="trim",
                close_fraction=0.25,
                new_notional_usd=750.0,
                entry_price=100.0,
                planned_max_loss_usd=0.0,
                leverage=10,
            ),
            scenario=None,
        ),
    )
    playbook.post_fill_risk_template = uma.build_empty_position_management_plan()
    playbook.trigger_confidence_raw = 0.64
    agent.execute_management_decision = lambda *args, **kwargs: {
        "decision": playbook.position_management.action_decision.to_dict(),
        "position_before": {"symbol": "BTC", "side": "long", "size": 4.0, "leverage": 10.0},
        "position_after": {"symbol": "BTC", "side": "long", "size": 4.0, "leverage": 10.0},
        "accepted": True,
    }

    agent._execute_immediate_playbook_action(playbook, {"symbol": "BTC", "side": "long", "size": 4.0, "leverage": 10.0})

    assert risk_calls == []



def test_compare_position_management_plans_keeps_small_observing_delta(uma):
    old_plan = make_management_plan(
        uma,
        action="add_to_long",
        notional=1000.0,
        leverage=10,
        stop_loss_price=99000.0,
        scenario=make_management_scenario(uma, observe_low=99500.0, observe_high=100000.0, trigger_level=100200.0),
    )
    new_plan = make_management_plan(
        uma,
        action="add_to_long",
        notional=1015.0,
        leverage=10,
        stop_loss_price=99020.0,
        scenario=make_management_scenario(uma, observe_low=99520.0, observe_high=100020.0, trigger_level=100220.0),
    )

    result = uma.compare_position_management_plans(old_plan, new_plan)

    assert result["should_replace"] is False
    assert result["hard_reasons"] == []
    assert result["soft_reasons"] == []


def test_set_position_management_session_from_plan_retains_pending_session_when_diff_small(uma):
    agent = make_agent_stub(uma)
    old_plan = make_management_plan(
        uma,
        action="add_to_long",
        notional=1000.0,
        leverage=10,
        stop_loss_price=99000.0,
        scenario=make_management_scenario(uma, observe_low=99500.0, observe_high=100000.0, trigger_level=100200.0),
    )
    new_plan = make_management_plan(
        uma,
        action="add_to_long",
        notional=1015.0,
        leverage=10,
        stop_loss_price=99020.0,
        scenario=make_management_scenario(uma, observe_low=99520.0, observe_high=100020.0, trigger_level=100220.0),
    )
    retained_events = []
    agent._audit_event = lambda event_type, payload=None: retained_events.append((event_type, payload or {}))
    agent.current_playbook = uma.GenericPlaybook(
        display_answer="display",
        current_bias="neutral",
        entry_plan=uma.EntryPlan(execute_now=False, action_decision=uma.build_empty_strategy_decision(), scenario=None),
        position_management=new_plan,
    )
    agent.position_management_session = uma.PositionManagementSession(
        plan_name="position_management",
        side="long",
        position_management=old_plan,
        start_time=0.0,
        baseline_size=1.0,
        expected_size=1.0,
        initial_size_abs=1.0,
        runtimes={uma.SCENARIO_RUNTIME_KEY: uma.ScenarioRuntime()},
        history_seconds=1800,
    )
    build_calls = []
    agent._build_position_management_session = lambda plan, snapshot, plan_name: build_calls.append((plan, snapshot, plan_name))
    snapshot = {"side": "long", "size": 1.0, "entry_price": 100000.0, "mid_price": 100100.0, "notional_usd": 1000.0}

    agent._set_position_management_session_from_plan(new_plan, snapshot, "position_management")

    assert agent.position_management_session.position_management is old_plan
    assert agent.current_playbook.position_management is old_plan
    assert build_calls == []
    assert any(event_type == "position_management_session_retained" for event_type, _ in retained_events)


def test_compare_position_management_plans_replaces_large_semantic_change(uma):
    old_plan = make_management_plan(
        uma,
        action="add_to_long",
        notional=1000.0,
        leverage=10,
        stop_loss_price=99000.0,
        scenario=make_management_scenario(uma),
    )
    new_plan = make_management_plan(
        uma,
        action="add_to_short",
        notional=1000.0,
        leverage=10,
        stop_loss_price=99000.0,
        scenario=make_management_scenario(uma),
    )

    result = uma.compare_position_management_plans(old_plan, new_plan)

    assert result["should_replace"] is True
    assert "action_changed" in result["hard_reasons"]


def test_should_force_news_context_disabled_for_pm_timeout_and_cancelled(uma):
    engine = object.__new__(uma.DiscretionaryLLMEngine)
    engine.force_active_news_context = True
    engine.force_passive_news_context = True

    assert engine._should_force_news_context("management_scenario_timeout", "context_only") is False
    assert engine._should_force_news_context("management_scenario_cancelled", "always") is False


def test_select_playbook_trade_symbol_context_rejects_active_symbol_mismatch(uma):
    agent = object.__new__(uma.UnifiedMarketAgent)
    context = {"trade_symbol_key": "BRENTOIL-USDC", "display_name": "BRENTOIL-USDC", "execution_symbol": "BRENTOIL"}

    assert agent._select_playbook_trade_symbol_context(agent, context, active_symbol="BTC") is None
    assert agent._select_playbook_trade_symbol_context(agent, context, active_symbol="BRENTOIL") == context
