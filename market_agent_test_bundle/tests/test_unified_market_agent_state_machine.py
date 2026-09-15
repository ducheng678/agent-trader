from __future__ import annotations

from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path

import pytest


def make_entry_decision(uma, *, action="no_trade", notional=0.0, stop_loss_price=0.0, leverage=10, entry_price=None):
    return uma.StrategyDecision(
        action=action,
        suggested_notional_usd=notional,
        entry_price=(float(entry_price) if entry_price is not None else 100000.0) if action in {"long", "short"} else 0.0,
        stop_loss_price=stop_loss_price,
        planned_margin_used_usd=0.0,
        planned_max_loss_usd=0.0,
        requested_leverage=leverage,
    )


def make_management_decision(
    uma,
    *,
    action="no_change",
    notional=0.0,
    leverage=10,
    close_fraction=0.0,
    stop_loss_price=0.0,
):
    is_exposure_action = action.startswith("reverse_to_") or action.startswith("add_to_")
    return uma.ManagementDecision(
        action=action,
        close_fraction=close_fraction,
        new_notional_usd=notional,
        entry_price=100000.0 if is_exposure_action else 0.0,
        stop_loss_price=stop_loss_price,
        planned_max_loss_usd=0.0,
        leverage=leverage,
        margin_basis_usd=0.0,
    )


def make_management_plan(
    uma,
    *,
    execute_now=False,
    now_action=None,
    stop_price=99000.0,
    stop_type="price_le",
    scenario=None,
):
    decision = now_action or uma.build_empty_management_decision()
    if stop_price > 0 and float(getattr(decision, "stop_loss_price", 0.0) or 0.0) <= 0.0:
        payload = decision.to_dict()
        payload["continue_entry_plan_after_close"] = bool(getattr(decision, "continue_entry_plan_after_close", False))
        decision = uma.ManagementDecision(
            **{
                **payload,
                "stop_loss_price": stop_price,
            }
        )
    return uma.PositionManagementPlan(
        execute_now=execute_now,
        action_decision=decision,
        scenario=scenario,
    )


def make_playbook(
    uma,
    *,
    entry_execute_now=False,
    entry_action_decision=None,
    entry_scenario=None,
    management_plan=None,
    post_fill_risk_template=None,
    selected_symbol="BTC",
):
    action_decision = entry_action_decision
    scenario = entry_scenario
    if scenario is not None and isinstance(scenario, uma.Scenario):
        scenario = uma.EntryScenario(
            observe_when_all=scenario.observe_when_all,
            execute_when_all={
                "condition": scenario.execute_when_all.condition,
                "timeout_seconds": scenario.execute_when_all.timeout_seconds,
            },
        )
    if action_decision is None:
        action_decision = uma.build_empty_strategy_decision()
    return uma.GenericPlaybook(
        display_answer="display",
        current_bias="neutral",
        selected_symbol=selected_symbol,
        selection_reason=f"{selected_symbol} best",
        entry_plan=uma.EntryPlan(
            execute_now=entry_execute_now,
            action_decision=action_decision,
            scenario=scenario,
        ),
        position_management=management_plan or make_management_plan(uma),
        post_fill_risk_template=post_fill_risk_template or uma.build_empty_position_management_plan(),
    )


class FakeReader:
    def __init__(
        self,
        *,
        all_positions=None,
        symbol_snapshots=None,
        mid_price=100000.0,
        frontend_open_orders=None,
        order_statuses=None,
        user_fills_by_time=None,
        ws_healthy=True,
        candles_by_interval=None,
    ):
        self._all_positions = all_positions or {
            "known": True,
            "account_address": "0xabc",
            "network": "mainnet",
            "margin_summary": {},
            "cross_margin_summary": {},
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
        self.account_address = str(self._all_positions.get("account_address", "0xabc") or "0xabc")
        self._symbol_snapshots = list(
            symbol_snapshots
            or [
                {
                    "known": True,
                    "account_address": "0xabc",
                    "network": "mainnet",
                    "symbol": "BTC",
                    "side": "flat",
                    "size": 0.0,
                    "entry_price": 0.0,
                    "mid_price": mid_price,
                    "notional_usd": 0.0,
                    "leverage": 0.0,
                    "max_leverage": 40,
                    "remaining_capital_usd": self._all_positions["remaining_capital_usd"],
                    "available_margin_usd": self._all_positions["available_margin_usd"],
                    "withdrawable_usd": self._all_positions["withdrawable_usd"],
                    "margin_used": 0.0,
                }
            ]
        )
        self._last_symbol_snapshot = dict(self._symbol_snapshots[-1])
        self._mid_price = mid_price
        self._frontend_open_orders = list(frontend_open_orders or [])
        self._order_statuses = {
            int(key): dict(value)
            for key, value in dict(order_statuses or {}).items()
        }
        self._user_fills_by_time = [dict(item) for item in list(user_fills_by_time or []) if isinstance(item, dict)]
        self._ws_healthy = bool(ws_healthy)
        self._candles_by_interval = {
            str(key): [dict(item) for item in list(value or []) if isinstance(item, dict)]
            for key, value in dict(candles_by_interval or {}).items()
        }
        self.subscribe_calls = []
        self.unsubscribe_calls = []
        self.disconnect_calls = 0
        self.user_fills_queries = []

    def get_all_positions(self):
        return dict(self._all_positions)

    def get_position_snapshot(self, symbol, all_positions=None, current_price=None):
        if self._symbol_snapshots:
            self._last_symbol_snapshot = dict(self._symbol_snapshots.pop(0))
        snap = dict(self._last_symbol_snapshot)
        if current_price is not None:
            snap["mid_price"] = current_price
        return snap

    def get_mid_price(self, symbol):
        return self._mid_price

    def get_frontend_open_orders(self, symbol=None):
        return [dict(item) for item in self._frontend_open_orders]

    def subscribe_user_fills(self, address, callback):
        self.subscribe_calls.append({"address": address, "callback": callback})
        return len(self.subscribe_calls)

    def unsubscribe_user_fills(self, address, subscription_id):
        self.unsubscribe_calls.append({"address": address, "subscription_id": subscription_id})
        return True

    def disconnect_ws(self):
        self.disconnect_calls += 1
        self._ws_healthy = False

    def user_fills_ws_is_healthy(self):
        return self._ws_healthy

    def get_user_fills_by_time(self, address, start_time_ms, end_time_ms=None, *, aggregate_by_time=False):
        self.user_fills_queries.append(
            {
                "address": address,
                "start_time_ms": start_time_ms,
                "end_time_ms": end_time_ms,
                "aggregate_by_time": aggregate_by_time,
            }
        )
        end_value = float('inf') if end_time_ms is None else int(end_time_ms)
        return [
            dict(item)
            for item in self._user_fills_by_time
            if int(item.get("time", 0) or 0) >= int(start_time_ms) and int(item.get("time", 0) or 0) <= end_value
        ]

    def get_info_client(self):
        reader = self

        class InfoClient:
            def query_order_by_oid(self, user, oid):
                return dict(reader._order_statuses.get(int(oid), {}))

        return InfoClient()

    def get_market_spec(self, symbol):
        return {"max_leverage": 40, "only_isolated": False}

    def get_candles_snapshot(self, symbol, interval, start_ms, end_ms):
        rows = [dict(item) for item in list(self._candles_by_interval.get(str(interval), []))]
        return [
            row
            for row in rows
            if int(row.get("t", 0) or 0) >= int(start_ms) and int(row.get("t", 0) or 0) <= int(end_ms)
        ]

    @staticmethod
    def format_all_positions(snapshot):
        return "formatted-all-positions"

    @staticmethod
    def format_symbol_position(snapshot):
        return "formatted-symbol-position"


class FakeEngine:
    def __init__(self, playbook, mode="raw_context_only"):
        self.playbook = playbook
        self.mode = mode
        self.active_reasoning_effort = "high"
        self.passive_reasoning_effort = "medium"
        self.calls = []
        self.last_call_debug = {
            "raw_output_text": "{\"mock\":true}",
            "parsed_output": playbook.to_dict(),
            "validated_playbook": playbook.to_dict(),
            "capped_playbook": playbook.to_dict(),
            "execution_view": playbook.to_dict(),
            "mode": mode,
        }

    def get_playbook(self, **kwargs):
        self.calls.append(kwargs)
        return self.playbook, self.mode

    @staticmethod
    def _get_cached_helper_market_mainline_context(**_kwargs):
        return None, {}


class FakeExecutor:
    def __init__(self, reader, symbol):
        self.reader = reader
        self.symbol = symbol
        self.enabled = False
        self.executions = []
        self.management_executions = []

    def execute(self, decision, plan_name=None, trigger_confidence_raw=None):
        self.executions.append((plan_name, decision.action))
        return {
            "mode": "dry_run",
            "symbol": self.symbol,
            "plan_name": plan_name,
            "decision": decision.to_dict(),
            "actions": [],
        }

    def execute_management(self, decision, plan_name=None, trigger_confidence_raw=None, **_kwargs):
        self.management_executions.append((plan_name, decision.action))
        return {
            "mode": "dry_run",
            "symbol": self.symbol,
            "plan_name": plan_name,
            "decision": decision.to_dict(),
            "actions": [],
        }

    @staticmethod
    def _result_has_exchange_error(result):
        return False

    def close_position(self, side, reason, plan_name=None):
        return {"side": side, "reason": reason, "plan_name": plan_name}


class FakeEvents:
    def __init__(self, recent_events=None):
        self._recent = list(recent_events or [])

    def recent(self):
        return list(self._recent)

    def poll(self):
        return []


def make_agent(uma, *, playbook=None, reader=None, engine=None, executor=None, events=None):
    agent = object.__new__(uma.UnifiedMarketAgent)
    agent.reader = reader or FakeReader()
    agent.trade_candidates = [
        {
            "candidate_key": "BTC",
            "trade_symbol_key": "BTC",
            "display_name": "BTC",
            "display_symbol": "BTC",
            "configured_execution_symbol": "BTC",
            "execution_symbol": "BTC",
            "tradable_on_hyperliquid": True,
        }
    ]
    agent.trade_symbol_context = dict(agent.trade_candidates[0])
    agent.symbol = "BTC"
    agent.engine = engine or FakeEngine(playbook or make_playbook(uma))
    agent.executor = executor or FakeExecutor(agent.reader, "BTC")
    agent.user_query_template = "剩余本金 {remaining_capital_usd}，风险预算 {max_planned_loss_usd}，交易 {symbol}"
    agent.max_planned_loss_usd = 100.0
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
    agent.risk_time_decay_tp_enabled = False
    agent.risk_time_decay_tp_timeframe_seconds = 900.0
    agent.risk_time_decay_normal_tp1_bars = 8.0
    agent.risk_time_decay_normal_tp1_mfe_r = 0.60
    agent.risk_time_decay_normal_tp1_current_r = 0.30
    agent.risk_time_decay_normal_tp2_bars = 12.0
    agent.risk_time_decay_normal_tp2_mfe_r = 1.50
    agent.risk_time_decay_normal_tp2_current_r = 1.00
    agent.risk_time_decay_low_tp1_bars = 18.0
    agent.risk_time_decay_low_tp1_mfe_r = 0.30
    agent.risk_time_decay_low_tp1_current_r = 0.15
    agent.risk_time_decay_low_tp2_bars = 36.0
    agent.risk_time_decay_low_tp2_mfe_r = 0.75
    agent.risk_time_decay_low_tp2_current_r = 0.50
    agent.risk_tp1_no_follow_through_enabled = True
    agent.risk_tp1_no_follow_through_normal_close_fraction = 0.50
    agent.risk_tp1_no_follow_through_normal_soft_stop_r = 0.40
    agent.risk_tp2_no_continuation_enabled = True
    agent.risk_tp2_no_continuation_normal_close_fraction = 0.50
    agent.risk_tp2_no_continuation_normal_soft_stop_r = 0.25
    agent.local_size_from_stop = True
    agent.local_risk_tolerance_usd = 1.0
    agent.loop_sleep_seconds = 0.0
    agent.playbook_poll_seconds = 0.0
    agent.price_history_seconds = 1800
    agent.risk_poll_seconds = 0.0
    agent.position_size_change_tol = 1e-8
    agent.risk_session_state_enabled = False
    agent.risk_session_state_path = Path("/tmp/fake_risk_session_state.json")
    agent.risk_session_restore_fill_lookback_seconds = 21600.0
    agent.loop_exception_sleep_seconds = 5.0
    agent.hyperliquid_transient_error_sleep_seconds = 30.0
    agent.enable_monitor = True
    agent.enable_active_query = True
    agent.enable_active_auto_requery = True
    agent.active_query_interval_seconds = 60.0
    agent.active_management_query_interval_seconds = 120.0
    agent.enable_passive_event_query = False
    agent.fast_replan_delay_seconds = 0.0
    agent.requery_on_playbook_end = True
    agent.events_path = Path("/tmp/fake_events.jsonl")
    agent.start_from = "end"
    agent.event_recent_limit = 20
    agent.event_context_max_items = 5
    agent.passive_event_context_max_items = 10
    agent.passive_recent_materially_new_event_limit = 10
    agent.passive_llm_relevant_event_buffer_size = 50
    agent.passive_relevant_events_log_base_path = Path("/tmp/passive_relevant_events.jsonl")
    agent.passive_relevant_events_log_path = agent.passive_relevant_events_log_base_path
    agent.llm_relevant_passive_events_by_symbol = {}
    agent.events = events or FakeEvents()
    agent.current_playbook = None
    agent.current_mode = None
    agent.current_playbook_reason = ""
    agent.risk_session = None
    agent.pending_entry_order_session = None
    agent.next_active_query_due_at = 10**12
    agent.last_playbook_query_at = None
    agent.last_playbook_tick_at = 0.0
    agent.last_risk_tick_at = 0.0
    agent.enable_user_fills_websocket = False
    agent.user_fills_address = getattr(agent.reader, "account_address", "0xabc")
    agent.user_fills_subscription_id = None
    agent.user_fills_event_buffer = deque()
    agent._user_fills_seen_keys = deque()
    agent._user_fills_seen_key_set = set()
    agent.user_fills_seen_capacity = 4000
    agent.user_fills_reconcile_grace_seconds = 3.0
    agent.user_fills_reconnect_retry_seconds = 10.0
    agent.user_fills_backfill_poll_seconds = 5.0
    agent.user_fills_backfill_lookback_seconds = 120.0
    agent.user_fills_last_subscribe_attempt_at = 0.0
    agent.user_fills_last_message_at = 0.0
    agent.user_fills_last_backfill_at = 0.0
    agent.user_fills_last_fill_time_ms = 0
    return agent


def test_market_catalog_warmup_calls_reader_once(uma):
    class CatalogReader(FakeReader):
        def __init__(self):
            super().__init__()
            self.market_catalog_calls = 0

        def get_market_catalog(self):
            self.market_catalog_calls += 1
            return {"xyz:BRENTOIL": {"sz_decimals": 2}}

    reader = CatalogReader()
    agent = make_agent(uma, reader=reader)
    audit_events = []
    agent._audit_event = lambda event_type, payload=None: audit_events.append((event_type, payload or {}))

    assert agent._warm_up_market_catalog() is True
    assert agent._warm_up_market_catalog() is True

    assert reader.market_catalog_calls == 1
    assert audit_events[0][0] == "market_catalog_warmup"
    assert audit_events[0][1]["markets_count"] == 1


def test_market_catalog_warmup_failure_is_nonfatal_and_not_retried(uma):
    class FailingCatalogReader(FakeReader):
        def __init__(self):
            super().__init__()
            self.market_catalog_calls = 0

        def get_market_catalog(self):
            self.market_catalog_calls += 1
            raise RuntimeError("meta unavailable")

    reader = FailingCatalogReader()
    agent = make_agent(uma, reader=reader)
    audit_events = []
    agent._audit_event = lambda event_type, payload=None: audit_events.append((event_type, payload or {}))

    assert agent._warm_up_market_catalog() is False
    assert agent._warm_up_market_catalog() is False

    assert reader.market_catalog_calls == 1
    assert audit_events[0][0] == "market_catalog_warmup_failed"
    assert "meta unavailable" in audit_events[0][1]["error"]


def make_position_snapshot(
    *,
    side="flat",
    notional=0.0,
    entry_price=100000.0,
    mid_price=100000.0,
    leverage=5.0,
    max_leverage=20,
    margin_used=20.0,
):
    size = 0.0
    if side in {"long", "short"} and entry_price > 0.0:
        size = abs(float(notional or 0.0)) / entry_price
        if side == "short":
            size = -size
    return {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "symbol": "BTC",
        "side": side,
        "size": size,
        "entry_price": entry_price if side in {"long", "short"} else 0.0,
        "mid_price": mid_price,
        "notional_usd": abs(float(notional or 0.0)) if side in {"long", "short"} else 0.0,
        "leverage": leverage if side in {"long", "short"} else 0.0,
        "max_leverage": max_leverage,
        "remaining_capital_usd": 300.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "account_equity_usd": 500.0,
        "margin_used": margin_used if side in {"long", "short"} else 0.0,
    }


def make_all_positions_for_snapshot(snapshot):
    has_position = snapshot.get("side") in {"long", "short"} and abs(float(snapshot.get("size", 0.0) or 0.0)) > 0.0
    return {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "margin_summary": {},
        "cross_margin_summary": {},
        "account_equity_usd": 500.0,
        "total_margin_used_usd": float(snapshot.get("margin_used", 0.0) or 0.0) if has_position else 0.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "remaining_capital_usd": 300.0,
        "remaining_capital_source": "withdrawable",
        "positions": [dict(snapshot)] if has_position else [],
        "positions_count": 1 if has_position else 0,
        "total_notional_usd": abs(float(snapshot.get("notional_usd", 0.0) or 0.0)) if has_position else 0.0,
    }


def install_profile_r_clip_fixture(uma, agent, *, atr_ref=1000.0):
    profile = uma.InstrumentMarketProfile(
        name="btc-test",
        timezone_name="UTC",
        helper_reset_time=None,
        low_liquidity_windows=(uma.LocalTimeWindow((1, 0, 0), (8, 0, 0)),),
        normal_liquidity_windows=(uma.LocalTimeWindow((8, 0, 0), (22, 0, 0)),),
        low_liquidity_weekdays=(5, 6),
        normal_liquidity_r_min_atr_multiple=1.5,
        normal_liquidity_r_max_atr_multiple=2.5,
        low_liquidity_r_min_atr_multiple=2.5,
        low_liquidity_r_max_atr_multiple=3.0,
    )
    agent.instrument_market_profiles = {key: profile for key in agent._profile_lookup_keys("BTC")}
    agent._profile_normal_liquidity_atr_ref = lambda **kwargs: {
        "available": True,
        "code": "ok",
        "atr_ref": atr_ref,
        "timeframe": "15m",
        "period": 14,
        "sample_count": 10,
        "sample_dates": ["2026-04-29"],
    }
    return profile


def test_profile_stop_clip_applies_normal_liquidity_r_max(uma):
    agent = make_agent(uma, reader=FakeReader())
    install_profile_r_clip_fixture(uma, agent, atr_ref=1000.0)

    correction = agent._maybe_clip_profile_stop_loss(
        side="long",
        entry_price=100000.0,
        stop_price=96000.0,
        symbol="BTC",
        now_utc=datetime(2026, 4, 29, 12, 0, tzinfo=timezone.utc),
    )

    assert correction["applied"] is True
    assert correction["liquidity_band"] == "normal_liquidity"
    assert correction["code"] == "profile_normal_liquidity_r_max_applied"
    assert correction["r_raw"] == pytest.approx(4000.0)
    assert correction["r_min"] == pytest.approx(1500.0)
    assert correction["r_max"] == pytest.approx(2500.0)
    assert correction["r_clipped"] == pytest.approx(2500.0)
    assert correction["stop_loss_price"] == pytest.approx(97500.0)


def test_profile_stop_clip_applies_low_liquidity_r_min(uma):
    agent = make_agent(uma, reader=FakeReader())
    install_profile_r_clip_fixture(uma, agent, atr_ref=1000.0)

    correction = agent._maybe_clip_profile_stop_loss(
        side="short",
        entry_price=100000.0,
        stop_price=101000.0,
        symbol="BTC",
        now_utc=datetime(2026, 4, 29, 2, 0, tzinfo=timezone.utc),
    )

    assert correction["applied"] is True
    assert correction["liquidity_band"] == "low_liquidity"
    assert correction["code"] == "profile_low_liquidity_r_min_applied"
    assert correction["r_raw"] == pytest.approx(1000.0)
    assert correction["r_min"] == pytest.approx(2500.0)
    assert correction["r_max"] == pytest.approx(3000.0)
    assert correction["r_clipped"] == pytest.approx(2500.0)
    assert correction["stop_loss_price"] == pytest.approx(102500.0)


def test_profile_weekend_uses_low_liquidity_even_during_normal_window(uma):
    agent = make_agent(uma, reader=FakeReader())
    install_profile_r_clip_fixture(uma, agent, atr_ref=1000.0)

    correction = agent._maybe_clip_profile_stop_loss(
        side="short",
        entry_price=100000.0,
        stop_price=101000.0,
        symbol="BTC",
        now_utc=datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc),
    )

    assert correction["applied"] is True
    assert correction["liquidity_band"] == "low_liquidity"
    assert correction["code"] == "profile_low_liquidity_r_min_applied"
    assert correction["r_min"] == pytest.approx(2500.0)
    assert correction["stop_loss_price"] == pytest.approx(102500.0)


def test_local_sizing_uses_profile_clipped_stop_before_sizing(uma):
    decision = make_entry_decision(
        uma,
        action="long",
        entry_price=100000.0,
        stop_loss_price=96000.0,
    )
    agent = make_agent(uma, reader=FakeReader())
    install_profile_r_clip_fixture(uma, agent, atr_ref=1000.0)
    agent._audit_event = lambda *args, **kwargs: None
    agent._maybe_clip_profile_stop_loss = lambda **kwargs: uma.UnifiedMarketAgent._maybe_clip_profile_stop_loss(
        agent,
        **{
            **kwargs,
            "now_utc": datetime(2026, 4, 29, 12, 0, tzinfo=timezone.utc),
        },
    )

    sizing = agent._derive_local_sizing(
        side="long",
        decision_context=decision,
        position_snapshot=make_position_snapshot(side="flat", max_leverage=20),
    )

    assert sizing["allowed"] is True
    assert decision.stop_loss_price == pytest.approx(97500.0)
    assert sizing["weighted_stop_loss_fraction"] == pytest.approx(0.025)
    assert sizing["max_notional_by_loss_usd"] == pytest.approx(4000.0)
    assert sizing["stop_profile"]["legs"][0]["stop_price"] == pytest.approx(97500.0)
    correction = sizing["stop_profile"]["profile_r_clip_correction"]
    assert correction["code"] == "profile_normal_liquidity_r_max_applied"


def test_query_new_playbook_sends_rendered_query_and_risk_constraints(uma):
    playbook = make_playbook(uma)
    reader = FakeReader(
        all_positions={
            "known": True,
            "account_address": "0xabc",
            "network": "mainnet",
            "margin_summary": {},
            "cross_margin_summary": {},
            "account_equity_usd": 500.0,
            "total_margin_used_usd": 0.0,
            "available_margin_usd": 321.5,
            "withdrawable_usd": 321.5,
            "remaining_capital_usd": 321.5,
            "remaining_capital_source": "withdrawable",
            "positions": [],
            "positions_count": 0,
            "total_notional_usd": 0.0,
        }
    )
    engine = FakeEngine(playbook)
    agent = make_agent(uma, playbook=playbook, reader=reader, engine=engine)

    agent.query_new_playbook("manual_once", None)

    assert len(engine.calls) == 1
    call = engine.calls[0]
    assert "321.5" not in call["user_query"]
    assert "{remaining_capital_usd}" in call["user_query"]
    assert "{max_planned_loss_usd}" in call["user_query"]
    assert "risk_constraints" not in call
    assert "all_positions" not in call
    assert "symbol_position" not in call
    assert "current_price" not in call
    assert call["trade_symbol_context"]["trade_symbol_key"] == "BTC"
    assert call["active_symbol"] == "BTC"
    assert call["has_live_position"] is False


def test_query_new_playbook_applies_local_sizing_from_entry_and_stop(uma):
    playbook = make_playbook(
        uma,
        entry_execute_now=False,
        entry_action_decision=make_entry_decision(
            uma,
            action="long",
            notional=123.0,
            stop_loss_price=99000.0,
            leverage=3,
        ),
        entry_scenario=uma.Scenario(
            observe_when_all=[uma.Condition(type="price_ge", level=99900.0, note="observe")],
            arm_when_all=[uma.Condition(type="price_ge", level=100100.0, note="arm")],
            cancel_when_any=[],
            timeout_seconds_after_arm=60,
            observation_starts_when="进入区间后观察",
        ),
    )
    reader = FakeReader()
    engine = FakeEngine(playbook)
    agent = make_agent(uma, playbook=playbook, reader=reader, engine=engine)

    agent.query_new_playbook("manual_once", None)

    decision = agent.current_playbook.position_management.action_decision
    assert decision.new_notional_usd == pytest.approx(10000.0)
    assert decision.leverage == 20
    assert decision.margin_basis_usd == pytest.approx(500.0)
    assert decision.planned_max_loss_usd == pytest.approx(100.0)


def _test_normalized_confidence(raw_confidence, *, threshold=0.46, full_scale=0.75):
    if raw_confidence is None:
        return None
    span = max(full_scale - threshold, 0.01)
    return min(max((float(raw_confidence) - threshold) / span, 0.0), 1.0)


def _test_adjust_same_side_target(
    *,
    target_side,
    target_notional,
    position_entry_price=100000.0,
    entry_price=100000.0,
    position_leverage=5.0,
):
    adjusted_target = max(0.0, float(target_notional or 0.0))
    if target_side not in {"long", "short"} or adjusted_target <= 0.0:
        return adjusted_target
    if position_entry_price <= 0.0 or entry_price <= 0.0:
        return adjusted_target
    raw_change_fraction = (float(entry_price) - float(position_entry_price)) / float(position_entry_price)
    comparison_change_fraction = raw_change_fraction if target_side == "long" else -raw_change_fraction
    if comparison_change_fraction >= 0.0:
        return adjusted_target
    if position_leverage <= 0.0:
        return adjusted_target
    numerator = 1.0 + comparison_change_fraction
    denominator = 1.0 + (float(position_leverage) * comparison_change_fraction)
    if numerator <= 0.0 or denominator <= 0.0:
        return adjusted_target
    adjusted = adjusted_target * numerator / denominator
    return adjusted if adjusted >= 0.0 else adjusted_target


def _build_full_chain_case(
    name,
    *,
    position_side,
    position_notional,
    entry_action,
    sizing_notional,
    raw_confidence,
    entry_execute_now=True,
    with_entry_scenario=False,
    sizing_allowed=True,
    max_margin_notional=None,
    position_entry_price=100000.0,
    entry_price=100000.0,
    position_leverage=5.0,
    tolerance_usd=1.0,
    close_fraction_tolerance=0.01,
):
    current_side = position_side
    current_notional = float(position_notional or 0.0) if current_side in {"long", "short"} else 0.0
    target_notional = max(0.0, float(sizing_notional or 0.0))
    normalized = _test_normalized_confidence(raw_confidence)
    action = "no_change"
    expected_notional = current_notional if current_side in {"long", "short"} else 0.0
    close_fraction = 0.0
    continue_after_close = False
    comparison_target = target_notional
    target_side = entry_action if entry_action in {"long", "short"} else ""
    no_change_should_refresh = False

    if entry_action == "no_trade":
        action = "no_change"
    elif entry_action in {"long", "short"}:
        is_opposite_signal = current_side in {"long", "short"} and target_side in {"long", "short"} and current_side != target_side
        if normalized is not None and normalized <= 0.0 and not is_opposite_signal:
            action = "no_change"
            expected_notional = current_notional if current_side in {"long", "short"} else 0.0
        else:
            if current_side == target_side:
                comparison_target = _test_adjust_same_side_target(
                    target_side=target_side,
                    target_notional=target_notional,
                    position_entry_price=position_entry_price,
                    entry_price=entry_price,
                    position_leverage=position_leverage,
                )
                if sizing_allowed and max_margin_notional is not None and max_margin_notional > 0.0:
                    comparison_target = min(comparison_target, float(max_margin_notional))
                if normalized is not None:
                    comparison_target = max(0.0, current_notional + normalized * (comparison_target - current_notional))
            if current_side == "flat":
                if normalized is not None:
                    comparison_target = max(0.0, normalized * target_notional)
                if target_notional > tolerance_usd and comparison_target > tolerance_usd:
                    action = target_side
                    expected_notional = comparison_target
                else:
                    action = "no_change"
                    expected_notional = 0.0
            elif current_side == target_side:
                if comparison_target > current_notional + tolerance_usd:
                    action = "add_to_long" if target_side == "long" else "add_to_short"
                    expected_notional = comparison_target
                elif current_notional > tolerance_usd and comparison_target <= tolerance_usd:
                    action = "close"
                    expected_notional = 0.0
                    close_fraction = 1.0
                    continue_after_close = True
                elif current_notional > tolerance_usd and comparison_target < current_notional - tolerance_usd:
                    close_fraction = min(max((current_notional - comparison_target) / max(current_notional, 1e-12), 0.0), 1.0)
                    if close_fraction >= 0.999999:
                        action = "close"
                        expected_notional = 0.0
                        continue_after_close = True
                    elif close_fraction < close_fraction_tolerance:
                        action = "no_change"
                        expected_notional = current_notional
                        close_fraction = 0.0
                        no_change_should_refresh = True
                    else:
                        action = "trim"
                        expected_notional = comparison_target
                else:
                    action = "no_change"
                    expected_notional = current_notional
                    no_change_should_refresh = True
            else:
                if normalized is not None:
                    comparison_target = max(0.0, normalized * target_notional)
                if raw_confidence is None:
                    if comparison_target > tolerance_usd:
                        if entry_execute_now:
                            action = "reverse_to_long" if target_side == "long" else "reverse_to_short"
                            expected_notional = comparison_target
                        else:
                            action = "close"
                            expected_notional = 0.0
                            close_fraction = 1.0
                            continue_after_close = True
                    else:
                        action = "close"
                        expected_notional = 0.0
                        close_fraction = 1.0
                        continue_after_close = bool(with_entry_scenario and not entry_execute_now)
                elif float(raw_confidence) < 0.40:
                    action = "no_change"
                    expected_notional = current_notional
                elif float(raw_confidence) >= 0.85:
                    if comparison_target > tolerance_usd and entry_execute_now:
                        action = "reverse_to_long" if target_side == "long" else "reverse_to_short"
                        expected_notional = comparison_target
                    else:
                        action = "close"
                        expected_notional = 0.0
                        close_fraction = 1.0
                        continue_after_close = bool(comparison_target > tolerance_usd)
                else:
                    strength = min(max((float(raw_confidence) - 0.40) / (0.85 - 0.40), 0.0), 1.0)
                    close_fraction = min(max(0.15 + 0.85 * strength - 0.05, 0.0), 0.95)
                    if close_fraction >= 0.85:
                        action = "close"
                        expected_notional = 0.0
                        close_fraction = 1.0
                    elif close_fraction < close_fraction_tolerance:
                        action = "no_change"
                        expected_notional = current_notional
                        close_fraction = 0.0
                    else:
                        action = "trim"
                        expected_notional = current_notional * (1.0 - close_fraction)

    immediate_same_side_refresh = bool(
        current_side in {"long", "short"}
        and entry_execute_now
        and entry_action == current_side
        and action == "no_change"
        and entry_action in {"long", "short"}
        and no_change_should_refresh
    )
    forced_immediate_close = bool(
        current_side in {"long", "short"}
        and with_entry_scenario
        and action == "close"
        and continue_after_close
    )
    expected_execute = bool(entry_execute_now and (action != "no_change" or immediate_same_side_refresh))
    if forced_immediate_close:
        expected_execute = True
    if action == "close" and entry_execute_now:
        continue_after_close = False
    sizing_calls = 0 if entry_action == "no_trade" else 1
    if expected_execute and continue_after_close:
        sizing_calls += 1
    if not expected_execute and with_entry_scenario and current_side in {"long", "short"} and action == "close":
        sizing_calls += 1

    case = {
        "name": name,
        "position_side": position_side,
        "position_notional": float(position_notional or 0.0),
        "entry_action": entry_action,
        "sizing_notional": float(sizing_notional or 0.0),
        "raw_confidence": raw_confidence,
        "entry_execute_now": entry_execute_now,
        "with_entry_scenario": with_entry_scenario,
        "sizing_allowed": sizing_allowed,
        "expected_action": action,
        "expected_notional": expected_notional,
        "expected_execute": expected_execute,
        "expected_sizing_calls": sizing_calls,
        "position_entry_price": position_entry_price,
        "entry_price": entry_price,
        "position_leverage": position_leverage,
    }
    if max_margin_notional is not None:
        case["max_margin_notional"] = float(max_margin_notional)
    if close_fraction > 0.0:
        case["expected_close_fraction"] = close_fraction
    if action == "close":
        case["expected_continue"] = continue_after_close
    if expected_execute and continue_after_close:
        case["expected_position_management_session"] = True
    if no_change_should_refresh:
        case["expected_refresh_no_change"] = True
    return case


def build_full_chain_extra_cases():
    cases = []

    def add(prefix, **kwargs):
        cases.append(_build_full_chain_case(f"generated_{prefix}", **kwargs))

    confidence_values = [None, 0.0, 0.10, 0.459, 0.46, 0.461, 0.50, 0.52, 0.605, 0.67, 0.749, 0.75, 0.90]
    for side in ("long", "short"):
        for raw in confidence_values:
            add(
                f"flat_{side}_confidence_{raw}",
                position_side="flat",
                position_notional=0.0,
                entry_action=side,
                sizing_notional=1000.0,
                raw_confidence=raw,
            )

    for side in ("long", "short"):
        for target in (0.0, 0.5, 1.0, 1.01, 2.0):
            for raw in (None, 0.52, 0.75):
                add(
                    f"flat_{side}_target_{target}_confidence_{raw}",
                    position_side="flat",
                    position_notional=0.0,
                    entry_action=side,
                    sizing_notional=target,
                    raw_confidence=raw,
                )

    same_side_targets = (1800.0, 1600.0, 1200.0, 1001.01, 1001.0, 1000.5, 999.5, 999.0, 998.99, 990.1, 990.0, 989.0, 700.0, 1.1, 1.0, 0.5)
    same_side_confidences = (None, 0.0, 0.459, 0.46, 0.461, 0.52, 0.605, 0.749, 0.75, 0.90)
    for side in ("long", "short"):
        for target in same_side_targets:
            for raw in same_side_confidences:
                add(
                    f"same_{side}_target_{target}_confidence_{raw}",
                    position_side=side,
                    position_notional=1000.0,
                    entry_action=side,
                    sizing_notional=target,
                    raw_confidence=raw,
                )

    for side, favorable_entry, unfavorable_entry in (("long", 99000.0, 101000.0), ("short", 101000.0, 99000.0)):
        for entry_price, label in ((favorable_entry, "favorable"), (unfavorable_entry, "unfavorable")):
            for raw in (None, 0.52, 0.605, 0.75):
                add(
                    f"same_{side}_entry_adjust_{label}_confidence_{raw}",
                    position_side=side,
                    position_notional=1000.0,
                    entry_action=side,
                    sizing_notional=1600.0,
                    raw_confidence=raw,
                    entry_price=entry_price,
                    position_entry_price=100000.0,
                    position_leverage=5.0,
                )

    for side in ("long", "short"):
        for cap in (900.0, 1000.0, 1001.0, 1200.0, 2000.0):
            for raw in (None, 0.52, 0.605, 0.75):
                add(
                    f"same_{side}_margin_cap_{cap}_confidence_{raw}",
                    position_side=side,
                    position_notional=1000.0,
                    entry_action=side,
                    sizing_notional=1600.0,
                    raw_confidence=raw,
                    max_margin_notional=cap,
                )

    reverse_confidences = (None, 0.0, 0.459, 0.46, 0.52, 0.605, 0.608, 0.67, 0.749, 0.75, 0.90)
    reverse_targets = (0.5, 1.0, 1.01, 900.0, 1000.0, 1200.0, 2000.0)
    for position_side, entry_action in (("long", "short"), ("short", "long")):
        for target in reverse_targets:
            for raw in reverse_confidences:
                add(
                    f"reverse_{position_side}_to_{entry_action}_target_{target}_confidence_{raw}",
                    position_side=position_side,
                    position_notional=1000.0,
                    entry_action=entry_action,
                    sizing_notional=target,
                    raw_confidence=raw,
                )

    for position_side, entry_action in (("long", "short"), ("short", "long")):
        for target in (1.0, 1200.0, 2000.0):
            for raw in (None, 0.52, 0.67, 0.75):
                add(
                    f"reverse_non_immediate_scenario_{position_side}_to_{entry_action}_target_{target}_confidence_{raw}",
                    position_side=position_side,
                    position_notional=1000.0,
                    entry_action=entry_action,
                    sizing_notional=target,
                    raw_confidence=raw,
                    entry_execute_now=False,
                    with_entry_scenario=True,
                )
                add(
                    f"reverse_non_immediate_no_scenario_{position_side}_to_{entry_action}_target_{target}_confidence_{raw}",
                    position_side=position_side,
                    position_notional=1000.0,
                    entry_action=entry_action,
                    sizing_notional=target,
                    raw_confidence=raw,
                    entry_execute_now=False,
                    with_entry_scenario=False,
                )

    for position_side, entry_action in (("flat", "long"), ("flat", "short"), ("long", "long"), ("short", "short"), ("long", "short"), ("short", "long")):
        for raw in (None, 0.52, 0.67, 0.75):
            add(
                f"sizing_denied_{position_side}_entry_{entry_action}_confidence_{raw}",
                position_side=position_side,
                position_notional=1000.0 if position_side in {"long", "short"} else 0.0,
                entry_action=entry_action,
                sizing_notional=1200.0,
                raw_confidence=raw,
                sizing_allowed=False,
            )

    for position_side in ("flat", "long", "short"):
        add(
            f"no_trade_{position_side}",
            position_side=position_side,
            position_notional=1000.0 if position_side in {"long", "short"} else 0.0,
            entry_action="no_trade",
            sizing_notional=1000.0,
            raw_confidence=0.90,
        )

    assert len(cases) >= 117
    return cases


@pytest.mark.parametrize(
    "case",
    [
        {
            "name": "flat_open_scaled",
            "position_side": "flat",
            "position_notional": 0.0,
            "entry_action": "short",
            "sizing_notional": 988.8114,
            "raw_confidence": 0.52,
            "expected_action": "short",
            "expected_notional": 204.58166896551725,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "confidence_gate_flat",
            "position_side": "flat",
            "position_notional": 0.0,
            "entry_action": "long",
            "sizing_notional": 1000.0,
            "raw_confidence": 0.10,
            "expected_action": "no_change",
            "expected_notional": 0.0,
            "expected_execute": False,
            "expected_sizing_calls": 1,
        },
        {
            "name": "flat_open_none_confidence_uses_full_target",
            "position_side": "flat",
            "position_notional": 0.0,
            "entry_action": "long",
            "sizing_notional": 1000.0,
            "raw_confidence": None,
            "expected_action": "long",
            "expected_notional": 1000.0,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "flat_open_at_threshold_gates_to_no_change",
            "position_side": "flat",
            "position_notional": 0.0,
            "entry_action": "long",
            "sizing_notional": 1000.0,
            "raw_confidence": 0.46,
            "expected_action": "no_change",
            "expected_notional": 0.0,
            "expected_execute": False,
            "expected_sizing_calls": 1,
        },
        {
            "name": "flat_open_at_full_scale_uses_full_target",
            "position_side": "flat",
            "position_notional": 0.0,
            "entry_action": "long",
            "sizing_notional": 1000.0,
            "raw_confidence": 0.75,
            "expected_action": "long",
            "expected_notional": 1000.0,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "flat_open_above_full_scale_is_clamped",
            "position_side": "flat",
            "position_notional": 0.0,
            "entry_action": "long",
            "sizing_notional": 1000.0,
            "raw_confidence": 0.95,
            "expected_action": "long",
            "expected_notional": 1000.0,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "flat_target_at_tolerance_no_change",
            "position_side": "flat",
            "position_notional": 0.0,
            "entry_action": "long",
            "sizing_notional": 1.0,
            "raw_confidence": None,
            "expected_action": "no_change",
            "expected_notional": 0.0,
            "expected_execute": False,
            "expected_sizing_calls": 1,
        },
        {
            "name": "flat_sizing_denied_uses_entry_fallback_once",
            "position_side": "flat",
            "position_notional": 0.0,
            "entry_action": "long",
            "sizing_notional": 1000.0,
            "sizing_allowed": False,
            "raw_confidence": None,
            "expected_action": "long",
            "expected_notional": 1000.0,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "same_side_add_scaled_delta",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "long",
            "sizing_notional": 1600.0,
            "raw_confidence": 0.605,
            "expected_action": "add_to_long",
            "expected_notional": 1300.0,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "same_side_margin_cap_limits_full_target",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "long",
            "sizing_notional": 1600.0,
            "max_margin_notional": 1200.0,
            "raw_confidence": None,
            "expected_action": "add_to_long",
            "expected_notional": 1200.0,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "same_side_margin_cap_is_confidence_scaled",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "long",
            "sizing_notional": 1600.0,
            "max_margin_notional": 1200.0,
            "raw_confidence": 0.605,
            "expected_action": "add_to_long",
            "expected_notional": 1100.0,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "same_side_none_confidence_uses_full_target",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "long",
            "sizing_notional": 1600.0,
            "raw_confidence": None,
            "expected_action": "add_to_long",
            "expected_notional": 1600.0,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "same_side_below_threshold_delta_becomes_no_change",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "long",
            "sizing_notional": 1600.0,
            "raw_confidence": 0.45,
            "expected_action": "no_change",
            "expected_notional": 1000.0,
            "expected_execute": False,
            "expected_sizing_calls": 1,
        },
        {
            "name": "same_side_at_threshold_delta_becomes_no_change",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "long",
            "sizing_notional": 1600.0,
            "raw_confidence": 0.46,
            "expected_action": "no_change",
            "expected_notional": 1000.0,
            "expected_execute": False,
            "expected_sizing_calls": 1,
        },
        {
            "name": "same_side_full_scale_add_uses_full_target",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "long",
            "sizing_notional": 1600.0,
            "raw_confidence": 0.75,
            "expected_action": "add_to_long",
            "expected_notional": 1600.0,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "same_side_trim_scaled_delta",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "long",
            "sizing_notional": 700.0,
            "raw_confidence": 0.605,
            "expected_action": "trim",
            "expected_notional": 850.0,
            "expected_close_fraction": 0.15,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "same_side_full_scale_trim_uses_full_target",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "long",
            "sizing_notional": 700.0,
            "raw_confidence": 0.75,
            "expected_action": "trim",
            "expected_notional": 700.0,
            "expected_close_fraction": 0.3,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "same_side_small_delta_refresh_no_change",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "long",
            "sizing_notional": 990.0,
            "raw_confidence": 0.605,
            "expected_action": "no_change",
            "expected_notional": 1000.0,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "same_side_delta_just_under_fraction_tolerance_no_change",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "long",
            "sizing_notional": 990.1,
            "raw_confidence": None,
            "expected_action": "no_change",
            "expected_notional": 1000.0,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "same_side_delta_at_fraction_tolerance_trims",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "long",
            "sizing_notional": 990.0,
            "raw_confidence": None,
            "expected_action": "trim",
            "expected_notional": 990.0,
            "expected_close_fraction": 0.01,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "same_side_delta_just_over_fraction_tolerance_trims",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "long",
            "sizing_notional": 989.0,
            "raw_confidence": None,
            "expected_action": "trim",
            "expected_notional": 989.0,
            "expected_close_fraction": 0.011,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "same_side_target_at_tolerance_closes_without_continue",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "long",
            "sizing_notional": 1.0,
            "raw_confidence": None,
            "expected_action": "close",
            "expected_notional": 0.0,
            "expected_close_fraction": 1.0,
            "expected_continue": False,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "opposite_event_medium_confidence_trims",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "short",
            "sizing_notional": 1200.0,
            "raw_confidence": 0.52,
            "expected_action": "trim",
            "expected_notional": 673.3333333333334,
            "expected_close_fraction": 0.32666666666666666,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "opposite_event_small_target_still_trims_existing_position",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "short",
            "sizing_notional": 1.0,
            "raw_confidence": 0.75,
            "expected_action": "trim",
            "expected_notional": 238.88888888888903,
            "expected_close_fraction": 0.761111111111111,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "opposite_missing_confidence_non_immediate_target_at_tolerance_closes",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "short",
            "entry_execute_now": False,
            "with_entry_scenario": True,
            "sizing_notional": 1.0,
            "raw_confidence": None,
            "expected_action": "close",
            "expected_notional": 0.0,
            "expected_close_fraction": 1.0,
            "expected_continue": True,
            "expected_execute": True,
            "expected_sizing_calls": 2,
            "expected_position_management_session": True,
        },
        {
            "name": "opposite_event_above_trim_threshold_trims",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "short",
            "sizing_notional": 1200.0,
            "raw_confidence": 0.46,
            "expected_action": "trim",
            "expected_notional": 786.6666666666666,
            "expected_close_fraction": 0.21333333333333337,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "opposite_event_mid_high_confidence_trims",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "short",
            "sizing_notional": 1200.0,
            "raw_confidence": 0.605,
            "expected_action": "trim",
            "expected_notional": 512.7777777777778,
            "expected_close_fraction": 0.4872222222222222,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "opposite_event_mid_high_boundary_trims",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "short",
            "sizing_notional": 1200.0,
            "raw_confidence": 0.608,
            "expected_action": "trim",
            "expected_notional": 507.1111111111111,
            "expected_close_fraction": 0.49288888888888893,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "opposite_event_non_immediate_trims_plan",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "short",
            "entry_execute_now": False,
            "with_entry_scenario": True,
            "sizing_notional": 1200.0,
            "raw_confidence": 0.67,
            "expected_action": "trim",
            "expected_notional": 390.0,
            "expected_close_fraction": 0.61,
            "expected_execute": False,
            "expected_sizing_calls": 1,
        },
        {
            "name": "opposite_event_high_confidence_trims_without_known_basis",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "short",
            "sizing_notional": 1200.0,
            "raw_confidence": 0.67,
            "expected_action": "trim",
            "expected_notional": 390.0,
            "expected_close_fraction": 0.61,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "reverse_none_confidence_uses_full_target",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "short",
            "sizing_notional": 1200.0,
            "raw_confidence": None,
            "expected_action": "reverse_to_short",
            "expected_notional": 1200.0,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "opposite_event_high_not_unknown_reverse_threshold_trims",
            "position_side": "long",
            "position_notional": 1000.0,
            "entry_action": "short",
            "sizing_notional": 1200.0,
            "raw_confidence": 0.75,
            "expected_action": "trim",
            "expected_notional": 238.88888888888903,
            "expected_close_fraction": 0.761111111111111,
            "expected_execute": True,
            "expected_sizing_calls": 1,
        },
        {
            "name": "no_trade_no_sizing",
            "position_side": "flat",
            "position_notional": 0.0,
            "entry_action": "no_trade",
            "sizing_notional": 1000.0,
            "raw_confidence": 0.90,
            "expected_action": "no_change",
            "expected_notional": 0.0,
            "expected_execute": False,
            "expected_sizing_calls": 0,
        },
    ] + build_full_chain_extra_cases(),
    ids=lambda case: case["name"],
)
def test_query_new_playbook_full_chain_materializes_branch_matrix_without_post_resizing(monkeypatch, uma, case):
    monkeypatch.setenv("TRIGGER_CONFIDENCE_RELEVANCE_THRESHOLD", "0.46")
    monkeypatch.setenv("TRIGGER_CONFIDENCE_FULL_SCALE", "0.75")
    monkeypatch.setenv("TRIGGER_CONFIDENCE_RELEVANCE_THRESHOLD_BTC_USDC", "0.46")
    monkeypatch.setenv("TRIGGER_CONFIDENCE_FULL_SCALE_BTC_USDC", "0.75")
    position = make_position_snapshot(
        side=case["position_side"],
        notional=case["position_notional"],
        entry_price=case.get("position_entry_price", 100000.0),
        leverage=case.get("position_leverage", 5.0),
    )
    all_positions = make_all_positions_for_snapshot(position)
    stop_loss = 101000.0 if case["entry_action"] == "short" else 99000.0 if case["entry_action"] == "long" else 0.0
    entry_scenario = None
    if case.get("with_entry_scenario"):
        entry_scenario = uma.EntryScenario(
            observe_when_all=[],
            execute_when_all={
                "condition": {"type": "price_le", "level": 99900.0, "low": 0.0, "high": 0.0, "timer_seconds": 0, "tolerance_bps": 0.0, "min_ratio": 0.0},
                "timeout_seconds": 300,
            },
        )
    playbook = make_playbook(
        uma,
        entry_execute_now=case.get("entry_execute_now", True),
        entry_action_decision=make_entry_decision(
            uma,
            action=case["entry_action"],
            notional=case["sizing_notional"],
            stop_loss_price=stop_loss,
            leverage=0,
            entry_price=case.get("entry_price", 100000.0),
        ),
        entry_scenario=entry_scenario,
    )
    playbook.trigger_confidence_raw = case["raw_confidence"]
    playbook.trigger_confidence = uma.normalize_confidence_value(case["raw_confidence"], symbol="BTC")
    reader = FakeReader(all_positions=all_positions, symbol_snapshots=[dict(position), dict(position)])
    engine = FakeEngine(playbook)
    agent = make_agent(uma, playbook=playbook, reader=reader, engine=engine)
    sizing_calls = []

    def fake_derive_local_sizing(**kwargs):
        sizing_calls.append(kwargs)
        if not case.get("sizing_allowed", True):
            return {
                "allowed": False,
                "code": "test_sizing_denied",
                "message": "test sizing denied",
            }
        max_margin_notional = case.get("max_margin_notional", case["sizing_notional"] * 10.0)
        return {
            "allowed": True,
            "suggested_notional_usd": case["sizing_notional"],
            "max_allowed_notional_usd": case["sizing_notional"],
            "max_notional_by_loss_usd": case["sizing_notional"],
            "max_notional_by_margin_usd": max_margin_notional,
            "requested_leverage": 20,
            "margin_basis_usd": 50.0,
            "max_leverage": 20,
            "max_planned_loss_usd": 100.0,
        }

    seen = {}

    def fake_execute_management(decision, plan_name, management_plan=None, trigger_confidence_raw=None, **_kwargs):
        seen["action"] = decision.action
        seen["new_notional_usd"] = decision.new_notional_usd
        seen["close_fraction"] = decision.close_fraction
        seen["continue_entry_plan_after_close"] = decision.continue_entry_plan_after_close
        if decision.action in {"short", "reverse_to_short", "add_to_short"}:
            after = make_position_snapshot(side="short", notional=max(decision.new_notional_usd, 1.0), leverage=decision.leverage)
        elif decision.action in {"long", "reverse_to_long", "add_to_long", "no_change"}:
            after_notional = max(decision.new_notional_usd, case["position_notional"])
            after_side = case["position_side"] if decision.action == "no_change" and case["position_side"] in {"long", "short"} else "long"
            after = make_position_snapshot(side=after_side, notional=after_notional, leverage=decision.leverage or position.get("leverage", 0.0))
        elif decision.action == "trim":
            after = make_position_snapshot(side=case["position_side"], notional=decision.new_notional_usd, leverage=position.get("leverage", 0.0))
        elif decision.action == "close":
            after = make_position_snapshot(side="flat", notional=0.0)
        else:
            after = make_position_snapshot(side="flat", notional=0.0)
        return {
            "accepted": True,
            "decision": decision.to_dict(),
            "position_before": dict(position),
            "position_after": after,
        }

    agent._derive_local_sizing = fake_derive_local_sizing
    agent.execute_management_decision = fake_execute_management

    agent.query_new_playbook("manual_once", None)

    assert len(sizing_calls) == case["expected_sizing_calls"]
    if case["expected_execute"]:
        assert seen["action"] == case["expected_action"]
        assert seen["new_notional_usd"] == pytest.approx(case["expected_notional"])
        if "expected_close_fraction" in case:
            assert seen["close_fraction"] == pytest.approx(case["expected_close_fraction"])
        if "expected_continue" in case:
            assert seen["continue_entry_plan_after_close"] is case["expected_continue"]
        if "expected_position_management_session" in case:
            assert (agent.position_management_session is not None) is case["expected_position_management_session"]
    else:
        assert seen == {}
        decision = agent.current_playbook.position_management.action_decision
        assert decision.action == case["expected_action"]
        assert decision.new_notional_usd == pytest.approx(case["expected_notional"])


def test_query_new_playbook_full_chain_uses_symbol_specific_confidence_calibration(monkeypatch, uma):
    monkeypatch.setenv("TRIGGER_CONFIDENCE_RELEVANCE_THRESHOLD", "0.10")
    monkeypatch.setenv("TRIGGER_CONFIDENCE_FULL_SCALE", "0.90")
    monkeypatch.setenv("TRIGGER_CONFIDENCE_RELEVANCE_THRESHOLD_BTC_USDC", "0.46")
    monkeypatch.setenv("TRIGGER_CONFIDENCE_FULL_SCALE_BTC_USDC", "0.75")
    position = make_position_snapshot(side="flat", notional=0.0)
    all_positions = make_all_positions_for_snapshot(position)
    playbook = make_playbook(
        uma,
        entry_execute_now=True,
        entry_action_decision=make_entry_decision(
            uma,
            action="long",
            notional=1000.0,
            stop_loss_price=99000.0,
            leverage=0,
        ),
    )
    playbook.trigger_confidence_raw = 0.52
    playbook.trigger_confidence = uma.normalize_confidence_value(0.52, symbol="BTC")
    reader = FakeReader(all_positions=all_positions, symbol_snapshots=[dict(position), dict(position)])
    engine = FakeEngine(playbook)
    agent = make_agent(uma, playbook=playbook, reader=reader, engine=engine)
    agent._derive_local_sizing = lambda **kwargs: {
        "allowed": True,
        "suggested_notional_usd": 1000.0,
        "max_allowed_notional_usd": 1000.0,
        "max_notional_by_loss_usd": 1000.0,
        "max_notional_by_margin_usd": 10000.0,
        "requested_leverage": 20,
        "margin_basis_usd": 50.0,
        "max_planned_loss_usd": 100.0,
    }
    seen = {}

    def fake_execute_management(decision, plan_name, management_plan=None, trigger_confidence_raw=None, **_kwargs):
        seen["action"] = decision.action
        seen["new_notional_usd"] = decision.new_notional_usd
        return {
            "accepted": True,
            "decision": decision.to_dict(),
            "position_before": dict(position),
            "position_after": make_position_snapshot(side="long", notional=max(decision.new_notional_usd, 1.0), leverage=decision.leverage),
        }

    agent.execute_management_decision = fake_execute_management

    agent.query_new_playbook("manual_once", None)

    assert seen["action"] == "long"
    assert seen["new_notional_usd"] == pytest.approx(206.8965517241379)


def test_query_new_playbook_writes_audit_history_with_llm_debug(uma, tmp_path):
    playbook = make_playbook(uma)
    engine = FakeEngine(playbook)
    agent = make_agent(uma, playbook=playbook, engine=engine)
    agent.enable_audit_log = True
    agent.audit_log_path = tmp_path / "audit.jsonl"

    agent.query_new_playbook("manual_once", None)

    lines = [line for line in agent.audit_log_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) >= 3
    records = [json.loads(line) for line in lines]
    event_names = {record["event"] for record in records}
    assert "playbook_query_requested" in event_names
    assert "llm_call_debug" in event_names
    assert "playbook_selected" in event_names


def test_passive_irrelevant_writes_audit_history_with_llm_debug(uma, tmp_path):
    playbook = uma.GenericPlaybook(
        display_answer="",
        current_bias="",
        trigger_event_relevance="unrelated",
        trigger_confidence=0.12,
        selected_symbol="BTC",
        entry_plan=uma.EntryPlan(
            execute_now=False,
            action_decision=uma.build_empty_strategy_decision(),
            scenario=None,
        ),
    )
    engine = FakeEngine(playbook)
    judge_messages = [
        {"role": "system", "content": [{"type": "input_text", "text": "judge prompt"}]},
        {
            "role": "user",
            "content": [
                {
                    "type": "input_text",
                    "text": json.dumps(
                        {
                            "trade_symbol": "BTC",
                            "trigger_event": {"source": "mni", "title": "off-topic"},
                            "recent_events": [{"source": "mni", "title": "prior"}],
                            "market_mainline_context": {"selected_instrument": "BTC"},
                        }
                    ),
                }
            ],
        },
    ]
    engine.last_call_debug.update(
        {
            "passive_event_judge_request_messages": judge_messages,
            "passive_event_judge_raw_output_text": '{"trigger_event_relevance":"unrelated","trigger_confidence":0.12,"action":"no_trade"}',
            "passive_event_judge_validated_output": {
                "trigger_event_relevance": "unrelated",
                "trigger_confidence": 0.12,
                "action": "no_trade",
            },
            "passive_step1_prefetched": True,
            "passive_step2_executed": False,
        }
    )
    agent = make_agent(uma, playbook=playbook, engine=engine)
    agent.enable_audit_log = True
    agent.audit_log_path = tmp_path / "audit.jsonl"
    trigger_event = {"source": "mni", "title": "off-topic"}

    agent.query_new_playbook("passive_event_trigger", trigger_event)

    records = [
        json.loads(line)
        for line in agent.audit_log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    event_names = [record["event"] for record in records]
    assert "playbook_query_requested" in event_names
    assert "passive_query_irrelevant" in event_names
    assert "llm_call_debug" in event_names
    assert "playbook_selected" not in event_names
    llm_debug = next(record["payload"] for record in records if record["event"] == "llm_call_debug")
    assert llm_debug["passive_event_judge_request_messages"] == judge_messages
    assert llm_debug["passive_event_judge_validated_output"]["trigger_event_relevance"] == "unrelated"
    assert llm_debug["validated_playbook"]["trigger_event_relevance"] == "unrelated"


def test_passive_event_judge_candidate_selection_uses_max_confidence():
    from market_agent.llm_engine import DiscretionaryLLMEngine

    selected = DiscretionaryLLMEngine._select_passive_event_judge_candidate(
        [
            {
                "sample_index": 1,
                "status": "ok",
                "validated_output": {
                    "trigger_event_relevance": "unrelated",
                    "trigger_confidence": 0.30,
                    "action": "no_trade",
                },
            },
            {
                "sample_index": 2,
                "status": "ok",
                "validated_output": {
                    "trigger_event_relevance": "relevant",
                    "trigger_confidence": 0.52,
                    "action": "short",
                },
            },
        ]
    )

    assert selected["sample_index"] == 2
    assert selected["validated_output"]["action"] == "short"


def test_passive_event_judge_candidate_selection_does_not_veto_relevant_with_duplicate():
    from market_agent.llm_engine import DiscretionaryLLMEngine

    selected = DiscretionaryLLMEngine._select_passive_event_judge_candidate(
        [
            {
                "sample_index": 1,
                "status": "ok",
                "validated_output": {
                    "trigger_event_relevance": "duplicate",
                    "trigger_confidence": 0.0,
                    "action": "no_trade",
                },
            },
            {
                "sample_index": 2,
                "status": "ok",
                "validated_output": {
                    "trigger_event_relevance": "relevant",
                    "trigger_confidence": 0.52,
                    "action": "short",
                },
            },
        ]
    )

    assert selected["sample_index"] == 2
    assert selected["validated_output"]["trigger_event_relevance"] == "relevant"


def test_query_new_playbook_rejects_llm_selected_symbol_outside_fixed_trade_symbol(uma):
    playbook = make_playbook(
        uma,
        selected_symbol="ETH",
        entry_execute_now=True,
        entry_action_decision=make_entry_decision(uma, action="long", notional=1200.0, stop_loss_price=99000.0, leverage=5),
    )
    reader = FakeReader(
        symbol_snapshots=[
            {
                "known": True,
                "account_address": "0xabc",
                "network": "mainnet",
                "symbol": "ETH",
                "side": "flat",
                "size": 0.0,
                "entry_price": 0.0,
                "mid_price": 100000.0,
                "notional_usd": 0.0,
                "leverage": 0.0,
                "max_leverage": 40,
                "remaining_capital_usd": 300.0,
                "available_margin_usd": 300.0,
                "withdrawable_usd": 300.0,
                "margin_used": 0.0,
            }
        ]
    )
    engine = FakeEngine(playbook)
    agent = make_agent(uma, playbook=playbook, reader=reader, engine=engine)

    agent.query_new_playbook("manual_once", None)

    assert agent.symbol == "BTC"
    assert agent.executor.symbol == "BTC"
    assert agent.current_playbook.selected_symbol == ""
    assert agent.executor.executions == []
    assert agent.executor.management_executions == []


def test_query_new_playbook_blocks_nontradable_selected_symbol_instead_of_switching(uma):
    scenario = uma.Scenario(
        observe_when_all=[uma.Condition(type="price_ge", level=100.0, note="observe")],
        arm_when_all=[uma.Condition(type="price_ge", level=101.0, note="arm")],
        cancel_when_any=[],
        timeout_seconds_after_arm=60,
        observation_starts_when="进入区间后观察",
    )
    playbook = make_playbook(uma, selected_symbol="SILVER", entry_action_decision=make_entry_decision(uma, action="short", notional=1200.0, leverage=8), entry_scenario=scenario)
    reader = FakeReader()
    engine = FakeEngine(playbook)
    agent = make_agent(uma, playbook=playbook, reader=reader, engine=engine)
    agent.trade_candidates = [
        {
            "candidate_key": "BTC",
            "display_name": "BTC",
            "configured_execution_symbol": "BTC",
            "execution_symbol": "BTC",
            "tradable_on_hyperliquid": True,
        },
        {
            "candidate_key": "SILVER",
            "display_name": "SILVER",
            "configured_execution_symbol": "",
            "execution_symbol": "",
            "tradable_on_hyperliquid": False,
        },
    ]

    agent.query_new_playbook("manual_once", None)

    assert agent.symbol == "BTC"
    assert agent.executor.symbol == "BTC"
    assert agent.current_playbook.selected_symbol == ""
    assert agent.executor.executions == []
    assert agent.executor.management_executions == []
    assert agent.position_management_session is None


def test_execute_decision_executes_without_local_risk_validation(uma):
    decision = make_entry_decision(uma, action="long", notional=15000.0, leverage=10, stop_loss_price=99000.0)
    management_plan = make_management_plan(uma, stop_price=99000.0)
    reader = FakeReader()
    executor = FakeExecutor(reader, "BTC")
    agent = make_agent(uma, reader=reader, executor=executor)

    result = agent.execute_decision(decision, "now_decision")

    assert result["accepted"] is True
    assert executor.executions == [("now_decision", "long")]


def test_execute_management_decision_executes_reverse_without_local_risk_validation(uma):
    reader = FakeReader(
        symbol_snapshots=[
            {
                "known": True,
                "account_address": "0xabc",
                "network": "mainnet",
                "symbol": "BTC",
                "side": "long",
                "size": 1.0,
                "entry_price": 100000.0,
                "mid_price": 100000.0,
                "notional_usd": 100000.0,
                "leverage": 5.0,
                "max_leverage": 20,
                "remaining_capital_usd": 20.0,
                "available_margin_usd": 20.0,
                "withdrawable_usd": 20.0,
                "margin_used": 30.0,
            }
        ]
    )
    executor = FakeExecutor(reader, "BTC")
    agent = make_agent(uma, reader=reader, executor=executor)
    decision = make_management_decision(
        uma,
        action="reverse_to_short",
        notional=15000.0,
        leverage=10,
        stop_loss_price=101000.0,
    )
    management_plan = make_management_plan(uma, stop_price=101000.0, stop_type="price_ge")

    result = agent.execute_management_decision(decision, "reverse_now", management_plan)

    assert result["accepted"] is True
    assert executor.management_executions == [("reverse_now", "reverse_to_short")]


def test_step_risk_session_returns_requery_reason_when_management_action_is_rejected(uma):
    management_scenario = uma.Scenario(
        observe_when_all=[uma.Condition(type="price_ge", level=100.0, note="observe")],
        arm_when_all=[uma.Condition(type="price_ge", level=100.0, note="reach")],
        cancel_when_any=[],
        timeout_seconds_after_arm=30,
        observation_starts_when="价格到达 100 后开始观察",
    )
    management_plan = make_management_plan(uma, stop_price=101000.0, stop_type="price_ge", scenario=management_scenario, now_action=make_management_decision(uma, action="reverse_to_short", notional=1000.0, leverage=10))
    agent = make_agent(uma)
    agent.current_playbook = make_playbook(uma, management_plan=management_plan)
    agent.position_management_session = uma.PositionManagementSession(
        plan_name="management",
        side="long",
        position_management=management_plan,
        start_time=1.0,
        baseline_size=1.0,
        expected_size=1.0,
        initial_size_abs=1.0,
        runtimes={uma.SCENARIO_RUNTIME_KEY: uma.ScenarioRuntime()},
        history=deque(),
        history_seconds=1800,
    )
    agent.execute_management_decision = lambda decision, plan_name, management_plan=None, trigger_confidence_raw=None, **_kwargs: {
        "accepted": False,
        "position_after": {
            "symbol": "BTC",
            "side": "long",
            "size": 1.0,
            "entry_price": 100000.0,
            "mid_price": 101.2,
            "notional_usd": 101.2,
        },
    }

    first_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 1.0,
        "entry_price": 100000.0,
        "mid_price": 100.0,
        "notional_usd": 100.0,
    }
    second_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 1.0,
        "entry_price": 100000.0,
        "mid_price": 101.2,
        "notional_usd": 101.2,
    }

    status = agent.step_position_management_session(first_snapshot, now=100.0)

    assert status == "management_action_rejected"


def test_step_position_management_session_returns_requery_reason_on_scenario_timeout(uma):
    management_scenario = uma.Scenario(
        observe_when_all=[],
        arm_when_all=[uma.Condition(type="sustained_ge", level=101.0, seconds=5, note="arm")],
        cancel_when_any=[],
        timeout_seconds_after_arm=1,
        observation_starts_when="立即开始观察。",
    )
    management_plan = make_management_plan(uma, scenario=management_scenario, now_action=make_management_decision(uma, action="add_to_long", notional=1000.0, leverage=10))
    agent = make_agent(uma)
    agent.position_management_session = uma.PositionManagementSession(
        plan_name="management",
        side="flat",
        position_management=management_plan,
        start_time=1.0,
        baseline_size=0.0,
        expected_size=0.0,
        initial_size_abs=0.0,
        runtimes={uma.SCENARIO_RUNTIME_KEY: uma.ScenarioRuntime()},
        history=deque(),
        history_seconds=1800,
    )

    first_snapshot = {
        "symbol": "BTC",
        "side": "flat",
        "size": 0.0,
        "entry_price": 0.0,
        "mid_price": 100.0,
        "notional_usd": 0.0,
    }
    second_snapshot = {
        "symbol": "BTC",
        "side": "flat",
        "size": 0.0,
        "entry_price": 0.0,
        "mid_price": 100.0,
        "notional_usd": 0.0,
    }

    assert agent.step_position_management_session(first_snapshot, now=100.0) is None
    status = agent.step_position_management_session(second_snapshot, now=102.0)

    assert status == "management_scenario_timeout"
    assert agent.position_management_session is None



def test_query_new_playbook_immediate_live_addon_arms_post_fill_risk_template_after_acceptance(uma):
    entry_only_playbook = uma.GenericPlaybook(
        display_answer="display",
        current_bias="bullish",
        selected_symbol="BTC",
        selection_reason="BTC best",
        entry_plan=uma.EntryPlan(
            execute_now=True,
            action_decision=make_entry_decision(uma, action="long", notional=1500.0, stop_loss_price=98500.0, leverage=0),
            scenario=None,
        ),
    )
    open_snapshot = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "symbol": "BTC",
        "side": "long",
        "size": 0.01,
        "entry_price": 100000.0,
        "mid_price": 100500.0,
        "notional_usd": 1005.0,
        "leverage": 5.0,
        "max_leverage": 20,
        "remaining_capital_usd": 300.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "margin_used": 20.0,
    }
    all_positions = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "margin_summary": {},
        "cross_margin_summary": {},
        "account_equity_usd": 500.0,
        "total_margin_used_usd": 20.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "remaining_capital_usd": 300.0,
        "remaining_capital_source": "withdrawable",
        "positions": [
            {
                "symbol": "BTC",
                "side": "long",
                "size": 0.01,
                "entry_price": 100000.0,
                "mid_price": 100500.0,
                "notional_usd": 1005.0,
                "unrealized_pnl": 0.0,
                "return_on_equity": 0.0,
                "leverage": 5.0,
                "max_leverage": 20,
                "liquidation_price": 0.0,
                "margin_used": 20.0,
                "only_isolated": False,
            }
        ],
        "positions_count": 1,
        "total_notional_usd": 1005.0,
    }
    playbook = entry_only_playbook
    reader = FakeReader(all_positions=all_positions, symbol_snapshots=[open_snapshot])
    engine = FakeEngine(playbook)
    agent = make_agent(uma, playbook=playbook, reader=reader, engine=engine)
    agent.execute_management_decision = lambda decision, plan_name, management_plan=None, trigger_confidence_raw=None, **_kwargs: {
        "accepted": True,
        "position_after": dict(open_snapshot, size=0.015, notional_usd=1507.5),
    }

    agent.query_new_playbook("manual_once", None)

    assert agent.risk_session is not None
    assert agent.risk_session.position_management is not None
    assert len(agent.risk_session.take_profit_legs) == 2
    assert [leg.name for leg in agent.risk_session.take_profit_legs] == ["stage_tp1", "stage_tp2"]
    assert agent.risk_session.stop_loss_legs == []
    assert agent.risk_session.active_soft_stop_price == pytest.approx(98500.0)


def test_query_new_playbook_immediate_live_addon_rejection_keeps_current_management_plan(uma):
    entry_only_playbook = uma.GenericPlaybook(
        display_answer="display",
        current_bias="bullish",
        selected_symbol="BTC",
        selection_reason="BTC best",
        entry_plan=uma.EntryPlan(
            execute_now=True,
            action_decision=make_entry_decision(uma, action="long", notional=1500.0, stop_loss_price=98500.0, leverage=0),
            scenario=None,
        ),
    )
    open_snapshot = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "symbol": "BTC",
        "side": "long",
        "size": 1.0,
        "entry_price": 100000.0,
        "mid_price": 100500.0,
        "notional_usd": 100500.0,
        "leverage": 5.0,
        "max_leverage": 20,
        "remaining_capital_usd": 300.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "margin_used": 20.0,
    }
    all_positions = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "margin_summary": {},
        "cross_margin_summary": {},
        "account_equity_usd": 500.0,
        "total_margin_used_usd": 20.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "remaining_capital_usd": 300.0,
        "remaining_capital_source": "withdrawable",
        "positions": [
            {
                "symbol": "BTC",
                "side": "long",
                "size": 1.0,
                "entry_price": 100000.0,
                "mid_price": 100500.0,
                "notional_usd": 100500.0,
                "unrealized_pnl": 0.0,
                "return_on_equity": 0.0,
                "leverage": 5.0,
                "max_leverage": 20,
                "liquidation_price": 0.0,
                "margin_used": 20.0,
                "only_isolated": False,
            }
        ],
        "positions_count": 1,
        "total_notional_usd": 100500.0,
    }
    playbook = entry_only_playbook
    reader = FakeReader(all_positions=all_positions, symbol_snapshots=[open_snapshot])
    engine = FakeEngine(playbook)
    agent = make_agent(uma, playbook=playbook, reader=reader, engine=engine)
    agent.execute_management_decision = lambda decision, plan_name, management_plan=None, trigger_confidence_raw=None, **_kwargs: {
        "accepted": False,
        "position_after": dict(open_snapshot),
    }

    agent.query_new_playbook("manual_once", None)

    assert agent.risk_session is None



def test_query_new_playbook_live_position_entry_only_plan_materializes_add_to_long(uma):
    open_snapshot = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "symbol": "BTC",
        "side": "long",
        "size": 1.0,
        "entry_price": 100000.0,
        "mid_price": 100500.0,
        "notional_usd": 1000.0,
        "leverage": 5.0,
        "max_leverage": 20,
        "remaining_capital_usd": 300.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "margin_used": 20.0,
    }
    all_positions = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "margin_summary": {},
        "cross_margin_summary": {},
        "account_equity_usd": 500.0,
        "total_margin_used_usd": 20.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "remaining_capital_usd": 300.0,
        "remaining_capital_source": "withdrawable",
        "positions": [dict(open_snapshot)],
        "positions_count": 1,
        "total_notional_usd": 1000.0,
    }
    playbook = uma.GenericPlaybook(
        display_answer="display",
        current_bias="bullish",
        selected_symbol="BTC",
        selection_reason="BTC best",
        entry_plan=uma.EntryPlan(
            execute_now=True,
            action_decision=make_entry_decision(uma, action="long", notional=2000.0, stop_loss_price=99000.0, leverage=0),
            scenario=None,
        ),
    )
    reader = FakeReader(all_positions=all_positions, symbol_snapshots=[dict(open_snapshot), dict(open_snapshot)])
    engine = FakeEngine(playbook)
    agent = make_agent(uma, playbook=playbook, reader=reader, engine=engine)
    agent._derive_local_sizing = lambda **kwargs: {
        "allowed": True,
        "suggested_notional_usd": 2000.0,
        "requested_leverage": 5,
        "margin_basis_usd": 400.0,
        "max_planned_loss_usd": 100.0,
    }
    agent.execute_management_decision = lambda decision, plan_name, management_plan=None, trigger_confidence_raw=None, **_kwargs: {
        "accepted": True,
        "position_after": dict(open_snapshot, size=2.0, notional_usd=2000.0),
    }

    agent.query_new_playbook("manual_once", None)

    assert agent.current_playbook.position_management.execute_now is True
    assert agent.current_playbook.position_management.action_decision.action == "add_to_long"
    assert agent.current_playbook.position_management.action_decision.new_notional_usd == pytest.approx(2000.0)
    assert agent.current_playbook.post_fill_risk_template.action_decision.stop_loss_price == pytest.approx(99000.0)
    assert agent.risk_session is not None


def test_query_new_playbook_live_position_entry_only_reverse_observation_preserves_existing_stop_plan(uma):
    open_snapshot = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "symbol": "BTC",
        "side": "long",
        "size": 1.0,
        "entry_price": 100000.0,
        "mid_price": 100500.0,
        "notional_usd": 1000.0,
        "leverage": 5.0,
        "max_leverage": 20,
        "remaining_capital_usd": 300.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "margin_used": 20.0,
    }
    all_positions = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "margin_summary": {},
        "cross_margin_summary": {},
        "account_equity_usd": 500.0,
        "total_margin_used_usd": 20.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "remaining_capital_usd": 300.0,
        "remaining_capital_source": "withdrawable",
        "positions": [dict(open_snapshot)],
        "positions_count": 1,
        "total_notional_usd": 1000.0,
    }
    existing_management = make_management_plan(uma, stop_price=99000.0)
    previous_playbook = make_playbook(uma, management_plan=existing_management)
    reverse_watch = uma.EntryScenario(
        observe_when_all=[uma.Condition(type="price_le", level=100000.0, note="observe")],
        arm_when_all=[uma.Condition(type="price_le", level=99800.0, note="arm")],
        cancel_when_any=[],
        timeout_seconds_after_arm=300,
    )
    new_playbook = uma.GenericPlaybook(
        display_answer="display",
        current_bias="bearish",
        selected_symbol="BTC",
        selection_reason="BTC best",
        entry_plan=uma.EntryPlan(
            execute_now=False,
            action_decision=make_entry_decision(uma, action="short", notional=800.0, stop_loss_price=101000.0, leverage=0),
            scenario=reverse_watch,
        ),
    )
    reader = FakeReader(all_positions=all_positions, symbol_snapshots=[dict(open_snapshot), dict(open_snapshot)])
    engine = FakeEngine(new_playbook)
    agent = make_agent(uma, playbook=new_playbook, reader=reader, engine=engine)
    agent.current_playbook = previous_playbook
    existing_stop_loss_legs = [
        uma.ExitLeg(
            name="sl-1",
            note="stop",
            when_all=[uma.Condition(type="price_le", level=99000.0, note="stop")],
            close_fraction=1.0,
        )
    ]
    agent.risk_session = uma.RiskSession(
        plan_name="position_management",
        side="long",
        initial_entry_price=100000.0,
        start_time=1.0,
        baseline_size=1.0,
        expected_size=1.0,
        initial_size_abs=1.0,
        position_management=existing_management,
        take_profit_legs=[],
        stop_loss_legs=list(existing_stop_loss_legs),
        runtimes={},
        history=deque(),
        history_seconds=1800,
    )
    agent._derive_local_sizing = lambda **kwargs: {
        "allowed": True,
        "suggested_notional_usd": 800.0,
        "requested_leverage": 4,
        "margin_basis_usd": 200.0,
        "max_planned_loss_usd": 100.0,
    }
    agent.execute_management_decision = lambda decision, plan_name, management_plan=None, trigger_confidence_raw=None, **_kwargs: {
        "accepted": False,
        "position_after": dict(open_snapshot),
    }

    agent.query_new_playbook("manual_once", None)

    assert agent.current_playbook.position_management.execute_now is True
    assert agent.current_playbook.position_management.action_decision.action == "close"
    assert agent.current_playbook.position_management.action_decision.continue_entry_plan_after_close is True
    assert agent.current_playbook.position_management.scenario is None
    assert agent.risk_session is not None
    assert agent.risk_session.stop_loss_legs[0].when_all[0].level == pytest.approx(99000.0)



def test_query_new_playbook_live_position_nonexecuted_refresh_without_active_risk_session_keeps_risk_session_none(uma):
    open_snapshot = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "symbol": "BTC",
        "side": "long",
        "size": 1.0,
        "entry_price": 100000.0,
        "mid_price": 100500.0,
        "notional_usd": 1000.0,
        "leverage": 5.0,
        "max_leverage": 20,
        "remaining_capital_usd": 300.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "margin_used": 20.0,
    }
    all_positions = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "margin_summary": {},
        "cross_margin_summary": {},
        "account_equity_usd": 500.0,
        "total_margin_used_usd": 20.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "remaining_capital_usd": 300.0,
        "remaining_capital_source": "withdrawable",
        "positions": [dict(open_snapshot)],
        "positions_count": 1,
        "total_notional_usd": 1000.0,
    }
    reverse_watch = uma.EntryScenario(
        observe_when_all=[uma.Condition(type="price_le", level=100000.0, note="observe")],
        arm_when_all=[uma.Condition(type="price_le", level=99800.0, note="arm")],
        cancel_when_any=[],
        timeout_seconds_after_arm=300,
    )
    new_playbook = uma.GenericPlaybook(
        display_answer="display",
        current_bias="bearish",
        selected_symbol="BTC",
        selection_reason="BTC best",
        entry_plan=uma.EntryPlan(
            execute_now=False,
            action_decision=make_entry_decision(uma, action="short", notional=800.0, stop_loss_price=101000.0, leverage=0),
            scenario=reverse_watch,
        ),
    )
    reader = FakeReader(all_positions=all_positions, symbol_snapshots=[dict(open_snapshot), dict(open_snapshot)])
    engine = FakeEngine(new_playbook)
    agent = make_agent(uma, playbook=new_playbook, reader=reader, engine=engine)
    agent._derive_local_sizing = lambda **kwargs: {
        "allowed": True,
        "suggested_notional_usd": 800.0,
        "requested_leverage": 4,
        "margin_basis_usd": 200.0,
        "max_planned_loss_usd": 100.0,
    }
    agent.execute_management_decision = lambda decision, plan_name, management_plan=None, trigger_confidence_raw=None, **_kwargs: {
        "accepted": False,
        "position_after": dict(open_snapshot),
    }

    agent.query_new_playbook("manual_once", None)

    assert agent.risk_session is None
    assert agent.current_playbook.position_management.execute_now is True
    assert agent.current_playbook.position_management.action_decision.action == "close"
    assert agent.current_playbook.position_management.scenario is None


def test_query_new_playbook_live_position_nonexecuted_refresh_preserves_guard_state(uma):
    open_snapshot = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "symbol": "BTC",
        "side": "long",
        "size": 1.0,
        "entry_price": 100000.0,
        "mid_price": 100500.0,
        "notional_usd": 1000.0,
        "leverage": 5.0,
        "max_leverage": 20,
        "remaining_capital_usd": 300.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "margin_used": 20.0,
    }
    all_positions = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "margin_summary": {},
        "cross_margin_summary": {},
        "account_equity_usd": 500.0,
        "total_margin_used_usd": 20.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "remaining_capital_usd": 300.0,
        "remaining_capital_source": "withdrawable",
        "positions": [dict(open_snapshot)],
        "positions_count": 1,
        "total_notional_usd": 1000.0,
    }
    existing_management = make_management_plan(uma, stop_price=99000.0)
    existing_take_profit_legs = [
        uma.ExitLeg(
            name="tp-1",
            note="tp",
            when_all=[uma.Condition(type="price_ge", level=101000.0, note="tp")],
            close_fraction=0.5,
        ),
        uma.ExitLeg(
            name="tp-2",
            note="tp",
            when_all=[uma.Condition(type="price_ge", level=102000.0, note="tp")],
            close_fraction=0.5,
        ),
    ]
    existing_stop_loss_legs = [
        uma.ExitLeg(
            name="sl-1",
            note="stop",
            when_all=[uma.Condition(type="price_le", level=99000.0, note="stop")],
            close_fraction=1.0,
        )
    ]
    previous_playbook = make_playbook(uma, management_plan=existing_management)
    reverse_watch = uma.EntryScenario(
        observe_when_all=[uma.Condition(type="price_le", level=100000.0, note="observe")],
        arm_when_all=[uma.Condition(type="price_le", level=99800.0, note="arm")],
        cancel_when_any=[],
        timeout_seconds_after_arm=300,
    )
    new_playbook = uma.GenericPlaybook(
        display_answer="display",
        current_bias="bearish",
        selected_symbol="BTC",
        selection_reason="BTC best",
        entry_plan=uma.EntryPlan(
            execute_now=False,
            action_decision=make_entry_decision(uma, action="short", notional=800.0, stop_loss_price=101000.0, leverage=0),
            scenario=reverse_watch,
        ),
    )
    reader = FakeReader(all_positions=all_positions, symbol_snapshots=[dict(open_snapshot), dict(open_snapshot)])
    engine = FakeEngine(new_playbook)
    agent = make_agent(uma, playbook=new_playbook, reader=reader, engine=engine)
    agent.current_playbook = previous_playbook
    agent.risk_session = uma.RiskSession(
        plan_name="position_management",
        side="long",
        initial_entry_price=100000.0,
        start_time=1.0,
        baseline_size=1.0,
        expected_size=1.0,
        initial_size_abs=1.5,
        position_management=existing_management,
        take_profit_legs=list(existing_take_profit_legs),
        stop_loss_legs=list(existing_stop_loss_legs),
        runtimes={},
        history=deque(),
        history_seconds=1800,
        executed_leg_names={"take_profit::tp-1"},
    )
    agent._derive_local_sizing = lambda **kwargs: {
        "allowed": True,
        "suggested_notional_usd": 800.0,
        "requested_leverage": 4,
        "margin_basis_usd": 200.0,
        "max_planned_loss_usd": 100.0,
    }
    agent.execute_management_decision = lambda decision, plan_name, management_plan=None, trigger_confidence_raw=None, **_kwargs: {
        "accepted": False,
        "position_after": dict(open_snapshot),
    }

    agent.query_new_playbook("manual_once", None)

    assert agent.risk_session is not None
    assert [leg.name for leg in agent.risk_session.take_profit_legs] == ["tp-1", "tp-2"]
    assert [leg.name for leg in agent.risk_session.stop_loss_legs] == ["sl-1"]
    assert agent.risk_session.executed_leg_names == {"take_profit::tp-1"}
    assert agent.risk_session.initial_size_abs == pytest.approx(1.5)
    assert agent.risk_session.position_management is existing_management
    assert agent.risk_session.position_management.scenario is None
    assert agent.current_playbook.position_management.execute_now is True
    assert agent.current_playbook.position_management.action_decision.action == "close"
    assert agent.current_playbook.position_management.scenario is None


def test_query_new_playbook_live_position_entry_only_same_side_refresh_executes_no_change_management(uma):
    open_snapshot = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "symbol": "BTC",
        "side": "long",
        "size": 1.0,
        "entry_price": 100000.0,
        "mid_price": 100500.0,
        "notional_usd": 1000.0,
        "leverage": 5.0,
        "max_leverage": 20,
        "remaining_capital_usd": 300.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "margin_used": 20.0,
    }
    all_positions = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "margin_summary": {},
        "cross_margin_summary": {},
        "account_equity_usd": 500.0,
        "total_margin_used_usd": 20.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "remaining_capital_usd": 300.0,
        "remaining_capital_source": "withdrawable",
        "positions": [dict(open_snapshot)],
        "positions_count": 1,
        "total_notional_usd": 1000.0,
    }
    playbook = uma.GenericPlaybook(
        display_answer="display",
        current_bias="bullish",
        selected_symbol="BTC",
        selection_reason="BTC best",
        entry_plan=uma.EntryPlan(
            execute_now=True,
            action_decision=make_entry_decision(uma, action="long", notional=1000.0, stop_loss_price=99500.0, leverage=0),
            scenario=None,
        ),
    )
    reader = FakeReader(all_positions=all_positions, symbol_snapshots=[dict(open_snapshot), dict(open_snapshot)])
    engine = FakeEngine(playbook)
    agent = make_agent(uma, playbook=playbook, reader=reader, engine=engine)
    agent._derive_local_sizing = lambda **kwargs: {
        "allowed": True,
        "suggested_notional_usd": 1000.0,
        "requested_leverage": 7,
        "margin_basis_usd": 142.857143,
        "max_planned_loss_usd": 100.0,
    }
    seen = {}

    def fake_execute_management(decision, plan_name, management_plan=None, trigger_confidence_raw=None, **_kwargs):
        seen["action"] = decision.action
        seen["requested_leverage"] = decision.leverage
        seen["stop_loss_price"] = decision.stop_loss_price
        return {"accepted": True, "position_after": dict(open_snapshot)}

    agent.execute_management_decision = fake_execute_management

    agent.query_new_playbook("manual_once", None)

    assert seen["action"] == "no_change"
    assert seen["requested_leverage"] == 7
    assert seen["stop_loss_price"] == pytest.approx(99500.0)
    assert agent.current_playbook.position_management.execute_now is False


def test_materialize_live_position_non_refresh_no_change_marks_risk_session_passthrough(uma):
    open_snapshot = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "symbol": "BTC",
        "side": "long",
        "size": 1.0,
        "entry_price": 100000.0,
        "mid_price": 100500.0,
        "notional_usd": 1000.0,
        "leverage": 5.0,
        "max_leverage": 20,
        "remaining_capital_usd": 300.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "margin_used": 20.0,
    }
    existing_management = make_management_plan(uma, stop_price=99000.0)
    existing_management.action_decision.stop_loss_price = 99000.0
    existing_take_profit_legs = [
        uma.ExitLeg(name="stage_tp2", note="tp2", when_all=[uma.Condition(type="price_ge", level=102000.0, note="tp2")], close_fraction=0.40),
    ]
    existing_stop_loss_legs = [
        uma.ExitLeg(name="stage_post_tp1_stop", note="sl", when_all=[uma.Condition(type="price_le", level=99000.0, note="sl")], close_fraction=1.0),
    ]
    playbook = uma.GenericPlaybook(
        display_answer="display",
        current_bias="bullish",
        trigger_confidence=0.0,
        selected_symbol="BTC",
        selection_reason="BTC best",
        entry_plan=uma.EntryPlan(
            execute_now=True,
            action_decision=make_entry_decision(uma, action="long", notional=1000.0, stop_loss_price=99500.0, leverage=0),
            scenario=None,
        ),
    )
    agent = make_agent(uma)
    agent.risk_session = uma.RiskSession(
        plan_name="position_management",
        side="long",
        initial_entry_price=100000.0,
        stop_loss_price=99000.0,
        baseline_size=1.0,
        expected_size=1.0,
        initial_size_abs=1.0,
        position_management=existing_management,
        take_profit_legs=list(existing_take_profit_legs),
        stop_loss_legs=list(existing_stop_loss_legs),
        runtimes={},
        history=deque(),
    )

    materialized = agent._materialize_live_position_management_from_entry_plan(playbook, dict(open_snapshot))

    assert getattr(materialized.position_management, "risk_session_passthrough", False) is True
    assert materialized.position_management.action_decision.stop_loss_price == pytest.approx(0.0)
    assert materialized.position_management.action_decision.leverage == 0


def test_startup_live_tpsl_restore_rebuilds_staged_session_and_resyncs_old_orders(uma):
    open_snapshot = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "symbol": "BTC",
        "side": "long",
        "size": 3.0,
        "entry_price": 100000.0,
        "mid_price": 100500.0,
        "notional_usd": 3000.0,
        "leverage": 6.0,
        "max_leverage": 20,
        "remaining_capital_usd": 300.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "margin_used": 500.0,
    }
    reader = FakeReader(
        symbol_snapshots=[dict(open_snapshot)],
        frontend_open_orders=[
            {
                "coin": "BTC",
                "reduceOnly": True,
                "isTrigger": False,
                "orderType": "Limit",
                "limitPx": "101000",
                "origSz": "1.0",
                "oid": 1001,
                "cloid": "0xtp1",
            },
            {
                "coin": "BTC",
                "reduceOnly": True,
                "isTrigger": True,
                "tpsl": "sl",
                "triggerPx": "99000",
                "origSz": "3.0",
                "oid": 1002,
                "cloid": "0xsl1",
            },
        ],
    )
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    cancel_calls = []
    place_calls = []

    def fake_cancel(order_refs, plan_name=None):
        cancel_calls.append((plan_name, [dict(item) for item in order_refs]))
        return {"accepted": True}

    def fake_place(side, close_size, trigger_price, tpsl, plan_name=None, leg_name=None):
        place_calls.append(
            {
                "side": side,
                "close_size": close_size,
                "trigger_price": trigger_price,
                "tpsl": tpsl,
                "plan_name": plan_name,
                "leg_name": leg_name,
            }
        )
        return {"accepted": True, "oid": 2000 + len(place_calls), "cloid": f"0xnew{len(place_calls)}", "close_size": close_size}

    executor.cancel_reduce_only_tpsl_orders = fake_cancel
    executor.place_reduce_only_tpsl_order = fake_place
    agent = make_agent(uma, reader=reader, executor=executor)
    agent.symbol = ""
    executor.symbol = ""
    agent._audit_event = lambda *args, **kwargs: None

    restored = agent._maybe_restore_startup_live_tpsl(open_snapshot)

    assert restored is True
    assert agent.symbol == "BTC"
    assert executor.symbol == "BTC"
    assert agent.risk_session is not None
    assert agent.risk_session.staged_exit_enabled is True
    assert agent.risk_session.tp1_hit is False
    assert agent.risk_session.tp2_hit is False
    assert agent.risk_session.initial_entry_price == pytest.approx(100000.0)
    assert getattr(agent.risk_session, "risk_entry_source", "") == "position_entry_price"
    assert agent.risk_session.initial_stop_price == pytest.approx(99000.0)
    assert agent.risk_session.initial_risk_price_distance == pytest.approx(1000.0)
    assert agent.risk_session.tp1_price == pytest.approx(101000.0)
    assert agent.risk_session.tp2_price == pytest.approx(102000.0)
    assert [leg.name for leg in agent.risk_session.take_profit_legs] == ["stage_tp1", "stage_tp2"]
    assert agent.risk_session.stop_loss_legs == []
    assert agent.risk_session.use_resting_exit_orders is True
    assert agent.risk_session.tp2_price == pytest.approx(102000.0)
    assert agent.risk_session.stop_loss_price == pytest.approx(99000.0)
    assert len(agent.risk_session.resting_exit_orders) == 2
    assert {item["key"] for item in agent.risk_session.resting_exit_orders} == {
        "take_profit::stage_tp1",
        "take_profit::stage_tp2",
    }
    assert len(cancel_calls) == 1
    assert len(cancel_calls[0][1]) == 2
    assert [(item["leg_name"], item["tpsl"], item["trigger_price"], item["close_size"]) for item in place_calls] == [
        ("stage_tp1", "tp", 101000.0, pytest.approx(0.9)),
        ("stage_tp2", "tp", 102000.0, pytest.approx(1.2)),
    ]

    agent.risk_session = None
    restored_again = agent._maybe_restore_startup_live_tpsl(open_snapshot)

    assert restored_again is False
    assert agent.risk_session is None


def test_startup_live_tpsl_restore_keeps_existing_staged_order_refs(uma):
    open_snapshot = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "symbol": "BTC",
        "side": "long",
        "size": 3.0,
        "entry_price": 100000.0,
        "mid_price": 100500.0,
        "notional_usd": 3000.0,
        "leverage": 6.0,
        "max_leverage": 20,
        "remaining_capital_usd": 300.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "margin_used": 500.0,
    }
    reader = FakeReader(
        symbol_snapshots=[dict(open_snapshot)],
        frontend_open_orders=[
            {"coin": "BTC", "reduceOnly": True, "isTrigger": False, "orderType": "Limit", "limitPx": "101000", "origSz": "0.9", "oid": 1001, "cloid": "0xtp1"},
            {"coin": "BTC", "reduceOnly": True, "isTrigger": False, "orderType": "Limit", "limitPx": "102000", "origSz": "1.2", "oid": 1002, "cloid": "0xtp2"},
            {"coin": "BTC", "reduceOnly": True, "isTrigger": True, "tpsl": "sl", "triggerPx": "99000", "origSz": "3.0", "oid": 1003, "cloid": "0xsl1"},
        ],
    )
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    calls = []
    executor.cancel_reduce_only_tpsl_orders = lambda *args, **kwargs: calls.append(("cancel", args, kwargs)) or {"accepted": True}
    executor.place_reduce_only_limit_order = (
        lambda side, close_size, limit_price, plan_name=None, leg_name="": calls.append(("limit", leg_name, limit_price, close_size))
        or {"accepted": True, "close_size": close_size, "limit_price": limit_price, "trigger_price": limit_price, "order_kind": "limit", "is_trigger": False}
    )
    executor.place_reduce_only_tpsl_order = lambda *args, **kwargs: calls.append(("place", args, kwargs)) or {"accepted": True, "order_kind": "trigger", "is_trigger": True}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent._audit_event = lambda *args, **kwargs: None

    restored = agent._maybe_restore_startup_live_tpsl(open_snapshot)

    assert restored is True
    assert agent.symbol == "BTC"
    assert executor.symbol == "BTC"
    assert agent.risk_session is not None
    assert agent.risk_session.staged_exit_enabled is True
    assert agent.risk_session.use_resting_exit_orders is True
    assert {item["key"] for item in agent.risk_session.resting_exit_orders} == {
        "take_profit::stage_tp1",
        "take_profit::stage_tp2",
    }
    assert calls[0][0] == "cancel"
    assert [item[0] for item in calls[1:]] == ["limit", "limit"]


def test_startup_live_tpsl_restore_resyncs_legacy_trigger_take_profits(uma):
    open_snapshot = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "symbol": "BTC",
        "side": "long",
        "size": 3.0,
        "entry_price": 100000.0,
        "mid_price": 100500.0,
        "notional_usd": 3000.0,
        "leverage": 6.0,
        "max_leverage": 20,
        "remaining_capital_usd": 300.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "margin_used": 500.0,
    }
    reader = FakeReader(
        symbol_snapshots=[dict(open_snapshot)],
        frontend_open_orders=[
            {"coin": "BTC", "reduceOnly": True, "isTrigger": True, "tpsl": "tp", "triggerPx": "101000", "origSz": "0.9", "oid": 1001, "cloid": "0xtp1"},
            {"coin": "BTC", "reduceOnly": True, "isTrigger": True, "tpsl": "tp", "triggerPx": "102000", "origSz": "1.2", "oid": 1002, "cloid": "0xtp2"},
            {"coin": "BTC", "reduceOnly": True, "isTrigger": True, "tpsl": "sl", "triggerPx": "99000", "origSz": "3.0", "oid": 1003, "cloid": "0xsl1"},
        ],
    )
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    calls = []
    executor.cancel_reduce_only_tpsl_orders = lambda refs, plan_name=None: calls.append(("cancel", [dict(item) for item in refs])) or {"accepted": True}
    executor.place_reduce_only_limit_order = (
        lambda side, close_size, limit_price, plan_name=None, leg_name="": calls.append(("limit", leg_name, limit_price, close_size))
        or {"accepted": True, "close_size": close_size, "limit_price": limit_price, "trigger_price": limit_price, "order_kind": "limit", "is_trigger": False, "oid": 2000 + len(calls)}
    )
    executor.place_reduce_only_tpsl_order = (
        lambda side, close_size, trigger_price, tpsl, plan_name=None, leg_name="": calls.append(("trigger", leg_name, tpsl, trigger_price, close_size))
        or {"accepted": True, "close_size": close_size, "trigger_price": trigger_price, "order_kind": "trigger", "is_trigger": True, "oid": 3000 + len(calls)}
    )
    agent = make_agent(uma, reader=reader, executor=executor)
    agent._audit_event = lambda *args, **kwargs: None

    restored = agent._maybe_restore_startup_live_tpsl(open_snapshot)

    assert restored is True
    assert calls[0][0] == "cancel"
    assert [item[0] for item in calls[1:]] == ["limit", "limit"]
    assert [(item["name"], item["order_kind"]) for item in agent.risk_session.resting_exit_orders] == [
        ("stage_tp1", "limit"),
        ("stage_tp2", "limit"),
    ]


def test_startup_live_tpsl_restore_uses_position_symbol_when_agent_symbol_empty(uma):
    open_snapshot = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "symbol": "BTC",
        "side": "long",
        "size": 3.0,
        "entry_price": 100000.0,
        "mid_price": 100500.0,
        "notional_usd": 3000.0,
        "leverage": 6.0,
        "max_leverage": 20,
        "remaining_capital_usd": 300.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "margin_used": 500.0,
    }
    reader = FakeReader(
        symbol_snapshots=[dict(open_snapshot)],
        frontend_open_orders=[
            {"coin": "BTC", "reduceOnly": True, "isTrigger": True, "tpsl": "tp", "triggerPx": "101000", "origSz": "0.9", "oid": 1001, "cloid": "0xtp1"},
            {"coin": "BTC", "reduceOnly": True, "isTrigger": True, "tpsl": "tp", "triggerPx": "102000", "origSz": "1.2", "oid": 1002, "cloid": "0xtp2"},
            {"coin": "BTC", "reduceOnly": True, "isTrigger": True, "tpsl": "sl", "triggerPx": "99000", "origSz": "3.0", "oid": 1003, "cloid": "0xsl1"},
        ],
    )

    def fake_get_sz_decimals(symbol):
        if not symbol:
            raise KeyError("empty symbol")
        assert symbol == "BTC"
        return 5

    reader.get_sz_decimals = fake_get_sz_decimals
    executor = FakeExecutor(reader, "BTC")
    agent = make_agent(uma, reader=reader, executor=executor)
    agent.symbol = ""
    agent._audit_event = lambda *args, **kwargs: None

    restored = agent._maybe_restore_startup_live_tpsl(open_snapshot)

    assert restored is True
    assert agent.risk_session is not None
    assert agent.risk_session.staged_exit_enabled is True


def test_startup_live_tpsl_restore_skips_unconfigured_position_symbol(uma):
    open_snapshot = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "symbol": "ETH",
        "side": "long",
        "size": 3.0,
        "entry_price": 3000.0,
        "mid_price": 3010.0,
        "notional_usd": 9030.0,
        "leverage": 6.0,
        "max_leverage": 20,
        "remaining_capital_usd": 300.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "margin_used": 500.0,
    }
    reader = FakeReader(symbol_snapshots=[dict(open_snapshot)])
    executor = FakeExecutor(reader, "BTC")
    agent = make_agent(uma, reader=reader, executor=executor)
    agent.engine.symbol = "BTC"
    audit_events = []
    agent._audit_event = lambda event_type, payload=None: audit_events.append((event_type, payload or {}))

    restored = agent._maybe_restore_startup_live_tpsl(open_snapshot)

    assert restored is False
    assert agent.symbol == "BTC"
    assert agent.engine.symbol == "BTC"
    assert executor.symbol == "BTC"
    assert agent.risk_session is None
    assert audit_events == [
        (
            "startup_live_tpsl_restore_skipped_unconfigured_symbol",
            {
                "configured_symbols": ["BTC"],
                "position_symbol": "ETH",
                "position_snapshot": open_snapshot,
            },
        )
    ]


def test_startup_live_tpsl_restore_skips_unconfigured_position_symbol_when_active_symbol_empty(uma):
    open_snapshot = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "symbol": "ETH",
        "side": "long",
        "size": 3.0,
        "entry_price": 3000.0,
        "mid_price": 3010.0,
        "notional_usd": 9030.0,
        "leverage": 6.0,
        "max_leverage": 20,
        "remaining_capital_usd": 300.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "margin_used": 500.0,
    }
    reader = FakeReader(symbol_snapshots=[dict(open_snapshot)])
    executor = FakeExecutor(reader, "")
    agent = make_agent(uma, reader=reader, executor=executor)
    agent.symbol = ""
    agent.engine.symbol = ""
    audit_events = []
    agent._audit_event = lambda event_type, payload=None: audit_events.append((event_type, payload or {}))

    restored = agent._maybe_restore_startup_live_tpsl(open_snapshot)

    assert restored is False
    assert agent.symbol == ""
    assert agent.engine.symbol == ""
    assert executor.symbol == ""
    assert agent.risk_session is None
    assert audit_events == [
        (
            "startup_live_tpsl_restore_skipped_unconfigured_symbol",
            {
                "configured_symbols": ["BTC"],
                "position_symbol": "ETH",
                "position_snapshot": open_snapshot,
            },
        )
    ]


def test_startup_live_tpsl_restore_rebuilds_post_tp1_session_without_reinitializing(uma):
    open_snapshot = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "symbol": "BTC",
        "side": "long",
        "size": 2.1,
        "entry_price": 100000.0,
        "mid_price": 101500.0,
        "notional_usd": 2100.0,
        "leverage": 6.0,
        "max_leverage": 20,
        "remaining_capital_usd": 300.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "margin_used": 350.0,
    }
    reader = FakeReader(
        symbol_snapshots=[dict(open_snapshot)],
        frontend_open_orders=[
            {"coin": "BTC", "reduceOnly": True, "isTrigger": True, "tpsl": "tp", "triggerPx": "102000", "origSz": "1.2", "oid": 1002, "cloid": "0xtp2"},
            {"coin": "BTC", "reduceOnly": True, "isTrigger": True, "tpsl": "sl", "triggerPx": "99600", "origSz": "2.1", "oid": 1003, "cloid": "0xsl2"},
        ],
    )
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    calls = []
    executor.cancel_reduce_only_tpsl_orders = lambda *args, **kwargs: calls.append(("cancel", args, kwargs)) or {"accepted": True}
    executor.place_reduce_only_limit_order = (
        lambda side, close_size, limit_price, plan_name=None, leg_name="": calls.append(("limit", leg_name, limit_price, close_size))
        or {"accepted": True, "close_size": close_size, "limit_price": limit_price, "trigger_price": limit_price, "order_kind": "limit", "is_trigger": False}
    )
    executor.place_reduce_only_tpsl_order = lambda *args, **kwargs: calls.append(("place", args, kwargs)) or {"accepted": True, "order_kind": "trigger", "is_trigger": True}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent._audit_event = lambda *args, **kwargs: None

    restored = agent._maybe_restore_startup_live_tpsl(open_snapshot)

    assert restored is True
    assert agent.risk_session is not None
    assert agent.risk_session.staged_exit_enabled is True
    assert agent.risk_session.tp1_hit is True
    assert agent.risk_session.tp2_hit is False
    assert agent.risk_session.initial_stop_price == pytest.approx(99000.0)
    assert agent.risk_session.initial_risk_price_distance == pytest.approx(1000.0)
    assert agent.risk_session.initial_size_abs == pytest.approx(3.0)
    assert agent.risk_session.tp1_price == pytest.approx(101000.0)
    assert agent.risk_session.tp2_price == pytest.approx(102000.0)
    assert [leg.name for leg in agent.risk_session.take_profit_legs] == ["stage_tp2"]
    assert agent.risk_session.stop_loss_legs == []
    assert agent.risk_session.stop_loss_price == pytest.approx(99600.0)
    assert {item["key"] for item in agent.risk_session.resting_exit_orders} == {
        "take_profit::stage_tp2",
    }
    assert calls[0][0] == "cancel"
    assert [item[0] for item in calls[1:]] == ["limit"]


def test_startup_live_tpsl_restore_rebuilds_tail_session_after_tp2(uma):
    open_snapshot = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "symbol": "BTC",
        "side": "long",
        "size": 0.9,
        "entry_price": 100000.0,
        "mid_price": 102500.0,
        "notional_usd": 900.0,
        "leverage": 6.0,
        "max_leverage": 20,
        "remaining_capital_usd": 300.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "margin_used": 150.0,
    }
    reader = FakeReader(
        symbol_snapshots=[dict(open_snapshot)],
        frontend_open_orders=[
            {"coin": "BTC", "reduceOnly": True, "isTrigger": True, "tpsl": "sl", "triggerPx": "101000", "origSz": "0.9", "oid": 1003, "cloid": "0xsl-tail"},
        ],
    )
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    calls = []
    executor.cancel_reduce_only_tpsl_orders = lambda *args, **kwargs: calls.append(("cancel", args, kwargs)) or {"accepted": True}
    executor.place_reduce_only_tpsl_order = lambda *args, **kwargs: calls.append(("place", args, kwargs)) or {"accepted": True}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent._audit_event = lambda *args, **kwargs: None

    restored = agent._maybe_restore_startup_live_tpsl(open_snapshot)

    assert restored is True
    assert agent.risk_session is not None
    assert agent.risk_session.staged_exit_enabled is True
    assert agent.risk_session.tp1_hit is True
    assert agent.risk_session.tp2_hit is True
    assert agent.risk_session.take_profit_legs == []
    assert agent.risk_session.stop_loss_legs == []
    assert agent.risk_session.locked_floor_price == pytest.approx(101000.0)
    assert agent.risk_session.trailing_soft_stop_price == pytest.approx(101000.0)
    assert agent.risk_session.trailing_hard_stop_price == pytest.approx(0.0)
    assert agent.risk_session.stop_loss_price == pytest.approx(101000.0)
    assert agent.risk_session.resting_exit_orders == []
    assert calls[0][0] == "cancel"


def _short_add_restore_snapshot(*, size=-14.44, entry_price=108.5747, mid_price=108.0):
    return {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "symbol": "BTC",
        "side": "short",
        "size": size,
        "entry_price": entry_price,
        "mid_price": mid_price,
        "notional_usd": abs(size) * mid_price,
        "leverage": 20.0,
        "max_leverage": 40,
        "remaining_capital_usd": 300.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "margin_used": 80.0,
    }


def _short_add_restore_orders(
    *,
    include_tp1=True,
    include_tp2=True,
    sl_price=109.15,
    sl_size=14.44,
    tp1_price=107.11,
    tp2_price=106.09,
    tp1_size=4.332,
    tp2_size=5.776,
    order_time_ms=0,
):
    def with_order_time(order):
        if order_time_ms:
            order["timestamp"] = int(order_time_ms)
        return order

    orders = []
    if include_tp1:
        orders.append(with_order_time({"coin": "BTC", "reduceOnly": True, "isTrigger": True, "tpsl": "tp", "triggerPx": str(tp1_price), "origSz": str(tp1_size), "oid": 4101, "cloid": "0xtp1"}))
    if include_tp2:
        orders.append(with_order_time({"coin": "BTC", "reduceOnly": True, "isTrigger": True, "tpsl": "tp", "triggerPx": str(tp2_price), "origSz": str(tp2_size), "oid": 4102, "cloid": "0xtp2"}))
    orders.append(with_order_time({"coin": "BTC", "reduceOnly": True, "isTrigger": True, "tpsl": "sl", "triggerPx": str(sl_price), "origSz": str(sl_size), "oid": 4103, "cloid": "0xsl"}))
    return orders


def _short_add_restore_low_liquidity_orders(*, order_time_ms=0):
    return _short_add_restore_orders(
        tp1_price=107.365,
        tp2_price=106.855,
        tp1_size=5.776,
        tp2_size=5.776,
        order_time_ms=order_time_ms,
    )


def _write_risk_state(agent, session, path, *, updated_at_ms=None):
    payload = agent._risk_session_state_payload(session)
    if updated_at_ms is not None:
        payload["updated_at_ms"] = int(updated_at_ms)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def _make_short_add_session(uma, agent, snapshot):
    session = agent._build_staged_risk_session_from_stop(
        position_after=dict(snapshot),
        plan_name="position_management",
        initial_entry_price=108.13,
        stop_loss_price=109.15,
        position_management=make_management_plan(uma, now_action=make_management_decision(uma, action="add_to_short", notional=1500.0, stop_loss_price=109.15)),
        risk_entry_source="add_fill_avg_price",
    )
    assert session is not None
    session.resting_exit_orders = [
        {"key": "take_profit::stage_tp1", "name": "stage_tp1", "leg_type": "take_profit", "tpsl": "tp", "trigger_price": 107.11, "close_size": 4.332, "oid": 4101, "cloid": "0xtp1"},
        {"key": "take_profit::stage_tp2", "name": "stage_tp2", "leg_type": "take_profit", "tpsl": "tp", "trigger_price": 106.09, "close_size": 5.776, "oid": 4102, "cloid": "0xtp2"},
        {"key": "stop_loss::stage_initial_stop", "name": "stage_initial_stop", "leg_type": "stop_loss", "tpsl": "sl", "trigger_price": 109.15, "close_size": 14.44, "oid": 4103, "cloid": "0xsl"},
    ]
    session.use_resting_exit_orders = True
    return session


def _make_short_restore_session(uma, agent, snapshot, *, risk_entry_source):
    session = agent._build_staged_risk_session_from_stop(
        position_after=dict(snapshot),
        plan_name="position_management",
        initial_entry_price=108.13,
        stop_loss_price=109.15,
        position_management=make_management_plan(
            uma,
            now_action=make_management_decision(uma, action="no_change", stop_loss_price=109.15),
        ),
        risk_entry_source=risk_entry_source,
    )
    assert session is not None
    session.resting_exit_orders = [
        {"key": "take_profit::stage_tp1", "name": "stage_tp1", "leg_type": "take_profit", "tpsl": "tp", "trigger_price": 107.11, "close_size": 4.332, "oid": 4101, "cloid": "0xtp1"},
        {"key": "take_profit::stage_tp2", "name": "stage_tp2", "leg_type": "take_profit", "tpsl": "tp", "trigger_price": 106.09, "close_size": 5.776, "oid": 4102, "cloid": "0xtp2"},
        {"key": "stop_loss::stage_initial_stop", "name": "stage_initial_stop", "leg_type": "stop_loss", "tpsl": "sl", "trigger_price": 109.15, "close_size": 14.44, "oid": 4103, "cloid": "0xsl"},
    ]
    session.use_resting_exit_orders = True
    return session


def test_startup_live_tpsl_restore_from_state_keeps_position_entry_anchor(uma, tmp_path):
    snapshot = _short_add_restore_snapshot(entry_price=108.13)
    reader = FakeReader(symbol_snapshots=[dict(snapshot)], frontend_open_orders=_short_add_restore_orders())
    agent = make_agent(uma, reader=reader, executor=FakeExecutor(reader, "BTC"))
    agent.risk_session_state_enabled = True
    agent.risk_session_state_path = tmp_path / "risk_session_state.json"
    agent._audit_event = lambda *args, **kwargs: None
    session = _make_short_restore_session(uma, agent, snapshot, risk_entry_source="position_entry_price")
    _write_risk_state(agent, session, agent.risk_session_state_path)

    restored = agent._maybe_restore_startup_live_tpsl(snapshot)

    assert restored is True
    assert agent.risk_session is not None
    assert agent.risk_session.initial_entry_price == pytest.approx(108.13)
    assert getattr(agent.risk_session, "risk_entry_source", "") == "position_entry_price"


def test_startup_live_tpsl_restore_from_state_keeps_strategy_entry_anchor(uma, tmp_path):
    snapshot = _short_add_restore_snapshot(entry_price=108.5747)
    reader = FakeReader(symbol_snapshots=[dict(snapshot)], frontend_open_orders=_short_add_restore_orders())
    agent = make_agent(uma, reader=reader, executor=FakeExecutor(reader, "BTC"))
    agent.risk_session_state_enabled = True
    agent.risk_session_state_path = tmp_path / "risk_session_state.json"
    agent._audit_event = lambda *args, **kwargs: None
    session = _make_short_restore_session(uma, agent, snapshot, risk_entry_source="strategy_entry_price")
    _write_risk_state(agent, session, agent.risk_session_state_path)

    restored = agent._maybe_restore_startup_live_tpsl(snapshot)

    assert restored is True
    assert agent.risk_session is not None
    assert agent.risk_session.initial_entry_price == pytest.approx(108.13)
    assert getattr(agent.risk_session, "risk_entry_source", "") == "strategy_entry_price"


def test_startup_live_tpsl_restore_from_state_keeps_add_fill_anchor(uma, tmp_path):
    snapshot = _short_add_restore_snapshot()
    reader = FakeReader(symbol_snapshots=[dict(snapshot)], frontend_open_orders=_short_add_restore_orders())
    executor = FakeExecutor(reader, "BTC")
    agent = make_agent(uma, reader=reader, executor=executor)
    agent.risk_session_state_enabled = True
    agent.risk_session_state_path = tmp_path / "risk_session_state.json"
    agent._audit_event = lambda *args, **kwargs: None
    session = _make_short_add_session(uma, agent, snapshot)
    _write_risk_state(agent, session, agent.risk_session_state_path, updated_at_ms=1_777_650_000_000)

    restored = agent._maybe_restore_startup_live_tpsl(snapshot)

    assert restored is True
    assert agent.risk_session is not None
    assert agent.risk_session.initial_entry_price == pytest.approx(108.13)
    assert getattr(agent.risk_session, "risk_entry_source", "") == "add_fill_avg_price"
    assert agent.risk_session.tp1_price == pytest.approx(107.11)
    assert agent.risk_session.tp2_price == pytest.approx(106.09)
    assert agent.risk_session.stop_loss_price == pytest.approx(109.15)


def test_startup_live_tpsl_restore_without_state_uses_recent_add_fill_anchor(uma):
    snapshot = _short_add_restore_snapshot()
    reader = FakeReader(
        symbol_snapshots=[dict(snapshot)],
        frontend_open_orders=_short_add_restore_orders(),
        user_fills_by_time=[
            {"coin": "BTC", "oid": 5001, "px": "108.13", "sz": "9.30", "dir": "Open Short", "time": 1_777_650_000_000},
        ],
    )
    agent = make_agent(uma, reader=reader, executor=FakeExecutor(reader, "BTC"))
    agent.risk_session_restore_fill_lookback_seconds = 10**9
    agent._audit_event = lambda *args, **kwargs: None

    restored = agent._maybe_restore_startup_live_tpsl(snapshot)

    assert restored is True
    assert agent.risk_session is not None
    assert agent.risk_session.initial_entry_price == pytest.approx(108.13)
    assert getattr(agent.risk_session, "risk_entry_source", "") == "add_fill_avg_price"


def test_startup_live_tpsl_restore_without_state_can_infer_strategy_entry_anchor(uma):
    snapshot = _short_add_restore_snapshot()
    reader = FakeReader(
        symbol_snapshots=[dict(snapshot)],
        frontend_open_orders=_short_add_restore_orders(),
    )
    agent = make_agent(uma, reader=reader, executor=FakeExecutor(reader, "BTC"))
    agent._audit_event = lambda *args, **kwargs: None

    restored = agent._maybe_restore_startup_live_tpsl(snapshot)

    assert restored is True
    assert agent.risk_session is not None
    assert agent.risk_session.initial_entry_price == pytest.approx(108.13)
    assert getattr(agent.risk_session, "risk_entry_source", "") == "strategy_entry_price"


def test_startup_live_tpsl_restore_strategy_anchor_uses_open_order_time_for_low_liquidity_params(uma):
    snapshot = _short_add_restore_snapshot()
    low_time_ms = int(datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
    reader = FakeReader(
        symbol_snapshots=[dict(snapshot)],
        frontend_open_orders=_short_add_restore_low_liquidity_orders(order_time_ms=low_time_ms),
    )
    agent = make_agent(uma, reader=reader, executor=FakeExecutor(reader, "BTC"))
    agent.risk_session_restore_fill_lookback_seconds = 10**9
    install_profile_r_clip_fixture(uma, agent)
    original_params = agent._staged_exit_params_for_symbol
    agent._staged_exit_params_for_symbol = lambda symbol, now_utc=None: original_params(
        symbol,
        now_utc=now_utc or datetime(2026, 4, 29, 12, 0, tzinfo=timezone.utc),
    )
    agent._audit_event = lambda *args, **kwargs: None

    restored = agent._maybe_restore_startup_live_tpsl(snapshot)

    assert restored is True
    assert agent.risk_session is not None
    assert agent.risk_session.initial_entry_price == pytest.approx(108.13)
    assert getattr(agent.risk_session, "risk_entry_source", "") == "strategy_entry_price"
    assert agent.risk_session.tp1_price == pytest.approx(107.365)
    assert agent.risk_session.tp2_price == pytest.approx(106.855)
    assert agent.risk_session.trailing_soft_atr_mult == pytest.approx(1.5)
    assert agent.risk_session.trailing_hard_atr_mult == pytest.approx(2.5)


def test_startup_live_tpsl_restore_strategy_anchor_without_order_time_tries_normal_and_low_variants(uma):
    snapshot = _short_add_restore_snapshot()
    reader = FakeReader(
        symbol_snapshots=[dict(snapshot)],
        frontend_open_orders=_short_add_restore_orders(),
    )
    agent = make_agent(uma, reader=reader, executor=FakeExecutor(reader, "BTC"))
    install_profile_r_clip_fixture(uma, agent)
    original_params = agent._staged_exit_params_for_symbol
    agent._staged_exit_params_for_symbol = lambda symbol, now_utc=None: original_params(
        symbol,
        now_utc=now_utc or datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc),
    )
    agent._audit_event = lambda *args, **kwargs: None

    restored = agent._maybe_restore_startup_live_tpsl(snapshot)

    assert restored is True
    assert agent.risk_session is not None
    assert agent.risk_session.initial_entry_price == pytest.approx(108.13)
    assert getattr(agent.risk_session, "risk_entry_source", "") == "strategy_entry_price"
    assert agent.risk_session.tp1_price == pytest.approx(107.11)
    assert agent.risk_session.tp2_price == pytest.approx(106.09)
    assert agent.risk_session.trailing_soft_atr_mult == pytest.approx(2.5)
    assert agent.risk_session.trailing_hard_atr_mult == pytest.approx(3.5)


def test_startup_live_tpsl_restore_add_anchor_uses_fill_time_for_low_liquidity_params(uma):
    snapshot = _short_add_restore_snapshot()
    low_time_ms = int(datetime(2026, 5, 2, 12, 0, tzinfo=timezone.utc).timestamp() * 1000)
    reader = FakeReader(
        symbol_snapshots=[dict(snapshot)],
        frontend_open_orders=_short_add_restore_low_liquidity_orders(),
        user_fills_by_time=[
            {"coin": "BTC", "oid": 5001, "px": "108.13", "sz": "9.30", "dir": "Open Short", "time": low_time_ms},
        ],
    )
    agent = make_agent(uma, reader=reader, executor=FakeExecutor(reader, "BTC"))
    install_profile_r_clip_fixture(uma, agent)
    original_params = agent._staged_exit_params_for_symbol
    agent._staged_exit_params_for_symbol = lambda symbol, now_utc=None: original_params(
        symbol,
        now_utc=now_utc or datetime(2026, 4, 29, 12, 0, tzinfo=timezone.utc),
    )
    agent._audit_event = lambda *args, **kwargs: None

    restored = agent._maybe_restore_startup_live_tpsl(snapshot)

    assert restored is True
    assert agent.risk_session is not None
    assert agent.risk_session.initial_entry_price == pytest.approx(108.13)
    assert getattr(agent.risk_session, "risk_entry_source", "") in {"add_fill_avg_price", "strategy_entry_price"}
    assert agent.risk_session.tp1_price == pytest.approx(107.365)
    assert agent.risk_session.tp2_price == pytest.approx(106.855)


def test_startup_live_tpsl_restore_from_state_replays_tp1_fill_and_resyncs_orders(uma, tmp_path):
    snapshot = _short_add_restore_snapshot(size=-10.108, entry_price=108.5747, mid_price=107.0)
    reader = FakeReader(
        symbol_snapshots=[dict(snapshot)],
        frontend_open_orders=_short_add_restore_orders(include_tp1=False, include_tp2=True, sl_price=109.15, sl_size=14.44),
        user_fills_by_time=[
            {"coin": "BTC", "oid": 4101, "cloid": "0xtp1", "px": "107.11", "sz": "4.332", "dir": "Close Short", "time": 1_777_650_010_000},
        ],
    )
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    cancel_calls = []
    place_calls = []
    executor.cancel_reduce_only_tpsl_orders = lambda order_refs, plan_name=None: cancel_calls.append([dict(item) for item in order_refs]) or {"accepted": True}
    executor.place_reduce_only_tpsl_order = lambda **kwargs: place_calls.append(dict(kwargs)) or {"accepted": True, "oid": 5000 + len(place_calls), "cloid": f"0xnew{len(place_calls)}", "close_size": kwargs["close_size"]}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent.risk_session_state_enabled = True
    agent.risk_session_state_path = tmp_path / "risk_session_state.json"
    agent._audit_event = lambda *args, **kwargs: None
    session = _make_short_add_session(uma, agent, _short_add_restore_snapshot())
    _write_risk_state(agent, session, agent.risk_session_state_path)

    restored = agent._maybe_restore_startup_live_tpsl(snapshot)

    assert restored is True
    assert agent.risk_session is not None
    assert agent.risk_session.tp1_hit is True
    assert agent.risk_session.tp2_hit is False
    assert agent.risk_session.stop_loss_price == pytest.approx(108.538)
    assert cancel_calls
    assert [(item["leg_name"], item["tpsl"], item["trigger_price"]) for item in place_calls] == [
        ("stage_tp2", "tp", pytest.approx(106.09)),
    ]


def test_startup_live_tpsl_restore_without_state_replays_add_and_tp1_fills(uma):
    snapshot = _short_add_restore_snapshot(size=-10.108, entry_price=108.5747, mid_price=107.0)
    reader = FakeReader(
        symbol_snapshots=[dict(snapshot)],
        frontend_open_orders=_short_add_restore_orders(include_tp1=False, include_tp2=True, sl_price=109.15, sl_size=14.44),
        user_fills_by_time=[
            {"coin": "BTC", "oid": 5001, "px": "108.13", "sz": "14.44", "dir": "Open Short", "time": 1_777_650_000_000},
            {"coin": "BTC", "oid": 4101, "cloid": "0xtp1", "px": "107.11", "sz": "4.332", "dir": "Close Short", "time": 1_777_650_010_000},
        ],
    )
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    place_calls = []
    executor.cancel_reduce_only_tpsl_orders = lambda order_refs, plan_name=None: {"accepted": True}
    executor.place_reduce_only_tpsl_order = lambda **kwargs: place_calls.append(dict(kwargs)) or {"accepted": True, "oid": 6000 + len(place_calls), "cloid": f"0xnew{len(place_calls)}", "close_size": kwargs["close_size"]}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent.risk_session_restore_fill_lookback_seconds = 10**9
    agent._audit_event = lambda *args, **kwargs: None

    restored = agent._maybe_restore_startup_live_tpsl(snapshot)

    assert restored is True
    assert agent.risk_session is not None
    assert agent.risk_session.initial_entry_price == pytest.approx(108.13)
    assert getattr(agent.risk_session, "risk_entry_source", "") == "add_fill_avg_price"
    assert agent.risk_session.tp1_hit is True
    assert agent.risk_session.stop_loss_price == pytest.approx(108.538)
    assert {item["leg_name"] for item in place_calls} == {"stage_tp2"}


def test_startup_live_tpsl_restore_from_state_replays_tp1_and_tp2_to_tail(uma, tmp_path):
    snapshot = _short_add_restore_snapshot(size=-4.332, entry_price=108.5747, mid_price=106.0)
    reader = FakeReader(
        symbol_snapshots=[dict(snapshot)],
        frontend_open_orders=[
            {"coin": "BTC", "reduceOnly": True, "isTrigger": True, "tpsl": "sl", "triggerPx": "109.15", "origSz": "14.44", "oid": 4103, "cloid": "0xsl"},
        ],
        user_fills_by_time=[
            {"coin": "BTC", "oid": 4101, "cloid": "0xtp1", "px": "107.11", "sz": "4.332", "dir": "Close Short", "time": 1_777_650_010_000},
            {"coin": "BTC", "oid": 4102, "cloid": "0xtp2", "px": "106.09", "sz": "5.776", "dir": "Close Short", "time": 1_777_650_020_000},
        ],
    )
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    place_calls = []
    executor.cancel_reduce_only_tpsl_orders = lambda order_refs, plan_name=None: {"accepted": True}
    executor.place_reduce_only_tpsl_order = lambda **kwargs: place_calls.append(dict(kwargs)) or {"accepted": True, "oid": 7000 + len(place_calls), "cloid": f"0xnew{len(place_calls)}", "close_size": kwargs["close_size"]}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent.risk_session_state_enabled = True
    agent.risk_session_state_path = tmp_path / "risk_session_state.json"
    agent._audit_event = lambda *args, **kwargs: None
    session = _make_short_add_session(uma, agent, _short_add_restore_snapshot())
    _write_risk_state(agent, session, agent.risk_session_state_path, updated_at_ms=1_777_650_000_000)

    restored = agent._maybe_restore_startup_live_tpsl(snapshot)

    assert restored is True
    assert agent.risk_session is not None
    assert agent.risk_session.tp1_hit is True
    assert agent.risk_session.tp2_hit is True
    assert agent.risk_session.take_profit_legs == []
    assert agent.risk_session.stop_loss_legs == []
    assert place_calls == []


def test_startup_live_tpsl_restore_clears_state_when_position_is_flat(uma, tmp_path):
    open_snapshot = _short_add_restore_snapshot()
    flat_snapshot = {**open_snapshot, "side": "flat", "size": 0.0, "notional_usd": 0.0}
    reader = FakeReader(symbol_snapshots=[dict(flat_snapshot)], frontend_open_orders=[])
    agent = make_agent(uma, reader=reader, executor=FakeExecutor(reader, "BTC"))
    agent.risk_session_state_enabled = True
    agent.risk_session_state_path = tmp_path / "risk_session_state.json"
    agent._audit_event = lambda *args, **kwargs: None
    session = _make_short_add_session(uma, agent, open_snapshot)
    _write_risk_state(agent, session, agent.risk_session_state_path)

    restored = agent._maybe_restore_startup_live_tpsl(flat_snapshot)

    assert restored is False
    assert agent.risk_session is None
    assert not agent.risk_session_state_path.exists()


def test_active_query_allowed_now_respects_active_playbook_disabled(uma):
    agent = make_agent(uma)
    agent.enable_active_query = True
    agent.enable_active_auto_requery = True
    agent.enable_active_playbook = False
    agent.risk_session = None
    agent.position_management_session = None

    assert agent.active_query_allowed_now() is False


def test_query_new_playbook_live_position_entry_only_smaller_same_side_materializes_trim(uma):
    open_snapshot = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "symbol": "BTC",
        "side": "long",
        "size": 1.0,
        "entry_price": 100000.0,
        "mid_price": 100500.0,
        "notional_usd": 1000.0,
        "leverage": 5.0,
        "max_leverage": 20,
        "remaining_capital_usd": 300.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "margin_used": 20.0,
    }
    all_positions = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "margin_summary": {},
        "cross_margin_summary": {},
        "account_equity_usd": 500.0,
        "total_margin_used_usd": 20.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "remaining_capital_usd": 300.0,
        "remaining_capital_source": "withdrawable",
        "positions": [dict(open_snapshot)],
        "positions_count": 1,
        "total_notional_usd": 1000.0,
    }
    playbook = uma.GenericPlaybook(
        display_answer="display",
        current_bias="neutral",
        selected_symbol="BTC",
        selection_reason="BTC best",
        entry_plan=uma.EntryPlan(
            execute_now=True,
            action_decision=make_entry_decision(uma, action="long", notional=700.0, stop_loss_price=99000.0, leverage=0),
            scenario=None,
        ),
    )
    reader = FakeReader(all_positions=all_positions, symbol_snapshots=[dict(open_snapshot), dict(open_snapshot)])
    engine = FakeEngine(playbook)
    agent = make_agent(uma, playbook=playbook, reader=reader, engine=engine)
    agent._derive_local_sizing = lambda **kwargs: {
        "allowed": True,
        "suggested_notional_usd": 700.0,
        "requested_leverage": 4,
        "margin_basis_usd": 175.0,
        "max_planned_loss_usd": 100.0,
    }
    seen = {}

    def fake_execute_management(decision, plan_name, management_plan=None, trigger_confidence_raw=None, **_kwargs):
        seen["action"] = decision.action
        seen["close_fraction"] = decision.close_fraction
        return {"accepted": True, "position_after": dict(open_snapshot, notional_usd=700.0)}

    agent.execute_management_decision = fake_execute_management

    agent.query_new_playbook("manual_once", None)

    assert seen["action"] == "trim"
    assert seen["close_fraction"] == pytest.approx(0.3)
    assert agent.current_playbook.position_management.execute_now is True


def test_query_new_playbook_live_position_entry_only_opposite_immediate_materializes_reverse(uma):
    open_snapshot = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "symbol": "BTC",
        "side": "long",
        "size": 1.0,
        "entry_price": 100000.0,
        "mid_price": 100500.0,
        "notional_usd": 1000.0,
        "leverage": 5.0,
        "max_leverage": 20,
        "remaining_capital_usd": 300.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "margin_used": 20.0,
    }
    all_positions = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "margin_summary": {},
        "cross_margin_summary": {},
        "account_equity_usd": 500.0,
        "total_margin_used_usd": 20.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "remaining_capital_usd": 300.0,
        "remaining_capital_source": "withdrawable",
        "positions": [dict(open_snapshot)],
        "positions_count": 1,
        "total_notional_usd": 1000.0,
    }
    playbook = uma.GenericPlaybook(
        display_answer="display",
        current_bias="bearish",
        selected_symbol="BTC",
        selection_reason="BTC best",
        entry_plan=uma.EntryPlan(
            execute_now=True,
            action_decision=make_entry_decision(uma, action="short", notional=800.0, stop_loss_price=101000.0, leverage=0),
            scenario=None,
        ),
    )
    reader = FakeReader(all_positions=all_positions, symbol_snapshots=[dict(open_snapshot), dict(open_snapshot)])
    engine = FakeEngine(playbook)
    agent = make_agent(uma, playbook=playbook, reader=reader, engine=engine)
    agent._derive_local_sizing = lambda **kwargs: {
        "allowed": True,
        "suggested_notional_usd": 800.0,
        "requested_leverage": 4,
        "margin_basis_usd": 200.0,
        "max_planned_loss_usd": 100.0,
    }
    seen = {}

    def fake_execute_management(decision, plan_name, management_plan=None, trigger_confidence_raw=None, **_kwargs):
        seen["action"] = decision.action
        seen["new_notional_usd"] = decision.new_notional_usd
        return {"accepted": True, "position_after": dict(open_snapshot, side="short", size=-0.8, notional_usd=800.0)}

    agent.execute_management_decision = fake_execute_management

    agent.query_new_playbook("manual_once", None)

    assert seen["action"] == "reverse_to_short"
    assert seen["new_notional_usd"] == pytest.approx(800.0)
    assert agent.current_playbook.post_fill_risk_template.action_decision.stop_loss_price == pytest.approx(101000.0)


def test_non_immediate_reverse_closes_then_refreshes_position_management_session(uma):
    open_snapshot = {
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "symbol": "BTC",
        "side": "long",
        "size": 1.0,
        "entry_price": 100000.0,
        "mid_price": 100500.0,
        "notional_usd": 1000.0,
        "leverage": 5.0,
        "max_leverage": 20,
        "remaining_capital_usd": 300.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "margin_used": 20.0,
    }
    flat_snapshot = dict(open_snapshot, side="flat", size=0.0, entry_price=0.0, notional_usd=0.0, leverage=0.0)
    refreshed_entry_scenario = uma.EntryScenario(
        observe_when_all=[],
        execute_when_all={
            "condition": {"type": "price_ge", "level": 999999.0, "low": 0.0, "high": 0.0, "timer_seconds": 0, "tolerance_bps": 0.0, "min_ratio": 0.0},
            "timeout_seconds": 600,
        },
    )
    playbook = uma.GenericPlaybook(
        display_answer="display",
        current_bias="bearish",
        selected_symbol="BTC",
        selection_reason="BTC best",
        entry_plan=uma.EntryPlan(
            execute_now=False,
            action_decision=make_entry_decision(uma, action="no_trade", notional=0.0, stop_loss_price=0.0, leverage=0),
            scenario=refreshed_entry_scenario,
        ),
    )
    reader = FakeReader(all_positions={
        "known": True,
        "account_address": "0xabc",
        "network": "mainnet",
        "margin_summary": {},
        "cross_margin_summary": {},
        "account_equity_usd": 500.0,
        "total_margin_used_usd": 0.0,
        "available_margin_usd": 300.0,
        "withdrawable_usd": 300.0,
        "remaining_capital_usd": 300.0,
        "remaining_capital_source": "withdrawable",
        "positions": [],
        "positions_count": 0,
        "total_notional_usd": 0.0,
    }, symbol_snapshots=[dict(open_snapshot)])
    agent = make_agent(uma, playbook=playbook, reader=reader)
    agent.current_playbook = playbook
    agent._derive_local_sizing = lambda **kwargs: {
        "allowed": True,
        "suggested_notional_usd": 800.0,
        "requested_leverage": 4,
        "margin_basis_usd": 200.0,
        "max_planned_loss_usd": 100.0,
    }
    close_then_continue = uma.ManagementDecision(
        action="close",
        close_fraction=1.0,
        new_notional_usd=0.0,
        entry_price=0.0,
        stop_loss_price=0.0,
        planned_max_loss_usd=0.0,
        leverage=0,
        margin_basis_usd=0.0,
        continue_entry_plan_after_close=True,
    )
    scenario = uma.Scenario(
        observe_when_all=[],
        execute_when_all={
            "condition": {"type": "price_ge", "level": 100000.0, "low": 0.0, "high": 0.0, "timer_seconds": 0, "tolerance_bps": 0.0, "min_ratio": 0.0},
            "timeout_seconds": 600,
        },
    )
    agent.position_management_session = uma.PositionManagementSession(
        plan_name="position_management",
        side="long",
        position_management=make_management_plan(uma, scenario=scenario, now_action=close_then_continue),
        start_time=1.0,
        baseline_size=1.0,
        expected_size=1.0,
        initial_size_abs=1.0,
        runtimes={uma.SCENARIO_RUNTIME_KEY: uma.ScenarioRuntime()},
        history=deque(),
        history_seconds=1800,
    )
    seen = {}

    def fake_execute_management(decision, plan_name, management_plan=None, trigger_confidence_raw=None, **_kwargs):
        seen["action"] = decision.action
        seen["continue"] = decision.continue_entry_plan_after_close
        return {"accepted": True, "position_after": dict(flat_snapshot)}

    agent.execute_management_decision = fake_execute_management

    result = agent.step_position_management_session(dict(open_snapshot), 1000.0)

    assert result is None
    assert seen == {"action": "close", "continue": True}
    assert agent.risk_session is None
    assert agent.position_management_session is None


def test_execute_immediate_playbook_action_keeps_existing_risk_session_when_effective_action_is_no_change(uma):
    open_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 1.0,
        "entry_price": 100000.0,
        "mid_price": 100500.0,
        "notional_usd": 1000.0,
    }
    requested_decision = make_management_decision(
        uma,
        action="reverse_to_short",
        notional=800.0,
        leverage=10,
        stop_loss_price=101000.0,
    )
    management_plan = make_management_plan(
        uma,
        execute_now=True,
        now_action=requested_decision,
        stop_price=101000.0,
        stop_type="price_ge",
    )
    post_fill_risk_template = make_management_plan(
        uma,
        execute_now=False,
        now_action=make_management_decision(uma, action="short", notional=800.0, leverage=10, stop_loss_price=101000.0),
        stop_price=101000.0,
        stop_type="price_ge",
    )
    playbook = make_playbook(
        uma,
        management_plan=management_plan,
        post_fill_risk_template=post_fill_risk_template,
    )
    reader = FakeReader(symbol_snapshots=[dict(open_snapshot)])
    agent = make_agent(uma, playbook=playbook, reader=reader)
    existing_risk_session = uma.RiskSession(
        plan_name="post_fill_risk_template",
        side="long",
        initial_entry_price=100000.0,
        start_time=1.0,
        baseline_size=1.0,
        expected_size=1.0,
        initial_size_abs=1.0,
        position_management=make_management_plan(uma),
        take_profit_legs=[],
        stop_loss_legs=[],
        runtimes={},
        history=deque(),
        history_seconds=1800,
    )
    existing_position_management_session = uma.PositionManagementSession(
        plan_name="position_management",
        side="long",
        position_management=management_plan,
        start_time=1.0,
        baseline_size=1.0,
        expected_size=1.0,
        initial_size_abs=1.0,
        runtimes={},
        history=deque(),
        history_seconds=1800,
    )
    agent.risk_session = existing_risk_session
    agent.position_management_session = existing_position_management_session
    seen = {}

    def fake_execute_management(decision, plan_name, management_plan=None, trigger_confidence_raw=None, **_kwargs):
        return {
            "accepted": True,
            "position_after": dict(open_snapshot),
            "requested_decision": requested_decision.to_dict(),
            "decision": {
                "action": "no_change",
                "close_fraction": 0.0,
                "new_notional_usd": 1000.0,
                "leverage": 10,
                "entry_price": 100000.0,
                "stop_loss_price": 101000.0,
                "planned_max_loss_usd": 0.0,
                "margin_basis_usd": 0.0,
            },
            "message": "Scaled reverse target is below 60% of current notional; no change.",
        }

    agent.execute_management_decision = fake_execute_management
    agent._set_risk_session_after_management_decision = lambda *args, **kwargs: seen.setdefault("risk_session_reset", True)
    agent._schedule_next_active_query = lambda position_after: seen.setdefault("scheduled_side", position_after.get("side"))
    agent._audit_event = lambda kind, payload: seen.setdefault("audit_kind", kind)

    result = agent._execute_immediate_playbook_action(playbook, dict(open_snapshot))

    assert result is not None
    assert result["result"]["decision"]["action"] == "no_change"
    assert "risk_session_reset" not in seen
    assert seen["audit_kind"] == "position_management_immediate_no_change"
    assert seen["scheduled_side"] == "long"
    assert playbook.position_management.execute_now is False
    assert playbook.position_management.action_decision.action == "no_change"
    assert agent.risk_session is existing_risk_session
    assert agent.position_management_session is existing_position_management_session


def test_set_risk_session_from_management_passthrough_preserves_staged_state(uma):
    open_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 1.4,
        "entry_price": 100000.0,
        "mid_price": 101500.0,
        "notional_usd": 142100.0,
    }
    reader = FakeReader(symbol_snapshots=[dict(open_snapshot)])
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    calls = []
    executor.cancel_reduce_only_tpsl_orders = lambda *args, **kwargs: calls.append(("cancel", args, kwargs)) or {"accepted": True}
    executor.place_reduce_only_tpsl_order = lambda *args, **kwargs: calls.append(("place", args, kwargs)) or {"accepted": True}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent._audit_event = lambda *args, **kwargs: None
    existing_plan = make_management_plan(uma, stop_price=99600.0)
    existing_session = uma.RiskSession(
        plan_name="position_management",
        side="long",
        stop_loss_price=99600.0,
        baseline_size=2.0,
        expected_size=2.0,
        initial_size_abs=2.0,
        position_management=existing_plan,
        take_profit_legs=[
            uma.ExitLeg(name="stage_tp2", note="tp2", when_all=[uma.Condition(type="price_ge", level=102000.0, note="tp2")], close_fraction=0.40),
        ],
        stop_loss_legs=[
            uma.ExitLeg(name="stage_post_tp1_stop", note="sl", when_all=[uma.Condition(type="price_le", level=99600.0, note="sl")], close_fraction=1.0),
        ],
        resting_exit_orders=[
            {"key": "take_profit::stage_tp2", "name": "stage_tp2", "leg_type": "take_profit", "tpsl": "tp", "close_size": 0.8, "oid": 2},
            {"key": "stop_loss::stage_post_tp1_stop", "name": "stage_post_tp1_stop", "leg_type": "stop_loss", "tpsl": "sl", "close_size": 1.4, "oid": 3},
        ],
        use_resting_exit_orders=True,
        take_profit_legs_scale_from_initial_size=True,
        staged_exit_enabled=True,
        tp1_hit=True,
        tp2_hit=False,
        initial_entry_price=100000.0,
        initial_stop_price=99000.0,
        initial_risk_price_distance=1000.0,
        tp1_price=101000.0,
        tp2_price=102000.0,
        post_tp1_stop_price=99600.0,
        locked_floor_price=101000.0,
        trailing_hard_stop_price=99600.0,
        executed_leg_names={"take_profit::stage_tp1"},
        runtimes={},
        history=deque(),
    )
    original_resting_orders = [dict(item) for item in existing_session.resting_exit_orders]
    agent.risk_session = existing_session
    incoming_plan = make_management_plan(uma, stop_price=99000.0)
    setattr(incoming_plan, "risk_session_passthrough", True)

    agent._set_risk_session_from_management(incoming_plan, dict(open_snapshot), "position_management")

    assert agent.risk_session is existing_session
    assert calls == []
    assert agent.risk_session.expected_size == pytest.approx(2.0)
    assert agent.risk_session.baseline_size == pytest.approx(2.0)
    assert agent.risk_session.prev_price is None
    assert agent.risk_session.tp1_hit is True
    assert agent.risk_session.tp2_hit is False
    assert agent.risk_session.executed_leg_names == {"take_profit::stage_tp1"}
    assert agent.risk_session.resting_exit_orders == original_resting_orders
    assert agent.risk_session.position_management is existing_plan
    assert incoming_plan.action_decision.entry_price != pytest.approx(100000.0)
    assert incoming_plan.action_decision.stop_loss_price == pytest.approx(99000.0)


def test_risk_leg_hit_preserves_position_management_session(uma):
    open_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 1.0,
        "entry_price": 100000.0,
        "mid_price": 101000.0,
        "notional_usd": 101000.0,
    }
    reduced_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 0.7,
        "entry_price": 100000.0,
        "mid_price": 101000.0,
        "notional_usd": 70700.0,
    }
    management_scenario = uma.Scenario(
        observe_when_all=[uma.Condition(type="price_ge", level=102000.0, note="observe")],
        execute_when_all={
            "condition": {"type": "price_ge", "level": 103000.0, "low": 0.0, "high": 0.0, "timer_seconds": 0, "tolerance_bps": 0.0, "min_ratio": 0.0},
            "timeout_seconds": 60,
        },
    )
    pending_plan = make_management_plan(uma, scenario=management_scenario, now_action=make_management_decision(uma, action="reverse_to_short", notional=800.0, leverage=10))
    reader = FakeReader(symbol_snapshots=[dict(reduced_snapshot)])
    agent = make_agent(uma, reader=reader)
    agent.executor.reduce_position = lambda side, qty, reason, plan_name=None: {
        "side": side,
        "qty": qty,
        "reason": reason,
        "plan_name": plan_name,
    }
    risk_plan = make_management_plan(uma, scenario=None)
    risk_take_profit_legs = [
        uma.ExitLeg(
            name="tp-1",
            note="tp1",
            when_all=[uma.Condition(type="price_ge", level=100900.0, note="tp1")],
            close_fraction=0.3,
        ),
        uma.ExitLeg(
            name="tp-2",
            note="tp2",
            when_all=[uma.Condition(type="price_ge", level=101500.0, note="tp2")],
            close_fraction=0.7,
        ),
    ]
    agent.risk_session = uma.RiskSession(
        plan_name="post_fill_risk_template",
        side="long",
        initial_entry_price=100000.0,
        start_time=1.0,
        baseline_size=1.0,
        expected_size=1.0,
        initial_size_abs=1.0,
        position_management=risk_plan,
        take_profit_legs=list(risk_take_profit_legs),
        stop_loss_legs=[],
        runtimes={},
        history=deque(),
        history_seconds=1800,
    )
    seen = {}

    def fake_refresh(snapshot, all_positions=None):
        seen["size"] = float(snapshot.get("size", 0.0) or 0.0)
        agent.position_management_session = uma.PositionManagementSession(
            plan_name="position_management",
            side="long",
            position_management=pending_plan,
            start_time=1.0,
            baseline_size=0.7,
            expected_size=0.7,
            initial_size_abs=0.7,
            runtimes={uma.SCENARIO_RUNTIME_KEY: uma.ScenarioRuntime()},
            history=deque(),
            history_seconds=1800,
        )

    agent._refresh_position_management_session_from_current_playbook = fake_refresh
    agent.position_management_session = uma.PositionManagementSession(
        plan_name="position_management",
        side="long",
        position_management=pending_plan,
        start_time=1.0,
        baseline_size=1.0,
        expected_size=1.0,
        initial_size_abs=1.0,
        runtimes={uma.SCENARIO_RUNTIME_KEY: uma.ScenarioRuntime()},
        history=deque(),
        history_seconds=1800,
    )

    status = agent.step_risk_session(dict(open_snapshot), now=100.0)

    assert status is None
    assert agent.risk_session is not None
    assert agent.position_management_session is not None
    assert seen == {}
    assert agent.position_management_session.expected_size == pytest.approx(1.0)
    assert agent.position_management_session.side == "long"


def test_step_risk_session_resting_reduce_only_fill_reposts_remaining_orders(uma):
    open_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 2.0,
        "entry_price": 100000.0,
        "mid_price": 101000.0,
        "notional_usd": 200000.0,
        "max_leverage": 40,
    }
    reader = FakeReader(symbol_snapshots=[dict(open_snapshot)])
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    calls = []
    executor.modify_reduce_only_tpsl_orders = lambda order_updates, side, plan_name=None: calls.append(("modify", plan_name, side, [dict(item) for item in order_updates])) or {"accepted": True, "updated_refs": [dict(item) for item in order_updates]}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent._audit_event = lambda *args, **kwargs: None
    agent.risk_session = uma.RiskSession(
        plan_name="risk",
        side="long",
        initial_entry_price=100000.0,
        baseline_size=3.0,
        expected_size=3.0,
        initial_size_abs=3.0,
        take_profit_legs=[
            uma.ExitLeg(name="tp-1", note="tp", when_all=[uma.Condition(type="price_ge", level=101000.0, note="tp")], close_fraction=1.0 / 3.0),
            uma.ExitLeg(name="tp-2", note="tp", when_all=[uma.Condition(type="price_ge", level=102000.0, note="tp")], close_fraction=1.0 / 3.0),
            uma.ExitLeg(name="tp-3", note="tp", when_all=[uma.Condition(type="price_ge", level=103000.0, note="tp")], close_fraction=1.0 / 3.0),
        ],
        stop_loss_legs=[
            uma.ExitLeg(name="sl-1", note="sl", when_all=[uma.Condition(type="price_le", level=99000.0, note="sl")], close_fraction=0.25),
            uma.ExitLeg(name="sl-2", note="sl", when_all=[uma.Condition(type="price_le", level=98000.0, note="sl")], close_fraction=0.75),
        ],
        history=deque(),
        runtimes={},
        resting_exit_orders=[
            {"key": "take_profit::tp-1", "name": "tp-1", "leg_type": "take_profit", "tpsl": "tp", "close_size": 1.0, "cloid": "0x1"},
            {"key": "take_profit::tp-2", "name": "tp-2", "leg_type": "take_profit", "tpsl": "tp", "close_size": 1.0, "cloid": "0x2"},
            {"key": "take_profit::tp-3", "name": "tp-3", "leg_type": "take_profit", "tpsl": "tp", "close_size": 1.0, "cloid": "0x3"},
            {"key": "stop_loss::sl-1", "name": "sl-1", "leg_type": "stop_loss", "tpsl": "sl", "close_size": 0.75, "cloid": "0x4"},
            {"key": "stop_loss::sl-2", "name": "sl-2", "leg_type": "stop_loss", "tpsl": "sl", "close_size": 2.25, "cloid": "0x5"},
        ],
        use_resting_exit_orders=True,
    )

    status = agent.step_risk_session(dict(open_snapshot), now=100.0)

    assert status == "take_profit_hit"
    assert agent.risk_session is not None
    assert "take_profit::tp-1" in agent.risk_session.executed_leg_names
    assert agent.risk_session.expected_size == pytest.approx(2.0)
    modify_calls = [item for item in calls if item[0] == "modify"]
    assert len(modify_calls) == 1
    assert modify_calls[0][2] == "long"
    assert {(item["name"], item["tpsl"], item["close_size"]) for item in modify_calls[0][3]} == {("sl-1", "sl", 0.5), ("sl-2", "sl", 1.5)}
    assert {item["name"] for item in agent.risk_session.resting_exit_orders} == {"tp-2", "tp-3", "sl-1", "sl-2"}
    stop_refs = {item["name"]: item["close_size"] for item in agent.risk_session.resting_exit_orders if item["leg_type"] == "stop_loss"}
    assert stop_refs == {"sl-1": 0.5, "sl-2": 1.5}








def test_step_risk_session_reconciles_missed_fill_even_without_ws_active(uma):
    open_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 2.25,
        "entry_price": 100000.0,
        "mid_price": 98950.0,
        "notional_usd": 222637.5,
        "max_leverage": 40,
    }
    reader = FakeReader(
        symbol_snapshots=[dict(open_snapshot)],
        frontend_open_orders=[
            {"coin": "BTC", "oid": 1001},
            {"coin": "BTC", "oid": 1002},
            {"coin": "BTC", "oid": 1003},
            {"coin": "BTC", "oid": 1005},
        ],
        order_statuses={1004: {"status": "filled"}},
    )
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    calls = []
    executor.modify_reduce_only_tpsl_orders = lambda order_updates, side, plan_name=None: calls.append(("modify", plan_name, side, [dict(item) for item in order_updates])) or {"accepted": True, "updated_refs": [dict(item) for item in order_updates]}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent._audit_event = lambda *args, **kwargs: None
    agent.enable_user_fills_websocket = False
    agent.user_fills_subscription_id = None
    agent.risk_session = uma.RiskSession(
        plan_name="risk",
        side="long",
        initial_entry_price=100000.0,
        baseline_size=3.0,
        expected_size=3.0,
        initial_size_abs=3.0,
        take_profit_legs=[
            uma.ExitLeg(name="tp-1", note="tp", when_all=[uma.Condition(type="price_ge", level=101000.0, note="tp")], close_fraction=1.0 / 3.0),
            uma.ExitLeg(name="tp-2", note="tp", when_all=[uma.Condition(type="price_ge", level=102000.0, note="tp")], close_fraction=1.0 / 3.0),
            uma.ExitLeg(name="tp-3", note="tp", when_all=[uma.Condition(type="price_ge", level=103000.0, note="tp")], close_fraction=1.0 / 3.0),
        ],
        stop_loss_legs=[
            uma.ExitLeg(name="sl-1", note="sl", when_all=[uma.Condition(type="price_le", level=99000.0, note="sl")], close_fraction=0.25),
            uma.ExitLeg(name="sl-2", note="sl", when_all=[uma.Condition(type="price_le", level=98000.0, note="sl")], close_fraction=0.75),
        ],
        history=deque(),
        runtimes={},
        resting_exit_orders=[
            {"key": "take_profit::tp-1", "name": "tp-1", "leg_type": "take_profit", "tpsl": "tp", "close_size": 1.0, "oid": 1001, "cloid": "0x1"},
            {"key": "take_profit::tp-2", "name": "tp-2", "leg_type": "take_profit", "tpsl": "tp", "close_size": 1.0, "oid": 1002, "cloid": "0x2"},
            {"key": "take_profit::tp-3", "name": "tp-3", "leg_type": "take_profit", "tpsl": "tp", "close_size": 1.0, "oid": 1003, "cloid": "0x3"},
            {"key": "stop_loss::sl-1", "name": "sl-1", "leg_type": "stop_loss", "tpsl": "sl", "close_size": 0.75, "oid": 1004, "cloid": "0x4"},
            {"key": "stop_loss::sl-2", "name": "sl-2", "leg_type": "stop_loss", "tpsl": "sl", "close_size": 2.25, "oid": 1005, "cloid": "0x5"},
        ],
        use_resting_exit_orders=True,
    )

    status = agent.step_risk_session(dict(open_snapshot), now=100.0, fill_events=[])

    assert status == "stop_loss_hit"
    assert agent.risk_session is not None
    assert "stop_loss::sl-1" in agent.risk_session.executed_leg_names
    modify_calls = [item for item in calls if item[0] == "modify"]
    assert len(modify_calls) == 1
    assert {(item["name"], item["tpsl"], item["close_size"]) for item in modify_calls[0][3]} == {("tp-1", "tp", 0.75), ("tp-2", "tp", 0.75), ("tp-3", "tp", 0.75)}


def test_step_risk_session_staged_tp1_hit_moves_stop_to_minus_point_four_r(uma):
    open_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 1.4,
        "entry_price": 100000.0,
        "mid_price": 101000.0,
        "notional_usd": 140000.0,
        "max_leverage": 40,
    }
    reader = FakeReader(symbol_snapshots=[dict(open_snapshot)])
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    calls = []
    executor.modify_reduce_only_tpsl_orders = lambda order_updates, side, plan_name=None: calls.append(("modify", plan_name, side, [dict(item) for item in order_updates])) or {"accepted": True, "updated_refs": [dict(item) for item in order_updates]}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent._audit_event = lambda *args, **kwargs: None
    initial_take_profit_legs = [
        uma.ExitLeg(name="stage_tp1", note="tp1", when_all=[uma.Condition(type="price_ge", level=101000.0, note="tp1")], close_fraction=0.30),
        uma.ExitLeg(name="stage_tp2", note="tp2", when_all=[uma.Condition(type="price_ge", level=102000.0, note="tp2")], close_fraction=0.40),
    ]
    initial_stop_loss_legs = [
        uma.ExitLeg(name="stage_initial_stop", note="sl", when_all=[uma.Condition(type="price_le", level=99000.0, note="sl")], close_fraction=1.0),
    ]
    position_management = uma.PositionManagementPlan(
        execute_now=False,
        action_decision=uma.build_empty_management_decision(),
        scenario=None,
    )
    session = uma.RiskSession(
        plan_name="risk",
        side="long",
        baseline_size=2.0,
        expected_size=2.0,
        initial_size_abs=2.0,
        position_management=position_management,
        take_profit_legs=list(initial_take_profit_legs),
        stop_loss_legs=list(initial_stop_loss_legs),
        resting_exit_orders=[
            {"key": "take_profit::stage_tp1", "name": "stage_tp1", "leg_type": "take_profit", "tpsl": "tp", "close_size": 0.6, "cloid": "0x1"},
            {"key": "take_profit::stage_tp2", "name": "stage_tp2", "leg_type": "take_profit", "tpsl": "tp", "close_size": 0.8, "cloid": "0x2"},
            {"key": "stop_loss::stage_initial_stop", "name": "stage_initial_stop", "leg_type": "stop_loss", "tpsl": "sl", "close_size": 2.0, "cloid": "0x3"},
        ],
        use_resting_exit_orders=True,
        take_profit_legs_scale_from_initial_size=True,
        staged_exit_enabled=True,
        initial_entry_price=100000.0,
        initial_stop_price=99000.0,
        initial_risk_price_distance=1000.0,
        tp1_price=101000.0,
        tp2_price=102000.0,
        post_tp1_stop_price=99600.0,
        locked_floor_price=101000.0,
        history=deque(),
        runtimes={},
    )
    agent.risk_session = session

    status = agent.step_risk_session(dict(open_snapshot), now=100.0, fill_events=[{"coin": "BTC", "oid": 1, "cloid": "0x1", "sz": "0.6", "px": "101000", "time": 100000}])

    assert status == "take_profit_hit"
    assert agent.risk_session is not None
    assert agent.risk_session.tp1_hit is True
    assert agent.risk_session.stop_loss_price == pytest.approx(99600.0)
    assert [leg.name for leg in agent.risk_session.take_profit_legs] == ["stage_tp2"]
    modify_calls = [item for item in calls if item[0] == "modify"]
    assert modify_calls == []


def test_step_risk_session_time_decay_tp1_executes_when_mfe_and_current_profit_hold(uma):
    before_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 2.0,
        "entry_price": 100.0,
        "mid_price": 100.35,
        "notional_usd": 200.7,
        "max_leverage": 40,
    }
    after_snapshot = dict(before_snapshot, size=1.4)
    reader = FakeReader(symbol_snapshots=[dict(after_snapshot)])
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    cancel_calls = []
    reduce_calls = []
    place_calls = []
    executor.cancel_reduce_only_tpsl_orders = lambda refs, plan_name=None: cancel_calls.append([dict(item) for item in refs]) or {"accepted": True}
    executor.reduce_position = lambda side, close_size, reason, plan_name=None, position_before=None: reduce_calls.append((side, close_size, reason, plan_name, dict(position_before or {}))) or {"accepted": True, "actions": [{"market_close": {"status": "ok"}}]}
    executor.place_reduce_only_limit_order = lambda side, close_size, limit_price, plan_name=None, leg_name="": place_calls.append((leg_name, "limit", limit_price, close_size)) or {"accepted": True, "close_size": close_size, "limit_price": limit_price, "oid": 2000 + len(place_calls), "order_kind": "limit"}
    executor.place_reduce_only_tpsl_order = lambda side, close_size, trigger_price, tpsl, plan_name=None, leg_name="": place_calls.append((leg_name, tpsl, trigger_price, close_size)) or {"accepted": True, "close_size": close_size, "trigger_price": trigger_price, "oid": 3000 + len(place_calls)}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent.risk_time_decay_tp_enabled = True
    agent._audit_event = lambda *args, **kwargs: None
    agent.risk_session = uma.RiskSession(
        plan_name="risk",
        side="long",
        stop_loss_price=99.0,
        start_time=0.0,
        baseline_size=2.0,
        expected_size=2.0,
        initial_size_abs=2.0,
        staged_exit_size_basis_abs=2.0,
        take_profit_legs=[
            uma.ExitLeg(name="stage_tp1", note="tp1", when_all=[uma.Condition(type="price_ge", level=101.0, note="tp1")], close_fraction=0.30),
            uma.ExitLeg(name="stage_tp2", note="tp2", when_all=[uma.Condition(type="price_ge", level=102.0, note="tp2")], close_fraction=0.40),
        ],
        stop_loss_legs=[
            uma.ExitLeg(name="stage_initial_stop", note="sl", when_all=[uma.Condition(type="price_le", level=99.0, note="sl")], close_fraction=1.0),
        ],
        resting_exit_orders=[
            {"key": "take_profit::stage_tp1", "name": "stage_tp1", "leg_type": "take_profit", "tpsl": "tp", "close_size": 0.6, "oid": 1},
            {"key": "take_profit::stage_tp2", "name": "stage_tp2", "leg_type": "take_profit", "tpsl": "tp", "close_size": 0.8, "oid": 2},
            {"key": "stop_loss::stage_initial_stop", "name": "stage_initial_stop", "leg_type": "stop_loss", "tpsl": "sl", "close_size": 2.0, "oid": 3},
        ],
        use_resting_exit_orders=True,
        take_profit_legs_scale_from_initial_size=True,
        staged_exit_enabled=True,
        staged_exit_liquidity_band="normal_liquidity",
        initial_entry_price=100.0,
        initial_stop_price=99.0,
        initial_risk_price_distance=1.0,
        tp1_price=101.0,
        tp2_price=102.0,
        tp1_hit=False,
        tp2_hit=False,
        post_tp1_stop_price=99.6,
        locked_floor_price=101.0,
        max_favorable_excursion_r=0.65,
        history=deque(),
        runtimes={},
    )

    status = agent.step_risk_session(dict(before_snapshot), now=(8 * 900) + 1, fill_events=[])

    assert status == "take_profit_hit"
    assert reduce_calls == [("long", pytest.approx(0.6), "time_decay_take_profit", "risk", before_snapshot)]
    assert [item["key"] for item in cancel_calls[0]] == ["take_profit::stage_tp1"]
    assert agent.risk_session is not None
    assert agent.risk_session.tp1_hit is True
    assert agent.risk_session.tp1_hit_at == pytest.approx((8 * 900) + 1)
    assert agent.risk_session.stop_loss_price == pytest.approx(99.6)
    assert [leg.name for leg in agent.risk_session.take_profit_legs] == ["stage_tp2"]
    assert ("stage_tp2", "limit", 102.0, pytest.approx(0.8)) in place_calls


def test_staged_tp_target_uses_exchange_size_precision_to_avoid_residual(uma):
    open_snapshot = {
        "symbol": "xyz:BRENTOIL",
        "side": "long",
        "size": 11.6,
        "entry_price": 99.98,
        "mid_price": 100.61,
        "notional_usd": 1167.0,
        "max_leverage": 20,
    }
    reader = FakeReader(symbol_snapshots=[dict(open_snapshot)])
    reader.get_sz_decimals = lambda symbol: 2
    executor = FakeExecutor(reader, "xyz:BRENTOIL")
    agent = make_agent(uma, reader=reader, executor=executor)
    agent.symbol = "xyz:BRENTOIL"
    agent.risk_time_decay_tp_enabled = True
    agent.risk_time_decay_tp_timeframe_seconds = 300.0
    agent.risk_time_decay_normal_tp1_bars = 6.0
    agent.risk_time_decay_normal_tp1_mfe_r = 0.60
    agent.risk_time_decay_normal_tp1_current_r = 0.30
    rs = uma.RiskSession(
        plan_name="risk",
        side="long",
        stop_loss_price=98.756,
        start_time=0.0,
        baseline_size=11.6,
        expected_size=11.6,
        initial_size_abs=16.56,
        staged_exit_size_basis_abs=16.56,
        take_profit_legs=[
            uma.ExitLeg(name="stage_tp1", note="tp1", when_all=[uma.Condition(type="price_ge", level=101.15, note="tp1")], close_fraction=0.30),
            uma.ExitLeg(name="stage_tp2", note="tp2", when_all=[uma.Condition(type="price_ge", level=102.31, note="tp2")], close_fraction=0.40),
        ],
        stop_loss_legs=[],
        take_profit_legs_scale_from_initial_size=True,
        staged_exit_enabled=True,
        staged_exit_liquidity_band="normal_liquidity",
        initial_entry_price=99.98,
        initial_stop_price=98.814,
        initial_risk_price_distance=1.166,
        tp1_price=101.15,
        tp2_price=102.31,
        tp1_completed_size_abs=4.96,
        tp1_hit=False,
        tp2_hit=False,
        max_favorable_excursion_r=0.82,
        history=deque(),
        runtimes={},
    )

    assert agent._staged_tp_target_size_abs(rs, "stage_tp1") == pytest.approx(4.96)
    assert agent._risk_session_tp1_completed(rs) is True
    assert agent._risk_session_time_decay_tp_candidate(
        rs,
        dict(open_snapshot),
        price=100.61,
        now=6 * 300 + 1,
    ) is None
    tp_specs = agent._iter_risk_session_exit_order_specs(rs, leg_type_filter="take_profit")

    assert [item["name"] for item in tp_specs] == ["stage_tp2"]
    assert tp_specs[0]["close_size"] == pytest.approx(6.62)


def test_tp1_no_follow_through_trim_size_uses_exchange_size_precision(uma):
    snapshot = {
        "symbol": "xyz:BRENTOIL",
        "side": "long",
        "size": 2.01,
        "entry_price": 100.0,
        "mid_price": 100.1,
        "notional_usd": 201.0,
        "max_leverage": 20,
    }
    reader = FakeReader(symbol_snapshots=[dict(snapshot)])
    reader.get_sz_decimals = lambda symbol: 2
    agent = make_agent(uma, reader=reader, executor=FakeExecutor(reader, "xyz:BRENTOIL"))
    agent.symbol = "xyz:BRENTOIL"
    agent.risk_time_decay_tp_timeframe_seconds = 300.0
    agent.risk_time_decay_normal_tp1_bars = 6.0
    agent.risk_time_decay_normal_tp1_mfe_r = 0.60
    agent.risk_tp1_no_follow_through_normal_close_fraction = 0.50
    rs = uma.RiskSession(
        plan_name="risk",
        side="long",
        stop_loss_price=99.0,
        start_time=0.0,
        baseline_size=2.01,
        expected_size=2.01,
        initial_size_abs=2.01,
        staged_exit_size_basis_abs=2.01,
        take_profit_legs=[
            uma.ExitLeg(name="stage_tp1", note="tp1", when_all=[uma.Condition(type="price_ge", level=101.0, note="tp1")], close_fraction=0.30),
        ],
        stop_loss_legs=[],
        take_profit_legs_scale_from_initial_size=True,
        staged_exit_enabled=True,
        staged_exit_liquidity_band="normal_liquidity",
        initial_entry_price=100.0,
        initial_stop_price=99.0,
        initial_risk_price_distance=1.0,
        max_favorable_excursion_r=0.20,
        history=deque(),
        runtimes={},
    )

    candidate = agent._risk_session_tp1_no_follow_through_candidate(rs, dict(snapshot), price=100.1, now=(6 * 300) + 1)

    assert candidate is not None
    assert candidate["close_size"] == pytest.approx(1.00)


def test_tp2_no_continuation_trim_size_uses_exchange_size_precision(uma):
    snapshot = {
        "symbol": "xyz:BRENTOIL",
        "side": "long",
        "size": 1.41,
        "entry_price": 100.0,
        "mid_price": 100.8,
        "notional_usd": 142.0,
        "max_leverage": 20,
    }
    reader = FakeReader(symbol_snapshots=[dict(snapshot)])
    reader.get_sz_decimals = lambda symbol: 2
    agent = make_agent(uma, reader=reader, executor=FakeExecutor(reader, "xyz:BRENTOIL"))
    agent.symbol = "xyz:BRENTOIL"
    agent.risk_time_decay_tp_timeframe_seconds = 300.0
    agent.risk_time_decay_normal_tp2_bars = 12.0
    agent.risk_time_decay_normal_tp2_mfe_r = 1.50
    agent.risk_tp2_no_continuation_normal_close_fraction = 0.50
    agent.risk_tp2_no_continuation_normal_soft_stop_r = 0.25
    rs = uma.RiskSession(
        plan_name="risk",
        side="long",
        stop_loss_price=99.6,
        start_time=0.0,
        baseline_size=1.41,
        expected_size=1.41,
        initial_size_abs=2.01,
        staged_exit_size_basis_abs=2.01,
        take_profit_legs=[
            uma.ExitLeg(name="stage_tp2", note="tp2", when_all=[uma.Condition(type="price_ge", level=102.0, note="tp2")], close_fraction=0.40),
        ],
        stop_loss_legs=[],
        take_profit_legs_scale_from_initial_size=True,
        staged_exit_enabled=True,
        staged_exit_liquidity_band="normal_liquidity",
        initial_entry_price=100.0,
        initial_stop_price=99.0,
        initial_risk_price_distance=1.0,
        tp1_hit=True,
        tp1_hit_at=100.0,
        tp1_completed_size_abs=0.60,
        max_favorable_excursion_r=1.20,
        active_soft_stop_price=99.6,
        history=deque(),
        runtimes={},
    )

    candidate = agent._risk_session_tp2_no_continuation_candidate(
        rs,
        dict(snapshot),
        price=100.8,
        now=100.0 + (12 * 300) + 1,
    )

    assert candidate is not None
    assert candidate["close_size"] == pytest.approx(0.70)


def test_tp2_no_continuation_can_use_tp1_no_follow_through_anchor(uma):
    snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 1.0,
        "entry_price": 100.0,
        "mid_price": 100.8,
        "notional_usd": 100.8,
        "max_leverage": 40,
    }
    reader = FakeReader(symbol_snapshots=[dict(snapshot)])
    agent = make_agent(uma, reader=reader, executor=FakeExecutor(reader, "BTC"))
    agent.risk_time_decay_tp_timeframe_seconds = 300.0
    agent.risk_time_decay_normal_tp2_bars = 12.0
    agent.risk_time_decay_normal_tp2_mfe_r = 1.50
    agent.risk_tp2_no_continuation_normal_close_fraction = 0.50
    agent.risk_tp2_no_continuation_normal_soft_stop_r = 0.25
    rs = uma.RiskSession(
        plan_name="risk",
        side="long",
        stop_loss_price=99.6,
        start_time=0.0,
        baseline_size=1.0,
        expected_size=1.0,
        initial_size_abs=2.0,
        staged_exit_size_basis_abs=1.0,
        take_profit_legs=[
            uma.ExitLeg(name="stage_tp1", note="tp1", when_all=[uma.Condition(type="price_ge", level=101.0, note="tp1")], close_fraction=0.30),
            uma.ExitLeg(name="stage_tp2", note="tp2", when_all=[uma.Condition(type="price_ge", level=102.0, note="tp2")], close_fraction=0.40),
        ],
        stop_loss_legs=[],
        take_profit_legs_scale_from_initial_size=True,
        staged_exit_enabled=True,
        staged_exit_liquidity_band="normal_liquidity",
        initial_entry_price=100.0,
        initial_stop_price=99.0,
        initial_risk_price_distance=1.0,
        tp1_hit=False,
        tp1_hit_at=0.0,
        tp1_no_follow_through_applied=True,
        tp1_no_follow_through_at=100.0,
        max_favorable_excursion_r=1.20,
        active_soft_stop_price=99.6,
        history=deque(),
        runtimes={},
    )

    candidate = agent._risk_session_tp2_no_continuation_candidate(
        rs,
        dict(snapshot),
        price=100.8,
        now=100.0 + (12 * 300) + 1,
    )

    assert candidate is not None
    assert candidate["continuation_anchor_source"] == "tp1_no_follow_through"
    assert candidate["tp1_completed"] is False
    assert candidate["tp1_no_follow_through_applied"] is True
    assert candidate["close_size"] == pytest.approx(0.5)


def test_regular_exit_order_specs_use_exchange_size_precision(uma):
    reader = FakeReader()
    reader.get_sz_decimals = lambda symbol: 2
    agent = make_agent(uma, reader=reader, executor=FakeExecutor(reader, "xyz:BRENTOIL"))
    agent.symbol = "xyz:BRENTOIL"
    rs = uma.RiskSession(
        plan_name="risk",
        side="long",
        stop_loss_price=99.0,
        start_time=0.0,
        baseline_size=2.01,
        expected_size=2.01,
        initial_size_abs=2.01,
        take_profit_legs=[],
        stop_loss_legs=[
            uma.ExitLeg(name="sl_a", note="sl", when_all=[uma.Condition(type="price_le", level=99.0, note="sl")], close_fraction=0.50),
            uma.ExitLeg(name="sl_b", note="sl", when_all=[uma.Condition(type="price_le", level=98.5, note="sl")], close_fraction=0.50),
        ],
        staged_exit_enabled=False,
        history=deque(),
        runtimes={},
    )

    specs = agent._iter_risk_session_exit_order_specs(rs, leg_type_filter="stop_loss")

    assert [item["close_size"] for item in specs] == [pytest.approx(1.00), pytest.approx(1.00)]


def test_risk_session_state_restore_normalizes_legacy_precision_sizes(uma):
    snapshot = {
        "symbol": "xyz:BRENTOIL",
        "side": "long",
        "size": 11.6,
        "entry_price": 99.98,
        "mid_price": 100.61,
        "notional_usd": 1167.0,
        "max_leverage": 20,
    }
    reader = FakeReader(symbol_snapshots=[dict(snapshot)])
    reader.get_sz_decimals = lambda symbol: 2
    agent = make_agent(uma, reader=reader, executor=FakeExecutor(reader, "xyz:BRENTOIL"))
    agent.symbol = "xyz:BRENTOIL"
    state_payload = {
        "symbol": "xyz:BRENTOIL",
        "plan_name": "risk",
        "side": "long",
        "stop_loss_price": 98.75,
        "baseline_size": 11.6,
        "expected_size": 11.6,
        "initial_size_abs": 16.56,
        "staged_exit_size_basis_abs": 16.56,
        "take_profit_legs": [
            {"name": "stage_tp1", "note": "tp1", "when_all": [{"type": "price_ge", "level": 101.15}], "close_fraction": 0.30},
            {"name": "stage_tp2", "note": "tp2", "when_all": [{"type": "price_ge", "level": 102.31}], "close_fraction": 0.40},
        ],
        "resting_exit_orders": [
            {"key": "take_profit::stage_tp1", "name": "stage_tp1", "leg_type": "take_profit", "tpsl": "tp", "trigger_price": 101.15, "close_size": 4.968},
            {"key": "take_profit::stage_tp2", "name": "stage_tp2", "leg_type": "take_profit", "tpsl": "tp", "trigger_price": 102.31, "close_size": 6.624},
        ],
        "use_resting_exit_orders": True,
        "take_profit_legs_scale_from_initial_size": True,
        "staged_exit_enabled": True,
        "tp1_completed_size_abs": 4.968,
        "tp2_completed_size_abs": 0.0,
        "initial_entry_price": 99.98,
        "initial_stop_price": 98.75,
        "initial_risk_price_distance": 1.23,
        "tp1_price": 101.15,
        "tp2_price": 102.31,
    }

    session = agent._risk_session_from_state_payload(state_payload, dict(snapshot))

    assert session is not None
    assert session.tp1_completed_size_abs == pytest.approx(4.96)
    assert [item["close_size"] for item in session.resting_exit_orders if item["leg_type"] == "take_profit"] == [
        pytest.approx(4.96),
        pytest.approx(6.62),
    ]


def test_startup_restore_fill_completion_uses_aligned_saved_ref_size(uma):
    reader = FakeReader()
    reader.get_sz_decimals = lambda symbol: 2
    agent = make_agent(uma, reader=reader, executor=FakeExecutor(reader, "xyz:BRENTOIL"))
    agent.symbol = "xyz:BRENTOIL"
    session = uma.RiskSession(
        plan_name="risk",
        side="long",
        baseline_size=16.56,
        expected_size=16.56,
        initial_size_abs=16.56,
        staged_exit_size_basis_abs=16.56,
        take_profit_legs=[
            uma.ExitLeg(name="stage_tp1", note="tp1", when_all=[uma.Condition(type="price_ge", level=101.15, note="tp1")], close_fraction=0.30),
        ],
        stop_loss_legs=[],
        resting_exit_orders=[
            {"key": "take_profit::stage_tp1", "name": "stage_tp1", "leg_type": "take_profit", "tpsl": "tp", "close_size": 4.968, "oid": 1001},
        ],
        use_resting_exit_orders=True,
        take_profit_legs_scale_from_initial_size=True,
        staged_exit_enabled=True,
        history=deque(),
        runtimes={},
    )

    completed = agent._startup_restore_completed_keys_from_fills(
        session,
        [{"coin": "xyz:BRENTOIL", "oid": 1001, "time": 100000, "sz": "4.96", "px": "101.15"}],
    )

    assert completed == {"take_profit::stage_tp1"}


def test_step_risk_session_time_decay_tp1_executes_after_timer_even_when_current_profit_is_low(uma):
    before_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 2.0,
        "entry_price": 100.0,
        "mid_price": 100.2,
        "notional_usd": 200.4,
        "max_leverage": 40,
    }
    after_snapshot = dict(before_snapshot, size=1.4)
    reader = FakeReader(symbol_snapshots=[dict(after_snapshot)])
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    cancel_calls = []
    reduce_calls = []
    place_calls = []
    executor.cancel_reduce_only_tpsl_orders = lambda refs, plan_name=None: cancel_calls.append([dict(item) for item in refs]) or {"accepted": True}
    executor.reduce_position = lambda side, close_size, reason, plan_name=None, position_before=None: reduce_calls.append((side, close_size, reason, plan_name, dict(position_before or {}))) or {"accepted": True, "actions": [{"market_close": {"status": "ok"}}]}
    executor.place_reduce_only_limit_order = lambda side, close_size, limit_price, plan_name=None, leg_name="": place_calls.append((leg_name, "limit", limit_price, close_size)) or {"accepted": True, "close_size": close_size, "limit_price": limit_price, "oid": 2000 + len(place_calls), "order_kind": "limit"}
    executor.place_reduce_only_tpsl_order = lambda side, close_size, trigger_price, tpsl, plan_name=None, leg_name="": place_calls.append((leg_name, tpsl, trigger_price, close_size)) or {"accepted": True, "close_size": close_size, "trigger_price": trigger_price, "oid": 3000 + len(place_calls)}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent.risk_time_decay_tp_enabled = True
    agent._audit_event = lambda *args, **kwargs: None
    agent.risk_session = uma.RiskSession(
        plan_name="risk",
        side="long",
        stop_loss_price=99.0,
        start_time=0.0,
        baseline_size=2.0,
        expected_size=2.0,
        initial_size_abs=2.0,
        staged_exit_size_basis_abs=2.0,
        take_profit_legs=[
            uma.ExitLeg(name="stage_tp1", note="tp1", when_all=[uma.Condition(type="price_ge", level=101.0, note="tp1")], close_fraction=0.30),
            uma.ExitLeg(name="stage_tp2", note="tp2", when_all=[uma.Condition(type="price_ge", level=102.0, note="tp2")], close_fraction=0.40),
        ],
        stop_loss_legs=[
            uma.ExitLeg(name="stage_initial_stop", note="sl", when_all=[uma.Condition(type="price_le", level=99.0, note="sl")], close_fraction=1.0),
        ],
        resting_exit_orders=[
            {"key": "take_profit::stage_tp1", "name": "stage_tp1", "leg_type": "take_profit", "tpsl": "tp", "close_size": 0.6, "oid": 1},
        ],
        use_resting_exit_orders=True,
        take_profit_legs_scale_from_initial_size=True,
        staged_exit_enabled=True,
        staged_exit_liquidity_band="normal_liquidity",
        initial_entry_price=100.0,
        initial_stop_price=99.0,
        initial_risk_price_distance=1.0,
        tp1_price=101.0,
        tp2_price=102.0,
        post_tp1_stop_price=99.6,
        locked_floor_price=101.0,
        max_favorable_excursion_r=0.65,
        history=deque(),
        runtimes={},
    )

    status = agent.step_risk_session(dict(before_snapshot), now=(8 * 900) + 1, fill_events=[])

    assert status == "take_profit_hit"
    assert reduce_calls == [("long", pytest.approx(0.6), "time_decay_take_profit", "risk", before_snapshot)]
    assert [item["key"] for item in cancel_calls[0]] == ["take_profit::stage_tp1"]
    assert agent.risk_session.tp1_hit is True
    assert ("stage_tp2", "limit", 102.0, pytest.approx(0.8)) in place_calls
    assert agent.risk_session.max_favorable_excursion_r == pytest.approx(0.65)


def test_step_risk_session_tp1_no_follow_through_trims_normal_liquidity_and_retargets_orders(uma):
    before_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 2.0,
        "entry_price": 100.0,
        "mid_price": 100.2,
        "notional_usd": 200.4,
        "max_leverage": 40,
    }
    after_snapshot = dict(before_snapshot, size=1.0)
    reader = FakeReader(symbol_snapshots=[dict(after_snapshot)])
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    cancel_calls = []
    reduce_calls = []
    place_calls = []
    executor.cancel_reduce_only_tpsl_orders = lambda refs, plan_name=None: cancel_calls.append((plan_name, [dict(item) for item in refs])) or {"accepted": True}
    executor.reduce_position = lambda side, close_size, reason, plan_name=None, position_before=None: reduce_calls.append((side, close_size, reason, plan_name, dict(position_before or {}))) or {"accepted": True, "actions": [{"market_close": {"status": "ok"}}]}
    executor.place_reduce_only_limit_order = (
        lambda side, close_size, limit_price, plan_name=None, leg_name="": place_calls.append((leg_name, "limit", limit_price, close_size))
        or {"accepted": True, "close_size": close_size, "limit_price": limit_price, "oid": 2000 + len(place_calls), "order_kind": "limit"}
    )
    executor.place_reduce_only_tpsl_order = (
        lambda side, close_size, trigger_price, tpsl, plan_name=None, leg_name="": place_calls.append((leg_name, tpsl, trigger_price, close_size))
        or {"accepted": True, "close_size": close_size, "trigger_price": trigger_price, "oid": 3000 + len(place_calls)}
    )
    agent = make_agent(uma, reader=reader, executor=executor)
    agent.risk_time_decay_tp_enabled = True
    agent.risk_time_decay_tp_timeframe_seconds = 300.0
    agent.risk_time_decay_normal_tp1_bars = 6.0
    agent.risk_time_decay_normal_tp1_mfe_r = 0.60
    agent.risk_time_decay_normal_tp1_current_r = 0.30
    agent._audit_event = lambda *args, **kwargs: None
    agent.risk_session = uma.RiskSession(
        plan_name="risk",
        side="long",
        stop_loss_price=99.0,
        start_time=0.0,
        baseline_size=2.0,
        expected_size=2.0,
        initial_size_abs=2.0,
        staged_exit_size_basis_abs=2.0,
        take_profit_legs=[
            uma.ExitLeg(name="stage_tp1", note="tp1", when_all=[uma.Condition(type="price_ge", level=101.0, note="tp1")], close_fraction=0.30),
            uma.ExitLeg(name="stage_tp2", note="tp2", when_all=[uma.Condition(type="price_ge", level=102.0, note="tp2")], close_fraction=0.40),
        ],
        stop_loss_legs=[
            uma.ExitLeg(name="stage_initial_stop", note="sl", when_all=[uma.Condition(type="price_le", level=99.0, note="sl")], close_fraction=1.0),
        ],
        resting_exit_orders=[
            {"key": "take_profit::stage_tp1", "name": "stage_tp1", "leg_type": "take_profit", "tpsl": "tp", "close_size": 0.6, "oid": 1},
            {"key": "take_profit::stage_tp2", "name": "stage_tp2", "leg_type": "take_profit", "tpsl": "tp", "close_size": 0.8, "oid": 2},
            {"key": "stop_loss::stage_initial_stop", "name": "stage_initial_stop", "leg_type": "stop_loss", "tpsl": "sl", "close_size": 2.0, "oid": 3},
        ],
        use_resting_exit_orders=True,
        take_profit_legs_scale_from_initial_size=True,
        staged_exit_enabled=True,
        staged_exit_liquidity_band="normal_liquidity",
        initial_entry_price=100.0,
        initial_stop_price=99.0,
        initial_risk_price_distance=1.0,
        tp1_price=101.0,
        tp2_price=102.0,
        tp1_hit=False,
        tp2_hit=False,
        post_tp1_stop_price=99.6,
        locked_floor_price=101.0,
        max_favorable_excursion_r=0.31,
        history=deque(),
        runtimes={},
    )

    status = agent.step_risk_session(dict(before_snapshot), now=(6 * 300) + 1, fill_events=[])

    assert status == "tp1_no_follow_through"
    assert reduce_calls == [("long", pytest.approx(1.0), "tp1_no_follow_through", "risk", before_snapshot)]
    assert [item["key"] for item in cancel_calls[0][1]] == [
        "take_profit::stage_tp1",
        "take_profit::stage_tp2",
        "stop_loss::stage_initial_stop",
    ]
    assert agent.risk_session is not None
    assert agent.risk_session.tp1_no_follow_through_applied is True
    assert agent.risk_session.tp1_no_follow_through_at == pytest.approx((6 * 300) + 1)
    assert agent.risk_session.expected_size == pytest.approx(1.0)
    assert agent.risk_session.tp1_hit is False
    assert agent.risk_session.tp1_hit_at == pytest.approx(0.0)
    assert agent.risk_session.executed_leg_names == set()
    assert [leg.name for leg in agent.risk_session.take_profit_legs] == ["stage_tp1", "stage_tp2"]
    assert agent.risk_session.staged_exit_size_basis_abs == pytest.approx(1.0)
    assert agent.risk_session.active_soft_stop_price == pytest.approx(99.6)
    assert ("stage_tp1", "limit", 101.0, pytest.approx(0.3)) in place_calls
    assert ("stage_tp2", "limit", 102.0, pytest.approx(0.4)) in place_calls


def test_step_risk_session_tp1_no_follow_through_trims_low_liquidity_and_tightens_stop(uma):
    before_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 2.0,
        "entry_price": 100.0,
        "mid_price": 100.1,
        "notional_usd": 200.2,
        "max_leverage": 40,
    }
    after_snapshot = dict(before_snapshot, size=1.0)
    reader = FakeReader(symbol_snapshots=[dict(after_snapshot)])
    executor = FakeExecutor(reader, "BTC")
    reduce_calls = []
    executor.reduce_position = lambda side, close_size, reason, plan_name=None, position_before=None: reduce_calls.append((side, close_size, reason, plan_name, dict(position_before or {}))) or {"accepted": True}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent.risk_time_decay_tp_enabled = True
    agent.risk_time_decay_tp_timeframe_seconds = 300.0
    agent.risk_time_decay_low_tp1_bars = 18.0
    agent.risk_time_decay_low_tp1_mfe_r = 0.30
    agent.risk_tp1_no_follow_through_normal_close_fraction = 0.50
    agent.risk_tp1_no_follow_through_normal_soft_stop_r = 0.40
    agent._audit_event = lambda *args, **kwargs: None
    agent.risk_session = uma.RiskSession(
        plan_name="risk",
        side="long",
        stop_loss_price=99.0,
        start_time=0.0,
        baseline_size=2.0,
        expected_size=2.0,
        initial_size_abs=2.0,
        staged_exit_size_basis_abs=2.0,
        take_profit_legs=[
            uma.ExitLeg(name="stage_tp1", note="tp1", when_all=[uma.Condition(type="price_ge", level=101.0, note="tp1")], close_fraction=0.40),
            uma.ExitLeg(name="stage_tp2", note="tp2", when_all=[uma.Condition(type="price_ge", level=101.25, note="tp2")], close_fraction=0.40),
        ],
        stop_loss_legs=[],
        resting_exit_orders=[],
        use_resting_exit_orders=False,
        take_profit_legs_scale_from_initial_size=True,
        staged_exit_enabled=True,
        staged_exit_liquidity_band="low_liquidity",
        initial_entry_price=100.0,
        initial_stop_price=99.0,
        initial_risk_price_distance=1.0,
        tp1_price=101.0,
        tp2_price=101.25,
        max_favorable_excursion_r=0.20,
        history=deque(),
        runtimes={},
    )

    status = agent.step_risk_session(dict(before_snapshot), now=(18 * 300) + 1, fill_events=[])

    assert status == "tp1_no_follow_through"
    assert reduce_calls == [("long", pytest.approx(1.0), "tp1_no_follow_through", "risk", before_snapshot)]
    assert agent.risk_session is not None
    assert agent.risk_session.tp1_no_follow_through_applied is True
    assert agent.risk_session.tp1_no_follow_through_at == pytest.approx((18 * 300) + 1)
    assert agent.risk_session.expected_size == pytest.approx(1.0)
    assert agent.risk_session.tp1_hit is False
    assert agent.risk_session.tp1_hit_at == pytest.approx(0.0)
    assert agent.risk_session.executed_leg_names == set()
    assert [leg.name for leg in agent.risk_session.take_profit_legs] == ["stage_tp1", "stage_tp2"]
    assert agent.risk_session.staged_exit_size_basis_abs == pytest.approx(1.0)
    assert agent.risk_session.active_soft_stop_price == pytest.approx(99.6)


def test_step_risk_session_tp2_no_continuation_trims_normal_liquidity_and_tightens_stop(uma):
    before_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 1.4,
        "entry_price": 100.0,
        "mid_price": 100.8,
        "notional_usd": 141.12,
        "max_leverage": 40,
    }
    after_snapshot = dict(before_snapshot, size=0.7)
    reader = FakeReader(symbol_snapshots=[dict(after_snapshot)])
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    cancel_calls = []
    reduce_calls = []
    place_calls = []
    executor.cancel_reduce_only_tpsl_orders = lambda refs, plan_name=None: cancel_calls.append((plan_name, [dict(item) for item in refs])) or {"accepted": True}
    executor.reduce_position = lambda side, close_size, reason, plan_name=None, position_before=None: reduce_calls.append((side, close_size, reason, plan_name, dict(position_before or {}))) or {"accepted": True, "actions": [{"market_close": {"status": "ok"}}]}
    executor.place_reduce_only_limit_order = (
        lambda side, close_size, limit_price, plan_name=None, leg_name="": place_calls.append((leg_name, "limit", limit_price, close_size))
        or {"accepted": True, "close_size": close_size, "limit_price": limit_price, "oid": 2000 + len(place_calls), "order_kind": "limit"}
    )
    executor.place_reduce_only_tpsl_order = (
        lambda side, close_size, trigger_price, tpsl, plan_name=None, leg_name="": place_calls.append((leg_name, tpsl, trigger_price, close_size))
        or {"accepted": True, "close_size": close_size, "trigger_price": trigger_price, "oid": 3000 + len(place_calls)}
    )
    agent = make_agent(uma, reader=reader, executor=executor)
    agent.risk_time_decay_tp_enabled = True
    agent.risk_time_decay_tp_timeframe_seconds = 300.0
    agent.risk_time_decay_normal_tp2_bars = 12.0
    agent.risk_time_decay_normal_tp2_mfe_r = 1.50
    agent.risk_time_decay_normal_tp2_current_r = 1.00
    agent.risk_tp2_no_continuation_normal_close_fraction = 0.50
    agent.risk_tp2_no_continuation_normal_soft_stop_r = 0.25
    agent._audit_event = lambda *args, **kwargs: None
    agent.risk_session = uma.RiskSession(
        plan_name="risk",
        side="long",
        stop_loss_price=99.6,
        start_time=0.0,
        baseline_size=1.4,
        expected_size=1.4,
        initial_size_abs=2.0,
        staged_exit_size_basis_abs=2.0,
        take_profit_legs=[
            uma.ExitLeg(name="stage_tp2", note="tp2", when_all=[uma.Condition(type="price_ge", level=102.0, note="tp2")], close_fraction=0.40),
        ],
        stop_loss_legs=[
            uma.ExitLeg(name="stage_post_tp1_stop", note="sl", when_all=[uma.Condition(type="price_le", level=99.6, note="sl")], close_fraction=1.0),
        ],
        resting_exit_orders=[
            {"key": "take_profit::stage_tp2", "name": "stage_tp2", "leg_type": "take_profit", "tpsl": "tp", "close_size": 0.8, "oid": 2},
            {"key": "stop_loss::stage_post_tp1_stop", "name": "stage_post_tp1_stop", "leg_type": "stop_loss", "tpsl": "sl", "close_size": 1.4, "oid": 3},
        ],
        use_resting_exit_orders=True,
        take_profit_legs_scale_from_initial_size=True,
        staged_exit_enabled=True,
        staged_exit_liquidity_band="normal_liquidity",
        initial_entry_price=100.0,
        initial_stop_price=99.0,
        initial_risk_price_distance=1.0,
        tp1_price=101.0,
        tp2_price=102.0,
        tp1_hit=True,
        tp1_hit_at=100.0,
        tp1_completed_size_abs=0.6,
        tp2_hit=False,
        post_tp1_stop_price=99.6,
        active_soft_stop_price=99.6,
        locked_floor_price=101.0,
        max_favorable_excursion_r=1.20,
        history=deque(),
        runtimes={},
    )

    status = agent.step_risk_session(dict(before_snapshot), now=100.0 + (12 * 300) + 1, fill_events=[])

    assert status == "tp2_no_continuation"
    assert reduce_calls == [("long", pytest.approx(0.7), "tp2_no_continuation", "risk", before_snapshot)]
    assert [item["key"] for item in cancel_calls[0][1]] == [
        "take_profit::stage_tp2",
        "stop_loss::stage_post_tp1_stop",
    ]
    assert agent.risk_session is not None
    assert agent.risk_session.tp2_no_continuation_applied is True
    assert agent.risk_session.expected_size == pytest.approx(0.7)
    assert agent.risk_session.tp1_hit is True
    assert agent.risk_session.tp2_hit is False
    assert agent.risk_session.staged_exit_size_basis_abs == pytest.approx(1.0)
    assert agent.risk_session.post_tp1_stop_price == pytest.approx(100.25)
    assert agent.risk_session.active_soft_stop_price == pytest.approx(100.25)
    assert ("stage_tp2", "limit", 102.0, pytest.approx(0.4)) in place_calls


def test_step_risk_session_tp2_no_continuation_flattens_when_tightened_stop_already_crossed(uma):
    before_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 1.4,
        "entry_price": 100.0,
        "mid_price": 100.1,
        "notional_usd": 140.14,
        "max_leverage": 40,
    }
    flat_snapshot = dict(before_snapshot, side="flat", size=0.0, entry_price=0.0, notional_usd=0.0)
    reader = FakeReader(symbol_snapshots=[dict(flat_snapshot)])
    executor = FakeExecutor(reader, "BTC")
    close_calls = []
    executor.close_position = lambda side, reason, plan_name=None, **kwargs: close_calls.append((side, reason, plan_name, dict(kwargs.get("position_before") or {}))) or {"accepted": True}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent.risk_time_decay_tp_enabled = True
    agent.risk_time_decay_tp_timeframe_seconds = 300.0
    agent.risk_time_decay_normal_tp2_bars = 12.0
    agent.risk_time_decay_normal_tp2_mfe_r = 1.50
    agent.risk_tp2_no_continuation_normal_soft_stop_r = 0.25
    agent._audit_event = lambda *args, **kwargs: None
    agent.risk_session = uma.RiskSession(
        plan_name="risk",
        side="long",
        stop_loss_price=99.6,
        start_time=0.0,
        baseline_size=1.4,
        expected_size=1.4,
        initial_size_abs=2.0,
        staged_exit_size_basis_abs=2.0,
        take_profit_legs=[
            uma.ExitLeg(name="stage_tp2", note="tp2", when_all=[uma.Condition(type="price_ge", level=102.0, note="tp2")], close_fraction=0.40),
        ],
        stop_loss_legs=[],
        resting_exit_orders=[],
        use_resting_exit_orders=False,
        take_profit_legs_scale_from_initial_size=True,
        staged_exit_enabled=True,
        staged_exit_liquidity_band="normal_liquidity",
        initial_entry_price=100.0,
        initial_stop_price=99.0,
        initial_risk_price_distance=1.0,
        tp1_price=101.0,
        tp2_price=102.0,
        tp1_hit=True,
        tp1_hit_at=100.0,
        tp1_completed_size_abs=0.6,
        post_tp1_stop_price=99.6,
        active_soft_stop_price=99.6,
        max_favorable_excursion_r=1.20,
        history=deque(),
        runtimes={},
    )

    status = agent.step_risk_session(dict(before_snapshot), now=100.0 + (12 * 300) + 1, fill_events=[])

    assert status == "tp2_no_continuation"
    assert close_calls == [("long", "tp2_no_continuation", "risk", before_snapshot)]
    assert agent.risk_session is None


def test_step_risk_session_tp2_no_continuation_trims_low_liquidity_and_tightens_stop(uma):
    before_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 1.2,
        "entry_price": 100.0,
        "mid_price": 100.7,
        "notional_usd": 120.84,
        "max_leverage": 40,
    }
    after_snapshot = dict(before_snapshot, size=0.6)
    reader = FakeReader(symbol_snapshots=[dict(after_snapshot)])
    executor = FakeExecutor(reader, "BTC")
    reduce_calls = []
    executor.reduce_position = lambda side, close_size, reason, plan_name=None, position_before=None: reduce_calls.append((side, close_size, reason, plan_name, dict(position_before or {}))) or {"accepted": True}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent.risk_time_decay_tp_enabled = True
    agent.risk_time_decay_tp_timeframe_seconds = 300.0
    agent.risk_time_decay_low_tp2_bars = 36.0
    agent.risk_time_decay_low_tp2_mfe_r = 0.75
    agent.risk_tp2_no_continuation_normal_close_fraction = 0.50
    agent.risk_tp2_no_continuation_normal_soft_stop_r = 0.25
    agent._audit_event = lambda *args, **kwargs: None
    agent.risk_session = uma.RiskSession(
        plan_name="risk",
        side="long",
        stop_loss_price=99.6,
        start_time=0.0,
        baseline_size=1.2,
        expected_size=1.2,
        initial_size_abs=2.0,
        staged_exit_size_basis_abs=2.0,
        take_profit_legs=[
            uma.ExitLeg(name="stage_tp2", note="tp2", when_all=[uma.Condition(type="price_ge", level=101.25, note="tp2")], close_fraction=0.40),
        ],
        stop_loss_legs=[],
        resting_exit_orders=[],
        use_resting_exit_orders=False,
        take_profit_legs_scale_from_initial_size=True,
        staged_exit_enabled=True,
        staged_exit_liquidity_band="low_liquidity",
        initial_entry_price=100.0,
        initial_stop_price=99.0,
        initial_risk_price_distance=1.0,
        tp1_price=100.75,
        tp2_price=101.25,
        tp1_hit=True,
        tp1_hit_at=100.0,
        tp1_completed_size_abs=0.8,
        max_favorable_excursion_r=0.60,
        history=deque(),
        runtimes={},
    )

    status = agent.step_risk_session(dict(before_snapshot), now=100.0 + (36 * 300) + 1, fill_events=[])

    assert status == "tp2_no_continuation"
    assert reduce_calls == [("long", pytest.approx(0.6), "tp2_no_continuation", "risk", before_snapshot)]
    assert agent.risk_session is not None
    assert agent.risk_session.tp2_no_continuation_applied is True
    assert agent.risk_session.expected_size == pytest.approx(0.6)
    assert agent.risk_session.post_tp1_stop_price == pytest.approx(100.25)
    assert agent.risk_session.active_soft_stop_price == pytest.approx(100.25)


def test_staged_risk_session_trim_risk_reduce_preserves_prices_and_resizes_remaining_orders(uma):
    before_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 8.57,
        "entry_price": 101.3608,
        "mid_price": 100.1,
        "notional_usd": 865.0,
        "max_leverage": 40,
    }
    after_snapshot = dict(before_snapshot, size=6.67, mid_price=100.1)
    reader = FakeReader(symbol_snapshots=[dict(before_snapshot), dict(after_snapshot)])
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    cancel_calls = []
    place_calls = []
    executor.cancel_reduce_only_tpsl_orders = lambda refs, plan_name=None: cancel_calls.append((plan_name, [dict(item) for item in refs])) or {"accepted": True}
    executor.place_reduce_only_tpsl_order = lambda side, close_size, trigger_price, tpsl, plan_name=None, leg_name="": place_calls.append((leg_name, tpsl, trigger_price, close_size)) or {"accepted": True, "close_size": close_size, "oid": 1000 + len(place_calls)}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent._audit_event = lambda *args, **kwargs: None
    agent.risk_session = uma.RiskSession(
        plan_name="risk",
        side="long",
        stop_loss_price=99.941,
        baseline_size=8.57,
        expected_size=8.57,
        initial_size_abs=8.57,
        staged_exit_size_basis_abs=8.57,
        take_profit_legs=[
            uma.ExitLeg(name="stage_tp1", note="tp1", when_all=[uma.Condition(type="price_ge", level=102.86, note="tp1")], close_fraction=0.30),
            uma.ExitLeg(name="stage_tp2", note="tp2", when_all=[uma.Condition(type="price_ge", level=104.32, note="tp2")], close_fraction=0.40),
        ],
        stop_loss_legs=[
            uma.ExitLeg(name="stage_initial_stop", note="sl", when_all=[uma.Condition(type="price_le", level=99.941, note="sl")], close_fraction=1.0),
        ],
        resting_exit_orders=[
            {"key": "take_profit::stage_tp1", "name": "stage_tp1", "leg_type": "take_profit", "tpsl": "tp", "close_size": 2.571, "oid": 1},
            {"key": "take_profit::stage_tp2", "name": "stage_tp2", "leg_type": "take_profit", "tpsl": "tp", "close_size": 3.428, "oid": 2},
            {"key": "stop_loss::stage_initial_stop", "name": "stage_initial_stop", "leg_type": "stop_loss", "tpsl": "sl", "close_size": 8.57, "oid": 3},
        ],
        use_resting_exit_orders=True,
        take_profit_legs_scale_from_initial_size=True,
        staged_exit_enabled=True,
        initial_entry_price=101.3608,
        initial_stop_price=99.941,
        initial_risk_price_distance=1.4198,
        tp1_price=102.86,
        tp2_price=104.32,
        post_tp1_stop_price=100.79,
        locked_floor_price=102.86,
        history=deque(),
        runtimes={},
    )

    reused = agent._reuse_staged_risk_session_after_trim(
        decision=make_management_decision(uma, action="trim", close_fraction=0.2),
        execution_result={"actions": [{"response": {"data": {"statuses": [{"filled": {"avgPx": "100.1"}}]}}}]},
        position_before=before_snapshot,
        position_after=after_snapshot,
    )

    assert reused is True
    assert agent.risk_session is not None
    assert agent.risk_session.initial_entry_price == pytest.approx(101.3608)
    assert agent.risk_session.initial_stop_price == pytest.approx(99.941)
    assert agent.risk_session.tp1_price == pytest.approx(102.86)
    assert agent.risk_session.tp2_price == pytest.approx(104.32)
    assert agent.risk_session.staged_exit_size_basis_abs == pytest.approx(6.67)
    assert agent.risk_session.tp1_hit is False
    assert agent.risk_session.tp2_hit is False
    assert len(cancel_calls) == 1
    assert place_calls == [
        ("stage_tp1", "tp", 102.86, pytest.approx(2.001)),
        ("stage_tp2", "tp", 104.32, pytest.approx(2.668)),
        ("stage_initial_stop", "sl", 99.941, pytest.approx(6.67)),
    ]


def test_sync_risk_session_places_take_profit_limits_and_stop_triggers(uma):
    reader = FakeReader()
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    calls = []
    executor.cancel_reduce_only_tpsl_orders = lambda refs, plan_name=None: {"accepted": True}
    executor.place_reduce_only_limit_order = (
        lambda side, close_size, limit_price, plan_name=None, leg_name="": calls.append(
            ("limit", leg_name, limit_price, close_size)
        )
        or {
            "accepted": True,
            "close_size": close_size,
            "limit_price": limit_price,
            "trigger_price": limit_price,
            "order_kind": "limit",
            "is_trigger": False,
            "oid": 8000 + len(calls),
        }
    )
    executor.place_reduce_only_tpsl_order = (
        lambda side, close_size, trigger_price, tpsl, plan_name=None, leg_name="": calls.append(
            ("trigger", leg_name, tpsl, trigger_price, close_size)
        )
        or {
            "accepted": True,
            "close_size": close_size,
            "trigger_price": trigger_price,
            "order_kind": "trigger",
            "is_trigger": True,
            "oid": 9000 + len(calls),
        }
    )
    agent = make_agent(uma, reader=reader, executor=executor)
    agent._audit_event = lambda *args, **kwargs: None
    session = uma.RiskSession(
        plan_name="risk",
        side="long",
        baseline_size=10.0,
        expected_size=10.0,
        take_profit_legs=[
            uma.ExitLeg(name="stage_tp1", note="tp1", when_all=[uma.Condition(type="price_ge", level=111.13)], close_fraction=0.30),
            uma.ExitLeg(name="stage_tp2", note="tp2", when_all=[uma.Condition(type="price_ge", level=112.04)], close_fraction=0.40),
        ],
        stop_loss_legs=[
            uma.ExitLeg(name="stage_initial_stop", note="sl", when_all=[uma.Condition(type="price_le", level=109.31)], close_fraction=1.0),
        ],
        take_profit_legs_scale_from_initial_size=True,
    )

    agent._sync_risk_session_resting_orders(session)

    assert calls == [
        ("limit", "stage_tp1", 111.13, pytest.approx(3.0)),
        ("limit", "stage_tp2", 112.04, pytest.approx(4.0)),
        ("trigger", "stage_initial_stop", "sl", 109.31, pytest.approx(10.0)),
    ]
    assert [(item["name"], item["order_kind"], item["is_trigger"]) for item in session.resting_exit_orders] == [
        ("stage_tp1", "limit", False),
        ("stage_tp2", "limit", False),
        ("stage_initial_stop", "trigger", True),
    ]


def test_staged_risk_session_trim_missing_initial_entry_falls_back_to_position_entry_without_rebuild(uma):
    before_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 8.57,
        "entry_price": 101.3608,
        "mid_price": 101.0,
        "notional_usd": 864.66,
        "max_leverage": 40,
    }
    after_snapshot = dict(before_snapshot, size=6.67, mid_price=100.1)
    reader = FakeReader(symbol_snapshots=[dict(before_snapshot), dict(after_snapshot)])
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    cancel_calls = []
    place_calls = []
    executor.cancel_reduce_only_tpsl_orders = lambda refs, plan_name=None: cancel_calls.append((plan_name, [dict(item) for item in refs])) or {"accepted": True}
    executor.place_reduce_only_tpsl_order = lambda side, close_size, trigger_price, tpsl, plan_name=None, leg_name="": place_calls.append((leg_name, tpsl, trigger_price, close_size)) or {"accepted": True, "close_size": close_size, "oid": 3000 + len(place_calls)}
    agent = make_agent(uma, reader=reader, executor=executor)
    audit_events = []
    agent._audit_event = lambda event_type, payload=None: audit_events.append((event_type, payload or {}))
    agent.risk_session = uma.RiskSession(
        plan_name="risk",
        side="long",
        stop_loss_price=100.79,
        baseline_size=8.57,
        expected_size=8.57,
        initial_size_abs=8.57,
        staged_exit_size_basis_abs=8.57,
        take_profit_legs=[
            uma.ExitLeg(name="stage_tp1", note="tp1", when_all=[uma.Condition(type="price_ge", level=102.86, note="tp1")], close_fraction=0.30),
            uma.ExitLeg(name="stage_tp2", note="tp2", when_all=[uma.Condition(type="price_ge", level=104.32, note="tp2")], close_fraction=0.40),
        ],
        stop_loss_legs=[
            uma.ExitLeg(name="stage_post_tp1_stop", note="sl", when_all=[uma.Condition(type="price_le", level=100.79, note="sl")], close_fraction=1.0),
        ],
        resting_exit_orders=[
            {"key": "take_profit::stage_tp1", "name": "stage_tp1", "leg_type": "take_profit", "tpsl": "tp", "close_size": 2.571, "oid": 1},
            {"key": "take_profit::stage_tp2", "name": "stage_tp2", "leg_type": "take_profit", "tpsl": "tp", "close_size": 3.428, "oid": 2},
            {"key": "stop_loss::stage_post_tp1_stop", "name": "stage_post_tp1_stop", "leg_type": "stop_loss", "tpsl": "sl", "close_size": 8.57, "oid": 3},
        ],
        use_resting_exit_orders=True,
        take_profit_legs_scale_from_initial_size=True,
        staged_exit_enabled=True,
        initial_entry_price=0.0,
        initial_stop_price=99.941,
        initial_risk_price_distance=1.4198,
        tp1_price=102.86,
        tp2_price=104.32,
        post_tp1_stop_price=100.79,
        locked_floor_price=102.86,
        history=deque(),
        runtimes={},
    )

    reused = agent._reuse_staged_risk_session_after_trim(
        decision=make_management_decision(uma, action="trim", close_fraction=0.2),
        execution_result={"actions": [{"response": {"data": {"statuses": [{"filled": {"avgPx": "100.1"}}]}}}]},
        position_before=before_snapshot,
        position_after=after_snapshot,
    )

    assert reused is True
    assert agent.risk_session.initial_entry_price == 0.0
    assert agent.risk_session.stop_loss_price == pytest.approx(100.79)
    assert agent.risk_session.tp1_price == pytest.approx(102.86)
    assert agent.risk_session.tp2_price == pytest.approx(104.32)
    assert agent.risk_session.staged_exit_size_basis_abs == pytest.approx(6.67)
    assert any(event_type == "risk_session_staged_trim_fallback_entry" for event_type, _ in audit_events)
    assert len(cancel_calls) == 1
    assert place_calls == [
        ("stage_tp1", "tp", 102.86, pytest.approx(2.001)),
        ("stage_tp2", "tp", 104.32, pytest.approx(2.668)),
        ("stage_post_tp1_stop", "sl", 100.79, pytest.approx(6.67)),
    ]


def test_staged_risk_session_trim_early_take_profit_consumes_tp_ladder(uma):
    before_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 10.0,
        "entry_price": 100.0,
        "mid_price": 103.0,
        "notional_usd": 1000.0,
        "max_leverage": 40,
    }
    after_snapshot = dict(before_snapshot, size=6.0, mid_price=103.0)
    reader = FakeReader(symbol_snapshots=[dict(before_snapshot), dict(after_snapshot)])
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    place_calls = []
    executor.cancel_reduce_only_tpsl_orders = lambda refs, plan_name=None: {"accepted": True}
    executor.place_reduce_only_tpsl_order = lambda side, close_size, trigger_price, tpsl, plan_name=None, leg_name="": place_calls.append((leg_name, tpsl, trigger_price, close_size)) or {"accepted": True, "close_size": close_size, "oid": 2000 + len(place_calls)}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent._audit_event = lambda *args, **kwargs: None
    agent.risk_session = uma.RiskSession(
        plan_name="risk",
        side="long",
        stop_loss_price=99.0,
        baseline_size=10.0,
        expected_size=10.0,
        initial_size_abs=10.0,
        staged_exit_size_basis_abs=10.0,
        take_profit_legs=[
            uma.ExitLeg(name="stage_tp1", note="tp1", when_all=[uma.Condition(type="price_ge", level=101.0, note="tp1")], close_fraction=0.30),
            uma.ExitLeg(name="stage_tp2", note="tp2", when_all=[uma.Condition(type="price_ge", level=102.0, note="tp2")], close_fraction=0.40),
        ],
        stop_loss_legs=[
            uma.ExitLeg(name="stage_initial_stop", note="sl", when_all=[uma.Condition(type="price_le", level=99.0, note="sl")], close_fraction=1.0),
        ],
        resting_exit_orders=[
            {"key": "take_profit::stage_tp1", "name": "stage_tp1", "leg_type": "take_profit", "tpsl": "tp", "close_size": 3.0, "oid": 1},
            {"key": "take_profit::stage_tp2", "name": "stage_tp2", "leg_type": "take_profit", "tpsl": "tp", "close_size": 4.0, "oid": 2},
            {"key": "stop_loss::stage_initial_stop", "name": "stage_initial_stop", "leg_type": "stop_loss", "tpsl": "sl", "close_size": 10.0, "oid": 3},
        ],
        use_resting_exit_orders=True,
        take_profit_legs_scale_from_initial_size=True,
        staged_exit_enabled=True,
        initial_entry_price=100.0,
        initial_stop_price=99.0,
        initial_risk_price_distance=1.0,
        tp1_price=101.0,
        tp2_price=102.0,
        post_tp1_stop_price=99.6,
        locked_floor_price=101.0,
        history=deque(),
        runtimes={},
    )

    reused = agent._reuse_staged_risk_session_after_trim(
        decision=make_management_decision(uma, action="trim", close_fraction=0.4),
        execution_result={"actions": [{"response": {"data": {"statuses": [{"filled": {"avgPx": "103.0"}}]}}}]},
        position_before=before_snapshot,
        position_after=after_snapshot,
    )

    assert reused is True
    assert agent.risk_session is not None
    assert agent.risk_session.tp1_hit is True
    assert agent.risk_session.tp2_hit is False
    assert agent.risk_session.tp1_completed_size_abs == pytest.approx(3.0)
    assert agent.risk_session.tp2_completed_size_abs == pytest.approx(1.0)
    assert agent.risk_session.stop_loss_price == pytest.approx(99.6)
    assert "take_profit::stage_tp1" in agent.risk_session.executed_leg_names
    assert place_calls == [
        ("stage_tp2", "tp", 102.0, pytest.approx(3.0)),
    ]


def test_step_risk_session_soft_trailing_stop_closes_tail_on_1m_close_confirmation(uma):
    close_snapshot = {
        "symbol": "BTC",
        "side": "flat",
        "size": 0.0,
        "entry_price": 0.0,
        "mid_price": 101300.0,
        "notional_usd": 0.0,
        "max_leverage": 40,
    }
    open_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 0.6,
        "entry_price": 100000.0,
        "mid_price": 101300.0,
        "notional_usd": 60000.0,
        "max_leverage": 40,
    }
    candles_15m = [
        {"t": 0, "h": 100500.0, "l": 99500.0, "c": 100200.0},
        {"t": 900000, "h": 101500.0, "l": 100100.0, "c": 101200.0},
        {"t": 1800000, "h": 102500.0, "l": 101000.0, "c": 102000.0},
        {"t": 2700000, "h": 102300.0, "l": 101100.0, "c": 101300.0},
    ]
    candles_1m = [
        {"t": 3480000, "h": 101600.0, "l": 101250.0, "c": 101500.0},
        {"t": 3540000, "h": 101550.0, "l": 101200.0, "c": 101300.0},
    ]
    reader = FakeReader(symbol_snapshots=[dict(close_snapshot)], candles_by_interval={"15m": candles_15m, "1m": candles_1m})
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    close_calls = []
    executor.close_position = lambda side, reason, plan_name=None: close_calls.append((side, reason, plan_name)) or {"accepted": True, "side": side, "reason": reason, "plan_name": plan_name}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent._audit_event = lambda *args, **kwargs: None
    agent.risk_session = uma.RiskSession(
        plan_name="risk",
        side="long",
        baseline_size=0.6,
        expected_size=0.6,
        initial_size_abs=2.0,
        take_profit_legs=[],
        stop_loss_legs=[
            uma.ExitLeg(name="stage_tail_hard_stop", note="sl", when_all=[uma.Condition(type="price_le", level=101000.0, note="sl")], close_fraction=1.0),
        ],
        use_resting_exit_orders=False,
        staged_exit_enabled=True,
        tp1_hit=True,
        tp2_hit=True,
        initial_entry_price=100000.0,
        initial_stop_price=99000.0,
        initial_risk_price_distance=1000.0,
        locked_floor_price=101000.0,
        trailing_timeframe="15m",
        trailing_atr_period=2,
        trailing_soft_atr_mult=0.5,
        trailing_hard_atr_mult=1.0,
        start_time=0.0,
        history=deque(),
        runtimes={},
    )

    status = agent.step_risk_session(dict(open_snapshot), now=3600.0, fill_events=[])

    assert status == "soft_trailing_stop_hit"
    assert close_calls == [("long", "soft_trailing_stop", "risk")]
    assert agent.risk_session is None


def test_staged_trailing_recalculation_preserves_existing_tight_tail_stop(uma):
    long_candles_15m = [
        {"t": index * 900000, "h": 102.0, "l": 100.0, "c": 101.0}
        for index in range(24)
    ]
    reader = FakeReader(candles_by_interval={"15m": long_candles_15m})
    agent = make_agent(uma, reader=reader)
    agent.risk_session = uma.RiskSession(
        plan_name="risk",
        side="long",
        baseline_size=0.6,
        expected_size=0.6,
        initial_size_abs=2.0,
        take_profit_legs=[],
        stop_loss_legs=[],
        use_resting_exit_orders=False,
        staged_exit_enabled=True,
        tp1_hit=True,
        tp2_hit=True,
        initial_entry_price=100.0,
        initial_stop_price=99.0,
        initial_risk_price_distance=1.0,
        locked_floor_price=101.0,
        trailing_soft_stop_price=103.0,
        trailing_timeframe="15m",
        trailing_atr_period=2,
        trailing_atr_lookback_bars=24,
        trailing_soft_atr_mult=0.5,
        trailing_hard_atr_mult=1.0,
        start_time=0.0,
        history=deque(),
        runtimes={},
    )

    updated = agent._update_staged_risk_session_trailing_state(agent.risk_session, now=(24 * 900) + 1)

    assert updated is True
    assert agent.risk_session.trailing_soft_stop_price == pytest.approx(103.0)
    assert agent.risk_session.stop_loss_price == pytest.approx(103.0)

    short_candles_15m = [
        {"t": index * 900000, "h": 97.0, "l": 95.0, "c": 96.0}
        for index in range(24)
    ]
    reader = FakeReader(candles_by_interval={"15m": short_candles_15m})
    agent = make_agent(uma, reader=reader)
    agent.risk_session = uma.RiskSession(
        plan_name="risk",
        side="short",
        baseline_size=-0.6,
        expected_size=-0.6,
        initial_size_abs=2.0,
        take_profit_legs=[],
        stop_loss_legs=[],
        use_resting_exit_orders=False,
        staged_exit_enabled=True,
        tp1_hit=True,
        tp2_hit=True,
        initial_entry_price=100.0,
        initial_stop_price=101.0,
        initial_risk_price_distance=1.0,
        locked_floor_price=98.0,
        trailing_soft_stop_price=96.5,
        trailing_timeframe="15m",
        trailing_atr_period=2,
        trailing_atr_lookback_bars=24,
        trailing_soft_atr_mult=0.5,
        trailing_hard_atr_mult=1.0,
        start_time=0.0,
        history=deque(),
        runtimes={},
    )

    updated = agent._update_staged_risk_session_trailing_state(agent.risk_session, now=(24 * 900) + 1)

    assert updated is True
    assert agent.risk_session.trailing_soft_stop_price == pytest.approx(96.5)
    assert agent.risk_session.stop_loss_price == pytest.approx(96.5)


def test_step_risk_session_reconciles_missed_user_fill_via_exchange_order_status(uma):
    open_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 2.25,
        "entry_price": 100000.0,
        "mid_price": 98950.0,
        "notional_usd": 222637.5,
        "max_leverage": 40,
    }
    reader = FakeReader(
        symbol_snapshots=[dict(open_snapshot)],
        frontend_open_orders=[
            {"coin": "BTC", "oid": 1001},
            {"coin": "BTC", "oid": 1002},
            {"coin": "BTC", "oid": 1003},
            {"coin": "BTC", "oid": 1005},
        ],
        order_statuses={1004: {"status": "filled"}},
    )
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    calls = []
    executor.modify_reduce_only_tpsl_orders = lambda order_updates, side, plan_name=None: calls.append(("modify", plan_name, side, [dict(item) for item in order_updates])) or {"accepted": True, "updated_refs": [dict(item) for item in order_updates]}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent._audit_event = lambda *args, **kwargs: None
    agent.enable_user_fills_websocket = True
    agent.user_fills_subscription_id = 1
    agent.risk_session = uma.RiskSession(
        plan_name="risk",
        side="long",
        initial_entry_price=100000.0,
        baseline_size=3.0,
        expected_size=3.0,
        initial_size_abs=3.0,
        take_profit_legs=[
            uma.ExitLeg(name="tp-1", note="tp", when_all=[uma.Condition(type="price_ge", level=101000.0, note="tp")], close_fraction=1.0 / 3.0),
            uma.ExitLeg(name="tp-2", note="tp", when_all=[uma.Condition(type="price_ge", level=102000.0, note="tp")], close_fraction=1.0 / 3.0),
            uma.ExitLeg(name="tp-3", note="tp", when_all=[uma.Condition(type="price_ge", level=103000.0, note="tp")], close_fraction=1.0 / 3.0),
        ],
        stop_loss_legs=[
            uma.ExitLeg(name="sl-1", note="sl", when_all=[uma.Condition(type="price_le", level=99000.0, note="sl")], close_fraction=0.25),
            uma.ExitLeg(name="sl-2", note="sl", when_all=[uma.Condition(type="price_le", level=98000.0, note="sl")], close_fraction=0.75),
        ],
        history=deque(),
        runtimes={},
        resting_exit_orders=[
            {"key": "take_profit::tp-1", "name": "tp-1", "leg_type": "take_profit", "tpsl": "tp", "close_size": 1.0, "oid": 1001, "cloid": "0x1"},
            {"key": "take_profit::tp-2", "name": "tp-2", "leg_type": "take_profit", "tpsl": "tp", "close_size": 1.0, "oid": 1002, "cloid": "0x2"},
            {"key": "take_profit::tp-3", "name": "tp-3", "leg_type": "take_profit", "tpsl": "tp", "close_size": 1.0, "oid": 1003, "cloid": "0x3"},
            {"key": "stop_loss::sl-1", "name": "sl-1", "leg_type": "stop_loss", "tpsl": "sl", "close_size": 0.75, "oid": 1004, "cloid": "0x4"},
            {"key": "stop_loss::sl-2", "name": "sl-2", "leg_type": "stop_loss", "tpsl": "sl", "close_size": 2.25, "oid": 1005, "cloid": "0x5"},
        ],
        use_resting_exit_orders=True,
    )

    status = agent.step_risk_session(dict(open_snapshot), now=100.0, fill_events=[])

    assert status == "stop_loss_hit"
    assert agent.risk_session is not None
    assert "stop_loss::sl-1" in agent.risk_session.executed_leg_names
    assert agent.risk_session.expected_size == pytest.approx(2.25)
    modify_calls = [item for item in calls if item[0] == "modify"]
    assert len(modify_calls) == 1
    assert {(item["name"], item["tpsl"], item["close_size"]) for item in modify_calls[0][3]} == {("tp-1", "tp", 0.75), ("tp-2", "tp", 0.75), ("tp-3", "tp", 0.75)}

def test_step_risk_session_waits_for_user_fills_before_classifying_resting_exit_hit(uma):
    open_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 2.0,
        "entry_price": 100000.0,
        "mid_price": 101000.0,
        "notional_usd": 200000.0,
        "max_leverage": 40,
    }
    reader = FakeReader(symbol_snapshots=[dict(open_snapshot), dict(open_snapshot)])
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    calls = []
    executor.modify_reduce_only_tpsl_orders = lambda order_updates, side, plan_name=None: calls.append(("modify", plan_name, side, [dict(item) for item in order_updates])) or {"accepted": True, "updated_refs": [dict(item) for item in order_updates]}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent._audit_event = lambda *args, **kwargs: None
    agent.enable_user_fills_websocket = True
    agent.user_fills_subscription_id = 1
    agent.user_fills_reconcile_grace_seconds = 3.0
    agent.risk_session = uma.RiskSession(
        plan_name="risk",
        side="long",
        initial_entry_price=100000.0,
        baseline_size=3.0,
        expected_size=3.0,
        initial_size_abs=3.0,
        take_profit_legs=[
            uma.ExitLeg(name="tp-1", note="tp", when_all=[uma.Condition(type="price_ge", level=101000.0, note="tp")], close_fraction=1.0 / 3.0),
            uma.ExitLeg(name="tp-2", note="tp", when_all=[uma.Condition(type="price_ge", level=102000.0, note="tp")], close_fraction=1.0 / 3.0),
            uma.ExitLeg(name="tp-3", note="tp", when_all=[uma.Condition(type="price_ge", level=103000.0, note="tp")], close_fraction=1.0 / 3.0),
        ],
        stop_loss_legs=[
            uma.ExitLeg(name="sl-1", note="sl", when_all=[uma.Condition(type="price_le", level=99000.0, note="sl")], close_fraction=0.25),
            uma.ExitLeg(name="sl-2", note="sl", when_all=[uma.Condition(type="price_le", level=98000.0, note="sl")], close_fraction=0.75),
        ],
        history=deque(),
        runtimes={},
        resting_exit_orders=[
            {"key": "take_profit::tp-1", "name": "tp-1", "leg_type": "take_profit", "tpsl": "tp", "close_size": 1.0, "oid": 1001, "cloid": "0x1"},
            {"key": "take_profit::tp-2", "name": "tp-2", "leg_type": "take_profit", "tpsl": "tp", "close_size": 1.0, "oid": 1002, "cloid": "0x2"},
            {"key": "take_profit::tp-3", "name": "tp-3", "leg_type": "take_profit", "tpsl": "tp", "close_size": 1.0, "oid": 1003, "cloid": "0x3"},
            {"key": "stop_loss::sl-1", "name": "sl-1", "leg_type": "stop_loss", "tpsl": "sl", "close_size": 0.75, "oid": 1004, "cloid": "0x4"},
            {"key": "stop_loss::sl-2", "name": "sl-2", "leg_type": "stop_loss", "tpsl": "sl", "close_size": 2.25, "oid": 1005, "cloid": "0x5"},
        ],
        use_resting_exit_orders=True,
    )

    status = agent.step_risk_session(dict(open_snapshot), now=100.0, fill_events=[])

    assert status is None
    assert agent.risk_session is not None
    assert agent.risk_session.pending_fill_reconcile_since is None
    assert agent.risk_session.executed_leg_names == set()
    assert calls == []

    status = agent.step_risk_session(
        dict(open_snapshot),
        now=101.0,
        fill_events=[{"coin": "BTC", "oid": 1001, "tid": 1, "time": 100000, "sz": "1.0", "px": "101000"}],
    )

    assert status == "take_profit_hit"
    assert agent.risk_session is not None
    assert "take_profit::tp-1" in agent.risk_session.executed_leg_names
    modify_calls = [item for item in calls if item[0] == "modify"]
    assert len(modify_calls) == 1
    assert {(item["name"], item["tpsl"], item["close_size"]) for item in modify_calls[0][3]} == {("sl-1", "sl", 0.5), ("sl-2", "sl", 1.5)}
def test_step_risk_session_resting_reduce_only_fill_does_not_repost_if_counterpart_orders_still_open(uma):
    open_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 2.0,
        "entry_price": 100000.0,
        "mid_price": 101000.0,
        "notional_usd": 200000.0,
        "max_leverage": 40,
    }
    reader = FakeReader(
        symbol_snapshots=[dict(open_snapshot)],
        frontend_open_orders=[
            {
                "coin": "BTC",
                "reduceOnly": True,
                "isTrigger": True,
                "orderType": "Stop Limit",
                "triggerPx": "99000",
                "origSz": "0.75",
            },
            {
                "coin": "BTC",
                "reduceOnly": True,
                "isTrigger": True,
                "orderType": "Stop Limit",
                "triggerPx": "98000",
                "origSz": "2.25",
            },
        ],
    )
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    calls = []

    def fake_cancel(order_refs, plan_name=None):
        calls.append(("cancel", plan_name, [dict(item) for item in order_refs]))
        return {
            "accepted": False,
            "message": "Exchange rejected risk order cancellation.",
            "actions": [
                {
                    "cancel_reduce_only_tpsl_orders": {
                        "response": {
                            "data": {
                                "statuses": [
                                    {"error": "Order was never placed, already canceled, or filled. asset=110049"},
                                    {"error": "Order was never placed, already canceled, or filled. asset=110049"},
                                ]
                            }
                        }
                    }
                }
            ],
        }

    executor.modify_reduce_only_tpsl_orders = lambda order_updates, side, plan_name=None: calls.append(("modify", plan_name, side, [dict(item) for item in order_updates])) or {"accepted": True, "updated_refs": [dict(item) for item in order_updates]}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent._audit_event = lambda *args, **kwargs: None
    agent.risk_session = uma.RiskSession(
        plan_name="risk",
        side="long",
        initial_entry_price=100000.0,
        baseline_size=3.0,
        expected_size=3.0,
        initial_size_abs=3.0,
        take_profit_legs=[
            uma.ExitLeg(name="tp-1", note="tp", when_all=[uma.Condition(type="price_ge", level=101000.0, note="tp")], close_fraction=1.0 / 3.0),
            uma.ExitLeg(name="tp-2", note="tp", when_all=[uma.Condition(type="price_ge", level=102000.0, note="tp")], close_fraction=1.0 / 3.0),
            uma.ExitLeg(name="tp-3", note="tp", when_all=[uma.Condition(type="price_ge", level=103000.0, note="tp")], close_fraction=1.0 / 3.0),
        ],
        stop_loss_legs=[
            uma.ExitLeg(name="sl-1", note="sl", when_all=[uma.Condition(type="price_le", level=99000.0, note="sl")], close_fraction=0.25),
            uma.ExitLeg(name="sl-2", note="sl", when_all=[uma.Condition(type="price_le", level=98000.0, note="sl")], close_fraction=0.75),
        ],
        history=deque(),
        runtimes={},
        resting_exit_orders=[
            {"key": "take_profit::tp-1", "name": "tp-1", "leg_type": "take_profit", "tpsl": "tp", "trigger_price": 101000.0, "close_size": 1.0, "cloid": "0x1"},
            {"key": "take_profit::tp-2", "name": "tp-2", "leg_type": "take_profit", "tpsl": "tp", "trigger_price": 102000.0, "close_size": 1.0, "cloid": "0x2"},
            {"key": "take_profit::tp-3", "name": "tp-3", "leg_type": "take_profit", "tpsl": "tp", "trigger_price": 103000.0, "close_size": 1.0, "cloid": "0x3"},
            {"key": "stop_loss::sl-1", "name": "sl-1", "leg_type": "stop_loss", "tpsl": "sl", "trigger_price": 99000.0, "close_size": 0.75, "cloid": "0x4"},
            {"key": "stop_loss::sl-2", "name": "sl-2", "leg_type": "stop_loss", "tpsl": "sl", "trigger_price": 98000.0, "close_size": 2.25, "cloid": "0x5"},
        ],
        use_resting_exit_orders=True,
    )

    status = agent.step_risk_session(dict(open_snapshot), now=100.0)

    assert status == "take_profit_hit"
    modify_calls = [item for item in calls if item[0] == "modify"]
    assert len(modify_calls) == 1
    stop_refs = {item["name"]: item["close_size"] for item in agent.risk_session.resting_exit_orders if item["leg_type"] == "stop_loss"}
    assert stop_refs == {"sl-1": 0.5, "sl-2": 1.5}


def test_resync_counterpart_orders_rebuilds_multi_order_key_mismatch_instead_of_zipping(uma):
    reader = FakeReader()
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    calls = []

    def fake_modify(order_updates, side, plan_name=None):
        calls.append(("modify", plan_name, side, [dict(item) for item in order_updates]))
        return {"accepted": True, "updated_refs": [dict(item) for item in order_updates]}

    def fake_cancel(order_refs, plan_name=None):
        calls.append(("cancel", plan_name, [dict(item) for item in order_refs]))
        return {"accepted": True}

    def fake_place(side, close_size, trigger_price, tpsl, plan_name=None, leg_name=""):
        calls.append(("place", plan_name, side, leg_name, tpsl, trigger_price, close_size))
        return {
            "accepted": True,
            "oid": 1000 + len(calls),
            "cloid": f"0x{len(calls)}",
            "close_size": close_size,
        }

    executor.modify_reduce_only_tpsl_orders = fake_modify
    executor.cancel_reduce_only_tpsl_orders = fake_cancel
    executor.place_reduce_only_tpsl_order = fake_place
    agent = make_agent(uma, reader=reader, executor=executor)
    agent._audit_event = lambda *args, **kwargs: None
    session = uma.RiskSession(
        plan_name="risk",
        side="long",
        initial_entry_price=100000.0,
        baseline_size=2.0,
        expected_size=2.0,
        take_profit_legs=[
            uma.ExitLeg(name="tp-a", note="tp", when_all=[uma.Condition(type="price_ge", level=101000.0, note="tp")], close_fraction=0.5),
            uma.ExitLeg(name="tp-b", note="tp", when_all=[uma.Condition(type="price_ge", level=102000.0, note="tp")], close_fraction=0.5),
        ],
        stop_loss_legs=[
            uma.ExitLeg(name="sl-1", note="sl", when_all=[uma.Condition(type="price_le", level=99000.0, note="sl")], close_fraction=1.0),
        ],
        history=deque(),
        runtimes={},
        executed_leg_names={"stop_loss::sl-1"},
        resting_exit_orders=[
            {"key": "take_profit::old-a", "name": "old-a", "leg_type": "take_profit", "tpsl": "tp", "trigger_price": 101000.0, "close_size": 1.0, "cloid": "0x1"},
            {"key": "take_profit::old-b", "name": "old-b", "leg_type": "take_profit", "tpsl": "tp", "trigger_price": 102000.0, "close_size": 1.0, "cloid": "0x2"},
        ],
        use_resting_exit_orders=True,
    )

    agent._resync_risk_session_counterpart_orders(session, "stop_loss")

    assert [item for item in calls if item[0] == "modify"] == []
    assert len([item for item in calls if item[0] == "cancel"]) == 1
    place_calls = [item for item in calls if item[0] == "place"]
    assert [(item[3], item[4], item[5], item[6]) for item in place_calls] == [
        ("tp-a", "tp", 101000.0, 1.0),
        ("tp-b", "tp", 102000.0, 1.0),
    ]
    assert {item["key"] for item in session.resting_exit_orders} == {"take_profit::tp-a", "take_profit::tp-b"}


def test_resync_counterpart_orders_rebuilds_limit_take_profit_refs(uma):
    reader = FakeReader()
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    calls = []

    executor.modify_reduce_only_tpsl_orders = lambda order_updates, side, plan_name=None: calls.append(("modify", [dict(item) for item in order_updates])) or {"accepted": True}
    executor.cancel_reduce_only_tpsl_orders = lambda order_refs, plan_name=None: calls.append(("cancel", [dict(item) for item in order_refs])) or {"accepted": True}
    executor.place_reduce_only_limit_order = (
        lambda side, close_size, limit_price, plan_name=None, leg_name="": calls.append(("limit", leg_name, limit_price, close_size))
        or {
            "accepted": True,
            "close_size": close_size,
            "limit_price": limit_price,
            "trigger_price": limit_price,
            "order_kind": "limit",
            "is_trigger": False,
            "oid": 7000 + len(calls),
        }
    )
    executor.place_reduce_only_tpsl_order = (
        lambda side, close_size, trigger_price, tpsl, plan_name=None, leg_name="": calls.append(("trigger", leg_name, tpsl, trigger_price, close_size))
        or {"accepted": True, "close_size": close_size, "trigger_price": trigger_price, "order_kind": "trigger", "is_trigger": True}
    )
    agent = make_agent(uma, reader=reader, executor=executor)
    agent._audit_event = lambda *args, **kwargs: None
    session = uma.RiskSession(
        plan_name="risk",
        side="long",
        initial_entry_price=100000.0,
        baseline_size=2.0,
        expected_size=2.0,
        take_profit_legs=[
            uma.ExitLeg(name="tp-a", note="tp", when_all=[uma.Condition(type="price_ge", level=101000.0, note="tp")], close_fraction=0.5),
            uma.ExitLeg(name="tp-b", note="tp", when_all=[uma.Condition(type="price_ge", level=102000.0, note="tp")], close_fraction=0.5),
        ],
        stop_loss_legs=[
            uma.ExitLeg(name="sl-1", note="sl", when_all=[uma.Condition(type="price_le", level=99000.0, note="sl")], close_fraction=1.0),
        ],
        history=deque(),
        runtimes={},
        executed_leg_names={"stop_loss::sl-1"},
        resting_exit_orders=[
            {"key": "take_profit::tp-a", "name": "tp-a", "leg_type": "take_profit", "tpsl": "tp", "trigger_price": 101000.0, "limit_price": 101000.0, "close_size": 1.0, "cloid": "0x1", "order_kind": "limit", "is_trigger": False},
            {"key": "take_profit::tp-b", "name": "tp-b", "leg_type": "take_profit", "tpsl": "tp", "trigger_price": 102000.0, "limit_price": 102000.0, "close_size": 1.0, "cloid": "0x2", "order_kind": "limit", "is_trigger": False},
        ],
        use_resting_exit_orders=True,
    )

    agent._resync_risk_session_counterpart_orders(session, "stop_loss")

    assert [item for item in calls if item[0] == "modify"] == []
    assert [item[0] for item in calls] == ["cancel", "limit", "limit"]
    assert [(item["name"], item["order_kind"], item["is_trigger"]) for item in session.resting_exit_orders] == [
        ("tp-a", "limit", False),
        ("tp-b", "limit", False),
    ]


def test_step_risk_session_resting_reduce_only_stop_fill_reposts_take_profit_orders(uma):
    open_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 2.25,
        "entry_price": 100000.0,
        "mid_price": 98950.0,
        "notional_usd": 222637.5,
        "max_leverage": 40,
    }
    reader = FakeReader(symbol_snapshots=[dict(open_snapshot)])
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    calls = []
    executor.modify_reduce_only_tpsl_orders = lambda order_updates, side, plan_name=None: calls.append(("modify", plan_name, side, [dict(item) for item in order_updates])) or {"accepted": True, "updated_refs": [dict(item) for item in order_updates]}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent._audit_event = lambda *args, **kwargs: None
    agent.risk_session = uma.RiskSession(
        plan_name="risk",
        side="long",
        initial_entry_price=100000.0,
        baseline_size=3.0,
        expected_size=3.0,
        initial_size_abs=3.0,
        take_profit_legs=[
            uma.ExitLeg(name="tp-1", note="tp", when_all=[uma.Condition(type="price_ge", level=101000.0, note="tp")], close_fraction=1.0 / 3.0),
            uma.ExitLeg(name="tp-2", note="tp", when_all=[uma.Condition(type="price_ge", level=102000.0, note="tp")], close_fraction=1.0 / 3.0),
            uma.ExitLeg(name="tp-3", note="tp", when_all=[uma.Condition(type="price_ge", level=103000.0, note="tp")], close_fraction=1.0 / 3.0),
        ],
        stop_loss_legs=[
            uma.ExitLeg(name="sl-1", note="sl", when_all=[uma.Condition(type="price_le", level=99000.0, note="sl")], close_fraction=0.25),
            uma.ExitLeg(name="sl-2", note="sl", when_all=[uma.Condition(type="price_le", level=98000.0, note="sl")], close_fraction=0.75),
        ],
        history=deque(),
        runtimes={},
        resting_exit_orders=[
            {"key": "take_profit::tp-1", "name": "tp-1", "leg_type": "take_profit", "tpsl": "tp", "close_size": 1.0, "cloid": "0x1"},
            {"key": "take_profit::tp-2", "name": "tp-2", "leg_type": "take_profit", "tpsl": "tp", "close_size": 1.0, "cloid": "0x2"},
            {"key": "take_profit::tp-3", "name": "tp-3", "leg_type": "take_profit", "tpsl": "tp", "close_size": 1.0, "cloid": "0x3"},
            {"key": "stop_loss::sl-1", "name": "sl-1", "leg_type": "stop_loss", "tpsl": "sl", "close_size": 0.75, "cloid": "0x4"},
            {"key": "stop_loss::sl-2", "name": "sl-2", "leg_type": "stop_loss", "tpsl": "sl", "close_size": 2.25, "cloid": "0x5"},
        ],
        use_resting_exit_orders=True,
    )

    status = agent.step_risk_session(dict(open_snapshot), now=100.0)

    assert status == "stop_loss_hit"
    assert agent.risk_session is not None
    assert "stop_loss::sl-1" in agent.risk_session.executed_leg_names
    assert agent.risk_session.expected_size == pytest.approx(2.25)
    modify_calls = [item for item in calls if item[0] == "modify"]
    assert len(modify_calls) == 1
    assert modify_calls[0][2] == "long"
    assert {(item["name"], item["tpsl"], item["close_size"]) for item in modify_calls[0][3]} == {("tp-1", "tp", 0.75), ("tp-2", "tp", 0.75), ("tp-3", "tp", 0.75)}
    assert {item["name"] for item in agent.risk_session.resting_exit_orders} == {"tp-1", "tp-2", "tp-3", "sl-2"}
    stop_refs = {item["name"]: item["close_size"] for item in agent.risk_session.resting_exit_orders if item["leg_type"] == "stop_loss"}
    assert stop_refs == {"sl-2": 2.25}
    take_profit_refs = {item["name"]: item["close_size"] for item in agent.risk_session.resting_exit_orders if item["leg_type"] == "take_profit"}
    assert take_profit_refs == {"tp-1": 0.75, "tp-2": 0.75, "tp-3": 0.75}



def test_on_user_fills_ws_message_accepts_snapshot_and_dedupes(uma):
    agent = make_agent(uma)

    snapshot_msg = {
        "data": {
            "isSnapshot": True,
            "fills": [
                {"coin": "BTC", "oid": 1001, "tid": 77, "time": 101000, "sz": "1.0", "px": "101000"}
            ],
        }
    }

    agent._on_user_fills_ws_message(snapshot_msg)
    agent._on_user_fills_ws_message(snapshot_msg)

    drained = agent._drain_pending_user_fill_events()

    assert len(drained) == 1
    assert drained[0]["oid"] == 1001
    assert agent.user_fills_last_fill_time_ms == 101000


def test_user_fills_ws_is_active_requires_reader_health(uma):
    reader = FakeReader(ws_healthy=False)
    agent = make_agent(uma, reader=reader)
    agent.enable_user_fills_websocket = True
    agent.user_fills_subscription_id = 1

    assert agent._user_fills_ws_is_active() is False

    reader._ws_healthy = True
    assert agent._user_fills_ws_is_active() is True


def test_step_risk_session_backfills_missed_user_fill_from_rest(uma):
    open_snapshot = {
        "symbol": "BTC",
        "side": "long",
        "size": 2.0,
        "entry_price": 100000.0,
        "mid_price": 101000.0,
        "notional_usd": 202000.0,
        "max_leverage": 40,
    }
    reader = FakeReader(
        symbol_snapshots=[dict(open_snapshot)],
        user_fills_by_time=[
            {"coin": "BTC", "oid": 1001, "tid": 1, "time": 101000, "sz": "1.0", "px": "101000"}
        ],
    )
    executor = FakeExecutor(reader, "BTC")
    executor.enabled = True
    calls = []
    executor.modify_reduce_only_tpsl_orders = lambda order_updates, side, plan_name=None: calls.append(("modify", plan_name, side, [dict(item) for item in order_updates])) or {"accepted": True, "updated_refs": [dict(item) for item in order_updates]}
    agent = make_agent(uma, reader=reader, executor=executor)
    agent.enable_user_fills_websocket = True
    agent.user_fills_subscription_id = 1
    agent.user_fills_backfill_poll_seconds = 0.0
    agent.risk_session = uma.RiskSession(
        plan_name="risk",
        side="long",
        initial_entry_price=100000.0,
        baseline_size=3.0,
        expected_size=3.0,
        initial_size_abs=3.0,
        take_profit_legs=[
            uma.ExitLeg(name="tp-1", note="tp", when_all=[uma.Condition(type="price_ge", level=101000.0, note="tp")], close_fraction=1.0 / 3.0),
            uma.ExitLeg(name="tp-2", note="tp", when_all=[uma.Condition(type="price_ge", level=102000.0, note="tp")], close_fraction=1.0 / 3.0),
            uma.ExitLeg(name="tp-3", note="tp", when_all=[uma.Condition(type="price_ge", level=103000.0, note="tp")], close_fraction=1.0 / 3.0),
        ],
        stop_loss_legs=[
            uma.ExitLeg(name="sl-1", note="sl", when_all=[uma.Condition(type="price_le", level=99000.0, note="sl")], close_fraction=0.25),
            uma.ExitLeg(name="sl-2", note="sl", when_all=[uma.Condition(type="price_le", level=98000.0, note="sl")], close_fraction=0.75),
        ],
        history=deque(),
        runtimes={},
        resting_exit_orders=[
            {"key": "take_profit::tp-1", "name": "tp-1", "leg_type": "take_profit", "tpsl": "tp", "close_size": 1.0, "oid": 1001, "cloid": "0x1"},
            {"key": "take_profit::tp-2", "name": "tp-2", "leg_type": "take_profit", "tpsl": "tp", "close_size": 1.0, "oid": 1002, "cloid": "0x2"},
            {"key": "take_profit::tp-3", "name": "tp-3", "leg_type": "take_profit", "tpsl": "tp", "close_size": 1.0, "oid": 1003, "cloid": "0x3"},
            {"key": "stop_loss::sl-1", "name": "sl-1", "leg_type": "stop_loss", "tpsl": "sl", "close_size": 0.75, "oid": 1004, "cloid": "0x4"},
            {"key": "stop_loss::sl-2", "name": "sl-2", "leg_type": "stop_loss", "tpsl": "sl", "close_size": 2.25, "oid": 1005, "cloid": "0x5"},
        ],
        use_resting_exit_orders=True,
    )

    status = agent.step_risk_session(dict(open_snapshot), now=101.0, fill_events=[])

    assert status == "take_profit_hit"
    assert reader.user_fills_queries
    assert agent.risk_session is not None
    assert "take_profit::tp-1" in agent.risk_session.executed_leg_names
    modify_calls = [item for item in calls if item[0] == "modify"]
    assert len(modify_calls) == 1


def test_build_management_session_refresh_no_change_uses_plan_entry(uma):
    agent = make_agent(uma, reader=FakeReader())
    agent._audit_event = lambda *args, **kwargs: None
    position_after = {
        "symbol": "BTC",
        "side": "long",
        "size": 2.0,
        "entry_price": 100000.0,
        "mid_price": 111700.0,
        "notional_usd": 223400.0,
    }
    decision = uma.ManagementDecision(
        action="no_change",
        close_fraction=0.0,
        new_notional_usd=223400.0,
        entry_price=111730.0,
        stop_loss_price=110720.0,
        planned_max_loss_usd=0.0,
        leverage=18,
        margin_basis_usd=0.0,
    )
    plan = uma.PositionManagementPlan(execute_now=False, action_decision=decision, scenario=None)

    session = agent._build_management_session(plan, position_after, "position_management")

    assert session is not None
    assert session.initial_entry_price == pytest.approx(111730.0)
    assert session.initial_stop_price == pytest.approx(110720.0)
    assert session.initial_risk_price_distance == pytest.approx(1010.0)
    assert session.tp1_price == pytest.approx(112740.0)
    assert session.tp2_price == pytest.approx(113750.0)


def test_build_management_session_refresh_no_change_protects_long_tp1_at_position_entry(uma):
    agent = make_agent(uma, reader=FakeReader())
    agent._audit_event = lambda *args, **kwargs: None
    position_after = {
        "symbol": "BTC",
        "side": "long",
        "size": 2.0,
        "entry_price": 110000.0,
        "mid_price": 100100.0,
        "notional_usd": 200200.0,
    }
    decision = uma.ManagementDecision(
        action="no_change",
        close_fraction=0.0,
        new_notional_usd=200200.0,
        entry_price=100000.0,
        stop_loss_price=99000.0,
        planned_max_loss_usd=0.0,
        leverage=18,
        margin_basis_usd=0.0,
    )
    plan = uma.PositionManagementPlan(execute_now=False, action_decision=decision, scenario=None)

    session = agent._build_management_session(plan, position_after, "position_management")

    assert session is not None
    assert session.initial_entry_price == pytest.approx(100000.0)
    assert getattr(session, "risk_entry_source", "") == "strategy_entry_price"
    assert session.initial_risk_price_distance == pytest.approx(1000.0)
    assert session.tp1_price == pytest.approx(110000.0)
    assert session.tp2_price == pytest.approx(102000.0)
    tp1_leg = next(leg for leg in session.take_profit_legs if leg.name == "stage_tp1")
    assert tp1_leg.when_all[0].level == pytest.approx(110000.0)


def test_build_management_session_refresh_no_change_protects_short_tp1_at_position_entry(uma):
    agent = make_agent(uma, reader=FakeReader())
    agent._audit_event = lambda *args, **kwargs: None
    position_after = {
        "symbol": "BTC",
        "side": "short",
        "size": -2.0,
        "entry_price": 90000.0,
        "mid_price": 99900.0,
        "notional_usd": 199800.0,
    }
    decision = uma.ManagementDecision(
        action="no_change",
        close_fraction=0.0,
        new_notional_usd=199800.0,
        entry_price=100000.0,
        stop_loss_price=101000.0,
        planned_max_loss_usd=0.0,
        leverage=18,
        margin_basis_usd=0.0,
    )
    plan = uma.PositionManagementPlan(execute_now=False, action_decision=decision, scenario=None)

    session = agent._build_management_session(plan, position_after, "position_management")

    assert session is not None
    assert session.initial_entry_price == pytest.approx(100000.0)
    assert getattr(session, "risk_entry_source", "") == "strategy_entry_price"
    assert session.initial_risk_price_distance == pytest.approx(1000.0)
    assert session.tp1_price == pytest.approx(90000.0)
    assert session.tp2_price == pytest.approx(98000.0)
    tp1_leg = next(leg for leg in session.take_profit_legs if leg.name == "stage_tp1")
    assert tp1_leg.when_all[0].level == pytest.approx(90000.0)


def test_build_management_session_add_uses_plan_entry_but_new_long_uses_fill_entry(uma):
    agent = make_agent(uma, reader=FakeReader())
    agent._audit_event = lambda *args, **kwargs: None
    position_after = {
        "symbol": "BTC",
        "side": "long",
        "size": 2.0,
        "entry_price": 100000.0,
        "mid_price": 111700.0,
        "notional_usd": 223400.0,
    }

    add_decision = uma.ManagementDecision(
        action="add_to_long",
        close_fraction=0.0,
        new_notional_usd=223400.0,
        entry_price=111730.0,
        stop_loss_price=110720.0,
        planned_max_loss_usd=0.0,
        leverage=18,
        margin_basis_usd=0.0,
    )
    add_plan = uma.PositionManagementPlan(execute_now=False, action_decision=add_decision, scenario=None)
    add_session = agent._build_management_session(add_plan, position_after, "position_management")

    assert add_session is not None
    assert add_session.initial_entry_price == pytest.approx(111730.0)
    assert add_session.initial_risk_price_distance == pytest.approx(1010.0)

    long_decision = uma.ManagementDecision(
        action="long",
        close_fraction=0.0,
        new_notional_usd=223400.0,
        entry_price=111730.0,
        stop_loss_price=99750.0,
        planned_max_loss_usd=0.0,
        leverage=18,
        margin_basis_usd=0.0,
    )
    long_plan = uma.PositionManagementPlan(execute_now=False, action_decision=long_decision, scenario=None)
    long_session = agent._build_management_session(long_plan, position_after, "position_management")

    assert long_session is not None
    assert long_session.initial_entry_price == pytest.approx(100000.0)
    assert long_session.initial_risk_price_distance == pytest.approx(250.0)


def test_exposure_risk_session_new_long_ignores_post_fill_template_entry(uma):
    agent = make_agent(uma, reader=FakeReader())
    agent._audit_event = lambda *args, **kwargs: None
    position_after = {
        "symbol": "BTC",
        "side": "long",
        "size": 2.0,
        "entry_price": 100000.0,
        "mid_price": 100050.0,
        "notional_usd": 200100.0,
    }
    decision = uma.ManagementDecision(
        action="long",
        close_fraction=0.0,
        new_notional_usd=200100.0,
        entry_price=111730.0,
        stop_loss_price=99750.0,
        planned_max_loss_usd=0.0,
        leverage=18,
        margin_basis_usd=0.0,
    )
    management_plan = uma.PositionManagementPlan(execute_now=True, action_decision=decision, scenario=None)
    post_fill_template = uma.PositionManagementPlan(
        execute_now=False,
        action_decision=uma.ManagementDecision(
            action="no_change",
            close_fraction=0.0,
            new_notional_usd=0.0,
            entry_price=111730.0,
            stop_loss_price=110720.0,
            planned_max_loss_usd=0.0,
            leverage=18,
            margin_basis_usd=0.0,
        ),
        scenario=None,
    )

    agent._set_risk_session_after_management_decision(
        decision,
        management_plan,
        post_fill_template,
        position_after,
        "position_management",
    )

    assert agent.risk_session is not None
    assert agent.risk_session.initial_entry_price == pytest.approx(100000.0)
    assert agent.risk_session.stop_loss_price == pytest.approx(99750.0)
    assert agent.risk_session.initial_risk_price_distance == pytest.approx(250.0)


def test_exposure_risk_session_add_uses_strategy_entry_without_fill_price(uma):
    agent = make_agent(uma, reader=FakeReader())
    agent._audit_event = lambda *args, **kwargs: None
    position_after = {
        "symbol": "BTC",
        "side": "long",
        "size": 2.0,
        "entry_price": 100000.0,
        "mid_price": 111700.0,
        "notional_usd": 223400.0,
    }
    decision = uma.ManagementDecision(
        action="add_to_long",
        close_fraction=0.0,
        new_notional_usd=223400.0,
        entry_price=111730.0,
        stop_loss_price=110720.0,
        planned_max_loss_usd=0.0,
        leverage=18,
        margin_basis_usd=0.0,
    )
    management_plan = uma.PositionManagementPlan(execute_now=True, action_decision=decision, scenario=None)
    post_fill_template = uma.PositionManagementPlan(
        execute_now=False,
        action_decision=uma.ManagementDecision(
            action="no_change",
            close_fraction=0.0,
            new_notional_usd=0.0,
            entry_price=222222.0,
            stop_loss_price=220000.0,
            planned_max_loss_usd=0.0,
            leverage=18,
            margin_basis_usd=0.0,
        ),
        scenario=None,
    )

    agent._set_risk_session_after_management_decision(
        decision,
        management_plan,
        post_fill_template,
        position_after,
        "position_management",
    )

    assert agent.risk_session is not None
    assert agent.risk_session.initial_entry_price == pytest.approx(111730.0)
    assert agent.risk_session.stop_loss_price == pytest.approx(110720.0)
    assert agent.risk_session.initial_risk_price_distance == pytest.approx(1010.0)


def test_exposure_risk_session_add_uses_fill_price_when_available(uma):
    agent = make_agent(uma, reader=FakeReader())
    seen = {}
    agent._audit_event = lambda kind, payload: seen.setdefault(kind, payload)
    position_after = {
        "symbol": "BTC",
        "side": "short",
        "size": -10.0,
        "entry_price": 108.713,
        "mid_price": 108.45,
        "notional_usd": 1084.5,
    }
    decision = uma.ManagementDecision(
        action="add_to_short",
        close_fraction=0.0,
        new_notional_usd=1084.5,
        entry_price=108.36,
        stop_loss_price=109.35,
        planned_max_loss_usd=0.0,
        leverage=16,
        margin_basis_usd=0.0,
    )
    management_plan = uma.PositionManagementPlan(execute_now=True, action_decision=decision, scenario=None)

    agent._set_risk_session_after_management_decision(
        decision,
        management_plan,
        None,
        position_after,
        "position_management",
        add_fill_entry_price=108.45,
    )

    assert agent.risk_session is not None
    assert agent.risk_session.initial_entry_price == pytest.approx(108.45)
    assert agent.risk_session.stop_loss_price == pytest.approx(109.35)
    assert agent.risk_session.initial_risk_price_distance == pytest.approx(0.90)
    assert agent.risk_session.tp1_price == pytest.approx(107.55)
    assert agent.risk_session.tp2_price == pytest.approx(106.65)
    assert seen["risk_session_created"]["risk_entry_source"] == "add_fill_avg_price"
    assert seen["risk_session_created"]["risk_entry_price"] == pytest.approx(108.45)


def test_exposure_risk_session_reverse_uses_filled_position_entry(uma):
    agent = make_agent(uma, reader=FakeReader())
    agent._audit_event = lambda *args, **kwargs: None
    position_after = {
        "symbol": "BTC",
        "side": "short",
        "size": -2.0,
        "entry_price": 99000.0,
        "mid_price": 98950.0,
        "notional_usd": 197900.0,
    }
    decision = uma.ManagementDecision(
        action="reverse_to_short",
        close_fraction=0.0,
        new_notional_usd=197900.0,
        entry_price=100000.0,
        stop_loss_price=101000.0,
        planned_max_loss_usd=0.0,
        leverage=18,
        margin_basis_usd=0.0,
    )
    management_plan = uma.PositionManagementPlan(execute_now=True, action_decision=decision, scenario=None)
    post_fill_template = uma.PositionManagementPlan(
        execute_now=False,
        action_decision=uma.ManagementDecision(
            action="no_change",
            close_fraction=0.0,
            new_notional_usd=0.0,
            entry_price=100000.0,
            stop_loss_price=101000.0,
            planned_max_loss_usd=0.0,
            leverage=18,
            margin_basis_usd=0.0,
        ),
        scenario=None,
    )

    agent._set_risk_session_after_management_decision(
        decision,
        management_plan,
        post_fill_template,
        position_after,
        "position_management",
    )

    assert agent.risk_session is not None
    assert agent.risk_session.initial_entry_price == pytest.approx(99000.0)
    assert agent.risk_session.stop_loss_price == pytest.approx(101000.0)
    assert agent.risk_session.initial_risk_price_distance == pytest.approx(2000.0)
