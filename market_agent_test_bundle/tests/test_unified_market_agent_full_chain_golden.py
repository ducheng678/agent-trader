from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).resolve().parent
if str(TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TEST_DIR))

from test_unified_market_agent_state_machine import (  # noqa: E402
    FakeEngine,
    make_agent,
    make_all_positions_for_snapshot,
    make_entry_decision,
    make_playbook,
    make_position_snapshot,
)


GOLDEN_PATH = Path(__file__).resolve().parents[1] / "golden" / "full_chain_replay_baseline.json"


def _round_value(value):
    if isinstance(value, float):
        return round(value, 8)
    if isinstance(value, dict):
        return {str(key): _round_value(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_round_value(item) for item in value]
    if isinstance(value, tuple):
        return [_round_value(item) for item in value]
    return value


class GoldenReader:
    def __init__(self, initial_position):
        self.position = dict(initial_position)
        self.account_address = str(self.position.get("account_address", "0xabc") or "0xabc")
        self.network = str(self.position.get("network", "mainnet") or "mainnet")
        self.open_orders = []
        self.candles = []

    def get_all_positions(self):
        return make_all_positions_for_snapshot(dict(self.position))

    def get_position_snapshot(self, symbol, all_positions=None, current_price=None):
        snapshot = dict(self.position)
        if current_price is not None:
            snapshot["mid_price"] = float(current_price)
            if snapshot.get("side") in {"long", "short"}:
                snapshot["notional_usd"] = abs(float(snapshot.get("size", 0.0) or 0.0)) * float(current_price)
        return snapshot

    def get_mid_price(self, symbol):
        return float(self.position.get("mid_price", 100000.0) or 100000.0)

    def get_market_spec(self, symbol):
        return {"max_leverage": 20, "only_isolated": False}

    def get_user_fee_rates(self):
        return {"known": True, "source": "test", "taker_fee_rate": 0.0}

    def get_frontend_open_orders(self, symbol=None):
        return [dict(item) for item in self.open_orders]

    def get_candles_snapshot(self, symbol, interval, start_ms, end_ms):
        return list(self.candles)

    @staticmethod
    def format_all_positions(snapshot):
        return "formatted-all-positions"

    @staticmethod
    def format_symbol_position(snapshot):
        return "formatted-symbol-position"


class GoldenExecutor:
    def __init__(self, reader, symbol="BTC"):
        self.reader = reader
        self.symbol = symbol
        self.enabled = True
        self.actions = []
        self._next_oid = 1000

    @staticmethod
    def _result_has_exchange_error(result):
        return False

    def _set_position(self, *, side, notional, leverage):
        price = self.reader.get_mid_price(self.symbol)
        notional = max(0.0, float(notional or 0.0))
        if side not in {"long", "short"} or notional <= 0.0:
            self.reader.position.update(
                {
                    "side": "flat",
                    "size": 0.0,
                    "notional_usd": 0.0,
                    "margin_used": 0.0,
                    "leverage": 0.0,
                }
            )
            return
        size = notional / max(price, 1e-12)
        if side == "short":
            size = -size
        margin_used = notional / max(int(leverage or 1), 1)
        self.reader.position.update(
            {
                "side": side,
                "size": size,
                "entry_price": price,
                "mid_price": price,
                "notional_usd": notional,
                "leverage": float(leverage or 0),
                "margin_used": margin_used,
                "remaining_capital_usd": max(0.0, 300.0 - margin_used),
                "available_margin_usd": max(0.0, 300.0 - margin_used),
                "withdrawable_usd": max(0.0, 300.0 - margin_used),
            }
        )

    def execute_management(self, decision, plan_name=None, trigger_confidence_raw=None, **_kwargs):
        before = self.reader.get_position_snapshot(self.symbol)
        action = str(decision.action or "")
        if action in {"long", "add_to_long", "reverse_to_long"}:
            self._set_position(side="long", notional=decision.new_notional_usd, leverage=decision.leverage)
        elif action in {"short", "add_to_short", "reverse_to_short"}:
            self._set_position(side="short", notional=decision.new_notional_usd, leverage=decision.leverage)
        elif action == "trim":
            current_side = str(before.get("side", "flat") or "flat")
            self._set_position(side=current_side, notional=decision.new_notional_usd, leverage=before.get("leverage", 0.0))
        elif action == "close":
            self._set_position(side="flat", notional=0.0, leverage=0)
        elif action == "no_change" and int(decision.leverage or 0) > 0 and before.get("side") in {"long", "short"}:
            self.reader.position["leverage"] = float(decision.leverage)
        self.actions.append(
            {
                "kind": "execute_management",
                "plan_name": plan_name,
                "action": action,
                "new_notional_usd": float(decision.new_notional_usd or 0.0),
            }
        )
        return {
            "mode": "test",
            "symbol": self.symbol,
            "plan_name": plan_name,
            "decision": decision.to_dict(),
            "position_before": before,
            "actions": [],
        }

    def close_position(self, side, reason, plan_name=None, **_kwargs):
        self._set_position(side="flat", notional=0.0, leverage=0)
        self.actions.append({"kind": "close_position", "plan_name": plan_name, "side": side, "reason": reason})
        return {"accepted": True, "side": side, "reason": reason, "plan_name": plan_name}

    def place_reduce_only_tpsl_order(self, *, side, close_size, trigger_price, tpsl, plan_name=None, leg_name=None):
        self._next_oid += 1
        ref = {
            "oid": self._next_oid,
            "cloid": f"golden-{self._next_oid}",
            "coin": self.symbol,
            "side": side,
            "reduceOnly": True,
            "isTrigger": True,
            "close_size": float(close_size or 0.0),
            "trigger_price": float(trigger_price or 0.0),
            "tpsl": str(tpsl or ""),
            "plan_name": plan_name,
            "leg_name": leg_name,
        }
        self.reader.open_orders.append(dict(ref))
        self.actions.append(
            {
                "kind": "place_tpsl",
                "plan_name": plan_name,
                "leg_name": leg_name,
                "tpsl": str(tpsl or ""),
                "close_size": float(close_size or 0.0),
                "trigger_price": float(trigger_price or 0.0),
            }
        )
        return {"accepted": True, "oid": ref["oid"], "cloid": ref["cloid"], "close_size": ref["close_size"]}

    def cancel_reduce_only_tpsl_orders(self, order_refs, plan_name=None):
        cloids = {str((item or {}).get("cloid", "") or "") for item in list(order_refs or [])}
        self.reader.open_orders = [item for item in self.reader.open_orders if str(item.get("cloid", "") or "") not in cloids]
        self.actions.append({"kind": "cancel_tpsl", "plan_name": plan_name, "cloids": sorted(cloids)})
        return {"accepted": True}


def _golden_cases():
    cases = []
    confidences = [None, 0.0, 0.10, 0.459, 0.46, 0.461, 0.50, 0.52, 0.605, 0.608, 0.67, 0.749, 0.75, 0.90]
    for action in ("long", "short"):
        for raw in confidences:
            cases.append({"name": f"flat_{action}_{raw}", "initial_side": "flat", "initial_notional": 0.0, "entry_action": action, "raw_confidence": raw})
            cases.append(
                {
                    "name": f"flat_non_immediate_{action}_{raw}",
                    "initial_side": "flat",
                    "initial_notional": 0.0,
                    "entry_action": action,
                    "raw_confidence": raw,
                    "entry_execute_now": False,
                    "with_entry_scenario": True,
                }
            )
    for side in ("long", "short"):
        for target_hint in (0.0, 0.5, 1.0, 500.0, 1000.0, 5000.0, 9999.0, 10000.0, 10001.0, 15000.0, 25000.0):
            for raw in (None, 0.0, 0.459, 0.46, 0.461, 0.52, 0.605, 0.749, 0.75, 0.90):
                cases.append({"name": f"same_{side}_{target_hint}_{raw}", "initial_side": side, "initial_notional": target_hint, "entry_action": side, "raw_confidence": raw})
    for initial_side, entry_action in (("long", "short"), ("short", "long")):
        for initial_notional in (0.5, 1.0, 500.0, 5000.0, 9999.0, 10000.0, 10001.0, 15000.0, 25000.0):
            for raw in (None, 0.0, 0.459, 0.46, 0.461, 0.52, 0.605, 0.608, 0.67, 0.749, 0.75, 0.90):
                cases.append(
                    {
                        "name": f"reverse_{initial_side}_to_{entry_action}_{initial_notional}_{raw}",
                        "initial_side": initial_side,
                        "initial_notional": initial_notional,
                        "entry_action": entry_action,
                        "raw_confidence": raw,
                    }
                )
                cases.append(
                    {
                        "name": f"reverse_non_immediate_{initial_side}_to_{entry_action}_{initial_notional}_{raw}",
                        "initial_side": initial_side,
                        "initial_notional": initial_notional,
                        "entry_action": entry_action,
                        "raw_confidence": raw,
                        "entry_execute_now": False,
                        "with_entry_scenario": True,
                    }
                )
    for action in ("long", "short"):
        cases.append({"name": f"{action}_tp1_hit", "initial_side": "flat", "initial_notional": 0.0, "entry_action": action, "raw_confidence": 0.75, "post_step": "tp1"})
        cases.append({"name": f"{action}_tp1_tp2_hit", "initial_side": "flat", "initial_notional": 0.0, "entry_action": action, "raw_confidence": 0.75, "post_step": "tp1_tp2"})
        cases.append({"name": f"{action}_initial_stop_hit", "initial_side": "flat", "initial_notional": 0.0, "entry_action": action, "raw_confidence": 0.75, "post_step": "sl"})
    assert len(cases) >= 500
    return cases


def _risk_snapshot(agent):
    rs = getattr(agent, "risk_session", None)
    if rs is None:
        return None
    return _round_value(
        {
            "plan_name": rs.plan_name,
            "side": rs.side,
            "expected_size": rs.expected_size,
            "initial_size_abs": rs.initial_size_abs,
            "initial_entry_price": rs.initial_entry_price,
            "initial_stop_price": rs.initial_stop_price,
            "initial_risk_price_distance": rs.initial_risk_price_distance,
            "tp1_price": rs.tp1_price,
            "tp2_price": rs.tp2_price,
            "stop_loss_price": rs.stop_loss_price,
            "tp1_hit": rs.tp1_hit,
            "tp2_hit": rs.tp2_hit,
            "take_profit_legs": [(leg.name, leg.close_fraction, leg.when_all[0].type, leg.when_all[0].level) for leg in rs.take_profit_legs],
            "stop_loss_legs": [(leg.name, leg.close_fraction, leg.when_all[0].type, leg.when_all[0].level) for leg in rs.stop_loss_legs],
            "resting_exit_orders": [
                {
                    "key": item.get("key"),
                    "name": item.get("name"),
                    "leg_type": item.get("leg_type"),
                    "tpsl": item.get("tpsl"),
                    "trigger_price": item.get("trigger_price"),
                    "close_size": item.get("close_size"),
                }
                for item in list(rs.resting_exit_orders or [])
            ],
        }
    )


def _position_snapshot(reader):
    pos = reader.get_position_snapshot("BTC")
    return _round_value(
        {
            "side": pos.get("side"),
            "size": pos.get("size"),
            "entry_price": pos.get("entry_price"),
            "mid_price": pos.get("mid_price"),
            "notional_usd": pos.get("notional_usd"),
            "leverage": pos.get("leverage"),
        }
    )


def _apply_exit_fill(agent, reader, leg_name, now):
    rs = agent.risk_session
    assert rs is not None
    ref = next(item for item in rs.resting_exit_orders if item.get("name") == leg_name)
    side = reader.position.get("side")
    old_size = abs(float(reader.position.get("size", 0.0) or 0.0))
    close_size = min(old_size, abs(float(ref.get("close_size", 0.0) or 0.0)))
    new_size = max(0.0, old_size - close_size)
    trigger_price = float(ref.get("trigger_price", reader.get_mid_price("BTC")) or reader.get_mid_price("BTC"))
    if new_size <= 1e-9:
        reader.position.update({"side": "flat", "size": 0.0, "mid_price": trigger_price, "notional_usd": 0.0, "margin_used": 0.0})
    else:
        reader.position.update(
            {
                "side": side,
                "size": new_size if side == "long" else -new_size,
                "mid_price": trigger_price,
                "notional_usd": new_size * trigger_price,
            }
        )
    fill = {"coin": "BTC", "oid": ref.get("oid"), "cloid": ref.get("cloid"), "sz": close_size, "px": trigger_price, "time": int(now * 1000)}
    return agent.step_risk_session(reader.get_position_snapshot("BTC"), now=now, fill_events=[fill])


def _apply_soft_stop(agent, reader, now):
    rs = agent.risk_session
    assert rs is not None
    stop_price = float(rs.initial_stop_price)
    close_price = stop_price - 1.0 if rs.side == "long" else stop_price + 1.0
    reader.position.update(
        {
            "mid_price": close_price,
            "notional_usd": abs(float(reader.position.get("size", 0.0) or 0.0)) * close_price,
        }
    )
    reader.candles = [
        {"t": int((now - 120.0) * 1000), "h": close_price, "l": close_price, "c": close_price},
        {"t": int((now - 60.0) * 1000), "h": close_price, "l": close_price, "c": close_price},
    ]
    return agent.step_risk_session(reader.get_position_snapshot("BTC"), now=now, fill_events=[])


def _run_full_chain_case(uma, monkeypatch, case):
    monkeypatch.setenv("TRIGGER_CONFIDENCE_RELEVANCE_THRESHOLD", "0.46")
    monkeypatch.setenv("TRIGGER_CONFIDENCE_FULL_SCALE", "0.75")
    monkeypatch.setenv("TRIGGER_CONFIDENCE_RELEVANCE_THRESHOLD_BTC_USDC", "0.46")
    monkeypatch.setenv("TRIGGER_CONFIDENCE_FULL_SCALE_BTC_USDC", "0.75")
    monkeypatch.setattr(uma.time, "time", lambda: 1000.0)
    initial = make_position_snapshot(side=case["initial_side"], notional=case["initial_notional"], entry_price=100000.0, mid_price=100000.0, leverage=5.0)
    reader = GoldenReader(initial)
    executor = GoldenExecutor(reader)
    stop_loss = 101000.0 if case["entry_action"] == "short" else 99000.0
    playbook = make_playbook(
        uma,
        entry_execute_now=bool(case.get("entry_execute_now", True)),
        entry_action_decision=make_entry_decision(
            uma,
            action=case["entry_action"],
            notional=1000.0,
            stop_loss_price=stop_loss,
            leverage=0,
            entry_price=100000.0,
        ),
        entry_scenario=(
            uma.EntryScenario(
                observe_when_all=[],
                execute_when_all={
                    "condition": {"type": "price_le" if case["entry_action"] == "short" else "price_ge", "level": 100000.0, "low": 0.0, "high": 0.0, "timer_seconds": 0, "tolerance_bps": 0.0, "min_ratio": 0.0},
                    "timeout_seconds": 300,
                },
            )
            if case.get("with_entry_scenario")
            else None
        ),
    )
    playbook.trigger_confidence_raw = case["raw_confidence"]
    playbook.trigger_confidence = uma.normalize_confidence_value(case["raw_confidence"], symbol="BTC")
    agent = make_agent(uma, playbook=playbook, reader=reader, engine=FakeEngine(playbook), executor=executor)
    audit_events = []
    agent._audit_event = lambda event, payload=None: audit_events.append({"event": event, "payload": _round_value(payload or {})})

    agent.query_new_playbook("manual_once", None)
    step_statuses = []
    if case.get("post_step") == "tp1" and agent.risk_session is not None:
        step_statuses.append(_apply_exit_fill(agent, reader, "stage_tp1", 1010.0))
    elif case.get("post_step") == "tp1_tp2" and agent.risk_session is not None:
        step_statuses.append(_apply_exit_fill(agent, reader, "stage_tp1", 1010.0))
        if agent.risk_session is not None:
            step_statuses.append(_apply_exit_fill(agent, reader, "stage_tp2", 1020.0))
    elif case.get("post_step") == "sl" and agent.risk_session is not None:
        step_statuses.append(_apply_soft_stop(agent, reader, 1010.0))

    decision = agent.current_playbook.position_management.action_decision if agent.current_playbook is not None else None
    return _round_value(
        {
            "case": case,
            "decision": decision.to_dict() if decision is not None else None,
            "position": _position_snapshot(reader),
            "risk_session": _risk_snapshot(agent),
            "position_management_session": getattr(agent, "position_management_session", None) is not None,
            "executor_actions": executor.actions,
            "step_statuses": step_statuses,
            "audit_events": [item["event"] for item in audit_events],
        }
    )


def test_full_chain_golden_replay_matches_baseline(uma, monkeypatch):
    results = [_run_full_chain_case(uma, monkeypatch, case) for case in _golden_cases()]
    if os.getenv("UPDATE_FULL_CHAIN_GOLDEN") == "1":
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    expected = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert results == expected
