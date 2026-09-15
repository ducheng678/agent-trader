#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import types
from collections import deque
from pathlib import Path
from typing import List, Tuple


def load_target_module(target_path: Path):
    if "openai" not in sys.modules:
        fake_openai = types.ModuleType("openai")
        fake_openai.OpenAI = object
        sys.modules["openai"] = fake_openai

    spec = importlib.util.spec_from_file_location("unified_market_agent_replay_target", target_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to import target module from {target_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class ReplayReader:
    def __init__(self, price_tape: List[Tuple[float, float]]):
        self.price_tape = list(price_tape)
        self.idx = 0
        self.account_address = "0xabc"
        self.network = "mainnet"
        self.position = {
            "known": True,
            "account_address": self.account_address,
            "network": self.network,
            "symbol": "BTC",
            "side": "flat",
            "size": 0.0,
            "entry_price": 0.0,
            "mid_price": self.price_tape[0][1],
            "notional_usd": 0.0,
            "leverage": 0.0,
            "max_leverage": 40,
            "remaining_capital_usd": 1000.0,
            "available_margin_usd": 1000.0,
            "withdrawable_usd": 1000.0,
            "margin_used": 0.0,
        }

    def advance(self):
        self.idx = min(self.idx + 1, len(self.price_tape) - 1)
        self.position["mid_price"] = self.price_tape[self.idx][1]
        self.position["notional_usd"] = abs(self.position["size"]) * self.position["mid_price"]

    def get_mid_price(self, symbol):
        return self.position["mid_price"]

    def get_position_snapshot(self, symbol, all_positions=None, current_price=None):
        snapshot = dict(self.position)
        if current_price is not None:
            snapshot["mid_price"] = current_price
        return snapshot

    def get_all_positions(self):
        if self.position["size"] == 0.0:
            positions = []
        else:
            positions = [
                {
                    "symbol": "BTC",
                    "side": self.position["side"],
                    "size": self.position["size"],
                    "entry_price": self.position["entry_price"],
                    "mid_price": self.position["mid_price"],
                    "notional_usd": self.position["notional_usd"],
                    "unrealized_pnl": 0.0,
                    "return_on_equity": 0.0,
                    "leverage": self.position["leverage"],
                    "max_leverage": self.position["max_leverage"],
                    "liquidation_price": 0.0,
                    "margin_used": self.position["margin_used"],
                    "only_isolated": False,
                }
            ]
        return {
            "known": True,
            "account_address": self.account_address,
            "network": self.network,
            "margin_summary": {},
            "cross_margin_summary": {},
            "account_equity_usd": 1000.0,
            "total_margin_used_usd": sum(p["margin_used"] for p in positions),
            "available_margin_usd": self.position["remaining_capital_usd"],
            "withdrawable_usd": self.position["remaining_capital_usd"],
            "remaining_capital_usd": self.position["remaining_capital_usd"],
            "remaining_capital_source": "withdrawable",
            "positions": positions,
            "positions_count": len(positions),
            "total_notional_usd": sum(abs(p["notional_usd"]) for p in positions),
        }

    def get_market_spec(self, symbol):
        return {"max_leverage": 40, "only_isolated": False}

    @staticmethod
    def format_all_positions(snapshot):
        return json.dumps(snapshot, ensure_ascii=False)

    @staticmethod
    def format_symbol_position(snapshot):
        return json.dumps(snapshot, ensure_ascii=False)


class ReplayExecutor:
    def __init__(self, reader, symbol):
        self.reader = reader
        self.symbol = symbol
        self.enabled = False
        self.allow_pyramiding = False
        self.actions = []

    @staticmethod
    def _result_has_exchange_error(payload):
        return False

    def execute_management(self, decision, plan_name=None, trigger_confidence_raw=None, **_kwargs):
        action = str(getattr(decision, "action", "") or "")
        price = self.reader.get_mid_price(self.symbol)
        if action in {"close", "trim", "no_change"}:
            result = {
                "mode": "dry_run",
                "symbol": self.symbol,
                "plan_name": plan_name,
                "decision": decision.to_dict(),
                "actions": [],
            }
            if action == "close":
                self.close_position(self.reader.position.get("side", "flat"), action, plan_name)
            return result
        target_side = "long" if action in {"long", "add_to_long", "reverse_to_long"} else "short"
        leverage = max(int(getattr(decision, "leverage", 0) or 0), 1)
        target_notional = max(0.0, float(getattr(decision, "new_notional_usd", 0.0) or 0.0))
        margin_used = target_notional / leverage if leverage > 0 else 0.0
        size = target_notional / max(price, 1e-12)
        if target_side == "short":
            size *= -1.0
        self.reader.position.update(
            {
                "side": target_side,
                "size": size,
                "entry_price": price,
                "mid_price": price,
                "notional_usd": abs(target_notional),
                "leverage": float(leverage),
                "margin_used": margin_used,
                "remaining_capital_usd": max(0.0, 1000.0 - margin_used),
                "available_margin_usd": max(0.0, 1000.0 - margin_used),
                "withdrawable_usd": max(0.0, 1000.0 - margin_used),
            }
        )
        self.actions.append(("execute_management", plan_name, action, price, target_notional))
        return {
            "mode": "dry_run",
            "symbol": self.symbol,
            "plan_name": plan_name,
            "decision": decision.to_dict(),
            "actions": [],
        }

    def execute(self, decision, plan_name=None):
        price = self.reader.get_mid_price(self.symbol)
        margin_used = decision.suggested_notional_usd / max(int(decision.requested_leverage or 1), 1)
        side = "long" if decision.action == "long" else "short"
        size = 1.0 if side == "long" else -1.0
        self.reader.position.update(
            {
                "side": side,
                "size": size,
                "entry_price": price,
                "mid_price": price,
                "notional_usd": abs(price),
                "leverage": float(decision.requested_leverage or 1),
                "margin_used": margin_used,
                "remaining_capital_usd": max(0.0, 1000.0 - margin_used),
                "available_margin_usd": max(0.0, 1000.0 - margin_used),
                "withdrawable_usd": max(0.0, 1000.0 - margin_used),
            }
        )
        self.actions.append(("execute", plan_name, decision.action, price))
        return {
            "mode": "dry_run",
            "symbol": self.symbol,
            "plan_name": plan_name,
            "decision": decision.to_dict(),
            "actions": [],
        }

    def reduce_position(self, side, close_size, reason, plan_name=None):
        price = self.reader.get_mid_price(self.symbol)
        self.reader.position.update(
            {
                "side": "flat",
                "size": 0.0,
                "notional_usd": 0.0,
                "margin_used": 0.0,
                "remaining_capital_usd": 1000.0,
                "available_margin_usd": 1000.0,
                "withdrawable_usd": 1000.0,
            }
        )
        self.actions.append(("close", plan_name, side, reason, price))
        return {"plan_name": plan_name, "side": side, "reason": reason}

    def close_position(self, side, reason, plan_name=None):
        return self.reduce_position(side, 0.0, reason, plan_name)

    def resolve_exit_levels(self, decision, ref_price):
        return {
            "reference_price": ref_price,
            "stop_loss_price": decision.stop_loss_price,
        }


def make_management_plan(uma):
    return uma.PositionManagementPlan(
        execute_now=False,
        action_decision=uma.build_empty_management_decision(),
        scenario=None,
    )


def make_demo_playbook(uma):
    entry = uma.StrategyDecision(
        action="long",
        suggested_notional_usd=1000.0,
        entry_price=101.0,
        stop_loss_price=99.0,
        planned_margin_used_usd=100.0,
        planned_max_loss_usd=20.0,
        requested_leverage=10,
    )
    scenario = uma.EntryScenario(
        observe_when_all=uma.ObserveWhenAll(low=100.0, high=100.0),
        execute_when_all=uma.ExecuteWhenAll(
            condition=uma.Condition(type="price_ge", level=101.0, note="buy"),
            timeout_seconds=30,
        ),
    )
    return uma.GenericPlaybook(
        display_answer="wait then buy",
        current_bias="bullish",
        selected_symbol="BTC",
        selection_reason="BTC selected for replay harness",
        entry_plan=uma.EntryPlan(
            execute_now=False,
            action_decision=entry,
            scenario=scenario,
        ),
        position_management=make_management_plan(uma),
        post_fill_risk_template=make_management_plan(uma),
    )


def make_agent(uma, reader, executor, playbook):
    agent = object.__new__(uma.UnifiedMarketAgent)
    agent.reader = reader
    agent.default_symbol = "BTC"
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
    agent.engine = None
    agent.executor = executor
    agent.user_query_template = "replay"
    agent.max_planned_loss_usd = 100.0
    agent.local_size_from_stop = True
    agent.local_risk_tolerance_usd = 1.0
    agent.auto_clamp_margin_drift_ratio = 0.02
    agent.loop_sleep_seconds = 0.0
    agent.playbook_poll_seconds = 0.0
    agent.price_history_seconds = 1800
    agent.risk_poll_seconds = 0.0
    agent.position_size_change_tol = 1e-8
    agent.enable_monitor = True
    agent.enable_active_query = False
    agent.enable_active_auto_requery = False
    agent.active_query_interval_seconds = 60.0
    agent.active_management_query_interval_seconds = 120.0
    agent.enable_passive_event_query = False
    agent.fast_replan_delay_seconds = 0.0
    agent.query_on_start = False
    agent.requery_on_playbook_end = False
    agent.enable_audit_log = False
    agent.enable_status_log = False
    agent.events = None
    agent.current_mode = "raw_context_only"
    agent.current_playbook_reason = "replay"
    agent.risk_session = None
    agent.position_management_session = None
    agent.next_active_query_due_at = 10**12
    agent.last_playbook_query_at = None
    agent.last_playbook_tick_at = 0.0
    agent.last_risk_tick_at = 0.0
    agent.last_position_management_tick_at = 0.0
    snapshot = reader.get_position_snapshot("BTC")
    playbook = agent._materialize_live_position_management_from_entry_plan(playbook, snapshot, reader.get_all_positions())
    agent.current_playbook = playbook
    agent.position_management_session = agent._build_position_management_session(playbook.position_management, snapshot, "position_management")
    return agent


def load_price_tape(path: Path | None) -> List[Tuple[float, float]]:
    if path is None:
        return [
            (100.0, 100.0),
            (101.0, 101.2),
            (102.0, 105.0),
        ]
    raw = json.loads(path.read_text(encoding="utf-8"))
    tape = []
    for row in raw:
        if not isinstance(row, (list, tuple)) or len(row) != 2:
            raise ValueError("Each price tape row must be [timestamp, price]")
        tape.append((float(row[0]), float(row[1])))
    if not tape:
        raise ValueError("Price tape is empty")
    return tape


def main():
    parser = argparse.ArgumentParser(description="Deterministic replay harness for unified_market_agent.py")
    parser.add_argument("--target", required=True, help="Path to unified_market_agent.py")
    parser.add_argument("--prices", help="Optional JSON file with [[timestamp, price], ...]")
    args = parser.parse_args()

    target = Path(args.target).expanduser().resolve()
    if not target.exists():
        raise FileNotFoundError(f"Target module not found: {target}")

    uma = load_target_module(target)
    price_tape = load_price_tape(Path(args.prices).expanduser().resolve() if args.prices else None)
    reader = ReplayReader(price_tape)
    executor = ReplayExecutor(reader, "BTC")
    playbook = make_demo_playbook(uma)
    agent = make_agent(uma, reader, executor, playbook)

    print("[replay_start]")
    print(json.dumps({"target": str(target), "price_tape": price_tape}, ensure_ascii=False, indent=2))

    for ts, px in price_tape:
        uma.time.time = lambda ts=ts: ts
        print(f"[tick] ts={ts} price={px}")
        if agent.risk_session is None and agent.position_management_session is not None:
            status = agent.step_position_management_session(reader.get_position_snapshot("BTC"), ts)
            if status is not None:
                print(f"[scenario_status] {status}")
        if agent.risk_session is not None:
            status = agent.step_risk_session(reader.get_position_snapshot("BTC"), ts)
            if status is not None:
                print(f"[risk_status] {status}")
        reader.advance()

    print("[executor_actions]")
    print(json.dumps(executor.actions, ensure_ascii=False, indent=2))
    final_position = reader.get_position_snapshot("BTC")
    print("[final_position]")
    print(json.dumps(final_position, ensure_ascii=False, indent=2))

    if not executor.actions:
        raise SystemExit("Replay failed: no management actions were produced.")
    if executor.actions[0][0] not in {"execute", "execute_management"}:
        raise SystemExit("Replay failed: first action was not a management execution.")

    print("[replay_ok] scenario arm -> management execution path works")


if __name__ == "__main__":
    main()
