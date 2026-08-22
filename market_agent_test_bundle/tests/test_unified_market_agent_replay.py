from __future__ import annotations

from collections import deque


def make_entry_decision(uma, *, notional=1000.0, leverage=10, stop_loss_price=99.0):
    return uma.StrategyDecision(
        action="long",
        suggested_notional_usd=notional,
        entry_price=101.0,
        stop_loss_price=stop_loss_price,
        planned_margin_used_usd=notional / max(leverage, 1),
        planned_max_loss_usd=20.0,
        requested_leverage=leverage,
    )


def make_management_plan(uma):
    return uma.PositionManagementPlan(
        execute_now=False,
        action_decision=uma.build_empty_management_decision(),
        scenario=None,
    )


class ReplayReader:
    def __init__(self, price_tape):
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
            "remaining_capital_usd": 300.0,
            "available_margin_usd": 300.0,
            "withdrawable_usd": 300.0,
            "margin_used": 0.0,
            "account_equity_usd": 500.0,
            "perp_account_equity_usd": 500.0,
            "cross_margin_basis_usd": 500.0,
            "isolated_margin_basis_usd": 500.0,
            "perp_account_equity_usd": 500.0,
            "cross_margin_basis_usd": 500.0,
            "isolated_margin_basis_usd": 500.0,
        }
        self.resting_orders = []
        self.executor = None

    def _apply_resting_orders(self):
        side = self.position["side"]
        size = abs(float(self.position["size"] or 0.0))
        price = float(self.position["mid_price"] or 0.0)
        if side not in {"long", "short"} or size <= 0.0:
            self.resting_orders = []
            return
        remaining_orders = []
        for order in list(self.resting_orders):
            tpsl = str(order.get("tpsl", "") or "")
            trigger_price = float(order.get("trigger_price", 0.0) or 0.0)
            close_size = min(size, abs(float(order.get("close_size", 0.0) or 0.0)))
            triggered = False
            if tpsl == "tp":
                triggered = price >= trigger_price if side == "long" else price <= trigger_price
            elif tpsl == "sl":
                triggered = price <= trigger_price if side == "long" else price >= trigger_price
            if not triggered or close_size <= 0.0:
                remaining_orders.append(order)
                continue
            size = max(0.0, size - close_size)
            if self.executor is not None:
                self.executor.actions.append(("fill_tpsl", order.get("plan_name"), order.get("leg_name"), tpsl, price))
            if size <= 1e-9:
                break
        self.resting_orders = remaining_orders if size > 1e-9 else []
        if size <= 1e-9:
            self.position.update(
                {
                    "side": "flat",
                    "size": 0.0,
                    "notional_usd": 0.0,
                    "margin_used": 0.0,
            "account_equity_usd": 500.0,
            "perp_account_equity_usd": 500.0,
            "cross_margin_basis_usd": 500.0,
            "isolated_margin_basis_usd": 500.0,
            "perp_account_equity_usd": 500.0,
            "cross_margin_basis_usd": 500.0,
            "isolated_margin_basis_usd": 500.0,
                    "remaining_capital_usd": 300.0,
                    "available_margin_usd": 300.0,
                    "withdrawable_usd": 300.0,
                }
            )
        else:
            signed_size = size if side == "long" else -size
            self.position.update(
                {
                    "side": side,
                    "size": signed_size,
                    "notional_usd": abs(signed_size) * price,
                }
            )

    def advance(self):
        self.idx = min(self.idx + 1, len(self.price_tape) - 1)
        self.position["mid_price"] = self.price_tape[self.idx][1]
        self._apply_resting_orders()
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
            "account_equity_usd": 500.0,
            "perp_account_equity_usd": 500.0,
            "cross_margin_basis_usd": 500.0,
            "isolated_margin_basis_usd": 500.0,
            "total_margin_used_usd": sum(p["margin_used"] for p in positions),
            "available_margin_usd": self.position["remaining_capital_usd"],
            "withdrawable_usd": self.position["remaining_capital_usd"],
            "remaining_capital_usd": self.position["remaining_capital_usd"],
            "remaining_capital_source": "withdrawable",
            "positions": positions,
            "positions_count": len(positions),
            "total_notional_usd": sum(abs(p["notional_usd"]) for p in positions),
        }

    def get_candles_snapshot(self, symbol, interval, start_ms, end_ms):
        if str(interval or "").lower() != "1m":
            return []
        rows = []
        for ts, price in self.price_tape:
            open_ms = max(0, int(float(ts) * 1000) - 60000)
            if start_ms <= open_ms <= end_ms:
                rows.append({"t": open_ms, "h": float(price), "l": float(price), "c": float(price)})
        return rows

    def get_market_spec(self, symbol):
        return {"max_leverage": 40, "only_isolated": False}

    @staticmethod
    def format_all_positions(snapshot):
        return "formatted-all-positions"

    @staticmethod
    def format_symbol_position(snapshot):
        return "formatted-symbol-position"


class ReplayExecutor:
    def __init__(self, reader, symbol):
        self.reader = reader
        self.symbol = symbol
        self.enabled = True
        self.actions = []
        self.reader.executor = self

    @staticmethod
    def _result_has_exchange_error(payload):
        return False

    def execute(self, decision, plan_name=None):
        price = self.reader.get_mid_price(self.symbol)
        margin_used = decision.suggested_notional_usd / max(int(decision.leverage or 1), 1)
        self.reader.position.update(
            {
                "side": "long",
                "size": 1.0,
                "entry_price": price,
                "mid_price": price,
                "notional_usd": abs(price),
                "leverage": float(decision.leverage),
                "margin_used": margin_used,
                "remaining_capital_usd": max(0.0, 300.0 - margin_used),
                "available_margin_usd": max(0.0, 300.0 - margin_used),
                "withdrawable_usd": max(0.0, 300.0 - margin_used),
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
            "account_equity_usd": 500.0,
            "perp_account_equity_usd": 500.0,
            "cross_margin_basis_usd": 500.0,
            "isolated_margin_basis_usd": 500.0,
            "perp_account_equity_usd": 500.0,
            "cross_margin_basis_usd": 500.0,
            "isolated_margin_basis_usd": 500.0,
                "remaining_capital_usd": 300.0,
                "available_margin_usd": 300.0,
                "withdrawable_usd": 300.0,
            }
        )
        self.actions.append(("reduce", plan_name, side, reason, price))
        return {"accepted": True, "plan_name": plan_name, "side": side, "reason": reason}

    def close_position(self, side, reason, plan_name=None, **_kwargs):
        return self.reduce_position(side, 0.0, reason, plan_name)

    def place_reduce_only_tpsl_order(self, *, side, close_size, trigger_price, tpsl, plan_name=None, leg_name=None):
        cloid = f"cloid-{len(self.reader.resting_orders) + 1}"
        self.reader.resting_orders.append(
            {
                "cloid": cloid,
                "side": side,
                "close_size": float(close_size or 0.0),
                "trigger_price": float(trigger_price or 0.0),
                "tpsl": str(tpsl or ""),
                "plan_name": plan_name,
                "leg_name": leg_name,
            }
        )
        self.actions.append(("place_tpsl", plan_name, leg_name, tpsl, float(close_size or 0.0), float(trigger_price or 0.0)))
        return {"accepted": True, "cloid": cloid, "close_size": float(close_size or 0.0)}

    def cancel_reduce_only_tpsl_orders(self, order_refs, plan_name=None):
        cloids = {str((item or {}).get("cloid", "") or "") for item in list(order_refs or [])}
        self.reader.resting_orders = [order for order in self.reader.resting_orders if str(order.get("cloid", "") or "") not in cloids]
        self.actions.append(("cancel_tpsl", plan_name, sorted(cloids)))
        return {"accepted": True}

    def execute_management(self, decision, plan_name=None, **_kwargs):
        price = self.reader.get_mid_price(self.symbol)
        margin_used = decision.new_notional_usd / max(int(decision.leverage or 1), 1)
        side = decision.action if decision.action in {"long", "short"} else "long"
        signed_size = 1.0 if side == "long" else -1.0
        self.reader.position.update(
            {
                "side": side,
                "size": signed_size,
                "entry_price": price,
                "mid_price": price,
                "notional_usd": abs(price),
                "leverage": float(decision.leverage),
                "margin_used": margin_used,
                "remaining_capital_usd": max(0.0, 300.0 - margin_used),
                "available_margin_usd": max(0.0, 300.0 - margin_used),
                "withdrawable_usd": max(0.0, 300.0 - margin_used),
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


def make_playbook(uma):
    entry = make_entry_decision(uma, notional=1000.0, leverage=10)
    scenario = uma.EntryScenario(
        observe_when_all=[uma.Condition(type="price_ge", level=100.0, note="observe")],
        arm_when_all=[uma.Condition(type="price_ge", level=100.0, note="arm")],
        cancel_when_any=[],
        timeout_seconds_after_arm=30,
    )
    return uma.GenericPlaybook(
        display_answer="wait then buy",
        current_bias="bullish",
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
    agent.symbol = "BTC"
    agent.engine = None
    agent.executor = executor
    agent.user_query_template = "replay"
    agent.max_planned_loss_usd = 100.0
    agent.local_risk_tolerance_usd = 1.0
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
    agent.requery_on_playbook_end = False
    agent.events = None
    snapshot = reader.get_position_snapshot("BTC")
    playbook = agent._materialize_live_position_management_from_entry_plan(playbook, snapshot, reader.get_all_positions())
    agent.current_playbook = playbook
    agent.current_mode = "raw_context_only"
    agent.current_playbook_reason = "replay"
    agent.position_management_session = agent._build_position_management_session(playbook.position_management, snapshot, "position_management")
    agent.risk_session = None
    agent.next_active_query_due_at = 10**12
    agent.last_playbook_query_at = None
    agent.last_playbook_tick_at = 0.0
    agent.last_risk_tick_at = 0.0
    agent.last_position_management_tick_at = 0.0
    return agent


def test_replay_open_then_take_profit_legs_scale_position(uma, monkeypatch):
    price_tape = [
        (100.0, 100.0),
        (101.0, 101.2),
        (102.0, 105.0),
    ]
    reader = ReplayReader(price_tape)
    executor = ReplayExecutor(reader, "BTC")
    playbook = make_playbook(uma)
    agent = make_agent(uma, reader, executor, playbook)

    for ts, px in price_tape:
        monkeypatch.setattr(uma.time, "time", lambda ts=ts: ts)
        if agent.risk_session is not None:
            agent.step_risk_session(reader.get_position_snapshot("BTC"), ts)
        if agent.position_management_session is not None:
            agent.step_position_management_session(reader.get_position_snapshot("BTC"), ts)
        reader.advance()

    assert executor.actions[0][0] == "execute"
    assert executor.actions[0][1].endswith("::armed")
    assert executor.actions[0][2] == "long"
    assert any(action[0] == "place_tpsl" and action[3] == "tp" for action in executor.actions)
    assert any(action[0] == "fill_tpsl" and action[3] == "tp" for action in executor.actions)
    assert round(reader.get_position_snapshot("BTC")["size"], 8) == 0.3
    assert agent.risk_session is not None


def test_replay_open_then_stop_loss_leg_closes_position(uma, monkeypatch):
    price_tape = [
        (100.0, 100.0),
        (101.0, 101.2),
        (102.0, 98.5),
    ]
    reader = ReplayReader(price_tape)
    executor = ReplayExecutor(reader, "BTC")
    playbook = make_playbook(uma)
    agent = make_agent(uma, reader, executor, playbook)

    for ts, px in price_tape:
        monkeypatch.setattr(uma.time, "time", lambda ts=ts: ts)
        if agent.risk_session is not None:
            agent.step_risk_session(reader.get_position_snapshot("BTC"), ts)
        if agent.position_management_session is not None:
            agent.step_position_management_session(reader.get_position_snapshot("BTC"), ts)
        reader.advance()

    assert executor.actions[0][0] == "execute"
    assert executor.actions[0][1].endswith("::armed")
    assert executor.actions[0][2] == "long"
    assert any(action[0] == "reduce" and action[3] == "soft_stop" for action in executor.actions)
    assert reader.get_position_snapshot("BTC")["size"] == 0.0
    assert agent.risk_session is None
