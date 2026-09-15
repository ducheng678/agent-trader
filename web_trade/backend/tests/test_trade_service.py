from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from market_agent.exchange import HyperliquidExecutor
from web_trade.backend.web_trade.ledger import SyntheticPositionLedger
from web_trade.backend.web_trade.service import WebTradeService


class FakeReader:
    def __init__(self):
        self.candle_requests = []
        self.user_fills_requests = []
        self.user_funding_requests = []
        self.historical_order_requests = []
        self.snapshot = {
            "symbol": "BTC",
            "side": "long",
            "size": 0.1,
            "entry_price": 100000.0,
            "mid_price": 101000.0,
            "notional_usd": 10100.0,
            "leverage": 10.0,
            "max_leverage": 40,
            "only_isolated": True,
            "liquidation_price": 92000.0,
            "margin_used": 1010.0,
            "unrealized_pnl": 140.0,
            "available_margin_usd": 500.0,
        }
        self.account_address = "0x1234567890abcdef1234567890abcdef12345678"
        self.network = "testnet"

    def get_position_snapshot(self, symbol):
        assert symbol == "BTC"
        return dict(self.snapshot)

    def get_selected_symbol_position_context(self, symbol):
        return {"position_snapshot": self.get_position_snapshot(symbol), "all_positions": {"positions": [self.get_position_snapshot(symbol)]}}

    def get_all_positions(self):
        return {
            "account_address": self.account_address,
            "network": self.network,
            "available_margin_usd": 500.0,
            "withdrawable_usd": 500.0,
            "positions": [self.get_position_snapshot("BTC")],
        }

    def get_market_catalog(self):
        return {
            "BTC": {"symbol": "BTC", "display_name": "BTC-USDC", "max_leverage": 40, "only_isolated": True},
            "ETH": {"symbol": "ETH", "display_name": "ETH-USDC", "max_leverage": 25, "only_isolated": True},
        }

    def get_ws_symbol(self, symbol):
        return symbol

    def get_frontend_open_orders(self, symbol=None):
        return []

    def get_user_fills_by_time(self, address, start_time_ms, end_time_ms=None, aggregate_by_time=False):
        self.user_fills_requests.append((address, start_time_ms, end_time_ms, aggregate_by_time))
        return [{"coin": "BTC", "side": "B", "px": "100000", "sz": "0.01", "time": end_time_ms, "oid": 123}]

    def get_user_funding_history(self, address, start_time_ms, end_time_ms=None):
        self.user_funding_requests.append((address, start_time_ms, end_time_ms))
        return [{"coin": "BTC", "usdc": "-1.25", "fundingRate": "0.0001", "time": end_time_ms}]

    def get_historical_orders(self, address):
        self.historical_order_requests.append(address)
        return [
            {"order": {"coin": "BTC", "side": "B", "limitPx": "100000", "sz": "0.01", "oid": 456}, "status": "filled", "statusTimestamp": 1699999999000},
            {"order": {"coin": "ETH", "side": "A", "limitPx": "2000", "sz": "0.5", "oid": 457}, "status": "canceled", "statusTimestamp": 1690000000000},
        ]

    def get_candles_snapshot(self, symbol, interval, start_ms, end_ms):
        self.candle_requests.append((symbol, interval, start_ms, end_ms))
        return [
            {"t": 1700000000000, "o": "100", "h": "110", "l": "95", "c": "105", "v": "12.5"},
            {"T": 1700000060000, "open": "105", "high": "115", "low": "104", "close": "108", "volume": "8"},
        ]

    def get_mids(self, ttl_seconds=2.0, dex=""):
        return {"BTC": "101000", "ETH": "2050"}

    def get_l2_book_snapshot(self, symbol):
        assert symbol == "BTC"
        return {
            "coin": "BTC",
            "time": 1700000123000,
            "levels": [
                [{"px": "100.0", "sz": "2.0", "n": 3}, {"px": "99.5", "sz": "1.0", "n": 1}],
                [{"px": "101.0", "sz": "1.5", "n": 2}],
            ],
        }

    def get_sz_decimals(self, symbol):
        return 4


class MultiDexReader(FakeReader):
    def list_perp_dex_names(self):
        return ["xyz", "mkts"]

    def get_market_catalog(self):
        return {
            "BTC": {"symbol": "BTC", "display_name": "BTC-USDC", "max_leverage": 50, "only_isolated": True},
            "ETH": {"symbol": "ETH", "display_name": "ETH-USDC", "max_leverage": 25, "only_isolated": True},
            "xyz:AAPL": {
                "symbol": "xyz:AAPL",
                "execution_symbol": "xyz:AAPL",
                "display_name": "xyz:AAPL-USDC",
                "dex": "xyz",
                "market_name": "AAPL",
                "max_leverage": 20,
                "only_isolated": True,
            },
            "xyz:BTC": {
                "symbol": "xyz:BTC",
                "execution_symbol": "xyz:BTC",
                "display_name": "xyz:BTC-USDC",
                "dex": "xyz",
                "market_name": "BTC",
                "max_leverage": 40,
                "only_isolated": True,
            },
            "mkts:BTC": {
                "symbol": "mkts:BTC",
                "execution_symbol": "mkts:BTC",
                "display_name": "mkts:BTC-USDC",
                "dex": "mkts",
                "market_name": "BTC",
                "max_leverage": 10,
                "only_isolated": True,
            },
        }

    def get_mids(self, ttl_seconds=2.0, dex=""):
        if dex == "xyz":
            return {"AAPL": "200", "BTC": "101000"}
        if dex == "mkts":
            return {"BTC": "99000"}
        return {"BTC": "100000", "ETH": "2050"}


class FakeExecutor:
    instances = []

    def __init__(self, reader, symbol):
        self.reader = reader
        self.symbol = symbol
        self.calls = []
        self.enabled = True
        self._exchange = SimpleNamespace(update_isolated_margin=self._update_isolated_margin)
        FakeExecutor.instances.append(self)

    def close_position(self, side, reason, plan_name=None, position_before=None):
        self.calls.append(("close_position", side, reason))
        self.reader.snapshot = {
            **self.reader.snapshot,
            "side": "flat",
            "size": 0.0,
            "notional_usd": 0.0,
            "leverage": 0.0,
            "margin_used": 0.0,
            "unrealized_pnl": 0.0,
        }
        return {"accepted": True, "actions": [{"market_close": {"status": "ok"}}]}

    def apply_requested_leverage(self, requested_leverage):
        self.calls.append(("apply_requested_leverage", requested_leverage))
        return {"applied_leverage": requested_leverage}

    def execute_position_target(self, **kwargs):
        self.calls.append(("execute_position_target", kwargs))
        self.reader.snapshot = {
            "symbol": "BTC",
            "side": kwargs["target_side"],
            "size": 0.05,
            "entry_price": 101200.0,
            "mid_price": 101000.0,
            "notional_usd": kwargs["target_notional_usd"],
            "leverage": float(kwargs["requested_leverage"]),
            "max_leverage": 40,
            "only_isolated": True,
            "liquidation_price": 81000.0,
            "margin_used": 1010.0,
            "unrealized_pnl": -20.0,
            "available_margin_usd": 500.0,
        }
        return {"accepted": True, "target_notional_usd": kwargs["target_notional_usd"]}

    def reduce_position(self, side, close_size, reason, plan_name=None, position_before=None):
        self.calls.append(("reduce_position", side, close_size, reason))
        return {"accepted": True, "close_size": close_size}

    def place_entry_limit_order(self, **kwargs):
        self.calls.append(("place_entry_limit_order", kwargs))
        return {"accepted": True, "entry_order_pending": True, "limit_price": kwargs["limit_price"]}

    def place_market_order(self, **kwargs):
        self.calls.append(("place_market_order", kwargs))
        return {"accepted": True, "filled": True, "notional_usd": kwargs["notional_usd"]}

    def place_reduce_only_limit_order(self, **kwargs):
        self.calls.append(("place_reduce_only_limit_order", kwargs))
        return {"accepted": True, "reduce_only_limit_order_pending": True, "limit_price": kwargs["limit_price"]}

    def place_reduce_only_tpsl_order(self, **kwargs):
        self.calls.append(("place_reduce_only_tpsl_order", kwargs))
        return {"accepted": True, "reduce_only_tpsl_order_pending": True, "trigger_price": kwargs["trigger_price"], "tpsl": kwargs["tpsl"]}

    def _ensure_exchange(self):
        return None

    def _update_isolated_margin(self, amount, symbol):
        self.calls.append(("update_isolated_margin", amount, symbol))
        self.reader.snapshot = {
            **self.reader.snapshot,
            "margin_used": self.reader.snapshot["margin_used"] + amount,
            "available_margin_usd": self.reader.snapshot["available_margin_usd"] - amount,
        }
        return {"status": "ok"}


class CloseRejectExecutor(FakeExecutor):
    def close_position(self, side, reason, plan_name=None, position_before=None):
        self.calls.append(("close_position", side, reason))
        return {
            "actions": [
                {
                    "market_close": {
                        "status": "ok",
                        "response": {"data": {"statuses": [{"error": "minTradeNtlRejected"}]}},
                    }
                }
            ]
        }

    def execute_position_target(self, **kwargs):
        raise AssertionError("reopen should not run after a rejected close")


class MarketRejectExecutor(FakeExecutor):
    def place_market_order(self, **kwargs):
        self.calls.append(("place_market_order", kwargs))
        return {"accepted": False, "message": "Exchange rejected market_open: insufficient margin."}


class StaleCloseExecutor(FakeExecutor):
    def close_position(self, side, reason, plan_name=None, position_before=None):
        self.calls.append(("close_position", side, reason))
        return {"accepted": True, "actions": [{"market_close": {"status": "ok"}}]}


class StaleSnapshotExecutor(StaleCloseExecutor):
    def execute_position_target(self, **kwargs):
        self.calls.append(("execute_position_target", kwargs))
        return {
            "accepted": True,
            "target_side": kwargs["target_side"],
            "target_notional_usd": kwargs["target_notional_usd"],
            "requested_leverage": kwargs["requested_leverage"],
            "mid": kwargs["execution_mid_price"],
            "open_qty": 0.08,
        }


def test_rebalance_leverage_scales_current_notional_by_leverage_ratio_not_lifecycle_roi_basis(tmp_path):
    reader = FakeReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)
    ledger.overlay_position(reader.get_position_snapshot("BTC"))
    reader.snapshot = {**reader.snapshot, "notional_usd": 12000.0, "leverage": 10.0, "margin_used": 1200.0}

    result = service.rebalance_leverage("BTC", 5)

    executor = FakeExecutor.instances[-1]
    target_call = [call for call in executor.calls if call[0] == "execute_position_target"][0]
    assert target_call[1]["target_notional_usd"] == pytest.approx(6000.0)
    assert result["rebalance_margin_usd"] == pytest.approx(1200.0)
    assert result["position"]["display_entry_price"] == pytest.approx(100000.0)
    assert result["position"]["carried_realized_pnl_usd"] == pytest.approx(140.0)
    assert result["position"]["synthetic_pnl_usd"] == pytest.approx(140.0)


def test_rebalance_leverage_returns_fresh_liquidation_price_after_reopen(tmp_path):
    reader = FakeReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)

    result = service.rebalance_leverage("BTC", 5)

    assert result["accepted"] is True
    assert result["position_before"]["liquidation_price"] == pytest.approx(92000.0)
    assert result["position"]["liquidation_price"] == pytest.approx(81000.0)


def test_rebalance_leverage_keeps_current_notional_when_target_matches_current_leverage(tmp_path):
    reader = FakeReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)
    ledger.overlay_position({**reader.get_position_snapshot("BTC"), "margin_used": 2.348571, "notional_usd": 23.48571})
    reader.snapshot = {
        **reader.snapshot,
        "size": 0.05,
        "mid_price": 333.795,
        "notional_usd": 16.68975,
        "leverage": 10.0,
        "margin_used": 1.677398,
        "unrealized_pnl": -0.0055,
    }

    result = service.rebalance_leverage("BTC", 10)

    executor = FakeExecutor.instances[-1]
    target_call = [call for call in executor.calls if call[0] == "execute_position_target"][0]
    assert target_call[1]["target_notional_usd"] == pytest.approx(16.68975)
    assert result["rebalance_margin_usd"] == pytest.approx(1.668975)


def test_rebalance_leverage_reopens_from_synthetic_flat_snapshot_when_rest_snapshot_is_stale(tmp_path):
    StaleCloseExecutor.instances.clear()
    reader = FakeReader()
    reader.snapshot = {
        **reader.snapshot,
        "size": 0.07,
        "mid_price": 333.13,
        "notional_usd": 23.3191,
        "leverage": 10.0,
        "margin_used": 2.33191,
    }
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=StaleCloseExecutor)

    result = service.rebalance_leverage("BTC", 11)

    executor = StaleCloseExecutor.instances[-1]
    target_call = [call for call in executor.calls if call[0] == "execute_position_target"][0]
    assert target_call[1]["target_notional_usd"] == pytest.approx(25.65101)
    assert target_call[1]["position_before"]["side"] == "flat"
    assert target_call[1]["position_before"]["size"] == pytest.approx(0.0)
    assert target_call[1]["position_before"]["notional_usd"] == pytest.approx(0.0)
    assert result["accepted"] is True


def test_rebalance_leverage_returns_expected_position_when_rest_snapshot_is_stale_after_reopen(tmp_path):
    StaleSnapshotExecutor.instances.clear()
    reader = FakeReader()
    reader.snapshot = {
        **reader.snapshot,
        "size": 0.07,
        "mid_price": 333.13,
        "notional_usd": 23.3191,
        "leverage": 10.0,
        "margin_used": 2.33191,
        "unrealized_pnl": 0.07,
    }
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=StaleSnapshotExecutor)

    result = service.rebalance_leverage("BTC", 11)

    assert result["accepted"] is True
    assert result["position"]["leverage"] == pytest.approx(11.0)
    assert result["position"]["notional_usd"] == pytest.approx(26.6504)
    assert result["position"]["size"] == pytest.approx(0.08)
    assert result["position"]["synthetic_pnl_usd"] == pytest.approx(0.07)
    assert result["position"]["display_entry_price"] == pytest.approx(100000.0)
    assert reader.snapshot["leverage"] == pytest.approx(10.0)


def test_rebalance_leverage_rejects_too_small_target_notional_before_close(tmp_path, monkeypatch):
    FakeExecutor.instances.clear()
    monkeypatch.setenv("WEB_TRADE_MIN_TRADE_NOTIONAL_USD", "10")
    reader = FakeReader()
    reader.snapshot = {
        **reader.snapshot,
        "size": 0.01,
        "mid_price": 500.0,
        "notional_usd": 5.0,
        "leverage": 5.0,
        "margin_used": 1.0,
    }
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)

    result = service.rebalance_leverage("BTC", 5)

    assert result["accepted"] is False
    assert result["stage"] == "target_below_min_trade_notional"
    assert result["target_notional_usd"] == pytest.approx(5.0)
    assert result["min_trade_notional_usd"] == pytest.approx(10.0)
    assert FakeExecutor.instances == []
    assert reader.snapshot["side"] == "long"


def test_rebalance_leverage_rejects_target_that_rounds_below_min_notional(tmp_path, monkeypatch):
    FakeExecutor.instances.clear()
    monkeypatch.setenv("WEB_TRADE_MIN_TRADE_NOTIONAL_USD", "10")
    reader = FakeReader()
    reader.get_sz_decimals = lambda symbol: 2
    reader.snapshot = {
        **reader.snapshot,
        "size": 0.07,
        "mid_price": 333.13,
        "notional_usd": 23.3191,
        "leverage": 10.0,
        "margin_used": 2.332397,
    }
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)

    result = service.rebalance_leverage("BTC", 5)

    assert result["accepted"] is False
    assert result["stage"] == "target_below_min_trade_notional"
    assert result["target_notional_usd"] == pytest.approx(11.65955)
    assert result["effective_target_notional_usd"] == pytest.approx(9.9939)
    assert FakeExecutor.instances == []


def test_rebalance_leverage_reports_close_stage_when_close_order_is_rejected(tmp_path):
    CloseRejectExecutor.instances.clear()
    reader = FakeReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=CloseRejectExecutor)

    result = service.rebalance_leverage("BTC", 5)

    assert result["accepted"] is False
    assert result["stage"] == "close_failed"
    assert result["close_result"]["actions"][0]["market_close"]["response"]["data"]["statuses"][0]["error"] == "minTradeNtlRejected"


def test_execute_position_target_marks_min_trade_rejection_unaccepted():
    class Reader:
        def get_position_snapshot(self, symbol):
            return {"symbol": symbol, "side": "flat", "size": 0.0, "notional_usd": 0.0}

        def get_sz_decimals(self, symbol):
            return 4

        def get_mid_price(self, symbol):
            return 100.0

        def get_market_spec(self, symbol):
            return {"max_leverage": 20, "only_isolated": True}

    attempts = []

    def market_open(symbol, is_buy, qty, price, slippage):
        attempts.append(qty)
        return {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {"error": "minTradeNtlRejected"},
                    ]
                }
            },
        }

    executor = HyperliquidExecutor(Reader(), "BTC")
    executor.enabled = True
    executor._exchange = SimpleNamespace(market_open=market_open, update_leverage=lambda *args, **kwargs: {"status": "ok"})

    result = executor.execute_position_target(
        target_side="long",
        target_notional_usd=5.0,
        requested_leverage=5,
        reason="test",
        plan_name="web_trade",
        execution_mid_price=100.0,
    )

    assert result["accepted"] is False
    assert result["message"] == "Exchange rejected market_open: below minimum trade notional."
    assert attempts == [0.05]


def test_place_market_order_stops_after_margin_rejection():
    class Reader:
        def get_position_snapshot(self, symbol):
            return {"symbol": symbol, "side": "long", "size": 0.00048, "notional_usd": 28.8}

        def get_sz_decimals(self, symbol):
            return 5

        def get_market_spec(self, symbol):
            return {"max_leverage": 40, "only_isolated": True}

    attempts = []

    def market_open(symbol, is_buy, qty, price, slippage):
        attempts.append(qty)
        if len(attempts) == 1:
            return {
                "status": "ok",
                "response": {
                    "data": {
                        "statuses": [
                            {"error": "perpMarginRejected"},
                        ]
                    }
                },
            }
        return {"status": "ok", "response": {"data": {"statuses": [{"filled": {"totalSz": str(qty)}}]}}}

    executor = HyperliquidExecutor(Reader(), "BTC")
    executor.enabled = True
    executor._exchange = SimpleNamespace(market_open=market_open, update_leverage=lambda *args, **kwargs: {"status": "ok"})

    result = executor.place_market_order(
        side="long",
        notional_usd=13.8,
        requested_leverage=40,
        reason="test",
        plan_name="web_trade",
        execution_mid_price=60000.0,
    )

    assert result["accepted"] is False
    assert result["message"] == "Exchange rejected market_open: insufficient margin."
    assert attempts == [0.00023]


def test_place_market_order_reports_partial_ioc_fill():
    class Reader:
        def get_position_snapshot(self, symbol):
            return {"symbol": symbol, "side": "long", "size": 0.00027, "notional_usd": 16.0}

        def get_sz_decimals(self, symbol):
            return 5

        def get_market_spec(self, symbol):
            return {"max_leverage": 40, "only_isolated": True}

    def market_open(symbol, is_buy, qty, price, slippage):
        return {
            "status": "ok",
            "response": {
                "data": {
                    "statuses": [
                        {"filled": {"totalSz": "0.00020", "avgPx": "59358.0", "oid": 482287291315}},
                    ]
                }
            },
        }

    executor = HyperliquidExecutor(Reader(), "BTC")
    executor.enabled = True
    executor._exchange = SimpleNamespace(market_open=market_open, update_leverage=lambda *args, **kwargs: {"status": "ok"})

    result = executor.place_market_order(
        side="long",
        notional_usd=20.4,
        requested_leverage=40,
        reason="test",
        plan_name="web_trade",
        execution_mid_price=60000.0,
    )

    assert result["accepted"] is True
    assert result["partial_fill"] is True
    assert result["requested_qty"] == pytest.approx(0.00034)
    assert result["filled_qty"] == pytest.approx(0.0002)
    assert result["message"] == "Market order partially filled: filled 0.0002 / requested 0.00034 BTC."


def test_update_isolated_margin_clamps_to_max_and_updates_net_roi_basis(tmp_path):
    reader = FakeReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)
    ledger.overlay_position(reader.get_position_snapshot("BTC"))

    add_result = service.update_isolated_margin("BTC", "add", 999.0, safety_buffer_usd=5.0)
    remove_result = service.update_isolated_margin("BTC", "remove", 999.0, safety_buffer_usd=5.0)

    assert add_result["applied_amount_usd"] == pytest.approx(495.0)
    assert remove_result["applied_amount_usd"] < 0
    view = ledger.overlay_position(reader.get_position_snapshot("BTC"))
    assert view["lifecycle_roi_basis_usd"] == pytest.approx(1010.0 + 495.0 + remove_result["applied_amount_usd"])


def test_account_positions_include_margin_limits(tmp_path):
    reader = FakeReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)

    account = service.account()

    limits = account["positions"][0]["margin_limits"]
    assert limits["enabled"] is True
    assert limits["max_add_margin_usd"] > 0
    assert limits["max_remove_margin_usd"] > 0


def test_account_history_uses_90_day_window_and_filters_order_history(tmp_path, monkeypatch):
    reader = FakeReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)
    monkeypatch.setattr("web_trade.backend.web_trade.service.time.time", lambda: 1700000000.0)

    history = service.account_history(window_days=90)

    expected_start = 1700000000000 - 90 * 24 * 60 * 60 * 1000
    assert reader.user_fills_requests == [(reader.account_address, expected_start, 1700000000000, False)]
    assert reader.user_funding_requests == [(reader.account_address, expected_start, 1700000000000)]
    assert reader.historical_order_requests == [reader.account_address]
    assert history["window_days"] == 90
    assert history["trade_history"][0]["oid"] == 123
    assert history["funding_history"][0]["coin"] == "BTC"
    assert [item["order"]["oid"] for item in history["order_history"]] == [456]


def test_margin_limits_uses_fresh_position_snapshot(tmp_path):
    reader = FakeReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)

    limits = service.margin_limits("BTC", safety_buffer_usd=5.0)

    assert limits["max_add_margin_usd"] == pytest.approx(495.0)


def test_place_market_order_uses_margin_and_leverage_for_direct_order_notional(tmp_path):
    FakeExecutor.instances.clear()
    reader = FakeReader()
    reader.snapshot = {
        **reader.snapshot,
        "side": "long",
        "notional_usd": 18.0,
        "margin_used": 0.45,
        "leverage": 40.0,
    }
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)

    result = service.place_order(
        symbol="BTC",
        order_type="market",
        side="long",
        margin_usd=250.0,
        leverage=20,
    )

    executor = FakeExecutor.instances[-1]
    market_call = [call for call in executor.calls if call[0] == "place_market_order"][0]
    target_calls = [call for call in executor.calls if call[0] == "execute_position_target"]
    assert result["accepted"] is True
    assert target_calls == []
    assert market_call[1]["side"] == "long"
    assert market_call[1]["notional_usd"] == pytest.approx(5000.0)
    assert market_call[1]["requested_leverage"] == 20
    assert result["margin_usd"] == pytest.approx(250.0)
    assert result["target_notional_usd"] == pytest.approx(5000.0)


def test_reverse_order_uses_target_position_and_starts_new_lifecycle(tmp_path):
    FakeExecutor.instances.clear()
    reader = FakeReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)
    ledger.overlay_position(reader.get_position_snapshot("BTC"))
    ledger.record_hidden_rebalance(
        symbol="BTC",
        realized_pnl_usd=140.0,
        target_leverage=10,
        target_notional_usd=10100.0,
    )

    result = service.place_order(
        symbol="BTC",
        order_type="market",
        side="short",
        margin_usd=1000.0,
        leverage=10,
        position_action="reverse",
    )

    executor = FakeExecutor.instances[-1]
    target_calls = [call for call in executor.calls if call[0] == "execute_position_target"]
    market_calls = [call for call in executor.calls if call[0] == "place_market_order"]
    assert len(target_calls) == 1
    assert market_calls == []
    assert target_calls[0][1]["target_side"] == "short"
    assert target_calls[0][1]["target_notional_usd"] == pytest.approx(10000.0)
    assert result["accepted"] is True
    assert result["position_action"] == "reverse"
    assert result["position"]["side"] == "short"
    assert result["position"]["display_entry_price"] == pytest.approx(101200.0)
    assert result["position"]["carried_realized_pnl_usd"] == pytest.approx(0.0)
    assert result["position"]["synthetic_pnl_usd"] == pytest.approx(-20.0)


def test_place_market_order_propagates_rejection_message(tmp_path):
    MarketRejectExecutor.instances.clear()
    reader = FakeReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=MarketRejectExecutor)

    result = service.place_order(
        symbol="BTC",
        order_type="market",
        side="long",
        margin_usd=0.53,
        leverage=40,
    )

    assert result["accepted"] is False
    assert result["message"] == "Exchange rejected market_open: insufficient margin."


def test_place_market_order_rejects_margin_above_fresh_available_margin(tmp_path):
    FakeExecutor.instances.clear()
    reader = FakeReader()
    reader.snapshot = {
        **reader.snapshot,
        "available_margin_usd": 0.308527,
        "withdrawable_usd": 0.308527,
        "remaining_capital_usd": 0.308527,
    }
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)

    result = service.place_order(
        symbol="BTC",
        order_type="market",
        side="long",
        margin_usd=0.53,
        leverage=40,
    )

    assert result["accepted"] is False
    assert result["stage"] == "insufficient_available_margin"
    assert result["available_margin_usd"] == pytest.approx(0.308527)
    assert result["requested_margin_usd"] == pytest.approx(0.53)
    assert FakeExecutor.instances == []


def test_place_market_order_rejects_below_min_trade_notional_before_exchange(tmp_path, monkeypatch):
    FakeExecutor.instances.clear()
    monkeypatch.setenv("WEB_TRADE_MIN_TRADE_NOTIONAL_USD", "10")
    reader = FakeReader()
    reader.get_sz_decimals = lambda symbol: 5
    reader.snapshot = {
        **reader.snapshot,
        "mid_price": 58310.0,
        "available_margin_usd": 0.236689,
    }
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)

    result = service.place_order(
        symbol="BTC",
        order_type="market",
        side="short",
        margin_usd=0.236,
        leverage=40,
    )

    assert result["accepted"] is False
    assert result["stage"] == "below_min_trade_notional"
    assert result["target_notional_usd"] == pytest.approx(9.44)
    assert result["min_trade_notional_usd"] == pytest.approx(10.0)
    assert FakeExecutor.instances == []


def test_reduce_only_limit_order_uses_limit_close_executor(tmp_path):
    FakeExecutor.instances.clear()
    reader = FakeReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)

    result = service.place_order(
        symbol="BTC",
        order_type="limit",
        side="short",
        margin_usd=505.0,
        leverage=10,
        limit_price=102000.0,
        reduce_only=True,
    )

    executor = FakeExecutor.instances[-1]
    limit_call = [call for call in executor.calls if call[0] == "place_reduce_only_limit_order"][0]
    assert result["accepted"] is True
    assert limit_call[1]["side"] == "long"
    assert limit_call[1]["close_size"] == pytest.approx(0.05)
    assert limit_call[1]["limit_price"] == pytest.approx(102000.0)


def test_reduce_only_market_close_all_uses_full_current_size_even_when_requested_notional_is_stale(tmp_path):
    FakeExecutor.instances.clear()
    reader = FakeReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)

    result = service.place_order(
        symbol="BTC",
        order_type="market",
        side="short",
        margin_usd=1000.0,
        leverage=10,
        reduce_only=True,
        close_all=True,
    )

    executor = FakeExecutor.instances[-1]
    reduce_call = [call for call in executor.calls if call[0] == "reduce_position"][0]
    assert result["accepted"] is True
    assert reduce_call[1] == "long"
    assert reduce_call[2] == pytest.approx(0.1)


def test_set_position_tpsl_places_reduce_only_trigger_orders(tmp_path):
    FakeExecutor.instances.clear()
    reader = FakeReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)

    result = service.set_position_tpsl(
        symbol="BTC",
        take_profit_price=105000.0,
        stop_loss_price=99000.0,
    )

    executor = FakeExecutor.instances[-1]
    trigger_calls = [call for call in executor.calls if call[0] == "place_reduce_only_tpsl_order"]
    assert result["accepted"] is True
    assert len(trigger_calls) == 2
    assert trigger_calls[0][1]["side"] == "long"
    assert trigger_calls[0][1]["close_size"] == pytest.approx(0.1)
    assert trigger_calls[0][1]["trigger_price"] == pytest.approx(105000.0)
    assert trigger_calls[0][1]["tpsl"] == "tp"
    assert trigger_calls[1][1]["trigger_price"] == pytest.approx(99000.0)
    assert trigger_calls[1][1]["tpsl"] == "sl"


def test_market_snapshot_normalizes_candles_for_frontend_chart(tmp_path):
    reader = FakeReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)

    snapshot = service.market_snapshot("BTC")

    assert snapshot["candles"] == [
        {"time": 1700000000, "open": 100.0, "high": 110.0, "low": 95.0, "close": 105.0, "volume": 12.5},
        {"time": 1700000060, "open": 105.0, "high": 115.0, "low": 104.0, "close": 108.0, "volume": 8.0},
    ]


def test_markets_include_ws_symbol_for_frontend_subscriptions(tmp_path):
    reader = FakeReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)

    markets = service.markets()

    assert markets[0]["symbol"] == "BTC"
    assert markets[0]["ws_symbol"] == "BTC"


def test_markets_prefer_xyz_dex_with_main_btc_eth_exceptions(tmp_path):
    reader = MultiDexReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)

    markets = service.markets()

    assert [market["symbol"] for market in markets] == ["BTC", "ETH", "xyz:AAPL"]
    assert [market["display_name"] for market in markets] == ["BTC-USDC", "ETH-USDC", "AAPL-USDC"]
    assert markets[0]["mid_price"] == pytest.approx(100000.0)
    assert markets[2]["execution_symbol"] == "xyz:AAPL"
    assert markets[2]["mid_price"] == pytest.approx(200.0)


def test_market_snapshot_uses_requested_interval(tmp_path):
    reader = FakeReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)

    snapshot = service.market_snapshot("BTC", interval="5m", window_seconds=7200)

    assert snapshot["interval"] == "5m"
    assert reader.candle_requests[-1][1] == "5m"
    assert reader.candle_requests[-1][3] - reader.candle_requests[-1][2] == 7200 * 1000


def test_market_bars_maps_tradingview_resolution_to_hyperliquid_interval(tmp_path):
    reader = FakeReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)

    result = service.market_bars("BTC", resolution="60", from_s=1_700_000_000, to_s=1_700_007_200)

    assert result["symbol"] == "BTC"
    assert result["resolution"] == "60"
    assert result["interval"] == "1h"
    assert reader.candle_requests[-1] == ("BTC", "1h", 1_700_000_000_000, 1_700_007_200_000)
    assert result["bars"][0]["time"] == 1_700_000_000


def test_market_bars_uses_count_back_when_from_is_missing(tmp_path):
    reader = FakeReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)

    service.market_bars("BTC", resolution="5", to_s=1_700_007_200, count_back=20)

    assert reader.candle_requests[-1] == ("BTC", "5m", 1_700_001_200_000, 1_700_007_200_000)


def test_favorite_markets_are_file_backed_deduped_and_validated(tmp_path):
    reader = FakeReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    favorites_path = tmp_path / "favorite_markets.json"
    service = WebTradeService(
        reader=reader,
        ledger=ledger,
        executor_factory=FakeExecutor,
        favorites_path=favorites_path,
    )

    favorites = service.update_favorite_markets(["ETH", "BTC", "ETH", "DOGE"])
    reloaded = WebTradeService(
        reader=reader,
        ledger=ledger,
        executor_factory=FakeExecutor,
        favorites_path=favorites_path,
    ).favorite_markets()

    assert [item["symbol"] for item in favorites] == ["ETH", "BTC"]
    assert [item["symbol"] for item in reloaded] == ["ETH", "BTC"]
    assert json.loads(favorites_path.read_text()) == {"symbols": ["ETH", "BTC"]}


def test_favorites_keep_main_btc_exception_and_resolve_other_unprefixed_symbols_to_xyz(tmp_path):
    reader = MultiDexReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    favorites_path = tmp_path / "favorite_markets.json"
    service = WebTradeService(
        reader=reader,
        ledger=ledger,
        executor_factory=FakeExecutor,
        favorites_path=favorites_path,
    )

    favorites = service.update_favorite_markets(["BTC", "AAPL", "mkts:BTC"])

    assert [item["symbol"] for item in favorites] == ["BTC", "xyz:AAPL"]
    assert [item["display_name"] for item in favorites] == ["BTC-USDC", "AAPL-USDC"]
    assert json.loads(favorites_path.read_text()) == {"symbols": ["BTC", "xyz:AAPL"]}


def test_market_book_normalizes_l2_snapshot_for_frontend_depth(tmp_path):
    reader = FakeReader()
    ledger = SyntheticPositionLedger(tmp_path / "ledger.json")
    service = WebTradeService(reader=reader, ledger=ledger, executor_factory=FakeExecutor)

    book = service.market_book("BTC")

    assert book == {
        "symbol": "BTC",
        "time": 1700000123000,
        "mid_price": 100.5,
        "bids": [
            {"price": 100.0, "size": 2.0, "total": 2.0, "orders": 3},
            {"price": 99.5, "size": 1.0, "total": 3.0, "orders": 1},
        ],
        "asks": [{"price": 101.0, "size": 1.5, "total": 1.5, "orders": 2}],
    }
