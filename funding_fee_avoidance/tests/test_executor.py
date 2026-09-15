from __future__ import annotations

from decimal import Decimal
from datetime import datetime, timedelta, timezone

import pytest

from funding_fee_avoidance.config import StrategyConfig
from funding_fee_avoidance.executor import HyperliquidHedgeExecutor
from funding_fee_avoidance.state_machine import stable_cloid


PRIMARY = "0x1111111111111111111111111111111111111111"
HEDGE = "0x2222222222222222222222222222222222222222"
DEADLINE = datetime(2099, 1, 1, tzinfo=timezone.utc)


class Reader:
    account_address = PRIMARY
    base = "https://example.test"


class Wallet:
    address = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"


class FakeExchange:
    def __init__(self):
        self.calls = []
        self.expires_after = None

    def set_expires_after(self, value):
        self.expires_after = value

    def update_leverage(self, leverage, symbol, is_cross=True):
        self.calls.append(("leverage", leverage, symbol, is_cross))
        return {"status": "ok"}

    def _slippage_price(self, symbol, is_buy, slippage, price):
        self.calls.append(("price", symbol, is_buy, slippage, price))
        return 100.0

    def order(
        self,
        symbol,
        is_buy,
        size,
        limit_price,
        order_type,
        reduce_only=False,
        cloid=None,
    ):
        self.calls.append(
            (
                "order",
                symbol,
                is_buy,
                size,
                limit_price,
                order_type,
                reduce_only,
                cloid,
            )
        )
        return {
            "status": "ok",
            "response": {"data": {"statuses": [{"filled": {"totalSz": "2"}}]}},
        }


def config(kind="subaccount"):
    return StrategyConfig(
        symbols=("xyz:SKHX",),
        primary_account_address=PRIMARY,
        hedge_account_address=HEDGE,
        hedge_account_kind=kind,
        hip3_extra_fee_rate_known=True,
        execution_enabled=True,
    )


def test_subaccount_executor_routes_actions_with_vault_address_only_to_hedge():
    captured = {}
    fake = FakeExchange()

    def factory(wallet, base, **kwargs):
        captured.update({"wallet": wallet, "base": base, **kwargs})
        return fake

    executor = HyperliquidHedgeExecutor(
        config(),
        Reader(),
        exchange_factory=factory,
        wallet_factory=lambda secret: Wallet(),
        env={"FUNDING_HEDGE_SECRET_KEY": "secret"},
    )
    cloid = stable_cloid("cycle", "open:1")
    result = executor.open_hedge(
        "xyz:SKHX", Decimal("-2"), cloid, submit_deadline=DEADLINE
    )

    assert result.accepted is True
    assert result.reported_filled_size == Decimal("2")
    assert captured["vault_address"] == HEDGE
    assert captured["account_address"] == HEDGE
    assert captured["perp_dexs"] == ["", "xyz"]
    assert captured["timeout"] == 5
    assert PRIMARY not in captured.values()
    assert fake.calls[0] == ("leverage", 1, "xyz:SKHX", True)
    assert fake.calls[1][0] == "price"
    assert fake.calls[2][0:7] == (
        "order",
        "xyz:SKHX",
        False,
        2.0,
        100.0,
        {"limit": {"tif": "Ioc"}},
        False,
    )
    assert fake.expires_after is not None


def test_close_uses_sdk_reduce_only_market_close_boundary():
    fake = FakeExchange()
    executor = HyperliquidHedgeExecutor(
        config(),
        Reader(),
        exchange_factory=lambda *args, **kwargs: fake,
        wallet_factory=lambda secret: Wallet(),
        env={"FUNDING_HEDGE_SECRET_KEY": "secret"},
    )

    executor.close_hedge(
        "xyz:SKHX",
        Decimal("-2"),
        Decimal("2"),
        stable_cloid("cycle", "close:1"),
    )

    assert fake.calls[0][0:3] == ("price", "xyz:SKHX", True)
    assert fake.calls[1][0:7] == (
        "order",
        "xyz:SKHX",
        True,
        2.0,
        100.0,
        {"limit": {"tif": "Ioc"}},
        True,
    )


def test_independent_wallet_signer_must_equal_hedge_address():
    executor = HyperliquidHedgeExecutor(
        config("wallet"),
        Reader(),
        exchange_factory=lambda *args, **kwargs: FakeExchange(),
        wallet_factory=lambda secret: Wallet(),
        env={"FUNDING_HEDGE_SECRET_KEY": "secret"},
    )

    with pytest.raises(RuntimeError, match="signer does not match"):
        executor.open_hedge(
            "xyz:SKHX",
            Decimal("-1"),
            stable_cloid("cycle", "open:1"),
            submit_deadline=DEADLINE,
        )


def test_leverage_configuration_failure_blocks_open_order():
    class LeverageRejectedExchange(FakeExchange):
        def update_leverage(self, leverage, symbol, is_cross=True):
            self.calls.append(("leverage", leverage, symbol, is_cross))
            return {"status": "err", "response": "invalid leverage"}

    fake = LeverageRejectedExchange()
    executor = HyperliquidHedgeExecutor(
        config(),
        Reader(),
        exchange_factory=lambda *args, **kwargs: fake,
        wallet_factory=lambda secret: Wallet(),
        env={"FUNDING_HEDGE_SECRET_KEY": "secret"},
    )

    with pytest.raises(RuntimeError, match="failed to set hedge leverage"):
        executor.open_hedge(
            "xyz:SKHX",
            Decimal("-1"),
            stable_cloid("cycle", "open:1"),
            submit_deadline=DEADLINE,
        )
    assert all(call[0] != "order" for call in fake.calls)


def test_expired_submit_deadline_blocks_position_changing_order():
    fake = FakeExchange()
    executor = HyperliquidHedgeExecutor(
        config(),
        Reader(),
        exchange_factory=lambda *args, **kwargs: fake,
        wallet_factory=lambda secret: Wallet(),
        env={"FUNDING_HEDGE_SECRET_KEY": "secret"},
    )

    with pytest.raises(RuntimeError, match="deadline"):
        executor.open_hedge(
            "xyz:SKHX",
            Decimal("-1"),
            stable_cloid("cycle", "open:1"),
            submit_deadline=datetime.now(tz=timezone.utc) - timedelta(seconds=1),
        )
    assert all(call[0] != "order" for call in fake.calls)


def test_close_loads_active_hip3_dex_even_after_symbol_removed_from_allowlist():
    captured = {}
    fake = FakeExchange()
    cfg = StrategyConfig(
        symbols=(),
        primary_account_address=PRIMARY,
        hedge_account_address=HEDGE,
        execution_enabled=True,
    )

    def factory(wallet, base, **kwargs):
        captured.update(kwargs)
        return fake

    executor = HyperliquidHedgeExecutor(
        cfg,
        Reader(),
        exchange_factory=factory,
        wallet_factory=lambda secret: Wallet(),
        env={"FUNDING_HEDGE_SECRET_KEY": "secret"},
    )
    executor.close_hedge(
        "xyz:SKHX",
        Decimal("-1"),
        Decimal("1"),
        stable_cloid("cycle", "close:1"),
    )

    assert captured["perp_dexs"] == ["", "xyz"]
    assert fake.calls[-1][0] == "order"
    assert fake.calls[-1][6] is True
