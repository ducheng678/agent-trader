from __future__ import annotations

import pytest

from web_trade.backend.web_trade.margin import calculate_margin_limits


def test_isolated_margin_limits_include_add_and_transfer_safe_remove_amounts():
    limits = calculate_margin_limits(
        {
            "symbol": "BTC",
            "side": "long",
            "only_isolated": True,
            "notional_usd": 1000.0,
            "max_leverage": 20,
            "margin_used": 150.0,
            "unrealized_pnl": 30.0,
            "available_margin_usd": 400.0,
        },
        safety_buffer_usd=2.0,
    )

    assert limits["enabled"] is True
    assert limits["max_add_margin_usd"] == pytest.approx(398.0)
    assert limits["isolated_position_equity_usd"] == pytest.approx(180.0)
    assert limits["required_remaining_margin_usd"] == pytest.approx(100.0)
    assert limits["max_remove_margin_usd"] == pytest.approx(78.0)


def test_margin_limits_disable_cross_or_flat_positions():
    cross_limits = calculate_margin_limits(
        {
            "symbol": "BTC",
            "side": "long",
            "only_isolated": False,
            "notional_usd": 1000.0,
            "margin_used": 100.0,
            "available_margin_usd": 300.0,
        }
    )
    flat_limits = calculate_margin_limits(
        {
            "symbol": "BTC",
            "side": "flat",
            "only_isolated": True,
            "notional_usd": 0.0,
            "margin_used": 0.0,
            "available_margin_usd": 300.0,
        }
    )

    assert cross_limits["enabled"] is False
    assert cross_limits["reason"] == "not_isolated"
    assert flat_limits["enabled"] is False
    assert flat_limits["reason"] == "no_open_position"
