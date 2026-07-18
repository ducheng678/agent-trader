from __future__ import annotations

import os

import pytest


RUN_REAL_HYPERLIQUID_TESTNET_TESTS = os.environ.get("RUN_REAL_HYPERLIQUID_TESTNET_TESTS", "").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}
RUN_REAL_HYPERLIQUID_TESTNET_ORDER_PROBE = os.environ.get(
    "RUN_REAL_HYPERLIQUID_TESTNET_ORDER_PROBE", ""
).strip().lower() in {"1", "true", "yes", "on"}


@pytest.mark.skipif(
    not RUN_REAL_HYPERLIQUID_TESTNET_TESTS,
    reason="Set RUN_REAL_HYPERLIQUID_TESTNET_TESTS=true to run live Hyperliquid testnet integration tests.",
)
def test_hyperliquid_testnet_reader_smoke(uma, monkeypatch):
    monkeypatch.setenv("HYPERLIQUID_NETWORK", "testnet")

    reader = uma.HyperliquidRestReader()
    snapshot = reader.get_position_snapshot("BTC")
    all_positions = reader.get_all_positions()
    safe_spot = reader.get_safe_spot_meta()

    assert snapshot["network"] == "testnet"
    assert snapshot["symbol"] == "BTC"
    assert snapshot["max_leverage"] >= 1
    assert "remaining_capital_usd" in snapshot
    assert all_positions["network"] == "testnet"
    assert "spot_usdc_total" in all_positions
    assert "spot_available_usdc" in all_positions
    assert isinstance(safe_spot.get("tokens"), list)
    assert isinstance(safe_spot.get("universe"), list)
    assert int(safe_spot.get("_dropped_invalid_universe_entries", 0) or 0) >= 0
    if float(all_positions.get("spot_available_usdc", 0.0) or 0.0) > 0:
        assert float(all_positions.get("remaining_capital_usd", 0.0) or 0.0) >= float(
            all_positions.get("spot_available_usdc", 0.0) or 0.0
        )


@pytest.mark.skipif(
    not RUN_REAL_HYPERLIQUID_TESTNET_ORDER_PROBE,
    reason="Set RUN_REAL_HYPERLIQUID_TESTNET_ORDER_PROBE=true to run a live Hyperliquid testnet order probe.",
)
def test_hyperliquid_testnet_order_probe_reaches_exchange(uma, monkeypatch):
    monkeypatch.setenv("HYPERLIQUID_NETWORK", "testnet")
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")

    reader = uma.HyperliquidRestReader()
    executor = uma.HyperliquidExecutor(reader, "BTC")
    mid = float(reader.get_mid_price("BTC") or 0.0)

    assert mid > 0

    decision = uma.StrategyDecision(
        action="long",
        suggested_notional_usd=10.0,
        requested_leverage=2,
        entry_price=mid,
        entry_price_low=mid,
        entry_price_high=mid,
        take_profit_price=mid * 1.01,
        stop_loss_price=mid * 0.99,
        planned_margin_used_usd=5.0,
        planned_max_loss_usd=0.1,
        need_web_confirmation=False,
        thesis="Hyperliquid testnet order probe",
        invalidation="Probe only",
        key_risks=["testnet integration"],
    )

    result = executor.execute(decision, plan_name="testnet_probe")

    assert result["mode"] == "live"
    assert "leverage_update" in result
    assert result["actions"]
    assert "open" in result["actions"][0]
    assert result["leverage_update"]["exchange"]["status"] in {"ok", "err"}
    assert result["actions"][0]["open"]["status"] in {"ok", "err"}
