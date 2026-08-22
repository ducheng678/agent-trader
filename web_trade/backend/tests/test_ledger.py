from __future__ import annotations

import pytest

from web_trade.backend.web_trade.ledger import SyntheticPositionLedger


def _open_snapshot(**overrides):
    snapshot = {
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
        "unrealized_pnl": 100.0,
    }
    snapshot.update(overrides)
    return snapshot


def test_overlay_initializes_lifecycle_entry_and_net_roi_basis(tmp_path):
    ledger = SyntheticPositionLedger(tmp_path / "positions.json")

    view = ledger.overlay_position(_open_snapshot())

    assert view["display_entry_price"] == pytest.approx(100000.0)
    assert view["lifecycle_roi_basis_usd"] == pytest.approx(1010.0)
    assert view["carried_realized_pnl_usd"] == pytest.approx(0.0)
    assert view["synthetic_pnl_usd"] == pytest.approx(100.0)
    assert view["synthetic_pnl_pct"] == pytest.approx(100.0 / 1010.0)
    assert view["liquidation_price"] == pytest.approx(92000.0)


def test_margin_add_and_remove_update_net_roi_basis(tmp_path):
    ledger = SyntheticPositionLedger(tmp_path / "positions.json")
    ledger.overlay_position(_open_snapshot())

    ledger.apply_margin_delta("BTC", 250.0)
    view_after_add = ledger.overlay_position(_open_snapshot(unrealized_pnl=125.0))

    assert view_after_add["lifecycle_roi_basis_usd"] == pytest.approx(1260.0)
    assert view_after_add["synthetic_pnl_pct"] == pytest.approx(125.0 / 1260.0)

    ledger.apply_margin_delta("BTC", -60.0)
    view_after_remove = ledger.overlay_position(_open_snapshot(unrealized_pnl=90.0))

    assert view_after_remove["lifecycle_roi_basis_usd"] == pytest.approx(1200.0)
    assert view_after_remove["synthetic_pnl_pct"] == pytest.approx(90.0 / 1200.0)


def test_hidden_rebalance_keeps_display_entry_and_accumulates_realized_pnl(tmp_path):
    ledger = SyntheticPositionLedger(tmp_path / "positions.json")
    ledger.overlay_position(_open_snapshot(entry_price=100000.0, unrealized_pnl=140.0))

    ledger.record_hidden_rebalance(
        symbol="BTC",
        realized_pnl_usd=140.0,
        target_leverage=5,
        target_notional_usd=5050.0,
    )
    view = ledger.overlay_position(
        _open_snapshot(
            entry_price=101500.0,
            notional_usd=5050.0,
            leverage=5.0,
            margin_used=1010.0,
            unrealized_pnl=-20.0,
            liquidation_price=81000.0,
        )
    )

    assert view["display_entry_price"] == pytest.approx(100000.0)
    assert view["carried_realized_pnl_usd"] == pytest.approx(140.0)
    assert view["synthetic_pnl_usd"] == pytest.approx(120.0)
    assert view["synthetic_pnl_pct"] == pytest.approx(120.0 / 1010.0)
    assert view["liquidation_price"] == pytest.approx(81000.0)
