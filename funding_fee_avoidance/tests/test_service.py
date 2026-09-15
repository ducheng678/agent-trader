from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from funding_fee_avoidance.config import StrategyConfig
from funding_fee_avoidance.models import (
    AccountPositionSnapshot,
    FundingObservation,
    HedgeAccountSnapshot,
    HedgeSnapshot,
)
from funding_fee_avoidance.service import FundingHedgeService


UTC = timezone.utc


def test_offline_report_can_only_recommend_hedge_account_order():
    now = datetime(2026, 7, 10, 12, 58, 30, tzinfo=UTC)
    snapshot = HedgeSnapshot(
        primary_position=AccountPositionSnapshot(
            "0x1111111111111111111111111111111111111111",
            "xyz:SKHX",
            Decimal("100"),
        ),
        hedge_account=HedgeAccountSnapshot(
            position=AccountPositionSnapshot(
                "0x2222222222222222222222222222222222222222",
                "xyz:SKHX",
                Decimal("0"),
            ),
            withdrawable_usd=Decimal("50000"),
            taker_fee_rate=Decimal("0.0001"),
            fee_rate_source="test",
            ownership_verified=True,
        ),
        funding=FundingObservation(
            symbol="xyz:SKHX",
            oracle_price=Decimal("100"),
            mark_price=Decimal("100"),
            funding_rate=Decimal("0.002"),
            observed_at=now,
            next_funding_at=now + timedelta(seconds=90),
            size_decimals=3,
        ),
    )
    config = StrategyConfig(
        symbols=("xyz:SKHX",),
        slippage_rate_per_order=Decimal("0"),
        risk_buffer_rate=Decimal("0"),
        cost_safety_multiplier=Decimal("1"),
        min_available_margin_usd=Decimal("0"),
        hip3_extra_fee_rate_known=True,
    )

    report = FundingHedgeService.evaluate_snapshots(config, [snapshot], now)

    assert report["mode"] == "report_only"
    assert report["orders_enabled"] is False
    assert report["primary_account_is_read_only"] is True
    assert report["order_route"] == "hedge_account_only"
    assert report["summary"]["open_hedge"] == 1
    assert report["decisions"][0]["order_size_delta"] == "-100.000"
