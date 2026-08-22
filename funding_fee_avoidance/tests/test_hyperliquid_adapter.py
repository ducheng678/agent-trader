from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

from funding_fee_avoidance.config import StrategyConfig
from funding_fee_avoidance.hyperliquid_adapter import (
    HyperliquidSnapshotAdapter,
    _funding_record,
    hip3_fee_rates,
)
from funding_fee_avoidance.models import HedgeCycleState, HedgeCycleStatus
from funding_fee_avoidance.models import HedgeAction
from funding_fee_avoidance.policy import FundingHedgePolicy
from funding_fee_avoidance.state_machine import stable_cloid


UTC = timezone.utc
PRIMARY = "0x1111111111111111111111111111111111111111"
HEDGE = "0x2222222222222222222222222222222222222222"
SETTLEMENT = datetime(2026, 7, 10, 13, 0, tzinfo=UTC)


class FakeReader:
    account_address = PRIMARY
    base = "https://example.test"

    def __init__(self, *, confirmed=False):
        self.confirmed = confirmed
        self.payloads = []

    def post_info(self, payload):
        self.payloads.append(payload)
        kind = payload["type"]
        if kind == "subAccounts":
            return [{"name": "funding-hedge", "subAccountUser": HEDGE}]
        if kind == "userFees":
            assert payload["user"] == HEDGE
            return {
                "userCrossRate": "0.00045",
                "userAddRate": "0.00015",
                "activeReferralDiscount": "0",
            }
        if kind == "perpDexs":
            return [None, {"name": "xyz", "deployerFeeScale": "1.0"}]
        if kind == "userAbstraction":
            assert payload["user"] == HEDGE
            return "disabled"
        if kind == "clearinghouseState":
            assert payload["dex"] == "xyz"
            if payload["user"] == PRIMARY:
                return {
                    "withdrawable": "0",
                    "assetPositions": [
                        {"position": {"coin": "xyz:SKHX", "szi": "123.456"}}
                    ],
                }
            assert payload["user"] == HEDGE
            return {"withdrawable": "50000", "assetPositions": []}
        if kind == "metaAndAssetCtxs":
            assert payload == {"type": "metaAndAssetCtxs", "dex": "xyz"}
            return [
                {
                    "universe": [
                        {
                            "name": "xyz:SKHX",
                            "szDecimals": 3,
                            "growthMode": "enabled",
                        }
                    ]
                },
                [
                    {
                        "oraclePx": "212.50",
                        "markPx": "212.55",
                        "funding": "0.0002",
                    }
                ],
            ]
        if kind == "frontendOpenOrders":
            assert payload["user"] == HEDGE
            assert payload["dex"] == "xyz"
            return []
        if kind == "orderStatus":
            return {"status": "unknownOid"}
        if kind == "userFunding":
            if not self.confirmed:
                return []
            return [
                {
                    "time": int((SETTLEMENT + timedelta(seconds=3)).timestamp() * 1000),
                    "delta": {
                        "type": "funding",
                        "coin": "xyz:SKHX",
                        "usdc": "4.25",
                        "szi": "-100",
                        "fundingRate": "0.0002",
                    },
                }
            ]
        raise AssertionError(payload)


class MetaFailureWithTrackedHedgeReader(FakeReader):
    def post_info(self, payload):
        if payload["type"] == "metaAndAssetCtxs":
            raise RuntimeError("temporary market-data failure")
        if payload["type"] == "clearinghouseState" and payload["user"] == HEDGE:
            self.payloads.append(payload)
            return {
                "withdrawable": "50000",
                "assetPositions": [
                    {"position": {"coin": "xyz:SKHX", "szi": "-100"}}
                ],
            }
        return super().post_info(payload)


def config() -> StrategyConfig:
    return StrategyConfig(
        symbols=("xyz:SKHX",),
        primary_account_address=PRIMARY,
        hedge_account_address=HEDGE,
        hip3_extra_fee_rate_known=True,
    )


def cycle() -> HedgeCycleState:
    key = f"{PRIMARY}:{HEDGE}:xyz:SKHX:{int(SETTLEMENT.timestamp() * 1000)}"
    return HedgeCycleState(
        cycle_key=key,
        status=HedgeCycleStatus.AWAITING_FUNDING,
        symbol="xyz:SKHX",
        primary_account_address=PRIMARY,
        hedge_account_address=HEDGE,
        settlement_at=SETTLEMENT,
        target_hedge_size=Decimal("-100"),
        actual_hedge_size=Decimal("-100"),
        open_cloid=stable_cloid(key, "open:1"),
        close_cloid="",
        created_at=SETTLEMENT - timedelta(minutes=2),
        updated_at=SETTLEMENT,
    )


def test_adapter_accepts_hip3_and_uses_fresh_dex_scoped_endpoints():
    reader = FakeReader()
    observed = SETTLEMENT - timedelta(seconds=30)
    snapshots, errors = HyperliquidSnapshotAdapter(config(), reader).load_snapshots(observed)

    assert errors == []
    assert len(snapshots) == 1
    snap = snapshots[0]
    assert snap.symbol == "xyz:SKHX"
    assert snap.primary_position.size == Decimal("123.456")
    assert snap.hedge_account.position.size == 0
    assert snap.hedge_account.ownership_verified is True
    assert snap.hedge_account.margin_available_verified is True
    assert snap.hedge_account.hip3_fee_formula_verified is True
    assert snap.hedge_account.taker_fee_rate == Decimal("0.000090")
    assert snap.funding.oracle_price == Decimal("212.50")
    assert snap.funding.funding_rate == Decimal("0.0002")
    assert snap.funding.size_decimals == 3
    assert {"type": "predictedFundings"} not in reader.payloads


def test_adapter_confirms_exact_coin_and_target_hour_user_funding():
    reader = FakeReader(confirmed=True)
    active = cycle()
    snapshots, errors = HyperliquidSnapshotAdapter(config(), reader).load_snapshots(
        SETTLEMENT + timedelta(seconds=10), {"xyz:SKHX": active}
    )

    assert errors == []
    assert snapshots[0].hedge_account.funding_confirmed is True
    assert snapshots[0].hedge_account.funding_delta_usd == Decimal("4.25")
    query = next(item for item in reader.payloads if item["type"] == "userFunding")
    assert query["user"] == HEDGE


def test_funding_match_rejects_other_coin_and_other_hour():
    raw = [
        {
            "time": int((SETTLEMENT - timedelta(hours=1)).timestamp() * 1000),
            "delta": {"coin": "xyz:SKHX", "usdc": "1"},
        },
        {
            "time": int(SETTLEMENT.timestamp() * 1000),
            "delta": {"coin": "xyz:SKHY", "usdc": "1"},
        },
    ]

    assert _funding_record(raw, "xyz:SKHX", SETTLEMENT) == (False, None, None)


def test_official_hip3_fee_formula_matches_growth_mode_xyz_example():
    maker, taker = hip3_fee_rates(
        {
            "userCrossRate": "0.00045",
            "userAddRate": "0.00015",
            "activeReferralDiscount": "0",
        },
        {"deployerFeeScale": "1.0"},
        {"growthMode": "enabled"},
    )

    assert maker == Decimal("0.000030")
    assert taker == Decimal("0.000090")


def test_unknown_growth_mode_fails_closed_in_fee_formula():
    import pytest

    with pytest.raises(ValueError, match="growthMode"):
        hip3_fee_rates(
            {
                "userCrossRate": "0.00045",
                "userAddRate": "0.00015",
                "activeReferralDiscount": "0",
            },
            {"deployerFeeScale": "1.0"},
            {"growthMode": "future-mode"},
        )


def test_market_data_failure_still_builds_emergency_close_snapshot_for_tracked_hedge():
    active = cycle()
    reader = MetaFailureWithTrackedHedgeReader()
    snapshots, errors = HyperliquidSnapshotAdapter(config(), reader).load_snapshots(
        SETTLEMENT - timedelta(seconds=30), {"xyz:SKHX": active}
    )

    assert len(snapshots) == 1
    assert snapshots[0].funding.market_data_available is False
    assert snapshots[0].hedge_account.position.size == Decimal("-100")
    assert any("only tracked-hedge shutdown" in error for error in errors)
    decision = FundingHedgePolicy(config()).evaluate(
        snapshots[0], SETTLEMENT - timedelta(seconds=30), active
    )
    assert decision.action is HedgeAction.CLOSE_HEDGE


def test_active_cycle_symbol_is_read_even_if_removed_from_open_allowlist():
    active = cycle()
    cfg = StrategyConfig(
        symbols=(),
        primary_account_address=PRIMARY,
        hedge_account_address=HEDGE,
        hip3_extra_fee_rate_known=True,
    )
    snapshots, _ = HyperliquidSnapshotAdapter(cfg, FakeReader()).load_snapshots(
        SETTLEMENT - timedelta(seconds=30), {"xyz:SKHX": active}
    )

    assert [item.symbol for item in snapshots] == ["xyz:SKHX"]
