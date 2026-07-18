from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal

import pytest

from funding_fee_avoidance.config import StrategyConfig
from funding_fee_avoidance.models import (
    AccountPositionSnapshot,
    FundingObservation,
    HedgeAccountSnapshot,
    HedgeAction,
    HedgeCycleState,
    HedgeCycleStatus,
    HedgeSnapshot,
)
from funding_fee_avoidance.policy import FundingHedgePolicy
from funding_fee_avoidance.state_machine import stable_cloid


UTC = timezone.utc
NOW = datetime(2026, 7, 10, 12, 58, 30, tzinfo=UTC)
PRIMARY = "0x1111111111111111111111111111111111111111"
HEDGE = "0x2222222222222222222222222222222222222222"


def config(**overrides) -> StrategyConfig:
    values = {
        "symbols": ("xyz:SKHX",),
        "primary_account_address": PRIMARY,
        "hedge_account_address": HEDGE,
        "hedge_open_lead_seconds": 120,
        "latest_open_cutoff_seconds": 15,
        "min_expected_saving_usd": Decimal("1"),
        "min_available_margin_usd": Decimal("0"),
        "margin_safety_multiplier": Decimal("1"),
        "slippage_rate_per_order": Decimal("0"),
        "risk_buffer_rate": Decimal("0"),
        "cost_safety_multiplier": Decimal("1"),
        "hip3_extra_fee_rate_known": True,
    }
    values.update(overrides)
    return StrategyConfig(**values)


def snapshot(**overrides) -> HedgeSnapshot:
    primary_size = overrides.pop("primary_size", Decimal("100"))
    hedge_size = overrides.pop("hedge_size", Decimal("0"))
    funding_rate = overrides.pop("funding_rate", Decimal("0.002"))
    oracle = overrides.pop("oracle_price", Decimal("100"))
    mark = overrides.pop("mark_price", Decimal("101"))
    observed_at = overrides.pop("observed_at", NOW - timedelta(seconds=5))
    next_funding = overrides.pop("next_funding_at", NOW + timedelta(seconds=90))
    withdrawable = overrides.pop("withdrawable_usd", Decimal("50000"))
    fee = overrides.pop("taker_fee_rate", Decimal("0.0001"))
    ownership = overrides.pop("ownership_verified", True)
    margin_verified = overrides.pop("margin_available_verified", True)
    funding_confirmed = overrides.pop("funding_confirmed", False)
    unknown_orders = overrides.pop("unknown_open_orders", False)
    primary_address = overrides.pop("primary_address", PRIMARY)
    hedge_address = overrides.pop("hedge_address", HEDGE)
    size_decimals = overrides.pop("size_decimals", 3)
    assert not overrides
    return HedgeSnapshot(
        primary_position=AccountPositionSnapshot(
            account_address=primary_address,
            symbol="xyz:SKHX",
            size=primary_size,
        ),
        hedge_account=HedgeAccountSnapshot(
            position=AccountPositionSnapshot(
                account_address=hedge_address,
                symbol="xyz:SKHX",
                size=hedge_size,
            ),
            withdrawable_usd=withdrawable,
            taker_fee_rate=fee,
            fee_rate_source="test",
            ownership_verified=ownership,
            margin_available_verified=margin_verified,
            unknown_open_orders=unknown_orders,
            funding_confirmed=funding_confirmed,
            funding_record_time=(NOW + timedelta(seconds=90) if funding_confirmed else None),
        ),
        funding=FundingObservation(
            symbol="xyz:SKHX",
            oracle_price=oracle,
            mark_price=mark,
            funding_rate=funding_rate,
            observed_at=observed_at,
            next_funding_at=next_funding,
            size_decimals=size_decimals,
        ),
    )


def cycle(target: Decimal = Decimal("-100"), **overrides) -> HedgeCycleState:
    settlement = overrides.pop("settlement_at", NOW + timedelta(seconds=90))
    key = f"{PRIMARY}:{HEDGE}:xyz:SKHX:{int(settlement.timestamp() * 1000)}"
    values = {
        "cycle_key": key,
        "status": HedgeCycleStatus.HEDGED,
        "symbol": "xyz:SKHX",
        "primary_account_address": PRIMARY,
        "hedge_account_address": HEDGE,
        "settlement_at": settlement,
        "target_hedge_size": target,
        "actual_hedge_size": target,
        "open_cloid": stable_cloid(key, "open:1"),
        "close_cloid": "",
        "created_at": NOW - timedelta(seconds=10),
        "updated_at": NOW,
    }
    values.update(overrides)
    return HedgeCycleState(**values)


def test_positive_funding_long_opens_equal_short_in_hedge_account():
    decision = FundingHedgePolicy(config()).evaluate(snapshot(), NOW)

    assert decision.action is HedgeAction.OPEN_HEDGE
    assert decision.target_hedge_size == Decimal("-100.000")
    assert decision.order_size_delta == Decimal("-100.000")
    assert decision.reduce_only is False
    assert decision.estimated_primary_funding_debit_usd == Decimal("20.000")
    assert decision.estimated_hedge_funding_credit_usd == Decimal("20.000000")
    assert decision.estimated_round_trip_cost_usd == Decimal("2.0200000")
    assert decision.estimated_net_saving_usd == Decimal("17.9800000")


def test_negative_funding_short_opens_long_hedge():
    decision = FundingHedgePolicy(config()).evaluate(
        snapshot(primary_size=Decimal("-100"), funding_rate=Decimal("-0.002")), NOW
    )

    assert decision.action is HedgeAction.OPEN_HEDGE
    assert decision.target_hedge_size == Decimal("100.000")
    assert decision.order_size_delta > 0


@pytest.mark.parametrize(
    ("size", "rate"),
    [
        (Decimal("100"), Decimal("-0.002")),
        (Decimal("-100"), Decimal("0.002")),
        (Decimal("100"), Decimal("0")),
    ],
)
def test_receiving_or_zero_funding_never_opens(size: Decimal, rate: Decimal):
    decision = FundingHedgePolicy(config()).evaluate(
        snapshot(primary_size=size, funding_rate=rate), NOW
    )

    assert decision.action is HedgeAction.HOLD
    assert decision.reason == "primary_receives_or_owes_no_funding"


def test_partial_hedge_ratio_preserves_partial_directional_exposure():
    decision = FundingHedgePolicy(config(hedge_ratio=Decimal("0.4"))).evaluate(
        snapshot(), NOW
    )

    assert decision.action is HedgeAction.OPEN_HEDGE
    assert decision.target_hedge_size == Decimal("-40.000")
    assert decision.estimated_hedge_funding_credit_usd == Decimal("8.000000")


def test_size_is_rounded_down_and_never_overhedges():
    decision = FundingHedgePolicy(config()).evaluate(
        snapshot(primary_size=Decimal("1.2399"), oracle_price=Decimal("10000"), size_decimals=2),
        NOW,
    )

    assert decision.target_hedge_size == Decimal("-1.23")
    assert abs(decision.target_hedge_size) <= Decimal("1.2399")


def test_funding_debit_uses_oracle_and_execution_cost_uses_mark():
    decision = FundingHedgePolicy(config()).evaluate(
        snapshot(oracle_price=Decimal("100"), mark_price=Decimal("150")), NOW
    )

    assert decision.estimated_primary_funding_debit_usd == Decimal("20.000")
    assert decision.estimated_round_trip_cost_usd == Decimal("3.0000000")


def test_stale_data_blocks_opening():
    decision = FundingHedgePolicy(config()).evaluate(
        snapshot(observed_at=NOW - timedelta(seconds=31)), NOW
    )

    assert decision.action is HedgeAction.HOLD
    assert decision.reason == "funding_data_is_stale"


def test_hip3_fee_must_be_explicitly_confirmed():
    decision = FundingHedgePolicy(
        config(hip3_extra_fee_rate_known=False)
    ).evaluate(snapshot(), NOW)

    assert decision.action is HedgeAction.HOLD
    assert decision.reason == "hip3_extra_fee_rate_not_confirmed"


def test_unverified_margin_blocks_opening():
    decision = FundingHedgePolicy(config()).evaluate(
        snapshot(margin_available_verified=False), NOW
    )

    assert decision.action is HedgeAction.HOLD
    assert decision.reason == "hedge_available_margin_not_verified"


def test_untracked_existing_hedge_requires_recovery_and_never_adds():
    decision = FundingHedgePolicy(config()).evaluate(
        snapshot(hedge_size=Decimal("-40")), NOW
    )

    assert decision.action is HedgeAction.RECOVERY_REQUIRED
    assert decision.order_size_delta == 0


def test_confirmed_funding_closes_only_hedge_reduce_only():
    snap = snapshot(hedge_size=Decimal("-100"), funding_confirmed=True)
    decision = FundingHedgePolicy(config()).evaluate(snap, NOW + timedelta(seconds=95), cycle())

    assert decision.action is HedgeAction.CLOSE_HEDGE
    assert decision.order_size_delta == Decimal("100")
    assert decision.reduce_only is True


def test_confirmation_timeout_closes_even_if_current_market_data_is_stale():
    snap = snapshot(
        hedge_size=Decimal("-100"),
        observed_at=NOW - timedelta(minutes=10),
    )
    decision = FundingHedgePolicy(config()).evaluate(
        snap,
        NOW + timedelta(seconds=90 + 121),
        cycle(),
    )

    assert decision.action is HedgeAction.CLOSE_HEDGE
    assert decision.reason == "funding_confirmation_timeout"


def test_funding_sign_reversal_immediately_unwinds_tracked_hedge():
    decision = FundingHedgePolicy(config()).evaluate(
        snapshot(hedge_size=Decimal("-100"), funding_rate=Decimal("-0.002")),
        NOW,
        cycle(),
    )

    assert decision.action is HedgeAction.CLOSE_HEDGE
    assert decision.reason == "funding_direction_reversed_or_primary_no_longer_pays"
    assert decision.estimated_hedge_funding_credit_usd == 0


def test_unknown_order_cannot_strand_a_tracked_hedge():
    decision = FundingHedgePolicy(config()).evaluate(
        snapshot(hedge_size=Decimal("-100"), unknown_open_orders=True),
        NOW,
        cycle(),
    )

    assert decision.action is HedgeAction.CLOSE_HEDGE
    assert decision.reason == "unknown_order_requires_tracked_hedge_shutdown"
    assert decision.reduce_only is True


def test_missing_market_data_closes_tracked_hedge_but_cannot_open_new_one():
    tracked = snapshot(hedge_size=Decimal("-100"))
    tracked = HedgeSnapshot(
        tracked.primary_position,
        tracked.hedge_account,
        FundingObservation(
            **{
                **tracked.funding.__dict__,
                "market_data_available": False,
            }
        ),
    )
    decision = FundingHedgePolicy(config()).evaluate(tracked, NOW, cycle())

    assert decision.action is HedgeAction.CLOSE_HEDGE
    assert decision.reason == "market_data_unavailable_for_tracked_hedge"


def test_primary_shrink_triggers_reduce_only_adjustment():
    snap = snapshot(primary_size=Decimal("50"), hedge_size=Decimal("-100"))
    decision = FundingHedgePolicy(config()).evaluate(snap, NOW, cycle())

    assert decision.action is HedgeAction.ADJUST_HEDGE
    assert decision.target_hedge_size == Decimal("-50.000")
    assert decision.order_size_delta == Decimal("50.000")
    assert decision.reduce_only is True


def test_primary_flat_triggers_reduce_only_close():
    decision = FundingHedgePolicy(config()).evaluate(
        snapshot(primary_size=Decimal("0"), hedge_size=Decimal("-100")),
        NOW,
        cycle(),
    )

    assert decision.action is HedgeAction.CLOSE_HEDGE
    assert decision.reason == "primary_position_is_flat"


def test_equal_primary_and_hedge_addresses_are_rejected():
    decision = FundingHedgePolicy(config()).evaluate(
        snapshot(hedge_address=PRIMARY), NOW
    )

    assert decision.action is HedgeAction.RECOVERY_REQUIRED
    assert decision.reason == "primary_and_hedge_accounts_are_equal"
