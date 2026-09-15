from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_DOWN
from typing import Optional, Tuple

from .config import StrategyConfig
from .models import (
    HedgeAction,
    HedgeCycleState,
    HedgeCycleStatus,
    HedgeDecision,
    HedgeSnapshot,
)


UTC = timezone.utc
ZERO = Decimal("0")
ONE = Decimal("1")


def _sign(value: Decimal) -> Decimal:
    if value > 0:
        return ONE
    if value < 0:
        return -ONE
    return ZERO


def _quantize_down(value: Decimal, decimals: int) -> Decimal:
    if value <= 0:
        return ZERO
    quantum = ONE.scaleb(-max(0, decimals))
    return value.quantize(quantum, rounding=ROUND_DOWN)


class FundingHedgePolicy:
    """Pure policy for a temporary hedge held in an independent account.

    It has no exchange write dependency.  The primary position appears only as
    input data; every non-zero order delta refers to the hedge account.
    """

    def __init__(self, config: StrategyConfig):
        self.config = config

    def _target_size(
        self, snapshot: HedgeSnapshot, *, apply_margin_cap: bool
    ) -> Tuple[Decimal, Decimal]:
        primary_size = snapshot.primary_position.size
        funding = snapshot.funding
        qty = abs(primary_size) * self.config.hedge_ratio

        if self.config.max_hedge_notional_usd > 0:
            qty = min(qty, self.config.max_hedge_notional_usd / funding.mark_price)

        if apply_margin_cap:
            margin_budget = max(
                ZERO,
                snapshot.hedge_account.withdrawable_usd
                - self.config.min_available_margin_usd,
            )
            margin_qty_cap = (
                margin_budget
                * self.config.assumed_hedge_leverage
                / funding.mark_price
                / self.config.margin_safety_multiplier
            )
            qty = min(qty, margin_qty_cap)

        qty = _quantize_down(qty, funding.size_decimals)
        return -_sign(primary_size) * qty, qty

    def evaluate(
        self,
        snapshot: HedgeSnapshot,
        now: datetime,
        cycle: Optional[HedgeCycleState] = None,
    ) -> HedgeDecision:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must include a timezone")
        evaluated_at = now.astimezone(UTC)
        primary = snapshot.primary_position
        hedge = snapshot.hedge_account.position
        funding = snapshot.funding
        active_cycle = cycle if cycle is not None and cycle.active else None
        settlement_at = (
            active_cycle.settlement_at.astimezone(UTC)
            if active_cycle is not None
            else funding.next_funding_at.astimezone(UTC)
        )
        seconds_to_funding = Decimal(
            str((settlement_at - evaluated_at).total_seconds())
        )
        data_age_seconds = Decimal(
            str((evaluated_at - funding.observed_at.astimezone(UTC)).total_seconds())
        )
        settlement_ms = int(settlement_at.timestamp() * 1000)
        cycle_key = (
            active_cycle.cycle_key
            if active_cycle is not None
            else ":".join(
                (
                    self.config.network,
                    primary.account_address.lower(),
                    hedge.account_address.lower(),
                    snapshot.symbol,
                    str(settlement_ms),
                )
            )
        )

        current_hedge_size = hedge.size
        target_hedge_size, target_qty = self._target_size(
            snapshot, apply_margin_cap=current_hedge_size == 0
        )
        primary_payment = (
            abs(primary.size)
            * funding.oracle_price
            * funding.funding_rate
            * _sign(primary.size)
        )
        primary_debit = max(ZERO, primary_payment)
        economics_size = current_hedge_size if current_hedge_size != 0 else target_hedge_size
        economics_qty = abs(economics_size)
        hedge_payment = (
            economics_qty
            * funding.oracle_price
            * funding.funding_rate
            * _sign(economics_size)
        )
        hedge_credit = max(ZERO, -hedge_payment)
        hedge_execution_notional = economics_qty * funding.mark_price
        round_trip_rate = (
            Decimal("2")
            * (
                snapshot.hedge_account.taker_fee_rate
                + self.config.hip3_extra_fee_rate_per_order
                + self.config.slippage_rate_per_order
            )
            + self.config.risk_buffer_rate
        )
        round_trip_cost = (
            hedge_execution_notional
            * round_trip_rate
            * self.config.cost_safety_multiplier
        )
        net_saving = hedge_credit - round_trip_cost
        required_margin = (
            hedge_execution_notional
            / self.config.assumed_hedge_leverage
            * self.config.margin_safety_multiplier
        )

        action = HedgeAction.HOLD
        reason = "no_action"
        order_delta = ZERO
        reduce_only = False
        quantum = ONE.scaleb(-max(0, funding.size_decimals))

        def choose(
            selected_action: HedgeAction,
            selected_reason: str,
            *,
            delta: Decimal = ZERO,
            is_reduce_only: bool = False,
        ) -> None:
            nonlocal action, reason, order_delta, reduce_only
            action = selected_action
            reason = selected_reason
            order_delta = delta
            reduce_only = is_reduce_only

        addresses_equal = (
            primary.account_address.lower() == hedge.account_address.lower()
        )
        if addresses_equal:
            choose(HedgeAction.RECOVERY_REQUIRED, "primary_and_hedge_accounts_are_equal")
        elif active_cycle is not None and (
            active_cycle.symbol != snapshot.symbol
            or active_cycle.primary_account_address.lower()
            != primary.account_address.lower()
            or active_cycle.hedge_account_address.lower()
            != hedge.account_address.lower()
            or active_cycle.network != self.config.network
            or active_cycle.hedge_account_kind != self.config.hedge_account_kind
        ):
            choose(HedgeAction.RECOVERY_REQUIRED, "active_cycle_identity_mismatch")
        elif current_hedge_size != 0:
            if active_cycle is None:
                choose(HedgeAction.RECOVERY_REQUIRED, "untracked_hedge_position")
            elif primary.size == 0:
                choose(
                    HedgeAction.CLOSE_HEDGE,
                    "primary_position_is_flat",
                    delta=-current_hedge_size,
                    is_reduce_only=True,
                )
            elif _sign(current_hedge_size) != -_sign(primary.size):
                choose(
                    HedgeAction.CLOSE_HEDGE,
                    "hedge_direction_no_longer_opposes_primary",
                    delta=-current_hedge_size,
                    is_reduce_only=True,
                )
            elif not funding.market_data_available:
                choose(
                    HedgeAction.CLOSE_HEDGE,
                    "market_data_unavailable_for_tracked_hedge",
                    delta=-current_hedge_size,
                    is_reduce_only=True,
                )
            elif primary_debit <= 0:
                choose(
                    HedgeAction.CLOSE_HEDGE,
                    "funding_direction_reversed_or_primary_no_longer_pays",
                    delta=-current_hedge_size,
                    is_reduce_only=True,
                )
            elif abs(current_hedge_size) > abs(target_hedge_size) + quantum / 2:
                choose(
                    HedgeAction.ADJUST_HEDGE,
                    "hedge_exceeds_current_allowed_ratio",
                    delta=target_hedge_size - current_hedge_size,
                    is_reduce_only=True,
                )
            elif snapshot.hedge_account.funding_confirmed or (
                active_cycle.funding_confirmed_at is not None
            ):
                choose(
                    HedgeAction.CLOSE_HEDGE,
                    "target_funding_record_confirmed",
                    delta=-current_hedge_size,
                    is_reduce_only=True,
                )
            elif evaluated_at >= settlement_at + timedelta(
                seconds=self.config.funding_confirmation_timeout_seconds
            ):
                choose(
                    HedgeAction.CLOSE_HEDGE,
                    "funding_confirmation_timeout",
                    delta=-current_hedge_size,
                    is_reduce_only=True,
                )
            elif snapshot.hedge_account.unknown_open_orders:
                choose(
                    HedgeAction.CLOSE_HEDGE,
                    "unknown_order_requires_tracked_hedge_shutdown",
                    delta=-current_hedge_size,
                    is_reduce_only=True,
                )
            else:
                choose(
                    HedgeAction.WAIT_FOR_FUNDING,
                    (
                        "awaiting_user_funding_record"
                        if evaluated_at >= settlement_at
                        else "hedge_open_before_settlement"
                    ),
                )
        elif snapshot.hedge_account.unknown_open_orders:
            choose(HedgeAction.RECOVERY_REQUIRED, "unknown_order_on_hedge_account")
        elif active_cycle is not None:
            if active_cycle.status in {
                HedgeCycleStatus.CLOSE_SUBMITTED,
                HedgeCycleStatus.CLOSE_PARTIAL,
            }:
                choose(HedgeAction.HOLD, "hedge_is_flat_after_close_submission")
            elif active_cycle.status in {
                HedgeCycleStatus.OPEN_SUBMITTED,
                HedgeCycleStatus.OPEN_PARTIAL,
            }:
                choose(HedgeAction.HOLD, "awaiting_open_order_reconciliation")
            elif active_cycle.status in {
                HedgeCycleStatus.HEDGED,
                HedgeCycleStatus.AWAITING_FUNDING,
            }:
                choose(HedgeAction.RECOVERY_REQUIRED, "tracked_hedge_position_disappeared")
            else:
                choose(HedgeAction.HOLD, "active_cycle_has_no_hedge_position")
        elif primary.size == 0:
            reason = "primary_position_is_flat"
        elif not funding.market_data_available:
            reason = "market_data_unavailable"
        elif abs(primary.size) * funding.oracle_price < self.config.min_position_notional_usd:
            reason = "primary_position_below_minimum_notional"
        elif data_age_seconds < -Decimal(self.config.max_clock_skew_seconds):
            reason = "funding_data_timestamp_is_in_the_future"
        elif data_age_seconds > Decimal(self.config.max_funding_data_age_seconds):
            reason = "funding_data_is_stale"
        elif primary_debit <= 0:
            reason = "primary_receives_or_owes_no_funding"
        elif seconds_to_funding > Decimal(self.config.hedge_open_lead_seconds):
            reason = "outside_hedge_open_window"
        elif seconds_to_funding <= Decimal(self.config.latest_open_cutoff_seconds):
            reason = "too_late_to_open_hedge_safely"
        elif snapshot.hedge_account.account_kind == "subaccount" and (
            self.config.require_verified_subaccount
            and not snapshot.hedge_account.ownership_verified
        ):
            reason = "hedge_subaccount_ownership_not_verified"
        elif not snapshot.hedge_account.margin_available_verified:
            reason = "hedge_available_margin_not_verified"
        elif ":" in snapshot.symbol and not (
            snapshot.hedge_account.hip3_fee_formula_verified
            or self.config.hip3_extra_fee_rate_known
        ):
            reason = "hip3_extra_fee_rate_not_confirmed"
        elif target_qty == 0:
            reason = "insufficient_available_margin_for_any_hedge"
        elif hedge_execution_notional < self.config.min_hedge_notional_usd:
            reason = "hedge_order_below_minimum_notional"
        elif required_margin > max(
            ZERO,
            snapshot.hedge_account.withdrawable_usd
            - self.config.min_available_margin_usd,
        ):
            reason = "insufficient_available_margin"
        elif net_saving < self.config.min_expected_saving_usd:
            reason = "expected_funding_does_not_cover_hedge_round_trip_cost"
        else:
            choose(
                HedgeAction.OPEN_HEDGE,
                "temporary_independent_account_hedge_is_cost_effective",
                delta=target_hedge_size,
                is_reduce_only=False,
            )

        return HedgeDecision(
            snapshot=snapshot,
            action=action,
            reason=reason,
            evaluated_at=evaluated_at,
            seconds_to_funding=seconds_to_funding,
            target_hedge_size=target_hedge_size,
            current_hedge_size=current_hedge_size,
            order_size_delta=order_delta,
            reduce_only=reduce_only,
            estimated_primary_funding_debit_usd=primary_debit,
            estimated_hedge_funding_credit_usd=hedge_credit,
            estimated_round_trip_cost_usd=round_trip_cost,
            estimated_net_saving_usd=net_saving,
            estimated_required_margin_usd=required_margin,
            cycle_key=cycle_key,
        )


# A narrow alias keeps imports readable while making the strategy change
# explicit in documentation and output.
FundingAvoidancePolicy = FundingHedgePolicy
