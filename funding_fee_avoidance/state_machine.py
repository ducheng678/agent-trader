from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Callable, Optional

from .config import StrategyConfig
from .executor import ExecutionResult, HyperliquidHedgeExecutor
from .models import (
    HedgeAction,
    HedgeCycleState,
    HedgeCycleStatus,
    HedgeDecision,
    HedgeSnapshot,
)
from .policy import FundingHedgePolicy
from .state_store import CycleStateStore, CycleTransaction


UTC = timezone.utc
TERMINAL_ORDER_STATUSES = {
    "filled",
    "canceled",
    "cancelled",
    "rejected",
    "margincanceled",
}


def stable_cloid(cycle_key: str, phase: str) -> str:
    digest = hashlib.blake2b(
        f"funding-hedge|{cycle_key}|{phase}".encode("utf-8"), digest_size=16
    ).hexdigest()
    return "0x" + digest


@dataclass(frozen=True)
class CoordinatorResult:
    decision: HedgeDecision
    cycle: Optional[HedgeCycleState]
    execution: Optional[ExecutionResult] = None
    execution_error: str = ""

    def execution_dict(self):
        if self.execution is None:
            return None
        return {
            "accepted": self.execution.accepted,
            "requested_size": str(self.execution.requested_size),
            "reported_filled_size": str(self.execution.reported_filled_size),
            "error": self.execution.error,
        }


class HedgeCoordinator:
    """Persistent single-action state machine for one polling iteration."""

    def __init__(
        self,
        config: StrategyConfig,
        store: CycleStateStore,
        executor: Optional[HyperliquidHedgeExecutor] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
        self.config = config
        self.store = store
        self.executor = executor
        self.policy = FundingHedgePolicy(config)
        self.clock = clock

    def _same_cycle_identity(self, cycle: HedgeCycleState, snapshot: HedgeSnapshot) -> bool:
        return (
            cycle.symbol == snapshot.symbol
            and cycle.primary_account_address.lower()
            == snapshot.primary_position.account_address.lower()
            and cycle.hedge_account_address.lower()
            == snapshot.hedge_account.position.account_address.lower()
            and cycle.network == self.config.network
            and cycle.hedge_account_kind == self.config.hedge_account_kind
        )

    def _reconcile(
        self,
        cycle: Optional[HedgeCycleState],
        snapshot: HedgeSnapshot,
        now: datetime,
    ) -> Optional[HedgeCycleState]:
        if cycle is None or not cycle.active:
            return cycle
        if not self._same_cycle_identity(cycle, snapshot):
            return replace(
                cycle,
                status=HedgeCycleStatus.RECOVERY_REQUIRED,
                updated_at=now,
                last_error="cycle identity differs from exchange snapshot",
            )

        actual = snapshot.hedge_account.position.size
        changes = {
            "actual_hedge_size": actual,
            "updated_at": now,
        }
        if snapshot.hedge_account.funding_confirmed:
            changes.update(
                {
                    "funding_confirmed_at": snapshot.hedge_account.funding_record_time
                    or now,
                    "funding_delta_usd": snapshot.hedge_account.funding_delta_usd,
                }
            )

        status = cycle.status
        last_error = cycle.last_error
        pending_adjust_cloid = cycle.pending_adjust_cloid
        pre_adjust_size = cycle.pre_adjust_hedge_size
        adjust_submitted_at = cycle.adjust_submitted_at

        if pending_adjust_cloid:
            adjust_status = snapshot.hedge_account.adjust_order_status.lower()
            if abs(actual) < abs(pre_adjust_size):
                pending_adjust_cloid = ""
                pre_adjust_size = Decimal("0")
                adjust_submitted_at = None
                status = HedgeCycleStatus.HEDGED
            elif adjust_status in {"canceled", "cancelled", "rejected", "margincanceled"}:
                pending_adjust_cloid = ""
                pre_adjust_size = Decimal("0")
                adjust_submitted_at = None
                status = HedgeCycleStatus.HEDGED
                last_error = f"adjustment order ended with {adjust_status}"
            elif adjust_status == "filled" and actual == pre_adjust_size:
                status = HedgeCycleStatus.RECOVERY_REQUIRED
                last_error = "adjustment reports filled but hedge position did not change"
            elif (
                cycle.adjust_submitted_at is not None
                and now - cycle.adjust_submitted_at
                >= timedelta(seconds=self.config.ambiguous_order_retry_seconds)
                and actual == pre_adjust_size
            ):
                pending_adjust_cloid = ""
                pre_adjust_size = Decimal("0")
                adjust_submitted_at = None
                status = HedgeCycleStatus.HEDGED
                last_error = "ambiguous adjustment timed out after position reconciliation"

        if status in {HedgeCycleStatus.OPEN_SUBMITTED, HedgeCycleStatus.OPEN_PARTIAL}:
            open_status = snapshot.hedge_account.open_order_status.lower()
            if actual != 0:
                status = (
                    HedgeCycleStatus.OPEN_PARTIAL
                    if abs(actual) < abs(cycle.target_hedge_size)
                    else HedgeCycleStatus.HEDGED
                )
            elif open_status in {"canceled", "cancelled", "rejected", "margincanceled"}:
                status = HedgeCycleStatus.ABORTED
                last_error = f"open order ended with {open_status} and no hedge position"
            elif open_status == "filled":
                status = HedgeCycleStatus.RECOVERY_REQUIRED
                last_error = "open order reports filled but hedge position is flat"
            elif (
                cycle.open_submitted_at is not None
                and now - cycle.open_submitted_at
                >= timedelta(
                    seconds=self.config.open_reconciliation_timeout_seconds
                )
            ):
                status = HedgeCycleStatus.ABORTED
                last_error = "ambiguous IOC open timed out with hedge position still flat"

        if status in {HedgeCycleStatus.CLOSE_SUBMITTED, HedgeCycleStatus.CLOSE_PARTIAL}:
            close_status = snapshot.hedge_account.close_order_status.lower()
            if actual == 0:
                status = HedgeCycleStatus.COMPLETED
            elif close_status in TERMINAL_ORDER_STATUSES:
                status = HedgeCycleStatus.CLOSE_PARTIAL
            elif (
                cycle.close_submitted_at is not None
                and now - cycle.close_submitted_at
                >= timedelta(seconds=self.config.ambiguous_order_retry_seconds)
            ):
                status = HedgeCycleStatus.CLOSE_PARTIAL
                last_error = "ambiguous close timed out after position reconciliation"

        if status in {HedgeCycleStatus.HEDGED, HedgeCycleStatus.OPEN_PARTIAL}:
            if now >= cycle.settlement_at:
                status = HedgeCycleStatus.AWAITING_FUNDING

        return replace(
            cycle,
            status=status,
            last_error=last_error,
            pending_adjust_cloid=pending_adjust_cloid,
            pre_adjust_hedge_size=pre_adjust_size,
            adjust_submitted_at=adjust_submitted_at,
            **changes,
        )

    @staticmethod
    def _put_if_changed(
        transaction: CycleTransaction,
        before: Optional[HedgeCycleState],
        after: Optional[HedgeCycleState],
    ) -> None:
        if after is not None and after != before:
            transaction.put(after)

    def _new_cycle(
        self, decision: HedgeDecision, now: datetime
    ) -> HedgeCycleState:
        snapshot = decision.snapshot
        return HedgeCycleState(
            cycle_key=decision.cycle_key,
            status=HedgeCycleStatus.ARMED,
            symbol=snapshot.symbol,
            primary_account_address=snapshot.primary_position.account_address,
            hedge_account_address=snapshot.hedge_account.position.account_address,
            settlement_at=snapshot.funding.next_funding_at,
            target_hedge_size=decision.target_hedge_size,
            actual_hedge_size=snapshot.hedge_account.position.size,
            open_cloid=stable_cloid(decision.cycle_key, "open:1"),
            close_cloid="",
            created_at=now,
            updated_at=now,
            network=self.config.network,
            hedge_account_kind=self.config.hedge_account_kind,
        )

    def process(
        self,
        snapshot: HedgeSnapshot,
        now: datetime,
        *,
        execute: bool = False,
    ) -> CoordinatorResult:
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("now must include a timezone")
        evaluated_at = (
            self.clock().astimezone(UTC)
            if execute and self.clock is not None
            else now.astimezone(UTC)
        )
        with self.store.transaction() as transaction:
            stored = transaction.get(snapshot.symbol)
            reconciled = self._reconcile(stored, snapshot, evaluated_at)
            active_cycle = reconciled if reconciled is not None and reconciled.active else None
            decision = self.policy.evaluate(snapshot, evaluated_at, active_cycle)
            if (
                reconciled is not None
                and reconciled.status.terminal
                and decision.action is HedgeAction.OPEN_HEDGE
                and decision.cycle_key == reconciled.cycle_key
            ):
                decision = replace(
                    decision,
                    action=HedgeAction.HOLD,
                    reason="terminal_cycle_already_exists_for_this_settlement",
                    order_size_delta=Decimal("0"),
                    reduce_only=False,
                )

            if not execute:
                return CoordinatorResult(decision=decision, cycle=reconciled)
            if not self.config.execution_enabled:
                raise RuntimeError(
                    "--execute also requires FUNDING_HEDGE_EXECUTE=true"
                )
            if self.executor is None:
                raise RuntimeError("hedge executor is not configured")

            self._put_if_changed(transaction, stored, reconciled)
            transaction.flush()
            execution: Optional[ExecutionResult] = None
            execution_error = ""

            if decision.action is HedgeAction.OPEN_HEDGE:
                if active_cycle is not None:
                    return CoordinatorResult(decision=decision, cycle=active_cycle)
                cycle = self._new_cycle(decision, evaluated_at)
                transaction.put(cycle)
                transaction.flush()
                cycle = replace(
                    cycle,
                    status=HedgeCycleStatus.OPEN_SUBMITTED,
                    updated_at=evaluated_at,
                    open_submitted_at=evaluated_at,
                )
                transaction.put(cycle)
                transaction.flush()  # intent exists before the external action
                try:
                    execution = self.executor.open_hedge(
                        snapshot.symbol,
                        decision.order_size_delta,
                        cycle.open_cloid,
                        submit_deadline=cycle.settlement_at
                        - timedelta(seconds=self.config.latest_open_cutoff_seconds),
                    )
                    if not execution.accepted:
                        cycle = replace(
                            cycle,
                            status=HedgeCycleStatus.ABORTED,
                            updated_at=evaluated_at,
                            last_error=execution.error or "open order rejected",
                        )
                except Exception as exc:
                    # An I/O exception is ambiguous.  Keep OPEN_SUBMITTED and
                    # reconcile CLOID + position before any future action.
                    execution_error = str(exc)
                    cycle = replace(
                        cycle,
                        updated_at=evaluated_at,
                        last_error=f"ambiguous open submission: {exc}",
                    )
                transaction.put(cycle)
                return CoordinatorResult(decision, cycle, execution, execution_error)

            cycle = active_cycle
            if decision.action is HedgeAction.WAIT_FOR_FUNDING and cycle is not None:
                status = (
                    HedgeCycleStatus.AWAITING_FUNDING
                    if evaluated_at >= cycle.settlement_at
                    else (
                        HedgeCycleStatus.OPEN_PARTIAL
                        if abs(snapshot.hedge_account.position.size)
                        < abs(cycle.target_hedge_size)
                        else HedgeCycleStatus.HEDGED
                    )
                )
                cycle = replace(
                    cycle,
                    status=status,
                    actual_hedge_size=snapshot.hedge_account.position.size,
                    updated_at=evaluated_at,
                    funding_confirmed_at=(
                        snapshot.hedge_account.funding_record_time
                        if snapshot.hedge_account.funding_confirmed
                        else cycle.funding_confirmed_at
                    ),
                    funding_delta_usd=(
                        snapshot.hedge_account.funding_delta_usd
                        if snapshot.hedge_account.funding_confirmed
                        else cycle.funding_delta_usd
                    ),
                )
                transaction.put(cycle)
                return CoordinatorResult(decision=decision, cycle=cycle)

            if decision.action is HedgeAction.ADJUST_HEDGE and cycle is not None:
                if cycle.pending_adjust_cloid:
                    return CoordinatorResult(decision=decision, cycle=cycle)
                count = cycle.adjustment_count + 1
                cloid = stable_cloid(cycle.cycle_key, f"adjust:{count}")
                cycle = replace(
                    cycle,
                    adjustment_count=count,
                    pending_adjust_cloid=cloid,
                    pre_adjust_hedge_size=snapshot.hedge_account.position.size,
                    updated_at=evaluated_at,
                    adjust_submitted_at=evaluated_at,
                )
                transaction.put(cycle)
                transaction.flush()
                try:
                    execution = self.executor.close_hedge(
                        snapshot.symbol,
                        snapshot.hedge_account.position.size,
                        abs(decision.order_size_delta),
                        cloid,
                    )
                    if not execution.accepted:
                        cycle = replace(
                            cycle,
                            pending_adjust_cloid="",
                            pre_adjust_hedge_size=Decimal("0"),
                            adjust_submitted_at=None,
                            last_error=execution.error or "adjustment order rejected",
                        )
                except Exception as exc:
                    execution_error = str(exc)
                    cycle = replace(
                        cycle,
                        last_error=f"ambiguous adjustment submission: {exc}",
                    )
                transaction.put(cycle)
                return CoordinatorResult(decision, cycle, execution, execution_error)

            if decision.action is HedgeAction.CLOSE_HEDGE and cycle is not None:
                close_status = snapshot.hedge_account.close_order_status.lower()
                if (
                    cycle.status is HedgeCycleStatus.CLOSE_SUBMITTED
                    and close_status not in TERMINAL_ORDER_STATUSES
                ):
                    return CoordinatorResult(decision=decision, cycle=cycle)
                count = cycle.close_attempt_count + 1
                cloid = stable_cloid(cycle.cycle_key, f"close:{count}")
                cycle = replace(
                    cycle,
                    status=HedgeCycleStatus.CLOSE_SUBMITTED,
                    close_attempt_count=count,
                    close_cloid=cloid,
                    actual_hedge_size=snapshot.hedge_account.position.size,
                    updated_at=evaluated_at,
                    close_submitted_at=evaluated_at,
                )
                transaction.put(cycle)
                transaction.flush()
                try:
                    execution = self.executor.close_hedge(
                        snapshot.symbol,
                        snapshot.hedge_account.position.size,
                        abs(snapshot.hedge_account.position.size),
                        cloid,
                    )
                    if not execution.accepted:
                        cycle = replace(
                            cycle,
                            status=HedgeCycleStatus.CLOSE_PARTIAL,
                            last_error=execution.error or "close order rejected",
                        )
                except Exception as exc:
                    execution_error = str(exc)
                    cycle = replace(
                        cycle,
                        last_error=f"ambiguous close submission: {exc}",
                    )
                transaction.put(cycle)
                return CoordinatorResult(decision, cycle, execution, execution_error)

            if decision.action is HedgeAction.RECOVERY_REQUIRED and cycle is not None:
                cycle = replace(
                    cycle,
                    status=HedgeCycleStatus.RECOVERY_REQUIRED,
                    updated_at=evaluated_at,
                    last_error=decision.reason,
                )
                transaction.put(cycle)
                return CoordinatorResult(decision=decision, cycle=cycle)

            return CoordinatorResult(decision=decision, cycle=reconciled)
