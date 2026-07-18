from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .config import StrategyConfig
from .executor import HyperliquidHedgeExecutor
from .hyperliquid_adapter import HyperliquidSnapshotAdapter
from .models import HedgeDecision, HedgeSnapshot, utc_text
from .policy import FundingHedgePolicy
from .state_machine import CoordinatorResult, HedgeCoordinator
from .state_store import CycleStateStore


UTC = timezone.utc


class FundingHedgeService:
    """Read both accounts, evaluate policy, and optionally execute one safe step."""

    def __init__(
        self,
        config: StrategyConfig,
        adapter: Optional[HyperliquidSnapshotAdapter] = None,
        store: Optional[CycleStateStore] = None,
        executor: Optional[HyperliquidHedgeExecutor] = None,
    ) -> None:
        self.config = config
        self.adapter = adapter or HyperliquidSnapshotAdapter(config)
        self.store = store or CycleStateStore(config.state_path)
        self.executor = executor

    @staticmethod
    def evaluate_snapshots(
        config: StrategyConfig,
        snapshots: Sequence[HedgeSnapshot],
        now: datetime,
        errors: Iterable[str] = (),
    ) -> Dict[str, Any]:
        policy = FundingHedgePolicy(config)
        decisions = [policy.evaluate(snapshot, now) for snapshot in snapshots]
        return FundingHedgeService._report(
            decisions=decisions,
            results=(),
            evaluated_at=now,
            errors=list(errors),
            execute=False,
        )

    # Singular name retained for the offline CLI/API used by the first draft.
    evaluate_snapshot = evaluate_snapshots

    def run_once(
        self,
        now: Optional[datetime] = None,
        *,
        execute: bool = False,
    ) -> Dict[str, Any]:
        evaluated_at = (now or datetime.now(tz=UTC)).astimezone(UTC)
        if execute and not self.config.execution_enabled:
            raise RuntimeError("--execute also requires FUNDING_HEDGE_EXECUTE=true")
        cycles = self.store.load_all()
        for cycle in cycles.values():
            if not cycle.active:
                continue
            if (
                cycle.primary_account_address.lower()
                != self.config.primary_account_address.lower()
                or cycle.hedge_account_address.lower()
                != self.config.hedge_account_address.lower()
                or cycle.network != self.config.network
                or cycle.hedge_account_kind != self.config.hedge_account_kind
            ):
                raise RuntimeError(
                    f"active cycle {cycle.cycle_key} does not match current "
                    "network/account configuration; restore the original configuration "
                    "before managing the tracked hedge"
                )
        snapshots, errors = self.adapter.load_snapshots(
            observed_at=evaluated_at, cycles=cycles
        )
        if execute and self.executor is None:
            self.executor = HyperliquidHedgeExecutor(
                self.config, self.adapter.reader
            )
        coordinator = HedgeCoordinator(
            self.config,
            self.store,
            self.executor if execute else None,
            clock=lambda: datetime.now(tz=UTC),
        )
        results = []
        open_attempted = False
        for snapshot in snapshots:
            preview = coordinator.process(snapshot, evaluated_at, execute=False)
            if (
                execute
                and preview.decision.action.value == "open_hedge"
                and open_attempted
            ):
                errors.append(
                    f"{snapshot.symbol}: new hedge deferred; at most one new hedge "
                    "is opened per polling iteration so margin can be refreshed"
                )
                results.append(preview)
                continue
            selected_snapshot = snapshot
            if execute and preview.decision.action.value == "open_hedge":
                # Re-read both positions, current funding, fees, margin and
                # orders immediately before a position-changing open. The
                # coordinator then performs a second wall-clock cutoff check.
                refreshed_at = datetime.now(tz=UTC)
                refreshed_cycles = self.store.load_all()
                refreshed, refresh_errors = self.adapter.load_snapshots(
                    observed_at=refreshed_at, cycles=refreshed_cycles
                )
                errors.extend(refresh_errors)
                match = next(
                    (item for item in refreshed if item.symbol == snapshot.symbol),
                    None,
                )
                if match is None:
                    errors.append(
                        f"{snapshot.symbol}: final pre-open refresh failed; no order sent"
                    )
                    results.append(preview)
                    continue
                selected_snapshot = match
            result = (
                coordinator.process(selected_snapshot, evaluated_at, execute=True)
                if execute
                else preview
            )
            if (
                execute
                and result.decision.action.value == "open_hedge"
                and (result.execution is not None or result.execution_error)
            ):
                open_attempted = True
            results.append(result)
        return self._report(
            decisions=[result.decision for result in results],
            results=results,
            evaluated_at=evaluated_at,
            errors=errors,
            execute=execute,
        )

    @staticmethod
    def _report(
        *,
        decisions: Sequence[HedgeDecision],
        results: Sequence[CoordinatorResult],
        evaluated_at: datetime,
        errors: List[str],
        execute: bool,
    ) -> Dict[str, Any]:
        action_counts: Dict[str, int] = {
            "open_hedge": 0,
            "wait_for_funding": 0,
            "close_hedge": 0,
            "adjust_hedge": 0,
            "recovery_required": 0,
            "hold": 0,
        }
        for item in decisions:
            action_counts[item.action.value] += 1
        result_by_key = {item.decision.cycle_key: item for item in results}
        rendered = []
        for decision in decisions:
            payload = decision.to_dict()
            result = result_by_key.get(decision.cycle_key)
            payload["cycle"] = (
                result.cycle.to_dict() if result is not None and result.cycle else None
            )
            payload["execution"] = result.execution_dict() if result else None
            payload["execution_error"] = result.execution_error if result else ""
            rendered.append(payload)
        return {
            "mode": "live_execute" if execute else "report_only",
            "orders_enabled": execute,
            "primary_account_is_read_only": True,
            "order_route": "hedge_account_only",
            "evaluated_at": utc_text(evaluated_at),
            "summary": {
                "symbols_evaluated": len(decisions),
                **action_counts,
                "errors": len(errors),
            },
            "errors": errors,
            "decisions": rendered,
        }


FundingAvoidanceService = FundingHedgeService
