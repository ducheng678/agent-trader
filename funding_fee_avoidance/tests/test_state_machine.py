from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from funding_fee_avoidance.config import StrategyConfig
from funding_fee_avoidance.executor import ExecutionResult
from funding_fee_avoidance.models import (
    AccountPositionSnapshot,
    FundingObservation,
    HedgeAccountSnapshot,
    HedgeAction,
    HedgeCycleStatus,
    HedgeSnapshot,
)
from funding_fee_avoidance.state_machine import HedgeCoordinator
from funding_fee_avoidance.state_store import CycleStateStore


UTC = timezone.utc
NOW = datetime(2026, 7, 10, 12, 58, 30, tzinfo=UTC)
PRIMARY = "0x1111111111111111111111111111111111111111"
HEDGE = "0x2222222222222222222222222222222222222222"


class FakeExecutor:
    def __init__(self):
        self.calls = []

    def open_hedge(self, symbol, signed_size, cloid, *, submit_deadline):
        self.calls.append(("open", symbol, signed_size, cloid, submit_deadline))
        return ExecutionResult(True, abs(signed_size), abs(signed_size), {"status": "ok"})

    def close_hedge(self, symbol, current_signed_size, size, cloid):
        self.calls.append(("close", symbol, size, cloid, current_signed_size))
        return ExecutionResult(True, size, size, {"status": "ok"})


def config(path, **overrides):
    values = {
        "symbols": ("xyz:SKHX",),
        "primary_account_address": PRIMARY,
        "hedge_account_address": HEDGE,
        "hip3_extra_fee_rate_known": True,
        "execution_enabled": True,
        "min_available_margin_usd": Decimal("0"),
        "margin_safety_multiplier": Decimal("1"),
        "slippage_rate_per_order": Decimal("0"),
        "risk_buffer_rate": Decimal("0"),
        "cost_safety_multiplier": Decimal("1"),
        "state_path": path,
    }
    values.update(overrides)
    return StrategyConfig(**values)


def snapshot(
    *,
    primary_size=Decimal("100"),
    hedge_size=Decimal("0"),
    confirmed=False,
    open_status="",
    close_status="",
):
    return HedgeSnapshot(
        primary_position=AccountPositionSnapshot(
            PRIMARY, "xyz:SKHX", primary_size
        ),
        hedge_account=HedgeAccountSnapshot(
            position=AccountPositionSnapshot(HEDGE, "xyz:SKHX", hedge_size),
            withdrawable_usd=Decimal("50000"),
            taker_fee_rate=Decimal("0.0001"),
            fee_rate_source="test",
            ownership_verified=True,
            funding_confirmed=confirmed,
            funding_record_time=(NOW + timedelta(seconds=90) if confirmed else None),
            open_order_status=open_status,
            close_order_status=close_status,
        ),
        funding=FundingObservation(
            "xyz:SKHX",
            Decimal("100"),
            Decimal("100"),
            Decimal("0.002"),
            NOW,
            NOW + timedelta(seconds=90),
            3,
        ),
    )


def test_dry_run_never_calls_executor_or_writes_cycle(tmp_path):
    state_path = tmp_path / "state.json"
    fake = FakeExecutor()
    coordinator = HedgeCoordinator(
        config(state_path), CycleStateStore(state_path), fake
    )

    result = coordinator.process(snapshot(), NOW, execute=False)

    assert result.decision.action is HedgeAction.OPEN_HEDGE
    assert fake.calls == []
    assert not state_path.exists()


def test_open_intent_is_persisted_and_restart_does_not_duplicate(tmp_path):
    state_path = tmp_path / "state.json"
    cfg = config(state_path)
    store = CycleStateStore(state_path)
    fake = FakeExecutor()
    first = HedgeCoordinator(cfg, store, fake).process(snapshot(), NOW, execute=True)

    assert len(fake.calls) == 1
    assert first.cycle.status is HedgeCycleStatus.OPEN_SUBMITTED
    saved = store.load_all()["xyz:SKHX"]
    assert saved.open_cloid == fake.calls[0][3]

    second = HedgeCoordinator(cfg, store, fake).process(
        snapshot(open_status="unknown"), NOW + timedelta(seconds=2), execute=True
    )

    assert second.decision.action is HedgeAction.HOLD
    assert len(fake.calls) == 1


def test_partial_actual_hedge_is_accepted_without_blind_topup(tmp_path):
    state_path = tmp_path / "state.json"
    cfg = config(state_path)
    store = CycleStateStore(state_path)
    fake = FakeExecutor()
    HedgeCoordinator(cfg, store, fake).process(snapshot(), NOW, execute=True)

    result = HedgeCoordinator(cfg, store, fake).process(
        snapshot(hedge_size=Decimal("-40"), open_status="filled"),
        NOW + timedelta(seconds=2),
        execute=True,
    )

    assert result.decision.action is HedgeAction.WAIT_FOR_FUNDING
    assert result.cycle.status is HedgeCycleStatus.OPEN_PARTIAL
    assert len(fake.calls) == 1


def test_user_funding_confirmation_submits_reduce_only_close_for_actual_size(tmp_path):
    state_path = tmp_path / "state.json"
    cfg = config(state_path)
    store = CycleStateStore(state_path)
    fake = FakeExecutor()
    HedgeCoordinator(cfg, store, fake).process(snapshot(), NOW, execute=True)
    HedgeCoordinator(cfg, store, fake).process(
        snapshot(hedge_size=Decimal("-40"), open_status="filled"),
        NOW + timedelta(seconds=2),
        execute=True,
    )

    result = HedgeCoordinator(cfg, store, fake).process(
        snapshot(hedge_size=Decimal("-40"), confirmed=True),
        NOW + timedelta(seconds=95),
        execute=True,
    )

    assert result.decision.action is HedgeAction.CLOSE_HEDGE
    assert fake.calls[-1][0:3] == ("close", "xyz:SKHX", Decimal("40"))
    assert result.cycle.status is HedgeCycleStatus.CLOSE_SUBMITTED


def test_primary_shrink_uses_reduce_only_adjustment_not_new_open(tmp_path):
    state_path = tmp_path / "state.json"
    cfg = config(state_path)
    store = CycleStateStore(state_path)
    fake = FakeExecutor()
    HedgeCoordinator(cfg, store, fake).process(snapshot(), NOW, execute=True)
    HedgeCoordinator(cfg, store, fake).process(
        snapshot(hedge_size=Decimal("-100"), open_status="filled"),
        NOW + timedelta(seconds=1),
        execute=True,
    )

    result = HedgeCoordinator(cfg, store, fake).process(
        snapshot(primary_size=Decimal("50"), hedge_size=Decimal("-100")),
        NOW + timedelta(seconds=2),
        execute=True,
    )

    assert result.decision.action is HedgeAction.ADJUST_HEDGE
    assert fake.calls[-1][0:3] == ("close", "xyz:SKHX", Decimal("50.000"))
    assert sum(1 for call in fake.calls if call[0] == "open") == 1


def test_flat_after_close_marks_cycle_completed_from_exchange_truth(tmp_path):
    state_path = tmp_path / "state.json"
    cfg = config(state_path)
    store = CycleStateStore(state_path)
    fake = FakeExecutor()
    HedgeCoordinator(cfg, store, fake).process(snapshot(), NOW, execute=True)
    HedgeCoordinator(cfg, store, fake).process(
        snapshot(hedge_size=Decimal("-100")), NOW + timedelta(seconds=1), execute=True
    )
    HedgeCoordinator(cfg, store, fake).process(
        snapshot(hedge_size=Decimal("-100"), confirmed=True),
        NOW + timedelta(seconds=95),
        execute=True,
    )

    result = HedgeCoordinator(cfg, store, fake).process(
        snapshot(hedge_size=Decimal("0"), close_status="filled"),
        NOW + timedelta(seconds=96),
        execute=True,
    )

    assert result.cycle.status is HedgeCycleStatus.COMPLETED
    assert len(fake.calls) == 2


def test_fresh_clock_at_submission_prevents_late_open(tmp_path):
    state_path = tmp_path / "state.json"
    cfg = config(state_path)
    fake = FakeExecutor()
    coordinator = HedgeCoordinator(
        cfg,
        CycleStateStore(state_path),
        fake,
        clock=lambda: NOW + timedelta(seconds=91),
    )

    result = coordinator.process(snapshot(), NOW, execute=True)

    assert result.decision.action is HedgeAction.HOLD
    assert result.decision.reason in {
        "funding_data_is_stale",
        "too_late_to_open_hedge_safely",
    }
    assert fake.calls == []


def test_ambiguous_ioc_open_times_out_without_reopening_same_settlement(tmp_path):
    state_path = tmp_path / "state.json"
    cfg = config(state_path, open_reconciliation_timeout_seconds=30)
    store = CycleStateStore(state_path)
    fake = FakeExecutor()
    HedgeCoordinator(cfg, store, fake).process(snapshot(), NOW, execute=True)

    result = HedgeCoordinator(cfg, store, fake).process(
        replace(
            snapshot(open_status="unknown"),
            funding=replace(snapshot().funding, observed_at=NOW + timedelta(seconds=31)),
        ),
        NOW + timedelta(seconds=31),
        execute=True,
    )

    assert result.decision.action is HedgeAction.HOLD
    assert result.decision.reason == "terminal_cycle_already_exists_for_this_settlement"
    assert result.cycle.status is HedgeCycleStatus.ABORTED
    assert len(fake.calls) == 1


def test_ambiguous_close_retries_with_new_cloid_after_actual_position_check(tmp_path):
    state_path = tmp_path / "state.json"
    cfg = config(state_path, ambiguous_order_retry_seconds=15)
    store = CycleStateStore(state_path)
    fake = FakeExecutor()
    HedgeCoordinator(cfg, store, fake).process(snapshot(), NOW, execute=True)
    HedgeCoordinator(cfg, store, fake).process(
        snapshot(hedge_size=Decimal("-100")),
        NOW + timedelta(seconds=1),
        execute=True,
    )
    first_close = HedgeCoordinator(cfg, store, fake).process(
        snapshot(hedge_size=Decimal("-100"), confirmed=True),
        NOW + timedelta(seconds=95),
        execute=True,
    )
    first_cloid = first_close.cycle.close_cloid

    retried = HedgeCoordinator(cfg, store, fake).process(
        snapshot(
            hedge_size=Decimal("-100"),
            confirmed=True,
            close_status="unknown",
        ),
        NOW + timedelta(seconds=111),
        execute=True,
    )

    close_calls = [call for call in fake.calls if call[0] == "close"]
    assert len(close_calls) == 2
    assert retried.cycle.close_attempt_count == 2
    assert retried.cycle.close_cloid != first_cloid
