from __future__ import annotations

import math
import threading
import time
import uuid
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Callable

from market_agent.openai_usage import (
    PricingBand,
    UsageTokens,
    estimate_workflow_usage_cost,
    get_openai_web_search_tool_price_usd_per_1k_decimal,
)
from market_agent.workflow_contracts import WorkflowMode
from market_agent.workflow_model_routing import AgentExecutionPolicy, policy_for


class BudgetExceededError(RuntimeError):
    pass


class BudgetOverflowError(BudgetExceededError):
    def __init__(self, message: str, settlement: BudgetSettlement) -> None:
        super().__init__(message)
        self.settlement = settlement


class ReservationOwnershipError(ValueError):
    pass


class ReservationStateError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BudgetReservation:
    reservation_id: str
    node_name: str
    model: str
    band: PricingBand
    reserved_cost: Decimal
    deadline_monotonic: float
    _ledger_nonce: object = field(repr=False, compare=False)
    _reserved_usage: UsageTokens = field(repr=False, compare=False)
    _web_search_tool_price_usd_per_1k: Decimal = field(repr=False, compare=False)


@dataclass(frozen=True, slots=True)
class BudgetSettlement:
    reservation_id: str
    charged_cost: Decimal
    timeout: bool


@dataclass(frozen=True, slots=True)
class NodeBudgetSnapshot:
    node_name: str
    remaining_cost: Decimal
    reserved_cost: Decimal
    settled_cost: Decimal
    remaining_attempts: int
    remaining_seconds: float
    exhausted: bool
    overdrawn: bool


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    mode: WorkflowMode
    remaining_cost: Decimal
    reserved_cost: Decimal
    settled_cost: Decimal
    remaining_attempts: int
    remaining_seconds: float
    deadline_monotonic: float
    nodes: tuple[NodeBudgetSnapshot, ...]
    exhausted: bool
    overdrawn: bool


@dataclass(slots=True)
class _NodeBudget:
    policy: AgentExecutionPolicy
    started_monotonic: float
    deadline_monotonic: float
    reserved_cost: Decimal = Decimal("0")
    settled_cost: Decimal = Decimal("0")
    attempts: int = 0
    tier_attempts: dict[str, int] = field(default_factory=dict)
    exhausted: bool = False
    overdrawn: bool = False


_WORKFLOW_CAPS: dict[WorkflowMode, tuple[float, int, Decimal]] = {
    WorkflowMode.ACTIVE: (300.0, 10, Decimal("0.75")),
    WorkflowMode.PASSIVE: (130.0, 10, Decimal("0.30")),
}


class WorkflowBudgetLedger:
    timeout_charge_policy = "full_reservation"

    def __init__(self, mode: WorkflowMode | str, *, clock: Callable[[], float] = time.monotonic) -> None:
        try:
            self._mode = mode if isinstance(mode, WorkflowMode) else WorkflowMode(mode)
        except (TypeError, ValueError) as exc:
            raise ValueError("unknown workflow mode") from exc
        if not callable(clock):
            raise ValueError("workflow clock must be callable")
        self._clock = clock
        self._lock = threading.RLock()
        self._nonce = object()
        self._last_monotonic = self._read_clock()
        cap_seconds, self._maximum_attempts, self._cost_cap = _WORKFLOW_CAPS[self._mode]
        self._deadline_monotonic = self._last_monotonic + cap_seconds
        self._reserved_cost = Decimal("0")
        self._settled_cost = Decimal("0")
        self._attempts = 0
        self._nodes: dict[str, _NodeBudget] = {}
        self._active: dict[str, BudgetReservation] = {}
        self._completed: set[str] = set()
        self._exhausted = False
        self._overdrawn = False

    @property
    def mode(self) -> WorkflowMode:
        return self._mode

    def _read_clock(self) -> float:
        value = self._clock()
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise ValueError("workflow clock must return finite monotonic seconds")
        return float(value)

    def _now(self) -> float:
        current = self._read_clock()
        if current < self._last_monotonic:
            raise ValueError("monotonic clock moved backwards")
        self._last_monotonic = current
        return current

    @staticmethod
    def _positive_int(value: int, field_name: str) -> int:
        if type(value) is not int or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")
        return value

    @staticmethod
    def _nonnegative_int(value: int, field_name: str) -> int:
        if type(value) is not int or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer")
        return value

    def reserve(
        self,
        *,
        node_name: str,
        model: str,
        band: PricingBand,
        usage: UsageTokens,
        maximum_tool_calls: int = 0,
        attempt_timeout_seconds: int | None = None,
    ) -> BudgetReservation:
        policy = policy_for(node_name)
        if not isinstance(usage, UsageTokens):
            raise ValueError("workflow reservations require UsageTokens")
        if model not in tuple(tier.model for tier in policy.tiers):
            raise ValueError("model is not allowed for workflow node")
        self._nonnegative_int(maximum_tool_calls, "maximum_tool_calls")
        if maximum_tool_calls > policy.maximum_tool_calls:
            raise BudgetExceededError("node tool call cap exceeded")
        if usage.output_tokens > policy.maximum_output_tokens:
            raise BudgetExceededError("node output token cap exceeded")
        timeout = policy.attempt_timeout_seconds if attempt_timeout_seconds is None else self._positive_int(attempt_timeout_seconds, "attempt_timeout_seconds")
        if timeout > policy.attempt_timeout_seconds:
            raise BudgetExceededError("attempt timeout exceeds node cap")
        with self._lock:
            reserved_usage = UsageTokens(
                input_tokens=usage.input_tokens,
                cached_input_tokens=usage.cached_input_tokens,
                cache_write_tokens=usage.cache_write_tokens,
                output_tokens=usage.output_tokens,
                web_search_tool_calls=maximum_tool_calls,
            )
            web_search_tool_price = get_openai_web_search_tool_price_usd_per_1k_decimal()
            reserved_cost = estimate_workflow_usage_cost(
                model,
                band,
                reserved_usage,
                web_search_tool_price_usd_per_1k=web_search_tool_price,
            )
            now = self._now()
            if self._exhausted:
                raise BudgetExceededError("workflow budget exhausted")
            node = self._nodes.get(node_name)
            node_deadline = now + policy.node_timeout_seconds if node is None else node.deadline_monotonic
            node_reserved_cost = Decimal("0") if node is None else node.reserved_cost
            node_settled_cost = Decimal("0") if node is None else node.settled_cost
            node_attempts = 0 if node is None else node.attempts
            tier_attempts = 0 if node is None else node.tier_attempts.get(model, 0)
            if node is not None and node.policy is not policy:
                raise ReservationStateError("node policy changed")
            if now >= node_deadline or now >= self._deadline_monotonic:
                raise BudgetExceededError("workflow deadline exhausted")
            if now + timeout > node_deadline or now + timeout > self._deadline_monotonic:
                raise BudgetExceededError("attempt timeout exceeds remaining deadline")
            if self._attempts >= self._maximum_attempts or node_attempts >= policy.maximum_total_attempts:
                raise BudgetExceededError("attempt cap exhausted")
            if tier_attempts >= policy.maximum_attempts_per_tier:
                raise BudgetExceededError("model tier attempt cap exhausted")
            if self._settled_cost + self._reserved_cost + reserved_cost > self._cost_cap:
                raise BudgetExceededError("workflow cost cap exceeded")
            if node_settled_cost + node_reserved_cost + reserved_cost > policy.node_cost_cap:
                raise BudgetExceededError("node cost cap exceeded")
            reservation = BudgetReservation(
                reservation_id=uuid.uuid4().hex,
                node_name=node_name,
                model=model,
                band=band,
                reserved_cost=reserved_cost,
                deadline_monotonic=min(now + timeout, node_deadline, self._deadline_monotonic),
                _ledger_nonce=self._nonce,
                _reserved_usage=reserved_usage,
                _web_search_tool_price_usd_per_1k=web_search_tool_price,
            )
            if node is None:
                node = _NodeBudget(policy, now, node_deadline)
                self._nodes[node_name] = node
            self._active[reservation.reservation_id] = reservation
            self._reserved_cost += reserved_cost
            node.reserved_cost += reserved_cost
            self._attempts += 1
            node.attempts += 1
            node.tier_attempts[model] = node.tier_attempts.get(model, 0) + 1
            return reservation

    def _active_reservation(self, reservation: BudgetReservation) -> BudgetReservation:
        if not isinstance(reservation, BudgetReservation) or reservation._ledger_nonce is not self._nonce:
            raise ReservationOwnershipError("reservation does not belong to this ledger")
        current = self._active.get(reservation.reservation_id)
        if current is None:
            if reservation.reservation_id in self._completed:
                raise ReservationStateError("reservation has already been settled")
            raise ReservationOwnershipError("reservation is stale")
        if current is not reservation:
            raise ReservationOwnershipError("reservation identity is invalid")
        return current

    def _close(
        self,
        reservation: BudgetReservation,
        charged_cost: Decimal,
        timeout: bool,
        *,
        overflowed: bool = False,
    ) -> tuple[BudgetSettlement, bool]:
        if not isinstance(charged_cost, Decimal) or charged_cost < 0 or not charged_cost.is_finite():
            raise ValueError("settlement cost is invalid")
        current = self._active_reservation(reservation)
        node = self._nodes[current.node_name]
        self._active.pop(current.reservation_id)
        self._completed.add(current.reservation_id)
        self._reserved_cost -= current.reserved_cost
        node.reserved_cost -= current.reserved_cost
        self._settled_cost += charged_cost
        node.settled_cost += charged_cost
        overflowed = overflowed or charged_cost > current.reserved_cost
        if overflowed:
            self._exhausted = True
            self._overdrawn = True
            node.exhausted = True
            node.overdrawn = True
        return BudgetSettlement(current.reservation_id, charged_cost, timeout), overflowed

    def settle(self, reservation: BudgetReservation, usage: UsageTokens) -> BudgetSettlement:
        if not isinstance(usage, UsageTokens):
            raise ValueError("workflow settlements require UsageTokens")
        with self._lock:
            current = self._active_reservation(reservation)
            reserved_usage = current._reserved_usage
            usage_overflowed = any(
                actual > reserved
                for actual, reserved in (
                    (usage.input_tokens, reserved_usage.input_tokens),
                    (usage.cached_input_tokens, reserved_usage.cached_input_tokens),
                    (usage.cache_write_tokens, reserved_usage.cache_write_tokens),
                    (usage.output_tokens, reserved_usage.output_tokens),
                    (usage.web_search_tool_calls, reserved_usage.web_search_tool_calls),
                )
            )
            charged_cost = estimate_workflow_usage_cost(
                current.model,
                current.band,
                usage,
                web_search_tool_price_usd_per_1k=current._web_search_tool_price_usd_per_1k,
            )
            settlement, overflowed = self._close(current, charged_cost, False, overflowed=usage_overflowed)
            if overflowed:
                raise BudgetOverflowError("actual usage exceeds reservation", settlement)
            return settlement

    def consume_timeout(self, reservation: BudgetReservation, usage: UsageTokens | None = None) -> BudgetSettlement:
        if usage is not None and not isinstance(usage, UsageTokens):
            raise ValueError("workflow timeout usage must use UsageTokens")
        with self._lock:
            current = self._active_reservation(reservation)
            settlement, _ = self._close(current, current.reserved_cost, True)
            return settlement

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            now = self._now()
            global_remaining = max(0, self._maximum_attempts - self._attempts)
            node_snapshots = tuple(
                NodeBudgetSnapshot(
                    node_name=node_name,
                    remaining_cost=max(Decimal("0"), node.policy.node_cost_cap - node.reserved_cost - node.settled_cost),
                    reserved_cost=node.reserved_cost,
                    settled_cost=node.settled_cost,
                    remaining_attempts=min(
                        global_remaining,
                        max(0, node.policy.maximum_total_attempts - node.attempts),
                        sum(
                            max(0, node.policy.maximum_attempts_per_tier - node.tier_attempts.get(tier.model, 0))
                            for tier in node.policy.tiers
                        ),
                    ),
                    remaining_seconds=max(0.0, node.deadline_monotonic - now),
                    exhausted=node.exhausted,
                    overdrawn=node.overdrawn,
                )
                for node_name, node in sorted(self._nodes.items())
            )
            return BudgetSnapshot(
                mode=self._mode,
                remaining_cost=max(Decimal("0"), self._cost_cap - self._reserved_cost - self._settled_cost),
                reserved_cost=self._reserved_cost,
                settled_cost=self._settled_cost,
                remaining_attempts=max(0, self._maximum_attempts - self._attempts),
                remaining_seconds=max(0.0, self._deadline_monotonic - now),
                deadline_monotonic=self._deadline_monotonic,
                nodes=node_snapshots,
                exhausted=self._exhausted,
                overdrawn=self._overdrawn,
            )
