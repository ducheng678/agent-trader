from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, List, Literal, Optional, Tuple


def _safe_float_for_model(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if value is None:
            return default
        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return default
            return float(stripped)
        return float(value)
    except (TypeError, ValueError):
        return default


@dataclass
class StrategyDecision:
    action: Literal["long", "short", "no_trade"]
    suggested_notional_usd: float
    entry_price: float
    stop_loss_price: float
    planned_margin_used_usd: float
    planned_max_loss_usd: float
    requested_leverage: int = 0

    def __init__(
        self,
        action: Literal["long", "short", "no_trade"],
        suggested_notional_usd: float,
        entry_price: float,
        stop_loss_price: float,
        planned_margin_used_usd: float,
        planned_max_loss_usd: float,
        requested_leverage: int = 0,
    ):
        self.action = action
        self.suggested_notional_usd = suggested_notional_usd
        self.entry_price = entry_price
        self.stop_loss_price = stop_loss_price
        self.planned_margin_used_usd = planned_margin_used_usd
        self.planned_max_loss_usd = planned_max_loss_usd
        self.requested_leverage = requested_leverage

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "entry_price": self.entry_price,
            "stop_loss_price": self.stop_loss_price,
        }


@dataclass
class ManagementDecision:
    action: Literal["no_change", "close", "trim", "long", "short", "reverse_to_long", "reverse_to_short", "add_to_long", "add_to_short"]
    close_fraction: float
    new_notional_usd: float
    entry_price: float
    planned_max_loss_usd: float
    leverage: int = 0
    stop_loss_price: float = 0.0
    margin_basis_usd: float = 0.0
    continue_entry_plan_after_close: bool = False

    def __init__(
        self,
        action: Literal["no_change", "close", "trim", "long", "short", "reverse_to_long", "reverse_to_short", "add_to_long", "add_to_short"],
        close_fraction: float,
        new_notional_usd: float,
        entry_price: float,
        planned_max_loss_usd: float,
        leverage: int = 0,
        stop_loss_price: float = 0.0,
        margin_basis_usd: float = 0.0,
        continue_entry_plan_after_close: bool = False,
    ):
        self.action = action
        self.close_fraction = close_fraction
        self.new_notional_usd = new_notional_usd
        self.entry_price = entry_price
        self.planned_max_loss_usd = planned_max_loss_usd
        self.leverage = leverage
        self.stop_loss_price = stop_loss_price
        self.margin_basis_usd = margin_basis_usd
        self.continue_entry_plan_after_close = continue_entry_plan_after_close

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "close_fraction": self.close_fraction,
            "new_notional_usd": self.new_notional_usd,
            "leverage": self.leverage,
            "entry_price": self.entry_price,
            "stop_loss_price": self.stop_loss_price,
            "planned_max_loss_usd": self.planned_max_loss_usd,
            "margin_basis_usd": self.margin_basis_usd,
        }


@dataclass
class Condition:
    type: str
    level: float = 0.0
    low: float = 0.0
    high: float = 0.0
    timer_seconds: int = 0
    tolerance_bps: float = 0.0
    min_ratio: float = 0.0
    note: str = ""

    def __init__(
        self,
        type: str,
        level: float = 0.0,
        low: float = 0.0,
        high: float = 0.0,
        timer_seconds: int = 0,
        tolerance_bps: float = 0.0,
        min_ratio: float = 0.0,
        note: str = "",
        seconds: Optional[int] = None,
    ):
        self.type = type
        self.level = level
        self.low = low
        self.high = high
        self.timer_seconds = int(timer_seconds if seconds is None else seconds)
        self.tolerance_bps = tolerance_bps
        self.min_ratio = min_ratio
        self.note = note

    @property
    def seconds(self) -> int:
        return int(self.timer_seconds or 0)

    @seconds.setter
    def seconds(self, value: int) -> None:
        self.timer_seconds = int(value or 0)


def condition_to_dict(condition: "Condition") -> dict:
    return {
        "type": condition.type,
        "level": float(condition.level or 0.0),
        "low": float(condition.low or 0.0),
        "high": float(condition.high or 0.0),
        "timer_seconds": int(condition.timer_seconds or 0),
        "tolerance_bps": float(condition.tolerance_bps or 0.0),
        "min_ratio": float(condition.min_ratio or 0.0),
    }


@dataclass
class ExecuteWhenAll:
    condition: Optional[Condition]
    timeout_seconds: int

    @property
    def conditions(self) -> List[Condition]:
        return [self.condition] if self.condition is not None else []

    @conditions.setter
    def conditions(self, value: List[Condition]) -> None:
        items = list(value or [])
        self.condition = items[0] if items else None


@dataclass
class ObserveWhenAll:
    low: float = 0.0
    high: float = 0.0


def _coerce_observe_when_all(data: Any) -> ObserveWhenAll:
    if isinstance(data, ObserveWhenAll):
        low = float(data.low or 0.0)
        high = float(data.high or 0.0)
    elif isinstance(data, dict):
        low = float(data.get("low", 0.0) or 0.0)
        high = float(data.get("high", 0.0) or 0.0)
    elif isinstance(data, list) and data:
        first = data[0]
        if isinstance(first, Condition):
            low = float(first.low or first.level or 0.0)
            high = float(first.high or first.level or 0.0)
        elif isinstance(first, dict):
            low = float(first.get("low", first.get("level", 0.0)) or 0.0)
            high = float(first.get("high", first.get("level", 0.0)) or 0.0)
        else:
            low = 0.0
            high = 0.0
    else:
        low = 0.0
        high = 0.0
    low = max(0.0, low)
    high = max(0.0, high)
    if low > 0.0 and high <= 0.0:
        high = low
    elif high > 0.0 and low <= 0.0:
        low = high
    if low > 0.0 and high > 0.0 and low > high:
        low, high = high, low
    return ObserveWhenAll(low=low, high=high)


def _coerce_single_condition(data: Any) -> Optional[Condition]:
    if isinstance(data, Condition):
        return data
    if isinstance(data, dict):
        return Condition(
            type=str(data.get("type", "") or "").strip(),
            level=float(data.get("level", 0.0) or 0.0),
            low=float(data.get("low", 0.0) or 0.0),
            high=float(data.get("high", 0.0) or 0.0),
            timer_seconds=int(data.get("timer_seconds", data.get("seconds", 0)) or 0),
            tolerance_bps=float(data.get("tolerance_bps", 0.0) or 0.0),
            min_ratio=float(data.get("min_ratio", 0.0) or 0.0),
            note=str(data.get("note", "") or ""),
        )
    if isinstance(data, list) and data:
        return _coerce_single_condition(data[0])
    return None


def observe_when_all_to_dict(observe_when_all: Any) -> dict:
    observe = _coerce_observe_when_all(observe_when_all)
    return {
        "low": float(observe.low or 0.0),
        "high": float(observe.high or 0.0),
    }


def observe_when_all_contains_price(observe_when_all: Any, price: Optional[float]) -> bool:
    numeric_price = _safe_float_for_model(price, None)
    if numeric_price is None:
        return False
    observe = _coerce_observe_when_all(observe_when_all)
    low = float(observe.low or 0.0)
    high = float(observe.high or 0.0)
    if low <= 0.0 and high <= 0.0:
        return True
    return low <= numeric_price <= high


SCENARIO_RUNTIME_KEY = "__scenario__"


@dataclass
class Scenario:
    observe_when_all: ObserveWhenAll
    execute_when_all: ExecuteWhenAll
    observation_starts_when: str = ""

    def __init__(
        self,
        observe_when_all: Optional[Any],
        execute_when_all: Optional[Any] = None,
        cancel_when_any: Optional[List[Condition]] = None,
        observation_starts_when: str = "",
        arm_when_all: Optional[List[Condition]] = None,
        timeout_seconds_after_arm: Optional[int] = None,
    ):
        self.observe_when_all = _coerce_observe_when_all(observe_when_all)
        self.observation_starts_when = observation_starts_when
        if isinstance(execute_when_all, ExecuteWhenAll):
            self.execute_when_all = execute_when_all
        elif isinstance(execute_when_all, dict):
            raw_condition = execute_when_all.get("condition")
            if raw_condition is None:
                raw_condition = (execute_when_all.get("conditions", []) or [None])[0]
            self.execute_when_all = ExecuteWhenAll(
                condition=_coerce_single_condition(raw_condition),
                timeout_seconds=max(1, int(execute_when_all.get("timeout_seconds", timeout_seconds_after_arm or 300))),
            )
        else:
            self.execute_when_all = ExecuteWhenAll(
                condition=_coerce_single_condition((arm_when_all or [None])[0]),
                timeout_seconds=max(1, int(timeout_seconds_after_arm or 300)),
            )

    @property
    def arm_when_all(self) -> List[Condition]:
        return [self.execute_when_all.condition] if self.execute_when_all.condition is not None else []

    @arm_when_all.setter
    def arm_when_all(self, value: List[Condition]) -> None:
        items = list(value or [])
        self.execute_when_all.condition = items[0] if items else None

    @property
    def timeout_seconds_after_arm(self) -> int:
        return int(self.execute_when_all.timeout_seconds or 0)

    @timeout_seconds_after_arm.setter
    def timeout_seconds_after_arm(self, value: int) -> None:
        self.execute_when_all.timeout_seconds = max(1, int(value or 0))

    @property
    def cancel_when_any(self) -> List[Condition]:
        return []

    @cancel_when_any.setter
    def cancel_when_any(self, value: List[Condition]) -> None:
        return None


@dataclass
class EntryScenario:
    observe_when_all: ObserveWhenAll
    execute_when_all: ExecuteWhenAll

    def __init__(
        self,
        observe_when_all: Optional[Any],
        execute_when_all: Optional[Any] = None,
        cancel_when_any: Optional[List[Condition]] = None,
        arm_when_all: Optional[List[Condition]] = None,
        timeout_seconds_after_arm: Optional[int] = None,
    ):
        self.observe_when_all = _coerce_observe_when_all(observe_when_all)
        if isinstance(execute_when_all, ExecuteWhenAll):
            self.execute_when_all = execute_when_all
        elif isinstance(execute_when_all, dict):
            raw_condition = execute_when_all.get("condition")
            if raw_condition is None:
                raw_condition = (execute_when_all.get("conditions", []) or [None])[0]
            self.execute_when_all = ExecuteWhenAll(
                condition=_coerce_single_condition(raw_condition),
                timeout_seconds=max(1, int(execute_when_all.get("timeout_seconds", timeout_seconds_after_arm or 300))),
            )
        else:
            self.execute_when_all = ExecuteWhenAll(
                condition=_coerce_single_condition((arm_when_all or [None])[0]),
                timeout_seconds=max(1, int(timeout_seconds_after_arm or 300)),
            )

    @property
    def arm_when_all(self) -> List[Condition]:
        return [self.execute_when_all.condition] if self.execute_when_all.condition is not None else []

    @arm_when_all.setter
    def arm_when_all(self, value: List[Condition]) -> None:
        items = list(value or [])
        self.execute_when_all.condition = items[0] if items else None

    @property
    def timeout_seconds_after_arm(self) -> int:
        return int(self.execute_when_all.timeout_seconds or 0)

    @timeout_seconds_after_arm.setter
    def timeout_seconds_after_arm(self, value: int) -> None:
        self.execute_when_all.timeout_seconds = max(1, int(value or 0))

    @property
    def cancel_when_any(self) -> List[Condition]:
        return []

    @cancel_when_any.setter
    def cancel_when_any(self, value: List[Condition]) -> None:
        return None


@dataclass
class ExitLeg:
    name: str
    note: str
    when_all: List[Condition]
    close_fraction: float

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "when_all": [condition_to_dict(c) for c in self.when_all],
            "close_fraction": self.close_fraction,
        }


def entry_scenario_to_dict(scenario: EntryScenario) -> dict:
    return {
        "observe_when_all": observe_when_all_to_dict(scenario.observe_when_all),
        "execute_when_all": {
            "condition": condition_to_dict(scenario.execute_when_all.condition) if scenario.execute_when_all.condition is not None else None,
            "timeout_seconds": int(scenario.execute_when_all.timeout_seconds or 0),
        },
    }


def scenario_to_dict(scenario: Scenario) -> dict:
    return {
        "observation_starts_when": scenario.observation_starts_when,
        "observe_when_all": observe_when_all_to_dict(scenario.observe_when_all),
        "execute_when_all": {
            "condition": condition_to_dict(scenario.execute_when_all.condition) if scenario.execute_when_all.condition is not None else None,
            "timeout_seconds": int(scenario.execute_when_all.timeout_seconds or 0),
        },
    }


@dataclass
class EntryPlan:
    execute_now: bool
    action_decision: StrategyDecision
    scenario: Optional[EntryScenario]

    def to_dict(self) -> dict:
        return {
            "execute_now": self.execute_now,
            "action_decision": self.action_decision.to_dict(),
            "scenario": entry_scenario_to_dict(self.scenario) if self.scenario is not None else None,
        }


@dataclass
class PositionManagementPlan:
    execute_now: bool
    action_decision: ManagementDecision
    scenario: Optional[Scenario]

    def to_dict(self) -> dict:
        return {
            "execute_now": self.execute_now,
            "action_decision": self.action_decision.to_dict(),
            "scenario": scenario_to_dict(self.scenario) if self.scenario is not None else None,
        }


@dataclass
class TargetPositionImmediateAction:
    action: str = "none"
    target_side: str = "none"
    target_notional_usd: float = 0.0
    target_notional_mode: str = "none"
    retain_fraction_of_current_position: float = 0.0

    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "target_side": self.target_side,
            "target_notional_usd": self.target_notional_usd,
            "target_notional_mode": self.target_notional_mode,
            "retain_fraction_of_current_position": self.retain_fraction_of_current_position,
        }


@dataclass
class TargetPositionPlan:
    position_state: str = "unknown"
    immediate_action_source: str = "none"
    immediate_action: Optional[TargetPositionImmediateAction] = None
    observation_source: str = "none"
    observation_plan_names: List[str] = None
    active_management_source: str = "none"
    active_management_summary: str = ""
    successor_management_source: str = "none"
    successor_management_summary: str = ""

    def __post_init__(self):
        if self.immediate_action is None:
            self.immediate_action = TargetPositionImmediateAction()
        if self.observation_plan_names is None:
            self.observation_plan_names = []

    def to_dict(self) -> dict:
        return {
            "position_state": self.position_state,
            "immediate_action_source": self.immediate_action_source,
            "immediate_action": self.immediate_action.to_dict(),
            "observation_source": self.observation_source,
            "observation_plan_names": list(self.observation_plan_names),
            "active_management_source": self.active_management_source,
            "active_management_summary": self.active_management_summary,
            "successor_management_source": self.successor_management_source,
            "successor_management_summary": self.successor_management_summary,
        }


@dataclass
class ScenarioRuntime:
    observing: bool = False
    observing_at: Optional[float] = None
    armed: bool = False
    armed_at: Optional[float] = None
    completed: bool = False


@dataclass
class RiskSession:
    decision: Optional[StrategyDecision] = None
    plan_name: str = ""
    side: str = ""
    stop_loss_price: float = 0.0
    start_time: float = 0.0
    baseline_size: float = 0.0
    position_management: Optional[PositionManagementPlan] = None
    expected_size: float = 0.0
    initial_size_abs: float = 0.0
    take_profit_legs: List[ExitLeg] = None
    stop_loss_legs: List[ExitLeg] = None
    runtimes: Dict[str, ScenarioRuntime] = None
    history: Deque[Tuple[float, float]] = None
    history_seconds: int = 1800
    prev_price: Optional[float] = None
    executed_plan_names: set = None
    executed_leg_names: set = None
    resting_exit_orders: List[Dict[str, Any]] = None
    use_resting_exit_orders: bool = False
    pending_fill_reconcile_since: Optional[float] = None
    take_profit_legs_scale_from_initial_size: bool = False
    staged_exit_enabled: bool = False
    staged_exit_size_basis_abs: float = 0.0
    tp1_completed_size_abs: float = 0.0
    tp2_completed_size_abs: float = 0.0
    initial_entry_price: float = 0.0
    initial_stop_price: float = 0.0
    initial_risk_price_distance: float = 0.0
    tp1_price: float = 0.0
    tp2_price: float = 0.0
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp1_hit_at: float = 0.0
    tp2_hit_at: float = 0.0
    staged_exit_liquidity_band: str = ""
    max_favorable_excursion_r: float = 0.0
    tp1_no_follow_through_applied: bool = False
    tp1_no_follow_through_at: float = 0.0
    tp2_no_continuation_applied: bool = False
    post_tp1_stop_price: float = 0.0
    locked_floor_price: float = 0.0
    active_soft_stop_price: float = 0.0
    active_hard_stop_price: float = 0.0
    cross_asset_soft_stop_symbol: str = ""
    cross_asset_entry_price: float = 0.0
    cross_asset_entry_time: float = 0.0
    cross_asset_peak_adverse_pct: float = 0.0
    soft_stop_breach_since: float = 0.0
    soft_stop_last_breach_price: float = 0.0
    exchange_hard_stop_buffer_usd: float = 0.0
    exchange_hard_stop_min_buffer_usd: float = 0.0
    exchange_hard_stop_atr_buffer_usd: float = 0.0
    exchange_hard_stop_r_buffer_usd: float = 0.0
    exchange_hard_stop_atr_value: float = 0.0
    trailing_timeframe: str = "15m"
    trailing_atr_period: int = 14
    trailing_atr_lookback_bars: int = 200
    trailing_soft_atr_mult: float = 2.5
    trailing_hard_atr_mult: float = 3.5
    trailing_highest_close: float = 0.0
    trailing_lowest_close: float = 0.0
    trailing_soft_stop_price: float = 0.0
    trailing_hard_stop_price: float = 0.0
    trailing_last_bar_ms: int = 0
    trailing_last_close_price: float = 0.0
    position_basis_confidence_raw: Optional[float] = None
    position_basis_validity: float = 0.0
    basis_profit_observation_active: bool = False
    basis_profit_observation_started_at: float = 0.0
    basis_profit_observation_basis_start: float = 0.0
    basis_profit_history: Deque[Tuple[float, float]] = None

    def __post_init__(self):
        if self.take_profit_legs is None:
            self.take_profit_legs = []
        if self.stop_loss_legs is None:
            self.stop_loss_legs = []
        if self.runtimes is None:
            self.runtimes = {}
        if self.history is None:
            self.history = deque()
        if self.executed_plan_names is None:
            self.executed_plan_names = set()
        if self.executed_leg_names is None:
            self.executed_leg_names = set()
        if self.resting_exit_orders is None:
            self.resting_exit_orders = []
        if self.basis_profit_history is None:
            self.basis_profit_history = deque()
        if self.expected_size == 0.0:
            self.expected_size = self.baseline_size
        if self.initial_size_abs <= 0.0:
            self.initial_size_abs = abs(float(self.baseline_size or 0.0))
        if self.staged_exit_size_basis_abs <= 0.0:
            self.staged_exit_size_basis_abs = abs(float(self.initial_size_abs or 0.0))
        if self.initial_stop_price <= 0.0:
            self.initial_stop_price = float(self.stop_loss_price or 0.0)
        if self.initial_risk_price_distance <= 0.0 and self.initial_entry_price > 0.0 and self.initial_stop_price > 0.0:
            self.initial_risk_price_distance = abs(float(self.initial_entry_price - self.initial_stop_price))

    def update_price(self, price: float, now: float) -> None:
        self.history.append((now, price))
        while self.history and now - self.history[0][0] > self.history_seconds:
            self.history.popleft()

    def is_observing(self) -> bool:
        return any(state.observing and not state.completed for state in self.runtimes.values())

    def is_armed(self) -> bool:
        return any(state.armed and not state.completed for state in self.runtimes.values())

    def scenarios_completed(self) -> bool:
        if not self.runtimes:
            return True
        return all(state.completed for state in self.runtimes.values())


@dataclass
class PositionManagementSession:
    plan_name: str = ""
    side: str = ""
    playbook_reason: str = ""
    trigger_confidence_raw: Optional[float] = None
    position_basis_confidence_raw: Optional[float] = None
    position_basis_validity: float = 0.0
    position_management: Optional[PositionManagementPlan] = None
    start_time: float = 0.0
    baseline_size: float = 0.0
    expected_size: float = 0.0
    initial_size_abs: float = 0.0
    runtimes: Dict[str, ScenarioRuntime] = None
    history: Deque[Tuple[float, float]] = None
    history_seconds: int = 1800
    prev_price: Optional[float] = None
    executed_plan_names: set = None

    def __post_init__(self):
        if self.runtimes is None:
            self.runtimes = {}
        if self.history is None:
            self.history = deque()
        if self.executed_plan_names is None:
            self.executed_plan_names = set()
        if self.expected_size == 0.0:
            self.expected_size = self.baseline_size
        if self.initial_size_abs <= 0.0:
            self.initial_size_abs = abs(float(self.baseline_size or 0.0))

    def update_price(self, price: float, now: float) -> None:
        self.history.append((now, price))
        while self.history and now - self.history[0][0] > self.history_seconds:
            self.history.popleft()

    def is_observing(self) -> bool:
        return any(state.observing and not state.completed for state in self.runtimes.values())

    def is_armed(self) -> bool:
        return any(state.armed and not state.completed for state in self.runtimes.values())

    def scenarios_completed(self) -> bool:
        if not self.runtimes:
            return True
        return all(state.completed for state in self.runtimes.values())


@dataclass
class PendingEntryOrderSession:
    plan_name: str = ""
    symbol: str = ""
    side: str = ""
    management_decision: Optional[ManagementDecision] = None
    position_management: Optional[PositionManagementPlan] = None
    post_fill_risk_template: Optional[PositionManagementPlan] = None
    oid: Optional[int] = None
    cloid: str = ""
    limit_price: float = 0.0
    requested_qty: float = 0.0
    created_at: float = 0.0
