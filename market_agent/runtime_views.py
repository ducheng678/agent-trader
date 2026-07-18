from typing import TYPE_CHECKING, Any, Dict, List, Optional

from market_agent.calibration import extract_raw_confidence_value, normalize_confidence_value
from market_agent.constants import (
    MANAGEMENT_EXPOSURE_ACTION_VALUES,
    TARGET_POSITION_IMMEDIATE_ACTION_VALUES,
    TARGET_POSITION_MODE_VALUES,
    TARGET_POSITION_SIDE_VALUES,
    TARGET_POSITION_SOURCE_VALUES,
    TARGET_POSITION_STATE_VALUES,
)
from market_agent.models import (
    EntryScenario,
    ManagementDecision,
    PositionManagementPlan,
    Scenario,
    StrategyDecision,
    TargetPositionImmediateAction,
    TargetPositionPlan,
    _coerce_observe_when_all,
    condition_to_dict,
    observe_when_all_to_dict,
)
from market_agent.positions import snapshot_has_open_position
from market_agent.presentation import default_observation_starts_when, describe_entry_window
from market_agent.utils import safe_float

if TYPE_CHECKING:
    from market_agent.playbook import GenericPlaybook


def build_empty_strategy_decision() -> StrategyDecision:
    return StrategyDecision(
        action="no_trade",
        suggested_notional_usd=0.0,
        entry_price=0.0,
        stop_loss_price=0.0,
        planned_margin_used_usd=0.0,
        planned_max_loss_usd=0.0,
        requested_leverage=0,
    )


def build_empty_management_decision() -> ManagementDecision:
    return ManagementDecision(
        action="no_change",
        close_fraction=0.0,
        new_notional_usd=0.0,
        entry_price=0.0,
        planned_max_loss_usd=0.0,
        leverage=0,
        stop_loss_price=0.0,
        margin_basis_usd=0.0,
        continue_entry_plan_after_close=False,
    )


def build_empty_position_management_plan() -> "PositionManagementPlan":
    return PositionManagementPlan(
        execute_now=False,
        action_decision=build_empty_management_decision(),
        scenario=None,
    )


def build_management_exposure_entry_decision(decision: ManagementDecision) -> StrategyDecision:
    if decision.action not in MANAGEMENT_EXPOSURE_ACTION_VALUES:
        raise ValueError(f"Management decision {decision.action} does not create or increase exposure")
    if decision.action in {"long", "short"}:
        target_side = decision.action
    else:
        target_side = "long" if decision.action in {"reverse_to_long", "add_to_long"} else "short"
    return StrategyDecision(
        action=target_side,
        suggested_notional_usd=decision.new_notional_usd,
        entry_price=safe_float(decision.entry_price, 0.0) or 0.0,
        stop_loss_price=safe_float(decision.stop_loss_price, 0.0) or 0.0,
        planned_margin_used_usd=0.0,
        planned_max_loss_usd=safe_float(decision.planned_max_loss_usd, 0.0) or 0.0,
        requested_leverage=int(decision.leverage or 0),
    )


def build_decision_execution_view(decision: Any, trigger_confidence_raw: Any = None, symbol: Any = "") -> dict:
    entry_window = describe_entry_window(
        safe_float(getattr(decision, "entry_price", 0.0), 0.0) or 0.0,
    )
    raw_confidence = extract_raw_confidence_value(trigger_confidence_raw)
    normalized_confidence = normalize_confidence_value(raw_confidence, symbol=symbol)
    view = {
        "action": getattr(decision, "action", ""),
        "trigger_confidence_raw": raw_confidence,
        "trigger_confidence": normalized_confidence,
        "leverage": int((getattr(decision, "leverage", None) if getattr(decision, "leverage", None) is not None else getattr(decision, "requested_leverage", 0)) or 0),
        "suggested_notional_usd": safe_float(getattr(decision, "suggested_notional_usd", 0.0), 0.0) or 0.0,
        "new_notional_usd": safe_float(getattr(decision, "new_notional_usd", 0.0), 0.0) or 0.0,
        "close_fraction": safe_float(getattr(decision, "close_fraction", 0.0), 0.0) or 0.0,
        "entry_price": entry_window["entry_price"],
        "entry_zone_text": entry_window["entry_zone_text"],
        "margin_basis_usd": safe_float(getattr(decision, "margin_basis_usd", 0.0), 0.0) or 0.0,
        "planned_max_loss_usd": safe_float(getattr(decision, "planned_max_loss_usd", 0.0), 0.0) or 0.0,
        "stop_loss_price": safe_float(getattr(decision, "stop_loss_price", 0.0), 0.0) or 0.0,
    }
    return view


def build_entry_scenario_execution_view(scenario: EntryScenario) -> dict:
    return {
        "observation_starts_when": default_observation_starts_when(scenario.observe_when_all, scenario.execute_when_all.condition),
        "observe_when_all": observe_when_all_to_dict(scenario.observe_when_all),
        "execute_when_all": {
            "condition": condition_to_dict(scenario.execute_when_all.condition) if scenario.execute_when_all.condition is not None else None,
            "timeout_seconds": int(scenario.execute_when_all.timeout_seconds or 0),
        },
    }


def build_scenario_execution_view(scenario: Scenario) -> dict:
    return {
        "observation_starts_when": scenario.observation_starts_when
        or default_observation_starts_when(scenario.observe_when_all, scenario.execute_when_all.condition),
        "observe_when_all": observe_when_all_to_dict(scenario.observe_when_all),
        "execute_when_all": {
            "condition": condition_to_dict(scenario.execute_when_all.condition) if scenario.execute_when_all.condition is not None else None,
            "timeout_seconds": int(scenario.execute_when_all.timeout_seconds or 0),
        },
    }


def build_runtime_target_view(decision: Any, source: str, trigger_confidence_raw: Any = None, symbol: Any = "") -> dict:
    view = build_decision_execution_view(decision, trigger_confidence_raw=trigger_confidence_raw, symbol=symbol)
    action = str(view.get("action", "") or "")
    target = {
        "source": source,
        "action": action,
        "decision_view": view,
        "target_side": "",
        "target_notional_usd": 0.0,
        "target_notional_mode": "none",
        "retain_fraction_of_current_position": 0.0,
    }
    if source == "entry_plan":
        if action in {"long", "short"}:
            target["target_side"] = action
            target["target_notional_usd"] = float(view.get("suggested_notional_usd", 0.0) or 0.0)
            target["target_notional_mode"] = "explicit_total_notional"
        elif action == "no_trade":
            target["target_side"] = "flat"
            target["target_notional_mode"] = "no_immediate_trade"
        return target
    if action == "no_change":
        target["target_side"] = "current_position"
        target["target_notional_mode"] = "keep_current_position"
    elif action == "close":
        target["target_side"] = "flat"
        target["target_notional_mode"] = "flat"
    elif action == "trim":
        retain_fraction = max(0.0, 1.0 - float(view.get("close_fraction", 0.0) or 0.0))
        target["target_side"] = "current_position"
        target["target_notional_mode"] = "retain_fraction_of_current_position"
        target["retain_fraction_of_current_position"] = retain_fraction
    elif action in {"long", "short"}:
        target["target_side"] = action
        target["target_notional_usd"] = float(view.get("new_notional_usd", 0.0) or 0.0)
        target["target_notional_mode"] = "explicit_total_notional"
    elif action in {"reverse_to_long", "add_to_long"}:
        target["target_side"] = "long"
        target["target_notional_usd"] = float(view.get("new_notional_usd", 0.0) or 0.0)
        target["target_notional_mode"] = "explicit_total_notional"
    elif action in {"reverse_to_short", "add_to_short"}:
        target["target_side"] = "short"
        target["target_notional_usd"] = float(view.get("new_notional_usd", 0.0) or 0.0)
        target["target_notional_mode"] = "explicit_total_notional"
    return target


def build_position_management_view(plan: "PositionManagementPlan", trigger_confidence_raw: Any = None, symbol: Any = "") -> dict:
    return {
        "execute_now": plan.execute_now,
        "action_decision": build_decision_execution_view(plan.action_decision, trigger_confidence_raw=trigger_confidence_raw, symbol=symbol),
        "scenario": build_scenario_execution_view(plan.scenario) if plan.scenario is not None else None,
    }


def build_target_position_plan_from_runtime_view(runtime_view: dict) -> TargetPositionPlan:
    runtime_view = dict(runtime_view or {})
    immediate = dict(runtime_view.get("immediate_action") or {})
    target = dict(immediate.get("target") or {})
    action = str(target.get("action", "none") or "none")
    if action not in TARGET_POSITION_IMMEDIATE_ACTION_VALUES:
        action = "none"
    target_side = str(target.get("target_side", "none") or "none")
    if target_side not in TARGET_POSITION_SIDE_VALUES:
        target_side = "none"
    target_mode = str(target.get("target_notional_mode", "none") or "none")
    if target_mode not in TARGET_POSITION_MODE_VALUES:
        target_mode = "none"
    position_state = str(runtime_view.get("position_state", "unknown") or "unknown")
    if position_state not in TARGET_POSITION_STATE_VALUES:
        position_state = "unknown"
    immediate_source = str(immediate.get("source", "none") or "none")
    if immediate_source not in TARGET_POSITION_SOURCE_VALUES:
        immediate_source = "none"
    observation_source = str(runtime_view.get("observation_source", "none") or "none")
    if observation_source not in TARGET_POSITION_SOURCE_VALUES:
        observation_source = "none"
    active_management_source = str(runtime_view.get("active_management_source", "none") or "none")
    if active_management_source not in TARGET_POSITION_SOURCE_VALUES:
        active_management_source = "none"
    successor_management_source = str(runtime_view.get("successor_management_source", "none") or "none")
    if successor_management_source not in TARGET_POSITION_SOURCE_VALUES:
        successor_management_source = "none"
    observation_plan_names = ["scenario"] if list(runtime_view.get("observation_scenarios") or []) else []
    active_management = runtime_view.get("active_management") or {}
    successor_management = runtime_view.get("successor_management") or {}
    return TargetPositionPlan(
        position_state=position_state,
        immediate_action_source=immediate_source,
        immediate_action=TargetPositionImmediateAction(
            action=action,
            target_side=target_side,
            target_notional_usd=max(0.0, float(target.get("target_notional_usd", 0.0) or 0.0)),
            target_notional_mode=target_mode,
            retain_fraction_of_current_position=min(max(float(target.get("retain_fraction_of_current_position", 0.0) or 0.0), 0.0), 1.0),
        ),
        observation_source=observation_source,
        observation_plan_names=observation_plan_names,
        active_management_source=active_management_source,
        active_management_summary=str((active_management.get("action_decision") or {}).get("action", "") or ""),
        successor_management_source=successor_management_source,
        successor_management_summary=str((successor_management.get("action_decision") or {}).get("action", "") or ""),
    )


def synthetic_symbol_position_for_target_state(position_state: str) -> Optional[dict]:
    state = str(position_state or "").strip().lower()
    if state == "open":
        return {"symbol": "", "side": "long", "size": 1.0, "notional_usd": 1.0}
    if state == "flat":
        return {"symbol": "", "side": "flat", "size": 0.0, "notional_usd": 0.0}
    return None


def build_effective_target_position(playbook: "GenericPlaybook", symbol_position: Optional[dict] = None) -> TargetPositionPlan:
    if isinstance(symbol_position, dict):
        return build_target_position_plan_from_runtime_view(build_playbook_runtime_view(playbook, symbol_position))
    declared = getattr(playbook, "target_position", None)
    if isinstance(declared, TargetPositionPlan):
        synthetic_position = synthetic_symbol_position_for_target_state(declared.position_state)
        if synthetic_position is not None:
            return build_target_position_plan_from_runtime_view(build_playbook_runtime_view(playbook, synthetic_position))
        return TargetPositionPlan(
            position_state=str(declared.position_state or "unknown"),
            immediate_action_source=str(declared.immediate_action_source or "none"),
            immediate_action=TargetPositionImmediateAction(**declared.immediate_action.to_dict()),
            observation_source=str(declared.observation_source or "none"),
            observation_plan_names=list(declared.observation_plan_names or []),
            active_management_source=str(declared.active_management_source or "none"),
            active_management_summary=str(declared.active_management_summary or ""),
            successor_management_source=str(declared.successor_management_source or "none"),
            successor_management_summary=str(declared.successor_management_summary or ""),
        )
    return build_target_position_plan_from_runtime_view(build_playbook_runtime_view(playbook, None))


def build_playbook_runtime_view(playbook: "GenericPlaybook", symbol_position: Optional[dict] = None) -> dict:
    has_open_position = snapshot_has_open_position(symbol_position or {}) if isinstance(symbol_position, dict) else False
    position_state = "open" if has_open_position else "flat" if isinstance(symbol_position, dict) else "unknown"
    ignored_sources: List[Dict[str, Any]] = []
    immediate_action: Dict[str, Any] = {"source": "none", "target": None}
    observation_source = "none"
    observation_scenarios: List[dict] = []
    active_management_source = "none"
    active_management = None
    successor_management_source = "none"
    successor_management = None

    if has_open_position:
        active_management_source = "position_management"
        active_management = build_position_management_view(playbook.position_management, trigger_confidence_raw=playbook.trigger_confidence_raw, symbol=playbook.selected_symbol)
        if playbook.entry_plan.execute_now:
            ignored_sources.append(
                {
                    "source": "entry_plan",
                    "reason": "selected symbol already has a live position; immediate runtime uses position_management.",
                    "decision_view": build_decision_execution_view(playbook.entry_plan.action_decision, trigger_confidence_raw=playbook.trigger_confidence_raw, symbol=playbook.selected_symbol),
                }
            )
        if playbook.position_management.execute_now:
            immediate_action = {
                "source": "position_management",
                "target": build_runtime_target_view(playbook.position_management.action_decision, "position_management", trigger_confidence_raw=playbook.trigger_confidence_raw, symbol=playbook.selected_symbol),
            }
        if playbook.position_management.scenario is not None:
            observation_source = "position_management"
            observation_scenarios = [build_scenario_execution_view(playbook.position_management.scenario)]
        if playbook.position_management.execute_now and playbook.position_management.action_decision.action in MANAGEMENT_EXPOSURE_ACTION_VALUES:
            successor_management_source = "post_fill_risk_template"
            successor_management = build_position_management_view(playbook.post_fill_risk_template, trigger_confidence_raw=playbook.trigger_confidence_raw, symbol=playbook.selected_symbol)
        else:
            successor_management_source = "position_management"
            successor_management = build_position_management_view(playbook.position_management, trigger_confidence_raw=playbook.trigger_confidence_raw, symbol=playbook.selected_symbol)
    else:
        active_management_source = "position_management" if position_management_plan_has_content(playbook.position_management) else "none"
        active_management = build_position_management_view(playbook.position_management, trigger_confidence_raw=playbook.trigger_confidence_raw, symbol=playbook.selected_symbol) if position_management_plan_has_content(playbook.position_management) else None
        if playbook.entry_plan.execute_now or playbook.entry_plan.scenario is not None:
            ignored_sources.append(
                {
                    "source": "entry_plan",
                    "reason": "runtime materializes entry_plan into position_management before execution.",
                    "decision_view": build_decision_execution_view(playbook.entry_plan.action_decision, trigger_confidence_raw=playbook.trigger_confidence_raw, symbol=playbook.selected_symbol),
                }
            )
        if playbook.position_management.execute_now:
            immediate_action = {
                "source": "position_management",
                "target": build_runtime_target_view(playbook.position_management.action_decision, "position_management", trigger_confidence_raw=playbook.trigger_confidence_raw, symbol=playbook.selected_symbol),
            }
        if playbook.position_management.scenario is not None:
            observation_source = "position_management"
            observation_scenarios = [build_scenario_execution_view(playbook.position_management.scenario)]
        if playbook.position_management.execute_now or playbook.position_management.scenario is not None:
            successor_management_source = "post_fill_risk_template"
            successor_management = build_position_management_view(playbook.post_fill_risk_template, trigger_confidence_raw=playbook.trigger_confidence_raw, symbol=playbook.selected_symbol)

    return {
        "trigger_confidence_raw": playbook.trigger_confidence_raw,
        "trigger_confidence": playbook.trigger_confidence,
        "position_state": position_state,
        "immediate_action": immediate_action,
        "ignored_immediate_sources": ignored_sources,
        "observation_source": observation_source,
        "observation_scenarios": observation_scenarios,
        "active_management_source": active_management_source,
        "active_management": active_management,
        "successor_management_source": successor_management_source,
        "successor_management": successor_management,
    }


def build_playbook_execution_view(playbook: "GenericPlaybook", symbol_position: Optional[dict] = None) -> dict:
    runtime_view = build_playbook_runtime_view(playbook, symbol_position)
    return {
        "current_bias": playbook.current_bias,
        "trigger_confidence": playbook.trigger_confidence_raw,
        "selected_symbol": playbook.selected_symbol,
        "selection_reason": playbook.selection_reason,
        "target_position": build_target_position_plan_from_runtime_view(runtime_view).to_dict(),
        "runtime_view": runtime_view,
        "entry_plan": {
            "execute_now": playbook.entry_plan.execute_now,
            "action_decision": build_decision_execution_view(playbook.entry_plan.action_decision, trigger_confidence_raw=playbook.trigger_confidence_raw, symbol=playbook.selected_symbol),
            "scenario": build_entry_scenario_execution_view(playbook.entry_plan.scenario) if playbook.entry_plan.scenario is not None else None,
        },
        "position_management": build_position_management_view(playbook.position_management, trigger_confidence_raw=playbook.trigger_confidence_raw, symbol=playbook.selected_symbol),
        "post_fill_risk_template": build_position_management_view(playbook.post_fill_risk_template, trigger_confidence_raw=playbook.trigger_confidence_raw, symbol=playbook.selected_symbol),
    }


def position_management_plan_has_content(plan: Optional[PositionManagementPlan]) -> bool:
    if plan is None:
        return False
    if plan.scenario is not None:
        return True
    decision = plan.action_decision if isinstance(plan.action_decision, ManagementDecision) else None
    if decision is None:
        return False
    if safe_float(getattr(decision, "stop_loss_price", 0.0), 0.0):
        return True
    if plan.execute_now and decision.action != "no_change":
        return True
    if plan.execute_now and int(decision.leverage or 0) > 0:
        return True
    return False


def _pm_compare_price_bps(old_value: Any, new_value: Any) -> float:
    old_numeric = safe_float(old_value, 0.0) or 0.0
    new_numeric = safe_float(new_value, 0.0) or 0.0
    if old_numeric <= 0 and new_numeric <= 0:
        return 0.0
    if old_numeric <= 0 or new_numeric <= 0:
        return float("inf")
    baseline = max(abs(old_numeric), abs(new_numeric), 1e-12)
    return abs(old_numeric - new_numeric) / baseline * 10000.0


def compare_position_management_plans(
    old_plan: Optional[PositionManagementPlan],
    new_plan: Optional[PositionManagementPlan],
) -> Dict[str, Any]:
    if old_plan is None or new_plan is None:
        return {
            "should_replace": True,
            "hard_reasons": ["missing_plan"],
            "soft_reasons": [],
        }

    hard_reasons: List[str] = []
    soft_reasons: List[str] = []

    if bool(old_plan.execute_now) != bool(new_plan.execute_now):
        hard_reasons.append("execute_now_changed")

    old_decision = old_plan.action_decision if isinstance(old_plan.action_decision, ManagementDecision) else build_empty_management_decision()
    new_decision = new_plan.action_decision if isinstance(new_plan.action_decision, ManagementDecision) else build_empty_management_decision()

    if str(old_decision.action or "") != str(new_decision.action or ""):
        hard_reasons.append("action_changed")

    old_scenario = old_plan.scenario if isinstance(old_plan.scenario, Scenario) else None
    new_scenario = new_plan.scenario if isinstance(new_plan.scenario, Scenario) else None
    if (old_scenario is None) != (new_scenario is None):
        hard_reasons.append("scenario_presence_changed")
    elif old_scenario is not None and new_scenario is not None:
        old_observe = _coerce_observe_when_all(old_scenario.observe_when_all)
        new_observe = _coerce_observe_when_all(new_scenario.observe_when_all)
        if max(
            _pm_compare_price_bps(old_observe.low, new_observe.low),
            _pm_compare_price_bps(old_observe.high, new_observe.high),
        ) > 75.0:
            soft_reasons.append("observe_when_all_price_changed")

    old_notional = max(0.0, float(old_decision.new_notional_usd or 0.0))
    new_notional = max(0.0, float(new_decision.new_notional_usd or 0.0))
    if max(old_notional, new_notional) > 0:
        notional_ratio_delta = abs(new_notional - old_notional) / max(old_notional, new_notional, 1e-12)
        if notional_ratio_delta > 0.03:
            soft_reasons.append("new_notional_usd_changed")

    if abs(float(old_decision.close_fraction or 0.0) - float(new_decision.close_fraction or 0.0)) > 0.02:
        soft_reasons.append("close_fraction_changed")

    if max(
        _pm_compare_price_bps(old_decision.entry_price, new_decision.entry_price),
        _pm_compare_price_bps(old_decision.stop_loss_price, new_decision.stop_loss_price),
    ) > 50.0:
        soft_reasons.append("decision_prices_changed")

    return {
        "should_replace": bool(hard_reasons or soft_reasons),
        "hard_reasons": sorted(set(hard_reasons)),
        "soft_reasons": sorted(set(soft_reasons)),
    }
