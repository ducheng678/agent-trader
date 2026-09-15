from typing import Optional

from market_agent.calibration import extract_raw_confidence_value, normalize_confidence_value
from market_agent.models import (
    EntryPlan,
    EntryScenario,
    PositionManagementPlan,
    Scenario,
    StrategyDecision,
    TargetPositionPlan,
)
from market_agent.runtime_views import (
    build_effective_target_position,
    build_empty_position_management_plan,
    build_empty_strategy_decision,
)


class GenericPlaybook:
    def __init__(
        self,
        display_answer: str,
        current_bias: str,
        trigger_event_relevance: str = "not_applicable",
        trigger_confidence: Optional[float] = None,
        selected_symbol: str = "",
        selection_reason: str = "",
        execute_now: Optional[bool] = None,
        now_decision: Optional[StrategyDecision] = None,
        scenario: Optional[Scenario] = None,
        entry_plan: Optional[EntryPlan] = None,
        position_management: Optional[PositionManagementPlan] = None,
        post_fill_risk_template: Optional[PositionManagementPlan] = None,
        target_position: Optional[TargetPositionPlan] = None,
    ):
        self.display_answer = display_answer
        self.current_bias = current_bias
        self.trigger_event_relevance = str(trigger_event_relevance or "not_applicable").strip().lower() or "not_applicable"
        self.selected_symbol = str(selected_symbol or "").strip().upper()
        self.trigger_confidence_raw = extract_raw_confidence_value(trigger_confidence)
        self.trigger_confidence = normalize_confidence_value(self.trigger_confidence_raw, symbol=self.selected_symbol)
        self.selection_reason = str(selection_reason or "").strip()
        if entry_plan is None:
            entry_plan = EntryPlan(
                execute_now=bool(execute_now),
                action_decision=now_decision or build_empty_strategy_decision(),
                scenario=scenario,
            )
        self.entry_plan = entry_plan
        self.position_management = position_management or build_empty_position_management_plan()
        self.post_fill_risk_template = post_fill_risk_template or build_empty_position_management_plan()
        self.target_position = target_position or TargetPositionPlan()

    @property
    def execute_now(self) -> bool:
        return self.entry_plan.execute_now

    @property
    def now_decision(self) -> StrategyDecision:
        return self.entry_plan.action_decision

    @property
    def scenario(self) -> Optional[EntryScenario]:
        return self.entry_plan.scenario

    def to_dict(self, symbol_position: Optional[dict] = None) -> dict:
        return {
            "display_answer": self.display_answer,
            "current_bias": self.current_bias,
            "trigger_event_relevance": self.trigger_event_relevance,
            "trigger_confidence": self.trigger_confidence_raw,
            "selected_symbol": self.selected_symbol,
            "selection_reason": self.selection_reason,
            "target_position": build_effective_target_position(self, symbol_position).to_dict(),
            "entry_plan": self.entry_plan.to_dict(),
            "position_management": self.position_management.to_dict(),
            "post_fill_risk_template": self.post_fill_risk_template.to_dict(),
        }
