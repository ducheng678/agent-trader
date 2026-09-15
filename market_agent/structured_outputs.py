from __future__ import annotations

import hashlib
import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from market_agent.calibration import extract_raw_confidence_value, get_trigger_confidence_calibration
from market_agent.constants import (
    CONDITION_TYPES,
    DEFAULT_DIAGNOSTIC_INSTRUMENT_UNIVERSE,
    ENTRY_ACTION_VALUES,
    MANAGEMENT_QUERY_OMIT_MARKET_SPEC_FIELDS,
    PM_SCENARIO_REQUERY_LOCK_REASONS,
    SEARCH_MODES,
)
from market_agent.events import _iter_jsonl_lines_reverse, parse_utc_iso, strip_item_id_for_llm
from market_agent.logging_utils import print_line
from market_agent.models import (
    Condition,
    EntryPlan,
    EntryScenario,
    ExecuteWhenAll,
    ManagementDecision,
    ObserveWhenAll,
    StrategyDecision,
    _coerce_observe_when_all,
)
from market_agent.openai_usage import (
    _response_attr,
    analyze_web_search_calls,
    count_web_search_tool_calls,
    estimate_openai_usage_cost,
    extract_response_usage,
    extract_web_search_call_details,
    merge_usage_costs,
    merge_usage_dicts,
    normalize_image_input_context,
    sanitize_response_input_messages,
)
from market_agent.playbook import GenericPlaybook
from market_agent.presentation import normalize_entry_price
from market_agent.runtime_views import (
    build_effective_target_position,
    build_empty_strategy_decision,
    build_playbook_execution_view,
)
from market_agent.schemas import (
    HELPER_MARKET_NEWS_CONTEXT_SCHEMA,
    PASSIVE_EVENT_JUDGE_SCHEMA,
    PASSIVE_TECHNICAL_PRICING_SCHEMA,
    PLAYBOOK_SCHEMA,
)
from market_agent.symbols import (
    build_default_query,
    canonicalize_execution_symbol,
    normalize_candidate_key,
    parse_symbol_universe,
)
from market_agent.utils import safe_float



class StructuredOutputMixin:
    @classmethod
    def _sanitize_playbook_trade_symbol_context(cls, trade_symbol_context: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(trade_symbol_context, dict):
            return {}
        context = dict(trade_symbol_context)
        context.pop("symbol_position", None)
        if isinstance(context.get("market_spec"), dict):
            context["market_spec"] = {
                key: value
                for key, value in dict(context.get("market_spec") or {}).items()
                if key not in MANAGEMENT_QUERY_OMIT_MARKET_SPEC_FIELDS
            }
        return context
    def _align_price_for_symbol(self, symbol: str, price: float) -> float:
        raw = max(0.0, float(price or 0.0))
        if raw <= 0.0:
            return 0.0
        reader = getattr(self, "reader", None)
        if reader is not None and hasattr(reader, "align_price_to_wire"):
            try:
                return float(reader.align_price_to_wire(symbol, raw) or 0.0)
            except Exception:
                pass
        if reader is not None and hasattr(reader, "get_sz_decimals"):
            try:
                sz_decimals = int(reader.get_sz_decimals(symbol) or 0)
                return round(float(f"{raw:.5g}"), max(0, 6 - sz_decimals))
            except Exception:
                pass
        return raw
    def _normalize_condition_prices_for_symbol(self, condition: Condition, symbol: str) -> Condition:
        condition.level = self._align_price_for_symbol(symbol, condition.level)
        condition.low = self._align_price_for_symbol(symbol, condition.low)
        condition.high = self._align_price_for_symbol(symbol, condition.high)
        return condition
    def _normalize_observe_when_all_for_symbol(self, observe_when_all: Any, symbol: str) -> ObserveWhenAll:
        observe = _coerce_observe_when_all(observe_when_all)
        observe.low = self._align_price_for_symbol(symbol, observe.low)
        observe.high = self._align_price_for_symbol(symbol, observe.high)
        if observe.low > 0.0 and observe.high > 0.0 and observe.low > observe.high:
            observe.low, observe.high = observe.high, observe.low
        return observe
    def _normalize_strategy_decision_prices_for_symbol(self, decision: StrategyDecision, symbol: str) -> StrategyDecision:
        decision.entry_price = self._align_price_for_symbol(symbol, decision.entry_price)
        decision.entry_price = normalize_entry_price(decision.entry_price)
        decision.stop_loss_price = self._align_price_for_symbol(symbol, decision.stop_loss_price)
        return decision
    def _normalize_management_decision_prices_for_symbol(self, decision: ManagementDecision, symbol: str) -> ManagementDecision:
        decision.entry_price = self._align_price_for_symbol(symbol, decision.entry_price)
        decision.entry_price = normalize_entry_price(decision.entry_price)
        decision.stop_loss_price = self._align_price_for_symbol(symbol, decision.stop_loss_price)
        return decision
    def _normalize_playbook_prices_for_symbol(self, playbook: GenericPlaybook, symbol: str) -> GenericPlaybook:
        self._normalize_strategy_decision_prices_for_symbol(playbook.entry_plan.action_decision, symbol)
        scenario = playbook.entry_plan.scenario
        if scenario is not None:
            scenario.observe_when_all = self._normalize_observe_when_all_for_symbol(scenario.observe_when_all, symbol)
            if scenario.execute_when_all.condition is not None:
                self._normalize_condition_prices_for_symbol(scenario.execute_when_all.condition, symbol)
        self._normalize_management_decision_prices_for_symbol(playbook.position_management.action_decision, symbol)
        scenario = playbook.position_management.scenario
        if scenario is not None:
            scenario.observe_when_all = self._normalize_observe_when_all_for_symbol(scenario.observe_when_all, symbol)
            if scenario.execute_when_all.condition is not None:
                self._normalize_condition_prices_for_symbol(scenario.execute_when_all.condition, symbol)
        self._normalize_management_decision_prices_for_symbol(playbook.post_fill_risk_template.action_decision, symbol)
        scenario = playbook.post_fill_risk_template.scenario
        if scenario is not None:
            scenario.observe_when_all = self._normalize_observe_when_all_for_symbol(scenario.observe_when_all, symbol)
            if scenario.execute_when_all.condition is not None:
                self._normalize_condition_prices_for_symbol(scenario.execute_when_all.condition, symbol)
        return playbook
    def _cap_decision(self, decision: StrategyDecision) -> StrategyDecision:
        decision.suggested_notional_usd = max(0.0, decision.suggested_notional_usd)
        decision.requested_leverage = min(max(int(decision.requested_leverage or 0), 0), 100)
        decision.entry_price = normalize_entry_price(decision.entry_price)
        decision.planned_margin_used_usd = max(0.0, float(decision.planned_margin_used_usd or 0.0))
        decision.planned_max_loss_usd = max(0.0, float(decision.planned_max_loss_usd or 0.0))
        decision.stop_loss_price = max(0.0, float(decision.stop_loss_price or 0.0))
        if decision.action == "no_trade":
            decision.suggested_notional_usd = 0.0
            decision.entry_price = 0.0
            decision.stop_loss_price = 0.0
            decision.planned_margin_used_usd = 0.0
            decision.planned_max_loss_usd = 0.0
            decision.requested_leverage = 0
        return decision
    def _cap_playbook(self, playbook: GenericPlaybook) -> GenericPlaybook:
        playbook.selected_symbol = str(playbook.selected_symbol or "").strip().upper()
        playbook.selection_reason = str(playbook.selection_reason or "").strip()
        playbook.entry_plan.action_decision = self._cap_decision(playbook.entry_plan.action_decision)
        if playbook.entry_plan.scenario is not None:
            scenario = playbook.entry_plan.scenario
            scenario.observe_when_all = _coerce_observe_when_all(scenario.observe_when_all)
        if playbook.entry_plan.execute_now:
            playbook.entry_plan.scenario = None
        if playbook.entry_plan.execute_now and playbook.entry_plan.action_decision.action == "no_trade":
            playbook.entry_plan.execute_now = False
        playbook.target_position = build_effective_target_position(playbook)
        return playbook

def validate_decision(data: dict) -> StrategyDecision:
    if not isinstance(data, dict):
        raise ValueError("Decision is not a JSON object")
    action = str(data.get("action", "no_trade") or "no_trade").strip()
    if action not in ENTRY_ACTION_VALUES:
        raise ValueError(f"Invalid action: {action}")
    entry_price = max(0.0, float(data.get("entry_price", 0.0) or 0.0))
    stop_loss_price = max(0.0, float(data.get("stop_loss_price", 0.0) or 0.0))
    return StrategyDecision(
        action=action,
        suggested_notional_usd=0.0,
        entry_price=entry_price,
        stop_loss_price=stop_loss_price,
        planned_margin_used_usd=0.0,
        planned_max_loss_usd=0.0,
        requested_leverage=0,
    )
def validate_passive_event_judge(data: dict, relevance_threshold: float = 0.20) -> Dict[str, Any]:
    if not isinstance(data, dict):
        raise ValueError("Passive event judge output is not a JSON object")
    unsupported_keys = sorted(str(key) for key in data.keys() if key not in {"trigger_event_relevance", "trigger_confidence", "action"})
    if unsupported_keys:
        raise ValueError(f"Unsupported passive event judge keys: {', '.join(unsupported_keys)}")
    trigger_event_relevance = str(data.get("trigger_event_relevance", "unrelated") or "unrelated").strip().lower()
    if trigger_event_relevance not in {"relevant", "unrelated", "duplicate"}:
        raise ValueError(f"Invalid trigger_event_relevance: {trigger_event_relevance}")
    trigger_confidence_raw = extract_raw_confidence_value(data.get("trigger_confidence"))
    if trigger_confidence_raw is None:
        raise ValueError("Passive event judge must include numeric trigger_confidence")
    trigger_confidence = min(max(float(trigger_confidence_raw), 0.0), 1.0)
    action = str(data.get("action", "no_trade") or "no_trade").strip().lower()
    if action not in ENTRY_ACTION_VALUES:
        raise ValueError(f"Invalid passive event judge action: {action}")
    if trigger_event_relevance == "duplicate":
        if trigger_confidence != 0.0:
            raise ValueError("Duplicate passive judge response must set trigger_confidence to 0")
        if action != "no_trade":
            raise ValueError("Duplicate passive judge response must set action to no_trade")
    if trigger_event_relevance == "unrelated":
        if action != "no_trade":
            raise ValueError("Unrelated passive judge response must set action to no_trade")
    return {
        "trigger_event_relevance": trigger_event_relevance,
        "trigger_confidence": trigger_confidence,
        "action": action,
    }
def validate_passive_technical_pricing(data: dict) -> Dict[str, float]:
    if not isinstance(data, dict):
        raise ValueError("Passive technical pricing output is not a JSON object")
    unsupported_keys = sorted(str(key) for key in data.keys() if key not in {"entry_price", "stop_loss_price"})
    if unsupported_keys:
        raise ValueError(f"Unsupported passive technical pricing keys: {', '.join(unsupported_keys)}")
    return {
        "entry_price": max(0.0, float(data.get("entry_price", 0.0) or 0.0)),
        "stop_loss_price": max(0.0, float(data.get("stop_loss_price", 0.0) or 0.0)),
    }
def validate_observe_when_all(data: Any) -> ObserveWhenAll:
    return _coerce_observe_when_all(data)
def validate_execute_when_all(data: dict) -> ExecuteWhenAll:
    if not isinstance(data, dict):
        data = {}
    condition_data = data.get("condition")
    if condition_data is None:
        condition_items = [x for x in (data.get("conditions", []) or []) if isinstance(x, dict)]
        condition_data = condition_items[0] if condition_items else None
    return ExecuteWhenAll(
        condition=validate_condition(condition_data) if isinstance(condition_data, dict) else None,
        timeout_seconds=max(1, int(data.get("timeout_seconds", 300))),
    )
def validate_entry_scenario(data: dict) -> EntryScenario:
    observe_when_all = validate_observe_when_all(data.get("observe_when_all", {}))
    execute_when_all = validate_execute_when_all(
        data.get(
            "execute_when_all",
            {
                "condition": ((data.get("arm_when_all", []) or [None])[0]),
                "timeout_seconds": data.get("timeout_seconds_after_arm", 300),
            },
        )
    )
    return EntryScenario(
        observe_when_all=observe_when_all,
        execute_when_all=execute_when_all,
    )
def validate_condition(data: dict) -> Condition:
    if not isinstance(data, dict):
        raise ValueError("Condition is not a JSON object")
    ctype = str(data.get("type", "")).strip()
    if ctype not in CONDITION_TYPES:
        raise ValueError(f"Invalid condition type: {ctype}")
    return Condition(
        type=ctype,
        level=float(data.get("level", 0.0) or 0.0),
        low=float(data.get("low", 0.0) or 0.0),
        high=float(data.get("high", 0.0) or 0.0),
        timer_seconds=int(data.get("timer_seconds", data.get("seconds", 0)) or 0),
        tolerance_bps=max(0.0, float(data.get("tolerance_bps", 0.0) or 0.0)),
        min_ratio=min(max(float(data.get("min_ratio", 0.0) or 0.0), 0.0), 1.0),
        note="",
    )
def validate_playbook(data: dict) -> GenericPlaybook:
    if not isinstance(data, dict):
        raise ValueError("Playbook is not a JSON object")
    new_root_keys = {"trigger_event_relevance", "trigger_confidence", "playbook"}
    old_root_keys = {"display_answer", "current_bias", "trigger_event_relevance", "trigger_confidence", "selected_symbol", "selection_reason", "entry_plan"}
    active_root_keys = new_root_keys if "playbook" in data else old_root_keys
    unsupported_keys = sorted(str(key) for key in data.keys() if key not in active_root_keys)
    if unsupported_keys:
        raise ValueError(f"Unsupported playbook root keys: {', '.join(unsupported_keys)}")
    trigger_event_relevance = str(data.get("trigger_event_relevance", "not_applicable") or "not_applicable").strip().lower()
    if trigger_event_relevance not in {"not_applicable", "relevant", "unrelated", "duplicate"}:
        raise ValueError(f"Invalid trigger_event_relevance: {trigger_event_relevance}")
    selected_symbol_hint = ""
    if isinstance(data.get("playbook"), dict):
        selected_symbol_hint = str((data.get("playbook") or {}).get("selected_symbol", "") or "").strip().upper()
    if not selected_symbol_hint:
        selected_symbol_hint = str(data.get("selected_symbol", "") or "").strip().upper()
    confidence_threshold = get_trigger_confidence_calibration(selected_symbol_hint)["relevance_threshold"]
    trigger_confidence_raw = extract_raw_confidence_value(data.get("trigger_confidence"))
    if trigger_event_relevance == "not_applicable":
        if trigger_confidence_raw is not None:
            raise ValueError("Active response must set trigger_confidence to null")
    else:
        if trigger_confidence_raw is None:
            raise ValueError("Passive response must include numeric trigger_confidence")
        if trigger_event_relevance == "duplicate":
            if float(trigger_confidence_raw) != 0.0:
                raise ValueError("Duplicate passive response must set trigger_confidence to 0")
        if trigger_event_relevance == "unrelated" and trigger_confidence_raw >= confidence_threshold:
            raise ValueError(f"Unrelated passive response must keep trigger_confidence below {confidence_threshold:.2f}")
        if trigger_event_relevance == "relevant" and trigger_confidence_raw < confidence_threshold:
            raise ValueError(f"Relevant passive response must set trigger_confidence to at least {confidence_threshold:.2f}")
    if "playbook" in data:
        playbook_payload = data.get("playbook", None)
        if trigger_event_relevance in {"unrelated", "duplicate"}:
            if playbook_payload is not None:
                raise ValueError(f"{trigger_event_relevance.title()} response must set playbook to null")
            return GenericPlaybook(
                display_answer="",
                current_bias="neutral",
                trigger_event_relevance=trigger_event_relevance,
                trigger_confidence=trigger_confidence_raw,
                selected_symbol="",
                selection_reason="",
                entry_plan=EntryPlan(
                    execute_now=False,
                    action_decision=build_empty_strategy_decision(),
                    scenario=None,
                ),
            )
        if not isinstance(playbook_payload, dict):
            raise ValueError("playbook must be a JSON object when trigger_event_relevance is relevant")
        data = {
            "trigger_event_relevance": trigger_event_relevance,
            "trigger_confidence": trigger_confidence_raw,
            "entry_plan": playbook_payload.get("entry_plan", {}),
        }
    elif trigger_event_relevance in {"unrelated", "duplicate"}:
        if set(data.keys()) != {"trigger_event_relevance", "trigger_confidence"}:
            raise ValueError(f"{trigger_event_relevance.title()} passive response must only include trigger_event_relevance and trigger_confidence")
        return GenericPlaybook(
            display_answer="",
            current_bias="neutral",
            trigger_event_relevance=trigger_event_relevance,
            trigger_confidence=trigger_confidence_raw,
            selected_symbol="",
            selection_reason="",
            entry_plan=EntryPlan(
                execute_now=False,
                action_decision=build_empty_strategy_decision(),
                scenario=None,
            ),
        )
    display_answer = str(data.get("display_answer", "")).strip()
    current_bias = str(data.get("current_bias", "")).strip()
    selected_symbol = str(data.get("selected_symbol", "") or "").strip().upper()
    selection_reason = str(data.get("selection_reason", "") or "").strip()
    entry_raw = data.get("entry_plan", {}) or {}
    if not isinstance(entry_raw, dict):
        raise ValueError("entry_plan is not a JSON object")
    entry_plan = EntryPlan(
        execute_now=bool(entry_raw.get("execute_now", False)),
        action_decision=validate_decision(entry_raw.get("action_decision", {})),
        scenario=(
            validate_entry_scenario(entry_raw.get("scenario", {}))
            if isinstance(entry_raw.get("scenario"), dict)
            else None
        ),
    )
    if entry_plan.execute_now and entry_plan.scenario is not None:
        raise ValueError("entry_plan.scenario must be null when execute_now is true")
    return GenericPlaybook(
        display_answer=display_answer,
        current_bias=current_bias,
        trigger_event_relevance=trigger_event_relevance,
        trigger_confidence=trigger_confidence_raw,
        selected_symbol=selected_symbol,
        selection_reason=selection_reason,
        entry_plan=entry_plan,
    )
