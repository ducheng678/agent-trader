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

from market_agent.structured_outputs import validate_passive_event_judge, validate_passive_technical_pricing



class PassiveWorkflowMixin:
    @staticmethod
    def _select_passive_event_judge_candidate(candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        valid_candidates = [
            candidate
            for candidate in list(candidates or [])
            if isinstance(candidate, dict)
            and str(candidate.get("status", "") or "") == "ok"
            and isinstance(candidate.get("validated_output"), dict)
        ]
        if not valid_candidates:
            messages = [
                str(candidate.get("error", "") or "").strip()
                for candidate in list(candidates or [])
                if isinstance(candidate, dict) and str(candidate.get("error", "") or "").strip()
            ]
            suffix = f": {'; '.join(messages)}" if messages else ""
            raise RuntimeError(f"Passive event judge produced no valid candidates{suffix}")

        def candidate_confidence(candidate: Dict[str, Any]) -> float:
            output = candidate.get("validated_output") if isinstance(candidate.get("validated_output"), dict) else {}
            return min(1.0, max(0.0, float(safe_float(output.get("trigger_confidence"), 0.0) or 0.0)))

        return max(valid_candidates, key=candidate_confidence)
    def _call_passive_event_judge_once(
        self,
        *,
        phase: str,
        trigger_event: Optional[Dict[str, Any]],
        recent_events: Optional[List[Dict[str, Any]]],
        market_mainline_context: Optional[Dict[str, Any]],
        reasoning_effort: str,
        trade_symbol: str,
        sample_index: int,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        selected_model = self._resolve_model("passive_event_trigger")
        playbook_trigger_event = self._sanitize_passive_recent_event_for_llm(trigger_event) if isinstance(trigger_event, dict) else None
        playbook_recent_events = [
            dict(item)
            for item in list(recent_events or [])
            if isinstance(item, dict)
        ]
        user_payload: Dict[str, Any] = {
            "trade_symbol": trade_symbol,
            "trigger_event": playbook_trigger_event,
        }
        if isinstance(market_mainline_context, dict) and market_mainline_context:
            user_payload["market_mainline_context"] = self._normalize_market_mainline_context(
                market_mainline_context,
                diagnostic_universe=getattr(self, "diagnostic_instrument_universe", DEFAULT_DIAGNOSTIC_INSTRUMENT_UNIVERSE),
                trade_symbol=trade_symbol,
            )
        if playbook_recent_events:
            user_payload["recent_events"] = playbook_recent_events
        input_messages = [
            {"role": "system", "content": [{"type": "input_text", "text": self._build_passive_event_judge_prompt(phase)}]},
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": json.dumps(user_payload, ensure_ascii=False, indent=2),
                    }
                ],
            },
        ]
        response = self._responses_create_with_retry(
            phase="passive_event_judge",
            timeout_seconds=self._resolve_request_timeout_seconds("passive_event_trigger"),
            model=selected_model,
            input=input_messages,
            tools=[],
            reasoning={"effort": reasoning_effort},
            text={"format": PASSIVE_EVENT_JUDGE_SCHEMA},
        )
        parsed_output = json.loads(response.output_text)
        validated_output = validate_passive_event_judge(parsed_output, relevance_threshold=self.passive_relevance_threshold)
        response_model = str(_response_attr(response, "model", selected_model) or selected_model)
        usage = extract_response_usage(response)
        web_search_tool_calls = count_web_search_tool_calls(response)
        web_search_calls = extract_web_search_call_details(response)
        usage_cost = estimate_openai_usage_cost(
            model=response_model,
            usage=usage,
            web_search_tool_calls=web_search_tool_calls,
        )
        return validated_output, {
            "request_messages": sanitize_response_input_messages(input_messages),
            "trigger_event_for_llm": playbook_trigger_event,
            "response_id": str(_response_attr(response, "id", "") or ""),
            "response_model": response_model,
            "raw_output_text": response.output_text,
            "parsed_output": parsed_output,
            "validated_output": validated_output,
            "usage": usage,
            "usage_cost": usage_cost,
            "web_search_tool_calls": web_search_tool_calls,
            "web_search_calls": web_search_calls,
            "tools": [],
            "phase": phase,
            "trigger_reason": "passive_event_trigger",
            "trade_symbol": trade_symbol,
            "sample_index": sample_index,
        }
    def _call_passive_event_judge(
        self,
        *,
        phase: str,
        trigger_event: Optional[Dict[str, Any]],
        recent_events: Optional[List[Dict[str, Any]]],
        market_mainline_context: Optional[Dict[str, Any]],
        reasoning_effort: str,
        trade_symbol: str,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        sample_count = 2

        def run_sample(sample_index: int) -> Dict[str, Any]:
            started_at = time.time()
            try:
                validated_output, debug = self._call_passive_event_judge_once(
                    phase=phase,
                    trigger_event=trigger_event,
                    recent_events=recent_events,
                    market_mainline_context=market_mainline_context,
                    reasoning_effort=reasoning_effort,
                    trade_symbol=trade_symbol,
                    sample_index=sample_index,
                )
                duration_seconds = max(0.0, time.time() - started_at)
                return {
                    "sample_index": sample_index,
                    "status": "ok",
                    "duration_seconds": duration_seconds,
                    "response_id": debug.get("response_id", ""),
                    "response_model": debug.get("response_model", ""),
                    "raw_output_text": debug.get("raw_output_text", ""),
                    "parsed_output": debug.get("parsed_output"),
                    "validated_output": validated_output,
                    "usage": debug.get("usage"),
                    "usage_cost": debug.get("usage_cost"),
                    "web_search_tool_calls": debug.get("web_search_tool_calls", 0),
                    "web_search_calls": debug.get("web_search_calls") or [],
                    "debug": debug,
                }
            except Exception as exc:
                return {
                    "sample_index": sample_index,
                    "status": "error",
                    "duration_seconds": max(0.0, time.time() - started_at),
                    "error": f"{type(exc).__name__}: {exc}",
                }

        with ThreadPoolExecutor(max_workers=sample_count, thread_name_prefix="passive-step1-sample") as executor:
            candidates = list(executor.map(run_sample, range(1, sample_count + 1)))

        selected_candidate = self._select_passive_event_judge_candidate(candidates)
        selected_debug = dict(selected_candidate.get("debug") or {})
        selected_output = dict(selected_candidate.get("validated_output") or {})
        candidate_summaries = [
            {
                key: value
                for key, value in dict(candidate).items()
                if key != "debug"
            }
            for candidate in candidates
        ]
        candidate_relevances = [
            str((candidate.get("validated_output") or {}).get("trigger_event_relevance", "") or "").strip().lower()
            for candidate in candidates
            if isinstance(candidate, dict) and candidate.get("status") == "ok"
        ]
        duplicate_candidate_count = sum(1 for relevance in candidate_relevances if relevance == "duplicate")
        unrelated_candidate_count = sum(1 for relevance in candidate_relevances if relevance == "unrelated")
        relevant_candidate_count = sum(1 for relevance in candidate_relevances if relevance == "relevant")
        duplicate_relevant_conflict = duplicate_candidate_count > 0 and relevant_candidate_count > 0
        unrelated_relevant_conflict = unrelated_candidate_count > 0 and relevant_candidate_count > 0
        selected_raw_confidence = min(
            1.0,
            max(0.0, float(safe_float(selected_output.get("trigger_confidence"), 0.0) or 0.0)),
        )
        confidence_adjustment: Dict[str, Any] = {
            "applied": False,
            "reason": "",
            "multiplier": 1.0,
            "raw_trigger_confidence": selected_raw_confidence,
            "effective_trigger_confidence": selected_raw_confidence,
        }
        selected_relevance = str(selected_output.get("trigger_event_relevance", "") or "").strip().lower()
        conflict_adjustment_reason = ""
        conflict_adjustment_multiplier = 1.0
        if selected_relevance == "relevant":
            if duplicate_relevant_conflict:
                conflict_adjustment_reason = "duplicate_relevant_conflict_multiplier"
                conflict_adjustment_multiplier = getattr(self, "passive_duplicate_relevant_conflict_multiplier", 0.5)
            elif unrelated_relevant_conflict:
                conflict_adjustment_reason = "unrelated_relevant_conflict_multiplier"
                conflict_adjustment_multiplier = getattr(self, "passive_unrelated_relevant_conflict_multiplier", 0.5)
        if conflict_adjustment_reason:
            multiplier = min(1.0, max(0.0, float(conflict_adjustment_multiplier or 0.0)))
            adjusted_confidence = min(1.0, max(0.0, selected_raw_confidence * multiplier))
            selected_output["trigger_confidence"] = adjusted_confidence
            confidence_adjustment = {
                "applied": True,
                "reason": conflict_adjustment_reason,
                "multiplier": multiplier,
                "raw_trigger_confidence": selected_raw_confidence,
                "effective_trigger_confidence": adjusted_confidence,
                "effective_trigger_event_relevance": selected_output.get("trigger_event_relevance"),
                "effective_action": selected_output.get("action"),
            }
        selected_effective_confidence = min(
            1.0,
            max(0.0, float(safe_float(selected_output.get("trigger_confidence"), 0.0) or 0.0)),
        )
        selected_effective_relevance = str(selected_output.get("trigger_event_relevance", "") or "").strip().lower()
        relevance_gate: Dict[str, Any] = {
            "applied": False,
            "reason": "",
            "threshold": self.passive_relevance_threshold,
            "trigger_confidence": selected_effective_confidence,
            "trigger_event_relevance_before": selected_effective_relevance,
            "action_before": selected_output.get("action"),
        }
        if selected_effective_relevance == "relevant" and selected_effective_confidence < self.passive_relevance_threshold:
            selected_output["trigger_event_relevance"] = "unrelated"
            selected_output["action"] = "no_trade"
            relevance_gate.update(
                {
                    "applied": True,
                    "reason": "below_passive_relevance_threshold",
                    "trigger_event_relevance_after": "unrelated",
                    "action_after": "no_trade",
                }
            )
        combined_usage: Optional[Dict[str, Any]] = None
        combined_usage_cost: Optional[Dict[str, Any]] = None
        combined_web_search_tool_calls = 0
        combined_web_search_calls: List[Dict[str, Any]] = []
        response_ids: List[str] = []
        for candidate in candidates:
            if not isinstance(candidate, dict) or candidate.get("status") != "ok":
                continue
            combined_usage = merge_usage_dicts(combined_usage, candidate.get("usage"))
            combined_usage_cost = merge_usage_costs(combined_usage_cost, candidate.get("usage_cost"))
            combined_web_search_tool_calls += int(candidate.get("web_search_tool_calls", 0) or 0)
            combined_web_search_calls.extend(list(candidate.get("web_search_calls") or []))
            response_id = str(candidate.get("response_id", "") or "").strip()
            if response_id:
                response_ids.append(response_id)

        selected_debug.update(
            {
                "response_id": "+".join(response_ids) or selected_debug.get("response_id", ""),
                "usage": combined_usage,
                "usage_cost": combined_usage_cost,
                "web_search_tool_calls": combined_web_search_tool_calls,
                "web_search_calls": combined_web_search_calls,
                "passive_event_judge_candidates": candidate_summaries,
                "passive_event_judge_selection_rule": "max_trigger_confidence",
                "passive_event_judge_selected_sample_index": selected_candidate.get("sample_index"),
                "passive_event_judge_sample_count": sample_count,
                "passive_event_judge_duplicate_candidate_count": duplicate_candidate_count,
                "passive_event_judge_unrelated_candidate_count": unrelated_candidate_count,
                "passive_event_judge_relevant_candidate_count": relevant_candidate_count,
                "passive_event_judge_duplicate_relevant_conflict": duplicate_relevant_conflict,
                "passive_event_judge_unrelated_relevant_conflict": unrelated_relevant_conflict,
                "passive_event_judge_confidence_adjustment": confidence_adjustment,
                "passive_event_judge_relevance_gate": relevance_gate,
                "passive_event_judge_selected_raw_trigger_confidence": selected_raw_confidence,
                "passive_event_judge_selected_effective_trigger_confidence": selected_output.get("trigger_confidence"),
                "passive_event_judge_duplicate_relevant_conflict_multiplier": getattr(self, "passive_duplicate_relevant_conflict_multiplier", 0.5),
                "passive_event_judge_unrelated_relevant_conflict_multiplier": getattr(self, "passive_unrelated_relevant_conflict_multiplier", 0.5),
                "validated_output": selected_output,
            }
        )
        return selected_output, selected_debug
    def _call_passive_technical_pricing(
        self,
        *,
        phase: str,
        action: str,
        chart_input_context: Optional[Dict[str, Any]],
        reasoning_effort: str,
    ) -> Tuple[Dict[str, float], Dict[str, Any]]:
        selected_model = self._resolve_model("passive_event_trigger")
        chart_context_debug = normalize_image_input_context(chart_input_context)
        chart_summaries = [dict(item) for item in list((chart_input_context or {}).get("chart_summaries") or []) if isinstance(item, dict)]
        user_payload = {
            "action": action,
            "chart_summaries": chart_summaries,
        }
        user_content = [
            {
                "type": "input_text",
                "text": json.dumps(user_payload, ensure_ascii=False, indent=2),
            }
        ]
        if any(isinstance(item, dict) and item.get("type") == "input_image" for item in list((chart_input_context or {}).get("input_images") or [])):
            user_content.append(
                {
                    "type": "input_text",
                    "text": "The following uploaded images are chart screenshots for the fixed action, in the same order as chart_summaries.",
                }
            )
        for image_item in list((chart_input_context or {}).get("input_images") or []):
            if isinstance(image_item, dict) and image_item.get("type") == "input_image":
                user_content.append(dict(image_item))
        input_messages = [
            {"role": "system", "content": [{"type": "input_text", "text": self._build_passive_technical_pricing_prompt(phase)}]},
            {"role": "user", "content": user_content},
        ]
        response = self._responses_create_with_retry(
            phase="passive_technical_pricing",
            timeout_seconds=self._resolve_request_timeout_seconds("passive_event_trigger"),
            model=selected_model,
            input=input_messages,
            tools=[],
            reasoning={"effort": reasoning_effort},
            text={"format": PASSIVE_TECHNICAL_PRICING_SCHEMA},
        )
        parsed_output = json.loads(response.output_text)
        validated_output = validate_passive_technical_pricing(parsed_output)
        response_model = str(_response_attr(response, "model", selected_model) or selected_model)
        usage = extract_response_usage(response)
        web_search_tool_calls = count_web_search_tool_calls(response)
        web_search_calls = extract_web_search_call_details(response)
        usage_cost = estimate_openai_usage_cost(
            model=response_model,
            usage=usage,
            web_search_tool_calls=web_search_tool_calls,
            image_input_context=chart_input_context,
        )
        return validated_output, {
            "request_messages": sanitize_response_input_messages(input_messages),
            "response_id": str(_response_attr(response, "id", "") or ""),
            "response_model": response_model,
            "raw_output_text": response.output_text,
            "parsed_output": parsed_output,
            "validated_output": validated_output,
            "usage": usage,
            "usage_cost": usage_cost,
            "chart_screenshot_debug": {
                "symbol": chart_context_debug.get("symbol", ""),
                "display_name": chart_context_debug.get("display_name", ""),
                "detail": chart_context_debug.get("detail", ""),
                "image_count": chart_context_debug.get("count", 0),
                "images": list(chart_context_debug.get("rendered_images") or []),
                "note": str((chart_input_context or {}).get("note", "") or ""),
            },
            "web_search_tool_calls": web_search_tool_calls,
            "web_search_calls": web_search_calls,
            "tools": [],
            "phase": phase,
            "trigger_reason": "passive_event_trigger",
            "chart_summaries": chart_summaries,
        }
    def _call_passive_two_step_model(
        self,
        *,
        phase: str,
        trigger_event: Optional[Dict[str, Any]],
        recent_events: Optional[List[Dict[str, Any]]],
        trade_symbol_context: Dict[str, Any],
        active_symbol: str,
        market_mainline_context: Optional[Dict[str, Any]],
        chart_input_context: Optional[Dict[str, Any]],
        reasoning_effort: str,
        trade_symbol_label: str,
        prefetched_passive_event_judge: Optional[Dict[str, Any]] = None,
    ) -> Tuple[GenericPlaybook, Dict[str, Any]]:
        trade_symbol_context = dict(trade_symbol_context or {}) if isinstance(trade_symbol_context, dict) else {}
        trade_symbol = str(
            trade_symbol_label
            or trade_symbol_context.get("display_name", "")
            or trade_symbol_context.get("trade_symbol_key", "")
            or trade_symbol_context.get("candidate_key", "")
            or ""
        ).strip()
        prefetched = prefetched_passive_event_judge if isinstance(prefetched_passive_event_judge, dict) else {}
        prefetched_symbol = normalize_candidate_key(prefetched.get("trade_symbol", ""))
        current_symbol = normalize_candidate_key(trade_symbol)
        trade_confidence_threshold = self._passive_trade_confidence_threshold(trade_symbol)

        def run_judge():
            if (
                prefetched
                and prefetched_symbol
                and current_symbol
                and prefetched_symbol == current_symbol
                and isinstance(prefetched.get("judge_output"), dict)
                and isinstance(prefetched.get("judge_debug"), dict)
            ):
                return dict(prefetched.get("judge_output") or {}), {
                    **dict(prefetched.get("judge_debug") or {}),
                    "prefetched": True,
                    "prefetch_started_at": prefetched.get("started_at"),
                    "prefetch_completed_at": prefetched.get("completed_at"),
                }
            return self._call_passive_event_judge(
                phase=phase,
                trigger_event=trigger_event,
                recent_events=recent_events,
                market_mainline_context=market_mainline_context,
                reasoning_effort=reasoning_effort,
                trade_symbol=trade_symbol,
            )

        def should_run_pricing(judge_result) -> bool:
            output, _ = judge_result
            relevance = str(output.get("trigger_event_relevance", "unrelated") or "unrelated").strip().lower()
            confidence = float(output.get("trigger_confidence", 0.0) or 0.0)
            action_value = str(output.get("action", "no_trade") or "no_trade").strip().lower()
            return relevance == "relevant" and action_value != "no_trade" and confidence >= trade_confidence_threshold

        def run_pricing(judge_result):
            output, _ = judge_result
            action_value = str(output.get("action", "no_trade") or "no_trade").strip().lower()
            return self._call_passive_technical_pricing(
                phase=phase,
                action=action_value,
                chart_input_context=chart_input_context,
                reasoning_effort=reasoning_effort,
            )

        workflow = getattr(self, "llm_workflow", None)
        if workflow is None:
            from market_agent.llm_workflow import LLMWorkflow

            workflow = LLMWorkflow()
            self.llm_workflow = workflow
        judge_result, pricing_result = workflow.run_passive(
            judge=run_judge,
            should_price=should_run_pricing,
            price=run_pricing,
            assemble=lambda result, pricing: (result, pricing),
        )
        judge_output, judge_debug = judge_result
        trigger_event_relevance = str(judge_output.get("trigger_event_relevance", "unrelated") or "unrelated").strip().lower()
        trigger_confidence = float(judge_output.get("trigger_confidence", 0.0) or 0.0)
        action = str(judge_output.get("action", "no_trade") or "no_trade").strip().lower()
        should_price = (
            trigger_event_relevance == "relevant"
            and action != "no_trade"
            and trigger_confidence >= trade_confidence_threshold
        )
        passive_trade_gate = {
            "should_price": bool(should_price),
            "threshold": trade_confidence_threshold,
            "trigger_confidence": trigger_confidence,
            "trigger_event_relevance": trigger_event_relevance,
            "action": action,
            "reason": "",
        }
        if trigger_event_relevance == "relevant" and action != "no_trade" and trigger_confidence < trade_confidence_threshold:
            passive_trade_gate["reason"] = "below_trade_confidence_threshold"
        pricing_output, pricing_debug = pricing_result or (
            {"entry_price": 0.0, "stop_loss_price": 0.0},
            {},
        )
        entry_price = float(pricing_output.get("entry_price", 0.0) or 0.0)
        stop_loss_price = float(pricing_output.get("stop_loss_price", 0.0) or 0.0)
        playbook = GenericPlaybook(
            display_answer="",
            current_bias="",
            trigger_event_relevance=trigger_event_relevance,
            trigger_confidence=trigger_confidence,
            selected_symbol=trade_symbol,
            selection_reason="",
            entry_plan=EntryPlan(
                execute_now=bool(should_price),
                action_decision=StrategyDecision(
                    action=action if should_price else "no_trade",
                    suggested_notional_usd=0.0,
                    entry_price=entry_price,
                    stop_loss_price=stop_loss_price,
                    planned_margin_used_usd=0.0,
                    planned_max_loss_usd=0.0,
                    requested_leverage=0,
                ),
                scenario=None,
            ),
        )
        assembled_output = {
            "trigger_event_relevance": trigger_event_relevance,
            "trigger_confidence": trigger_confidence,
            "playbook": {
                "entry_plan": {
                    "execute_now": bool(should_price),
                    "action_decision": {
                        "action": action if should_price else "no_trade",
                        "entry_price": entry_price,
                        "stop_loss_price": stop_loss_price,
                    },
                    "scenario": None,
                }
            },
        }
        combined_usage = merge_usage_dicts(judge_debug.get("usage"), pricing_debug.get("usage"))
        combined_usage_cost = merge_usage_costs(judge_debug.get("usage_cost"), pricing_debug.get("usage_cost"))
        combined_web_search_calls = int(judge_debug.get("web_search_tool_calls", 0) or 0) + int(pricing_debug.get("web_search_tool_calls", 0) or 0)
        combined_web_search_details = list(judge_debug.get("web_search_calls") or []) + list(pricing_debug.get("web_search_calls") or [])
        request_messages = pricing_debug.get("request_messages") if should_price else judge_debug.get("request_messages")
        response_ids = [str(item) for item in (judge_debug.get("response_id"), pricing_debug.get("response_id")) if str(item or "").strip()]
        return playbook, {
            "request_messages": request_messages or [],
            "passive_event_judge_request_messages": judge_debug.get("request_messages") or [],
            "passive_technical_pricing_request_messages": pricing_debug.get("request_messages") or [],
            "response_id": "+".join(response_ids),
            "response_model": pricing_debug.get("response_model") or judge_debug.get("response_model") or self._resolve_model("passive_event_trigger"),
            "raw_output_text": json.dumps(assembled_output, ensure_ascii=False),
            "parsed_output": assembled_output,
            "passive_event_judge_raw_output_text": judge_debug.get("raw_output_text", ""),
            "passive_event_judge_parsed_output": judge_debug.get("parsed_output"),
            "passive_event_judge_validated_output": judge_debug.get("validated_output"),
            "passive_event_judge_candidates": judge_debug.get("passive_event_judge_candidates") or [],
            "passive_event_judge_selection_rule": judge_debug.get("passive_event_judge_selection_rule", ""),
            "passive_event_judge_selected_sample_index": judge_debug.get("passive_event_judge_selected_sample_index"),
            "passive_event_judge_sample_count": judge_debug.get("passive_event_judge_sample_count", 1),
            "passive_event_judge_duplicate_candidate_count": judge_debug.get("passive_event_judge_duplicate_candidate_count", 0),
            "passive_event_judge_unrelated_candidate_count": judge_debug.get("passive_event_judge_unrelated_candidate_count", 0),
            "passive_event_judge_relevant_candidate_count": judge_debug.get("passive_event_judge_relevant_candidate_count", 0),
            "passive_event_judge_duplicate_relevant_conflict": bool(judge_debug.get("passive_event_judge_duplicate_relevant_conflict", False)),
            "passive_event_judge_unrelated_relevant_conflict": bool(judge_debug.get("passive_event_judge_unrelated_relevant_conflict", False)),
            "passive_event_judge_confidence_adjustment": judge_debug.get("passive_event_judge_confidence_adjustment") or {},
            "passive_event_judge_selected_raw_trigger_confidence": judge_debug.get("passive_event_judge_selected_raw_trigger_confidence"),
            "passive_event_judge_selected_effective_trigger_confidence": judge_debug.get("passive_event_judge_selected_effective_trigger_confidence"),
            "passive_event_judge_duplicate_relevant_conflict_multiplier": judge_debug.get("passive_event_judge_duplicate_relevant_conflict_multiplier", 1.0),
            "passive_event_judge_unrelated_relevant_conflict_multiplier": judge_debug.get("passive_event_judge_unrelated_relevant_conflict_multiplier", 1.0),
            "passive_event_judge_trigger_event_for_llm": judge_debug.get("trigger_event_for_llm"),
            "passive_technical_pricing_raw_output_text": pricing_debug.get("raw_output_text", ""),
            "passive_technical_pricing_parsed_output": pricing_debug.get("parsed_output"),
            "passive_technical_pricing_validated_output": pricing_debug.get("validated_output"),
            "usage": combined_usage,
            "usage_cost": combined_usage_cost,
            "chart_screenshot_debug": pricing_debug.get("chart_screenshot_debug") or {},
            "web_search_tool_calls": combined_web_search_calls,
            "web_search_calls": combined_web_search_details,
            "tools": [],
            "phase": phase,
            "trigger_reason": "passive_event_trigger",
            "trade_symbol": trade_symbol,
            "chart_summaries": pricing_debug.get("chart_summaries") or [],
            "trade_symbol_context": trade_symbol_context,
            "active_symbol": active_symbol,
            "passive_two_step": True,
            "passive_step2_executed": bool(should_price),
            "passive_trade_gate": passive_trade_gate,
            "passive_step1_prefetched": bool(judge_debug.get("prefetched")),
        }
