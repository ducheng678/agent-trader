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

from market_agent.agent_context import AgentContextMixin
from market_agent.memory_state import MemoryStateMixin
from market_agent.model_routing import ModelRoutingMixin, normalize_reasoning_effort
from market_agent.passive_workflow import PassiveWorkflowMixin
from market_agent.prompt_context import PromptContextMixin
from market_agent.retrieval_rag import RetrievalRAGMixin
from market_agent.structured_outputs import StructuredOutputMixin, validate_playbook
from market_agent.tool_calling import ToolCallingMixin

class DiscretionaryLLMEngine(
    ModelRoutingMixin,
    PromptContextMixin,
    ToolCallingMixin,
    MemoryStateMixin,
    StructuredOutputMixin,
    AgentContextMixin,
    PassiveWorkflowMixin,
    RetrievalRAGMixin,
):
    def __init__(self):
        api_key = (os.getenv("OPENAI_API_KEY", "") or "").strip()
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 未设置")
        try:
            from market_agent.langchain_runtime import LangChainResponsesRuntime
            from market_agent.llm_workflow import LLMWorkflow
        except ImportError as exc:
            raise RuntimeError("Missing LangChain dependencies. Install: pip install -r requirements.txt") from exc
        self.client = LangChainResponsesRuntime(api_key=api_key)
        self.llm_workflow = LLMWorkflow()
        self.active_openai_request_timeout_seconds = max(1.0, float(os.getenv("OPENAI_ACTIVE_REQUEST_TIMEOUT_SECONDS", os.getenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "180")) or 180.0))
        self.passive_openai_request_timeout_seconds = max(1.0, float(os.getenv("OPENAI_PASSIVE_REQUEST_TIMEOUT_SECONDS", os.getenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "60")) or 60.0))
        self.openai_max_attempts = max(1, int(os.getenv("OPENAI_MAX_ATTEMPTS", "3") or 3))
        self.openai_retry_delay_seconds = max(0.0, float(os.getenv("OPENAI_RETRY_DELAY_SECONDS", "2") or 2.0))
        self.active_model = os.getenv("OPENAI_ACTIVE_MODEL", "gpt-5.4")
        self.passive_model = os.getenv("OPENAI_PASSIVE_MODEL", "gpt-5.4")
        self.prompt_cache_enabled = str(os.getenv("OPENAI_PROMPT_CACHE_ENABLED", "true")).strip().lower() in {"1", "true", "yes", "on"}
        self.prompt_cache_key_prefix = str(os.getenv("OPENAI_PROMPT_CACHE_KEY_PREFIX", "market-agent") or "market-agent").strip()
        self.symbol = ""
        self.default_search_mode = os.getenv("OPENAI_SEARCH_MODE", "context_only").lower()
        self.active_search_mode = os.getenv("OPENAI_ACTIVE_SEARCH_MODE", self.default_search_mode).lower()
        self.passive_search_mode = os.getenv("OPENAI_PASSIVE_SEARCH_MODE", self.default_search_mode).lower()
        self.active_reasoning_effort = normalize_reasoning_effort(os.getenv("OPENAI_ACTIVE_REASONING_EFFORT", "high"), "high")
        self.passive_reasoning_effort = normalize_reasoning_effort(os.getenv("OPENAI_PASSIVE_REASONING_EFFORT", "medium"), "medium")
        self.local_size_from_stop = str(os.getenv("LOCAL_SIZE_FROM_STOP", "true")).strip().lower() in {"1", "true", "yes", "on"}
        self.force_active_news_context = str(os.getenv("OPENAI_FORCE_ACTIVE_NEWS_CONTEXT", "true")).strip().lower() in {"1", "true", "yes", "on"}
        self.force_passive_news_context = str(os.getenv("OPENAI_FORCE_PASSIVE_NEWS_CONTEXT", "false")).strip().lower() in {"1", "true", "yes", "on"}
        self.include_chart_images = str(os.getenv("OPENAI_INCLUDE_CHART_IMAGES", "true")).strip().lower() in {"1", "true", "yes", "on"}
        self.include_passive_chart_images = str(os.getenv("OPENAI_INCLUDE_PASSIVE_CHART_IMAGES", "false")).strip().lower() in {"1", "true", "yes", "on"}
        self.passive_recent_materially_new_event_limit = max(0, int(os.getenv("PASSIVE_RECENT_MATERIALLY_NEW_EVENT_LIMIT", "0") or 0))
        self.passive_relevance_threshold = min(1.0, max(0.0, float(safe_float(os.getenv("PASSIVE_RELEVANCE_THRESHOLD"), 0.20) or 0.0)))
        self.passive_duplicate_relevant_conflict_multiplier = min(
            1.0,
            max(0.0, float(safe_float(os.getenv("PASSIVE_DUPLICATE_RELEVANT_CONFLICT_MULTIPLIER"), 0.5) or 0.0)),
        )
        self.passive_unrelated_relevant_conflict_multiplier = min(
            1.0,
            max(0.0, float(safe_float(os.getenv("PASSIVE_UNRELATED_RELEVANT_CONFLICT_MULTIPLIER"), 0.5) or 0.0)),
        )
        self.execute_now_confidence_threshold = min(1.0, max(0.0, float(os.getenv("ENTRY_EXECUTE_NOW_CONFIDENCE_THRESHOLD", "0.70") or 0.70)))
        self.last_call_debug: Dict[str, Any] = {}
        self.audit_callback: Optional[Any] = None
        self.chart_context_builder: Optional[Any] = None
        self.helper_market_mainline_latest_path = Path(os.getenv("HELPER_MARKET_MAINLINE_LATEST_PATH", "logs/latest_helper_market_mainline.json"))
        self.helper_market_mainline_latest_path.parent.mkdir(parents=True, exist_ok=True)
        self.helper_materially_new_first_events_path = Path(
            os.getenv("HELPER_MATERIALLY_NEW_FIRST_EVENTS_PATH", "logs/helper_materially_new_first_events.jsonl")
        )
        self.helper_materially_new_first_events_path.parent.mkdir(parents=True, exist_ok=True)
        self.helper_prior_materially_new_trigger_threshold = max(
            0,
            int(os.getenv("HELPER_PRIOR_MATERIALLY_NEW_TRIGGER_THRESHOLD", "30") or 30),
        )
        self.helper_prior_materially_new_max_items = max(
            0,
            int(os.getenv("HELPER_PRIOR_MATERIALLY_NEW_MAX_ITEMS", "5") or 5),
        )
        self.diagnostic_instrument_universe = parse_symbol_universe(
            os.getenv("HELPER_DIAGNOSTIC_INSTRUMENT_UNIVERSE", ""),
            DEFAULT_DIAGNOSTIC_INSTRUMENT_UNIVERSE,
        )
        self.latest_helper_market_mainline_context: Dict[str, Any] = {}
        self.latest_helper_market_mainline_debug: Dict[str, Any] = {}
        self.latest_helper_materially_new_first_events: List[Dict[str, Any]] = []
        self._hydrate_helper_context_from_disk()
    def get_playbook(

        self,
        user_query: str,
        event_tape: List[Dict[str, Any]],
        trigger_reason: str,
        trigger_event: Optional[Dict[str, Any]],
        recent_events: Optional[List[Dict[str, Any]]] = None,
        trade_symbol_context: Optional[Dict[str, Any]] = None,
        active_symbol: Optional[str] = None,
        has_live_position: bool = False,
        prefetched_passive_event_judge: Optional[Dict[str, Any]] = None,
        ) -> Tuple[GenericPlaybook, str]:
        search_mode = self._resolve_search_mode(trigger_reason)
        reasoning_effort = self._resolve_reasoning_effort(trigger_reason)
        phase = "fast"
        tools: List[dict] = []
        market_mainline_context: Optional[Dict[str, Any]] = None
        market_mainline_debug: Dict[str, Any] = {}
        market_mainline_usage_is_current_call = False
        chart_input_context: Optional[Dict[str, Any]] = None
        has_live_position = bool(has_live_position)
        if search_mode == "context_only":
            phase = "context_only"
        elif search_mode == "always":
            phase = "verified"
        if trigger_reason == "passive_event_trigger":
            market_mainline_context, market_mainline_debug = self._get_cached_helper_market_mainline_context(
                trade_symbol_context=dict(trade_symbol_context or {}),
                active_symbol=(active_symbol or self.symbol),
            )
        elif self._should_force_news_context(trigger_reason, search_mode) and not has_live_position:
            if list(event_tape or []):
                market_mainline_context, market_mainline_debug = self._build_market_news_context(
                    user_query=user_query,
                    recent_events=event_tape,
                    trade_symbol_context=dict(trade_symbol_context or {}),
                    active_symbol=(active_symbol or self.symbol),
                    trigger_reason=trigger_reason,
                    trigger_event=trigger_event,
                )
                market_mainline_usage_is_current_call = True
            else:
                market_mainline_context, market_mainline_debug = self._get_cached_helper_market_mainline_context(
                    trade_symbol_context=dict(trade_symbol_context or {}),
                    active_symbol=(active_symbol or self.symbol),
                )
            self.last_call_debug = {
                "market_mainline_context": market_mainline_context,
                "market_mainline_call_debug": market_mainline_debug,
            }
            audit_callback = getattr(self, "audit_callback", None)
            if callable(audit_callback):
                try:
                    audit_callback(
                        "market_mainline_call_debug",
                        {
                            **dict(market_mainline_debug),
                            "market_mainline_context": market_mainline_context,
                        },
                    )
                except Exception:
                    pass
            tools = []
        include_chart_images = bool(getattr(self, "include_chart_images", False))
        if trigger_reason == "passive_event_trigger":
            include_chart_images = bool(getattr(self, "include_passive_chart_images", False))
        builder = getattr(self, "chart_context_builder", None)
        passive_prefetched_should_price = (
            self._passive_prefetched_judge_should_price(
                prefetched_passive_event_judge,
                trade_symbol=str(
                    (prefetched_passive_event_judge or {}).get("trade_symbol", "")
                    or (trade_symbol_context or {}).get("display_name", "")
                    or (trade_symbol_context or {}).get("trade_symbol_key", "")
                    or (trade_symbol_context or {}).get("candidate_key", "")
                    or ""
                ),
            )
            if trigger_reason == "passive_event_trigger"
            else None
        )
        skip_chart_context = trigger_reason == "passive_event_trigger" and passive_prefetched_should_price is False
        passive_chart_prefetch_debug: Dict[str, Any] = {}
        if trigger_reason == "passive_event_trigger" and not skip_chart_context:
            chart_input_context, passive_chart_prefetch_debug = self._resolve_prefetched_passive_chart_context(
                prefetched_passive_event_judge,
                include_images=include_chart_images,
            )
        if chart_input_context is None and callable(builder) and not skip_chart_context:
            visual_candidate = self._select_visual_trade_symbol_context(
                dict(trade_symbol_context or {}),
                active_symbol=(active_symbol or self.symbol),
                market_news_debug=market_mainline_debug,
            )
            if visual_candidate is not None:
                try:
                    visual_candidate_payload = dict(visual_candidate)
                    if trigger_reason == "passive_event_trigger":
                        visual_candidate_payload["_chart_mode"] = "passive"
                    chart_input_context = builder(visual_candidate_payload)
                    if isinstance(chart_input_context, dict) and not include_chart_images:
                        chart_input_context = {
                            **dict(chart_input_context),
                            "input_images": [],
                            "debug_images": [],
                            "image_count": 0,
                        }
                except Exception:
                    chart_input_context = None
        playbook_selection_context = self._sanitize_playbook_trade_symbol_context(dict(trade_symbol_context or {}))
        playbook_trade_symbol_context = self._select_playbook_trade_symbol_context(
            playbook_selection_context,
            active_symbol=(active_symbol or self.symbol),
            market_mainline_context=market_mainline_context,
        )
        if playbook_trade_symbol_context is None:
            skipped_playbook = GenericPlaybook(
                display_answer="",
                current_bias="neutral",
                trigger_event_relevance="unrelated" if trigger_reason == "passive_event_trigger" else "not_applicable",
                trigger_confidence=0.0 if trigger_reason == "passive_event_trigger" else None,
                entry_plan=EntryPlan(
                    execute_now=False,
                    action_decision=build_empty_strategy_decision(),
                    scenario=None,
                ),
            )
            self.last_call_debug = {
                "market_mainline_context": market_mainline_context,
                "market_mainline_call_debug": market_mainline_debug,
                "trade_symbol_context": {},
                "validated_playbook": skipped_playbook.to_dict(),
                "capped_playbook": skipped_playbook.to_dict(),
                "execution_view": build_playbook_execution_view(skipped_playbook),
                "mode": "skipped_no_trade_symbol_context",
                "usage": {},
                "usage_cost": {},
                "web_search_tool_calls": 0,
                "web_search_calls": [],
                "llm_payload_market_only": True,
                "reasoning_effort": reasoning_effort,
                "skip_reason": "no_playbook_trade_symbol_context",
            }
            return skipped_playbook, "skipped_no_trade_symbol_context"
        playbook_symbol_label = str(
            (playbook_trade_symbol_context or {}).get("display_name", "")
            or (playbook_trade_symbol_context or {}).get("trade_symbol_key", "")
            or (playbook_trade_symbol_context or {}).get("candidate_key", "")
            or ""
        ).strip()
        playbook_user_query = build_default_query(symbol=playbook_symbol_label)
        playbook_active_symbol = canonicalize_execution_symbol((playbook_trade_symbol_context or {}).get("execution_symbol", "")) or (active_symbol or self.symbol)
        if trigger_reason == "passive_event_trigger":
            playbook, call_debug = self._call_passive_two_step_model(
                phase=phase,
                trigger_event=trigger_event,
                recent_events=recent_events,
                trade_symbol_context=playbook_trade_symbol_context,
                active_symbol=playbook_active_symbol,
                market_mainline_context=market_mainline_context,
                chart_input_context=chart_input_context,
                reasoning_effort=reasoning_effort,
                trade_symbol_label=playbook_symbol_label,
                prefetched_passive_event_judge=prefetched_passive_event_judge,
            )
            if passive_chart_prefetch_debug:
                call_debug["passive_chart_context_prefetch"] = passive_chart_prefetch_debug
        else:
            playbook, call_debug = self._call_model(
                user_query=playbook_user_query,
                event_tape=event_tape,
                tools=tools,
                phase=phase,
                trigger_reason=trigger_reason,
                trade_symbol_context=playbook_trade_symbol_context,
                active_symbol=playbook_active_symbol,
                market_mainline_context=market_mainline_context,
                chart_input_context=chart_input_context,
                reasoning_effort=reasoning_effort,
                trade_symbol_label=playbook_symbol_label,
            )
        local_selected_symbol = str(
            (playbook_trade_symbol_context or {}).get("display_name", "")
            or (playbook_trade_symbol_context or {}).get("trade_symbol_key", "")
            or (playbook_trade_symbol_context or {}).get("candidate_key", "")
            or ""
        ).strip()
        if local_selected_symbol:
            playbook.selected_symbol = local_selected_symbol
        playbook_execution_symbol = canonicalize_execution_symbol((playbook_trade_symbol_context or {}).get("execution_symbol", "")) or canonicalize_execution_symbol(active_symbol or self.symbol)
        if playbook_execution_symbol:
            playbook = self._normalize_playbook_prices_for_symbol(playbook, playbook_execution_symbol)
        mode_map = {"fast": "raw_context_only", "context_only": "context_enriched_with_web", "verified": "verified_with_web"}
        validated_playbook = playbook.to_dict()
        capped_playbook = self._cap_playbook(playbook)
        if market_mainline_usage_is_current_call:
            combined_usage = merge_usage_dicts(
                market_mainline_debug.get("usage"),
                call_debug.get("usage"),
            )
            combined_usage_cost = merge_usage_costs(
                market_mainline_debug.get("usage_cost"),
                call_debug.get("usage_cost"),
            )
            combined_web_search_calls = int(market_mainline_debug.get("web_search_tool_calls", 0) or 0) + int(call_debug.get("web_search_tool_calls", 0) or 0)
            combined_web_search_details = list(market_mainline_debug.get("web_search_calls") or []) + list(call_debug.get("web_search_calls") or [])
        else:
            combined_usage = dict(call_debug.get("usage") or {})
            combined_usage_cost = dict(call_debug.get("usage_cost") or {})
            combined_web_search_calls = int(call_debug.get("web_search_tool_calls", 0) or 0)
            combined_web_search_details = list(call_debug.get("web_search_calls") or [])
        self.last_call_debug = {
            **call_debug,
            "market_mainline_context": market_mainline_context,
            "market_mainline_call_debug": market_mainline_debug,
            "market_mainline_usage_is_current_call": market_mainline_usage_is_current_call,
            "market_mainline_web_search_budget": market_mainline_debug.get("web_search_budget"),
            "market_mainline_web_search_analysis": market_mainline_debug.get("web_search_analysis"),
            "chart_screenshot_debug": call_debug.get("chart_screenshot_debug") or {},
            "validated_playbook": validated_playbook,
            "capped_playbook": capped_playbook.to_dict(),
            "execution_view": build_playbook_execution_view(capped_playbook),
            "mode": mode_map[phase],
            "usage": combined_usage,
            "usage_cost": combined_usage_cost if combined_usage_cost else call_debug.get("usage_cost"),
            "web_search_tool_calls": combined_web_search_calls,
            "web_search_calls": combined_web_search_details,
            "llm_payload_market_only": True,
            "reasoning_effort": reasoning_effort,
        }
        return capped_playbook, mode_map[phase]
    def _call_model(
        self,
        user_query: str,
        event_tape: List[Dict[str, Any]],
        tools: List[dict],
        phase: str,
        trigger_reason: str,
        trade_symbol_context: Dict[str, Any],
        active_symbol: str,
        market_mainline_context: Optional[Dict[str, Any]] = None,
        chart_input_context: Optional[Dict[str, Any]] = None,
        reasoning_effort: str = "medium",
        trade_symbol_label: str = "",
    ) -> Tuple[GenericPlaybook, dict]:
        trade_symbol_context = dict(trade_symbol_context or {}) if isinstance(trade_symbol_context, dict) else {}
        trade_symbol = str(
            trade_symbol_label
            or trade_symbol_context.get("display_name", "")
            or trade_symbol_context.get("trade_symbol_key", "")
            or trade_symbol_context.get("candidate_key", "")
            or ""
        ).strip()
        chart_context_debug = normalize_image_input_context(chart_input_context)
        selected_model = self._resolve_model(trigger_reason)
        chart_summaries = [dict(item) for item in list((chart_input_context or {}).get("chart_summaries") or []) if isinstance(item, dict)]
        user_payload = {
            "trade_symbol": trade_symbol,
            "chart_summaries": chart_summaries,
        }
        if trigger_reason == "passive_event_trigger" and isinstance(market_mainline_context, dict) and market_mainline_context:
            user_payload["market_mainline_context"] = self._normalize_market_mainline_context(
                market_mainline_context,
                diagnostic_universe=getattr(self, "diagnostic_instrument_universe", DEFAULT_DIAGNOSTIC_INSTRUMENT_UNIVERSE),
                trade_symbol=trade_symbol,
            )
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
                    "text": "The following uploaded images are chart screenshots for trade_symbol, in the same order as chart_summaries.",
                }
            )
        for image_item in list((chart_input_context or {}).get("input_images") or []):
            if isinstance(image_item, dict) and image_item.get("type") == "input_image":
                user_content.append(dict(image_item))
        input_messages = [
            {"role": "system", "content": [{"type": "input_text", "text": self._build_system_prompt(phase, trigger_reason)}]},
            {"role": "user", "content": user_content},
        ]
        response = self._responses_create_with_retry(
            phase="playbook",
            timeout_seconds=self._resolve_request_timeout_seconds(trigger_reason),
            model=selected_model,
            input=input_messages,
            tools=tools,
            reasoning={"effort": reasoning_effort},
            text={"format": PLAYBOOK_SCHEMA},
        )
        parsed_output = json.loads(response.output_text)
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
        return validate_playbook(parsed_output), {
            "request_messages": sanitize_response_input_messages(input_messages),
            "response_id": str(_response_attr(response, "id", "") or ""),
            "response_model": response_model,
            "raw_output_text": response.output_text,
            "parsed_output": parsed_output,
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
            "tools": tools,
            "phase": phase,
            "trigger_reason": trigger_reason,
            "trade_symbol": trade_symbol,
            "chart_summaries": chart_summaries,
            "trade_symbol_context": trade_symbol_context,
            "active_symbol": active_symbol,
        }
