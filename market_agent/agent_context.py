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



class AgentContextMixin:
    @staticmethod
    def _trade_symbol_matches_local_selection(trade_symbol_context: Dict[str, Any], selected_symbol: str) -> bool:
        raw = str(selected_symbol or "").strip().upper()
        normalized = normalize_candidate_key(raw)
        if not raw:
            return False
        tokens = {
            str(trade_symbol_context.get("trade_symbol_key", "") or trade_symbol_context.get("candidate_key", "") or "").strip().upper(),
            normalize_candidate_key(trade_symbol_context.get("trade_symbol_key", "") or trade_symbol_context.get("candidate_key", "")),
            str(trade_symbol_context.get("canonical_symbol_key", "") or "").strip().upper(),
            normalize_candidate_key(trade_symbol_context.get("canonical_symbol_key", "")),
            str(trade_symbol_context.get("market_name", "") or "").strip().upper(),
            normalize_candidate_key(trade_symbol_context.get("market_name", "")),
            str(trade_symbol_context.get("display_symbol", "") or "").strip().upper(),
            normalize_candidate_key(trade_symbol_context.get("display_symbol", "")),
            str(trade_symbol_context.get("display_name", "") or "").strip().upper(),
            normalize_candidate_key(trade_symbol_context.get("display_name", "")),
            str(trade_symbol_context.get("configured_execution_symbol", "") or "").strip().upper(),
            normalize_candidate_key(trade_symbol_context.get("configured_execution_symbol", "")),
            str(trade_symbol_context.get("execution_symbol", "") or "").strip().upper(),
            normalize_candidate_key(trade_symbol_context.get("execution_symbol", "")),
        }
        return raw in tokens or normalized in tokens
    def _select_playbook_trade_symbol_context(
        self,
        trade_symbol_context: Dict[str, Any],
        *,
        active_symbol: str,
        market_mainline_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        context = dict(trade_symbol_context or {}) if isinstance(trade_symbol_context, dict) else {}
        if not context:
            return None
        active = canonicalize_execution_symbol(active_symbol)
        if active:
            execution_symbol = canonicalize_execution_symbol(context.get("execution_symbol", ""))
            if execution_symbol and execution_symbol != active:
                return None
        return context
    def _passive_trade_confidence_threshold(self, trade_symbol: Any = "") -> float:
        calibration = get_trigger_confidence_calibration(trade_symbol)
        return min(1.0, max(0.0, float(safe_float(calibration.get("relevance_threshold"), 0.0) or 0.0)))
    def _passive_prefetched_judge_should_price(
        self,
        prefetched_passive_event_judge: Optional[Dict[str, Any]],
        *,
        trade_symbol: str = "",
    ) -> Optional[bool]:
        if not isinstance(prefetched_passive_event_judge, dict):
            return None
        if not bool(prefetched_passive_event_judge.get("passive_realtime_allowed", True)):
            return False
        judge_output = prefetched_passive_event_judge.get("judge_output")
        if not isinstance(judge_output, dict):
            return None
        relevance = str(judge_output.get("trigger_event_relevance", "unrelated") or "unrelated").strip().lower()
        action = str(judge_output.get("action", "no_trade") or "no_trade").strip().lower()
        trigger_confidence = min(1.0, max(0.0, float(safe_float(judge_output.get("trigger_confidence"), 0.0) or 0.0)))
        threshold_symbol = str(trade_symbol or prefetched_passive_event_judge.get("trade_symbol", "") or "").strip()
        trade_threshold = self._passive_trade_confidence_threshold(threshold_symbol)
        return relevance == "relevant" and action != "no_trade" and trigger_confidence >= trade_threshold
    @staticmethod
    def _copy_chart_input_context(chart_input_context: Dict[str, Any], *, include_images: bool) -> Dict[str, Any]:
        copied = dict(chart_input_context or {})
        copied["input_images"] = [dict(item) for item in list(copied.get("input_images") or []) if isinstance(item, dict)]
        copied["debug_images"] = [dict(item) for item in list(copied.get("debug_images") or []) if isinstance(item, dict)]
        copied["chart_summaries"] = [dict(item) for item in list(copied.get("chart_summaries") or []) if isinstance(item, dict)]
        if not include_images:
            copied["input_images"] = []
            copied["debug_images"] = []
            copied["image_count"] = 0
        return copied
    def _resolve_prefetched_passive_chart_context(
        self,
        prefetched_passive_event_judge: Optional[Dict[str, Any]],
        *,
        include_images: bool,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        if not isinstance(prefetched_passive_event_judge, dict):
            return None, {}
        future = prefetched_passive_event_judge.get("chart_context_future")
        if not hasattr(future, "result"):
            return None, {}
        is_done = getattr(future, "done", None)
        debug: Dict[str, Any] = {
            "started": True,
            "completed_before_wait": bool(is_done()) if callable(is_done) else False,
            "reused": False,
        }
        try:
            result = future.result()
        except Exception as exc:
            debug["error"] = str(exc)
            return None, debug
        if isinstance(result, dict):
            debug.update(
                {
                    "duration_seconds": result.get("duration_seconds"),
                    "error": result.get("error", ""),
                    "completed_at": result.get("completed_at"),
                }
            )
            chart_input_context = result.get("chart_input_context")
        else:
            chart_input_context = None
        if isinstance(chart_input_context, dict) and chart_input_context:
            copied = self._copy_chart_input_context(chart_input_context, include_images=include_images)
            debug["reused"] = True
            debug["chart_summary_count"] = len(copied.get("chart_summaries") or [])
            debug["image_count"] = int(copied.get("image_count") or len(copied.get("debug_images") or []))
            return copied, debug
        return None, debug
    @staticmethod
    def _select_visual_trade_symbol_context(
        trade_symbol_context: Dict[str, Any],
        *,
        active_symbol: str,
        market_news_debug: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        context = dict(trade_symbol_context or {}) if isinstance(trade_symbol_context, dict) else {}
        if not context:
            return None
        winner_display_name = str((market_news_debug or {}).get("winner_display_name", "") or "").strip().upper()
        if winner_display_name:
            label = str(context.get("display_name", "") or context.get("trade_symbol_key", "") or context.get("candidate_key", "") or "").strip().upper()
            if label == winner_display_name:
                return context
            return None
        active = canonicalize_execution_symbol(active_symbol)
        if active:
            execution_symbol = canonicalize_execution_symbol(context.get("execution_symbol", ""))
            if execution_symbol and execution_symbol != active:
                return None
        return context
