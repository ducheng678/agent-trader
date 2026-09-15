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

def normalize_reasoning_effort(value: Optional[str], default: str = "medium") -> str:
    aliases = {"minimal": "low"}
    allowed = {"none", "low", "medium", "high", "xhigh"}
    raw = (value or "").strip().lower()
    if not raw:
        return default
    if raw in aliases:
        mapped = aliases[raw]
        print(f"[warn] OPENAI_REASONING_EFFORT={value!r} is deprecated; mapping to {mapped!r}")
        raw = mapped
    if raw not in allowed:
        print(f"[warn] invalid OPENAI_REASONING_EFFORT={value!r}; fallback to {default!r}")
        return default
    return raw

def bump_reasoning_effort_one_level(value: Optional[str], default: str = "medium") -> str:
    normalized = normalize_reasoning_effort(value, default)
    levels = ["none", "low", "medium", "high", "xhigh"]
    index = levels.index(normalized)
    return levels[min(index + 1, len(levels) - 1)]


class ModelRoutingMixin:
    def _resolve_search_mode(self, trigger_reason: str) -> str:
        mode = self.passive_search_mode if trigger_reason == "passive_event_trigger" else self.active_search_mode
        if mode not in SEARCH_MODES:
            raise ValueError("search mode must be off, context_only, or always")
        return mode
    def _resolve_reasoning_effort(self, trigger_reason: str) -> str:
        return self.passive_reasoning_effort if trigger_reason == "passive_event_trigger" else self.active_reasoning_effort
    def _resolve_model(self, trigger_reason: str) -> str:
        return self.passive_model if trigger_reason == "passive_event_trigger" else self.active_model
    def _resolve_request_timeout_seconds(self, trigger_reason: str) -> float:
        active_timeout = max(
            1.0,
            float(
                getattr(
                    self,
                    "active_openai_request_timeout_seconds",
                    os.getenv("OPENAI_ACTIVE_REQUEST_TIMEOUT_SECONDS", os.getenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "180")),
                )
                or 180.0
            ),
        )
        passive_timeout = max(
            1.0,
            float(
                getattr(
                    self,
                    "passive_openai_request_timeout_seconds",
                    os.getenv("OPENAI_PASSIVE_REQUEST_TIMEOUT_SECONDS", os.getenv("OPENAI_REQUEST_TIMEOUT_SECONDS", "60")),
                )
                or 60.0
            ),
        )
        return passive_timeout if trigger_reason == "passive_event_trigger" else active_timeout
