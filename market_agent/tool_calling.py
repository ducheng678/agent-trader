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



class ToolCallingMixin:
    def _responses_create_with_retry(self, *, phase: str, timeout_seconds: Optional[float] = None, **create_kwargs) -> Any:
        request_kwargs = dict(create_kwargs)
        if not request_kwargs.get("prompt_cache_key"):
            prompt_cache_key = self._build_prompt_cache_key(phase=phase, create_kwargs=request_kwargs)
            if prompt_cache_key:
                request_kwargs["prompt_cache_key"] = prompt_cache_key
        max_attempts = max(1, int(getattr(self, "openai_max_attempts", 3) or 3))
        timeout_seconds = max(1.0, float(timeout_seconds if timeout_seconds is not None else 180.0))
        retry_delay_seconds = max(0.0, float(getattr(self, "openai_retry_delay_seconds", 0.0) or 0.0))
        quiet_success_log = phase in {"passive_event_judge"}
        last_exc: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            if not quiet_success_log:
                print_line(f"[openai_call_start] phase={phase} attempt={attempt}/{max_attempts}")
            try:
                def invoke_once():
                    if hasattr(self.client, "create"):
                        return self.client.create(timeout=timeout_seconds, **request_kwargs)
                    return self.client.responses.create(timeout=timeout_seconds, **request_kwargs)

                workflow = getattr(self, "llm_workflow", None)
                response = workflow.run_single(invoke_once) if workflow is not None else invoke_once()
            except Exception as exc:
                last_exc = exc
                message = f"{type(exc).__name__}: {exc}".strip()
                if attempt < max_attempts:
                    retry_in_seconds = retry_delay_seconds * attempt
                    print_line(
                        f"[openai_call_retry] phase={phase} attempt={attempt}/{max_attempts} "
                        f"error={message} retry_in={retry_in_seconds:.1f}s"
                    )
                    if retry_in_seconds > 0:
                        time.sleep(retry_in_seconds)
                    continue
                print_line(f"[openai_call_failed] phase={phase} attempt={attempt}/{max_attempts} error={message}")
                raise RuntimeError(f"OpenAI {phase} failed after {max_attempts} attempts: {message}") from exc
            if not quiet_success_log:
                print_line(f"[openai_call_done] phase={phase} attempt={attempt}/{max_attempts}")
            return response
        if last_exc is not None:
            raise RuntimeError(f"OpenAI {phase} failed after {max_attempts} attempts") from last_exc
        raise RuntimeError(f"OpenAI {phase} failed before any attempt completed")
