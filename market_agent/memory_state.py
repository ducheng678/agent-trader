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



class MemoryStateMixin:
    def _should_force_news_context(self, trigger_reason: str, search_mode: str) -> bool:
        if search_mode == "off":
            return False
        if trigger_reason in PM_SCENARIO_REQUERY_LOCK_REASONS:
            return False
        if trigger_reason == "passive_event_trigger":
            return bool(getattr(self, "force_passive_news_context", False))
        return bool(getattr(self, "force_active_news_context", True))
    def _sanitize_helper_event_ref(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(event, dict):
            return None
        event_timestamp = str(event.get("event_timestamp", "") or event.get("published_at", "") or event.get("seen_at", "") or "").strip()
        source = str(event.get("source", "") or "").strip()
        title = str(event.get("title", "") or "").strip()
        url = str(event.get("url", "") or "").strip()
        item_id = str(event.get("item_id", "") or "").strip()
        if not (event_timestamp and source and title):
            return None
        sanitized = {
            "event_timestamp": event_timestamp,
            "source": source,
            "title": title,
        }
        if url:
            sanitized["url"] = url
        if item_id:
            sanitized["item_id"] = item_id
        return sanitized
    def _normalize_helper_materiality_events(
        self,
        events: Optional[Any],
    ) -> List[Dict[str, Any]]:
        normalized: List[Dict[str, Any]] = []
        seen_keys: set = set()
        for item in list(events or []):
            sanitized = self._sanitize_helper_event_ref(item if isinstance(item, dict) else {})
            if not sanitized:
                continue
            dedupe_key = (
                sanitized.get("item_id", ""),
                sanitized.get("url", ""),
                sanitized["title"],
                sanitized["event_timestamp"],
            )
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            normalized.append(sanitized)
        return normalized
    @staticmethod
    def _sanitize_passive_recent_event_for_llm(event: Dict[str, Any]) -> Dict[str, Any]:
        cleaned = strip_item_id_for_llm(dict(event))
        if isinstance(cleaned, dict):
            for key in ("source", "event_timestamp"):
                cleaned.pop(key, None)
            return cleaned
        return {}
    def _load_passive_recent_events_from_helper_materiality(
        self,
        trade_symbol: str,
        *,
        max_items: int,
    ) -> List[Dict[str, Any]]:
        if not str(trade_symbol or "").strip() or max_items <= 0:
            return []
        path = getattr(self, "helper_materially_new_first_events_path", None)
        if not isinstance(path, Path) or not path.exists():
            return []
        selected_events: List[Dict[str, Any]] = []
        for line in _iter_jsonl_lines_reverse(path):
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            events = [
                self._sanitize_passive_recent_event_for_llm(item)
                for item in self._normalize_helper_materiality_events(payload.get("materially_new_first_events") or [])
            ]
            events = [item for item in events if isinstance(item, dict) and item]
            if not events:
                continue
            remaining = max_items - len(selected_events)
            if remaining <= 0:
                break
            selected_events = events[-remaining:] + selected_events
            if len(selected_events) >= max_items:
                break
        return selected_events[-max_items:]
    def _load_helper_prior_materially_new_events(
        self,
        *,
        max_items: int,
    ) -> List[Dict[str, Any]]:
        if max_items <= 0:
            return []
        path = getattr(self, "helper_materially_new_first_events_path", None)
        if not isinstance(path, Path) or not path.exists():
            return []
        selected_events: List[Dict[str, Any]] = []
        for line in _iter_jsonl_lines_reverse(path):
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if not isinstance(payload, dict):
                continue
            flat_events: List[Dict[str, Any]] = []
            for item in self._normalize_helper_materiality_events(payload.get("materially_new_first_events") or []):
                slim = {
                    "event_timestamp": str(item.get("event_timestamp", "") or "").strip(),
                    "source": str(item.get("source", "") or "").strip(),
                    "title": str(item.get("title", "") or "").strip(),
                }
                if all(slim.values()):
                    flat_events.append(slim)
            if not flat_events:
                continue
            flat_events.sort(key=lambda item: parse_utc_iso(item.get("event_timestamp")) or datetime.min.replace(tzinfo=timezone.utc))
            remaining = max_items - len(selected_events)
            if remaining <= 0:
                break
            selected_events = flat_events[-remaining:] + selected_events
            if len(selected_events) >= max_items:
                break
        return selected_events[-max_items:]
    def _load_helper_materially_new_first_events(self) -> List[Dict[str, Any]]:
        path = getattr(self, "helper_materially_new_first_events_path", None)
        if not isinstance(path, Path) or not path.exists():
            return []
        try:
            raw_text = path.read_text(encoding="utf-8").strip()
        except Exception:
            return []
        if not raw_text:
            return []
        payload: Optional[Dict[str, Any]] = None
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            payload = None
        if payload is None:
            lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
            for line in reversed(lines):
                try:
                    parsed = json.loads(line)
                except Exception:
                    continue
                if isinstance(parsed, dict):
                    payload = parsed
                    break
        if payload is None:
            return []
        if not isinstance(payload, dict):
            return []
        return self._normalize_helper_materiality_events(payload.get("materially_new_first_events") or [])
    def _load_latest_helper_market_mainline_context_from_disk(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        path = getattr(self, "helper_market_mainline_latest_path", None)
        if not isinstance(path, Path) or not path.exists():
            return {}, {}
        try:
            lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except Exception:
            return {}, {}
        if not lines:
            return {}, {}
        try:
            payload = json.loads(lines[-1])
        except Exception:
            return {}, {}
        if not isinstance(payload, dict):
            return {}, {}
        context = self._normalize_market_mainline_context(
            payload.get("market_mainline_context") or {},
            diagnostic_universe=getattr(self, "diagnostic_instrument_universe", DEFAULT_DIAGNOSTIC_INSTRUMENT_UNIVERSE),
        )
        debug = dict(payload.get("market_mainline_call_debug") or {})
        return context, debug
    def _hydrate_helper_context_from_disk(self) -> None:
        context, debug = self._load_latest_helper_market_mainline_context_from_disk()
        if context:
            self.latest_helper_market_mainline_context = context
            self.latest_helper_market_mainline_debug = debug
        self.latest_helper_materially_new_first_events = self._load_helper_materially_new_first_events()
    def _persist_helper_market_mainline_snapshot(self, context: Dict[str, Any], debug: Dict[str, Any]) -> None:
        path = getattr(self, "helper_market_mainline_latest_path", None)
        if not isinstance(path, Path):
            return
        payload = {
            "market_mainline_context": self._normalize_market_mainline_context(
                context or {},
                diagnostic_universe=getattr(self, "diagnostic_instrument_universe", DEFAULT_DIAGNOSTIC_INSTRUMENT_UNIVERSE),
            ),
            "market_mainline_call_debug": dict(debug or {}),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    def _persist_helper_materiality_events(self, events: List[Dict[str, Any]]) -> None:
        path = getattr(self, "helper_materially_new_first_events_path", None)
        if not isinstance(path, Path):
            return
        payload = {
            "materially_new_first_events": self._normalize_helper_materiality_events(events or []),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    def _helper_materiality_checkpoint_timestamp(self) -> Optional[datetime]:
        latest_dt: Optional[datetime] = None
        for item in list(getattr(self, "latest_helper_materially_new_first_events", []) or []):
            dt = parse_utc_iso(str((item or {}).get("event_timestamp", "") or ""))
            if dt is None:
                continue
            if latest_dt is None or dt > latest_dt:
                latest_dt = dt
        return latest_dt
    def _get_cached_helper_market_mainline_context(
        self,
        *,
        trade_symbol_context: Dict[str, Any],
        active_symbol: str,
    ) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
        cached_context = dict(getattr(self, "latest_helper_market_mainline_context", {}) or {})
        cached_debug = dict(getattr(self, "latest_helper_market_mainline_debug", {}) or {})
        if not cached_context:
            context, debug = self._load_latest_helper_market_mainline_context_from_disk()
            if context:
                self.latest_helper_market_mainline_context = dict(context)
                self.latest_helper_market_mainline_debug = dict(debug)
                cached_context = dict(context)
                cached_debug = dict(debug)
        if not cached_context:
            return None, {}
        active = canonicalize_execution_symbol(active_symbol)
        context = trade_symbol_context if isinstance(trade_symbol_context, dict) else {}
        execution_symbol = canonicalize_execution_symbol(context.get("execution_symbol", ""))
        cached_trade_symbol = str(cached_debug.get("trade_symbol") or cached_debug.get("winner_display_name") or "").strip()
        if not cached_trade_symbol or not self._trade_symbol_matches_local_selection(context, cached_trade_symbol):
            return None, {}
        if not active or not execution_symbol:
            return cached_context, cached_debug
        if execution_symbol == active:
            return cached_context, cached_debug
        return None, {}
