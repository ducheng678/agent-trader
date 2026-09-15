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

from market_agent.model_routing import bump_reasoning_effort_one_level

MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(https?://[^)\s]+(?:\s+\"[^\"]*\")?\)")
BARE_URL_RE = re.compile(r"https?://\S+")
DOMAIN_CITATION_PARENS_RE = re.compile(r"\s*\(([A-Za-z0-9.-]+\.[A-Za-z]{2,})(?:/[^)]*)?\)")
EMPTY_PARENS_RE = re.compile(r"\(\s*\)")
WHITESPACE_RE = re.compile(r"\s+")
RELATION_ENTITY_RE = re.compile(r"\b(?:U\.S\.|US|[A-Z][A-Za-z]+)[-.–—](?:[A-Z][A-Za-z]+|[A-Z]{2,})\b")
CAPITALIZED_ENTITY_RE = re.compile(
    r"\b(?:U\.S\.|US|[A-Z]{2,8}|[A-Z][A-Za-z0-9'’]*)"
    r"(?:\s+(?:of|al|el|the|and|in|on|to|for|&|[A-Z]{2,8}|[A-Z][A-Za-z0-9'’]*)){0,5}"
)
ENTITY_CONNECTOR_TERMS = {"of", "al", "el", "the", "and", "in", "on", "to", "for", "&"}
MAINLINE_OVERLAP_STOP_TERMS = {
    "a",
    "an",
    "analysis",
    "ap",
    "axios",
    "bbc",
    "bbg",
    "bloomberg",
    "brent",
    "cnbc",
    "cnn",
    "crude",
    "current",
    "demand",
    "fox",
    "front",
    "front month",
    "front-month",
    "futures",
    "investing",
    "april",
    "august",
    "december",
    "february",
    "january",
    "july",
    "june",
    "latest",
    "market",
    "markets",
    "marketscreener",
    "march",
    "may",
    "mni",
    "month",
    "november",
    "move",
    "net",
    "news",
    "nyt",
    "october",
    "oil",
    "physical",
    "premium",
    "price",
    "prices",
    "report",
    "reported",
    "reports",
    "reuters",
    "risk",
    "rtrs",
    "september",
    "say",
    "says",
    "source",
    "sources",
    "supply",
    "that",
    "this",
    "trade symbol",
    "trade_symbol",
    "update",
    "us",
    "u s",
    "u.s.",
    "wsj",
}

def strip_links_for_llm_text(value: Any) -> str:
    text = str(value or "")
    if not text:
        return ""
    text = MARKDOWN_LINK_RE.sub(r"\1", text)
    text = BARE_URL_RE.sub("", text)
    text = DOMAIN_CITATION_PARENS_RE.sub("", text)
    text = EMPTY_PARENS_RE.sub("", text)
    return WHITESPACE_RE.sub(" ", text).strip()
def _normalize_overlap_term(value: Any) -> str:
    term = str(value or "").strip()
    if not term:
        return ""
    term = term.replace("’", "'")
    term = re.sub(r"\bU\.S\.", "US", term)
    term = re.sub(r"'s\b", "", term, flags=re.IGNORECASE)
    term = re.sub(r"[^A-Za-z0-9&' -]+", " ", term)
    term = WHITESPACE_RE.sub(" ", term).strip().lower()
    parts = [part for part in term.strip(" -").split() if part]
    while parts and parts[0] in ENTITY_CONNECTOR_TERMS:
        parts.pop(0)
    while parts and parts[-1] in ENTITY_CONNECTOR_TERMS:
        parts.pop()
    return " ".join(parts).strip(" -")
def _is_meaningful_overlap_term(term: str) -> bool:
    normalized = _normalize_overlap_term(term)
    if not normalized or normalized in MAINLINE_OVERLAP_STOP_TERMS:
        return False
    if len(normalized) <= 1:
        return False
    parts = [part for part in normalized.split() if part not in ENTITY_CONNECTOR_TERMS]
    if not parts:
        return False
    if all(part in MAINLINE_OVERLAP_STOP_TERMS for part in parts):
        return False
    return True
def extract_concrete_mainline_terms(value: Any) -> List[str]:
    """Extract concrete named terms for mainline-overlap debugging.

    This is intentionally heuristic and conservative. Generic commodity and
    news-reporting words are filtered because they do not prove two events share
    the same causal channel.
    """

    text = strip_links_for_llm_text(value)
    if not text:
        return []
    text = re.sub(r"\bU\.S\.", "US", text)
    terms = set()

    def add_term(raw: Any, *, add_parts: bool = False) -> None:
        normalized = _normalize_overlap_term(raw)
        if _is_meaningful_overlap_term(normalized):
            terms.add(normalized)
        if add_parts:
            for part in normalized.split():
                if _is_meaningful_overlap_term(part):
                    terms.add(part)

    for match in RELATION_ENTITY_RE.finditer(text):
        add_term(match.group(0), add_parts=True)
    for match in CAPITALIZED_ENTITY_RE.finditer(text):
        phrase = match.group(0)
        add_term(phrase, add_parts=True)

    return sorted(terms)
def _event_text_for_mainline_overlap(event: Optional[Dict[str, Any]]) -> str:
    if not isinstance(event, dict):
        return ""
    pieces = []
    for key in ("title", "summary"):
        value = str(event.get(key, "") or "").strip()
        if value:
            pieces.append(value)
    return "\n".join(pieces)
def _mainline_text_for_overlap(market_mainline_context: Optional[Dict[str, Any]]) -> str:
    if not isinstance(market_mainline_context, dict):
        return ""
    return str(market_mainline_context.get("current_move_logic_mainline", "") or "").strip()
def evaluate_mainline_overlap_confidence_adjustment(
    *,
    trigger_event: Optional[Dict[str, Any]],
    market_mainline_context: Optional[Dict[str, Any]],
    trigger_confidence: Any,
    multiplier: float = 0.5,
    min_confidence: float = 0.5,
) -> Dict[str, Any]:
    raw_confidence = min(1.0, max(0.0, float(safe_float(trigger_confidence, 0.0) or 0.0)))
    multiplier_value = safe_float(multiplier, 1.0)
    min_confidence_value = safe_float(min_confidence, 0.5)
    multiplier = min(1.0, max(0.0, float(multiplier_value if multiplier_value is not None else 1.0)))
    min_confidence = min(1.0, max(0.0, float(min_confidence_value if min_confidence_value is not None else 0.5)))
    mainline_text = _mainline_text_for_overlap(market_mainline_context)
    trigger_text = _event_text_for_mainline_overlap(trigger_event)
    mainline_terms = extract_concrete_mainline_terms(mainline_text)
    trigger_terms = extract_concrete_mainline_terms(trigger_text)
    overlap_terms = sorted(set(mainline_terms) & set(trigger_terms))
    eligible = bool(mainline_text and trigger_text and mainline_terms and trigger_terms and raw_confidence >= min_confidence)
    applied = bool(eligible and not overlap_terms and multiplier < 1.0)
    effective_confidence = min(1.0, max(0.0, raw_confidence * multiplier)) if applied else raw_confidence
    return {
        "applied": applied,
        "reason": "no_concrete_mainline_term_overlap" if applied else "",
        "eligible": eligible,
        "multiplier": multiplier,
        "min_confidence": min_confidence,
        "raw_trigger_confidence": raw_confidence,
        "effective_trigger_confidence": effective_confidence,
        "mainline_terms": mainline_terms,
        "trigger_terms": trigger_terms,
        "overlap_terms": overlap_terms,
    }

class RetrievalRAGMixin:
    @staticmethod
    def _normalize_diagnostic_instruments(
        instruments: Optional[List[Any]],
        *,
        excluded_instrument: str = "",
        allowed_universe: Optional[List[str]] = None,
    ) -> List[str]:
        allowed_map = {
            str(item).strip().upper(): str(item).strip().upper()
            for item in list(allowed_universe or [])
            if str(item).strip()
        }
        excluded_key = str(excluded_instrument or "").strip().upper()
        normalized: List[str] = []
        seen = set()
        for item in list(instruments or []):
            instrument = " ".join(str(item or "").strip().split()).upper()
            if not instrument or instrument == excluded_key:
                continue
            if allowed_map and instrument not in allowed_map:
                continue
            canonical = allowed_map.get(instrument, instrument)
            if canonical in seen:
                continue
            seen.add(canonical)
            normalized.append(canonical)
        return normalized
    @classmethod
    def _normalize_market_mainline_context(
        cls,
        context: Optional[Dict[str, Any]],
        *,
        diagnostic_universe: Optional[List[str]] = None,
        trade_symbol: str = "",
    ) -> Dict[str, Any]:
        raw = dict(context or {}) if isinstance(context, dict) else {}
        diagnostic_items = raw.get("diagnostic_instruments")
        return {
            "current_move_logic_mainline": strip_links_for_llm_text(raw.get("current_move_logic_mainline", "")),
            "diagnostic_instruments": cls._normalize_diagnostic_instruments(
                list(diagnostic_items or []),
                excluded_instrument=trade_symbol,
                allowed_universe=diagnostic_universe,
            ),
        }
    @classmethod
    def _compose_market_news_context_locally(
        cls,
        current_move_logic_mainline: str,
        diagnostic_instruments: Optional[List[str]] = None,
        *,
        diagnostic_universe: Optional[List[str]] = None,
        trade_symbol: str = "",
    ) -> Dict[str, Any]:
        mainline = str(current_move_logic_mainline or "").strip()
        return {
            "current_move_logic_mainline": mainline,
            "diagnostic_instruments": cls._normalize_diagnostic_instruments(
                list(diagnostic_instruments or []),
                excluded_instrument=trade_symbol,
                allowed_universe=diagnostic_universe,
            ),
        }
    def _build_market_news_context(
        self,
        *,
        user_query: str,
        recent_events: List[Dict[str, Any]],
        trade_symbol_context: Dict[str, Any],
        active_symbol: str,
        trigger_reason: str,
        trigger_event: Optional[Dict[str, Any]],
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        context = dict(trade_symbol_context or {}) if isinstance(trade_symbol_context, dict) else {}
        trade_symbol = str(
            context.get("display_symbol", "")
            or context.get("display_name", "")
            or context.get("trade_symbol_key", "")
            or context.get("candidate_key", "")
            or ""
        ).strip()
        diagnostic_instrument_universe = list(
            getattr(self, "diagnostic_instrument_universe", DEFAULT_DIAGNOSTIC_INSTRUMENT_UNIVERSE)
            or DEFAULT_DIAGNOSTIC_INSTRUMENT_UNIVERSE
        )
        prompt = (
            "You are the market-news helper for a trading system. "
            "Use recent_events as the event tape. trade_symbol is the only trading instrument this helper serves. Use diagnostic_instrument_universe only for diagnostic_instruments. "
            "Use prior_materially_new_events only as background for already-known facts. "
            "From recent_events, keep only thoroughly materially new first fact-level events after comparing within recent_events and against prior_materially_new_events, discarding follow-ups, lagged confirmations, recaps, commentary, explainers, opinions, interview-only items, and market-wrap items. "
            "Keep first-hand factual developments relevant to trade_symbol in the retained event set. "
            "Combine materially_new_first_events with web_search to produce current_move_logic_mainline for trade_symbol. "
            "Then choose diagnostic_instruments only from diagnostic_instrument_universe to help diagnose, confirm, contradict, or conditionally reframe that mainline. "
            "If web_search conflicts with materially_new_first_events, use web_search as the controlling source for current_move_logic_mainline and explicitly reconcile the mainline to the current verifiable facts, but do not add web_search-only events to materially_new_first_events. "
            "Do not let recent_events that are not materially new for trade_symbol override trade_symbol's mainline. "
            "diagnostic_instruments are instruments whose reaction to the same news flow or regime change helps diagnose how the market interprets trade_symbol's move; they are not close substitutes, same-complex defaults, or high-correlation peers. "
            "Pick diagnostic_instruments only from diagnostic_instrument_universe, ranking by interpretive value rather than raw correlation, with diverse and orthogonal information about risk sentiment, growth, inflation/rates, USD/liquidity, safe-haven demand, or sector cost pressure. "
            "Same-direction, opposite-direction, and conditional relationships are all useful, but return only symbols, not explanations. "
            "Return 3 diagnostic_instruments when possible, using fewer only if distinct cross-asset signals are unavailable. "
            "Avoid redundant selections that express the same signal, and do not select same-complex instruments unless they are essential for interpreting a sector cost-pressure or beneficiary/loser dynamic. "
            "Do not return trade_symbol, its base asset, obvious aliases or equivalents of trade_symbol, futures codes, free-text drivers, logistics phrases, commentary, or citations as diagnostic_instruments. "
            "Return only market_mainline_context and materially_new_first_events."
        )
        prior_materially_new_events: List[Dict[str, Any]] = []
        prior_trigger_threshold = max(0, int(getattr(self, "helper_prior_materially_new_trigger_threshold", 30) or 0))
        prior_max_items = max(0, int(getattr(self, "helper_prior_materially_new_max_items", 5) or 0))
        if prior_trigger_threshold > 0 and prior_max_items > 0 and len(list(recent_events or [])) < prior_trigger_threshold:
            prior_materially_new_events = [
                strip_item_id_for_llm(dict(item))
                for item in self._load_helper_prior_materially_new_events(max_items=prior_max_items)
                if isinstance(item, dict)
            ]
        request_payload = {
            "recent_events": [
                strip_item_id_for_llm(dict(item))
                for item in list(recent_events or [])
                if isinstance(item, dict)
            ],
            "prior_materially_new_events": prior_materially_new_events,
            "trade_symbol": trade_symbol,
            "diagnostic_instrument_universe": diagnostic_instrument_universe,
        }
        response = self._responses_create_with_retry(
            phase="market_news_context",
            timeout_seconds=self._resolve_request_timeout_seconds(trigger_reason),
            model=self.active_model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": json.dumps(request_payload, ensure_ascii=False, indent=2)}]},
            ],
            tools=[{"type": "web_search"}],
            parallel_tool_calls=False,
            reasoning={"effort": bump_reasoning_effort_one_level(self._resolve_reasoning_effort(trigger_reason), "medium")},
            text={"format": HELPER_MARKET_NEWS_CONTEXT_SCHEMA},
        )
        response_model = str(_response_attr(response, "model", self.active_model) or self.active_model)
        usage = extract_response_usage(response)
        web_search_tool_calls = count_web_search_tool_calls(response)
        web_search_calls = extract_web_search_call_details(response)
        web_search_analysis = analyze_web_search_calls(
            web_search_calls,
            context,
            max_total_calls=max(1, web_search_tool_calls),
            max_calls_per_topic=max(1, web_search_tool_calls),
        )
        web_search_analysis["max_total_calls"] = None
        web_search_analysis["max_calls_per_topic"] = None
        web_search_analysis["over_budget"] = False
        web_search_analysis["topic_budget_violations"] = {}
        usage_cost = estimate_openai_usage_cost(
            model=response_model,
            usage=usage,
            web_search_tool_calls=web_search_tool_calls,
        )
        raw_output = json.loads(response.output_text)
        raw_market_mainline_context = dict(raw_output.get("market_mainline_context") or {})
        parsed_output = self._compose_market_news_context_locally(
            str(raw_market_mainline_context.get("current_move_logic_mainline", "") or "").strip(),
            list(raw_market_mainline_context.get("diagnostic_instruments") or []),
            diagnostic_universe=diagnostic_instrument_universe,
            trade_symbol=trade_symbol,
        )
        materially_new_first_events = self._normalize_helper_materiality_events(raw_output.get("materially_new_first_events") or [])
        self.latest_helper_market_mainline_context = dict(parsed_output)
        self.latest_helper_materially_new_first_events = list(materially_new_first_events)
        self.latest_helper_market_mainline_debug = {
            "response_id": str(_response_attr(response, "id", "") or ""),
            "response_model": response_model,
            "raw_output_text": response.output_text,
            "parsed_output": parsed_output,
            "usage": usage,
            "usage_cost": usage_cost,
            "web_search_tool_calls": web_search_tool_calls,
            "web_search_calls": list(web_search_calls or []),
            "web_search_budget": {
                "max_total_calls": None,
                "trade_symbol_configured": bool(trade_symbol),
                "trade_symbol_search_budget": None,
                "winner_follow_up_budget": None,
            },
            "web_search_analysis": web_search_analysis,
            "trade_symbol": trade_symbol,
            "winner_display_name": trade_symbol,
            "tools": [{"type": "web_search"}],
            "phase": "market_news_context",
            "trigger_reason": trigger_reason,
            "active_symbol": active_symbol,
        }
        self._persist_helper_market_mainline_snapshot(self.latest_helper_market_mainline_context, self.latest_helper_market_mainline_debug)
        self._persist_helper_materiality_events(self.latest_helper_materially_new_first_events)
        return parsed_output, {
            **dict(self.latest_helper_market_mainline_debug),
        }
