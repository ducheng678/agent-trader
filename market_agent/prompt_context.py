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

PROMPT_CACHE_KEY_COMPONENT_RE = re.compile(r"[^A-Za-z0-9_-]+")
PROMPT_CACHE_KEY_MAX_LENGTH = 64

def _system_prompt_text(input_messages: List[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for message in input_messages or []:
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "") or "").strip().lower() not in {"system", "developer"}:
            continue
        content = message.get("content")
        if isinstance(content, str):
            if content:
                parts.append(content)
            continue
        for item in content or []:
            if not isinstance(item, dict):
                continue
            if str(item.get("type", "") or "").strip() not in {"input_text", "text"}:
                continue
            text = str(item.get("text", "") or "")
            if text:
                parts.append(text)
    return "\n".join(parts).strip()

def build_prompt_cache_key(
    *,
    prefix: str,
    phase: str,
    model: str,
    input_messages: List[Dict[str, Any]],
) -> str:
    if not _system_prompt_text(input_messages):
        return ""
    safe_prefix = PROMPT_CACHE_KEY_COMPONENT_RE.sub("-", str(prefix or "").strip()).strip("-_") or "market-agent"
    safe_phase = PROMPT_CACHE_KEY_COMPONENT_RE.sub("-", str(phase or "").strip()).strip("-_") or "request"
    safe_model = PROMPT_CACHE_KEY_COMPONENT_RE.sub("-", str(model or "").strip()).strip("-_") or "model"
    cache_key = f"{safe_prefix}-{safe_phase}-{safe_model}"
    if len(cache_key) <= PROMPT_CACHE_KEY_MAX_LENGTH:
        return cache_key
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()[:16]
    max_prefix_length = PROMPT_CACHE_KEY_MAX_LENGTH - len(digest) - 1
    return f"{cache_key[:max_prefix_length].rstrip('-_')}-{digest}"


class PromptContextMixin:
    def _build_prompt_cache_key(self, *, phase: str, create_kwargs: Dict[str, Any]) -> str:
        if not bool(getattr(self, "prompt_cache_enabled", True)):
            return ""
        return build_prompt_cache_key(
            prefix=str(getattr(self, "prompt_cache_key_prefix", "market-agent") or "market-agent"),
            phase=phase,
            model=str(create_kwargs.get("model", "") or ""),
            input_messages=list(create_kwargs.get("input") or []),
        )
    def _build_passive_event_judge_prompt(self, phase: str) -> str:
        verify = self._phase_verify_guidance(phase)
        return (
            "When recent_events is not empty, use it as the canonical materially-new fact tape for trade_symbol. "
            "If trigger_event has no direct effect on trade_symbol, fill trigger_confidence as 0, set trigger_event_relevance as unrelated, and set action as no_trade. "
            "If trigger_event only repeats a development in recent_events that is relevant to trade_symbol, or adds no materially new fact for trade_symbol beyond that development, fill trigger_confidence as 0, set trigger_event_relevance as duplicate, and set action as no_trade. "
            "Before scoring trigger_confidence, treat trigger_event as duplicate if it mainly describes pre-action threats, warnings, preparations, intelligence, deliberations, or anything else that occurred before an already-known event in recent_events, unless it adds an explicit new post-action development. "
            "Otherwise, fill trigger_event_relevance as relevant and set trigger_confidence to a raw 0-to-1 direct, first-order, tradeable event-impact score for trade_symbol, using trigger_event together with market_mainline_context. "
            "If market_mainline_context is supplied, treat it as background on trade_symbol's current move logic and cross-asset diagnostic instruments, and judge trigger_confidence by whether trigger_event reinforces, weakens, or materially changes that mainline for trade_symbol. "
            "The score of trigger_confidence directs tradeable impact rather than general importance, narrative salience, broad macro significance, or vague thematic relevance. "
            "For single-site local facility, terminal, port, depot, refinery, or loading disruptions, keep trigger_confidence below 0.55 when the facility is outside the named country, chokepoint, conflict, or export route driving market_mainline_context. "
            "Do not apply this cap to core export infrastructure inside that mainline geography, especially shipment halts, empty jetties, pipeline hits, or confirmed export curtailment. "
            "Commentary, interviews, lawsuits, opinionated remarks, person-focused news, and second-order industry discussion from trigger_event should usually receive low trigger_confidence unless they contain a clear new fact that directly changes the tradeable outlook for trade_symbol. "
            "Market-wrap or price-action recap items are no_trade unless they add a new first-order non-price fact for trade_symbol. "
            "Use trigger_event together with market_mainline_context as the passive directional context. "
            "Before assigning long or short, first identify the single marginal change that trigger_event makes to the expected future price path of trade_symbol relative to recent_events and market_mainline_context. "
            "If different parts of trigger_event imply opposite directions and the net marginal update cannot be resolved without emphasizing one cue over another, set the event-implied direction as genuinely unclear and use no_trade rather than choosing the side with higher standalone salience. "
            "First determine the event-implied net direction for trade_symbol from trigger_event together with market_mainline_context: bullish, bearish, or genuinely unclear. "
            "If the event-implied net direction is bullish, set action to long. If the event-implied net direction is bearish, set action to short. "
            "Use no_trade only when the event-implied direction for trade_symbol is genuinely unclear, neutral, or too weak to justify a trade. "
            "When trigger_event is directionally relevant, action should normally align with the net direction implied by trigger_event together with market_mainline_context. "
            f"{verify}"
        )
    def _build_passive_technical_pricing_prompt(self, phase: str) -> str:
        verify = self._phase_verify_guidance(phase)
        return (
            "Foremost, do not chase strength or weakness! "
            "Focus on entry, stop-loss, and execution logic for the action. "
            "Use chart_summaries only to judge entry quality, whether immediate execution is still coherent at current price, and to keep entry_price and stop_loss_price realistic. "
            "chart_summaries may include local candle-derived summaries aligned with supplied chart screenshots. Treat chart_summaries as the source of truth for price/technical analysis. "
            "If chart screenshots are supplied, use those images only for visual structure interpretation such as shape, congestion, breakout quality, and pullback texture. "
            "Do not infer exact numeric values from chart pixels when chart_summaries or other text fields provide them. "
            "entry_price must stay logically coherent with the action and current chart context. "
            f"{verify}"
        )
    @staticmethod
    def _phase_verify_guidance(phase: str) -> str:
        if phase == "verified":
            return (
                "Do not use web search for price, K-line, candlestick, technical-indicator, or historical market data lookup; local Hyperliquid market context is the source of truth for those. "
            )
        if phase == "context_only":
            return (
                "Treat trigger_event as the source-of-truth event tape for the passive trigger. "
                "Do NOT use web search to verify whether the trigger event itself happened. "
                "Do not use web search for price, K-line, candlestick, technical-indicator, or historical market data lookup."
            )
        return "Do not assume facts beyond the supplied context."
    def _build_system_prompt(self, phase: str, trigger_reason: str = "") -> str:
        verify = self._phase_verify_guidance(phase)
        execute_now_confidence_pct = max(0, min(100, int(round(self.execute_now_confidence_threshold * 100.0))))
        trigger_guidance = (
            "When recent_events is not empty, use it as the canonical materially-new fact tape for trade_symbol. "
            "If trigger_event adds no materially new fact beyond recent_events and only repeats an already known development, fill trigger_confidence as 0, trigger_event_relevance as duplicate and set the whole playbook to null. "
            "Otherwise, set root-level trigger_confidence to a raw 0-to-1 direct, first-order, tradeable event-impact score for trade_symbol, using trigger_event together with market_mainline_context. "
            "If market_mainline_context is supplied, treat it as background on trade_symbol's current move logic and cross-asset diagnostic instruments, and judge trigger_confidence by whether trigger_event reinforces, weakens, or materially changes that mainline for trade_symbol. "
            "The score of trigger_confidence directs tradeable impact rather than general importance, narrative salience, broad macro significance, or vague thematic relevance. "
            "Commentary, interviews, lawsuits, opinionated remarks, person-focused news, and second-order industry discussion from trigger_event should usually receive low trigger_confidence unless they contain a clear new fact that directly changes the tradeable outlook for trade_symbol. "
            "If trigger_confidence is below 0.20, fill trigger_event_relevance as unrelated and set the whole playbook to null. "
            "If trigger_confidence is 0.20 or above, fill trigger_event_relevance as relevant, set execute_now to true, and set scenario to null. "
            "And fill entry_plan.action_decision with that action under the following instruction:\n "
            "Use trigger_event together with market_mainline_context as the passive directional context. "
            "First determine the event-implied net direction for trade_symbol from trigger_event together with market_mainline_context: bullish, bearish, or genuinely unclear. "
            "If the event-implied net direction is bullish, set entry_plan.action_decision to long. If the event-implied net direction is bearish, set entry_plan.action_decision to short. "
            "Use no_trade only when the event-implied direction for trade_symbol is genuinely unclear, neutral, or too weak to justify a trade. "
            "When trigger_event is directionally relevant, entry_plan.action_decision should normally align with the net direction implied by trigger_event together with market_mainline_context. "
            "Do not use no_trade merely because the move already looks extended, and do not flip to the opposite side merely because the move looks extended, overbought, oversold, or technically stretched. "
            "Use chart_summaries only to judge entry quality, whether immediate execution is still coherent at current price, and to keep entry_price and stop_loss_price realistic. "
            "Do not let chart_summaries reverse a clear event-implied direction. "
            if trigger_reason == "passive_event_trigger"
            else (
                "Think the probability of a positive return for an entry here. "
                f"Only set execute_now to true and set scenario to null when you estimate that probability is at least {execute_now_confidence_pct}%. "
                f"If you think that probability is below {execute_now_confidence_pct}%, set execute_now to false, still fill entry_plan.action_decision with the eventual action, and fill scenario with the trigger conditions that determine when that same action_decision should execute. "
                "Set root-level trigger_confidence to null. "
                "Fill trigger_event_relevance as not_applicable and fill playbook with entry_plan. "
            )
        )
        chart_image_guidance = (
            "If chart screenshots are supplied, use those images only for visual structure interpretation such as shape, congestion, breakout quality, and pullback texture. "
            "Do not infer exact numeric values from chart pixels when chart_summaries or other text fields provide them. "
        )
        return (
            "Foremost, do not chase strength or weakness! Avoid buying after an already extended rise or shorting after an already extended drop unless the entry and execution logic are still clearly coherent. "
            "entry_plan is the only market-intent plan you should design from a flat baseline. Focus on market direction, entry, stop-loss, and execution logic. "
            "Use only long, short, or no_trade in entry_plan.action_decision. Every entry_decision long/short action must include entry_price and stop_loss_price. "
            "chart_summaries may include local candle-derived summaries aligned with supplied chart screenshots. Treat chart_summaries as the source of truth for price/technical analysis. "
            f"{chart_image_guidance}"
            "Set scenario.observe_when_all.low and scenario.observe_when_all.high as a single observation zone. Observation starts only when price trades inside that low-high range. "
            "Use execute_when_all.condition for the post-observation execution gate. When execute_when_all.condition is satisfied after observation has started, the system executes entry_plan.action_decision immediately. Put the abandonment timer in execute_when_all.timeout_seconds, and keep execute_when_all.timeout_seconds around 900 seconds. "
            "For execute_when_all.condition, level means the single trigger price point for the execution rule, not a range. "
            "Use level for price_ge, price_le, cross_above, cross_below, sustained_ge, and sustained_le; use low and high for price_between and sustained_between; use timer_seconds only when the condition needs a time window; use tolerance_bps and min_ratio only when they meaningfully refine the rule. "
            "entry_price and execute_when_all.condition.level must stay logically coherent when level is used: for example, do not place a long entry_price materially below an upward execution trigger, and do not place a short entry_price materially above a downward execution trigger. "
            "For price_between or sustained_between, keep entry_price logically coherent with the low/high band instead of level. "
            "For practical market semantics such as holding above a level, losing a level, or failing on a retest, use sustained_* / cross_* with optional min_ratio and tolerance_bps when useful; do not interpret these ideas as requiring every sampled tick to be perfectly on one side unless you explicitly want a very strict rule. "
            f"{trigger_guidance}"
            f"{verify}"
        )
