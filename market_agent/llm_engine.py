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


MARKDOWN_LINK_RE = re.compile(r"\[([^\]]+)\]\(https?://[^)\s]+(?:\s+\"[^\"]*\")?\)")
BARE_URL_RE = re.compile(r"https?://\S+")
DOMAIN_CITATION_PARENS_RE = re.compile(r"\s*\(([A-Za-z0-9.-]+\.[A-Za-z]{2,})(?:/[^)]*)?\)")
EMPTY_PARENS_RE = re.compile(r"\(\s*\)")
WHITESPACE_RE = re.compile(r"\s+")

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
    system_prompt = _system_prompt_text(input_messages)
    if not system_prompt:
        return ""
    safe_prefix = PROMPT_CACHE_KEY_COMPONENT_RE.sub("-", str(prefix or "").strip()).strip("-_") or "market-agent"
    safe_phase = PROMPT_CACHE_KEY_COMPONENT_RE.sub("-", str(phase or "").strip()).strip("-_") or "request"
    digest_input = "\x1f".join((str(model or ""), str(phase or ""), system_prompt))
    digest = hashlib.sha256(digest_input.encode("utf-8")).hexdigest()[:16]
    return f"{safe_prefix}-{safe_phase}-{digest}"[:PROMPT_CACHE_KEY_MAX_LENGTH].rstrip("-_")


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


class DiscretionaryLLMEngine:
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

    def _build_prompt_cache_key(self, *, phase: str, create_kwargs: Dict[str, Any]) -> str:
        if not bool(getattr(self, "prompt_cache_enabled", True)):
            return ""
        return build_prompt_cache_key(
            prefix=str(getattr(self, "prompt_cache_key_prefix", "market-agent") or "market-agent"),
            phase=phase,
            model=str(create_kwargs.get("model", "") or ""),
            input_messages=list(create_kwargs.get("input") or []),
        )

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

    @staticmethod
    def _compose_market_news_context_locally(
        current_move_logic_mainline: str,
        diagnostic_instruments: Optional[List[str]] = None,
        *,
        diagnostic_universe: Optional[List[str]] = None,
        trade_symbol: str = "",
    ) -> Dict[str, Any]:
        mainline = str(current_move_logic_mainline or "").strip()
        return {
            "current_move_logic_mainline": mainline,
            "diagnostic_instruments": DiscretionaryLLMEngine._normalize_diagnostic_instruments(
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
            f"{trigger_guidance}"
            "chart_summaries may include local candle-derived summaries aligned with supplied chart screenshots. Treat chart_summaries as the source of truth for price/technical analysis. "
            f"{chart_image_guidance}"
            "Set scenario.observe_when_all.low and scenario.observe_when_all.high as a single observation zone. Observation starts only when price trades inside that low-high range. "
            "Use execute_when_all.condition for the post-observation execution gate. When execute_when_all.condition is satisfied after observation has started, the system executes entry_plan.action_decision immediately. Put the abandonment timer in execute_when_all.timeout_seconds, and keep execute_when_all.timeout_seconds around 900 seconds. "
            "For execute_when_all.condition, level means the single trigger price point for the execution rule, not a range. "
            "Use level for price_ge, price_le, cross_above, cross_below, sustained_ge, and sustained_le; use low and high for price_between and sustained_between; use timer_seconds only when the condition needs a time window; use tolerance_bps and min_ratio only when they meaningfully refine the rule. "
            "entry_price and execute_when_all.condition.level must stay logically coherent when level is used: for example, do not place a long entry_price materially below an upward execution trigger, and do not place a short entry_price materially above a downward execution trigger. "
            "For price_between or sustained_between, keep entry_price logically coherent with the low/high band instead of level. "
            "For practical market semantics such as holding above a level, losing a level, or failing on a retest, use sustained_* / cross_* with optional min_ratio and tolerance_bps when useful; do not interpret these ideas as requiring every sampled tick to be perfectly on one side unless you explicitly want a very strict rule. "
            f"{verify}"
        )


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
