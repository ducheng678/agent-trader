import json
import re
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from market_agent.events import current_utc_iso, normalize_event_record, parse_utc_iso
from market_agent.openai_usage import _tokenize_search_query, _trade_symbol_topic_aliases
from market_agent.symbols import (
    canonicalize_execution_symbol,
    normalize_candidate_key,
    split_execution_symbol,
)


class PassiveEventsMixin:
    def _passive_event_relevance_trade_symbol_context(self) -> Dict[str, Any]:
        context = dict(getattr(self, "trade_symbol_context", {}) or {})
        active_symbol = canonicalize_execution_symbol(getattr(self, "symbol", ""))
        if active_symbol and canonicalize_execution_symbol(context.get("execution_symbol", "")) != active_symbol:
            _, market_name = split_execution_symbol(active_symbol)
            market_name = market_name or active_symbol
            display_symbol = f"{market_name}-USDC" if market_name and not re.search(r"[-_]USDC$", market_name, re.IGNORECASE) else market_name
            canonical_symbol_key = normalize_candidate_key(display_symbol or market_name or active_symbol)
            context = {
                "execution_symbol": active_symbol,
                "display_name": display_symbol or active_symbol,
                "display_symbol": display_symbol or active_symbol,
                "trade_symbol_key": canonical_symbol_key,
                "canonical_symbol_key": canonical_symbol_key,
                "market_name": market_name,
                "market_spec": {"market_name": market_name},
            }
        return context
    @staticmethod
    def _passive_event_buffer_key(event: Optional[Dict[str, Any]]) -> str:
        if not isinstance(event, dict):
            return ""
        source = str(event.get("source", "") or "").strip()
        item_id = str(event.get("item_id", "") or "").strip()
        if source and item_id:
            return f"{source}:{item_id}"
        url = str(event.get("url", "") or "").strip()
        if source and url:
            return f"{source}:{url}"
        title = str(event.get("title", "") or "").strip()
        event_timestamp = str(event.get("event_timestamp", "") or event.get("published_at", "") or event.get("seen_at", "") or "").strip()
        if source or title or event_timestamp:
            return f"{source}:{title}:{event_timestamp}"
        return ""
    def _normalize_passive_event_symbol(self, symbol: Any) -> str:
        return str(symbol or "").strip().upper()
    def _passive_relevant_events_log_path_for_symbol(self, symbol: Any) -> Optional[Path]:
        normalized_symbol = self._normalize_passive_event_symbol(symbol)
        if not normalized_symbol:
            return None
        base_path = getattr(self, "passive_relevant_events_log_base_path", None) or getattr(self, "passive_relevant_events_log_path", None)
        if not isinstance(base_path, Path):
            return None
        safe_symbol = re.sub(r"[^A-Za-z0-9._-]+", "_", normalized_symbol)
        suffix = base_path.suffix or ".jsonl"
        return base_path.parent / f"{base_path.stem}_{safe_symbol}{suffix}"
    def _passive_relevant_event_buffer_for_symbol(self, symbol: Any) -> Deque[Dict[str, Any]]:
        normalized_symbol = self._normalize_passive_event_symbol(symbol)
        if not normalized_symbol:
            normalized_symbol = "UNKNOWN"
        bucket = self.llm_relevant_passive_events_by_symbol.get(normalized_symbol)
        if bucket is None:
            bucket = deque(maxlen=self.passive_llm_relevant_event_buffer_size)
            self.llm_relevant_passive_events_by_symbol[normalized_symbol] = bucket
        return bucket
    def _resolve_recent_passive_event_symbol(
        self,
        *,
        query_symbol: Any,
        query_trade_symbol_context: Optional[Dict[str, Any]],
        trade_symbol_context: Optional[Dict[str, Any]],
    ) -> str:
        normalized_query_symbol = canonicalize_execution_symbol(str(query_symbol or ""))
        normalized_active_symbol = canonicalize_execution_symbol(str(getattr(self, "symbol", "") or ""))
        for target_symbol in (normalized_query_symbol, normalized_active_symbol):
            if not target_symbol:
                continue
            for item in (trade_symbol_context, query_trade_symbol_context):
                context = item if isinstance(item, dict) else {}
                execution_symbol = canonicalize_execution_symbol(context.get("execution_symbol", ""))
                if execution_symbol == target_symbol:
                    return self._normalize_passive_event_symbol(context.get("display_name") or execution_symbol)
        for item in (trade_symbol_context, query_trade_symbol_context):
            context = item if isinstance(item, dict) else {}
            execution_symbol = canonicalize_execution_symbol(context.get("execution_symbol", ""))
            display_name = str(context.get("display_name") or context.get("display_symbol") or "").strip()
            if execution_symbol or display_name:
                return self._normalize_passive_event_symbol(display_name or execution_symbol)
        return self._normalize_passive_event_symbol(query_symbol)
    def _refresh_passive_llm_recent_events_from_helper(self) -> None:
        max_items = max(0, int(getattr(self, "passive_recent_materially_new_event_limit", 0) or 0))
        if max_items <= 0:
            self.passive_llm_recent_events = []
            self.passive_llm_recent_events_symbol = ""
            self.passive_llm_recent_events_source = ""
            return
        trade_symbol = self._resolve_recent_passive_event_symbol(
            query_symbol=getattr(self, "symbol", ""),
            query_trade_symbol_context=dict(getattr(self, "trade_symbol_context", {}) or {}),
            trade_symbol_context=None,
        )
        trade_symbol = self._normalize_passive_event_symbol(trade_symbol)
        if not trade_symbol:
            self.passive_llm_recent_events = []
            self.passive_llm_recent_events_symbol = ""
            self.passive_llm_recent_events_source = ""
            return
        events = self.engine._load_passive_recent_events_from_helper_materiality(
            trade_symbol,
            max_items=max_items,
        )
        self.passive_llm_recent_events = [dict(item) for item in list(events or []) if isinstance(item, dict)]
        self.passive_llm_recent_events_symbol = trade_symbol
        self.passive_llm_recent_events_source = "helper_materiality"
        self._persist_passive_llm_recent_events_state()
    def _hydrate_passive_llm_recent_events_state(self) -> bool:
        max_items = max(0, int(getattr(self, "passive_recent_materially_new_event_limit", 0) or 0))
        if max_items <= 0:
            return False
        path = getattr(self, "passive_llm_recent_events_state_path", None)
        if not isinstance(path, Path) or not path.exists():
            return False
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except Exception as exc:
            print(f"[warn] passive_llm_recent_events_state_hydrate_failed path={path} error={exc}")
            return False
        if not isinstance(payload, dict):
            return False
        symbol = self._normalize_passive_event_symbol(payload.get("symbol"))
        raw_events = payload.get("events")
        if not symbol or not isinstance(raw_events, list):
            return False
        cap = max_items * 2
        events: List[Dict[str, Any]] = []
        for item in raw_events:
            if not isinstance(item, dict):
                continue
            event = dict(item)
            key = self._passive_event_buffer_key(event)
            if not key:
                continue
            events = [existing for existing in events if self._passive_event_buffer_key(existing) != key]
            events.append(event)
        if cap > 0:
            events = events[-cap:]
        if not events:
            return False
        self.passive_llm_recent_events = events
        self.passive_llm_recent_events_symbol = symbol
        self.passive_llm_recent_events_source = "runtime_state"
        return True
    def _persist_passive_llm_recent_events_state(self) -> None:
        path = getattr(self, "passive_llm_recent_events_state_path", None)
        if not isinstance(path, Path):
            return
        symbol = self._normalize_passive_event_symbol(getattr(self, "passive_llm_recent_events_symbol", ""))
        if not symbol:
            return
        max_items = max(0, int(getattr(self, "passive_recent_materially_new_event_limit", 0) or 0))
        if max_items <= 0:
            return
        cap = max_items * 2
        events = [
            dict(item)
            for item in list(getattr(self, "passive_llm_recent_events", []) or [])
            if isinstance(item, dict)
        ]
        if cap > 0:
            events = events[-cap:]
        if not events:
            return
        payload = {
            "updated_at": current_utc_iso(),
            "kind": "passive_llm_recent_events_state",
            "symbol": symbol,
            "source": str(getattr(self, "passive_llm_recent_events_source", "") or ""),
            "max_items": max_items,
            "cap": cap,
            "events": events,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(f"{path.name}.tmp")
            with tmp_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False)
                handle.write("\n")
            tmp_path.replace(path)
        except Exception as exc:
            print(f"[warn] passive_llm_recent_events_state_persist_failed path={path} error={exc}")
    def _append_passive_llm_recent_event(self, event: Optional[Dict[str, Any]], symbol: Any) -> None:
        if not isinstance(event, dict):
            return
        max_items = max(0, int(getattr(self, "passive_recent_materially_new_event_limit", 0) or 0))
        if max_items <= 0:
            return
        normalized_symbol = self._normalize_passive_event_symbol(symbol)
        if not normalized_symbol:
            return
        if self.passive_llm_recent_events_symbol != normalized_symbol:
            self.passive_llm_recent_events = []
            self.passive_llm_recent_events_symbol = normalized_symbol
        normalized_event = dict(event)
        events = [dict(item) for item in list(getattr(self, "passive_llm_recent_events", []) or []) if isinstance(item, dict)]
        events.append(normalized_event)
        cap = max_items * 2
        if cap > 0:
            events = events[-cap:]
        self.passive_llm_recent_events = events
        self.passive_llm_recent_events_source = "runtime_state"
        self._persist_passive_llm_recent_events_state()
    def _remember_llm_relevant_passive_event(self, event: Optional[Dict[str, Any]], symbol: Any) -> None:
        if not isinstance(event, dict):
            return
        normalized_symbol = self._normalize_passive_event_symbol(symbol)
        if not normalized_symbol:
            return
        normalized = normalize_event_record(dict(event))
        key = self._passive_event_buffer_key(normalized)
        if not key:
            return
        normalized["buffer_symbol"] = normalized_symbol
        bucket = self._passive_relevant_event_buffer_for_symbol(normalized_symbol)
        items = [item for item in list(bucket) if self._passive_event_buffer_key(item) != key]
        items.append(normalized)
        self.llm_relevant_passive_events_by_symbol[normalized_symbol] = deque(items, maxlen=self.passive_llm_relevant_event_buffer_size)
        self._persist_llm_relevant_passive_event(normalized, normalized_symbol)
    def _hydrate_llm_relevant_passive_event_buffer_from_log(self) -> None:
        max_items = max(0, int(getattr(self, "event_context_max_items", 0) or 0))
        if max_items <= 0:
            return
        base_path = getattr(self, "passive_relevant_events_log_base_path", None) or getattr(self, "passive_relevant_events_log_path", None)
        if not isinstance(base_path, Path):
            return
        if not base_path.parent.exists():
            return
        pattern = f"{base_path.stem}_*{base_path.suffix or '.jsonl'}"
        hydrated: Dict[str, Deque[Dict[str, Any]]] = {}
        for path in sorted(base_path.parent.glob(pattern)):
            try:
                with path.open("r", encoding="utf-8") as handle:
                    lines = [line.strip() for line in handle if line.strip()]
            except Exception as exc:
                print(f"[warn] passive_relevant_event_buffer_hydrate_failed path={path} error={exc}")
                continue
            bucket_items: List[Dict[str, Any]] = []
            for line in lines[-max_items:]:
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if not isinstance(payload, dict):
                    continue
                symbol = self._normalize_passive_event_symbol(payload.get("symbol"))
                if not symbol:
                    suffix = path.name[len(base_path.stem) + 1 :]
                    if suffix.endswith(base_path.suffix or ".jsonl"):
                        suffix = suffix[: -len(base_path.suffix or ".jsonl")]
                    symbol = self._normalize_passive_event_symbol(suffix)
                event = payload.get("event")
                if not symbol or not isinstance(event, dict):
                    continue
                normalized = normalize_event_record(dict(event))
                key = self._passive_event_buffer_key(normalized)
                if not key:
                    continue
                normalized["buffer_symbol"] = symbol
                bucket_items = [item for item in bucket_items if self._passive_event_buffer_key(item) != key]
                bucket_items.append(normalized)
                hydrated[symbol] = deque(bucket_items, maxlen=self.passive_llm_relevant_event_buffer_size)
        self.llm_relevant_passive_events_by_symbol = hydrated
    def _recent_relevant_passive_events(self, *, symbol: Any, max_items: int, exclude_event: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        if max_items <= 0:
            return []
        exclude_key = self._passive_event_buffer_key(exclude_event)
        bucket = self.llm_relevant_passive_events_by_symbol.get(self._normalize_passive_event_symbol(symbol), deque())
        items = [
            dict(item)
            for item in list(bucket)
            if isinstance(item, dict) and self._passive_event_buffer_key(item) != exclude_key
        ]
        return items[-max_items:]
    def _persist_llm_relevant_passive_event(self, event: Dict[str, Any], symbol: Any) -> None:
        path = self._passive_relevant_events_log_path_for_symbol(symbol)
        if path is None:
            return
        payload = {
            "recorded_at": current_utc_iso(),
            "kind": "llm_relevant_passive_event",
            "symbol": self._normalize_passive_event_symbol(symbol),
            "source": str((event or {}).get("source", "") or ""),
            "item_id": str((event or {}).get("item_id", "") or ""),
            "title": str((event or {}).get("title", "") or ""),
            "url": str((event or {}).get("url", "") or ""),
            "event_timestamp": str((event or {}).get("event_timestamp", "") or ""),
            "event": dict(event or {}),
        }
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    def _event_relevance_text(self, event: Dict[str, Any]) -> str:
        raw = event.get("raw") if isinstance(event, dict) else None
        parts: List[str] = []
        for key in ("title", "summary", "url", "source", "category"):
            value = (event or {}).get(key) if isinstance(event, dict) else None
            if value not in (None, ""):
                parts.append(str(value))
        if isinstance(raw, dict):
            for key in ("title", "summary", "headline", "description", "path", "url"):
                value = raw.get(key)
                if value not in (None, ""):
                    parts.append(str(value))
        return " ".join(parts)
    def _event_allows_passive_query(self, event: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        raw = event.get("raw") if isinstance(event, dict) else None
        raw = raw if isinstance(raw, dict) else {}
        alert_enabled = raw.get("alert_enabled", True)
        alert_enabled_text = str(alert_enabled).strip().lower()
        source_role = str(raw.get("source_role", "") or "").strip().lower()
        disabled = (
            alert_enabled is False
            or alert_enabled_text in {"0", "false", "no", "off"}
            or source_role in {"backfill", "late_surface_backfill"}
        )
        return not disabled, {
            "allowed": not disabled,
            "alert_enabled": alert_enabled,
            "source_role": source_role,
            "reason": "alert_disabled_or_backfill" if disabled else "alert_enabled",
        }
    def _passive_event_published_age_on_seen_seconds(self, event: Dict[str, Any]) -> Optional[float]:
        if not isinstance(event, dict):
            return None
        raw = event.get("raw")
        raw = raw if isinstance(raw, dict) else {}
        published_at = str(event.get("published_at", "") or raw.get("published_at", "") or "").strip()
        seen_at = str(event.get("seen_at", "") or raw.get("seen_at", "") or "").strip()
        published_dt = parse_utc_iso(published_at)
        seen_dt = parse_utc_iso(seen_at)
        if published_dt is None or seen_dt is None:
            return None
        return (seen_dt - published_dt).total_seconds()
    def _event_allows_passive_realtime_trigger(self, event: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        max_age_hours = max(0.0, float(getattr(self, "passive_max_published_age_on_seen_hours", 0.0) or 0.0))
        max_age_seconds = max_age_hours * 3600.0
        age_seconds = self._passive_event_published_age_on_seen_seconds(event)
        title = str((event or {}).get("title", "") or "")
        source = str((event or {}).get("source", "") or "")
        published_at = str((event or {}).get("published_at", "") or "")
        seen_at = str((event or {}).get("seen_at", "") or "")
        debug = {
            "allowed": True,
            "reason": "fresh_or_age_gate_disabled",
            "source": source,
            "title": title,
            "published_at": published_at,
            "seen_at": seen_at,
            "age_seconds": age_seconds,
            "threshold_seconds": max_age_seconds,
        }
        if max_age_seconds <= 0:
            debug["reason"] = "age_gate_disabled"
            return True, debug
        if age_seconds is None:
            debug["reason"] = "missing_published_or_seen_at"
            return True, debug
        if age_seconds < 0:
            debug["reason"] = "published_after_seen_clock_skew"
            return True, debug
        if age_seconds > max_age_seconds:
            debug["allowed"] = False
            debug["reason"] = "published_age_on_seen_exceeds_threshold"
            return False, debug
        return True, debug
    def _event_is_relevant_for_passive_query(self, event: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        text = self._event_relevance_text(event)
        tokens = set(_tokenize_search_query(text))
        trade_symbol_context = self._passive_event_relevance_trade_symbol_context()
        alias_map = _trade_symbol_topic_aliases(trade_symbol_context)
        macro_tokens = {
            "fed", "fomc", "powell", "cpi", "ppi", "inflation", "payroll", "payrolls", "nfp",
            "jobs", "jobless", "unemployment", "retail", "sales", "pmi", "ism", "gdp", "rates",
            "rate", "yield", "yields", "treasury", "treasuries", "tsy", "bond", "bonds", "credit", "dollar", "usd", "tariff", "sanction", "sanctions",
            "war", "ceasefire", "strike", "strikes", "missile", "missiles", "opec", "iran", "iranian"
        }
        best_topic = ""
        best_score = 0
        matched_aliases: List[str] = []
        for topic, aliases in alias_map.items():
            matched = sorted(tokens & aliases)
            score = len(matched)
            if score > best_score:
                best_score = score
                best_topic = topic
                matched_aliases = matched
        matched_macro = sorted(tokens & macro_tokens)
        relevant = bool(best_score > 0 or matched_macro)
        return relevant, {
            "relevant": relevant,
            "token_count": len(tokens),
            "best_topic": best_topic,
            "best_score": best_score,
            "matched_aliases": matched_aliases,
            "matched_macro_tokens": matched_macro,
        }
    def _filter_recent_events_for_active_helper(self, recent_events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        max_items = max(0, int(getattr(self, "event_context_max_items", 0) or 0))
        if max_items <= 0:
            return []
        candidate_window = list(recent_events or [])
        helper_engine = getattr(self, "engine", None)
        checkpoint_dt = None
        if helper_engine is not None and hasattr(helper_engine, "_helper_materiality_checkpoint_timestamp"):
            try:
                checkpoint_dt = helper_engine._helper_materiality_checkpoint_timestamp()
            except Exception:
                checkpoint_dt = None
        if checkpoint_dt is not None:
            candidate_window = [
                dict(item)
                for item in candidate_window
                if isinstance(item, dict)
                and (parse_utc_iso(str(item.get("event_timestamp", "") or item.get("published_at", "") or item.get("seen_at", "") or "")) or datetime.min.replace(tzinfo=timezone.utc)) > checkpoint_dt
            ]
        candidate_window = candidate_window[-(max_items * 4):]
        filtered = [
            dict(item)
            for item in candidate_window
            if isinstance(item, dict) and self._event_is_relevant_for_passive_query(item)[0]
        ]
        return filtered[-max_items:]
