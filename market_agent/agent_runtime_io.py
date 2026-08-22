import json
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

from market_agent.events import current_utc_iso
from market_agent.playbook import GenericPlaybook
from market_agent.presentation import _status_position_brief, build_status_summary
from market_agent.runtime_views import build_playbook_execution_view
from market_agent.symbols import canonicalize_execution_symbol, render_query_template
from market_agent.utils import format_query_amount


class AgentRuntimeIOMixin:
    def _warm_up_market_catalog(self) -> bool:
        if bool(getattr(self, "_market_catalog_warmup_attempted", False)):
            return bool(getattr(self, "_market_catalog_warmup_succeeded", False))
        self._market_catalog_warmup_attempted = True
        reader = getattr(self, "reader", None)
        if reader is None or not hasattr(reader, "get_market_catalog"):
            self._market_catalog_warmup_succeeded = False
            return False
        started = time.time()
        try:
            catalog = reader.get_market_catalog()
            count = len(catalog) if isinstance(catalog, dict) else 0
        except Exception as exc:
            elapsed = max(0.0, time.time() - started)
            self._market_catalog_warmup_succeeded = False
            print(f"[market_catalog_warmup] failed elapsed={format_query_amount(elapsed)}s error={exc}")
            self._audit_event(
                "market_catalog_warmup_failed",
                {
                    "elapsed_seconds": elapsed,
                    "error": str(exc),
                },
            )
            return False
        elapsed = max(0.0, time.time() - started)
        self._market_catalog_warmup_succeeded = True
        print(f"[market_catalog_warmup] ok markets={count} elapsed={format_query_amount(elapsed)}s")
        self._audit_event(
            "market_catalog_warmup",
            {
                "markets_count": count,
                "elapsed_seconds": elapsed,
            },
        )
        return True

    def log_startup(self) -> None:
        print(f"[effective_symbol] {self.symbol}")
        trade_symbol_context = dict(getattr(self, "trade_symbol_context", {}) or {})
        print(
            "[trade_symbol] "
            + json.dumps(
                {
                    "trade_symbol_key": trade_symbol_context.get("trade_symbol_key") or trade_symbol_context.get("candidate_key"),
                    "display_name": trade_symbol_context.get("display_name"),
                    "execution_symbol": trade_symbol_context.get("execution_symbol"),
                    "tradable_on_hyperliquid": trade_symbol_context.get("tradable_on_hyperliquid"),
                },
                ensure_ascii=False,
            )
        )
        print(f"[query_template] {self.user_query_template or '<default>'}")
        print(f"[max_planned_loss_ratio] {self.max_planned_loss_ratio:.4f}")
        print(f"[max_planned_loss_usd_fallback] {self.max_planned_loss_usd_fallback:.2f}")
        print(f"[local_risk_tolerance_usd] {self.local_risk_tolerance_usd:.2f}")
        print(f"[local_no_change_close_fraction_tolerance] {self.local_no_change_close_fraction_tolerance:.4f}")
        print(f"[openai_reasoning_effort] active={self.engine.active_reasoning_effort} passive={self.engine.passive_reasoning_effort}")
        print(f"[enable_active_query] {self.enable_active_query}")
        print(f"[enable_active_playbook] {self.enable_active_playbook}")
        print(f"[enable_active_auto_requery] {self.enable_active_auto_requery}")
        helper_profile = self._market_profile_for_symbol(self._helper_reset_symbol_hint())
        if helper_profile is not None and helper_profile.helper_reset_time is not None:
            helper_reset_time = helper_profile.helper_reset_time
            helper_reset_timezone = helper_profile.helper_reset_timezone_name or helper_profile.timezone_name
            print(
                "[helper_reset_profile] "
                f"{helper_profile.name} {helper_reset_timezone} "
                f"{helper_reset_time[0]:02d}:{helper_reset_time[1]:02d}:{helper_reset_time[2]:02d}"
            )
            pre_disabled_reset_time = helper_profile.pre_disabled_weekday_reset_time
            if pre_disabled_reset_time is not None:
                print(
                    "[pre_disabled_weekday_reset_profile] "
                    f"{helper_profile.name} {helper_profile.timezone_name} "
                    f"{pre_disabled_reset_time[0]:02d}:"
                    f"{pre_disabled_reset_time[1]:02d}:"
                    f"{pre_disabled_reset_time[2]:02d}"
                )
        else:
            print("[helper_reset_profile] <disabled:no_profile>")
        print(
            "[next_helper_reset_at] "
            + (
                self.next_helper_reset_at.isoformat()
                if isinstance(getattr(self, "next_helper_reset_at", None), datetime)
                else "<disabled>"
            )
        )
        print(f"[active_query_interval_seconds] {self.active_query_interval_seconds}")
        print(f"[active_management_query_interval_seconds] {self.active_management_query_interval_seconds}")
        print(f"[loop_exception_sleep_seconds] {self.loop_exception_sleep_seconds}")
        print(f"[hyperliquid_transient_error_sleep_seconds] {self.hyperliquid_transient_error_sleep_seconds}")
        print(f"[enable_passive_event_query] {self.enable_passive_event_query}")
        print(f"[passive_max_published_age_on_seen_hours] {self.passive_max_published_age_on_seen_hours:.4f}")
        disabled_weekdays = (
            list(helper_profile.low_liquidity_trade_disabled_weekdays or ())
            if helper_profile is not None
            else []
        )
        print(f"[low_liquidity_trade_disabled_weekdays] {disabled_weekdays}")
        print(f"[openai_include_chart_images] {self.engine.include_chart_images}")
        print(f"[openai_include_passive_chart_images] {self.engine.include_passive_chart_images}")
        print(f"[events_path] {self.events_path.resolve()}")
        print(f"[recent_event_window_hours] {self.event_recent_window_hours}")
        print(f"[recent_event_max_items] {self.event_context_max_items}")
        print(f"[recent_event_buffer_size] {self.event_buffer_size}")
        print(f"[audit_log_enabled] {self.enable_audit_log}")
        print(f"[audit_log_path] {self.audit_log_path.resolve()}")
        print(f"[status_log_enabled] {self.enable_status_log}")
        print(f"[status_log_path] {self.status_log_path.resolve()}")
        print(f"[live_trading] {self.executor.enabled}")
        print("[startup_all_positions]")
        print(self.reader.format_all_positions(self.reader.get_all_positions()))
        print()
    def _emit_startup_ready(self) -> None:
        next_helper_reset_at = getattr(self, "next_helper_reset_at", None)
        state_path = getattr(self, "passive_llm_recent_events_state_path", None)
        payload = {
            "passive_recent_events_count": len(list(getattr(self, "passive_llm_recent_events", []) or [])),
            "passive_recent_events_symbol": str(getattr(self, "passive_llm_recent_events_symbol", "") or ""),
            "passive_recent_events_source": str(getattr(self, "passive_llm_recent_events_source", "") or ""),
            "passive_recent_events_state_hydrated": bool(getattr(self, "passive_llm_recent_events_state_hydrated", False)),
            "passive_recent_events_state_path": str(state_path) if state_path is not None else "",
            "next_helper_reset_at": (
                next_helper_reset_at.isoformat()
                if isinstance(next_helper_reset_at, datetime)
                else None
            ),
            "startup_live_tpsl_restore_attempted": bool(getattr(self, "_startup_live_tpsl_restore_attempted", False)),
            "risk_session_active": getattr(self, "risk_session", None) is not None,
        }
        self._audit_event("startup_ready", payload)
    def _ensure_audit_defaults(self) -> None:
        if not hasattr(self, "enable_audit_log"):
            self.enable_audit_log = False
        if not hasattr(self, "audit_log_path"):
            self.audit_log_path = Path("logs/unified_market_agent_audit.jsonl")
        if not hasattr(self, "enable_status_log"):
            self.enable_status_log = False
        if not hasattr(self, "status_log_path"):
            self.status_log_path = Path("logs/unified_market_agent_status.jsonl")
    def _build_llm_cost_projection(self, per_query_cost_usd: float) -> Dict[str, Any]:
        active_queries_per_day = (
            86400.0 / max(self.active_query_interval_seconds, 1.0)
            if self.enable_active_query and self.enable_active_auto_requery and self.active_query_interval_seconds > 0
            else 0.0
        )
        management_queries_per_day = (
            86400.0 / max(self.active_management_query_interval_seconds, 1.0)
            if self.enable_active_query and self.enable_active_auto_requery and self.active_management_query_interval_seconds > 0
            else 0.0
        )
        return {
            "active_no_position": {
                "queries_per_day": active_queries_per_day,
                "estimated_cost_per_day_usd": per_query_cost_usd * active_queries_per_day,
                "estimated_cost_per_30d_usd": per_query_cost_usd * active_queries_per_day * 30.0,
            },
            "active_with_position": {
                "queries_per_day": management_queries_per_day,
                "estimated_cost_per_day_usd": per_query_cost_usd * management_queries_per_day,
                "estimated_cost_per_30d_usd": per_query_cost_usd * management_queries_per_day * 30.0,
            },
            "note": "Passive event-triggered queries are not included in this projection because their frequency depends on the event stream. All cost figures here are estimates based on Responses usage fields and configured pricing.",
        }
    def _augment_engine_debug_with_cost_metrics(self, engine_debug: Dict[str, Any]) -> Dict[str, Any]:
        usage_cost = engine_debug.get("usage_cost") if isinstance(engine_debug.get("usage_cost"), dict) else {}
        if not usage_cost or not usage_cost.get("known"):
            return engine_debug
        engine_debug["cost_projection"] = self._build_llm_cost_projection(
            float(usage_cost.get("estimated_total_cost_usd", usage_cost.get("total_cost_usd", 0.0)) or 0.0)
        )
        return engine_debug
    def _print_json_block(self, tag: str, payload: Any) -> None:
        if not getattr(self, "console_verbose_json", False):
            return
        print(f"[{tag}]")
        if isinstance(payload, str):
            print(payload)
        else:
            print(json.dumps(payload, ensure_ascii=False, indent=2))
        print()
    def _emit_status_line(self, event_type: str, summary: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self._ensure_audit_defaults()
        if not self.enable_status_log:
            return
        entry = {
            "ts": current_utc_iso(),
            "event": event_type,
            "symbol": getattr(self, "symbol", ""),
            "mode": getattr(self, "current_mode", None),
            "playbook_reason": getattr(self, "current_playbook_reason", ""),
            "summary": summary,
            "payload": payload or {},
        }
        print(f"[status] {entry['ts']} {summary}")
        self.status_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.status_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    def _emit_status_from_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        summary = build_status_summary(event_type, payload)
        if not summary:
            return
        self._emit_status_line(event_type, str(summary.get("summary", "") or event_type), payload)
    @staticmethod
    def _request_exception_text(exc: Exception) -> str:
        parts = [str(exc or "")]
        response = getattr(exc, "response", None)
        request = getattr(exc, "request", None)
        if response is not None:
            parts.append(str(getattr(response, "url", "") or ""))
            request = request or getattr(response, "request", None)
        if request is not None:
            parts.append(str(getattr(request, "url", "") or ""))
        return " ".join(part for part in parts if part)
    @classmethod
    def _is_transient_hyperliquid_error(cls, exc: Exception) -> bool:
        text = cls._request_exception_text(exc).lower()
        is_hyperliquid = "api.hyperliquid.xyz" in text or "hyperliquid" in text
        if not is_hyperliquid:
            return False
        if isinstance(exc, (requests.exceptions.Timeout, requests.exceptions.ConnectionError)):
            return True
        if isinstance(exc, requests.exceptions.HTTPError):
            response = getattr(exc, "response", None)
            status_code = int(getattr(response, "status_code", 0) or 0) if response is not None else 0
            return status_code == 429 or status_code >= 500
        if isinstance(exc, requests.exceptions.RequestException):
            return True
        return False
    def _runtime_error_sleep_seconds(self, exc: Exception) -> float:
        if self._is_transient_hyperliquid_error(exc):
            return max(
                float(getattr(self, "loop_exception_sleep_seconds", 5.0) or 5.0),
                float(getattr(self, "hyperliquid_transient_error_sleep_seconds", 30.0) or 30.0),
            )
        return float(getattr(self, "loop_exception_sleep_seconds", 5.0) or 5.0)
    def _record_runtime_error(
        self,
        stage: str,
        exc: Exception,
        extra: Optional[Dict[str, Any]] = None,
        *,
        retry_after_seconds: Optional[float] = None,
    ) -> None:
        retry_after = (
            float(retry_after_seconds)
            if retry_after_seconds is not None
            else float(getattr(self, "loop_exception_sleep_seconds", 5.0) or 5.0)
        )
        payload: Dict[str, Any] = {
            "stage": stage,
            "error_type": type(exc).__name__,
            "message": str(exc),
            "retry_after_seconds": retry_after,
            "traceback": traceback.format_exc(),
        }
        if extra:
            payload.update(extra)
        print(
            f"[runtime_error] stage={stage} type={type(exc).__name__} "
            f"retry_in={format_query_amount(retry_after)}s message={exc}"
        )
        self._audit_event("runtime_error", payload)
    def _audit_event(self, event_type: str, payload: Optional[Dict[str, Any]] = None) -> None:
        self._ensure_audit_defaults()
        self._emit_status_from_event(event_type, payload)
        if not self.enable_audit_log:
            return
        entry = {
            "ts": current_utc_iso(),
            "event": event_type,
            "symbol": getattr(self, "symbol", ""),
            "mode": getattr(self, "current_mode", None),
            "playbook_reason": getattr(self, "current_playbook_reason", ""),
            "payload": payload or {},
        }
        self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        with self.audit_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
    def render_user_query(self, all_positions: dict, trade_symbol_context: Optional[Dict[str, Any]] = None) -> str:
        context = trade_symbol_context if isinstance(trade_symbol_context, dict) else {}
        primary_label = str(
            context.get("display_name", "")
            or context.get("display_symbol", "")
            or context.get("trade_symbol_key", "")
            or context.get("candidate_key", "")
            or ""
        ).strip()
        tradable_symbol = canonicalize_execution_symbol(context.get("execution_symbol", ""))
        variables = {
            "symbol": primary_label,
            "active_symbol": primary_label,
            "trade_symbol": primary_label,
            "tradable_trade_symbol": tradable_symbol,
        }
        return render_query_template(self.user_query_template, primary_label, variables)
    def print_playbook(self, playbook: GenericPlaybook, mode: str, all_positions: dict, symbol_position: dict) -> None:
        self._print_json_block("all_positions", self.reader.format_all_positions(all_positions))
        print(f"[selected_symbol] {playbook.selected_symbol or '<empty>'}")
        position_brief = _status_position_brief(symbol_position)
        if position_brief:
            print(f"[position] {position_brief}")
        position_label = canonicalize_execution_symbol(self.symbol or "") or "<none>"
        self._print_json_block(f"{position_label}_position", self.reader.format_symbol_position(symbol_position))
        print()
        self._print_json_block("playbook_json", playbook.to_dict())
        self._print_json_block("playbook_execution_view", build_playbook_execution_view(playbook, symbol_position))
