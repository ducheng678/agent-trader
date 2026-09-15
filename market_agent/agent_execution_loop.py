from concurrent.futures import Future, ThreadPoolExecutor
import inspect
import json
from threading import Event
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from market_agent.calibration import extract_raw_confidence_value
from market_agent.conditions import evaluate_condition
from market_agent.logging_utils import print_line
from market_agent.models import (
    SCENARIO_RUNTIME_KEY,
    ManagementDecision,
    PositionManagementPlan,
    StrategyDecision,
    observe_when_all_contains_price,
)
from market_agent.openai_usage import merge_usage_costs, merge_usage_dicts
from market_agent.playbook import GenericPlaybook
from market_agent.positions import snapshot_has_open_position
from market_agent.presentation import (
    _status_decision_brief,
    _status_event_brief,
    _status_execution_result_brief,
    _status_trade_symbol_price_brief,
)
from market_agent.runtime_views import (
    build_decision_execution_view,
    build_empty_position_management_plan,
    build_playbook_execution_view,
    build_scenario_execution_view,
)
from market_agent.symbols import canonicalize_execution_symbol
from market_agent.utils import format_display_price, safe_float


class ExecutionLoopMixin:
    @staticmethod
    def _call_with_optional_position_before(
        method: Any,
        *args: Any,
        position_before: Optional[dict] = None,
        execution_mid_price: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        try:
            parameters = inspect.signature(method).parameters
            supports_var_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values())
        except (TypeError, ValueError):
            parameters = {}
            supports_var_kwargs = False
        if position_before is not None and ("position_before" in parameters or supports_var_kwargs):
            kwargs["position_before"] = position_before
        if execution_mid_price is not None and ("execution_mid_price" in parameters or supports_var_kwargs):
            kwargs["execution_mid_price"] = execution_mid_price
        return method(*args, **kwargs)

    def _fetch_position_context(
        self,
        symbol: Optional[str] = None,
        *,
        thread_name_prefix: str = "position-context",
    ) -> Tuple[Dict[str, Any], dict, Optional[float]]:
        target_symbol = canonicalize_execution_symbol(symbol or self.symbol or "")
        with ThreadPoolExecutor(max_workers=2, thread_name_prefix=thread_name_prefix) as executor:
            positions_future = executor.submit(self.reader.get_all_positions)
            mid_future = executor.submit(self.reader.get_mid_price, target_symbol) if target_symbol else None
            all_positions = positions_future.result()
            execution_mid_price = safe_float(mid_future.result(), None) if mid_future is not None else None
        position_before = self.reader.get_position_snapshot(
            target_symbol,
            all_positions=all_positions,
            current_price=execution_mid_price,
        ) if target_symbol else self._empty_runtime_snapshot(all_positions)
        return all_positions, position_before, execution_mid_price

    def _fetch_execution_position_context(self) -> Tuple[dict, Optional[float]]:
        _, position_before, execution_mid_price = self._fetch_selected_symbol_position_context(self.symbol)
        return position_before, execution_mid_price

    def _fetch_selected_symbol_position_context(
        self,
        symbol: Optional[str] = None,
    ) -> Tuple[Dict[str, Any], dict, Optional[float]]:
        target_symbol = canonicalize_execution_symbol(symbol or self.symbol or "")
        if target_symbol and hasattr(self.reader, "get_selected_symbol_position_context"):
            context = self.reader.get_selected_symbol_position_context(target_symbol)
            if not isinstance(context, dict):
                raise RuntimeError("selected symbol position context did not return a dict")
            all_positions = context.get("all_positions") if isinstance(context.get("all_positions"), dict) else {}
            position_snapshot = context.get("position_snapshot") if isinstance(context.get("position_snapshot"), dict) else {}
            execution_mid_price = safe_float(context.get("mid_price"), None)
            if not position_snapshot:
                position_snapshot = self.reader.get_position_snapshot(
                    target_symbol,
                    all_positions=all_positions,
                    current_price=execution_mid_price,
                )
            return all_positions, position_snapshot, execution_mid_price
        return self._fetch_position_context(
            target_symbol,
            thread_name_prefix="selected-symbol-context-fallback",
        )

    def _fetch_pre_execution_position_context(
        self,
        symbol: Optional[str] = None,
        *,
        reason: str = "",
    ) -> Tuple[Dict[str, Any], dict, Optional[float]]:
        target_symbol = canonicalize_execution_symbol(symbol or self.symbol or "")
        use_selected_context = (
            reason == "passive_event_trigger"
            and target_symbol
            and hasattr(self.reader, "get_selected_symbol_position_context")
        )
        if not use_selected_context:
            return self._fetch_position_context(
                target_symbol,
                thread_name_prefix="pre-execution-context",
        )
        return self._fetch_selected_symbol_position_context(target_symbol)

    @staticmethod
    def _management_exposure_target_side(action: str) -> str:
        action = str(action or "").strip()
        if action in {"long", "add_to_long", "reverse_to_long"}:
            return "long"
        if action in {"short", "add_to_short", "reverse_to_short"}:
            return "short"
        return ""

    def _passive_entry_deviation_guard(
        self,
        decision: ManagementDecision,
        *,
        plan_name: str,
        position_before: dict,
        execution_mid_price: Optional[float],
    ) -> Optional[dict]:
        action = str(getattr(decision, "action", "") or "").strip()
        target_side = self._management_exposure_target_side(action)
        if not target_side:
            return None
        entry_price = safe_float(getattr(decision, "entry_price", None), None)
        stop_loss_price = safe_float(getattr(decision, "stop_loss_price", None), None)
        mid_price = safe_float(execution_mid_price, None)
        if entry_price is None or stop_loss_price is None or mid_price is None:
            return None
        if entry_price <= 0 or stop_loss_price <= 0 or mid_price <= 0:
            return None
        risk_distance = abs(entry_price - stop_loss_price)
        if risk_distance <= 0:
            return None
        tolerance_r = 0.25
        tolerance_usd = risk_distance * tolerance_r
        if target_side == "long":
            adverse_deviation = mid_price - entry_price
            allowed_price = entry_price + tolerance_usd
            blocked = mid_price > allowed_price
            comparator = "execution_mid_price > entry_price + 0.25R"
        else:
            adverse_deviation = entry_price - mid_price
            allowed_price = entry_price - tolerance_usd
            blocked = mid_price < allowed_price
            comparator = "execution_mid_price < entry_price - 0.25R"
        if not blocked:
            return None
        guard = {
            "enabled": True,
            "blocked": True,
            "reason": "entry_deviation_exceeds_0_25R",
            "action": action,
            "target_side": target_side,
            "entry_price": entry_price,
            "stop_loss_price": stop_loss_price,
            "execution_mid_price": mid_price,
            "risk_distance": risk_distance,
            "tolerance_r": tolerance_r,
            "tolerance_usd": tolerance_usd,
            "allowed_price": allowed_price,
            "adverse_deviation_usd": adverse_deviation,
            "adverse_deviation_r": adverse_deviation / risk_distance,
            "comparator": comparator,
        }
        return {
            "mode": "live" if bool(getattr(self.executor, "enabled", False)) else "dry_run",
            "symbol": self.symbol,
            "plan_name": plan_name,
            "position_before": position_before,
            "decision": decision.to_dict(),
            "actions": [],
            "local_blocked": True,
            "entry_deviation_guard": guard,
            "message": (
                "Entry deviation guard blocked passive market execution: "
                f"{format_display_price(mid_price)} is more than 0.25R away from "
                f"strategy entry {format_display_price(entry_price)} for {target_side}."
            ),
        }

    def _passive_basis_chase_guard(
        self,
        decision: ManagementDecision,
        *,
        plan_name: str,
        position_before: dict,
        execution_mid_price: Optional[float],
    ) -> Optional[dict]:
        if not bool(getattr(self, "risk_basis_chase_guard_enabled", False)):
            return None
        action = str(getattr(decision, "action", "") or "").strip()
        target_side = self._management_exposure_target_side(action)
        if not target_side:
            return None
        basis_context = self._market_basis_context_for_side(
            symbol=str(getattr(self, "symbol", "") or ""),
            side=target_side,
            snapshot_mid_price=execution_mid_price,
        )
        if not bool(basis_context.get("available")):
            return None
        has_open_position = snapshot_has_open_position(position_before)
        first_entry_actions = {"long", "short"}
        if action in first_entry_actions and not has_open_position:
            threshold = max(0.0, float(getattr(self, "risk_basis_chase_first_entry_threshold_usd", 1.5) or 1.5))
            threshold_kind = "first_entry"
        else:
            threshold = max(0.0, float(getattr(self, "risk_basis_chase_add_reverse_threshold_usd", 1.0) or 1.0))
            threshold_kind = "add_or_reverse"
        favorable_basis = float(basis_context.get("favorable_basis", 0.0) or 0.0)
        if threshold <= 0.0 or favorable_basis < threshold:
            return None
        guard = {
            "enabled": True,
            "blocked": True,
            "reason": "oracle_mid_favorable_basis_exceeds_chase_threshold",
            "action": action,
            "target_side": target_side,
            "threshold_kind": threshold_kind,
            "threshold_usd": threshold,
            "favorable_basis_usd": favorable_basis,
            "basis_context": basis_context,
        }
        self._audit_event("basis_chase_guard_blocked", guard)
        oracle_px = safe_float(basis_context.get("oraclePx"), None)
        mid_px = safe_float(basis_context.get("midPx"), None)
        return {
            "mode": "live" if bool(getattr(self.executor, "enabled", False)) else "dry_run",
            "symbol": self.symbol,
            "plan_name": plan_name,
            "position_before": position_before,
            "decision": decision.to_dict(),
            "actions": [],
            "local_blocked": True,
            "basis_chase_guard": guard,
            "message": (
                "Basis chase guard blocked passive market execution: "
                f"favorable oracle-mid basis {format_display_price(favorable_basis)} >= "
                f"threshold {format_display_price(threshold)} for {target_side} "
                f"(oracle {format_display_price(oracle_px)}, mid {format_display_price(mid_px)})."
            ),
        }

    def _build_prefetched_passive_query_trade_symbol_context(self, prefetched_context: Dict[str, Any]) -> Dict[str, Any]:
        context = dict(prefetched_context or {}) if isinstance(prefetched_context, dict) else {}
        configured_context = dict(getattr(self, "trade_symbol_context", {}) or {})
        if not configured_context:
            raise RuntimeError("TRADE_SYMBOL context is not configured.")
        configured_symbol = canonicalize_execution_symbol(configured_context.get("execution_symbol", ""))
        execution_symbol = canonicalize_execution_symbol(context.get("execution_symbol", ""))
        tokens = [
            context.get("display_name", ""),
            context.get("display_symbol", ""),
            context.get("trade_symbol_key", ""),
            context.get("candidate_key", ""),
            context.get("canonical_symbol_key", ""),
            execution_symbol,
        ]
        for token in tokens:
            selected_context = self._find_trade_symbol_by_selected_symbol(str(token or ""), dict(getattr(self, "trade_symbol_context", {}) or {}))
            if selected_context is not None:
                return dict(selected_context)
        if execution_symbol and configured_symbol and execution_symbol != configured_symbol:
            raise RuntimeError(
                f"Passive prefetch symbol {execution_symbol} does not match configured TRADE_SYMBOL {configured_symbol}."
            )
        return configured_context

    @staticmethod
    def _update_trade_symbol_context_price(
        trade_symbol_context: Dict[str, Any],
        symbol: str,
        current_price: Optional[float],
    ) -> Dict[str, Any]:
        target_symbol = canonicalize_execution_symbol(symbol)
        context = dict(trade_symbol_context or {}) if isinstance(trade_symbol_context, dict) else {}
        if canonicalize_execution_symbol(context.get("execution_symbol", "")) == target_symbol:
            context["current_price"] = current_price
        return context

    def _passive_event_judge_prefetch_event_key(self, event: Optional[Dict[str, Any]]) -> str:
        if hasattr(self, "_passive_event_buffer_key"):
            try:
                return str(self._passive_event_buffer_key(event) or "")
            except Exception:
                pass
        if not isinstance(event, dict):
            return ""
        source = str(event.get("source", "") or "").strip()
        item_id = str(event.get("item_id", "") or "").strip()
        title = str(event.get("title", "") or "").strip()
        event_timestamp = str(event.get("event_timestamp", "") or event.get("published_at", "") or event.get("seen_at", "") or "").strip()
        return f"{source}:{item_id}:{title}:{event_timestamp}"

    def _passive_event_judge_prefetch_executor(self) -> ThreadPoolExecutor:
        executor = getattr(self, "_passive_event_judge_executor", None)
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="passive-step1")
            self._passive_event_judge_executor = executor
        return executor

    def _passive_chart_context_prefetch_executor(self) -> ThreadPoolExecutor:
        executor = getattr(self, "_passive_chart_context_executor", None)
        if executor is None:
            executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="passive-chart")
            self._passive_chart_context_executor = executor
        return executor

    def _passive_event_judge_wake_event(self) -> Event:
        wake_event = getattr(self, "_passive_event_judge_ready_event", None)
        if not isinstance(wake_event, Event):
            wake_event = Event()
            self._passive_event_judge_ready_event = wake_event
        return wake_event

    def _signal_passive_event_judge_ready(self, _future: Future) -> None:
        self._passive_event_judge_wake_event().set()

    def _wait_for_loop_wake(self, timeout_seconds: float) -> None:
        wake_event = self._passive_event_judge_wake_event()
        if wake_event.wait(max(0.0, float(timeout_seconds or 0.0))):
            wake_event.clear()

    def _shutdown_passive_event_judge_prefetch(self) -> None:
        executor = getattr(self, "_passive_event_judge_executor", None)
        chart_executor = getattr(self, "_passive_chart_context_executor", None)
        self._passive_event_judge_executor = None
        self._passive_chart_context_executor = None
        self._passive_event_judge_future = None
        self._passive_event_judge_request = None
        self._passive_event_judge_queued_event = None
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        if chart_executor is not None:
            chart_executor.shutdown(wait=False, cancel_futures=True)

    def _passive_event_judge_prefetch_pending(self) -> bool:
        future = getattr(self, "_passive_event_judge_future", None)
        return isinstance(future, Future) and not future.done()

    @staticmethod
    def _normalize_passive_event_judge_prefetch_events(events: Any) -> List[Dict[str, Any]]:
        if isinstance(events, dict):
            return [dict(events)]
        if isinstance(events, list):
            return [dict(event) for event in events if isinstance(event, dict)]
        return []

    def _build_lightweight_passive_trade_symbol_context(self) -> Dict[str, Any]:
        preferred_symbol = canonicalize_execution_symbol(getattr(self, "symbol", "") or "")
        configured_context = dict(getattr(self, "trade_symbol_context", {}) or {})
        if not configured_context:
            raise RuntimeError("TRADE_SYMBOL context is not configured.")
        configured_symbol = canonicalize_execution_symbol(configured_context.get("execution_symbol", ""))
        if preferred_symbol and configured_symbol and preferred_symbol != configured_symbol:
            raise RuntimeError(
                f"Active symbol {preferred_symbol} does not match configured TRADE_SYMBOL {configured_symbol}."
            )
        item = configured_context
        execution_symbol = canonicalize_execution_symbol(item.get("execution_symbol", ""))
        return {
            "trade_symbol_key": str(item.get("trade_symbol_key", "") or item.get("candidate_key", "") or "").strip().upper(),
            "canonical_symbol_key": str(item.get("canonical_symbol_key", "") or item.get("trade_symbol_key", "") or item.get("candidate_key", "") or "").strip().upper(),
            "market_name": str(item.get("market_name", "") or "").strip(),
            "display_symbol": str(item.get("display_symbol", "") or item.get("display_name", "") or "").strip(),
            "display_name": str(item.get("display_name", "") or "").strip(),
            "configured_execution_symbol": canonicalize_execution_symbol(item.get("configured_execution_symbol", "")),
            "execution_symbol": execution_symbol,
            "tradable_on_hyperliquid": bool(execution_symbol),
            "current_price": None,
            "market_spec": {},
        }

    def _build_passive_event_judge_prefetch_request(self, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        if not isinstance(event, dict):
            return None
        event_payload = dict(event)
        passive_realtime_allowed = bool(event_payload.pop("_passive_realtime_allowed", True))
        passive_realtime_debug = event_payload.pop("_passive_realtime_debug", {})
        if not isinstance(passive_realtime_debug, dict):
            passive_realtime_debug = {}
        trade_symbol_context = self._build_lightweight_passive_trade_symbol_context()
        active_symbol = canonicalize_execution_symbol(getattr(self, "symbol", "") or "")
        market_mainline_context, market_mainline_debug = self.engine._get_cached_helper_market_mainline_context(
            trade_symbol_context=trade_symbol_context,
            active_symbol=active_symbol,
        )
        selection_context = self.engine._sanitize_playbook_trade_symbol_context(trade_symbol_context)
        selected_context = self.engine._select_playbook_trade_symbol_context(
            selection_context,
            active_symbol=active_symbol,
            market_mainline_context=market_mainline_context,
        )
        if selected_context is None:
            return None
        trade_symbol = str(
            selected_context.get("display_name", "")
            or selected_context.get("trade_symbol_key", "")
            or selected_context.get("candidate_key", "")
            or ""
        ).strip()
        if not trade_symbol:
            return None
        passive_recent_symbol = self._normalize_passive_event_symbol(getattr(self, "passive_llm_recent_events_symbol", ""))
        trade_symbol_normalized = self._normalize_passive_event_symbol(trade_symbol)
        if passive_recent_symbol and passive_recent_symbol == trade_symbol_normalized:
            recent_events = [
                dict(item)
                for item in list(getattr(self, "passive_llm_recent_events", []) or [])
                if isinstance(item, dict)
            ]
        else:
            recent_events = []
        search_mode = self.engine._resolve_search_mode("passive_event_trigger")
        if search_mode == "context_only":
            phase = "context_only"
        elif search_mode == "always":
            phase = "verified"
        else:
            phase = "fast"
        return {
            "event_key": self._passive_event_judge_prefetch_event_key(event_payload),
            "trigger_event": event_payload,
            "recent_events": recent_events,
            "recent_events_source": str(getattr(self, "passive_llm_recent_events_source", "") or ""),
            "market_mainline_context": market_mainline_context,
            "market_mainline_call_debug": market_mainline_debug,
            "trade_symbol": trade_symbol,
            "trade_symbol_context": dict(selected_context),
            "phase": phase,
            "reasoning_effort": self.engine._resolve_reasoning_effort("passive_event_trigger"),
            "passive_realtime_allowed": passive_realtime_allowed,
            "passive_realtime_debug": passive_realtime_debug,
            "started_at": time.time(),
        }

    def _build_prefetched_passive_chart_context(self, selected_context: Dict[str, Any]) -> Dict[str, Any]:
        started_at = time.time()
        chart_input_context: Dict[str, Any] = {}
        error = ""
        try:
            builder = getattr(self.engine, "chart_context_builder", None)
            if callable(builder):
                chart_candidate = dict(selected_context or {})
                chart_candidate["_chart_mode"] = "passive"
                built_context = builder(chart_candidate)
                if isinstance(built_context, dict):
                    chart_input_context = dict(built_context)
                    if not bool(getattr(self.engine, "include_passive_chart_images", False)):
                        chart_input_context = {
                            **chart_input_context,
                            "input_images": [],
                            "debug_images": [],
                            "image_count": 0,
                        }
        except Exception as exc:
            error = str(exc)
            chart_input_context = {}
        completed_at = time.time()
        return {
            "chart_input_context": chart_input_context,
            "started_at": started_at,
            "completed_at": completed_at,
            "duration_seconds": max(0.0, completed_at - started_at),
            "error": error,
        }

    def _attach_passive_chart_context_prefetch(self, request: Dict[str, Any]) -> None:
        selected_context = request.get("trade_symbol_context")
        if not isinstance(selected_context, dict) or not selected_context:
            return
        builder = getattr(self.engine, "chart_context_builder", None)
        if not callable(builder):
            return
        future = self._passive_chart_context_prefetch_executor().submit(
            self._build_prefetched_passive_chart_context,
            dict(selected_context),
        )
        request["chart_context_future"] = future
        request["chart_context_started_at"] = time.time()

    def _run_passive_event_judge_prefetch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        judge_output, judge_debug = self.engine._call_passive_event_judge(
            phase=str(request.get("phase", "fast") or "fast"),
            trigger_event=request.get("trigger_event"),
            recent_events=list(request.get("recent_events") or []),
            market_mainline_context=request.get("market_mainline_context"),
            reasoning_effort=str(request.get("reasoning_effort", "") or self.engine._resolve_reasoning_effort("passive_event_trigger")),
            trade_symbol=str(request.get("trade_symbol", "") or ""),
        )
        return {
            **dict(request),
            "judge_output": judge_output,
            "judge_debug": judge_debug,
            "completed_at": time.time(),
        }

    @staticmethod
    def _passive_event_judge_result_confidence(result: Dict[str, Any]) -> float:
        judge_output = result.get("judge_output") if isinstance(result, dict) else None
        if not isinstance(judge_output, dict):
            return 0.0
        return min(1.0, max(0.0, float(safe_float(judge_output.get("trigger_confidence"), 0.0) or 0.0)))

    def _run_passive_event_judge_batch_prefetch(self, request: Dict[str, Any]) -> Dict[str, Any]:
        event_requests = [
            dict(item)
            for item in list(request.get("event_requests") or [])
            if isinstance(item, dict)
        ]
        if not event_requests:
            raise RuntimeError("Passive event judge batch has no valid event requests")

        def run_event_request(index_and_request: Tuple[int, Dict[str, Any]]) -> Dict[str, Any]:
            index, event_request = index_and_request
            try:
                result = self._run_passive_event_judge_prefetch(event_request)
                result["batch_index"] = index
                return result
            except Exception as exc:
                return {
                    **dict(event_request),
                    "batch_index": index,
                    "prefetch_failed": True,
                    "error": f"{type(exc).__name__}: {exc}",
                    "completed_at": time.time(),
                }

        max_workers = max(1, len(event_requests))
        with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="passive-step1-event") as executor:
            event_results = list(executor.map(run_event_request, enumerate(event_requests)))

        valid_results = [
            result
            for result in event_results
            if isinstance(result, dict)
            and not result.get("prefetch_failed")
            and isinstance(result.get("judge_output"), dict)
        ]
        if valid_results:
            relevant_results = [
                result
                for result in valid_results
                if str((result.get("judge_output") or {}).get("trigger_event_relevance", "") or "").strip().lower() == "relevant"
            ]
            realtime_relevant_results = [
                result
                for result in relevant_results
                if bool(result.get("passive_realtime_allowed", True))
            ]
            duplicate_results = [
                result
                for result in valid_results
                if str((result.get("judge_output") or {}).get("trigger_event_relevance", "") or "").strip().lower() == "duplicate"
            ]
            unrelated_results = [
                result
                for result in valid_results
                if str((result.get("judge_output") or {}).get("trigger_event_relevance", "") or "").strip().lower() == "unrelated"
            ]
            selection_pool = realtime_relevant_results or relevant_results or unrelated_results or duplicate_results or valid_results
            selected_result = max(
                selection_pool,
                key=lambda result: (
                    self._passive_event_judge_result_confidence(result),
                    int(result.get("batch_index", 0) or 0),
                ),
            )
        else:
            selected_result = event_results[-1]

        batch_completed_at = time.time()
        batch_candidates = []
        combined_usage: Optional[Dict[str, Any]] = None
        combined_usage_cost: Optional[Dict[str, Any]] = None
        combined_web_search_tool_calls = 0
        combined_web_search_calls: List[Dict[str, Any]] = []
        response_ids: List[str] = []
        for result in event_results:
            judge_output = result.get("judge_output") if isinstance(result, dict) else {}
            judge_debug = result.get("judge_debug") if isinstance(result, dict) else {}
            if isinstance(judge_debug, dict) and not result.get("prefetch_failed"):
                combined_usage = merge_usage_dicts(combined_usage, judge_debug.get("usage"))
                combined_usage_cost = merge_usage_costs(combined_usage_cost, judge_debug.get("usage_cost"))
                combined_web_search_tool_calls += int(judge_debug.get("web_search_tool_calls", 0) or 0)
                combined_web_search_calls.extend(list(judge_debug.get("web_search_calls") or []))
                response_id = str(judge_debug.get("response_id", "") or "").strip()
                if response_id:
                    response_ids.append(response_id)
            batch_candidates.append(
                {
                    "batch_index": result.get("batch_index"),
                    "event_key": result.get("event_key", ""),
                    "source": str((result.get("trigger_event") or {}).get("source", "") or ""),
                    "title": str((result.get("trigger_event") or {}).get("title", "") or ""),
                    "status": "error" if result.get("prefetch_failed") else "ok",
                    "error": str(result.get("error", "") or ""),
                    "action": (judge_output or {}).get("action") if isinstance(judge_output, dict) else None,
                    "trigger_event_relevance": (judge_output or {}).get("trigger_event_relevance") if isinstance(judge_output, dict) else None,
                    "trigger_confidence": self._passive_event_judge_result_confidence(result),
                    "passive_realtime_allowed": bool(result.get("passive_realtime_allowed", True)),
                    "passive_realtime_debug": result.get("passive_realtime_debug") or {},
                    "duration_seconds": max(
                        0.0,
                        float(result.get("completed_at", batch_completed_at) or batch_completed_at)
                        - float(result.get("started_at", request.get("started_at", batch_completed_at)) or request.get("started_at", batch_completed_at) or batch_completed_at),
                    ),
                }
            )
        stale_relevant_tape_events = []
        for result in event_results:
            judge_output = result.get("judge_output") if isinstance(result, dict) else {}
            if (
                isinstance(judge_output, dict)
                and str(judge_output.get("trigger_event_relevance", "") or "").strip().lower() == "relevant"
                and not bool(result.get("passive_realtime_allowed", True))
                and not result.get("prefetch_failed")
            ):
                judge_debug = result.get("judge_debug") if isinstance(result.get("judge_debug"), dict) else {}
                tape_event = judge_debug.get("trigger_event_for_llm")
                if not isinstance(tape_event, dict):
                    tape_event = result.get("trigger_event") if isinstance(result.get("trigger_event"), dict) else {}
                stale_relevant_tape_events.append(
                    {
                        "event_key": result.get("event_key", ""),
                        "trigger_event": dict(tape_event),
                        "trade_symbol": str(result.get("trade_symbol", "") or ""),
                        "judge_output": dict(judge_output),
                        "passive_realtime_debug": result.get("passive_realtime_debug") or {},
                    }
                )

        selected = dict(selected_result)
        selected["started_at"] = request.get("started_at", selected.get("started_at", time.time()))
        selected["completed_at"] = batch_completed_at
        selected["batch_event_count"] = len(event_requests)
        selected["batch_candidates"] = batch_candidates
        selected["batch_selection_rule"] = "realtime_relevant_max_then_relevant_tape_only_then_unrelated_then_duplicate"
        selected["batch_selected_index"] = selected.get("batch_index")
        selected["batch_selected_event_key"] = selected.get("event_key", "")
        selected["stale_relevant_tape_events"] = stale_relevant_tape_events
        if isinstance(selected.get("judge_debug"), dict):
            selected_judge_debug = dict(selected.get("judge_debug") or {})
            selected_judge_debug.update(
                {
                    "response_id": "+".join(response_ids) or selected_judge_debug.get("response_id", ""),
                    "usage": combined_usage,
                    "usage_cost": combined_usage_cost,
                    "web_search_tool_calls": combined_web_search_tool_calls,
                    "web_search_calls": combined_web_search_calls,
                    "passive_event_judge_batch_event_count": len(event_requests),
                    "passive_event_judge_batch_candidates": batch_candidates,
                    "passive_event_judge_batch_selection_rule": "realtime_relevant_max_then_relevant_tape_only_then_unrelated_then_duplicate",
                    "passive_event_judge_batch_selected_index": selected.get("batch_index"),
                    "passive_event_judge_batch_selected_event_key": selected.get("event_key", ""),
                }
            )
            selected["judge_debug"] = selected_judge_debug
        for key in ("chart_context_future", "chart_context_started_at"):
            if key in request and key not in selected:
                selected[key] = request.get(key)
        return selected

    def _build_passive_event_judge_batch_prefetch_request(self, events: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        event_requests = [
            request
            for request in (self._build_passive_event_judge_prefetch_request(event) for event in events)
            if isinstance(request, dict)
        ]
        if not event_requests:
            return None
        started_at = time.time()
        latest_event = events[-1] if events else {}
        batch_request: Dict[str, Any] = {
            "event_key": "|".join(str(item.get("event_key", "") or "") for item in event_requests),
            "trigger_event": dict(latest_event),
            "event_requests": event_requests,
            "batch_event_count": len(event_requests),
            "started_at": started_at,
            "trade_symbol": str(event_requests[0].get("trade_symbol", "") or ""),
            "trade_symbol_context": dict(event_requests[0].get("trade_symbol_context") or {}),
            "phase": str(event_requests[0].get("phase", "") or ""),
            "reasoning_effort": str(event_requests[0].get("reasoning_effort", "") or ""),
            "passive_realtime_allowed": any(bool(item.get("passive_realtime_allowed", True)) for item in event_requests),
        }
        if any(bool(item.get("passive_realtime_allowed", True)) for item in event_requests):
            self._attach_passive_chart_context_prefetch(batch_request)
        return batch_request

    def _start_passive_event_judge_prefetch(self, event: Any) -> bool:
        events = self._normalize_passive_event_judge_prefetch_events(event)
        if not events:
            return False
        future = getattr(self, "_passive_event_judge_future", None)
        if isinstance(future, Future) and not future.done():
            self._passive_event_judge_queued_event = [dict(item) for item in events]
            self._audit_event(
                "passive_event_judge_prefetch_queued",
                {
                    "trigger_event": events[-1],
                    "event_count": len(events),
                    "event_keys": [self._passive_event_judge_prefetch_event_key(item) for item in events],
                },
            )
            return True
        request = self._build_passive_event_judge_batch_prefetch_request(events)
        if not request:
            return False
        self._passive_event_judge_request = dict(request)
        self._passive_event_judge_future = self._passive_event_judge_prefetch_executor().submit(
            self._run_passive_event_judge_batch_prefetch,
            dict(request),
        )
        self._passive_event_judge_future.add_done_callback(self._signal_passive_event_judge_ready)
        print_line(f"[passive_event_judge_prefetch_started] count={len(events)} latest={_status_event_brief(events[-1])}")
        self._audit_event(
            "passive_event_judge_prefetch_started",
            {
                "trigger_event": events[-1],
                "events": events,
                "event_count": len(events),
                "event_key": request.get("event_key", ""),
                "trade_symbol": request.get("trade_symbol", ""),
                "phase": request.get("phase", ""),
                "chart_context_prefetch_started": bool(request.get("chart_context_future")),
            },
        )
        return True

    def _consume_ready_passive_event_judge_prefetch(self) -> Optional[Dict[str, Any]]:
        future = getattr(self, "_passive_event_judge_future", None)
        if not isinstance(future, Future) or not future.done():
            return None
        request = dict(getattr(self, "_passive_event_judge_request", {}) or {})
        self._passive_event_judge_future = None
        self._passive_event_judge_request = None
        try:
            result = future.result()
            duration = max(0.0, float(result.get("completed_at", time.time()) or time.time()) - float(result.get("started_at", time.time()) or time.time()))
            batch_count = max(1, int(result.get("batch_event_count", 1) or 1))
            batch_prefix = f"batch_count={batch_count} selected=" if batch_count > 1 else ""
            print_line(
                "[passive_event_judge_prefetch_done] "
                f"{batch_prefix}{_status_event_brief(result.get('trigger_event'))} | "
                f"relevance={result.get('judge_output', {}).get('trigger_event_relevance')} "
                f"action={result.get('judge_output', {}).get('action')} "
                f"confidence={result.get('judge_output', {}).get('trigger_confidence')} "
                f"{duration:.2f}s"
            )
            self._audit_event(
                "passive_event_judge_prefetch_done",
                {
                    "trigger_event": result.get("trigger_event"),
                    "event_key": result.get("event_key", ""),
                    "trade_symbol": result.get("trade_symbol", ""),
                    "duration_seconds": duration,
                    "recent_events_source": result.get("recent_events_source", ""),
                    "judge_output": result.get("judge_output"),
                    "judge_candidates": (result.get("judge_debug") or {}).get("passive_event_judge_candidates") or [],
                    "judge_selection_rule": (result.get("judge_debug") or {}).get("passive_event_judge_selection_rule", ""),
                    "judge_selected_sample_index": (result.get("judge_debug") or {}).get("passive_event_judge_selected_sample_index"),
                    "judge_duplicate_candidate_count": (result.get("judge_debug") or {}).get("passive_event_judge_duplicate_candidate_count", 0),
                    "judge_unrelated_candidate_count": (result.get("judge_debug") or {}).get("passive_event_judge_unrelated_candidate_count", 0),
                    "judge_relevant_candidate_count": (result.get("judge_debug") or {}).get("passive_event_judge_relevant_candidate_count", 0),
                    "judge_duplicate_relevant_conflict": bool((result.get("judge_debug") or {}).get("passive_event_judge_duplicate_relevant_conflict", False)),
                    "judge_unrelated_relevant_conflict": bool((result.get("judge_debug") or {}).get("passive_event_judge_unrelated_relevant_conflict", False)),
                    "judge_confidence_adjustment": (result.get("judge_debug") or {}).get("passive_event_judge_confidence_adjustment") or {},
                    "judge_selected_raw_trigger_confidence": (result.get("judge_debug") or {}).get("passive_event_judge_selected_raw_trigger_confidence"),
                    "judge_selected_effective_trigger_confidence": (result.get("judge_debug") or {}).get("passive_event_judge_selected_effective_trigger_confidence"),
                    "judge_duplicate_relevant_conflict_multiplier": (result.get("judge_debug") or {}).get("passive_event_judge_duplicate_relevant_conflict_multiplier", 1.0),
                    "judge_unrelated_relevant_conflict_multiplier": (result.get("judge_debug") or {}).get("passive_event_judge_unrelated_relevant_conflict_multiplier", 1.0),
                    "batch_event_count": batch_count,
                    "batch_candidates": result.get("batch_candidates") or [],
                    "batch_selection_rule": result.get("batch_selection_rule", ""),
                    "batch_selected_index": result.get("batch_selected_index"),
                    "batch_selected_event_key": result.get("batch_selected_event_key", ""),
                    "passive_realtime_allowed": bool(result.get("passive_realtime_allowed", True)),
                    "passive_realtime_debug": result.get("passive_realtime_debug") or {},
                    "stale_relevant_tape_events": result.get("stale_relevant_tape_events") or [],
                    "chart_context_prefetch_done": bool(
                        isinstance(result.get("chart_context_future"), Future)
                        and result.get("chart_context_future").done()
                    ),
                },
            )
        except Exception as exc:
            result = {
                **request,
                "prefetch_failed": True,
                "error": str(exc),
                "completed_at": time.time(),
            }
            self._audit_event(
                "passive_event_judge_prefetch_failed",
                {
                    "trigger_event": request.get("trigger_event"),
                    "event_key": request.get("event_key", ""),
                    "trade_symbol": request.get("trade_symbol", ""),
                    "message": str(exc),
                },
            )
        queued_event = getattr(self, "_passive_event_judge_queued_event", None)
        self._passive_event_judge_queued_event = None
        queued_events = self._normalize_passive_event_judge_prefetch_events(queued_event)
        if queued_events:
            self._start_passive_event_judge_prefetch(queued_event)
            if result and not result.get("prefetch_failed"):
                self._audit_event(
                    "passive_event_judge_prefetch_discarded",
                    {
                        "discarded_event": result.get("trigger_event"),
                        "replacement_event": queued_events[-1],
                        "replacement_event_count": len(queued_events),
                    },
                )
            return None
        return result

    def _append_stale_relevant_passive_tape_events(self, result: Optional[Dict[str, Any]]) -> int:
        if not isinstance(result, dict):
            return 0
        appended: List[Dict[str, Any]] = []
        seen_keys: set = set()
        for item in list(result.get("stale_relevant_tape_events") or []):
            if not isinstance(item, dict):
                continue
            event = item.get("trigger_event")
            if not isinstance(event, dict):
                continue
            trade_symbol = str(item.get("trade_symbol", "") or result.get("trade_symbol", "") or "").strip()
            if not trade_symbol:
                continue
            event_key = str(item.get("event_key", "") or "").strip()
            dedupe_key = event_key or json.dumps(event, sort_keys=True, ensure_ascii=False)
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            self._append_passive_llm_recent_event(event, trade_symbol)
            appended.append(
                {
                    "event_key": event_key,
                    "trigger_event": event,
                    "trade_symbol": trade_symbol,
                    "judge_output": item.get("judge_output") if isinstance(item.get("judge_output"), dict) else {},
                    "passive_realtime_debug": item.get("passive_realtime_debug") if isinstance(item.get("passive_realtime_debug"), dict) else {},
                }
            )
        if appended:
            print_line(
                "[passive_stale_relevant_tape_update] "
                f"count={len(appended)} latest={_status_event_brief(appended[-1].get('trigger_event'))}"
            )
            self._audit_event(
                "passive_stale_relevant_tape_update",
                {"count": len(appended), "events": appended},
            )
        return len(appended)

    def _passive_prefetched_trade_gate_block(self, result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(result, dict) or result.get("prefetch_failed"):
            return None
        if not bool(result.get("passive_realtime_allowed", True)):
            return None
        judge_output = result.get("judge_output")
        if not isinstance(judge_output, dict):
            return None
        relevance = str(judge_output.get("trigger_event_relevance", "") or "").strip().lower()
        if relevance != "relevant":
            return None
        action = str(judge_output.get("action", "no_trade") or "no_trade").strip().lower()
        confidence = min(1.0, max(0.0, float(safe_float(judge_output.get("trigger_confidence"), 0.0) or 0.0)))
        trade_symbol = str(result.get("trade_symbol", "") or "").strip()
        threshold_fn = getattr(self.engine, "_passive_trade_confidence_threshold", None)
        threshold = (
            min(1.0, max(0.0, float(safe_float(threshold_fn(trade_symbol), 0.0) or 0.0)))
            if callable(threshold_fn)
            else 0.0
        )
        if action == "no_trade":
            reason = "step1_no_trade"
        elif confidence < threshold:
            reason = "below_trade_confidence_threshold"
        else:
            return None
        return {
            "reason": reason,
            "threshold": threshold,
            "trigger_confidence": confidence,
            "trigger_event_relevance": relevance,
            "action": action,
            "trade_symbol": trade_symbol,
        }

    def _append_realtime_relevant_passive_tape_event(self, result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if not isinstance(result, dict):
            return None
        judge_output = result.get("judge_output")
        if not isinstance(judge_output, dict):
            return None
        if str(judge_output.get("trigger_event_relevance", "") or "").strip().lower() != "relevant":
            return None
        trade_symbol = str(result.get("trade_symbol", "") or "").strip()
        if not trade_symbol:
            return None
        judge_debug = result.get("judge_debug") if isinstance(result.get("judge_debug"), dict) else {}
        tape_event = judge_debug.get("trigger_event_for_llm")
        if not isinstance(tape_event, dict):
            tape_event = result.get("trigger_event") if isinstance(result.get("trigger_event"), dict) else {}
        if not tape_event:
            return None
        self._append_passive_llm_recent_event(tape_event, trade_symbol)
        trigger_event = result.get("trigger_event") if isinstance(result.get("trigger_event"), dict) else tape_event
        self._remember_llm_relevant_passive_event(trigger_event, trade_symbol)
        return {
            "event_key": str(result.get("event_key", "") or ""),
            "trigger_event": dict(tape_event),
            "trade_symbol": trade_symbol,
            "judge_output": dict(judge_output),
        }

    def _resolve_immediate_playbook_action(
        self,
        playbook: GenericPlaybook,
        symbol_position: dict,
    ) -> Optional[Dict[str, Any]]:
        has_open_position = snapshot_has_open_position(symbol_position)
        position_state = "open" if has_open_position else "flat"
        if playbook.entry_plan.execute_now:
            self._audit_event(
                "playbook_immediate_action_ignored",
                {
                    "position_state": position_state,
                    "selected_symbol": playbook.selected_symbol,
                    "ignored_source": "entry_plan",
                    "reason": "runtime materializes entry_plan into position_management before execution.",
                    "ignored_decision_view": build_decision_execution_view(playbook.entry_plan.action_decision, trigger_confidence_raw=playbook.trigger_confidence_raw, symbol=playbook.selected_symbol),
                },
            )
        if playbook.position_management.execute_now:
            return {
                "kind": "management",
                "plan_name": "position_management_now",
                "decision": playbook.position_management.action_decision,
            }
        return None
    def _arm_follow_up_plan_for_current_state(
        self,
        playbook: GenericPlaybook,
        symbol_position: dict,
    ) -> None:
        if not snapshot_has_open_position(symbol_position):
            self._replace_risk_session(None)
        self._set_position_management_session_from_plan(playbook.position_management, symbol_position, "position_management")
        if getattr(self, "position_management_session", None) is not None:
            self._audit_event(
                "position_management_session_created",
                {
                    "plan_name": "position_management",
                    "position_management": playbook.position_management.to_dict(),
                    "position_snapshot": symbol_position,
                },
            )
    def _execute_immediate_playbook_action(
        self,
        playbook: GenericPlaybook,
        symbol_position: dict,
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        immediate_action = self._resolve_immediate_playbook_action(playbook, symbol_position)
        if immediate_action is None:
            return None
        management_exec = self.execute_management_decision(
            playbook.position_management.action_decision,
            str(immediate_action.get("plan_name") or "position_management_now"),
            playbook.position_management,
            trigger_confidence_raw=playbook.trigger_confidence_raw,
            execution_context=execution_context,
        )
        position_after = management_exec["position_after"]
        position_before = management_exec.get("position_before") if isinstance(management_exec, dict) else None
        if not isinstance(position_before, dict):
            position_before = symbol_position
        effective_decision_payload = dict(management_exec.get("decision") or {})
        effective_action = str(effective_decision_payload.get("action") or playbook.position_management.action_decision.action or "").strip()
        should_apply_rebuilt_management = self._management_decision_reached_effective_state(
            playbook.position_management.action_decision,
            position_before,
            position_after,
        )
        action_decision = playbook.position_management.action_decision
        add_fill_entry_price = 0.0
        if str(getattr(action_decision, "action", "") or "") in {"add_to_long", "add_to_short"}:
            add_fill_entry_price = self._extract_filled_avg_price_from_execution_result(management_exec)
        if effective_action == "no_change":
            playbook.position_management.execute_now = False
            playbook.position_management.action_decision = ManagementDecision(
                action="no_change",
                close_fraction=min(max(float(effective_decision_payload.get("close_fraction", 0.0) or 0.0), 0.0), 1.0),
                new_notional_usd=max(0.0, float(effective_decision_payload.get("new_notional_usd", 0.0) or 0.0)),
                entry_price=max(0.0, float(effective_decision_payload.get("entry_price", 0.0) or 0.0)),
                stop_loss_price=max(0.0, float(effective_decision_payload.get("stop_loss_price", 0.0) or 0.0)),
                planned_max_loss_usd=max(0.0, float(effective_decision_payload.get("planned_max_loss_usd", 0.0) or 0.0)),
                leverage=max(0, int(effective_decision_payload.get("leverage", 0) or 0)),
                margin_basis_usd=max(0.0, float(effective_decision_payload.get("margin_basis_usd", 0.0) or 0.0)),
                continue_entry_plan_after_close=bool(effective_decision_payload.get("continue_entry_plan_after_close", False)),
            )
            playbook.post_fill_risk_template = build_empty_position_management_plan()
            self._audit_event(
                "position_management_immediate_no_change",
                {
                    "selected_symbol": playbook.selected_symbol,
                    "requested_decision": management_exec.get("requested_decision"),
                    "decision": effective_decision_payload,
                    "result": management_exec,
                    "tp_sl_refresh_applied": should_apply_rebuilt_management,
                },
            )
            if should_apply_rebuilt_management:
                self._set_risk_session_after_management_decision(
                    playbook.position_management.action_decision,
                    playbook.position_management,
                    playbook.post_fill_risk_template,
                    position_after,
                    "position_management",
                    add_fill_entry_price=add_fill_entry_price,
                )
        elif should_apply_rebuilt_management:
            if bool(management_exec.get("entry_order_pending", False)) and not snapshot_has_open_position(position_after):
                self._set_pending_entry_order_session(
                    plan_name="position_management",
                    management_decision=playbook.position_management.action_decision,
                    position_management=playbook.position_management,
                    post_fill_risk_template=playbook.post_fill_risk_template,
                    execution_result=management_exec,
                )
                self._replace_risk_session(None)
                self.position_management_session = None
            else:
                continue_after_close = bool(playbook.position_management.action_decision.continue_entry_plan_after_close) and not snapshot_has_open_position(position_after)
                staged_trim_reused = self._reuse_staged_risk_session_after_trim(
                    decision=playbook.position_management.action_decision,
                    execution_result=management_exec,
                    position_before=position_before,
                    position_after=position_after,
                )
                if not staged_trim_reused:
                    self._set_risk_session_after_management_decision(
                        playbook.position_management.action_decision,
                        playbook.position_management,
                        playbook.post_fill_risk_template,
                        position_after,
                        "position_management",
                        add_fill_entry_price=add_fill_entry_price,
                    )
                if continue_after_close:
                    self._refresh_position_management_session_from_current_playbook(position_after)
                else:
                    self.position_management_session = None
        else:
            self._audit_event(
                "position_management_immediate_rejected",
                {
                    "selected_symbol": playbook.selected_symbol,
                    "decision_view": build_decision_execution_view(playbook.position_management.action_decision),
                    "result": management_exec,
                    "tp_sl_refresh_applied": False,
                },
            )
            self._arm_follow_up_plan_for_current_state(playbook, position_after)
        self._schedule_next_active_query(position_after)
        return {"kind": "management", "result": management_exec, "position_after": position_after}
    def query_new_playbook(
        self,
        reason: str,
        trigger_event: Optional[Dict[str, Any]],
        recent_events: Optional[List[Dict[str, Any]]] = None,
        prefetched_passive_event_judge: Optional[Dict[str, Any]] = None,
    ) -> None:
        now = time.time()
        if reason != "passive_event_trigger" or getattr(self, "pending_entry_order_session", None) is not None:
            self.step_pending_entry_order_session(now)
        if getattr(self, "pending_entry_order_session", None) is not None:
            self._cancel_pending_entry_order(f"new_playbook_query:{reason}")
        prefetched_context = (
            dict((prefetched_passive_event_judge or {}).get("trade_symbol_context") or {})
            if reason == "passive_event_trigger" and isinstance(prefetched_passive_event_judge, dict)
            else {}
        )
        passive_prefetch_available = bool(
            reason == "passive_event_trigger"
            and isinstance(prefetched_passive_event_judge, dict)
            and not prefetched_passive_event_judge.get("prefetch_failed")
        )
        defer_initial_positions = bool(
            passive_prefetch_available
            and prefetched_context
            and canonicalize_execution_symbol(prefetched_context.get("execution_symbol", ""))
        )
        if defer_initial_positions:
            all_positions: Dict[str, Any] = {}
            query_trade_symbol_context = self._build_prefetched_passive_query_trade_symbol_context(prefetched_context)
            open_positions: List[Dict[str, Any]] = []
            query_symbol = canonicalize_execution_symbol(prefetched_context.get("execution_symbol", "")) or canonicalize_execution_symbol(self.symbol or "")
            trade_symbol_context = dict(prefetched_context) if prefetched_context else self._build_lightweight_passive_trade_symbol_context()
        else:
            all_positions = self.reader.get_all_positions()
            query_trade_symbol_context, _ = self._resolve_query_trade_symbol_context(all_positions, reason=reason)
            open_positions = [pos for pos in (all_positions.get("positions", []) or []) if snapshot_has_open_position(pos)]
            query_symbol = self._find_management_symbol(all_positions) or ""
            trade_symbol_context = self._build_trade_symbol_context(all_positions, query_trade_symbol_context)
        event_tape = self.events.recent()
        buffer_symbol = self._resolve_recent_passive_event_symbol(
            query_symbol=query_symbol,
            query_trade_symbol_context=query_trade_symbol_context,
            trade_symbol_context=trade_symbol_context,
        )
        if reason == "passive_event_trigger":
            if recent_events is not None:
                passive_recent_events = [
                    dict(item)
                    for item in list(recent_events or [])
                    if isinstance(item, dict)
                ]
                passive_recent_events_source = str(
                    (prefetched_passive_event_judge or {}).get("recent_events_source", "")
                    or getattr(self, "passive_llm_recent_events_source", "")
                    or ""
                )
            else:
                passive_recent_symbol = self._normalize_passive_event_symbol(getattr(self, "passive_llm_recent_events_symbol", ""))
                buffer_symbol_normalized = self._normalize_passive_event_symbol(buffer_symbol)
                if passive_recent_symbol and passive_recent_symbol == buffer_symbol_normalized:
                    passive_recent_events = [
                        dict(item)
                        for item in list(getattr(self, "passive_llm_recent_events", []) or [])
                        if isinstance(item, dict)
                    ]
                    passive_recent_events_source = str(getattr(self, "passive_llm_recent_events_source", "") or "")
                else:
                    passive_recent_events = []
                    passive_recent_events_source = ""
        else:
            passive_recent_events = [
                dict(item)
                for item in list(
                    recent_events
                    or self._recent_relevant_passive_events(
                        symbol=buffer_symbol,
                        max_items=self.event_context_max_items,
                    )
                )
                if isinstance(item, dict)
            ]
            passive_recent_events_source = "passive_relevant_events"
        effective_query = self.render_user_query(all_positions, query_trade_symbol_context)
        had_open_position = bool(open_positions)
        self.current_playbook_reason = reason
        if not passive_prefetch_available:
            print(f"[query_reason] {reason}")
        default_search_mode = str(getattr(self.engine, "default_search_mode", "off") or "off")
        helper_search_mode = str(
            getattr(
                self.engine,
                "passive_search_mode" if reason == "passive_event_trigger" else "active_search_mode",
                default_search_mode,
            )
            or default_search_mode
        ).lower()
        if helper_search_mode == "off":
            helper_force_news_context = False
        elif reason == "passive_event_trigger":
            helper_force_news_context = bool(getattr(self.engine, "force_passive_news_context", False))
        else:
            helper_force_news_context = bool(getattr(self.engine, "force_active_news_context", True))
        helper_query_enabled = helper_force_news_context and not had_open_position
        helper_only_refresh = reason == "helper_reset_refresh"
        helper_query_invoked = helper_only_refresh and helper_query_enabled
        if helper_query_enabled and reason != "passive_event_trigger":
            event_tape = self._filter_recent_events_for_active_helper(event_tape)
        trade_symbol_price = _status_trade_symbol_price_brief(trade_symbol_context)
        if trade_symbol_price:
            if reason != "passive_event_trigger":
                if helper_query_invoked:
                    print(f"[helper_trade_symbol] {trade_symbol_price}")
                elif not helper_query_enabled:
                    print(f"[query_trade_symbol] {trade_symbol_price}")
        if trigger_event is not None and not passive_prefetch_available:
            print(f"[trigger_event] {_status_event_brief(trigger_event)}")
        if helper_query_invoked and len(event_tape) > 1:
            print(f"[recent_events] count={len(event_tape)} latest={_status_event_brief(event_tape[-1])}")
        self._print_json_block("effective_query", effective_query)
        self._print_json_block("trade_symbol_context", trade_symbol_context)
        if trigger_event is not None:
            self._print_json_block("trigger_event_payload", trigger_event)
        if passive_recent_events:
            self._print_json_block("recent_events_payload", passive_recent_events)
        display_symbol = query_symbol
        if defer_initial_positions:
            current_price = safe_float(prefetched_context.get("current_price"), None)
            symbol_position = self._empty_runtime_snapshot(all_positions, symbol=display_symbol)
            if current_price is not None:
                symbol_position["mid_price"] = current_price
        else:
            current_price = self.reader.get_mid_price(display_symbol) if display_symbol else None
            symbol_position = self.reader.get_position_snapshot(
                display_symbol,
                all_positions=all_positions,
                current_price=current_price,
            ) if display_symbol else self._empty_runtime_snapshot(all_positions)
        active_playbook_disabled = (
            reason != "passive_event_trigger"
            and not helper_only_refresh
            and not bool(getattr(self, "enable_active_playbook", True))
        )
        if reason != "passive_event_trigger" and (helper_only_refresh or active_playbook_disabled):
            helper_context: Dict[str, Any] = {}
            helper_debug: Dict[str, Any] = {}
            if helper_query_invoked:
                helper_context, helper_debug = self.engine._build_market_news_context(
                    user_query=effective_query,
                    recent_events=event_tape,
                    trade_symbol_context=trade_symbol_context,
                    active_symbol=(query_symbol or self.symbol or ""),
                    trigger_reason=reason,
                    trigger_event=trigger_event,
                )
                helper_selected = str((helper_debug or {}).get("trade_symbol", "") or (helper_debug or {}).get("winner_display_name", "") or "").strip()
                if helper_selected:
                    print(f"[helper_selected] {helper_selected}")
                if helper_context:
                    self._print_json_block("market_mainline_context", helper_context)
                if helper_debug:
                    self._print_json_block("market_mainline_call_debug", helper_debug)
                self._refresh_passive_llm_recent_events_from_helper()
            elif active_playbook_disabled:
                print(f"[active_playbook_disabled] reason={reason} | helper_not_allowed")
            else:
                print(f"[helper_reset] reason={reason} | helper_skipped")
            self._audit_event(
                "active_playbook_skipped",
                {
                    "reason": reason,
                    "helper_query_enabled": helper_query_enabled,
                    "helper_query_invoked": helper_query_invoked,
                    "had_open_position": had_open_position,
                    "trigger_event": trigger_event,
                    "recent_events": passive_recent_events,
                    "recent_events_source": passive_recent_events_source,
                    "trade_symbol_context": trade_symbol_context,
                    "market_mainline_context": helper_context,
                    "market_mainline_call_debug": helper_debug,
                },
            )
            self.current_playbook = None
            self.current_mode = None
            self.current_playbook_reason = reason
            self.last_playbook_query_at = time.time()
            self._schedule_next_active_query(symbol_position)
            return
        playbook, mode = self.engine.get_playbook(
            user_query=effective_query,
            event_tape=event_tape,
            trigger_reason=reason,
            trigger_event=trigger_event,
            recent_events=passive_recent_events,
            trade_symbol_context=trade_symbol_context,
            active_symbol=(query_symbol or self.symbol or ""),
            has_live_position=had_open_position,
            prefetched_passive_event_judge=prefetched_passive_event_judge if reason == "passive_event_trigger" else None,
        )
        playbook_trade_context = None
        if isinstance(self.engine.last_call_debug, dict):
            playbook_trade_context = self.engine.last_call_debug.get("trade_symbol_context")
        if isinstance(playbook_trade_context, dict) and playbook_trade_context:
            playbook_query_context = dict(playbook_trade_context)
        else:
            playbook_query_context = dict(trade_symbol_context or {})
        if reason == "passive_event_trigger" or helper_query_enabled:
            narrowed_price = _status_trade_symbol_price_brief(playbook_query_context)
            if narrowed_price:
                print(f"[playbook_trade_symbol] {narrowed_price}")
        self._audit_event(
            "playbook_query_requested",
            {
                "reason": reason,
                "effective_query": effective_query,
                "trigger_event": trigger_event,
                "recent_events": passive_recent_events,
                "recent_events_source": passive_recent_events_source,
                "trade_symbol_context": playbook_query_context,
                "query_symbol": query_symbol,
            },
        )
        if reason == "passive_event_trigger" and str(getattr(playbook, "trigger_event_relevance", "") or "").strip().lower() in {"unrelated", "duplicate"}:
            passive_relevance = str(getattr(playbook, "trigger_event_relevance", "") or "").strip().lower()
            if not passive_prefetch_available:
                print(
                    f"[passive_query_irrelevant] {_status_event_brief(trigger_event)} "
                    f"| relevance={passive_relevance or '<empty>'} | selection={playbook.selected_symbol or '<empty>'}"
                )
            self._audit_event(
                "passive_query_irrelevant",
                {
                    "reason": reason,
                    "trigger_event_relevance": passive_relevance,
                    "trigger_event": trigger_event,
                    "recent_events": passive_recent_events,
                    "trade_symbol_context": playbook_query_context,
                    "playbook": playbook.to_dict(),
                },
            )
            if getattr(self.engine, "last_call_debug", None):
                self.engine.last_call_debug["validated_playbook"] = playbook.to_dict()
                self.engine.last_call_debug["capped_playbook"] = playbook.to_dict()
                self.engine.last_call_debug["execution_view"] = build_playbook_execution_view(playbook, symbol_position)
                self.engine.last_call_debug["flattened_positions"] = []
            engine_debug = self._augment_engine_debug_with_cost_metrics(dict(getattr(self.engine, "last_call_debug", {}) or {}))
            if engine_debug:
                self._audit_event("llm_call_debug", engine_debug)
            self.last_playbook_query_at = time.time()
            return
        selected_symbol = self._normalize_selected_symbol(playbook, query_trade_symbol_context)
        if reason == "passive_event_trigger" and str(getattr(playbook, "trigger_event_relevance", "") or "").strip().lower() == "relevant":
            recent_tape_event = None
            if isinstance(getattr(self.engine, "last_call_debug", None), dict):
                candidate_event = self.engine.last_call_debug.get("passive_event_judge_trigger_event_for_llm")
                if isinstance(candidate_event, dict):
                    recent_tape_event = candidate_event
            self._append_passive_llm_recent_event(recent_tape_event or trigger_event, selected_symbol or buffer_symbol)
            self._remember_llm_relevant_passive_event(trigger_event, selected_symbol or buffer_symbol)
        selected_context = self._find_trade_symbol_by_selected_symbol(selected_symbol, query_trade_symbol_context)
        selected_execution_symbol = canonicalize_execution_symbol((selected_context or {}).get("execution_symbol", ""))
        selected_is_tradable = bool(selected_execution_symbol)
        if selected_context is not None and not selected_is_tradable:
            print(
                f"[nontradable_selected_symbol] selected_symbol={selected_symbol} "
                "cannot be executed on Hyperliquid right now; local execution disabled."
            )
            playbook = self._disable_nontradable_entry(playbook, selected_context)
            self._audit_event(
                "playbook_nontradable_selection",
                {
                    "selected_symbol": selected_symbol,
                    "selection_reason": playbook.selection_reason,
                    "trade_symbol_context": selected_context,
                },
            )
        flattened_positions: List[Dict[str, Any]] = []
        immediate_execution_context: Optional[Dict[str, Any]] = None
        if selected_is_tradable:
            if selected_execution_symbol != self.symbol:
                self._set_active_symbol(selected_execution_symbol, reason="llm_selected_symbol")
            all_positions, symbol_position, current_price = self._fetch_pre_execution_position_context(
                self.symbol,
                reason=reason,
            )
            had_open_position = snapshot_has_open_position(symbol_position)
            playbook = self._materialize_live_position_management_from_entry_plan(playbook, symbol_position, all_positions)
            trade_symbol_context = self._update_trade_symbol_context_price(trade_symbol_context, self.symbol, current_price)
            immediate_execution_context = {
                "all_positions": all_positions,
                "position_before": symbol_position,
                "execution_mid_price": current_price,
            }
        if getattr(self.engine, "last_call_debug", None):
            self.engine.last_call_debug["validated_playbook"] = playbook.to_dict()
            self.engine.last_call_debug["capped_playbook"] = playbook.to_dict()
            self.engine.last_call_debug["execution_view"] = build_playbook_execution_view(playbook, symbol_position)
            self.engine.last_call_debug["flattened_positions"] = flattened_positions
        engine_debug = self._augment_engine_debug_with_cost_metrics(dict(getattr(self.engine, "last_call_debug", {}) or {}))
        if engine_debug.get("raw_output_text"):
            self._print_json_block("llm_raw_output_text", engine_debug["raw_output_text"])
        if engine_debug.get("market_mainline_context") is not None:
            self._print_json_block("market_mainline_context", engine_debug["market_mainline_context"])
        if engine_debug.get("market_mainline_web_search_budget") is not None:
            self._print_json_block("market_mainline_web_search_budget", engine_debug["market_mainline_web_search_budget"])
        if engine_debug.get("market_mainline_web_search_analysis") is not None:
            self._print_json_block("market_mainline_web_search_analysis", engine_debug["market_mainline_web_search_analysis"])
        if engine_debug.get("chart_screenshot_debug"):
            self._print_json_block("llm_chart_screenshot_debug", engine_debug["chart_screenshot_debug"])
        if engine_debug.get("parsed_output") is not None:
            self._print_json_block("llm_parsed_output", engine_debug["parsed_output"])
        if engine_debug.get("validated_playbook") is not None:
            self._print_json_block("llm_validated_playbook", engine_debug["validated_playbook"])
        if engine_debug.get("capped_playbook") is not None:
            self._print_json_block("llm_capped_playbook", engine_debug["capped_playbook"])
        if engine_debug.get("execution_view") is not None:
            self._print_json_block("llm_execution_view", engine_debug["execution_view"])
        if engine_debug.get("usage") is not None:
            self._print_json_block("llm_usage", engine_debug["usage"])
        if engine_debug.get("web_search_calls") is not None:
            self._print_json_block("llm_web_search_calls", engine_debug["web_search_calls"])
        if engine_debug.get("usage_cost") is not None:
            self._print_json_block("llm_usage_cost", engine_debug["usage_cost"])
        if engine_debug.get("cost_rollup") is not None:
            self._print_json_block("llm_cost_rollup", engine_debug["cost_rollup"])
        if engine_debug.get("cost_projection") is not None:
            self._print_json_block("llm_cost_projection", engine_debug["cost_projection"])
        self._audit_event("llm_call_debug", engine_debug)
        self.current_playbook = playbook
        self.current_mode = mode
        self.current_playbook_reason = reason
        self.last_playbook_query_at = time.time()
        self.print_playbook(playbook, mode, all_positions, symbol_position)
        self._audit_event(
            "playbook_selected",
            {
                "mode": mode,
                "playbook": playbook.to_dict(),
                "execution_view": build_playbook_execution_view(playbook, symbol_position),
                "had_open_position": had_open_position,
                "trade_symbol_context": trade_symbol_context,
            },
        )

        immediate_execution = self._execute_immediate_playbook_action(
            playbook,
            symbol_position,
            execution_context=immediate_execution_context,
        )
        if immediate_execution is not None:
            return
        self._arm_follow_up_plan_for_current_state(playbook, symbol_position)
        self._schedule_next_active_query(symbol_position)
    def execute_decision(
        self,
        decision: StrategyDecision,
        plan_name: str,
        trigger_confidence_raw: Optional[float] = None,
    ) -> dict:
        result = self.executor.execute(decision, plan_name=plan_name, trigger_confidence_raw=trigger_confidence_raw)
        _, position_after, _ = self._fetch_selected_symbol_position_context(self.symbol)
        result["position_after"] = position_after
        if int(decision.requested_leverage or 0) > 0 and snapshot_has_open_position(position_after) and hasattr(self.executor, "reconcile_requested_leverage_after_execution"):
            leverage_reconcile = self.executor.reconcile_requested_leverage_after_execution(position_after, int(decision.requested_leverage or 0))
            if leverage_reconcile:
                result.update({k: v for k, v in leverage_reconcile.items() if k != "position_after"})
                position_after = leverage_reconcile.get("position_after", position_after)
                result["position_after"] = position_after
        result["accepted"] = not self.executor._result_has_exchange_error(result)
        if not result["accepted"] and not str(result.get("message", "") or "").strip():
            result["message"] = "Exchange rejected at least one order action."
        basis_update = None
        if result.get("accepted", False):
            if snapshot_has_open_position(position_after) and decision.action in {"long", "short"} and extract_raw_confidence_value(trigger_confidence_raw) is not None:
                basis_update = self._set_position_basis_state(
                    side=decision.action,
                    confidence_raw=trigger_confidence_raw,
                    validity=1.0,
                    reason=f"entry_{decision.action}_basis_reset",
                    position_snapshot=position_after,
                )
            elif not snapshot_has_open_position(position_after):
                basis_update = self._clear_position_basis_state(
                    reason=f"entry_{decision.action}_flat",
                    position_snapshot=position_after,
                )
            if basis_update is not None:
                result["position_basis_update"] = basis_update
        print(f"[execution] {_status_decision_brief(build_decision_execution_view(decision, trigger_confidence_raw=trigger_confidence_raw, symbol=self.symbol))} | {_status_execution_result_brief(result)}")
        self._print_json_block("execution", result)
        if decision.action in {"long", "short"} and snapshot_has_open_position(position_after):
            self._replace_risk_session(None)
            print(f"[risk_monitor_plan_missing] {plan_name} | side={decision.action} | no post-fill risk template")
            self._print_json_block("risk_monitor_plan_missing", {
                "plan_name": plan_name,
                "side": decision.action,
                "initial_entry_price": float(position_after.get("entry_price") or position_after.get("current_price") or 0.0),
                "message": "No post-fill risk template was available after entry execution; legacy risk session fallback disabled.",
            })
            self._audit_event(
                "risk_monitor_plan_missing",
                {
                    "plan_name": plan_name,
                    "side": decision.action,
                    "position_after": position_after,
                    "message": "No post-fill risk template was available after entry execution; legacy risk session fallback disabled.",
                },
            )
        else:
            pass
        self._audit_event(
            "entry_execution_result",
            {
                "plan_name": plan_name,
                "decision": decision.to_dict(),
                "decision_view": build_decision_execution_view(decision, trigger_confidence_raw=trigger_confidence_raw, symbol=self.symbol),
                "result": result,
            },
        )
        return result
    def execute_management_decision(
        self,
        decision: ManagementDecision,
        plan_name: str,
        management_plan: Optional[PositionManagementPlan] = None,
        trigger_confidence_raw: Optional[float] = None,
        execution_context: Optional[Dict[str, Any]] = None,
    ) -> dict:
        context = dict(execution_context or {}) if isinstance(execution_context, dict) else {}
        position_before = context.get("position_before") if isinstance(context.get("position_before"), dict) else None
        execution_mid_price = safe_float(context.get("execution_mid_price"), None)
        if not isinstance(position_before, dict):
            position_before, execution_mid_price = self._fetch_execution_position_context()
        original_decision = decision
        management_playbook_reason = str(getattr(self, "current_playbook_reason", "") or "")
        current_pm_session = getattr(self, "position_management_session", None)
        if (
            management_plan is not None
            and current_pm_session is not None
            and getattr(current_pm_session, "position_management", None) is management_plan
        ):
            management_playbook_reason = str(
                getattr(current_pm_session, "playbook_reason", "") or management_playbook_reason
            )
        guard_result = None
        if management_playbook_reason == "passive_event_trigger":
            guard_result = self._passive_entry_deviation_guard(
                decision,
                plan_name=plan_name,
                position_before=position_before,
                execution_mid_price=execution_mid_price,
            )
        if guard_result is not None:
            result = guard_result
        elif management_playbook_reason == "passive_event_trigger":
            guard_result = self._passive_basis_chase_guard(
                decision,
                plan_name=plan_name,
                position_before=position_before,
                execution_mid_price=execution_mid_price,
            )
            if guard_result is not None:
                result = guard_result
            else:
                result = None
        else:
            result = None
        if result is None:
            if (
                management_playbook_reason == "passive_event_trigger"
                and str(getattr(decision, "action", "") or "") in {"long", "short"}
                and not snapshot_has_open_position(position_before)
                and hasattr(self.executor, "execute_position_target")
            ):
                result = self._call_with_optional_position_before(
                    self.executor.execute_position_target,
                    target_side=str(decision.action or ""),
                    target_notional_usd=float(decision.new_notional_usd or 0.0),
                    requested_leverage=int(decision.leverage or 0),
                    reason="management_market_open_from_flat_passive",
                    plan_name=plan_name,
                    position_before=position_before,
                    execution_mid_price=execution_mid_price,
                )
                result["decision"] = decision.to_dict()
            elif hasattr(self.executor, "execute_management"):
                result = self._call_with_optional_position_before(
                    self.executor.execute_management,
                    decision,
                    plan_name=plan_name,
                    trigger_confidence_raw=trigger_confidence_raw,
                    position_before=position_before,
                    execution_mid_price=execution_mid_price,
                )
            else:
                result = {
                    "mode": "dry_run",
                    "symbol": self.symbol,
                    "plan_name": plan_name,
                    "decision": decision.to_dict(),
                    "actions": [],
                    "message": "Executor does not implement execute_management.",
                }
        requested_decision_dict = original_decision.to_dict()
        decision_dict = decision.to_dict()
        if decision_dict != requested_decision_dict:
            result["requested_decision"] = requested_decision_dict
        result["decision"] = decision_dict
        _, position_after, _ = self._fetch_selected_symbol_position_context(self.symbol)
        result["position_after"] = position_after
        if int(decision.leverage or 0) > 0 and snapshot_has_open_position(result["position_after"]) and hasattr(self.executor, "reconcile_requested_leverage_after_execution"):
            leverage_reconcile = self.executor.reconcile_requested_leverage_after_execution(result["position_after"], int(decision.leverage or 0))
            if leverage_reconcile:
                result.update({k: v for k, v in leverage_reconcile.items() if k != "position_after"})
                result["position_after"] = leverage_reconcile.get("position_after", result["position_after"])
        result["accepted"] = False if result.get("local_blocked") else not self.executor._result_has_exchange_error(result)
        if not result["accepted"] and not str(result.get("message", "") or "").strip():
            result["message"] = "Exchange rejected at least one order action."
        basis_update = self._update_position_basis_after_management_execution(
            decision=decision,
            trigger_confidence_raw=trigger_confidence_raw,
            position_before=position_before,
            position_after=result["position_after"],
            accepted=bool(result.get("accepted", False)),
        )
        if basis_update is not None:
            result["position_basis_update"] = basis_update
        print(f"[management_execution] {_status_decision_brief(build_decision_execution_view(decision, trigger_confidence_raw=trigger_confidence_raw, symbol=self.symbol))} | {_status_execution_result_brief(result)}")
        self._print_json_block("management_execution", result)
        self._audit_event(
            "management_execution_result",
            {
                "plan_name": plan_name,
                "requested_decision": requested_decision_dict if decision_dict != requested_decision_dict else None,
                "decision": decision_dict,
                "decision_view": build_decision_execution_view(decision, trigger_confidence_raw=trigger_confidence_raw, symbol=self.symbol),
                "result": result,
            },
        )
        return result
    def active_query_allowed_now(self) -> bool:
        if not self.enable_active_query or not self.enable_active_auto_requery:
            return False
        if not bool(getattr(self, "enable_active_playbook", True)):
            return False
        if self.risk_session is not None and self.risk_session.is_observing():
            return False
        if getattr(self, "position_management_session", None) is not None and self.position_management_session.is_observing():
            return False
        return True
    def step_risk_session(self, snapshot: dict, now: float, fill_events: Optional[List[Dict[str, Any]]] = None) -> Optional[str]:
        if self.risk_session is None:
            return None
        if now - self.last_risk_tick_at < self.risk_poll_seconds:
            return None
        self.last_risk_tick_at = now
        self._maintain_user_fills_subscription(now)
        rs = self.risk_session
        size = float(snapshot.get("size", 0.0) or 0.0)
        price = safe_float(snapshot.get("mid_price"), None)
        expected_sign = 1 if rs.side == "long" else -1
        size_changed = abs(size - rs.expected_size) > self.position_size_change_tol
        ws_active = bool(rs.use_resting_exit_orders and self._user_fills_ws_is_active())
        self._maybe_audit_risk_session_market_context(snapshot, now)

        if rs.use_resting_exit_orders:
            pending_fill_events = list(fill_events or [])
            pending_fill_events.extend(
                self._backfill_recent_user_fills(
                    now,
                    force=(size_changed or not ws_active),
                )
            )
            fill_status = self._process_risk_session_user_fill_events(snapshot, pending_fill_events, now)
            if fill_status is not None:
                return fill_status
            if size_changed:
                reconcile_status = self._reconcile_risk_session_exchange_fill(snapshot, now)
                if reconcile_status is not None:
                    return reconcile_status
            if ws_active:
                if not size_changed:
                    soft_stop_status = self._maybe_execute_risk_session_soft_stop(
                        rs,
                        snapshot,
                        price=price,
                        now=now,
                    )
                    if soft_stop_status is not None:
                        return soft_stop_status
                    basis_profit_lock_status = self._maybe_execute_risk_session_basis_profit_lock(
                        rs,
                        snapshot,
                        price=price,
                        now=now,
                    )
                    if basis_profit_lock_status is not None:
                        return basis_profit_lock_status
                    time_decay_status = self._maybe_execute_risk_session_time_decay_take_profit(
                        rs,
                        snapshot,
                        price=price,
                        now=now,
                    )
                    if time_decay_status is not None:
                        return time_decay_status
                    tp1_no_follow_status = self._maybe_execute_risk_session_tp1_no_follow_through(
                        rs,
                        snapshot,
                        price=price,
                        now=now,
                    )
                    if tp1_no_follow_status is not None:
                        return tp1_no_follow_status
                    tp2_no_continuation_status = self._maybe_execute_risk_session_tp2_no_continuation(
                        rs,
                        snapshot,
                        price=price,
                        now=now,
                    )
                    if tp2_no_continuation_status is not None:
                        return tp2_no_continuation_status
                rs.pending_fill_reconcile_since = None
                if price is not None:
                    rs.prev_price = price
                return None

        if size == 0:
            print(f"[risk_monitor_done] {rs.plan_name} position already flat")
            self._emit_status_line("risk_monitor_done", f"持仓已平，结束风险监控: {rs.plan_name}")
            rs.pending_fill_reconcile_since = None
            self._clear_position_basis_state(reason="risk_session_position_flat", position_snapshot=snapshot)
            self._replace_risk_session(None)
            return None
        if expected_sign > 0 and size < 0:
            rs.pending_fill_reconcile_since = None
            self._clear_position_basis_state(reason="risk_session_side_flipped", position_snapshot=snapshot)
            self._replace_risk_session(None)
            return None
        if expected_sign < 0 and size > 0:
            rs.pending_fill_reconcile_since = None
            self._clear_position_basis_state(reason="risk_session_side_flipped", position_snapshot=snapshot)
            self._replace_risk_session(None)
            return None

        if rs.use_resting_exit_orders and size_changed:
            matched = self._match_risk_session_reduction(rs, size)
            if matched is None:
                rs.pending_fill_reconcile_since = None
                if size == 0.0:
                    self._clear_position_basis_state(reason="risk_session_unmatched_position_flat", position_snapshot=snapshot)
                self._replace_risk_session(None)
                return None
            leg_type, keys = matched
            key_set = set(keys)
            rs.executed_leg_names.update(key_set)
            leg_names = [key.split("::", 1)[1] if "::" in key else key for key in keys]
            print(f"[{leg_type}_leg_hit] {rs.plan_name} legs={','.join(leg_names)} size={size:.8f}")
            self._audit_event(
                "management_exit_leg_hit",
                {
                    "plan_name": rs.plan_name,
                    "leg_type": leg_type,
                    "leg_names": leg_names,
                    "matched_keys": list(keys),
                    "price": price,
                    "at": now,
                    "source": "resting_reduce_only_orders",
                    "remaining_size": size,
                },
            )
            rs.resting_exit_orders = [
                ref for ref in list(rs.resting_exit_orders or [])
                if str(ref.get("key", "") or "") not in key_set
            ]
            rs.expected_size = size
            rs.side = str(snapshot.get("side", rs.side))
            if price is not None:
                rs.prev_price = price
            rs.pending_fill_reconcile_since = None
            self._update_staged_risk_session_after_completed_keys(rs, list(keys), now=now)
            self._resync_risk_session_counterpart_orders(rs, leg_type)
            rs.use_resting_exit_orders = bool(rs.resting_exit_orders)
            self._persist_risk_session_state()
            return f"{leg_type}_hit"

        if size_changed:
            rs.pending_fill_reconcile_since = None
            if size == 0.0:
                self._clear_position_basis_state(reason="risk_session_size_changed_flat", position_snapshot=snapshot)
            self._replace_risk_session(None)
            return None

        soft_stop_status = self._maybe_execute_risk_session_soft_stop(
            rs,
            snapshot,
            price=price,
            now=now,
        )
        if soft_stop_status is not None:
            return soft_stop_status

        basis_profit_lock_status = self._maybe_execute_risk_session_basis_profit_lock(
            rs,
            snapshot,
            price=price,
            now=now,
        )
        if basis_profit_lock_status is not None:
            return basis_profit_lock_status

        time_decay_status = self._maybe_execute_risk_session_time_decay_take_profit(
            rs,
            snapshot,
            price=price,
            now=now,
        )
        if time_decay_status is not None:
            return time_decay_status

        tp1_no_follow_status = self._maybe_execute_risk_session_tp1_no_follow_through(
            rs,
            snapshot,
            price=price,
            now=now,
        )
        if tp1_no_follow_status is not None:
            return tp1_no_follow_status

        tp2_no_continuation_status = self._maybe_execute_risk_session_tp2_no_continuation(
            rs,
            snapshot,
            price=price,
            now=now,
        )
        if tp2_no_continuation_status is not None:
            return tp2_no_continuation_status

        previous_trailing_soft_stop = float(getattr(rs, "trailing_soft_stop_price", 0.0) or 0.0)
        previous_trailing_hard_stop = float(getattr(rs, "trailing_hard_stop_price", 0.0) or 0.0)
        previous_trailing_bar_ms = int(getattr(rs, "trailing_last_bar_ms", 0) or 0)
        trailing_updated = self._update_staged_risk_session_trailing_state(rs, now=now)
        if trailing_updated:
            if rs.use_resting_exit_orders:
                self._sync_risk_session_resting_orders(rs)
            self._persist_risk_session_state()
            trailing_payload = {
                "plan_name": rs.plan_name,
                "side": rs.side,
                "timeframe": str(getattr(rs, "trailing_timeframe", "") or ""),
                "soft_stop_price": float(getattr(rs, "trailing_soft_stop_price", 0.0) or 0.0),
                "hard_stop_price": float(getattr(rs, "trailing_hard_stop_price", 0.0) or 0.0),
                "previous_soft_stop_price": previous_trailing_soft_stop,
                "previous_hard_stop_price": previous_trailing_hard_stop,
                "trailing_last_bar_ms": int(getattr(rs, "trailing_last_bar_ms", 0) or 0),
                "previous_trailing_last_bar_ms": previous_trailing_bar_ms,
                "trailing_last_close_price": float(getattr(rs, "trailing_last_close_price", 0.0) or 0.0),
                "trailing_lowest_close": float(getattr(rs, "trailing_lowest_close", 0.0) or 0.0),
                "trailing_highest_close": float(getattr(rs, "trailing_highest_close", 0.0) or 0.0),
                "stop_loss_price": float(getattr(rs, "stop_loss_price", 0.0) or 0.0),
                "use_resting_exit_orders": bool(getattr(rs, "use_resting_exit_orders", False)),
            }
            self._audit_event("risk_session_trailing_stop_updated", trailing_payload)
            self._emit_status_line(
                "risk_session_trailing_stop_updated",
                (
                    f"风控更新 trailing soft SL: {rs.plan_name} | "
                    f"soft_sl {format_display_price(trailing_payload['soft_stop_price'])} | "
                    f"hard_sl {format_display_price(trailing_payload['hard_stop_price'])} | "
                    f"last_close {format_display_price(trailing_payload['trailing_last_close_price'])} | "
                    f"{trailing_payload['timeframe']} trailing"
                ),
                trailing_payload,
            )
        trailing_soft_stop_candidate = self._staged_risk_session_soft_stop_trigger_candidate(rs, now=now)
        if trailing_soft_stop_candidate is not None:
            close_result = self.executor.close_position(rs.side, "soft_trailing_stop", rs.plan_name)
            self._audit_event(
                "risk_session_soft_trailing_stop_triggered",
                {
                    "plan_name": rs.plan_name,
                    "side": rs.side,
                    **trailing_soft_stop_candidate,
                    "result": close_result,
                },
            )
            refreshed_snapshot = self.reader.get_position_snapshot(self.symbol)
            if snapshot_has_open_position(refreshed_snapshot):
                rs.expected_size = float(refreshed_snapshot.get("size", 0.0) or 0.0)
                rs.side = str(refreshed_snapshot.get("side", rs.side) or rs.side)
                rs.pending_fill_reconcile_since = now
            else:
                self._clear_position_basis_state(reason="soft_trailing_stop_flat", position_snapshot=refreshed_snapshot)
                self._replace_risk_session(None)
            return "soft_trailing_stop_hit"

        rs.pending_fill_reconcile_since = None
        if price is not None:
            rs.prev_price = price
        return None
    def step_position_management_session(self, snapshot: dict, now: float) -> Optional[str]:
        if self.position_management_session is None:
            return None
        last_pm_tick_at = float(getattr(self, "last_position_management_tick_at", 0.0) or 0.0)
        if now - last_pm_tick_at < self.risk_poll_seconds:
            return None
        self.last_position_management_tick_at = now
        session = self.position_management_session
        price = safe_float(snapshot.get("mid_price"), None)
        if price is None:
            return None

        session.update_price(price, now)

        scenario = session.position_management.scenario
        if scenario is None:
            if session.scenarios_completed():
                self.position_management_session = None
            return None
        state = session.runtimes[SCENARIO_RUNTIME_KEY]
        if not state.completed:
            if not state.observing:
                if observe_when_all_contains_price(scenario.observe_when_all, price):
                    state.observing = True
                    state.observing_at = now
                    print(f"[management_scenario_observing] price={format_display_price(price)}")
                    self._audit_event(
                        "management_scenario_observing",
                        {"scenario": build_scenario_execution_view(scenario), "price": price, "observing_at": now},
                    )
                else:
                    session.prev_price = price
                    return None
            timeout_anchor = state.observing_at
            if timeout_anchor and now - timeout_anchor > scenario.execute_when_all.timeout_seconds:
                state.completed = True
                print(f"[management_scenario_timeout] price={format_display_price(price)}")
                self._audit_event("management_scenario_timeout", {"price": price, "at": now})
                session.prev_price = price
                if session.scenarios_completed():
                    self.position_management_session = None
                return "management_scenario_timeout"
            if not state.armed:
                execute_condition = scenario.execute_when_all.condition
                if execute_condition is None or evaluate_condition(execute_condition, price, session.prev_price, session.history, now=now, since_ts=state.observing_at):
                    state.armed = True
                    state.armed_at = now
                    print(f"[management_scenario_armed] price={format_display_price(price)}")
                    self._audit_event(
                        "management_scenario_armed",
                        {"scenario": build_scenario_execution_view(scenario), "price": price, "armed_at": now},
                    )
                else:
                    session.prev_price = price
                    return None
            key = SCENARIO_RUNTIME_KEY + "::armed"
            if key not in session.executed_plan_names:
                session.executed_plan_names.add(key)
                state.completed = True
                print("[management_scenario_execute]")
                self._audit_event(
                    "management_scenario_execute",
                    {"price": price, "at": now},
                )
                management_exec = self.execute_management_decision(
                    session.position_management.action_decision,
                    key,
                    session.position_management,
                    trigger_confidence_raw=session.trigger_confidence_raw,
                )
                position_after = management_exec["position_after"]
                if not management_exec.get("accepted", False):
                    print(f"[management_scenario_execution_rejected] {key}")
                    self._audit_event(
                        "management_scenario_execution_rejected",
                        {"result": management_exec},
                    )
                    if session.scenarios_completed():
                        self.position_management_session = None
                    self._schedule_next_active_query(position_after)
                    return "management_action_rejected"
                if bool(management_exec.get("entry_order_pending", False)) and not snapshot_has_open_position(position_after):
                    self._set_pending_entry_order_session(
                        plan_name=key,
                        management_decision=session.position_management.action_decision,
                        position_management=session.position_management,
                        post_fill_risk_template=getattr(self.current_playbook, "post_fill_risk_template", None),
                        execution_result=management_exec,
                    )
                    self._replace_risk_session(None)
                    self.position_management_session = None
                    self._schedule_next_active_query(position_after)
                    return None
                self._set_risk_session_after_management_decision(
                    session.position_management.action_decision,
                    session.position_management,
                    getattr(self.current_playbook, "post_fill_risk_template", None),
                    position_after,
                    key,
                    add_fill_entry_price=self._extract_filled_avg_price_from_execution_result(management_exec),
                )
                self.position_management_session = None
                self._schedule_next_active_query(position_after)
                if snapshot_has_open_position(position_after):
                    return "management_action_executed"
                return None

        session.prev_price = price
        if session.scenarios_completed():
            self.position_management_session = None
        return None
    def run_forever(self) -> None:
        self._warm_up_market_catalog()
        self.log_startup()
        self._emit_startup_ready()
        pending_reason: Optional[str] = None
        pending_event: Optional[Dict[str, Any]] = None
        pending_recent_events: Optional[List[Dict[str, Any]]] = None
        pending_prefetched_passive_event_judge: Optional[Dict[str, Any]] = None
        queued_passive_event: Optional[Dict[str, Any]] = None
        queued_passive_recent_events: Optional[List[Dict[str, Any]]] = None
        queued_fresh_after_stale_events: Optional[List[Dict[str, Any]]] = None
        while True:
            try:
                now = time.time()
                now_utc = datetime.fromtimestamp(now, tz=timezone.utc)
                passive_prefetch_started_this_loop = False
                ready_prefetched_passive_event_judge = self._consume_ready_passive_event_judge_prefetch()
                if ready_prefetched_passive_event_judge is not None:
                    self._append_stale_relevant_passive_tape_events(ready_prefetched_passive_event_judge)
                    if not bool(ready_prefetched_passive_event_judge.get("passive_realtime_allowed", True)):
                        selected_event = ready_prefetched_passive_event_judge.get("trigger_event")
                        judge_output = ready_prefetched_passive_event_judge.get("judge_output") or {}
                        print_line(
                            "[passive_realtime_blocked_after_step1] "
                            f"{_status_event_brief(selected_event)} | "
                            f"relevance={judge_output.get('trigger_event_relevance')} "
                            f"action={judge_output.get('action')} "
                            f"confidence={judge_output.get('trigger_confidence')}"
                        )
                        self._audit_event(
                            "passive_realtime_blocked_after_step1",
                            {
                                "trigger_event": selected_event,
                                "event_key": ready_prefetched_passive_event_judge.get("event_key", ""),
                                "trade_symbol": ready_prefetched_passive_event_judge.get("trade_symbol", ""),
                                "judge_output": judge_output,
                                "passive_realtime_debug": ready_prefetched_passive_event_judge.get("passive_realtime_debug") or {},
                                "batch_event_count": ready_prefetched_passive_event_judge.get("batch_event_count", 1),
                                "batch_candidates": ready_prefetched_passive_event_judge.get("batch_candidates") or [],
                            },
                        )
                        ready_prefetched_passive_event_judge = None
                    if ready_prefetched_passive_event_judge is not None:
                        passive_trade_gate = self._passive_prefetched_trade_gate_block(ready_prefetched_passive_event_judge)
                        if passive_trade_gate is not None:
                            selected_event = ready_prefetched_passive_event_judge.get("trigger_event")
                            tape_update = self._append_realtime_relevant_passive_tape_event(ready_prefetched_passive_event_judge)
                            print_line(
                                "[passive_trade_gate_blocked] "
                                f"{_status_event_brief(selected_event)} | "
                                f"relevance={passive_trade_gate.get('trigger_event_relevance')} "
                                f"action={passive_trade_gate.get('action')} "
                                f"confidence={float(passive_trade_gate.get('trigger_confidence') or 0.0):.2f} "
                                f"threshold={float(passive_trade_gate.get('threshold') or 0.0):.2f} "
                                f"reason={passive_trade_gate.get('reason')}"
                            )
                            self._audit_event(
                                "passive_trade_gate_blocked",
                                {
                                    "trigger_event": selected_event,
                                    "event_key": ready_prefetched_passive_event_judge.get("event_key", ""),
                                    "trade_symbol": ready_prefetched_passive_event_judge.get("trade_symbol", ""),
                                    "judge_output": ready_prefetched_passive_event_judge.get("judge_output") or {},
                                    "trade_gate": passive_trade_gate,
                                    "tape_update": tape_update,
                                    "batch_event_count": ready_prefetched_passive_event_judge.get("batch_event_count", 1),
                                    "batch_candidates": ready_prefetched_passive_event_judge.get("batch_candidates") or [],
                                },
                            )
                            ready_prefetched_passive_event_judge = None
                    if ready_prefetched_passive_event_judge is None and queued_fresh_after_stale_events:
                        fresh_after_stale_events = [dict(item) for item in queued_fresh_after_stale_events if isinstance(item, dict)]
                        queued_fresh_after_stale_events = None
                        if fresh_after_stale_events:
                            if not self._start_passive_event_judge_prefetch(fresh_after_stale_events):
                                queued_passive_event = fresh_after_stale_events[-1]
                                queued_passive_recent_events = None
                            else:
                                passive_prefetch_started_this_loop = True
                new_events = self.events.poll()
                if new_events:
                    latest_event = new_events[-1]
                    print(f"[new_events] count={len(new_events)} latest={_status_event_brief(latest_event)}")
                    self._print_json_block("new_events", new_events)
                    if self.enable_passive_event_query:
                        fresh_step1_events: List[Dict[str, Any]] = []
                        stale_step1_events: List[Dict[str, Any]] = []
                        ignored_events: List[Dict[str, Any]] = []
                        stale_events: List[Dict[str, Any]] = []
                        apply_relevance_filter = bool(getattr(self, "passive_event_relevance_filter", False))
                        for event in new_events:
                            passive_allowed, passive_allow_debug = self._event_allows_passive_query(event)
                            if not passive_allowed:
                                ignored_events.append({"event": event, "relevance": passive_allow_debug})
                                continue
                            if apply_relevance_filter:
                                relevant, relevance_debug = self._event_is_relevant_for_passive_query(event)
                                if not relevant:
                                    ignored_events.append({"event": event, "relevance": relevance_debug})
                                    continue
                            realtime_allowed, realtime_debug = self._event_allows_passive_realtime_trigger(event)
                            if not realtime_allowed:
                                stale_events.append({"event": event, "realtime": realtime_debug})
                            event_for_step1 = dict(event)
                            event_for_step1["_passive_realtime_allowed"] = bool(realtime_allowed)
                            event_for_step1["_passive_realtime_debug"] = dict(realtime_debug or {})
                            if realtime_allowed:
                                fresh_step1_events.append(event_for_step1)
                            else:
                                stale_step1_events.append(event_for_step1)
                        if ignored_events:
                            ignored_preview = [
                                {
                                    "title": str((item.get("event") or {}).get("title", "") or ""),
                                    "source": str((item.get("event") or {}).get("source", "") or ""),
                                    "relevance": item.get("relevance", {}),
                                }
                                for item in ignored_events[-3:]
                            ]
                            ignored_titles = "; ".join(_status_event_brief(item.get("event")) for item in ignored_events[-3:])
                            print(f"[passive_events_ignored] count={len(ignored_events)} latest={ignored_titles}")
                            self._print_json_block("passive_events_ignored", ignored_preview)
                            self._audit_event("passive_events_ignored", {"count": len(ignored_events), "events": ignored_events})
                        if stale_events:
                            stale_preview = [
                                {
                                    "title": str((item.get("event") or {}).get("title", "") or ""),
                                    "source": str((item.get("event") or {}).get("source", "") or ""),
                                    "published_at": str((item.get("event") or {}).get("published_at", "") or ""),
                                    "seen_at": str((item.get("event") or {}).get("seen_at", "") or ""),
                                    "realtime": item.get("realtime", {}),
                                }
                                for item in stale_events[-3:]
                            ]
                            stale_titles = "; ".join(_status_event_brief(item.get("event")) for item in stale_events[-3:])
                            threshold_seconds = max(
                                0.0,
                                float(getattr(self, "passive_max_published_age_on_seen_hours", 0.0) or 0.0) * 3600.0,
                            )
                            print(
                                "[passive_events_realtime_blocked] "
                                f"count={len(stale_events)} threshold_seconds={threshold_seconds:.1f} "
                                f"latest={stale_titles}"
                            )
                            payload = {"count": len(stale_events), "events": stale_events, "threshold_seconds": threshold_seconds}
                            self._print_json_block("passive_events_realtime_blocked", stale_preview)
                            self._audit_event("passive_events_realtime_blocked", payload)
                            self._emit_status_line(
                                "passive_events_realtime_blocked",
                                f"passive realtime blocked {len(stale_events)} stale event(s) | latest {stale_titles}",
                                {"count": len(stale_events), "events": stale_preview, "threshold_seconds": threshold_seconds},
                            )
                        if stale_step1_events and fresh_step1_events:
                            ready_prefetched_passive_event_judge = None
                            queued_fresh_after_stale_events = [dict(item) for item in fresh_step1_events]
                            if not self._start_passive_event_judge_prefetch(stale_step1_events):
                                queued_passive_event = stale_step1_events[-1]
                                queued_passive_recent_events = None
                            else:
                                passive_prefetch_started_this_loop = True
                                queued_passive_event = None
                                queued_passive_recent_events = None
                        elif stale_step1_events or fresh_step1_events:
                            step1_events = stale_step1_events or fresh_step1_events
                            ready_prefetched_passive_event_judge = None
                            if not self._start_passive_event_judge_prefetch(step1_events):
                                queued_passive_event = step1_events[-1]
                                queued_passive_recent_events = None
                            else:
                                passive_prefetch_started_this_loop = True
                                queued_passive_event = None
                                queued_passive_recent_events = None

                if pending_reason is None and ready_prefetched_passive_event_judge is not None and self.enable_passive_event_query:
                    pending_reason = "passive_event_trigger"
                    pending_event = ready_prefetched_passive_event_judge.get("trigger_event")
                    pending_recent_events = list(ready_prefetched_passive_event_judge.get("recent_events") or [])
                    pending_prefetched_passive_event_judge = None if ready_prefetched_passive_event_judge.get("prefetch_failed") else ready_prefetched_passive_event_judge
                    self._audit_event(
                        "passive_event_judge_prefetch_prioritized",
                        {
                            "trigger_event": pending_event,
                            "event_key": ready_prefetched_passive_event_judge.get("event_key", ""),
                            "trade_symbol": ready_prefetched_passive_event_judge.get("trade_symbol", ""),
                            "judge_output": ready_prefetched_passive_event_judge.get("judge_output"),
                            "judge_duplicate_candidate_count": (ready_prefetched_passive_event_judge.get("judge_debug") or {}).get("passive_event_judge_duplicate_candidate_count", 0),
                            "judge_unrelated_candidate_count": (ready_prefetched_passive_event_judge.get("judge_debug") or {}).get("passive_event_judge_unrelated_candidate_count", 0),
                            "judge_relevant_candidate_count": (ready_prefetched_passive_event_judge.get("judge_debug") or {}).get("passive_event_judge_relevant_candidate_count", 0),
                            "judge_duplicate_relevant_conflict": bool((ready_prefetched_passive_event_judge.get("judge_debug") or {}).get("passive_event_judge_duplicate_relevant_conflict", False)),
                            "judge_unrelated_relevant_conflict": bool((ready_prefetched_passive_event_judge.get("judge_debug") or {}).get("passive_event_judge_unrelated_relevant_conflict", False)),
                            "judge_confidence_adjustment": (ready_prefetched_passive_event_judge.get("judge_debug") or {}).get("passive_event_judge_confidence_adjustment") or {},
                            "judge_selected_raw_trigger_confidence": (ready_prefetched_passive_event_judge.get("judge_debug") or {}).get("passive_event_judge_selected_raw_trigger_confidence"),
                            "judge_selected_effective_trigger_confidence": (ready_prefetched_passive_event_judge.get("judge_debug") or {}).get("passive_event_judge_selected_effective_trigger_confidence"),
                            "judge_duplicate_relevant_conflict_multiplier": (ready_prefetched_passive_event_judge.get("judge_debug") or {}).get("passive_event_judge_duplicate_relevant_conflict_multiplier", 1.0),
                            "judge_unrelated_relevant_conflict_multiplier": (ready_prefetched_passive_event_judge.get("judge_debug") or {}).get("passive_event_judge_unrelated_relevant_conflict_multiplier", 1.0),
                            "recent_events_source": ready_prefetched_passive_event_judge.get("recent_events_source", ""),
                            "prefetch_failed": bool(ready_prefetched_passive_event_judge.get("prefetch_failed")),
                            "batch_event_count": ready_prefetched_passive_event_judge.get("batch_event_count", 1),
                            "batch_candidates": ready_prefetched_passive_event_judge.get("batch_candidates") or [],
                            "batch_selection_rule": ready_prefetched_passive_event_judge.get("batch_selection_rule", ""),
                            "batch_selected_index": ready_prefetched_passive_event_judge.get("batch_selected_index"),
                        },
                    )
                    ready_prefetched_passive_event_judge = None

                passive_query_in_flight = (
                    passive_prefetch_started_this_loop
                    or self._passive_event_judge_prefetch_pending()
                    or ready_prefetched_passive_event_judge is not None
                    or pending_reason == "passive_event_trigger"
                )

                pending_user_fill_events: List[Dict[str, Any]] = []
                loop_positions: Dict[str, Any] = {}
                runtime_symbol = ""
                symbol_snapshot: Dict[str, Any] = {}
                price = None
                if pending_reason is None and not passive_query_in_flight:
                    self._maintain_user_fills_subscription(now)
                    pending_user_fill_events = self._drain_pending_user_fill_events()
                    self.step_pending_entry_order_session(now)

                    loop_positions = self.reader.get_all_positions()
                    runtime_symbol = self._runtime_symbol(loop_positions)
                    symbol_snapshot = self.reader.get_position_snapshot(runtime_symbol, all_positions=loop_positions) if runtime_symbol else self._empty_runtime_snapshot(loop_positions)
                    price = safe_float(symbol_snapshot.get("mid_price"), None)
                    self._maybe_restore_startup_live_tpsl(symbol_snapshot)

                passive_judge_waiting = passive_query_in_flight

                if pending_reason is None and not passive_judge_waiting and self._perform_helper_reset(now_utc):
                    pending_reason = "helper_reset_refresh"
                    pending_event = None
                    pending_recent_events = None
                    pending_prefetched_passive_event_judge = None

                if pending_reason is None and not passive_query_in_flight:
                    risk_status = self.step_risk_session(symbol_snapshot, now, fill_events=pending_user_fill_events)
                    if risk_status in {"take_profit_hit", "stop_loss_hit"}:
                        risk_status = None
                    if risk_status is not None and self.enable_active_query:
                        pending_reason = risk_status
                        pending_event = None
                        pending_recent_events = None
                        pending_prefetched_passive_event_judge = None

                pm_status = None
                if pending_reason is None and not passive_query_in_flight and getattr(self, "position_management_session", None) is not None and price is not None:
                    pm_symbol = canonicalize_execution_symbol(runtime_symbol or self.symbol or "")
                    _, symbol_snapshot, _ = self._fetch_selected_symbol_position_context(pm_symbol)
                    pm_status = self.step_position_management_session(symbol_snapshot, now)
                    if pm_status is not None and self.enable_active_query:
                        pending_reason = pm_status
                        pending_event = None
                        pending_recent_events = None
                        pending_prefetched_passive_event_judge = None

                if pending_reason is None and queued_passive_event is not None and self.enable_passive_event_query and not self._passive_event_judge_prefetch_pending():
                    pending_reason = "passive_event_trigger"
                    pending_event = queued_passive_event
                    pending_recent_events = list(queued_passive_recent_events or [])
                    pending_prefetched_passive_event_judge = None
                    queued_passive_event = None
                    queued_passive_recent_events = None

                passive_judge_pending = passive_query_in_flight

                if pending_reason is None and not passive_judge_pending and self.current_playbook is None and (self.enable_active_query or self.enable_passive_event_query):
                    if self.enable_active_query and bool(getattr(self, "enable_active_playbook", True)):
                        pending_reason = "no_active_playbook"
                        pending_recent_events = None
                        pending_prefetched_passive_event_judge = None

                if pending_reason is None and not passive_judge_pending and self.active_query_allowed_now() and now >= self.next_active_query_due_at:
                    pending_reason = "active_periodic_refresh"
                    pending_event = None
                    pending_recent_events = None
                    pending_prefetched_passive_event_judge = None

                if pending_reason is not None:
                    reason = pending_reason
                    event = pending_event
                    event_batch = list(pending_recent_events or [])
                    prefetched_passive_event_judge = pending_prefetched_passive_event_judge
                    pending_reason = None
                    pending_event = None
                    pending_recent_events = None
                    pending_prefetched_passive_event_judge = None
                    self.query_new_playbook(
                        reason,
                        event,
                        recent_events=event_batch,
                        prefetched_passive_event_judge=prefetched_passive_event_judge,
                    )
                    time.sleep(max(0.0, self.fast_replan_delay_seconds))
                self._wait_for_loop_wake(self.loop_sleep_seconds)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                sleep_seconds = self._runtime_error_sleep_seconds(exc)
                self._record_runtime_error(
                    "main_loop",
                    exc,
                    {
                        "pending_reason": pending_reason,
                        "has_current_playbook": self.current_playbook is not None,
                        "has_risk_session": self.risk_session is not None,
                        "has_position_management_session": getattr(self, "position_management_session", None) is not None,
                    },
                    retry_after_seconds=sleep_seconds,
                )
                pending_reason = None
                pending_event = None
                pending_recent_events = None
                pending_prefetched_passive_event_judge = None
                time.sleep(sleep_seconds)
    def run_once(
        self,
        trigger_reason: str = "manual_once",
        trigger_event: Optional[Dict[str, Any]] = None,
        recent_events: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._warm_up_market_catalog()
        self.log_startup()
        loop_positions = self.reader.get_all_positions()
        runtime_symbol = self._runtime_symbol(loop_positions)
        symbol_snapshot = self.reader.get_position_snapshot(runtime_symbol, all_positions=loop_positions) if runtime_symbol else self._empty_runtime_snapshot(loop_positions)
        self._maybe_restore_startup_live_tpsl(symbol_snapshot)
        self.query_new_playbook(trigger_reason, trigger_event, recent_events=recent_events)
