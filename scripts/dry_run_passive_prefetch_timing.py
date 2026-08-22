#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType
from typing import Any, Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return str(value)


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            try:
                value = json.loads(text)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                rows.append(value)
    return rows


def _load_event(args: argparse.Namespace) -> Dict[str, Any]:
    if args.event_json:
        text = sys.stdin.read() if args.event_json == "-" else Path(args.event_json).read_text(encoding="utf-8")
        value = json.loads(text)
        if isinstance(value, dict) and isinstance(value.get("event"), dict):
            return dict(value["event"])
        if not isinstance(value, dict):
            raise ValueError("--event-json must contain a JSON object")
        return dict(value)

    events_path = Path(args.events_path)
    events = _load_jsonl(events_path)
    if not events:
        raise ValueError(f"No events found in {events_path}")

    matches = events
    if args.source:
        source = str(args.source).strip().lower()
        matches = [event for event in matches if str(event.get("source", "") or "").strip().lower() == source]
    if args.item_id:
        item_id = str(args.item_id).strip()
        matches = [event for event in matches if str(event.get("item_id", "") or "").strip() == item_id]
    if args.title_contains:
        needle = str(args.title_contains).strip().lower()
        matches = [event for event in matches if needle in str(event.get("title", "") or "").lower()]

    if not matches:
        raise ValueError("No event matched the supplied filters")
    return dict(matches[-1])


def _brief_event(event: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "source": event.get("source"),
        "item_id": event.get("item_id"),
        "title": event.get("title"),
        "published_at": event.get("published_at"),
        "seen_at": event.get("seen_at"),
        "event_timestamp": event.get("event_timestamp"),
        "category": event.get("category"),
    }


def _brief_position(snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = snapshot if isinstance(snapshot, dict) else {}
    keys = (
        "symbol",
        "side",
        "size",
        "entry_price",
        "mid_price",
        "notional_usd",
        "leverage",
        "max_leverage",
        "only_isolated",
        "margin_used",
        "account_equity_usd",
        "perp_account_equity_usd",
        "available_margin_usd",
        "withdrawable_usd",
        "remaining_capital_usd",
        "isolated_margin_basis_usd",
        "cross_margin_basis_usd",
        "isolated_available_margin_usd",
        "cross_available_margin_usd",
    )
    return {key: raw.get(key) for key in keys if key in raw}


def _brief_all_positions(snapshot: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    raw = snapshot if isinstance(snapshot, dict) else {}
    keys = (
        "partial",
        "partial_scope",
        "positions_count",
        "total_notional_usd",
        "total_margin_used_usd",
        "account_equity_usd",
        "perp_account_equity_usd",
        "available_margin_usd",
        "withdrawable_usd",
        "remaining_capital_usd",
        "isolated_margin_basis_usd",
        "cross_margin_basis_usd",
        "isolated_available_margin_usd",
        "cross_available_margin_usd",
    )
    return {key: raw.get(key) for key in keys if key in raw}


def _mark(timings: Dict[str, Any], key: str) -> None:
    timings[f"{key}_wall"] = time.time()
    timings[f"{key}_perf"] = time.perf_counter()
    timings[f"{key}_utc"] = _utc_now_iso()


def _elapsed(timings: Dict[str, Any], start: str, end: str) -> Optional[float]:
    start_value = timings.get(f"{start}_perf")
    end_value = timings.get(f"{end}_perf")
    if isinstance(start_value, (int, float)) and isinstance(end_value, (int, float)):
        return max(0.0, float(end_value) - float(start_value))
    return None


def _wrap_for_timing(agent: Any, timings: Dict[str, Any]) -> None:
    timings.setdefault("position_context_calls", [])
    timings.setdefault("selected_symbol_position_context_calls", [])
    timings.setdefault("timed_sections", [])

    def record_section(name: str, start: float, error: str = "") -> None:
        timings["timed_sections"].append(
            {
                "name": name,
                "duration_seconds": max(0.0, time.perf_counter() - start),
                "done_utc": _utc_now_iso(),
                "error": error,
            }
        )

    def wrap_callable(obj: Any, attr: str, name: str) -> None:
        original = getattr(obj, attr, None)
        if not callable(original):
            return

        def timed(*args: Any, **kwargs: Any) -> Any:
            started_at = time.perf_counter()
            error = ""
            try:
                return original(*args, **kwargs)
            except Exception as exc:
                error = str(exc)
                raise
            finally:
                record_section(name, started_at, error)

        setattr(obj, attr, timed)

    wrap_callable(agent, "_build_prefetched_passive_query_candidates", "agent._build_prefetched_passive_query_candidates")
    wrap_callable(agent, "_resolve_recent_passive_event_symbol", "agent._resolve_recent_passive_event_symbol")
    wrap_callable(agent, "render_user_query", "agent.render_user_query")
    wrap_callable(agent, "_should_apply_passive_event_relevance_filter", "agent._should_apply_passive_event_relevance_filter")
    wrap_callable(agent, "_event_is_relevant_for_passive_query", "agent._event_is_relevant_for_passive_query")
    wrap_callable(agent, "_empty_runtime_snapshot", "agent._empty_runtime_snapshot")
    wrap_callable(agent.engine, "_get_cached_helper_market_mainline_context", "engine._get_cached_helper_market_mainline_context")
    if getattr(agent, "events", None) is not None:
        wrap_callable(agent.events, "recent", "events.recent")

    original_get_playbook = agent.engine.get_playbook

    def timed_get_playbook(*args: Any, **kwargs: Any) -> Any:
        _mark(timings, "engine_get_playbook_start")
        started_at = time.perf_counter()
        error = ""
        try:
            return original_get_playbook(*args, **kwargs)
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            _mark(timings, "engine_get_playbook_done")
            record_section("engine.get_playbook", started_at, error)

    agent.engine.get_playbook = timed_get_playbook

    original_chart_resolve = agent.engine._resolve_prefetched_passive_chart_context

    def timed_chart_resolve(*args: Any, **kwargs: Any) -> Any:
        _mark(timings, "chart_context_resolve_start")
        started_at = time.perf_counter()
        error = ""
        try:
            return original_chart_resolve(*args, **kwargs)
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            _mark(timings, "chart_context_resolve_done")
            record_section("engine._resolve_prefetched_passive_chart_context", started_at, error)

    agent.engine._resolve_prefetched_passive_chart_context = timed_chart_resolve

    original_two_step = agent.engine._call_passive_two_step_model

    def timed_two_step(*args: Any, **kwargs: Any) -> Any:
        _mark(timings, "passive_two_step_start")
        started_at = time.perf_counter()
        error = ""
        try:
            return original_two_step(*args, **kwargs)
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            _mark(timings, "passive_two_step_done")
            record_section("engine._call_passive_two_step_model", started_at, error)

    agent.engine._call_passive_two_step_model = timed_two_step

    original_pricing = agent.engine._call_passive_technical_pricing

    def timed_pricing(*args: Any, **kwargs: Any) -> Any:
        _mark(timings, "step2_pricing_start")
        try:
            return original_pricing(*args, **kwargs)
        finally:
            _mark(timings, "step2_pricing_done")

    agent.engine._call_passive_technical_pricing = timed_pricing

    original_flatten = agent._flatten_unselected_positions

    def timed_flatten(*args: Any, **kwargs: Any) -> Any:
        _mark(timings, "flatten_unselected_start")
        started_at = time.perf_counter()
        error = ""
        try:
            return original_flatten(*args, **kwargs)
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            _mark(timings, "flatten_unselected_done")
            record_section("agent._flatten_unselected_positions", started_at, error)

    agent._flatten_unselected_positions = timed_flatten

    original_materialize = agent._materialize_live_position_management_from_entry_plan

    def timed_materialize(*args: Any, **kwargs: Any) -> Any:
        _mark(timings, "materialize_start")
        started_at = time.perf_counter()
        error = ""
        try:
            return original_materialize(*args, **kwargs)
        except Exception as exc:
            error = str(exc)
            raise
        finally:
            _mark(timings, "materialize_done")
            record_section("agent._materialize_live_position_management_from_entry_plan", started_at, error)

    agent._materialize_live_position_management_from_entry_plan = timed_materialize

    original_fetch_context = agent._fetch_position_context

    def timed_fetch_position_context(self: Any, *args: Any, **kwargs: Any) -> Any:
        prefix = str(kwargs.get("thread_name_prefix", "") or "")
        key = "pre_execution_context" if prefix == "pre-execution-context" else "position_context"
        call_record: Dict[str, Any] = {
            "prefix": prefix,
            "symbol": args[0] if args else kwargs.get("symbol"),
            "start_utc": _utc_now_iso(),
            "stack": [
                {
                    "file": str(frame.filename),
                    "line": frame.lineno,
                    "name": frame.name,
                    "text": frame.line,
                }
                for frame in traceback.extract_stack(limit=8)[:-1]
            ],
        }
        call_start = time.perf_counter()
        _mark(timings, f"{key}_start")
        try:
            return original_fetch_context(*args, **kwargs)
        finally:
            _mark(timings, f"{key}_done")
            call_record["duration_seconds"] = max(0.0, time.perf_counter() - call_start)
            call_record["done_utc"] = _utc_now_iso()
            timings["position_context_calls"].append(call_record)

    agent._fetch_position_context = MethodType(timed_fetch_position_context, agent)

    original_selected_symbol_context = getattr(agent.reader, "get_selected_symbol_position_context", None)
    if callable(original_selected_symbol_context):
        def timed_selected_symbol_context(*args: Any, **kwargs: Any) -> Any:
            call_record: Dict[str, Any] = {
                "symbol": args[0] if args else kwargs.get("symbol"),
                "start_utc": _utc_now_iso(),
            }
            call_start = time.perf_counter()
            _mark(timings, "selected_symbol_position_context_start")
            error = ""
            try:
                return original_selected_symbol_context(*args, **kwargs)
            except Exception as exc:
                error = str(exc)
                raise
            finally:
                _mark(timings, "selected_symbol_position_context_done")
                call_record["duration_seconds"] = max(0.0, time.perf_counter() - call_start)
                call_record["done_utc"] = _utc_now_iso()
                call_record["error"] = error
                timings["selected_symbol_position_context_calls"].append(call_record)
                record_section("reader.get_selected_symbol_position_context", call_start, error)

        agent.reader.get_selected_symbol_position_context = timed_selected_symbol_context

    def dry_run_execute_immediate_playbook_action(self: Any, playbook: Any, symbol_position: dict, execution_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        _mark(timings, "execution_would_start")
        timings["execution_suppressed"] = True
        timings["execution_context_available"] = isinstance(execution_context, dict)
        if isinstance(execution_context, dict):
            timings["execution_context_summary"] = {
                "execution_mid_price": execution_context.get("execution_mid_price"),
                "position_before": _brief_position(execution_context.get("position_before")),
                "all_positions": _brief_all_positions(execution_context.get("all_positions")),
            }
        decision = None
        try:
            action = self._resolve_immediate_playbook_action(playbook, symbol_position)
            decision = action.get("decision").to_dict() if isinstance(action, dict) and hasattr(action.get("decision"), "to_dict") else action
        except Exception as exc:
            decision = {"error": str(exc)}
        timings["would_execute"] = decision
        print("[dry_run_execution_suppressed] immediate execution was not submitted")
        return {"kind": "dry_run_suppressed", "would_execute": decision}

    agent._execute_immediate_playbook_action = MethodType(dry_run_execute_immediate_playbook_action, agent)


def _build_summary(timings: Dict[str, Any], agent: Any, prefetched: Optional[Dict[str, Any]], query_error: Optional[str]) -> Dict[str, Any]:
    debug = dict(getattr(agent.engine, "last_call_debug", {}) or {})
    chart_debug = debug.get("passive_chart_context_prefetch") if isinstance(debug.get("passive_chart_context_prefetch"), dict) else {}
    judge_output = dict((prefetched or {}).get("judge_output") or {}) if isinstance(prefetched, dict) else {}
    playbook = debug.get("validated_playbook") if isinstance(debug.get("validated_playbook"), dict) else {}
    entry_plan = playbook.get("entry_plan") if isinstance(playbook.get("entry_plan"), dict) else {}
    action_decision = entry_plan.get("action_decision") if isinstance(entry_plan.get("action_decision"), dict) else {}
    summary = {
        "prefetch_start_to_step1_done_seconds": _elapsed(timings, "prefetch_start_call", "step1_prefetch_done"),
        "step1_done_to_query_start_seconds": _elapsed(timings, "step1_prefetch_done", "query_new_playbook_start"),
        "query_start_to_step2_start_seconds": _elapsed(timings, "query_new_playbook_start", "step2_pricing_start"),
        "step2_duration_seconds": _elapsed(timings, "step2_pricing_start", "step2_pricing_done"),
        "step2_done_to_execution_would_start_seconds": _elapsed(timings, "step2_pricing_done", "execution_would_start"),
        "prefetch_start_to_execution_would_start_seconds": _elapsed(timings, "prefetch_start_call", "execution_would_start"),
        "prefetch_start_to_query_return_seconds": _elapsed(timings, "prefetch_start_call", "query_new_playbook_done"),
        "query_error": query_error,
        "step2_executed": bool(debug.get("passive_step2_executed")),
        "judge_output": judge_output,
        "entry_action": action_decision.get("action"),
        "entry_price": action_decision.get("entry_price"),
        "stop_loss_price": action_decision.get("stop_loss_price"),
        "chart_prefetch": {
            "started": bool(chart_debug.get("started")),
            "completed_before_wait": chart_debug.get("completed_before_wait"),
            "reused": chart_debug.get("reused"),
            "duration_seconds": chart_debug.get("duration_seconds"),
            "chart_summary_count": chart_debug.get("chart_summary_count"),
            "image_count": chart_debug.get("image_count"),
            "error": chart_debug.get("error"),
        },
        "execution_suppressed": bool(timings.get("execution_suppressed")),
        "execution_context_available": bool(timings.get("execution_context_available")),
        "execution_context_summary": timings.get("execution_context_summary"),
        "would_execute": timings.get("would_execute"),
    }
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dry-run the live passive prefetch -> Step2 -> pre-execution path without submitting orders."
    )
    parser.add_argument("--events-path", default=os.getenv("EVENTS_JSONL_PATH", "data/free_sources_watch/events.jsonl"))
    parser.add_argument("--event-json", default="", help="Path to a JSON event object, or '-' for stdin")
    parser.add_argument("--source", default="")
    parser.add_argument("--item-id", default="")
    parser.add_argument("--title-contains", default="")
    parser.add_argument("--timeout-seconds", type=float, default=120.0)
    parser.add_argument("--poll-seconds", type=float, default=0.05)
    parser.add_argument("--output", type=Path, default=Path("tmp_bad_case_queries/passive_prefetch_timing_dry_run.json"))
    parser.add_argument("--query", default=os.getenv("MANUAL_STRATEGY_QUERY", ""))
    parser.add_argument("--force", action="store_true", help="Run even if the local passive relevance filter would ignore the event")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    os.environ.setdefault("ENABLE_LIVE_TRADING", "false")
    os.environ.setdefault("ENABLE_HYPERLIQUID_USER_FILLS_WEBSOCKET", "false")
    os.environ.setdefault("ENABLE_PLAYBOOK_STATUS_LOG", "false")
    os.environ.setdefault("PLAYBOOK_AUDIT_LOG_PATH", "logs/passive_prefetch_timing_dry_run_audit.jsonl")

    from unified_market_agent import UnifiedMarketAgent

    event = _load_event(args)
    agent = UnifiedMarketAgent(args.query)
    agent.executor.enabled = False

    local_relevance = {"checked": False, "relevant": True, "debug": {}}
    if hasattr(agent, "_event_is_relevant_for_passive_query"):
        local_relevance["checked"] = True
        relevant, debug = agent._event_is_relevant_for_passive_query(event)
        local_relevance["relevant"] = bool(relevant)
        local_relevance["debug"] = debug
        if not relevant and not args.force:
            raise SystemExit(
                json.dumps(
                    {
                        "error": "event_failed_local_passive_relevance_filter",
                        "event": _brief_event(event),
                        "local_relevance": local_relevance,
                        "hint": "Use --force to time the path anyway.",
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )

    timings: Dict[str, Any] = {}
    _wrap_for_timing(agent, timings)

    _mark(timings, "prefetch_start_call")
    started = agent._start_passive_event_judge_prefetch(event)
    _mark(timings, "prefetch_start_return")
    if not started:
        raise SystemExit("Failed to start passive event judge prefetch")

    prefetched: Optional[Dict[str, Any]] = None
    deadline = time.time() + max(1.0, float(args.timeout_seconds))
    while time.time() < deadline:
        result = agent._consume_ready_passive_event_judge_prefetch()
        if result is not None:
            prefetched = result
            _mark(timings, "step1_prefetch_done")
            break
        time.sleep(max(0.01, float(args.poll_seconds)))
    if prefetched is None:
        raise SystemExit("Timed out waiting for passive event judge prefetch")

    query_error: Optional[str] = None
    _mark(timings, "query_new_playbook_start")
    try:
        agent.query_new_playbook(
            "passive_event_trigger",
            event,
            recent_events=list(prefetched.get("recent_events") or []),
            prefetched_passive_event_judge=prefetched,
        )
    except Exception as exc:
        query_error = str(exc)
        raise
    finally:
        _mark(timings, "query_new_playbook_done")

    summary = _build_summary(timings, agent, prefetched, query_error)
    output = {
        "event": _brief_event(event),
        "passive_pre_execution_context_mode": (
            "selected_symbol"
            if timings.get("selected_symbol_position_context_calls")
            else "full_positions"
            if timings.get("position_context_calls")
            else "none"
        ),
        "local_relevance": local_relevance,
        "summary": summary,
        "timings": timings,
        "engine_debug_excerpt": {
            "passive_step1_prefetched": (getattr(agent.engine, "last_call_debug", {}) or {}).get("passive_step1_prefetched"),
            "passive_step2_executed": (getattr(agent.engine, "last_call_debug", {}) or {}).get("passive_step2_executed"),
            "passive_chart_context_prefetch": (getattr(agent.engine, "last_call_debug", {}) or {}).get("passive_chart_context_prefetch"),
            "selected_trade_symbol_context": (getattr(agent.engine, "last_call_debug", {}) or {}).get("selected_trade_symbol_context"),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=_json_default), encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2, default=_json_default))
    print(f"[dry_run_timing_output] {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
