#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import unified_market_agent as uma


TARGET_TITLE = "Oil Prices Soar as Iran Standoff Shows No End in Sight"
TARGET_SYMBOL = "BRENTOIL-USDC"


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_no, raw_line in enumerate(handle, 1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except Exception:
                continue
            if isinstance(payload, dict):
                yield line_no, payload


def _event_ts(event: Dict[str, Any]) -> str:
    return str(
        event.get("event_timestamp")
        or event.get("published_at")
        or event.get("seen_at")
        or ""
    )


def _find_original_query_and_debug(audit_path: Path, title: str) -> Tuple[Dict[str, Any], Dict[str, Any], int, int]:
    query_payload: Optional[Dict[str, Any]] = None
    query_line = 0
    for line_no, payload in _load_jsonl(audit_path):
        if payload.get("event") != "playbook_query_requested":
            continue
        body = payload.get("payload") or {}
        trigger = body.get("trigger_event") or {}
        if str(trigger.get("title") or "") == title:
            query_payload = body
            query_line = line_no
            break
    if not query_payload:
        raise RuntimeError(f"Could not find playbook_query_requested for title={title!r}")

    debug_payload: Optional[Dict[str, Any]] = None
    debug_line = 0
    for line_no, payload in _load_jsonl(audit_path):
        if line_no <= query_line:
            continue
        if payload.get("event") != "llm_call_debug":
            continue
        body = payload.get("payload") or {}
        if body.get("trigger_reason") != "passive_event_trigger":
            continue
        debug_payload = body
        debug_line = line_no
        break
    if not debug_payload:
        raise RuntimeError("Could not find following passive llm_call_debug")
    return query_payload, debug_payload, query_line, debug_line


def _helper_recent_events_before(
    helper_path: Path,
    *,
    symbol: str,
    before_ts: str,
    limit: int,
) -> List[Dict[str, Any]]:
    before_dt = uma.parse_utc_iso(before_ts)
    if before_dt is None:
        raise RuntimeError(f"Invalid trigger timestamp: {before_ts}")
    rows: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str, str, str]] = set()
    for _, payload in _load_jsonl(helper_path):
        raw_events = payload.get("materially_new_first_events")
        if not isinstance(raw_events, list):
            buckets = payload.get("materially_new_first_events_by_instrument") or {}
            raw_events = list(buckets.get(symbol) or [])
        for event in list(raw_events or []):
            if not isinstance(event, dict):
                continue
            normalized = uma.normalize_event_record(dict(event))
            event_dt = uma.parse_utc_iso(_event_ts(normalized))
            if event_dt is None or event_dt >= before_dt:
                continue
            key = (
                str(normalized.get("source") or ""),
                str(normalized.get("item_id") or ""),
                str(normalized.get("title") or ""),
                str(normalized.get("event_timestamp") or ""),
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(normalized)
    rows.sort(key=lambda item: (_event_ts(item), str(item.get("title") or "")))
    return rows[-limit:]


def run(args: argparse.Namespace) -> Dict[str, Any]:
    query_payload, debug_payload, query_line, debug_line = _find_original_query_and_debug(
        args.audit_path,
        args.title,
    )
    trigger_event = dict(query_payload.get("trigger_event") or {})
    if not trigger_event:
        raise RuntimeError("Original query has no trigger_event")
    trade_symbol_context = dict(query_payload.get("trade_symbol_context") or {})
    if not trade_symbol_context:
        trade_symbol_context = dict(debug_payload.get("trade_symbol_context") or {})
    if not trade_symbol_context:
        candidate_contexts = [
            dict(item)
            for item in list(query_payload.get("candidate_contexts") or [])
            if isinstance(item, dict)
        ]
        if not candidate_contexts:
            selected_context = dict(debug_payload.get("selected_trade_symbol_context") or {})
            if selected_context:
                candidate_contexts = [selected_context]
        if not candidate_contexts:
            raise RuntimeError("No trade symbol context found")
        trade_symbol_context = candidate_contexts[0]
    market_mainline_context = dict(debug_payload.get("market_mainline_context") or {})

    recent_events = _helper_recent_events_before(
        args.helper_materiality_path,
        symbol=args.symbol,
        before_ts=_event_ts(trigger_event),
        limit=args.recent_limit,
    )

    engine = uma.DiscretionaryLLMEngine()
    display_name = str(trade_symbol_context.get("display_name") or args.symbol)
    execution_symbol = str(trade_symbol_context.get("execution_symbol") or args.symbol)
    search_mode = engine._resolve_search_mode("passive_event_trigger")
    phase = "context_only"
    if search_mode == "off":
        phase = "fast"
    elif search_mode == "always":
        phase = "verified"

    playbook, call_debug = engine._call_passive_two_step_model(
        phase=phase,
        trigger_event=trigger_event,
        recent_events=recent_events,
        trade_symbol_context=trade_symbol_context,
        active_symbol=execution_symbol,
        market_mainline_context=market_mainline_context,
        chart_input_context=None,
        reasoning_effort=engine._resolve_reasoning_effort("passive_event_trigger"),
        trade_symbol_label=display_name,
    )
    playbook.selected_symbol = display_name
    capped = engine._cap_playbook(playbook)
    summary = {
        "audit_query_line": query_line,
        "audit_debug_line": debug_line,
        "trigger_title": trigger_event.get("title"),
        "trigger_timestamp": _event_ts(trigger_event),
        "recent_limit": args.recent_limit,
        "recent_event_count": len(recent_events),
        "recent_event_titles": [
            {
                "event_timestamp": _event_ts(item),
                "source": item.get("source"),
                "title": item.get("title"),
            }
            for item in recent_events
        ],
        "has_wsj_extended_blockade": any(
            str(item.get("title") or "") == "Trump Tells Aides to Prepare for Extended Blockade of Iran"
            for item in recent_events
        ),
        "trigger_event_relevance": capped.trigger_event_relevance,
        "trigger_confidence_raw": capped.trigger_confidence_raw,
        "trigger_confidence_normalized": capped.trigger_confidence,
        "selected_symbol": capped.selected_symbol,
        "entry_action": capped.entry_plan.action_decision.action,
        "response_model": call_debug.get("response_model"),
        "usage_cost": call_debug.get("usage_cost"),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(
            {
                "summary": summary,
                "trigger_event": trigger_event,
                "recent_events": recent_events,
                "market_mainline_context": market_mainline_context,
                "validated_playbook": playbook.to_dict(),
                "capped_playbook": capped.to_dict(),
                "llm_call_debug": call_debug,
            },
            ensure_ascii=False,
            indent=2,
            default=_json_default,
        ),
        encoding="utf-8",
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay the NYT Brent passive trigger with helper materially-new recent events.")
    parser.add_argument("--title", default=TARGET_TITLE)
    parser.add_argument("--symbol", default=TARGET_SYMBOL)
    parser.add_argument("--recent-limit", type=int, default=20)
    parser.add_argument("--audit-path", type=Path, default=Path("logs/unified_market_agent_audit.jsonl"))
    parser.add_argument("--helper-materiality-path", type=Path, default=Path("logs/helper_materially_new_first_events.jsonl"))
    parser.add_argument("--output-path", type=Path, default=Path("tmp_bad_case_queries/replay_nyt_passive_limit20/result.json"))
    return parser.parse_args()


def main() -> None:
    summary = run(parse_args())
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
