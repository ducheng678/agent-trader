from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from watchers.common import OUTPUT_ROOT
from soak_all_watchers import default_output_path, probe_watcher, run_soak
from watch_free_sources_modular import build_watchers


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def default_gate_report_path(now: datetime | None = None) -> Path:
    stamp = (now or utc_now()).strftime("%Y%m%dT%H%M%SZ")
    return OUTPUT_ROOT / f"gate_live_report_{stamp}.json"


def default_live_log_path(now: datetime | None = None) -> Path:
    stamp = (now or utc_now()).strftime("%Y%m%dT%H%M%SZ")
    return OUTPUT_ROOT / f"watch_live_{stamp}.log"


def count_fail_rounds(summary: dict) -> int:
    return sum(item.get("fail_rounds", 0) for item in summary.get("sources", {}).values())


def verify_latest_round_against_live(
    latest_round: dict,
    *,
    probe_fn: Callable[[object], object] = probe_watcher,
) -> dict:
    expected = {
        result["source"]: {
            "sample_title": result["sample_title"],
            "item_count": result["item_count"],
        }
        for result in latest_round.get("results", [])
    }
    mismatches: list[dict] = []
    errors: list[dict] = []
    checked_sources = 0

    for watcher in build_watchers():
        try:
            result = probe_fn(watcher)
            checked_sources += 1
            source = result.source
            expected_entry = expected.get(source, {"sample_title": "", "item_count": None})
            title_match = expected_entry["sample_title"] == result.sample_title
            count_match = expected_entry["item_count"] == result.item_count
            if not title_match or not count_match:
                mismatches.append(
                    {
                        "source": source,
                        "expected_title": expected_entry["sample_title"],
                        "actual_title": result.sample_title,
                        "expected_count": expected_entry["item_count"],
                        "actual_count": result.item_count,
                        "title_match": title_match,
                        "count_match": count_match,
                    }
                )
        except Exception as exc:
            errors.append({"source": getattr(watcher, "source_name", "unknown"), "error": str(exc)})

    return {
        "checked_sources": checked_sources,
        "mismatch_count": len(mismatches),
        "error_count": len(errors),
        "mismatches": mismatches,
        "errors": errors,
    }


def should_start_live(soak_summary: dict, authenticity_report: dict) -> bool:
    return (
        count_fail_rounds(soak_summary) == 0
        and authenticity_report.get("mismatch_count", 0) == 0
        and authenticity_report.get("error_count", 0) == 0
    )


def start_live_watcher(log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handle = log_path.open("a", encoding="utf-8")
    process = subprocess.Popen(
        [sys.executable, "-u", "watch_free_sources_modular.py"],
        cwd=Path(__file__).resolve().parent,
        stdout=handle,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process.pid


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run watcher soak gate, authenticity check, then launch live watcher")
    parser.add_argument("--rounds", type=int, default=30, help="Number of soak rounds before authenticity check")
    parser.add_argument("--interval-seconds", type=int, default=60, help="Sleep between soak rounds")
    parser.add_argument("--soak-output", default="", help="Optional soak JSONL path")
    parser.add_argument("--gate-report", default="", help="Optional gate report JSON path")
    parser.add_argument("--live-log", default="", help="Optional live watcher log path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    now = utc_now()
    soak_output = Path(args.soak_output) if args.soak_output else default_output_path(now)
    gate_report_path = Path(args.gate_report) if args.gate_report else default_gate_report_path(now)
    live_log_path = Path(args.live_log) if args.live_log else default_live_log_path(now)

    watchers = build_watchers()
    print(
        f"[gate-start] watcher_count={len(watchers)} rounds={args.rounds} "
        f"interval_seconds={args.interval_seconds} soak_output={soak_output}"
    )
    history, summary = run_soak(
        watchers=watchers,
        rounds=args.rounds,
        interval_seconds=args.interval_seconds,
        output_path=soak_output,
    )

    latest_round = history[-1]
    authenticity_report = verify_latest_round_against_live(latest_round)
    live_pid: int | None = None
    if should_start_live(summary, authenticity_report):
        live_pid = start_live_watcher(live_log_path)
        print(f"[gate-live-started] pid={live_pid} live_log={live_log_path}")
    else:
        print(
            f"[gate-live-skipped] fail_rounds={count_fail_rounds(summary)} "
            f"mismatch_count={authenticity_report['mismatch_count']} "
            f"error_count={authenticity_report['error_count']}"
        )

    gate_report = {
        "type": "gate_report",
        "finished_at": utc_now().isoformat(),
        "soak_output": str(soak_output),
        "rounds": args.rounds,
        "interval_seconds": args.interval_seconds,
        "soak_summary": summary,
        "authenticity": authenticity_report,
        "live_started": live_pid is not None,
        "live_pid": live_pid,
        "live_log": str(live_log_path) if live_pid is not None else "",
    }
    gate_report_path.parent.mkdir(parents=True, exist_ok=True)
    gate_report_path.write_text(json.dumps(gate_report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[gate-report] {gate_report_path}")
    return 0 if live_pid is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
