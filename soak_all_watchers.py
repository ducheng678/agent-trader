from __future__ import annotations

import argparse
import json
import math
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from watchers.common import OUTPUT_ROOT
from watch_free_sources_modular import build_watchers


@dataclass
class WatcherProbeResult:
    source: str
    ok: bool
    item_count: int = 0
    sample_title: str = ""
    error: str = ""


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def compute_rounds(*, rounds: int | None, duration_hours: float, interval_seconds: int) -> int:
    if rounds is not None:
        if rounds <= 0:
            raise ValueError("--rounds must be > 0")
        return rounds
    if duration_hours <= 0:
        raise ValueError("--duration-hours must be > 0")
    if interval_seconds <= 0:
        raise ValueError("--interval-seconds must be > 0")
    return max(1, math.ceil((duration_hours * 3600) / interval_seconds))


def default_output_path(now: datetime | None = None) -> Path:
    stamp = (now or utc_now()).strftime("%Y%m%dT%H%M%SZ")
    return OUTPUT_ROOT / f"all_watchers_soak_{stamp}.jsonl"


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def probe_watcher(watcher) -> WatcherProbeResult:
    try:
        if hasattr(watcher, "_safe_fetch_statuses"):
            posts = watcher._safe_fetch_statuses()
            originals = [post for post in posts if not watcher._is_repost(post)]
            sample_title = watcher._summarize_post_line(originals[-1]) if originals else ""
            return WatcherProbeResult(source=watcher.source_name, ok=True, item_count=len(originals), sample_title=sample_title)
        if hasattr(watcher, "_fetch_events"):
            items = watcher._fetch_events()
            sample_title = items[0].title if items else ""
            return WatcherProbeResult(source=watcher.source_name, ok=True, item_count=len(items), sample_title=sample_title)
        html = watcher._fetch_html()
        items = watcher.extract_events(html)
        sample_title = items[0].title if items else ""
        return WatcherProbeResult(source=watcher.source_name, ok=True, item_count=len(items), sample_title=sample_title)
    except Exception as e:
        return WatcherProbeResult(source=watcher.source_name, ok=False, error=str(e))


def build_summary(history: list[dict], *, finished_at: str) -> dict:
    per_source: dict[str, dict] = {}
    for round_entry in history:
        for result in round_entry["results"]:
            source = result["source"]
            metrics = per_source.setdefault(
                source,
                {
                    "ok_rounds": 0,
                    "fail_rounds": 0,
                    "last_item_count": None,
                    "last_error": "",
                },
            )
            if result["ok"]:
                metrics["ok_rounds"] += 1
                metrics["last_item_count"] = result["item_count"]
            else:
                metrics["fail_rounds"] += 1
                metrics["last_error"] = result["error"]
    return {
        "type": "summary",
        "finished_at": finished_at,
        "rounds": len(history),
        "sources": per_source,
    }


def run_soak(
    *,
    watchers: list,
    rounds: int,
    interval_seconds: int,
    output_path: Path,
    probe_fn: Callable[[object], WatcherProbeResult] = probe_watcher,
    sleep_fn: Callable[[float], None] = time.sleep,
    now_fn: Callable[[], datetime] = utc_now,
) -> tuple[list[dict], dict]:
    history: list[dict] = []
    for index in range(rounds):
        started_at = now_fn().isoformat()
        results = [probe_fn(watcher) for watcher in watchers]
        round_payload = {
            "type": "round",
            "round": index + 1,
            "started_at": started_at,
            "results": [asdict(item) for item in results],
        }
        append_jsonl(output_path, round_payload)
        history.append(round_payload)

        ok_count = sum(1 for item in results if item.ok)
        fail_count = len(results) - ok_count
        print(
            f"[round {index + 1}/{rounds}] ok={ok_count} fail={fail_count} "
            f"timestamp={started_at} log={output_path}"
        )
        for item in results:
            if item.ok:
                continue
            print(f"  fail source={item.source} error={item.error}")

        if index < rounds - 1:
            sleep_fn(interval_seconds)

    summary = build_summary(history, finished_at=now_fn().isoformat())
    append_jsonl(output_path, summary)
    return history, summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Long-running soak probe for all enabled watchers")
    parser.add_argument("--interval-seconds", type=int, default=60, help="Sleep between rounds")
    parser.add_argument("--duration-hours", type=float, default=2.0, help="Target runtime in hours when --rounds is omitted")
    parser.add_argument("--rounds", type=int, default=None, help="Explicit round count; overrides --duration-hours")
    parser.add_argument("--output", default="", help="Optional JSONL output path")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    watchers = build_watchers()
    rounds = compute_rounds(rounds=args.rounds, duration_hours=args.duration_hours, interval_seconds=args.interval_seconds)
    output_path = Path(args.output) if args.output else default_output_path()
    print(
        f"[startup] watcher_count={len(watchers)} rounds={rounds} interval_seconds={args.interval_seconds} "
        f"output={output_path}"
    )
    _, summary = run_soak(
        watchers=watchers,
        rounds=rounds,
        interval_seconds=args.interval_seconds,
        output_path=output_path,
    )
    fail_rounds = sum(item["fail_rounds"] for item in summary["sources"].values())
    print(f"[done] rounds={summary['rounds']} fail_rounds={fail_rounds} output={output_path}")
    return 0 if fail_rounds == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
