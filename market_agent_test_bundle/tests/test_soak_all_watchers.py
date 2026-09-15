from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from soak_all_watchers import WatcherProbeResult, build_summary, compute_rounds, probe_watcher, run_soak


def test_compute_rounds_defaults_to_two_hours_at_sixty_seconds():
    assert compute_rounds(rounds=None, duration_hours=2.0, interval_seconds=60) == 120


def test_probe_watcher_handles_truth_event_and_html_watchers():
    class TruthWatcher:
        source_name = "truth_social:test"

        def _safe_fetch_statuses(self):
            return [{"id": "1"}, {"id": "2", "reblog": {"id": "x"}}]

        def _is_repost(self, post):
            return bool(post.get("reblog"))

        def _summarize_post_line(self, post):
            return f"post-{post['id']}"

    class EventItem:
        def __init__(self, title: str):
            self.title = title

    class EventWatcher:
        source_name = "bloomberg"

        def _fetch_events(self):
            return [EventItem("headline")]

    class HtmlItem:
        def __init__(self, title: str):
            self.title = title

    class HtmlWatcher:
        source_name = "white_house"

        def _fetch_html(self):
            return "<html></html>"

        def extract_events(self, html):
            assert html == "<html></html>"
            return [HtmlItem("fact sheet")]

    assert probe_watcher(TruthWatcher()).item_count == 1
    assert probe_watcher(EventWatcher()).sample_title == "headline"
    assert probe_watcher(HtmlWatcher()).sample_title == "fact sheet"


def test_run_soak_writes_rounds_and_summary(tmp_path):
    output_path = tmp_path / "all_soak.jsonl"
    sleeps: list[float] = []
    timestamps = iter(
        [
            datetime(2026, 4, 1, 0, 0, 0, tzinfo=timezone.utc),
            datetime(2026, 4, 1, 0, 1, 0, tzinfo=timezone.utc),
            datetime(2026, 4, 1, 0, 2, 0, tzinfo=timezone.utc),
        ]
    )

    def fake_now():
        return next(timestamps)

    class DummyWatcher:
        def __init__(self, source_name: str):
            self.source_name = source_name

    def fake_probe(watcher):
        if watcher.source_name == "bad":
            return WatcherProbeResult(source="bad", ok=False, error="403")
        return WatcherProbeResult(source=watcher.source_name, ok=True, item_count=3, sample_title="ok")

    history, summary = run_soak(
        watchers=[DummyWatcher("good"), DummyWatcher("bad")],
        rounds=2,
        interval_seconds=60,
        output_path=output_path,
        probe_fn=fake_probe,
        sleep_fn=sleeps.append,
        now_fn=fake_now,
    )

    assert len(history) == 2
    assert sleeps == [60]
    assert summary["sources"]["good"]["ok_rounds"] == 2
    assert summary["sources"]["bad"]["fail_rounds"] == 2

    lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [line["type"] for line in lines] == ["round", "round", "summary"]


def test_build_summary_tracks_latest_status():
    history = [
        {
            "type": "round",
            "round": 1,
            "started_at": "2026-04-01T00:00:00+00:00",
            "results": [
                {"source": "fed", "ok": True, "item_count": 20, "sample_title": "a", "error": ""},
                {"source": "sec", "ok": False, "item_count": 0, "sample_title": "", "error": "timeout"},
            ],
        },
        {
            "type": "round",
            "round": 2,
            "started_at": "2026-04-01T00:01:00+00:00",
            "results": [
                {"source": "fed", "ok": True, "item_count": 21, "sample_title": "b", "error": ""},
                {"source": "sec", "ok": True, "item_count": 25, "sample_title": "c", "error": ""},
            ],
        },
    ]

    summary = build_summary(history, finished_at="2026-04-01T00:02:00+00:00")

    assert summary["sources"]["fed"]["last_item_count"] == 21
    assert summary["sources"]["sec"]["fail_rounds"] == 1
    assert summary["sources"]["sec"]["last_item_count"] == 25
