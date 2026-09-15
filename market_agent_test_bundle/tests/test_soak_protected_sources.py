from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from probe_protected_sources import ProbeResult
from soak_protected_sources import build_summary, compute_rounds, run_soak


def test_compute_rounds_defaults_to_two_hours_at_sixty_seconds():
    rounds = compute_rounds(rounds=None, duration_hours=2.0, interval_seconds=60)

    assert rounds == 120


def test_run_soak_writes_rounds_and_summary(tmp_path):
    output_path = tmp_path / "soak.jsonl"
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

    def fake_run_probes(names):
        assert names == ["irna", "bloomberg"]
        return [
            ProbeResult(source="irna", target="https://en.irna.ir/", enabled=True, has_cookie=False, ok=True, item_count=20),
            ProbeResult(source="bloomberg", target="https://www.bloomberg.com/markets/economics", enabled=True, has_cookie=False, ok=False, error="403"),
        ]

    history, summary = run_soak(
        sources=["irna", "bloomberg"],
        rounds=2,
        interval_seconds=60,
        output_path=output_path,
        run_probes=fake_run_probes,
        sleep_fn=sleeps.append,
        now_fn=fake_now,
    )

    assert len(history) == 2
    assert sleeps == [60]
    assert summary["rounds"] == 2
    assert summary["sources"]["irna"]["ok_rounds"] == 2
    assert summary["sources"]["bloomberg"]["fail_rounds"] == 2

    lines = [json.loads(line) for line in output_path.read_text(encoding="utf-8").splitlines()]
    assert [line["type"] for line in lines] == ["round", "round", "summary"]
    assert lines[-1]["sources"]["bloomberg"]["last_error"] == "403"


def test_build_summary_tracks_last_item_count_and_failures():
    history = [
        {
            "type": "round",
            "round": 1,
            "started_at": "2026-04-01T00:00:00+00:00",
            "results": [
                {"source": "dol", "ok": True, "item_count": 20, "has_cookie": False, "error": ""},
                {"source": "bls", "ok": False, "item_count": 0, "has_cookie": False, "error": "timeout"},
            ],
        },
        {
            "type": "round",
            "round": 2,
            "started_at": "2026-04-01T00:01:00+00:00",
            "results": [
                {"source": "dol", "ok": True, "item_count": 21, "has_cookie": False, "error": ""},
                {"source": "bls", "ok": True, "item_count": 1, "has_cookie": False, "error": ""},
            ],
        },
    ]

    summary = build_summary(history, finished_at="2026-04-01T00:02:00+00:00")

    assert summary["sources"]["dol"]["last_item_count"] == 21
    assert summary["sources"]["dol"]["ok_rounds"] == 2
    assert summary["sources"]["bls"]["fail_rounds"] == 1
    assert summary["sources"]["bls"]["last_item_count"] == 1
