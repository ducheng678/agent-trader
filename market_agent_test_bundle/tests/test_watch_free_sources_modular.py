from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from watchers.common import Event, StateStore, apply_warmup_mode
from watch_free_sources_modular import get_poll_timeout_seconds, process_poll_cycle, warmup_watchers


class DummyWarmupWatcher:
    def __init__(self, source_name: str, items: list[Event], cursor_value: str | None = None):
        self.source_name = source_name
        self.interval_seconds = 60
        self.last_poll_at = 0.0
        self.items = items
        self.cursor_value = cursor_value

    def warmup(self, state: StateStore) -> list[Event]:
        return apply_warmup_mode(
            self.source_name,
            state,
            [item.item_id for item in self.items],
            emitted_events=list(reversed(self.items)),
            cursor_value=self.cursor_value,
        )

    def should_poll(self, ts: float) -> bool:
        return False

    def poll(self, state: StateStore) -> list[Event]:
        raise AssertionError("poll should not be called in warmup tests")


class SleepPollWatcher:
    def __init__(self, source_name: str, sleep_seconds: float, start_times: list[float], *, event_id: str = "evt"):
        self.source_name = source_name
        self.interval_seconds = 999
        self.last_poll_at = 0.0
        self.sleep_seconds = sleep_seconds
        self.start_times = start_times
        self.event_id = event_id

    def warmup(self, state: StateStore) -> list[Event]:
        return []

    def should_poll(self, ts: float) -> bool:
        return (ts - self.last_poll_at) >= self.interval_seconds

    def poll(self, state: StateStore) -> list[Event]:
        self.start_times.append(time.monotonic())
        time.sleep(self.sleep_seconds)
        self.last_poll_at = time.time()
        if state.is_seen(self.source_name, self.event_id):
            return []
        state.mark_seen(self.source_name, self.event_id)
        return [
            Event(
                source=self.source_name,
                item_id=self.event_id,
                title=f"title-{self.source_name}",
                url=f"https://example.test/{self.source_name}",
            )
        ]


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def test_warmup_mark_seen_marks_state_without_emitting(tmp_path, monkeypatch):
    monkeypatch.setenv("WATCH_WARMUP_MODE", "mark_seen")
    state = StateStore(tmp_path / "state.json")
    from watchers.common import JsonlWriter

    writer = JsonlWriter(tmp_path / "events.jsonl")
    watcher = DummyWarmupWatcher(
        "dummy",
        [
            Event(source="dummy", item_id="2", title="two", url="https://example.test/2"),
            Event(source="dummy", item_id="1", title="one", url="https://example.test/1"),
        ],
        cursor_value="2",
    )

    warmup_watchers([watcher], state, writer)

    assert state.is_seen("dummy", "1")
    assert state.is_seen("dummy", "2")
    assert state.get_cursor("dummy") == "2"
    assert _read_jsonl(tmp_path / "events.jsonl") == []


def test_warmup_emit_recent_emits_in_chronological_order(tmp_path, monkeypatch):
    monkeypatch.setenv("WATCH_WARMUP_MODE", "emit_recent")
    state = StateStore(tmp_path / "state.json")
    from watchers.common import JsonlWriter

    writer = JsonlWriter(tmp_path / "events.jsonl")
    watcher = DummyWarmupWatcher(
        "dummy",
        [
            Event(source="dummy", item_id="2", title="two", url="https://example.test/2"),
            Event(source="dummy", item_id="1", title="one", url="https://example.test/1"),
        ],
        cursor_value="2",
    )

    warmup_watchers([watcher], state, writer)

    assert [item["item_id"] for item in _read_jsonl(tmp_path / "events.jsonl")] == ["1", "2"]
    assert state.is_seen("dummy", "1")
    assert state.is_seen("dummy", "2")
    assert state.get_cursor("dummy") == "2"


def test_warmup_default_emits_recent_without_rewriting_existing_events(tmp_path, monkeypatch):
    monkeypatch.delenv("WATCH_WARMUP_MODE", raising=False)
    state = StateStore(tmp_path / "state.json")
    from watchers.common import JsonlWriter

    writer = JsonlWriter(tmp_path / "events.jsonl")
    assert writer.append(Event(source="dummy", item_id="1", title="one", url="https://example.test/1"))
    watcher = DummyWarmupWatcher(
        "dummy",
        [
            Event(source="dummy", item_id="2", title="two", url="https://example.test/2"),
            Event(source="dummy", item_id="1", title="one", url="https://example.test/1"),
        ],
        cursor_value="2",
    )

    warmup_watchers([watcher], state, writer)

    assert [item["item_id"] for item in _read_jsonl(tmp_path / "events.jsonl")] == ["1", "2"]
    assert state.is_seen("dummy", "1")
    assert state.is_seen("dummy", "2")
    assert state.get_cursor("dummy") == "2"


def test_warmup_cursor_only_sets_cursor_without_marking_seen(tmp_path, monkeypatch):
    monkeypatch.setenv("WATCH_WARMUP_MODE", "cursor_only")
    state = StateStore(tmp_path / "state.json")
    from watchers.common import JsonlWriter

    writer = JsonlWriter(tmp_path / "events.jsonl")
    watcher = DummyWarmupWatcher(
        "dummy",
        [
            Event(source="dummy", item_id="2", title="two", url="https://example.test/2"),
            Event(source="dummy", item_id="1", title="one", url="https://example.test/1"),
        ],
        cursor_value="2",
    )

    warmup_watchers([watcher], state, writer)

    assert state.get_cursor("dummy") == "2"
    assert not state.is_seen("dummy", "1")
    assert not state.is_seen("dummy", "2")
    assert _read_jsonl(tmp_path / "events.jsonl") == []


def test_process_poll_cycle_runs_ready_watchers_concurrently(tmp_path, monkeypatch):
    monkeypatch.setenv("WATCH_POLL_TIMEOUT_SECONDS", "1")
    state = StateStore(tmp_path / "state.json")
    from watchers.common import JsonlWriter

    writer = JsonlWriter(tmp_path / "events.jsonl")
    start_times: list[float] = []
    watch_list = [
        SleepPollWatcher("slow_a", 0.15, start_times, event_id="a"),
        SleepPollWatcher("slow_b", 0.15, start_times, event_id="b"),
    ]
    inflight = {}

    with ThreadPoolExecutor(max_workers=2) as executor:
        process_poll_cycle(watch_list, state, writer, executor, inflight, now_ts=time.time(), now_monotonic=time.monotonic())
        time.sleep(0.22)
        process_poll_cycle(watch_list, state, writer, executor, inflight, now_ts=time.time(), now_monotonic=time.monotonic())

    assert len(start_times) == 2
    assert max(start_times) - min(start_times) < 0.08
    assert {item["item_id"] for item in _read_jsonl(tmp_path / "events.jsonl")} == {"a", "b"}
    assert not inflight


def test_process_poll_cycle_discards_results_that_finish_after_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("WATCH_POLL_TIMEOUT_SECONDS", "0.05")
    state = StateStore(tmp_path / "state.json")
    from watchers.common import JsonlWriter

    writer = JsonlWriter(tmp_path / "events.jsonl")
    start_times: list[float] = []
    watch_list = [SleepPollWatcher("too_slow", 0.20, start_times, event_id="late")]
    inflight = {}

    with ThreadPoolExecutor(max_workers=1) as executor:
        process_poll_cycle(watch_list, state, writer, executor, inflight, now_ts=time.time(), now_monotonic=time.monotonic())
        time.sleep(0.08)
        process_poll_cycle(watch_list, state, writer, executor, inflight, now_ts=time.time(), now_monotonic=time.monotonic())
        time.sleep(0.20)
        process_poll_cycle(watch_list, state, writer, executor, inflight, now_ts=time.time(), now_monotonic=time.monotonic())

    assert len(start_times) == 1
    assert _read_jsonl(tmp_path / "events.jsonl") == []
    assert not state.is_seen("too_slow", "late")
    assert not inflight


def test_process_poll_cycle_does_not_reappend_event_already_in_log(tmp_path, monkeypatch):
    monkeypatch.setenv("WATCH_POLL_TIMEOUT_SECONDS", "1")
    state = StateStore(tmp_path / "state.json")
    from watchers.common import JsonlWriter

    writer = JsonlWriter(tmp_path / "events.jsonl")
    assert writer.append(
        Event(
            source="dup_source",
            item_id="dup",
            title="title-dup_source",
            url="https://example.test/dup_source",
        )
    )
    start_times: list[float] = []
    watch_list = [SleepPollWatcher("dup_source", 0.01, start_times, event_id="dup")]
    inflight = {}

    with ThreadPoolExecutor(max_workers=1) as executor:
        process_poll_cycle(watch_list, state, writer, executor, inflight, now_ts=time.time(), now_monotonic=time.monotonic())
        time.sleep(0.05)
        process_poll_cycle(watch_list, state, writer, executor, inflight, now_ts=time.time(), now_monotonic=time.monotonic())

    assert [item["item_id"] for item in _read_jsonl(tmp_path / "events.jsonl")] == ["dup"]
    assert state.is_seen("dup_source", "dup")
    assert not inflight


def test_get_poll_timeout_seconds_honors_source_specific_override(monkeypatch):
    class Watcher:
        source_name = "aaa"

    monkeypatch.setenv("WATCH_POLL_TIMEOUT_SECONDS", "30")
    monkeypatch.setenv("AAA_POLL_TIMEOUT_SECONDS", "7")

    assert get_poll_timeout_seconds(Watcher()) == 7.0
