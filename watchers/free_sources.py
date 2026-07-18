import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from .aaa_watcher import build_aaa_watchers
from .adp_watcher import build_adp_watchers
from .axios_watcher import build_axios_watchers
from .bls_watcher import build_bls_watchers
from .bloomberg_watcher import build_bloomberg_watchers
from .census_retail_watcher import build_census_retail_watchers
from .coindesk_watcher import build_coindesk_watchers
from .common import EVENTS_PATH, OUTPUT_ROOT, REQUEST_TIMEOUT, STATE_PATH, JsonlWriter, StateStore, now_iso, poll_deadline, print_event
from .conference_board_watcher import build_conference_board_watchers
from .dol_watcher import build_dol_watchers
from .factbase_watcher import build_factbase_watchers
from .fars_watcher import build_fars_watchers
from .fed_watcher import build_fed_watchers
from .irna_watcher import build_irna_watchers
from .mni_watcher import build_mni_watchers
from .nyt_watcher import build_nyt_watchers
from .reuters_watcher import build_reuters_watchers
from .sec_watcher import build_sec_watchers
from .spglobal_pmi_watcher import build_spglobal_pmi_watchers
from .schwab_watcher import build_schwab_watchers
from .treasury_watcher import build_treasury_watchers
from .truth_social_watcher import build_truth_social_watchers
from .white_house_watcher import build_white_house_watchers
from .wsj_watcher import build_wsj_watchers

def _build_watchers_safe(label: str, builder):
    try:
        return builder()
    except Exception as e:
        print(f"[ERROR] watcher factory failed for {label}: {e}")
        return []


def build_watchers():
    return [
        *_build_watchers_safe("truth_social", build_truth_social_watchers),
        *_build_watchers_safe("white_house", build_white_house_watchers),
        *_build_watchers_safe("aaa", build_aaa_watchers),
        *_build_watchers_safe("adp", build_adp_watchers),
        *_build_watchers_safe("fed", build_fed_watchers),
        *_build_watchers_safe("sec", build_sec_watchers),
        *_build_watchers_safe("dol", build_dol_watchers),
        *_build_watchers_safe("bls", build_bls_watchers),
        *_build_watchers_safe("census_retail", build_census_retail_watchers),
        *_build_watchers_safe("treasury", build_treasury_watchers),
        *_build_watchers_safe("conference_board", build_conference_board_watchers),
        *_build_watchers_safe("spglobal_pmi", build_spglobal_pmi_watchers),
        *_build_watchers_safe("schwab", build_schwab_watchers),
        *_build_watchers_safe("coindesk", build_coindesk_watchers),
        *_build_watchers_safe("axios", build_axios_watchers),
        *_build_watchers_safe("factbase", build_factbase_watchers),
        *_build_watchers_safe("mni", build_mni_watchers),
        *_build_watchers_safe("fars", build_fars_watchers),
        *_build_watchers_safe("irna", build_irna_watchers),
        *_build_watchers_safe("bloomberg", build_bloomberg_watchers),
        *_build_watchers_safe("reuters", build_reuters_watchers),
        *_build_watchers_safe("nyt", build_nyt_watchers),
        *_build_watchers_safe("wsj", build_wsj_watchers),
    ]


@dataclass
class PollOutcome:
    events: list[Any]
    source_state: dict[str, Any]


@dataclass
class PollTask:
    watcher: Any
    future: Future
    started_monotonic: float
    timeout_seconds: float
    timed_out: bool = False


def _source_timeout_env_key(source_name: str) -> str:
    base = source_name.split(":", 1)[0]
    return f"{re.sub(r'[^A-Za-z0-9]+', '_', base).upper()}_POLL_TIMEOUT_SECONDS"


def get_poll_timeout_seconds(watcher: Any) -> float:
    source_key = _source_timeout_env_key(watcher.source_name)
    raw_value = os.getenv(source_key, os.getenv("WATCH_POLL_TIMEOUT_SECONDS", str(REQUEST_TIMEOUT)))
    timeout_seconds = float(raw_value)
    if timeout_seconds <= 0:
        raise ValueError(f"Poll timeout for {watcher.source_name} must be > 0 seconds")
    return timeout_seconds


def get_max_poll_workers(watcher_count: int) -> int:
    configured = int(os.getenv("WATCH_MAX_POLL_WORKERS", "0") or "0")
    if configured > 0:
        return max(1, configured)
    return max(1, watcher_count)


def warmup_watchers(watch_list, state: StateStore, writer: JsonlWriter) -> None:
    for watcher in watch_list:
        try:
            warmup_events = watcher.warmup(state) or []
            for event in warmup_events:
                if writer.append(event):
                    print_event(event)
            state.save()
        except Exception as e:
            print(f"[ERROR] warmup failed for {watcher.source_name}: {e}")


def _run_poll_once(watcher: Any, state: StateStore, timeout_seconds: float) -> PollOutcome:
    local_state = state.fork()
    with poll_deadline(timeout_seconds):
        events = watcher.poll(local_state)
    return PollOutcome(events=events, source_state=local_state.export_source_state(watcher.source_name))


def _submit_ready_poll_tasks(watch_list, state: StateStore, executor: ThreadPoolExecutor, inflight: dict[str, PollTask], *, now_ts: float, now_monotonic: float) -> None:
    for watcher in watch_list:
        if watcher.source_name in inflight:
            continue
        if not watcher.should_poll(now_ts):
            continue
        timeout_seconds = get_poll_timeout_seconds(watcher)
        future = executor.submit(_run_poll_once, watcher, state, timeout_seconds)
        inflight[watcher.source_name] = PollTask(
            watcher=watcher,
            future=future,
            started_monotonic=now_monotonic,
            timeout_seconds=timeout_seconds,
        )


def _finalize_poll_task(task: PollTask, state: StateStore, writer: JsonlWriter) -> None:
    if task.timed_out:
        try:
            task.future.result()
        except Exception:
            pass
        return

    try:
        outcome = task.future.result()
    except Exception as e:
        task.watcher.last_poll_at = time.time()
        print(f"[ERROR] polling failed for {task.watcher.source_name}: {e}")
        return

    state.apply_source_state(task.watcher.source_name, outcome.source_state)
    for event in outcome.events:
        if writer.append(event):
            print_event(event)
    state.save()


def _reap_completed_poll_tasks(inflight: dict[str, PollTask], state: StateStore, writer: JsonlWriter) -> None:
    completed_sources = [source_name for source_name, task in inflight.items() if task.future.done()]
    for source_name in completed_sources:
        task = inflight.pop(source_name)
        _finalize_poll_task(task, state, writer)


def _mark_timed_out_poll_tasks(inflight: dict[str, PollTask], *, now_monotonic: float) -> None:
    for task in inflight.values():
        if task.timed_out or task.future.done():
            continue
        if (now_monotonic - task.started_monotonic) < task.timeout_seconds:
            continue
        task.timed_out = True
        task.watcher.last_poll_at = time.time()
        print(f"[ERROR] polling timed out for {task.watcher.source_name} after {task.timeout_seconds:.1f}s")


def process_poll_cycle(
    watch_list,
    state: StateStore,
    writer: JsonlWriter,
    executor: ThreadPoolExecutor,
    inflight: dict[str, PollTask],
    *,
    now_ts: float | None = None,
    now_monotonic: float | None = None,
) -> None:
    now_ts = time.time() if now_ts is None else now_ts
    now_monotonic = time.monotonic() if now_monotonic is None else now_monotonic
    _reap_completed_poll_tasks(inflight, state, writer)
    _mark_timed_out_poll_tasks(inflight, now_monotonic=now_monotonic)
    _submit_ready_poll_tasks(watch_list, state, executor, inflight, now_ts=now_ts, now_monotonic=now_monotonic)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    state = StateStore(STATE_PATH)
    writer = JsonlWriter(EVENTS_PATH)
    watch_list = build_watchers()

    print(f"[{now_iso()}] starting modular free-source watcher stack")
    print(f"[startup] output_dir   : {OUTPUT_ROOT.resolve()}")
    print(f"[startup] events_jsonl : {EVENTS_PATH.resolve()}")
    print(f"[startup] state_json   : {STATE_PATH.resolve()}")
    print(f"[startup] watcher_count: {len(watch_list)}")
    for watcher in watch_list:
        print(f"  - {watcher.source_name}  (interval={watcher.interval_seconds}s)")

    warmup_watchers(watch_list, state, writer)
    print("[startup] warmup complete; entering polling loop")
    max_workers = get_max_poll_workers(len(watch_list))
    inflight: dict[str, PollTask] = {}
    with ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="watch-free-sources") as executor:
        while True:
            process_poll_cycle(watch_list, state, writer, executor, inflight)
            time.sleep(1)


if __name__ == "__main__":
    main()
