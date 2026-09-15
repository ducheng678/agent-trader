import json
import os
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

from market_agent.constants import EVENT_TIME_KEYS


def current_utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def parse_utc_iso(value: str) -> Optional[datetime]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        try:
            return datetime.strptime(str(value).strip(), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            return None


def _iter_jsonl_lines_reverse(path: Path, chunk_size: int = 65536):
    if chunk_size <= 0:
        chunk_size = 65536
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        buffer = b""
        while position > 0:
            read_size = min(chunk_size, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            buffer = chunk + buffer
            parts = buffer.split(b"\n")
            buffer = parts[0]
            for raw_line in reversed(parts[1:]):
                if raw_line.strip():
                    yield raw_line.decode("utf-8")
        if buffer.strip():
            yield buffer.decode("utf-8")


def _extract_event_datetime(event: Dict[str, Any]) -> Tuple[Optional[datetime], str]:
    sources: List[Tuple[str, Any]] = []
    for key in EVENT_TIME_KEYS:
        sources.append((key, event.get(key)))
    raw = event.get("raw")
    if isinstance(raw, dict):
        for key in EVENT_TIME_KEYS:
            sources.append((f"raw.{key}", raw.get(key)))
    for key, value in sources:
        parsed = parse_utc_iso(str(value or ""))
        if parsed is not None:
            return parsed, key
    return None, ""


def normalize_event_record(event: Dict[str, Any], *, seen_at: Optional[str] = None) -> Dict[str, Any]:
    normalized = dict(event or {})
    seen_iso = str(normalized.get("seen_at", "") or seen_at or current_utc_iso()).strip()
    normalized["seen_at"] = seen_iso
    event_dt, source = _extract_event_datetime(normalized)
    if event_dt is not None:
        normalized["event_timestamp"] = event_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        normalized["event_time_source"] = source
    elif not normalized.get("event_timestamp"):
        normalized["event_timestamp"] = seen_iso
        normalized["event_time_source"] = "seen_at"
    return normalized


def strip_item_id_for_llm(value: Any) -> Any:
    if isinstance(value, dict):
        source = dict(value)
        if not str(source.get("event_timestamp", "") or "").strip():
            for time_key in ("published_at", "seen_at", "timestamp", "ts", "created_at", "updated_at", "event_time", "time"):
                fallback_time = str(source.get(time_key, "") or "").strip()
                if fallback_time:
                    source["event_timestamp"] = fallback_time
                    break
        if str(source.get("summary", "") or "").strip() == str(source.get("title", "") or "").strip():
            source.pop("summary", None)
        return {
            str(key): strip_item_id_for_llm(item)
            for key, item in source.items()
            if str(key)
            not in {
                "item_id",
                "url",
                "raw",
                "attachments",
                "local_files",
                "category",
                "published_at",
                "seen_at",
                "event_time_source",
                "timestamp",
                "ts",
                "created_at",
                "updated_at",
                "event_time",
                "time",
            }
        }
    if isinstance(value, list):
        return [strip_item_id_for_llm(item) for item in value]
    return value


def build_recent_event_context(
    events: List[Dict[str, Any]],
    *,
    window_hours: float,
    max_items: int,
    now: Optional[datetime] = None,
) -> List[Dict[str, Any]]:
    ref = now or parse_utc_iso(current_utc_iso()) or datetime.now(timezone.utc)
    filtered: List[Tuple[datetime, Dict[str, Any]]] = []
    for raw in events:
        event = normalize_event_record(raw)
        event_dt = parse_utc_iso(str(event.get("event_timestamp", "") or ""))
        if event_dt is None:
            event_dt = parse_utc_iso(str(event.get("seen_at", "") or ""))
        if event_dt is None:
            continue
        age_hours = max(0.0, (ref - event_dt).total_seconds() / 3600.0)
        if window_hours > 0 and age_hours > window_hours:
            continue
        enriched = dict(event)
        enriched["event_age_hours"] = round(age_hours, 3)
        enriched["is_within_recent_window"] = True
        filtered.append((event_dt, enriched))
    filtered.sort(key=lambda item: item[0])
    if max_items > 0:
        filtered = filtered[-max_items:]
    return [item[1] for item in filtered]


class EventFileWatcher:
    def __init__(
        self,
        path: Path,
        start_from: str,
        max_recent: int,
        *,
        recent_window_hours: float = 72.0,
        max_context_items: Optional[int] = None,
    ):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self.max_recent = max_recent
        self.recent_window_hours = max(0.0, float(recent_window_hours or 0.0))
        self.max_context_items = int(max_context_items or max_recent or 0)
        self.recent_events: Deque[Dict[str, Any]] = deque(maxlen=max_recent)
        self.offset = 0
        self._load_existing_events()
        if start_from == "end":
            self.offset = self.path.stat().st_size
        else:
            self.offset = 0

    def _load_existing_events(self) -> None:
        try:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(obj, dict):
                        self.recent_events.append(normalize_event_record(obj, seen_at=self.current_utc_iso()))
        except FileNotFoundError:
            return

    def poll(self) -> List[Dict[str, Any]]:
        new_events: List[Dict[str, Any]] = []
        with self.path.open("r", encoding="utf-8") as f:
            if self.offset > self.path.stat().st_size:
                self.offset = 0
            f.seek(self.offset)
            while True:
                line = f.readline()
                if not line:
                    break
                self.offset = f.tell()
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    print(f"[warn] skipped malformed JSONL line: {line[:120]}")
                    continue
                if isinstance(obj, dict):
                    normalized = normalize_event_record(obj, seen_at=self.current_utc_iso())
                    self.recent_events.append(normalized)
                    new_events.append(normalized)
        return new_events

    def recent(self) -> List[Dict[str, Any]]:
        return build_recent_event_context(
            list(self.recent_events),
            window_hours=self.recent_window_hours,
            max_items=self.max_context_items,
            now=parse_utc_iso(self.current_utc_iso()),
        )

    def current_utc_iso(self) -> str:
        return current_utc_iso()
