import copy
import hashlib
import json
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse
from xml.etree.ElementTree import ParseError

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

try:
    from curl_cffi import requests as curl_requests
except Exception:
    curl_requests = None


load_dotenv()

USER_AGENT = os.getenv(
    "WATCH_USER_AGENT",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
)
REQUEST_TIMEOUT = int(os.getenv("WATCH_REQUEST_TIMEOUT", "30"))
WEB_POLL_SECONDS = int(os.getenv("WEB_POLL_SECONDS", "60"))
WARMUP_PREVIEW_COUNT = int(os.getenv("WARMUP_PREVIEW_COUNT", "3"))
MAX_SEEN_PER_SOURCE = int(os.getenv("MAX_SEEN_PER_SOURCE", "5000"))
HTTP_RETRY_COUNT = int(os.getenv("HTTP_RETRY_COUNT", "3"))
HTTP_RETRY_BACKOFF_SECONDS = float(os.getenv("HTTP_RETRY_BACKOFF_SECONDS", "2"))
VALID_WARMUP_MODES = {"mark_seen", "emit_recent", "cursor_only"}
CURL_IMPERSONATE = os.getenv("WATCH_CURL_IMPERSONATE", "chrome136")

OUTPUT_ROOT = Path(os.getenv("WATCH_OUTPUT_DIR", "data/free_sources_watch"))
STATE_PATH = OUTPUT_ROOT / "state.json"
EVENTS_PATH = OUTPUT_ROOT / "events.jsonl"
_POLL_DEADLINE = threading.local()


@dataclass
class Event:
    source: str
    item_id: str
    title: str
    url: str
    published_at: Optional[str] = None
    summary: str = ""
    category: Optional[str] = None
    attachments: list[dict[str, Any]] = field(default_factory=list)
    local_files: list[str] = field(default_factory=list)
    raw: dict[str, Any] = field(default_factory=dict)
    seen_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"))


class StateStore:
    def __init__(self, path: Path):
        self.path = path
        self.data = {"seen": {}, "cursors": {}}
        self._lock = threading.RLock()
        self._load()

    def _load(self) -> None:
        with self._lock:
            if self.path.exists():
                try:
                    self.data = json.loads(self.path.read_text(encoding="utf-8"))
                except Exception:
                    pass
            self.data.setdefault("seen", {})
            self.data.setdefault("cursors", {})

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=2), encoding="utf-8")

    def mark_seen(self, source: str, item_id: str) -> None:
        with self._lock:
            seen = self.data.setdefault("seen", {}).setdefault(source, [])
            if item_id not in seen:
                seen.append(item_id)
                if len(seen) > MAX_SEEN_PER_SOURCE:
                    self.data["seen"][source] = seen[-MAX_SEEN_PER_SOURCE:]

    def mark_many_seen(self, source: str, item_ids: list[str]) -> None:
        for item_id in item_ids:
            self.mark_seen(source, item_id)

    def is_seen(self, source: str, item_id: str) -> bool:
        with self._lock:
            return item_id in self.data.setdefault("seen", {}).setdefault(source, [])

    def get_cursor(self, source: str, default: Optional[str] = None) -> Optional[str]:
        with self._lock:
            return self.data.setdefault("cursors", {}).get(source, default)

    def set_cursor(self, source: str, value: Optional[str]) -> None:
        with self._lock:
            self.data.setdefault("cursors", {})[source] = value

    def fork(self) -> "StateStore":
        with self._lock:
            clone = object.__new__(StateStore)
            clone.path = self.path
            clone.data = copy.deepcopy(self.data)
            clone._lock = threading.RLock()
            return clone

    def export_source_state(self, source: str) -> dict[str, Any]:
        with self._lock:
            seen_map = self.data.setdefault("seen", {})
            cursor_map = self.data.setdefault("cursors", {})
            return {
                "seen": list(seen_map.get(source, [])),
                "has_cursor": source in cursor_map,
                "cursor": cursor_map.get(source),
            }

    def apply_source_state(self, source: str, source_state: dict[str, Any]) -> None:
        with self._lock:
            seen_items = list(source_state.get("seen", []))
            self.data.setdefault("seen", {})[source] = seen_items[-MAX_SEEN_PER_SOURCE:]
            if source_state.get("has_cursor"):
                self.data.setdefault("cursors", {})[source] = source_state.get("cursor")


class JsonlWriter:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()
        self._written_keys: set[tuple[str, str]] = set()
        self._load_existing()

    def _load_existing(self) -> None:
        with self._lock:
            if not self.path.exists():
                return
            for line in self.path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                source = clean_text(str(payload.get("source", "")))
                item_id = clean_text(str(payload.get("item_id", "")))
                if source and item_id:
                    self._written_keys.add((source, item_id))

    def append(self, event: Event) -> bool:
        key = (event.source, event.item_id)
        with self._lock:
            if key in self._written_keys:
                return False
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(asdict(event), ensure_ascii=False) + "\n")
            self._written_keys.add(key)
            return True


class BaseWatcher:
    def __init__(self, source_name: str, interval_seconds: int):
        self.source_name = source_name
        self.interval_seconds = interval_seconds
        self.last_poll_at = 0.0

    def should_poll(self, ts: float) -> bool:
        return (ts - self.last_poll_at) >= self.interval_seconds

    def _print_warmup_preview(self, items: list[Event]) -> None:
        preview = items[:WARMUP_PREVIEW_COUNT]
        print(f"[warmup][{self.source_name}] latest {len(preview)} items:")
        for item in preview:
            print(f"  title={item.title[:140]}")
            if item.url:
                print(f"    url={item.url}")

    def warmup(self, state: StateStore) -> list[Event]:
        raise NotImplementedError

    def poll(self, state: StateStore) -> list[Event]:
        raise NotImplementedError


def apply_source_proxy_to_session(
    source_name: str,
    session: requests.Session,
    *,
    healthcheck_url: str = "",
) -> None:
    from .proxy_failover import build_source_proxy_failover, load_source_proxy_config

    proxy_config = load_source_proxy_config(source_name)
    if not proxy_config.has_proxy:
        return
    session.proxies.update(
        {
            "http": proxy_config.http_proxy or proxy_config.https_proxy,
            "https": proxy_config.https_proxy or proxy_config.http_proxy,
        }
    )
    session._source_name = source_name
    session._source_proxy_failover = build_source_proxy_failover(
        source_name,
        proxy_config,
        healthcheck_url=healthcheck_url,
    )
    session._source_proxy_failover_max_rotations = int(proxy_config.max_rotations or 0)
    session._source_proxy_failover_sleep_seconds = float(proxy_config.sleep_seconds or 0.0)


def _http_status_from_error(error: BaseException) -> Optional[int]:
    response = getattr(error, "response", None)
    status_code = getattr(response, "status_code", None)
    try:
        return int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        return None


DEFAULT_SOURCE_PROXY_FAILOVER_STATUSES = {403, 407, 429, 500, 502, 503, 504}


def should_failover_source_proxy_error(
    error: BaseException,
    *,
    status_codes: Optional[set[int]] = None,
) -> bool:
    status_codes = DEFAULT_SOURCE_PROXY_FAILOVER_STATUSES if status_codes is None else status_codes
    status_code = _http_status_from_error(error)
    if status_code in status_codes:
        return True
    message = str(error or "").lower()
    status_tokens = {str(code) for code in status_codes}
    text_tokens: list[str] = [
        "ssl",
        "tls",
        "eof",
        "timed out",
        "timeout",
        "connection",
    ]
    if 403 in status_codes:
        text_tokens.append("forbidden")
    if 407 in status_codes:
        text_tokens.append("proxy")
    if 503 in status_codes:
        text_tokens.append("service unavailable")
    return any(
        token in message
        for token in (*status_tokens, *text_tokens)
    )


def rotate_source_proxy_for_session(session: requests.Session, reason: BaseException | str) -> bool:
    failover = getattr(session, "_source_proxy_failover", None)
    if failover is None:
        return False
    source_name = str(getattr(session, "_source_name", "") or "unknown")
    try:
        switched = failover.rotate_to_next_node()
    except Exception as exc:
        print(f"[source_proxy_failover][{source_name}] rotate failed: {type(exc).__name__}: {exc}")
        return False
    if not switched:
        return False
    reason_text = type(reason).__name__ if isinstance(reason, BaseException) else str(reason)
    probe_summary = str(getattr(failover, "last_probe_summary", "") or "").strip()
    probe_suffix = f" ({probe_summary})" if probe_summary else ""
    print(
        f"[source_proxy_failover][{source_name}] switched "
        f"{failover.selector_group} -> {switched}{probe_suffix} after {reason_text}"
    )
    sleep_seconds = max(0.0, float(getattr(session, "_source_proxy_failover_sleep_seconds", 0.0) or 0.0))
    if sleep_seconds > 0:
        time.sleep(sleep_seconds)
    return True


class HtmlPageWatcher(BaseWatcher):
    def __init__(self, source_name: str, url: str, interval_seconds: int):
        super().__init__(source_name, interval_seconds)
        self.url = url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})
        apply_source_proxy_to_session(self.source_name, self.session, healthcheck_url=self.url)

    def _fetch_html(self) -> str:
        resp = fetch_with_retries(self.session, self.url)
        return resp.text

    def extract_events(self, html: str) -> list[Event]:
        raise NotImplementedError

    def warmup(self, state: StateStore) -> list[Event]:
        html = self._fetch_html()
        items = self.extract_events(html)
        self._print_warmup_preview(items)
        return apply_warmup_mode(
            self.source_name,
            state,
            [item.item_id for item in items],
            emitted_events=list(reversed(items)),
        )

    def poll(self, state: StateStore) -> list[Event]:
        html = self._fetch_html()
        items = self.extract_events(html)
        self.last_poll_at = time.time()
        new_items = [item for item in items if not state.is_seen(self.source_name, item.item_id)]
        for item in new_items:
            state.mark_seen(self.source_name, item.item_id)
        return list(reversed(new_items))


class RssWatcher(BaseWatcher):
    def __init__(self, source_name: str, feed_url: str, interval_seconds: int, category: Optional[str] = None):
        super().__init__(source_name, interval_seconds)
        self.feed_url = feed_url
        self.category = category
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": USER_AGENT,
            "Accept": "application/rss+xml, application/xml, text/xml, application/atom+xml;q=0.9, */*;q=0.1",
        })
        apply_source_proxy_to_session(self.source_name, self.session, healthcheck_url=self.feed_url)

    def _parse_xml_root(self, resp: requests.Response) -> ET.Element:
        raw = resp.content or b""
        raw = raw.lstrip(b"\xef\xbb\xbf\x00 \t\r\n")
        try:
            return ET.fromstring(raw)
        except ParseError:
            text = resp.text or ""
            text = text.lstrip("\ufeff\x00 \t\r\n")
            xml_start = text.find("<?xml")
            rss_start = text.find("<rss")
            feed_start = text.find("<feed")
            starts = [i for i in [xml_start, rss_start, feed_start] if i >= 0]
            if starts:
                text = text[min(starts):]
            return ET.fromstring(text.encode("utf-8"))

    def _iter_items(self, root: ET.Element):
        items = root.findall("./channel/item")
        if items:
            return items, "rss"
        atom_items = root.findall("{http://www.w3.org/2005/Atom}entry")
        if atom_items:
            return atom_items, "atom"
        items = root.findall(".//item")
        if items:
            return items, "rss"
        return [], "rss"

    def _fetch_events(self) -> list[Event]:
        resp = fetch_with_retries(
            self.session,
            self.feed_url,
            accept="application/rss+xml, application/xml, text/xml, application/atom+xml;q=0.9, */*;q=0.1",
        )
        root = self._parse_xml_root(resp)
        items, feed_kind = self._iter_items(root)

        events: list[Event] = []
        for item in items:
            if feed_kind == "atom":
                title = clean_text(item.findtext("{http://www.w3.org/2005/Atom}title", default=""))
                url = ""
                for link in item.findall("{http://www.w3.org/2005/Atom}link"):
                    href = clean_text(link.attrib.get("href", ""))
                    rel = clean_text(link.attrib.get("rel", "alternate"))
                    if href and rel in {"", "alternate"}:
                        url = href
                        break
                    if href and not url:
                        url = href
                guid = clean_text(item.findtext("{http://www.w3.org/2005/Atom}id", default=""))
                description = clean_text(item.findtext("{http://www.w3.org/2005/Atom}summary", default=""))
                if not description:
                    description = clean_text(item.findtext("{http://www.w3.org/2005/Atom}content", default=""))
                pub_date_raw = clean_text(item.findtext("{http://www.w3.org/2005/Atom}updated", default=""))
                if not pub_date_raw:
                    pub_date_raw = clean_text(item.findtext("{http://www.w3.org/2005/Atom}published", default=""))
            else:
                title = clean_text(item.findtext("title", default=""))
                url = clean_text(item.findtext("link", default=""))
                guid = clean_text(item.findtext("guid", default=""))
                description = clean_text(item.findtext("description", default=""))
                pub_date_raw = clean_text(item.findtext("pubDate", default=""))

            published_at = None
            if pub_date_raw:
                try:
                    published_at = parsedate_to_datetime(pub_date_raw).astimezone().isoformat()
                except Exception:
                    published_at = pub_date_raw

            if not title or not url:
                continue

            item_id = guid or stable_id(self.source_name, url, title)
            events.append(
                Event(
                    source=self.source_name,
                    item_id=item_id,
                    title=title,
                    url=url,
                    published_at=published_at,
                    summary=description,
                    category=self.category,
                    raw={
                        "feed_url": self.feed_url,
                        "content_type": resp.headers.get("Content-Type", ""),
                        "feed_kind": feed_kind,
                    },
                )
            )
        return events

    def warmup(self, state: StateStore) -> list[Event]:
        items = self._fetch_events()
        self._print_warmup_preview(items)
        return apply_warmup_mode(
            self.source_name,
            state,
            [item.item_id for item in items],
            emitted_events=list(reversed(items)),
        )

    def poll(self, state: StateStore) -> list[Event]:
        items = self._fetch_events()
        self.last_poll_at = time.time()
        new_items = [item for item in items if not state.is_seen(self.source_name, item.item_id)]
        for item in new_items:
            state.mark_seen(self.source_name, item.item_id)
        return list(reversed(new_items))


def _sleep_backoff(attempt: int) -> None:
    delay_seconds = HTTP_RETRY_BACKOFF_SECONDS * max(1, attempt)
    remaining_seconds = remaining_poll_budget_seconds()
    if remaining_seconds is not None:
        if remaining_seconds <= 0:
            raise TimeoutError("poll time budget exhausted during backoff")
        delay_seconds = min(delay_seconds, remaining_seconds)
    time.sleep(delay_seconds)


def fetch_with_retries(
    session: requests.Session,
    url: str,
    *,
    accept: Optional[str] = None,
    proxy_failover_statuses: Optional[set[int]] = None,
) -> requests.Response:
    headers = {}
    if accept:
        headers["Accept"] = accept
    proxy_failover_statuses = (
        DEFAULT_SOURCE_PROXY_FAILOVER_STATUSES if proxy_failover_statuses is None else proxy_failover_statuses
    )

    last_err: Optional[Exception] = None
    max_rotations = max(0, int(getattr(session, "_source_proxy_failover_max_rotations", 0) or 0))
    rotations = 0
    max_attempts = max(HTTP_RETRY_COUNT, max_rotations + 1)
    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, timeout=bounded_timeout_seconds(REQUEST_TIMEOUT), headers=headers or None)
            if resp.status_code in (proxy_failover_statuses - {403, 407}):
                if rotations < max_rotations and rotate_source_proxy_for_session(session, f"HTTP {resp.status_code}"):
                    rotations += 1
                    continue
                if attempt < max_attempts:
                    _sleep_backoff(attempt)
                    continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            last_err = e
            if (
                rotations < max_rotations
                and should_failover_source_proxy_error(e, status_codes=proxy_failover_statuses)
                and rotate_source_proxy_for_session(session, e)
            ):
                rotations += 1
                continue
            if attempt < max_attempts:
                _sleep_backoff(attempt)
                continue
            raise

    assert last_err is not None
    raise last_err


def browser_impersonated_get(
    url: str,
    *,
    headers: Optional[dict[str, str]] = None,
    accept: Optional[str] = None,
    proxies: Optional[dict[str, str]] = None,
):
    if curl_requests is None:
        raise RuntimeError("curl_cffi is not installed")
    request_headers = dict(headers or {})
    request_headers.setdefault("User-Agent", USER_AGENT)
    if accept:
        request_headers["Accept"] = accept
    return curl_requests.get(
        url,
        headers=request_headers or None,
        proxies=proxies or None,
        impersonate=CURL_IMPERSONATE,
        timeout=bounded_timeout_seconds(REQUEST_TIMEOUT),
    )


def parse_csv_env(name: str, default: str = "") -> list[str]:
    value = os.getenv(name, default)
    return [item.strip() for item in (value or "").split(",") if item.strip()]


def env_flag_enabled(name: str, default: bool = True) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def get_warmup_mode() -> str:
    mode = clean_text(os.getenv("WATCH_WARMUP_MODE", "emit_recent")).lower() or "emit_recent"
    if mode not in VALID_WARMUP_MODES:
        raise ValueError(f"Invalid WATCH_WARMUP_MODE={mode!r}; expected one of {sorted(VALID_WARMUP_MODES)}")
    return mode


def apply_warmup_mode(
    source_name: str,
    state: StateStore,
    item_ids: list[str],
    *,
    emitted_events: Optional[list[Event]] = None,
    cursor_value: Optional[str] = None,
) -> list[Event]:
    mode = get_warmup_mode()
    emitted = list(emitted_events or [])
    if mode == "mark_seen":
        if cursor_value is not None:
            state.set_cursor(source_name, cursor_value)
        state.mark_many_seen(source_name, item_ids)
        return []
    if mode == "emit_recent":
        if cursor_value is not None:
            state.set_cursor(source_name, cursor_value)
        state.mark_many_seen(source_name, item_ids)
        return emitted
    if cursor_value is not None:
        state.set_cursor(source_name, cursor_value)
        return []

    print(f"[warmup][{source_name}] cursor_only unsupported for this source; falling back to mark_seen")
    state.mark_many_seen(source_name, item_ids)
    return []


@contextmanager
def poll_deadline(timeout_seconds: Optional[float]):
    previous_deadline = getattr(_POLL_DEADLINE, "deadline", None)
    next_deadline = previous_deadline
    if timeout_seconds is not None and timeout_seconds > 0:
        candidate = time.monotonic() + timeout_seconds
        next_deadline = candidate if previous_deadline is None else min(previous_deadline, candidate)
    _POLL_DEADLINE.deadline = next_deadline
    try:
        yield
    finally:
        _POLL_DEADLINE.deadline = previous_deadline


def remaining_poll_budget_seconds() -> Optional[float]:
    deadline = getattr(_POLL_DEADLINE, "deadline", None)
    if deadline is None:
        return None
    return max(0.0, deadline - time.monotonic())


def bounded_timeout_seconds(default_timeout: float) -> float:
    remaining_seconds = remaining_poll_budget_seconds()
    if remaining_seconds is None:
        return default_timeout
    if remaining_seconds <= 0:
        raise TimeoutError("poll time budget exhausted")
    return max(0.001, min(default_timeout, remaining_seconds))


def make_instance_source_name(base: str, index: int, total: int) -> str:
    return base if total <= 1 else f"{base}:{index + 1}"


def now_iso() -> str:
    return datetime.utcnow().isoformat() + "Z"


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def html_to_text(html: str) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)


SUMMARY_META_ATTRS = (
    ("name", "description"),
    ("property", "og:description"),
    ("name", "twitter:description"),
)
SUMMARY_SKIP_PHRASES = {
    "share",
    "share this page",
    "read more",
    "watch live",
    "learn more",
}


def _extract_meta_description(soup: BeautifulSoup) -> str:
    for attr_name, attr_value in SUMMARY_META_ATTRS:
        node = soup.find("meta", attrs={attr_name: attr_value})
        if node is None:
            continue
        content = clean_text(node.get("content", ""))
        if content:
            return content
    return ""


def best_effort_html_summary(
    html: str,
    *,
    paragraph_limit: int = 2,
    char_limit: int = 600,
    min_paragraph_length: int = 40,
) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    meta_description = _extract_meta_description(soup)
    if len(meta_description) >= min(min_paragraph_length, 20):
        return meta_description[:char_limit].rstrip()
    seen: set[str] = set()
    paragraphs: list[str] = []
    for container in [soup.find("article"), soup.find("main"), soup]:
        if container is None:
            continue
        for node in container.find_all(["p", "li"]):
            text = clean_text(node.get_text(" ", strip=True))
            if len(text) < min_paragraph_length:
                continue
            lowered = text.lower()
            if any(lowered == phrase or lowered.startswith(f"{phrase}:") for phrase in SUMMARY_SKIP_PHRASES):
                continue
            if text in seen:
                continue
            seen.add(text)
            paragraphs.append(text)
            if len(paragraphs) >= paragraph_limit or len(" ".join(paragraphs)) >= char_limit:
                return clean_text(" ".join(paragraphs))[:char_limit].rstrip()
    return clean_text(" ".join(paragraphs))[:char_limit].rstrip()


def stable_id(*parts: str) -> str:
    return hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()


def path_category(url: str) -> str:
    parsed = urlparse(url)
    return parsed.path.strip("/").split("/", 1)[0] if parsed.path.strip("/") else ""


def print_event(event: Event) -> None:
    print("\n" + "=" * 100)
    print(f"[{event.source}] {event.title}")
    if event.published_at:
        print(f"published_at: {event.published_at}")
    print(f"url        : {event.url}")
    if event.category:
        print(f"category   : {event.category}")
    if event.summary:
        print(f"summary    : {event.summary[:500]}")
    if event.attachments:
        print(f"attachments: {len(event.attachments)}")
        for idx, item in enumerate(event.attachments, start=1):
            print(f"  [{idx}] type={item.get('type')} url={item.get('url') or item.get('preview_url')}")
    if event.local_files:
        print("local_files:")
        for path in event.local_files:
            print(f"  {path}")
    print("=" * 100, flush=True)
