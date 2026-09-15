import os
import re
import time
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .common import (
    Event,
    HtmlPageWatcher,
    REQUEST_TIMEOUT,
    WEB_POLL_SECONDS,
    WARMUP_PREVIEW_COUNT,
    apply_warmup_mode,
    best_effort_html_summary,
    bounded_timeout_seconds,
    clean_text,
    env_flag_enabled,
    fetch_with_retries,
    make_instance_source_name,
    parse_csv_env,
    path_category,
    stable_id,
)

DEFAULT_COINDESK_URL = "https://www.coindesk.com/latest-crypto-news"
COINDESK_ARTICLE_PATH_RE = re.compile(r"^/.+/20\d{2}/\d{2}/\d{2}/")
COINDESK_SKIP_PATH_PARTS = {"/podcasts/", "/videos/", "/learn/", "/consensus/", "/opinion/"}
COINDESK_DEFAULT_429_COOLDOWN_SECONDS = 900


def _retry_after_seconds(value: str) -> Optional[float]:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return max(0.0, float(text))
    except ValueError:
        pass
    try:
        retry_at = parsedate_to_datetime(text)
        if retry_at.tzinfo is None:
            retry_at = retry_at.astimezone()
        return max(0.0, retry_at.timestamp() - time.time())
    except Exception:
        return None


class CoinDeskWatcher(HtmlPageWatcher):
    def __init__(self, source_name: str, url: str, interval_seconds: int = WEB_POLL_SECONDS):
        super().__init__(source_name, url, interval_seconds)
        self.summary_cache: dict[str, str] = {}
        self.cooldown_until = 0.0
        self.consecutive_429_count = 0
        self.default_429_cooldown_seconds = max(
            60.0,
            float(os.getenv("COINDESK_429_COOLDOWN_SECONDS", str(COINDESK_DEFAULT_429_COOLDOWN_SECONDS)) or COINDESK_DEFAULT_429_COOLDOWN_SECONDS),
        )
        self.max_429_cooldown_seconds = max(
            self.default_429_cooldown_seconds,
            float(os.getenv("COINDESK_429_COOLDOWN_MAX_SECONDS", "21600") or "21600"),
        )

    def should_poll(self, ts: float) -> bool:
        if ts < self.cooldown_until:
            return False
        return super().should_poll(ts)

    def _record_429_cooldown(self, exc: requests.HTTPError) -> None:
        response = getattr(exc, "response", None)
        retry_after = _retry_after_seconds((getattr(response, "headers", {}) or {}).get("Retry-After", ""))
        if retry_after is not None and retry_after > 0:
            cooldown_seconds = retry_after
        else:
            cooldown_seconds = min(
                self.max_429_cooldown_seconds,
                self.default_429_cooldown_seconds * (2 ** self.consecutive_429_count),
            )
        self.consecutive_429_count += 1
        self.cooldown_until = max(self.cooldown_until, time.time() + cooldown_seconds)
        print(
            f"[source_cooldown][{self.source_name}] HTTP 429; "
            f"pause polling for {cooldown_seconds:.1f}s "
            f"(consecutive_429={self.consecutive_429_count})"
        )

    def _fetch_listing_html(self) -> str:
        resp = self.session.get(
            self.url,
            timeout=bounded_timeout_seconds(REQUEST_TIMEOUT),
        )
        if resp.status_code == 429:
            exc = requests.HTTPError(f"429 Client Error: Too Many Requests for url: {self.url}", response=resp)
            self._record_429_cooldown(exc)
            return ""
        resp.raise_for_status()
        if self.consecutive_429_count:
            self.consecutive_429_count = 0
        return resp.text

    def _fetch_article_summary(self, url: str) -> str:
        if url in self.summary_cache:
            return self.summary_cache[url]
        summary = ""
        try:
            resp = fetch_with_retries(
                self.session,
                url,
                accept="text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
            )
            summary = best_effort_html_summary(resp.text)
        except Exception:
            summary = ""
        self.summary_cache[url] = summary
        return summary

    def _enrich_summaries(self, items: list[Event], *, limit: Optional[int] = None) -> list[Event]:
        target_items = items if limit is None or limit <= 0 else items[:limit]
        for item in target_items:
            if item.summary or not item.url:
                continue
            item.summary = self._fetch_article_summary(item.url)
        return items

    def extract_events(self, html: str) -> list[Event]:
        soup = BeautifulSoup(html, "html.parser")
        seen_urls: set[str] = set()
        items: list[Event] = []
        for a in soup.find_all("a", href=True):
            href = urljoin(self.url, a["href"])
            parsed = urlparse(href)
            if parsed.netloc not in {"www.coindesk.com", "coindesk.com"}:
                continue
            if any(part in parsed.path for part in COINDESK_SKIP_PATH_PARTS):
                continue
            if not COINDESK_ARTICLE_PATH_RE.match(parsed.path):
                continue
            title = clean_text(a.get_text(" ", strip=True))
            if len(title) < 12:
                continue
            if href in seen_urls:
                continue
            seen_urls.add(href)
            items.append(
                Event(
                    source=self.source_name,
                    item_id=stable_id(self.source_name, href),
                    title=title,
                    url=href,
                    category=path_category(href),
                    raw={"path": parsed.path},
                )
            )
            if len(items) >= 30:
                break
        return items

    def warmup(self, state) -> list[Event]:
        html = self._fetch_listing_html()
        items = self.extract_events(html)
        self._enrich_summaries(items, limit=WARMUP_PREVIEW_COUNT)
        self._print_warmup_preview(items)
        return apply_warmup_mode(
            self.source_name,
            state,
            [item.item_id for item in items],
            emitted_events=list(reversed(items)),
        )

    def poll(self, state) -> list[Event]:
        html = self._fetch_listing_html()
        items = self.extract_events(html)
        self.last_poll_at = time.time()
        new_items = [item for item in items if not state.is_seen(self.source_name, item.item_id)]
        self._enrich_summaries(new_items)
        for item in new_items:
            state.mark_seen(self.source_name, item.item_id)
        return list(reversed(new_items))


def build_coindesk_watchers() -> list[CoinDeskWatcher]:
    if not env_flag_enabled("COINDESK_ENABLED", True):
        return []
    urls = parse_csv_env("COINDESK_URLS", DEFAULT_COINDESK_URL)
    total = len(urls)
    interval = int(os.getenv("COINDESK_POLL_SECONDS", str(WEB_POLL_SECONDS)))
    return [CoinDeskWatcher(make_instance_source_name("coindesk", i, total), url, interval) for i, url in enumerate(urls)]
