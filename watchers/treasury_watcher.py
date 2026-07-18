import os
import re
import time
from typing import Optional
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from .common import (
    Event,
    HtmlPageWatcher,
    WEB_POLL_SECONDS,
    WARMUP_PREVIEW_COUNT,
    apply_warmup_mode,
    best_effort_html_summary,
    clean_text,
    fetch_with_retries,
    make_instance_source_name,
    parse_csv_env,
    stable_id,
)

DEFAULT_TREASURY_URL = "https://home.treasury.gov/news/press-releases"
TREASURY_PRESS_PATH_RE = re.compile(r"^/news/press-releases/[a-z0-9-]+/?$", re.IGNORECASE)
TREASURY_SKIP_PATH_RE = re.compile(
    r"^/news/press-releases/(?:statements-remarks(?:/.*)?|readouts(?:/.*)?|testimonies(?:/.*)?)$",
    re.IGNORECASE,
)
TREASURY_SKIP_TITLE_RE = re.compile(
    r"^(?:remarks and statements|view all remarks and statements|statements\s*&\s*remarks|readouts|testimonies|secretary statements\s*&\s*remarks)$",
    re.IGNORECASE,
)


class TreasuryWatcher(HtmlPageWatcher):
    def __init__(self, source_name: str, url: str, interval_seconds: int = WEB_POLL_SECONDS):
        super().__init__(source_name, url, interval_seconds)
        self.summary_cache: dict[str, str] = {}

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
            if parsed.netloc not in {"home.treasury.gov", "www.home.treasury.gov"}:
                continue
            if not TREASURY_PRESS_PATH_RE.match(parsed.path):
                continue
            title = clean_text(a.get_text(" ", strip=True))
            if TREASURY_SKIP_PATH_RE.match(parsed.path) or TREASURY_SKIP_TITLE_RE.match(title):
                continue
            if len(title) < 10 or title.lower() in {"press releases", "view all press releases"}:
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
                    category="press_release",
                    raw={"path": parsed.path},
                )
            )
            if len(items) >= 30:
                break
        return items

    def warmup(self, state) -> list[Event]:
        html = self._fetch_html()
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
        html = self._fetch_html()
        items = self.extract_events(html)
        self.last_poll_at = time.time()
        new_items = [item for item in items if not state.is_seen(self.source_name, item.item_id)]
        self._enrich_summaries(new_items)
        for item in new_items:
            state.mark_seen(self.source_name, item.item_id)
        return list(reversed(new_items))


def build_treasury_watchers() -> list[TreasuryWatcher]:
    urls = parse_csv_env("TREASURY_URLS", DEFAULT_TREASURY_URL)
    total = len(urls)
    interval = int(os.getenv("TREASURY_POLL_SECONDS", str(WEB_POLL_SECONDS)))
    return [TreasuryWatcher(make_instance_source_name("treasury", i, total), url, interval) for i, url in enumerate(urls)]
