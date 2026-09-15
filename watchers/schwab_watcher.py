import os
import re
import time
from datetime import datetime, timezone
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

DEFAULT_SCHWAB_URL = "https://www.schwab.com/learn/market-commentary"
SCHWAB_ARTICLE_PATH_RE = re.compile(r"^/learn/story/[a-z0-9-]+/?$", re.IGNORECASE)
SCHWAB_TYPE_DATE_RE = re.compile(
    r"^(Article|Podcast|Video|Watchlist|Interactive)\s*\|\s*([A-Za-z]{3,9}\s+\d{1,2},\s+\d{4})$",
    re.IGNORECASE,
)


def _parse_card_date(text: str) -> Optional[str]:
    match = SCHWAB_TYPE_DATE_RE.match(clean_text(text))
    if not match:
        return None
    try:
        parsed = datetime.strptime(match.group(2), "%b %d, %Y").replace(tzinfo=timezone.utc)
        return parsed.isoformat()
    except Exception:
        return match.group(2)


class SchwabWatcher(HtmlPageWatcher):
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

    def _extract_card_metadata(self, anchor) -> tuple[str, str, Optional[str]]:
        title = ""
        summary = ""
        published_at = None

        heading = anchor.find(["h1", "h2", "h3", "h4", "h5", "h6"])
        if heading is not None:
            title = clean_text(heading.get_text(" ", strip=True))
        if not title:
            title = clean_text(anchor.get("aria-label", ""))

        text_nodes: list[str] = []
        for node in anchor.find_all(["h1", "h2", "h3", "h4", "h5", "h6", "p", "span"]):
            text = clean_text(node.get_text(" ", strip=True))
            if text:
                text_nodes.append(text)
        if not title and text_nodes:
            title = text_nodes[0]

        for text in text_nodes:
            if text == title:
                continue
            maybe_published_at = _parse_card_date(text)
            if maybe_published_at:
                published_at = maybe_published_at
                continue
            if not summary:
                summary = text
        return title, summary, published_at

    def extract_events(self, html: str) -> list[Event]:
        soup = BeautifulSoup(html, "html.parser")
        seen_urls: set[str] = set()
        items: list[Event] = []
        for anchor in soup.find_all("a", href=True):
            href = urljoin(self.url, anchor["href"])
            parsed = urlparse(href)
            if parsed.netloc not in {"www.schwab.com", "schwab.com"}:
                continue
            if not SCHWAB_ARTICLE_PATH_RE.match(parsed.path):
                continue
            if href in seen_urls:
                continue

            title, summary, published_at = self._extract_card_metadata(anchor)
            if len(title) < 10:
                continue

            seen_urls.add(href)
            items.append(
                Event(
                    source=self.source_name,
                    item_id=stable_id(self.source_name, href),
                    title=title,
                    url=href,
                    published_at=published_at,
                    summary=summary,
                    category="market_commentary",
                    raw={"path": parsed.path},
                )
            )
            if len(items) >= 60:
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


def build_schwab_watchers() -> list[SchwabWatcher]:
    urls = parse_csv_env("SCHWAB_URLS", DEFAULT_SCHWAB_URL)
    total = len(urls)
    interval = int(os.getenv("SCHWAB_POLL_SECONDS", str(WEB_POLL_SECONDS)))
    return [SchwabWatcher(make_instance_source_name("schwab", i, total), url, interval) for i, url in enumerate(urls)]
