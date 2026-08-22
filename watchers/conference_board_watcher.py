import os
import re
import time
from datetime import datetime
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

DEFAULT_CONFERENCE_BOARD_URL = "https://www.conference-board.org/press/index.cfm?centerid=34"
CB_ALLOWED_PATH_RE = re.compile(r"^/(?:press/|topics/|research/|north-america/press/)", re.IGNORECASE)
CB_RELEASE_DATE_RE = re.compile(
    r"\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},\s+\d{4}\b"
)
CB_SKIP_PATH_RE = re.compile(r"^/press/(?:media-contact|experts?)/?$", re.IGNORECASE)
CB_SKIP_TITLE_RE = re.compile(r"media contacts?|experts page", re.IGNORECASE)


def _split_release_title_and_date(title: str) -> tuple[str, Optional[str]]:
    matches = list(CB_RELEASE_DATE_RE.finditer(title))
    if not matches:
        return title, None
    match = matches[-1]
    release_title = clean_text(title[: match.start()].rstrip(" -|"))
    try:
        published_at = datetime.strptime(match.group(0), "%B %d, %Y").date().isoformat()
    except ValueError:
        published_at = match.group(0)
    return release_title or title, published_at


class ConferenceBoardWatcher(HtmlPageWatcher):
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
            if parsed.netloc not in {"www.conference-board.org", "conference-board.org"}:
                continue
            title = clean_text(a.get_text(" ", strip=True))
            title, published_at = _split_release_title_and_date(title)
            if not CB_ALLOWED_PATH_RE.match(parsed.path) or parsed.path.rstrip("/") == "/press":
                continue
            if not published_at:
                continue
            if CB_SKIP_PATH_RE.match(parsed.path) or CB_SKIP_TITLE_RE.search(title):
                continue
            if len(title) < 10 or title.lower() in {"press", "read more"}:
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
                    published_at=published_at,
                    category="economic_release",
                    raw={"path": parsed.path},
                )
            )
            if len(items) >= 40:
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


def build_conference_board_watchers() -> list[ConferenceBoardWatcher]:
    urls = parse_csv_env("CONFERENCE_BOARD_URLS", DEFAULT_CONFERENCE_BOARD_URL)
    total = len(urls)
    interval = int(os.getenv("CONFERENCE_BOARD_POLL_SECONDS", str(WEB_POLL_SECONDS)))
    return [ConferenceBoardWatcher(make_instance_source_name("conference_board", i, total), url, interval) for i, url in enumerate(urls)]
