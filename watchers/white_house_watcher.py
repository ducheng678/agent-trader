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
    path_category,
    stable_id,
)

WH_ARTICLE_PATH_RE = re.compile(r"^/(?:articles|briefings-statements|presidential-actions|fact-sheets|remarks|research)/20\d{2}/\d{2}/")
DEFAULT_WHITEHOUSE_URL = "https://www.whitehouse.gov/news/"


class WhiteHouseWatcher(HtmlPageWatcher):
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
            if parsed.netloc not in {"www.whitehouse.gov", "whitehouse.gov"}:
                continue
            if not WH_ARTICLE_PATH_RE.match(parsed.path):
                continue
            title = clean_text(a.get_text(" ", strip=True))
            if len(title) < 12 or title.lower() in {"read the latest", "view all"}:
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


def build_white_house_watchers() -> list[WhiteHouseWatcher]:
    urls = parse_csv_env("WHITEHOUSE_URLS", DEFAULT_WHITEHOUSE_URL)
    total = len(urls)
    return [WhiteHouseWatcher(make_instance_source_name("white_house", i, total), url) for i, url in enumerate(urls)]
