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
    browser_impersonated_get,
    clean_text,
    fetch_with_retries,
    make_instance_source_name,
    parse_csv_env,
    stable_id,
)

DEFAULT_DOL_URL = "https://www.dol.gov/newsroom/releases"
DOL_RELEASE_PATH_RE = re.compile(r"^/newsroom/releases/[^/]+/[^/]+/?$", re.IGNORECASE)
DOL_CHALLENGE_MARKERS = ("Challenge Validation", "cp_clge_done", "custmsg crypto")


class DolWatcher(HtmlPageWatcher):
    def __init__(self, source_name: str, url: str, interval_seconds: int = WEB_POLL_SECONDS):
        super().__init__(source_name, url, interval_seconds)
        cookie_header = clean_text(os.getenv("DOL_COOKIE_HEADER", ""))
        if cookie_header:
            self.session.headers.update({"Cookie": cookie_header})
        self.summary_cache: dict[str, str] = {}

    def _fetch_html(self) -> str:
        html = self._fetch_html_with_browser_impersonation()
        if all(marker in html for marker in DOL_CHALLENGE_MARKERS):
            raise RuntimeError(
                "DOL returned a challenge page; set DOL_COOKIE_HEADER from a browser session or use a less-protected DOL endpoint"
            )
        return html

    def _fetch_html_with_browser_impersonation(self) -> str:
        try:
            response = browser_impersonated_get(self.url, headers=dict(self.session.headers))
            return response.text
        except Exception:
            return super()._fetch_html()

    def _fetch_article_summary(self, url: str) -> str:
        if url in self.summary_cache:
            return self.summary_cache[url]
        summary = ""
        try:
            response = browser_impersonated_get(
                url,
                headers=dict(self.session.headers),
                accept="text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
            )
            if not all(marker in (response.text or "") for marker in DOL_CHALLENGE_MARKERS):
                summary = best_effort_html_summary(response.text)
        except Exception:
            summary = ""
        if not summary:
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
            if parsed.netloc not in {"www.dol.gov", "dol.gov"}:
                continue
            if not DOL_RELEASE_PATH_RE.match(parsed.path):
                continue
            title = clean_text(a.get_text(" ", strip=True))
            if len(title) < 12 or title.lower() == "news releases":
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
                    category="labor_release",
                    raw={"path": parsed.path},
                )
            )
            if len(items) >= 50:
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


def build_dol_watchers() -> list[DolWatcher]:
    urls = parse_csv_env("DOL_URLS", DEFAULT_DOL_URL)
    total = len(urls)
    interval = int(os.getenv("DOL_POLL_SECONDS", str(WEB_POLL_SECONDS)))
    return [DolWatcher(make_instance_source_name("dol", i, total), url, interval) for i, url in enumerate(urls)]
