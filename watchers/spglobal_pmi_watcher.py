import os
import re
import time

import requests
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

DEFAULT_SPGLOBAL_PMI_URL = ""
SPGLOBAL_PMI_ARTICLE_PATH_RE = re.compile(
    r"^/marketintelligence/en/mi/research-analysis/(?!pmi\.html$)[a-z0-9-]+(?:\.html)?/?$",
    re.IGNORECASE,
)
SPGLOBAL_PMI_BLOCK_MARKERS = ("access denied", "security controls triggered")


class SpglobalPmiWatcher(HtmlPageWatcher):
    def __init__(self, source_name: str, url: str, interval_seconds: int = WEB_POLL_SECONDS):
        super().__init__(source_name, url, interval_seconds)
        cookie_header = clean_text(os.getenv("SPGLOBAL_PMI_COOKIE_HEADER", ""))
        if cookie_header:
            self.session.headers.update({"Cookie": cookie_header})
        self.summary_cache: dict[str, str] = {}

    def _fetch_html(self) -> str:
        accept = "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1"
        try:
            resp = browser_impersonated_get(self.url, headers=dict(self.session.headers), accept=accept)
        except Exception:
            resp = fetch_with_retries(self.session, self.url, accept=accept)
        text = resp.text or ""
        if getattr(resp, "status_code", None) == 403 and any(marker in text.lower() for marker in SPGLOBAL_PMI_BLOCK_MARKERS):
            raise RuntimeError(
                "S&P Global PMI blocked access to the configured page; set SPGLOBAL_PMI_COOKIE_HEADER from a browser session or use another allowed official endpoint"
            )
        return text

    def _fetch_article_summary(self, url: str) -> str:
        if url in self.summary_cache:
            return self.summary_cache[url]
        summary = ""
        accept = "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1"
        try:
            try:
                resp = browser_impersonated_get(url, headers=dict(self.session.headers), accept=accept)
            except Exception:
                resp = fetch_with_retries(self.session, url, accept=accept)
            summary = best_effort_html_summary(resp.text)
        except Exception:
            summary = ""
        self.summary_cache[url] = summary
        return summary

    def _enrich_summaries(self, items: list[Event], *, limit: int | None = None) -> list[Event]:
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
        for anchor in soup.find_all("a", href=True):
            href = anchor["href"]
            title = clean_text(anchor.get_text(" ", strip=True))
            full_url = requests.compat.urljoin(self.url, href)
            parsed = requests.utils.urlparse(full_url)
            if parsed.netloc not in {"www.spglobal.com", "spglobal.com"}:
                continue
            if not SPGLOBAL_PMI_ARTICLE_PATH_RE.match(parsed.path):
                continue
            if len(title) < 15 or full_url in seen_urls:
                continue
            seen_urls.add(full_url)
            items.append(
                Event(
                    source=self.source_name,
                    item_id=stable_id(self.source_name, full_url),
                    title=title,
                    url=full_url,
                    category="economic_release",
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


def build_spglobal_pmi_watchers() -> list[SpglobalPmiWatcher]:
    urls = parse_csv_env("SPGLOBAL_PMI_URLS", DEFAULT_SPGLOBAL_PMI_URL)
    if not urls:
        return []
    total = len(urls)
    interval = int(os.getenv("SPGLOBAL_PMI_POLL_SECONDS", str(WEB_POLL_SECONDS)))
    return [SpglobalPmiWatcher(make_instance_source_name("spglobal_pmi", i, total), url, interval) for i, url in enumerate(urls)]
