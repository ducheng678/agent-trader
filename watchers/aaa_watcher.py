import os
import re
import time
from typing import Optional
from urllib.parse import urljoin, urlparse

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
    rotate_source_proxy_for_session,
    parse_csv_env,
    stable_id,
)

DEFAULT_AAA_URL = "https://gasprices.aaa.com/"
AAA_ARTICLE_PATH_RE = re.compile(r"^/[a-z0-9-]+/?$", re.IGNORECASE)
AAA_SKIP_PATHS = {
    "/",
    "/about-aaa/",
    "/subscribe/",
    "/state-gas-price-averages/",
    "/ev-charging-prices/",
    "/aaa-gas-cost-calculator/",
    "/news/",
    "/fuel-saving-tips/",
    "/fuel-quality/",
    "/premium-fuel-research/",
    "/top-trends/",
    "/contact/",
    "/privacy-policy/",
}


class AAAWatcher(HtmlPageWatcher):
    def __init__(self, source_name: str, url: str, interval_seconds: int = WEB_POLL_SECONDS):
        super().__init__(source_name, url, interval_seconds)
        self.summary_cache: dict[str, str] = {}

    def _fetch_html_with_browser_fallback(self, url: str) -> str:
        accept = "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1"
        try:
            resp = fetch_with_retries(
                self.session,
                url,
                accept=accept,
                proxy_failover_statuses={407, 429, 500, 502, 503, 504},
            )
            return resp.text
        except requests.HTTPError as e:
            status_code = getattr(getattr(e, "response", None), "status_code", None)
            if status_code != 403:
                raise
        max_rotations = max(0, int(getattr(self.session, "_source_proxy_failover_max_rotations", 0) or 0))
        rotations = 0
        while True:
            browser_kwargs = {"proxies": dict(self.session.proxies)} if self.session.proxies else {}
            try:
                resp = browser_impersonated_get(url, accept=accept, **browser_kwargs)
                resp.raise_for_status()
                return resp.text
            except Exception as exc:
                if rotations >= max_rotations or not rotate_source_proxy_for_session(self.session, exc):
                    raise
                rotations += 1

    def _fetch_html(self) -> str:
        return self._fetch_html_with_browser_fallback(self.url)

    def _fetch_article_summary(self, url: str) -> str:
        if url in self.summary_cache:
            return self.summary_cache[url]
        summary = ""
        try:
            summary = best_effort_html_summary(self._fetch_html_with_browser_fallback(url))
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
            if parsed.netloc not in {"gasprices.aaa.com"}:
                continue
            normalized_path = parsed.path.rstrip("/") + "/" if parsed.path not in {"", "/"} else "/"
            if normalized_path in AAA_SKIP_PATHS:
                continue
            if not AAA_ARTICLE_PATH_RE.match(parsed.path):
                continue
            title = clean_text(a.get_text(" ", strip=True))
            if len(title) < 12 or title.lower() in {"read more »", "read more"}:
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
                    category="fuel_prices",
                    raw={"path": parsed.path},
                )
            )
            if len(items) >= 20:
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


def build_aaa_watchers() -> list[AAAWatcher]:
    urls = parse_csv_env("AAA_URLS", DEFAULT_AAA_URL)
    total = len(urls)
    interval = int(os.getenv("AAA_POLL_SECONDS", str(WEB_POLL_SECONDS)))
    return [AAAWatcher(make_instance_source_name("aaa", i, total), url, interval) for i, url in enumerate(urls)]
