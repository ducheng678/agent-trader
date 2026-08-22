import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from typing import Optional
from urllib.parse import urlparse

import requests

from .common import (
    Event,
    USER_AGENT,
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
from .watcher_keywords import (
    dedupe_keywords,
    keywords_from_trade_symbols as _shared_keywords_from_trade_symbols,
    profile_matches as _shared_profile_matches,
    trade_topic_profiles as _shared_trade_topic_profiles,
)

DEFAULT_REUTERS_TARGET = "https://www.reuters.com/markets/"
DEFAULT_REUTERS_SITEMAP_INDEX = "https://www.reuters.com/arc/outboundfeeds/news-sitemap-index/?outputType=xml"
REUTERS_DEFAULT_KEYWORDS = [
    "economy",
    "economic",
    "inflation",
    "prices",
    "jobs",
    "labor",
    "labour",
    "trade",
    "tariff",
    "tax",
    "fed",
    "rates",
    "bond",
    "bonds",
    "currency",
    "currencies",
    "dxy",
    "dollar index",
    "treasury",
    "oil",
    "crude",
    "brent",
    "silver",
    "bitcoin",
]
REUTERS_STOPWORDS = {"markets", "world", "business", "latest", "news", "the", "and", "for", "with", "from"}
REUTERS_SKIP_PATH_PARTS = ("/graphics/", "/video/", "/podcast/", "/plus/")
REUTERS_NON_ENGLISH_PREFIXES = {
    "/ar/",
    "/de/",
    "/es/",
    "/fr/",
    "/it/",
    "/ja/",
    "/pt/",
    "/pt-br/",
    "/ru/",
    "/tr/",
    "/zh-hans/",
    "/zh-hant/",
}
SUMMARY_BLOCK_MARKERS = {
    "access denied",
    "too many requests",
    "verify you are human",
    "captcha",
    "request blocked",
    "forbidden",
    "are you a robot",
}


@dataclass
class ReutersSitemapEntry:
    url: str
    title: str
    published_at: Optional[str] = None
    keywords: str = ""


class ReutersWatcher:
    def __init__(self, source_name: str, target_url: str, interval_seconds: int = WEB_POLL_SECONDS):
        self.source_name = source_name
        self.target_url = target_url
        self.interval_seconds = interval_seconds
        self.last_poll_at = 0.0
        self.sitemap_index_url = os.getenv("REUTERS_SITEMAP_INDEX_URL", DEFAULT_REUTERS_SITEMAP_INDEX)
        self.max_sitemap_pages = max(1, int(os.getenv("REUTERS_SITEMAP_PAGES", "3") or "3"))
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "application/xml, text/xml;q=0.9, */*;q=0.1",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.reuters.com/",
            }
        )
        self.explicit_keywords = [item.lower() for item in parse_csv_env("REUTERS_KEYWORDS", "")]
        self.keywords = _keywords_for_target(target_url)
        self.trade_topic_profiles = _trade_topic_profiles()
        self.summary_cache: dict[str, str] = {}
        self.summary_fetch_backoff_until = 0.0
        self.summary_block_cooldown_seconds = max(300, int(os.getenv("REUTERS_SUMMARY_BLOCK_COOLDOWN_SECONDS", "21600") or "21600"))

    def should_poll(self, ts: float) -> bool:
        return (ts - self.last_poll_at) >= self.interval_seconds

    def _fetch_xml_root(self, url: str) -> ET.Element:
        resp = fetch_with_retries(self.session, url, accept="application/xml, text/xml;q=0.9, */*;q=0.1")
        raw = (resp.content or b"").lstrip(b"\xef\xbb\xbf\x00 \t\r\n")
        if not raw:
            raise ValueError(f"empty XML from {url}")
        return ET.fromstring(raw)

    def _iter_sitemap_urls(self) -> list[str]:
        root = self._fetch_xml_root(self.sitemap_index_url)
        urls: list[str] = []
        for node in root.findall(".//"):
            if _local_name(node.tag) != "loc":
                continue
            value = clean_text(node.text or "")
            if value:
                urls.append(value)
        return urls[: self.max_sitemap_pages]

    def _iter_entries_from_sitemap(self, sitemap_url: str) -> list[ReutersSitemapEntry]:
        root = self._fetch_xml_root(sitemap_url)
        entries: list[ReutersSitemapEntry] = []
        for url_node in root.findall(".//"):
            if _local_name(url_node.tag) != "url":
                continue
            loc = ""
            title = ""
            published_at = None
            keywords = ""
            for child in list(url_node):
                child_name = _local_name(child.tag)
                if child_name == "loc":
                    loc = clean_text(child.text or "")
                elif child_name == "news":
                    for news_child in list(child):
                        news_name = _local_name(news_child.tag)
                        if news_name == "title":
                            title = clean_text(news_child.text or "")
                        elif news_name == "publication_date":
                            published_at = clean_text(news_child.text or "") or None
                        elif news_name == "keywords":
                            keywords = clean_text(news_child.text or "")
            if loc and title:
                entries.append(ReutersSitemapEntry(url=loc, title=title, published_at=published_at, keywords=keywords))
        return entries

    def _entry_matches(self, entry: ReutersSitemapEntry) -> bool:
        parsed = urlparse(entry.url)
        if parsed.netloc not in {"www.reuters.com", "reuters.com"}:
            return False
        if _is_non_english_reuters_path(parsed.path):
            return False
        if any(part in parsed.path for part in REUTERS_SKIP_PATH_PARTS):
            return False
        haystack = f"{entry.url} {entry.title} {entry.keywords}".lower()
        if self.explicit_keywords and any(keyword in haystack for keyword in self.explicit_keywords):
            return True
        if self.trade_topic_profiles:
            if any(_profile_matches(profile, haystack) for profile in self.trade_topic_profiles):
                return True
        return any(keyword in haystack for keyword in self.keywords)

    def _entry_to_event(self, entry: ReutersSitemapEntry, sitemap_url: str) -> Event:
        return Event(
            source=self.source_name,
            item_id=stable_id(self.source_name, entry.url),
            title=entry.title,
            url=entry.url,
            published_at=entry.published_at,
            category="markets_news",
            raw={
                "matched_keywords": self.explicit_keywords or self.keywords,
                "sitemap_url": sitemap_url,
                "keywords": entry.keywords,
            },
        )

    def _summary_fetch_is_backed_off(self) -> bool:
        return time.time() < self.summary_fetch_backoff_until

    def _record_summary_fetch_block(self) -> None:
        self.summary_fetch_backoff_until = max(
            self.summary_fetch_backoff_until,
            time.time() + float(self.summary_block_cooldown_seconds),
        )

    def _looks_like_summary_fetch_block(self, status_code: Optional[int], html: str) -> bool:
        if status_code in {401, 403}:
            return True
        lowered = (html or "").lower()
        return any(marker in lowered for marker in SUMMARY_BLOCK_MARKERS)

    def _fallback_summary(self, item: Event) -> str:
        title = clean_text(item.title)
        keywords = clean_text(str(item.raw.get("keywords", "")))
        if keywords:
            normalized_keywords = []
            for part in keywords.split(","):
                keyword = clean_text(part)
                lowered = keyword.lower()
                if not keyword:
                    continue
                if any(token in lowered for token in {"guid:", "vguid:", "usn:", "newsml_"}):
                    continue
                if any(ch.isdigit() for ch in keyword) and len(keyword) > 8:
                    continue
                if ":" in keyword:
                    continue
                normalized_keywords.append(keyword)
            if normalized_keywords:
                return clean_text(f"{title}. Topics: {', '.join(normalized_keywords)}.")
        return title

    def _fetch_article_summary(self, url: str) -> str:
        if url in self.summary_cache:
            return self.summary_cache[url]
        if self._summary_fetch_is_backed_off():
            self.summary_cache[url] = ""
            return ""
        summary = ""
        blocked = False
        accept_header = "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1"
        try:
            resp = fetch_with_retries(
                self.session,
                url,
                accept=accept_header,
            )
            if self._looks_like_summary_fetch_block(getattr(resp, "status_code", None), getattr(resp, "text", "")):
                blocked = True
            else:
                summary = best_effort_html_summary(resp.text)
        except requests.HTTPError as exc:
            response = getattr(exc, "response", None)
            blocked = self._looks_like_summary_fetch_block(
                getattr(response, "status_code", None),
                getattr(response, "text", ""),
            )
        except Exception:
            summary = ""
        if blocked and not summary:
            headers = {"Referer": self.target_url}
            cookie_header = clean_text(self.session.headers.get("Cookie", ""))
            if cookie_header:
                headers["Cookie"] = cookie_header
            try:
                response = browser_impersonated_get(url, headers=headers, accept=accept_header)
                if not self._looks_like_summary_fetch_block(
                    getattr(response, "status_code", None),
                    getattr(response, "text", ""),
                ):
                    summary = best_effort_html_summary(getattr(response, "text", ""))
                    blocked = False
            except Exception:
                pass
        if blocked and not summary:
            self._record_summary_fetch_block()
        self.summary_cache[url] = summary
        return summary

    def _enrich_summaries(self, items: list[Event], *, limit: Optional[int] = None) -> list[Event]:
        target_items = items if limit is None or limit <= 0 else items[:limit]
        for item in target_items:
            if item.summary or not item.url:
                continue
            item.summary = self._fetch_article_summary(item.url)
            if not item.summary:
                item.summary = self._fallback_summary(item)
        return items

    def _fetch_events(self) -> list[Event]:
        seen_urls: set[str] = set()
        items: list[Event] = []
        for sitemap_url in self._iter_sitemap_urls():
            for entry in self._iter_entries_from_sitemap(sitemap_url):
                if not self._entry_matches(entry):
                    continue
                if entry.url in seen_urls:
                    continue
                seen_urls.add(entry.url)
                items.append(self._entry_to_event(entry, sitemap_url))
        items.sort(key=lambda item: (item.published_at or "", item.url), reverse=True)
        return items[:150]

    def warmup(self, state) -> list[Event]:
        items = self._fetch_events()
        self._enrich_summaries(items, limit=WARMUP_PREVIEW_COUNT)
        preview = items[:WARMUP_PREVIEW_COUNT]
        print(f"[warmup][{self.source_name}] latest {len(preview)} items:")
        for item in preview:
            print(f"  title={item.title[:140]}")
            print(f"    url={item.url}")
        return apply_warmup_mode(
            self.source_name,
            state,
            [item.item_id for item in items],
            emitted_events=list(reversed(items)),
        )

    def poll(self, state) -> list[Event]:
        items = self._fetch_events()
        self.last_poll_at = time.time()
        new_items = [item for item in items if not state.is_seen(self.source_name, item.item_id)]
        self._enrich_summaries(new_items)
        for item in new_items:
            state.mark_seen(self.source_name, item.item_id)
        return list(reversed(new_items))


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1].lower() if tag else ""


def _is_non_english_reuters_path(path: str) -> bool:
    lowered = (path or "").lower()
    return any(lowered.startswith(prefix) for prefix in REUTERS_NON_ENGLISH_PREFIXES)


def _keywords_for_target(target_url: str) -> list[str]:
    explicit = parse_csv_env("REUTERS_KEYWORDS", "")
    trade_symbol_keywords = _keywords_from_trade_symbols()
    if explicit or trade_symbol_keywords:
        return dedupe_keywords([*explicit, *trade_symbol_keywords])
    parsed = urlparse(target_url)
    parts = [part.strip().lower() for part in parsed.path.split("/") if part.strip()]
    keywords = [part for part in parts if part not in REUTERS_STOPWORDS and len(part) >= 3]
    return keywords or list(REUTERS_DEFAULT_KEYWORDS)


def _keywords_from_trade_symbols() -> list[str]:
    return _shared_keywords_from_trade_symbols()


def _trade_topic_profiles():
    return _shared_trade_topic_profiles()


def _profile_matches(profile, haystack: str) -> bool:
    return _shared_profile_matches(profile, haystack)


def build_reuters_watchers() -> list[ReutersWatcher]:
    urls = parse_csv_env("REUTERS_URLS", DEFAULT_REUTERS_TARGET)
    total = len(urls)
    interval = int(os.getenv("REUTERS_POLL_SECONDS", str(WEB_POLL_SECONDS)))
    return [ReutersWatcher(make_instance_source_name("reuters", i, total), url, interval) for i, url in enumerate(urls)]
