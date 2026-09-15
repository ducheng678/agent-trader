import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
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
    env_flag_enabled,
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

DEFAULT_BLOOMBERG_TARGET = "https://www.bloomberg.com/markets/economics"
DEFAULT_BLOOMBERG_SITEMAP = "https://www.bloomberg.com/sitemaps/news/latest.xml"
DEFAULT_BLOOMBERG_RSS_FEEDS = [
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://feeds.bloomberg.com/politics/news.rss",
    "https://feeds.bloomberg.com/business/news.rss",
    "https://feeds.bloomberg.com/economics/news.rss",
    "https://feeds.bloomberg.com/industries/news.rss",
]
DEFAULT_BLOOMBERG_NEWS_SITEMAPS = [
    "https://www.bloomberg.com/feeds/sitemap_news.xml",
    "https://www.bloomberg.com/feeds/markets/sitemap_news.xml",
    "https://www.bloomberg.com/feeds/bbiz/sitemap_news.xml",
]
BLOOMBERG_ARTICLE_PATH_RE = re.compile(r"^/(news/articles|news/newsletters|news/live-blog)/", re.IGNORECASE)
BLOOMBERG_STOPWORDS = {
    "markets",
    "economics",
    "latest",
    "news",
    "and",
    "the",
    "for",
    "with",
    "from",
}
BLOOMBERG_DEFAULT_KEYWORDS = [
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
    "spend",
    "central bank",
    "fed",
    "rates",
    "rate",
    "bond",
    "bonds",
    "currency",
    "currencies",
    "dollar",
    "yuan",
    "euro",
    "pound",
    "gdp",
    "pmi",
    "manufacturing",
    "services",
    "payroll",
    "employment",
    "oil",
    "gas",
    "treasury",
    "deficit",
    "stimulus",
    "imports",
    "exports",
]
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
class BloombergSitemapEntry:
    url: str
    title: str
    published_at: Optional[str] = None
    summary: str = ""
    feed_kind: str = "sitemap"


class BloombergWatcher:
    def __init__(
        self,
        source_name: str,
        target_url: str,
        interval_seconds: int = WEB_POLL_SECONDS,
        *,
        endpoint_url: Optional[str] = None,
        endpoint_kind: str = "news_sitemap",
        source_role: str = "realtime_candidate",
        alert_enabled: bool = True,
    ):
        self.source_name = source_name
        self.target_url = target_url
        self.interval_seconds = interval_seconds
        self.last_poll_at = 0.0
        self.endpoint_url = endpoint_url or os.getenv("BLOOMBERG_SITEMAP_URL", DEFAULT_BLOOMBERG_SITEMAP)
        self.endpoint_kind = clean_text(endpoint_kind).lower() or "news_sitemap"
        self.source_role = clean_text(source_role).lower() or "realtime_candidate"
        self.alert_enabled = bool(alert_enabled)
        self.sitemap_url = self.endpoint_url
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.1",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.bloomberg.com/",
            }
        )
        cookie_header = clean_text(os.getenv("BLOOMBERG_COOKIE_HEADER", ""))
        if cookie_header:
            self.session.headers.update({"Cookie": cookie_header})
        self.explicit_keywords = [item.lower() for item in parse_csv_env("BLOOMBERG_KEYWORDS", "")]
        self.keywords = _keywords_for_target(target_url)
        self.trade_topic_profiles = _trade_topic_profiles()
        self.summary_cache: dict[str, str] = {}
        self.summary_fetch_backoff_until = 0.0
        self.summary_block_cooldown_seconds = max(300, int(os.getenv("BLOOMBERG_SUMMARY_BLOCK_COOLDOWN_SECONDS", "21600") or "21600"))

    def should_poll(self, ts: float) -> bool:
        return (ts - self.last_poll_at) >= self.interval_seconds

    def _fetch_xml_root(self) -> ET.Element:
        resp = fetch_with_retries(
            self.session,
            self.endpoint_url,
            accept="application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.1",
        )
        raw = (resp.content or b"").lstrip(b"\xef\xbb\xbf\x00 \t\r\n")
        if not raw:
            raise ValueError(f"empty XML from {self.endpoint_url}")
        return ET.fromstring(raw)

    def _iter_sitemap_entries(self, root: ET.Element) -> list[BloombergSitemapEntry]:
        entries: list[BloombergSitemapEntry] = []
        for url_node in root.findall(".//"):
            if _local_name(url_node.tag) != "url":
                continue
            loc = ""
            title = ""
            published_at = None
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
            if loc and title:
                entries.append(BloombergSitemapEntry(url=loc, title=title, published_at=published_at))
        return entries

    def _rss_child_text(self, item: ET.Element, name: str) -> str:
        for child in list(item):
            if _local_name(child.tag) == name:
                return clean_text(child.text or "")
        return ""

    def _parse_rss_datetime(self, value: str) -> Optional[str]:
        text = clean_text(value)
        if not text:
            return None
        try:
            return parsedate_to_datetime(text).astimezone().isoformat()
        except Exception:
            return text

    def _iter_rss_entries(self, root: ET.Element) -> list[BloombergSitemapEntry]:
        entries: list[BloombergSitemapEntry] = []
        rss_items = [item for item in root.findall(".//") if _local_name(item.tag) == "item"]
        if rss_items:
            for item in rss_items:
                title = self._rss_child_text(item, "title")
                url = self._rss_child_text(item, "link")
                published_at = self._parse_rss_datetime(self._rss_child_text(item, "pubdate"))
                summary = self._rss_child_text(item, "description")
                if title and url:
                    entries.append(
                        BloombergSitemapEntry(
                            url=url,
                            title=title,
                            published_at=published_at,
                            summary=summary,
                            feed_kind="rss",
                        )
                    )
            return entries

        atom_entries = [item for item in root.findall(".//") if _local_name(item.tag) == "entry"]
        for item in atom_entries:
            title = self._rss_child_text(item, "title")
            url = ""
            for link in list(item):
                if _local_name(link.tag) != "link":
                    continue
                href = clean_text(link.attrib.get("href", ""))
                rel = clean_text(link.attrib.get("rel", "alternate"))
                if href and rel in {"", "alternate"}:
                    url = href
                    break
                if href and not url:
                    url = href
            published_at = clean_text(self._rss_child_text(item, "updated") or self._rss_child_text(item, "published")) or None
            summary = clean_text(self._rss_child_text(item, "summary") or self._rss_child_text(item, "content"))
            if title and url:
                entries.append(
                    BloombergSitemapEntry(
                        url=url,
                        title=title,
                        published_at=published_at,
                        summary=summary,
                        feed_kind="atom",
                    )
                )
        return entries

    def _iter_entries(self) -> list[BloombergSitemapEntry]:
        root = self._fetch_xml_root()
        if self.endpoint_kind == "rss":
            return self._iter_rss_entries(root)
        return self._iter_sitemap_entries(root)

    def _entry_matches(self, entry: BloombergSitemapEntry) -> bool:
        parsed = urlparse(entry.url)
        if parsed.netloc not in {"www.bloomberg.com", "bloomberg.com"}:
            return False
        if not BLOOMBERG_ARTICLE_PATH_RE.match(parsed.path):
            return False
        haystack = f"{entry.url} {entry.title}".lower()
        if self.explicit_keywords and any(keyword in haystack for keyword in self.explicit_keywords):
            return True
        if self.trade_topic_profiles:
            if any(_profile_matches(profile, haystack) for profile in self.trade_topic_profiles):
                return True
        return any(keyword in haystack for keyword in self.keywords)

    def _entry_to_event(self, entry: BloombergSitemapEntry) -> Event:
        return Event(
            source=self.source_name,
            item_id=stable_id(self.source_name, entry.url),
            title=entry.title,
            url=entry.url,
            published_at=entry.published_at,
            summary=entry.summary,
            category="economics",
            raw={
                "matched_keywords": self.explicit_keywords or self.keywords,
                "discovered_via": self.endpoint_url,
                "endpoint_kind": self.endpoint_kind,
                "feed_kind": entry.feed_kind,
                "source_role": self.source_role,
                "alert_enabled": self.alert_enabled,
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
        return clean_text(item.title)

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
        entries = self._iter_entries()
        seen_urls: set[str] = set()
        items: list[Event] = []
        for entry in entries:
            if not self._entry_matches(entry):
                continue
            if entry.url in seen_urls:
                continue
            seen_urls.add(entry.url)
            items.append(self._entry_to_event(entry))
        items.sort(key=lambda item: (item.published_at or "", item.url), reverse=True)
        return items[:120]

    def warmup(self, state) -> list[Event]:
        items = self._fetch_events()
        self._enrich_summaries(items, limit=WARMUP_PREVIEW_COUNT)
        preview = items[:WARMUP_PREVIEW_COUNT]
        print(f"[warmup][{self.source_name}] role={self.source_role} latest {len(preview)} items:")
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


def _keywords_for_target(target_url: str) -> list[str]:
    explicit = parse_csv_env("BLOOMBERG_KEYWORDS", "")
    trade_symbol_keywords = _keywords_from_trade_symbols()
    if explicit or trade_symbol_keywords:
        return dedupe_keywords([*explicit, *trade_symbol_keywords])
    parsed = urlparse(target_url)
    parts = [part.strip().lower() for part in parsed.path.split("/") if part.strip()]
    if "economics" in parts:
        return list(BLOOMBERG_DEFAULT_KEYWORDS)
    keywords = [part for part in parts if part not in BLOOMBERG_STOPWORDS and len(part) >= 3]
    return keywords or list(BLOOMBERG_DEFAULT_KEYWORDS)


def _keywords_from_trade_symbols() -> list[str]:
    return _shared_keywords_from_trade_symbols()


def _trade_topic_profiles():
    return _shared_trade_topic_profiles()


def _profile_matches(profile, haystack: str) -> bool:
    return _shared_profile_matches(profile, haystack)


def build_bloomberg_watchers() -> list[BloombergWatcher]:
    if not env_flag_enabled("BLOOMBERG_ENABLED", True):
        return []
    target_urls = parse_csv_env("BLOOMBERG_URLS", "")
    if not target_urls:
        return []
    target_url = target_urls[0]
    interval = int(os.getenv("BLOOMBERG_POLL_SECONDS", str(WEB_POLL_SECONDS)))
    watchers: list[BloombergWatcher] = []

    if os.getenv("BLOOMBERG_ENABLE_RSS_FEEDS", "true").lower() == "true":
        rss_feeds = parse_csv_env("BLOOMBERG_RSS_FEEDS", ",".join(DEFAULT_BLOOMBERG_RSS_FEEDS))
        total = len(rss_feeds)
        for i, feed_url in enumerate(rss_feeds):
            watchers.append(
                BloombergWatcher(
                    make_instance_source_name("bloomberg_rss", i, total),
                    target_url,
                    interval,
                    endpoint_url=feed_url,
                    endpoint_kind="rss",
                    source_role="realtime_candidate",
                    alert_enabled=True,
                )
            )

    if os.getenv("BLOOMBERG_ENABLE_NEWS_SITEMAP_REPAIR", "true").lower() == "true":
        sitemap_urls = parse_csv_env("BLOOMBERG_NEWS_SITEMAP_URLS", ",".join(DEFAULT_BLOOMBERG_NEWS_SITEMAPS))
        total = len(sitemap_urls)
        for i, sitemap_url in enumerate(sitemap_urls):
            watchers.append(
                BloombergWatcher(
                    make_instance_source_name("bloomberg_news_sitemap", i, total),
                    target_url,
                    interval,
                    endpoint_url=sitemap_url,
                    endpoint_kind="news_sitemap",
                    source_role="coverage_repair",
                    alert_enabled=True,
                )
            )

    if os.getenv("BLOOMBERG_ENABLE_BACKFILL_SITEMAP", "true").lower() == "true":
        backfill_urls = parse_csv_env(
            "BLOOMBERG_BACKFILL_SITEMAP_URLS",
            os.getenv("BLOOMBERG_SITEMAP_URL", DEFAULT_BLOOMBERG_SITEMAP),
        )
        alert_enabled = os.getenv("BLOOMBERG_BACKFILL_ALERT_ENABLED", "false").lower() == "true"
        total = len(backfill_urls)
        for i, sitemap_url in enumerate(backfill_urls):
            watchers.append(
                BloombergWatcher(
                    make_instance_source_name("bloomberg_backfill", i, total),
                    target_url,
                    interval,
                    endpoint_url=sitemap_url,
                    endpoint_kind="news_sitemap",
                    source_role="backfill",
                    alert_enabled=alert_enabled,
                )
            )

    return watchers
