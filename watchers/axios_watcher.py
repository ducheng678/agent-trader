import os
import re
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
from .watcher_keywords import keywords_from_trade_symbols

DEFAULT_AXIOS_TARGET = "https://www.axios.com/world/iran"
DEFAULT_AXIOS_SITEMAP = "https://www.axios.com/sitemap.xml"
AXIOS_ARTICLE_PATH_RE = re.compile(r"^/\d{4}/\d{2}/\d{2}/[a-z0-9-]+/?$")
AXIOS_SUMMARY_BLOCK_MARKERS = {
    "access denied",
    "too many requests",
    "verify you are human",
    "captcha",
    "request blocked",
    "forbidden",
    "are you a robot",
}
STOPWORDS = {
    "world",
    "politics",
    "policy",
    "business",
    "technology",
    "health",
    "energy",
    "climate",
    "science",
    "sports",
    "local",
    "news",
    "latest",
    "the",
    "and",
    "or",
    "of",
    "to",
    "for",
    "in",
    "on",
}


@dataclass
class SitemapEntry:
    url: str
    lastmod: Optional[str] = None
    title: str = ""
    keywords: str = ""


class AxiosWatcher:
    def __init__(self, source_name: str, target_url: str, interval_seconds: int = WEB_POLL_SECONDS):
        self.source_name = source_name
        self.target_url = target_url
        self.interval_seconds = interval_seconds
        self.last_poll_at = 0.0
        self.sitemap_url = os.getenv("AXIOS_SITEMAP_URL", DEFAULT_AXIOS_SITEMAP)
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "application/xml, text/xml;q=0.9, */*;q=0.1",
                "Accept-Language": "en-US,en;q=0.9",
                "Cache-Control": "no-cache",
                "Pragma": "no-cache",
                "Referer": "https://www.axios.com/",
            }
        )
        self.explicit_keywords = [item.lower() for item in parse_csv_env("AXIOS_KEYWORDS", "")]
        self.target_keywords = _dedupe_keywords([*_keywords_for_target(target_url), *keywords_from_trade_symbols()])
        combined_keywords = list(self.explicit_keywords) + list(self.target_keywords)
        self.keywords = _dedupe_keywords(combined_keywords) or ["axios"]
        self.summary_cache: dict[str, str] = {}
        self.summary_fetch_backoff_until = 0.0
        self.summary_block_cooldown_seconds = max(
            300,
            int(os.getenv("AXIOS_SUMMARY_BLOCK_COOLDOWN_SECONDS", "21600") or "21600"),
        )

    def should_poll(self, ts: float) -> bool:
        return (ts - self.last_poll_at) >= self.interval_seconds

    def _fetch_xml_root(self, url: str) -> ET.Element:
        resp = fetch_with_retries(self.session, url, accept="application/xml, text/xml;q=0.9, */*;q=0.1")
        raw = (resp.content or b"").lstrip(b"\xef\xbb\xbf\x00 \t\r\n")
        if not raw:
            raise ValueError(f"empty XML from {url}")
        return ET.fromstring(raw)

    def _iter_sitemap_entries(self) -> list[SitemapEntry]:
        root = self._fetch_xml_root(self.sitemap_url)
        tag = _local_name(root.tag)
        if tag == "sitemapindex":
            nested_urls = []
            for sitemap in root.findall(".//"):
                if _local_name(sitemap.tag) == "loc" and sitemap.text:
                    nested_urls.append(clean_text(sitemap.text))
            entries: list[SitemapEntry] = []
            max_nested = int(os.getenv("AXIOS_MAX_NESTED_SITEMAPS", "8"))
            for nested in nested_urls[:max_nested]:
                try:
                    entries.extend(self._parse_urlset(self._fetch_xml_root(nested)))
                except Exception:
                    continue
            return entries
        return self._parse_urlset(root)

    def _parse_urlset(self, root: ET.Element) -> list[SitemapEntry]:
        entries: list[SitemapEntry] = []
        for url_node in root.findall(".//"):
            if _local_name(url_node.tag) != "url":
                continue
            loc = None
            lastmod = None
            title = ""
            keywords = ""
            for child in list(url_node):
                lname = _local_name(child.tag)
                if lname == "loc":
                    loc = clean_text(child.text or "")
                elif lname == "lastmod":
                    lastmod = clean_text(child.text or "") or None
                elif lname == "news":
                    for news_child in list(child):
                        news_name = _local_name(news_child.tag)
                        if news_name == "publication_date":
                            lastmod = clean_text(news_child.text or "") or lastmod
                        elif news_name == "title":
                            title = clean_text(news_child.text or "")
                        elif news_name == "keywords":
                            keywords = clean_text(news_child.text or "")
            if loc:
                entries.append(SitemapEntry(url=loc, lastmod=lastmod, title=title, keywords=keywords))
        return entries

    def _entry_haystack(self, entry: SitemapEntry, *, include_metadata: bool = True) -> str:
        parsed = urlparse(entry.url)
        base = f"{parsed.path} {entry.url} {entry.title}".lower()
        if not include_metadata:
            return base
        return f"{base} {entry.keywords}".lower()

    def _matching_keywords(
        self,
        entry: SitemapEntry,
        keywords: Optional[list[str]] = None,
        *,
        include_metadata: bool = True,
    ) -> list[str]:
        haystack = self._entry_haystack(entry, include_metadata=include_metadata)
        active_keywords = keywords if keywords is not None else self.keywords
        return [keyword for keyword in active_keywords if keyword and keyword in haystack]

    def _entry_matches(self, entry: SitemapEntry) -> bool:
        parsed = urlparse(entry.url)
        if parsed.netloc not in {"www.axios.com", "axios.com"}:
            return False
        if not AXIOS_ARTICLE_PATH_RE.match(parsed.path):
            return False
        return bool(self._matching_keywords(entry))

    def _entry_to_event(self, entry: SitemapEntry, matched_keywords: list[str]) -> Event:
        slug = urlparse(entry.url).path.rstrip("/").split("/")[-1]
        title = entry.title or _prettify_slug(slug)
        primary_matches = self._matching_keywords(entry, self.target_keywords)
        title_url_matches = self._matching_keywords(entry, include_metadata=False)
        primary_title_url_matches = self._matching_keywords(
            entry,
            self.target_keywords,
            include_metadata=False,
        )
        return Event(
            source=self.source_name,
            item_id=stable_id(self.source_name, entry.url),
            title=title,
            url=entry.url,
            published_at=entry.lastmod,
            category="axios",
            raw={
                "matched_keywords": matched_keywords,
                "primary_matched_keywords": primary_matches,
                "title_url_matched_keywords": title_url_matches,
                "primary_title_url_matched_keywords": primary_title_url_matches,
                "match_count": len(matched_keywords),
                "primary_match_count": len(primary_matches),
                "title_url_match_count": len(title_url_matches),
                "primary_title_url_match_count": len(primary_title_url_matches),
                "discovered_via": self.sitemap_url,
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
        return any(marker in lowered for marker in AXIOS_SUMMARY_BLOCK_MARKERS)

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
        entries = self._iter_sitemap_entries()
        best_entries_by_url: dict[str, tuple[SitemapEntry, list[str]]] = {}
        for entry in entries:
            matched_keywords = self._matching_keywords(entry)
            if not matched_keywords:
                continue
            existing = best_entries_by_url.get(entry.url)
            if existing is None or _entry_quality(entry) > _entry_quality(existing[0]):
                best_entries_by_url[entry.url] = (entry, matched_keywords)
        items = []
        for entry, matched_keywords in best_entries_by_url.values():
            items.append(self._entry_to_event(entry, matched_keywords))
        items.sort(
            key=lambda item: (
                item.published_at or "",
                int(item.raw.get("primary_title_url_match_count", 0)),
                int(item.raw.get("primary_match_count", 0)),
                int(item.raw.get("title_url_match_count", 0)),
                int(item.raw.get("match_count", 0)),
                item.url,
            ),
            reverse=True,
        )
        return items[:100]

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


def _prettify_slug(slug: str) -> str:
    return clean_text(slug.replace("-", " ")).title()


def _entry_quality(entry: SitemapEntry) -> tuple[int, int, int]:
    return (
        1 if entry.title else 0,
        1 if entry.keywords else 0,
        1 if entry.lastmod else 0,
    )


def _dedupe_keywords(keywords: list[str]) -> list[str]:
    deduped: list[str] = []
    for keyword in keywords:
        normalized = clean_text(keyword).lower()
        if normalized and normalized not in deduped:
            deduped.append(normalized)
    return deduped


def _keywords_for_target(target_url: str) -> list[str]:
    parsed = urlparse(target_url)
    parts = [p.strip().lower() for p in parsed.path.split("/") if p.strip()]
    keywords = [p for p in parts if p not in STOPWORDS and len(p) >= 3]
    if not keywords:
        keywords = ["axios"]
    return keywords


def build_axios_watchers() -> list[AxiosWatcher]:
    urls = parse_csv_env("AXIOS_URLS", DEFAULT_AXIOS_TARGET)
    total = len(urls)
    interval = int(os.getenv("AXIOS_POLL_SECONDS", str(WEB_POLL_SECONDS)))
    return [AxiosWatcher(make_instance_source_name("axios", i, total), url, interval) for i, url in enumerate(urls)]
