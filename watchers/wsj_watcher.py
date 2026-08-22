import os
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Optional

import requests

from .common import (
    BaseWatcher,
    Event,
    USER_AGENT,
    WEB_POLL_SECONDS,
    WARMUP_PREVIEW_COUNT,
    apply_warmup_mode,
    clean_text,
    fetch_with_retries,
    make_instance_source_name,
    parse_csv_env,
    stable_id,
)
from .watcher_keywords import dedupe_keywords, keywords_from_trade_symbols as _keywords_from_trade_symbols

DEFAULT_WSJ_URL = "https://www.wsj.com/economy"
DEFAULT_WSJ_FEEDS = [
    "https://feeds.content.dowjones.io/public/rss/RSSWorldNews",
    "https://feeds.content.dowjones.io/public/rss/RSSMarketsMain",
    "https://feeds.content.dowjones.io/public/rss/WSJcomUSBusiness",
    "https://feeds.content.dowjones.io/public/rss/RSSUSNews",
    "https://feeds.content.dowjones.io/public/rss/socialeconomyfeed",
]
WSJ_SKIP_TITLE_PARTS = ("market talk", "roundup")


@dataclass
class WsjFeedEntry:
    url: str
    title: str
    published_at: Optional[str] = None
    summary: str = ""
    feed_url: str = ""


class WsjWatcher(BaseWatcher):
    def __init__(self, source_name: str, target_url: str, interval_seconds: int = WEB_POLL_SECONDS):
        super().__init__(source_name, interval_seconds)
        self.target_url = target_url
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
                "Accept": "application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.1",
                "Referer": "https://www.wsj.com/",
            }
        )
        self.feed_urls = parse_csv_env("WSJ_FEEDS", ",".join(DEFAULT_WSJ_FEEDS))
        explicit = parse_csv_env("WSJ_KEYWORDS", "")
        self.keywords = dedupe_keywords([*explicit, *_keywords_from_trade_symbols()])

    def _fetch_feed_entries(self, feed_url: str) -> list[WsjFeedEntry]:
        resp = fetch_with_retries(
            self.session,
            feed_url,
            accept="application/rss+xml, application/xml, text/xml;q=0.9, */*;q=0.1",
        )
        root = ET.fromstring((resp.content or b"").lstrip(b"\xef\xbb\xbf\x00 \t\r\n"))
        entries: list[WsjFeedEntry] = []
        for item in root.findall("./channel/item"):
            title = clean_text(item.findtext("title", default=""))
            url = clean_text(item.findtext("link", default=""))
            summary = clean_text(item.findtext("description", default=""))
            pub_raw = clean_text(item.findtext("pubDate", default=""))
            published_at = None
            if pub_raw:
                try:
                    published_at = parsedate_to_datetime(pub_raw).astimezone().isoformat()
                except Exception:
                    published_at = pub_raw
            if title and url:
                entries.append(WsjFeedEntry(url=url, title=title, published_at=published_at, summary=summary, feed_url=feed_url))
        return entries

    def _entry_matches(self, entry: WsjFeedEntry) -> bool:
        haystack = f"{entry.title} {entry.summary} {entry.url}".lower()
        if any(part in haystack for part in WSJ_SKIP_TITLE_PARTS):
            return False
        return any(keyword in haystack for keyword in self.keywords)

    def _fetch_events(self) -> list[Event]:
        seen_urls: set[str] = set()
        items: list[Event] = []
        for feed_url in self.feed_urls:
            for entry in self._fetch_feed_entries(feed_url):
                if not self._entry_matches(entry):
                    continue
                if entry.url in seen_urls:
                    continue
                seen_urls.add(entry.url)
                items.append(
                    Event(
                        source=self.source_name,
                        item_id=stable_id(self.source_name, entry.url),
                        title=entry.title,
                        url=entry.url,
                        published_at=entry.published_at,
                        summary=entry.summary,
                        category="wsj_news",
                        raw={"feed_url": feed_url, "matched_keywords": self.keywords},
                    )
                )
        items.sort(key=lambda item: (item.published_at or "", item.url), reverse=True)
        return items[:100]

    def warmup(self, state) -> list[Event]:
        items = self._fetch_events()
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
        for item in new_items:
            state.mark_seen(self.source_name, item.item_id)
        return list(reversed(new_items))

def build_wsj_watchers() -> list[WsjWatcher]:
    urls = parse_csv_env("WSJ_URLS", DEFAULT_WSJ_URL)
    total = len(urls)
    interval = int(os.getenv("WSJ_POLL_SECONDS", str(WEB_POLL_SECONDS)))
    return [WsjWatcher(make_instance_source_name("wsj", i, total), url, interval) for i, url in enumerate(urls)]
