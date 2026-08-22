import os
from email.utils import parsedate_to_datetime

import requests

from .common import Event, RssWatcher, WEB_POLL_SECONDS, browser_impersonated_get, clean_text, fetch_with_retries, make_instance_source_name, parse_csv_env, stable_id

DEFAULT_BLS_FEEDS = "https://www.bls.gov/feed/bls_latest.rss"
BLS_POLL_SECONDS = int(os.getenv("BLS_POLL_SECONDS", str(WEB_POLL_SECONDS)))
BLS_DENIED_MARKERS = ("Access Denied", "Bureau of Labor Statistics")


class BlsWatcher(RssWatcher):
    def __init__(self, source_name: str, feed_url: str, interval_seconds: int = BLS_POLL_SECONDS):
        super().__init__(source_name, feed_url, interval_seconds, category="economic_release")
        cookie_header = clean_text(os.getenv("BLS_COOKIE_HEADER", ""))
        if cookie_header:
            self.session.headers.update({"Cookie": cookie_header})

    def _fetch_events(self):
        try:
            resp = self._fetch_feed_response()
        except requests.HTTPError as e:
            response = getattr(e, "response", None)
            if response is not None and response.status_code == 403 and all(marker in (response.text or "") for marker in BLS_DENIED_MARKERS):
                raise RuntimeError(
                    "BLS denied access to the configured feed; set BLS_COOKIE_HEADER from a browser session or use another allowed BLS endpoint"
                ) from e
            raise
        if resp.status_code == 403 and all(marker in (resp.text or "") for marker in BLS_DENIED_MARKERS):
            raise RuntimeError(
                "BLS denied access to the configured feed; set BLS_COOKIE_HEADER from a browser session or use another allowed BLS endpoint"
            )
        root = self._parse_xml_root(resp)
        items, feed_kind = self._iter_items(root)

        events: list[Event] = []
        for item in items:
            if feed_kind == "atom":
                title = clean_text(item.findtext("{http://www.w3.org/2005/Atom}title", default=""))
                url = ""
                for link in item.findall("{http://www.w3.org/2005/Atom}link"):
                    href = clean_text(link.attrib.get("href", ""))
                    rel = clean_text(link.attrib.get("rel", "alternate"))
                    if href and rel in {"", "alternate"}:
                        url = href
                        break
                    if href and not url:
                        url = href
                guid = clean_text(item.findtext("{http://www.w3.org/2005/Atom}id", default=""))
                description = clean_text(item.findtext("{http://www.w3.org/2005/Atom}summary", default=""))
                if not description:
                    description = clean_text(item.findtext("{http://www.w3.org/2005/Atom}content", default=""))
                pub_date_raw = clean_text(item.findtext("{http://www.w3.org/2005/Atom}updated", default=""))
                if not pub_date_raw:
                    pub_date_raw = clean_text(item.findtext("{http://www.w3.org/2005/Atom}published", default=""))
            else:
                title = clean_text(item.findtext("title", default=""))
                url = clean_text(item.findtext("link", default=""))
                guid = clean_text(item.findtext("guid", default=""))
                description = clean_text(item.findtext("description", default=""))
                pub_date_raw = clean_text(item.findtext("pubDate", default=""))

            published_at = None
            if pub_date_raw:
                try:
                    published_at = parsedate_to_datetime(pub_date_raw).astimezone().isoformat()
                except Exception:
                    published_at = pub_date_raw

            if not title or not url:
                continue

            item_id = guid or stable_id(self.source_name, url, title)
            events.append(
                Event(
                    source=self.source_name,
                    item_id=item_id,
                    title=title,
                    url=url,
                    published_at=published_at,
                    summary=description,
                    category=self.category,
                    raw={
                        "feed_url": self.feed_url,
                        "content_type": resp.headers.get("Content-Type", ""),
                        "feed_kind": feed_kind,
                    },
                )
            )
        return events

    def _fetch_feed_response(self):
        accept = "application/rss+xml, application/xml, text/xml, application/atom+xml;q=0.9, */*;q=0.1"
        try:
            return browser_impersonated_get(self.feed_url, headers=dict(self.session.headers), accept=accept)
        except Exception:
            return fetch_with_retries(self.session, self.feed_url, accept=accept)


def build_bls_watchers() -> list[BlsWatcher]:
    feeds = parse_csv_env("BLS_FEEDS", DEFAULT_BLS_FEEDS)
    total = len(feeds)
    return [BlsWatcher(make_instance_source_name("bls", i, total), feed) for i, feed in enumerate(feeds)]
