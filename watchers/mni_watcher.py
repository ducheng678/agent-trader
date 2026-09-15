import os
import re
import time
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from .common import (
    Event,
    HtmlPageWatcher,
    REQUEST_TIMEOUT,
    USER_AGENT,
    WEB_POLL_SECONDS,
    WARMUP_PREVIEW_COUNT,
    apply_warmup_mode,
    best_effort_html_summary,
    browser_impersonated_get,
    bounded_timeout_seconds,
    clean_text,
    make_instance_source_name,
    parse_csv_env,
    stable_id,
)

DEFAULT_MNI_URL = "https://www.mnimarkets.com/"
MNI_ARTICLE_PATH_RE = re.compile(r"^/articles/[a-z0-9-]+(?:-\d+)?/?$", re.IGNORECASE)
MNI_EPOCH_MS_RE = re.compile(r"-(\d{13})(?:\D|$)")
MNI_EPOCH_MS_TOKEN_RE = re.compile(r"(?:^|[-_:])\d{13}(?:$|[-_:])")
MNI_UUID_RE = re.compile(r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.IGNORECASE)
MNI_COMPACT_HEX_ID_RE = re.compile(r"^[0-9a-f]{24,64}$", re.IGNORECASE)
MNI_INTERNAL_PREFIX_CHARS_RE = re.compile(r"^[a-z0-9_.:-]+$", re.IGNORECASE)
MNI_TITLE_DATE_RE = re.compile(r"-\s*(\d{2})-(\d{2})-(\d{4})\s*\|\s*MNI", re.IGNORECASE)
MNI_DISPLAY_TIME_RE = re.compile(r"\b([A-Z][a-z]{2})-(\d{1,2})\s+(\d{1,2}):(\d{2})\b")
MNI_SUMMARY_META_ATTRS = (
    ("name", "description"),
    ("property", "og:description"),
    ("name", "twitter:description"),
)
MNI_MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}


def _format_utc_ms_timestamp(epoch_ms: int) -> str:
    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).isoformat(timespec="milliseconds").replace(
        "+00:00",
        "Z",
    )


def _infer_published_at_from_epoch_ms(*values: Optional[str]) -> Optional[str]:
    for value in values:
        if not value:
            continue
        match = MNI_EPOCH_MS_RE.search(value)
        if not match:
            continue
        try:
            return _format_utc_ms_timestamp(int(match.group(1)))
        except (OverflowError, OSError, ValueError):
            continue
    return None


def _infer_published_at_from_article_html(html: str) -> Optional[str]:
    published_at = _infer_published_at_from_epoch_ms(html)
    if published_at:
        return published_at

    soup = BeautifulSoup(html, "html.parser")
    title_text = clean_text(soup.title.get_text(" ", strip=True)) if soup.title else ""
    title_date = MNI_TITLE_DATE_RE.search(title_text)
    if not title_date:
        return None
    month_num, day, year = (int(part) for part in title_date.groups())

    time_text = ""
    for node in soup.find_all(string=MNI_DISPLAY_TIME_RE):
        time_text = clean_text(str(node))
        if time_text:
            break
    time_match = MNI_DISPLAY_TIME_RE.search(time_text)
    if not time_match:
        return None
    month_name, time_day, hour, minute = time_match.groups()
    if MNI_MONTHS.get(month_name.lower()) != month_num or int(time_day) != day:
        return None
    try:
        dt = datetime(year, month_num, day, int(hour), int(minute), tzinfo=timezone.utc)
    except ValueError:
        return None
    return dt.isoformat(timespec="minutes").replace("+00:00", "Z")


def _looks_like_mni_internal_prefix(prefix: str) -> bool:
    prefix = clean_text(prefix)
    if not prefix or len(prefix) > 180:
        return False
    if not MNI_INTERNAL_PREFIX_CHARS_RE.match(prefix):
        return False
    return bool(
        MNI_UUID_RE.search(prefix) or MNI_EPOCH_MS_TOKEN_RE.search(prefix) or MNI_COMPACT_HEX_ID_RE.match(prefix)
    )


def _clean_mni_summary(summary: str) -> str:
    text = clean_text(summary or "")
    prefix, separator, body = text.partition(" - ")
    if separator and _looks_like_mni_internal_prefix(prefix):
        return body.strip()
    return text


def _best_mni_metadata_summary(html: str) -> str:
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    summaries: list[str] = []
    seen: set[str] = set()
    for attr_name, attr_value in MNI_SUMMARY_META_ATTRS:
        node = soup.find("meta", attrs={attr_name: attr_value})
        if node is None:
            continue
        summary = _clean_mni_summary(node.get("content", ""))
        if not summary or summary in seen:
            continue
        summaries.append(summary)
        seen.add(summary)
    if not summaries:
        return ""
    return max(summaries, key=len).rstrip()


class MniWatcher(HtmlPageWatcher):
    def __init__(self, source_name: str, url: str, interval_seconds: int = WEB_POLL_SECONDS):
        super().__init__(source_name, url, interval_seconds)
        self.direct_session = requests.Session()
        self.direct_session.trust_env = False
        self.direct_session.headers.update({"User-Agent": USER_AGENT})
        self.summary_cache: dict[str, str] = {}
        self.published_at_cache: dict[str, Optional[str]] = {}

    @staticmethod
    def _validate_mni_html(url: str, html: str) -> None:
        text = html or ""
        if not text.strip():
            raise RuntimeError("MNI returned empty HTML")
        if 'id="html-root"' not in text and "id='html-root'" not in text:
            raise RuntimeError("MNI returned non-standard/block HTML")
        parsed = urlparse(url)
        if parsed.path in {"", "/"} and "/articles/" not in text:
            raise RuntimeError("MNI homepage returned no article links")

    def _fetch_mni_html_once(
        self,
        url: str,
        *,
        accept: str,
        session: requests.Session,
        proxies: Optional[dict[str, str]],
    ) -> str:
        try:
            resp = browser_impersonated_get(
                url,
                accept=accept,
                proxies=proxies,
            )
        except Exception:
            resp = session.get(
                url,
                headers={"Accept": accept},
                timeout=bounded_timeout_seconds(REQUEST_TIMEOUT),
            )
        resp.raise_for_status()
        html = resp.text
        self._validate_mni_html(url, html)
        return html

    def _fetch_mni_html_via_proxy(
        self,
        url: str,
        *,
        accept: str,
        reason: Optional[BaseException] = None,
    ) -> str:
        last_err: Optional[BaseException] = None
        try:
            return self._fetch_mni_html_once(
                url,
                accept=accept,
                session=self.session,
                proxies=dict(self.session.proxies or {}),
            )
        except Exception as exc:
            last_err = exc

        failover = getattr(self.session, "_source_proxy_failover", None)
        if failover is None:
            if last_err is not None:
                raise last_err
            raise RuntimeError("MNI proxy fetch failed")

        max_rotations = max(0, int(getattr(self.session, "_source_proxy_failover_max_rotations", 0) or 0))
        if max_rotations <= 0:
            if last_err is not None:
                raise last_err
            raise RuntimeError("MNI proxy fetch failed")

        original_selection = ""
        try:
            group_detail = failover.get_proxy_detail(failover.selector_group)
            all_names = [str(item).strip() for item in list(group_detail.get("all") or []) if str(item).strip()]
            concrete_nodes = [
                name
                for name in all_names
                if failover._is_concrete_node(name) and failover._is_leaf_proxy(name)
            ]
            original_selection = str(group_detail.get("now") or "").strip()
            current_selection = failover._resolve_nested_current_selection(original_selection)
            candidates = failover._ordered_failover_candidates(concrete_nodes, current_selection)
        except Exception as exc:
            if last_err is not None:
                raise last_err
            raise exc

        reason_text = type(reason or last_err).__name__ if isinstance(reason or last_err, BaseException) else "error"
        tested = 0
        for candidate in candidates:
            if tested >= max_rotations:
                break
            tested += 1
            try:
                failover.set_selector(failover.selector_group, candidate)
                html = self._fetch_mni_html_once(
                    url,
                    accept=accept,
                    session=self.session,
                    proxies=dict(self.session.proxies or {}),
                )
                print(
                    f"[source_proxy_failover][{self.source_name}] switched "
                    f"{failover.selector_group} -> {candidate} (mni=200) after {reason_text}"
                )
                return html
            except Exception as exc:
                last_err = exc
                continue

        if original_selection:
            try:
                failover.set_selector(failover.selector_group, original_selection)
            except Exception:
                pass
        if last_err is not None:
            raise last_err
        raise RuntimeError("MNI proxy fetch failed")

    def _fetch_mni_html(self, url: str, *, accept: str = "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1") -> str:
        direct_err: Optional[BaseException] = None
        try:
            return self._fetch_mni_html_once(
                url,
                accept=accept,
                session=self.direct_session,
                proxies=None,
            )
        except Exception as exc:
            direct_err = exc
        if not self.session.proxies:
            raise direct_err
        return self._fetch_mni_html_via_proxy(url, accept=accept, reason=direct_err)

    def _fetch_html(self) -> str:
        return self._fetch_mni_html(self.url)

    def _fetch_article_summary(self, url: str) -> str:
        if url in self.summary_cache:
            return self.summary_cache[url]
        summary = ""
        published_at = _infer_published_at_from_epoch_ms(url)
        try:
            html = self._fetch_mni_html(
                url,
                accept="text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
            )
            summary = _best_mni_metadata_summary(html) or _clean_mni_summary(best_effort_html_summary(html))
            published_at = published_at or _infer_published_at_from_article_html(html)
        except Exception:
            summary = ""
        self.summary_cache[url] = summary
        self.published_at_cache[url] = published_at
        return summary

    def _enrich_summaries(self, items: list[Event], *, limit: Optional[int] = None) -> list[Event]:
        target_items = items if limit is None or limit <= 0 else items[:limit]
        for item in target_items:
            if item.published_at is None and item.url:
                item.published_at = _infer_published_at_from_epoch_ms(item.url, item.summary)
            if item.summary or not item.url:
                item.summary = _clean_mni_summary(item.summary)
                continue
            item.summary = self._fetch_article_summary(item.url)
            if item.published_at is None:
                item.published_at = self.published_at_cache.get(item.url) or _infer_published_at_from_epoch_ms(
                    item.url,
                    item.summary,
                )
        return items

    def extract_events(self, html: str) -> list[Event]:
        soup = BeautifulSoup(html, "html.parser")
        seen_urls: set[str] = set()
        items: list[Event] = []
        for a in soup.find_all("a", href=True):
            href = urljoin(self.url, a["href"])
            parsed = urlparse(href)
            if parsed.netloc not in {"www.mnimarkets.com", "mnimarkets.com"}:
                continue
            if not MNI_ARTICLE_PATH_RE.match(parsed.path):
                continue
            title = clean_text(a.get_text(" ", strip=True))
            if len(title) < 10:
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
                    published_at=_infer_published_at_from_epoch_ms(parsed.path),
                    category="market_research",
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


def build_mni_watchers() -> list[MniWatcher]:
    urls = parse_csv_env("MNI_URLS", DEFAULT_MNI_URL)
    total = len(urls)
    interval = int(os.getenv("MNI_POLL_SECONDS", str(WEB_POLL_SECONDS)))
    return [MniWatcher(make_instance_source_name("mni", i, total), url, interval) for i, url in enumerate(urls)]
