import os
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from .common import (
    Event,
    HtmlPageWatcher,
    WEB_POLL_SECONDS,
    clean_text,
    fetch_with_retries,
    make_instance_source_name,
    parse_csv_env,
    poll_deadline,
    stable_id,
)
from .telegram_channel_watcher import extract_telegram_public_page_events, is_telegram_public_page

DEFAULT_IRNA_URL = "https://en.irna.ir/"
DEFAULT_IRNA_TELEGRAM_URL = "https://t.me/s/Irna_en"
IRNA_ARTICLE_PATH_RE = re.compile(r"^/news/\d+(?:/[^?#]+)?/?$", re.IGNORECASE)
IRNA_CHALLENGE_MARKERS = ("Transferring to the website...", "__arcsjs", "__arcsjsc")


class IrnaWatcher(HtmlPageWatcher):
    def __init__(self, source_name: str, url: str, interval_seconds: int = WEB_POLL_SECONDS):
        super().__init__(source_name, url, interval_seconds)
        self.telegram_url = clean_text(os.getenv("IRNA_TELEGRAM_URL", DEFAULT_IRNA_TELEGRAM_URL)) or DEFAULT_IRNA_TELEGRAM_URL
        cookie_header = clean_text(os.getenv("IRNA_COOKIE_HEADER", ""))
        if cookie_header:
            self.session.headers.update({"Cookie": cookie_header})

    def _fetch_telegram_html(self) -> str:
        resp = fetch_with_retries(self.session, self.telegram_url)
        return resp.text

    def _fetch_parseable_telegram_html(self, failure_reason: str) -> str:
        telegram_html = self._fetch_telegram_html()
        if self._is_telegram_page(telegram_html):
            return telegram_html
        raise RuntimeError(f"IRNA {failure_reason} and Telegram fallback was not parseable")

    def _is_challenge_page(self, html: str) -> bool:
        return all(marker in html for marker in IRNA_CHALLENGE_MARKERS)

    def _is_telegram_page(self, html: str) -> bool:
        return is_telegram_public_page(html)

    def _site_fetch_timeout_seconds(self) -> float:
        return max(0.0, float(os.getenv("IRNA_SITE_FETCH_TIMEOUT_SECONDS", "8") or "8"))

    def _fetch_site_html(self) -> str:
        timeout_seconds = self._site_fetch_timeout_seconds()
        if timeout_seconds <= 0:
            return super()._fetch_html()
        with poll_deadline(timeout_seconds):
            return super()._fetch_html()

    def _fetch_html(self) -> str:
        if urlparse(self.url).netloc in {"t.me", "telegram.me"}:
            return self._fetch_telegram_html()
        try:
            html = self._fetch_site_html()
        except Exception as site_exc:
            try:
                return self._fetch_parseable_telegram_html("site fetch failed")
            except Exception as telegram_exc:
                raise RuntimeError(
                    f"IRNA site fetch failed ({type(site_exc).__name__}: {site_exc}) "
                    f"and Telegram fallback failed ({type(telegram_exc).__name__}: {telegram_exc})"
                ) from site_exc
        if self._is_challenge_page(html):
            return self._fetch_parseable_telegram_html("returned a JS/Cookie challenge page")
        return html

    def _extract_site_events(self, html: str) -> list[Event]:
        soup = BeautifulSoup(html, "html.parser")
        seen_urls: set[str] = set()
        items: list[Event] = []
        for a in soup.find_all("a", href=True):
            href = urljoin(self.url, a["href"])
            parsed = urlparse(href)
            if parsed.netloc not in {"en.irna.ir", "www.irna.ir", "irna.ir"}:
                continue
            if not IRNA_ARTICLE_PATH_RE.match(parsed.path):
                continue
            title = clean_text(a.get_text(" ", strip=True))
            if len(title) < 8:
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
                    category="iran_news",
                    raw={"path": parsed.path},
                )
            )
            if len(items) >= 40:
                break
        return items

    def _extract_telegram_events(self, html: str) -> list[Event]:
        return extract_telegram_public_page_events(
            html,
            source_name=self.source_name,
            discovered_via=self.telegram_url,
            category="iran_news",
        )

    def extract_events(self, html: str) -> list[Event]:
        if self._is_telegram_page(html):
            return self._extract_telegram_events(html)
        return self._extract_site_events(html)


def build_irna_watchers() -> list[IrnaWatcher]:
    urls = parse_csv_env("IRNA_URLS", "")
    if not urls:
        return []
    total = len(urls)
    interval = int(os.getenv("IRNA_POLL_SECONDS", str(WEB_POLL_SECONDS)))
    return [IrnaWatcher(make_instance_source_name("irna", i, total), url, interval) for i, url in enumerate(urls)]
