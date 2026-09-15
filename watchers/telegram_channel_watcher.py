import re

from bs4 import BeautifulSoup

from .common import Event, HtmlPageWatcher, WEB_POLL_SECONDS, clean_text, stable_id


TELEGRAM_PUBLIC_PAGE_MARKERS = ("tgme_widget_message_wrap", "tgme_widget_message_text")
TELEGRAM_TITLE_PREFIX_RE = re.compile(r"^[\W_]+", re.UNICODE)


def is_telegram_public_page(html: str) -> bool:
    return all(marker in html for marker in TELEGRAM_PUBLIC_PAGE_MARKERS)


def telegram_public_page_title(text_node, full_text: str) -> str:
    if text_node is not None:
        for bold_node in text_node.find_all("b"):
            candidate = clean_text(bold_node.get_text(" ", strip=True))
            candidate = TELEGRAM_TITLE_PREFIX_RE.sub("", candidate)
            if len(candidate) >= 8:
                return candidate[:160]
        for chunk in text_node.stripped_strings:
            candidate = clean_text(chunk)
            candidate = TELEGRAM_TITLE_PREFIX_RE.sub("", candidate)
            if not candidate or candidate.startswith("@"):
                continue
            if len(candidate) >= 8:
                return candidate[:160]
    fallback = TELEGRAM_TITLE_PREFIX_RE.sub("", full_text)
    return fallback[:160]


def extract_telegram_public_page_events(
    html: str,
    *,
    source_name: str,
    discovered_via: str,
    category: str = "iran_news",
    max_items: int = 40,
    raw: dict | None = None,
) -> list[Event]:
    soup = BeautifulSoup(html, "html.parser")
    raw_base = {"discovered_via": discovered_via, "fallback": "telegram_public_page"}
    if raw:
        raw_base.update(raw)
    seen_urls: set[str] = set()
    items: list[Event] = []
    for message in soup.select(".tgme_widget_message"):
        permalink_node = message.select_one(".tgme_widget_message_date")
        permalink = clean_text(permalink_node.get("href", "")) if permalink_node else ""
        if not permalink or permalink in seen_urls:
            continue
        text_node = message.select_one(".tgme_widget_message_text")
        text = clean_text(text_node.get_text(" ", strip=True)) if text_node else ""
        if len(text) < 8:
            continue
        published_at = ""
        time_node = message.select_one(".tgme_widget_message_date time")
        if time_node is not None:
            published_at = clean_text(time_node.get("datetime", ""))
        seen_urls.add(permalink)
        items.append(
            Event(
                source=source_name,
                item_id=stable_id(source_name, permalink),
                title=telegram_public_page_title(text_node, text),
                url=permalink,
                published_at=published_at or None,
                summary=text,
                category=category,
                raw=dict(raw_base),
            )
        )
        if len(items) >= max_items:
            break
    return items


class TelegramChannelWatcher(HtmlPageWatcher):
    def __init__(
        self,
        source_name: str,
        url: str,
        interval_seconds: int = WEB_POLL_SECONDS,
        *,
        category: str = "iran_news",
    ):
        super().__init__(source_name, url, interval_seconds)
        self.category = category

    def extract_events(self, html: str) -> list[Event]:
        if not is_telegram_public_page(html):
            return []
        return extract_telegram_public_page_events(
            html,
            source_name=self.source_name,
            discovered_via=self.url,
            category=self.category,
        )
