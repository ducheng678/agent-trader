import os
import re
import time
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

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

DEFAULT_ADP_URL = "https://adpemploymentreport.com/"
DEFAULT_ADP_REPORT_JSON = "ner_production.json"
MONTH_RE = re.compile(r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}$")
CHANGE_RE = re.compile(r"^(up|down)\s+\d[\d,]*$", re.IGNORECASE)


class AdpWatcher(BaseWatcher):
    def __init__(self, source_name: str, url: str, interval_seconds: int = WEB_POLL_SECONDS):
        super().__init__(source_name, interval_seconds)
        self.url = url
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": USER_AGENT})

    def _fetch_html(self) -> str:
        resp = fetch_with_retries(self.session, self.url)
        return resp.text

    def _report_json_url(self) -> str:
        explicit = os.getenv("ADP_REPORT_JSON_URL", "").strip()
        if explicit:
            return explicit
        path = os.getenv("ADP_REPORT_JSON_PATH", DEFAULT_ADP_REPORT_JSON).strip() or DEFAULT_ADP_REPORT_JSON
        return urljoin(self.url, path)

    def _fetch_report_json(self) -> dict[str, Any]:
        resp = fetch_with_retries(self.session, self._report_json_url(), accept="application/json")
        payload = resp.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"unexpected ADP report JSON payload: {type(payload)}")
        return payload

    def _extract_current_report(self, html: str) -> Event:
        soup = BeautifulSoup(html, "html.parser")
        tokens = [clean_text(item) for item in soup.stripped_strings if clean_text(item)]
        press_release_url = self.url
        for anchor in soup.find_all("a", href=True):
            text = clean_text(anchor.get_text(" ", strip=True)).lower()
            if "view press release" in text:
                press_release_url = urljoin(self.url, anchor["href"])
                break

        month_index = next((i for i, token in enumerate(tokens) if MONTH_RE.match(token)), -1)
        if month_index >= 0:
            focus_tokens = tokens[max(0, month_index - 5) : month_index + 20]
        else:
            heading_index = max((i for i, token in enumerate(tokens) if "national employment report" in token.lower()), default=0)
            focus_tokens = tokens[heading_index : heading_index + 40]

        report_month = next((token for token in focus_tokens if MONTH_RE.match(token)), "")
        headline = next((token for token in focus_tokens if token.lower().startswith("private employers ")), "")
        change_value = next((token for token in focus_tokens if CHANGE_RE.match(token)), "")
        if not change_value:
            for left, right in zip(focus_tokens, focus_tokens[1:]):
                candidate = f"{left} {right}"
                if CHANGE_RE.match(candidate):
                    change_value = candidate
                    break
        lead = ""
        if headline:
            headline_index = focus_tokens.index(headline)
            for token in focus_tokens[headline_index + 1 :]:
                lower = token.lower()
                if lower.startswith("view press release") or lower.startswith("subscribe"):
                    break
                if len(token) >= 20 and not MONTH_RE.match(token) and not CHANGE_RE.match(token):
                    lead = token
                    break

        if not headline:
            raise RuntimeError("could not locate current ADP report headline")

        summary_parts = [part for part in [report_month, change_value, lead] if part]
        summary = " | ".join(summary_parts) if summary_parts else headline
        item_key = press_release_url if press_release_url != self.url else f"{report_month}|{headline}"
        return Event(
            source=self.source_name,
            item_id=stable_id(self.source_name, item_key),
            title=headline,
            url=press_release_url,
            published_at=report_month or None,
            summary=summary,
            category="employment_report",
            raw={
                "report_month": report_month,
                "change_value": change_value,
                "page_url": self.url,
                "press_release_url": press_release_url,
            },
        )

    def _extract_current_report_json(self, payload: dict[str, Any]) -> Event:
        overview = payload.get("reportOverview") if isinstance(payload.get("reportOverview"), dict) else {}
        cards = overview.get("cards") if isinstance(overview.get("cards"), list) else []
        primary_card = next((card for card in cards if isinstance(card, dict)), {})

        report_month = clean_text(f"{payload.get('reportMonth', '')} {payload.get('reportYear', '')}")
        headline = clean_text(str(overview.get("title", "")))
        lead = clean_text(str(overview.get("description", "")))
        press_release_url = clean_text(str(payload.get("reportPressReleaseLink", ""))) or self.url

        metric_direction = clean_text(str(primary_card.get("metricDirection", ""))).lower()
        metric_value = clean_text(str(primary_card.get("metricValue", "")))
        change_value = clean_text(f"{metric_direction} {metric_value}") if metric_direction and metric_value else metric_value

        if not headline:
            raise RuntimeError("could not locate current ADP report headline")

        summary_parts = [part for part in [report_month, change_value, lead] if part]
        summary = " | ".join(summary_parts) if summary_parts else headline
        item_key = press_release_url if press_release_url != self.url else f"{report_month}|{headline}"
        return Event(
            source=self.source_name,
            item_id=stable_id(self.source_name, item_key),
            title=headline,
            url=press_release_url,
            published_at=report_month or None,
            summary=summary,
            category="employment_report",
            raw={
                "report_month": report_month,
                "change_value": change_value,
                "page_url": self.url,
                "json_url": self._report_json_url(),
                "press_release_url": press_release_url,
                "report_download_url": clean_text(str(payload.get("reportDownloadLink", ""))),
            },
        )

    def _fetch_events(self) -> list[Event]:
        html = self._fetch_html()
        try:
            return [self._extract_current_report(html)]
        except RuntimeError as exc:
            if "could not locate current ADP report headline" not in str(exc):
                raise
        return [self._extract_current_report_json(self._fetch_report_json())]

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


def build_adp_watchers() -> list[AdpWatcher]:
    urls = parse_csv_env("ADP_URLS", DEFAULT_ADP_URL)
    total = len(urls)
    interval = int(os.getenv("ADP_POLL_SECONDS", str(WEB_POLL_SECONDS)))
    return [AdpWatcher(make_instance_source_name("adp", i, total), url, interval) for i, url in enumerate(urls)]
