import re
from typing import Optional

from bs4 import BeautifulSoup

from .common import Event, HtmlPageWatcher, WEB_POLL_SECONDS, clean_text, make_instance_source_name, parse_csv_env, stable_id

DEFAULT_FACTBASE_URL = "https://rollcall.com/factbase/trump/topic/calendar/"
DATE_HEADING_RE = re.compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+[A-Za-z]+\s+\d{1,2}\s+\d{4}$")
DATE_WEEKDAY_RE = re.compile(r"^(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),$")
DATE_MONTHDAY_RE = re.compile(r"^[A-Za-z]+\s+\d{1,2}\s+\d{4}$")
TIME_RE = re.compile(r"^\d{1,2}:\d{2}\s*(AM|PM)$", re.IGNORECASE)
FACTBASE_SKIP_LINES = {
    "The White House",
    "Closed Press",
    "Open Press",
    "White House Press Pool",
    "In-Town Pool",
    "Out-of-Town Travel Pool",
    "Pre-Credentialed Media",
    "On Camera",
    "Pool Call Time",
    "In-Town Pool Call Time",
    "Out-of-Town Travel Pool Call Time",
}


class FactbaseWatcher(HtmlPageWatcher):
    def __init__(self, source_name: str, url: str, interval_seconds: int = WEB_POLL_SECONDS):
        super().__init__(source_name, url, interval_seconds)

    def extract_events(self, html: str) -> list[Event]:
        soup = BeautifulSoup(html, "html.parser")
        text = soup.get_text("\n", strip=True)
        raw_lines = [clean_text(line) for line in text.splitlines()]
        lines = [line for line in raw_lines if line]

        start_idx = 0
        for i, line in enumerate(lines):
            if "President's Public Schedule" in line:
                start_idx = i
                break
        lines = lines[start_idx : start_idx + 500]

        items: list[Event] = []
        seen_ids: set[str] = set()
        current_date: Optional[str] = None
        i = 0
        while i < len(lines):
            line = lines[i]
            if DATE_HEADING_RE.match(line):
                current_date = line
                i += 1
                continue
            if DATE_WEEKDAY_RE.match(line) and i + 1 < len(lines) and DATE_MONTHDAY_RE.match(lines[i + 1]):
                current_date = f"{line} {lines[i + 1]}"
                i += 2
                continue

            is_time = bool(TIME_RE.match(line)) or line.startswith("TBD:")
            if not is_time:
                i += 1
                continue

            time_text = line
            j = i + 1
            while j < len(lines) and lines[j] == time_text:
                j += 1

            title: Optional[str] = None
            location: Optional[str] = None
            press_type: Optional[str] = None
            k = j
            while k < len(lines):
                candidate = lines[k]
                if DATE_HEADING_RE.match(candidate) or TIME_RE.match(candidate) or candidate.startswith("TBD:"):
                    break
                if candidate in FACTBASE_SKIP_LINES:
                    if candidate not in {"The White House"}:
                        press_type = press_type or candidate
                    k += 1
                    continue
                if candidate == "Full Text and Analysis":
                    k += 1
                    continue
                if candidate.endswith("Full Text and Analysis"):
                    title = candidate.replace(" Full Text and Analysis", "").strip()
                    k += 1
                    continue
                if title is None:
                    title = candidate
                elif location is None and candidate != title:
                    location = candidate
                    break
                k += 1

            if current_date and title:
                item_id = stable_id(self.source_name, current_date, time_text, title)
                if item_id not in seen_ids:
                    seen_ids.add(item_id)
                    summary_parts = [current_date, time_text]
                    if location:
                        summary_parts.append(location)
                    if press_type:
                        summary_parts.append(press_type)
                    items.append(Event(source=self.source_name, item_id=item_id, title=title, url=self.url, summary=" | ".join(summary_parts), category="public_schedule", raw={"schedule_date": current_date, "time": time_text, "location": location, "press_type": press_type}))
            i = max(i + 1, k)
            if len(items) >= 40:
                break
        return items


def build_factbase_watchers() -> list[FactbaseWatcher]:
    urls = parse_csv_env("FACTBASE_URLS", DEFAULT_FACTBASE_URL)
    total = len(urls)
    return [FactbaseWatcher(make_instance_source_name("factbase", i, total), url) for i, url in enumerate(urls)]
