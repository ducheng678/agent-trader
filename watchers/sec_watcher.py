from .common import RssWatcher, WEB_POLL_SECONDS, make_instance_source_name, parse_csv_env

DEFAULT_SEC_FEED = "https://www.sec.gov/news/pressreleases.rss"


class SecWatcher(RssWatcher):
    def __init__(self, source_name: str, feed_url: str, interval_seconds: int = WEB_POLL_SECONDS):
        super().__init__(source_name, feed_url, interval_seconds, category="press_release")


def build_sec_watchers() -> list[SecWatcher]:
    feeds = parse_csv_env("SEC_FEEDS", DEFAULT_SEC_FEED)
    total = len(feeds)
    return [SecWatcher(make_instance_source_name("sec", i, total), feed) for i, feed in enumerate(feeds)]
