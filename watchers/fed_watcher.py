from .common import RssWatcher, WEB_POLL_SECONDS, make_instance_source_name, parse_csv_env

DEFAULT_FED_FEED = "https://www.federalreserve.gov/feeds/press_all.xml"


class FedWatcher(RssWatcher):
    def __init__(self, source_name: str, feed_url: str, interval_seconds: int = WEB_POLL_SECONDS):
        super().__init__(source_name, feed_url, interval_seconds, category="press_release")


def build_fed_watchers() -> list[FedWatcher]:
    feeds = parse_csv_env("FED_FEEDS", DEFAULT_FED_FEED)
    total = len(feeds)
    return [FedWatcher(make_instance_source_name("fed", i, total), feed) for i, feed in enumerate(feeds)]
