import os

from .common import WEB_POLL_SECONDS, env_flag_enabled, make_instance_source_name, parse_csv_env
from .telegram_channel_watcher import TelegramChannelWatcher


class FarsWatcher(TelegramChannelWatcher):
    pass


def build_fars_watchers() -> list[FarsWatcher]:
    if not env_flag_enabled("FARS_ENABLED", True):
        return []
    urls = parse_csv_env("FARS_URLS", "")
    if not urls:
        return []
    total = len(urls)
    interval = int(os.getenv("FARS_POLL_SECONDS", str(WEB_POLL_SECONDS)))
    return [FarsWatcher(make_instance_source_name("fars", i, total), url, interval) for i, url in enumerate(urls)]
