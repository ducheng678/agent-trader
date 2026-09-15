import os
import re
import time
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

from .common import (
    BaseWatcher,
    Event,
    OUTPUT_ROOT,
    REQUEST_TIMEOUT,
    StateStore,
    WARMUP_PREVIEW_COUNT,
    apply_warmup_mode,
    bounded_timeout_seconds,
    clean_text,
    html_to_text,
    parse_csv_env,
)
from .proxy_failover import load_source_proxy_config


load_dotenv()

TRUTH_POLL_SECONDS = int(os.getenv("TRUTH_POLL_SECONDS", "3"))
TRUTH_MEDIA_DIR = Path(os.getenv("TRUTH_MEDIA_DIR", str(OUTPUT_ROOT / "truth_media")))
TRUTH_MEDIA_TIMEOUT = int(os.getenv("TRUTH_MEDIA_TIMEOUT", "30"))
TRUTH_BASE_URL = os.getenv("TRUTHSOCIAL_BASE_URL", "https://truthsocial.com").rstrip("/")
TRUTH_API_BASE_URL = os.getenv("TRUTHSOCIAL_API_BASE_URL", f"{TRUTH_BASE_URL}/api").rstrip("/")
TRUTH_USER_AGENT = os.getenv(
    "TRUTHSOCIAL_USER_AGENT",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 12_2_1) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36",
)
TRUTH_BLOCK_MARKERS = ("you have been blocked", "attention required", "cloudflare")
TRUTH_GEOBLOCK_MARKERS = ("unavailable in your area", "not available in your area")

CONTENT_TYPE_TO_EXT = {
    "image/jpeg": ".jpg",
    "image/jpg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
    "image/bmp": ".bmp",
    "image/tiff": ".tiff",
    "video/mp4": ".mp4",
    "video/webm": ".webm",
}


class TruthSocialApiError(RuntimeError):
    pass


class TruthSocialAuthError(TruthSocialApiError):
    pass


class TruthSocialGeoblockError(TruthSocialApiError):
    pass


class TruthSocialBlockError(TruthSocialApiError):
    pass


class TruthSocialRequestsClient:
    def __init__(self, token: str, *, base_url: str = TRUTH_BASE_URL, api_base_url: str = TRUTH_API_BASE_URL):
        if not token:
            raise RuntimeError("TRUTHSOCIAL_TOKEN not found in environment or .env")
        self.base_url = base_url.rstrip("/")
        self.api_base_url = api_base_url.rstrip("/")
        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {token}",
                "User-Agent": TRUTH_USER_AGENT,
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": f"{self.base_url}/",
            }
        )
        cookie_header = clean_text(os.getenv("TRUTHSOCIAL_COOKIE_HEADER", ""))
        if cookie_header:
            self.session.headers.update({"Cookie": cookie_header})

    def lookup(self, handle: str) -> dict[str, Any]:
        payload = self._request_json("/v1/accounts/lookup", params={"acct": handle})
        if not isinstance(payload, dict) or "id" not in payload:
            raise TruthSocialApiError(f"lookup failed for @{handle}: {payload!r}")
        return payload

    def list_statuses(self, user_id: str, *, exclude_replies: bool = True) -> list[dict[str, Any]]:
        payload = self._request_json(
            f"/v1/accounts/{user_id}/statuses",
            params={"exclude_replies": "true" if exclude_replies else "false"},
        )
        if payload is None:
            return []
        if isinstance(payload, dict):
            raise TruthSocialApiError(f"statuses returned dict instead of list: {payload!r}")
        if not isinstance(payload, list):
            raise TruthSocialApiError(f"statuses returned unexpected type: {type(payload)!r}")
        return payload

    def _request_json(self, path: str, *, params: Optional[dict[str, str]] = None) -> Any:
        url = self._api_url(path)
        try:
            response = self.session.get(url, params=params, timeout=bounded_timeout_seconds(REQUEST_TIMEOUT))
        except requests.RequestException as e:
            raise TruthSocialApiError(f"Truth Social request failed for {url}: {e}") from e
        self._raise_for_truth_errors(response)
        try:
            payload = response.json()
        except ValueError as e:
            snippet = clean_text((response.text or "")[:200])
            raise TruthSocialApiError(f"Truth Social returned non-JSON for {url}: {snippet}") from e
        if isinstance(payload, dict) and payload.get("error"):
            raise TruthSocialApiError(f"Truth Social API error for {url}: {payload['error']}")
        return payload

    def _raise_for_truth_errors(self, response: requests.Response) -> None:
        text = (response.text or "").lower()
        if response.status_code == 401:
            raise TruthSocialAuthError("Truth Social rejected the bearer token with 401 Unauthorized")
        if response.status_code == 403:
            if any(marker in text for marker in TRUTH_GEOBLOCK_MARKERS):
                raise TruthSocialGeoblockError("Truth Social is unavailable in this geographic region")
            if any(marker in text for marker in TRUTH_BLOCK_MARKERS):
                raise TruthSocialBlockError(
                    "Truth Social blocked the request with an anti-bot or Cloudflare page; "
                    "set TRUTHSOCIAL_COOKIE_HEADER from a browser session if plain requests are blocked in this environment"
                )
            raise TruthSocialAuthError("Truth Social returned 403 Forbidden for the configured bearer token")
        try:
            response.raise_for_status()
        except requests.HTTPError as e:
            raise TruthSocialApiError(f"Truth Social request failed with HTTP {response.status_code}") from e

    def _api_url(self, path: str) -> str:
        return f"{self.api_base_url}{path}"


class TruthSocialRequestsWatcher(BaseWatcher):
    def __init__(self, handle: str, interval_seconds: int = TRUTH_POLL_SECONDS):
        self.handle = handle.lstrip("@")
        self._user_id: Optional[str] = None
        source_name = f"truth_social:{self.handle}"
        super().__init__(source_name, interval_seconds)
        token = os.getenv("TRUTHSOCIAL_TOKEN", "").strip()
        self.client = TruthSocialRequestsClient(token)
        proxy_config = load_source_proxy_config(self.source_name)
        self.media_session = requests.Session()
        self.media_session.headers.update(
            {
                "User-Agent": TRUTH_USER_AGENT,
                "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": f"{TRUTH_BASE_URL}/",
            }
        )
        cookie_header = clean_text(os.getenv("TRUTHSOCIAL_COOKIE_HEADER", ""))
        if cookie_header:
            self.media_session.headers.update({"Cookie": cookie_header})
        if proxy_config.http_proxy or proxy_config.https_proxy:
            self.media_session.proxies.update(
                {
                    "http": proxy_config.http_proxy or proxy_config.https_proxy,
                    "https": proxy_config.https_proxy or proxy_config.http_proxy,
                }
            )

    def warmup(self, state: StateStore) -> list[Event]:
        posts = self._safe_fetch_statuses()
        if not posts:
            print(f"[warmup][{self.source_name}] no posts found")
            return []

        originals_desc = sorted(
            [p for p in posts if not self._is_repost(p)],
            key=lambda p: p.get("created_at", ""),
            reverse=True,
        )
        print(f"[warmup][{self.source_name}] latest {min(WARMUP_PREVIEW_COUNT, len(originals_desc))} original posts:")
        for post in originals_desc[:WARMUP_PREVIEW_COUNT]:
            summary = self._summarize_post_line(post)
            print(f"  id={post.get('id')} time={post.get('created_at')} text={summary}")
            attachments = self._extract_attachments(post)
            for idx, media in enumerate(attachments, start=1):
                media_url = media.get("preview_url") or media.get("url") or ""
                if media_url:
                    print(f"    media[{idx}] {media.get('type', 'unknown')}: {media_url}")
            try:
                downloaded = self._download_images_for_post(post)
                if downloaded:
                    print("    downloaded_images:")
                    for path in downloaded:
                        print(f"      {path.resolve()}")
            except Exception as e:
                print(f"    [WARN] warmup image download failed: {e}")

        cursor_value = str(max(int(p["id"]) for p in posts))
        emitted = apply_warmup_mode(
            self.source_name,
            state,
            [str(p.get("id")) for p in originals_desc],
            emitted_events=[self._post_to_event(post) for post in reversed(originals_desc)],
            cursor_value=cursor_value,
        )
        print(f"[warmup][{self.source_name}] chosen since_id = {cursor_value}")
        return emitted

    def poll(self, state: StateStore) -> list[Event]:
        since_id = state.get_cursor(self.source_name)
        posts = self._safe_fetch_statuses(since_id=since_id)
        self.last_poll_at = time.time()
        if not posts:
            return []

        state.set_cursor(self.source_name, str(max(int(p["id"]) for p in posts)))
        events: list[Event] = []
        for post in posts:
            if self._is_repost(post):
                continue
            item_id = str(post.get("id"))
            if state.is_seen(self.source_name, item_id):
                continue
            events.append(self._post_to_event(post))
            state.mark_seen(self.source_name, item_id)
        return events

    def _safe_fetch_statuses(self, since_id: Optional[str] = None) -> list[dict[str, Any]]:
        user_id = self._user_id
        if not user_id:
            user = self.client.lookup(self.handle)
            user_id = str(user["id"])
            self._user_id = user_id
        posts = sorted(self.client.list_statuses(user_id), key=lambda x: int(x["id"]))
        if since_id is not None:
            posts = [p for p in posts if int(p["id"]) > int(since_id)]
        return posts

    @staticmethod
    def _is_repost(post: dict[str, Any]) -> bool:
        return post.get("reblog") is not None

    @staticmethod
    def _extract_media_items(post: dict[str, Any]) -> list[dict[str, Any]]:
        items = post.get("media_attachments") or []
        return items if isinstance(items, list) else []

    def _extract_attachments(self, post: dict[str, Any]) -> list[dict[str, Any]]:
        items = self._extract_media_items(post)
        return [
            {
                "id": media.get("id"),
                "type": media.get("type"),
                "url": media.get("url") or media.get("remote_url") or "",
                "preview_url": media.get("preview_url") or "",
                "description": media.get("description") or "",
            }
            for media in items
        ]

    def _post_to_event(self, post: dict[str, Any]) -> Event:
        text = html_to_text(post.get("content", ""))
        attachments = self._extract_attachments(post)
        local_files: list[str] = []
        try:
            local_files = [str(path.resolve()) for path in self._download_images_for_post(post)]
        except Exception as e:
            print(f"[WARN] image download failed for {self.source_name} post {post.get('id')}: {e}")

        title = text[:140] if text else self._summarize_post_line(post)
        summary = text if text else self._summarize_post_line(post)
        return Event(
            source=self.source_name,
            item_id=str(post.get("id")),
            title=title,
            url=post.get("url") or f"{TRUTH_BASE_URL}/@{self.handle}/posts/{post.get('id')}",
            published_at=post.get("created_at"),
            summary=summary,
            category="truth_social_post",
            attachments=attachments,
            local_files=local_files,
            raw={
                "handle": self.handle,
                "content_html": post.get("content", ""),
                "account": post.get("account", {}),
            },
        )

    def _download_images_for_post(self, post: dict[str, Any]) -> list[Path]:
        media_items = self._extract_media_items(post)
        if not media_items:
            return []
        post_id = str(post.get("id") or "unknown_post")
        created_at = str(post.get("created_at") or "unknown_time").replace(":", "-")
        post_dir = TRUTH_MEDIA_DIR / self.handle / f"{created_at}_{post_id}"
        saved_paths: list[Path] = []

        for i, media in enumerate(media_items, start=1):
            media_type = str(media.get("type") or "unknown").lower()
            if media_type != "image":
                continue
            media_urls = self._media_download_urls(media)
            if not media_urls:
                continue
            media_id = str(media.get("id") or i)
            filename = self._safe_filename(f"{i:02d}_{media_id}")
            last_error: Optional[Exception] = None
            for media_url in media_urls:
                ext = self._infer_extension_from_url(media_url)
                dest_path = post_dir / f"{filename}{ext}"
                if not dest_path.suffix:
                    dest_path = post_dir / filename
                if dest_path.exists():
                    saved_paths.append(dest_path)
                    last_error = None
                    break
                try:
                    saved_paths.append(self._download_binary_file(media_url, dest_path))
                    last_error = None
                    break
                except requests.RequestException as exc:
                    last_error = exc
                    continue
            if last_error is not None:
                raise last_error
        return saved_paths

    @staticmethod
    def _media_download_urls(media: dict[str, Any]) -> list[str]:
        urls: list[str] = []
        for key in ("url", "remote_url", "preview_url"):
            value = str(media.get(key) or "").strip()
            if value and value not in urls:
                urls.append(value)
        return urls

    def _download_binary_file(self, url: str, dest_path: Path, timeout: int = TRUTH_MEDIA_TIMEOUT) -> Path:
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        with self.media_session.get(url, stream=True, timeout=bounded_timeout_seconds(timeout)) as response:
            response.raise_for_status()
            content_type = response.headers.get("Content-Type", "")
            inferred_ext = self._infer_extension_from_content_type(content_type)
            final_path = dest_path
            if not final_path.suffix and inferred_ext:
                final_path = final_path.with_suffix(inferred_ext)
            with open(final_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
        return final_path

    @staticmethod
    def _summarize_post_line(post: dict[str, Any]) -> str:
        text = html_to_text(post.get("content", ""))
        if text:
            return clean_text(text)[:120]
        media_items = TruthSocialRequestsWatcher._extract_media_items(post)
        if media_items:
            media_types = ", ".join((str(m.get("type", "unknown")) for m in media_items))
            return f"[media-only post: {len(media_items)} attachment(s), types={media_types}]"
        return "[empty post]"

    @staticmethod
    def _safe_filename(name: str) -> str:
        name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
        return name or "file"

    @staticmethod
    def _infer_extension_from_url(url: str) -> str:
        path = urlparse(url).path
        suffix = Path(path).suffix.lower()
        return suffix if suffix and len(suffix) <= 10 else ""

    @staticmethod
    def _infer_extension_from_content_type(content_type: str) -> str:
        content_type = (content_type or "").split(";", 1)[0].strip().lower()
        return CONTENT_TYPE_TO_EXT.get(content_type, "")


def build_truth_social_watchers() -> list[TruthSocialRequestsWatcher]:
    handles = parse_csv_env("TRUTHSOCIAL_HANDLES")
    if not handles:
        single = os.getenv("TRUTHSOCIAL_HANDLE", "").strip()
        if single:
            handles = [single]
    if not handles:
        return []
    return [TruthSocialRequestsWatcher(handle=h) for h in handles]
