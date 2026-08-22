import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from truthbrush import Api, LoginErrorException, GeoblockException, CFBlockException


load_dotenv()


DOWNLOAD_ROOT = Path(os.getenv("TRUTH_MEDIA_DIR", "./downloads/truth_media"))
DOWNLOAD_TIMEOUT_SECONDS = 30


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


def html_to_text(html: str) -> str:
    if not html:
        return ""
    return BeautifulSoup(html, "html.parser").get_text(" ", strip=True)


def is_repost(post: dict) -> bool:
    # Truth Social / Mastodon 风格：转发帖会带 reblog 字段
    return post.get("reblog") is not None


def extract_media_items(post: dict) -> list[dict]:
    items = post.get("media_attachments") or []
    if not isinstance(items, list):
        return []
    return items


def safe_filename(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return name or "file"


def infer_extension_from_url(url: str) -> str:
    path = urlparse(url).path
    suffix = Path(path).suffix.lower()
    if suffix and len(suffix) <= 10:
        return suffix
    return ""


def infer_extension_from_content_type(content_type: str) -> str:
    if not content_type:
        return ""
    content_type = content_type.split(";", 1)[0].strip().lower()
    return CONTENT_TYPE_TO_EXT.get(content_type, "")


def choose_media_download_url(media: dict) -> str:
    return (
        media.get("url")
        or media.get("remote_url")
        or media.get("preview_url")
        or ""
    )


def download_binary_file(url: str, dest_path: Path, timeout: int = DOWNLOAD_TIMEOUT_SECONDS) -> Path:
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    request = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0",
            "Accept": "*/*",
        },
    )

    with urlopen(request, timeout=timeout) as response:
        content_type = response.headers.get("Content-Type", "")
        inferred_ext = infer_extension_from_content_type(content_type)

        final_path = dest_path
        if not final_path.suffix and inferred_ext:
            final_path = final_path.with_suffix(inferred_ext)

        with open(final_path, "wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)

    return final_path


def download_images_for_post(post: dict, root_dir: Path = DOWNLOAD_ROOT) -> list[Path]:
    media_items = extract_media_items(post)
    if not media_items:
        return []

    post_id = str(post.get("id") or "unknown_post")
    created_at = str(post.get("created_at") or "unknown_time").replace(":", "-")
    post_dir = root_dir / f"{created_at}_{post_id}"

    saved_paths: list[Path] = []

    for i, media in enumerate(media_items, start=1):
        media_type = str(media.get("type") or "unknown").lower()
        if media_type != "image":
            continue

        media_url = choose_media_download_url(media)
        if not media_url:
            continue

        media_id = str(media.get("id") or i)
        ext = infer_extension_from_url(media_url)
        filename = safe_filename(f"{i:02d}_{media_id}")
        dest_path = post_dir / f"{filename}{ext}"

        # 避免重复下载同一个文件
        if dest_path.exists():
            saved_paths.append(dest_path)
            continue

        if not dest_path.suffix:
            # 先无后缀下载，后面会根据 Content-Type 自动补后缀
            dest_path = post_dir / filename

        saved_path = download_binary_file(media_url, dest_path)
        saved_paths.append(saved_path)

    return saved_paths


def print_media(post: dict) -> None:
    media_items = extract_media_items(post)
    if not media_items:
        return

    print("media:")
    for i, media in enumerate(media_items, start=1):
        media_type = media.get("type", "unknown")
        url = media.get("url") or media.get("remote_url") or ""
        preview_url = media.get("preview_url") or ""
        desc = media.get("description") or ""

        print(f"  [{i}] type={media_type}")
        if url:
            print(f"      url        : {url}")
        if preview_url:
            print(f"      preview_url: {preview_url}")
        if desc:
            print(f"      desc       : {desc}")



def summarize_post_line(post: dict) -> str:
    text = html_to_text(post.get("content", ""))
    media_items = extract_media_items(post)

    if text:
        return text[:120]

    if media_items:
        media_types = ", ".join((m.get("type", "unknown") for m in media_items))
        return f"[media-only post: {len(media_items)} attachment(s), types={media_types}]"

    return "[empty post]"



def print_post(post: dict) -> None:
    print("\n" + "=" * 80)
    print(f"[NEW POST] id={post.get('id')}")
    print(f"time: {post.get('created_at')}")
    if post.get("url"):
        print(f"url : {post['url']}")

    media_items = extract_media_items(post)
    if media_items:
        print(f"media_count: {len(media_items)}")

    print("text:")
    text = html_to_text(post.get("content", ""))
    print(text if text else "[no text]")

    print_media(post)

    try:
        downloaded = download_images_for_post(post)
        if downloaded:
            print("downloaded_images:")
            for path in downloaded:
                print(f"  {path.resolve()}")
    except Exception as e:
        print(f"[WARN] image download failed: {e}")

    print("=" * 80, flush=True)



def safe_fetch_statuses(api: Api, handle: str, since_id: Optional[str] = None):
    """
    不走 truthbrush 自带的 pull_statuses，自己安全处理 None / dict / 非 list 返回。
    这里不直接过滤 repost，因为 since_id 必须按“所有新帖”推进，避免漏游标。
    """
    user = api.lookup(handle)
    if not user or "id" not in user:
        print(f"[ERROR] lookup failed for @{handle}: {user}")
        return []

    user_id = user["id"]
    url = f"/v1/accounts/{user_id}/statuses?exclude_replies=true"
    result = api._get(url, params={})

    if result is None:
        print("[ERROR] statuses response is None (likely non-JSON or blocked page)")
        return []

    if isinstance(result, dict):
        print(f"[ERROR] statuses returned dict instead of list: {result}")
        return []

    if not isinstance(result, list):
        print(f"[ERROR] statuses returned unexpected type: {type(result)}")
        return []

    posts = sorted(result, key=lambda x: int(x["id"]))

    if since_id is not None:
        posts = [p for p in posts if int(p["id"]) > int(since_id)]

    return posts



def main():
    handle = "realDonaldTrump"
    poll_seconds = 3
    since_id: Optional[str] = None

    token = os.getenv("TRUTHSOCIAL_TOKEN")
    if not token:
        raise RuntimeError("TRUTHSOCIAL_TOKEN not found in .env")

    print(f"[{datetime.now().isoformat()}] starting watcher for @{handle}")
    print("[startup] token detected: yes")
    print(f"[startup] image download dir: {DOWNLOAD_ROOT.resolve()}")

    try:
        api = Api(token=token)
    except Exception as e:
        print(f"[ERROR] failed to initialize Api: {e}")
        return

    # warmup
    try:
        warmup_posts = safe_fetch_statuses(api, handle)
        if warmup_posts:
            warmup_posts = sorted(
                warmup_posts,
                key=lambda p: p.get("created_at", ""),
                reverse=True,
            )

            print("[warmup] latest 3 original posts:")
            shown = 0
            for p in warmup_posts:
                if is_repost(p):
                    continue
                summary = summarize_post_line(p)
                print(f"id={p.get('id')}  time={p.get('created_at')}  text={summary}")
                media_items = extract_media_items(p)
                for i, media in enumerate(media_items, start=1):
                    media_type = media.get("type", "unknown")
                    preview_url = media.get("preview_url") or media.get("url") or media.get("remote_url") or ""
                    if preview_url:
                        print(f"    media[{i}] {media_type}: {preview_url}")

                try:
                    downloaded = download_images_for_post(p)
                    if downloaded:
                        print("    downloaded_images:")
                        for path in downloaded:
                            print(f"      {path.resolve()}")
                except Exception as e:
                    print(f"    [WARN] warmup image download failed: {e}")

                shown += 1
                if shown >= 3:
                    break

            # since_id 必须按“所有帖子”推进，不只是原创帖
            since_id = str(max(int(p["id"]) for p in warmup_posts))
            print(f"[warmup] chosen since_id = {since_id}")
        else:
            print("[warmup] no posts found or response abnormal")
    except GeoblockException as e:
        print(f"[ERROR] geoblocked: {e}")
        return
    except CFBlockException as e:
        print(f"[ERROR] cloudflare blocked: {e}")
        return
    except LoginErrorException as e:
        print(f"[ERROR] login failed: {e}")
        return
    except Exception as e:
        print(f"[ERROR] warmup failed: {e}")
        return

    while True:
        try:
            posts = safe_fetch_statuses(api, handle, since_id=since_id)
            if posts:
                # 先推进 since_id，再过滤输出
                since_id = str(max(int(p["id"]) for p in posts))

                original_posts = [p for p in posts if not is_repost(p)]
                for post in original_posts:
                    print_post(post)

        except Exception as e:
            print(f"[ERROR] polling failed: {e}")

        time.sleep(poll_seconds)


if __name__ == "__main__":
    main()
