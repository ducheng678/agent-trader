from __future__ import annotations

import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import truthbrush.api as truthbrush_api


def test_truth_social_watcher_applies_source_specific_proxy(monkeypatch):
    monkeypatch.setenv("TRUTHSOCIAL_TOKEN", "test-token")
    monkeypatch.setenv("TRUTHSOCIAL_HTTP_PROXY", "socks5://127.0.0.1:1080")
    monkeypatch.setenv("TRUTHSOCIAL_HTTPS_PROXY", "socks5://127.0.0.1:1080")

    import watchers.truth_social_watcher as watcher_mod

    original = dict(getattr(truthbrush_api, "proxies", {}) or {})
    try:
        watcher = watcher_mod.TruthSocialWatcher("realDonaldTrump")
        assert watcher.api is not None
        assert truthbrush_api.proxies["http"] == "socks5://127.0.0.1:1080"
        assert truthbrush_api.proxies["https"] == "socks5://127.0.0.1:1080"
    finally:
        truthbrush_api.proxies = original


def test_truth_social_watcher_applies_source_specific_proxy_to_media_session(monkeypatch):
    monkeypatch.setenv("TRUTHSOCIAL_TOKEN", "test-token")
    monkeypatch.setenv("TRUTHSOCIAL_HTTP_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("TRUTHSOCIAL_HTTPS_PROXY", "http://127.0.0.1:7897")

    import watchers.truth_social_watcher as watcher_mod

    watcher = watcher_mod.TruthSocialWatcher("realDonaldTrump")

    assert watcher.media_session.proxies["http"] == "http://127.0.0.1:7897"
    assert watcher.media_session.proxies["https"] == "http://127.0.0.1:7897"
    assert watcher.media_session.headers["Referer"] == "https://truthsocial.com/"


def test_truth_social_watcher_falls_back_to_preview_image_after_403(monkeypatch, tmp_path):
    monkeypatch.setenv("TRUTHSOCIAL_TOKEN", "test-token")
    monkeypatch.setenv("TRUTH_MEDIA_DIR", str(tmp_path / "truth_media"))

    import watchers.truth_social_watcher as watcher_mod

    watcher_mod.TRUTH_MEDIA_DIR = tmp_path / "truth_media"
    watcher = watcher_mod.TruthSocialWatcher("realDonaldTrump")
    requested_urls: list[str] = []

    class DummyResponse:
        def __init__(self, url: str, status_code: int, body: bytes = b""):
            self.url = url
            self.status_code = status_code
            self.headers = {"Content-Type": "image/jpeg"}
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def raise_for_status(self):
            if self.status_code >= 400:
                response = requests.Response()
                response.status_code = self.status_code
                response.url = self.url
                raise requests.HTTPError(f"{self.status_code} error", response=response)

        def iter_content(self, chunk_size: int):
            yield self.body

    def fake_get(url: str, **kwargs):
        requested_urls.append(url)
        if url == "https://static.truthsocial.com/full.jpg":
            return DummyResponse(url, 403)
        return DummyResponse(url, 200, b"preview-image")

    monkeypatch.setattr(watcher.media_session, "get", fake_get)

    saved = watcher._download_images_for_post(
        {
            "id": "101",
            "created_at": "2026-04-25T17:00:00Z",
            "media_attachments": [
                {
                    "id": "media-1",
                    "type": "image",
                    "url": "https://static.truthsocial.com/full.jpg",
                    "preview_url": "https://static.truthsocial.com/preview.jpg",
                }
            ],
        }
    )

    assert requested_urls == [
        "https://static.truthsocial.com/full.jpg",
        "https://static.truthsocial.com/preview.jpg",
    ]
    assert len(saved) == 1
    assert saved[0].read_bytes() == b"preview-image"
