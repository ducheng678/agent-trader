from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from watchers.common import StateStore
from probe_protected_sources import ProbeResult, ProbeSpec, main as probe_main, probe_source
from watchers.truth_social_requests_watcher import TruthSocialAuthError, TruthSocialRequestsClient, TruthSocialRequestsWatcher


def test_truth_social_requests_watcher_poll_filters_reposts_and_seen(monkeypatch, tmp_path):
    monkeypatch.setenv("TRUTHSOCIAL_TOKEN", "test-token")
    monkeypatch.setenv("TRUTH_MEDIA_DIR", str(tmp_path / "truth_media"))

    watcher = TruthSocialRequestsWatcher("realDonaldTrump")
    monkeypatch.setattr(watcher.client, "lookup", lambda handle: {"id": "acct-1"})
    monkeypatch.setattr(
        watcher.client,
        "list_statuses",
        lambda user_id: [
            {"id": "104", "created_at": "2026-03-31T10:00:00Z", "content": "<p>older</p>", "url": "https://truthsocial.com/@realDonaldTrump/posts/104"},
            {
                "id": "105",
                "created_at": "2026-03-31T10:05:00Z",
                "content": "<p>fresh</p>",
                "url": "https://truthsocial.com/@realDonaldTrump/posts/105",
                "account": {"username": "realDonaldTrump"},
            },
            {
                "id": "106",
                "created_at": "2026-03-31T10:06:00Z",
                "content": "<p>repost</p>",
                "url": "https://truthsocial.com/@realDonaldTrump/posts/106",
                "reblog": {"id": "500"},
            },
        ],
    )
    monkeypatch.setattr(watcher, "_download_images_for_post", lambda post: [])

    state = StateStore(tmp_path / "state.json")
    state.set_cursor(watcher.source_name, "104")
    state.mark_seen(watcher.source_name, "104")

    events = watcher.poll(state)

    assert [event.item_id for event in events] == ["105"]
    assert state.get_cursor(watcher.source_name) == "106"
    assert state.is_seen(watcher.source_name, "105")
    assert not state.is_seen(watcher.source_name, "106")


def test_truth_social_requests_client_raises_helpful_auth_error(monkeypatch):
    monkeypatch.setenv("TRUTHSOCIAL_TOKEN", "bad-token")
    client = TruthSocialRequestsClient("bad-token")

    class FakeResponse:
        status_code = 401
        text = "Unauthorized"

        def raise_for_status(self):
            raise AssertionError("should not reach raise_for_status for 401")

    monkeypatch.setattr(client.session, "get", lambda *args, **kwargs: FakeResponse())

    with pytest.raises(TruthSocialAuthError, match="401 Unauthorized"):
        client.lookup("realDonaldTrump")


def test_probe_source_reports_skip_success_and_failure():
    class DummySession:
        def __init__(self):
            self.headers = {}

    class DummyWatcher:
        def __init__(self, source_name: str, target: str, cookie: str = "", *, target_attr: str = "url"):
            self.source_name = source_name
            setattr(self, target_attr, target)
            self.session = DummySession()
            if cookie:
                self.session.headers["Cookie"] = cookie

    class DummyTruthWatcher:
        def __init__(self, source_name: str, handle: str, cookie: str = ""):
            self.source_name = source_name
            self.handle = handle
            self.client = type("DummyClient", (), {"session": DummySession()})()
            if cookie:
                self.client.session.headers["Cookie"] = cookie

    skip_spec = ProbeSpec(
        name="irna",
        builder=lambda: [],
        cookie_env="IRNA_COOKIE_HEADER",
        target_attr="url",
        probe_fn=lambda watcher: (0, ""),
    )
    skipped = probe_source(skip_spec)
    assert skipped[0].enabled is False
    assert skipped[0].error == "not enabled"

    ok_watcher = DummyWatcher("irna", "https://en.irna.ir/", cookie="session=1")
    ok_spec = ProbeSpec(
        name="irna",
        builder=lambda: [ok_watcher],
        cookie_env="IRNA_COOKIE_HEADER",
        target_attr="url",
        probe_fn=lambda watcher: (3, "sample-title"),
    )
    ok_results = probe_source(ok_spec)
    assert ok_results[0].ok is True
    assert ok_results[0].item_count == 3
    assert ok_results[0].has_cookie is True

    fail_watcher = DummyWatcher("bloomberg", "https://www.bloomberg.com/markets/economics")
    fail_spec = ProbeSpec(
        name="bloomberg",
        builder=lambda: [fail_watcher],
        cookie_env="BLOOMBERG_COOKIE_HEADER",
        target_attr="url",
        probe_fn=lambda watcher: (_ for _ in ()).throw(RuntimeError("anti-bot page")),
    )
    fail_results = probe_source(fail_spec)
    assert fail_results[0].ok is False
    assert "anti-bot page" in fail_results[0].error

    truth_watcher = DummyTruthWatcher("truth_social:realDonaldTrump", "realDonaldTrump", cookie="session=1")
    truth_spec = ProbeSpec(
        name="truth_social",
        builder=lambda: [truth_watcher],
        cookie_env="TRUTHSOCIAL_COOKIE_HEADER",
        target_attr="handle",
        probe_fn=lambda watcher: (5, "latest post"),
    )
    truth_results = probe_source(truth_spec)
    assert truth_results[0].target == "@realDonaldTrump"
    assert truth_results[0].has_cookie is True
    assert truth_results[0].item_count == 5


def test_probe_main_outputs_json(monkeypatch, capsys):
    monkeypatch.setattr(
        "probe_protected_sources.run_selected_probes",
        lambda names: [
            ProbeResult(
                source="dol",
                target="https://www.dol.gov/newsroom/releases",
                enabled=True,
                has_cookie=False,
                ok=True,
                item_count=4,
                sample_title="release",
                error="",
            )
        ],
    )

    exit_code = probe_main(["--sources", "dol", "--json"])
    captured = capsys.readouterr().out
    payload = json.loads(captured)

    assert exit_code == 0
    assert payload[0]["source"] == "dol"
    assert payload[0]["item_count"] == 4
