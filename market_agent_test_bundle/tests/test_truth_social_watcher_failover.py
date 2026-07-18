from __future__ import annotations

import sys
from pathlib import Path

import pytest
import requests

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from watchers.common import fetch_with_retries
from watchers.proxy_failover import MihomoProxyFailover, load_source_proxy_config
from watchers.truth_social_watcher import CFBlockException, TruthSocialWatcher
from watchers.watcher_proxy_config import build_proxy_targets_from_watchers


def test_mihomo_proxy_failover_rotates_to_next_concrete_node(monkeypatch):
    controller = MihomoProxyFailover("127.0.0.1:9097", selector_group="PROXY")
    switched = {}

    def fake_get(name: str):
        if name == "PROXY":
            return {
                "name": "PROXY",
                "type": "Selector",
                "now": "TrojanFlare Auto",
                "all": ["TrojanFlare Auto", "DIRECT", "node-a", "node-b"],
            }
        if name == "TrojanFlare Auto":
            return {
                "name": "TrojanFlare Auto",
                "type": "URLTest",
                "now": "node-a",
            }
        raise AssertionError(name)

    def fake_set(selector_name: str, target_name: str):
        switched["selector_name"] = selector_name
        switched["target_name"] = target_name

    monkeypatch.setattr(controller, "get_proxy_detail", fake_get)
    monkeypatch.setattr(controller, "set_selector", fake_set)

    assert controller.rotate_to_next_node() == "node-b"
    assert switched == {"selector_name": "PROXY", "target_name": "node-b"}


def test_mihomo_proxy_failover_rotates_source_specific_auto_group(monkeypatch):
    controller = MihomoProxyFailover("127.0.0.1:9097", selector_group="TruthSocial Proxy")
    switched = {}

    def fake_get(name: str):
        if name == "TruthSocial Proxy":
            return {
                "name": "TruthSocial Proxy",
                "type": "Selector",
                "now": "TruthSocial Auto",
                "all": ["TruthSocial Auto", "DIRECT", "node-a", "node-b"],
            }
        if name == "TruthSocial Auto":
            return {
                "name": "TruthSocial Auto",
                "type": "URLTest",
                "now": "node-a",
                "all": ["node-a", "node-b"],
            }
        raise AssertionError(name)

    def fake_set(selector_name: str, target_name: str):
        switched["selector_name"] = selector_name
        switched["target_name"] = target_name

    monkeypatch.setattr(controller, "get_proxy_detail", fake_get)
    monkeypatch.setattr(controller, "set_selector", fake_set)

    assert controller.rotate_to_next_node() == "node-b"
    assert switched == {"selector_name": "TruthSocial Proxy", "target_name": "node-b"}


def test_mihomo_proxy_failover_selects_fastest_healthchecked_node(monkeypatch):
    controller = MihomoProxyFailover(
        "127.0.0.1:9097",
        selector_group="ConferenceBoard Proxy",
        healthcheck_url="https://www.conference-board.org/press/index.cfm?centerid=34",
        probe_timeout_ms=1000,
        probe_concurrency=2,
    )
    switched = {}

    def fake_get(name: str):
        if name == "ConferenceBoard Proxy":
            return {
                "name": "ConferenceBoard Proxy",
                "type": "Selector",
                "now": "node-a",
                "all": ["ConferenceBoard Auto", "DIRECT", "node-a", "node-b", "node-c"],
            }
        return {
            "name": name,
            "type": "Trojan",
        }

    def fake_set(selector_name: str, target_name: str):
        switched["selector_name"] = selector_name
        switched["target_name"] = target_name

    delays = {"node-b": 900, "node-c": 120}
    monkeypatch.setattr(controller, "get_proxy_detail", fake_get)
    monkeypatch.setattr(controller, "set_selector", fake_set)
    monkeypatch.setattr(controller, "_probe_node_delay", lambda node_name: delays.get(node_name))

    assert controller.rotate_to_next_node() == "node-c"
    assert controller.last_probe_summary == "probe=120ms"
    assert switched == {"selector_name": "ConferenceBoard Proxy", "target_name": "node-c"}


def test_load_source_proxy_config_supports_collapsed_truthsocial_prefix(monkeypatch):
    monkeypatch.setenv("TRUTHSOCIAL_HTTP_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("TRUTHSOCIAL_HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("TRUTHSOCIAL_PROXY_FAILOVER_ENABLED", "true")
    monkeypatch.setenv("TRUTHSOCIAL_PROXY_SELECTOR_GROUP", "PROXY")
    config = load_source_proxy_config("truth_social:rapidresponse47")
    assert config.http_proxy == "http://127.0.0.1:7897"
    assert config.https_proxy == "http://127.0.0.1:7897"
    assert config.failover_enabled is True
    assert config.selector_group == "PROXY"


def test_load_source_proxy_config_defaults_truthsocial_selector_from_proxy_env(monkeypatch):
    monkeypatch.setenv("TRUTHSOCIAL_HTTP_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("TRUTHSOCIAL_HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.delenv("TRUTHSOCIAL_PROXY_SELECTOR_GROUP", raising=False)

    config = load_source_proxy_config("truth_social:rapidresponse47")

    assert config.selector_group == "TruthSocial Proxy"


def test_load_source_proxy_config_uses_local_mihomo_for_configured_sources(monkeypatch):
    monkeypatch.setenv("PROXY_WATCHER_SOURCES", "aaa,conference_board")
    monkeypatch.setenv("MIHOMO_MIXED_PORT", "7897")

    config = load_source_proxy_config("conference_board")
    aaa_config = load_source_proxy_config("aaa")

    assert config.http_proxy == "http://127.0.0.1:7897"
    assert config.https_proxy == "http://127.0.0.1:7897"
    assert config.selector_group == "ConferenceBoard Proxy"
    assert aaa_config.selector_group == "AAA Proxy"


def test_fetch_with_retries_rotates_source_proxy_after_403(monkeypatch):
    session = requests.Session()
    rotations: list[str] = []
    session._source_name = "aaa"
    session._source_proxy_failover_max_rotations = 1
    session._source_proxy_failover_sleep_seconds = 0
    session._source_proxy_failover = type(
        "DummyFailover",
        (),
        {
            "selector_group": "AAA Proxy",
            "rotate_to_next_node": lambda self: rotations.append("node-b") or "node-b",
        },
    )()
    calls = {"count": 0}

    def fake_get(url, timeout=None, headers=None):
        calls["count"] += 1
        response = requests.Response()
        response.url = url
        response.status_code = 403 if calls["count"] == 1 else 200
        response._content = b"ok"
        return response

    monkeypatch.setattr(session, "get", fake_get)

    response = fetch_with_retries(session, "https://gasprices.aaa.com/")

    assert response.status_code == 200
    assert calls["count"] == 2
    assert rotations == ["node-b"]


def test_proxy_target_is_derived_from_watcher_origin(monkeypatch):
    monkeypatch.setenv("TRUTHSOCIAL_HTTP_PROXY", "http://127.0.0.1:7897")
    monkeypatch.delenv("TRUTHSOCIAL_PROXY_HEALTHCHECK_URL", raising=False)
    monkeypatch.delenv("TRUTHSOCIAL_PROXY_RULE_DOMAINS", raising=False)
    monkeypatch.delenv("TRUTHSOCIAL_PROXY_SELECTOR_GROUP", raising=False)

    watcher = type(
        "DummyWatcher",
        (),
        {
            "source_name": "truth_social:rapidresponse47",
            "base_url": "https://truthsocial.com",
        },
    )()

    targets = build_proxy_targets_from_watchers([watcher])

    assert len(targets) == 1
    assert targets[0].selector_group == "TruthSocial Proxy"
    assert targets[0].auto_group == "TruthSocial Auto"
    assert targets[0].healthcheck_url == "https://truthsocial.com/"
    assert targets[0].rule_domains == ("truthsocial.com",)


def test_truth_social_watcher_retries_after_cfblock_with_proxy_failover(monkeypatch, tmp_path):
    monkeypatch.setenv("TRUTHSOCIAL_TOKEN", "test-token")
    monkeypatch.setenv("TRUTH_MEDIA_DIR", str(tmp_path / "truth_media"))
    monkeypatch.setenv("TRUTHSOCIAL_HTTP_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("TRUTHSOCIAL_HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("MIHOMO_EXTERNAL_CONTROLLER", "127.0.0.1:9097")
    monkeypatch.setenv("TRUTHSOCIAL_PROXY_FAILOVER_MAX_ROTATIONS", "1")
    monkeypatch.setenv("TRUTHSOCIAL_PROXY_FAILOVER_SLEEP_SECONDS", "0")

    watcher = TruthSocialWatcher("rapidresponse47")
    watcher.proxy_failover_max_rotations = 1
    rotations: list[str] = []
    watcher.proxy_failover = type(
        "DummyFailover",
        (),
        {
            "selector_group": "PROXY",
            "rotate_to_next_node": lambda self: rotations.append("node-b") or "node-b",
        },
    )()

    calls = {"count": 0}

    def fake_fetch_once(since_id=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise CFBlockException("blocked")
        return [
            {
                "id": "101",
                "created_at": "2026-04-20T20:23:49Z",
                "content": "<p>fresh</p>",
                "url": "https://truthsocial.com/@rapidresponse47/posts/101",
            }
        ]

    monkeypatch.setattr(watcher, "_fetch_statuses_once", fake_fetch_once)

    posts = watcher._safe_fetch_statuses()

    assert [post["id"] for post in posts] == ["101"]
    assert calls["count"] == 2
    assert rotations == ["node-b"]


def test_truth_social_watcher_retries_after_lookup_failed_with_proxy_failover(monkeypatch, tmp_path):
    monkeypatch.setenv("TRUTHSOCIAL_TOKEN", "test-token")
    monkeypatch.setenv("TRUTH_MEDIA_DIR", str(tmp_path / "truth_media"))
    monkeypatch.setenv("TRUTHSOCIAL_HTTP_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("TRUTHSOCIAL_HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("MIHOMO_EXTERNAL_CONTROLLER", "127.0.0.1:9097")
    monkeypatch.setenv("TRUTHSOCIAL_PROXY_FAILOVER_MAX_ROTATIONS", "1")
    monkeypatch.setenv("TRUTHSOCIAL_PROXY_FAILOVER_SLEEP_SECONDS", "0")

    watcher = TruthSocialWatcher("rapidresponse47")
    watcher.proxy_failover_max_rotations = 1
    rotations: list[str] = []
    watcher.proxy_failover = type(
        "DummyFailover",
        (),
        {
            "selector_group": "PROXY",
            "rotate_to_next_node": lambda self: rotations.append("node-b") or "node-b",
        },
    )()

    calls = {"count": 0}

    def fake_fetch_once(since_id=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("lookup failed for @rapidresponse47: None")
        return [
            {
                "id": "101",
                "created_at": "2026-04-20T20:23:49Z",
                "content": "<p>fresh</p>",
                "url": "https://truthsocial.com/@rapidresponse47/posts/101",
            }
        ]

    monkeypatch.setattr(watcher, "_fetch_statuses_once", fake_fetch_once)

    posts = watcher._safe_fetch_statuses()

    assert [post["id"] for post in posts] == ["101"]
    assert calls["count"] == 2
    assert rotations == ["node-b"]


def test_truth_social_watcher_retries_after_statuses_none_with_proxy_failover(monkeypatch, tmp_path):
    monkeypatch.setenv("TRUTHSOCIAL_TOKEN", "test-token")
    monkeypatch.setenv("TRUTH_MEDIA_DIR", str(tmp_path / "truth_media"))
    monkeypatch.setenv("TRUTHSOCIAL_HTTP_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("TRUTHSOCIAL_HTTPS_PROXY", "http://127.0.0.1:7897")
    monkeypatch.setenv("MIHOMO_EXTERNAL_CONTROLLER", "127.0.0.1:9097")
    monkeypatch.setenv("TRUTHSOCIAL_PROXY_FAILOVER_MAX_ROTATIONS", "1")
    monkeypatch.setenv("TRUTHSOCIAL_PROXY_FAILOVER_SLEEP_SECONDS", "0")

    watcher = TruthSocialWatcher("rapidresponse47")
    watcher.proxy_failover_max_rotations = 1
    rotations: list[str] = []
    watcher.proxy_failover = type(
        "DummyFailover",
        (),
        {
            "selector_group": "PROXY",
            "rotate_to_next_node": lambda self: rotations.append("node-b") or "node-b",
        },
    )()

    calls = {"count": 0}

    def fake_fetch_once(since_id=None):
        calls["count"] += 1
        if calls["count"] == 1:
            raise RuntimeError("statuses fetch returned no payload for @rapidresponse47")
        return [
            {
                "id": "101",
                "created_at": "2026-04-20T20:23:49Z",
                "content": "<p>fresh</p>",
                "url": "https://truthsocial.com/@rapidresponse47/posts/101",
            }
        ]

    monkeypatch.setattr(watcher, "_fetch_statuses_once", fake_fetch_once)

    posts = watcher._safe_fetch_statuses()

    assert [post["id"] for post in posts] == ["101"]
    assert calls["count"] == 2
    assert rotations == ["node-b"]
