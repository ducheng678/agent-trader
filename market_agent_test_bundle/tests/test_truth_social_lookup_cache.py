from watchers.truth_social_requests_watcher import TruthSocialRequestsWatcher
from watchers.truth_social_watcher import TruthSocialWatcher


def test_truth_social_requests_watcher_caches_lookup_user_id(monkeypatch):
    monkeypatch.setenv("TRUTHSOCIAL_TOKEN", "token")
    watcher = TruthSocialRequestsWatcher("rapidresponse47")
    calls = {"lookup": 0, "list_statuses": 0}

    def fake_lookup(handle):
        calls["lookup"] += 1
        return {"id": "user-123"}

    def fake_list_statuses(user_id):
        calls["list_statuses"] += 1
        assert user_id == "user-123"
        return [{"id": "1"}, {"id": "2"}]

    watcher.client.lookup = fake_lookup
    watcher.client.list_statuses = fake_list_statuses

    first = watcher._safe_fetch_statuses()
    second = watcher._safe_fetch_statuses()

    assert [post["id"] for post in first] == ["1", "2"]
    assert [post["id"] for post in second] == ["1", "2"]
    assert calls["lookup"] == 1
    assert calls["list_statuses"] == 2


def test_truth_social_watcher_caches_lookup_user_id(monkeypatch):
    monkeypatch.setenv("TRUTHSOCIAL_TOKEN", "token")
    watcher = TruthSocialWatcher("rapidresponse47")
    calls = {"lookup": 0, "get": 0}

    def fake_lookup(handle):
        calls["lookup"] += 1
        return {"id": "user-456"}

    def fake_get(path, params=None):
        calls["get"] += 1
        assert path == "/v1/accounts/user-456/statuses?exclude_replies=true"
        return [{"id": "1"}, {"id": "2"}]

    watcher.api.lookup = fake_lookup
    watcher.api._get = fake_get

    first = watcher._fetch_statuses_once()
    second = watcher._fetch_statuses_once()

    assert [post["id"] for post in first] == ["1", "2"]
    assert [post["id"] for post in second] == ["1", "2"]
    assert calls["lookup"] == 1
    assert calls["get"] == 2
