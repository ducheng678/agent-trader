from types import SimpleNamespace
import sys

from market_agent.backend.container import BackendContainer
from market_agent.backend.prompt_activation_store import PostgresPromptActivationStore


def test_readiness_rejects_unwired_shared_prompt_authority():
    store = object.__new__(PostgresPromptActivationStore)
    store.healthcheck = lambda: True
    container = object.__new__(BackendContainer)
    container.prompt_activation_store = store
    container.prompt_release_manager = SimpleNamespace(activation_store=None, current=lambda: object())
    assert container._probe_prompt_activation() == "failed"
    container.prompt_release_manager.activation_store = store
    assert container._probe_prompt_activation() == "ok"


def test_readiness_fails_closed_when_shared_prompt_store_is_unavailable():
    store = object.__new__(PostgresPromptActivationStore)
    store.healthcheck = lambda: False
    container = object.__new__(BackendContainer)
    container.prompt_activation_store = store
    container.prompt_release_manager = SimpleNamespace(activation_store=store, current=lambda: object())
    assert container._probe_prompt_activation() == "failed"


def test_container_starts_and_closes_prompt_audit_recovery(tmp_path, monkeypatch):
    from market_agent.backend.settings import BackendSettings

    calls = []

    class Scheduler:
        def __init__(self, manager, **kwargs):
            self.manager = manager

        def start(self):
            calls.append("start")

        def close(self):
            calls.append("close")

    monkeypatch.setitem(sys.modules, "market_agent.backend.prompt_audit_recovery",
                        SimpleNamespace(PromptAuditRecoveryScheduler=Scheduler))
    container = BackendContainer.create(BackendSettings(
        database_path=tmp_path / "backend.db", environment="test"))
    try:
        assert calls == ["start"]
        assert container.prompt_audit_recovery.manager is container.prompt_release_manager
    finally:
        container.shutdown()
    assert calls == ["start", "close"]
