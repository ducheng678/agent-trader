from __future__ import annotations

import time

import pytest
from fastapi.testclient import TestClient


def test_llm_engine_uses_named_architecture_modules():
    from market_agent.agent_context import AgentContextMixin
    from market_agent.agent_runtime import DiscretionaryLLMEngine
    from market_agent.memory_state import MemoryStateMixin
    from market_agent.model_routing import ModelRoutingMixin
    from market_agent.passive_workflow import PassiveWorkflowMixin
    from market_agent.prompt_context import PromptContextMixin
    from market_agent.retrieval_rag import RetrievalRAGMixin
    from market_agent.structured_outputs import StructuredOutputMixin
    from market_agent.tool_calling import ToolCallingMixin
    from market_agent.llm_engine import DiscretionaryLLMEngine as LegacyDiscretionaryLLMEngine

    assert LegacyDiscretionaryLLMEngine is DiscretionaryLLMEngine
    for module in (
        AgentContextMixin,
        MemoryStateMixin,
        ModelRoutingMixin,
        PassiveWorkflowMixin,
        PromptContextMixin,
        RetrievalRAGMixin,
        StructuredOutputMixin,
        ToolCallingMixin,
    ):
        assert issubclass(DiscretionaryLLMEngine, module)


def test_backend_cache_database_queue_events_and_api(tmp_path):
    from market_agent.backend.api import create_app
    from market_agent.backend.container import BackendContainer
    from market_agent.backend.settings import BackendSettings

    settings = BackendSettings(
        database_path=tmp_path / "backend.sqlite3",
        cache_max_entries=2,
        cache_default_ttl_seconds=60.0,
        task_workers=1,
        api_token="test-token",
        environment="test",
    )
    container = BackendContainer.create(settings)
    container.task_queue.register("echo", lambda payload: {"echo": payload["value"]})
    container.cache.set("recent", {"ok": True})
    assert container.cache.get("recent") == {"ok": True}

    app = create_app(container)
    client = TestClient(app)
    headers = {"Authorization": "Bearer test-token", "X-Request-ID": "request-123"}
    try:
        assert client.get("/health/live").status_code == 200
        assert client.post("/v1/tasks/echo", json={"payload": {"value": "hello"}}).status_code == 401

        accepted = client.post(
            "/v1/tasks/echo",
            headers=headers,
            json={"payload": {"value": "hello"}, "idempotency_key": "echo-1"},
        )
        assert accepted.status_code == 202
        job_id = accepted.json()["job_id"]

        duplicate = client.post(
            "/v1/tasks/echo",
            headers=headers,
            json={"payload": {"value": "hello"}, "idempotency_key": "echo-1"},
        )
        assert duplicate.status_code == 202
        assert duplicate.json()["job_id"] == job_id

        status = None
        for _ in range(100):
            status = client.get(f"/v1/tasks/{job_id}", headers=headers)
            if status.json()["status"] in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        assert status is not None
        assert status.status_code == 200
        assert status.json()["status"] == "succeeded"
        assert status.json()["result"] == {"echo": "hello"}
        assert status.headers["X-Request-ID"] == "request-123"

        events = client.get(f"/v1/tasks/{job_id}/events", headers=headers)
        assert events.status_code == 200
        assert any(event["event_type"] == "task_succeeded" for event in events.json()["items"])

        metrics = client.get("/metrics", headers=headers)
        assert metrics.status_code == 200
        assert "market_agent_task_submitted_total" in metrics.text

        unknown = client.post("/v1/tasks/missing", headers=headers, json={"payload": {}})
        assert unknown.status_code == 404
    finally:
        container.shutdown()


def test_backend_rejects_production_settings_without_api_token(tmp_path):
    from market_agent.backend.errors import ConfigurationError
    from market_agent.backend.settings import BackendSettings

    with pytest.raises(ConfigurationError):
        BackendSettings(
            database_path=tmp_path / "backend.sqlite3",
            environment="production",
        ).validate()


def test_backend_retries_transient_tasks_and_records_event(tmp_path):
    from market_agent.backend.container import BackendContainer
    from market_agent.backend.settings import BackendSettings

    container = BackendContainer.create(
        BackendSettings(
            database_path=tmp_path / "backend.sqlite3",
            task_workers=1,
            task_max_attempts=2,
            task_retry_delay_seconds=0.0,
            environment="test",
        )
    )
    attempts = {"count": 0}

    def flaky(payload):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise RuntimeError("transient failure")
        return {"value": payload["value"]}

    container.task_queue.register("flaky", flaky)
    try:
        submission = container.task_queue.submit("flaky", {"value": "recovered"}, request_id="retry-1")
        job = submission.job
        for _ in range(100):
            job = container.task_queue.get_job(job.job_id)
            if job.status in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        assert job.status == "succeeded"
        assert job.attempt_count == 2
        assert any(event.event_type == "task_retry_scheduled" for event in container.task_queue.list_events(job.job_id))
    finally:
        container.shutdown()


def test_backend_recovers_durable_accepted_task_after_restart(tmp_path):
    from market_agent.backend.container import BackendContainer
    from market_agent.backend.settings import BackendSettings

    settings = BackendSettings(
        database_path=tmp_path / "backend.sqlite3",
        task_workers=1,
        task_max_attempts=2,
        task_retry_delay_seconds=0.0,
        environment="test",
    )
    first = BackendContainer.create(settings)
    try:
        pending, reused = first.repository.create_or_get_job(
            "recovered",
            {"value": "after-restart"},
            "recover-1",
            2,
            "recover-request",
        )
        assert not reused
        assert pending.status == "accepted"
    finally:
        first.shutdown()

    second = BackendContainer.create(settings)
    second.task_queue.register("recovered", lambda payload: {"value": payload["value"]})
    try:
        job = second.task_queue.get_job(pending.job_id)
        for _ in range(100):
            job = second.task_queue.get_job(pending.job_id)
            if job.status in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        assert job.status == "succeeded"
        assert any(event.event_type == "task_recovery_queued" for event in second.task_queue.list_events(job.job_id))
    finally:
        second.shutdown()
