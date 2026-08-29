from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

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
    from market_agent.backend.errors import RetryableTaskError
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
            raise RetryableTaskError("transient failure")
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


def _wait_for_terminal(task_queue, job_id, timeout_seconds=3.0):
    deadline = time.monotonic() + timeout_seconds
    job = task_queue.get_job(job_id)
    while job.status not in {"succeeded", "failed"} and time.monotonic() < deadline:
        time.sleep(0.01)
        job = task_queue.get_job(job_id)
    return job


def test_generate_playbook_payload_rejects_string_boolean():
    from pydantic import ValidationError as PydanticValidationError

    from market_agent.backend.api_contracts import GeneratePlaybookPayload

    with pytest.raises(PydanticValidationError):
        GeneratePlaybookPayload.model_validate(
            {
                "user_query": "trade BTC",
                "event_tape": [],
                "trigger_reason": "event",
                "has_live_position": "false",
            }
        )


def test_agent_service_serializes_one_stateful_engine():
    from market_agent.backend.agent_service import AgentPlaybookService

    state_lock = threading.Lock()
    state = {"active": 0, "max_active": 0, "factory_calls": 0}

    class Playbook:
        def to_dict(self):
            return {"ok": True}

    class Engine:
        def get_playbook(self, **kwargs):
            with state_lock:
                state["active"] += 1
                state["max_active"] = max(state["max_active"], state["active"])
            time.sleep(0.03)
            with state_lock:
                state["active"] -= 1
            return Playbook(), {"query": kwargs["user_query"]}

    def factory():
        state["factory_calls"] += 1
        return Engine()

    service = AgentPlaybookService(engine_factory=factory)
    payload = {"user_query": "trade BTC", "event_tape": [], "trigger_reason": "event"}
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(service.generate_playbook, (payload, payload)))

    assert state["factory_calls"] == 1
    assert state["max_active"] == 1
    assert all(result["playbook"] == {"ok": True} for result in results)


def test_backend_does_not_retry_unclassified_errors(tmp_path):
    from market_agent.backend.container import BackendContainer
    from market_agent.backend.settings import BackendSettings

    container = BackendContainer.create(
        BackendSettings(
            database_path=tmp_path / "backend.sqlite3",
            task_workers=1,
            task_max_attempts=3,
            task_retry_delay_seconds=0.0,
            environment="test",
        )
    )
    attempts = {"count": 0}

    def fail(payload):
        attempts["count"] += 1
        raise RuntimeError(str(payload["value"]))

    container.task_queue.register("permanent", fail)
    try:
        submission = container.task_queue.submit("permanent", {"value": "invalid"})
        job = _wait_for_terminal(container.task_queue, submission.job.job_id)
        assert job.status == "failed"
        assert job.attempt_count == 1
        assert attempts["count"] == 1
        assert job.error["retryable"] is False
    finally:
        container.shutdown()


def test_generate_playbook_validation_failure_is_not_retried(tmp_path):
    from market_agent.backend.container import BackendContainer
    from market_agent.backend.settings import BackendSettings

    container = BackendContainer.create(
        BackendSettings(
            database_path=tmp_path / "backend.sqlite3",
            task_workers=1,
            task_max_attempts=3,
            task_retry_delay_seconds=0.0,
            environment="test",
        )
    )
    try:
        submission = container.task_queue.submit("generate_playbook", {})
        job = _wait_for_terminal(container.task_queue, submission.job.job_id)
        assert job.status == "failed"
        assert job.attempt_count == 1
        assert job.error["code"] == "validation_error"
        assert job.error["retryable"] is False
        assert all("input" not in error for error in job.error["details"]["errors"])
    finally:
        container.shutdown()


def test_task_queue_backpressure_preserves_idempotent_replay(tmp_path):
    from market_agent.backend.container import BackendContainer
    from market_agent.backend.errors import IdempotencyConflictError, TaskQueueFullError
    from market_agent.backend.settings import BackendSettings

    container = BackendContainer.create(
        BackendSettings(
            database_path=tmp_path / "backend.sqlite3",
            task_workers=1,
            task_queue_capacity=1,
            task_retry_delay_seconds=0.0,
            environment="test",
        )
    )
    started = threading.Event()
    release = threading.Event()

    def blocking(payload):
        started.set()
        if not release.wait(3.0):
            raise RuntimeError("test release timed out")
        return dict(payload)

    container.task_queue.register("blocking", blocking)
    try:
        first = container.task_queue.submit("blocking", {"value": 1}, idempotency_key="first")
        assert started.wait(1.0)
        second = container.task_queue.submit("blocking", {"value": 2}, idempotency_key="second")
        with pytest.raises(TaskQueueFullError):
            container.task_queue.submit("blocking", {"value": 3})
        replay = container.task_queue.submit("blocking", {"value": 1}, idempotency_key="first")
        assert replay.reused is True
        assert replay.job.job_id == first.job.job_id
        with pytest.raises(IdempotencyConflictError):
            container.task_queue.submit("blocking", {"value": 99}, idempotency_key="first")
        release.set()
        assert _wait_for_terminal(container.task_queue, first.job.job_id).status == "succeeded"
        assert _wait_for_terminal(container.task_queue, second.job.job_id).status == "succeeded"
    finally:
        release.set()
        container.shutdown()


def test_recovery_drains_more_than_one_repository_page(tmp_path):
    from market_agent.backend.container import BackendContainer
    from market_agent.backend.settings import BackendSettings

    container = BackendContainer.create(
        BackendSettings(
            database_path=tmp_path / "backend.sqlite3",
            task_workers=2,
            task_queue_capacity=2,
            task_retry_delay_seconds=0.0,
            environment="test",
        )
    )
    pending = [
        container.repository.create_or_get_job("paged", {"value": value}, f"paged-{value}", 2, "recovery")[0]
        for value in range(7)
    ]
    original = container.repository.list_recoverable_jobs

    def list_two(task_name, limit=1000):
        return original(task_name, limit=min(2, limit))

    container.repository.list_recoverable_jobs = list_two
    container.task_queue.register("paged", lambda payload: dict(payload))
    try:
        jobs = [_wait_for_terminal(container.task_queue, job.job_id) for job in pending]
        assert all(job.status == "succeeded" for job in jobs)
    finally:
        container.shutdown()


def test_api_errors_are_correlated_sanitized_and_low_cardinality(tmp_path, monkeypatch):
    from market_agent.backend.api import create_app
    from market_agent.backend.container import BackendContainer
    from market_agent.backend.settings import BackendSettings

    container = BackendContainer.create(
        BackendSettings(database_path=tmp_path / "backend.sqlite3", environment="test")
    )
    client = TestClient(create_app(container), raise_server_exceptions=False)
    try:
        invalid = client.post(
            "/v1/tasks/generate_playbook",
            json={"payload": {}, "idempotency_key": "   "},
        )
        assert invalid.status_code == 422
        assert invalid.headers["X-Request-ID"] == invalid.json()["request_id"]
        assert all("input" not in error for error in invalid.json()["details"]["errors"])

        conflicting = client.post(
            "/v1/tasks/generate_playbook",
            headers={"Idempotency-Key": "header-key"},
            json={"payload": {}, "idempotency_key": "body-key"},
        )
        assert conflicting.status_code == 422
        assert conflicting.json()["error"] == "validation_error"

        def fail_get_job(job_id):
            raise RuntimeError(job_id)

        monkeypatch.setattr(container.task_queue, "get_job", fail_get_job)
        incoming_request_id = "x" * 129
        failed = client.get(
            "/v1/tasks/customer-specific-job-id",
            headers={"X-Request-ID": incoming_request_id},
        )
        assert failed.status_code == 500
        assert failed.json()["error"] == "internal_server_error"
        assert failed.headers["X-Request-ID"] == failed.json()["request_id"]
        assert failed.headers["X-Request-ID"] != incoming_request_id
        rendered = container.metrics.render_prometheus()
        assert 'path="/v1/tasks/{job_id}"' in rendered
        assert "customer-specific-job-id" not in rendered
    finally:
        container.shutdown()


def test_database_begin_failure_preserves_original_error():
    import sqlite3

    from market_agent.backend.database import JobRepository

    commands = []

    class Connection:
        closed = False

        def execute(self, statement):
            commands.append(statement)
            raise sqlite3.OperationalError("begin failed")

        def close(self):
            self.closed = True

    connection = Connection()
    repository = JobRepository.__new__(JobRepository)
    repository._lock = threading.RLock()
    repository._connect = lambda: connection

    with pytest.raises(sqlite3.OperationalError, match="begin failed"):
        repository._transaction(lambda _: None)

    assert commands == ["BEGIN IMMEDIATE"]
    assert connection.closed is True


def test_message_bus_wildcard_topic_is_delivered_once():
    from market_agent.backend.message_bus import InMemoryMessageBus, MessageEnvelope

    bus = InMemoryMessageBus()
    received = []
    bus.subscribe("*", received.append)
    bus.publish(MessageEnvelope(topic="*", payload={}))
    assert len(received) == 1


def test_settings_reject_nonfinite_timing_and_unauthenticated_external_bind(tmp_path):
    from market_agent.backend.errors import ConfigurationError
    from market_agent.backend.settings import BackendSettings

    with pytest.raises(ConfigurationError):
        BackendSettings(
            database_path=tmp_path / "nan.sqlite3",
            cache_default_ttl_seconds=float("nan"),
            environment="test",
        ).validate()
    with pytest.raises(ConfigurationError):
        BackendSettings(
            database_path=tmp_path / "infinite.sqlite3",
            task_retry_delay_seconds=float("inf"),
            environment="test",
        ).validate()
    with pytest.raises(ConfigurationError):
        BackendSettings(
            database_path=tmp_path / "external.sqlite3",
            api_host="0.0.0.0",
            environment="development",
        ).validate()


def test_message_bus_continues_after_subscriber_failure():
    from market_agent.backend.message_bus import InMemoryMessageBus, MessageEnvelope

    bus = InMemoryMessageBus()
    received = []

    def fail(_):
        raise RuntimeError("subscriber failed")

    bus.subscribe("task.done", fail)
    bus.subscribe("*", received.append)
    with pytest.raises(RuntimeError, match="1 message subscriber"):
        bus.publish(MessageEnvelope(topic="task.done", payload={}))
    assert len(received) == 1
