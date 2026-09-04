from __future__ import annotations

import time


class _RedisStreams:
    def __init__(self):
        self.streams = {}
        self.groups = {}
        self.pending = {}

    def ping(self):
        return True

    def get(self, key):
        return None

    def set(self, key, value, ex=None, nx=False):
        return True

    def delete(self, *keys):
        return 1

    def xgroup_create(self, name, groupname, id="0", mkstream=False):
        self.groups.setdefault((name, groupname), set())

    def xadd(self, name, fields, id="*"):
        identifier = f"{len(self.streams.get(name, ())) + 1}-0"
        self.streams.setdefault(name, []).append((identifier, dict(fields)))
        return identifier

    def xreadgroup(self, groupname, consumername, streams, count=1, block=None):
        result = []
        for stream, _ in streams.items():
            delivered = self.groups.setdefault((stream, groupname), set())
            messages = []
            for identifier, fields in self.streams.get(stream, ()):
                if identifier in delivered:
                    continue
                delivered.add(identifier)
                self.pending[(stream, groupname, identifier)] = consumername
                messages.append((identifier, fields))
                if len(messages) >= count:
                    break
            if messages:
                result.append((stream, messages))
        return result

    def xautoclaim(self, name, groupname, consumername, min_idle_time, start_id="0-0", count=None):
        messages = []
        for (stream, group, identifier), _owner in tuple(self.pending.items()):
            if stream == name and group == groupname:
                self.pending[(stream, group, identifier)] = consumername
                fields = next(fields for item, fields in self.streams[name] if item == identifier)
                messages.append((identifier, fields))
                if len(messages) >= (count or 1):
                    break
        return (start_id, messages)

    def xack(self, name, groupname, *ids):
        for identifier in ids:
            self.pending.pop((name, groupname, identifier), None)
        return len(ids)


def test_signed_admin_capability_is_tenant_and_trace_bound():
    import pytest

    from market_agent.workflow_capabilities import (
        CapabilityDeniedError,
        CapabilityScope,
        SignedCapabilityVerifier,
    )

    scope = CapabilityScope(actor_id="operator", task_id="admin-api", tenant_id="tenant-a", trace_id="1" * 32)
    verifier = SignedCapabilityVerifier("s" * 32)
    token = verifier.issue(scope=scope, actions=("prompt.activate",), ttl_seconds=30)
    assert verifier.authorize(token, scope=scope, action="prompt.activate").resource == "prompt.activate"
    with pytest.raises(CapabilityDeniedError):
        verifier.authorize(token, scope=scope.model_copy(update={"tenant_id": "tenant-b"}), action="prompt.activate")
    with pytest.raises(CapabilityDeniedError):
        verifier.authorize(token, scope=scope, action="workflow.cancel")


def test_redis_stream_reclaims_a_delivery_from_another_worker():
    from market_agent.backend.message_bus import MessageEnvelope
    from market_agent.backend.redis_adapters import RedisStreamMessageBus

    client = _RedisStreams()
    first = RedisStreamMessageBus(client, tenant_id="tenant-a")
    second = RedisStreamMessageBus(client, tenant_id="tenant-a")
    message = MessageEnvelope(
        topic="task.dispatch", request_id="3" * 32,
        payload={"trace_id": "3" * 32, "task_name": "echo", "job_id": "job-1"},
    )
    first.publish(message)
    delivered = first.consume(topic="task.dispatch", group="workers", consumer="worker-a", count=1)
    assert delivered and delivered[0].envelope.message_id == message.message_id
    _cursor, reclaimed = second.recover_pending(
        topic="task.dispatch", group="workers", consumer="worker-b", min_idle_ms=1, count=1
    )
    assert reclaimed and reclaimed[0].envelope.message_id == message.message_id


def test_local_knowledge_jsonl_loader_preserves_provenance(tmp_path):
    from market_agent.local_knowledge_base import LocalKnowledgeBase

    path = tmp_path / "knowledge.jsonl"
    path.write_text(
        '{"document_id":"policy-1","text":"The supported answer is stable.","answer":"stable","provenance":"approved-v1"}\n',
        encoding="utf-8",
    )
    knowledge = LocalKnowledgeBase.from_jsonl(path)
    assert knowledge.configured
    assert knowledge.lookup("What is the supported answer?").citations == ("policy-1",)


def test_backend_trace_query_contains_queue_and_ingress_components(tmp_path):
    from market_agent.backend.container import BackendContainer
    from market_agent.backend.settings import BackendSettings

    trace_id = "2" * 32
    container = BackendContainer.create(BackendSettings(database_path=tmp_path / "backend.sqlite3", environment="test"))
    container.task_queue.register("trace-task", lambda payload: {"ok": payload["ok"]})
    try:
        submission = container.task_queue.submit("trace-task", {"ok": True, "trace_id": trace_id}, request_id=trace_id)
        for _ in range(100):
            if container.task_queue.get_job(submission.job.job_id).status in {"succeeded", "failed"}:
                break
            time.sleep(0.01)
        page = container.observability.query(trace_id)
        assert any(item.event.event == "queue_started" for item in page.items)
        assert any(item.event.trace.trace_id == trace_id for item in page.items)
    finally:
        container.shutdown()


def test_unified_trace_repository_keeps_all_component_events_under_one_trace():
    from market_agent.backend.trace_observability import BackendObservability
    from market_agent.workflow_tracing import TraceContext

    trace = TraceContext(trace_id="4" * 32, span_id="5" * 16)
    observability = BackendObservability.create(event_capacity=32, maximum_query=32)
    for component, event, status in (
        ("ingress", "request_started", "started"),
        ("queue", "queue_started", "started"),
        ("coordinator", "agent_completed", "succeeded"),
        ("cache", "cache_miss", "unknown"),
        ("memory", "memory_completed", "unknown"),
        ("service", "commit_completed", "succeeded"),
    ):
        observability.record_component(trace, event=event, status=status, component=component)
    page = observability.query(trace.trace_id, limit=32)
    assert [item.event.event for item in page.items] == [
        "request_started", "queue_started", "agent_completed", "cache_miss",
        "memory_completed", "commit_completed",
    ]


def test_evaluation_cli_runs_the_versioned_seed_gate():
    from market_agent.evaluation_cli import run

    code, report = run([
        "--manifest", "evals/datasets/offline-safety-v1.manifest.json",
        "--code-revision", "test",
        "--prompt-release-hash", "0" * 64,
        "--model-policy-hash", "0" * 64,
        "--allow-missing-baseline",
    ])
    assert code == 0
    assert report["allowed"] is True
    assert report["metrics"]["trace_rate"] == 1.0
