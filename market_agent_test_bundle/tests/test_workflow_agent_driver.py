from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from hashlib import sha256
import importlib
import json

import pytest
from pydantic import BaseModel, ConfigDict, Field

from market_agent.local_knowledge_base import KnowledgeDocument, LocalKnowledgeBase
from market_agent.workflow_agent_contracts import AgentInvocation, AgentUsage, ModelTier
from market_agent.workflow_circuit_breaker import CircuitBreaker
from market_agent.workflow_fallback import FallbackPolicy
from market_agent.workflow_observation import CoreNodeName, ExecutionObservationCollector, TokenUsage
from market_agent.workflow_prompt_release import PromptRelease, PromptReleaseRegistry, canonical_json
from market_agent.workflow_response_cache import CacheMetadata, CachedResponse, ExactCacheKey, ExactResponseCache
from market_agent.workflow_retry_policy import ProviderError, RetryPolicy
from market_agent.workflow_semantic_request_cache import SemanticCacheEntry, SemanticRequestCache

TRACE_ID = "1" * 32

class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str = Field(min_length=1)


class CitedAnswer(Answer):
    citations: list[str] = Field(default_factory=list)


class Clock:
    def __init__(self):
        self.time = 1.0
        self.waits = []

    def now(self):
        return self.time

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.time += seconds


class Observer:
    def __init__(self):
        self.events = []

    def record(self, event):
        self.events.append(event)


class Client:
    """Transport double: records the actual driver request and returns raw provider data."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def invoke(self, request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            return outcome(request)
        return outcome


def api():
    # A missing implementation is an explicit assertion failure on the first RED run.
    assert importlib.util.find_spec("market_agent.workflow_agent_driver") is not None, "AgentDriver is not implemented"
    return importlib.import_module("market_agent.workflow_agent_driver")


def schema(model=Answer, **overrides):
    return api().OutputSchema(schema_id="answer-v1", model=model, **overrides)


def prompt_release():
    content = dict(
        schema_version="v1", release_id="release-v1", stable_system_prefix="Stable released prefix.",
        supported_task_kinds=("extract", "analyze", "coordinator"),
        supported_model_tiers=(ModelTier.SOL, ModelTier.TERRA, ModelTier.LUNA),
        temperature_profile=((ModelTier.SOL, 0.0), (ModelTier.TERRA, 0.2), (ModelTier.LUNA, 0.0)),
    )
    encoded = json.dumps(content, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return PromptRelease(digest=sha256(encoded.encode("utf-8")).hexdigest(), **content)


def invocation(output_schema=None, **overrides):
    values = dict(
        trace_id=TRACE_ID, run_id="run-1", task_id="task-1", task_kind="extract",
        prompt_release_id="release-v1", prompt_release_digest=prompt_release().digest,
        allowed_model_tier=ModelTier.LUNA, deadline_epoch=100.0,
        max_attempts=6, cost_limit_usd=1.0, output_schema_id="answer-v1",
        output_schema_digest=(output_schema or schema()).digest,
        user_payload={"query": "supported answer", "context": "private payload"},
    )
    values.update(overrides)
    return AgentInvocation(**values)


def response(content='{"answer":"known"}', tier=ModelTier.LUNA, cost=0.01):
    return api().ModelResponse(
        content=content,
        usage=AgentUsage(
            input_tokens=4, output_tokens=2, cost_usd=cost, model_tier=tier,
            pricing_version="openai-standard-2026-08-01",
            pricing_model_id=f"gpt-5.6-{tier.value}", pricing_band="short",
        ),
    )


def metadata(output_schema=None, **overrides):
    values = dict(
        tenant_scope="tenant-a", prompt_release_digest=prompt_release().digest,
        output_schema_digest=(output_schema or schema()).digest,
        model_compatibility_key="luna-v1", category="reference", expires_at=50.0,
        vector_version="vector-v1", model_version="luna-v1",
    )
    values.update(overrides)
    return CacheMetadata(**values)


def make_driver(client, *, output_schema=None, **overrides):
    output_schema = output_schema or schema()
    clock, observer = Clock(), Observer()
    release = prompt_release()
    values = dict(
        model_client=client, audit_observer=observer, clock=clock, random=lambda low, high: high,
        prompt_releases=PromptReleaseRegistry(releases=(release,)), output_schemas=(output_schema,),
        retry_policy=RetryPolicy(max_attempts=2, base_delay=0.25),
        circuit_breaker=CircuitBreaker(failure_threshold=3, cooldown=10.0),
        fallback_policy=FallbackPolicy((ModelTier.SOL, ModelTier.TERRA, ModelTier.LUNA)),
        model_costs={ModelTier.SOL: 0.1, ModelTier.TERRA: 0.1, ModelTier.LUNA: 0.1},
    )
    values.update(overrides)
    return api().AgentDriver(**values), values["audit_observer"], values["clock"]


def cache_request(meta=None):
    return api().CacheRequest(metadata=meta or metadata(), vector=(1.0, 0.0))


def memory_summary(**changes):
    from market_agent.workflow_memory_retrieval import CoreExperienceSummary, SummaryItem
    values = dict(tenant_id="tenant-a", scope="default", conflict_state="clear", confidence=0.8,
                  issued_at=datetime.fromtimestamp(0, timezone.utc), expires_at=datetime.fromtimestamp(50, timezone.utc),
                  selected_ids=("rule-1",), evidence_ids=("event-1", "event-2"),
                  rules=(SummaryItem(record_id="rule-1", text="Check cited observations before entry.",
                                     evidence_ids=("event-1", "event-2")),))
    values.update(changes)
    return CoreExperienceSummary(**values)


def test_driver_cancellation_prevents_and_discards_provider_work():
    cancelled = {"value": True}
    client = Client(response())
    driver, _, _ = make_driver(client)

    before = driver.execute(invocation(), cancellation_check=lambda: cancelled["value"])
    assert before.failure is not None and before.failure.code == "cancelled"
    assert client.requests == []

    cancelled["value"] = False

    def cancel_during_call(_request):
        cancelled["value"] = True
        return response()

    client.outcomes[:] = [cancel_during_call]
    during = driver.execute(invocation(), cancellation_check=lambda: cancelled["value"])
    assert during.failure is not None and during.failure.code == "cancelled"
    assert len(client.requests) == 1


def test_driver_rechecks_summary_expiry_when_reused_with_reported_age_zero():
    client = Client(response(), response())
    driver, _, clock = make_driver(client)
    summary = memory_summary()
    binding = dict(memory_context=summary, memory_tenant_id="tenant-a", memory_scope="default")
    assert driver.execute(invocation(), **binding).failure is None
    clock.time = 50.0
    assert driver.execute(invocation(), **binding).failure is None
    assert len(client.requests[0].messages) == 3
    assert len(client.requests[1].messages) == 2


def test_driver_drops_summary_that_expires_during_retry_wait():
    client = Client(ProviderError(status_code=503), response())
    driver, _, _ = make_driver(client)
    summary = memory_summary(expires_at=datetime.fromtimestamp(1.1, timezone.utc))
    result = driver.execute(invocation(), memory_context=summary, memory_tenant_id="tenant-a", memory_scope="default")
    assert result.failure is None
    assert len(client.requests[0].messages) == 3
    assert len(client.requests[1].messages) == 2


def test_driver_checks_memory_expiry_after_slow_prompt_audit():
    clock = Clock()
    class SlowObserver(Observer):
        def record(self, event):
            super().record(event)
            if event.event_type == "prompt_composed":
                clock.time = 50.0
    client = Client(response())
    driver, _, _ = make_driver(client, clock=clock, audit_observer=SlowObserver())
    result = driver.execute(invocation(), memory_context=memory_summary(),
                            memory_tenant_id="tenant-a", memory_scope="default")
    assert result.failure is None
    assert len(client.requests[0].messages) == 2


def test_driver_discards_memory_bound_output_when_summary_expires_in_provider():
    clock = Clock()

    def expires_before_response(_):
        clock.time = 50.0
        return response()

    client = Client(expires_before_response)
    driver, observer, _ = make_driver(client, clock=clock)
    result = driver.execute(invocation(), memory_context=memory_summary(),
                            memory_tenant_id="tenant-a", memory_scope="default")

    assert result.failure.code == "memory_context_expired"
    assert len(client.requests[0].messages) == 3
    assert [event.event_type for event in observer.events][-2:] == ["memory_expired", "task_failed"]
    assert "schema_validated" not in [event.event_type for event in observer.events]


def test_driver_discards_memory_bound_output_when_summary_expires_in_schema_audit():
    clock = Clock()

    class SlowObserver(Observer):
        def record(self, event):
            super().record(event)
            if event.event_type == "schema_validated":
                clock.time = 50.0

    driver, observer, _ = make_driver(Client(response()), clock=clock, audit_observer=SlowObserver())
    result = driver.execute(invocation(), memory_context=memory_summary(),
                            memory_tenant_id="tenant-a", memory_scope="default")

    assert result.failure.code == "memory_context_expired"
    assert [event.event_type for event in observer.events][-2:] == ["memory_expired", "task_failed"]
    assert "task_completed" not in [event.event_type for event in observer.events]


def test_driver_discards_memory_bound_output_when_summary_expires_in_completion_audit():
    clock = Clock()

    class SlowObserver(Observer):
        def record(self, event):
            super().record(event)
            if event.event_type == "task_completed":
                clock.time = 50.0

    driver, observer, _ = make_driver(Client(response()), clock=clock, audit_observer=SlowObserver())
    result = driver.execute(invocation(), memory_context=memory_summary(),
                            memory_tenant_id="tenant-a", memory_scope="default")

    assert result.failure.code == "memory_context_expired"
    terminal = observer.events[-3:]
    assert [event.event_type for event in terminal] == ["task_completed", "memory_expired", "task_failed"]
    # The completion callback saw a candidate only, never an immutable success.
    # Consumers can safely treat the final rejected/failed pair as terminal.
    assert terminal[0].status == "accepted"
    assert terminal[0].payload.outcome_code == "selected"
    assert terminal[0].output_hash is None
    assert terminal[1].status == "rejected"
    assert terminal[2].status == "failed"
    assert "final_decision" not in [event.event_type for event in observer.events]


def test_driver_finalizes_memory_bound_success_only_after_candidate_completion_audit():
    driver, observer, _ = make_driver(Client(response()))

    result = driver.execute(invocation(), memory_context=memory_summary(),
                            memory_tenant_id="tenant-a", memory_scope="default")

    assert result.output == {"answer": "known"}
    terminal = observer.events[-2:]
    assert [event.event_type for event in terminal] == ["task_completed", "final_decision"]
    assert terminal[0].status == "accepted"
    assert terminal[0].payload.outcome_code == "selected"
    assert terminal[0].output_hash is None
    # Both observer-visible terminal records are candidates.  The driver only
    # returns the output after the final callback and expiry check have ended.
    assert terminal[1].status == "accepted"
    assert terminal[1].payload.outcome_code == "selected"
    assert terminal[1].output_hash is None


def test_driver_discards_memory_bound_output_when_summary_expires_in_final_decision_audit():
    clock = Clock()

    class SlowObserver(Observer):
        def record(self, event):
            super().record(event)
            if event.event_type == "final_decision":
                clock.time = 50.0

    driver, observer, _ = make_driver(Client(response()), clock=clock, audit_observer=SlowObserver())
    result = driver.execute(invocation(), memory_context=memory_summary(),
                            memory_tenant_id="tenant-a", memory_scope="default")

    assert result.failure.code == "memory_context_expired"
    terminal = observer.events[-4:]
    assert [event.event_type for event in terminal] == [
        "task_completed", "final_decision", "memory_expired", "task_failed",
    ]
    # No observer-visible event claims a successful answer before the last
    # expiry check.  In particular, the model output is never hashed here.
    assert [(event.status, event.payload.outcome_code, event.output_hash) for event in terminal[:2]] == [
        ("accepted", "selected", None), ("accepted", "selected", None),
    ]
    assert terminal[2].status == "rejected"
    assert terminal[3].status == "failed"


def test_driver_accepts_only_summary_as_dynamic_memory_after_stable_system_prefix():
    client = Client(response(), response())
    driver, observer, _ = make_driver(client)
    driver.execute(invocation())
    summary = memory_summary()
    result = driver.execute(invocation(), memory_context=summary, memory_tenant_id="tenant-a", memory_scope="default")
    assert result.output == {"answer": "known"}
    baseline, enriched = client.requests
    assert enriched.messages[0] == baseline.messages[0]
    assert enriched.messages[1] == baseline.messages[1]
    assert enriched.messages[2][0] == "user"
    dynamic = json.loads(enriched.messages[2][1])
    assert dynamic["trust"] == "untrusted_memory"
    assert dynamic["evidence_ids"] == ["event-1", "event-2"]
    assert dynamic["summary_hash"] == summary.summary_hash
    assert "Check cited observations" not in json.dumps([entry.model_dump(mode="json") for entry in observer.events])


@pytest.mark.parametrize("value", ["raw memory", {"rule": "raw rule"}, object()])
def test_driver_rejects_raw_memory_context_before_model_call(value):
    client = Client(response())
    driver, _, _ = make_driver(client)
    result = driver.execute(invocation(), memory_context=value, memory_tenant_id="tenant-a", memory_scope="default")
    assert result.failure.code == "invalid_memory_context"
    assert client.requests == []


@pytest.mark.parametrize("binding", [{}, {"memory_tenant_id": "tenant-b", "memory_scope": "default"},
                                     {"memory_tenant_id": "tenant-a", "memory_scope": "private"}])
def test_driver_requires_explicit_matching_tenant_and_scope_for_memory(binding):
    client = Client(response())
    driver, _, _ = make_driver(client)
    result = driver.execute(invocation(), memory_context=memory_summary(), **binding)
    assert result.failure.code == "invalid_memory_context"
    assert client.requests == []


@pytest.mark.parametrize("state", ["no_memory", "failed", "conflict"])
def test_driver_unsafe_memory_injects_nothing_and_fallback_abstains(state):
    client = Client()
    driver, observer, _ = make_driver(client)
    summary = memory_summary(conflict_state=state, rules=(), confidence=0.0)
    result = driver.execute(invocation(cost_limit_usd=0.0), memory_context=summary,
                            memory_tenant_id="tenant-a", memory_scope="default")
    assert result.output == {"answer": "不知道"}
    assert result.origin == "abstention"
    assert client.requests == []
    assert observer.events[-1].event_type == "task_completed"


def test_memory_context_cannot_reuse_a_response_cache_without_memory_binding():
    approved = {schema().digest: {"fixed", "semantic"}}
    exact = ExactResponseCache(safe_answers=approved)
    key = ExactCacheKey.from_metadata(sha256(canonical_json(invocation().user_payload).encode()).hexdigest(), metadata())
    exact.put(key, {"answer": "fixed"}, metadata(), now=0.0)
    semantic = SemanticRequestCache(safe_answers=approved)
    semantic.put(SemanticCacheEntry("seed", (1.0, 0.0), {"answer": "semantic"}, metadata(), 0.0, "vector-v1", "luna-v1"))
    client = Client(response())
    driver, _, _ = make_driver(client, exact_cache=exact, semantic_cache=semantic,
                               cache_context=lambda inv: cache_request(), safe_answers=approved)
    result = driver.execute(invocation(), memory_context=memory_summary(), memory_tenant_id="tenant-a", memory_scope="default")
    assert result.origin == "model" and len(client.requests) == 1


def test_driver_import_and_summary_execution_need_no_memory_storage_or_writer_modules():
    import subprocess
    import sys
    # A fresh interpreter catches direct and transitive imports, even when the
    # main test process already loaded storage for unrelated integration cases.
    script = '''
import importlib.abc
import sys
from datetime import datetime, timezone
class DenyStorage(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname in {"market_agent.workflow_memory_sqlite", "market_agent.workflow_memory_lifecycle",
                        "market_agent.workflow_memory_promotion", "market_agent.workflow_object_store"}:
            raise AssertionError("driver imported memory writer: " + fullname)
sys.meta_path.insert(0, DenyStorage())
from market_agent.workflow_agent_driver import AgentDriver
from market_agent.workflow_memory_retrieval import CoreExperienceSummary
assert AgentDriver is not None
now = datetime.now(timezone.utc)
assert CoreExperienceSummary(tenant_id="tenant-a", scope="default", conflict_state="no_memory", issued_at=now, expires_at=now).as_dynamic_context() == ""
'''
    result = subprocess.run([sys.executable, "-c", script], text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("changes", [{"confidence": 0.1}, {"freshness_seconds": 100000},
    {"rules": ()}, {"contradicting_evidence_ids": ("contradiction",)}])
def test_driver_omits_weak_stale_or_conflicting_summary_even_if_labeled_clear(changes):
    client = Client(response())
    driver, _, _ = make_driver(client)
    result = driver.execute(invocation(), memory_context=memory_summary(**changes),
                             memory_tenant_id="tenant-a", memory_scope="default")
    assert result.output == {"answer": "known"}
    assert len(client.requests[0].messages) == 2


def test_driver_enforces_memory_budget_and_summary_hash_before_dispatch():
    from market_agent.workflow_memory_retrieval import CoreExperienceSummary, SummaryItem
    large = memory_summary(rules=(SummaryItem(record_id="rule-1", text="观" * 3000,
                                               evidence_ids=("event-1", "event-2")),))
    client = Client(response())
    driver, _, _ = make_driver(client)
    result = driver.execute(invocation(), memory_context=large, memory_tenant_id="tenant-a", memory_scope="default")
    assert result.output == {"answer": "known"}
    assert len(client.requests[0].messages) == 2
    forged = CoreExperienceSummary.model_construct(**dict(memory_summary().model_dump(), summary_hash="0" * 64))
    result = driver.execute(invocation(), memory_context=forged, memory_tenant_id="tenant-a", memory_scope="default")
    assert result.failure.code == "invalid_memory_context"
    assert len(client.requests) == 1


def test_driver_prefers_safe_semantic_cache_and_preserves_trace():
    """Calling the provider before a valid semantic hit breaks the ordered flow."""
    approved = {schema().digest: {"stable"}}
    cache = SemanticRequestCache(safe_answers=approved)
    cache.put(SemanticCacheEntry("seed", (1.0, 0.0), {"answer": "stable"}, metadata(), 0.0, "vector-v1", "luna-v1"))
    client = Client()
    driver, observer, _ = make_driver(client, semantic_cache=cache, cache_context=lambda inv: cache_request(), safe_answers=approved)

    result = driver.execute(invocation())

    assert result.origin == "semantic_cache"
    assert result.output == {"answer": "stable"}
    assert result.trace_id == TRACE_ID
    assert client.requests == []
    assert observer.events[-1].event_type == "task_completed"
    assert {event.trace_id for event in observer.events} == {TRACE_ID}


def test_fixed_cache_precedes_semantic_cache():
    """A valid fixed seed must win even when semantic storage also has an answer."""
    approved = {schema().digest: {"fixed", "semantic"}}
    exact = ExactResponseCache(safe_answers=approved)
    request_hash = sha256(canonical_json(invocation().user_payload).encode()).hexdigest()
    exact.put(ExactCacheKey.from_metadata(request_hash, metadata()), {"answer": "fixed"}, metadata(), now=0.0)
    semantic = SemanticRequestCache(safe_answers=approved)
    semantic.put(SemanticCacheEntry("seed", (1.0, 0.0), {"answer": "semantic"}, metadata(), 0.0, "vector-v1", "luna-v1"))
    driver, observer, _ = make_driver(Client(), exact_cache=exact, semantic_cache=semantic, cache_context=lambda inv: cache_request(), safe_answers=approved)

    result = driver.execute(invocation())

    assert result.origin == "fixed_cache"
    assert result.output == {"answer": "fixed"}
    assert "semantic_cache_hit" not in [event.event_type for event in observer.events]


@pytest.mark.parametrize("content", [
    'Here is the answer: {"answer":"known"}', '```json\n{"answer":"known"}\n```',
    '{"answer":"known","extra":1}', '{"answer":4}', '[]', 'null',
    '{"answer":"first","answer":"second"}', '{"answer":NaN}',
    '{"answer":"known"} {"answer":"other"}', '{"answer":Infinity}',
])
def test_driver_rejects_malformed_provider_output_without_retry_or_fallback(content):
    """Coercion, duplicate keys, wrappers, or schema errors must never become answers."""
    client = Client(response(content))
    driver, observer, clock = make_driver(client)

    result = driver.execute(invocation())

    assert result.failure.code == "malformed_output"
    assert result.failure.retryable is False
    assert result.output is None
    assert len(client.requests) == 1
    assert clock.waits == []
    assert "fallback_selected" not in [event.event_type for event in observer.events]


def test_malformed_provider_output_still_records_authoritative_attempt_usage() -> None:
    observations = ExecutionObservationCollector("run-1", TRACE_ID)
    provider_response = api().ModelResponse(
        content='{"answer": 4}',
        usage=AgentUsage(
            input_tokens=40,
            cached_input_tokens=12,
            output_tokens=7,
            web_search_tool_calls=2,
            cost_usd=0.01,
            model_tier=ModelTier.LUNA,
            provider_request_id="response-malformed",
            model_id="gpt-5.6-luna-2026-08-15",
            pricing_version="openai-standard-2026-08-01",
        ),
    )
    driver, _, _ = make_driver(
        Client(provider_response), attempt_observer=observations.record_attempt
    )

    result = driver.execute(invocation(execution_node="recover"))

    assert result.failure is not None and result.failure.code == "malformed_output"
    usage = observations.usage()
    assert len(usage.attempts) == 1
    assert usage.attempts[0].provider_request_id == "response-malformed"
    assert usage.attempts[0].node is CoreNodeName.RECOVER
    assert usage.attempts[0].tokens == TokenUsage(
        input_tokens=40,
        cached_input_tokens=12,
        output_tokens=7,
        web_search_tool_calls=2,
    )


def test_retry_without_provider_usage_counts_attempt_and_does_not_invent_tokens() -> None:
    observations = ExecutionObservationCollector("run-1", TRACE_ID)
    client = Client(ProviderError(status_code=503), response())
    driver, _, _ = make_driver(client, attempt_observer=observations.record_attempt)

    result = driver.execute(invocation())

    assert result.failure is None
    usage = observations.usage()
    assert usage.provider_attempt_count == 2
    assert usage.unverified_provider_attempt_count == 1
    assert usage.attempts[0].tokens is None
    assert usage.attempts[0].source == "provider_usage_unavailable"
    assert usage.aggregate == TokenUsage(input_tokens=4, output_tokens=2)


def test_usage_over_reservation_is_recorded_before_budget_failure() -> None:
    observations = ExecutionObservationCollector("run-1", TRACE_ID)
    provider_response = api().ModelResponse(
        content='{"answer":"known"}',
        usage=AgentUsage(
            input_tokens=0, output_tokens=100_000, cost_usd=0.12,
            model_tier=ModelTier.LUNA, provider="openai",
            provider_request_id="response-over-reservation",
            model_id="gpt-5.6-luna",
            pricing_version="openai-standard-2026-08-01",
            pricing_model_id="gpt-5.6-luna", pricing_band="short",
        ),
    )
    driver, _, _ = make_driver(
        Client(provider_response), attempt_observer=observations.record_attempt
    )

    result = driver.execute(invocation(pricing_band="short"))

    assert result.failure is not None and result.failure.code == "budget_exhausted"
    assert observations.usage().attempts[0].source == "provider_response"
    assert observations.usage().attempts[0].tokens == TokenUsage(
        input_tokens=0, output_tokens=100_000
    )
    assert observations.usage().estimated_cost_usd == 0.12


def test_fixed_cache_hit_records_explicit_zero_provider_execution() -> None:
    approved = {schema().digest: {"fixed"}}
    exact = ExactResponseCache(safe_answers=approved)
    request_hash = sha256(canonical_json(invocation().user_payload).encode()).hexdigest()
    exact.put(
        ExactCacheKey.from_metadata(request_hash, metadata()),
        {"answer": "fixed"},
        metadata(),
        now=0.0,
    )
    observations = ExecutionObservationCollector("run-1", TRACE_ID)
    driver, _, _ = make_driver(
        Client(),
        exact_cache=exact,
        cache_context=lambda inv: cache_request(),
        safe_answers=approved,
        attempt_observer=observations.record_attempt,
    )

    result = driver.execute(invocation())

    assert result.origin == "fixed_cache"
    usage = observations.usage()
    assert usage.provider_attempt_count == 0
    assert usage.execution_count == 1
    assert usage.attempts[0].source == "fixed_cache"
    assert usage.aggregate == TokenUsage(input_tokens=0, output_tokens=0)


@pytest.mark.parametrize("depth", [3000, 10000])
def test_deep_json_returns_typed_malformed_output_and_terminal_audit(depth):
    """Parsing and later result freezing must not leak recursion failures or lose the trace."""
    content = '{"answer":' + '[' * depth + '0' + ']' * depth + '}'
    client = Client(response(content))
    driver, observer, clock = make_driver(client)
    result = driver.execute(invocation())

    assert result.failure is not None
    assert result.failure.code == "malformed_output"
    assert result.failure.retryable is False
    assert result.output is None
    assert len(client.requests) == 1
    assert clock.waits == []
    assert observer.events[-1].event_type == "task_failed"
    assert {event.trace_id for event in observer.events} == {result.trace_id} == {TRACE_ID}
    assert not {"fallback_selected", "task_completed", "retry_scheduled"} & {event.event_type for event in observer.events}


def test_model_request_has_stable_prefix_then_canonical_user_context():
    """Dynamic data in the system prefix or a lost schema pin would break provider isolation."""
    client = Client(response(), response())
    driver, _, _ = make_driver(client)
    assert driver.execute(invocation()).output == {"answer": "known"}
    driver.execute(invocation(user_payload={"query": "different"}, trace_id="2" * 32))
    first, second = client.requests
    assert first.messages[0][0] == "system"
    assert first.messages[0][1].startswith("Stable released prefix.")
    assert first.messages[0] == second.messages[0]
    assert "不知道" in first.messages[0][1]
    assert "internally" in first.messages[0][1]
    assert "private payload" not in first.messages[0][1]
    assert first.messages[1] == ("user", '{"context":"private payload","query":"supported answer"}')
    assert first.trace_id == TRACE_ID
    assert first.output_schema_digest == schema().digest
    assert first.temperature == 0.0
    assert first.cost_limit_usd == 0.1


def test_pricing_band_uses_final_rendered_request_not_context_summary_estimate():
    client = Client(response())
    driver, _, _ = make_driver(client)

    result = driver.execute(invocation(
        pricing_band="short",
        user_payload={"summary_token_estimate": 0, "rendered_payload": "x" * 272_000},
    ))

    assert result.failure is None
    assert client.requests[0].pricing_band == "long"


def test_transient_retry_keeps_tier_and_honors_retry_after():
    """Retry must wait before calling the same tier and account for the failed reservation."""
    client = Client(ProviderError(status_code=429, retry_after=1.5), response())
    driver, observer, clock = make_driver(client)
    result = driver.execute(invocation())
    assert result.output == {"answer": "known"}
    assert result.usage.cost_usd == pytest.approx(0.1000032)
    assert [request.model_tier for request in client.requests] == [ModelTier.LUNA, ModelTier.LUNA]
    assert [request.attempt for request in client.requests] == [0, 1]
    assert clock.waits == [1.5]
    assert "retry_scheduled" in [event.event_type for event in observer.events]


@pytest.mark.parametrize("error", [ProviderError(status_code=401), ProviderError(status_code=403), ProviderError(code="safety"), ValueError("secret provider prose")])
def test_permanent_errors_never_retry_downgrade_or_open_circuit(error):
    client = Client(error)
    breaker = CircuitBreaker(failure_threshold=1)
    driver, observer, clock = make_driver(client, circuit_breaker=breaker)
    result = driver.execute(invocation())
    assert result.failure.code == "provider_error"
    assert len(client.requests) == 1
    assert clock.waits == []
    assert breaker.acquire("luna", "extract", 2.0).kind == "allow"
    assert "fallback_selected" not in [event.event_type for event in observer.events]


def test_retry_exhaustion_downgrades_one_tier_at_a_time():
    client = Client(TimeoutError(), TimeoutError(), TimeoutError(), TimeoutError(), response())
    driver, observer, _ = make_driver(client)
    result = driver.execute(invocation(task_kind="coordinator", allowed_model_tier=ModelTier.SOL))
    assert result.output == {"answer": "known"}
    assert [request.model_tier for request in client.requests] == [ModelTier.SOL, ModelTier.SOL, ModelTier.TERRA, ModelTier.TERRA, ModelTier.LUNA]
    assert result.usage.cost_usd == pytest.approx(0.4000032)
    assert sum(event.event_type == "model_downgraded" for event in observer.events) == 2


@pytest.mark.parametrize("task_kind,allowed,want", [("extract", ModelTier.SOL, ModelTier.LUNA), ("analyze", ModelTier.SOL, ModelTier.TERRA), ("coordinator", ModelTier.LUNA, ModelTier.LUNA)])
def test_task_routing_respects_difficulty_and_allowed_ceiling(task_kind, allowed, want):
    client = Client(response(tier=want))
    driver, _, _ = make_driver(client)
    result = driver.execute(invocation(task_kind=task_kind, allowed_model_tier=allowed))
    assert result.usage.model_tier == want
    assert client.requests[0].model_tier == want


def test_open_circuit_skips_provider_and_downgrades():
    breaker = CircuitBreaker(failure_threshold=1, cooldown=10.0)
    breaker.record("sol", "coordinator", False, 0.0)
    client = Client(response(tier=ModelTier.TERRA))
    driver, observer, _ = make_driver(client, circuit_breaker=breaker)
    result = driver.execute(invocation(task_kind="coordinator", allowed_model_tier=ModelTier.SOL))
    assert result.usage.model_tier == ModelTier.TERRA
    assert [r.model_tier for r in client.requests] == [ModelTier.TERRA]
    assert "circuit_opened" in [e.event_type for e in observer.events]


def test_successful_half_open_probe_closes_circuit_and_is_audited():
    breaker = CircuitBreaker(failure_threshold=1, cooldown=1.0)
    breaker.record("luna", "extract", False, 0.0)
    driver, observer, _ = make_driver(Client(response()), circuit_breaker=breaker)
    assert driver.execute(invocation()).output == {"answer": "known"}
    assert breaker.acquire("luna", "extract", 1.0).kind == "allow"
    assert "circuit_probe" in [e.event_type for e in observer.events]
    assert "circuit_closed" in [e.event_type for e in observer.events]


@pytest.mark.parametrize("limits,want_calls", [({"max_attempts": 1}, 1), ({"cost_limit_usd": 0.1}, 1), ({"cost_limit_usd": 0.09}, 0), ({"deadline_epoch": 1.1}, 1), ({"deadline_epoch": 1.0}, 0), ({"attempt": 2, "max_attempts": 3}, 1)])
def test_exhausted_limits_never_schedule_an_unfunded_or_late_retry(limits, want_calls):
    client = Client(*[TimeoutError()] * 6)
    driver, observer, clock = make_driver(client)
    result = driver.execute(invocation(**limits))
    assert result.origin == "abstention"
    assert result.output == {"answer": "不知道"}
    assert len(client.requests) == want_calls
    assert clock.waits == []
    assert "retry_scheduled" not in [e.event_type for e in observer.events]


def test_local_knowledge_is_cited_and_never_promoted_to_cache():
    output_schema = schema(CitedAnswer)
    knowledge = LocalKnowledgeBase([KnowledgeDocument("doc-policy", "The supported answer is stable knowledge.", "stable knowledge")])
    fallback = FallbackPolicy((ModelTier.LUNA,), knowledge_base=knowledge)
    semantic = SemanticRequestCache()
    driver, observer, _ = make_driver(Client(TimeoutError()), output_schema=output_schema, fallback_policy=fallback, semantic_cache=semantic)
    result = driver.execute(invocation(output_schema, max_attempts=1))
    assert result.origin == "local_knowledge"
    assert result.output == {"answer": "stable knowledge", "citations": ("doc-policy",)}
    assert "local_knowledge_retrieved" in [e.event_type for e in observer.events]
    assert semantic.lookup((1.0, 0.0), metadata(output_schema), 1.0) is None


def test_unrelated_local_document_falls_through_to_exact_unknown():
    output_schema = schema(CitedAnswer)
    knowledge = LocalKnowledgeBase([KnowledgeDocument("doc-policy", "The supported answer is stable.", "stable")])
    fallback = FallbackPolicy((ModelTier.LUNA,), knowledge_base=knowledge)
    driver, observer, _ = make_driver(Client(TimeoutError()), output_schema=output_schema, fallback_policy=fallback)
    result = driver.execute(invocation(output_schema, max_attempts=1, user_payload={"query": "What is the capital of France?"}))

    assert result.origin == "abstention"
    assert result.output == {"answer": "不知道", "citations": ()}
    assert "local_knowledge_retrieved" not in [event.event_type for event in observer.events]
    assert "abstained" in [event.event_type for event in observer.events]


def test_knowledge_missing_citations_or_wrong_schema_falls_back_to_exact_unknown():
    class Uncited(FallbackPolicy):
        def resolve_local_knowledge(self, query):
            return type("Invalid", (), {"answer": "untrusted", "citations": ()})()
    fallback = Uncited((ModelTier.LUNA,), knowledge_base=LocalKnowledgeBase())
    driver, _, _ = make_driver(Client(TimeoutError()), fallback_policy=fallback)
    assert driver.execute(invocation(max_attempts=1)).output == {"answer": "不知道"}


@pytest.mark.parametrize("meta_change,response_value", [({}, {"answer": "BUY"}), ({"tenant_scope": "tenant-b"}, {"answer": "不知道"}), ({"prompt_release_digest": "b" * 64}, {"answer": "不知道"}), ({"category": "trade_decision"}, {"answer": "不知道"}), ({"expires_at": 1.0}, {"answer": "不知道"})])
def test_untrusted_cache_hits_are_revalidated_before_use(meta_change, response_value):
    class CorruptCache:
        def get(self, key, metadata, *, now):
            return CachedResponse(key, response_value, replace(metadata, **meta_change), 0.0)
    client = Client(response())
    driver, observer, _ = make_driver(client, exact_cache=CorruptCache(), cache_context=lambda inv: cache_request())
    result = driver.execute(invocation())
    assert result.origin == "model"
    assert result.output == {"answer": "known"}
    assert any(e.event_type == "fixed_cache_miss" and e.status == "rejected" for e in observer.events)


def test_audit_contains_only_trace_bound_hashes_and_redacted_metadata():
    secret = "sk-live-private-provider-secret"
    client = Client(TimeoutError(secret), response())
    driver, observer, _ = make_driver(client)
    result = driver.execute(invocation(user_payload={"query": secret}))
    encoded = json.dumps([e.model_dump(mode="json") for e in observer.events])
    assert result.trace_id == TRACE_ID
    assert {e.trace_id for e in observer.events} == {TRACE_ID}
    assert secret not in encoded
    assert "known" not in encoded
    assert all(e.input_hash and len(e.input_hash) == 64 for e in observer.events)
    assert observer.events[-1].output_hash == sha256(b'{"answer":"known"}').hexdigest()


@pytest.mark.parametrize("overrides", [{"prompt_release_digest": "b" * 64}, {"output_schema_digest": "b" * 64}, {"output_schema_id": "missing"}, {"task_kind": "unsupported"}])
def test_invalid_release_schema_or_task_binding_never_calls_provider(overrides):
    client = Client()
    driver, _, _ = make_driver(client)
    result = driver.execute(invocation(**overrides))
    assert result.failure.code == "invalid_invocation"
    assert client.requests == []


def test_core_verification_hook_gets_only_validated_result_and_cannot_replace_it():
    seen = []
    output_schema = schema(reflection_required=True)
    driver, observer, _ = make_driver(Client(response()), output_schema=output_schema, verification_hook=lambda result: seen.append(result))
    result = driver.execute(invocation(output_schema))
    assert seen == [result]
    assert any(e.event_type == "core_result_ready" and e.payload.reason_code == "reflection_required" for e in observer.events)


def test_memory_bound_reflection_rechecks_expiry_before_hook_after_core_candidate_audit():
    clock = Clock()
    seen = []

    class SlowObserver(Observer):
        def record(self, event):
            super().record(event)
            if event.event_type == "core_result_ready":
                clock.time = 50.0

    output_schema = schema(reflection_required=True)
    driver, observer, _ = make_driver(
        Client(response()), output_schema=output_schema, clock=clock, audit_observer=SlowObserver(),
        verification_hook=lambda result: seen.append(result),
    )

    result = driver.execute(invocation(output_schema), memory_context=memory_summary(),
                            memory_tenant_id="tenant-a", memory_scope="default")

    assert result.failure.code == "memory_context_expired"
    assert seen == []
    terminal = observer.events[-3:]
    assert [event.event_type for event in terminal] == ["core_result_ready", "memory_expired", "task_failed"]
    assert terminal[0].output_hash is None


def test_memory_bound_reflection_hook_receives_candidate_while_summary_is_current():
    seen = []
    output_schema = schema(reflection_required=True)
    driver, observer, _ = make_driver(
        Client(response()), output_schema=output_schema, verification_hook=lambda result: seen.append(result),
    )

    result = driver.execute(invocation(output_schema), memory_context=memory_summary(),
                            memory_tenant_id="tenant-a", memory_scope="default")

    assert result.output == {"answer": "known"}
    assert seen == [result]
    core = next(event for event in observer.events if event.event_type == "core_result_ready")
    assert core.output_hash is None


def test_memory_bound_reflection_discards_candidate_if_hook_consumes_remaining_authority():
    clock = Clock()
    seen = []
    output_schema = schema(reflection_required=True)

    def expires_in_hook(result):
        seen.append(result)
        clock.time = 50.0

    driver, observer, _ = make_driver(
        Client(response()), output_schema=output_schema, clock=clock, verification_hook=expires_in_hook,
    )

    result = driver.execute(invocation(output_schema), memory_context=memory_summary(),
                            memory_tenant_id="tenant-a", memory_scope="default")

    assert result.failure.code == "memory_context_expired"
    assert seen[0].output == {"answer": "known"}
    assert [event.event_type for event in observer.events][-2:] == ["memory_expired", "task_failed"]


def test_audit_failure_stops_before_provider_call():
    class BrokenObserver:
        def record(self, event):
            raise OSError("private database path")
    client = Client()
    driver, _, _ = make_driver(client, audit_observer=BrokenObserver())
    result = driver.execute(invocation())
    assert result.failure.code == "audit_unavailable"
    assert client.requests == []


@pytest.mark.parametrize("failure_event", ["circuit_probe", "prompt_composed"])
def test_audit_failure_reopens_acquired_probe_and_allows_recovery_after_cooldown(failure_event):
    """A pre-dispatch audit failure must not leak the only half-open probe."""
    breaker = CircuitBreaker(failure_threshold=1, cooldown=1.0)
    breaker.record("luna", "extract", False, 0.0)
    clock = Clock()

    class FailingObserver(Observer):
        fail_at = failure_event

        def record(self, event):
            if event.event_type == "circuit_probe":
                assert breaker.acquire("luna", "extract", clock.now()).kind == "reject"
            if event.event_type == self.fail_at:
                raise OSError("private audit failure")
            super().record(event)

    observer = FailingObserver()
    client = Client(response())
    driver, _, _ = make_driver(client, clock=clock, audit_observer=observer, circuit_breaker=breaker)

    result = driver.execute(invocation())

    assert result.failure.code == "audit_unavailable"
    assert result.output is None
    assert client.requests == []
    assert breaker.acquire("luna", "extract", clock.now()).kind == "reject"
    observer.fail_at = None
    clock.time = 1.5
    assert driver.execute(invocation()).origin == "abstention"
    assert client.requests == []
    clock.time = 2.0
    recovered = driver.execute(invocation())
    assert recovered.output == {"answer": "known"}
    assert recovered.origin == "model"
    assert len(client.requests) == 1
    assert breaker.acquire("luna", "extract", clock.now()).kind == "allow"


@pytest.mark.parametrize("origin", ["fixed_cache", "semantic_cache"])
@pytest.mark.parametrize("returned_at", [49.9, 50.0, 51.0])
def test_cache_expiry_is_rechecked_after_slow_adapter_returns(origin, returned_at):
    """An entry valid when lookup starts may expire while the adapter retrieves it."""
    clock = Clock()

    class SlowExactCache(ExactResponseCache):
        def get(self, key, metadata, *, now):
            entry = super().get(key, metadata, now=now)
            clock.time = returned_at
            return entry

    class SlowSemanticCache(SemanticRequestCache):
        def lookup(self, query, metadata, now):
            entry = super().lookup(query, metadata, now)
            clock.time = returned_at
            return entry

    meta = metadata()
    if origin == "fixed_cache":
        cache = SlowExactCache()
        request_hash = sha256(canonical_json(invocation().user_payload).encode()).hexdigest()
        cache.put(ExactCacheKey.from_metadata(request_hash, meta), {"answer": "不知道"}, meta, now=0.0)
        cache_options = {"exact_cache": cache}
    else:
        cache = SlowSemanticCache()
        cache.put(SemanticCacheEntry("seed", (1.0, 0.0), {"answer": "不知道"}, meta, 0.0, "vector-v1", "luna-v1"))
        cache_options = {"semantic_cache": cache}
    client = Client(response())
    driver, observer, _ = make_driver(client, clock=clock, cache_context=lambda inv: cache_request(meta), **cache_options)

    result = driver.execute(invocation())

    if returned_at < 50.0:
        assert result.origin == origin
        assert result.output == {"answer": "不知道"}
        assert client.requests == []
    else:
        assert result.origin == "model"
        assert result.output == {"answer": "known"}
        assert len(client.requests) == 1
        assert not any(event.event_type == origin + "_hit" for event in observer.events)
        assert any(event.event_type == origin + "_miss" and event.status == "rejected" for event in observer.events)


def test_driver_has_no_harness_persistence_execution_or_exchange_imports():
    """An authority-bearing import would let the driver bypass its scoped adapters."""
    import ast
    from pathlib import Path
    tree = ast.parse(Path(api().__file__).read_text(encoding="utf-8"))
    forbidden = ("workflow_harness", "workflow_state", "workflow_execution", "memory", "exchange", "queue", "sqlite", "sqlalchemy", "hyperliquid")
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.module == "market_agent.workflow_memory_retrieval":
                assert [alias.name for alias in node.names] == ["CoreExperienceSummary"]
                continue
            imported.append(node.module or "")
    assert not [name for name in imported if any(part in name for part in forbidden)]


@pytest.mark.parametrize("code", ["authentication", "authorization", "validation", "schema", "safety", "malformed_output"])
def test_permanent_provider_codes_override_transient_http_status(code):
    """A contradictory HTTP 503 must not turn a safety/schema denial into fallback."""
    client = Client(ProviderError(status_code=503, code=code), response())
    driver, _, clock = make_driver(client)
    result = driver.execute(invocation())
    assert result.failure is not None and result.failure.code == "provider_error"
    assert len(client.requests) == 1
    assert clock.waits == []


@pytest.mark.parametrize("changes", [
    {"vector_version": "vector-v2"}, {"model_version": "luna-v2"},
    {"request_vector": (0.0, 1.0)},
    {"request_vector": (0.95, (1.0 - 0.95**2) ** 0.5)},
])
def test_semantic_adapter_cannot_bypass_version_or_similarity_gates(changes):
    class CorruptSemantic:
        def lookup(self, query, metadata, now):
            candidate = SemanticCacheEntry("seed", (1.0, 0.0), {"answer": "不知道"}, metadata, 0.0, "vector-v1", "luna-v1")
            return replace(candidate, **changes)
    driver, observer, _ = make_driver(Client(response()), semantic_cache=CorruptSemantic(), cache_context=lambda inv: cache_request())
    result = driver.execute(invocation())
    assert result.origin == "model"
    assert any(e.event_type == "semantic_cache_miss" and e.status == "rejected" for e in observer.events)


def test_deadline_is_rechecked_after_audit_and_before_provider_dispatch():
    clock = Clock()
    class SlowObserver(Observer):
        def record(self, event):
            super().record(event)
            if event.event_type == "prompt_composed":
                clock.time = 100.0
    client = Client(response())
    driver, _, _ = make_driver(client, clock=clock, audit_observer=SlowObserver())
    result = driver.execute(invocation())
    assert client.requests == []
    assert result.output == {"answer": "不知道"}


def test_excessively_nested_provider_json_returns_typed_failure():
    content = '{"answer":' + '[' * 2000 + '0' + ']' * 2000 + '}'
    driver, _, _ = make_driver(Client(response(content)))
    result = driver.execute(invocation())
    assert result.failure.code == "malformed_output"


def test_strict_validation_enforces_extra_field_denial_independent_of_schema_hints():
    class Misconfigured(Answer):
        model_config = ConfigDict(extra="ignore", json_schema_extra={"additionalProperties": False})
    output_schema = schema(Misconfigured)
    driver, _, _ = make_driver(Client(response('{"answer":"known","hidden":"secret"}')), output_schema=output_schema)
    result = driver.execute(invocation(output_schema))
    assert result.failure is not None and result.failure.code == "malformed_output"


def test_schema_must_be_closed_and_support_exact_unknown_at_configuration_time():
    class Open(BaseModel):
        answer: str
    class NoUnknown(Answer):
        answer: int
    with pytest.raises(ValueError):
        schema(Open)
    with pytest.raises(ValueError):
        schema(NoUnknown)


def test_nested_output_is_validated_without_coercing_or_ignoring_fields():
    class Detail(BaseModel):
        model_config = ConfigDict(extra="forbid")
        count: int
    class Nested(Answer):
        detail: Detail | None = None
    output_schema = schema(Nested)
    driver, _, _ = make_driver(Client(response('{"answer":"known","detail":{"count":"2"}}')), output_schema=output_schema)
    assert driver.execute(invocation(output_schema)).failure.code == "malformed_output"


@pytest.mark.parametrize("usage", [None, {"input_tokens": 1}, AgentUsage(input_tokens=1, output_tokens=1, cost_usd=0.01, model_tier=ModelTier.SOL)])
def test_invalid_usage_or_wrong_model_cannot_be_a_success(usage):
    client = Client(api().ModelResponse(content='{"answer":"known"}', usage=usage))
    driver, _, _ = make_driver(client)
    assert driver.execute(invocation()).failure.code == "malformed_output"


def test_failed_half_open_probe_reopens_without_a_second_probe():
    breaker = CircuitBreaker(failure_threshold=1, cooldown=1.0)
    breaker.record("luna", "extract", False, 0.0)
    client = Client(TimeoutError())
    driver, observer, _ = make_driver(client, circuit_breaker=breaker)
    result = driver.execute(invocation(max_attempts=1))
    assert result.output == {"answer": "不知道"}
    assert breaker.acquire("luna", "extract", 1.5).kind == "reject"
    assert any(e.event_type == "circuit_recorded" and e.status == "failed" for e in observer.events)


def test_cache_failure_does_not_change_provider_circuit_state():
    class BrokenCache:
        def get(self, key, metadata, *, now):
            raise ConnectionError("local cache is unavailable")
    breaker = CircuitBreaker(failure_threshold=1)
    driver, _, _ = make_driver(Client(), exact_cache=BrokenCache(), cache_context=lambda inv: cache_request(), circuit_breaker=breaker)
    assert driver.execute(invocation(cost_limit_usd=0.0)).origin == "abstention"
    assert breaker.acquire("luna", "extract", 1.0).kind == "allow"


def test_driver_does_not_widen_safe_answers_from_later_policy_mutation():
    approved = {schema().digest: {"stable"}}
    exact = ExactResponseCache(safe_answers={schema().digest: {"unreviewed"}})
    key = ExactCacheKey.from_metadata(sha256(canonical_json(invocation().user_payload).encode()).hexdigest(), metadata())
    exact.put(key, {"answer": "unreviewed"}, metadata(), now=0.0)
    driver, _, _ = make_driver(Client(response()), exact_cache=exact, cache_context=lambda inv: cache_request(), safe_answers=approved)
    approved[schema().digest].add("unreviewed")
    assert driver.execute(invocation()).origin == "model"


def test_knowledge_that_cannot_fit_the_declared_schema_abstains():
    fallback = FallbackPolicy((ModelTier.LUNA,), knowledge_base=LocalKnowledgeBase([KnowledgeDocument("doc", "supported answer", "known")]))
    driver, _, _ = make_driver(Client(TimeoutError()), fallback_policy=fallback)
    result = driver.execute(invocation(max_attempts=1))
    assert result.origin == "abstention"
    assert result.output == {"answer": "不知道"}


def test_driver_events_can_be_recorded_by_existing_audit_writer(tmp_path):
    from market_agent.workflow_audit import AuditStore, AuditWriter
    store = AuditStore(tmp_path / "driver-audit.sqlite3")
    driver, _, _ = make_driver(Client(TimeoutError(), response()), audit_observer=AuditWriter(store))
    result = driver.execute(invocation())
    events = store.list(trace_id=TRACE_ID)
    assert result.output == {"answer": "known"}
    assert events[-1].event_type == "task_completed"
    assert [e.sequence for e in events] == list(range(1, len(events) + 1))
