from dataclasses import replace
from decimal import Decimal
from types import SimpleNamespace

import pytest

from market_agent import workflow_production_application as production
from market_agent.backend.settings import BackendSettings
from market_agent.workflow_embedding_client import EmbeddingResponse, embedding_cost
from market_agent.workflow_harness import HarnessKernel
from market_agent.workflow_observation import ExecutionObservationCollector
from market_agent.workflow_prompt_release import PromptReleaseRegistry
from market_agent.workflow_semantic_request_cache import SemanticRequestCache
from market_agent_test_bundle.tests import test_workflow_agent_driver as driver_tests
from market_agent_test_bundle.tests.test_task13_sourced_information import (
    ANSWER, CITATION, accept, fixture, request,
)


class Embeddings:
    def __init__(self, outcome=None):
        self.requests = []
        self.outcome = outcome

    def embed(self, *_args, **_kwargs):
        raise AssertionError("legacy vector-only embedding must never run")

    def invoke(self, value):
        self.requests.append(value)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        if callable(self.outcome):
            self.outcome(value)
        return EmbeddingResponse(
            vector=(1.0, 0.0), prompt_tokens=10, total_tokens=10,
            model_id=value.model_id, provider_request_id=f"embedding-{len(self.requests)}",
            estimated_cost_usd=float(embedding_cost(10)),
        )


def application(embedding=None):
    app, model, cache, clock, manager = fixture([{"answer": ANSWER, "citations": [CITATION]}])
    dependency = app._get_dependencies()
    embeddings = embedding or Embeddings()
    app._dependencies = replace(dependency, embedding_client=embeddings,
                                settings=BackendSettings(environment="test", embedding_dimension=2))
    return app, embeddings, model, cache, clock


def test_historical_lookup_is_typed_metered_and_accepted_refill_reuses_vector(monkeypatch):
    app, embeddings, model, cache, _ = application()
    req = request()
    execution = app.execute_workflow(req)
    assert len(embeddings.requests) == 1
    sent = embeddings.requests[0]
    assert (sent.workflow_id, sent.trace_id, sent.task_id) == (req.workflow_id, req.trace_id, "historical-query")
    assert sent.dimensions == 2 and sent.text == req.user_query
    assert sent.deadline_epoch <= 105.0
    attempt = execution.usage.attempts[0]
    assert attempt.source == "embedding_response"
    assert attempt.estimated_cost_usd == float(embedding_cost(10))
    assert execution.usage.provider_attempt_count == 2
    assert HarnessKernel._explicit_checkpoint_free_usage(execution.usage)
    accept(monkeypatch, app, req, execution)
    assert len(embeddings.requests) == 1
    assert len(cache._entries) == 1
    cached = app.execute_workflow(request(workflow_id="workflow-2", trace_id="2" * 32))
    assert cached.completion_kind == "historical_cache"
    assert cached.usage.estimated_cost_usd == float(embedding_cost(10))
    assert HarnessKernel._explicit_checkpoint_free_usage(cached.usage)
    assert len(model.requests) == 1


def test_failed_historical_embedding_keeps_reserved_cost_and_cannot_refill(monkeypatch):
    app, embeddings, _, cache, _ = application(Embeddings(RuntimeError("transport failed")))
    req = request()
    execution = app.execute_workflow(req)
    assert len(embeddings.requests) == 1
    attempt = execution.usage.attempts[0]
    assert attempt.source == "embedding_usage_unavailable"
    assert Decimal(str(attempt.estimated_cost_usd)) == embedding_cost(len(req.user_query.encode("utf-8")))
    accept(monkeypatch, app, req, execution)
    assert not cache._entries
    assert len(embeddings.requests) == 1


@pytest.mark.parametrize("cancel_before", [False, True])
def test_embedding_cancellation_prevents_vector_use_but_keeps_completed_charge(monkeypatch, cancel_before):
    signal = SimpleNamespace(cancelled=cancel_before)
    signal.is_cancelled = lambda: signal.cancelled
    def cancel(_request):
        signal.cancelled = True
    app, embeddings, model, cache, _ = application(Embeddings(cancel))
    req = request()
    execution = app.execute_workflow(req, cancellation_signal=signal)
    assert len(embeddings.requests) == (0 if cancel_before else 1)
    assert not model.requests
    assert execution.result.informational_answer.answer == "不知道"
    assert execution.usage.estimated_cost_usd == (0 if cancel_before else float(embedding_cost(10)))
    accept(monkeypatch, app, req, execution)
    assert not cache._entries


@pytest.mark.parametrize("semantic_limit", [0, 1])
def test_memory_embedding_cost_is_settled_before_graph_planning(monkeypatch, semantic_limit):
    app, embeddings, _, _, clock = application()
    app._dependencies = replace(app._get_dependencies(), memory_repository=object(),
                                semantic_embedding_attempt_limit=semantic_limit)
    captured = {}
    monkeypatch.setattr(production, "retrieve_memory", lambda query, _repository: captured.setdefault("query", query))
    monkeypatch.setattr(production, "build_core_experience_summary", lambda *_args, **_kwargs: None)
    class Runtime:
        def __init__(self, **kwargs):
            captured.update(kwargs)
        def services_for(self, _request):
            raise RuntimeError("planning reached")
    monkeypatch.setattr(production, "CoordinatorRuntime", Runtime)
    with pytest.raises(RuntimeError, match="planning reached"):
        app.execute_workflow(request(active_symbol="BTC"))
    assert len(embeddings.requests) == 1
    assert embeddings.requests[0].task_id == "core-memory-embedding"
    assert captured["query"].embedding == (1.0, 0.0)
    budget = captured["budget"]
    assert budget.settled_cost == float(embedding_cost(10))
    assert budget.remaining_attempts == 7 - semantic_limit
    assert budget.remaining_attempts + 2 + semantic_limit + captured["execution_observer"].usage().provider_attempt_count == 10
    assert budget.remaining_cost == pytest.approx(.75 - .12 - float(embedding_cost(10)))
    assert captured["execution_observer"].usage().provider_attempt_count == 1


def test_historical_compatibility_invalidates_when_luna_version_changes():
    kwargs = dict(tenant_id="tenant", now=100., prompt_release_digest="a" * 64)
    original = BackendSettings(environment="test")
    changed = replace(original, workflow_luna_model_version="luna-new")
    first = production._historical_metadata(settings=original, **kwargs)
    second = production._historical_metadata(settings=changed, **kwargs)
    assert not first.compatible_with(second)


def production_driver(monkeypatch, *, embedding=None, model=None, cancellation_check=lambda: False, cache=None):
    clock = driver_tests.Clock()
    embeddings = embedding or Embeddings()
    model = model or driver_tests.Client(driver_tests.response('{"answer":"不知道"}'))
    observed = ExecutionObservationCollector("run-1", "1" * 32)
    cache = cache or SemanticRequestCache()
    monkeypatch.setenv("OPENAI_API_KEY", "offline-test-key")
    monkeypatch.setattr(production, "OpenAIEmbeddingClient", lambda **_kwargs: embeddings)
    monkeypatch.setattr(production, "OpenAIModelClient", lambda **_kwargs: model)
    monkeypatch.setattr(production, "_SystemClock", lambda: clock)
    monkeypatch.setattr(production, "output_schemas", lambda: (driver_tests.schema(),))
    dependencies = production._production_dependencies(
        settings=BackendSettings(environment="test", embedding_dimension=2),
        memory_repository=None, semantic_cache=cache, historical_answer_cache=None,
        prompt_release_manager=PromptReleaseRegistry(releases=(driver_tests.prompt_release(),)),
        completion_hook=lambda *_args: None, audit_writer=driver_tests.Observer(),
    )
    driver = production._driver_for_run(dependencies, "tenant-a", observed.record_attempt, cancellation_check)
    return driver, embeddings, model, cache, observed, clock


def test_production_semantic_cache_is_queried_and_charged_once_before_safe_write(monkeypatch):
    driver, embeddings, model, cache, observed, clock = production_driver(monkeypatch)
    queries = []
    original_lookup = cache.lookup
    monkeypatch.setattr(cache, "lookup", lambda vector, metadata, now: (
        queries.append((vector, metadata)), original_lookup(vector, metadata, now)
    )[1])
    invocation = driver_tests.invocation()
    result = driver.execute(invocation)
    assert result.failure is None and result.origin == "model"
    assert len(embeddings.requests) == 1
    assert len(queries) == 1 and queries[0][0] == (1., 0.)
    assert len(cache._entries) == 1
    sent = embeddings.requests[0]
    assert (sent.workflow_id, sent.trace_id, sent.node) == (
        invocation.run_id, invocation.trace_id, invocation.execution_node)
    assert sent.task_id.startswith("semantic-embedding-") and sent.task_id != invocation.task_id
    assert sent.attempt == 0
    assert sent.deadline_epoch <= clock.now() + 5
    assert sent.cost_limit_usd <= invocation.cost_limit_usd
    assert result.usage.cost_usd == observed.usage().estimated_cost_usd
    # Different request JSON avoids the exact cache, while the admitted vector
    # locates the previously safe stored response via the actual semantic cache.
    next_driver, next_embeddings, next_model, _, _, _ = production_driver(monkeypatch, cache=cache)
    second = next_driver.execute(invocation.model_copy(update={"user_payload": {"query": "another safe request"}}))
    assert second.origin == "semantic_cache"
    assert second.usage.cost_usd == float(embedding_cost(10))
    assert len(model.requests) == 1
    assert len(next_embeddings.requests) == 1 and not next_model.requests


@pytest.mark.parametrize("outcome", [None, RuntimeError("transport uncertain")])
def test_run_scoped_semantic_allowance_admits_only_one_paid_embedding(monkeypatch, outcome):
    from concurrent.futures import ThreadPoolExecutor
    driver, embeddings, _, _, observed, _ = production_driver(monkeypatch, embedding=Embeddings(outcome))
    with ThreadPoolExecutor(3) as pool:
        contexts = tuple(pool.map(driver._cache_context, (
            driver_tests.invocation(task_id=f"task-{number}") for number in range(3)
        )))
    assert len(embeddings.requests) == 1
    assert sum(context.embedding_cost_usd > 0 for context in contexts) == 1
    assert observed.usage().provider_attempt_count == 1


def test_predispatch_rejection_restores_semantic_attempt_allowance(monkeypatch):
    driver, embeddings, _, _, _, _ = production_driver(monkeypatch)
    rejected = driver._cache_context(driver_tests.invocation(cost_limit_usd=.000000001))
    allowed = driver._cache_context(driver_tests.invocation(task_id="another-task"))
    assert rejected.vector is None and rejected.embedding_cost_usd == 0
    assert allowed.vector == (1., 0.) and len(embeddings.requests) == 1


def test_production_embedding_and_model_usage_pass_actual_graph_retry_validation(monkeypatch):
    from market_agent.workflow_graph import _observed_node
    from market_agent.workflow_observation import CoreNodeName
    from market_agent_test_bundle.tests.test_graph_embedding_work_items import _state
    driver, embeddings, _, _, observed, _ = production_driver(monkeypatch)
    req = request(workflow_id="run-1")
    state, plan = _state(req, observed)
    state.update(_observed_node(state, CoreNodeName.PLAN, lambda _: {"plan": plan}))
    result = driver.execute(driver_tests.invocation(task_id="technical-1"))
    assert result.failure is None
    update = _observed_node(state, CoreNodeName.DISPATCH, lambda _: {"reports": ()})
    assert "failure_reason" not in update
    checkpoints = observed.checkpoints()
    assert len(checkpoints) == 2
    assert checkpoints[-1].task_ids == ("technical-1", embeddings.requests[0].task_id)
    assert [item.attempts_consumed for item in checkpoints[-1].retry_state] == [1, 1]
    HarnessKernel._validate_checkpoint_sequence(checkpoints, SimpleNamespace(revision=0))


@pytest.mark.parametrize("outcome", [None, RuntimeError("transport failed")])
def test_semantic_embedding_consumes_invocation_budget_before_model_admission(monkeypatch, outcome):
    driver, embeddings, model, cache, observed, _ = production_driver(monkeypatch, embedding=Embeddings(outcome))
    result = driver.execute(driver_tests.invocation(cost_limit_usd=.01))
    assert len(embeddings.requests) == 1
    assert not model.requests
    assert result.origin == "abstention"
    assert result.usage.cost_usd == observed.usage().estimated_cost_usd > 0


@pytest.mark.parametrize("kind", ["cancelled", "oversized", "tiny_budget", "expired", "informational", "unsupported_node"])
def test_production_embedding_rejects_before_dispatch(monkeypatch, kind):
    driver, embeddings, _, _, _, clock = production_driver(monkeypatch, cancellation_check=lambda: kind == "cancelled")
    changes = {
        "oversized": {"user_payload": {"query": "x" * 9000}},
        "tiny_budget": {"cost_limit_usd": .000000001},
        "expired": {"deadline_epoch": clock.now()},
        "informational": {"output_schema_id": "workflow.informational.v1"},
        "unsupported_node": {"execution_node": "plan"},
    }.get(kind, {})
    context = driver._cache_context(driver_tests.invocation(**changes))
    assert not embeddings.requests
    assert context is None or context.vector is None


def test_production_driver_propagates_cancellation_and_captures_owner_fence(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from market_agent.backend.execution_fence import ExecutionFence, execution_fence_context
    fence = ExecutionFence()
    with execution_fence_context(fence):
        driver, embeddings, model, _, observed, _ = production_driver(monkeypatch)
    fence.latch_lost()
    with ThreadPoolExecutor(1) as pool:
        context = pool.submit(driver._cache_context, driver_tests.invocation()).result()
    assert context is None and not embeddings.requests


def test_production_cache_usage_failure_stops_before_model(monkeypatch):
    driver, embeddings, model, _, _, _ = production_driver(monkeypatch)
    def fail(_attempt):
        raise RuntimeError("ledger unavailable")
    # Reconstruct with the production closure's observation dependency failing.
    factory = production._production_dependencies(
        settings=BackendSettings(environment="test", embedding_dimension=2),
        memory_repository=None, semantic_cache=SemanticRequestCache(), historical_answer_cache=None,
        prompt_release_manager=PromptReleaseRegistry(releases=(driver_tests.prompt_release(),)),
        completion_hook=lambda *_args: None, audit_writer=driver_tests.Observer(),
    ).driver_factory
    driver = factory("tenant-a", fail)
    result = driver.execute(driver_tests.invocation())
    assert result.failure.code == "usage_observation_unavailable"
    assert len(embeddings.requests) == 1 and not model.requests


def test_memory_bound_invocation_skips_production_semantic_embedding(monkeypatch):
    driver, embeddings, model, _, observed, _ = production_driver(monkeypatch)
    result = driver.execute(driver_tests.invocation(), memory_context=driver_tests.memory_summary(),
                            memory_tenant_id="tenant-a", memory_scope="default")
    assert result.failure is None and result.origin == "model"
    assert len(model.requests) == 1 and not embeddings.requests
    assert observed.usage().provider_attempt_count == 1


def test_historical_input_over_bound_never_dispatches_or_refills(monkeypatch):
    app, embeddings, _, cache, _ = application()
    req = request(user_query="water freezes zero Celsius " + "水" * 3000)
    execution = app.execute_workflow(req)
    assert not embeddings.requests
    accept(monkeypatch, app, req, execution)
    assert not cache._entries


def test_historical_late_embedding_charge_is_kept_but_vector_not_refilled(monkeypatch):
    app, embeddings, _, cache, clock = application()
    def late(_request):
        clock.time += 6
    embeddings.outcome = late
    req = request()
    execution = app.execute_workflow(req)
    assert execution.usage.attempts[0].source == "embedding_response"
    assert execution.usage.attempts[0].estimated_cost_usd == float(embedding_cost(10))
    accept(monkeypatch, app, req, execution)
    assert not cache._entries and len(embeddings.requests) == 1


def test_missing_candidate_vector_after_restart_omits_refill_without_new_provider_work(monkeypatch):
    app, embeddings, _, cache, _ = application()
    req = request()
    execution = app.execute_workflow(req)
    app._informational_candidates.clear()
    accept(monkeypatch, app, req, execution)
    assert not cache._entries and len(embeddings.requests) == 1
