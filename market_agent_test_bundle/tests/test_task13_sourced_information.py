from datetime import datetime, timezone
import json
from types import SimpleNamespace

import pytest

from market_agent.backend.settings import BackendSettings
from market_agent.local_knowledge_base import KnowledgeDocument, LocalKnowledgeBase
from market_agent.workflow_agent_contracts import AgentUsage, ModelTier
from market_agent.workflow_agent_driver import AgentDriver, ModelResponse
from market_agent.workflow_circuit_breaker import CircuitBreaker
from market_agent.workflow_contracts import KnowledgeStatus, TerminalMode, WorkflowRequest, canonical_workflow_result_digest
from market_agent.workflow_fallback import FallbackPolicy
from market_agent.workflow_embedding_client import EmbeddingResponse, embedding_cost
from market_agent.workflow_historical_answer_cache import InMemoryHistoricalAnswerCache
from market_agent.workflow_informational_agent import informational_output_schema, informational_release
from market_agent.workflow_memory_result_writer import AcceptedOutcomeProof
from market_agent.workflow_production_application import ProductionDependencies, ProductionWorkflowApplication, _capture_workflow_prompt_pin
from market_agent.workflow_prompt_config import PromptPin
from market_agent.workflow_prompt_release import PromptReleaseRegistry
from market_agent.workflow_retry_policy import ProviderError, RetryPolicy
from market_agent_test_bundle.tests.test_workflow_production_application import _terminal_receipt


ANSWER = "Water freezes at zero degrees Celsius."
CITATION = "approved-water-v1"


class Clock:
    time = 100.0

    def now(self):
        return self.time

    def sleep(self, seconds):
        self.time += seconds


class Client:
    def __init__(self, outcomes):
        self.requests = []
        self.outcomes = list(outcomes)

    def invoke(self, request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if callable(outcome):
            outcome = outcome()
        return ModelResponse(content=json.dumps(outcome), usage=AgentUsage(
            input_tokens=10, output_tokens=10, cost_usd=0.001,
            model_tier=request.model_tier, provider="openai",
            pricing_version="openai-standard-2026-08-01",
            pricing_model_id=f"gpt-5.6-{request.model_tier.value}", pricing_band="short",
        ))


def request(**changes):
    values = dict(workflow_id="workflow-1", trace_id="1" * 32,
                  user_query="What freezes at zero degrees Celsius?", trigger_reason="manual_once")
    values.update(changes)
    return WorkflowRequest(**values)


def fixture(outcomes=(), *, corpus=True):
    clock, client = Clock(), Client(outcomes)
    kb = LocalKnowledgeBase((KnowledgeDocument(CITATION, ANSWER, ANSWER),) if corpus else ())
    cache = InMemoryHistoricalAnswerCache()
    writer = SimpleNamespace(healthy=True, record=lambda _event: None)
    release = informational_release()
    manager = SimpleNamespace(current=lambda: PromptPin(
        release_id=release.release_id, release_digest=release.digest,
        output_schema_hash=informational_output_schema().digest,
        manifest_hash="a" * 64, release=release,
    ))
    def factory(_tenant, observer):
        return AgentDriver(
            model_client=client, audit_observer=writer, clock=clock, random=lambda a, b: a,
            prompt_releases=PromptReleaseRegistry(releases=(release,)),
            output_schemas=(informational_output_schema(),),
            retry_policy=RetryPolicy(max_attempts=3), circuit_breaker=CircuitBreaker(),
            fallback_policy=FallbackPolicy((ModelTier.TERRA, ModelTier.LUNA), knowledge_base=kb),
            model_costs={ModelTier.TERRA: .05, ModelTier.LUNA: .01}, attempt_observer=observer,
        )
    def embed(value):
        return EmbeddingResponse(
            vector=(1.0, 0.0), prompt_tokens=10, total_tokens=10,
            model_id=value.model_id, provider_request_id="embedding-" + value.workflow_id,
            estimated_cost_usd=float(embedding_cost(10)),
        )
    dependencies = ProductionDependencies(
        settings=BackendSettings(environment="test", embedding_dimension=2), driver_factory=factory,
        audit_writer=writer, memory_repository=None,
        embedding_client=SimpleNamespace(invoke=embed),
        completion_hook=lambda *_args: None, historical_answer_cache=cache,
        prompt_release_manager=manager, clock=clock.now, local_knowledge_base=kb,
    )
    return ProductionWorkflowApplication(lambda: dependencies), client, cache, clock, manager


def accept(monkeypatch, app, req, execution):
    monkeypatch.setattr("market_agent.workflow_memory_result_writer.verify_committed_transition_receipt", lambda _: True)
    proof = AcceptedOutcomeProof.bind(
        req, execution.result, terminal_receipt=_terminal_receipt(
            req, execution.prompt_release_digest, canonical_workflow_result_digest(execution.result)),
        prompt_release_digest=execution.prompt_release_digest,
        accepted_at=datetime.now(timezone.utc),
    )
    app.commit_accepted_result(req, execution.result, proof)


def test_static_miss_model_answer_is_cited_bounded_and_refilled_only_after_acceptance(monkeypatch):
    app, client, cache, clock, _ = fixture([{"answer": ANSWER, "citations": [CITATION]}])
    req = request()
    execution = app.execute_workflow(req)
    assert execution.completion_kind == "informational"
    assert execution.result.terminal_mode is TerminalMode.INFORMATIONAL
    assert execution.result.informational_answer.source_references == (CITATION,)
    assert execution.result.route_history == ("informational_model",)
    assert len(client.requests) == 1
    sent = client.requests[0]
    assert sent.model_tier is ModelTier.TERRA and sent.temperature == 0.0
    assert sent.deadline_epoch <= clock.now() + 30
    assert next(item for item in execution.usage.attempts if item.source == "provider_response").tokens.input_tokens == 10
    assert not cache._entries
    accept(monkeypatch, app, req, execution)
    assert len(cache._entries) == 1
    second = app.execute_workflow(request(workflow_id="workflow-2", trace_id="2" * 32))
    assert second.completion_kind == "historical_cache"
    assert len(client.requests) == 1
    assert second.result.informational_answer.answer == ANSWER


def test_provider_failure_downgrades_to_luna_then_local_cited_answer_without_cache(monkeypatch):
    app, client, cache, _, _ = fixture([ProviderError(status_code=503)] * 3)
    req = request()
    execution = app.execute_workflow(req)
    assert [item.model_tier for item in client.requests] == [ModelTier.TERRA, ModelTier.LUNA, ModelTier.LUNA]
    assert execution.result.informational_answer.answer == ANSWER
    assert execution.result.route_history == ("informational_local_knowledge",)
    accept(monkeypatch, app, req, execution)
    assert not cache._entries


@pytest.mark.parametrize("output", [
    {"answer": ANSWER, "citations": ["forged-source"]},
    {"answer": "Water freezes at one thousand degrees Celsius.", "citations": [CITATION]},
    {"answer": ANSWER, "citations": []},
    {"answer": ANSWER, "citations": [CITATION], "order": "buy"},
])
def test_unapproved_output_degrades_locally_and_never_refills(monkeypatch, output):
    app, _, cache, _, _ = fixture([output])
    req = request()
    execution = app.execute_workflow(req)
    assert execution.result.informational_answer.answer == ANSWER
    assert execution.result.route_history == ("informational_local_knowledge",)
    accept(monkeypatch, app, req, execution)
    assert not cache._entries


@pytest.mark.parametrize("phase", ["before", "during"])
def test_cancellation_discards_answer_and_prevents_cache(monkeypatch, phase):
    signal = SimpleNamespace(cancelled=phase == "before")
    signal.is_cancelled = lambda: signal.cancelled
    def cancel():
        signal.cancelled = True
        return {"answer": ANSWER, "citations": [CITATION]}
    app, client, cache, _, _ = fixture([cancel])
    req = request()
    execution = app.execute_workflow(req, cancellation_signal=signal)
    assert execution.result.knowledge_status is KnowledgeStatus.INSUFFICIENT
    assert execution.result.informational_answer.answer == "不知道"
    assert len(client.requests) == (phase == "during")
    accept(monkeypatch, app, req, execution)
    assert not cache._entries


def test_missing_local_evidence_abstains_without_calling_provider(monkeypatch):
    app, client, cache, _, _ = fixture(corpus=False)
    req = request()
    execution = app.execute_workflow(req)
    assert execution.result.terminal_mode is TerminalMode.UNKNOWN
    assert execution.result.informational_answer.answer == "不知道"
    assert not client.requests
    accept(monkeypatch, app, req, execution)
    assert not cache._entries


def test_informational_component_is_part_of_workflow_prompt_pin():
    _, _, _, _, manager = fixture()
    pin, digest = _capture_workflow_prompt_pin(manager)
    component = pin.component("workflow.informational.v1", informational_output_schema().digest)
    assert component.release == informational_release()
    assert digest == pin.release_digest


def test_volatile_context_preserves_graph_route(monkeypatch):
    app, client, _, _, _ = fixture()
    import market_agent.workflow_production_application as production
    monkeypatch.setattr(production, "_request_context_records", lambda *_: (_ for _ in ()).throw(RuntimeError("graph path")))
    with pytest.raises(RuntimeError, match="graph path"):
        app.execute_workflow(request(active_symbol="BTC"))
    assert not client.requests


@pytest.mark.parametrize("query", ["Should I buy BTC now?", "BTC 今天可以做多吗", "What is the current water price?"])
def test_trading_or_time_sensitive_text_preserves_graph_route(monkeypatch, query):
    app, client, _, _, _ = fixture()
    import market_agent.workflow_production_application as production
    monkeypatch.setattr(production, "_request_context_records", lambda *_: (_ for _ in ()).throw(RuntimeError("graph path")))
    with pytest.raises(RuntimeError, match="graph path"):
        app.execute_workflow(request(user_query=query))
    assert not client.requests


def test_late_cancellation_does_not_refill_accepted_model_candidate(monkeypatch):
    signal = SimpleNamespace(cancelled=False)
    signal.is_cancelled = lambda: signal.cancelled
    app, _, cache, _, _ = fixture([{"answer": ANSWER, "citations": [CITATION]}])
    req = request()
    execution = app.execute_workflow(req, cancellation_signal=signal)
    signal.cancelled = True
    accept(monkeypatch, app, req, execution)
    assert not cache._entries


def test_cache_hit_is_rechecked_against_current_approved_local_source(monkeypatch):
    app, client, cache, _, _ = fixture([{"answer": ANSWER, "citations": [CITATION]}])
    req = request()
    execution = app.execute_workflow(req)
    accept(monkeypatch, app, req, execution)
    assert len(cache._entries) == 1
    monkeypatch.setattr(app._get_dependencies().local_knowledge_base, "lookup", lambda _: None)
    second = app.execute_workflow(request(workflow_id="workflow-2", trace_id="2" * 32))
    assert second.completion_kind == "informational"
    assert second.result.terminal_mode is TerminalMode.UNKNOWN
    assert len(client.requests) == 1


def test_expired_provider_answer_is_discarded_without_local_or_cache(monkeypatch):
    def late():
        clock.time += 400
        return {"answer": ANSWER, "citations": [CITATION]}
    app, client, cache, clock, _ = fixture([late])
    req = request()
    execution = app.execute_workflow(req)
    assert execution.result.terminal_mode is TerminalMode.UNKNOWN
    accept(monkeypatch, app, req, execution)
    assert not cache._entries


def test_model_abstention_never_refills(monkeypatch):
    app, _, cache, _, _ = fixture([{"answer": "不知道", "citations": []}])
    req = request()
    execution = app.execute_workflow(req)
    assert execution.result.terminal_mode is TerminalMode.UNKNOWN
    accept(monkeypatch, app, req, execution)
    assert not cache._entries
