from decimal import Decimal
from types import SimpleNamespace

import pytest

from market_agent.workflow_embedding_client import (
    EMBEDDING_PRICING_VERSION,
    EmbeddingRequest,
    EmbeddingResponse,
    OpenAIEmbeddingClient,
)
from market_agent.workflow_embedding_executor import BoundedEmbeddingExecutor, EmbeddingCancelled
from market_agent.workflow_observation import ExecutionObservationCollector, TokenUsage


def request(**changes):
    values = dict(text="hello world", workflow_id="workflow", trace_id="trace",
                  task_id="embedding:history", deadline_epoch=110.0,
                  cost_limit_usd=0.01, dimensions=2, cancellation_check=lambda: False)
    values.update(changes)
    return EmbeddingRequest(**values)


def response(**changes):
    values = dict(vector=(0.6, 0.8), prompt_tokens=2, total_tokens=2,
                  model_id="text-embedding-3-small", provider_request_id="req-embedding",
                  pricing_version=EMBEDDING_PRICING_VERSION, estimated_cost_usd=0.00000004)
    values.update(changes)
    return EmbeddingResponse(**values)


class Client:
    def __init__(self, effect=None):
        self.calls = []
        self.effect = effect

    def invoke(self, value):
        self.calls.append(value)
        return self.effect() if self.effect else response()


def test_embedding_response_is_accounted_exactly_in_workflow_ledger():
    collector = ExecutionObservationCollector("workflow", "trace")
    client = Client()
    result = BoundedEmbeddingExecutor(client, clock=lambda: 100.0).execute(
        request(), observation_callback=collector.record_attempt)
    usage = collector.usage()
    assert result.vector == (0.6, 0.8)
    assert usage.aggregate == TokenUsage(input_tokens=2, output_tokens=0)
    assert Decimal(str(usage.estimated_cost_usd)) == Decimal("0.00000004")
    assert usage.provider_attempt_count == 1
    assert usage.attempts[0].source == "embedding_response"
    assert usage.attempts[0].model_tier is None
    assert client.calls[0].deadline_epoch == 110.0


@pytest.mark.parametrize("changes,exception", [
    ({"cancellation_check": lambda: True}, EmbeddingCancelled),
    ({"deadline_epoch": 99.0}, TimeoutError),
    ({"cost_limit_usd": 0.000000001}, ValueError),
])
def test_rejected_request_does_not_dispatch_or_emit_fake_usage(changes, exception):
    collector = ExecutionObservationCollector("workflow", "trace")
    client = Client()
    with pytest.raises(exception):
        BoundedEmbeddingExecutor(client, clock=lambda: 100.0).execute(
            request(**changes), observation_callback=collector.record_attempt)
    assert client.calls == []
    assert collector.usage().attempts == ()


@pytest.mark.parametrize("cancel", [True, False])
def test_cancelled_or_late_response_is_charged_but_no_vector_escapes(cancel):
    state = {"cancelled": False, "now": 100.0}
    def effect():
        state.update(cancelled=cancel, now=100.0 if cancel else 111.0)
        return response()
    collector = ExecutionObservationCollector("workflow", "trace")
    with pytest.raises(EmbeddingCancelled if cancel else TimeoutError):
        BoundedEmbeddingExecutor(Client(effect), clock=lambda: state["now"]).execute(
            request(cancellation_check=lambda: state["cancelled"]),
            observation_callback=collector.record_attempt)
    assert collector.usage().estimated_cost_usd == 0.00000004
    assert collector.usage().provider_attempt_count == 1


def test_provider_failure_consumes_reservation_without_fabricating_tokens():
    def fail():
        raise RuntimeError("provider unavailable")
    collector = ExecutionObservationCollector("workflow", "trace")
    with pytest.raises(RuntimeError):
        BoundedEmbeddingExecutor(Client(fail), clock=lambda: 100.0).execute(
            request(), observation_callback=collector.record_attempt)
    usage = collector.usage()
    assert usage.estimated_cost_usd > 0
    assert usage.attempts[0].tokens is None
    assert usage.attempts[0].source == "embedding_usage_unavailable"
    assert usage.unverified_provider_attempt_count == 1


@pytest.mark.parametrize("vector", [(), (0.0, 0.0), (float("nan"), 0.1), (1.0,)])
def test_invalid_vector_is_discarded_after_recording_real_usage(vector):
    collector = ExecutionObservationCollector("workflow", "trace")
    with pytest.raises(ValueError):
        BoundedEmbeddingExecutor(Client(lambda: response(vector=vector)), clock=lambda: 100.0).execute(
            request(), observation_callback=collector.record_attempt)
    assert collector.usage().estimated_cost_usd == 0.00000004


def test_ledger_rejects_zero_or_mismatched_embedding_cost_and_llm_tier():
    collector = ExecutionObservationCollector("workflow", "trace")
    BoundedEmbeddingExecutor(Client(), clock=lambda: 100.0).execute(
        request(), observation_callback=collector.record_attempt)
    item = collector.usage().attempts[0]
    for update in ({"estimated_cost_usd": 0.0}, {"model_tier": "luna"},
                   {"pricing_model_id": "gpt-5.6-luna"},
                   {"tokens": TokenUsage(input_tokens=2, output_tokens=1)}):
        with pytest.raises(ValueError):
            item.model_copy(update=update)


def install_openai(monkeypatch, **changes):
    import openai
    wire = SimpleNamespace(model="text-embedding-3-small", _request_id="request-123",
                           data=[SimpleNamespace(embedding=[0.6, 0.8], index=0)],
                           usage=SimpleNamespace(prompt_tokens=2, total_tokens=2))
    for name, value in changes.items():
        setattr(wire, name, value)
    seen = {}
    class SDKClient:
        def __init__(self, **kwargs):
            seen["constructor"] = kwargs
            self.embeddings = self
        def create(self, **kwargs):
            seen["request"] = kwargs
            return wire
        def close(self):
            seen["closed"] = True
    monkeypatch.setattr(openai, "OpenAI", SDKClient)
    return seen


def test_openai_adapter_preserves_usage_and_disables_retries(monkeypatch):
    seen = install_openai(monkeypatch)
    client = OpenAIEmbeddingClient(api_key="test", model_id="text-embedding-3-small",
                                   dimensions=2, clock=lambda: 100.0)
    result = client.invoke(request())
    assert result.prompt_tokens == result.total_tokens == 2
    assert result.provider_request_id == "request-123"
    assert result.estimated_cost_usd == 0.00000004
    assert seen["constructor"]["max_retries"] == 0
    assert seen["constructor"]["timeout"] == 10.0
    assert seen["request"]["encoding_format"] == "float"
    assert seen["closed"] is True


@pytest.mark.parametrize("changes", [
    {"usage": None}, {"model": "wrong-model"},
    {"usage": SimpleNamespace(prompt_tokens=True, total_tokens=1)},
    {"usage": SimpleNamespace(prompt_tokens=2, total_tokens=3)},
    {"usage": SimpleNamespace(prompt_tokens=0, total_tokens=0)},
])
def test_openai_adapter_fails_closed_on_unverified_provider_identity_or_usage(monkeypatch, changes):
    install_openai(monkeypatch, **changes)
    collector = ExecutionObservationCollector("workflow", "trace")
    client = OpenAIEmbeddingClient(api_key="test", model_id="text-embedding-3-small",
                                   dimensions=2, clock=lambda: 100.0)
    with pytest.raises(ValueError):
        BoundedEmbeddingExecutor(client, clock=lambda: 100.0).execute(
            request(), observation_callback=collector.record_attempt)
    assert collector.usage().unverified_provider_attempt_count == 1
    assert collector.usage().estimated_cost_usd > 0.0


def informational_result():
    from market_agent.workflow_contracts import (
        Action, InformationalAnswer, KnowledgeStatus, TerminalMode, WorkflowResult,
    )
    return WorkflowResult(
        workflow_id="workflow", trace_id="trace", terminal_mode=TerminalMode.INFORMATIONAL,
        final_action=Action.NO_TRADE, knowledge_status=KnowledgeStatus.KNOWN,
        uncertainty_reason=None, evidence_references=("doc-1",),
        informational_answer=InformationalAnswer(
            answer="Static answer", source_references=("doc-1",),
            knowledge_status=KnowledgeStatus.KNOWN, uncertainty_reason=None,
        ),
    )


def nonprovider_attempt(source="historical_cache"):
    from market_agent.workflow_observation import AttemptUsage, CoreNodeName
    return AttemptUsage(
        workflow_id="workflow", trace_id="trace", task_id="cached-answer", attempt=0,
        node=CoreNodeName.PLAN, provider="none", provider_request_id="cache-1",
        model_id="none", model_tier=None, pricing_version="none",
        tokens=TokenUsage.zero(), estimated_cost_usd=0.0, latency_ms=0, source=source,
    )


@pytest.mark.parametrize("failed", [False, True])
def test_historical_hit_keeps_embedding_usage_without_graph_checkpoints(failed):
    from market_agent.workflow_observation import WorkflowExecution
    collector = ExecutionObservationCollector("workflow", "trace")
    def effect():
        if failed:
            raise ValueError("provider failure")
        return response()
    try:
        BoundedEmbeddingExecutor(Client(effect), clock=lambda: 100.0).execute(
            request(), observation_callback=collector.record_attempt)
    except ValueError:
        pass
    collector.record_attempt(nonprovider_attempt())
    execution = WorkflowExecution(result=informational_result(), usage=collector.usage(),
                                  checkpoints=(), completion_kind="historical_cache")
    assert execution.usage.provider_attempt_count == 1
    assert execution.usage.estimated_cost_usd > 0


def llm_attempt():
    return nonprovider_attempt().model_copy(update={
        "source": "provider_response", "model_tier": "luna", "provider": "openai",
        "model_id": "gpt-5.6-luna", "pricing_model_id": "gpt-5.6-luna",
        "pricing_version": "openai-standard-2026-08-01", "pricing_band": "short",
        "tokens": TokenUsage(input_tokens=1, output_tokens=0), "estimated_cost_usd": 0.00000020,
    })


def test_historical_hit_still_forbids_llm_usage():
    from market_agent.workflow_observation import WorkflowExecution, WorkflowUsage
    usage = WorkflowUsage.from_attempts("workflow", "trace", (llm_attempt(),))
    with pytest.raises(ValueError):
        WorkflowExecution(result=informational_result(), usage=usage,
                          checkpoints=(), completion_kind="historical_cache")


@pytest.mark.parametrize("kind", ["model", "local"])
def test_direct_informational_execution_requires_no_graph_and_keeps_actual_usage(kind):
    from market_agent.workflow_observation import WorkflowExecution, WorkflowUsage
    item = llm_attempt() if kind == "model" else nonprovider_attempt("local_knowledge")
    usage = WorkflowUsage.from_attempts("workflow", "trace", (item,))
    execution = WorkflowExecution(result=informational_result(), usage=usage,
                                  checkpoints=(), completion_kind="informational")
    assert execution.usage.attempts == (item,)


@pytest.mark.parametrize("change", ["no_sources", "foreign_sources", "identity", "empty_usage"])
def test_direct_informational_execution_fails_closed_on_missing_bindings(change):
    from market_agent.workflow_observation import WorkflowExecution, WorkflowUsage
    result = informational_result()
    usage = WorkflowUsage.from_attempts("workflow", "trace", (nonprovider_attempt("local_knowledge"),))
    if change in {"no_sources", "foreign_sources"}:
        result = result.model_copy(update={"informational_answer": result.informational_answer.model_copy(
            update={"source_references": () if change == "no_sources" else ("foreign-doc",)})})
    elif change == "identity":
        result = result.model_copy(update={"workflow_id": "other"})
    else:
        usage = WorkflowUsage.empty("workflow", "trace")
    with pytest.raises(ValueError):
        WorkflowExecution(result=result, usage=usage, checkpoints=(), completion_kind="informational")


def test_provider_usage_above_reservation_is_recorded_exactly_and_vector_rejected():
    collector = ExecutionObservationCollector("workflow", "trace")
    with pytest.raises(ValueError, match="reservation"):
        BoundedEmbeddingExecutor(Client(lambda: response(prompt_tokens=12, total_tokens=12,
                                                        estimated_cost_usd=0.00000024)),
                                 clock=lambda: 100.0).execute(
            request(), observation_callback=collector.record_attempt)
    assert collector.usage().estimated_cost_usd == 0.00000024


def test_failed_observation_callback_never_releases_vector_or_records_twice():
    seen = []
    def reject(value):
        seen.append(value)
        raise RuntimeError("ledger unavailable")
    with pytest.raises(RuntimeError, match="ledger"):
        BoundedEmbeddingExecutor(Client(), clock=lambda: 100.0).execute(
            request(), observation_callback=reject)
    assert len(seen) == 1


@pytest.mark.parametrize("kind", ["historical_cache", "informational"])
def test_direct_routes_reject_graph_checkpoints(kind):
    from market_agent.workflow_observation import CoreNodeName, NodeOutcome, WorkflowExecution
    collector = ExecutionObservationCollector("workflow", "trace")
    collector.record_attempt(nonprovider_attempt())
    collector.checkpoint(plan_revision=0, node=CoreNodeName.PLAN, outcome=NodeOutcome.COMPLETED,
                         task_ids=(), completed_task_ids=(), failed_task_ids=(), retry_state=(),
                         action_fingerprint="a" * 64)
    with pytest.raises(ValueError):
        WorkflowExecution(result=informational_result(), usage=collector.usage(),
                          checkpoints=collector.checkpoints(), completion_kind=kind)


def test_direct_informational_abstention_preserves_provider_usage():
    from market_agent.workflow_contracts import KnowledgeStatus, TerminalMode
    from market_agent.workflow_observation import WorkflowExecution, WorkflowUsage
    result = informational_result().model_copy(update={
        "terminal_mode": TerminalMode.UNKNOWN, "knowledge_status": KnowledgeStatus.INSUFFICIENT,
        "uncertainty_reason": "Cancelled before an answer could be accepted",
        "informational_answer": None,
    })
    usage = WorkflowUsage.from_attempts("workflow", "trace", (llm_attempt(),))
    execution = WorkflowExecution(result=result, usage=usage, checkpoints=(), completion_kind="informational")
    assert execution.usage.estimated_cost_usd > 0


@pytest.mark.parametrize("changes", [
    {"model_id": "text-embedding-3-large"}, {"dimensions": 1537},
    {"deadline_epoch": float("inf")}, {"cost_limit_usd": float("nan")},
    {"attempt": True}, {"text": ""}, {"cancellation_check": None},
])
def test_unbounded_or_unpriced_request_is_rejected(changes):
    with pytest.raises(ValueError):
        request(**changes)


def test_oversized_input_rejected_before_provider_dispatch():
    client = Client()
    with pytest.raises(ValueError, match="input allowance"):
        BoundedEmbeddingExecutor(client, clock=lambda: 100.0).execute(
            request(text="x" * 8193), observation_callback=lambda _: None)
    assert client.calls == []
