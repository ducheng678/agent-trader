"""Paid host embeddings are durable, distinct work without specialist retry authority."""

from types import SimpleNamespace

from market_agent.workflow_contracts import (
    AgentTask, CoordinatorPlan, ModelTier, TaskDifficulty, TaskType,
    WorkflowMode, WorkflowRequest,
)
from market_agent.workflow_graph import WorkflowServices, _observed_node
from market_agent.workflow_harness import HarnessKernel
from market_agent.workflow_observation import (
    AttemptUsage, CoreNodeName, ExecutionObservationCollector, TokenUsage,
)


def _request():
    return WorkflowRequest(
        workflow_id="workflow-embedding", trace_id="1" * 32,
        user_query="分析市场", trigger_reason="manual_once",
    )


def _attempt(request, task_id, node, *, source="embedding_response"):
    return AttemptUsage(
        workflow_id=request.workflow_id, trace_id=request.trace_id,
        task_id=task_id, attempt=0, node=node, provider="openai",
        provider_request_id="req-" + task_id, model_id="text-embedding-3-small",
        model_tier=None, pricing_version="openai-embedding-2026-09-15",
        pricing_model_id="text-embedding-3-small", pricing_band=None,
        tokens=(TokenUsage(input_tokens=2, output_tokens=0)
                if source == "embedding_response" else None),
        estimated_cost_usd=(0.00000004 if source == "embedding_response" else 0.00016),
        latency_ms=1, source=source,
    )


def _state(request, observer):
    task = AgentTask(
        task_id="technical-1", workflow_id=request.workflow_id,
        trace_id=request.trace_id, task_type=TaskType.TECHNICAL,
        objective="Analyze technical evidence.", context_summary_id="summary-1",
        allowed_data=("context_summary",), allowed_tools=(),
        expected_output="technical-v1", acceptance_criteria=("Return a report.",),
        difficulty=TaskDifficulty.NORMAL, model_tier=ModelTier.TERRA,
        prompt_version="technical-v1", attempt_timeout_seconds=30,
        maximum_retries=1, reserved_cost=0.05, remaining_workflow_cost=0.10,
        analysis_steps=("Inspect.", "Assess.", "Return."),
        escalation_rule="return_to_coordinator",
        conflict_return_rule="return_typed_conflict",
    )
    plan = CoordinatorPlan(
        workflow_id=request.workflow_id, trace_id=request.trace_id,
        revision=0, mode=WorkflowMode.ACTIVE, tasks=(task,),
    )
    services = WorkflowServices(
        plan=lambda _request: plan, dispatch=lambda _plan: (),
        decide=lambda *_args: None, technical=lambda _reports: None,
        execution_observer=observer,
    )
    return {"request": request, "services": services}, plan


def test_pregraph_embedding_is_registered_at_plan_checkpoint():
    request = _request()
    observer = ExecutionObservationCollector(request.workflow_id, request.trace_id)
    observer.record_attempt(_attempt(request, "core-memory-embedding", CoreNodeName.PLAN))
    state, plan = _state(request, observer)
    update = _observed_node(state, CoreNodeName.PLAN, lambda _state: {"plan": plan})
    assert update == {"plan": plan}
    checkpoint = observer.checkpoints()[0]
    assert checkpoint.task_ids == ("technical-1", "core-memory-embedding")
    assert checkpoint.work_items[1].task_kind == "embedding"
    assert checkpoint.work_items[1].worker_id == "embedding-agent"
    assert checkpoint.completed_task_ids == ("core-memory-embedding",)
    HarnessKernel._validate_checkpoint_sequence((checkpoint,), SimpleNamespace(revision=0))


def test_semantic_embedding_has_distinct_dispatch_task_and_zero_retry_budget():
    request = _request()
    observer = ExecutionObservationCollector(request.workflow_id, request.trace_id)
    state, plan = _state(request, observer)
    state.update(_observed_node(state, CoreNodeName.PLAN, lambda _state: {"plan": plan}))
    host_task_id = "semantic-embedding-" + "a" * 40
    observer.record_attempt(_attempt(request, host_task_id, CoreNodeName.DISPATCH))
    update = _observed_node(state, CoreNodeName.DISPATCH, lambda _state: {"reports": ()})
    assert update == {"reports": ()}
    checkpoints = observer.checkpoints()
    assert checkpoints[-1].task_ids == ("technical-1", host_task_id)
    assert checkpoints[-1].work_items[1].maximum_retries == 0
    assert checkpoints[-1].retry_state[1].attempts_consumed == 1
    HarnessKernel._validate_checkpoint_sequence(checkpoints, SimpleNamespace(revision=0))


def test_failed_optional_embedding_is_registered_failed_not_silently_free():
    request = _request()
    observer = ExecutionObservationCollector(request.workflow_id, request.trace_id)
    observer.record_attempt(_attempt(
        request, "historical-query", CoreNodeName.PLAN,
        source="embedding_usage_unavailable",
    ))
    state, plan = _state(request, observer)
    _observed_node(state, CoreNodeName.PLAN, lambda _state: {"plan": plan})
    checkpoint = observer.checkpoints()[0]
    assert checkpoint.failed_task_ids == ("historical-query",)
    assert checkpoint.work_items[1].execution_state == "failed"
    HarnessKernel._validate_checkpoint_sequence((checkpoint,), SimpleNamespace(revision=0))


def test_host_cache_attempt_cannot_spoof_embedding_work_registry():
    request = _request()
    observer = ExecutionObservationCollector(request.workflow_id, request.trace_id)
    observer.record_attempt(_attempt(request, "historical-query", CoreNodeName.PLAN).model_copy(update={
        "source": "fixed_cache", "provider": "host", "model_id": "fixed_cache",
        "pricing_model_id": None, "pricing_version": "host-zero-v1",
        "tokens": TokenUsage.zero(), "estimated_cost_usd": 0.0,
    }))
    state, plan = _state(request, observer)
    update = _observed_node(state, CoreNodeName.PLAN, lambda _state: {"plan": plan})
    assert "failure_reason" in update
    assert observer.checkpoints() == ()
