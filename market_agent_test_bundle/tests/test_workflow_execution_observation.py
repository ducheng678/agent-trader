from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from market_agent.workflow_observation import (
    AttemptUsage,
    CheckpointDecision,
    CheckpointPermit,
    CoreNodeName,
    ExecutionObservationCollector,
    NodeOutcome,
    ObservedWorkItem,
    TaskRetryState,
    TokenUsage,
    WorkflowUsage,
)


RUN_ID = "workflow-1"
TRACE_ID = "1" * 32


def _attempt(**updates: object) -> AttemptUsage:
    values: dict[str, object] = {
        "workflow_id": RUN_ID,
        "trace_id": TRACE_ID,
        "task_id": "task-1",
        "attempt": 0,
        "node": CoreNodeName.DISPATCH,
        "provider": "openai",
        "provider_request_id": "response-1",
        "model_id": "gpt-5.6-terra",
        "model_tier": "terra",
        "pricing_version": "openai-standard-2026-08-01",
        "pricing_model_id": "gpt-5.6-terra",
        "pricing_band": "short",
        "tokens": TokenUsage(
            input_tokens=100,
            cached_input_tokens=25,
            output_tokens=20,
            web_search_tool_calls=1,
        ),
        "estimated_cost_usd": 0.010395,
        "latency_ms": 125,
        "source": "provider_response",
    }
    values.update(updates)
    return AttemptUsage(**values)


def test_workflow_usage_preserves_mixed_model_attempts_and_exact_aggregate() -> None:
    first = _attempt()
    second = _attempt(
        task_id="task-2",
        attempt=1,
        provider_request_id="response-2",
        model_id="gpt-5.6-luna",
        model_tier="luna",
        pricing_model_id="gpt-5.6-luna",
        tokens=TokenUsage(input_tokens=40, cache_write_tokens=10, output_tokens=5),
        estimated_cost_usd=0.0000165,
        latency_ms=75,
    )

    usage = WorkflowUsage.from_attempts(RUN_ID, TRACE_ID, (first, second))

    assert usage.attempts == (first, second)
    assert usage.aggregate == TokenUsage(
        input_tokens=140,
        cached_input_tokens=25,
        cache_write_tokens=10,
        output_tokens=25,
        web_search_tool_calls=1,
    )
    assert Decimal(str(usage.estimated_cost_usd)) == Decimal("0.0104115")
    assert usage.provider_attempt_count == 2
    assert usage.total_latency_ms == 200
    assert usage.model_versions == (
        ("gpt-5.6-luna", "openai-standard-2026-08-01"),
        ("gpt-5.6-terra", "openai-standard-2026-08-01"),
    )


def test_missing_provider_usage_counts_attempt_without_fabricating_tokens() -> None:
    unavailable = _attempt(
        provider_request_id="attempt-without-response-usage",
        tokens=None,
        estimated_cost_usd=0.05,
        source="provider_usage_unavailable",
    )

    usage = WorkflowUsage.from_attempts(RUN_ID, TRACE_ID, (unavailable,))

    assert usage.provider_attempt_count == 1
    assert usage.unverified_provider_attempt_count == 1
    assert usage.aggregate == TokenUsage(input_tokens=0, output_tokens=0)
    assert usage.estimated_cost_usd == 0.05


def test_zero_provider_cache_path_is_explicit_and_not_missing_usage() -> None:
    cache = _attempt(
        provider="host",
        provider_request_id="cache-key-1",
        model_id="local-cache",
        model_tier=None,
        tokens=TokenUsage(input_tokens=0, output_tokens=0),
        estimated_cost_usd=0.0,
        latency_ms=0,
        source="fixed_cache",
        pricing_model_id=None,
        pricing_band=None,
    )

    usage = WorkflowUsage.from_attempts(RUN_ID, TRACE_ID, (cache,))

    assert usage.provider_attempt_count == 0
    assert usage.unverified_provider_attempt_count == 0
    assert usage.execution_count == 1


def test_collector_is_append_only_and_rejects_conflicting_duplicate_attempt() -> None:
    collector = ExecutionObservationCollector(RUN_ID, TRACE_ID)
    attempt = _attempt()

    collector.record_attempt(attempt)
    collector.record_attempt(attempt)
    assert collector.usage().attempts == (attempt,)

    with pytest.raises(ValueError, match="conflicting duplicate"):
        collector.record_attempt(
            attempt.model_copy(update={"latency_ms": 126})
        )


def test_checkpoint_binds_cumulative_usage_progress_retry_and_action_fingerprint() -> None:
    collector = ExecutionObservationCollector(RUN_ID, TRACE_ID)
    collector.record_attempt(_attempt())

    collector.checkpoint(
        plan_revision=2,
        node=CoreNodeName.DISPATCH,
        outcome=NodeOutcome.COMPLETED,
        task_ids=("task-1",),
        completed_task_ids=("task-1",),
        failed_task_ids=(),
        retry_state=(TaskRetryState(task_id="task-1", attempts_consumed=1,
                                    retries_consumed=0, retries_remaining=1),),
        work_items=(ObservedWorkItem(task_id="task-1", task_kind="technical",
                                     worker_id="technical-agent",
                                     owner_node=CoreNodeName.DISPATCH,
                                     maximum_retries=1, execution_state="succeeded",
                                     attempt_ids=("response-1",)),),
        action_fingerprint="a" * 64,
    )
    checkpoint = collector.checkpoints()[0]

    assert checkpoint.ordinal == 1
    assert checkpoint.usage == collector.usage()
    assert checkpoint.task_ids == ("task-1",)
    assert checkpoint.completed_task_ids == ("task-1",)
    assert checkpoint.retry_state[0].attempts_consumed == 1

    with pytest.raises(ValidationError):
        checkpoint.model_copy(update={"action_fingerprint": "not-a-digest"})


def test_checkpoint_requires_typed_authority_and_binds_actual_retry_counters() -> None:
    seen = []

    def authorize(checkpoint):
        seen.append(checkpoint)
        return CheckpointPermit(
            workflow_id=checkpoint.workflow_id,
            trace_id=checkpoint.trace_id,
            checkpoint_ordinal=checkpoint.ordinal,
            checkpoint_digest=checkpoint.canonical_digest(),
            decision=CheckpointDecision.CONTINUE,
            reason_code="checkpoint_authorized",
        )

    collector = ExecutionObservationCollector(
        RUN_ID, TRACE_ID, checkpoint_sink=authorize
    )
    collector.record_attempt(_attempt(attempt=0))
    permit = collector.checkpoint(
        plan_revision=0,
        node=CoreNodeName.PLAN,
        outcome=NodeOutcome.COMPLETED,
        task_ids=("task-1",),
        completed_task_ids=(),
        failed_task_ids=(),
        retry_state=(TaskRetryState(
            task_id="task-1",
            attempts_consumed=1,
            retries_consumed=0,
            retries_remaining=1,
        ),),
        work_items=(ObservedWorkItem(task_id="task-1", task_kind="technical",
                                     worker_id="technical-agent",
                                     owner_node=CoreNodeName.DISPATCH,
                                     maximum_retries=1, execution_state="running",
                                     attempt_ids=("response-1",)),),
        action_fingerprint="e" * 64,
    )

    assert permit.decision is CheckpointDecision.CONTINUE
    assert permit.checkpoint_ordinal == 1
    assert seen == [collector.checkpoints()[0]]


def test_provider_response_cost_is_recomputed_and_tampering_is_rejected() -> None:
    exact = _attempt(
        tokens=TokenUsage(input_tokens=100, cached_input_tokens=25, output_tokens=20),
        pricing_model_id="gpt-5.6-terra",
        pricing_band="short",
        estimated_cost_usd=0.000395,
    )
    assert exact.estimated_cost_usd == 0.000395

    with pytest.raises(ValidationError, match="exact pinned pricing"):
        exact.model_copy(update={"estimated_cost_usd": 0.0001})
