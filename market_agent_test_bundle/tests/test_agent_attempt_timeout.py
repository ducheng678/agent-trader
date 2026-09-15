from __future__ import annotations

from hashlib import sha256
import json

from pydantic import BaseModel, ConfigDict, Field

from market_agent.workflow_agent_contracts import AgentInvocation, AgentUsage, ModelTier
from market_agent.workflow_agent_driver import AgentDriver, ModelResponse, OutputSchema
from market_agent.workflow_circuit_breaker import CircuitBreaker
from market_agent.workflow_fallback import FallbackPolicy
from market_agent.workflow_prompt_release import PromptRelease, PromptReleaseRegistry, canonical_json
from market_agent.workflow_retry_policy import RetryPolicy


TRACE_ID = "1" * 32


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str = Field(min_length=1)


class Clock:
    def __init__(self, now: float = 1.0):
        self.time = now
        self.waits: list[float] = []

    def now(self) -> float:
        return self.time

    def sleep(self, seconds: float) -> None:
        self.waits.append(seconds)
        self.time += seconds


class RecordingClient:
    def __init__(self, *outcomes: object):
        self.outcomes = list(outcomes)
        self.requests = []

    def invoke(self, request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class Observer:
    def record(self, event) -> None:
        pass


def release() -> PromptRelease:
    values = dict(
        schema_version="v1", release_id="release-v1", stable_system_prefix="Released prompt.",
        supported_task_kinds=("extract",), supported_model_tiers=(ModelTier.LUNA,),
        temperature_profile=((ModelTier.LUNA, 0.0),),
    )
    digest = sha256(json.dumps(values, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode()).hexdigest()
    return PromptRelease(digest=digest, **values)


def output_schema() -> OutputSchema:
    return OutputSchema(schema_id="answer-v1", model=Answer)


def invocation(**overrides) -> AgentInvocation:
    schema = output_schema()
    values = dict(
        trace_id=TRACE_ID, run_id="run-1", task_id="task-1", task_kind="extract",
        prompt_release_id="release-v1", prompt_release_digest=release().digest,
        allowed_model_tier=ModelTier.LUNA, deadline_epoch=100.0,
        max_attempts=2, cost_limit_usd=1.0, output_schema_id=schema.schema_id,
        output_schema_digest=schema.digest, user_payload={"query": "q"},
    )
    values.update(overrides)
    return AgentInvocation(**values)


def response() -> ModelResponse:
    return ModelResponse(
        content='{"answer":"known"}',
        usage=AgentUsage(input_tokens=1, output_tokens=1, cost_usd=0.01, model_tier=ModelTier.LUNA),
    )


def driver(client: RecordingClient, clock: Clock) -> AgentDriver:
    return AgentDriver(
        model_client=client, audit_observer=Observer(), clock=clock,
        random=lambda low, high: high, prompt_releases=PromptReleaseRegistry(releases=(release(),)),
        output_schemas=(output_schema(),), retry_policy=RetryPolicy(max_attempts=2, base_delay=0.25),
        circuit_breaker=CircuitBreaker(failure_threshold=3, cooldown=10.0),
        fallback_policy=FallbackPolicy((ModelTier.LUNA,)), model_costs={ModelTier.LUNA: 0.1},
    )


def test_first_provider_request_uses_the_per_attempt_deadline():
    clock, client = Clock(), RecordingClient(response())

    assert driver(client, clock).execute(invocation(attempt_timeout_seconds=3.0)).failure is None

    assert client.requests[0].deadline_epoch == 4.0


def test_retry_gets_a_fresh_per_attempt_deadline():
    clock, client = Clock(), RecordingClient(TimeoutError(), response())

    assert driver(client, clock).execute(invocation(attempt_timeout_seconds=3.0)).failure is None

    assert [request.deadline_epoch for request in client.requests] == [4.0, 4.5]


def test_workflow_deadline_takes_precedence_over_attempt_window():
    clock, client = Clock(), RecordingClient(response())

    assert driver(client, clock).execute(invocation(deadline_epoch=2.0, attempt_timeout_seconds=3.0)).failure is None

    assert client.requests[0].deadline_epoch == 2.0


def test_provider_is_not_called_after_the_workflow_deadline():
    clock, client = Clock(now=2.0), RecordingClient(response())

    result = driver(client, clock).execute(invocation(deadline_epoch=2.0, attempt_timeout_seconds=3.0))

    assert result.output == {"answer": "不知道"}
    assert client.requests == []


def test_specialist_build_invocation_retains_task_attempt_timeout_and_ordinal():
    from market_agent.workflow_agents.common import build_invocation
    from market_agent.workflow_contracts import (
        AgentTask, ContextSummary, ModelTier as TaskModelTier, SummaryCompleteness,
        TaskDifficulty, TaskType,
    )
    from market_agent.workflow_agents.common import profile_for

    profile = profile_for(TaskType.TECHNICAL)
    task = AgentTask(
        task_id="task-technical", workflow_id="workflow-1", trace_id=TRACE_ID,
        task_type=TaskType.TECHNICAL, objective="Check supplied evidence.",
        context_summary_id="summary-technical", allowed_data=profile.allowed_data, allowed_tools=(),
        expected_output=profile.profile_id, acceptance_criteria=("Return cited evidence.",),
        difficulty=TaskDifficulty.NORMAL, model_tier=TaskModelTier.TERRA, prompt_version=profile.profile_id,
        attempt_timeout_seconds=37, maximum_retries=1, reserved_cost=0.05, remaining_workflow_cost=0.10,
        analysis_steps=profile.analysis_steps, escalation_rule="return_to_coordinator",
        conflict_return_rule="return_typed_conflict",
    )
    context = ContextSummary(
        summary_id=task.context_summary_id, task_id=task.task_id, workflow_id=task.workflow_id,
        trace_id=task.trace_id, user_objective=task.objective, immutable_constraints=("No execution.",),
        source_references=("source-1",), token_estimate=1, completeness=SummaryCompleteness.COMPLETE,
        summary_version="test-v1", summarizer_model=TaskModelTier.LUNA, source_record_hash="a" * 64,
    )

    built = build_invocation(task, context, deadline_epoch=100.0, attempt=1)

    assert built.attempt_timeout_seconds == 37.0
    assert built.attempt == 1
