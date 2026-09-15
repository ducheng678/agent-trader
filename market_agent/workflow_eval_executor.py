"""Explicit, host-owned adapter for scoring an actual workflow execution.

Constructing this adapter requires an application instance; importing the module
does not make model calls or start background work.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from time import perf_counter
from typing import Any, Protocol

from market_agent.workflow_contracts import KnowledgeStatus, TerminalMode, WorkflowRequest
from market_agent.workflow_eval_dataset import EvaluationAnswer, RecordedObservation
from market_agent.workflow_evaluation import EvaluationBinding, ExecutedObservation
from market_agent.workflow_observation import WorkflowExecution


class WorkflowExecutionBoundary(Protocol):
    @property
    def evaluation_binding(self) -> EvaluationBinding: ...

    def execute_workflow(self, request: WorkflowRequest) -> WorkflowExecution: ...


RequestBuilder = Callable[[Mapping[str, object], str], WorkflowRequest]


def _default_request(task_input: Mapping[str, object], trace_id: str) -> WorkflowRequest:
    query = task_input.get("query", task_input.get("user_query"))
    if not isinstance(query, str):
        raise ValueError("evaluation task_input must contain a query")
    # Identity comes from the signed case, not from a caller-controlled field
    # inside task_input. Only the task's question enters the default request.
    return WorkflowRequest(
        workflow_id=trace_id,
        trace_id=trace_id,
        user_query=query,
        trigger_reason="evaluation_case",
    )


class WorkflowEvaluationExecutor:
    def __init__(self, application: WorkflowExecutionBoundary, *,
                 request_builder: RequestBuilder = _default_request):
        if not callable(getattr(application, "execute_workflow", None)) or not callable(request_builder):
            raise TypeError("evaluation executor requires a workflow application and request builder")
        try:
            EvaluationBinding.model_validate(application.evaluation_binding)
        except (AttributeError, TypeError, ValueError) as error:
            raise ValueError("workflow application requires a host-owned evaluation binding") from error
        self._application = application
        self._request_builder = request_builder

    @property
    def attested_binding(self) -> EvaluationBinding:
        return EvaluationBinding.model_validate(self._application.evaluation_binding)

    def __call__(self, task_input: dict[str, object], trace_id: str) -> ExecutedObservation:
        request = WorkflowRequest.model_validate(self._request_builder(task_input, trace_id))
        if request.trace_id != trace_id:
            raise ValueError("evaluation request builder changed the signed trace identity")
        started = perf_counter()
        execution = WorkflowExecution.model_validate(self._application.execute_workflow(request))
        result, usage = execution.result, execution.usage
        if result.trace_id != trace_id or usage.unverified_provider_attempt_count:
            raise ValueError("workflow execution identity or priced provider usage is unverified")
        if execution.prompt_release_digest is None:
            raise ValueError("workflow execution did not attest the executed prompt release")
        if result.knowledge_status is KnowledgeStatus.INSUFFICIENT or result.terminal_mode is TerminalMode.UNKNOWN:
            conclusion, action, facts, evidence = "不知道", "no_trade", (), ()
        elif result.terminal_mode is TerminalMode.NO_TRADE:
            conclusion, action, facts, evidence = "no_trade", "no_trade", (), result.evidence_references
        else:
            conclusion, action = "answer", result.final_action.value
            answer = result.informational_answer
            facts = (answer.answer,) if answer is not None else ()
            evidence = answer.source_references if answer is not None else result.evidence_references
        output = EvaluationAnswer(
            conclusion=conclusion, action=action, facts=facts,
            evidence_ids=evidence, trace_id=result.trace_id,
        )
        observation = RecordedObservation(
            output=output.model_dump(mode="json"),
            latency_seconds=max(0.0, perf_counter() - started),
            input_tokens=usage.aggregate.input_tokens,
            output_tokens=usage.aggregate.output_tokens,
            cost_usd=usage.estimated_cost_usd,
        )
        return ExecutedObservation(observation=observation, prompt_release_hash=execution.prompt_release_digest)
