from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from market_agent.backend.settings import BackendSettings
from market_agent.workflow_production_application import (
    ProductionDependencies,
    ProductionWorkflowApplication,
)

from market_agent.workflow_contracts import (
    Action,
    InformationalAnswer,
    KnowledgeStatus,
    TerminalMode,
    WorkflowRequest,
    WorkflowResult,
    canonical_workflow_request_digest,
    canonical_workflow_result_digest,
)
from market_agent.workflow_execution_backend import (
    CommittedExecutionSnapshot,
    CommittedTransitionReceipt,
)
from market_agent.workflow_harness_contracts import HarnessSessionView, RunState
from market_agent.workflow_observation import (
    AttemptUsage,
    CheckpointDecision,
    CheckpointPermit,
    CoreNodeName,
    NodeOutcome,
    ObservedWorkItem,
    TaskRetryState,
    TokenUsage,
    WorkflowExecution,
)
import market_agent.workflow_production_application as production_module

TRACE_ID = "1" * 32
RUN_ID = "workflow-1"


def _request() -> WorkflowRequest:
    return WorkflowRequest(
        workflow_id=RUN_ID,
        trace_id=TRACE_ID,
        user_query="系统会自动下单吗",
        trigger_reason="manual_once",
    )


def _result() -> WorkflowResult:
    return WorkflowResult(
        workflow_id=RUN_ID,
        trace_id=TRACE_ID,
        terminal_mode=TerminalMode.INFORMATIONAL,
        final_action=Action.NO_TRADE,
        knowledge_status=KnowledgeStatus.KNOWN,
        uncertainty_reason=None,
        informational_answer=InformationalAnswer(
            knowledge_status=KnowledgeStatus.KNOWN,
            uncertainty_reason=None,
            answer="不会。系统没有下单权限。",
        ),
    )


def _terminal_receipt(
    request: WorkflowRequest,
    prompt_digest: str,
    result_digest: str,
) -> CommittedTransitionReceipt:
    request_digest = canonical_workflow_request_digest(request)
    pre = HarnessSessionView(
        sequence=1, state_revision=1, plan_revision=0,
        run_id=request.workflow_id, trace_id=request.trace_id,
        request_digest=request_digest, prompt_release_digest=prompt_digest,
        accepted_result_digest=result_digest,
        run_state=RunState.SUMMARIZING, last_event_hash="b" * 64,
    )
    post = pre.model_copy(update={
        "sequence": 2, "state_revision": 2,
        "run_state": RunState.SUCCEEDED, "last_event_hash": "c" * 64,
    })

    def snapshot(view: HarnessSessionView) -> CommittedExecutionSnapshot:
        return CommittedExecutionSnapshot(
            run_id=request.workflow_id, trace_id=request.trace_id,
            plan_id="plan-1", plan_digest="d" * 64, plan_revision=0,
            sequence=view.sequence, state_revision=view.state_revision,
            view_digest="e" * 64, event_head_hash=view.last_event_hash,
            folded_view=view, trust_key_id="host-rsa-2026-01",
            signature="0" * 512,
        )

    return CommittedTransitionReceipt(
        pre=snapshot(pre), post=snapshot(post), transition_digest="f" * 64,
        trust_key_id="host-rsa-2026-01", signature="0" * 512,
    )


def _application(committed: list[object]) -> ProductionWorkflowApplication:
    dependencies = ProductionDependencies(
        settings=BackendSettings(environment="test"),
        driver_factory=lambda _tenant: object(),
        audit_writer=object(),
        memory_repository=None,
        embedding_client=None,
        completion_hook=lambda _request, result, _proof: committed.append(result),
        prompt_release_manager=SimpleNamespace(
            current=lambda: SimpleNamespace(release_digest="a" * 64)
        ),
    )
    return ProductionWorkflowApplication(lambda: dependencies)


def test_accepted_result_commit_is_explicit_and_identity_bound(monkeypatch) -> None:
    committed: list[object] = []
    application = _application(committed)
    request = _request()
    result = _result()

    from market_agent.workflow_memory_result_writer import AcceptedOutcomeProof
    monkeypatch.setattr(
        "market_agent.workflow_memory_result_writer.verify_committed_transition_receipt",
        lambda _receipt: True,
    )
    proof = AcceptedOutcomeProof.bind(
        request, result, terminal_receipt=_terminal_receipt(
            request, "a" * 64, canonical_workflow_result_digest(result)
        ),
        prompt_release_digest="a" * 64,
        accepted_at=datetime.now(timezone.utc),
    )
    with pytest.raises(RuntimeError, match="ingress"):
        application.commit_accepted_result(request, result, proof)

    monkeypatch.setattr(
        production_module,
        "_lookup_historical_answer",
        lambda **_kwargs: result,
    )
    application.execute_workflow(request)
    application.commit_accepted_result(request, result, proof)
    assert committed == [result]

    with pytest.raises(RuntimeError, match="identity"):
        application.commit_accepted_result(
            request,
            result.model_copy(update={"workflow_id": "workflow-other"}), proof,
        )
    assert committed == [result]


def test_accepted_result_commit_rejects_changed_request_content(monkeypatch) -> None:
    committed: list[object] = []
    application = _application(committed)
    request = _request()
    result = _result()
    monkeypatch.setattr(
        production_module,
        "_lookup_historical_answer",
        lambda **_kwargs: result,
    )
    application.execute_workflow(request)

    from market_agent.workflow_memory_result_writer import AcceptedOutcomeProof
    monkeypatch.setattr(
        "market_agent.workflow_memory_result_writer.verify_committed_transition_receipt",
        lambda _receipt: True,
    )
    proof = AcceptedOutcomeProof.bind(
        request, result, terminal_receipt=_terminal_receipt(
            request, "a" * 64, canonical_workflow_result_digest(result)
        ),
        prompt_release_digest="a" * 64,
        accepted_at=datetime.now(timezone.utc),
    )
    changed = request.model_copy(update={"user_query": "已更改的请求"})

    with pytest.raises(RuntimeError, match="request"):
        application.commit_accepted_result(changed, result, proof)

    assert committed == []


def test_production_execution_binds_driver_usage_and_graph_checkpoints(monkeypatch) -> None:
    streamed = []
    driver_observers = []

    class FakeCoordinator:
        decide = staticmethod(lambda *_args: None)
        technical = staticmethod(lambda *_args: None)
        verifier = staticmethod(lambda *_args: None)

    class FakeWorkflow:
        def invoke(self, request, services):
            observer = services.execution_observer
            item = ObservedWorkItem(
                task_id="task-1", task_kind="technical",
                worker_id="technical-agent", owner_node=CoreNodeName.DISPATCH,
                maximum_retries=0, execution_state="pending",
            )
            observer.checkpoint(
                plan_revision=0, node=CoreNodeName.PLAN,
                outcome=NodeOutcome.COMPLETED, task_ids=("task-1",),
                completed_task_ids=(), failed_task_ids=(),
                retry_state=(TaskRetryState(task_id="task-1", attempts_consumed=0,
                                            retries_consumed=0, retries_remaining=0),),
                work_items=(item,), action_fingerprint="d" * 64,
            )
            observer.record_attempt(AttemptUsage(
                workflow_id=request.workflow_id,
                trace_id=request.trace_id,
                task_id="task-1",
                attempt=0,
                node=CoreNodeName.DISPATCH,
                provider="openai",
                provider_request_id="response-1",
                model_id="gpt-5.6-terra",
                model_tier="terra",
                pricing_version="openai-standard-2026-08-01",
                pricing_model_id="gpt-5.6-terra",
                pricing_band="short",
                tokens=TokenUsage(input_tokens=9, output_tokens=4),
                estimated_cost_usd=0.000066,
                latency_ms=5,
                source="provider_response",
            ))
            for ordinal, node in enumerate((
                CoreNodeName.DISPATCH, CoreNodeName.RECOVER, CoreNodeName.DECIDE,
                CoreNodeName.REFLECT, CoreNodeName.RISK, CoreNodeName.ASSEMBLE,
            ), start=1):
                observer.checkpoint(
                    plan_revision=0, node=node, outcome=NodeOutcome.COMPLETED,
                    task_ids=("task-1",), completed_task_ids=("task-1",),
                    failed_task_ids=(), retry_state=(TaskRetryState(
                        task_id="task-1", attempts_consumed=1,
                        retries_consumed=0, retries_remaining=0,
                    ),), work_items=(item.model_copy(update={
                        "execution_state": "succeeded", "attempt_ids": ("response-1",)
                    }),), action_fingerprint=f"{ordinal}" * 64,
                )
            return _result()

    monkeypatch.setattr(production_module, "AgentCoordinatorServices", lambda **_kwargs: FakeCoordinator())
    monkeypatch.setattr(production_module, "_lookup_historical_answer", lambda **_kwargs: None)
    monkeypatch.setattr(production_module, "_request_context_records", lambda *_args: ())
    monkeypatch.setattr(production_module, "_retrieve_core_memory", lambda **_kwargs: None)

    def driver_factory(_tenant, attempt_observer):
        driver_observers.append(attempt_observer)
        return object()

    dependencies = ProductionDependencies(
        settings=BackendSettings(environment="test"),
        driver_factory=driver_factory,
        audit_writer=SimpleNamespace(healthy=True),
        memory_repository=None,
        embedding_client=None,
        completion_hook=lambda _request, _result, _proof: None,
        prompt_release_manager=SimpleNamespace(
            current=lambda: SimpleNamespace(release_digest="a" * 64)
        ),
        workflow_factory=FakeWorkflow,
        clock=lambda: 1.0,
    )
    application = ProductionWorkflowApplication(lambda: dependencies)

    def checkpoint_sink(checkpoint):
        streamed.append(checkpoint)
        return CheckpointPermit(
            workflow_id=checkpoint.workflow_id, trace_id=checkpoint.trace_id,
            checkpoint_ordinal=checkpoint.ordinal,
            checkpoint_digest=checkpoint.canonical_digest(),
            decision=CheckpointDecision.CONTINUE,
            reason_code="checkpoint_authorized",
        )

    execution = application.execute_workflow(_request(), checkpoint_sink=checkpoint_sink)

    assert isinstance(execution, WorkflowExecution)
    assert len(driver_observers) == 1
    assert execution.usage.aggregate == TokenUsage(input_tokens=9, output_tokens=4)
    assert execution.checkpoints == tuple(streamed)
