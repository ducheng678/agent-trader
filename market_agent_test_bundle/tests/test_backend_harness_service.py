from __future__ import annotations

import json

from market_agent.backend.api import workflow_status
from market_agent.backend.database import JobRecord
from market_agent.backend.harness_service import HarnessWorkflowService
from market_agent.workflow_contracts import (
    Action,
    InformationalAnswer,
    KnowledgeStatus,
    TerminalMode,
    WorkflowResult,
)
from market_agent.workflow_harness import RunHandle
from market_agent.workflow_harness_application import (
    HarnessWorkflowApplication,
    HarnessWorkflowExecution,
)
from market_agent.workflow_execution_backend import (
    CommittedExecutionSnapshot,
    CommittedTransitionReceipt,
)
from market_agent.workflow_contracts import (
    WorkflowRequest,
    canonical_workflow_request_digest,
    canonical_workflow_result_digest,
)
from market_agent.workflow_harness_contracts import HarnessSessionView, RunState
from market_agent.workflow_cancellation import WorkflowCancellationRegistry
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
    WorkflowExecution,
    WorkflowUsage,
)


TRACE_ID = "1" * 32
RUN_ID = "workflow-1"


def _payload() -> dict[str, object]:
    return {
        "workflow_id": RUN_ID,
        "trace_id": TRACE_ID,
        "user_query": "系统会自动下单吗",
        "trigger_reason": "manual_once",
    }


def _handle() -> RunHandle:
    return RunHandle(
        run_id=RUN_ID,
        trace_id=TRACE_ID,
        plan_id="plan-1",
        plan_revision=0,
        sequence=1,
        state_revision=1,
        run_state=RunState.CREATED,
        backend_synchronized=True,
    )


def _known_result() -> WorkflowResult:
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


def _service(execution: HarnessWorkflowExecution) -> HarnessWorkflowService:
    application = object.__new__(HarnessWorkflowApplication)
    application.execute = lambda _request: execution
    return HarnessWorkflowService(application)


def test_successful_harness_result_survives_the_queue_adapter() -> None:
    result = _known_result()
    execution = HarnessWorkflowExecution(
        handle=_handle(),
        view=HarnessSessionView(
            run_id=RUN_ID,
            trace_id=TRACE_ID,
            run_state=RunState.SUCCEEDED,
            sequence=10,
            state_revision=5,
        ),
        decisions=(),
        workflow_result=result,
        workflow_error=None,
    )

    response = _service(execution).execute(_payload())

    assert response["workflow_result"] == result.model_dump(mode="json")


def test_degraded_harness_result_never_exposes_candidate_model_output() -> None:
    execution = HarnessWorkflowExecution(
        handle=_handle(),
        view=HarnessSessionView(
            run_id=RUN_ID,
            trace_id=TRACE_ID,
            run_state=RunState.DEGRADED,
            sequence=10,
            state_revision=5,
        ),
        decisions=(),
        workflow_result=_known_result(),
        workflow_error=None,
    )

    response = _service(execution).execute(_payload())

    safe = WorkflowResult.model_validate_json(json.dumps(response["workflow_result"]))
    assert safe.terminal_mode is TerminalMode.UNKNOWN
    assert safe.final_action is Action.NO_TRADE
    assert safe.knowledge_status is KnowledgeStatus.INSUFFICIENT
    assert safe.informational_answer is None


def _job(result: dict[str, object] | None, *, status: str = "succeeded") -> JobRecord:
    return JobRecord(
        job_id="job-1",
        task_name="execute_harness_workflow",
        status=status,
        payload={},
        idempotency_key=RUN_ID,
        payload_fingerprint="fingerprint",
        result=result,
        error=None,
        attempt_count=1,
        max_attempts=3,
        request_id=TRACE_ID,
        created_at="2026-09-02T00:00:00+00:00",
        updated_at="2026-09-02T00:00:01+00:00",
    )


def test_workflow_status_exposes_only_a_matching_persisted_result() -> None:
    result = _known_result().model_dump(mode="json")
    view = HarnessSessionView(
        run_id=RUN_ID,
        trace_id=TRACE_ID,
        run_state=RunState.SUCCEEDED,
        sequence=10,
        state_revision=5,
    )

    response = workflow_status(view, _job({"workflow_result": result}))

    assert response.result == _known_result()
    assert response.dispatcher_status == "succeeded"


def test_workflow_status_fails_closed_for_unfinished_or_cross_wired_jobs() -> None:
    result = _known_result().model_dump(mode="json")
    view = HarnessSessionView(
        run_id=RUN_ID,
        trace_id=TRACE_ID,
        run_state=RunState.SUCCEEDED,
        sequence=10,
        state_revision=5,
    )

    assert workflow_status(view, _job({"workflow_result": result}, status="running")).result is None
    cross_wired = _known_result().model_copy(update={"workflow_id": "workflow-other"}).model_dump(mode="json")
    assert workflow_status(view, _job({"workflow_result": cross_wired})).result is None
    cancelled = view.model_copy(update={"run_state": RunState.CANCELLED})
    assert workflow_status(cancelled, _job({"workflow_result": result})).result is None


def test_cancellation_registry_is_run_scoped_and_sticky() -> None:
    registry = WorkflowCancellationRegistry()
    first = registry.signal(RUN_ID)
    other = registry.signal("workflow-other")

    registry.cancel(RUN_ID)

    assert first.is_cancelled()
    assert registry.signal(RUN_ID).is_cancelled()
    assert not other.is_cancelled()


class _TerminalKernel:
    def __init__(self, terminal: RunState) -> None:
        self.terminal = terminal
        self.finished = False
        self.request = None

    def create(self, request):
        self.request = WorkflowRequest.model_validate(request)
        return _handle()

    def snapshot(self, _run_id):
        return HarnessSessionView(
            run_id=RUN_ID,
            trace_id=TRACE_ID,
            run_state=self.terminal if self.finished else RunState.RUNNING,
            sequence=10,
            state_revision=5,
            last_event_hash="b" * 64 if self.finished else None,
        )

    def advance(self, _run_id, **_kwargs):
        self.finished = True
        return object()

    def cancel(self, _run_id, _reason):
        self.terminal = RunState.CANCELLED
        self.finished = True

    def terminal_receipt(self, _run_id):
        request = self.request
        result = _known_result()
        request_digest = canonical_workflow_request_digest(request)
        result_digest = canonical_workflow_result_digest(result)
        pre_view = HarnessSessionView(
            run_id=RUN_ID,
            trace_id=TRACE_ID,
            run_state=RunState.RUNNING,
            sequence=9,
            state_revision=4,
            request_digest=request_digest,
            prompt_release_digest="a" * 64,
            accepted_result_digest=result_digest,
            last_event_hash="a" * 64,
        )
        post_view = pre_view.model_copy(update={
            "sequence": 10,
            "state_revision": 5,
            "run_state": RunState.SUCCEEDED,
            "last_event_hash": "b" * 64,
        })

        def snapshot(view):
            return CommittedExecutionSnapshot(
                run_id=RUN_ID,
                trace_id=TRACE_ID,
                plan_id="plan-1",
                plan_digest="c" * 64,
                plan_revision=0,
                sequence=view.sequence,
                state_revision=view.state_revision,
                view_digest="d" * 64,
                event_head_hash=view.last_event_hash,
                folded_view=view,
                trust_key_id="host-rsa-2026-01",
                signature="0" * 512,
            )

        return CommittedTransitionReceipt(
            pre=snapshot(pre_view),
            post=snapshot(post_view),
            transition_digest="e" * 64,
            trust_key_id="host-rsa-2026-01",
            signature="0" * 512,
        )


def _application_for_terminal(terminal: RunState, committed: list[WorkflowResult]):
    application = object.__new__(HarnessWorkflowApplication)
    application._kernel = _TerminalKernel(terminal)
    application._run_workflow = lambda _request: _known_result()
    application._run_observed_workflow = lambda _request, _sink: WorkflowExecution(
        result=_known_result(),
        usage=WorkflowUsage.from_attempts(RUN_ID, TRACE_ID, ()),
        checkpoints=(),
        completion_kind="historical_cache",
        prompt_release_digest="a" * 64,
    )
    application._completion_candidate_factory = lambda *_args: {"accepted": True}
    application._accepted_result_committer = lambda _request, result, _proof: committed.append(result)
    return application


def test_candidate_side_effects_commit_only_after_harness_success(monkeypatch) -> None:
    committed: list[WorkflowResult] = []
    monkeypatch.setattr(
        "market_agent.workflow_memory_result_writer.verify_committed_transition_receipt",
        lambda _receipt: True,
    )

    succeeded = _application_for_terminal(RunState.SUCCEEDED, committed).execute(_payload())
    assert succeeded.view.run_state is RunState.SUCCEEDED
    assert committed == [_known_result()]

    committed.clear()
    degraded = _application_for_terminal(RunState.DEGRADED, committed).execute(_payload())
    assert degraded.view.run_state is RunState.DEGRADED
    assert committed == []


def test_pre_cancelled_harness_run_never_calls_runner_or_committer() -> None:
    registry = WorkflowCancellationRegistry()
    registry.cancel(RUN_ID)
    calls: list[str] = []
    application = object.__new__(HarnessWorkflowApplication)
    application._kernel = _TerminalKernel(RunState.SUCCEEDED)
    application._run_workflow = lambda _request: calls.append("run")
    application._completion_candidate_factory = lambda *_args: {"accepted": True}
    application._accepted_result_committer = lambda *_args: calls.append("commit")
    application._cancellation_signal_factory = registry.signal

    execution = application.execute(_payload())

    assert execution.view.run_state is RunState.CANCELLED
    assert execution.workflow_result is None
    assert calls == []


def test_observed_runner_streams_checkpoints_and_terminal_usage_to_harness() -> None:
    class ObservedKernel(_TerminalKernel):
        def __init__(self) -> None:
            super().__init__(RunState.SUCCEEDED)
            self.checkpoints = []
            self.terminal_usage = None

        def record_checkpoint(self, _run_id, checkpoint):
            self.checkpoints.append(checkpoint)
            return CheckpointPermit(
                workflow_id=checkpoint.workflow_id,
                trace_id=checkpoint.trace_id,
                checkpoint_ordinal=checkpoint.ordinal,
                checkpoint_digest=checkpoint.canonical_digest(),
                decision=CheckpointDecision.CONTINUE,
                reason_code="checkpoint_authorized",
            )

        def advance(self, _run_id, **kwargs):
            if kwargs.get("workflow_usage") is not None:
                self.terminal_usage = kwargs["workflow_usage"]
            return super().advance(_run_id, **kwargs)

    kernel = ObservedKernel()

    def observed_runner(request, checkpoint_sink):
        observations = ExecutionObservationCollector(
            request.workflow_id,
            request.trace_id,
            checkpoint_sink=checkpoint_sink,
        )
        pending_item = ObservedWorkItem(
            task_id="task-1", task_kind="technical", worker_id="technical-agent",
            owner_node=CoreNodeName.DISPATCH, maximum_retries=0,
            execution_state="pending",
        )
        observations.checkpoint(
            plan_revision=0, node=CoreNodeName.PLAN,
            outcome=NodeOutcome.COMPLETED, task_ids=("task-1",),
            completed_task_ids=(), failed_task_ids=(),
            retry_state=(TaskRetryState(task_id="task-1", attempts_consumed=0,
                                        retries_consumed=0, retries_remaining=0),),
            work_items=(pending_item,), action_fingerprint="c" * 64,
        )
        observations.record_attempt(AttemptUsage(
            workflow_id=request.workflow_id,
            trace_id=request.trace_id,
            task_id="task-1",
            attempt=0,
            node=CoreNodeName.DISPATCH,
            provider="openai",
            provider_request_id="response-1",
            model_id="gpt-5.6-luna",
            model_tier="luna",
            pricing_version="openai-standard-2026-08-01",
            pricing_model_id="gpt-5.6-luna",
            pricing_band="short",
            tokens=TokenUsage(input_tokens=8, output_tokens=3),
            estimated_cost_usd=0.0000052,
            latency_ms=4,
            source="provider_response",
        ))
        observations.checkpoint(
            plan_revision=0,
            node=CoreNodeName.DISPATCH,
            outcome=NodeOutcome.COMPLETED,
            task_ids=("task-1",),
            completed_task_ids=("task-1",),
            failed_task_ids=(),
            retry_state=(TaskRetryState(task_id="task-1", attempts_consumed=1,
                                        retries_consumed=0, retries_remaining=0),),
            work_items=(ObservedWorkItem(task_id="task-1", task_kind="technical",
                                         worker_id="technical-agent",
                                         owner_node=CoreNodeName.DISPATCH,
                                         maximum_retries=0, execution_state="succeeded",
                                         attempt_ids=("response-1",)),),
            action_fingerprint="d" * 64,
        )
        return WorkflowExecution(
            result=_known_result(),
            usage=observations.usage(),
            checkpoints=observations.checkpoints(),
        )

    application = object.__new__(HarnessWorkflowApplication)
    application._kernel = kernel
    application._run_workflow = lambda _request: (_ for _ in ()).throw(
        AssertionError("legacy runner must not execute")
    )
    application._run_observed_workflow = observed_runner
    application._completion_candidate_factory = lambda *_args: {"accepted": True}
    application._accepted_result_committer = None
    application._cancellation_signal_factory = None

    execution = application.execute(_payload())

    assert execution.workflow_error is None
    assert len(kernel.checkpoints) == 2
    assert kernel.terminal_usage == execution.workflow_usage
    assert execution.workflow_usage.aggregate == TokenUsage(input_tokens=8, output_tokens=3)


def test_observed_runner_failure_never_falls_back_to_legacy_settlement() -> None:
    class ObservedKernel(_TerminalKernel):
        def __init__(self) -> None:
            super().__init__(RunState.DEGRADED)
            self.advance_calls = []

        def advance(self, _run_id, **kwargs):
            self.advance_calls.append(kwargs)
            return super().advance(_run_id, **kwargs)

    kernel = ObservedKernel()
    application = object.__new__(HarnessWorkflowApplication)
    application._kernel = kernel
    application._run_workflow = lambda _request: (_ for _ in ()).throw(
        AssertionError("legacy runner must not execute")
    )
    application._run_observed_workflow = lambda *_args: (_ for _ in ()).throw(
        RuntimeError("observed failure")
    )
    application._completion_candidate_factory = None
    application._accepted_result_committer = None
    application._cancellation_signal_factory = None

    execution = application.execute(_payload())

    assert execution.workflow_error == "RuntimeError"
    assert any(call.get("observed_execution") is True for call in kernel.advance_calls)


def test_observed_restart_uses_durable_checkpoint_without_replaying_provider() -> None:
    observations = ExecutionObservationCollector(RUN_ID, TRACE_ID)
    observations.checkpoint(
        plan_revision=0, node=CoreNodeName.PLAN,
        outcome=NodeOutcome.COMPLETED, task_ids=("task-1",),
        completed_task_ids=(), failed_task_ids=(),
        retry_state=(TaskRetryState(task_id="task-1", attempts_consumed=0,
                                    retries_consumed=0, retries_remaining=1),),
        work_items=(ObservedWorkItem(
            task_id="task-1", task_kind="technical", worker_id="technical-agent",
            owner_node=CoreNodeName.DISPATCH, maximum_retries=1,
            execution_state="pending",
        ),), action_fingerprint="f" * 64,
    )

    class RestartKernel(_TerminalKernel):
        def __init__(self) -> None:
            super().__init__(RunState.DEGRADED)
            self.advance_calls = []

        def checkpoint_history(self, _run_id):
            return observations.checkpoints()

        def advance(self, _run_id, **kwargs):
            self.advance_calls.append(kwargs)
            return super().advance(_run_id, **kwargs)

    calls = []
    kernel = RestartKernel()
    application = object.__new__(HarnessWorkflowApplication)
    application._kernel = kernel
    application._run_workflow = lambda _request: calls.append("legacy")
    application._run_observed_workflow = lambda *_args: calls.append("provider")
    application._completion_candidate_factory = None
    application._accepted_result_committer = None
    application._cancellation_signal_factory = None

    execution = application.execute(_payload())

    assert calls == []
    assert execution.workflow_error == "ObservedExecutionInterrupted"
    assert execution.workflow_usage == observations.usage()
    assert kernel.advance_calls[0]["observed_execution"] is True
