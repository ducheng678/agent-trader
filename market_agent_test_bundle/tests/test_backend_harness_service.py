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
from market_agent.workflow_harness_contracts import HarnessSessionView, RunState
from market_agent.workflow_cancellation import WorkflowCancellationRegistry


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

    def create(self, _request):
        return _handle()

    def snapshot(self, _run_id):
        return HarnessSessionView(
            run_id=RUN_ID,
            trace_id=TRACE_ID,
            run_state=self.terminal if self.finished else RunState.RUNNING,
            sequence=10,
            state_revision=5,
        )

    def advance(self, _run_id, **_kwargs):
        self.finished = True
        return object()

    def cancel(self, _run_id, _reason):
        self.terminal = RunState.CANCELLED
        self.finished = True


def _application_for_terminal(terminal: RunState, committed: list[WorkflowResult]):
    application = object.__new__(HarnessWorkflowApplication)
    application._kernel = _TerminalKernel(terminal)
    application._run_workflow = lambda _request: _known_result()
    application._completion_candidate_factory = lambda *_args: {"accepted": True}
    application._accepted_result_committer = lambda _request, result: committed.append(result)
    return application


def test_candidate_side_effects_commit_only_after_harness_success() -> None:
    committed: list[WorkflowResult] = []

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
