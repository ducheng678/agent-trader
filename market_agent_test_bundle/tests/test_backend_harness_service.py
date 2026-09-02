from __future__ import annotations

import json

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
