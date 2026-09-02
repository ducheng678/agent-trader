"""Queue adapter for host-owned Harness workflow execution."""

from __future__ import annotations

from typing import Any

from market_agent.workflow_contracts import WorkflowRequest
from market_agent.workflow_harness_application import HarnessWorkflowApplication
from market_agent.workflow_harness_contracts import RunState
from market_agent.workflow_playbook_assembler import unknown_playbook


class HarnessWorkflowService:
    """Validate queued requests and return only durable execution metadata."""

    def __init__(self, application: HarnessWorkflowApplication) -> None:
        if type(application) is not HarnessWorkflowApplication:
            raise TypeError("a host-owned HarnessWorkflowApplication is required")
        self._application = application

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = WorkflowRequest.model_validate(payload)
        execution = self._application.execute(request)
        result = execution.workflow_result
        if execution.view.run_state is not RunState.SUCCEEDED or result is None:
            state = execution.view.run_state.value if execution.view.run_state else "unknown"
            result = unknown_playbook(
                workflow_id=execution.handle.run_id,
                trace_id=execution.handle.trace_id,
                reason=f"Harness did not accept the workflow result: {state}",
                route_history=("harness_safe_terminal", state),
            )
        return {
            "run_id": execution.handle.run_id,
            "trace_id": execution.handle.trace_id,
            "state": execution.view.run_state.value if execution.view.run_state else None,
            "sequence": execution.view.sequence,
            "state_revision": execution.view.state_revision,
            "workflow_error": execution.workflow_error,
            "workflow_result": result.model_dump(mode="json"),
        }
