"""Queue adapter for host-owned Harness workflow execution."""

from __future__ import annotations

from typing import Any

from market_agent.workflow_contracts import WorkflowRequest
from market_agent.workflow_harness_application import HarnessWorkflowApplication


class HarnessWorkflowService:
    """Validate queued requests and return only durable execution metadata."""

    def __init__(self, application: HarnessWorkflowApplication) -> None:
        if type(application) is not HarnessWorkflowApplication:
            raise TypeError("a host-owned HarnessWorkflowApplication is required")
        self._application = application

    def execute(self, payload: dict[str, Any]) -> dict[str, Any]:
        request = WorkflowRequest.model_validate(payload)
        execution = self._application.execute(request)
        return {
            "run_id": execution.handle.run_id,
            "trace_id": execution.handle.trace_id,
            "state": execution.view.run_state.value if execution.view.run_state else None,
            "sequence": execution.view.sequence,
            "state_revision": execution.view.state_revision,
            "workflow_error": execution.workflow_error,
        }
