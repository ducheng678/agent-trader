"""Host-owned bridge from a committed Harness run to the coordinated workflow.

The bridge deliberately has no model-output parser.  Harness transitions are
advanced by fixed host policy; workflow output is only returned as data for a
later host validator/renderer and never chooses an edge.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from market_agent.workflow_contracts import WorkflowRequest, WorkflowResult
from market_agent.workflow_execution_backend import ExecutionRegistrationError
from market_agent.workflow_harness import HarnessDecision, HarnessKernel, RunHandle
from market_agent.workflow_harness_contracts import HarnessSessionView, RunState


@dataclass(frozen=True, slots=True)
class HarnessWorkflowExecution:
    """One immutable record of a host-orchestrated workflow attempt."""

    handle: RunHandle
    view: HarnessSessionView
    decisions: tuple[HarnessDecision, ...]
    workflow_result: WorkflowResult | None
    workflow_error: str | None

    @property
    def safe(self) -> bool:
        return self.view.run_state in {RunState.SUCCEEDED, RunState.DEGRADED}


WorkflowRunner = Callable[[WorkflowRequest], WorkflowResult]
AcceptedResultCommitter = Callable[[WorkflowRequest, WorkflowResult], object]
CancellationSignalFactory = Callable[[str], object]
CompletionCandidateFactory = Callable[
    [WorkflowRequest, WorkflowResult, HarnessSessionView], dict[str, object]
]


class HarnessWorkflowApplication:
    """Invoke a supplied workflow only after Harness reaches RUNNING.

    Until a host confidence/evidence adapter supplies a signed candidate, the
    kernel's existing fail-closed confidence policy deliberately settles every
    completed callback through DEGRADED/no-trade.  This makes the bridge safe
    to deploy before richer evidence adapters are enabled.
    """

    def __init__(
        self,
        *,
        kernel: HarnessKernel,
        run_workflow: WorkflowRunner,
        completion_candidate_factory: CompletionCandidateFactory | None = None,
        accepted_result_committer: AcceptedResultCommitter | None = None,
        cancellation_signal_factory: CancellationSignalFactory | None = None,
    ) -> None:
        if type(kernel) is not HarnessKernel or not callable(run_workflow):
            raise TypeError("Harness application requires a kernel and host workflow runner")
        if completion_candidate_factory is not None and not callable(completion_candidate_factory):
            raise TypeError("completion candidate factory must be host-owned and callable")
        if accepted_result_committer is not None and not callable(accepted_result_committer):
            raise TypeError("accepted result committer must be host-owned and callable")
        if cancellation_signal_factory is not None and not callable(cancellation_signal_factory):
            raise TypeError("cancellation signal factory must be host-owned and callable")
        self._kernel = kernel
        self._run_workflow = run_workflow
        self._completion_candidate_factory = completion_candidate_factory
        self._accepted_result_committer = accepted_result_committer
        self._cancellation_signal_factory = cancellation_signal_factory

    @property
    def kernel(self) -> HarnessKernel:
        """Expose the immutable host authority for composition validation."""
        return self._kernel

    def execute(self, request: WorkflowRequest) -> HarnessWorkflowExecution:
        request = WorkflowRequest.model_validate(request)
        signal_factory = getattr(self, "_cancellation_signal_factory", None)
        signal = (
            signal_factory(request.workflow_id)
            if signal_factory is not None
            else None
        )
        try:
            handle = self._kernel.create(request)
        except ExecutionRegistrationError as error:
            if str(error) != "run already exists":
                raise
            existing = self._kernel.snapshot(request.workflow_id)
            # A cancelled/terminal run must not re-enter the execution backend
            # merely because a queued delivery arrived after cancellation.
            handle = (
                self._kernel.handle(request.workflow_id)
                if existing.run_state in {
                    RunState.SUCCEEDED,
                    RunState.DEGRADED,
                    RunState.FAILED,
                    RunState.CANCELLED,
                }
                else self._kernel.resume(request.workflow_id)
            )
        decisions: list[HarnessDecision] = []
        view = self._kernel.snapshot(handle.run_id)
        for _ in range(4):
            if view.run_state is RunState.RUNNING:
                break
            decision = self._kernel.advance(handle.run_id, expected_state_revision=view.state_revision)
            decisions.append(decision)
            view = self._kernel.snapshot(handle.run_id)

        result: WorkflowResult | None = None
        error: str | None = None
        completion_candidate: dict[str, object] | None = None
        if view.run_state is RunState.RUNNING:
            try:
                if signal is not None and signal.is_cancelled():
                    self._kernel.cancel(handle.run_id, "cooperative_cancellation")
                    raise RuntimeError("workflow was cancelled before execution")
                candidate = self._run_workflow(request)
                candidate = WorkflowResult.model_validate(candidate)
                if (candidate.workflow_id, candidate.trace_id) != (request.workflow_id, request.trace_id):
                    raise ValueError("workflow result identity does not match Harness run")
                result = candidate
                if self._completion_candidate_factory is not None:
                    completion_candidate = self._completion_candidate_factory(
                        request, result, view
                    )
                    if type(completion_candidate) is not dict:
                        raise TypeError("host completion candidate must be an exact dictionary")
            except Exception as exc:  # Host error is recorded by deterministic degradation below.
                error = type(exc).__name__
            # No candidate is derived from model output.  The kernel therefore
            # applies its signed confidence policy and degrades safely if a
            # production evidence adapter has not authorized completion.
            for _ in range(3):
                view = self._kernel.snapshot(handle.run_id)
                if signal is not None and signal.is_cancelled() and view.run_state is not RunState.CANCELLED:
                    self._kernel.cancel(handle.run_id, "cooperative_cancellation")
                    view = self._kernel.snapshot(handle.run_id)
                if view.run_state in {RunState.SUCCEEDED, RunState.DEGRADED, RunState.FAILED, RunState.CANCELLED}:
                    break
                decision = self._kernel.advance(
                    handle.run_id,
                    candidate=completion_candidate,
                    expected_state_revision=view.state_revision,
                )
                decisions.append(decision)
                completion_candidate = None
        view = self._kernel.snapshot(handle.run_id)
        if (
            view.run_state is RunState.SUCCEEDED
            and result is not None
            and self._accepted_result_committer is not None
            and not (signal is not None and signal.is_cancelled())
        ):
            try:
                self._accepted_result_committer(request, result)
            except Exception as exc:
                # Fail closed: a result whose required durable commit failed
                # is not eligible to leave the queue adapter.
                error = type(exc).__name__
                result = None
        return HarnessWorkflowExecution(
            handle=handle,
            view=view,
            decisions=tuple(decisions),
            workflow_result=result,
            workflow_error=error,
        )
