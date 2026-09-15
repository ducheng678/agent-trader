"""Host-owned bridge from a committed Harness run to the coordinated workflow.

The bridge deliberately has no model-output parser.  Harness transitions are
advanced by fixed host policy; workflow output is only returned as data for a
later host validator/renderer and never chooses an edge.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timezone

from market_agent.backend.errors import ExecutionLeaseLostError, RetryableTaskError
from market_agent.backend.execution_fence import current_execution_fence
from market_agent.workflow_contracts import (
    WorkflowRequest,
    WorkflowResult,
    canonical_workflow_result_digest,
)
from market_agent.workflow_execution_backend import ExecutionRegistrationError
from market_agent.workflow_harness import HarnessDecision, HarnessKernel, RunHandle
from market_agent.workflow_harness_contracts import HarnessSessionView, RunState
from market_agent.workflow_memory_result_writer import AcceptedOutcomeProof
from market_agent.workflow_observation import NodeCheckpoint, WorkflowExecution, WorkflowUsage
from market_agent.workflow_result_store import WorkflowResultStore


@dataclass(frozen=True, slots=True)
class HarnessWorkflowExecution:
    """One immutable record of a host-orchestrated workflow attempt."""

    handle: RunHandle
    view: HarnessSessionView
    decisions: tuple[HarnessDecision, ...]
    workflow_result: WorkflowResult | None
    workflow_error: str | None
    workflow_usage: WorkflowUsage | None = None
    checkpoints: tuple[NodeCheckpoint, ...] = ()

    @property
    def safe(self) -> bool:
        return self.view.run_state in {RunState.SUCCEEDED, RunState.DEGRADED}


WorkflowRunner = Callable[[WorkflowRequest], WorkflowResult]
ObservedWorkflowRunner = Callable[
    [WorkflowRequest, Callable[[NodeCheckpoint], object]], WorkflowExecution
]
AcceptedResultCommitter = Callable[
    [WorkflowRequest, WorkflowResult, AcceptedOutcomeProof], object
]
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
        run_observed_workflow: ObservedWorkflowRunner | None = None,
        completion_candidate_factory: CompletionCandidateFactory | None = None,
        accepted_result_committer: AcceptedResultCommitter | None = None,
        cancellation_signal_factory: CancellationSignalFactory | None = None,
        result_store: WorkflowResultStore | None = None,
    ) -> None:
        if type(kernel) is not HarnessKernel or not callable(run_workflow):
            raise TypeError("Harness application requires a kernel and host workflow runner")
        if run_observed_workflow is not None and not callable(run_observed_workflow):
            raise TypeError("observed workflow runner must be host-owned and callable")
        if completion_candidate_factory is not None and not callable(completion_candidate_factory):
            raise TypeError("completion candidate factory must be host-owned and callable")
        if accepted_result_committer is not None and not callable(accepted_result_committer):
            raise TypeError("accepted result committer must be host-owned and callable")
        if cancellation_signal_factory is not None and not callable(cancellation_signal_factory):
            raise TypeError("cancellation signal factory must be host-owned and callable")
        self._kernel = kernel
        self._run_workflow = run_workflow
        self._run_observed_workflow = run_observed_workflow
        self._completion_candidate_factory = completion_candidate_factory
        self._accepted_result_committer = accepted_result_committer
        self._cancellation_signal_factory = cancellation_signal_factory
        self._result_store = result_store

    @property
    def kernel(self) -> HarnessKernel:
        """Expose the immutable host authority for composition validation."""
        return self._kernel

    @property
    def result_store(self) -> WorkflowResultStore | None:
        """The journal bound by the host composition, for readiness validation."""
        return getattr(self, "_result_store", None)

    def execute(self, request: WorkflowRequest) -> HarnessWorkflowExecution:
        execution_fence = current_execution_fence()

        def check_execution_fence() -> None:
            if execution_fence is not None:
                execution_fence.raise_if_lost()

        check_execution_fence()
        request = WorkflowRequest.model_validate(request)
        result_store = getattr(self, "_result_store", None)
        prepared = None
        if result_store is not None:
            try:
                prepared = result_store.load(request)
            except ValueError:
                raise
            except Exception as exc:
                raise RetryableTaskError("durable workflow result is unavailable") from exc
        signal_factory = getattr(self, "_cancellation_signal_factory", None)
        signal = (
            signal_factory(request.workflow_id)
            if signal_factory is not None
            else None
        )
        check_execution_fence()
        try:
            handle = self._kernel.create(request)
        except ExecutionRegistrationError as error:
            if str(error) != "run already exists":
                raise
            existing = self._kernel.snapshot(request.workflow_id)
            check_execution_fence()
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
            if view.run_state in {RunState.RUNNING, RunState.SUCCEEDED, RunState.DEGRADED,
                                  RunState.FAILED, RunState.CANCELLED}:
                break
            check_execution_fence()
            decision = self._kernel.advance(handle.run_id, expected_state_revision=view.state_revision)
            decisions.append(decision)
            view = self._kernel.snapshot(handle.run_id)

        result: WorkflowResult | None = None
        workflow_usage: WorkflowUsage | None = None
        prompt_release_digest: str | None = None
        checkpoints: tuple[NodeCheckpoint, ...] = ()
        error: str | None = None
        completion_candidate: dict[str, object] | None = None
        observed_runner = getattr(self, "_run_observed_workflow", None)
        if prepared is not None:
            result = prepared.execution.result
            workflow_usage = prepared.execution.usage
            checkpoints = prepared.execution.checkpoints
            prompt_release_digest = prepared.execution.prompt_release_digest
        if view.run_state is RunState.RUNNING:
            try:
                check_execution_fence()
                if signal is not None and signal.is_cancelled():
                    check_execution_fence()
                    self._kernel.cancel(handle.run_id, "cooperative_cancellation")
                    raise RuntimeError("workflow was cancelled before execution")
                if prepared is not None:
                    candidate = prepared.execution.result
                elif observed_runner is not None:
                    history_reader = getattr(self._kernel, "checkpoint_history", None)
                    durable = tuple(history_reader(handle.run_id)) if callable(history_reader) else ()
                    if durable:
                        checkpoints = durable
                        workflow_usage = durable[-1].usage
                        error = "ObservedExecutionInterrupted"
                        candidate = None
                    else:
                        def record_checkpoint(checkpoint: NodeCheckpoint) -> object:
                            check_execution_fence()
                            permit = self._kernel.record_checkpoint(handle.run_id, checkpoint)
                            check_execution_fence()
                            return permit

                        check_execution_fence()
                        observed = WorkflowExecution.model_validate(
                            observed_runner(request, record_checkpoint)
                        )
                        check_execution_fence()
                        candidate = observed.result
                        workflow_usage = observed.usage
                        checkpoints = observed.checkpoints
                        prompt_release_digest = observed.prompt_release_digest
                        if result_store is not None:
                            try:
                                check_execution_fence()
                                prepared = result_store.stage(request, observed)
                            except ExecutionLeaseLostError:
                                raise
                            except Exception as exc:
                                raise RetryableTaskError("workflow result could not be staged") from exc
                else:
                    check_execution_fence()
                    candidate = self._run_workflow(request)
                check_execution_fence()
                if candidate is not None:
                    candidate = WorkflowResult.model_validate(candidate)
                    if (candidate.workflow_id, candidate.trace_id) != (request.workflow_id, request.trace_id):
                        raise ValueError("workflow result identity does not match Harness run")
                    result = candidate
                    if self._completion_candidate_factory is not None:
                        check_execution_fence()
                        completion_candidate = self._completion_candidate_factory(
                            request, result, view
                        )
                        if type(completion_candidate) is not dict:
                            raise TypeError("host completion candidate must be an exact dictionary")
            except ExecutionLeaseLostError:
                # Owner loss is not a workflow failure or a user cancellation.
                # A replacement owner retains authority over the shared run.
                raise
            except RetryableTaskError:
                check_execution_fence()
                # Do not make success (or degradation) irreversible before the
                # host-owned result journal is durable.
                raise
            except Exception as exc:  # Host error is recorded by deterministic degradation below.
                check_execution_fence()
                error = type(exc).__name__
                if observed_runner is not None:
                    history_reader = getattr(self._kernel, "checkpoint_history", None)
                    durable = tuple(history_reader(handle.run_id)) if callable(history_reader) else ()
                    if durable:
                        checkpoints = durable
                        workflow_usage = durable[-1].usage
            # No candidate is derived from model output.  The kernel therefore
            # applies its signed confidence policy and degrades safely if a
            # production evidence adapter has not authorized completion.
            for _ in range(3):
                check_execution_fence()
                view = self._kernel.snapshot(handle.run_id)
                if signal is not None and signal.is_cancelled() and view.run_state is not RunState.CANCELLED:
                    check_execution_fence()
                    self._kernel.cancel(handle.run_id, "cooperative_cancellation")
                    view = self._kernel.snapshot(handle.run_id)
                if view.run_state in {RunState.SUCCEEDED, RunState.DEGRADED, RunState.FAILED, RunState.CANCELLED}:
                    break
                advance_values = {
                    "candidate": completion_candidate,
                    "expected_state_revision": view.state_revision,
                }
                if workflow_usage is not None:
                    advance_values["workflow_usage"] = workflow_usage
                if observed_runner is not None:
                    advance_values["observed_execution"] = True
                    if prompt_release_digest is not None:
                        advance_values["prompt_release_digest"] = prompt_release_digest
                    if result is not None:
                        advance_values["accepted_result_digest"] = (
                            canonical_workflow_result_digest(result)
                        )
                check_execution_fence()
                decision = self._kernel.advance(handle.run_id, **advance_values)
                decisions.append(decision)
                completion_candidate = None
        view = self._kernel.snapshot(handle.run_id)
        check_execution_fence()
        if view.run_state is RunState.SUCCEEDED and result is None and result_store is not None:
            raise RetryableTaskError("successful workflow has no recoverable durable result")
        if (
            view.run_state is RunState.SUCCEEDED
            and result is not None
            and self._accepted_result_committer is not None
            and not (signal is not None and signal.is_cancelled())
        ):
            try:
                check_execution_fence()
                if prompt_release_digest is None or view.last_event_hash is None:
                    raise RuntimeError("accepted outcome proof is unavailable")
                receipt_reader = getattr(self._kernel, "terminal_receipt", None)
                if not callable(receipt_reader):
                    raise RuntimeError("accepted outcome receipt is unavailable")
                proof = AcceptedOutcomeProof.bind(
                    request,
                    result,
                    terminal_receipt=receipt_reader(handle.run_id),
                    prompt_release_digest=prompt_release_digest,
                    accepted_at=(prepared.proof.accepted_at
                                 if prepared is not None and prepared.proof is not None
                                 else datetime.now(timezone.utc)),
                )
                if result_store is not None:
                    check_execution_fence()
                    proof = result_store.bind_proof(request, proof)
                if prepared is None or not prepared.committed:
                    check_execution_fence()
                    self._accepted_result_committer(request, result, proof)
                    if result_store is not None:
                        check_execution_fence()
                        result_store.mark_committed(request, proof)
            except ExecutionLeaseLostError:
                raise
            except Exception as exc:
                check_execution_fence()
                if result_store is not None:
                    raise RetryableTaskError("accepted result commit is pending") from exc
                # Legacy hosts without a journal still fail closed.
                error = type(exc).__name__
                result = None
        check_execution_fence()
        return HarnessWorkflowExecution(
            handle=handle,
            view=view,
            decisions=tuple(decisions),
            workflow_result=result,
            workflow_error=error,
            workflow_usage=workflow_usage,
            checkpoints=checkpoints,
        )
