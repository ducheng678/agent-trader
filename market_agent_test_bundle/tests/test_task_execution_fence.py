from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from market_agent.backend.cache import TTLCache
from market_agent.backend.database import JobRepository
from market_agent.backend.errors import ExecutionLeaseLostError, RetryableTaskError
from market_agent.backend.execution_fence import (
    ExecutionFence, current_execution_fence, execution_fence_context,
)
from market_agent.backend.message_bus import InMemoryMessageBus, MessageEnvelope
from market_agent.backend.observability import MetricsRegistry
from market_agent.backend.task_queue import BackgroundTaskQueue


def eventually(predicate, timeout=3):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if predicate():
            return
        time.sleep(0.005)
    assert predicate()


def queue(repository, **overrides):
    worker = BackgroundTaskQueue(
        repository, TTLCache(32, 60), InMemoryMessageBus(), MetricsRegistry(),
        1, 1, 3, overrides.pop("retry_delay_seconds", 0),
        lease_seconds=0.3, **overrides,
    )
    # Keep recovery explicit so the retired owner's state can be inspected.
    worker._start_recovery = lambda *_args: None
    return worker


@pytest.mark.parametrize("outcome", ["success", "retry", "failure"])
@pytest.mark.parametrize("renewal_error", ["exception", "false"])
def test_uncertain_owner_never_transitions_or_retries_and_replacement_can_run(
    tmp_path, monkeypatch, outcome, renewal_error,
):
    repository = JobRepository(tmp_path / "jobs.db")
    worker = queue(repository)
    replacement = queue(JobRepository(tmp_path / "jobs.db"))
    started, finish, fail_renewal = threading.Event(), threading.Event(), threading.Event()
    heartbeat_finished = threading.Event()
    attempts = []
    original_renew = repository.renew_job_lease
    original_increment = worker._metrics.increment

    def renew(*args):
        if fail_renewal.is_set() and threading.current_thread() is worker._lease_thread:
            if renewal_error == "exception":
                raise OSError("lease database transport failed")
            heartbeat_finished.set()
            return False
        return original_renew(*args)

    def increment(name, *args, **kwargs):
        original_increment(name, *args, **kwargs)
        if name == "market_agent_task_lease_renew_failed_total":
            heartbeat_finished.set()

    monkeypatch.setattr(repository, "renew_job_lease", renew)
    monkeypatch.setattr(worker._metrics, "increment", increment)

    def handler(payload):
        attempts.append(dict(payload))
        started.set()
        assert finish.wait(3)
        if outcome == "retry":
            raise RetryableTaskError("provider asks for retry")
        if outcome == "failure":
            raise RuntimeError("provider rejected the request")
        return {"owner": "retired"}

    worker.register("work", handler)
    replacement.register("work", lambda payload: {"owner": "replacement", **payload})
    try:
        submitted = worker.submit("work", {"workflow_id": "same-run"})
        assert started.wait(2)
        before = repository.list_events(submitted.job.job_id)
        fail_renewal.set()
        assert heartbeat_finished.wait(2)
        # Synchronize with the heartbeat loop after its return/exception.
        eventually(lambda: (
            worker._execution_fences[submitted.job.job_id].is_lost()
            if renewal_error == "false"
            else "market_agent_task_lease_renew_failed_total" in worker._metrics.render_prometheus()
        ))
        fail_renewal.clear()  # Recovery of transport cannot revive this owner.
        finish.set()
        worker.shutdown()
        assert len(attempts) == 1
        assert repository.get_job(submitted.job.job_id).status == "running"
        assert repository.list_events(submitted.job.job_id) == before

        replacement._handle_dispatch_message(MessageEnvelope(
            topic="task.dispatch", payload={"task_name": "work", "job_id": submitted.job.job_id},
        ))
        eventually(lambda: repository.get_job(submitted.job.job_id).status == "succeeded")
        assert repository.get_job(submitted.job.job_id).result == {
            "owner": "replacement", "workflow_id": "same-run",
        }
    finally:
        finish.set()
        worker.shutdown()
        replacement.shutdown()


def test_execution_fence_context_restores_outer_scope_after_exception():
    outer, inner = ExecutionFence(), ExecutionFence()
    assert current_execution_fence() is None
    with execution_fence_context(outer):
        with pytest.raises(ExecutionLeaseLostError):
            with execution_fence_context(inner):
                assert current_execution_fence() is inner
                inner.latch_lost()
                inner.latch_lost()
                inner.raise_if_lost()
        assert current_execution_fence() is outer
        assert not outer.is_lost()
    assert current_execution_fence() is None


@pytest.mark.parametrize("fails", [False, True])
def test_reused_executor_thread_has_no_fence_leak(tmp_path, fails):
    repository = JobRepository(tmp_path / "jobs.db")
    worker = queue(repository)
    fences = []

    def handler(payload):
        fence = current_execution_fence()
        assert fence is not None and not fence.is_lost()
        fences.append(fence)
        if payload["first"] and fails:
            raise RuntimeError("first handler failed")
        return dict(payload)

    worker.register("work", handler)
    try:
        first = worker.submit("work", {"first": True}).job
        eventually(lambda: not worker._active_job_ids)
        assert worker._executor.submit(current_execution_fence).result(timeout=2) is None
        second = worker.submit("work", {"first": False}).job
        eventually(lambda: not worker._active_job_ids)
        assert worker._executor.submit(current_execution_fence).result(timeout=2) is None
        assert len(fences) == 2 and fences[0] is not fences[1]
        assert repository.get_job(first.job_id).status == ("failed" if fails else "succeeded")
        assert repository.get_job(second.job_id).result == {"first": False}
    finally:
        worker.shutdown()


def test_loss_between_started_event_and_handler_prevents_invocation(tmp_path, monkeypatch):
    repository = JobRepository(tmp_path / "jobs.db")
    worker = queue(repository)
    calls = []
    original_publish = worker._publish

    def publish(topic, job, payload):
        original_publish(topic, job, payload)
        if topic == "task.started":
            current_execution_fence().latch_lost()

    monkeypatch.setattr(worker, "_publish", publish)
    worker.register("work", lambda payload: calls.append(payload))
    try:
        job = worker.submit("work", {}).job
        eventually(lambda: not worker._active_job_ids)
        assert calls == []
        assert repository.get_job(job.job_id).status == "running"
    finally:
        worker.shutdown()


def test_infrastructure_failure_after_lost_fence_cannot_mark_failed(tmp_path, monkeypatch):
    repository = JobRepository(tmp_path / "jobs.db")
    worker = queue(repository)
    original_get = repository.get_job

    def fail_before_handler(*_args, **_kwargs):
        raise RuntimeError("trace adapter crashed")

    def lose_during_failure_lookup(job_id):
        # The terminal infrastructure path must recheck after reading state.
        current_execution_fence().latch_lost()
        return original_get(job_id)

    monkeypatch.setattr(worker, "_record_trace", fail_before_handler)
    monkeypatch.setattr(repository, "get_job", lose_during_failure_lookup)
    worker.register("work", lambda payload: payload)
    try:
        job = worker.submit("work", {}).job
        eventually(lambda: not worker._active_job_ids)
        assert original_get(job.job_id).status == "running"
        assert worker._executor.submit(current_execution_fence).result(timeout=2) is None
    finally:
        worker.shutdown()


@pytest.mark.parametrize("shutting_down", [False, True])
def test_loss_during_retry_sleep_prevents_retry_and_shutdown_terminal_write(tmp_path, shutting_down):
    repository = JobRepository(tmp_path / "jobs.db")
    worker = queue(repository, retry_delay_seconds=0.2)
    attempts = []

    def handler(_payload):
        attempts.append(current_execution_fence())
        raise RetryableTaskError("try later")

    worker.register("work", handler)
    try:
        job = worker.submit("work", {}).job
        eventually(lambda: attempts and repository.get_job(job.job_id).status == "accepted")
        before = repository.list_events(job.job_id)
        attempts[0].latch_lost()
        if shutting_down:
            worker.shutdown(wait=False)
        eventually(lambda: not worker._active_job_ids)
        assert len(attempts) == 1
        assert repository.get_job(job.job_id).status == "accepted"
        assert repository.list_events(job.job_id) == before
    finally:
        worker.shutdown()


def workflow_request():
    from market_agent.workflow_contracts import WorkflowRequest
    return WorkflowRequest(
        workflow_id="same-run", trace_id="1" * 32,
        user_query="Inspect the supplied example.", trigger_reason="manual_once",
    )


def production_application(monkeypatch, invoke):
    from market_agent.backend.settings import BackendSettings
    from market_agent import workflow_production_application as module
    from market_agent.workflow_observation import CoreNodeName, NodeOutcome, ObservedWorkItem, TaskRetryState

    callbacks = []

    def coordinator(**kwargs):
        callbacks.append(kwargs["cancellation_check"])
        return SimpleNamespace(decide=lambda *_args: None, technical=lambda *_args: None,
                               verifier=lambda *_args: None)

    monkeypatch.setattr(module, "AgentCoordinatorServices", coordinator)
    monkeypatch.setattr(module, "_lookup_historical_answer", lambda **_kwargs: None)
    monkeypatch.setattr(module, "_retrieve_core_memory", lambda **_kwargs: None)

    def observed_invoke(request, services):
        services.execution_observer.checkpoint(
            plan_revision=0, node=CoreNodeName.PLAN, outcome=NodeOutcome.COMPLETED,
            task_ids=("task-1",), completed_task_ids=(), failed_task_ids=(),
            retry_state=(TaskRetryState(task_id="task-1", attempts_consumed=0,
                                       retries_consumed=0, retries_remaining=0),),
            work_items=(ObservedWorkItem(
                task_id="task-1", task_kind="technical", worker_id="technical-agent",
                owner_node=CoreNodeName.DISPATCH, maximum_retries=0, execution_state="pending",
            ),), action_fingerprint="d" * 64,
        )
        return invoke(request, services)

    dependencies = module.ProductionDependencies(
        settings=BackendSettings(environment="test"),
        driver_factory=lambda *_args: object(), audit_writer=SimpleNamespace(healthy=True),
        memory_repository=None, embedding_client=None, completion_hook=lambda *_args: None,
        prompt_release_manager=SimpleNamespace(current=lambda: SimpleNamespace(release_digest="a" * 64)),
        workflow_factory=lambda: SimpleNamespace(invoke=observed_invoke),
    )
    return module.ProductionWorkflowApplication(lambda: dependencies), callbacks


def unknown_result(request):
    from market_agent.workflow_playbook_assembler import unknown_playbook
    return unknown_playbook(workflow_id=request.workflow_id, trace_id=request.trace_id, reason="offline test")


def test_production_callbacks_combine_api_cancellation_and_owner_fence_across_threads(monkeypatch):
    from market_agent.workflow_cancellation import WorkflowCancellationRegistry

    registry = WorkflowCancellationRegistry()
    signal = registry.signal("same-run")
    graph_callbacks = []

    def invoke(request, services):
        graph_callbacks.append(services.cancelled)
        return unknown_result(request)

    application, coordinator_callbacks = production_application(monkeypatch, invoke)
    old, replacement = ExecutionFence(), ExecutionFence()
    with execution_fence_context(old):
        application.execute_workflow(workflow_request(), cancellation_signal=signal)
    with execution_fence_context(replacement):
        application.execute_workflow(workflow_request(), cancellation_signal=signal)

    with ThreadPoolExecutor(1) as pool:
        assert pool.submit(current_execution_fence).result() is None
        for callback in (*coordinator_callbacks, *graph_callbacks):
            assert pool.submit(callback).result() is False
        old.latch_lost()
        assert not signal.is_cancelled()
        for callbacks in (coordinator_callbacks, graph_callbacks):
            assert pool.submit(callbacks[0]).result() is True
            assert pool.submit(callbacks[1]).result() is False
        registry.cancel("same-run")
        for callbacks in (coordinator_callbacks, graph_callbacks):
            assert pool.submit(callbacks[1]).result() is True


def test_inflight_production_handler_stops_next_provider_attempt_after_heartbeat_exception(tmp_path, monkeypatch):
    from test_workflow_agent_driver import Client, invocation, make_driver, response
    from market_agent.workflow_cancellation import WorkflowCancellationRegistry
    from market_agent.workflow_retry_policy import ProviderError

    repository = JobRepository(tmp_path / "jobs.db")
    worker = queue(repository)
    entered_provider, finish_provider = threading.Event(), threading.Event()
    captured_fences, driver_results = [], []
    original_renew = repository.renew_job_lease
    registry = WorkflowCancellationRegistry()

    def renew(*args):
        if entered_provider.is_set() and threading.current_thread() is worker._lease_thread:
            raise OSError("renewal uncertain while provider is running")
        return original_renew(*args)

    def provider(_request):
        entered_provider.set()
        assert finish_provider.wait(3)
        raise ProviderError(status_code=503)

    client = Client(provider, response())
    driver, _, _ = make_driver(client)

    def invoke(request, services):
        captured_fences.append(current_execution_fence())
        with ThreadPoolExecutor(1) as pool:
            driver_results.append(pool.submit(
                driver.execute, invocation(), cancellation_check=services.cancelled,
            ).result(timeout=3))
        return unknown_result(request)

    application, callbacks = production_application(monkeypatch, invoke)
    monkeypatch.setattr(repository, "renew_job_lease", renew)
    worker.register("work", lambda _payload: application.execute_workflow(
        workflow_request(), cancellation_signal=registry.signal("same-run"),
    ).result.model_dump(mode="json"))
    try:
        job = worker.submit("work", {}).job
        assert entered_provider.wait(2)
        eventually(lambda: captured_fences[0].is_lost())
        assert callbacks[0]() and not registry.signal("same-run").is_cancelled()
        finish_provider.set()
        eventually(lambda: not worker._active_job_ids)
        assert len(client.requests) == 1
        assert driver_results[0].failure.code == "cancelled"
        assert repository.get_job(job.job_id).status == "running"
    finally:
        finish_provider.set()
        worker.shutdown()


@pytest.mark.parametrize("service_name", ["harness", "agent"])
def test_service_fences_entry_and_exit_but_direct_calls_remain_compatible(service_name):
    from market_agent.backend.agent_service import AgentPlaybookService
    from market_agent.backend.harness_service import HarnessWorkflowService
    from market_agent.workflow_harness_application import HarnessWorkflowApplication
    from market_agent.workflow_harness_contracts import RunState

    calls = []
    should_lose = False

    def work(*_args, **_kwargs):
        calls.append(1)
        if should_lose:
            current_execution_fence().latch_lost()
        if service_name == "agent":
            return SimpleNamespace(to_dict=lambda: {"answer": "known"}), "offline"
        return SimpleNamespace(
            workflow_result=unknown_result(workflow_request()),
            view=SimpleNamespace(run_state=RunState.SUCCEEDED, sequence=1, state_revision=1),
            handle=SimpleNamespace(run_id="same-run", trace_id="1" * 32), workflow_error=None,
        )

    if service_name == "harness":
        application = object.__new__(HarnessWorkflowApplication)
        application.execute = work
        execute = HarnessWorkflowService(application).execute
        payload = workflow_request().model_dump(mode="json")
    else:
        execute = AgentPlaybookService(application_factory=lambda: SimpleNamespace(get_playbook=work)).generate_playbook
        payload = {"user_query": "Inspect the supplied example.", "event_tape": [], "trigger_reason": "manual_once"}

    assert execute(payload)
    lost = ExecutionFence()
    lost.latch_lost()
    with execution_fence_context(lost), pytest.raises(ExecutionLeaseLostError):
        execute(payload)
    assert len(calls) == 1
    should_lose = True
    with execution_fence_context(ExecutionFence()), pytest.raises(ExecutionLeaseLostError):
        execute(payload)
    assert len(calls) == 2


@pytest.mark.parametrize("loss_phase", ["raise", "return", "candidate", "terminal_read", "checkpoint", "stage"])
def test_harness_owner_loss_does_not_advance_or_cancel_shared_terminal_state(monkeypatch, loss_phase):
    from test_backend_harness_service import _application_for_terminal, _payload
    from market_agent.workflow_cancellation import WorkflowCancellationRegistry
    from market_agent.workflow_harness_contracts import RunState

    committed, checkpoints, stages = [], [], []
    application = _application_for_terminal(RunState.SUCCEEDED, committed)
    kernel = application._kernel
    fence = ExecutionFence()
    registry = WorkflowCancellationRegistry()
    application._cancellation_signal_factory = registry.signal
    original_runner = application._run_observed_workflow
    original_snapshot = kernel.snapshot
    runner_returned = False

    def runner(request, sink):
        nonlocal runner_returned
        if loss_phase in {"raise", "return", "checkpoint"}:
            fence.latch_lost()
        if loss_phase == "raise":
            raise ExecutionLeaseLostError("old owner retired")
        if loss_phase == "checkpoint":
            sink(object())
        runner_returned = True
        return original_runner(request, sink)

    def candidate(*_args):
        if loss_phase == "candidate":
            fence.latch_lost()
        return {"accepted": True}

    def snapshot(run_id):
        if loss_phase == "terminal_read" and runner_returned:
            fence.latch_lost()
        return original_snapshot(run_id)

    def stage(*args):
        stages.append(args)
        fence.latch_lost()
        return None

    kernel.record_checkpoint = lambda *_args: checkpoints.append(1)
    kernel.snapshot = snapshot
    application._run_observed_workflow = runner
    application._completion_candidate_factory = candidate
    if loss_phase in {"raise", "return", "checkpoint", "stage"}:
        application._result_store = SimpleNamespace(
            load=lambda _request: None, stage=stage,
            bind_proof=lambda _request, proof: proof, mark_committed=lambda *_args: None,
        )
    monkeypatch.setattr("market_agent.workflow_memory_result_writer.verify_committed_transition_receipt", lambda _receipt: True)

    with execution_fence_context(fence), pytest.raises(ExecutionLeaseLostError):
        application.execute(_payload())
    assert not kernel.finished
    assert original_snapshot("workflow-1").run_state is RunState.RUNNING
    assert not registry.signal("workflow-1").is_cancelled()
    assert checkpoints == [] and committed == []
    assert len(stages) == (1 if loss_phase == "stage" else 0)


def test_harness_rethrows_lease_loss_without_a_context_fence():
    from test_backend_harness_service import _application_for_terminal, _payload
    from market_agent.workflow_harness_contracts import RunState

    application = _application_for_terminal(RunState.DEGRADED, [])

    def runner(*_args):
        raise ExecutionLeaseLostError("another layer detected lease loss")

    application._run_observed_workflow = runner
    assert current_execution_fence() is None
    with pytest.raises(ExecutionLeaseLostError):
        application.execute(_payload())
    assert not application._kernel.finished


@pytest.mark.parametrize("loss_phase", ["before", "hook"])
def test_production_commit_stops_subsequent_history_embedding_after_owner_loss(tmp_path, monkeypatch, loss_phase):
    from dataclasses import replace
    from test_workflow_production_application import _application
    from test_workflow_result_recovery import _setup
    from market_agent import workflow_production_application as module

    harness, store, request = _setup(tmp_path, monkeypatch)
    harness.execute(request)
    saved = store.load(request)
    fence, committed, history = ExecutionFence(), [], []
    production = _application(committed)

    def commit(*_args):
        committed.append(1)
        fence.latch_lost()

    production._dependencies = replace(production._get_dependencies(), result_store=store, completion_hook=commit)
    monkeypatch.setattr(module, "_store_historical_answer", lambda **_kwargs: history.append(1))
    if loss_phase == "before":
        fence.latch_lost()
    with execution_fence_context(fence), pytest.raises(ExecutionLeaseLostError):
        production.commit_accepted_result(request, saved.execution.result, saved.proof)
    assert committed == ([] if loss_phase == "before" else [1])
    assert history == []


def test_replacement_recovers_expired_lease_while_old_handler_remains_inflight(tmp_path, monkeypatch):
    from market_agent.backend import database

    database_time = ["2026-09-15T00:00:00+00:00"]
    monkeypatch.setattr(database, "_utc_now", lambda: database_time[0])
    repository = JobRepository(tmp_path / "jobs.db")
    worker = queue(repository)
    replacement = queue(JobRepository(tmp_path / "jobs.db"))
    started, finish = threading.Event(), threading.Event()
    old_fences, new_fences = [], []
    original_renew = repository.renew_job_lease

    def renew(*args):
        if started.is_set() and threading.current_thread() is worker._lease_thread:
            raise OSError("lease transport stays unavailable")
        return original_renew(*args)

    def old_handler(_payload):
        old_fences.append(current_execution_fence())
        started.set()
        assert finish.wait(3)
        raise RetryableTaskError("old handler asks for another attempt")

    def new_handler(payload):
        new_fences.append(current_execution_fence())
        assert not new_fences[-1].is_lost()
        return {"replacement": payload}

    monkeypatch.setattr(repository, "renew_job_lease", renew)
    worker.register("work", old_handler)
    replacement.register("work", new_handler)
    try:
        job = worker.submit("work", {"workflow_id": "same-run"}).job
        assert started.wait(2)
        eventually(lambda: old_fences[0].is_lost())
        database_time[0] = "2026-09-15T00:01:00+00:00"
        replacement._handle_dispatch_message(MessageEnvelope(
            topic="task.dispatch", payload={"task_name": "work", "job_id": job.job_id},
        ))
        eventually(lambda: repository.get_job(job.job_id).status == "succeeded")
        assert not finish.is_set()
        assert old_fences[0] is not new_fences[0]
        before = repository.list_events(job.job_id)
        finish.set()
        eventually(lambda: not worker._active_job_ids)
        assert len(old_fences) == len(new_fences) == 1
        assert repository.get_job(job.job_id).result == {"replacement": {"workflow_id": "same-run"}}
        assert repository.list_events(job.job_id) == before
    finally:
        finish.set()
        worker.shutdown()
        replacement.shutdown()


def test_synchronous_renewal_exception_is_not_persisted_as_infrastructure_failure(tmp_path, monkeypatch):
    repository = JobRepository(tmp_path / "jobs.db")
    worker = queue(repository)
    calls = []

    def renew(*_args):
        raise OSError("lease database unavailable before execution")

    monkeypatch.setattr(repository, "renew_job_lease", renew)
    worker.register("work", lambda payload: calls.append(payload))
    try:
        job = worker.submit("work", {}).job
        eventually(lambda: not worker._active_job_ids)
        assert calls == []
        assert repository.get_job(job.job_id).status == "accepted"
        assert worker._executor.submit(current_execution_fence).result(timeout=2) is None
    finally:
        worker.shutdown()


def test_infrastructure_terminal_write_remains_token_guarded_without_context(tmp_path):
    repository = JobRepository(tmp_path / "jobs.db")
    worker = queue(repository)
    job = repository.create_or_get_job("work", {}, None, 3, "")[0]
    old = repository.claim_job(job.job_id, "old", 30)
    repository.mark_running(job.job_id, 1, execution_token="old")
    repository.release_job_lease(job.job_id, "old")
    repository.claim_job(job.job_id, "replacement", 30, recovery=True)
    before = repository.list_events(job.job_id)
    try:
        assert current_execution_fence() is None
        worker._record_terminal_failure(old, RuntimeError("old infrastructure error"), "TaskInfrastructureError")
        assert repository.get_job(job.job_id).execution_token == "replacement"
        assert repository.list_events(job.job_id) == before
    finally:
        worker.shutdown()
