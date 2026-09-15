from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from market_agent.evaluation_cli import run as cli_run
from market_agent.workflow_eval_dataset import EvaluationDataset, RecordedObservation
from market_agent.workflow_eval_metrics import ReleaseGate, ReleaseThresholds
from market_agent.workflow_evaluation import EvaluationBinding, EvaluationRun, EvaluationRunner, ExecutedObservation
from market_agent.workflow_eval_executor import WorkflowEvaluationExecutor
from market_agent.workflow_contracts import Action, KnowledgeStatus, TerminalMode, WorkflowResult
from market_agent.workflow_observation import WorkflowExecution, WorkflowUsage
from market_agent.workflow_production_application import ProductionDependencies, ProductionWorkflowApplication
from market_agent.backend.settings import BackendSettings


def _dataset():
    return EvaluationDataset.load("evals/datasets/offline-safety-v1.manifest.json")


def _binding():
    return EvaluationBinding(code_revision="test-revision", prompt_release_hash="1" * 64, model_policy_hash="2" * 64)


class _CurrentHost:
    def __init__(self, observations):
        self.observations = observations
        self.calls = []
        self.attested_binding = _binding()
        self.prompt_release_hash = self.attested_binding.prompt_release_hash

    def run_workflow(self, task_input, trace_id):
        assert set(task_input).isdisjoint({"expected", "recording", "allowed_evidence_ids", "case_hash"})
        self.calls.append((deepcopy(task_input), trace_id))
        return self.observations[trace_id]

    def __call__(self, task_input, trace_id):
        return ExecutedObservation(
            observation=self.run_workflow(task_input, trace_id),
            prompt_release_hash=self.prompt_release_hash,
        )


def test_executed_evaluation_uses_current_host_results_and_actual_usage():
    dataset = _dataset()
    observations = {}
    for case in dataset.cases:
        payload = case.recording.model_dump(mode="json")
        payload["input_tokens"] = 7
        payload["output_tokens"] = 3
        payload["cost_usd"] = 0.002
        payload["observed_violations"] = tuple(payload["observed_violations"])
        if case.case_id == "seed-regression":
            payload["output"] = {"conclusion": "不知道", "action": "no_trade", "facts": [], "evidence_ids": [], "trace_id": case.trace_id}
        observations[case.trace_id] = RecordedObservation.model_validate(payload)
    host = _CurrentHost(observations)

    recorded = EvaluationRunner().run(dataset, binding=_binding(), execution_mode="recorded")
    executed = EvaluationRunner().run(dataset, binding=_binding(), execution_mode="executed", executor=host)

    assert executed.execution_mode == "executed"
    assert recorded.execution_mode == "recorded"
    assert executed.run_hash != recorded.run_hash
    assert len(host.calls) == len(dataset.cases)
    assert {trace for _task, trace in host.calls} == {case.trace_id for case in dataset.cases}
    assert executed.metrics.total_tokens == len(dataset.cases) * 10
    assert executed.metrics.total_cost_usd == pytest.approx(len(dataset.cases) * 0.002)
    assert not next(score for score in executed.scores if score.case_id == "seed-regression").success
    assert next(score for score in recorded.scores if score.case_id == "seed-regression").success


def test_executed_failure_is_scored_invalid_not_substituted_with_recording():
    dataset = _dataset()

    class FailingExecutor:
        attested_binding = _binding()

        def __call__(self, _input, _trace):
            raise RuntimeError("provider did not respond")

    run = EvaluationRunner().run(dataset, binding=_binding(), execution_mode="executed", executor=FailingExecutor())
    assert all(not score.success and "invalid_schema" in score.hard_violations for score in run.scores)
    assert run.metrics.total_tokens == 0
    assert run.binding_verified is False
    loose = ReleaseThresholds(require_baseline=False, minimum_cases=1, minimum_success_rate=0.0,
                              minimum_success_lower_bound=0.0)
    assert "execution_binding_unverified" in ReleaseGate(loose).evaluate(run).reasons


def test_recorded_scoring_cannot_pass_current_release_gate_or_compare_to_executed():
    dataset = _dataset()
    runner = EvaluationRunner()
    recorded = runner.run(dataset, binding=_binding(), execution_mode="recorded")
    host = _CurrentHost({case.trace_id: case.recording for case in dataset.cases})
    executed = runner.run(dataset, binding=_binding(), execution_mode="executed", executor=host)
    thresholds = ReleaseThresholds(require_baseline=False, minimum_cases=1, minimum_success_rate=0.0, minimum_success_lower_bound=0.0)
    assert "executed_mode_required" in ReleaseGate(thresholds).evaluate(recorded).reasons
    with pytest.raises(ValueError, match="mode"):
        runner.compare(executed, recorded)
    assert "incompatible_baseline" in ReleaseGate(thresholds).evaluate(executed, recorded).reasons


def test_cli_executed_mode_requires_explicit_executor_factory(tmp_path):
    argv = ["--manifest", "evals/datasets/offline-safety-v1.manifest.json", "--code-revision", "test",
            "--prompt-release-hash", "0" * 64, "--model-policy-hash", "0" * 64,
            "--output-dir", str(tmp_path), "--allow-missing-baseline"]
    code, report = cli_run([*argv, "--execution-mode", "executed"])
    assert code == 2
    assert report["allowed"] is False
    assert list(tmp_path.iterdir()) == []

    code, report = cli_run([*argv, "--execution-mode", "recorded"])
    assert code == 1
    assert "executed_mode_required" in report["reasons"]
    assert report["execution_mode"] == "recorded"


def test_explicit_production_adapter_executes_workflow_and_reads_bound_usage():
    class Application:
        def __init__(self):
            self.requests = []
            self.evaluation_binding = _binding()

        def execute_workflow(self, request):
            self.requests.append(request)
            result = WorkflowResult(
                workflow_id=request.workflow_id, trace_id=request.trace_id,
                terminal_mode=TerminalMode.UNKNOWN, final_action=Action.NO_TRADE,
                knowledge_status=KnowledgeStatus.INSUFFICIENT,
                uncertainty_reason="evidence_missing",
            )
            return WorkflowExecution(result=result, usage=WorkflowUsage.empty(request.workflow_id, request.trace_id),
                                     checkpoints=(), completion_kind="historical_cache",
                                     prompt_release_digest=self.evaluation_binding.prompt_release_hash)

    application = Application()
    case = _dataset().cases[0]
    executed = WorkflowEvaluationExecutor(application)(deepcopy(dict(case.task_input)), case.trace_id)
    observation = executed.observation
    assert executed.prompt_release_hash == _binding().prompt_release_hash
    assert len(application.requests) == 1
    assert application.requests[0].trace_id == case.trace_id
    assert application.requests[0].user_query == case.task_input["query"]
    assert observation.output["conclusion"] == "不知道"
    assert observation.output["trace_id"] == case.trace_id
    assert observation.input_tokens == observation.output_tokens == 0
    assert observation.cost_usd == 0.0


def test_cli_injected_factory_runs_current_host_not_implicit_recording(tmp_path):
    dataset = _dataset()
    host = _CurrentHost({case.trace_id: case.recording for case in dataset.cases})
    code, report = cli_run([
        "--manifest", "evals/datasets/offline-safety-v1.manifest.json", "--code-revision", _binding().code_revision,
        "--prompt-release-hash", _binding().prompt_release_hash, "--model-policy-hash", _binding().model_policy_hash,
        "--output-dir", str(tmp_path), "--allow-missing-baseline", "--execution-mode", "executed",
    ], executor_factory=lambda: host)
    assert code == 0
    assert report["allowed"] is True
    assert report["execution_mode"] == "executed"
    assert len(host.calls) == len(dataset.cases)


def test_executed_evaluation_rejects_cli_binding_not_attested_by_host():
    host = _CurrentHost({case.trace_id: case.recording for case in _dataset().cases})
    host.attested_binding = _binding().model_copy(update={"code_revision": "different-build"})
    with pytest.raises(ValueError, match="binding"):
        EvaluationRunner().run(_dataset(), binding=_binding(), execution_mode="executed", executor=host)


def test_executed_evaluation_rejects_prompt_version_drift_across_cases():
    host = _CurrentHost({case.trace_id: case.recording for case in _dataset().cases})
    host.attested_binding = _binding()
    host.prompt_release_hash = "9" * 64
    with pytest.raises(ValueError, match="prompt.*binding"):
        EvaluationRunner().run(_dataset(), binding=_binding(), execution_mode="executed", executor=host)


def test_executed_evaluation_rejects_host_binding_change_mid_run():
    class FlippingHost(_CurrentHost):
        def __call__(self, task_input, trace_id):
            observed = super().__call__(task_input, trace_id)
            self.attested_binding = _binding().model_copy(update={"model_policy_hash": "8" * 64})
            return observed

    host = FlippingHost({case.trace_id: case.recording for case in _dataset().cases})
    with pytest.raises(ValueError, match="binding changed"):
        EvaluationRunner().run(_dataset(), binding=_binding(), execution_mode="executed", executor=host)


def test_cli_mismatched_host_binding_produces_no_release_artifact(tmp_path):
    host = _CurrentHost({case.trace_id: case.recording for case in _dataset().cases})
    code, report = cli_run([
        "--manifest", "evals/datasets/offline-safety-v1.manifest.json",
        "--code-revision", "unrelated-build",
        "--prompt-release-hash", host.attested_binding.prompt_release_hash,
        "--model-policy-hash", host.attested_binding.model_policy_hash,
        "--output-dir", str(tmp_path), "--allow-missing-baseline", "--execution-mode", "executed",
    ], executor_factory=lambda: host)
    assert code == 2 and report["allowed"] is False
    assert list(tmp_path.iterdir()) == []


def test_legacy_executed_artifact_cannot_gain_verified_binding_on_reload():
    dataset = _dataset()
    host = _CurrentHost({case.trace_id: case.recording for case in dataset.cases})
    run = EvaluationRunner().run(dataset, binding=_binding(), execution_mode="executed", executor=host)
    legacy = run.model_dump(mode="json", exclude={"run_hash", "binding_verified"})
    legacy["evaluator_version"] = "offline-evaluator-v2"
    restored = EvaluationRun.model_validate_json(json.dumps(legacy))
    assert EvaluationRun.model_validate_json(restored.model_dump_json()).run_hash == restored.run_hash
    assert restored.binding_verified is False
    assert "execution_binding_unverified" in ReleaseGate(
        ReleaseThresholds(require_baseline=False, minimum_cases=1, minimum_success_rate=0.0,
                          minimum_success_lower_bound=0.0)
    ).evaluate(restored).reasons


def test_real_production_application_exposes_host_bound_evaluation_identity_offline():
    class OfflinePromptManager:
        def current(self):
            return SimpleNamespace(release_digest="1" * 64)

    dependencies = ProductionDependencies(
        settings=BackendSettings(environment="development"),
        driver_factory=lambda **_kwargs: None, audit_writer=object(),
        memory_repository=None, embedding_client=None,
        completion_hook=lambda *_args: None,
        prompt_release_manager=OfflinePromptManager(),
    )
    application = ProductionWorkflowApplication(
        lambda: dependencies, evaluation_code_revision="trusted-build-revision",
    )
    actual_binding = application.evaluation_binding
    assert actual_binding.code_revision == "trusted-build-revision"
    assert actual_binding.prompt_release_hash == "1" * 64

    def offline_execution(request):
        result = WorkflowResult(
            workflow_id=request.workflow_id, trace_id=request.trace_id,
            terminal_mode=TerminalMode.UNKNOWN, final_action=Action.NO_TRADE,
            knowledge_status=KnowledgeStatus.INSUFFICIENT, uncertainty_reason="evidence_missing",
        )
        return WorkflowExecution(
            result=result, usage=WorkflowUsage.empty(request.workflow_id, request.trace_id),
            checkpoints=(), completion_kind="historical_cache",
            prompt_release_digest=application.evaluation_binding.prompt_release_hash,
        )

    application.execute_workflow = offline_execution
    case = _dataset().cases[0]
    observed = WorkflowEvaluationExecutor(application)(deepcopy(dict(case.task_input)), case.trace_id)
    assert observed.prompt_release_hash == actual_binding.prompt_release_hash
    assert observed.observation.output["trace_id"] == case.trace_id

    unbound = ProductionWorkflowApplication(lambda: dependencies)
    with pytest.raises(ValueError, match="host-owned evaluation binding") as error:
        WorkflowEvaluationExecutor(unbound)
    assert "host-supplied code revision" in str(error.value.__cause__)
