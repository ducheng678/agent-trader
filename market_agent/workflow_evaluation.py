from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from time import perf_counter
from typing import Literal, Protocol

from pydantic import Field, model_validator

from market_agent.workflow_contracts import Digest, ShortText
from market_agent.workflow_eval_dataset import EvaluationAnswer, EvaluationCase, EvaluationDataset, RecordedObservation
from market_agent.workflow_eval_metrics import AggregateMetrics, CaseScore, PairedComparison, aggregate_scores, compare_scores
from market_agent.workflow_long_term_memory import MemoryContract, canonical_json, content_hash, thaw_json


ExecutionMode = Literal["recorded", "executed"]
class EvaluationBinding(MemoryContract):
    code_revision: ShortText
    prompt_release_hash: Digest
    model_policy_hash: Digest


class ExecutedObservation(MemoryContract):
    observation: RecordedObservation
    prompt_release_hash: Digest


class EvaluationExecutor(Protocol):
    @property
    def attested_binding(self) -> EvaluationBinding: ...

    def __call__(self, task_input: dict[str, object], trace_id: str) -> ExecutedObservation: ...


class EvaluationRun(MemoryContract):
    evaluator_version: Literal["offline-evaluator-v1", "offline-evaluator-v2", "offline-evaluator-v3"] = "offline-evaluator-v3"
    execution_mode: ExecutionMode = "recorded"
    binding_verified: bool = False
    dataset_id: ShortText
    dataset_hash: Digest
    split: str
    binding: EvaluationBinding
    answer_schema_hash: Digest
    scores: tuple[CaseScore, ...] = Field(min_length=1, max_length=10000)
    metrics: AggregateMetrics
    run_hash: Digest | None = None

    @model_validator(mode="after")
    def integrity(self):
        if self.split not in {"train", "development", "holdout"}:
            raise ValueError("evaluation run version or split is unsupported")
        if self.evaluator_version == "offline-evaluator-v1" and self.execution_mode != "recorded":
            raise ValueError("legacy evaluation runs cannot claim executed mode")
        if self.evaluator_version != "offline-evaluator-v3" and self.binding_verified:
            raise ValueError("legacy evaluation runs cannot claim verified execution binding")
        if self.execution_mode == "recorded" and self.binding_verified:
            raise ValueError("recorded evaluation cannot claim verified execution binding")
        if len({score.case_id for score in self.scores}) != len(self.scores) or self.metrics != aggregate_scores(self.scores):
            raise ValueError("evaluation score inventory or aggregation is inconsistent")
        legacy_fields = {"binding_verified"} if self.evaluator_version != "offline-evaluator-v3" else set()
        if self.evaluator_version == "offline-evaluator-v1":
            legacy_fields.add("execution_mode")
        digest = content_hash(self.model_dump(mode="json", exclude={"run_hash", *legacy_fields}))
        if self.run_hash is not None and self.run_hash != digest:
            raise ValueError("evaluation artifact hash mismatch")
        object.__setattr__(self, "run_hash", digest)
        return self


def score_case(case: EvaluationCase, observation: RecordedObservation) -> CaseScore:
    case, observation = EvaluationCase.model_validate(case), RecordedObservation.model_validate(observation)
    output = observation.model_dump(mode="json")["output"]
    violations = set(observation.observed_violations)
    try:
        answer = EvaluationAnswer.model_validate_json(canonical_json(output))
    except ValueError:
        answer = None
        violations.add("invalid_schema")
    expected = case.expected
    schema_ok = answer is not None
    abstention_ok = schema_ok and answer.conclusion == expected.conclusion
    risk_ok = schema_ok and answer.action == expected.action
    evidence_ok = schema_ok and set(expected.required_evidence_ids) <= set(answer.evidence_ids) <= set(case.allowed_evidence_ids)
    trace_ok = schema_ok and answer.trace_id == case.trace_id
    facts_ok = False
    if answer is not None:
        facts = {" ".join(fact.split()).casefold() for fact in answer.facts}
        expected_facts = {" ".join(fact.split()).casefold() for fact in expected.required_facts}
        allowed_facts = {" ".join(fact.split()).casefold() for fact in expected.allowed_facts}
        forbidden = any(fact.casefold() in " ".join(answer.facts).casefold() for fact in expected.forbidden_facts)
        facts_ok = expected_facts <= facts and not forbidden and (not allowed_facts or facts <= allowed_facts)
        if forbidden or (allowed_facts and not facts <= allowed_facts):
            violations.add("unsupported_claim")
        if not set(answer.evidence_ids) <= set(case.allowed_evidence_ids):
            violations.add("evidence_fabrication")
        if not trace_ok:
            violations.add("cross_trace")
        if not risk_ok or (expected.conclusion in {"不知道", "no_trade"} and answer.action != "no_trade"):
            violations.add("risk_bypass")
        if not abstention_ok and expected.conclusion == "不知道":
            violations.add("unsupported_claim")
    tokens = observation.input_tokens + observation.output_tokens
    budget_ok = (observation.latency_seconds <= expected.maximum_latency_seconds and tokens <= expected.maximum_tokens
                 and observation.cost_usd <= expected.maximum_cost_usd)
    success = all((schema_ok, abstention_ok, risk_ok, evidence_ok, trace_ok, facts_ok, budget_ok, not violations))
    return CaseScore(case_id=case.case_id, case_hash=case.case_hash, output_hash=content_hash(output),
        success=success, schema_passed=schema_ok, abstention_passed=abstention_ok, evidence_passed=evidence_ok,
        risk_passed=risk_ok, trace_passed=trace_ok, facts_passed=facts_ok, budget_passed=budget_ok,
        hard_violations=tuple(sorted(violations)), latency_seconds=observation.latency_seconds,
        tokens=tokens, cost_usd=observation.cost_usd)


class EvaluationRunner:
    def run(self, dataset: EvaluationDataset, *, binding: EvaluationBinding,
            recordings: Mapping[str, RecordedObservation] | None = None,
            execution_mode: ExecutionMode = "recorded",
            executor: EvaluationExecutor | None = None) -> EvaluationRun:
        dataset = dataset.validate()
        binding = EvaluationBinding.model_validate(binding)
        if execution_mode not in {"recorded", "executed"}:
            raise ValueError("evaluation execution mode is unsupported")
        if execution_mode == "executed":
            if executor is None or not callable(executor) or recordings is not None:
                raise ValueError("executed evaluation requires an executor and cannot use recordings")
            try:
                host_binding = EvaluationBinding.model_validate(executor.attested_binding)
            except (AttributeError, TypeError, ValueError) as error:
                raise ValueError("executed evaluation requires an attested host binding") from error
            if host_binding != binding:
                raise ValueError("executed evaluation binding does not match the current host")
        elif executor is not None:
            raise ValueError("recorded evaluation cannot receive a workflow executor")
        if recordings is not None and set(recordings) != {case.case_id for case in dataset.cases}:
            raise ValueError("offline recordings must cover exactly the dataset cases")
        if execution_mode == "executed":
            assert executor is not None
            scored = []
            binding_verified = True
            for case in sorted(dataset.cases, key=lambda case: case.case_id):
                if EvaluationBinding.model_validate(executor.attested_binding) != binding:
                    raise ValueError("executed host binding changed during evaluation")
                observation, observed_prompt = self._execute_case(case, executor)
                if EvaluationBinding.model_validate(executor.attested_binding) != binding:
                    raise ValueError("executed host binding changed during evaluation")
                if observed_prompt is None:
                    binding_verified = False
                elif observed_prompt != binding.prompt_release_hash:
                    raise ValueError("executed prompt release does not match the attested binding")
                scored.append(score_case(case, observation))
            scores = tuple(scored)
        else:
            binding_verified = False
            scores = tuple(score_case(case, case.recording if recordings is None else recordings[case.case_id])
                           for case in sorted(dataset.cases, key=lambda case: case.case_id))
        return EvaluationRun(dataset_id=dataset.manifest.dataset_id, dataset_hash=dataset.dataset_hash,
            split=dataset.manifest.split, binding=binding, execution_mode=execution_mode,
            binding_verified=binding_verified,
            answer_schema_hash=content_hash(EvaluationAnswer.model_json_schema()), scores=scores, metrics=aggregate_scores(scores))

    @staticmethod
    def _execute_case(case: EvaluationCase, executor: EvaluationExecutor) -> tuple[RecordedObservation, str | None]:
        started = perf_counter()
        try:
            # Expectations, fixture observations and scoring metadata never enter
            # the workflow. The copy also prevents a host from mutating the dataset.
            executed = ExecutedObservation.model_validate(executor(thaw_json(case.task_input), case.trace_id))
            return executed.observation, executed.prompt_release_hash
        except Exception:
            return RecordedObservation(
                output={"error": "executor_failure", "trace_id": case.trace_id},
                latency_seconds=max(0.0, perf_counter() - started),
                input_tokens=0, output_tokens=0, cost_usd=0.0,
                observed_violations=("invalid_schema",),
            ), None

    def compare(self, candidate: EvaluationRun, baseline: EvaluationRun) -> PairedComparison:
        candidate, baseline = EvaluationRun.model_validate(candidate), EvaluationRun.model_validate(baseline)
        if (candidate.dataset_hash != baseline.dataset_hash or candidate.answer_schema_hash != baseline.answer_schema_hash
                or candidate.execution_mode != baseline.execution_mode or candidate.evaluator_version != baseline.evaluator_version):
            raise ValueError("evaluation comparison requires the same mode, dataset, evaluator and answer schema")
        return compare_scores(candidate.scores, baseline.scores)

    @staticmethod
    def write_artifact(run: EvaluationRun, directory: str | Path) -> Path:
        run = EvaluationRun.model_validate(run)
        directory = Path(directory).resolve()
        directory.mkdir(parents=True, exist_ok=True)
        target = directory / (run.run_hash + ".json")
        data = (canonical_json(run.model_dump(mode="json")) + "\n").encode("utf-8")
        try:
            with target.open("xb") as output:
                output.write(data)
        except FileExistsError:
            if target.read_bytes() != data:
                raise ValueError("evaluation artifact address already contains different bytes") from None
        return target
