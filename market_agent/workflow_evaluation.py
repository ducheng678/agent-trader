from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from pydantic import Field, model_validator

from market_agent.workflow_contracts import Digest, ShortText
from market_agent.workflow_eval_dataset import EvaluationAnswer, EvaluationCase, EvaluationDataset, RecordedObservation
from market_agent.workflow_eval_metrics import AggregateMetrics, CaseScore, PairedComparison, aggregate_scores, compare_scores
from market_agent.workflow_long_term_memory import MemoryContract, canonical_json, content_hash


class EvaluationBinding(MemoryContract):
    code_revision: ShortText
    prompt_release_hash: Digest
    model_policy_hash: Digest


class EvaluationRun(MemoryContract):
    evaluator_version: str = "offline-evaluator-v1"
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
        if self.evaluator_version != "offline-evaluator-v1" or self.split not in {"train", "development", "holdout"}:
            raise ValueError("evaluation run version or split is unsupported")
        if len({score.case_id for score in self.scores}) != len(self.scores) or self.metrics != aggregate_scores(self.scores):
            raise ValueError("evaluation score inventory or aggregation is inconsistent")
        digest = content_hash(self.model_dump(mode="json", exclude={"run_hash"}))
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
            recordings: Mapping[str, RecordedObservation] | None = None) -> EvaluationRun:
        dataset = dataset.validate()
        binding = EvaluationBinding.model_validate(binding)
        if recordings is not None and set(recordings) != {case.case_id for case in dataset.cases}:
            raise ValueError("offline recordings must cover exactly the dataset cases")
        scores = tuple(score_case(case, case.recording if recordings is None else recordings[case.case_id])
                       for case in sorted(dataset.cases, key=lambda case: case.case_id))
        return EvaluationRun(dataset_id=dataset.manifest.dataset_id, dataset_hash=dataset.dataset_hash,
            split=dataset.manifest.split, binding=binding,
            answer_schema_hash=content_hash(EvaluationAnswer.model_json_schema()), scores=scores, metrics=aggregate_scores(scores))

    def compare(self, candidate: EvaluationRun, baseline: EvaluationRun) -> PairedComparison:
        candidate, baseline = EvaluationRun.model_validate(candidate), EvaluationRun.model_validate(baseline)
        if candidate.dataset_hash != baseline.dataset_hash or candidate.answer_schema_hash != baseline.answer_schema_hash:
            raise ValueError("offline comparison requires the same dataset and answer schema")
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
