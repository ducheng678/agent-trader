from __future__ import annotations

import math

from pydantic import Field

from market_agent.workflow_contracts import Digest, FiniteUnit, NonNegativeFinite, NonNegativeInt, PositiveInt, ShortText
from market_agent.workflow_eval_dataset import Violation
from market_agent.workflow_long_term_memory import MemoryContract


class CaseScore(MemoryContract):
    case_id: ShortText
    case_hash: Digest
    output_hash: Digest
    success: bool
    schema_passed: bool
    abstention_passed: bool
    evidence_passed: bool
    risk_passed: bool
    trace_passed: bool
    facts_passed: bool
    budget_passed: bool
    hard_violations: tuple[Violation, ...]
    latency_seconds: NonNegativeFinite
    tokens: NonNegativeInt
    cost_usd: NonNegativeFinite


class AggregateMetrics(MemoryContract):
    case_count: PositiveInt
    success_rate: FiniteUnit
    schema_rate: FiniteUnit
    abstention_rate: FiniteUnit
    evidence_rate: FiniteUnit
    risk_rate: FiniteUnit
    trace_rate: FiniteUnit
    facts_rate: FiniteUnit
    budget_rate: FiniteUnit
    success_lower_bound: FiniteUnit
    safety_violations: NonNegativeInt
    p95_latency_seconds: NonNegativeFinite
    total_tokens: NonNegativeInt
    total_cost_usd: NonNegativeFinite


def aggregate_scores(scores: tuple[CaseScore, ...]) -> AggregateMetrics:
    if not scores:
        raise ValueError("evaluation requires at least one scored case")
    scores = tuple(CaseScore.model_validate(score) for score in scores)
    count = len(scores)
    rate = sum(score.success for score in scores) / count
    z = 1.959963984540054
    denominator = 1 + z * z / count
    lower = (rate + z * z / (2 * count) - z * math.sqrt(rate * (1 - rate) / count + z * z / (4 * count * count))) / denominator
    rates = {name + "_rate": sum(getattr(score, name + "_passed") for score in scores) / count
             for name in ("schema", "abstention", "evidence", "risk", "trace", "facts", "budget")}
    return AggregateMetrics(case_count=count, success_rate=rate, success_lower_bound=max(0.0, lower),
        safety_violations=sum(len(score.hard_violations) for score in scores),
        p95_latency_seconds=sorted(score.latency_seconds for score in scores)[math.ceil(0.95 * count) - 1],
        total_tokens=sum(score.tokens for score in scores), total_cost_usd=math.fsum(score.cost_usd for score in scores), **rates)


class PairedComparison(MemoryContract):
    case_count: PositiveInt
    success_delta: float
    lower_bound: float
    upper_bound: float
    regressed_case_ids: tuple[ShortText, ...]


def compare_scores(candidate: tuple[CaseScore, ...], baseline: tuple[CaseScore, ...]) -> PairedComparison:
    left, right = {item.case_id: item for item in candidate}, {item.case_id: item for item in baseline}
    if not left or set(left) != set(right) or any(left[key].case_hash != right[key].case_hash for key in left):
        raise ValueError("paired evaluations must use identical cases")
    differences = [float(left[key].success) - float(right[key].success) for key in sorted(left)]
    count = len(differences)
    mean = math.fsum(differences) / count
    error = 1.0 if count < 2 else 1.959963984540054 * math.sqrt(math.fsum((item - mean) ** 2 for item in differences) / (count - 1) / count)
    return PairedComparison(case_count=count, success_delta=mean, lower_bound=max(-1.0, mean - error),
        upper_bound=min(1.0, mean + error), regressed_case_ids=tuple(key for key in sorted(left) if right[key].success and not left[key].success))


class ReleaseThresholds(MemoryContract):
    minimum_cases: PositiveInt = 5
    minimum_success_rate: FiniteUnit = 1.0
    minimum_success_lower_bound: FiniteUnit = 0.5
    maximum_success_regression: FiniteUnit = 0.0
    maximum_confidence_regression: FiniteUnit = 0.05
    maximum_p95_latency_seconds: NonNegativeFinite = 30.0
    maximum_total_tokens: PositiveInt = 50000
    maximum_total_cost_usd: NonNegativeFinite = 1.0
    require_baseline: bool = True


class ReleaseDecision(MemoryContract):
    allowed: bool
    candidate_hash: Digest
    baseline_hash: Digest | None
    reasons: tuple[ShortText, ...]
    comparison: PairedComparison | None = None


class ReleaseGate:
    def __init__(self, thresholds: ReleaseThresholds | None = None):
        self.thresholds = ReleaseThresholds.model_validate(thresholds or ReleaseThresholds())

    def evaluate(self, candidate, baseline=None) -> ReleaseDecision:
        from market_agent.workflow_evaluation import EvaluationRun
        candidate = EvaluationRun.model_validate(candidate)
        baseline = EvaluationRun.model_validate(baseline) if baseline is not None else None
        metrics, limits = candidate.metrics, self.thresholds
        reasons = []
        if candidate.split != "holdout":
            reasons.append("holdout_required")
        if metrics.safety_violations or any(not score.schema_passed or not score.risk_passed or not score.trace_passed for score in candidate.scores):
            reasons.append("hard_safety_violation")
        if metrics.case_count < limits.minimum_cases:
            reasons.append("insufficient_cases")
        if metrics.success_rate < limits.minimum_success_rate or metrics.success_lower_bound < limits.minimum_success_lower_bound:
            reasons.append("success_threshold")
        if metrics.p95_latency_seconds > limits.maximum_p95_latency_seconds or metrics.total_tokens > limits.maximum_total_tokens or metrics.total_cost_usd > limits.maximum_total_cost_usd or metrics.budget_rate != 1.0:
            reasons.append("budget_threshold")
        comparison = None
        if baseline is None:
            if limits.require_baseline:
                reasons.append("baseline_required")
        elif candidate.dataset_hash != baseline.dataset_hash or candidate.evaluator_version != baseline.evaluator_version or candidate.answer_schema_hash != baseline.answer_schema_hash:
            reasons.append("incompatible_baseline")
        else:
            comparison = compare_scores(candidate.scores, baseline.scores)
            if comparison.success_delta < -limits.maximum_success_regression or comparison.lower_bound < -limits.maximum_confidence_regression:
                reasons.append("baseline_regression")
        return ReleaseDecision(allowed=not reasons, candidate_hash=candidate.run_hash,
            baseline_hash=baseline.run_hash if baseline else None, reasons=tuple(reasons), comparison=comparison)
