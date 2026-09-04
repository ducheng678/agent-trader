"""Executable offline evaluation and release-quality gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence

from market_agent.workflow_evaluation import EvaluationBinding, EvaluationRun, EvaluationRunner
from market_agent.workflow_eval_dataset import EvaluationDataset
from market_agent.workflow_eval_metrics import ReleaseGate, ReleaseThresholds
from market_agent.workflow_long_term_memory import canonical_json


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the versioned Market Agent offline evaluation gate")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--prompt-release-hash", required=True)
    parser.add_argument("--model-policy-hash", required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("runtime/evaluations"))
    parser.add_argument("--allow-missing-baseline", action="store_true")
    return parser


def run(argv: Sequence[str] | None = None) -> tuple[int, dict[str, object]]:
    args = _parser().parse_args(argv)
    try:
        dataset = EvaluationDataset.load(args.manifest)
        binding = EvaluationBinding(
            code_revision=args.code_revision,
            prompt_release_hash=args.prompt_release_hash,
            model_policy_hash=args.model_policy_hash,
        )
        runner = EvaluationRunner()
        candidate = runner.run(dataset, binding=binding)
        baseline = None
        if args.baseline is not None:
            baseline = EvaluationRun.model_validate_json(args.baseline.read_text(encoding="utf-8"))
            if baseline.dataset_hash != candidate.dataset_hash:
                raise ValueError("baseline artifact uses a different dataset")
        thresholds = ReleaseThresholds(
            require_baseline=not args.allow_missing_baseline,
            minimum_cases=max(1, min(10000, len(dataset.cases))),
        )
        decision = ReleaseGate(thresholds).evaluate(candidate, baseline)
        artifact = runner.write_artifact(candidate, args.output_dir)
        report = {
            "allowed": decision.allowed,
            "reasons": list(decision.reasons),
            "run_hash": candidate.run_hash,
            "artifact": str(artifact),
            "dataset_id": dataset.manifest.dataset_id,
            "dataset_hash": dataset.dataset_hash,
            "metrics": candidate.metrics.model_dump(mode="json"),
        }
        return (0 if decision.allowed else 1), report
    except Exception as error:
        return 2, {"allowed": False, "error": type(error).__name__, "message": str(error)}


def main(argv: Sequence[str] | None = None) -> int:
    code, report = run(argv)
    sys.stdout.write(canonical_json(report) + "\n")
    return code


if __name__ == "__main__":  # pragma: no cover - exercised by the module CLI
    raise SystemExit(main())
