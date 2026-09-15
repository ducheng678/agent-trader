"""Check committed offline recordings without claiming a release evaluation.

This check exercises dataset integrity and the current scoring code. It does
not execute the agent workflow; only an executed, host-attested evaluation may
be submitted to the release gate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from market_agent.workflow_eval_dataset import EvaluationDataset
from market_agent.workflow_eval_metrics import aggregate_scores
from market_agent.workflow_evaluation import score_case
from market_agent.workflow_long_term_memory import canonical_json


def run(argv: Sequence[str] | None = None) -> tuple[int, dict[str, object]]:
    parser = argparse.ArgumentParser(description="Verify versioned recorded evaluation fixtures")
    parser.add_argument("--manifest", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        dataset = EvaluationDataset.load(args.manifest)
        scores = tuple(score_case(case, case.recording)
                       for case in sorted(dataset.cases, key=lambda item: item.case_id))
        metrics = aggregate_scores(scores)
        failed_cases = [score.case_id for score in scores if not score.success]
        report: dict[str, object] = {
            "check": "recorded-fixtures-v1",
            "release_eligible": False,
            "passed": not failed_cases,
            "dataset_id": dataset.manifest.dataset_id,
            "dataset_hash": dataset.dataset_hash,
            "failed_cases": failed_cases,
            "metrics": metrics.model_dump(mode="json"),
        }
        return (1 if failed_cases else 0), report
    except Exception as error:
        return 2, {
            "check": "recorded-fixtures-v1",
            "release_eligible": False,
            "passed": False,
            "error": type(error).__name__,
            "message": str(error),
        }


def main(argv: Sequence[str] | None = None) -> int:
    code, report = run(argv)
    sys.stdout.write(canonical_json(report) + "\n")
    return code


if __name__ == "__main__":  # pragma: no cover - exercised by the module CLI
    raise SystemExit(main())
