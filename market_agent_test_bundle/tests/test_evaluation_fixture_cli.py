from __future__ import annotations

import market_agent.evaluation_fixture_cli as fixture_cli
from market_agent.workflow_eval_dataset import RecordedObservation


MANIFEST = "evals/datasets/offline-safety-v1.manifest.json"


def test_recorded_fixture_check_passes_without_claiming_release():
    code, report = fixture_cli.run(["--manifest", MANIFEST])

    assert code == 0
    assert report["check"] == "recorded-fixtures-v1"
    assert report["release_eligible"] is False
    assert report["passed"] is True
    assert report["failed_cases"] == []
    assert report["metrics"]["success_rate"] == 1.0
    assert "allowed" not in report


def test_recorded_fixture_check_detects_scored_regression(monkeypatch):
    original = fixture_cli.score_case

    def changed_observation(case, observation):
        if case.case_id == "seed-regression":
            payload = observation.model_dump(mode="json")
            payload["output"] = {
                "conclusion": "no_trade",
                "action": "no_trade",
                "facts": [],
                "evidence_ids": [],
                "trace_id": case.trace_id,
            }
            payload["observed_violations"] = tuple(payload["observed_violations"])
            observation = RecordedObservation.model_validate(payload)
        return original(case, observation)

    monkeypatch.setattr(fixture_cli, "score_case", changed_observation)
    code, report = fixture_cli.run(["--manifest", MANIFEST])

    assert code == 1, report.get("message")
    assert report["release_eligible"] is False
    assert report["passed"] is False
    assert report["failed_cases"] == ["seed-regression"]


def test_recorded_fixture_check_rejects_invalid_manifest(tmp_path):
    code, report = fixture_cli.run(["--manifest", str(tmp_path / "missing.json")])

    assert code == 2
    assert report["passed"] is False
    assert report["release_eligible"] is False
