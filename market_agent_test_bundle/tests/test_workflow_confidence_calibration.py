from __future__ import annotations

from decimal import Decimal

import pytest
from pydantic import ValidationError

from market_agent.workflow_confidence_calibration import (
    AcceptedEvidenceRecord, ConfidenceCalibratorArtifact, ConfidenceFeatureSpec,
    ConfidenceFoldedState, ConfidenceGate, ConfidenceObservation, ConflictRecord,
    SourceRegistryRecord,
)
from market_agent.workflow_harness_contracts import ProgressTargetSet

HASH = "a" * 64


def progress_targets() -> ProgressTargetSet:
    return ProgressTargetSet(required_dependency_ids=("collect",), required_output_field_paths=("result.summary",), required_evidence_slot_ids=("primary-source",), required_source_coverage_weights=(("official-feed", 1.0),), known_conflict_slot_ids=("claim-1",), risk_invariant_ids=("no-unknown-side-effect",))


def feature_specs() -> tuple[ConfidenceFeatureSpec, ...]:
    return (
        ConfidenceFeatureSpec(feature_name="required_evidence_coverage", coefficient=Decimal("0.40"), normalization="unit_interval", monotonicity="increasing", missing_value_behavior="fail_closed"),
        ConfidenceFeatureSpec(feature_name="required_source_coverage", coefficient=Decimal("0.30"), normalization="unit_interval", monotonicity="increasing", missing_value_behavior="fail_closed"),
        ConfidenceFeatureSpec(feature_name="conflict_resolution", coefficient=Decimal("0.15"), normalization="unit_interval", monotonicity="increasing", missing_value_behavior="fail_closed"),
    )


def artifact(**overrides: object) -> ConfidenceCalibratorArtifact:
    values: dict[str, object] = {"artifact_version": "v1", "schema_hash": HASH, "policy_hash": "b" * 64, "dataset_hash": "c" * 64, "applicability_domains": ("market-analysis",), "feature_specs": feature_specs(), "intercept": Decimal("0.00"), "success_threshold": Decimal("0.85"), "recovery_threshold": Decimal("0.45")}
    values.update(overrides)
    values["artifact_hash"] = ConfidenceCalibratorArtifact.compute_artifact_hash(**values)
    values["signature"] = ConfidenceCalibratorArtifact.compute_signature(**values)
    return ConfidenceCalibratorArtifact(**values)


def observation(**overrides: object) -> ConfidenceObservation:
    values: dict[str, object] = {"request_class": "informational", "applicability_domain": "market-analysis", "progress_targets": progress_targets(), "accepted_evidence": (AcceptedEvidenceRecord(evidence_id="evidence-1", source_id="official-feed", required_slot_id="primary-source", provenance_hash="d" * 64, accepted_by_host=True),), "conflicts": (ConflictRecord(conflict_id="claim-1", evidence_ids=("evidence-1",), resolved=True, provenance_hash="e" * 64),), "source_registry": (SourceRegistryRecord(source_id="official-feed", registry_hash="f" * 64, enabled=True),), "folded_state": ConfidenceFoldedState(completed_dependency_ids=("collect",), valid_output_field_paths=("result.summary",), satisfied_risk_invariant_ids=("no-unknown-side-effect",), event_fold_hash="1" * 64), "recovery_used": False, "model_confidence": Decimal("0.01")}
    values.update(overrides)
    return ConfidenceObservation(**values)


@pytest.fixture
def gate() -> ConfidenceGate:
    return ConfidenceGate()


def test_model_self_confidence_never_changes_harness_score(gate: ConfidenceGate):
    low = gate.evaluate(observation(model_confidence=Decimal("0.01")), artifact())
    high = gate.evaluate(observation(model_confidence=Decimal("0.99")), artifact())
    assert low.score == Decimal("0.85")
    assert high.score == Decimal("0.85")
    assert low.feature_vector == high.feature_vector


def test_missing_or_out_of_domain_artifact_fails_closed(gate: ConfidenceGate):
    decision = gate.evaluate(observation(), artifact(applicability_domains=("other-domain",)))
    assert not decision.may_succeed
    assert decision.next_action == "safe_retrieval"
    assert decision.reason_code == "calibration_unavailable"


def test_initial_thresholds_are_exact(gate: ConfidenceGate):
    assert gate.decide(score=Decimal("0.85"), recovered=False).may_succeed
    assert gate.decide(score=Decimal("0.45"), recovered=False).next_action == "one_recovery"
    assert gate.decide(score=Decimal("0.4499"), recovered=False).next_action == "degrade_unknown"


def test_recovered_non_success_cannot_request_another_recovery(gate: ConfidenceGate):
    decision = gate.decide(score=Decimal("0.84"), recovered=True)
    assert not decision.may_succeed
    assert decision.next_action == "degrade_unknown"


def test_active_request_failures_degrade_to_no_trade(gate: ConfidenceGate):
    decision = gate.evaluate(observation(request_class="trading", accepted_evidence=()), artifact())
    assert not decision.may_succeed
    assert decision.next_action == "degrade_no_trade"


def test_valid_host_metadata_produces_hand_derived_decimal_feature_vector(gate: ConfidenceGate):
    decision = gate.evaluate(observation(), artifact())
    assert tuple((item.feature_name, item.value) for item in decision.feature_vector.features) == (("required_evidence_coverage", Decimal("1")), ("required_source_coverage", Decimal("1")), ("conflict_resolution", Decimal("1")))
    assert decision.score == Decimal("0.85")
    assert decision.may_succeed
    assert decision.next_action == "succeed"


@pytest.mark.parametrize("observation_overrides", [{"accepted_evidence": ()}, {"conflicts": ()}, {"folded_state": ConfidenceFoldedState(completed_dependency_ids=(), valid_output_field_paths=("result.summary",), satisfied_risk_invariant_ids=("no-unknown-side-effect",), event_fold_hash="1" * 64)}])
def test_missing_required_host_validated_values_fail_closed(gate: ConfidenceGate, observation_overrides: dict[str, object]):
    decision = gate.evaluate(observation(**observation_overrides), artifact())
    assert not decision.may_succeed
    assert decision.next_action == "safe_retrieval"
    assert decision.reason_code == "calibration_unavailable"


def test_unresolved_conflicts_fail_closed(gate: ConfidenceGate):
    unresolved = ConflictRecord(conflict_id="claim-1", evidence_ids=("evidence-1",), resolved=False, provenance_hash="e" * 64)
    decision = gate.evaluate(observation(conflicts=(unresolved,)), artifact())
    assert not decision.may_succeed
    assert decision.next_action == "safe_retrieval"


def test_artifact_hash_or_signature_tampering_fails_closed(gate: ConfidenceGate):
    valid = artifact()
    unsafe = ConfidenceCalibratorArtifact.model_construct(**{**valid.__dict__, "artifact_hash": "0" * 64})
    decision = gate.evaluate(observation(), unsafe)
    assert not decision.may_succeed
    assert decision.reason_code == "calibration_unavailable"


def test_public_contracts_are_frozen_strict_and_revalidate_model_copies():
    item = feature_specs()[0]
    with pytest.raises(ValidationError):
        item.model_copy(update={"coefficient": 0.4})
    with pytest.raises(ValidationError):
        observation().model_copy(update={"request_class": "forged"})


def test_feature_specs_reject_unknown_duplicate_out_of_order_and_float_values():
    with pytest.raises(ValidationError):
        ConfidenceFeatureSpec(feature_name="unknown", coefficient=Decimal("0.1"), normalization="unit_interval", monotonicity="increasing", missing_value_behavior="fail_closed")
    with pytest.raises(ValidationError):
        artifact(feature_specs=(feature_specs()[1], feature_specs()[0], feature_specs()[2]))
    with pytest.raises(ValidationError):
        ConfidenceFeatureSpec(feature_name="required_evidence_coverage", coefficient=0.4, normalization="unit_interval", monotonicity="increasing", missing_value_behavior="fail_closed")
