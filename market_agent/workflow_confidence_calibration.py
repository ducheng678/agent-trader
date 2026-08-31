"""Fail-closed, Decimal-only confidence calibration over host-validated metadata."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from hashlib import sha256
import json
from typing import Any, Literal

from pydantic import Field, StrictBool, field_validator, model_validator

from market_agent.workflow_contracts import ContractModel, Digest, ShortText
from market_agent.workflow_harness_contracts import ProgressTargetSet


SUCCESS_THRESHOLD = Decimal("0.85")
RECOVERY_THRESHOLD = Decimal("0.45")
MAX_RECORDS = 64
FEATURE_ORDER = (
    "required_evidence_coverage",
    "required_source_coverage",
    "conflict_resolution",
)
FeatureName = Literal[
    "required_evidence_coverage",
    "required_source_coverage",
    "conflict_resolution",
]
RequestClass = Literal["informational", "active", "trading"]
NextAction = Literal["succeed", "one_recovery", "safe_retrieval", "degrade_unknown", "degrade_no_trade"]


class CalibrationError(ValueError):
    """A deliberately content-free public validation failure."""


def _decimal(value: object, *, minimum: Decimal | None = None, maximum: Decimal | None = None) -> Decimal:
    if type(value) is not Decimal:
        raise CalibrationError("invalid confidence decimal")
    if not value.is_finite() or (minimum is not None and value < minimum) or (maximum is not None and value > maximum):
        raise CalibrationError("invalid confidence decimal")
    return value


def _unique(values: tuple[str, ...], message: str) -> None:
    if len(values) != len(set(values)):
        raise CalibrationError(message)


def _canonical(value: object) -> object:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, ContractModel):
        return _canonical(value.model_dump(mode="python", round_trip=True))
    if type(value) is tuple:
        return [_canonical(item) for item in value]
    if type(value) is dict:
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if type(value) in {str, int, bool} or value is None:
        return value
    raise CalibrationError("invalid calibration value")


def _digest(value: object) -> str:
    return sha256(json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")).hexdigest()


class _PublicConfidenceContract(ContractModel):
    """Strict public contracts whose copied values are always revalidated."""

    def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False) -> Any:
        values = self.model_dump(mode="python", round_trip=True)
        if update:
            values.update(update)
        return type(self).model_validate(values)


class ConfidenceFeatureSpec(_PublicConfidenceContract):
    feature_name: FeatureName
    coefficient: Decimal
    normalization: Literal["unit_interval"]
    monotonicity: Literal["increasing"]
    missing_value_behavior: Literal["fail_closed"]

    @field_validator("coefficient", mode="before")
    @classmethod
    def validate_coefficient(cls, value: object) -> Decimal:
        return _decimal(value, minimum=Decimal("0"), maximum=Decimal("1"))


class AcceptedEvidenceRecord(_PublicConfidenceContract):
    evidence_id: ShortText
    source_id: ShortText
    required_slot_id: ShortText
    provenance_hash: Digest
    accepted_by_host: Literal[True]


class ConflictRecord(_PublicConfidenceContract):
    conflict_id: ShortText
    evidence_ids: tuple[ShortText, ...] = Field(min_length=1, max_length=MAX_RECORDS)
    resolved: StrictBool
    provenance_hash: Digest

    @model_validator(mode="after")
    def validate_evidence_ids(self) -> ConflictRecord:
        _unique(self.evidence_ids, "invalid conflict record")
        return self


class SourceRegistryRecord(_PublicConfidenceContract):
    source_id: ShortText
    registry_hash: Digest
    enabled: StrictBool


class ConfidenceFoldedState(_PublicConfidenceContract):
    completed_dependency_ids: tuple[ShortText, ...] = Field(max_length=MAX_RECORDS)
    valid_output_field_paths: tuple[ShortText, ...] = Field(max_length=MAX_RECORDS)
    satisfied_risk_invariant_ids: tuple[ShortText, ...] = Field(max_length=MAX_RECORDS)
    event_fold_hash: Digest

    @model_validator(mode="after")
    def validate_unique_values(self) -> ConfidenceFoldedState:
        _unique(self.completed_dependency_ids, "invalid folded state")
        _unique(self.valid_output_field_paths, "invalid folded state")
        _unique(self.satisfied_risk_invariant_ids, "invalid folded state")
        return self


class ConfidenceObservation(_PublicConfidenceContract):
    request_class: RequestClass
    applicability_domain: ShortText
    progress_targets: ProgressTargetSet
    accepted_evidence: tuple[AcceptedEvidenceRecord, ...] = Field(max_length=MAX_RECORDS)
    conflicts: tuple[ConflictRecord, ...] = Field(max_length=MAX_RECORDS)
    source_registry: tuple[SourceRegistryRecord, ...] = Field(max_length=MAX_RECORDS)
    folded_state: ConfidenceFoldedState
    recovery_used: StrictBool
    model_confidence: Decimal | None = None

    @field_validator("model_confidence", mode="before")
    @classmethod
    def validate_ignored_model_confidence(cls, value: object) -> Decimal | None:
        if value is None:
            return None
        return _decimal(value, minimum=Decimal("0"), maximum=Decimal("1"))

    @model_validator(mode="after")
    def validate_metadata_identity(self) -> ConfidenceObservation:
        _unique(tuple(item.evidence_id for item in self.accepted_evidence), "invalid confidence observation")
        _unique(tuple(item.conflict_id for item in self.conflicts), "invalid confidence observation")
        _unique(tuple(item.source_id for item in self.source_registry), "invalid confidence observation")
        ProgressTargetSet.model_validate(self.progress_targets.model_dump(mode="python", round_trip=True))
        return self


class ConfidenceFeatureValue(_PublicConfidenceContract):
    feature_name: FeatureName
    value: Decimal

    @field_validator("value", mode="before")
    @classmethod
    def validate_value(cls, value: object) -> Decimal:
        return _decimal(value, minimum=Decimal("0"), maximum=Decimal("1"))


class ConfidenceFeatureVector(_PublicConfidenceContract):
    artifact_hash: Digest
    features: tuple[ConfidenceFeatureValue, ...] = Field(min_length=1, max_length=len(FEATURE_ORDER))

    @model_validator(mode="after")
    def validate_order(self) -> ConfidenceFeatureVector:
        names = tuple(item.feature_name for item in self.features)
        _unique(names, "invalid confidence feature vector")
        if tuple(sorted(names, key=FEATURE_ORDER.index)) != names:
            raise CalibrationError("invalid confidence feature vector")
        return self


class ConfidenceCalibratorArtifact(_PublicConfidenceContract):
    artifact_version: Literal["v1"]
    schema_hash: Digest
    policy_hash: Digest
    dataset_hash: Digest
    applicability_domains: tuple[ShortText, ...] = Field(min_length=1, max_length=16)
    feature_specs: tuple[ConfidenceFeatureSpec, ...] = Field(min_length=1, max_length=len(FEATURE_ORDER))
    intercept: Decimal
    success_threshold: Decimal
    recovery_threshold: Decimal
    artifact_hash: Digest
    signature: Digest

    @field_validator("intercept", mode="before")
    @classmethod
    def validate_intercept(cls, value: object) -> Decimal:
        return _decimal(value, minimum=Decimal("0"), maximum=Decimal("1"))

    @field_validator("success_threshold", "recovery_threshold", mode="before")
    @classmethod
    def validate_thresholds(cls, value: object) -> Decimal:
        return _decimal(value, minimum=Decimal("0"), maximum=Decimal("1"))

    @classmethod
    def _payload(cls, values: dict[str, object], *, include_artifact_hash: bool) -> dict[str, object]:
        keys = ("schema_version", "artifact_version", "schema_hash", "policy_hash", "dataset_hash", "applicability_domains", "feature_specs", "intercept", "success_threshold", "recovery_threshold")
        payload = {key: values.get(key, "v1" if key == "schema_version" else None) for key in keys}
        if include_artifact_hash:
            payload["artifact_hash"] = values.get("artifact_hash")
        return payload

    @classmethod
    def compute_artifact_hash(cls, **values: object) -> str:
        return _digest(cls._payload(dict(values), include_artifact_hash=False))

    @classmethod
    def compute_signature(cls, **values: object) -> str:
        return _digest(cls._payload(dict(values), include_artifact_hash=True))

    @model_validator(mode="after")
    def validate_artifact(self) -> ConfidenceCalibratorArtifact:
        _unique(self.applicability_domains, "invalid calibration artifact")
        names = tuple(item.feature_name for item in self.feature_specs)
        _unique(names, "invalid calibration artifact")
        if tuple(sorted(names, key=FEATURE_ORDER.index)) != names:
            raise CalibrationError("invalid calibration artifact")
        if self.intercept + sum((item.coefficient for item in self.feature_specs), Decimal("0")) > Decimal("1"):
            raise CalibrationError("invalid calibration artifact")
        if self.success_threshold != SUCCESS_THRESHOLD or self.recovery_threshold != RECOVERY_THRESHOLD:
            raise CalibrationError("invalid calibration artifact")
        values = self.model_dump(mode="python", round_trip=True)
        if self.artifact_hash != self.compute_artifact_hash(**values) or self.signature != self.compute_signature(**values):
            raise CalibrationError("invalid calibration artifact")
        return self


class ConfidenceDecision(_PublicConfidenceContract):
    score: Decimal | None = None
    feature_vector: ConfidenceFeatureVector | None = None
    may_succeed: StrictBool
    next_action: NextAction
    reason_code: Literal["calibrated", "calibration_unavailable"]

    @field_validator("score", mode="before")
    @classmethod
    def validate_score(cls, value: object) -> Decimal | None:
        if value is None:
            return None
        return _decimal(value, minimum=Decimal("0"), maximum=Decimal("1"))

    @model_validator(mode="after")
    def validate_shape(self) -> ConfidenceDecision:
        if self.may_succeed != (self.next_action == "succeed"):
            raise CalibrationError("invalid confidence decision")
        if self.reason_code == "calibrated" and (self.score is None or self.feature_vector is None):
            raise CalibrationError("invalid confidence decision")
        if self.reason_code == "calibration_unavailable" and (self.score is not None or self.feature_vector is not None):
            raise CalibrationError("invalid confidence decision")
        return self


class ConfidenceGate:
    """Pure policy: calibration can only narrow outcomes; it grants no other authority."""

    SUCCESS = SUCCESS_THRESHOLD
    ABSTAIN = RECOVERY_THRESHOLD

    def evaluate(self, observation: object, artifact: object) -> ConfidenceDecision:
        request_class = _request_class(observation)
        try:
            if type(observation) is not ConfidenceObservation or type(artifact) is not ConfidenceCalibratorArtifact:
                raise CalibrationError("invalid confidence input")
            fresh_observation = ConfidenceObservation.model_validate(observation.model_dump(mode="python", round_trip=True))
            fresh_artifact = ConfidenceCalibratorArtifact.model_validate(artifact.model_dump(mode="python", round_trip=True))
            if fresh_observation.applicability_domain not in fresh_artifact.applicability_domains:
                raise CalibrationError("calibration domain mismatch")
            vector = self._compute_features(fresh_observation, fresh_artifact)
            score = fresh_artifact.intercept + sum((spec.coefficient * value.value for spec, value in zip(fresh_artifact.feature_specs, vector.features, strict=True)), Decimal("0"))
            _decimal(score, minimum=Decimal("0"), maximum=Decimal("1"))
            return self.decide(score=score, recovered=fresh_observation.recovery_used, request_class=fresh_observation.request_class, feature_vector=vector)
        except (CalibrationError, InvalidOperation, TypeError, ValueError):
            return _unavailable(request_class)

    def decide(self, *, score: object, recovered: object, request_class: object = "informational", feature_vector: ConfidenceFeatureVector | None = None) -> ConfidenceDecision:
        request = _request_class_value(request_class)
        try:
            calibrated_score = _decimal(score, minimum=Decimal("0"), maximum=Decimal("1"))
            if type(recovered) is not bool:
                raise CalibrationError("invalid recovery state")
            if calibrated_score >= self.SUCCESS:
                return ConfidenceDecision(score=calibrated_score, feature_vector=feature_vector or _empty_vector(), may_succeed=True, next_action="succeed", reason_code="calibrated")
            if calibrated_score >= self.ABSTAIN and not recovered:
                return ConfidenceDecision(score=calibrated_score, feature_vector=feature_vector or _empty_vector(), may_succeed=False, next_action="one_recovery", reason_code="calibrated")
            return ConfidenceDecision(score=calibrated_score, feature_vector=feature_vector or _empty_vector(), may_succeed=False, next_action=_degrade_action(request), reason_code="calibrated")
        except (CalibrationError, InvalidOperation, TypeError, ValueError):
            return _unavailable(request)

    def _compute_features(self, observation: ConfidenceObservation, artifact: ConfidenceCalibratorArtifact) -> ConfidenceFeatureVector:
        targets = observation.progress_targets
        evidence_by_slot = {item.required_slot_id: item for item in observation.accepted_evidence}
        evidence_ids = frozenset(item.evidence_id for item in observation.accepted_evidence)
        registry = {item.source_id: item for item in observation.source_registry}
        conflict_by_id = {item.conflict_id: item for item in observation.conflicts}
        if not set(targets.required_dependency_ids).issubset(observation.folded_state.completed_dependency_ids):
            raise CalibrationError("missing required host metadata")
        if not set(targets.required_output_field_paths).issubset(observation.folded_state.valid_output_field_paths):
            raise CalibrationError("missing required host metadata")
        if not set(targets.risk_invariant_ids).issubset(observation.folded_state.satisfied_risk_invariant_ids):
            raise CalibrationError("missing required host metadata")
        if not set(targets.required_evidence_slot_ids).issubset(evidence_by_slot):
            raise CalibrationError("missing required host metadata")
        required_sources = tuple(source_id for source_id, _ in targets.required_source_coverage_weights)
        if any(source_id not in registry or not registry[source_id].enabled for source_id in required_sources):
            raise CalibrationError("missing required host metadata")
        if any(source_id not in {item.source_id for item in observation.accepted_evidence} for source_id in required_sources):
            raise CalibrationError("missing required host metadata")
        if set(targets.known_conflict_slot_ids) != set(conflict_by_id):
            raise CalibrationError("conflict metadata mismatch")
        if any(not item.resolved or not set(item.evidence_ids).issubset(evidence_ids) for item in observation.conflicts):
            raise CalibrationError("unresolved conflict")
        values = {"required_evidence_coverage": Decimal("1"), "required_source_coverage": Decimal("1"), "conflict_resolution": Decimal("1")}
        return ConfidenceFeatureVector(artifact_hash=artifact.artifact_hash, features=tuple(ConfidenceFeatureValue(feature_name=spec.feature_name, value=values[spec.feature_name]) for spec in artifact.feature_specs))


def _request_class(value: object) -> RequestClass:
    if type(value) is ConfidenceObservation:
        return value.request_class
    return "informational"


def _request_class_value(value: object) -> RequestClass:
    return value if value in {"informational", "active", "trading"} and type(value) is str else "informational"


def _degrade_action(request_class: RequestClass) -> NextAction:
    return "degrade_no_trade" if request_class in {"active", "trading"} else "degrade_unknown"


def _unavailable(request_class: RequestClass) -> ConfidenceDecision:
    return ConfidenceDecision(may_succeed=False, next_action="degrade_no_trade" if request_class in {"active", "trading"} else "safe_retrieval", reason_code="calibration_unavailable")


def _empty_vector() -> ConfidenceFeatureVector:
    return ConfidenceFeatureVector(artifact_hash="0" * 64, features=(ConfidenceFeatureValue(feature_name="required_evidence_coverage", value=Decimal("0")),))
