"""Host-authorized, fail-closed Decimal confidence policy."""
from __future__ import annotations
from decimal import Decimal, Context, ROUND_FLOOR, localcontext, Inexact, Rounded, Overflow, Underflow
from hashlib import sha256
import json
from typing import Any, Literal, Protocol
from pydantic import Field, StrictBool, field_validator, model_validator
from market_agent.workflow_contracts import ContractModel, Digest, ShortText

FEATURE_ORDER=("required_evidence_coverage","required_source_coverage","conflict_resolution")
SUCCESS=Decimal("0.85"); RECOVERY=Decimal("0.45"); _CTX=Context(prec=34,rounding=ROUND_FLOOR); _CTX.traps[Inexact]=True; _CTX.traps[Rounded]=True; _CTX.traps[Overflow]=True; _CTX.traps[Underflow]=True
class CalibrationError(ValueError): pass

def _dec(v:object)->Decimal:
    if type(v) is not Decimal or not v.is_finite() or len(v.as_tuple().digits)>28 or v.adjusted()<-18 or v.adjusted()>0 or v.as_tuple().exponent < -18 or v.as_tuple().exponent > 0: raise CalibrationError("invalid confidence decimal")
    with localcontext(_CTX): return +v

def _canon(v:object)->object:
    if isinstance(v,Decimal): return format(v,"f")
    if isinstance(v,ContractModel): return _canon(v.model_dump(mode="python",round_trip=True))
    if type(v) in {tuple, list}:return [_canon(x) for x in v]
    if type(v) is dict:return {str(k):_canon(x) for k,x in sorted(v.items())}
    if type(v) in {str,int,bool} or v is None:return v
    raise CalibrationError("invalid confidence value")
def artifact_payload(a:object)->bytes:
    if not isinstance(a,ConfidenceCalibratorArtifact): raise CalibrationError("invalid calibration artifact")
    d=a.model_dump(mode="python",round_trip=True,exclude={"signature"})
    return json.dumps(_canon(d),sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode()
def _snapshot_hash(value: object) -> str:
    return sha256(json.dumps(_canon(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()).hexdigest()

def confidence_snapshot_hashes(observation: "ConfidenceObservation") -> tuple[str, str]:
    accepted = _snapshot_hash({"evidence": tuple(sorted((_canon(item) for item in observation.accepted_evidence), key=str)), "sources": tuple(sorted((_canon(item) for item in observation.source_registry), key=str)), "folded": _canon(observation.folded_state)})
    provenance = _snapshot_hash({"evidence": tuple(sorted(item.provenance_hash for item in observation.accepted_evidence)), "conflicts": tuple(sorted((item.conflict_id, item.resolved, tuple(sorted(item.evidence_ids)), item.provenance_hash) for item in observation.conflicts)), "sources": tuple(sorted(item.registry_hash for item in observation.source_registry))})
    return accepted, provenance
class ArtifactSignatureVerifier(Protocol):
    def verify(self,key_id:str,payload:bytes,signature:str)->bool: ...
class _Public(ContractModel):
    def model_copy(self,*,update:dict[str,Any]|None=None,deep:bool=False)->Any:
        v=self.model_dump(mode="python",round_trip=True); v.update(update or {}); return type(self).model_validate(v)
class ConfidenceFeatureSpec(_Public):
    feature_name:Literal["required_evidence_coverage","required_source_coverage","conflict_resolution"]
    coefficient:Decimal
    normalization:Literal["unit_interval"]="unit_interval"
    monotonicity:Literal["increasing"]="increasing"
    missing_value_behavior:Literal["fail_closed"]="fail_closed"
    @field_validator("coefficient",mode="before")
    @classmethod
    def vd(cls,v:object)->Decimal:
        v=_dec(v)
        if not Decimal(0)<=v<=Decimal(1):raise CalibrationError("invalid confidence artifact")
        return v
class AcceptedEvidenceRecord(_Public):
    evidence_id:ShortText; source_id:ShortText; required_slot_id:ShortText; provenance_hash:Digest; accepted_by_host:Literal[True]
class ConflictRecord(_Public):
    conflict_id:ShortText; evidence_ids:tuple[ShortText,...]=Field(min_length=1,max_length=64); resolved:StrictBool; provenance_hash:Digest
    @model_validator(mode="after")
    def unique_evidence_ids(self):
        if len(self.evidence_ids) != len(set(self.evidence_ids)): raise CalibrationError("invalid conflict")
        return self
class SourceRegistryRecord(_Public): source_id:ShortText; registry_hash:Digest; enabled:StrictBool
class ConfidenceFoldedState(_Public):
    completed_dependency_ids:tuple[ShortText,...]=Field(max_length=64); valid_output_field_paths:tuple[ShortText,...]=Field(max_length=64); satisfied_risk_invariant_ids:tuple[ShortText,...]=Field(max_length=64); event_fold_hash:Digest
class ConfidenceTargetSnapshot(_Public):
    required_dependency_ids:tuple[ShortText,...]=Field(max_length=64); required_output_field_paths:tuple[ShortText,...]=Field(max_length=64); required_evidence_slot_ids:tuple[ShortText,...]=Field(max_length=64); required_source_ids:tuple[ShortText,...]=Field(max_length=64); known_conflict_slot_ids:tuple[ShortText,...]=Field(max_length=64); risk_invariant_ids:tuple[ShortText,...]=Field(max_length=64)
    @model_validator(mode="after")
    def u(self):
        for x in self.model_dump(exclude={"schema_version"}).values():
            if len(x)!=len(set(x)):raise CalibrationError("invalid confidence targets")
        return self
class ConfidenceObservation(_Public):
    applicability_domain:ShortText; targets:ConfidenceTargetSnapshot; accepted_evidence:tuple[AcceptedEvidenceRecord,...]=Field(max_length=64); conflicts:tuple[ConflictRecord,...]=Field(max_length=64); source_registry:tuple[SourceRegistryRecord,...]=Field(max_length=64); folded_state:ConfidenceFoldedState; accepted_record_snapshot_hash:Digest; provenance_snapshot_hash:Digest; model_confidence:Decimal|None=None
    @field_validator("model_confidence",mode="before")
    @classmethod
    def mi(cls,v):return None if v is None else _dec(v)
class ConfidenceCalibratorArtifact(_Public):
    artifact_id:ShortText; artifact_version:Literal["v1"]; schema_hash:Digest; policy_hash:Digest; dataset_hash:Digest; applicability_domains:tuple[ShortText,...]=Field(min_length=1,max_length=16); feature_specs:tuple[ConfidenceFeatureSpec,...]=Field(min_length=3,max_length=3); intercept:Decimal; issued_epoch:int; expires_epoch:int; key_id:ShortText; artifact_hash:Digest; signature:Digest
    @field_validator("intercept",mode="before")
    @classmethod
    def vi(cls,v):return _dec(v)
    @model_validator(mode="after")
    def va(self):
        n=tuple(x.feature_name for x in self.feature_specs)
        with localcontext(_CTX):
            total = self.intercept + sum((item.coefficient for item in self.feature_specs), Decimal(0))
        if n!=FEATURE_ORDER or self.issued_epoch<0 or self.expires_epoch<=self.issued_epoch or total > Decimal(1):raise CalibrationError("invalid calibration artifact")
        return self
class TrustedConfidencePolicy(_Public):
    artifact_id:ShortText; artifact_version:Literal["v1"]; artifact_hash:Digest; schema_hash:Digest; policy_hash:Digest; dataset_hash:Digest; key_id:ShortText; applicability_domain:ShortText; accepted_record_snapshot_hash:Digest; provenance_snapshot_hash:Digest; feature_order:tuple[Literal["required_evidence_coverage","required_source_coverage","conflict_resolution"],...]=FEATURE_ORDER; issued_epoch:int; expires_epoch:int
    @model_validator(mode="after")
    def vp(self):
        if self.feature_order!=FEATURE_ORDER or self.issued_epoch<0 or self.expires_epoch<=self.issued_epoch:raise CalibrationError("invalid trusted confidence policy")
        return self
class HardGateSnapshot(_Public):
    permission:StrictBool; risk:StrictBool; budget:StrictBool; loop:StrictBool; evidence:StrictBool; audit_integrity:StrictBool; run_id:ShortText; trace_hash:Digest; plan_revision:int; policy_hash:Digest
    @model_validator(mode="after")
    def vg(self):
        if self.plan_revision<0:raise CalibrationError("invalid hard gate snapshot")
        return self
class TrustedRequestContext(_Public):
    request_class:Literal["informational","active","trading"]; evaluation_epoch:int; recovery_used:StrictBool; hard_gates:HardGateSnapshot
    @model_validator(mode="after")
    def vc(self):
        if self.evaluation_epoch<0:raise CalibrationError("invalid request context")
        return self
class ConfidenceFeatureValue(_Public):
    feature_name:Literal["required_evidence_coverage","required_source_coverage","conflict_resolution"]; value:Decimal
    @field_validator("value",mode="before")
    @classmethod
    def vv(cls,v):
        v=_dec(v)
        if not 0<=v<=1:raise CalibrationError("invalid feature value")
        return v
class ConfidenceFeatureVector(_Public):
    artifact_hash:Digest; features:tuple[ConfidenceFeatureValue,...]=Field(min_length=3,max_length=3)
    @model_validator(mode="after")
    def vf(self):
        if tuple(x.feature_name for x in self.features)!=FEATURE_ORDER:raise CalibrationError("invalid confidence feature vector")
        return self
class ConfidenceDecision(_Public):
    score:Decimal|None=None; feature_vector:ConfidenceFeatureVector|None=None; artifact_hash:Digest|None=None; may_succeed:StrictBool; next_action:Literal["succeed","one_recovery","safe_retrieval","degrade_unknown","degrade_no_trade"]; reason_code:Literal["calibrated","calibration_unavailable","hard_gate_blocked"]

    @field_validator("score", mode="before")
    @classmethod
    def bounded_score(cls, value):
        if value is None: return None
        value = _dec(value)
        if value < 0 or value > 1: raise CalibrationError("invalid confidence decision")
        return value
    @model_validator(mode="after")
    def decision_shape(self):
        if self.may_succeed != (self.next_action == "succeed") or (self.may_succeed and self.reason_code != "calibrated"): raise CalibrationError("invalid confidence decision")
        if self.reason_code == "calibrated" and (self.score is None or self.feature_vector is None or self.artifact_hash is None or self.feature_vector.artifact_hash != self.artifact_hash): raise CalibrationError("invalid confidence decision")
        if self.may_succeed and (self.score is None or self.feature_vector is None): raise CalibrationError("invalid confidence decision")
        if not self.may_succeed and self.reason_code != "calibrated" and (self.score is not None or self.feature_vector is not None or self.artifact_hash is not None): raise CalibrationError("invalid confidence decision")
        return self
def _fallback(c:object,reason:str="calibration_unavailable")->ConfidenceDecision:
    return ConfidenceDecision(may_succeed=False,next_action="safe_retrieval" if c == "informational" else "degrade_no_trade",reason_code=reason)
class ConfidenceGate:
    SUCCESS=SUCCESS; ABSTAIN=RECOVERY
    def __init__(self, *, trusted_policy=None, signature_verifier=None, request_context=None):
        self._sealed = True; self._context = None; self._policy = None; self._verifier = None
        try:
            if type(trusted_policy) is not TrustedConfidencePolicy or type(request_context) is not TrustedRequestContext or type(signature_verifier) is dict or not callable(getattr(signature_verifier, "verify", None)): return
            self._policy=TrustedConfidencePolicy.model_validate(trusted_policy.model_dump()); self._context=TrustedRequestContext.model_validate(request_context.model_dump()); self._verifier=signature_verifier; self._sealed=False
        except Exception: return
    def trusted_context_snapshot(self) -> TrustedRequestContext | None:
        if self._sealed or self._context is None: return None
        try: return TrustedRequestContext.model_validate(self._context.model_dump(mode="python", round_trip=True))
        except Exception: return None

    def trusted_policy_snapshot(self) -> TrustedConfidencePolicy | None:
        if self._sealed or self._policy is None: return None
        try: return TrustedConfidencePolicy.model_validate(self._policy.model_dump(mode="python", round_trip=True))
        except Exception: return None

    def evaluate(self,observation:object,artifact:object)->ConfidenceDecision:
        if self._sealed: return _fallback(None)
        c=self._context.request_class
        try:
            o=ConfidenceObservation.model_validate(observation.model_dump(mode="python",round_trip=True));a=ConfidenceCalibratorArtifact.model_validate(artifact.model_dump(mode="python",round_trip=True));p=self._policy;g=self._context.hard_gates
            if not all((g.permission,g.risk,g.budget,g.loop,g.evidence,g.audit_integrity)) or g.policy_hash!=p.policy_hash:return _fallback(c,"hard_gate_blocked")
            try:
                signature_valid = self._verifier.verify(a.key_id, artifact_payload(a), a.signature)
            except Exception:
                self._sealed = True
                return _fallback(None)
            if type(signature_valid) is not bool:
                self._sealed = True
                return _fallback(None)
            if (a.artifact_id,a.artifact_version,a.artifact_hash,a.schema_hash,a.policy_hash,a.dataset_hash,a.key_id,tuple(a.feature_specs[i].feature_name for i in range(3)),a.issued_epoch,a.expires_epoch)!=(p.artifact_id,p.artifact_version,p.artifact_hash,p.schema_hash,p.policy_hash,p.dataset_hash,p.key_id,p.feature_order,p.issued_epoch,p.expires_epoch) or o.applicability_domain!=p.applicability_domain or o.applicability_domain not in a.applicability_domains or confidence_snapshot_hashes(o)!=(o.accepted_record_snapshot_hash,o.provenance_snapshot_hash) or o.accepted_record_snapshot_hash!=p.accepted_record_snapshot_hash or o.provenance_snapshot_hash!=p.provenance_snapshot_hash or not a.issued_epoch<=self._context.evaluation_epoch<=a.expires_epoch or not signature_valid:raise CalibrationError("untrusted artifact")
            es={x.evidence_id:x for x in o.accepted_evidence}; slots={x.required_slot_id:x for x in o.accepted_evidence}; rs={x.source_id:x for x in o.source_registry}; cs={x.conflict_id:x for x in o.conflicts}
            if len(es)!=len(o.accepted_evidence) or len(slots)!=len(o.accepted_evidence) or len(rs)!=len(o.source_registry) or len(cs)!=len(o.conflicts) or any(not r.enabled for r in rs.values()) or any(x.source_id not in rs for x in o.accepted_evidence) or set(slots)!=set(o.targets.required_evidence_slot_ids) or set(rs)!=set(o.targets.required_source_ids) or {slots[s].source_id for s in o.targets.required_evidence_slot_ids}!=set(o.targets.required_source_ids) or set(o.targets.known_conflict_slot_ids)!=set(cs) or any(not x.resolved or not set(x.evidence_ids)<=set(es) for x in cs.values()) or not set(o.targets.required_dependency_ids)<=set(o.folded_state.completed_dependency_ids) or not set(o.targets.required_output_field_paths)<=set(o.folded_state.valid_output_field_paths) or not set(o.targets.risk_invariant_ids)<=set(o.folded_state.satisfied_risk_invariant_ids):raise CalibrationError("incomplete host metadata")
            v=ConfidenceFeatureVector(artifact_hash=a.artifact_hash,features=tuple(ConfidenceFeatureValue(feature_name=n,value=Decimal(1)) for n in FEATURE_ORDER))
            with localcontext(_CTX):score=+(a.intercept+sum((x.coefficient for x in a.feature_specs),Decimal(0)))
            return self._decide(score=score,recovered=self._context.recovery_used,feature_vector=v)
        except Exception:return _fallback(c)
    def decide(self, *, score: object, recovered: object, feature_vector: ConfidenceFeatureVector) -> ConfidenceDecision:
        if self._sealed: return _fallback(None)
        return _fallback(self._context.request_class)

    def _decide(self, *, score: object, recovered: object, feature_vector: ConfidenceFeatureVector) -> ConfidenceDecision:
        try:
            if type(feature_vector) is not ConfidenceFeatureVector:
                raise CalibrationError("invalid feature vector")
            v=ConfidenceFeatureVector.model_validate(feature_vector.model_dump())
            if v.artifact_hash != self._policy.artifact_hash:
                return _fallback(self._context.request_class)
            s=_dec(score)
            if s>=SUCCESS:return ConfidenceDecision(score=s,feature_vector=v,artifact_hash=v.artifact_hash,may_succeed=True,next_action="succeed",reason_code="calibrated")
            if s>=RECOVERY and recovered is False:return ConfidenceDecision(score=s,feature_vector=v,artifact_hash=v.artifact_hash,may_succeed=False,next_action="one_recovery",reason_code="calibrated")
            return ConfidenceDecision(score=s,feature_vector=v,artifact_hash=v.artifact_hash,may_succeed=False,next_action="degrade_unknown" if self._context.request_class=="informational" else "degrade_no_trade",reason_code="calibrated")
        except Exception: return _fallback(self._context.request_class)
