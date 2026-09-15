from decimal import Decimal, getcontext
from hashlib import sha256
import pytest
from market_agent.workflow_confidence_calibration import *
H="a"*64
class Verifier:
    def verify(self,key_id,payload,signature): return key_id=="host-key" and signature==sha256(b"host-secret"+payload).hexdigest()
def targets(): return ConfidenceTargetSnapshot(required_dependency_ids=("collect",),required_output_field_paths=("result.summary",),required_evidence_slot_ids=("primary",),required_source_ids=("official",),known_conflict_slot_ids=("claim",),risk_invariant_ids=("safe",))
def obs(**x):
 d=dict(applicability_domain="market",targets=targets(),accepted_evidence=(AcceptedEvidenceRecord(evidence_id="e",source_id="official",required_slot_id="primary",provenance_hash="b"*64,accepted_by_host=True),),conflicts=(ConflictRecord(conflict_id="claim",evidence_ids=("e",),resolved=True,provenance_hash="c"*64),),source_registry=(SourceRegistryRecord(source_id="official",registry_hash="d"*64,enabled=True),),folded_state=ConfidenceFoldedState(completed_dependency_ids=("collect",),valid_output_field_paths=("result.summary",),satisfied_risk_invariant_ids=("safe",),event_fold_hash="e"*64),accepted_record_snapshot_hash="f"*64,provenance_snapshot_hash="1"*64)
 d.update(x);raw=ConfidenceObservation.model_construct(**d);d["accepted_record_snapshot_hash"],d["provenance_snapshot_hash"]=confidence_snapshot_hashes(raw);return ConfidenceObservation(**d)
def specs():return tuple(ConfidenceFeatureSpec(feature_name=n,coefficient=Decimal(v)) for n,v in zip(FEATURE_ORDER,(".4",".3",".15")))
def art(**x):
 d=dict(artifact_id="cal-v1",artifact_version="v1",schema_hash=H,policy_hash="b"*64,dataset_hash="c"*64,applicability_domains=("market",),feature_specs=specs(),intercept=Decimal("0"),issued_epoch=10,expires_epoch=20,key_id="host-key",artifact_hash="d"*64,signature="0"*64);d.update(x)
 raw=ConfidenceCalibratorArtifact.model_construct(**d);d["signature"]=sha256(b"host-secret"+artifact_payload(raw)).hexdigest();return ConfidenceCalibratorArtifact(**d)
def gate(a=None,**x):
 a=a or art();p=TrustedConfidencePolicy(artifact_id=a.artifact_id,artifact_version=a.artifact_version,artifact_hash=a.artifact_hash,schema_hash=a.schema_hash,policy_hash=a.policy_hash,dataset_hash=a.dataset_hash,key_id=a.key_id,applicability_domain="market",accepted_record_snapshot_hash=obs().accepted_record_snapshot_hash,provenance_snapshot_hash=obs().provenance_snapshot_hash,issued_epoch=a.issued_epoch,expires_epoch=a.expires_epoch)
 g=HardGateSnapshot(permission=True,risk=True,budget=True,loop=True,evidence=True,audit_integrity=True,run_id="run",trace_hash="e"*64,plan_revision=1,policy_hash=a.policy_hash)
 return ConfidenceGate(trusted_policy=p,signature_verifier=Verifier(),request_context=TrustedRequestContext(request_class=x.get("request_class","informational"),evaluation_epoch=x.get("epoch",15),recovery_used=x.get("recovery_used",False),hard_gates=x.get("hard_gates",g)))
def test_gate_requires_host_owned_trusted_policy_before_success():
 assert gate().evaluate(obs(),art()).may_succeed
def test_self_authenticated_artifact_hash_cannot_authorize_success():
 a=art(artifact_hash="9"*64);assert not gate(art()).evaluate(obs(),a).may_succeed
def test_bad_signature_key_domain_stale_future_and_missing_pins_fail_closed():
 a=art();bad=ConfidenceCalibratorArtifact.model_construct(**{**a.__dict__,"signature":"0"*64});assert not gate(a).evaluate(obs(),bad).may_succeed
 assert not gate(a).evaluate(obs(applicability_domain="other"),a).may_succeed
 assert not gate(a,epoch=9).evaluate(obs(),a).may_succeed
 assert not gate(a,epoch=21).evaluate(obs(),a).may_succeed
def test_all_hard_gates_must_pass_even_at_score_one():
 a=art(intercept=Decimal(".15"));g=HardGateSnapshot(permission=False,risk=True,budget=True,loop=True,evidence=True,audit_integrity=True,run_id="run",trace_hash="e"*64,plan_revision=1,policy_hash=a.policy_hash);d=gate(a,hard_gates=g).evaluate(obs(),a);assert not d.may_succeed and d.next_action=="safe_retrieval"
def test_exact_complete_vector_and_decimal_thresholds_are_context_independent():
 a=art();scores=[]
 for p in (2,6,28):
  getcontext().prec=p;d=gate(a).evaluate(obs(),a);scores.append((d.score,d.next_action,len(d.feature_vector.features)))
 assert scores==[(Decimal(".85"),"succeed",3)]*3
 v=ConfidenceFeatureVector(artifact_hash=a.artifact_hash,features=tuple(ConfidenceFeatureValue(feature_name=n,value=Decimal(1)) for n in FEATURE_ORDER));assert gate(a).decide(score=Decimal(".45"),recovered=False,feature_vector=v).next_action=="safe_retrieval";assert gate(a).decide(score=Decimal(".849999999999999999"),recovered=False,feature_vector=v).next_action=="safe_retrieval"
def test_duplicate_slot_or_disabled_or_orphan_evidence_fails_closed():
 a=art();e=AcceptedEvidenceRecord(evidence_id="x",source_id="official",required_slot_id="primary",provenance_hash="2"*64,accepted_by_host=True);assert not gate(a).evaluate(obs(accepted_evidence=(obs().accepted_evidence[0],e)),a).may_succeed
 assert not gate(a).evaluate(obs(source_registry=(SourceRegistryRecord(source_id="official",registry_hash="d"*64,enabled=False),)),a).may_succeed

def test_public_decide_cannot_bypass_host_hard_gates():
    a = art()
    blocked = HardGateSnapshot(permission=False, risk=True, budget=True, loop=True, evidence=True, audit_integrity=True, run_id="run", trace_hash="e" * 64, plan_revision=1, policy_hash=a.policy_hash)
    v = ConfidenceFeatureVector(artifact_hash=a.artifact_hash, features=tuple(ConfidenceFeatureValue(feature_name=n, value=Decimal(1)) for n in FEATURE_ORDER))
    assert not gate(a, hard_gates=blocked).decide(score=Decimal("1"), recovered=False, feature_vector=v).may_succeed


def test_snapshot_recomputes_provenance_and_rejects_unchanged_claim():
    a=art(); changed=obs().model_copy(update={"accepted_evidence":(AcceptedEvidenceRecord(evidence_id="e",source_id="official",required_slot_id="primary",provenance_hash="9"*64,accepted_by_host=True),)})
    assert not gate(a).evaluate(changed,a).may_succeed

def test_decimal_over_one_coefficient_sum_fails_closed():
    a=art(); forged=ConfidenceCalibratorArtifact.model_construct(**{**a.__dict__,"feature_specs":tuple(ConfidenceFeatureSpec(feature_name=n,coefficient=Decimal(".4")) for n in FEATURE_ORDER)})
    assert not gate(a).evaluate(obs(),forged).may_succeed

def test_invalid_context_creates_sealed_no_trade_gate():
    a=art(); g=ConfidenceGate(trusted_policy=gate(a)._policy,signature_verifier=Verifier(),request_context=None)
    assert g.evaluate(obs(),a).next_action=="degrade_no_trade"

def test_decision_direct_construction_enforces_cross_field_invariants():
    from pydantic import ValidationError
    with pytest.raises(ValidationError): ConfidenceDecision(score=Decimal("2"),may_succeed=False,next_action="succeed",reason_code="calibrated")

@pytest.mark.parametrize("override",[
 {"source_registry":(SourceRegistryRecord(source_id="official",registry_hash="d"*64,enabled=True),SourceRegistryRecord(source_id="official",registry_hash="9"*64,enabled=False))},
 {"conflicts":(ConflictRecord(conflict_id="claim",evidence_ids=("e",),resolved=True,provenance_hash="c"*64),ConflictRecord(conflict_id="claim",evidence_ids=("e",),resolved=True,provenance_hash="9"*64))},
 {"source_registry":(SourceRegistryRecord(source_id="extra",registry_hash="d"*64,enabled=True),)},
])
def test_canonical_snapshot_rejects_duplicate_or_exact_set_violations(override):
    a=art(); assert not gate(a).evaluate(obs(**override),a).may_succeed


def test_decimal_precision_cannot_round_invalid_artifact_into_range():
    old=getcontext().prec
    try:
        getcontext().prec=2
        with pytest.raises(Exception): art(feature_specs=tuple(ConfidenceFeatureSpec(feature_name=n,coefficient=Decimal(".334")) for n in FEATURE_ORDER),intercept=Decimal(".002"))
        with pytest.raises(Exception): ConfidenceFeatureSpec(feature_name="required_evidence_coverage",coefficient=Decimal("1e-45"))
    finally: getcontext().prec=old

def test_decision_calibrated_payload_and_model_copy_require_bound_vector_hash():
    from pydantic import ValidationError
    a=art(); v=ConfidenceFeatureVector(artifact_hash=a.artifact_hash,features=tuple(ConfidenceFeatureValue(feature_name=n,value=Decimal(1)) for n in FEATURE_ORDER))
    with pytest.raises(ValidationError): ConfidenceDecision(score=Decimal(".45"),may_succeed=False,next_action="one_recovery",reason_code="calibrated")
    good=ConfidenceDecision(score=Decimal(".45"),feature_vector=v,artifact_hash=a.artifact_hash,may_succeed=False,next_action="one_recovery",reason_code="calibrated")
    with pytest.raises(ValidationError): good.model_copy(update={"artifact_hash":"0"*64})
    with pytest.raises(ValidationError): good.model_copy(update={"reason_code":"calibration_unavailable"})


def test_invalid_verifiers_seal_gate_to_no_trade():
    a=art()
    for verifier in (None, {}, object()):
        sealed=ConfidenceGate(trusted_policy=gate(a)._policy,signature_verifier=verifier,request_context=gate(a)._context)
        assert sealed.evaluate(obs(),a).next_action=="degrade_no_trade"

def test_success_reason_and_vector_pin_cannot_be_forged():
    from pydantic import ValidationError
    a=art(); v=ConfidenceFeatureVector(artifact_hash="9"*64,features=tuple(ConfidenceFeatureValue(feature_name=n,value=Decimal(1)) for n in FEATURE_ORDER))
    with pytest.raises(ValidationError): ConfidenceDecision(score=Decimal(".9"),feature_vector=v,artifact_hash=v.artifact_hash,may_succeed=True,next_action="succeed",reason_code="hard_gate_blocked")
    assert not gate(a)._decide(score=Decimal(".9"),recovered=False,feature_vector=v).may_succeed

def test_truthy_or_raising_verifier_permanently_seals_gate():
    class TruthyVerifier:
        def verify(self, key_id, payload, signature): return "verified"
    class WrongArityVerifier:
        def verify(self): return True
    a=art()
    for verifier in (TruthyVerifier(), WrongArityVerifier()):
        g=ConfidenceGate(trusted_policy=gate(a)._policy,signature_verifier=verifier,request_context=gate(a)._context)
        assert g.evaluate(obs(),a).next_action=="degrade_no_trade"
        assert g.evaluate(obs(),a).next_action=="degrade_no_trade"

def test_private_decide_rejects_malformed_vector_without_raising():
    a=art()
    decision=gate(a)._decide(score=Decimal(".9"),recovered=False,feature_vector=None)
    assert not decision.may_succeed
    assert decision.next_action=="safe_retrieval"


def test_trusted_snapshots_are_fresh_read_only_and_secret_free():
    a = art()
    current = gate(a)
    context = current.trusted_context_snapshot()
    policy = current.trusted_policy_snapshot()
    assert context is not None and policy is not None
    assert (context.hard_gates.run_id, context.hard_gates.trace_hash, context.hard_gates.plan_revision, context.hard_gates.policy_hash) == ("run", "e" * 64, 1, a.policy_hash)
    assert (policy.artifact_hash, policy.schema_hash, policy.policy_hash, policy.dataset_hash) == (a.artifact_hash, a.schema_hash, a.policy_hash, a.dataset_hash)
    with pytest.raises(Exception):
        context.model_copy(update={"request_class": "forged"})
    assert current.trusted_context_snapshot() == context
    assert "verifier" not in policy.model_dump()


def test_sealed_gate_exposes_no_trusted_snapshots():
    sealed = ConfidenceGate()
    assert sealed.trusted_context_snapshot() is None
    assert sealed.trusted_policy_snapshot() is None
