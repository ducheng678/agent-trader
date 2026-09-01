from __future__ import annotations

from copy import deepcopy
from hashlib import sha256
import json
from typing import Annotated, Callable, Literal

from pydantic import BaseModel, Field, StrictBool, StrictFloat, StrictInt, model_validator

from market_agent.workflow_agent_contracts import AgentInvocation, ModelTier, StrictModel
from market_agent.workflow_agent_driver import AgentDriver
from market_agent.workflow_agents.common import JsonContractSchema
from market_agent.workflow_contracts import ContextSummary, Digest, ShortText, Text
from market_agent.workflow_prompt_release import PromptRelease, canonical_json


CheckCode = Literal["schema_valid", "numeric_consistency", "direction_consistency", "evidence_support", "uncertainty_consistency", "risk_invariants"]
CoreKind = Literal["decision_planner", "escalation", "coordinator_summary"]
Disposition = Literal["accept", "retry_original", "return_to_coordinator", "safe_reject"]
_REQUIRED = ("schema_valid", "numeric_consistency", "direction_consistency", "evidence_support", "uncertainty_consistency", "risk_invariants")
PROMPT_PROFILE_ID = "workflow.reflection.v1"


def _hash(value: object) -> str:
    return sha256(canonical_json(value).encode()).hexdigest()


class ObjectiveCheck(StrictModel):
    code: CheckCode
    status: Literal["pass", "fail", "not_verifiable"]
    field_paths: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=12)
    evidence_ids: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=20)
    observed_hash: Digest | None = None


class ObjectiveReview(StrictModel):
    target_hash: Digest
    output_schema_hash: Digest
    checks: tuple[ObjectiveCheck, ...] = Field(min_length=6, max_length=6)

    @model_validator(mode="after")
    def exact_checks(self):
        if {check.code for check in self.checks} != set(_REQUIRED):
            raise ValueError("reflection must report each objective check exactly once")
        return self


class ReflectionRequest(StrictModel):
    trace_id: ShortText
    task_id: ShortText
    target_kind: CoreKind
    target_hash: Digest
    output_schema_hash: Digest
    model_tier: Literal["luna"] = "luna"
    target_json: Annotated[str, Field(max_length=12000)]
    evidence_summary: ContextSummary
    deterministic_checks: tuple[ObjectiveCheck, ...]


class ReflectionOutput(StrictModel):
    conclusion: Literal["supported", "不知道"]
    review: ObjectiveReview | None = None

    @model_validator(mode="after")
    def coherent(self):
        if (self.conclusion == "supported") != (self.review is not None):
            raise ValueError("reflection conclusion and review must agree")
        return self


def reflection_output_schema():
    return JsonContractSchema(schema_id=PROMPT_PROFILE_ID, model=ReflectionOutput,
                              abstention={"conclusion": "不知道", "review": None})


def reflection_release() -> PromptRelease:
    values = dict(schema_version="v1", release_id=PROMPT_PROFILE_ID,
        stable_system_prefix="Check only schema, numbers, direction, cited evidence, uncertainty, and risk invariants. Reason step by step internally; output strict checks. If uncertain, answer 不知道. Never judge strategy, profitability, prose, or confidence; never repair the target.",
        supported_task_kinds=("extract",), supported_model_tiers=(ModelTier.LUNA,),
        temperature_profile=((ModelTier.LUNA, 0.0),))
    return PromptRelease(digest=_hash(values), **values)


def run_reflection(request: ReflectionRequest, driver: AgentDriver, *, deadline_epoch: float,
                   cost_limit_usd: float, grant: object, authorize: Callable) -> ObjectiveReview:
    request = ReflectionRequest.model_validate(request)
    if grant is None or not callable(authorize):
        raise PermissionError("reflection requires a host-issued grant")
    authorize(request, grant)
    release, schema = reflection_release(), reflection_output_schema()
    invocation = AgentInvocation(trace_id=request.trace_id, run_id=request.evidence_summary.workflow_id,
        task_id=request.task_id, task_kind="extract", prompt_release_id=release.release_id,
        prompt_release_digest=release.digest, allowed_model_tier=ModelTier.LUNA,
        deadline_epoch=deadline_epoch, max_attempts=1, cost_limit_usd=cost_limit_usd,
        output_schema_id=schema.schema_id, output_schema_digest=schema.digest,
        user_payload=request.model_dump(mode="json"))
    result = driver.execute(invocation)
    if result.failure is not None or result.trace_id != request.trace_id:
        raise ValueError("objective reflection is unavailable")
    output = ReflectionOutput.model_validate_json(canonical_json(result.output))
    if output.review is None:
        raise ValueError("objective reflection is not verifiable")
    return output.review


class ReflectionResult(StrictModel):
    target_kind: CoreKind
    target_hash: Digest
    output_schema_hash: Digest
    checks: tuple[ObjectiveCheck, ...] = Field(min_length=6, max_length=6)
    available: bool = True
    disposition: Disposition
    reflection_hash: Digest | None = None

    @model_validator(mode="after")
    def governed_disposition(self):
        if {item.code for item in self.checks} != set(_REQUIRED):
            raise ValueError("reflection requires the complete objective check inventory")
        expected = _disposition(self.checks, self.available)
        if self.disposition != expected:
            raise ValueError("only deterministic policy selects reflection disposition")
        digest = _hash(self.model_dump(mode="json", exclude={"reflection_hash"}))
        if self.reflection_hash is not None and self.reflection_hash != digest:
            raise ValueError("reflection hash mismatch")
        object.__setattr__(self, "reflection_hash", digest)
        return self

    @property
    def error_tuple(self) -> tuple[int, int, int]:
        return (sum(item.code == "risk_invariants" and item.status == "fail" for item in self.checks),
                sum(item.status == "fail" for item in self.checks),
                sum(item.status == "not_verifiable" for item in self.checks))


def _disposition(checks, available):
    if not available or any(item.code == "risk_invariants" and item.status == "fail" for item in checks):
        return "safe_reject"
    if any(item.code == "evidence_support" and item.status != "pass" for item in checks):
        return "return_to_coordinator"
    if any(item.status == "fail" for item in checks):
        return "retry_original"
    if any(item.status == "not_verifiable" for item in checks):
        return "return_to_coordinator"
    return "accept"


def _payload(target: BaseModel) -> dict:
    if not isinstance(target, BaseModel):
        raise TypeError("reflection targets must be typed outputs")
    return target.model_dump(mode="json")


def _evidence(payload: dict) -> set[str]:
    return set(payload.get("evidence_refs", ())) | set(payload.get("evidence_references", ())) | set(payload.get("source_references", ()))


def _checks(target: BaseModel, output_model: type[BaseModel], context: ContextSummary) -> tuple[ObjectiveCheck, ...]:
    payload = _payload(target)
    checks = {code: ObjectiveCheck(code=code, status="pass") for code in _REQUIRED}
    try:
        output_model.model_validate_json(canonical_json(payload), strict=True, extra="forbid")
    except ValueError:
        checks["schema_valid"] = ObjectiveCheck(code="schema_valid", status="fail")
    known = set(context.source_references) | {fact.source_id for fact in context.market_facts}
    if not known or not _evidence(payload) <= known:
        checks["evidence_support"] = ObjectiveCheck(code="evidence_support", status="not_verifiable")
    if context.conflicts:
        checks["evidence_support"] = ObjectiveCheck(code="evidence_support", status="fail", evidence_ids=tuple(sorted(known))[:20])
    entry, stop, action = payload.get("entry_price"), payload.get("stop_price"), payload.get("action")
    if entry is not None and stop is not None and ((action == "long" and stop >= entry) or (action == "short" and stop <= entry)):
        checks["numeric_consistency"] = ObjectiveCheck(code="numeric_consistency", status="fail", field_paths=("/stop_price",))
    if payload.get("knowledge_status") == "insufficient" and action not in (None, "no_trade"):
        checks["uncertainty_consistency"] = ObjectiveCheck(code="uncertainty_consistency", status="fail", field_paths=("/action",))
    return tuple(checks[code] for code in _REQUIRED)


def reflect_output(target: BaseModel, *, target_kind: CoreKind, context: ContextSummary,
                   output_model: type[BaseModel] | None = None,
                   reviewer: Callable[[ReflectionRequest], ObjectiveReview] | None = None) -> ReflectionResult:
    context = ContextSummary.model_validate(context)
    model = output_model or type(target)
    local = _checks(target, model, context)
    payload = _payload(target)
    request = ReflectionRequest(trace_id=context.trace_id, task_id=context.task_id, target_kind=target_kind,
        target_hash=_hash(payload), output_schema_hash=_hash(model.model_json_schema()),
        target_json=canonical_json(payload), evidence_summary=context, deterministic_checks=local)
    available = False
    combined = local
    if reviewer is not None:
        try:
            review = ObjectiveReview.model_validate(reviewer(request))
            sources = set(context.source_references) | {fact.source_id for fact in context.market_facts}
            if (review.target_hash, review.output_schema_hash) != (request.target_hash, request.output_schema_hash):
                raise ValueError("reflection target binding mismatch")
            if any(not set(item.evidence_ids) <= sources for item in review.checks):
                raise ValueError("reflection references unavailable evidence")
            remote = {item.code: item for item in review.checks}
            order = {"pass": 0, "not_verifiable": 1, "fail": 2}
            combined = tuple(max((item, remote[item.code]), key=lambda check: order[check.status]) for item in local)
            available = True
        except Exception:
            available = False
    return ReflectionResult(target_kind=target_kind, target_hash=request.target_hash,
        output_schema_hash=request.output_schema_hash, checks=combined, available=available,
        disposition=_disposition(combined, available))


class CorrectionContext(StrictModel):
    target_hash: Digest
    reflection_hash: Digest
    error_codes: tuple[CheckCode, ...] = Field(max_length=6)
    field_paths: tuple[ShortText, ...] = Field(max_length=12)
    evidence_ids: tuple[ShortText, ...] = Field(max_length=20)
    retry_ordinal: Literal[1, 2]
    prior_output_summary: Annotated[str, Field(max_length=1600)]
    original_task_summary: Annotated[str, Field(max_length=1000)]


def build_correction_context(target: BaseModel, reflection: ReflectionResult, *, task_summary: str,
                             retry_ordinal: Literal[1, 2] = 1) -> CorrectionContext:
    reflection = ReflectionResult.model_validate(reflection)
    payload = _payload(target)
    if _hash(payload) != reflection.target_hash or reflection.disposition != "retry_original":
        raise ValueError("correction requires the matching rejected target")
    errors = tuple(item for item in reflection.checks if item.status != "pass")
    projection = {key: payload[key] for key in ("action", "knowledge_status", "entry_price", "stop_price", "execute_now", "uncertainty_reason") if key in payload}
    return CorrectionContext(target_hash=reflection.target_hash, reflection_hash=reflection.reflection_hash,
        error_codes=tuple(item.code for item in errors),
        field_paths=tuple(sorted({path for item in errors for path in item.field_paths}))[:12],
        evidence_ids=tuple(sorted({ref for item in errors for ref in item.evidence_ids}))[:20],
        retry_ordinal=retry_ordinal, prior_output_summary=canonical_json(projection)[:1600],
        original_task_summary=task_summary[:1000])


class FieldReplacement(StrictModel):
    path: ShortText
    value: str | StrictFloat | StrictInt | StrictBool | None


class CorrectionPatch(StrictModel):
    target_hash: Digest
    replacements: tuple[FieldReplacement, ...] = Field(min_length=1, max_length=8)


def apply_correction_patch(target: BaseModel, patch: CorrectionPatch, *, allowed_paths: tuple[str, ...],
                           output_model: type[BaseModel] | None = None) -> BaseModel:
    patch = CorrectionPatch.model_validate(patch)
    payload = _payload(target)
    if patch.target_hash != _hash(payload):
        raise ValueError("patch target hash mismatch")
    paths = [replacement.path for replacement in patch.replacements]
    if len(paths) != len(set(paths)) or not set(paths) <= set(allowed_paths):
        raise ValueError("patch must replace unique allowlisted fields only")
    updated = deepcopy(payload)
    for replacement in patch.replacements:
        parts = replacement.path.split("/")[1:]
        if not replacement.path.startswith("/") or not parts or any(part in ("", "schema_version", "trace_id", "task_id", "workflow_id", "evidence_refs", "evidence_references") for part in parts):
            raise ValueError("patch cannot change identity, schema, or evidence")
        parent = updated
        for part in parts[:-1]:
            if not isinstance(parent, dict) or part not in parent:
                raise ValueError("patch path does not exist")
            parent = parent[part]
        if not isinstance(parent, dict) or parts[-1] not in parent:
            raise ValueError("patch may only replace existing fields")
        parent[parts[-1]] = replacement.value
    return (output_model or type(target)).model_validate_json(canonical_json(updated), strict=True, extra="forbid")


class CorrectionOutcome(StrictModel):
    disposition: Disposition
    output_json: str | None = None
    attempted_modes: tuple[Literal["patch", "rewrite"], ...] = Field(max_length=2)
    seen_hashes: tuple[Digest, ...] = Field(max_length=3)
    final_reflection: ReflectionResult


def _improves(before: BaseModel, after: BaseModel, prior: ReflectionResult, review: ReflectionResult, seen: set[str]) -> bool:
    old, new = _payload(before), _payload(after)
    digest = _hash(new)
    if digest in seen or review.target_hash != digest or not review.available or review.error_tuple >= prior.error_tuple:
        return False
    if not _evidence(old) <= _evidence(new):
        return False
    if new.get("action") not in (old.get("action"), "no_trade"):
        return False
    if old.get("entry_price") is not None and old.get("stop_price") is not None and new.get("action") != "no_trade":
        if new.get("entry_price") is None or new.get("stop_price") is None:
            return False
        if abs(new["entry_price"] - new["stop_price"]) > abs(old["entry_price"] - old["stop_price"]):
            return False
    old_critical = {item.code for item in prior.checks if item.code == "risk_invariants" and item.status == "fail"}
    return {item.code for item in review.checks if item.code == "risk_invariants" and item.status == "fail"} <= old_critical


def correct_output(target: BaseModel, reflection: ReflectionResult, *, task_summary: str,
                   generate_patch: Callable[[CorrectionContext], CorrectionPatch],
                   generate_rewrite: Callable[[CorrectionContext], BaseModel],
                   reflect: Callable[[BaseModel], ReflectionResult], allowed_paths: tuple[str, ...],
                   output_model: type[BaseModel] | None = None) -> CorrectionOutcome:
    reflection = ReflectionResult.model_validate(reflection)
    if reflection.target_hash != _hash(_payload(target)):
        raise ValueError("correction target mismatch")
    seen = [reflection.target_hash]
    modes = []
    current, review = target, reflection
    if review.disposition != "retry_original":
        return CorrectionOutcome(disposition=review.disposition,
            output_json=canonical_json(_payload(current)) if review.disposition == "accept" else None,
            seen_hashes=tuple(seen), final_reflection=review)
    for ordinal, mode in enumerate(("patch", "rewrite"), 1):
        context = build_correction_context(current, review, task_summary=task_summary, retry_ordinal=ordinal)
        modes.append(mode)
        try:
            if mode == "patch":
                candidate = apply_correction_patch(current, generate_patch(context), allowed_paths=tuple(set(allowed_paths) & set(context.field_paths)), output_model=output_model)
            else:
                candidate = generate_rewrite(context)
                candidate = (output_model or type(target)).model_validate_json(canonical_json(_payload(candidate)), strict=True, extra="forbid")
        except Exception:
            continue
        try:
            next_review = ReflectionResult.model_validate(reflect(candidate))
        except Exception:
            break
        digest = _hash(_payload(candidate))
        if not _improves(current, candidate, review, next_review, set(seen)):
            break
        seen.append(digest)
        current, review = candidate, next_review
        if review.disposition == "accept":
            return CorrectionOutcome(disposition="accept", output_json=canonical_json(_payload(current)),
                attempted_modes=tuple(modes), seen_hashes=tuple(seen), final_reflection=review)
        if review.disposition != "retry_original":
            break
    return CorrectionOutcome(disposition="safe_reject", attempted_modes=tuple(modes),
                              seen_hashes=tuple(seen), final_reflection=review)
