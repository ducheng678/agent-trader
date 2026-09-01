"""Bounded deterministic loop prevention over validated semantic observations."""

from __future__ import annotations

from collections import OrderedDict, deque
from enum import Enum
from hashlib import sha256
import json
import math
import re
from typing import Any, Literal, Sequence

from pydantic import Field, StrictBool, StrictFloat, StrictInt, field_validator, model_validator

from market_agent.workflow_contracts import ContractModel, Digest, NonNegativeInt, ShortText
from market_agent.workflow_harness_contracts import ProgressVector


FINGERPRINT_SCHEMA_VERSION = "v1"
STATE_WINDOW_SIZE = 12
ACTION_WINDOW_SIZE = 5
RECOVERY_SIGNATURE_CAPACITY = 12
MAX_ARGUMENTS = 16
MAX_VALUE_LENGTH = 256
MAX_NUMERIC_ABS = 1_000_000_000

POSITIVE_FIELDS = (
    "completed_dependency_count", "valid_required_field_count",
    "filled_required_evidence_slot_count", "fresh_authoritative_source_coverage",
)
NEGATIVE_FIELDS = (
    "missing_evidence_count", "validation_error_count",
    "unresolved_conflict_count", "risk_invariant_failure_count",
)
ALL_PROGRESS_FIELDS = POSITIVE_FIELDS + NEGATIVE_FIELDS


class LoopScope(str, Enum):
    ATTEMPT = "attempt"
    WORK_ITEM = "work_item"
    STAGE = "stage"
    RUN = "run"


class ObservationKind(str, Enum):
    SEMANTIC_CHECKPOINT = "semantic_checkpoint"
    HEARTBEAT = "heartbeat"
    INFRASTRUCTURE = "infrastructure"

    @property
    def is_semantic(self) -> bool:
        return self is ObservationKind.SEMANTIC_CHECKPOINT


class SemanticArgumentName(str, Enum):
    LABEL = "label"
    SYMBOL = "symbol"
    EVENT_TYPE = "event_type"
    ATTEMPT_LIMIT = "attempt_limit"
    OPERATION = "operation"
    TARGET = "target"
    CONDITION = "condition"
    MODE = "mode"
    LIMIT = "limit"


SemanticScalar = StrictInt | ShortText
_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
_CODE_RE = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_SYMBOL_RE = re.compile(r"^[A-Za-z][A-Za-z0-9]{0,15}$")
_WORKER_ID_RE = re.compile(r"^[a-z][a-z0-9-]{0,31}$")
_JWT_RE = re.compile(r"^[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}$")


def _digest(value: object) -> str:
    return sha256(json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")).hexdigest()


class _PublicContract(ContractModel):
    """Frozen, strict public values whose copies re-enter validation."""

    def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False) -> Any:
        values = self.model_dump(mode="python", round_trip=True)
        if update:
            values.update(update)
        return type(self).model_validate(values)


class _Fingerprint(_PublicContract):
    digest: Digest

    @property
    def value(self) -> str:
        return self.digest


class ActionFingerprint(_Fingerprint):
    pass


class ResultFingerprint(_Fingerprint):
    pass


class StateFingerprint(_Fingerprint):
    pass


class _ActionFingerprintInput(_PublicContract):
    worker_id: ShortText
    worker_version: ShortText
    action_kind: ShortText
    canonical_arguments: tuple[tuple[SemanticArgumentName, SemanticScalar], ...] = Field(max_length=MAX_ARGUMENTS)
    context_hash: Digest
    dependency_hash: Digest
    plan_revision: NonNegativeInt
    prompt_hash: Digest
    tool_hash: Digest
    output_schema_hash: Digest
    model_route: ShortText
    correction_ordinal: NonNegativeInt

    @model_validator(mode="after")
    def validate_arguments(self) -> _ActionFingerprintInput:
        names = tuple(name for name, _ in self.canonical_arguments)
        if len(set(names)) != len(names):
            raise ValueError("invalid action fingerprint input")
        for name, value in self.canonical_arguments:
            _validate_argument_value(name, value)
        return self


class _ResultFingerprintInput(_PublicContract):
    outcome_kind: ShortText
    validated_output_hash: Digest | None
    normalized_error_class: ShortText | None
    normalized_error_code: ShortText | None
    accepted_evidence_ids: tuple[ShortText, ...] = Field(max_length=64)
    tool_result_hashes: tuple[Digest, ...] = Field(max_length=64)
    result_schema_version: ShortText

    @model_validator(mode="after")
    def validate_unique(self) -> _ResultFingerprintInput:
        if len(set(self.accepted_evidence_ids)) != len(self.accepted_evidence_ids) or len(set(self.tool_result_hashes)) != len(self.tool_result_hashes):
            raise ValueError("invalid result fingerprint input")
        return self


class _StateFingerprintInput(_PublicContract):
    run_state: ShortText
    work_item_state: ShortText
    attempt_state: ShortText
    stage_id: ShortText
    plan_revision: NonNegativeInt
    unresolved_work_ids: tuple[ShortText, ...] = Field(max_length=64)
    dependency_versions: tuple[tuple[ShortText, NonNegativeInt], ...] = Field(max_length=64)
    progress: ProgressVector
    normalized_error_class: ShortText | None

    @model_validator(mode="after")
    def validate_unique(self) -> _StateFingerprintInput:
        if len(set(self.unresolved_work_ids)) != len(self.unresolved_work_ids):
            raise ValueError("invalid state fingerprint input")
        ids = tuple(item[0] for item in self.dependency_versions)
        if len(set(ids)) != len(ids):
            raise ValueError("invalid state fingerprint input")
        return self


class ActionObservationFingerprint(_Fingerprint):
    scope: LoopScope
    action: ActionFingerprint
    result: ResultFingerprint

    @classmethod
    def from_parts(cls, action: ActionFingerprint, result: ResultFingerprint, *, scope: LoopScope | str) -> ActionObservationFingerprint:
        scoped = LoopScope(scope)
        return cls(digest=_digest({"scope": scoped.value, "action": action.digest, "result": result.digest, "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION}), scope=scoped, action=action, result=result)

    @model_validator(mode="after")
    def validate_digest(self) -> ActionObservationFingerprint:
        if self.digest != _digest({"scope": self.scope.value, "action": self.action.digest, "result": self.result.digest, "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION}):
            raise ValueError("action observation digest must bind scope, action, and result")
        return self


def _normalize_rotation(period: Sequence[str]) -> tuple[str, ...]:
    values = tuple(period)
    return min(values[index:] + values[:index] for index in range(len(values)))


def _is_primitive(period: Sequence[str]) -> bool:
    values = tuple(period)
    return not any(len(values) % size == 0 and values == values[:size] * (len(values) // size) for size in range(1, len(values)))


class CycleSignature(_Fingerprint):
    scope: LoopScope
    plan_revision: NonNegativeInt
    fingerprint_schema_version: ShortText
    period: tuple[Digest, ...] = Field(min_length=1, max_length=6)

    @model_validator(mode="after")
    def validate_signature(self) -> CycleSignature:
        if tuple(self.period) != _normalize_rotation(self.period) or not _is_primitive(self.period):
            raise ValueError("cycle period must be primitive and rotation-normalized")
        expected = _digest({"scope": self.scope.value, "plan_revision": self.plan_revision, "fingerprint_schema_version": self.fingerprint_schema_version, "period": self.period})
        if self.digest != expected:
            raise ValueError("cycle digest must bind scope, plan, schema, and period")
        return self


class SeverityPolicy(_PublicContract):
    policy_version: ShortText
    critical_positive_regressions: tuple[ShortText, ...] = ("filled_required_evidence_slot_count", "fresh_authoritative_source_coverage")
    critical_negative_regressions: tuple[ShortText, ...] = ("validation_error_count", "risk_invariant_failure_count")

    @model_validator(mode="after")
    def validate_fields(self) -> SeverityPolicy:
        if not set(self.critical_positive_regressions).issubset(POSITIVE_FIELDS) or not set(self.critical_negative_regressions).issubset(NEGATIVE_FIELDS):
            raise ValueError("severity fields must be declared progress dimensions")
        if len(set(self.critical_positive_regressions)) != len(self.critical_positive_regressions) or len(set(self.critical_negative_regressions)) != len(self.critical_negative_regressions):
            raise ValueError("severity fields must be unique")
        return self


class ProgressDecision(_PublicContract):
    advanced: StrictBool
    critical_regression: StrictBool = False
    worsened_fields: tuple[ShortText, ...] = ()

    @model_validator(mode="after")
    def validate_shape(self) -> ProgressDecision:
        if self.advanced and (self.critical_regression or self.worsened_fields):
            raise ValueError("advanced progress cannot contain a regression")
        if self.critical_regression and not self.worsened_fields:
            raise ValueError("critical regression requires a worsened dimension")
        return self


class SemanticCheckpoint(_PublicContract):
    scope: LoopScope
    observation_kind: ObservationKind
    state_fingerprint: StateFingerprint
    progress: ProgressVector
    plan_revision: NonNegativeInt
    fingerprint_schema_version: ShortText = FINGERPRINT_SCHEMA_VERSION
    worker_id: ShortText | None = None
    normalized_failure: ShortText | None = None
    failure_context_hash: Digest | None = None
    failure_dependency_hash: Digest | None = None
    correction_ordinal: NonNegativeInt = 0
    model_route: ShortText | None = None

    @property
    def is_semantic(self) -> bool:
        return self.observation_kind.is_semantic

    @model_validator(mode="after")
    def validate_failure_shape(self) -> SemanticCheckpoint:
        fields = (self.failure_context_hash, self.failure_dependency_hash, self.model_route)
        if self.normalized_failure is None and (self.worker_id is not None or any(item is not None for item in fields)):
            raise ValueError("failure metadata requires a normalized failure")
        if self.normalized_failure is not None and (self.worker_id is None or any(item is None for item in fields)):
            raise ValueError("normalized failure requires complete semantic route metadata")
        if self.worker_id is not None and not _WORKER_ID_RE.fullmatch(self.worker_id):
            raise ValueError("worker identifier must be canonical lowercase ASCII")
        return self


class LoopDecision(_PublicContract):
    allowed: StrictBool
    ignored: StrictBool = False
    stop_reason: ShortText | None = None
    cycle_signature: CycleSignature | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> LoopDecision:
        if self.ignored and (not self.allowed or self.stop_reason is not None or self.cycle_signature is not None):
            raise ValueError("ignored decisions must be neutral and allowed")
        if self.allowed and not self.ignored and self.stop_reason is not None:
            raise ValueError("allowed decisions cannot have a stop reason")
        if not self.allowed and self.stop_reason is None:
            raise ValueError("stopped decisions require a reason")
        return self


def _reject_extra(extra: dict[str, object], message: str) -> None:
    if extra:
        raise ValueError(message)


def _validate_argument_tree(value: object) -> None:
    seen: set[int] = set()
    pending: list[tuple[object, int]] = [(value, 0)]
    while pending:
        item, depth = pending.pop()
        if depth > 4:
            raise ValueError("invalid action fingerprint input")
        if isinstance(item, (dict, list, tuple)):
            marker = id(item)
            if marker in seen:
                raise ValueError("invalid action fingerprint input")
            seen.add(marker)
            if len(item) > MAX_ARGUMENTS:
                raise ValueError("invalid action fingerprint input")
            children = item.values() if isinstance(item, dict) else item
            pending.extend((child, depth + 1) for child in children)
        else:
            _validate_scalar(item)


def _validate_scalar(value: object) -> None:
    if isinstance(value, bytes):
        raise ValueError("invalid action fingerprint input")
    if isinstance(value, str):
        if len(value) > MAX_VALUE_LENGTH or _looks_sensitive(value):
            raise ValueError("invalid action fingerprint input")
        return
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        if abs(value) > MAX_NUMERIC_ABS:
            raise ValueError("invalid action fingerprint input")
        return
    if isinstance(value, float):
        if not math.isfinite(value) or abs(value) > MAX_NUMERIC_ABS:
            raise ValueError("invalid action fingerprint input")
        return
    raise ValueError("invalid action fingerprint input")


def _validate_argument_value(name: SemanticArgumentName, value: object) -> None:
    if name in {SemanticArgumentName.ATTEMPT_LIMIT, SemanticArgumentName.LIMIT}:
        if type(value) is not int or not 0 <= value <= 1_000_000:
            raise ValueError("invalid action fingerprint input")
        return
    if type(value) is not str or not _safe_code(value):
        raise ValueError("invalid action fingerprint input")
    if name is SemanticArgumentName.SYMBOL:
        if not _SYMBOL_RE.fullmatch(value):
            raise ValueError("invalid action fingerprint input")
    elif not _CODE_RE.fullmatch(value):
        raise ValueError("invalid action fingerprint input")


def _looks_sensitive(value: str) -> bool:
    lower = value.lower()
    return bool(
        _JWT_RE.fullmatch(value)
        or re.fullmatch(r"ghp_[A-Za-z0-9]{20,}", value)
        or re.fullmatch(r"AKIA[A-Z0-9]{16}", value)
        or re.fullmatch(r"sk-live-[A-Za-z0-9-]+", value)
        or re.match(r"(?i)^bearer\s+[A-Za-z0-9._-]{12,}$", value)
        or value.startswith("-----BEGIN ")
        or bool(re.search(r"(^|[_-])(secret|password|credential)([_-]|$)", lower))
    )


def _safe_code(value: str) -> bool:
    return len(value) <= 32 and not _looks_sensitive(value)


def _bounded_sequence(value: object, limit: int, message: str) -> tuple[object, ...]:
    if type(value) not in (tuple, list) or len(value) > limit:
        raise ValueError(message)
    return tuple(value)


def _arguments(value: object) -> tuple[tuple[SemanticArgumentName, object], ...]:
    _validate_argument_tree(value)
    if not isinstance(value, dict) or len(value) > MAX_ARGUMENTS:
        raise ValueError("invalid action fingerprint input")
    if any(not isinstance(name, str) for name in value):
        raise ValueError("invalid action fingerprint input")
    if any(isinstance(item, (dict, list, tuple)) for item in value.values()):
        raise ValueError("invalid action fingerprint input")
    try:
        normalized = []
        for name, item in value.items():
            semantic_name = SemanticArgumentName(name)
            _validate_argument_value(semantic_name, item)
            if semantic_name is SemanticArgumentName.SYMBOL:
                item = item.upper()
            elif isinstance(item, str):
                item = item.lower()
            normalized.append((semantic_name, item))
        return tuple(sorted(normalized, key=lambda item: item[0].value))
    except ValueError:
        raise ValueError("invalid action fingerprint input") from None


def build_action_fingerprint(*, worker_id: str, worker_version: str, action_kind: str, canonical_arguments: object, context_hash: str, dependency_hash: str, plan_revision: int, prompt_hash: str, tool_hash: str, output_schema_hash: str, model_route: str, correction_ordinal: int, **extra: object) -> ActionFingerprint:
    """Hash only the fixed semantic action schema; reject all undeclared inputs."""
    try:
        _reject_extra(extra, "invalid action fingerprint input")
        if not _WORKER_ID_RE.fullmatch(worker_id) or not all(_IDENTIFIER_RE.fullmatch(value) and _safe_code(value) for value in (worker_version, action_kind, model_route)):
            raise ValueError("invalid action fingerprint input")
        values = _ActionFingerprintInput(worker_id=worker_id, worker_version=worker_version, action_kind=action_kind, canonical_arguments=_arguments(canonical_arguments), context_hash=context_hash, dependency_hash=dependency_hash, plan_revision=plan_revision, prompt_hash=prompt_hash, tool_hash=tool_hash, output_schema_hash=output_schema_hash, model_route=model_route, correction_ordinal=correction_ordinal)
    except Exception:
        raise ValueError("invalid action fingerprint input") from None
    return ActionFingerprint(digest=_digest({**values.model_dump(mode="json"), "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION}))


def build_result_fingerprint(*, outcome_kind: str, validated_output_hash: str | None, normalized_error_class: str | None, normalized_error_code: str | None, accepted_evidence_ids: Sequence[str] = (), tool_result_hashes: Sequence[str] = (), result_schema_version: str, **extra: object) -> ResultFingerprint:
    try:
        _reject_extra(extra, "invalid result fingerprint input")
        evidence = _bounded_sequence(accepted_evidence_ids, 64, "invalid result fingerprint input")
        tool_hashes = _bounded_sequence(tool_result_hashes, 64, "invalid result fingerprint input")
        if not all(isinstance(item, str) and _IDENTIFIER_RE.fullmatch(item) and _safe_code(item) for item in evidence):
            raise ValueError("invalid result fingerprint input")
        if not all(isinstance(item, str) and re.fullmatch(r"[0-9a-f]{64}", item) for item in tool_hashes):
            raise ValueError("invalid result fingerprint input")
        if not all(value is None or (isinstance(value, str) and _CODE_RE.fullmatch(value) and _safe_code(value)) for value in (outcome_kind, normalized_error_class, normalized_error_code, result_schema_version)):
            raise ValueError("invalid result fingerprint input")
        values = _ResultFingerprintInput(outcome_kind=outcome_kind, validated_output_hash=validated_output_hash, normalized_error_class=normalized_error_class, normalized_error_code=normalized_error_code, accepted_evidence_ids=tuple(sorted(evidence)), tool_result_hashes=tuple(sorted(tool_hashes)), result_schema_version=result_schema_version)
    except Exception:
        raise ValueError("invalid result fingerprint input") from None
    return ResultFingerprint(digest=_digest({**values.model_dump(mode="json"), "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION}))


def build_state_fingerprint(*, run_state: str, work_item_state: str, attempt_state: str, stage_id: str, plan_revision: int, unresolved_work_ids: Sequence[str], dependency_versions: Sequence[tuple[str, int]], progress: ProgressVector, normalized_error_class: str | None, **extra: object) -> StateFingerprint:
    try:
        _reject_extra(extra, "invalid state fingerprint input")
        unresolved = _bounded_sequence(unresolved_work_ids, 64, "invalid state fingerprint input")
        dependencies = _bounded_sequence(dependency_versions, 64, "invalid state fingerprint input")
        if not all(isinstance(item, str) and _IDENTIFIER_RE.fullmatch(item) and _safe_code(item) for item in unresolved):
            raise ValueError("invalid state fingerprint input")
        if not all(isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str) and _IDENTIFIER_RE.fullmatch(item[0]) and _safe_code(item[0]) and type(item[1]) is int and item[1] >= 0 for item in dependencies):
            raise ValueError("invalid state fingerprint input")
        if not all(isinstance(value, str) and _CODE_RE.fullmatch(value) and _safe_code(value) for value in (run_state, work_item_state, attempt_state, stage_id)) or (normalized_error_class is not None and (not isinstance(normalized_error_class, str) or not _CODE_RE.fullmatch(normalized_error_class) or not _safe_code(normalized_error_class))):
            raise ValueError("invalid state fingerprint input")
        values = _StateFingerprintInput(run_state=run_state, work_item_state=work_item_state, attempt_state=attempt_state, stage_id=stage_id, plan_revision=plan_revision, unresolved_work_ids=tuple(sorted(unresolved)), dependency_versions=tuple(sorted(dependencies)), progress=progress, normalized_error_class=normalized_error_class)
    except Exception:
        raise ValueError("invalid state fingerprint input") from None
    return StateFingerprint(digest=_digest({**values.model_dump(mode="json"), "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION}))


def detect_cycle(states: Sequence[str]) -> tuple[str, ...] | None:
    recent = tuple(states)[-STATE_WINDOW_SIZE:]
    for length in range(1, min(6, len(recent) // 2) + 1):
        if recent[-2 * length:-length] == recent[-length:]:
            return _normalize_rotation(recent[-length:])
    return None


def compare_progress(before: ProgressVector, after: ProgressVector, policy: SeverityPolicy) -> ProgressDecision:
    worsened = tuple(field for field in POSITIVE_FIELDS if getattr(after, field) < getattr(before, field)) + tuple(field for field in NEGATIVE_FIELDS if getattr(after, field) > getattr(before, field))
    critical = any(field in policy.critical_positive_regressions for field in worsened) or any(field in policy.critical_negative_regressions for field in worsened)
    if critical:
        return ProgressDecision(advanced=False, critical_regression=True, worsened_fields=worsened)
    advanced = not worsened and any(getattr(after, field) != getattr(before, field) for field in ALL_PROGRESS_FIELDS)
    return ProgressDecision(advanced=advanced, worsened_fields=worsened)


class LoopGuard:
    """Pure bounded policy, partitioned by the closed scope enum."""
    def __init__(self, *, severity_policy: SeverityPolicy) -> None:
        self._severity_policy = severity_policy
        self._actions = {scope: deque(maxlen=ACTION_WINDOW_SIZE) for scope in LoopScope}
        self._states = {scope: deque(maxlen=STATE_WINDOW_SIZE) for scope in LoopScope}
        self._progress: dict[LoopScope, ProgressVector | None] = {scope: None for scope in LoopScope}
        self._no_progress = {scope: 0 for scope in LoopScope}
        self._failures = {scope: deque(maxlen=3) for scope in LoopScope}
        self._observed_cycles = {scope: OrderedDict() for scope in LoopScope}
        self._recoveries = {scope: OrderedDict() for scope in LoopScope}

    def observe_action_result(self, observation: ActionObservationFingerprint) -> LoopDecision:
        actions = self._actions[observation.scope]
        actions.append(observation)
        if sum(item.digest == observation.digest for item in actions) >= 3:
            return _stopped("repeated_action_result")
        if sum(item.action.digest == observation.action.digest for item in actions) >= 3:
            return _stopped("repeated_action")
        return _allowed()

    def observe_checkpoint(self, checkpoint: SemanticCheckpoint) -> LoopDecision:
        if not checkpoint.is_semantic:
            return LoopDecision(allowed=True, ignored=True)
        scope = checkpoint.scope
        previous = self._progress[scope]
        comparison = compare_progress(previous, checkpoint.progress, self._severity_policy) if previous else ProgressDecision(advanced=False)
        self._progress[scope] = checkpoint.progress
        states = self._states[scope]
        if comparison.advanced:
            self._no_progress[scope] = 0
            states.clear()
        elif previous is not None:
            self._no_progress[scope] = min(2, self._no_progress[scope] + 1)
        if comparison.critical_regression:
            return _stopped("critical_progress_regression")
        duplicate = checkpoint.state_fingerprint.digest in states and not comparison.advanced
        states.append(checkpoint.state_fingerprint.digest)
        signature = self._cycle(checkpoint, states)
        if signature is not None:
            if duplicate and len(signature.period) == 1:
                return _stopped("duplicate_state_no_progress")
            if signature.digest in self._recoveries[scope]:
                return _stopped("recovered_cycle_returned", signature)
            return _stopped("state_cycle", signature)
        if duplicate:
            return _stopped("duplicate_state_no_progress")
        failure = self._failure(checkpoint)
        if failure:
            return failure
        if self._no_progress[scope] >= 2:
            return _stopped("no_progress")
        return _allowed()

    def authorize_recovery(self, signature: CycleSignature) -> LoopDecision:
        scope = signature.scope
        observed = self._observed_cycles[scope]
        if observed.get(signature.digest) != signature:
            return _stopped("unregistered_cycle_signature", signature)
        recoveries = self._recoveries[scope]
        if signature.digest in recoveries:
            return _stopped("recovery_exhausted", signature)
        if len(recoveries) >= RECOVERY_SIGNATURE_CAPACITY:
            return _stopped("recovery_capacity_exhausted", signature)
        recoveries[signature.digest] = signature
        return _allowed()

    def _cycle(self, checkpoint: SemanticCheckpoint, states: Sequence[str]) -> CycleSignature | None:
        period = detect_cycle(states)
        if period is None:
            return None
        values = {"scope": checkpoint.scope.value, "plan_revision": checkpoint.plan_revision, "fingerprint_schema_version": checkpoint.fingerprint_schema_version, "period": period}
        signature = CycleSignature(digest=_digest(values), scope=checkpoint.scope, plan_revision=checkpoint.plan_revision, fingerprint_schema_version=checkpoint.fingerprint_schema_version, period=period)
        observed = self._observed_cycles[checkpoint.scope]
        if signature.digest not in observed and len(observed) < RECOVERY_SIGNATURE_CAPACITY:
            observed[signature.digest] = signature
        return signature

    def _failure(self, checkpoint: SemanticCheckpoint) -> LoopDecision | None:
        failures = self._failures[checkpoint.scope]
        if checkpoint.normalized_failure is None:
            failures.clear()
            return None
        failures.append((checkpoint.worker_id, checkpoint.normalized_failure, checkpoint.failure_context_hash, checkpoint.failure_dependency_hash, checkpoint.correction_ordinal, checkpoint.model_route))
        if len(failures) < 3:
            return None
        workers = tuple(item[0] for item in failures)
        facts = tuple(item[1:] for item in failures)
        if workers[0] == workers[2] != workers[1] and len(set(facts)) == 1:
            return _stopped("cross_worker_failure_oscillation")
        return None


def _allowed() -> LoopDecision:
    return LoopDecision(allowed=True)


def _stopped(reason: str, signature: CycleSignature | None = None) -> LoopDecision:
    return LoopDecision(allowed=False, stop_reason=reason, cycle_signature=signature)
