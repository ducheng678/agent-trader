"""Deterministic, in-memory loop prevention for semantic harness observations."""

from __future__ import annotations

from collections import OrderedDict, deque
from hashlib import sha256
import json
from typing import Any, Literal, Sequence

from pydantic import Field, field_validator, model_validator

from market_agent.workflow_contracts import ContractModel, Digest, NonNegativeInt, ShortText
from market_agent.workflow_harness_contracts import ProgressVector


FINGERPRINT_SCHEMA_VERSION = "v1"
STATE_WINDOW_SIZE = 12
ACTION_WINDOW_SIZE = 5
RECOVERY_SIGNATURE_CAPACITY = 12

POSITIVE_FIELDS = (
    "completed_dependency_count",
    "valid_required_field_count",
    "filled_required_evidence_slot_count",
    "fresh_authoritative_source_coverage",
)
NEGATIVE_FIELDS = (
    "missing_evidence_count",
    "validation_error_count",
    "unresolved_conflict_count",
    "risk_invariant_failure_count",
)
ALL_PROGRESS_FIELDS = POSITIVE_FIELDS + NEGATIVE_FIELDS
LoopScope = Literal["attempt", "work_item", "stage", "run"]


def _canonical_digest(value: object) -> str:
    """Hash a deterministic JSON projection; callers supply semantic values only."""

    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


def _canonical_mapping(value: object) -> object:
    """Return a JSON-only semantic projection without transient or sensitive values."""

    normalized = json.loads(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    )
    return _remove_excluded_values(normalized)


def _remove_excluded_values(value: object) -> object:
    if isinstance(value, dict):
        return {
            key: _remove_excluded_values(item)
            for key, item in value.items()
            if not _is_excluded_key(key)
        }
    if isinstance(value, list):
        return [_remove_excluded_values(item) for item in value]
    return value


def _is_excluded_key(key: object) -> bool:
    """Reject common raw, secret, reasoning, and ephemeral identifier keys."""

    if not isinstance(key, str):
        return True
    normalized = key.lower().replace("-", "_")
    if normalized.endswith("_hash") or normalized.endswith("_digest"):
        return False
    tokens = set(normalized.split("_"))
    return bool(
        tokens
        & {
            "event",
            "attempt",
            "call",
            "trace",
            "span",
            "lease",
            "timestamp",
            "random",
            "secret",
            "raw",
            "reasoning",
        }
    ) or normalized in {"content", "private_reasoning"}


class _PublicContract(ContractModel):
    """Frozen public values that revalidate model_copy updates at the boundary."""

    def model_copy(self, *, update: dict[str, Any] | None = None, deep: bool = False) -> Any:
        values = self.model_dump(mode="python", round_trip=True)
        if update:
            values.update(update)
        return type(self).model_validate(values)


class _Fingerprint(_PublicContract):
    digest: Digest

    @property
    def value(self) -> str:
        """Compatibility-friendly read-only spelling for the canonical digest."""

        return self.digest


class ActionFingerprint(_Fingerprint):
    pass


class ResultFingerprint(_Fingerprint):
    pass


class StateFingerprint(_Fingerprint):
    pass


class ActionObservationFingerprint(_Fingerprint):
    action: ActionFingerprint
    result: ResultFingerprint

    @classmethod
    def from_parts(
        cls, action: ActionFingerprint, result: ResultFingerprint
    ) -> ActionObservationFingerprint:
        return cls(
            digest=_canonical_digest(
                {
                    "action_fingerprint": action.digest,
                    "result_fingerprint": result.digest,
                    "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
                }
            ),
            action=action,
            result=result,
        )

    @model_validator(mode="after")
    def validate_digest(self) -> ActionObservationFingerprint:
        expected = _canonical_digest(
            {
                "action_fingerprint": self.action.digest,
                "result_fingerprint": self.result.digest,
                "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
            }
        )
        if self.digest != expected:
            raise ValueError("action observation digest must bind its action and result")
        return self


class CycleSignature(_Fingerprint):
    scope: LoopScope
    plan_revision: NonNegativeInt
    fingerprint_schema_version: ShortText
    period: tuple[Digest, ...] = Field(min_length=1, max_length=6)

    @field_validator("period")
    @classmethod
    def validate_canonical_rotation(cls, period: tuple[str, ...]) -> tuple[str, ...]:
        canonical = _rotation_normalize(period)
        if tuple(period) != canonical:
            raise ValueError("cycle period must use its lexicographically smallest rotation")
        return tuple(period)

    @model_validator(mode="after")
    def validate_digest(self) -> CycleSignature:
        expected = _canonical_digest(
            {
                "scope": self.scope,
                "plan_revision": self.plan_revision,
                "fingerprint_schema_version": self.fingerprint_schema_version,
                "period": self.period,
            }
        )
        if self.digest != expected:
            raise ValueError("cycle digest must bind its scope, plan, schema, and period")
        return self


class SeverityPolicy(_PublicContract):
    policy_version: ShortText
    critical_positive_regressions: tuple[ShortText, ...] = (
        "filled_required_evidence_slot_count",
        "fresh_authoritative_source_coverage",
    )
    critical_negative_regressions: tuple[ShortText, ...] = (
        "validation_error_count",
        "risk_invariant_failure_count",
    )

    @model_validator(mode="after")
    def validate_declared_fields(self) -> SeverityPolicy:
        if (
            not set(self.critical_positive_regressions).issubset(POSITIVE_FIELDS)
            or not set(self.critical_negative_regressions).issubset(NEGATIVE_FIELDS)
        ):
            raise ValueError("severity fields must be declared progress dimensions")
        if len(set(self.critical_positive_regressions)) != len(
            self.critical_positive_regressions
        ) or len(set(self.critical_negative_regressions)) != len(
            self.critical_negative_regressions
        ):
            raise ValueError("severity fields must be unique")
        return self


class ProgressDecision(_PublicContract):
    advanced: bool
    critical_regression: bool = False
    worsened_fields: tuple[ShortText, ...] = ()

    @model_validator(mode="after")
    def validate_decision(self) -> ProgressDecision:
        if self.advanced and (self.critical_regression or self.worsened_fields):
            raise ValueError("advanced progress cannot contain a regression")
        if self.critical_regression and not self.worsened_fields:
            raise ValueError("critical regression requires a worsened dimension")
        return self


class SemanticCheckpoint(_PublicContract):
    """Only validated semantic observations enter LoopGuard state windows."""

    scope: LoopScope
    state_fingerprint: StateFingerprint
    progress: ProgressVector
    plan_revision: NonNegativeInt
    fingerprint_schema_version: ShortText = FINGERPRINT_SCHEMA_VERSION
    semantic: bool = True
    worker_id: ShortText | None = None
    normalized_failure: ShortText | None = None
    failure_context_hash: Digest | None = None
    failure_dependency_hash: Digest | None = None
    correction_ordinal: NonNegativeInt = 0
    model_route: ShortText | None = None


class LoopDecision(_PublicContract):
    allowed: bool
    stop_reason: ShortText | None = None
    cycle_signature: CycleSignature | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> LoopDecision:
        if self.allowed and self.stop_reason is not None:
            raise ValueError("allowed decisions cannot have a stop reason")
        if not self.allowed and self.stop_reason is None:
            raise ValueError("stopped decisions require a reason")
        return self


def build_action_fingerprint(
    *,
    worker_id: str,
    worker_version: str,
    action_kind: str,
    canonical_arguments: object,
    context_hash: str,
    dependency_hash: str,
    plan_revision: int,
    prompt_hash: str,
    tool_hash: str,
    output_schema_hash: str,
    model_route: str,
    correction_ordinal: int,
    **_ephemeral: object,
) -> ActionFingerprint:
    """Build an allowlisted action hash, intentionally ignoring transient/raw inputs."""

    return ActionFingerprint(
        digest=_canonical_digest(
            {
                "worker_id": worker_id,
                "worker_version": worker_version,
                "action_kind": action_kind,
                "canonical_arguments": _canonical_mapping(canonical_arguments),
                "context_hash": context_hash,
                "dependency_hash": dependency_hash,
                "plan_revision": plan_revision,
                "prompt_hash": prompt_hash,
                "tool_hash": tool_hash,
                "output_schema_hash": output_schema_hash,
                "model_route": model_route,
                "correction_ordinal": correction_ordinal,
                "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
            }
        )
    )


def build_result_fingerprint(
    *,
    outcome_kind: str,
    validated_output_hash: str | None,
    normalized_error_class: str | None,
    normalized_error_code: str | None,
    accepted_evidence_ids: Sequence[str] = (),
    tool_result_hashes: Sequence[str] = (),
    result_schema_version: str,
    **_ephemeral: object,
) -> ResultFingerprint:
    """Build an allowlisted result hash without retaining raw model/tool content."""

    return ResultFingerprint(
        digest=_canonical_digest(
            {
                "outcome_kind": outcome_kind,
                "validated_output_hash": validated_output_hash,
                "normalized_error_class": normalized_error_class,
                "normalized_error_code": normalized_error_code,
                "accepted_evidence_ids": sorted(set(accepted_evidence_ids)),
                "tool_result_hashes": sorted(set(tool_result_hashes)),
                "result_schema_version": result_schema_version,
                "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
            }
        )
    )


def build_state_fingerprint(
    *,
    run_state: str,
    work_item_state: str,
    attempt_state: str,
    stage_id: str,
    plan_revision: int,
    unresolved_work_ids: Sequence[str],
    dependency_versions: Sequence[tuple[str, int]],
    progress: ProgressVector,
    normalized_error_class: str | None,
    **_ephemeral: object,
) -> StateFingerprint:
    """Build a state hash from semantic state, never run/work/attempt identity."""

    return StateFingerprint(
        digest=_canonical_digest(
            {
                "run_state": run_state,
                "work_item_state": work_item_state,
                "attempt_state": attempt_state,
                "stage_id": stage_id,
                "plan_revision": plan_revision,
                "unresolved_work_ids": sorted(set(unresolved_work_ids)),
                "dependency_versions": sorted(set(dependency_versions)),
                "progress": progress.model_dump(mode="json"),
                "normalized_error_class": normalized_error_class,
                "fingerprint_schema_version": FINGERPRINT_SCHEMA_VERSION,
            }
        )
    )


def _rotation_normalize(period: Sequence[str]) -> tuple[str, ...]:
    values = tuple(period)
    return min(values[index:] + values[:index] for index in range(len(values)))


def detect_cycle(states: Sequence[str]) -> tuple[str, ...] | None:
    """Find the shortest repeated tail, canonicalized independently of its rotation."""

    recent = tuple(states)[-STATE_WINDOW_SIZE:]
    for length in range(1, min(6, len(recent) // 2) + 1):
        if recent[-2 * length : -length] == recent[-length:]:
            return _rotation_normalize(recent[-length:])
    return None


def compare_progress(
    before: ProgressVector, after: ProgressVector, policy: SeverityPolicy
) -> ProgressDecision:
    """Compare frozen objective progress dimensions under a versioned severity policy."""

    worsened = tuple(
        field
        for field in POSITIVE_FIELDS
        if getattr(after, field) < getattr(before, field)
    ) + tuple(
        field
        for field in NEGATIVE_FIELDS
        if getattr(after, field) > getattr(before, field)
    )
    critical = any(field in policy.critical_positive_regressions for field in worsened) or any(
        field in policy.critical_negative_regressions for field in worsened
    )
    if critical:
        return ProgressDecision(
            advanced=False, critical_regression=True, worsened_fields=worsened
        )
    positive_ok = all(getattr(after, field) >= getattr(before, field) for field in POSITIVE_FIELDS)
    negative_ok = all(getattr(after, field) <= getattr(before, field) for field in NEGATIVE_FIELDS)
    strict = any(getattr(after, field) != getattr(before, field) for field in ALL_PROGRESS_FIELDS)
    return ProgressDecision(
        advanced=positive_ok and negative_ok and strict, worsened_fields=worsened
    )


class LoopGuard:
    """A deterministic bounded policy. It neither mutates authority nor budgets."""

    def __init__(self, *, severity_policy: SeverityPolicy) -> None:
        self._severity_policy = severity_policy
        self._semantic_actions: deque[ActionObservationFingerprint] = deque(
            maxlen=ACTION_WINDOW_SIZE
        )
        self._state_windows = {
            scope: deque(maxlen=STATE_WINDOW_SIZE)
            for scope in ("attempt", "work_item", "stage", "run")
        }
        self._previous_progress: dict[LoopScope, ProgressVector | None] = {
            scope: None for scope in ("attempt", "work_item", "stage", "run")
        }
        self._no_progress_counts: dict[LoopScope, int] = {
            scope: 0 for scope in ("attempt", "work_item", "stage", "run")
        }
        self._failure_workers: deque[tuple[str, str, str, str, int, str]] = deque(maxlen=3)
        self._recoveries: OrderedDict[str, CycleSignature] = OrderedDict()

    def observe_action_result(self, observation: ActionObservationFingerprint) -> LoopDecision:
        self._semantic_actions.append(observation)
        matching_observations = sum(
            item.digest == observation.digest for item in self._semantic_actions
        )
        if matching_observations >= 3:
            return _stopped("repeated_action_result")
        matching_actions = sum(
            item.action.digest == observation.action.digest for item in self._semantic_actions
        )
        if matching_actions >= 3:
            return _stopped("repeated_action")
        return _allowed()

    def observe_checkpoint(self, checkpoint: SemanticCheckpoint) -> LoopDecision:
        if not checkpoint.semantic:
            return _stopped("infrastructure_ignored")

        scope = checkpoint.scope
        previous = self._previous_progress[scope]
        comparison = (
            compare_progress(previous, checkpoint.progress, self._severity_policy)
            if previous is not None
            else ProgressDecision(advanced=False)
        )
        self._previous_progress[scope] = checkpoint.progress
        if comparison.advanced:
            self._no_progress_counts[scope] = 0
        elif previous is not None:
            self._no_progress_counts[scope] = min(2, self._no_progress_counts[scope] + 1)

        if comparison.critical_regression:
            return _stopped("critical_progress_regression")

        states = self._state_windows[scope]
        duplicate_without_progress = (
            bool(states)
            and states[-1] == checkpoint.state_fingerprint.digest
            and not comparison.advanced
        )
        states.append(checkpoint.state_fingerprint.digest)
        if duplicate_without_progress:
            return _stopped("duplicate_state_no_progress")

        signature = self._cycle_signature(checkpoint, states)
        if signature is not None:
            if signature.digest in self._recoveries:
                return _stopped("recovered_cycle_returned", signature)
            return _stopped("state_cycle", signature)

        failure_decision = self._observe_failure(checkpoint)
        if failure_decision is not None:
            return failure_decision
        if self._no_progress_counts[scope] >= 2:
            return _stopped("no_progress")
        return _allowed()

    def authorize_recovery(self, signature: CycleSignature) -> LoopDecision:
        if signature.digest in self._recoveries:
            return _stopped("recovery_exhausted", signature)
        if len(self._recoveries) >= RECOVERY_SIGNATURE_CAPACITY:
            return _stopped("recovery_capacity_exhausted", signature)
        self._recoveries[signature.digest] = signature
        return _allowed()

    def _cycle_signature(
        self, checkpoint: SemanticCheckpoint, states: Sequence[str]
    ) -> CycleSignature | None:
        period = detect_cycle(states)
        if period is None:
            return None
        values = {
            "scope": checkpoint.scope,
            "plan_revision": checkpoint.plan_revision,
            "fingerprint_schema_version": checkpoint.fingerprint_schema_version,
            "period": period,
        }
        return CycleSignature(digest=_canonical_digest(values), **values)

    def _observe_failure(self, checkpoint: SemanticCheckpoint) -> LoopDecision | None:
        if (
            checkpoint.normalized_failure is None
            or checkpoint.worker_id is None
            or checkpoint.failure_context_hash is None
            or checkpoint.failure_dependency_hash is None
            or checkpoint.model_route is None
        ):
            self._failure_workers.clear()
            return None
        self._failure_workers.append(
            (
                checkpoint.worker_id,
                checkpoint.normalized_failure,
                checkpoint.failure_context_hash,
                checkpoint.failure_dependency_hash,
                checkpoint.correction_ordinal,
                checkpoint.model_route,
            )
        )
        if len(self._failure_workers) < 3:
            return None
        workers = tuple(item[0] for item in self._failure_workers)
        failures = tuple(item[1] for item in self._failure_workers)
        contexts = tuple(item[2:] for item in self._failure_workers)
        if (
            len(set(failures)) == 1
            and len(set(contexts)) == 1
            and workers[0] == workers[2]
            and workers[0] != workers[1]
        ):
            return _stopped("cross_worker_failure_oscillation")
        return None


def _allowed() -> LoopDecision:
    return LoopDecision(allowed=True)


def _stopped(reason: str, signature: CycleSignature | None = None) -> LoopDecision:
    return LoopDecision(allowed=False, stop_reason=reason, cycle_signature=signature)
