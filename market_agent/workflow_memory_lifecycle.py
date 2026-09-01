"""Deterministic, dry-run-first forgetting through a trusted repository adapter."""
from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Literal, Protocol, Unpack

from pydantic import AwareDatetime, TypeAdapter, model_validator

from market_agent.workflow_contracts import Digest, FiniteUnit, NonNegativeInt, PositiveInt, ShortText
from market_agent.workflow_long_term_memory import (
    ArtifactReference, ArtifactStore, Lifecycle, MemoryAuthorityError, MemoryConflictError,
    MemoryContract, MemoryRepository, MutationContext, Record, WriteArguments, content_hash,
)


class LifecyclePolicy(MemoryContract):
    standard_retention_seconds: PositiveInt = 90 * 86400
    short_retention_seconds: PositiveInt = 7 * 86400
    standard_half_life_seconds: PositiveInt = 30 * 86400
    short_half_life_seconds: PositiveInt = 86400
    archive_grace_seconds: PositiveInt = 86400
    tombstone_grace_seconds: PositiveInt = 7 * 86400
    min_confidence: FiniteUnit = 0.1
    max_live_records: PositiveInt = 10000


class LifecycleScope(MemoryContract):
    tenant_id: ShortText
    scope: ShortText | None = None


class LifecycleEntry(MemoryContract):
    record: Record
    changed_at: AwareDatetime | None = None
    referenced_by: tuple[ShortText, ...] = ()
    is_knowledge_head: bool = False


class LifecycleAction(MemoryContract):
    record_id: ShortText
    expected_hash: Digest
    kind: Literal["archive", "tombstone", "purge"]
    reason: Literal["expired", "retention", "decay", "capacity", "archive_grace", "tombstone_grace"]


class LifecyclePlan(MemoryContract):
    scope: LifecycleScope
    now: AwareDatetime
    policy_hash: Digest
    actions: tuple[LifecycleAction, ...] = ()
    plan_hash: Digest | None = None

    @model_validator(mode="after")
    def validate_plan(self):
        if len({action.record_id for action in self.actions}) != len(self.actions):
            raise ValueError("a plan may advance each record only once")
        order = {"archive": 0, "tombstone": 1, "purge": 2}
        if tuple(sorted(self.actions, key=lambda item: (order[item.kind], item.record_id))) != self.actions:
            raise ValueError("lifecycle actions must be ordered archive, tombstone, purge")
        digest = content_hash(self.model_dump(mode="json", exclude={"plan_hash"}))
        if self.plan_hash is not None and self.plan_hash != digest:
            raise ValueError("lifecycle plan hash mismatch")
        object.__setattr__(self, "plan_hash", digest)
        return self

    @property
    def archive_ids(self):
        return tuple(action.record_id for action in self.actions if action.kind == "archive")

    @property
    def tombstone_ids(self):
        return tuple(action.record_id for action in self.actions if action.kind == "tombstone")

    @property
    def purge_ids(self):
        return tuple(action.record_id for action in self.actions if action.kind == "purge")


class LifecycleLimits(MemoryContract):
    max_actions: NonNegativeInt = 100
    max_cleanup: NonNegativeInt = 100


class CleanupTask(MemoryContract):
    task_id: Digest
    tenant_id: ShortText
    scope: ShortText
    trace_id: ShortText
    record_id: ShortText
    record_hash: Digest
    kind: Literal["vector", "cache", "artifact"]
    artifact: ArtifactReference | None = None

    @model_validator(mode="after")
    def validate_artifact(self):
        if (self.kind == "artifact") != (self.artifact is not None):
            raise ValueError("only artifact cleanup carries an artifact reference")
        if self.artifact is not None and self.artifact.tenant_id != self.tenant_id:
            raise ValueError("artifact cleanup must share tenant")
        return self


class LifecycleResult(MemoryContract):
    applied_ids: tuple[ShortText, ...] = ()
    skipped_ids: tuple[ShortText, ...] = ()
    cleaned_ids: tuple[Digest, ...] = ()
    pending_cleanup: NonNegativeInt = 0


class LifecycleRepository(MemoryRepository, Protocol):
    def validate_mutation(self, **context: Unpack[WriteArguments]) -> MutationContext: ...
    def lifecycle_snapshot(self, scope: LifecycleScope) -> tuple[LifecycleEntry, ...]: ...
    def apply_lifecycle(self, plan: LifecyclePlan, policy: LifecyclePolicy, limits: LifecycleLimits,
                        **context: Unpack[WriteArguments]) -> LifecycleResult: ...
    def list_cleanup(self, *, tenant_id: str, scope: str | None = None) -> tuple[CleanupTask, ...]: ...
    def begin_cleanup(self, task: CleanupTask, **context: Unpack[WriteArguments]) -> bool: ...
    def finish_cleanup(self, task: CleanupTask, **context: Unpack[WriteArguments]) -> None: ...


def effective_confidence(record: Record, now: datetime, policy: LifecyclePolicy) -> float:
    now = TypeAdapter(AwareDatetime).validate_python(now, strict=True)
    policy = LifecyclePolicy.model_validate(policy)
    confidence = getattr(record, "confidence", 1.0)
    if record.retention_class == "permanent":
        return confidence
    half_life = (policy.short_half_life_seconds if record.retention_class == "short"
                 else policy.standard_half_life_seconds)
    age = max(0.0, (now - record.observed_at).total_seconds())
    return confidence * 2 ** (-age / half_life)


def build_lifecycle_plan(entries: tuple[LifecycleEntry, ...], scope: LifecycleScope,
                         now: datetime, policy: LifecyclePolicy) -> LifecyclePlan:
    """Pure selection also used inside the repository's write transaction."""
    now = TypeAdapter(AwareDatetime).validate_python(now, strict=True)
    actions = {}
    candidates = []
    live_count = 0
    for entry in entries:
        record = entry.record
        if record.tenant_id != scope.tenant_id or (scope.scope is not None and record.scope != scope.scope):
            continue
        live = record.lifecycle in (Lifecycle.ACTIVE, Lifecycle.PROPOSED)
        live_count += live
        if record.legal_hold or entry.referenced_by or record.retention_class == "permanent" or record.observed_at > now:
            continue
        kind = reason = None
        if live:
            retention = (policy.short_retention_seconds if record.retention_class == "short"
                         else policy.standard_retention_seconds)
            confidence = effective_confidence(record, now, policy)
            if record.expires_at is not None and record.expires_at <= now:
                reason = "expired"
            elif (now - record.observed_at).total_seconds() >= retention:
                reason = "retention"
            elif confidence < policy.min_confidence:
                reason = "decay"
            if reason:
                kind = "archive"
            else:
                candidates.append((confidence, record.observed_at, record.record_id, record))
        else:
            # Unknown legacy transition times cannot prove the required grace
            # elapsed; such rows remain protected until their age is established.
            elapsed = (now - entry.changed_at).total_seconds() if entry.changed_at else 0
            if record.lifecycle is Lifecycle.ARCHIVED and elapsed >= policy.archive_grace_seconds:
                kind, reason = "tombstone", "archive_grace"
            elif (record.lifecycle is Lifecycle.TOMBSTONED and not entry.is_knowledge_head
                  and elapsed >= policy.tombstone_grace_seconds):
                kind, reason = "purge", "tombstone_grace"
        if kind:
            actions[record.record_id] = LifecycleAction(record_id=record.record_id,
                expected_hash=content_hash(record.model_dump(mode="json")), kind=kind, reason=reason)
    excess = max(0, live_count - len([action for action in actions.values() if action.kind == "archive"])
                 - policy.max_live_records)
    for _, _, _, record in sorted(candidates)[:excess]:
        actions[record.record_id] = LifecycleAction(record_id=record.record_id,
            expected_hash=content_hash(record.model_dump(mode="json")), kind="archive", reason="capacity")
    order = {"archive": 0, "tombstone": 1, "purge": 2}
    return LifecyclePlan(scope=scope, now=now, policy_hash=content_hash(policy.model_dump(mode="json")),
                         actions=tuple(sorted(actions.values(), key=lambda item: (order[item.kind], item.record_id))))


class LifecycleWorker:
    def __init__(self, repository: LifecycleRepository, *, policy: LifecyclePolicy | None = None,
                 artifact_store: ArtifactStore | None = None,
                 cleanup_adapters: Mapping[str, Callable] | None = None):
        self._repository = repository
        self._policy = LifecyclePolicy.model_validate(policy or LifecyclePolicy())
        self._artifacts = artifact_store
        self._cleanup_adapters = dict(cleanup_adapters or {})
        if set(self._cleanup_adapters) - {"vector", "cache"}:
            raise ValueError("cleanup adapters support only vector and cache derivatives")

    def plan(self, scope: str | LifecycleScope, now: datetime) -> LifecyclePlan:
        scope = LifecycleScope(tenant_id=scope) if type(scope) is str else LifecycleScope.model_validate(scope)
        return build_lifecycle_plan(self._repository.lifecycle_snapshot(scope), scope, now, self._policy)

    def apply(self, plan: LifecyclePlan, limits: LifecycleLimits,
              **context: Unpack[WriteArguments]) -> LifecycleResult:
        plan, limits = LifecyclePlan.model_validate(plan), LifecycleLimits.model_validate(limits)
        ctx = self._repository.validate_mutation(**context)
        if ctx.tenant_id != plan.scope.tenant_id:
            raise MemoryAuthorityError("lifecycle tenant does not match mutation scope")
        if plan.policy_hash != content_hash(self._policy.model_dump(mode="json")):
            raise MemoryConflictError("lifecycle policy changed since planning")
        result = self._repository.apply_lifecycle(plan, self._policy, limits, **context)
        cleaned = []
        tasks = self._repository.list_cleanup(tenant_id=ctx.tenant_id, scope=plan.scope.scope)
        attempted = 0
        for task in tasks:
            adapter = self._cleanup_adapters.get(task.kind)
            if task.kind == "artifact":
                if self._artifacts is None:
                    continue
            elif adapter is None:
                continue
            if attempted >= limits.max_cleanup:
                break
            cleanup_context = dict(context, trace_id=task.trace_id, idempotency_key=task.task_id)
            if not self._repository.begin_cleanup(task, **cleanup_context):
                continue
            attempted += 1
            try:
                if task.kind == "artifact":
                    self._artifacts.delete(task.artifact, **cleanup_context)
                else:
                    # Adapters must treat task_id as an idempotency key: a crash
                    # after deletion but before acknowledgement replays this call.
                    adapter(task, **cleanup_context)
            except Exception:
                # Failure leaves the durable outbox entry pending for a later run.
                continue
            self._repository.finish_cleanup(task, **cleanup_context)
            cleaned.append(task.task_id)
        pending = self._repository.list_cleanup(tenant_id=ctx.tenant_id, scope=plan.scope.scope)
        return result.model_copy(update={"cleaned_ids": tuple(cleaned), "pending_cleanup": len(pending)})
