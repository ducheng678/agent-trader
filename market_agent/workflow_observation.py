"""Host-owned, append-only observations for coordinated workflow execution.

Business results never carry settlement authority.  This module instead keeps
provider attempts and graph checkpoints in a closed immutable stream that can
be bound to the Harness event chain.
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from hashlib import sha256
import json
from threading import RLock
from typing import Any, Callable, Literal, Mapping, Self

from pydantic import BaseModel, ConfigDict, model_validator

from market_agent.openai_usage import (
    UsageTokens as PricedUsageTokens,
    estimate_workflow_usage_cost,
)

from market_agent.workflow_contracts import (
    Digest,
    NonNegativeFinite,
    NonNegativeInt,
    PositiveInt,
    ShortText,
    WorkflowResult,
)


class ObservationModel(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
        revalidate_instances="always",
    )
    schema_version: Literal["v1"] = "v1"

    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        values = {name: getattr(self, name) for name in type(self).model_fields}
        values.update(update or {})
        return type(self).model_validate(values)


class CoreNodeName(str, Enum):
    PLAN = "plan"
    DISPATCH = "dispatch"
    RECOVER = "recover"
    DECIDE = "decide"
    REFLECT = "reflect"
    RISK = "risk"
    ASSEMBLE = "assemble"


class NodeOutcome(str, Enum):
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class CheckpointDecision(str, Enum):
    CONTINUE = "continue"
    STOP = "stop"
    DEGRADE = "degrade"


class TokenUsage(ObservationModel):
    input_tokens: NonNegativeInt
    cached_input_tokens: NonNegativeInt = 0
    cache_write_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt
    web_search_tool_calls: NonNegativeInt = 0

    @model_validator(mode="after")
    def validate_cached_tokens(self) -> TokenUsage:
        if self.cached_input_tokens > self.input_tokens:
            raise ValueError("cached input tokens cannot exceed input tokens")
        return self

    @classmethod
    def zero(cls) -> TokenUsage:
        return cls(input_tokens=0, output_tokens=0)


ProviderSource = Literal["provider_response", "provider_usage_unavailable"]
NonProviderSource = Literal[
    "fixed_cache", "semantic_cache", "local_knowledge", "abstention", "historical_cache"
]
UsageSource = ProviderSource | NonProviderSource
ModelTierName = Literal["luna", "terra", "sol"]


class AttemptUsage(ObservationModel):
    workflow_id: ShortText
    trace_id: ShortText
    task_id: ShortText
    attempt: NonNegativeInt
    node: CoreNodeName
    provider: ShortText
    provider_request_id: ShortText
    model_id: ShortText
    model_tier: ModelTierName | None
    pricing_version: ShortText
    pricing_model_id: Literal[
        "gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"
    ] | None = None
    pricing_band: Literal["short", "long"] | None = None
    tokens: TokenUsage | None
    estimated_cost_usd: NonNegativeFinite
    latency_ms: NonNegativeInt
    source: UsageSource

    @model_validator(mode="after")
    def validate_source_shape(self) -> AttemptUsage:
        provider_source = self.source in {
            "provider_response",
            "provider_usage_unavailable",
        }
        if provider_source and self.model_tier is None:
            raise ValueError("provider attempts require a model tier")
        if provider_source and (
            self.pricing_model_id is None or self.pricing_band is None
        ):
            raise ValueError("provider attempts require pinned pricing identity")
        if provider_source and self.pricing_model_id != f"gpt-5.6-{self.model_tier}":
            raise ValueError("provider pricing model does not match model tier")
        if self.source == "provider_response" and self.tokens is None:
            raise ValueError("provider responses require verified token usage")
        if self.source == "provider_response" and self.pricing_version != "openai-standard-2026-08-01":
            raise ValueError("provider response pricing version is not pinned")
        if self.source == "provider_usage_unavailable" and self.tokens is not None:
            raise ValueError("unavailable provider usage cannot fabricate tokens")
        if not provider_source:
            if self.model_tier is not None:
                raise ValueError("non-provider execution cannot carry a model tier")
            if self.tokens != TokenUsage.zero() or self.estimated_cost_usd != 0.0:
                raise ValueError("non-provider execution must carry explicit zero usage")
            if self.pricing_model_id is not None or self.pricing_band is not None:
                raise ValueError("non-provider execution cannot carry pricing identity")
        if self.source == "provider_response":
            assert self.tokens is not None
            expected = estimate_workflow_usage_cost(
                self.pricing_model_id,
                self.pricing_band,
                PricedUsageTokens(
                    input_tokens=self.tokens.input_tokens,
                    cached_input_tokens=self.tokens.cached_input_tokens,
                    cache_write_tokens=self.tokens.cache_write_tokens,
                    output_tokens=self.tokens.output_tokens,
                    web_search_tool_calls=self.tokens.web_search_tool_calls,
                ),
            )
            if Decimal(str(self.estimated_cost_usd)) != expected:
                raise ValueError("provider response cost does not match exact pinned pricing")
        return self

    @property
    def is_provider_attempt(self) -> bool:
        return self.source in {"provider_response", "provider_usage_unavailable"}


def _aggregate(attempts: tuple[AttemptUsage, ...]) -> TokenUsage:
    fields = (
        "input_tokens",
        "cached_input_tokens",
        "cache_write_tokens",
        "output_tokens",
        "web_search_tool_calls",
    )
    totals = {
        field: sum(getattr(attempt.tokens, field) for attempt in attempts if attempt.tokens is not None)
        for field in fields
    }
    return TokenUsage(**totals)


def _usage_values(attempts: tuple[AttemptUsage, ...]) -> dict[str, object]:
    cost = sum((Decimal(str(item.estimated_cost_usd)) for item in attempts), Decimal("0"))
    provider_attempts = tuple(item for item in attempts if item.is_provider_attempt)
    return {
        "aggregate": _aggregate(attempts),
        "estimated_cost_usd": float(cost),
        "provider_attempt_count": len(provider_attempts),
        "unverified_provider_attempt_count": sum(
            item.source == "provider_usage_unavailable" for item in attempts
        ),
        "execution_count": len(attempts),
        "total_latency_ms": sum(item.latency_ms for item in attempts),
        "model_versions": tuple(
            sorted({(item.model_id, item.pricing_version) for item in provider_attempts})
        ),
    }


class WorkflowUsage(ObservationModel):
    workflow_id: ShortText
    trace_id: ShortText
    attempts: tuple[AttemptUsage, ...] = ()
    aggregate: TokenUsage
    estimated_cost_usd: NonNegativeFinite
    provider_attempt_count: NonNegativeInt
    unverified_provider_attempt_count: NonNegativeInt
    execution_count: NonNegativeInt
    total_latency_ms: NonNegativeInt
    model_versions: tuple[tuple[ShortText, ShortText], ...] = ()

    @classmethod
    def from_attempts(
        cls,
        workflow_id: str,
        trace_id: str,
        attempts: tuple[AttemptUsage, ...],
    ) -> WorkflowUsage:
        checked = tuple(AttemptUsage.model_validate(item) for item in attempts)
        return cls(
            workflow_id=workflow_id,
            trace_id=trace_id,
            attempts=checked,
            **_usage_values(checked),
        )

    @classmethod
    def empty(cls, workflow_id: str, trace_id: str) -> WorkflowUsage:
        return cls.from_attempts(workflow_id, trace_id, ())

    @model_validator(mode="after")
    def validate_canonical_fold(self) -> WorkflowUsage:
        keys: dict[tuple[str, int, str], AttemptUsage] = {}
        for attempt in self.attempts:
            if (attempt.workflow_id, attempt.trace_id) != (
                self.workflow_id,
                self.trace_id,
            ):
                raise ValueError("attempt identity does not match workflow usage")
            key = (attempt.task_id, attempt.attempt, attempt.provider_request_id)
            if key in keys:
                raise ValueError("workflow usage attempts must be unique")
            keys[key] = attempt
        expected = _usage_values(self.attempts)
        for field in (
            "aggregate",
            "estimated_cost_usd",
            "provider_attempt_count",
            "unverified_provider_attempt_count",
            "execution_count",
            "total_latency_ms",
            "model_versions",
        ):
            if getattr(self, field) != expected[field]:
                raise ValueError("workflow usage is not the exact canonical fold")
        return self


class TaskRetryState(ObservationModel):
    task_id: ShortText
    attempts_consumed: NonNegativeInt
    retries_consumed: NonNegativeInt
    retries_remaining: NonNegativeInt

    @model_validator(mode="after")
    def validate_retry_accounting(self) -> TaskRetryState:
        expected_retries = max(0, self.attempts_consumed - 1)
        if self.retries_consumed != expected_retries:
            raise ValueError("retry counters do not match attempts consumed")
        return self


class ObservedWorkItem(ObservationModel):
    task_id: ShortText
    task_kind: ShortText
    worker_id: ShortText
    owner_node: CoreNodeName
    dependency_ids: tuple[ShortText, ...] = ()
    maximum_retries: NonNegativeInt
    execution_state: Literal["pending", "running", "succeeded", "failed", "cancelled"]
    attempt_ids: tuple[ShortText, ...] = ()

    @model_validator(mode="after")
    def validate_dependencies(self) -> ObservedWorkItem:
        if (
            self.task_id in self.dependency_ids
            or len(self.dependency_ids) != len(set(self.dependency_ids))
            or len(self.attempt_ids) != len(set(self.attempt_ids))
        ):
            raise ValueError("work-item dependencies must be unique and external")
        return self


class NodeCheckpoint(ObservationModel):
    workflow_id: ShortText
    trace_id: ShortText
    plan_revision: NonNegativeInt
    ordinal: PositiveInt
    node: CoreNodeName
    outcome: NodeOutcome
    task_ids: tuple[ShortText, ...]
    completed_task_ids: tuple[ShortText, ...] = ()
    failed_task_ids: tuple[ShortText, ...] = ()
    action_fingerprint: Digest
    work_items: tuple[ObservedWorkItem, ...] = ()
    retry_state: tuple[TaskRetryState, ...] = ()
    usage: WorkflowUsage

    @model_validator(mode="after")
    def validate_checkpoint(self) -> NodeCheckpoint:
        if (self.usage.workflow_id, self.usage.trace_id) != (
            self.workflow_id,
            self.trace_id,
        ):
            raise ValueError("checkpoint usage identity does not match")
        if len(self.task_ids) != len(set(self.task_ids)):
            raise ValueError("checkpoint task identities must be unique")
        task_ids = set(self.task_ids)
        if not set(self.completed_task_ids) <= task_ids or not set(self.failed_task_ids) <= task_ids:
            raise ValueError("checkpoint progress must reference checkpoint tasks")
        if set(self.completed_task_ids) & set(self.failed_task_ids):
            raise ValueError("checkpoint completed and failed progress must be disjoint")
        retry_ids = tuple(state.task_id for state in self.retry_state)
        work_item_ids = tuple(item.task_id for item in self.work_items)
        if task_ids and (set(retry_ids) != task_ids or set(work_item_ids) != task_ids):
            raise ValueError("checkpoint tasks require complete retry and work-item bindings")
        if len(retry_ids) != len(set(retry_ids)) or len(work_item_ids) != len(set(work_item_ids)):
            raise ValueError("checkpoint task bindings must be unique")
        return self

    def canonical_digest(self) -> str:
        return sha256(json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")).hexdigest()


class CheckpointPermit(ObservationModel):
    workflow_id: ShortText
    trace_id: ShortText
    checkpoint_ordinal: PositiveInt
    checkpoint_digest: Digest
    decision: CheckpointDecision
    reason_code: ShortText

    def authorizes_continuation(self, checkpoint: NodeCheckpoint) -> bool:
        return (
            self.workflow_id == checkpoint.workflow_id
            and self.trace_id == checkpoint.trace_id
            and self.checkpoint_ordinal == checkpoint.ordinal
            and self.checkpoint_digest == checkpoint.canonical_digest()
            and self.decision is CheckpointDecision.CONTINUE
        )


CheckpointSink = Callable[[NodeCheckpoint], CheckpointPermit]


class ExecutionObservationCollector:
    """Append-only run-scoped collector; model output has no access to it."""

    def __init__(
        self,
        workflow_id: str,
        trace_id: str,
        *,
        checkpoint_sink: CheckpointSink | None = None,
        prior_checkpoints: tuple[NodeCheckpoint, ...] = (),
    ) -> None:
        if not workflow_id or not trace_id or (
            checkpoint_sink is not None and not callable(checkpoint_sink)
        ):
            raise ValueError("collector requires workflow identity and a callable sink")
        self._workflow_id = workflow_id
        self._trace_id = trace_id
        self._checkpoints: list[NodeCheckpoint] = [
            NodeCheckpoint.model_validate(item) for item in prior_checkpoints
        ]
        if any(
            (item.workflow_id, item.trace_id) != (workflow_id, trace_id)
            for item in self._checkpoints
        ) or tuple(item.ordinal for item in self._checkpoints) != tuple(
            range(1, len(self._checkpoints) + 1)
        ):
            raise ValueError("prior checkpoints do not form this workflow prefix")
        seeded_attempts = (
            self._checkpoints[-1].usage.attempts if self._checkpoints else ()
        )
        self._attempts: list[AttemptUsage] = list(seeded_attempts)
        self._attempts_by_key: dict[tuple[str, int, str], AttemptUsage] = {
            (item.task_id, item.attempt, item.provider_request_id): item
            for item in seeded_attempts
        }
        self._checkpoint_sink = checkpoint_sink
        self._lock = RLock()

    def record_attempt(self, value: AttemptUsage) -> AttemptUsage:
        attempt = AttemptUsage.model_validate(value)
        if (attempt.workflow_id, attempt.trace_id) != (
            self._workflow_id,
            self._trace_id,
        ):
            raise ValueError("attempt belongs to another workflow")
        key = (attempt.task_id, attempt.attempt, attempt.provider_request_id)
        with self._lock:
            existing = self._attempts_by_key.get(key)
            if existing is not None:
                if existing != attempt:
                    raise ValueError("conflicting duplicate attempt observation")
                return existing
            self._attempts.append(attempt)
            self._attempts_by_key[key] = attempt
            return attempt

    def usage(self) -> WorkflowUsage:
        with self._lock:
            return WorkflowUsage.from_attempts(
                self._workflow_id, self._trace_id, tuple(self._attempts)
            )

    def checkpoint(
        self,
        *,
        plan_revision: int,
        node: CoreNodeName,
        outcome: NodeOutcome,
        task_ids: tuple[str, ...],
        completed_task_ids: tuple[str, ...],
        failed_task_ids: tuple[str, ...],
        retry_state: tuple[TaskRetryState, ...],
        action_fingerprint: str,
        work_items: tuple[ObservedWorkItem, ...] = (),
    ) -> CheckpointPermit:
        with self._lock:
            checkpoint = NodeCheckpoint(
                workflow_id=self._workflow_id,
                trace_id=self._trace_id,
                plan_revision=plan_revision,
                ordinal=len(self._checkpoints) + 1,
                node=node,
                outcome=outcome,
                task_ids=task_ids,
                completed_task_ids=completed_task_ids,
                failed_task_ids=failed_task_ids,
                action_fingerprint=action_fingerprint,
                work_items=work_items,
                retry_state=retry_state,
                usage=self.usage(),
            )
            if self._checkpoint_sink is not None:
                permit = CheckpointPermit.model_validate(
                    self._checkpoint_sink(checkpoint)
                )
                if (
                    permit.workflow_id != checkpoint.workflow_id
                    or permit.trace_id != checkpoint.trace_id
                    or permit.checkpoint_ordinal != checkpoint.ordinal
                    or permit.checkpoint_digest != checkpoint.canonical_digest()
                ):
                    raise ValueError("checkpoint permit does not bind the checkpoint")
            else:
                permit = CheckpointPermit(
                    workflow_id=checkpoint.workflow_id,
                    trace_id=checkpoint.trace_id,
                    checkpoint_ordinal=checkpoint.ordinal,
                    checkpoint_digest=checkpoint.canonical_digest(),
                    decision=CheckpointDecision.CONTINUE,
                    reason_code="local_checkpoint_authorized",
                )
            self._checkpoints.append(checkpoint)
            return permit

    def checkpoints(self) -> tuple[NodeCheckpoint, ...]:
        with self._lock:
            return tuple(self._checkpoints)


class WorkflowExecution(ObservationModel):
    result: WorkflowResult
    usage: WorkflowUsage
    checkpoints: tuple[NodeCheckpoint, ...]
    completion_kind: Literal["graph", "historical_cache"] = "graph"
    prompt_release_digest: Digest | None = None

    @model_validator(mode="after")
    def validate_execution(self) -> WorkflowExecution:
        identity = (self.result.workflow_id, self.result.trace_id)
        if identity != (self.usage.workflow_id, self.usage.trace_id):
            raise ValueError("workflow execution usage identity does not match result")
        if any((item.workflow_id, item.trace_id) != identity for item in self.checkpoints):
            raise ValueError("workflow execution checkpoint identity does not match result")
        if tuple(item.ordinal for item in self.checkpoints) != tuple(
            range(1, len(self.checkpoints) + 1)
        ):
            raise ValueError("workflow execution checkpoints are not contiguous")
        if self.checkpoints and self.checkpoints[-1].usage != self.usage:
            raise ValueError("terminal checkpoint does not bind aggregate usage")
        legal = (
            CoreNodeName.PLAN, CoreNodeName.DISPATCH, CoreNodeName.RECOVER,
            CoreNodeName.DECIDE, CoreNodeName.REFLECT, CoreNodeName.RISK,
            CoreNodeName.ASSEMBLE,
        )
        if tuple(item.node for item in self.checkpoints) != legal[:len(self.checkpoints)]:
            raise ValueError("workflow checkpoints are not a legal graph prefix")
        for prior, current in zip(self.checkpoints, self.checkpoints[1:]):
            revision_delta = current.plan_revision - prior.plan_revision
            if revision_delta < 0 or revision_delta > 1 or (
                revision_delta and current.node is not CoreNodeName.RECOVER
            ):
                raise ValueError("only recovery may advance the plan revision")
            if current.usage.attempts[:len(prior.usage.attempts)] != prior.usage.attempts:
                raise ValueError("workflow checkpoint usage is not append-only")
        if self.completion_kind == "graph" and len(self.checkpoints) == 0:
            raise ValueError("graph execution requires checkpoints")
        if self.completion_kind == "historical_cache" and (
            self.checkpoints or self.usage.provider_attempt_count != 0
        ):
            raise ValueError("historical cache execution must be explicit and provider-free")
        return self
