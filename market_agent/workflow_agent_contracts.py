from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Literal, Mapping, Self

from pydantic import BaseModel, ConfigDict, Field, StrictBool, field_validator, model_validator

from market_agent.workflow_contracts import (
    Digest,
    FrozenJsonMapping,
    NonNegativeFinite,
    NonNegativeInt,
    PositiveFinite,
    PositiveInt,
    ShortText,
)


class StrictModel(BaseModel):
    """Immutable, closed models at the LLM-driver boundary."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        strict=True,
        allow_inf_nan=False,
        str_strip_whitespace=True,
        revalidate_instances="always",
    )
    schema_version: Literal["v1"] = "v1"

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        """Copy through the validation boundary instead of trusting Pydantic's fast path."""

        values = {field_name: getattr(self, field_name) for field_name in type(self).model_fields}
        if update is not None:
            values.update(update)
        return type(self).model_validate(values)


class ModelTier(str, Enum):
    LUNA = "luna"
    TERRA = "terra"
    SOL = "sol"


BoundedAttempts = Annotated[PositiveInt, Field(le=10)]
BoundedCostUsd = Annotated[NonNegativeFinite, Field(le=10.0)]


def thaw_json(value: object) -> object:
    """Restore frozen request JSON to standard JSON containers for revalidation."""

    if isinstance(value, dict):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [thaw_json(item) for item in value]
    return value


class AgentInvocation(StrictModel):
    """Pinned, capability-free request for one bounded model invocation."""

    trace_id: ShortText
    run_id: ShortText = "run"
    task_id: ShortText = "task"
    task_kind: ShortText = "extract"
    prompt_release_id: ShortText = "default"
    prompt_release_digest: Digest = "0" * 64
    allowed_model_tier: ModelTier = ModelTier.LUNA
    deadline_epoch: PositiveFinite = 1.0
    attempt: NonNegativeInt = 0
    max_attempts: BoundedAttempts = 1
    cost_limit_usd: BoundedCostUsd = 0.0
    output_schema_id: ShortText = "default"
    output_schema_digest: Digest = "0" * 64
    user_payload: FrozenJsonMapping = Field(default_factory=dict)

    @field_validator("user_payload", mode="before")
    @classmethod
    def require_json_object(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise ValueError("user_payload must be a JSON object")
        return thaw_json(value)

    @model_validator(mode="after")
    def validate_attempt_limit(self) -> AgentInvocation:
        if self.attempt >= self.max_attempts:
            raise ValueError("attempt must be below max_attempts")
        return self


class AgentUsage(StrictModel):
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    cost_usd: NonNegativeFinite
    model_tier: ModelTier


class AgentFailure(StrictModel):
    trace_id: ShortText
    code: ShortText
    message: ShortText
    retryable: StrictBool


class AgentResult(StrictModel):
    """Normalized structured result; raw provider content is never representable."""

    trace_id: ShortText
    origin: ShortText = "model"
    output: FrozenJsonMapping | None = None
    usage: AgentUsage | None = None
    failure: AgentFailure | None = None

    @model_validator(mode="after")
    def validate_terminal_variant(self) -> AgentResult:
        if self.failure is not None:
            if self.output is not None or self.usage is not None:
                raise ValueError("failed results cannot include output or usage")
            if self.failure.trace_id != self.trace_id:
                raise ValueError("failure trace_id must match result trace_id")
            return self
        if self.output is None or self.usage is None:
            raise ValueError("successful results require output and usage")
        return self
