from __future__ import annotations

import json
from hashlib import sha256
from typing import Annotated

from pydantic import Field, StrictFloat, ValidationError, model_validator

from market_agent.workflow_agent_contracts import AgentInvocation, ModelTier, StrictModel
from market_agent.workflow_contracts import Digest, ShortText


def canonical_json(value: object) -> str:
    """Serialize validated dynamic user content in one deterministic form."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)


Temperature = Annotated[StrictFloat, Field(ge=0.0, le=2.0)]


class PromptRelease(StrictModel):
    release_id: ShortText
    digest: Digest
    stable_system_prefix: ShortText
    supported_task_kinds: tuple[ShortText, ...] = Field(min_length=1, max_length=32)
    supported_model_tiers: tuple[ModelTier, ...] = Field(min_length=1, max_length=3)
    temperature_profile: tuple[tuple[ModelTier, Temperature], ...] = Field(min_length=1, max_length=3)

    @model_validator(mode="after")
    def validate_temperature_profile(self) -> PromptRelease:
        temperatures = dict(self.temperature_profile)
        if len(set(self.supported_model_tiers)) != len(self.supported_model_tiers):
            raise ValueError("supported model tiers cannot repeat a model tier")
        if len(temperatures) != len(self.temperature_profile):
            raise ValueError("temperature profile cannot repeat a model tier")
        if set(temperatures) != set(self.supported_model_tiers):
            raise ValueError("temperature profile must cover exactly the supported model tiers")
        # Pin every immutable release field, including schema/version and routing
        # policy. A caller-supplied label cannot authorize different prompt content.
        content = self.model_dump(mode="json", exclude={"digest"})
        if self.digest != sha256(canonical_json(content).encode("utf-8")).hexdigest():
            raise ValueError("prompt release digest does not match canonical release content")
        return self

    def temperature_for(self, tier: ModelTier) -> float:
        for configured_tier, temperature in self.temperature_profile:
            if configured_tier is tier:
                return temperature
        raise ValueError("model tier is not supported by prompt release")


class PromptReleaseRegistry(StrictModel):
    releases: tuple[PromptRelease, ...] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def reject_duplicate_release_ids(self) -> PromptReleaseRegistry:
        release_ids = tuple(release.release_id for release in self.releases)
        if len(release_ids) != len(set(release_ids)):
            raise ValueError("prompt release IDs must be unique")
        return self

    def select(self, invocation: AgentInvocation) -> PromptRelease:
        invocation = AgentInvocation.model_validate(invocation)
        release = next((item for item in self.releases if item.release_id == invocation.prompt_release_id), None)
        if release is None:
            raise ValidationError.from_exception_data(
                "PromptReleaseRegistry",
                [{"type": "value_error", "loc": ("prompt_release_id",), "input": invocation.prompt_release_id, "ctx": {"error": ValueError("unknown prompt release")}}],
            )
        release = PromptRelease.model_validate(release)
        if release.digest != invocation.prompt_release_digest:
            raise ValidationError.from_exception_data(
                "PromptReleaseRegistry",
                [{"type": "value_error", "loc": ("prompt_release_digest",), "input": invocation.prompt_release_digest, "ctx": {"error": ValueError("prompt release digest does not match")}}],
            )
        if invocation.task_kind not in release.supported_task_kinds:
            raise ValidationError.from_exception_data(
                "PromptReleaseRegistry",
                [{"type": "value_error", "loc": ("task_kind",), "input": invocation.task_kind, "ctx": {"error": ValueError("task kind is not supported by prompt release")}}],
            )
        if invocation.allowed_model_tier not in release.supported_model_tiers:
            raise ValidationError.from_exception_data(
                "PromptReleaseRegistry",
                [{"type": "value_error", "loc": ("allowed_model_tier",), "input": invocation.allowed_model_tier, "ctx": {"error": ValueError("model tier is not supported by prompt release")}}],
            )
        return release

    def render(self, invocation: AgentInvocation) -> tuple[str, str]:
        validated_invocation = AgentInvocation.model_validate(invocation)
        release = self.select(validated_invocation)
        return release.stable_system_prefix, canonical_json(validated_invocation.user_payload)
