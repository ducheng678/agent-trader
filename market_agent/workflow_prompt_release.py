from __future__ import annotations

import json

from pydantic import Field, ValidationError, model_validator

from market_agent.workflow_agent_contracts import AgentInvocation, ModelTier, StrictModel
from market_agent.workflow_contracts import Digest, ShortText


def canonical_json(value: object) -> str:
    """Serialize validated dynamic user content in one deterministic form."""

    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)


class PromptRelease(StrictModel):
    release_id: ShortText
    digest: Digest
    stable_system_prefix: ShortText
    supported_task_kinds: tuple[ShortText, ...] = Field(min_length=1, max_length=32)
    supported_model_tiers: tuple[ModelTier, ...] = Field(min_length=1, max_length=3)


class PromptReleaseRegistry(StrictModel):
    releases: tuple[PromptRelease, ...] = Field(min_length=1, max_length=128)

    @model_validator(mode="after")
    def reject_duplicate_release_ids(self) -> PromptReleaseRegistry:
        release_ids = tuple(release.release_id for release in self.releases)
        if len(release_ids) != len(set(release_ids)):
            raise ValueError("prompt release IDs must be unique")
        return self

    def render(self, invocation: AgentInvocation) -> tuple[str, str]:
        release = next((item for item in self.releases if item.release_id == invocation.prompt_release_id), None)
        if release is None:
            raise ValidationError.from_exception_data(
                "PromptReleaseRegistry",
                [{"type": "value_error", "loc": ("prompt_release_id",), "input": invocation.prompt_release_id, "ctx": {"error": ValueError("unknown prompt release")}}],
            )
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
        return release.stable_system_prefix, canonical_json(invocation.user_payload)
