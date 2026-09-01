from __future__ import annotations

import json
from hashlib import sha256

import pytest
from pydantic import ValidationError

from market_agent.workflow_agent_contracts import AgentInvocation, ModelTier
from market_agent.workflow_prompt_release import PromptRelease, PromptReleaseRegistry


def make_invocation(**overrides: object) -> AgentInvocation:
    values = {
        "trace_id": "trace-1",
        "run_id": "run-1",
        "task_id": "task-1",
        "task_kind": "extract",
        "prompt_release_id": "release-1",
        "prompt_release_digest": make_release().digest,
        "allowed_model_tier": ModelTier.LUNA,
        "user_payload": {"z": [2, 1], "a": "context"},
    }
    values.update(overrides)
    return AgentInvocation(**values)


def make_release(**overrides: object) -> PromptRelease:
    values = {
        "release_id": "release-1",
        "stable_system_prefix": "Return only the declared JSON object.",
        "supported_task_kinds": ("extract",),
        "supported_model_tiers": (ModelTier.LUNA,),
        "temperature_profile": ((ModelTier.LUNA, 0.0),),
    }
    values.update(overrides)
    if "digest" not in values:
        content = json.dumps({"schema_version": "v1", **values}, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        values["digest"] = sha256(content.encode("utf-8")).hexdigest()
    return PromptRelease(**values)


def test_render_returns_stable_system_prefix_before_canonical_dynamic_user_json():
    """Swapping prompt positions would destroy the provider cacheable prefix."""
    registry = PromptReleaseRegistry(releases=(make_release(),))

    system_prefix, user_content = registry.render(make_invocation())

    assert system_prefix == "Return only the declared JSON object."
    assert user_content == '{"a":"context","z":[2,1]}'
    assert json.loads(user_content) == {"a": "context", "z": [2, 1]}


def test_invocation_rejects_dynamic_system_values():
    """Accepting dynamic system context would make a pinned prefix unstable."""
    with pytest.raises(ValidationError):
        make_invocation(system_context={"tenant": "tenant-1"})


def test_render_rejects_a_model_tier_not_supported_by_the_release():
    """Ignoring release tier compatibility would route to an unapproved model."""
    registry = PromptReleaseRegistry(releases=(make_release(),))

    with pytest.raises(ValidationError):
        registry.render(make_invocation(allowed_model_tier=ModelTier.TERRA))


def test_render_revalidates_copied_payload_before_canonicalizing_it():
    """A copied invocation must not retain a caller-owned mutable JSON object."""
    registry = PromptReleaseRegistry(releases=(make_release(),))
    payload = {"a": "original"}
    copied = make_invocation().model_copy(update={"user_payload": payload})
    payload["a"] = "tampered"

    assert registry.render(copied)[1] == '{"a":"original"}'


def test_prompt_release_exposes_a_temperature_for_each_supported_tier():
    """Missing a tier temperature would leave a permitted driver call underspecified."""
    release = make_release(
        supported_model_tiers=(ModelTier.LUNA, ModelTier.TERRA),
        temperature_profile=((ModelTier.LUNA, 0.0), (ModelTier.TERRA, 0.7)),
    )
    registry = PromptReleaseRegistry(releases=(release,))

    assert registry.select(make_invocation(prompt_release_digest=release.digest)).temperature_for(ModelTier.LUNA) == 0.0
    with pytest.raises(ValidationError):
        make_release(
            supported_model_tiers=(ModelTier.LUNA, ModelTier.TERRA),
            temperature_profile=((ModelTier.LUNA, 0.0),),
        )


@pytest.mark.parametrize("temperature", [float("nan"), 2.1])
def test_prompt_release_rejects_nonfinite_or_out_of_range_temperatures(temperature: float):
    """An invalid temperature must not reach a model call through a release."""
    with pytest.raises(ValidationError):
        make_release(temperature_profile=((ModelTier.LUNA, temperature),))


@pytest.mark.parametrize("change", [
    {"stable_system_prefix": "Changed instructions."},
    {"supported_task_kinds": ("extract", "analyze")},
    {"temperature_profile": ((ModelTier.LUNA, 0.5),)},
    {"supported_model_tiers": (ModelTier.LUNA, ModelTier.TERRA),
     "temperature_profile": ((ModelTier.LUNA, 0.0), (ModelTier.TERRA, 0.7))},
    {"release_id": "release-2"},
])
@pytest.mark.parametrize("copied", [False, True])
def test_changed_release_content_cannot_reuse_a_pinned_digest(change, copied):
    """A stale digest must not alias changed prompt content, even via model_copy."""
    release = make_release()
    with pytest.raises(ValidationError, match="digest"):
        if copied:
            release.model_copy(update=change)
        else:
            make_release(digest=release.digest, **change)


@pytest.mark.parametrize("render", [False, True])
def test_registry_revalidates_forged_release_content_before_use(render):
    """Unchecked Pydantic construction must not smuggle a changed prompt past its pin."""
    release = make_release()
    forged = PromptRelease.model_construct(**{**release.__dict__, "stable_system_prefix": "Unpinned instructions."})
    with pytest.raises(ValidationError, match="digest"):
        if render:
            PromptReleaseRegistry.model_construct(releases=(forged,)).render(make_invocation())
        else:
            PromptReleaseRegistry(releases=(forged,))
