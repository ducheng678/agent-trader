from __future__ import annotations

import json

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
        "prompt_release_digest": "a" * 64,
        "allowed_model_tier": ModelTier.LUNA,
        "user_payload": {"z": [2, 1], "a": "context"},
    }
    values.update(overrides)
    return AgentInvocation(**values)


def make_release(**overrides: object) -> PromptRelease:
    values = {
        "release_id": "release-1",
        "digest": "a" * 64,
        "stable_system_prefix": "Return only the declared JSON object.",
        "supported_task_kinds": ("extract",),
        "supported_model_tiers": (ModelTier.LUNA,),
    }
    values.update(overrides)
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
