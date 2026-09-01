from __future__ import annotations

import pytest
from pydantic import ValidationError

from market_agent.workflow_agent_contracts import AgentInvocation, AgentResult, AgentUsage, ModelTier


def test_invocation_rejects_missing_trace_and_unbounded_limits():
    """Removing trace authority or a bounded execution limit must be rejected."""
    with pytest.raises(ValidationError):
        AgentInvocation(trace_id="", max_attempts=0, cost_limit_usd=-1)


def test_result_rejects_raw_text_and_extra_fields():
    """Provider prose is not a driver result, even when it resembles JSON."""
    with pytest.raises(ValidationError):
        AgentResult.model_validate(
            {
                "trace_id": "trace-1",
                "output": {"answer": "known"},
                "usage": {
                    "input_tokens": 1,
                    "output_tokens": 1,
                    "cost_usd": 0.01,
                    "model_tier": ModelTier.LUNA,
                },
                "raw_text": "```json {} ```",
            }
        )


def test_invocation_and_result_are_immutable():
    """Allowing a caller to alter the pinned request or result breaks auditability."""
    invocation = AgentInvocation(trace_id="trace-1", user_payload={"answer": "known"})
    result = AgentResult(
        trace_id="trace-1",
        output={"answer": "known"},
        usage=AgentUsage(input_tokens=1, output_tokens=1, cost_usd=0.01, model_tier=ModelTier.LUNA),
    )

    with pytest.raises(ValidationError):
        invocation.trace_id = "trace-2"
    with pytest.raises(TypeError):
        result.output["answer"] = "changed"


def test_model_copy_revalidates_invocation_and_result_updates():
    """Bypassing validation through model_copy would permit unauditable data."""
    invocation = AgentInvocation(trace_id="trace-1")
    result = AgentResult(
        trace_id="trace-1",
        output={"answer": "known"},
        usage=AgentUsage(input_tokens=1, output_tokens=1, cost_usd=0.01, model_tier=ModelTier.LUNA),
    )

    with pytest.raises(ValidationError):
        invocation.model_copy(update={"max_attempts": 0})
    with pytest.raises(ValidationError):
        result.model_copy(update={"raw_text": "```json {} ```"})
