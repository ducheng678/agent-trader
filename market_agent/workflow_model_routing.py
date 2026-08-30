from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Mapping


class UnknownWorkflowNodeError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ModelRouteTier:
    model: str
    effort: str

    def __post_init__(self) -> None:
        if self.model not in {"gpt-5.6-luna", "gpt-5.6-terra", "gpt-5.6-sol"}:
            raise ValueError("unknown workflow model")
        if self.effort not in {"low", "medium", "high"}:
            raise ValueError("invalid workflow effort")


@dataclass(frozen=True, slots=True)
class AgentExecutionPolicy:
    node_name: str
    tiers: tuple[ModelRouteTier, ...]
    attempt_timeout_seconds: int
    node_timeout_seconds: int
    maximum_attempts_per_tier: int
    maximum_total_attempts: int
    maximum_output_tokens: int
    node_cost_cap: Decimal
    maximum_tool_calls: int

    def __post_init__(self) -> None:
        if not self.node_name or not self.tiers:
            raise ValueError("workflow policy requires a node and tier")
        if self.attempt_timeout_seconds <= 0 or self.node_timeout_seconds < self.attempt_timeout_seconds:
            raise ValueError("workflow policy timeout is invalid")
        if self.maximum_attempts_per_tier <= 0 or self.maximum_total_attempts <= 0:
            raise ValueError('workflow policy attempt cap is invalid')
        if self.maximum_total_attempts > self.maximum_attempts_per_tier * len(self.tiers):
            raise ValueError('workflow policy attempt cap is unreachable')
        if self.maximum_output_tokens <= 0 or self.maximum_tool_calls < 0:
            raise ValueError("workflow policy output or tool cap is invalid")
        if not self.node_cost_cap.is_finite() or self.node_cost_cap < 0:
            raise ValueError("workflow policy cost cap is invalid")

    @property
    def max_output_tokens(self) -> int:
        return self.maximum_output_tokens

    @property
    def max_total_attempts(self) -> int:
        return self.maximum_total_attempts

    @property
    def max_tool_calls(self) -> int:
        return self.maximum_tool_calls


def _policy(
    node_name: str,
    tiers: tuple[ModelRouteTier, ...],
    attempt_timeout_seconds: int,
    node_timeout_seconds: int,
    maximum_attempts_per_tier: int,
    maximum_total_attempts: int,
    maximum_output_tokens: int,
    node_cost_cap: str,
    maximum_tool_calls: int,
) -> AgentExecutionPolicy:
    return AgentExecutionPolicy(
        node_name=node_name,
        tiers=tiers,
        attempt_timeout_seconds=attempt_timeout_seconds,
        node_timeout_seconds=node_timeout_seconds,
        maximum_attempts_per_tier=maximum_attempts_per_tier,
        maximum_total_attempts=maximum_total_attempts,
        maximum_output_tokens=maximum_output_tokens,
        node_cost_cap=Decimal(node_cost_cap),
        maximum_tool_calls=maximum_tool_calls,
    )


LUNA = ModelRouteTier("gpt-5.6-luna", "low")
TERRA = ModelRouteTier("gpt-5.6-terra", "medium")
SOL = ModelRouteTier("gpt-5.6-sol", "high")


_POLICIES: Mapping[str, AgentExecutionPolicy] = MappingProxyType(
    {
        'event_filter': _policy('event_filter', (LUNA,), 20, 55, 2, 2, 600, '0.02', 0),
        "market_context": _policy("market_context", (TERRA, LUNA), 60, 150, 2, 3, 1200, "0.20", 3),
        "fundamental": _policy("fundamental", (TERRA, LUNA), 35, 95, 2, 3, 900, "0.08", 0),
        "technical": _policy("technical", (TERRA, LUNA), 40, 105, 2, 3, 1400, "0.12", 0),
        "decision_planner": _policy("decision_planner", (TERRA, LUNA), 30, 85, 2, 3, 1100, "0.10", 0),
        "escalation": _policy("escalation", (SOL, TERRA, LUNA), 60, 155, 2, 4, 1400, "0.25", 0),
        "reflect_decision": _policy("reflect_decision", (LUNA,), 20, 45, 1, 1, 400, "0.02", 0),
        "reflect_escalation_if_used": _policy("reflect_escalation_if_used", (LUNA,), 20, 45, 1, 1, 400, "0.02", 0),
        "reflect_coordinator_summary": _policy("reflect_coordinator_summary", (LUNA,), 20, 45, 1, 1, 400, "0.02", 0),
    }
)


def policy_for(node_name: str) -> AgentExecutionPolicy:
    if not isinstance(node_name, str):
        raise UnknownWorkflowNodeError("workflow node name must be a string")
    try:
        return _POLICIES[node_name]
    except KeyError as exc:
        raise UnknownWorkflowNodeError(f"unknown workflow node: {node_name}") from exc


def policies() -> Mapping[str, AgentExecutionPolicy]:
    return _POLICIES
