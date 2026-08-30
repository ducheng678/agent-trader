from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
from decimal import Decimal

import pytest

from market_agent.workflow_contracts import WorkflowMode


def test_policy_for_uses_explicit_ordered_routes_and_luna_only_reflection():
    from market_agent.workflow_model_routing import UnknownWorkflowNodeError, policy_for

    assert [tier.model for tier in policy_for("fundamental").tiers] == ["gpt-5.6-terra", "gpt-5.6-luna"]
    assert [tier.model for tier in policy_for("escalation").tiers] == ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"]
    for node_name in ("fundamental_reflection", "technical_reflection", "decision_reflection"):
        assert [tier.model for tier in policy_for(node_name).tiers] == ["gpt-5.6-luna"]
    with pytest.raises(UnknownWorkflowNodeError):
        policy_for("fundamental_like")


def test_policy_is_immutable_and_carries_authoritative_node_caps():
    from market_agent.workflow_model_routing import policy_for

    policy = policy_for("fundamental")
    assert (policy.attempt_timeout_seconds, policy.node_timeout_seconds) == (35, 95)
    assert (policy.maximum_attempts_per_tier, policy.maximum_total_attempts) == (2, 3)
    assert (policy.maximum_output_tokens, policy.node_cost_cap) == (900, Decimal("0.08"))
    with pytest.raises(FrozenInstanceError):
        policy.node_cost_cap = Decimal("1.00")


def test_workflow_pricing_requires_explicit_band_and_preserves_cache_write_cost():
    from market_agent.openai_usage import UsageTokens, estimate_workflow_usage_cost

    usage = UsageTokens(input_tokens=1_000_000, output_tokens=1_000_000)
    assert estimate_workflow_usage_cost("gpt-5.6-terra", "short", usage) == Decimal("14.00")
    assert estimate_workflow_usage_cost("gpt-5.6-terra", "long", usage) == Decimal("22.00")
    assert estimate_workflow_usage_cost("gpt-5.6-terra", "short", UsageTokens(input_tokens=1_000_000, cache_write_tokens=1_000_000)) == Decimal("4.50")
    with pytest.raises(ValueError):
        estimate_workflow_usage_cost("gpt-5.6-terra", "automatic", usage)
    with pytest.raises(ValueError):
        UsageTokens(input_tokens=1.5)


def test_reservation_prevents_node_preoverspend_and_settlement_releases_unused_cost():
    from market_agent.workflow_budget import BudgetExceededError, WorkflowBudgetLedger
    from market_agent.openai_usage import UsageTokens

    ledger = WorkflowBudgetLedger(WorkflowMode.ACTIVE)
    reservation = ledger.reserve(node_name="fundamental", model="gpt-5.6-terra", band="short", usage=UsageTokens(input_tokens=20_000, output_tokens=900))
    reserved_cost = reservation.reserved_cost
    assert ledger.snapshot().reserved_cost == reserved_cost
    settled = ledger.settle(reservation, UsageTokens(input_tokens=20_000, output_tokens=100))
    after = ledger.snapshot()
    assert settled.charged_cost < reserved_cost
    assert (after.reserved_cost, after.settled_cost) == (Decimal("0"), settled.charged_cost)
    with pytest.raises(BudgetExceededError):
        ledger.reserve(node_name="fundamental", model="gpt-5.6-terra", band="short", usage=UsageTokens(input_tokens=40_000, output_tokens=900))


def test_parallel_reservations_cannot_overrun_node_cap():
    from market_agent.workflow_budget import BudgetExceededError, WorkflowBudgetLedger
    from market_agent.openai_usage import UsageTokens

    ledger = WorkflowBudgetLedger(WorkflowMode.ACTIVE)

    def reserve_once():
        try:
            return ledger.reserve(node_name="fundamental", model="gpt-5.6-terra", band="short", usage=UsageTokens(input_tokens=10_000, output_tokens=900))
        except BudgetExceededError:
            return None

    with ThreadPoolExecutor(max_workers=8) as executor:
        reservations = list(executor.map(lambda _: reserve_once(), range(8)))
    assert sum(reservation is not None for reservation in reservations) == 2
    assert ledger.snapshot().reserved_cost <= Decimal("0.08")


def test_timeout_consumes_reservation_or_known_usage_and_attempts_are_bounded():
    from market_agent.workflow_budget import BudgetExceededError, WorkflowBudgetLedger
    from market_agent.openai_usage import UsageTokens

    ledger = WorkflowBudgetLedger(WorkflowMode.ACTIVE)
    timed_out = ledger.reserve(node_name="fundamental", model="gpt-5.6-terra", band="short", usage=UsageTokens(input_tokens=10_000, output_tokens=900))
    assert ledger.consume_timeout(timed_out).charged_cost == timed_out.reserved_cost
    known_usage = ledger.reserve(node_name="fundamental", model="gpt-5.6-terra", band="short", usage=UsageTokens(input_tokens=10_000, output_tokens=900))
    assert ledger.consume_timeout(known_usage, UsageTokens(input_tokens=10_000, output_tokens=1)).charged_cost < known_usage.reserved_cost
    with pytest.raises(BudgetExceededError):
        ledger.reserve(node_name="fundamental", model="gpt-5.6-terra", band="short", usage=UsageTokens(input_tokens=10_000, output_tokens=900))


def test_ledger_rejects_foreign_and_duplicate_settlement():
    from market_agent.workflow_budget import ReservationOwnershipError, ReservationStateError, WorkflowBudgetLedger
    from market_agent.openai_usage import UsageTokens

    first = WorkflowBudgetLedger(WorkflowMode.ACTIVE)
    second = WorkflowBudgetLedger(WorkflowMode.ACTIVE)
    reservation = first.reserve(node_name="fundamental", model="gpt-5.6-terra", band="short", usage=UsageTokens(input_tokens=1_000, output_tokens=900))
    with pytest.raises(ReservationOwnershipError):
        second.settle(reservation, UsageTokens(input_tokens=1_000, output_tokens=1))
    first.settle(reservation, UsageTokens(input_tokens=1_000, output_tokens=1))
    with pytest.raises(ReservationStateError):
        first.settle(reservation, UsageTokens(input_tokens=1_000, output_tokens=1))


def test_reservation_deadlines_use_monotonic_time_and_snapshot_is_immutable():
    from market_agent.workflow_budget import BudgetExceededError, WorkflowBudgetLedger
    from market_agent.openai_usage import UsageTokens

    now = [100.0]
    ledger = WorkflowBudgetLedger(WorkflowMode.ACTIVE, clock=lambda: now[0])
    ledger.reserve(node_name="fundamental", model="gpt-5.6-terra", band="short", usage=UsageTokens(input_tokens=1_000, output_tokens=900))
    now[0] = 161.0
    with pytest.raises(BudgetExceededError):
        ledger.reserve(node_name="fundamental", model="gpt-5.6-terra", band="short", usage=UsageTokens(input_tokens=1_000, output_tokens=900))
    with pytest.raises(FrozenInstanceError):
        ledger.snapshot().remaining_cost = Decimal("1")


def test_legacy_usage_pricing_exposes_the_workflow_model_short_band():
    from market_agent.openai_usage import get_openai_model_pricing

    pricing = get_openai_model_pricing("gpt-5.6-sol")
    assert pricing == {
        "input": 4.0,
        "cached_input": 0.4,
        "cache_write": 5.0,
        "output": 20.0,
    }


def test_reservation_accounts_for_authorized_maximum_web_tool_calls():
    from market_agent.openai_usage import UsageTokens, estimate_workflow_usage_cost

    cost = estimate_workflow_usage_cost(
        "gpt-5.6-terra",
        "short",
        UsageTokens(input_tokens=1_000, output_tokens=1, web_search_tool_calls=3),
    )
    assert cost == Decimal("0.032012")


def test_market_context_reservation_charges_authorized_tool_call_ceiling():
    from market_agent.workflow_budget import WorkflowBudgetLedger
    from market_agent.openai_usage import UsageTokens, estimate_workflow_usage_cost

    usage = UsageTokens(input_tokens=1_000, output_tokens=1_200)
    reservation = WorkflowBudgetLedger(WorkflowMode.ACTIVE).reserve(
        node_name="market_context",
        model="gpt-5.6-terra",
        band="short",
        usage=usage,
        maximum_tool_calls=3,
    )
    expected = estimate_workflow_usage_cost(
        "gpt-5.6-terra",
        "short",
        UsageTokens(input_tokens=1_000, output_tokens=1_200, web_search_tool_calls=3),
    )
    assert reservation.reserved_cost == expected
