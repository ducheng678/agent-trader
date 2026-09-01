from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest

from market_agent.openai_usage import UsageTokens
from market_agent.workflow_contracts import WorkflowMode


@pytest.fixture
def ledger():
    from market_agent.workflow_budget import WorkflowBudgetLedger

    return WorkflowBudgetLedger(WorkflowMode.ACTIVE)


@pytest.fixture
def reservation(ledger):
    return ledger.reserve(
        node_name="fundamental",
        model="gpt-5.6-terra",
        band="short",
        usage=UsageTokens(input_tokens=1, output_tokens=1),
    )


@pytest.fixture
def overflow_usage():
    return UsageTokens(input_tokens=2, output_tokens=1)


def consume_global_attempts_across_distinct_nodes(ledger):
    from market_agent.workflow_model_routing import policies

    node_names = tuple(policies())
    for node_name in node_names:
        ledger.reserve(
            node_name=node_name,
            model="gpt-5.6-luna",
            band="short",
            usage=UsageTokens(),
        )
    ledger.reserve(
        node_name=node_names[0],
        model="gpt-5.6-luna",
        band="short",
        usage=UsageTokens(),
    )


def test_policy_catalog_matches_the_core_reflection_graph_and_is_immutable():
    from market_agent.workflow_model_routing import UnknownWorkflowNodeError, policies, policy_for

    expected_nodes = {
        "event_filter", "market_context", "fundamental", "technical", "decision_planner", "escalation",
        "reflect_decision", "reflect_escalation_if_used", "reflect_coordinator_summary",
    }
    assert set(policies()) == expected_nodes
    assert [tier.model for tier in policy_for("fundamental").tiers] == ["gpt-5.6-terra", "gpt-5.6-luna"]
    assert [tier.model for tier in policy_for("escalation").tiers] == ["gpt-5.6-sol", "gpt-5.6-terra", "gpt-5.6-luna"]
    for node_name in ("reflect_decision", "reflect_escalation_if_used", "reflect_coordinator_summary"):
        policy = policy_for(node_name)
        assert [tier.model for tier in policy.tiers] == ["gpt-5.6-luna"]
        assert policy.maximum_tool_calls == 0
    with pytest.raises(UnknownWorkflowNodeError):
        policy_for("fundamental_reflection")
    with pytest.raises(UnknownWorkflowNodeError):
        policy_for("technical_reflection")
    with pytest.raises(TypeError):
        policies()["new_node"] = policy_for("fundamental")


def test_policy_is_immutable_and_carries_authoritative_node_caps():
    from market_agent.workflow_model_routing import policy_for

    policy = policy_for("fundamental")
    assert (policy.attempt_timeout_seconds, policy.node_timeout_seconds) == (35, 95)
    assert (policy.maximum_attempts_per_tier, policy.maximum_total_attempts) == (2, 3)
    assert (policy.maximum_output_tokens, policy.node_cost_cap) == (900, Decimal("0.08"))
    with pytest.raises(FrozenInstanceError):
        policy.node_cost_cap = Decimal("1.00")


@pytest.mark.parametrize(
    ("model", "band", "field", "expected"),
    [
        ("gpt-5.6-sol", "short", "input", Decimal("4.00")), ("gpt-5.6-sol", "short", "cached_input", Decimal("0.40")),
        ("gpt-5.6-sol", "short", "cache_write", Decimal("5.00")), ("gpt-5.6-sol", "short", "output", Decimal("20.00")),
        ("gpt-5.6-sol", "long", "input", Decimal("8.00")), ("gpt-5.6-sol", "long", "cached_input", Decimal("0.80")),
        ("gpt-5.6-sol", "long", "cache_write", Decimal("10.00")), ("gpt-5.6-sol", "long", "output", Decimal("30.00")),
        ("gpt-5.6-terra", "short", "input", Decimal("2.00")), ("gpt-5.6-terra", "short", "cached_input", Decimal("0.20")),
        ("gpt-5.6-terra", "short", "cache_write", Decimal("2.50")), ("gpt-5.6-terra", "short", "output", Decimal("12.00")),
        ("gpt-5.6-terra", "long", "input", Decimal("4.00")), ("gpt-5.6-terra", "long", "cached_input", Decimal("0.40")),
        ("gpt-5.6-terra", "long", "cache_write", Decimal("5.00")), ("gpt-5.6-terra", "long", "output", Decimal("18.00")),
        ("gpt-5.6-luna", "short", "input", Decimal("0.20")), ("gpt-5.6-luna", "short", "cached_input", Decimal("0.02")),
        ("gpt-5.6-luna", "short", "cache_write", Decimal("0.25")), ("gpt-5.6-luna", "short", "output", Decimal("1.20")),
        ("gpt-5.6-luna", "long", "input", Decimal("0.40")), ("gpt-5.6-luna", "long", "cached_input", Decimal("0.04")),
        ("gpt-5.6-luna", "long", "cache_write", Decimal("0.50")), ("gpt-5.6-luna", "long", "output", Decimal("1.80")),
    ],
)
def test_workflow_pricing_uses_all_explicit_decimal_price_components(model, band, field, expected):
    from market_agent.openai_usage import workflow_model_pricing

    assert getattr(workflow_model_pricing(model, band), field) == expected


def test_workflow_pricing_mapping_is_immutable_at_both_levels():
    from market_agent.openai_usage import WORKFLOW_MODEL_PRICING_USD_PER_1M

    with pytest.raises(TypeError):
        WORKFLOW_MODEL_PRICING_USD_PER_1M["other"] = {}
    with pytest.raises(TypeError):
        WORKFLOW_MODEL_PRICING_USD_PER_1M["gpt-5.6-terra"]["short"] = None


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


def test_legacy_pricing_does_not_select_an_implicit_workflow_band():
    from market_agent.openai_usage import get_openai_model_pricing

    assert get_openai_model_pricing("gpt-5.6-sol") is None
    assert get_openai_model_pricing("gpt-5.4") == {"input": 1.25, "cached_input": 0.125, "output": 10.0}


def test_web_tool_prices_use_one_decimal_native_environment_source(monkeypatch):
    from market_agent.openai_usage import UsageTokens, estimate_workflow_usage_cost, get_openai_web_search_tool_price_usd_per_1k, get_openai_web_search_tool_price_usd_per_1k_decimal

    monkeypatch.setenv("OPENAI_WEB_SEARCH_TOOL_PRICE_PER_1K_USD", "3.125")
    assert get_openai_web_search_tool_price_usd_per_1k_decimal() == Decimal("3.125")
    assert get_openai_web_search_tool_price_usd_per_1k() == 3.125
    assert estimate_workflow_usage_cost("gpt-5.6-luna", "short", UsageTokens(web_search_tool_calls=2)) == Decimal("0.00625")


def test_reservation_prevents_node_preoverspend_and_settlement_releases_unused_cost():
    from market_agent.openai_usage import UsageTokens
    from market_agent.workflow_budget import BudgetExceededError, WorkflowBudgetLedger

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


def test_failed_first_reservation_does_not_start_a_node_deadline_or_insert_a_node():
    from market_agent.openai_usage import UsageTokens
    from market_agent.workflow_budget import BudgetExceededError, WorkflowBudgetLedger

    now = [100.0]
    ledger = WorkflowBudgetLedger(WorkflowMode.ACTIVE, clock=lambda: now[0])
    with pytest.raises(BudgetExceededError):
        ledger.reserve(node_name="fundamental", model="gpt-5.6-terra", band="short", usage=UsageTokens(input_tokens=50_000, output_tokens=900))
    assert ledger.snapshot().nodes == ()
    now[0] = 160.0
    reservation = ledger.reserve(node_name="fundamental", model="gpt-5.6-terra", band="short", usage=UsageTokens(input_tokens=1_000, output_tokens=900))
    assert reservation.deadline_monotonic == 195.0


def test_parallel_reservations_cannot_overrun_node_cap():
    from market_agent.openai_usage import UsageTokens
    from market_agent.workflow_budget import BudgetExceededError, WorkflowBudgetLedger

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


def test_timeouts_always_charge_the_full_reservation_even_with_known_partial_usage():
    from market_agent.openai_usage import UsageTokens
    from market_agent.workflow_budget import BudgetExceededError, WorkflowBudgetLedger

    ledger = WorkflowBudgetLedger(WorkflowMode.ACTIVE)
    timed_out = ledger.reserve(node_name="fundamental", model="gpt-5.6-terra", band="short", usage=UsageTokens(input_tokens=10_000, output_tokens=900))
    assert ledger.consume_timeout(timed_out).charged_cost == timed_out.reserved_cost
    known_usage = ledger.reserve(node_name="fundamental", model="gpt-5.6-terra", band="short", usage=UsageTokens(input_tokens=10_000, output_tokens=900))
    assert ledger.consume_timeout(known_usage, UsageTokens(input_tokens=10_000, output_tokens=1)).charged_cost == known_usage.reserved_cost
    with pytest.raises(BudgetExceededError):
        ledger.reserve(node_name="fundamental", model="gpt-5.6-terra", band="short", usage=UsageTokens(input_tokens=10_000, output_tokens=900))


def test_settlement_overflow_is_accounted_then_exhausts_the_ledger_and_node():
    from market_agent.openai_usage import UsageTokens
    from market_agent.workflow_budget import BudgetExceededError, BudgetOverflowError, WorkflowBudgetLedger

    ledger = WorkflowBudgetLedger(WorkflowMode.ACTIVE)
    reservation = ledger.reserve(node_name="fundamental", model="gpt-5.6-terra", band="short", usage=UsageTokens(input_tokens=1_000, output_tokens=1))
    with pytest.raises(BudgetOverflowError):
        ledger.settle(reservation, UsageTokens(input_tokens=50_000, output_tokens=1))
    snapshot = ledger.snapshot()
    node = snapshot.nodes[0]
    assert snapshot.reserved_cost == Decimal("0")
    assert snapshot.settled_cost == Decimal("0.100012")
    assert snapshot.exhausted and snapshot.overdrawn
    assert node.reserved_cost == Decimal("0")
    assert node.settled_cost == Decimal("0.100012")
    assert node.exhausted and node.overdrawn
    with pytest.raises(BudgetExceededError):
        ledger.reserve(node_name="event_filter", model="gpt-5.6-luna", band="short", usage=UsageTokens(output_tokens=1))


def test_overflow_error_exposes_committed_settlement(ledger, reservation, overflow_usage):
    from market_agent.workflow_budget import BudgetOverflowError, ReservationStateError

    with pytest.raises(BudgetOverflowError) as raised:
        ledger.settle(reservation, overflow_usage)
    assert raised.value.settlement.reservation_id == reservation.reservation_id
    assert ledger.snapshot().settled_cost == raised.value.settlement.charged_cost
    with pytest.raises(ReservationStateError):
        ledger.settle(reservation, overflow_usage)


def test_node_remaining_attempts_respects_workflow_global_cap(ledger):
    consume_global_attempts_across_distinct_nodes(ledger)
    snapshot = ledger.snapshot()
    assert snapshot.remaining_attempts == 0
    assert all(node.remaining_attempts == 0 for node in snapshot.nodes)


def test_ledger_rejects_same_ledger_forgery_unknown_and_stale_reservations():
    from market_agent.openai_usage import UsageTokens
    from market_agent.workflow_budget import ReservationOwnershipError, ReservationStateError, WorkflowBudgetLedger

    ledger = WorkflowBudgetLedger(WorkflowMode.ACTIVE)
    reservation = ledger.reserve(node_name="fundamental", model="gpt-5.6-terra", band="short", usage=UsageTokens(input_tokens=1_000, output_tokens=900))
    with pytest.raises(ReservationOwnershipError):
        ledger.settle(replace(reservation), UsageTokens(input_tokens=1_000, output_tokens=1))
    with pytest.raises(ReservationOwnershipError):
        ledger.settle(replace(reservation, reservation_id="unknown"), UsageTokens(input_tokens=1_000, output_tokens=1))
    ledger.settle(reservation, UsageTokens(input_tokens=1_000, output_tokens=1))
    with pytest.raises(ReservationStateError):
        ledger.settle(reservation, UsageTokens(input_tokens=1_000, output_tokens=1))


def test_reservation_enforces_request_and_node_caps_without_state_changes():
    from market_agent.openai_usage import UsageTokens
    from market_agent.workflow_budget import BudgetExceededError, WorkflowBudgetLedger

    ledger = WorkflowBudgetLedger(WorkflowMode.ACTIVE)
    with pytest.raises(BudgetExceededError):
        ledger.reserve(node_name="market_context", model="gpt-5.6-terra", band="short", usage=UsageTokens(output_tokens=1_201), maximum_tool_calls=3)
    with pytest.raises(BudgetExceededError):
        ledger.reserve(node_name="market_context", model="gpt-5.6-terra", band="short", usage=UsageTokens(output_tokens=1_200), maximum_tool_calls=4)
    with pytest.raises(BudgetExceededError):
        ledger.reserve(node_name="fundamental", model="gpt-5.6-terra", band="short", usage=UsageTokens(output_tokens=1), attempt_timeout_seconds=36)
    assert ledger.snapshot().nodes == ()


def test_monotonic_clock_rollback_is_rejected_without_corrupting_future_accounting():
    from market_agent.openai_usage import UsageTokens
    from market_agent.workflow_budget import WorkflowBudgetLedger

    now = [100.0]
    ledger = WorkflowBudgetLedger(WorkflowMode.ACTIVE, clock=lambda: now[0])
    ledger.reserve(node_name="fundamental", model="gpt-5.6-terra", band="short", usage=UsageTokens(input_tokens=1_000, output_tokens=900))
    now[0] = 99.0
    with pytest.raises(ValueError, match="moved backwards"):
        ledger.snapshot()
    now[0] = 101.0
    assert ledger.snapshot().remaining_seconds == 299.0


def test_reservation_deadlines_use_monotonic_time_and_snapshot_is_immutable():
    from market_agent.openai_usage import UsageTokens
    from market_agent.workflow_budget import BudgetExceededError, WorkflowBudgetLedger

    now = [100.0]
    ledger = WorkflowBudgetLedger(WorkflowMode.ACTIVE, clock=lambda: now[0])
    ledger.reserve(node_name="fundamental", model="gpt-5.6-terra", band="short", usage=UsageTokens(input_tokens=1_000, output_tokens=900))
    now[0] = 161.0
    with pytest.raises(BudgetExceededError):
        ledger.reserve(node_name="fundamental", model="gpt-5.6-terra", band="short", usage=UsageTokens(input_tokens=1_000, output_tokens=900))
    with pytest.raises(FrozenInstanceError):
        ledger.snapshot().remaining_cost = Decimal('1')


@pytest.mark.parametrize(
    ('reserved', 'actual'),
    [
        (UsageTokens(input_tokens=1), UsageTokens(input_tokens=2)),
        (UsageTokens(input_tokens=2, cached_input_tokens=1), UsageTokens(input_tokens=2, cached_input_tokens=2)),
        (UsageTokens(cache_write_tokens=1), UsageTokens(cache_write_tokens=2)),
        (UsageTokens(output_tokens=1), UsageTokens(output_tokens=2)),
    ],
)
def test_each_reserved_token_dimension_overflow_is_accounted_before_raising(reserved, actual):
    from market_agent.workflow_budget import BudgetOverflowError, WorkflowBudgetLedger

    ledger = WorkflowBudgetLedger(WorkflowMode.ACTIVE)
    reservation = ledger.reserve(node_name='fundamental', model='gpt-5.6-terra', band='short', usage=reserved)
    with pytest.raises(BudgetOverflowError):
        ledger.settle(reservation, actual)
    snapshot = ledger.snapshot()
    assert snapshot.reserved_cost == Decimal('0')
    assert snapshot.settled_cost > Decimal('0')
    assert snapshot.exhausted and snapshot.overdrawn


def test_tool_overflow_is_accounted_before_raising_even_when_total_cost_is_lower():
    from market_agent.openai_usage import UsageTokens
    from market_agent.workflow_budget import BudgetOverflowError, WorkflowBudgetLedger

    ledger = WorkflowBudgetLedger(WorkflowMode.ACTIVE)
    reservation = ledger.reserve(
        node_name='market_context', model='gpt-5.6-terra', band='short',
        usage=UsageTokens(input_tokens=20_000, output_tokens=1_200), maximum_tool_calls=1,
    )
    with pytest.raises(BudgetOverflowError):
        ledger.settle(reservation, UsageTokens(input_tokens=1, output_tokens=1, web_search_tool_calls=2))
    snapshot = ledger.snapshot()
    assert snapshot.reserved_cost == Decimal('0')
    assert snapshot.settled_cost < reservation.reserved_cost
    assert snapshot.exhausted and snapshot.overdrawn


@pytest.mark.parametrize('changed_price', ['100', '0', 'not-a-decimal'])
def test_settlement_uses_tool_price_pinned_at_reservation(monkeypatch, changed_price):
    from market_agent.openai_usage import UsageTokens
    from market_agent.workflow_budget import WorkflowBudgetLedger

    monkeypatch.setenv('OPENAI_WEB_SEARCH_TOOL_PRICE_PER_1K_USD', '2.5')
    ledger = WorkflowBudgetLedger(WorkflowMode.ACTIVE)
    reservation = ledger.reserve(
        node_name='market_context', model='gpt-5.6-terra', band='short',
        usage=UsageTokens(web_search_tool_calls=0), maximum_tool_calls=1,
    )
    monkeypatch.setenv('OPENAI_WEB_SEARCH_TOOL_PRICE_PER_1K_USD', changed_price)
    settlement = ledger.settle(reservation, UsageTokens(web_search_tool_calls=1))
    assert reservation.reserved_cost == Decimal('0.0025')
    assert settlement.charged_cost == Decimal('0.0025')


def test_policy_attempt_caps_are_reachable_and_snapshot_counts_reachable_attempts():
    from market_agent.openai_usage import UsageTokens
    from market_agent.workflow_budget import WorkflowBudgetLedger
    from market_agent.workflow_model_routing import AgentExecutionPolicy, ModelRouteTier, policies

    for policy in policies().values():
        assert policy.maximum_total_attempts <= policy.maximum_attempts_per_tier * len(policy.tiers)
    with pytest.raises(ValueError):
        AgentExecutionPolicy('bad', (ModelRouteTier('gpt-5.6-luna', 'low'),), 1, 1, 1, 2, 1, Decimal('0'), 0)

    ledger = WorkflowBudgetLedger(WorkflowMode.ACTIVE)
    ledger.reserve(node_name='fundamental', model='gpt-5.6-terra', band='short', usage=UsageTokens())
    ledger.reserve(node_name='fundamental', model='gpt-5.6-terra', band='short', usage=UsageTokens())
    assert ledger.snapshot().nodes[0].remaining_attempts == 1


def test_concurrent_settlements_use_each_reservation_pinned_tool_price_after_environment_becomes_invalid(monkeypatch):
    from market_agent.workflow_budget import WorkflowBudgetLedger

    ledger = WorkflowBudgetLedger(WorkflowMode.ACTIVE)
    monkeypatch.setenv('OPENAI_WEB_SEARCH_TOOL_PRICE_PER_1K_USD', '1')
    first = ledger.reserve(node_name='market_context', model='gpt-5.6-terra', band='short', usage=UsageTokens(), maximum_tool_calls=1)
    monkeypatch.setenv('OPENAI_WEB_SEARCH_TOOL_PRICE_PER_1K_USD', '2')
    second = ledger.reserve(node_name='market_context', model='gpt-5.6-terra', band='short', usage=UsageTokens(), maximum_tool_calls=1)
    monkeypatch.setenv('OPENAI_WEB_SEARCH_TOOL_PRICE_PER_1K_USD', 'not-a-decimal')

    with ThreadPoolExecutor(max_workers=2) as executor:
        settlements = list(executor.map(lambda reservation: ledger.settle(reservation, UsageTokens(web_search_tool_calls=1)), (first, second)))
    assert sorted(settlement.charged_cost for settlement in settlements) == [Decimal('0.001'), Decimal('0.002')]


@pytest.mark.parametrize('invalid_price', ['-1', 'NaN', 'Infinity'])
def test_invalid_tool_price_is_rejected_at_reservation_without_creating_budget_state(monkeypatch, invalid_price):
    from market_agent.workflow_budget import WorkflowBudgetLedger

    monkeypatch.setenv('OPENAI_WEB_SEARCH_TOOL_PRICE_PER_1K_USD', invalid_price)
    ledger = WorkflowBudgetLedger(WorkflowMode.ACTIVE)
    with pytest.raises(ValueError, match='web search tool price'):
        ledger.reserve(node_name='market_context', model='gpt-5.6-terra', band='short', usage=UsageTokens(), maximum_tool_calls=1)
    assert ledger.snapshot().nodes == ()


def test_foreign_reservation_cannot_change_the_other_ledger():
    from market_agent.workflow_budget import ReservationOwnershipError, WorkflowBudgetLedger

    owner = WorkflowBudgetLedger(WorkflowMode.ACTIVE)
    foreign = WorkflowBudgetLedger(WorkflowMode.ACTIVE)
    reservation = owner.reserve(node_name='fundamental', model='gpt-5.6-terra', band='short', usage=UsageTokens(input_tokens=1))
    with pytest.raises(ReservationOwnershipError):
        foreign.settle(reservation, UsageTokens(input_tokens=1))
    assert foreign.snapshot().reserved_cost == Decimal('0')
    assert owner.snapshot().reserved_cost == reservation.reserved_cost
