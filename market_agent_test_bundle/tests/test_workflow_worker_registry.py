from __future__ import annotations

import pytest

from market_agent.workflow_harness_contracts import OutcomeKind, TaskKind, WorkerSpec
from market_agent.workflow_worker_registry import (
    DuplicateWorkerError,
    InvalidWorkerError,
    UnknownWorkerError,
    WorkerRegistry,
)


HASH = "a" * 64


def worker_spec(*, worker_id: str = "information-worker") -> WorkerSpec:
    return WorkerSpec(
        worker_id=worker_id,
        version="worker-v1",
        supported_task_kinds=(TaskKind.INFORMATIONAL,),
        analysis_phases=("collect", "verify", "summarize"),
        input_schema_id="InformationInput",
        input_schema_hash=HASH,
        output_schema_id="InformationOutput",
        output_schema_hash=HASH,
        prompt_release="information-v1",
        prompt_profile="default",
        model_routing_policy_key="information-route-v1",
        context_selector="information-context-v1",
        context_token_budget=800,
        writable_invocation_state_key="information_result",
        cacheable=True,
        freshness_class="request",
        maximum_turns=2,
        maximum_tool_calls=1,
        maximum_input_tokens=800,
        maximum_output_tokens=300,
        timeout_seconds=10.0,
        maximum_attempts=1,
        maximum_cost=0.01,
        success_outcome=OutcomeKind.ANSWER,
        failure_outcome=OutcomeKind.NONE,
        degradation_outcome=OutcomeKind.UNKNOWN,
    )


def test_registry_preserves_immutable_worker_specs_in_declaration_order():
    first = worker_spec(worker_id="first-worker")
    second = worker_spec(worker_id="second-worker")

    registry = WorkerRegistry((first, second))

    assert registry.all() == (first, second)
    assert registry.get("first-worker") is not first
    with pytest.raises(TypeError):
        registry._specs["third-worker"] = first


def test_registry_rejects_duplicate_worker_identifiers():
    with pytest.raises(DuplicateWorkerError, match="worker identifiers must be unique"):
        WorkerRegistry((worker_spec(), worker_spec()))


def test_registry_rejects_unknown_worker_identifier_with_typed_error():
    registry = WorkerRegistry((worker_spec(),))

    with pytest.raises(UnknownWorkerError, match="unknown worker identifier: missing-worker"):
        registry.get("missing-worker")


@pytest.mark.parametrize(
    "forged_update",
    (
        {"analysis_phases": ("only",)},
        {"maximum_cost": -1.0},
        {"input_schema_hash": "not-a-digest"},
        {"untrusted_capability": "trade.execute"},
    ),
)
def test_registry_revalidates_model_copy_bypasses_with_a_typed_error(
    forged_update: dict[str, object],
):
    forged = worker_spec().model_copy(update=forged_update)

    with pytest.raises(InvalidWorkerError, match="invalid worker specification"):
        WorkerRegistry((forged,))
