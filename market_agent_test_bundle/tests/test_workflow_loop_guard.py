from __future__ import annotations

import hashlib
import json

import pytest
from pydantic import ValidationError

from market_agent.workflow_harness_contracts import ProgressVector
from market_agent.workflow_loop_guard import (
    ActionFingerprint,
    ActionObservationFingerprint,
    CycleSignature,
    LoopGuard,
    LoopScope,
    ObservationKind,
    ResultFingerprint,
    SemanticCheckpoint,
    SeverityPolicy,
    build_action_fingerprint,
    build_result_fingerprint,
    build_state_fingerprint,
    compare_progress,
)


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def progress(**overrides: object) -> ProgressVector:
    values: dict[str, object] = {
        "completed_dependency_count": 0,
        "valid_required_field_count": 0,
        "filled_required_evidence_slot_count": 0,
        "fresh_authoritative_source_coverage": 0.0,
        "missing_evidence_count": 1,
        "validation_error_count": 0,
        "unresolved_conflict_count": 0,
        "risk_invariant_failure_count": 0,
    }
    values.update(overrides)
    return ProgressVector(**values)


def severity_policy() -> SeverityPolicy:
    return SeverityPolicy(
        policy_version="severity-v1",
        critical_positive_regressions=(
            "filled_required_evidence_slot_count",
            "fresh_authoritative_source_coverage",
        ),
        critical_negative_regressions=(
            "validation_error_count",
            "risk_invariant_failure_count",
        ),
    )


def action(label: str, *, worker_id: str = "worker-a") -> ActionFingerprint:
    return build_action_fingerprint(
        worker_id=worker_id,
        worker_version="v1",
        action_kind="inspect",
        canonical_arguments={"label": label},
        context_hash=digest({"context": "fixed"}),
        dependency_hash=digest({"dependency": "fixed"}),
        plan_revision=1,
        prompt_hash=digest({"prompt": "v1"}),
        tool_hash=digest({"tool": "v1"}),
        output_schema_hash=digest({"schema": "v1"}),
        model_route="fixed-route",
        correction_ordinal=0,
    )


def result(label: str, *, error_code: str | None = None) -> ResultFingerprint:
    return build_result_fingerprint(
        outcome_kind="failed" if error_code else "answer",
        validated_output_hash=digest({"output": label}),
        normalized_error_class="source_failure" if error_code else None,
        normalized_error_code=error_code,
        accepted_evidence_ids=("evidence-1",),
        tool_result_hashes=(digest({"tool-result": label}),),
        result_schema_version="result-v1",
    )


def action_observation(
    action_label: str,
    result_label: str,
    *,
    worker_id: str = "worker-a",
    scope: str = "attempt",
) -> ActionObservationFingerprint:
    return ActionObservationFingerprint.from_parts(
        action(action_label, worker_id=worker_id), result(result_label), scope=scope
    )


def checkpoint(
    label: str,
    *,
    current_progress: ProgressVector | None = None,
    observation_kind: ObservationKind = ObservationKind.SEMANTIC_CHECKPOINT,
    scope: str = "attempt",
    worker_id: str = "worker-a",
    failure: str | None = None,
) -> SemanticCheckpoint:
    return SemanticCheckpoint(
        scope=LoopScope(scope),
        state_fingerprint=build_state_fingerprint(
            run_state="running",
            work_item_state="running",
            attempt_state="validating",
            stage_id="analysis",
            plan_revision=1,
            unresolved_work_ids=(label,),
            dependency_versions=(("input", 1),),
            progress=current_progress or progress(),
            normalized_error_class=failure,
        ),
        progress=current_progress or progress(),
        plan_revision=1,
        fingerprint_schema_version="v1",
        observation_kind=observation_kind,
        worker_id=worker_id if failure else None,
        normalized_failure=failure,
        failure_context_hash=digest({"context": "fixed"}) if failure else None,
        failure_dependency_hash=digest({"dependency": "fixed"}) if failure else None,
        correction_ordinal=0,
        model_route="fixed-route" if failure else None,
    )


@pytest.fixture
def loop_guard() -> LoopGuard:
    return LoopGuard(severity_policy=severity_policy())


def test_third_identical_action_result_stops_work(loop_guard: LoopGuard):
    observation = action_observation("same-action", "same-result")
    assert loop_guard.observe_action_result(observation).allowed
    assert loop_guard.observe_action_result(observation).allowed
    assert (
        loop_guard.observe_action_result(observation).stop_reason
        == "repeated_action_result"
    )


def test_three_same_actions_in_latest_five_stop_even_when_results_differ(
    loop_guard: LoopGuard,
):
    assert loop_guard.observe_action_result(action_observation("same", "one")).allowed
    assert loop_guard.observe_action_result(action_observation("other", "two")).allowed
    assert loop_guard.observe_action_result(action_observation("same", "three")).allowed
    assert (
        loop_guard.observe_action_result(action_observation("same", "four")).stop_reason
        == "repeated_action"
    )


@pytest.mark.parametrize(
    "states", [("a", "b", "a", "b"), ("a", "b", "c", "a", "b", "c")],
)
def test_shortest_repeating_cycle_is_canonical(loop_guard: LoopGuard, states: tuple[str, ...]):
    decision = None
    for state in states:
        decision = loop_guard.observe_checkpoint(checkpoint(state))
    assert decision is not None and decision.stop_reason == "state_cycle"


def test_cycle_signature_is_rotation_normalized(loop_guard: LoopGuard):
    first = None
    for state in ("a", "b", "c", "a", "b", "c"):
        first = loop_guard.observe_checkpoint(checkpoint(state))
    second_guard = LoopGuard(severity_policy=severity_policy())
    second = None
    for state in ("b", "c", "a", "b", "c", "a"):
        second = second_guard.observe_checkpoint(checkpoint(state))
    assert first is not None and second is not None
    assert first.cycle_signature is not None
    assert first.cycle_signature == second.cycle_signature


def test_duplicate_state_without_progress_stops_attempt(loop_guard: LoopGuard):
    assert loop_guard.observe_checkpoint(checkpoint("same")).allowed
    assert (
        loop_guard.observe_checkpoint(checkpoint("same")).stop_reason
        == "duplicate_state_no_progress"
    )


def test_heartbeats_do_not_enter_semantic_windows_or_advance_no_progress(
    loop_guard: LoopGuard,
):
    assert loop_guard.observe_checkpoint(checkpoint("same")).allowed
    for _ in range(10):
        ignored = loop_guard.observe_checkpoint(
            checkpoint("same", observation_kind=ObservationKind.HEARTBEAT)
        )
        assert ignored.allowed and ignored.ignored
    assert (
        loop_guard.observe_checkpoint(checkpoint("same")).stop_reason
        == "duplicate_state_no_progress"
    )


def test_two_no_progress_semantic_checkpoints_stop_work(loop_guard: LoopGuard):
    assert loop_guard.observe_checkpoint(checkpoint("first")).allowed
    assert loop_guard.observe_checkpoint(checkpoint("second")).allowed
    assert (
        loop_guard.observe_checkpoint(checkpoint("third")).stop_reason
        == "no_progress"
    )


def test_cross_worker_same_failure_oscillation_stops_rescheduling(loop_guard: LoopGuard):
    assert loop_guard.observe_checkpoint(
        checkpoint("first", worker_id="worker-a", failure="source_failure")
    ).allowed
    assert loop_guard.observe_checkpoint(
        checkpoint("second", worker_id="worker-b", failure="source_failure")
    ).allowed
    assert (
        loop_guard.observe_checkpoint(
            checkpoint("third", worker_id="worker-a", failure="source_failure")
        ).stop_reason
        == "cross_worker_failure_oscillation"
    )


def test_changed_failure_context_cannot_be_misclassified_as_cross_worker_oscillation(
    loop_guard: LoopGuard,
):
    assert loop_guard.observe_checkpoint(
        checkpoint("first", worker_id="worker-a", failure="source_failure")
    ).allowed
    assert loop_guard.observe_checkpoint(
        checkpoint("second", worker_id="worker-b", failure="source_failure")
    ).allowed
    changed = checkpoint("third", worker_id="worker-a", failure="source_failure").model_copy(
        update={"failure_context_hash": digest({"context": "changed"})}
    )
    assert loop_guard.observe_checkpoint(changed).stop_reason != "cross_worker_failure_oscillation"


def test_unrelated_evidence_is_not_progress():
    before = progress(filled_required_evidence_slot_count=1)
    after = before.model_copy()
    assert not compare_progress(before, after, severity_policy()).advanced


def test_critical_regression_fails_closed_despite_positive_improvement():
    before = progress(valid_required_field_count=1, validation_error_count=0)
    after = progress(valid_required_field_count=2, validation_error_count=1)
    decision = compare_progress(before, after, severity_policy())
    assert not decision.advanced
    assert decision.critical_regression


def test_progress_requires_oriented_monotonic_strict_improvement():
    decision = compare_progress(
        progress(missing_evidence_count=2),
        progress(missing_evidence_count=1),
        severity_policy(),
    )
    assert decision.advanced


def test_fingerprint_builders_are_canonical_and_ignore_ephemeral_inputs():
    first = build_action_fingerprint(
        worker_id="worker-a",
        worker_version="v1",
        action_kind="inspect",
        canonical_arguments={"symbol": "BTC", "label": "one"},
        context_hash=digest({"context": 1}),
        dependency_hash=digest({"dependency": 1}),
        plan_revision=1,
        prompt_hash=digest({"prompt": 1}),
        tool_hash=digest({"tool": 1}),
        output_schema_hash=digest({"schema": 1}),
        model_route="route",
        correction_ordinal=0,
    )
    second = build_action_fingerprint(
        worker_id="worker-a",
        worker_version="v1",
        action_kind="inspect",
        canonical_arguments={"label": "one", "symbol": "BTC"},
        context_hash=digest({"context": 1}),
        dependency_hash=digest({"dependency": 1}),
        plan_revision=1,
        prompt_hash=digest({"prompt": 1}),
        tool_hash=digest({"tool": 1}),
        output_schema_hash=digest({"schema": 1}),
        model_route="route",
        correction_ordinal=0,
    )
    assert first == second


def test_fingerprint_contract_rejects_raw_content_and_secret_values():
    common = {
        "worker_id": "worker-a",
        "worker_version": "v1",
        "action_kind": "inspect",
        "context_hash": digest({"context": 1}),
        "dependency_hash": digest({"dependency": 1}),
        "plan_revision": 1,
        "prompt_hash": digest({"prompt": 1}),
        "tool_hash": digest({"tool": 1}),
        "output_schema_hash": digest({"schema": 1}),
        "model_route": "route",
        "correction_ordinal": 0,
    }
    with pytest.raises(ValueError):
        build_action_fingerprint(
            **common,
            canonical_arguments={"symbol": "BTC", "raw_content": "private prose"},
        )
    with pytest.raises(ValueError):
        build_action_fingerprint(
            **common,
            canonical_arguments={"symbol": "BTC", "label": "auth_secret"},
        )


def test_fingerprint_contracts_are_frozen_strict_and_revalidate_model_copy():
    fingerprint = action("one")
    with pytest.raises(ValidationError):
        fingerprint.model_copy(update={"digest": "not-a-digest"})
    with pytest.raises(ValidationError):
        ActionFingerprint(digest=digest({"x": 1}), unexpected=True)


def test_one_recovery_is_authorized_per_cycle_signature(loop_guard: LoopGuard):
    decision = None
    for state in ("a", "b", "a", "b"):
        decision = loop_guard.observe_checkpoint(checkpoint(state))
    assert decision is not None and decision.cycle_signature is not None
    signature = decision.cycle_signature
    assert loop_guard.authorize_recovery(signature).allowed
    assert loop_guard.authorize_recovery(signature).stop_reason == "recovery_exhausted"


def test_returning_to_recovered_cycle_terminates(loop_guard: LoopGuard):
    first = None
    for state in ("a", "b", "a", "b"):
        first = loop_guard.observe_checkpoint(checkpoint(state))
    assert first is not None and first.cycle_signature is not None
    assert loop_guard.authorize_recovery(first.cycle_signature).allowed
    for state in ("a", "b", "a", "b"):
        decision = loop_guard.observe_checkpoint(checkpoint(state))
    assert decision.stop_reason == "recovered_cycle_returned"


def test_public_cycle_signature_rejects_noncanonical_periods():
    with pytest.raises(ValidationError):
        CycleSignature(
            digest=digest({"cycle": "bad"}),
            scope="attempt",
            plan_revision=1,
            fingerprint_schema_version="v1",
            period=("b" * 64, "a" * 64),
        )


def test_fingerprint_contract_rejects_ephemeral_hash_variants_and_preserves_declared_semantics():
    common = {
        "worker_id": "worker-a",
        "worker_version": "v1",
        "action_kind": "inspect",
        "context_hash": digest({"context": 1}),
        "dependency_hash": digest({"dependency": 1}),
        "plan_revision": 1,
        "prompt_hash": digest({"prompt": 1}),
        "tool_hash": digest({"tool": 1}),
        "output_schema_hash": digest({"schema": 1}),
        "model_route": "route",
        "correction_ordinal": 0,
    }
    with pytest.raises(ValueError):
        build_action_fingerprint(
            **common,
            canonical_arguments={"event_type": "earnings", "attempt_limit": 2},
            trace_id_hash=digest({"trace": 1}),
        )
    first = build_action_fingerprint(
        **common, canonical_arguments={"event_type": "earnings", "attempt_limit": 2}
    )
    assert first != build_action_fingerprint(
        **common, canonical_arguments={"event_type": "filing", "attempt_limit": 2}
    )
    assert first != build_action_fingerprint(
        **common, canonical_arguments={"event_type": "earnings", "attempt_limit": 3}
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"label": "sk-live-secret"},
        {"label": {"nested": {"too": {"deep": {"for": {"schema": "x"}}}}}},
        {"label": "x" * 257},
    ],
)
def test_action_fingerprint_input_rejects_secrets_and_unbounded_values_without_leaking_them(arguments):
    with pytest.raises(ValueError) as error:
        build_action_fingerprint(
            worker_id="worker-a",
            worker_version="v1",
            action_kind="inspect",
            canonical_arguments=arguments,
            context_hash=digest({"context": 1}),
            dependency_hash=digest({"dependency": 1}),
            plan_revision=1,
            prompt_hash=digest({"prompt": 1}),
            tool_hash=digest({"tool": 1}),
            output_schema_hash=digest({"schema": 1}),
            model_route="route",
            correction_ordinal=0,
        )
    assert "sk-live-secret" not in str(error.value)


def test_action_fingerprint_input_rejects_cycles_without_recursion_error():
    cyclic: dict[str, object] = {}
    cyclic["label"] = cyclic
    with pytest.raises(ValueError, match="invalid action fingerprint input"):
        build_action_fingerprint(
            worker_id="worker-a", worker_version="v1", action_kind="inspect",
            canonical_arguments=cyclic, context_hash=digest({"context": 1}),
            dependency_hash=digest({"dependency": 1}), plan_revision=1,
            prompt_hash=digest({"prompt": 1}), tool_hash=digest({"tool": 1}),
            output_schema_hash=digest({"schema": 1}), model_route="route", correction_ordinal=0,
        )


def test_observation_kind_is_closed_neutral_and_cannot_be_model_copy_bypassed(loop_guard: LoopGuard):
    heartbeat = checkpoint("same", observation_kind=ObservationKind.HEARTBEAT)
    decision = loop_guard.observe_checkpoint(heartbeat)
    assert decision.allowed and decision.ignored
    with pytest.raises(ValidationError):
        heartbeat.model_copy(update={"observation_kind": "forged"})
    with pytest.raises(ValidationError):
        heartbeat.model_copy(update={"semantic": True})


def test_scope_isolation_keeps_action_histories_and_failures_separate(loop_guard: LoopGuard):
    observation = action_observation("same", "result", scope="attempt")
    assert loop_guard.observe_action_result(observation).allowed
    assert loop_guard.observe_action_result(observation).allowed
    assert loop_guard.observe_action_result(
        action_observation("same", "result", scope="stage")
    ).allowed
    assert loop_guard.observe_action_result(
        action_observation("same", "result", scope="stage")
    ).allowed
    assert loop_guard.observe_action_result(observation).stop_reason == "repeated_action_result"

    assert loop_guard.observe_checkpoint(
        checkpoint("one", scope="work_item", worker_id="worker-a", failure="source_failure")
    ).allowed
    assert loop_guard.observe_checkpoint(
        checkpoint("two", scope="work_item", worker_id="worker-b", failure="source_failure")
    ).allowed
    ignored = loop_guard.observe_checkpoint(
        checkpoint("infra", scope="stage", observation_kind=ObservationKind.INFRASTRUCTURE)
    )
    assert ignored.allowed and ignored.ignored
    assert loop_guard.observe_checkpoint(
        checkpoint("three", scope="work_item", worker_id="worker-a", failure="source_failure")
    ).stop_reason == "cross_worker_failure_oscillation"


def test_nonadjacent_duplicate_in_same_scope_stops_without_progress(loop_guard: LoopGuard):
    assert loop_guard.observe_checkpoint(checkpoint("a", scope="attempt")).allowed
    assert loop_guard.observe_checkpoint(checkpoint("b", scope="attempt")).allowed
    assert (
        loop_guard.observe_checkpoint(checkpoint("a", scope="attempt")).stop_reason
        == "duplicate_state_no_progress"
    )


def test_recovery_rejects_fabricated_and_nonprimitive_cycle_signatures(loop_guard: LoopGuard):
    decision = None
    for state in ("a", "b", "a", "b"):
        decision = loop_guard.observe_checkpoint(checkpoint(state))
    assert decision is not None and decision.cycle_signature is not None
    signature = decision.cycle_signature
    with pytest.raises(ValidationError):
        signature.model_copy(update={"period": signature.period + signature.period})
    period = ("a" * 64, "b" * 64)
    forged = CycleSignature(
        digest=digest({"scope": "stage", "plan_revision": 1, "fingerprint_schema_version": "v1", "period": period}),
        scope=LoopScope.STAGE, plan_revision=1, fingerprint_schema_version="v1", period=period,
    )
    assert loop_guard.authorize_recovery(forged).stop_reason == "unregistered_cycle_signature"


@pytest.mark.parametrize(
    ("arguments"),
    [
        {"label": 1}, {"symbol": 1.0}, {"event_type": False},
        {"attempt_limit": True}, {"attempt_limit": 1.0}, {"attempt_limit": "1"},
        {"operation": 1}, {"target": False}, {"condition": 1.0},
        {"mode": 1}, {"limit": True}, {"limit": 1.0}, {"limit": "1"},
    ],
)
def test_each_semantic_argument_name_rejects_wrong_scalar_domain(arguments):
    with pytest.raises(ValueError, match="invalid action fingerprint input"):
        action_input(arguments)


def action_input(arguments: object, **overrides: object) -> ActionFingerprint:
    values: dict[str, object] = {
        "worker_id": "worker-a", "worker_version": "v1", "action_kind": "inspect",
        "canonical_arguments": arguments, "context_hash": digest({"context": 1}),
        "dependency_hash": digest({"dependency": 1}), "plan_revision": 1,
        "prompt_hash": digest({"prompt": 1}), "tool_hash": digest({"tool": 1}),
        "output_schema_hash": digest({"schema": 1}), "model_route": "route",
        "correction_ordinal": 0,
    }
    values.update(overrides)
    return build_action_fingerprint(**values)


def test_symbol_normalization_and_integer_representation_are_canonical():
    assert action_input({"symbol": "btc", "limit": 1}) == action_input(
        {"limit": 1, "symbol": "BTC"}
    )
    with pytest.raises(ValueError):
        action_input({"limit": 1.0})


class OversizedSequence:
    def __len__(self) -> int:
        return 65

    def __iter__(self):
        raise AssertionError("must not materialize oversized input")


class UnsizedGenerator:
    def __iter__(self):
        yield "evidence-1"


@pytest.mark.parametrize("sequence", [OversizedSequence(), UnsizedGenerator()])
def test_result_sequences_are_bounded_before_sorting_or_materialization(sequence):
    with pytest.raises(ValueError, match="invalid result fingerprint input"):
        build_result_fingerprint(
            outcome_kind="answer", validated_output_hash=digest({"out": 1}),
            normalized_error_class=None, normalized_error_code=None,
            accepted_evidence_ids=sequence, tool_result_hashes=(), result_schema_version="v1",
        )


@pytest.mark.parametrize(
    "value",
    [
        "ghp_abcdefghijklmnopqrstuvwxyz", "AKIAABCDEFGHIJKLMNOP", "aaa.bbb.ccc",
        "-----BEGIN PRIVATE KEY-----", "Bearer abcdefghijklmnopqrstuvwxyz",
        "sk-live-abcdefghijklmnopqrstuvwxyz", "p" * 80,
    ],
)
def test_semantic_string_fields_reject_credential_shaped_values_without_echoing(value):
    with pytest.raises(ValueError) as error:
        action_input({"label": value})
    assert value not in str(error.value)


def test_failure_requires_worker_and_nonfailures_cannot_carry_workers():
    with pytest.raises(ValidationError):
        checkpoint("missing", failure="source_failure").model_copy(update={"worker_id": None})
    with pytest.raises(ValidationError):
        checkpoint("success").model_copy(update={"worker_id": "worker-a"})


def test_failure_oscillation_rejects_missing_identity_but_stops_valid_a_b_a(loop_guard: LoopGuard):
    with pytest.raises(ValidationError):
        checkpoint("missing", failure="source_failure").model_copy(update={"worker_id": None})
    assert loop_guard.observe_checkpoint(
        checkpoint("one", worker_id="worker-a", failure="source_failure")
    ).allowed
    assert loop_guard.observe_checkpoint(
        checkpoint("two", worker_id="worker-b", failure="source_failure")
    ).allowed
    assert loop_guard.observe_checkpoint(
        checkpoint("three", worker_id="worker-a", failure="source_failure")
    ).stop_reason == "cross_worker_failure_oscillation"


class LyingList(list[str]):
    calls = 0

    def __len__(self) -> int:
        type(self).calls += 1
        raise AssertionError("subclass length must not be called")

    def __iter__(self):
        type(self).calls += 1
        raise AssertionError("subclass iteration must not be called")


class LyingTuple(tuple[str, ...]):
    calls = 0

    def __len__(self) -> int:
        type(self).calls += 1
        raise AssertionError("subclass length must not be called")

    def __iter__(self):
        type(self).calls += 1
        raise AssertionError("subclass iteration must not be called")


@pytest.mark.parametrize("sequence", [LyingList(["evidence-1"]), LyingTuple(("evidence-1",))])
def test_sequence_subclasses_are_rejected_without_calling_attacker_methods(sequence):
    type(sequence).calls = 0
    with pytest.raises(ValueError, match="invalid result fingerprint input"):
        build_result_fingerprint(
            outcome_kind="answer", validated_output_hash=digest({"out": 1}),
            normalized_error_class=None, normalized_error_code=None,
            accepted_evidence_ids=sequence, tool_result_hashes=(), result_schema_version="v1",
        )
    assert type(sequence).calls == 0


def test_precise_secret_checks_accept_legitimate_token_and_dotted_codes():
    accepted = action_input(
        {"event_type": "token_unlock", "symbol": "btc"},
        worker_version="v1.2.3", model_route="route.v1.2",
    )
    assert accepted == action_input(
        {"symbol": "BTC", "event_type": "token_unlock"},
        worker_version="v1.2.3", model_route="route.v1.2",
    )


@pytest.mark.parametrize(
    "secret",
    [
        "aaaaaaaa.bbbbbbbb.cccccccc", "ghp_abcdefghijklmnopqrstuvwxyz123456",
        "AKIAABCDEFGHIJKLMNOP", "sk-live-abcdefghijklmnopqrstuvwxyz",
        "Bearer abcdefghijklmnop", "-----BEGIN PRIVATE KEY-----", "a" * 48,
    ],
)
def test_precise_secret_checks_reject_credentials_without_echoing(secret):
    with pytest.raises(ValueError) as error:
        action_input({"label": secret})
    assert secret not in str(error.value)


def test_worker_ids_are_lowercase_ascii_and_shared_by_actions_and_failures():
    with pytest.raises(ValueError):
        action_input({"label": "one"}, worker_id="Worker-A")
    with pytest.raises(ValueError):
        action_input({"label": "one"}, worker_id="worker-а")
    with pytest.raises(ValidationError):
        checkpoint("one", worker_id="Worker-A", failure="source_failure")
    with pytest.raises(ValidationError):
        checkpoint("one", worker_id="worker-а", failure="source_failure")
