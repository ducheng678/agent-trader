from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256

import pytest
from pydantic import BaseModel, ConfigDict, Field

from market_agent.workflow_agent_contracts import AgentInvocation, AgentUsage, ModelTier
from market_agent.workflow_agent_driver import AgentDriver, ModelResponse, OutputSchema
from market_agent.workflow_agents.common import build_invocation, profile_for
from market_agent.workflow_audit import AuditEvent
from market_agent.workflow_circuit_breaker import CircuitBreaker
from market_agent.workflow_coordinator_services import AgentCoordinatorServices
from market_agent.workflow_fallback import FallbackPolicy
from market_agent.workflow_long_term_memory import (
    DecisionRecord,
    EventRecord,
    KnowledgeRevision,
    OutcomeRecord,
    Provenance,
)
from market_agent.workflow_memory_result_writer import AcceptedOutcomeProof, MemoryResultWriter
from market_agent.workflow_memory_sqlite import SQLiteMemoryRepository
from market_agent.workflow_execution_backend import (
    CommittedExecutionSnapshot,
    CommittedTransitionReceipt,
)
from market_agent.workflow_harness_contracts import HarnessSessionView, RunState
from market_agent.workflow_prompt_config import PromptPin, PromptReleaseManager, PromptReleaseManifest
from market_agent.workflow_prompt_config import WorkflowPromptPin
from market_agent.workflow_prompt_release import PromptRelease, canonical_json
from market_agent.workflow_reflection_agent import CorrectionContext
from market_agent.workflow_reflection_agent import reflection_output_schema, reflection_release
from market_agent.workflow_retry_policy import RetryPolicy


NOW = datetime(2026, 9, 2, tzinfo=timezone.utc)


def _workflow_request(*, query: str = "Does this system trade?"):
    from market_agent.workflow_contracts import WorkflowRequest

    return WorkflowRequest(
        workflow_id="wf-1", trace_id="1" * 32,
        user_query=query, trigger_reason="manual_once",
    )


def _unsigned_terminal_receipt(
    request, prompt_digest: str, result_digest: str,
) -> CommittedTransitionReceipt:
    from market_agent.workflow_contracts import canonical_workflow_request_digest

    request_digest = canonical_workflow_request_digest(request)
    pre_view = HarnessSessionView(
        sequence=1, state_revision=1, plan_revision=0,
        run_id=request.workflow_id, trace_id=request.trace_id,
        request_digest=request_digest, prompt_release_digest=prompt_digest,
        accepted_result_digest=result_digest,
        run_state=RunState.SUMMARIZING, last_event_hash="a" * 64,
    )
    post_view = pre_view.model_copy(update={
        "sequence": 2,
        "state_revision": 2,
        "run_state": RunState.SUCCEEDED,
        "last_event_hash": "b" * 64,
    })

    def snapshot(view: HarnessSessionView) -> CommittedExecutionSnapshot:
        return CommittedExecutionSnapshot(
            run_id=request.workflow_id, trace_id=request.trace_id,
            plan_id="plan-1", plan_digest="c" * 64, plan_revision=0,
            sequence=view.sequence, state_revision=view.state_revision,
            view_digest="d" * 64, event_head_hash=view.last_event_hash,
            folded_view=view, trust_key_id="host-rsa-2026-01",
            signature="0" * 512,
        )

    return CommittedTransitionReceipt(
        pre=snapshot(pre_view), post=snapshot(post_view),
        transition_digest="e" * 64,
        trust_key_id="host-rsa-2026-01", signature="0" * 512,
    )


def _untrusted_proof(request, result, prompt_digest: str) -> AcceptedOutcomeProof:
    from market_agent.workflow_contracts import canonical_workflow_request_digest

    return AcceptedOutcomeProof(
        workflow_id=result.workflow_id,
        trace_id=result.trace_id,
        request_digest=canonical_workflow_request_digest(request),
        result_digest=sha256(canonical_json(
            result.model_dump(mode="json")
        ).encode("utf-8")).hexdigest(),
        harness_state="succeeded",
        terminal_receipt=_unsigned_terminal_receipt(
            request, prompt_digest,
            sha256(canonical_json(result.model_dump(mode="json")).encode("utf-8")).hexdigest(),
        ),
        prompt_release_digest=prompt_digest,
        accepted_at=NOW,
    )


def _release(identifier: str, prefix: str) -> PromptRelease:
    values = dict(
        schema_version="v1",
        release_id=identifier,
        stable_system_prefix=prefix,
        supported_task_kinds=("extract", "analyze", "coordinator"),
        supported_model_tiers=(ModelTier.LUNA, ModelTier.TERRA, ModelTier.SOL),
        temperature_profile=((ModelTier.LUNA, 0.0), (ModelTier.TERRA, 0.0), (ModelTier.SOL, 0.0)),
    )
    return PromptRelease(
        digest=sha256(canonical_json(values).encode("utf-8")).hexdigest(), **values
    )


def _manifest(release: PromptRelease) -> PromptReleaseManifest:
    values = {
        "schema_version": "v1",
        "release": release.model_dump(mode="python"),
        "output_schema_hash": "0" * 64,
    }
    return PromptReleaseManifest(
        manifest_hash=sha256(canonical_json(values).encode("utf-8")).hexdigest(), **values
    )


def _pin(manager: PromptReleaseManager, release_id: str) -> PromptPin:
    manager.activate(release_id)
    return manager.current()


class _Answer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str = Field(min_length=1)


class _Clock:
    def now(self) -> float:
        return 1.0

    def sleep(self, _seconds: float) -> None:
        return None


class _Audit:
    def record(self, _event: AuditEvent) -> None:
        return None


class _Client:
    def __init__(self) -> None:
        self.requests = []

    def invoke(self, request):
        self.requests.append(request)
        return ModelResponse(
            content='{"answer":"known"}',
            usage=AgentUsage(
                input_tokens=2, output_tokens=1, cost_usd=0.001, model_tier=ModelTier.LUNA,
                pricing_version="openai-standard-2026-08-01", pricing_model_id="gpt-5.6-luna",
                pricing_band="short",
            ),
        )


def test_explicit_ingress_prompt_pin_survives_mid_run_activation(tmp_path) -> None:
    """Refreshing current() per call would make the provider system prefix change."""
    first, second = _release("release-a", "Stable global release A."), _release("release-b", "Stable global release B.")
    manager = PromptReleaseManager(manifests=(_manifest(first), _manifest(second)),
                                   registry_path=tmp_path / "prompts.db")
    pin = _pin(manager, first.release_id)
    client = _Client()
    schema = OutputSchema(schema_id="answer-v1", model=_Answer)
    driver = AgentDriver(
        model_client=client, audit_observer=_Audit(), clock=_Clock(), random=lambda low, high: high,
        prompt_releases=manager, output_schemas=(schema,),
        retry_policy=RetryPolicy(max_attempts=1, base_delay=0.1),
        circuit_breaker=CircuitBreaker(failure_threshold=3, cooldown=10.0),
        fallback_policy=FallbackPolicy((ModelTier.SOL, ModelTier.TERRA, ModelTier.LUNA)),
        model_costs={ModelTier.SOL: 0.1, ModelTier.TERRA: 0.1, ModelTier.LUNA: 0.1},
    )
    invocation = AgentInvocation(
        trace_id="1" * 32, run_id="run-1", task_id="task-1", task_kind="extract",
        prompt_release_id=first.release_id, prompt_release_digest=first.digest,
        allowed_model_tier=ModelTier.LUNA, deadline_epoch=10.0, max_attempts=1, cost_limit_usd=1.0,
        output_schema_id=schema.schema_id, output_schema_digest=schema.digest, user_payload={"query": "q"},
    )
    manager.activate(second.release_id)

    result = driver.execute(invocation, prompt_pin=pin)

    assert result.failure is None
    assert client.requests[0].messages[0][1].startswith(first.stable_system_prefix)


def test_retry_keeps_the_same_prompt_pin_when_release_activates_between_attempts(tmp_path) -> None:
    """A retry that reads current() again would switch from release A to B."""
    first, second = _release("release-a", "Stable global release A."), _release("release-b", "Stable global release B.")
    manager = PromptReleaseManager(manifests=(_manifest(first), _manifest(second)),
                                   registry_path=tmp_path / "prompts.db")
    pin = _pin(manager, first.release_id)
    schema = OutputSchema(schema_id="answer-v1", model=_Answer)

    class RetryClient(_Client):
        def invoke(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                manager.activate(second.release_id)
                raise TimeoutError("retry")
            return ModelResponse(
                content='{"answer":"known"}',
                usage=AgentUsage(
                    input_tokens=2, output_tokens=1, cost_usd=0.001,
                    model_tier=ModelTier.LUNA,
                    pricing_version="openai-standard-2026-08-01",
                    pricing_model_id="gpt-5.6-luna", pricing_band="short",
                ),
            )

    client = RetryClient()
    driver = AgentDriver(
        model_client=client, audit_observer=_Audit(), clock=_Clock(),
        random=lambda low, high: low, prompt_releases=manager, output_schemas=(schema,),
        retry_policy=RetryPolicy(max_attempts=2, base_delay=0.1),
        circuit_breaker=CircuitBreaker(failure_threshold=3, cooldown=10.0),
        fallback_policy=FallbackPolicy((ModelTier.SOL, ModelTier.TERRA, ModelTier.LUNA)),
        model_costs={ModelTier.SOL: 0.1, ModelTier.TERRA: 0.1, ModelTier.LUNA: 0.1},
    )
    invocation = AgentInvocation(
        trace_id="1" * 32, run_id="run-1", task_id="task-1", task_kind="extract",
        prompt_release_id=first.release_id, prompt_release_digest=first.digest,
        allowed_model_tier=ModelTier.LUNA, deadline_epoch=10.0, max_attempts=2,
        cost_limit_usd=1.0, output_schema_id=schema.schema_id,
        output_schema_digest=schema.digest, user_payload={"query": "q"},
    )

    result = driver.execute(invocation, prompt_pin=pin)

    assert result.failure is None
    assert len(client.requests) == 2
    assert client.requests[0].messages[0] == client.requests[1].messages[0]
    assert client.requests[1].messages[0][1].startswith(first.stable_system_prefix)


def test_specialist_invocation_is_bound_to_frozen_base_and_profile_component(tmp_path) -> None:
    """Rebuilding from the mutable profile instead of the workflow pin loses the base release."""
    from market_agent.workflow_contracts import (
        AgentTask, ContextSummary, ModelTier as TaskModelTier, SummaryCompleteness,
        TaskDifficulty, TaskType,
    )

    first, second = _release("release-a", "Stable global release A."), _release("release-b", "Stable global release B.")
    manager = PromptReleaseManager(manifests=(_manifest(first), _manifest(second)),
                                   registry_path=tmp_path / "prompts.db")
    base = _pin(manager, first.release_id)
    profile = profile_for(TaskType.TECHNICAL)
    schema = profile.output_schema()
    workflow_pin = WorkflowPromptPin.capture(base, (
        (profile.release(), schema.digest),
        (reflection_release(), reflection_output_schema().digest),
    ))
    task = AgentTask(
        task_id="task-technical", workflow_id="workflow-1", trace_id="1" * 32,
        task_type=TaskType.TECHNICAL, objective="Check the supplied setup.",
        context_summary_id="summary-technical", allowed_data=profile.allowed_data,
        allowed_tools=(), expected_output=profile.profile_id,
        acceptance_criteria=("Return cited evidence.",), difficulty=TaskDifficulty.NORMAL,
        model_tier=TaskModelTier.TERRA, prompt_version=profile.profile_id,
        attempt_timeout_seconds=30, maximum_retries=1, reserved_cost=0.05,
        remaining_workflow_cost=0.10, analysis_steps=profile.analysis_steps,
        escalation_rule="return_to_coordinator", conflict_return_rule="return_typed_conflict",
    )
    context = ContextSummary(
        summary_id=task.context_summary_id, task_id=task.task_id,
        workflow_id=task.workflow_id, trace_id=task.trace_id,
        user_objective=task.objective, immutable_constraints=("No execution authority.",),
        source_references=("source-1",), token_estimate=10,
        completeness=SummaryCompleteness.COMPLETE, summary_version="test-v1",
        summarizer_model=TaskModelTier.LUNA, source_record_hash="c" * 64,
    )
    manager.activate(second.release_id)

    invocation = build_invocation(task, context, deadline_epoch=10.0, prompt_pin=workflow_pin)
    component = workflow_pin.component(profile.profile_id, schema.digest)

    assert invocation.prompt_release_digest == component.release.digest
    effective_prefix = workflow_pin.system_prefix(profile.profile_id, schema.digest)
    assert effective_prefix.startswith(first.stable_system_prefix)
    assert profile.stable_prefix in effective_prefix


def test_objective_patch_converts_an_invalid_decision_to_safe_no_trade() -> None:
    """Removing the patch generator must make numeric correction unavailable."""
    context = CorrectionContext(
        target_hash="a" * 64,
        reflection_hash="b" * 64,
        error_codes=("numeric_consistency",),
        field_paths=(
            "/action",
            "/execute_now",
            "/entry_price",
            "/stop_price",
            "/observation_scenario",
        ),
        evidence_ids=("source-1",),
        retry_ordinal=1,
        prior_output_summary='{"action":"long","entry_price":100.0,"stop_price":101.0}',
        original_task_summary="make a bounded decision",
    )

    patch = AgentCoordinatorServices._patch(context)

    assert {item.path: item.value for item in patch.replacements} == {
        "/action": "no_trade",
        "/execute_now": False,
        "/entry_price": None,
        "/stop_price": None,
        "/observation_scenario": None,
    }


def test_objective_patch_rejects_changes_outside_reported_error_paths() -> None:
    """A one-field verifier error must not authorize a coupled six-field patch."""
    context = CorrectionContext(
        target_hash="a" * 64,
        reflection_hash="b" * 64,
        error_codes=("numeric_consistency",),
        field_paths=("/stop_price",),
        evidence_ids=("source-1",),
        retry_ordinal=1,
        prior_output_summary='{"action":"long","entry_price":100.0,"stop_price":101.0}',
        original_task_summary="make a bounded decision",
    )

    with pytest.raises(ValueError, match="reported paths"):
        AgentCoordinatorServices._patch(context)


def test_correction_uses_objective_errors_then_stops_before_actions_on_cancellation() -> None:
    """Removing the cancellation boundary would allow patch/rewrite work after cancellation."""
    from market_agent.workflow_contracts import Action, DecisionDraft, KnowledgeStatus
    from market_agent.workflow_reflection_agent import (
        ObjectiveCheck, ReflectionResult, correct_output,
    )

    decision = DecisionDraft(
        knowledge_status=KnowledgeStatus.KNOWN, uncertainty_reason=None,
        action=Action.LONG, execute_now=True, entry_price=100.0, stop_price=101.0,
        decision_confidence=0.7, selected_setup="long",
    )
    target_hash = sha256(canonical_json(decision.model_dump(mode="json")).encode()).hexdigest()
    checks = tuple(ObjectiveCheck(
        code=code,
        status="fail" if code == "numeric_consistency" else "pass",
        field_paths=("/stop_price",) if code == "numeric_consistency" else (),
    ) for code in (
        "schema_valid", "numeric_consistency", "direction_consistency",
        "evidence_support", "uncertainty_consistency", "risk_invariants",
    ))
    reflection = ReflectionResult(
        target_kind="decision_planner", target_hash=target_hash,
        output_schema_hash=sha256(canonical_json(DecisionDraft.model_json_schema()).encode()).hexdigest(),
        checks=checks, available=True, disposition="retry_original",
    )
    actions: list[str] = []

    outcome = correct_output(
        decision, reflection, task_summary="bounded task",
        generate_patch=lambda _context: actions.append("patch"),
        generate_rewrite=lambda _context: actions.append("rewrite"),
        reflect=lambda _candidate: actions.append("reflect"),
        allowed_paths=("/stop_price",), output_model=DecisionDraft,
        cancellation_check=lambda: True,
    )

    assert actions == []
    assert outcome.disposition == "safe_reject"
    assert outcome.attempted_modes == ()


def test_production_commit_keeps_ingress_prompt_digest_after_activation(tmp_path, monkeypatch) -> None:
    """Reading current() during commit would stamp the accepted result with release B."""
    from market_agent.backend.settings import BackendSettings
    from market_agent.workflow_contracts import (
        Action, InformationalAnswer, KnowledgeStatus, TerminalMode, WorkflowRequest, WorkflowResult,
    )
    from market_agent.workflow_production_application import (
        ProductionDependencies, ProductionWorkflowApplication, _capture_workflow_prompt_pin,
    )
    import market_agent.workflow_production_application as production_module

    first, second = _release("release-a", "Stable global release A."), _release("release-b", "Stable global release B.")
    manager = PromptReleaseManager(manifests=(_manifest(first), _manifest(second)),
                                   registry_path=tmp_path / "prompts.db")
    _pin(manager, first.release_id)
    expected_digest = _capture_workflow_prompt_pin(manager)[1]
    request = WorkflowRequest(
        workflow_id="workflow-1", trace_id="1" * 32,
        user_query="Does this system trade?", trigger_reason="manual_once",
    )
    result = WorkflowResult(
        workflow_id=request.workflow_id, trace_id=request.trace_id,
        terminal_mode=TerminalMode.INFORMATIONAL, final_action=Action.NO_TRADE,
        knowledge_status=KnowledgeStatus.KNOWN, uncertainty_reason=None,
        informational_answer=InformationalAnswer(
            knowledge_status=KnowledgeStatus.KNOWN, uncertainty_reason=None,
            answer="No.",
        ),
    )
    stored: list[str] = []
    monkeypatch.setattr(production_module, "_lookup_historical_answer", lambda **_kwargs: result)
    monkeypatch.setattr(production_module, "_store_historical_answer",
                        lambda **kwargs: stored.append(kwargs["prompt_release_digest"]))
    dependencies = ProductionDependencies(
        settings=BackendSettings(environment="test"),
        driver_factory=lambda *_args: object(), audit_writer=object(),
        memory_repository=None, embedding_client=None,
        completion_hook=lambda _request, _result, _proof: None, prompt_release_manager=manager,
    )
    application = ProductionWorkflowApplication(lambda: dependencies)

    application.execute_workflow(request)
    manager.activate(second.release_id)
    monkeypatch.setattr(
        "market_agent.workflow_memory_result_writer.verify_committed_transition_receipt",
        lambda _receipt: True,
    )
    proof = AcceptedOutcomeProof.bind(
        request, result,
        terminal_receipt=_unsigned_terminal_receipt(
            request, expected_digest,
            sha256(canonical_json(result.model_dump(mode="json")).encode("utf-8")).hexdigest(),
        ),
        prompt_release_digest=expected_digest, accepted_at=NOW,
    )
    application.commit_accepted_result(request, result, proof=proof)

    assert stored == [expected_digest]
    assert manager.current().release_digest == second.digest

    with pytest.raises(RuntimeError, match="prompt"):
        application.commit_accepted_result(
            request, result,
            proof=proof.model_copy(update={"prompt_release_digest": "f" * 64}),
        )


def test_memory_writer_marks_host_accepted_outcome_verified(tmp_path, monkeypatch) -> None:
    """Changing accepted outcome verification back to false must fail this test."""
    from market_agent.workflow_contracts import (
        Action, InformationalAnswer, KnowledgeStatus, TerminalMode, WorkflowResult,
    )

    authority = object()
    with SQLiteMemoryRepository(tmp_path / "memory.db", writer_authority=authority) as repository:
        writer = MemoryResultWriter(repository=repository, authority=authority, tenant_id="tenant-a")
        result = WorkflowResult(
            workflow_id="wf-1", trace_id="1" * 32,
            terminal_mode=TerminalMode.INFORMATIONAL, final_action=Action.NO_TRADE,
            knowledge_status=KnowledgeStatus.KNOWN, uncertainty_reason=None,
            informational_answer=InformationalAnswer(
                knowledge_status=KnowledgeStatus.KNOWN, uncertainty_reason=None, answer="不会。",
            ),
        )

        request = _workflow_request()
        monkeypatch.setattr(
            "market_agent.workflow_memory_result_writer.verify_committed_transition_receipt",
            lambda _receipt: True,
        )
        proof = AcceptedOutcomeProof.bind(
            request, result,
            terminal_receipt=_unsigned_terminal_receipt(
                request, "e" * 64,
                sha256(canonical_json(result.model_dump(mode="json")).encode("utf-8")).hexdigest(),
            ),
            prompt_release_digest="e" * 64,
            accepted_at=NOW,
        )
        writer.record(request, result, proof)

        outcomes = [item for item in repository.list_records(tenant_id="tenant-a") if isinstance(item, OutcomeRecord)]
        assert len(outcomes) == 1
        assert outcomes[0].verified is True


def test_memory_writer_rejects_mismatched_acceptance_proof_before_writes(tmp_path, monkeypatch) -> None:
    """Trusting a caller-supplied verified flag would persist a mismatched result."""
    from market_agent.workflow_contracts import (
        Action, InformationalAnswer, KnowledgeStatus, TerminalMode, WorkflowResult,
    )

    authority = object()
    with SQLiteMemoryRepository(tmp_path / "memory.db", writer_authority=authority) as repository:
        writer = MemoryResultWriter(repository=repository, authority=authority, tenant_id="tenant-a")
        result = WorkflowResult(
            workflow_id="wf-1", trace_id="1" * 32,
            terminal_mode=TerminalMode.INFORMATIONAL, final_action=Action.NO_TRADE,
            knowledge_status=KnowledgeStatus.KNOWN, uncertainty_reason=None,
            informational_answer=InformationalAnswer(
                knowledge_status=KnowledgeStatus.KNOWN, uncertainty_reason=None, answer="不会。",
            ),
        )
        request = _workflow_request()
        monkeypatch.setattr(
            "market_agent.workflow_memory_result_writer.verify_committed_transition_receipt",
            lambda _receipt: True,
        )
        proof = _untrusted_proof(request, result, "e" * 64).model_copy(
            update={"result_digest": "f" * 64}
        )

        with pytest.raises(ValueError, match="result digest"):
            writer.record(request, result, proof)

        assert repository.list_records(tenant_id="tenant-a") == ()


def test_memory_writer_rejects_arbitrary_receipt_digest_before_writes(tmp_path) -> None:
    """A syntactically valid 64-hex string is not host acceptance authority."""
    from market_agent.workflow_contracts import (
        Action, InformationalAnswer, KnowledgeStatus, TerminalMode, WorkflowResult,
    )

    authority = object()
    with SQLiteMemoryRepository(tmp_path / "memory.db", writer_authority=authority) as repository:
        writer = MemoryResultWriter(repository=repository, authority=authority, tenant_id="tenant-a")
        result = WorkflowResult(
            workflow_id="wf-1", trace_id="1" * 32,
            terminal_mode=TerminalMode.INFORMATIONAL, final_action=Action.NO_TRADE,
            knowledge_status=KnowledgeStatus.KNOWN, uncertainty_reason=None,
            informational_answer=InformationalAnswer(
                knowledge_status=KnowledgeStatus.KNOWN, uncertainty_reason=None, answer="不会。",
            ),
        )
        request = _workflow_request()
        proof = _untrusted_proof(request, result, "e" * 64)

        with pytest.raises(ValueError, match="host-signed"):
            writer.record(request, result, proof)

        assert repository.list_records(tenant_id="tenant-a") == ()


def test_builtin_objective_numeric_failure_is_repaired_by_patch_without_rewrite() -> None:
    """The built-in verifier must report the complete coupled invariant patch scope."""
    from market_agent.workflow_contracts import (
        Action, ContextSummary, DecisionDraft, KnowledgeStatus, ModelTier as TaskModelTier,
        SummaryCompleteness,
    )
    from market_agent.workflow_reflection_agent import (
        ObjectiveReview, correct_output, reflect_output,
    )

    decision = DecisionDraft(
        knowledge_status=KnowledgeStatus.KNOWN, uncertainty_reason=None,
        action=Action.LONG, execute_now=True, entry_price=100.0, stop_price=101.0,
        decision_confidence=0.7, selected_setup="long",
    )
    context = ContextSummary(
        summary_id="summary-1", task_id="decision-1", workflow_id="workflow-1",
        trace_id="1" * 32, user_objective="make a bounded decision",
        immutable_constraints=("No execution authority.",), source_references=("source-1",),
        token_estimate=10, completeness=SummaryCompleteness.COMPLETE,
        summary_version="test-v1", summarizer_model=TaskModelTier.LUNA,
        source_record_hash="c" * 64,
    )

    def reviewer(request):
        return ObjectiveReview(
            target_hash=request.target_hash,
            output_schema_hash=request.output_schema_hash,
            checks=request.deterministic_checks,
        )

    def reflect(candidate):
        return reflect_output(
            candidate, target_kind="decision_planner", context=context,
            output_model=DecisionDraft, reviewer=reviewer,
        )

    initial = reflect(decision)
    rewrites: list[CorrectionContext] = []

    def rewrite(correction: CorrectionContext) -> DecisionDraft:
        rewrites.append(correction)
        return DecisionDraft(
            knowledge_status=KnowledgeStatus.INSUFFICIENT,
            uncertainty_reason="不知道：objective verification failed",
            action=Action.NO_TRADE,
            execute_now=False,
            decision_confidence=0.0,
        )

    outcome = correct_output(
        decision, initial, task_summary="make a bounded decision",
        generate_patch=AgentCoordinatorServices._patch,
        generate_rewrite=rewrite,
        reflect=reflect,
        allowed_paths=(
            "/action", "/execute_now", "/entry_price", "/stop_price",
            "/observation_scenario", "/uncertainty_reason",
        ),
        output_model=DecisionDraft,
    )

    repaired = DecisionDraft.model_validate_json(outcome.output_json)
    assert outcome.disposition == "accept"
    assert outcome.attempted_modes == ("patch",)
    assert repaired.action is Action.NO_TRADE
    assert rewrites == []


def test_promotion_scheduler_requires_verified_outcome_and_bounded_host_evaluation(tmp_path) -> None:
    """Removing host verification or the one-candidate bound must fail this test."""
    from market_agent.workflow_memory_promotion import PromotionScheduler

    authority = object()
    with SQLiteMemoryRepository(tmp_path / "memory.db", writer_authority=authority) as repository:
        event = EventRecord(
            record_id="event-1", tenant_id="tenant-a", observed_at=NOW, source="exchange",
            payload={"fact": "funding"},
            provenance=Provenance(source_id="exchange", source_kind="external", independent_group="exchange"),
        )
        repository.append_event(event, tenant_id="tenant-a", trace_id="trace-1",
                                idempotency_key="event", authority=authority)
        repository.append_event(EventRecord(
            record_id="event-2", tenant_id="tenant-a", observed_at=NOW,
            source="independent", payload={"fact": "risk"},
            provenance=Provenance(
                source_id="independent", source_kind="external",
                independent_group="independent",
            ),
        ), tenant_id="tenant-a", trace_id="trace-1",
            idempotency_key="event-2", authority=authority)
        candidate = KnowledgeRevision(
            record_id="knowledge-1", tenant_id="tenant-a", observed_at=NOW,
            knowledge_id="funding", revision=1, rule="Check funding before entry.",
            confidence=0.9, effective_at=NOW,
            evidence_ids=("event-1", "event-2"), outcome_id="outcome-1",
        )
        evaluations = []
        scheduler = PromotionScheduler(repository=repository, authority=authority,
                                       tenant_id="tenant-a", max_candidates_per_run=1,
                                       evaluation_observer=evaluations.append)

        assert scheduler.evaluate((candidate,), now=NOW, trace_id="trace-1") == ()
        assert evaluations[-1].status == "rejected"
        assert evaluations[-1].reason_code == "verified_outcome_required"
        repository.append_decision(DecisionRecord(
            record_id="decision-1", tenant_id="tenant-a", observed_at=NOW,
            decision="no_trade", status="final", evidence_ids=("event-1",),
        ), tenant_id="tenant-a", trace_id="trace-1", idempotency_key="decision", authority=authority)
        repository.append_outcome(OutcomeRecord(
            record_id="outcome-1", tenant_id="tenant-a", observed_at=NOW,
            decision_id="decision-1", result="no_trade", verified=True, evidence_ids=("event-1",),
        ), tenant_id="tenant-a", trace_id="trace-1", idempotency_key="outcome", authority=authority)
        promoted = scheduler.evaluate((candidate, candidate.model_copy(update={"record_id": "knowledge-2"})),
                                      now=NOW, trace_id="trace-1")

        assert len(promoted) == 1
        assert promoted[0].record_id == "knowledge-1"
        assert evaluations[-1].status == "promoted"

        assert scheduler.evaluate((candidate,), now=NOW, trace_id="trace-1") == promoted
        assert [item.status for item in evaluations].count("promoted") == 1

        cancelled_candidate = candidate.model_copy(update={
            "record_id": "knowledge-cancelled", "knowledge_id": "cancelled",
        })
        assert scheduler.evaluate(
            (cancelled_candidate,), now=NOW, trace_id="trace-cancelled",
            cancellation_check=lambda: True,
        ) == ()
        assert evaluations[-1].reason_code == "cancelled"
        assert repository.get_by_id("knowledge-cancelled", tenant_id="tenant-a") is None


def test_promotion_scheduler_audits_expected_rejection_but_surfaces_infrastructure_failure(tmp_path) -> None:
    """Catching every exception would hide a storage outage as ordinary ineligibility."""
    from market_agent.workflow_memory_promotion import PromotionScheduler

    authority = object()
    evaluations = []
    with SQLiteMemoryRepository(tmp_path / "memory.db", writer_authority=authority) as repository:
        for identifier in ("event-1", "event-2"):
            repository.append_event(EventRecord(
                record_id=identifier, tenant_id="tenant-a", observed_at=NOW,
                source="exchange", payload={"fact": identifier},
                provenance=Provenance(
                    source_id=identifier, source_kind="external", independent_group="same-source",
                ),
            ), tenant_id="tenant-a", trace_id="trace-1",
                idempotency_key=identifier, authority=authority)
        repository.append_decision(DecisionRecord(
            record_id="decision-1", tenant_id="tenant-a", observed_at=NOW,
            decision="no_trade", status="final", evidence_ids=("event-1",),
        ), tenant_id="tenant-a", trace_id="trace-1", idempotency_key="decision", authority=authority)
        repository.append_outcome(OutcomeRecord(
            record_id="outcome-1", tenant_id="tenant-a", observed_at=NOW,
            decision_id="decision-1", result="no_trade", verified=True,
            evidence_ids=("event-1",),
        ), tenant_id="tenant-a", trace_id="trace-1", idempotency_key="outcome", authority=authority)
        candidate = KnowledgeRevision(
            record_id="knowledge-1", tenant_id="tenant-a", observed_at=NOW,
            knowledge_id="funding", revision=1, rule="Check funding before entry.",
            confidence=0.9, effective_at=NOW,
            evidence_ids=("event-1", "event-2"), outcome_id="outcome-1",
        )
        scheduler = PromotionScheduler(
            repository=repository, authority=authority, tenant_id="tenant-a",
            evaluation_observer=evaluations.append,
        )

        assert scheduler.evaluate((candidate,), now=NOW, trace_id="trace-1") == ()
        assert evaluations[-1].status == "rejected"
        assert evaluations[-1].reason_code == "repository_rejected"

    class BrokenRepository:
        def get_by_id(self, *_args, **_kwargs):
            raise RuntimeError("database unavailable")

    broken = PromotionScheduler(
        repository=BrokenRepository(), authority=authority, tenant_id="tenant-a",
        evaluation_observer=evaluations.append,
    )
    with pytest.raises(RuntimeError, match="database unavailable"):
        broken.evaluate((candidate,), now=NOW, trace_id="trace-1")
