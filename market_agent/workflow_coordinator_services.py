"""Concrete coordinator decision and objective reflection services."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from hashlib import sha256

from market_agent.workflow_agent_driver import AgentDriver
from market_agent.workflow_agents.common import profile_for, run_node
from market_agent.workflow_capabilities import CapabilityGrant, CapabilityIssuer, CapabilityScope
from market_agent.workflow_contracts import (
    Action,
    AgentReport,
    AgentTask,
    ContextSummary,
    CoordinatorPlan,
    DecisionDraft,
    KnowledgeStatus,
    ModelTier,
    SummaryCompleteness,
    TaskDifficulty,
    TaskType,
    TechnicalAnalysis,
    WorkflowRequest,
)
from market_agent.workflow_prompt_release import canonical_json
from market_agent.workflow_prompt_config import WorkflowPromptPin
from market_agent.workflow_memory_retrieval import CoreExperienceSummary
from market_agent.workflow_reflection_agent import (
    CorrectionContext,
    CorrectionPatch,
    FieldReplacement,
    ObjectiveReview,
    ReflectionRequest,
    run_reflection,
)
from market_agent.workflow_decision_verifier import ObjectiveDecisionVerifier


def _digest(value: object) -> str:
    return sha256(canonical_json(value).encode("utf-8")).hexdigest()


class AgentCoordinatorServices:
    """Run the coordinator on Terra and verify its core decision once with Luna."""

    def __init__(self, *, driver: AgentDriver, issuer: CapabilityIssuer,
                 tenant_id: str, deadline_epoch: float,
                 memory_context: CoreExperienceSummary | None = None,
                 memory_scope: str | None = None,
                 prompt_pin: WorkflowPromptPin | None = None,
                 clock: Callable[[], float] = time.time,
                 cancellation_check: Callable[[], bool] = lambda: False) -> None:
        if type(driver) is not AgentDriver or type(issuer) is not CapabilityIssuer:
            raise TypeError("coordinator services require host-owned driver and issuer")
        self._driver = driver
        self._issuer = issuer
        self._tenant = tenant_id
        self._deadline = deadline_epoch
        self._clock = clock
        self._memory = memory_context
        self._memory_scope = memory_scope
        self._prompt_pin = prompt_pin
        self._cancellation_check = cancellation_check
        self._decision_contexts: dict[str, ContextSummary] = {}
        self.verifier = ObjectiveDecisionVerifier(
            context_factory=self._verification_context,
            reviewer=self._review,
            generate_patch=self._patch,
            generate_rewrite=self._rewrite,
            cancellation_check=self._cancellation_check,
        )

    def decide(self, request: WorkflowRequest, plan: CoordinatorPlan,
               reports: tuple[AgentReport, ...]) -> DecisionDraft | None:
        if any(report.status.value != "completed" for report in reports):
            return None
        task = self._decision_task(request, plan)
        context = self._decision_context(request, task, reports)
        self._decision_contexts[request.workflow_id] = context
        grant = self._grant(task)
        result = run_node(task, context, self._driver, deadline_epoch=self._deadline,
                          grant=grant, authorize=self._authorize_task,
                          execution_node="decide",
                          memory_context=self._memory,
                          memory_tenant_id=self._tenant if self._memory is not None else None,
                          memory_scope=self._memory_scope if self._memory is not None else None,
                          prompt_pin=self._prompt_pin,
                          cancellation_check=self._cancellation_check)
        if result.status.value != "completed":
            return None
        # ``summary`` is the canonical JSON handoff.  Strict models require
        # enum decoding through the JSON boundary; validating a decoded Python
        # dict would reject the otherwise valid string enum values.
        return DecisionDraft.model_validate_json(result.summary)

    @staticmethod
    def technical(reports: tuple[AgentReport, ...]) -> TechnicalAnalysis | None:
        for report in reports:
            if report.status.value != "completed":
                continue
            try:
                value = json.loads(report.summary)
                if {"market_regime", "extension_state", "long_setup", "short_setup", "data_quality"} <= set(value):
                    return TechnicalAnalysis.model_validate(value)
            except (ValueError, TypeError):
                continue
        return None

    def _decision_task(self, request: WorkflowRequest, plan: CoordinatorPlan) -> AgentTask:
        profile = profile_for(TaskType.DECISION_PLANNER)
        identifier = sha256(f"{request.workflow_id}:{plan.revision}:decision".encode()).hexdigest()[:24]
        return AgentTask(
            task_id=identifier, workflow_id=request.workflow_id, trace_id=request.trace_id,
            task_type=TaskType.DECISION_PLANNER, objective=request.user_query,
            context_summary_id="summary-" + identifier, allowed_data=profile.allowed_data,
            allowed_tools=(), expected_output=profile.profile_id,
            acceptance_criteria=("Return one evidence-bound decision.", "Use no_trade when support is insufficient."),
            difficulty=TaskDifficulty.NORMAL, model_tier=ModelTier.TERRA,
            prompt_version=profile.profile_id, attempt_timeout_seconds=30,
            maximum_retries=1, reserved_cost=0.10, remaining_workflow_cost=0.10,
            analysis_steps=profile.analysis_steps, escalation_rule="return_to_coordinator",
            conflict_return_rule="return_typed_conflict",
        )

    @staticmethod
    def _decision_context(request: WorkflowRequest, task: AgentTask,
                          reports: tuple[AgentReport, ...]) -> ContextSummary:
        evidence = tuple(sorted({item for report in reports for item in report.evidence_refs}))
        summaries = tuple(report.summary[:2000] for report in reports)
        material = {"reports": summaries, "evidence": evidence, "revision": task.task_id}
        return ContextSummary(
            summary_id=task.context_summary_id, task_id=task.task_id,
            workflow_id=request.workflow_id, trace_id=request.trace_id,
            user_objective=request.user_query,
            immutable_constraints=("No execution authority.", "Abstain when evidence is insufficient."),
            prior_conclusions=summaries, token_estimate=min(8000, sum(len(item) for item in summaries) // 3),
            completeness=SummaryCompleteness.COMPLETE, summary_version="coordinator-v1",
            summarizer_model=ModelTier.LUNA, source_record_hash=_digest(material),
            source_references=evidence,
        )

    def _grant(self, task: AgentTask) -> CapabilityGrant:
        return self._issuer.issue(
            scope=CapabilityScope(actor_id="coordinator", task_id=task.task_id,
                                  tenant_id=self._tenant, trace_id=task.trace_id),
            ttl_seconds=max(1.0, min(300.0, self._deadline - self._clock())),
            readable_resources=("context_summary",),
        )

    def _authorize_task(self, task: AgentTask, grant: CapabilityGrant) -> None:
        scope = CapabilityScope(actor_id="coordinator", task_id=task.task_id,
                                tenant_id=self._tenant, trace_id=task.trace_id)
        self._issuer.authorize_read(grant, scope=scope, resource="context_summary")

    def _verification_context(self, request: WorkflowRequest, plan: CoordinatorPlan,
                              reports: tuple[AgentReport, ...], decision: DecisionDraft) -> ContextSummary:
        del plan, reports, decision
        context = self._decision_contexts.get(request.workflow_id)
        if context is None:
            raise ValueError("decision context is unavailable")
        return context

    def _review(self, request: ReflectionRequest) -> ObjectiveReview:
        scope = CapabilityScope(actor_id="reflector", task_id=request.task_id,
                                tenant_id=self._tenant, trace_id=request.trace_id)
        grant = self._issuer.issue(scope=scope,
            ttl_seconds=max(1.0, min(300.0, self._deadline - self._clock())),
            readable_resources=("reflection_target",))

        def authorize(value: ReflectionRequest, supplied: CapabilityGrant) -> None:
            checked = CapabilityScope(actor_id="reflector", task_id=value.task_id,
                                      tenant_id=self._tenant, trace_id=value.trace_id)
            self._issuer.authorize_read(supplied, scope=checked, resource="reflection_target")

        return run_reflection(request, self._driver, deadline_epoch=self._deadline,
                              cost_limit_usd=0.02, grant=grant, authorize=authorize,
                              prompt_pin=self._prompt_pin,
                              cancellation_check=self._cancellation_check)

    @staticmethod
    def _patch(context: CorrectionContext) -> CorrectionPatch:
        """Return the sole safe deterministic correction for a failed draft.

        Objective verification may identify an invalid numeric or uncertainty
        field, but it must never invent a replacement price or evidence.  The
        only universally valid field-level repair is therefore to remove the
        actionable fields and abstain.  It remains a *patch* (not a new model
        answer), is fully derived from the verifier error tuple, and is later
        accepted only when its objective review improves.
        """
        context = CorrectionContext.model_validate(context)
        if not context.error_codes:
            raise ValueError("objective patch requires verifier errors")
        safe_codes = {
            "numeric_consistency",
            "direction_consistency",
            "uncertainty_consistency",
        }
        if not set(context.error_codes) <= safe_codes:
            raise ValueError("objective errors do not authorize a deterministic patch")
        required_paths = {
            "/action",
            "/execute_now",
            "/entry_price",
            "/stop_price",
            "/observation_scenario",
        }
        if not required_paths <= set(context.field_paths):
            raise ValueError("safe no-trade patch exceeds objectively reported paths")
        return CorrectionPatch(
            target_hash=context.target_hash,
            replacements=(
                FieldReplacement(path="/action", value="no_trade"),
                FieldReplacement(path="/execute_now", value=False),
                FieldReplacement(path="/entry_price", value=None),
                FieldReplacement(path="/stop_price", value=None),
                FieldReplacement(path="/observation_scenario", value=None),
            ),
        )

    @staticmethod
    def _rewrite(context: CorrectionContext) -> DecisionDraft:
        return DecisionDraft(
            knowledge_status=KnowledgeStatus.INSUFFICIENT,
            uncertainty_reason="不知道：objective verification failed: " + ",".join(context.error_codes),
            action=Action.NO_TRADE, execute_now=False, decision_confidence=0.0,
        )
