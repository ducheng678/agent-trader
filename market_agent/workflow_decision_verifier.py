"""Objective, bounded verification and correction for core decision outputs."""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from market_agent.workflow_contracts import (
    AgentReport,
    ContextSummary,
    CoordinatorPlan,
    DecisionDraft,
    WorkflowRequest,
)
from market_agent.workflow_reflection_agent import (
    CorrectionContext,
    CorrectionPatch,
    ObjectiveReview,
    ReflectionRequest,
    ReflectionResult,
    correct_output,
    reflect_output,
)


DecisionContextFactory = Callable[
    [WorkflowRequest, CoordinatorPlan, tuple[AgentReport, ...], DecisionDraft],
    ContextSummary,
]
ReflectionReviewer = Callable[[ReflectionRequest], ObjectiveReview]
PatchGenerator = Callable[[CorrectionContext], CorrectionPatch]
RewriteGenerator = Callable[[CorrectionContext], DecisionDraft]


@dataclass(frozen=True, slots=True)
class ObjectiveDecisionVerifier:
    """Use Luna reflection only on the decision, then bound correction to two attempts."""

    context_factory: DecisionContextFactory
    reviewer: ReflectionReviewer
    generate_patch: PatchGenerator
    generate_rewrite: RewriteGenerator
    allowed_patch_paths: tuple[str, ...] = (
        "/action",
        "/execute_now",
        "/entry_price",
        "/stop_price",
        "/observation_scenario",
        "/decision_confidence",
        "/selected_setup",
        "/conflict_codes",
        "/knowledge_status",
        "/uncertainty_reason",
    )

    def __call__(
        self,
        request: WorkflowRequest,
        plan: CoordinatorPlan,
        reports: tuple[AgentReport, ...],
        decision: DecisionDraft,
    ) -> DecisionDraft | None:
        request = WorkflowRequest.model_validate(request)
        plan = CoordinatorPlan.model_validate(plan)
        reports = tuple(AgentReport.model_validate(report) for report in reports)
        decision = DecisionDraft.model_validate(decision)
        context = ContextSummary.model_validate(
            self.context_factory(request, plan, reports, decision)
        )
        if (context.workflow_id, context.trace_id) != (plan.workflow_id, plan.trace_id):
            raise ValueError("decision verification context crossed workflow identity")

        def reflect(candidate: DecisionDraft) -> ReflectionResult:
            return reflect_output(
                candidate,
                target_kind="decision_planner",
                context=context,
                output_model=DecisionDraft,
                reviewer=self.reviewer,
            )

        initial = reflect(decision)
        if initial.disposition == "accept":
            return decision
        if initial.disposition != "retry_original":
            return None
        outcome = correct_output(
            decision,
            initial,
            task_summary=request.user_query,
            generate_patch=self.generate_patch,
            generate_rewrite=self.generate_rewrite,
            reflect=reflect,
            allowed_paths=self.allowed_patch_paths,
            output_model=DecisionDraft,
        )
        if outcome.disposition != "accept" or outcome.output_json is None:
            return None
        return DecisionDraft.model_validate(json.loads(outcome.output_json))
