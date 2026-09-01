from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Literal

from pydantic import Field

from market_agent.workflow_agent_contracts import StrictModel
from market_agent.workflow_agent_driver import AgentDriver
from market_agent.workflow_agents.common import checked_context, profile_for, run_node
from market_agent.workflow_context_summary import ContextHandoff
from market_agent.workflow_contracts import (
    Action, AgentReport, AgentTask, ContextSummary, CoordinatorPlan, DecisionDraft,
    InformationalAnswer, KnowledgeStatus, ModelTier, ReportStatus, RiskAssessment,
    ShortText, TaskDifficulty, TaskType, TerminalMode, WorkflowBudgetState, WorkflowMode,
    WorkflowRequest, WorkflowResult,
)
from market_agent.workflow_model_routing import policy_for


class CoordinatorDirective(StrictModel):
    action: Literal["continue", "wait", "reschedule", "schedule_reconciliation", "safe_unknown"]
    task_ids: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=10)
    reason_codes: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=10)
    downgrade: bool = False


@dataclass(frozen=True, slots=True)
class DispatchSpec:
    task: AgentTask
    context: ContextSummary
    grant: object


def _task(request: WorkflowRequest, kind: TaskType, budget: WorkflowBudgetState, revision: int,
          parent: str | None = None, remaining_task_slots: int = 1) -> AgentTask:
    if type(remaining_task_slots) is not int or remaining_task_slots < 1:
        raise ValueError("remaining task slots must be a positive integer")
    profile = profile_for(kind)
    policy = policy_for("escalation" if kind is TaskType.RECONCILIATION else kind.value)
    identifier = sha256(f"{request.workflow_id}:{revision}:{kind.value}:{parent or ''}".encode()).hexdigest()[:24]
    # Keep enough authority for every task that is still required by the plan.
    # This also leaves any unallocated budget available for bounded recovery.
    reserved = min(float(policy.node_cost_cap), budget.remaining_cost / remaining_task_slots)
    if reserved <= 0 or budget.remaining_attempts < 1:
        raise ValueError("insufficient workflow budget for a specialist task")
    return AgentTask(task_id=identifier, parent_task_id=parent, workflow_id=request.workflow_id, trace_id=request.trace_id,
        task_type=kind, objective=request.user_query, context_summary_id="pending-" + identifier,
        allowed_data=profile.allowed_data, allowed_tools=(), expected_output=profile.profile_id,
        acceptance_criteria=("Return only the declared strict output.", "Cite supplied source evidence; unknown when unsupported."),
        difficulty=TaskDifficulty.HIGH if profile.tier.value == "sol" else TaskDifficulty.LOW if profile.tier.value == "luna" else TaskDifficulty.NORMAL,
        model_tier=ModelTier("gpt-5.6-" + profile.tier.value), prompt_version=profile.profile_id,
        attempt_timeout_seconds=policy.attempt_timeout_seconds,
        maximum_retries=min(3, policy.maximum_total_attempts - 1,
                            budget.remaining_attempts - remaining_task_slots),
        reserved_cost=reserved, remaining_workflow_cost=budget.remaining_cost,
        analysis_steps=profile.analysis_steps, escalation_rule="return_to_coordinator",
        conflict_return_rule="return_typed_conflict")


def plan_request(request: WorkflowRequest, budget: WorkflowBudgetState, *,
                 task_types: tuple[TaskType, ...] | None = None) -> CoordinatorPlan:
    request, budget = WorkflowRequest.model_validate(request), WorkflowBudgetState.model_validate(budget)
    kinds = task_types or ((TaskType.EVENT_FILTER, TaskType.FUNDAMENTAL, TaskType.TECHNICAL)
                          if budget.mode is WorkflowMode.PASSIVE else
                          (TaskType.MARKET_CONTEXT, TaskType.FUNDAMENTAL, TaskType.TECHNICAL))
    if not 1 <= len(kinds) <= 5 or len(set(kinds)) != len(kinds):
        raise ValueError("plans require one to five distinct catalogued specialists")
    # Preserve one global attempt for deterministic coordinator recovery when
    # the caller supplied capacity beyond the initial task set.  Per-task
    # retries are allocated from the remainder and can never consume it.
    remaining_attempts = budget.remaining_attempts - (1 if budget.remaining_attempts > len(kinds) else 0)
    remaining_cost = budget.remaining_cost
    tasks = []
    for index, kind in enumerate(kinds):
        slots = len(kinds) - index
        available = budget.model_copy(update={
            "remaining_attempts": remaining_attempts,
            "remaining_cost": remaining_cost,
        })
        task = _task(request, kind, available, 0, remaining_task_slots=slots)
        tasks.append(task)
        remaining_attempts -= task.maximum_retries + 1
        remaining_cost -= task.reserved_cost
    tasks = tuple(tasks)
    if sum(task.reserved_cost for task in tasks) > budget.remaining_cost + 1e-12 or len(tasks) > budget.remaining_attempts:
        raise ValueError("planned dispatch exceeds the remaining workflow budget")
    return CoordinatorPlan(workflow_id=request.workflow_id, trace_id=request.trace_id,
                           revision=0, mode=budget.mode, tasks=tasks)


def bind_contexts(plan: CoordinatorPlan, contexts: Mapping[str, ContextSummary | ContextHandoff]) -> CoordinatorPlan:
    plan = CoordinatorPlan.model_validate(plan)
    tasks = []
    for task in plan.tasks:
        context = contexts[task.task_id]
        summary = ContextHandoff.model_validate(context).summary if type(context) is ContextHandoff else ContextSummary.model_validate(context)
        task = AgentTask.model_validate(dict(task.model_dump(mode="python"), context_summary_id=summary.summary_id))
        checked_context(task, context)
        tasks.append(task)
    return CoordinatorPlan.model_validate(dict(plan.model_dump(mode="python"), tasks=tuple(tasks)))


def dispatch_tasks(plan: CoordinatorPlan, contexts: Mapping[str, ContextSummary | ContextHandoff],
                   driver: AgentDriver | Callable | None, grants: Mapping[str, object], *,
                   deadline_epoch: float | None = None, authorize: Callable | None = None) -> tuple:
    bound = bind_contexts(plan, contexts)
    specs = tuple(DispatchSpec(task, checked_context(task, contexts[task.task_id]), grants[task.task_id]) for task in bound.tasks)
    if any(spec.grant is None for spec in specs):
        raise PermissionError("every dispatch requires a host-issued capability")
    if driver is None:
        return specs
    reports = []
    for spec in specs:
        try:
            if isinstance(driver, AgentDriver):
                if deadline_epoch is None or authorize is None:
                    raise PermissionError("driver dispatch needs a deadline and host authorizer")
                report = run_node(spec.task, spec.context, driver, deadline_epoch=deadline_epoch,
                                  grant=spec.grant, authorize=authorize)
            else:
                report = driver(spec)
            report = AgentReport.model_validate(report)
            if (report.task_id, report.workflow_id, report.trace_id) != (spec.task.task_id, bound.workflow_id, bound.trace_id):
                raise ValueError("dispatcher returned a mismatched report")
        except Exception as error:
            report = AgentReport(task_id=spec.task.task_id, workflow_id=bound.workflow_id, trace_id=bound.trace_id,
                status=ReportStatus.FAILED, knowledge_status=KnowledgeStatus.INSUFFICIENT,
                uncertainty_reason="Authorized specialist dispatch did not yield a valid report.", summary="不知道",
                error_category="permission_denied" if isinstance(error, PermissionError) else "invalid_dispatch_result",
                retryable=False, safe_fallback=Action.NO_TRADE)
        reports.append(report)
    return tuple(reports)


def reconcile_reports(plan: CoordinatorPlan, reports: Sequence[AgentReport], budget: WorkflowBudgetState) -> CoordinatorDirective:
    plan, budget = CoordinatorPlan.model_validate(plan), WorkflowBudgetState.model_validate(budget)
    known = {task.task_id: task for task in plan.tasks}
    validated = {}
    conflicts, failed, missing = set(), set(), set()
    fatal = {"permission_denied", "audit_unavailable", "invalid_invocation", "authentication", "configuration", "budget_exhausted"}
    for raw in reports:
        report = AgentReport.model_validate(raw)
        if report.task_id not in known or (report.workflow_id, report.trace_id) != (plan.workflow_id, plan.trace_id):
            raise ValueError("report does not belong to this plan")
        if report.task_id in validated and validated[report.task_id] != report:
            conflicts.add(report.task_id)
        validated[report.task_id] = report
        if report.status is ReportStatus.CONFLICT:
            conflicts.add(report.task_id)
        elif report.status is ReportStatus.FAILED:
            if report.error_category in fatal or not report.retryable:
                return CoordinatorDirective(action="safe_unknown", task_ids=(report.task_id,), reason_codes=("non_retryable_failure",))
            failed.add(report.task_id)
        elif report.status is ReportStatus.UNCERTAIN:
            missing.add(report.task_id)
    if set(validated) != set(known):
        return CoordinatorDirective(action="wait", task_ids=tuple(sorted(set(known) - set(validated))), reason_codes=("pending_reports",))
    directions = {}
    for identifier, report in validated.items():
        if report.status is ReportStatus.COMPLETED:
            try:
                direction = json.loads(report.summary).get("action")
                if direction in ("long", "short"):
                    directions[identifier] = direction
            except (ValueError, AttributeError):
                pass
    if len(set(directions.values())) > 1:
        conflicts.update(directions)
    if not (conflicts or failed or missing):
        return CoordinatorDirective(action="continue")
    if budget.remaining_attempts < 1 or budget.remaining_cost <= 0 or plan.revision >= 3:
        return CoordinatorDirective(action="safe_unknown", reason_codes=("recovery_budget_exhausted",))
    if conflicts:
        return CoordinatorDirective(action="schedule_reconciliation", task_ids=tuple(sorted(conflicts)), reason_codes=("conflicting_evidence",))
    return CoordinatorDirective(action="reschedule", task_ids=tuple(sorted(failed or missing)),
        reason_codes=("transient_failure" if failed else "missing_evidence",), downgrade=bool(failed and plan.revision > 0))


def reschedule(plan: CoordinatorPlan, directive: CoordinatorDirective, budget: WorkflowBudgetState) -> CoordinatorPlan:
    plan, directive, budget = CoordinatorPlan.model_validate(plan), CoordinatorDirective.model_validate(directive), WorkflowBudgetState.model_validate(budget)
    if directive.action not in ("reschedule", "schedule_reconciliation") or plan.revision >= 3:
        raise ValueError("recovery is not authorized or the revision limit is exhausted")
    originals = {task.task_id: task for task in plan.tasks}
    if not directive.task_ids or not set(directive.task_ids) <= set(originals):
        raise ValueError("recovery must name tasks from the current plan")
    selected = directive.task_ids[:1] if directive.action == "schedule_reconciliation" else directive.task_ids
    tasks = []
    remaining_attempts = budget.remaining_attempts
    remaining_cost = budget.remaining_cost
    for index, identifier in enumerate(selected):
        original = originals[identifier]
        kind = TaskType.RECONCILIATION if directive.action == "schedule_reconciliation" else original.task_type
        request = WorkflowRequest(workflow_id=plan.workflow_id, trace_id=plan.trace_id,
                                  user_query=original.objective, trigger_reason="bounded_recovery")
        slots = len(selected) - index
        available = budget.model_copy(update={
            "remaining_attempts": remaining_attempts,
            "remaining_cost": remaining_cost,
        })
        task = _task(request, kind, available, plan.revision + 1, parent=identifier,
                     remaining_task_slots=slots)
        if directive.downgrade:
            tier = ModelTier.TERRA if original.model_tier is ModelTier.SOL else ModelTier.LUNA
            task = AgentTask.model_validate(dict(task.model_dump(mode="python"), model_tier=tier))
        tasks.append(task)
        remaining_attempts -= task.maximum_retries + 1
        remaining_cost -= task.reserved_cost
    if sum(task.reserved_cost for task in tasks) > budget.remaining_cost + 1e-12 or len(tasks) > budget.remaining_attempts:
        raise ValueError("recovery exceeds remaining workflow budget")
    return CoordinatorPlan(workflow_id=plan.workflow_id, trace_id=plan.trace_id, revision=plan.revision + 1,
                           mode=plan.mode, tasks=tuple(tasks), unresolved_conflicts=directive.task_ids if directive.action == "schedule_reconciliation" else ())


def summarize_result(plan: CoordinatorPlan, reports: Sequence[AgentReport], *, decision: DecisionDraft | None = None,
                     risk: RiskAssessment | None = None, informational_answer: InformationalAnswer | None = None,
                     route_history: tuple[str, ...] = ()) -> WorkflowResult:
    plan = CoordinatorPlan.model_validate(plan)
    known = {task.task_id for task in plan.tasks}
    seen, evidence = {}, set()
    safe = True
    for raw in reports:
        report = AgentReport.model_validate(raw)
        if report.task_id not in known or (report.workflow_id, report.trace_id) != (plan.workflow_id, plan.trace_id):
            raise ValueError("summary report identity mismatch")
        if report.task_id in seen and seen[report.task_id] != report:
            safe = False
        seen[report.task_id] = report
        safe = safe and report.status is ReportStatus.COMPLETED
        evidence.update(report.evidence_refs)
    base = dict(workflow_id=plan.workflow_id, trace_id=plan.trace_id,
                evidence_references=tuple(sorted(evidence))[:50], route_history=route_history[:50])
    safe = safe and set(seen) == known and bool(evidence)
    if safe and informational_answer is not None:
        answer = InformationalAnswer.model_validate(informational_answer)
        if answer.knowledge_status is KnowledgeStatus.KNOWN and set(answer.source_references) <= evidence:
            return WorkflowResult(**base, terminal_mode=TerminalMode.INFORMATIONAL, final_action=Action.NO_TRADE,
                                  informational_answer=answer, knowledge_status=KnowledgeStatus.KNOWN, uncertainty_reason=None)
    if safe and decision is not None:
        decision = DecisionDraft.model_validate(decision)
        risk = RiskAssessment.model_validate(risk) if risk is not None else None
        if decision.knowledge_status is KnowledgeStatus.KNOWN and not decision.conflict_codes:
            if decision.action is Action.NO_TRADE:
                return WorkflowResult(**base, terminal_mode=TerminalMode.NO_TRADE, final_action=Action.NO_TRADE,
                                      knowledge_status=KnowledgeStatus.KNOWN, uncertainty_reason=None)
            if risk is not None and risk.accepted and not risk.requires_escalation and risk.knowledge_status is KnowledgeStatus.KNOWN:
                return WorkflowResult(**base, terminal_mode=TerminalMode.PLAYBOOK, final_action=decision.action,
                                      playbook_payload=decision, knowledge_status=KnowledgeStatus.KNOWN, uncertainty_reason=None)
    return WorkflowResult(**base, terminal_mode=TerminalMode.UNKNOWN, final_action=Action.NO_TRADE,
                          knowledge_status=KnowledgeStatus.INSUFFICIENT,
                          uncertainty_reason="不知道：required evidence, conflict resolution, or risk approval is missing.")
