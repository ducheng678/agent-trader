from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from market_agent.workflow_agent_driver import AgentDriver
from market_agent.workflow_context_summary import ContextHandoff
from market_agent.workflow_coordinator_agent import (
    bind_contexts,
    CoordinatorDirective,
    dispatch_tasks,
    plan_request,
    reconcile_reports,
    reschedule,
)
from market_agent.workflow_contracts import (
    AgentReport,
    ContextSummary,
    CoordinatorPlan,
    DecisionDraft,
    TechnicalAnalysis,
    WorkflowBudgetState,
    WorkflowRequest,
    WorkflowResult,
)
from market_agent.workflow_graph import DecisionBuilder, DecisionVerifier, TechnicalSelector, WorkflowServices
from market_agent.workflow_memory_retrieval import CoreExperienceSummary
from market_agent.workflow_risk_gate import RiskPolicy


ContextProvider = Callable[[CoordinatorPlan], Mapping[str, ContextSummary | ContextHandoff]]
RecoveryContextProvider = Callable[
    [CoordinatorPlan, tuple[AgentReport, ...], CoordinatorDirective],
    Mapping[str, ContextSummary | ContextHandoff],
]
GrantProvider = Callable[[CoordinatorPlan], Mapping[str, object]]
HostAuthorizer = Callable[..., object]
ResultFinalizer = Callable[[WorkflowResult], None]


@dataclass(frozen=True, slots=True)
class CoordinatorRuntime:
    budget: WorkflowBudgetState
    contexts: ContextProvider
    grants: GrantProvider
    driver: AgentDriver | Callable
    deadline_epoch: float
    authorize: HostAuthorizer
    decide: DecisionBuilder
    technical: TechnicalSelector
    verify: DecisionVerifier
    recovery_contexts: RecoveryContextProvider | None = None
    finalize: ResultFinalizer | None = None
    memory_context: CoreExperienceSummary | None = None
    memory_tenant_id: str | None = None
    memory_scope: str | None = None
    risk_policy: RiskPolicy = RiskPolicy()
    cancellation_check: Callable[[], bool] = lambda: False

    def services_for(self, request: WorkflowRequest) -> WorkflowServices:
        request = WorkflowRequest.model_validate(request)
        budget = WorkflowBudgetState.model_validate(self.budget)
        if budget.remaining_attempts < 1 or budget.remaining_cost <= 0.0:
            raise ValueError("workflow budget is exhausted before planning")

        def plan(value: WorkflowRequest) -> CoordinatorPlan:
            if value.workflow_id != request.workflow_id or value.trace_id != request.trace_id:
                raise PermissionError("runtime cannot plan another workflow trace")
            return plan_request(value, budget)

        def dispatch(value: CoordinatorPlan) -> tuple[AgentReport, ...]:
            contexts = self.contexts(value)
            grants = self.grants(value)
            bound = bind_contexts(value, contexts)
            return tuple(dispatch_tasks(
                bound,
                contexts,
                self.driver,
                grants,
                deadline_epoch=self.deadline_epoch,
                authorize=self.authorize,
                memory_context=self.memory_context,
                memory_tenant_id=self.memory_tenant_id,
                memory_scope=self.memory_scope,
                cancellation_check=self.cancellation_check,
            ))

        def remaining_after(state: WorkflowBudgetState, plan_value: CoordinatorPlan) -> WorkflowBudgetState:
            # Task reservations and their maximum call counts are charged before
            # a recovery plan is admitted.  Treating unreported failed calls as
            # free would let recovery exceed the global cost/attempt ceiling.
            reserved = sum(task.reserved_cost for task in plan_value.tasks)
            attempts = sum(task.maximum_retries + 1 for task in plan_value.tasks)
            return state.model_copy(update={
                "remaining_cost": max(0.0, state.remaining_cost - reserved),
                "reserved_cost": state.reserved_cost + reserved,
                "remaining_attempts": max(0, state.remaining_attempts - attempts),
            })

        def recover(value: CoordinatorPlan, reports: tuple[AgentReport, ...]) -> tuple[CoordinatorPlan, tuple[AgentReport, ...]] | None:
            current_plan, current_reports = value, reports
            available = remaining_after(budget, current_plan)
            while True:
                directive = reconcile_reports(current_plan, current_reports, available)
                if directive.action == "continue":
                    return current_plan, current_reports
                if directive.action not in ("reschedule", "schedule_reconciliation"):
                    return None

                recovery_plan = reschedule(current_plan, directive, available)
                context_provider = self.recovery_contexts
                contexts = (context_provider(recovery_plan, current_reports, directive)
                            if context_provider is not None else self.contexts(recovery_plan))
                grants = self.grants(recovery_plan)
                bound = bind_contexts(recovery_plan, contexts)
                recovery_reports = tuple(dispatch_tasks(
                    bound,
                    contexts,
                    self.driver,
                    grants,
                    deadline_epoch=self.deadline_epoch,
                    authorize=self.authorize,
                    memory_context=self.memory_context,
                    memory_tenant_id=self.memory_tenant_id,
                    memory_scope=self.memory_scope,
                    cancellation_check=self.cancellation_check,
                ))

                replaced = set(directive.task_ids)
                surviving_tasks = tuple(task for task in current_plan.tasks if task.task_id not in replaced)
                surviving_reports = tuple(report for report in current_reports if report.task_id not in replaced)
                current_plan = CoordinatorPlan(
                    workflow_id=bound.workflow_id,
                    trace_id=bound.trace_id,
                    revision=bound.revision,
                    mode=bound.mode,
                    tasks=surviving_tasks + bound.tasks,
                    unresolved_conflicts=bound.unresolved_conflicts,
                )
                current_reports = surviving_reports + recovery_reports
                available = remaining_after(available, bound)

        return WorkflowServices(
            plan=plan,
            dispatch=dispatch,
            recover=recover,
            decide=self.decide,
            technical=self.technical,
            verify=self.verify,
            finalize=self.finalize,
            cancelled=self.cancellation_check,
            risk_policy=self.risk_policy,
        )
