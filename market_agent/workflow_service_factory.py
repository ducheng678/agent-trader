from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from market_agent.workflow_agent_driver import AgentDriver
from market_agent.workflow_context_summary import ContextHandoff
from market_agent.workflow_coordinator_agent import (
    bind_contexts,
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
from market_agent.workflow_graph import DecisionBuilder, TechnicalSelector, WorkflowServices
from market_agent.workflow_risk_gate import RiskPolicy


ContextProvider = Callable[[CoordinatorPlan], Mapping[str, ContextSummary | ContextHandoff]]
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
    finalize: ResultFinalizer | None = None
    risk_policy: RiskPolicy = RiskPolicy()

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
            ))

        def remaining_after(state: WorkflowBudgetState, plan_value: CoordinatorPlan) -> WorkflowBudgetState:
            # Task reservations and their maximum call counts are charged before
            # a recovery plan is admitted.  Treating unreported failed calls as
            # free would let recovery exceed the global cost/attempt ceiling.
            reserved = sum(task.reserved_cost for task in plan_value.tasks)
            attempts = sum(task.maximum_retries + 1 for task in plan_value.tasks)
            return budget.model_copy(update={
                "remaining_cost": max(0.0, state.remaining_cost - reserved),
                "reserved_cost": state.reserved_cost + reserved,
                "remaining_attempts": max(0, state.remaining_attempts - attempts),
            })

        def recover(value: CoordinatorPlan, reports: tuple[AgentReport, ...]) -> tuple[CoordinatorPlan, tuple[AgentReport, ...]] | None:
            available = remaining_after(budget, value)
            directive = reconcile_reports(value, reports, available)
            if directive.action == "continue":
                return value, reports
            if directive.action not in ("reschedule", "schedule_reconciliation"):
                return None
            next_plan = reschedule(value, directive, available)
            contexts = self.contexts(next_plan)
            grants = self.grants(next_plan)
            bound = bind_contexts(next_plan, contexts)
            next_reports = tuple(dispatch_tasks(
                bound,
                contexts,
                self.driver,
                grants,
                deadline_epoch=self.deadline_epoch,
                authorize=self.authorize,
            ))
            # A recovered plan replaces failed/conflicting work.  Its task set
            # and reports remain paired so final decision assembly cannot mix
            # identities from different coordinator revisions.
            final_directive = reconcile_reports(bound, next_reports, remaining_after(available, next_plan))
            return (bound, next_reports) if final_directive.action == "continue" else None

        return WorkflowServices(
            plan=plan,
            dispatch=dispatch,
            recover=recover,
            decide=self.decide,
            technical=self.technical,
            finalize=self.finalize,
            risk_policy=self.risk_policy,
        )
