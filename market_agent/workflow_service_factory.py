from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from market_agent.workflow_agent_driver import AgentDriver
from market_agent.workflow_context_summary import ContextHandoff
from market_agent.workflow_coordinator_agent import bind_contexts, dispatch_tasks, plan_request
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

        return WorkflowServices(
            plan=plan,
            dispatch=dispatch,
            decide=self.decide,
            technical=self.technical,
            finalize=self.finalize,
            risk_policy=self.risk_policy,
        )
