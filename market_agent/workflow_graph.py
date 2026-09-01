from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from market_agent.workflow_contracts import (
    AgentReport,
    CoordinatorPlan,
    DecisionDraft,
    RiskAssessment,
    TechnicalAnalysis,
    WorkflowRequest,
    WorkflowResult,
)
from market_agent.workflow_playbook_assembler import assemble_playbook, unknown_playbook
from market_agent.workflow_risk_gate import RiskPolicy, evaluate_risk


PlanBuilder = Callable[[WorkflowRequest], CoordinatorPlan]
TaskDispatcher = Callable[[CoordinatorPlan], tuple[AgentReport, ...]]
DecisionBuilder = Callable[[WorkflowRequest, CoordinatorPlan, tuple[AgentReport, ...]], DecisionDraft | None]
TechnicalSelector = Callable[[tuple[AgentReport, ...]], TechnicalAnalysis | None]
AuditFinalizer = Callable[[WorkflowResult], None]


@dataclass(frozen=True, slots=True)
class WorkflowServices:
    plan: PlanBuilder
    dispatch: TaskDispatcher
    decide: DecisionBuilder
    technical: TechnicalSelector
    finalize: AuditFinalizer | None = None
    risk_policy: RiskPolicy = RiskPolicy()


class GraphState(TypedDict, total=False):
    request: WorkflowRequest
    services: WorkflowServices
    plan: CoordinatorPlan
    reports: tuple[AgentReport, ...]
    decision: DecisionDraft | None
    technical: TechnicalAnalysis | None
    risk: RiskAssessment | None
    result: WorkflowResult
    failure_reason: str


def _safe_failure(state: GraphState, reason: str) -> dict[str, object]:
    request = state["request"]
    return {
        "failure_reason": reason,
        "result": unknown_playbook(
            workflow_id=request.workflow_id,
            trace_id=request.trace_id,
            reason=reason,
            route_history=("coordinated_graph", "safe_unknown"),
        ),
    }


def _plan(state: GraphState) -> dict[str, object]:
    try:
        plan = state["services"].plan(state["request"])
        request = state["request"]
        if plan.workflow_id != request.workflow_id or plan.trace_id != request.trace_id:
            return _safe_failure(state, "coordinator returned a cross-trace plan")
        return {"plan": plan}
    except Exception:
        return _safe_failure(state, "coordinator planning is unavailable")


def _route_after_plan(state: GraphState) -> str:
    return "finalize" if "result" in state else "dispatch"


def _dispatch(state: GraphState) -> dict[str, object]:
    try:
        reports = state["services"].dispatch(state["plan"])
        if not isinstance(reports, tuple):
            raise TypeError("dispatcher must return a tuple")
        request = state["request"]
        if any(report.workflow_id != request.workflow_id or report.trace_id != request.trace_id for report in reports):
            return _safe_failure(state, "dispatcher returned a cross-trace report")
        return {"reports": reports}
    except Exception:
        return _safe_failure(state, "task dispatch is unavailable")


def _route_after_dispatch(state: GraphState) -> str:
    return "finalize" if "result" in state else "decide"


def _decide(state: GraphState) -> dict[str, object]:
    try:
        decision = state["services"].decide(state["request"], state["plan"], state["reports"])
        technical = state["services"].technical(state["reports"])
        return {"decision": decision, "technical": technical}
    except Exception:
        return _safe_failure(state, "decision assembly is unavailable")


def _route_after_decide(state: GraphState) -> str:
    return "finalize" if "result" in state else "risk"


def _risk(state: GraphState) -> dict[str, object]:
    decision = state.get("decision")
    if decision is None:
        return _safe_failure(state, "decision draft is unavailable")
    return {"risk": evaluate_risk(decision, technical=state.get("technical"), policy=state["services"].risk_policy)}


def _assemble(state: GraphState) -> dict[str, object]:
    request = state["request"]
    references = tuple(sorted({reference for report in state.get("reports", ()) for reference in report.evidence_refs}))
    return {
        "result": assemble_playbook(
            workflow_id=request.workflow_id,
            trace_id=request.trace_id,
            decision=state.get("decision"),
            risk=state.get("risk"),
            evidence_references=references,
            route_history=("coordinated_graph", "plan", "dispatch", "risk", "assemble"),
        )
    }


def _finalize(state: GraphState) -> dict[str, object]:
    result = state["result"]
    finalizer = state["services"].finalize
    if finalizer is not None:
        try:
            finalizer(result)
        except Exception:
            return _safe_failure(state, "audit finalization is unavailable")
    return {"result": result}


def build_workflow_graph():
    graph = StateGraph(GraphState)
    graph.add_node("plan", _plan)
    graph.add_node("dispatch", _dispatch)
    graph.add_node("decide", _decide)
    graph.add_node("risk", _risk)
    graph.add_node("assemble", _assemble)
    graph.add_node("finalize", _finalize)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", _route_after_plan, {"dispatch": "dispatch", "finalize": "finalize"})
    graph.add_conditional_edges("dispatch", _route_after_dispatch, {"decide": "decide", "finalize": "finalize"})
    graph.add_conditional_edges("decide", _route_after_decide, {"risk": "risk", "finalize": "finalize"})
    graph.add_edge("risk", "assemble")
    graph.add_edge("assemble", "finalize")
    graph.add_edge("finalize", END)
    return graph.compile()


class CoordinatedWorkflow:
    def __init__(self) -> None:
        self._graph = build_workflow_graph()

    def invoke(self, request: WorkflowRequest, services: WorkflowServices) -> WorkflowResult:
        if type(request) is not WorkflowRequest or type(services) is not WorkflowServices:
            raise TypeError("coordinated workflow requires strict request and services")
        state = self._graph.invoke({"request": request, "services": services})
        result = state.get("result")
        if type(result) is not WorkflowResult:
            return unknown_playbook(
                workflow_id=request.workflow_id,
                trace_id=request.trace_id,
                reason="graph did not produce a valid result",
                route_history=("coordinated_graph", "safe_unknown"),
            )
        return result
