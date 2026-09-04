from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
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
from market_agent.workflow_observation import (
    CheckpointDecision,
    CoreNodeName,
    ExecutionObservationCollector,
    NodeOutcome,
    ObservedWorkItem,
    TaskRetryState,
)
from market_agent.workflow_risk_gate import RiskPolicy, evaluate_risk


PlanBuilder = Callable[[WorkflowRequest], CoordinatorPlan]
TaskDispatcher = Callable[[CoordinatorPlan], tuple[AgentReport, ...]]
RecoveryDispatcher = Callable[[CoordinatorPlan, tuple[AgentReport, ...]], tuple[CoordinatorPlan, tuple[AgentReport, ...]] | None]
DecisionBuilder = Callable[[WorkflowRequest, CoordinatorPlan, tuple[AgentReport, ...]], DecisionDraft | None]
DecisionVerifier = Callable[[WorkflowRequest, CoordinatorPlan, tuple[AgentReport, ...], DecisionDraft], DecisionDraft | None]
TechnicalSelector = Callable[[tuple[AgentReport, ...]], TechnicalAnalysis | None]
AuditFinalizer = Callable[[WorkflowResult], None]
CancellationCheck = Callable[[], bool]


@dataclass(frozen=True, slots=True)
class WorkflowServices:
    plan: PlanBuilder
    dispatch: TaskDispatcher
    decide: DecisionBuilder
    technical: TechnicalSelector
    verify: DecisionVerifier | None = None
    recover: RecoveryDispatcher | None = None
    finalize: AuditFinalizer | None = None
    cancelled: CancellationCheck = lambda: False
    risk_policy: RiskPolicy = RiskPolicy()
    execution_observer: ExecutionObservationCollector | None = None


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


def _cancelled(state: GraphState) -> bool:
    try:
        return bool(state["services"].cancelled())
    except Exception:
        return True


def _plan(state: GraphState) -> dict[str, object]:
    if _cancelled(state):
        return _safe_failure(state, "workflow cancelled before planning")
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
    if _cancelled(state):
        return _safe_failure(state, "workflow cancelled before dispatch")
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
    return "finalize" if "result" in state else "recover"


def _recover(state: GraphState) -> dict[str, object]:
    if _cancelled(state):
        return _safe_failure(state, "workflow cancelled before recovery")
    recover = state["services"].recover
    if recover is None:
        return {}
    try:
        outcome = recover(state["plan"], state["reports"])
        if outcome is None:
            return _safe_failure(state, "coordinator recovery could not produce a safe report set")
        plan, reports = outcome
        request = state["request"]
        if (plan.workflow_id, plan.trace_id) != (request.workflow_id, request.trace_id):
            return _safe_failure(state, "coordinator recovery returned a cross-trace plan")
        if not isinstance(reports, tuple) or any(
            (report.workflow_id, report.trace_id) != (request.workflow_id, request.trace_id) for report in reports
        ):
            return _safe_failure(state, "coordinator recovery returned cross-trace reports")
        return {"plan": plan, "reports": reports}
    except Exception:
        return _safe_failure(state, "coordinator recovery is unavailable")


def _route_after_recover(state: GraphState) -> str:
    return "finalize" if "result" in state else "decide"


def _decide(state: GraphState) -> dict[str, object]:
    if _cancelled(state):
        return _safe_failure(state, "workflow cancelled before decision")
    try:
        decision = state["services"].decide(state["request"], state["plan"], state["reports"])
        technical = state["services"].technical(state["reports"])
        return {"decision": decision, "technical": technical}
    except Exception:
        return _safe_failure(state, "decision assembly is unavailable")


def _route_after_decide(state: GraphState) -> str:
    return "finalize" if "result" in state else "reflect"


def _reflect(state: GraphState) -> dict[str, object]:
    if _cancelled(state):
        return _safe_failure(state, "workflow cancelled before reflection")
    decision = state.get("decision")
    verify = state["services"].verify
    if decision is None or verify is None:
        return _safe_failure(state, "core decision verification is unavailable")
    try:
        accepted = verify(state["request"], state["plan"], state["reports"], decision)
        if accepted is None:
            return _safe_failure(state, "core decision failed objective verification")
        accepted = DecisionDraft.model_validate(accepted)
        return {"decision": accepted}
    except Exception:
        return _safe_failure(state, "core decision verification is unavailable")


def _route_after_reflect(state: GraphState) -> str:
    return "finalize" if "result" in state else "risk"


def _risk(state: GraphState) -> dict[str, object]:
    if _cancelled(state):
        return _safe_failure(state, "workflow cancelled before risk evaluation")
    decision = state.get("decision")
    if decision is None:
        return _safe_failure(state, "decision draft is unavailable")
    return {"risk": evaluate_risk(decision, technical=state.get("technical"), policy=state["services"].risk_policy)}


def _route_after_risk(state: GraphState) -> str:
    return "finalize" if "result" in state else "assemble"


def _assemble(state: GraphState) -> dict[str, object]:
    if _cancelled(state):
        return _safe_failure(state, "workflow cancelled before assembly")
    request = state["request"]
    references = tuple(sorted({reference for report in state.get("reports", ()) for reference in report.evidence_refs}))
    return {
        "result": assemble_playbook(
            workflow_id=request.workflow_id,
            trace_id=request.trace_id,
            decision=state.get("decision"),
            risk=state.get("risk"),
            evidence_references=references,
            route_history=("coordinated_graph", "plan", "dispatch", "recover", "risk", "assemble"),
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


def _checkpoint_value(value: object) -> object:
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    if isinstance(value, tuple):
        return [_checkpoint_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _checkpoint_value(item) for key, item in value.items()}
    return value


def _inferred_core_work_item(
    task_id: str,
    node: CoreNodeName,
    *,
    decision_task_ids: tuple[str, ...],
    attempt_ids: tuple[str, ...],
    execution_state: str,
) -> ObservedWorkItem | None:
    """Bind model-only core nodes into the durable task inventory.

    Specialist work comes from the coordinator plan.  Decision and reflection
    invocations are created after that plan, so their host-observed attempts
    must add a fixed, node-owned inventory entry at their first checkpoint.
    No arbitrary provider task is admitted: only the two known core nodes can
    create such an entry.
    """

    if node is CoreNodeName.DECIDE:
        return ObservedWorkItem(
            task_id=task_id,
            task_kind="decision_planner",
            worker_id="decision_planner-agent",
            owner_node=CoreNodeName.DECIDE,
            maximum_retries=1,
            execution_state=execution_state,
            attempt_ids=attempt_ids,
        )
    if node is CoreNodeName.REFLECT:
        return ObservedWorkItem(
            task_id=task_id,
            task_kind="reflection",
            worker_id="reflection-agent",
            owner_node=CoreNodeName.REFLECT,
            dependency_ids=decision_task_ids[:1],
            maximum_retries=0,
            execution_state=execution_state,
            attempt_ids=attempt_ids,
        )
    return None


def _observed_node(
    state: GraphState,
    node: CoreNodeName,
    action: Callable[[GraphState], dict[str, object]],
) -> dict[str, object]:
    update = action(state)
    observer = state["services"].execution_observer
    if observer is None:
        return update
    combined = {**state, **update}
    plan = combined.get("plan")
    tasks = tuple(plan.tasks) if isinstance(plan, CoordinatorPlan) else ()
    # A recovery plan carries both survivors and newly scheduled work.  The
    # previous durable checkpoint, not the current revision number, is the
    # authority for a survivor's original owner.
    prior_checkpoints = observer.checkpoints()
    prior_checkpoint = prior_checkpoints[-1] if prior_checkpoints else None
    prior_task_ids = prior_checkpoint.task_ids if prior_checkpoint is not None else ()
    prior_work_items = {
        item.task_id: item
        for item in (prior_checkpoint.work_items if prior_checkpoint is not None else ())
    }
    prior_retry_state = {
        item.task_id: item
        for item in (prior_checkpoint.retry_state if prior_checkpoint is not None else ())
    }
    planned_task_ids = tuple(task.task_id for task in tasks)
    attempts = observer.usage().attempts
    attempts_by_task: dict[str, int] = {}
    attempt_ids_by_task: dict[str, tuple[str, ...]] = {}
    first_node_by_task: dict[str, CoreNodeName] = {}
    for attempt in attempts:
        attempts_by_task[attempt.task_id] = attempts_by_task.get(attempt.task_id, 0) + 1
        attempt_ids_by_task[attempt.task_id] = (
            *attempt_ids_by_task.get(attempt.task_id, ()), attempt.provider_request_id
        )
        first_node_by_task.setdefault(attempt.task_id, attempt.node)
    decision_task_ids = tuple(
        task_id for task_id, owner in first_node_by_task.items()
        if owner is CoreNodeName.DECIDE
    )
    durable_specialist_ids = set(prior_task_ids) | set(planned_task_ids)
    inferred_task_ids = tuple(
        task_id for task_id in first_node_by_task
        if task_id not in durable_specialist_ids
    )
    appended_task_ids = tuple(
        task_id for task_id in (*planned_task_ids, *inferred_task_ids)
        if task_id not in prior_task_ids
    )
    task_ids = (*prior_task_ids, *appended_task_ids)
    reports = tuple(combined.get("reports", ()))
    reports_by_id = {report.task_id: report for report in reports}
    completed_set = set(
        prior_checkpoint.completed_task_ids if prior_checkpoint is not None else ()
    )
    completed_set.update(
        task_id
        for task_id in planned_task_ids
        if task_id in reports_by_id and reports_by_id[task_id].status.value == "completed"
    )
    failed_set = set(
        prior_checkpoint.failed_task_ids if prior_checkpoint is not None else ()
    )
    failed_set.update(
        task_id
        for task_id in planned_task_ids
        if task_id in reports_by_id and reports_by_id[task_id].status.value == "failed"
    )
    failure_reason = update.get("failure_reason")
    for task_id in inferred_task_ids:
        owner = first_node_by_task[task_id]
        if owner is CoreNodeName.DECIDE and node is not CoreNodeName.PLAN:
            if combined.get("decision") is not None and failure_reason is None:
                completed_set.add(task_id)
            elif node is CoreNodeName.DECIDE:
                failed_set.add(task_id)
        elif owner is CoreNodeName.REFLECT and node in {
            CoreNodeName.REFLECT, CoreNodeName.RISK, CoreNodeName.ASSEMBLE,
        }:
            if failure_reason is None:
                completed_set.add(task_id)
            elif node is CoreNodeName.REFLECT:
                failed_set.add(task_id)
        elif failure_reason is not None and owner is node:
            failed_set.add(task_id)
    completed = tuple(task_id for task_id in task_ids if task_id in completed_set)
    failed = tuple(task_id for task_id in task_ids if task_id in failed_set)
    retry_limits = {
        task.task_id: task.maximum_retries for task in tasks
    }
    retry_limits.update({
        task_id: 1 if first_node_by_task[task_id] is CoreNodeName.DECIDE else 0
        for task_id in inferred_task_ids
    })
    retry_limits.update({
        task_id: item.maximum_retries
        for task_id, item in prior_work_items.items()
        if task_id not in retry_limits
    })
    retry_state = tuple(
        (
            prior_retry_state[task_id]
            if task_id not in planned_task_ids and task_id not in inferred_task_ids
            else TaskRetryState(
                task_id=task_id,
                attempts_consumed=attempts_by_task.get(task_id, 0),
                retries_consumed=max(0, attempts_by_task.get(task_id, 0) - 1),
                retries_remaining=max(
                    0,
                    retry_limits[task_id] - max(0, attempts_by_task.get(task_id, 0) - 1),
                ),
            )
        )
        for task_id in task_ids
    )
    planned_work_items = tuple(
        ObservedWorkItem(
            task_id=task.task_id,
            task_kind=task.task_type.value,
            worker_id=f"{task.task_type.value}-agent",
            owner_node=(
                prior_work_items[task.task_id].owner_node
                if task.task_id in prior_work_items
                else CoreNodeName.RECOVER if plan.revision > 0 else CoreNodeName.DISPATCH
            ),
            dependency_ids=((task.parent_task_id,) if task.parent_task_id is not None else ()),
            maximum_retries=task.maximum_retries,
            execution_state=(
                "succeeded" if task.task_id in completed else
                "failed" if task.task_id in failed else
                "running" if attempts_by_task.get(task.task_id, 0) else "pending"
            ),
            attempt_ids=attempt_ids_by_task.get(task.task_id, ()),
        )
        for task in tasks
    )
    inferred_work_items = tuple(
        item
        for task_id in inferred_task_ids
        if (item := _inferred_core_work_item(
            task_id,
            first_node_by_task[task_id],
            decision_task_ids=decision_task_ids,
            attempt_ids=attempt_ids_by_task[task_id],
            execution_state=(
                "succeeded" if task_id in completed else
                "failed" if task_id in failed else "running"
            ),
        )) is not None
    )
    if len(inferred_work_items) != len(inferred_task_ids):
        return _safe_failure(state, "unrecognized model execution node")
    current_work_items = {
        item.task_id: item for item in (*planned_work_items, *inferred_work_items)
    }
    work_items = tuple(
        current_work_items[task_id]
        if task_id in current_work_items
        else prior_work_items[task_id]
        for task_id in task_ids
    )
    outcome = (
        NodeOutcome.CANCELLED
        if isinstance(failure_reason, str) and "cancelled" in failure_reason
        else NodeOutcome.FAILED
        if failure_reason is not None
        else NodeOutcome.COMPLETED
    )
    material = {
        "workflow_id": state["request"].workflow_id,
        "trace_id": state["request"].trace_id,
        "node": node.value,
        "plan_revision": plan.revision if isinstance(plan, CoordinatorPlan) else 0,
        "task_ids": task_ids,
        "failure_reason": failure_reason,
        "reports": reports,
        "decision": combined.get("decision"),
        "risk": combined.get("risk"),
        "result": combined.get("result"),
    }
    fingerprint = sha256(
        json.dumps(
            _checkpoint_value(material),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()
    try:
        permit = observer.checkpoint(
            plan_revision=plan.revision if isinstance(plan, CoordinatorPlan) else 0,
            node=node,
            outcome=outcome,
            task_ids=task_ids,
            completed_task_ids=completed,
            failed_task_ids=failed,
            retry_state=retry_state,
            work_items=work_items,
            action_fingerprint=fingerprint,
        )
    except Exception:
        return _safe_failure(state, "execution checkpoint authority is unavailable")
    if permit.decision is not CheckpointDecision.CONTINUE:
        return _safe_failure(
            combined,
            f"Harness checkpoint authority required {permit.decision.value}: {permit.reason_code}",
        )
    return update


def build_workflow_graph():
    graph = StateGraph(GraphState)
    graph.add_node("plan", lambda state: _observed_node(state, CoreNodeName.PLAN, _plan))
    graph.add_node("dispatch", lambda state: _observed_node(state, CoreNodeName.DISPATCH, _dispatch))
    graph.add_node("recover", lambda state: _observed_node(state, CoreNodeName.RECOVER, _recover))
    graph.add_node("decide", lambda state: _observed_node(state, CoreNodeName.DECIDE, _decide))
    graph.add_node("reflect", lambda state: _observed_node(state, CoreNodeName.REFLECT, _reflect))
    graph.add_node("risk", lambda state: _observed_node(state, CoreNodeName.RISK, _risk))
    graph.add_node("assemble", lambda state: _observed_node(state, CoreNodeName.ASSEMBLE, _assemble))
    graph.add_node("finalize", _finalize)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges("plan", _route_after_plan, {"dispatch": "dispatch", "finalize": "finalize"})
    graph.add_conditional_edges("dispatch", _route_after_dispatch, {"recover": "recover", "finalize": "finalize"})
    graph.add_conditional_edges("recover", _route_after_recover, {"decide": "decide", "finalize": "finalize"})
    graph.add_conditional_edges("decide", _route_after_decide, {"reflect": "reflect", "finalize": "finalize"})
    graph.add_conditional_edges("reflect", _route_after_reflect, {"risk": "risk", "finalize": "finalize"})
    graph.add_conditional_edges("risk", _route_after_risk, {"assemble": "assemble", "finalize": "finalize"})
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
