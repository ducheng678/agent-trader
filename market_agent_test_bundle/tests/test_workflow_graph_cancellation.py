from market_agent.workflow_contracts import (
    Action,
    AgentReport,
    AgentTask,
    CoordinatorPlan,
    DecisionDraft,
    KnowledgeStatus,
    ModelTier,
    ReportStatus,
    TaskDifficulty,
    TaskType,
    TerminalMode,
    WorkflowMode,
    WorkflowRequest,
)
from market_agent.workflow_observation import (
    CheckpointDecision,
    CheckpointPermit,
    CoreNodeName,
    ExecutionObservationCollector,
)
from market_agent.workflow_graph import CoordinatedWorkflow, WorkflowServices


def test_graph_stops_before_planning_when_cancelled() -> None:
    calls: list[str] = []

    def unexpected(*_args):
        calls.append("called")
        raise AssertionError("cancelled graph called a workflow service")

    services = WorkflowServices(
        plan=unexpected,
        dispatch=unexpected,
        decide=unexpected,
        technical=unexpected,
        cancelled=lambda: True,
    )
    request = WorkflowRequest(
        workflow_id="workflow-1",
        trace_id="1" * 32,
        user_query="分析市场",
        trigger_reason="manual_once",
    )

    result = CoordinatedWorkflow().invoke(request, services)

    assert result.terminal_mode is TerminalMode.UNKNOWN
    assert result.knowledge_status is KnowledgeStatus.INSUFFICIENT
    assert calls == []


def test_graph_emits_one_typed_cumulative_checkpoint_for_each_executed_core_node() -> None:
    request = WorkflowRequest(
        workflow_id="workflow-1",
        trace_id="1" * 32,
        user_query="分析市场",
        trigger_reason="manual_once",
    )
    task = AgentTask(
        task_id="technical-1",
        workflow_id=request.workflow_id,
        trace_id=request.trace_id,
        task_type=TaskType.TECHNICAL,
        objective="Analyze technical evidence.",
        context_summary_id="summary-1",
        allowed_data=("context_summary",),
        allowed_tools=(),
        expected_output="technical-v1",
        acceptance_criteria=("Return a bounded report.",),
        difficulty=TaskDifficulty.NORMAL,
        model_tier=ModelTier.TERRA,
        prompt_version="technical-v1",
        attempt_timeout_seconds=30,
        maximum_retries=1,
        reserved_cost=0.05,
        remaining_workflow_cost=0.10,
        analysis_steps=("Inspect evidence.", "Assess setup.", "Return conclusion."),
        escalation_rule="return_to_coordinator",
        conflict_return_rule="return_typed_conflict",
    )
    plan = CoordinatorPlan(
        workflow_id=request.workflow_id,
        trace_id=request.trace_id,
        revision=2,
        mode=WorkflowMode.ACTIVE,
        tasks=(task,),
    )
    report = AgentReport(
        task_id=task.task_id,
        workflow_id=request.workflow_id,
        trace_id=request.trace_id,
        status=ReportStatus.COMPLETED,
        knowledge_status=KnowledgeStatus.KNOWN,
        uncertainty_reason=None,
        summary="Supported bounded evidence.",
    )
    decision = DecisionDraft(
        knowledge_status=KnowledgeStatus.KNOWN,
        uncertainty_reason=None,
        action=Action.NO_TRADE,
        execute_now=False,
        decision_confidence=0.8,
    )
    observations = ExecutionObservationCollector(request.workflow_id, request.trace_id)
    services = WorkflowServices(
        plan=lambda _request: plan,
        dispatch=lambda _plan: (report,),
        recover=lambda current, reports: (current, reports),
        decide=lambda *_args: decision,
        technical=lambda _reports: None,
        verify=lambda *_args: decision,
        execution_observer=observations,
    )

    result = CoordinatedWorkflow().invoke(request, services)

    assert result.terminal_mode is TerminalMode.NO_TRADE
    checkpoints = observations.checkpoints()
    assert tuple(checkpoint.node for checkpoint in checkpoints) == (
        CoreNodeName.PLAN,
        CoreNodeName.DISPATCH,
        CoreNodeName.RECOVER,
        CoreNodeName.DECIDE,
        CoreNodeName.REFLECT,
        CoreNodeName.RISK,
        CoreNodeName.ASSEMBLE,
    )
    assert tuple(checkpoint.ordinal for checkpoint in checkpoints) == tuple(range(1, 8))
    assert all(checkpoint.plan_revision == 2 for checkpoint in checkpoints)
    assert all(checkpoint.task_ids == (task.task_id,) for checkpoint in checkpoints)


def test_graph_obeys_harness_degrade_permit_before_next_node() -> None:
    request = WorkflowRequest(
        workflow_id="workflow-1",
        trace_id="1" * 32,
        user_query="分析市场",
        trigger_reason="manual_once",
    )
    calls: list[str] = []
    task = AgentTask(
        task_id="technical-1", workflow_id=request.workflow_id,
        trace_id=request.trace_id, task_type=TaskType.TECHNICAL,
        objective="Analyze technical evidence.", context_summary_id="summary-1",
        allowed_data=("context_summary",), allowed_tools=(),
        expected_output="technical-v1", acceptance_criteria=("Return report.",),
        difficulty=TaskDifficulty.NORMAL, model_tier=ModelTier.TERRA,
        prompt_version="technical-v1", attempt_timeout_seconds=30,
        maximum_retries=1, reserved_cost=0.05, remaining_workflow_cost=0.10,
        analysis_steps=("Inspect.", "Assess.", "Return."),
        escalation_rule="return_to_coordinator",
        conflict_return_rule="return_typed_conflict",
    )
    plan = CoordinatorPlan(
        workflow_id=request.workflow_id,
        trace_id=request.trace_id,
        revision=0,
        mode=WorkflowMode.ACTIVE,
        tasks=(task,),
    )

    def permit(checkpoint):
        return CheckpointPermit(
            workflow_id=checkpoint.workflow_id,
            trace_id=checkpoint.trace_id,
            checkpoint_ordinal=checkpoint.ordinal,
            checkpoint_digest=checkpoint.canonical_digest(),
            decision=CheckpointDecision.DEGRADE,
            reason_code="budget_exhausted",
        )

    observations = ExecutionObservationCollector(
        request.workflow_id, request.trace_id, checkpoint_sink=permit
    )
    result = CoordinatedWorkflow().invoke(request, WorkflowServices(
        plan=lambda _request: (calls.append("plan") or plan),
        dispatch=lambda _plan: (calls.append("dispatch") or ()),
        decide=lambda *_args: None,
        technical=lambda _reports: None,
        execution_observer=observations,
    ))

    assert calls == ["plan"]
    assert result.terminal_mode is TerminalMode.UNKNOWN
    assert "budget_exhausted" in result.uncertainty_reason
    assert tuple(item.node for item in observations.checkpoints()) == (CoreNodeName.PLAN,)


def test_risk_degrade_permit_routes_to_finalize_without_assembly() -> None:
    request = WorkflowRequest(
        workflow_id="workflow-1", trace_id="1" * 32,
        user_query="分析市场", trigger_reason="manual_once",
    )
    task = AgentTask(
        task_id="technical-1", workflow_id=request.workflow_id,
        trace_id=request.trace_id, task_type=TaskType.TECHNICAL,
        objective="Analyze technical evidence.", context_summary_id="summary-1",
        allowed_data=("context_summary",), allowed_tools=(),
        expected_output="technical-v1", acceptance_criteria=("Return report.",),
        difficulty=TaskDifficulty.NORMAL, model_tier=ModelTier.TERRA,
        prompt_version="technical-v1", attempt_timeout_seconds=30,
        maximum_retries=1, reserved_cost=0.05, remaining_workflow_cost=0.10,
        analysis_steps=("Inspect.", "Assess.", "Return."),
        escalation_rule="return_to_coordinator",
        conflict_return_rule="return_typed_conflict",
    )
    plan = CoordinatorPlan(
        workflow_id=request.workflow_id, trace_id=request.trace_id,
        revision=0, mode=WorkflowMode.ACTIVE, tasks=(task,),
    )
    report = AgentReport(
        task_id=task.task_id, workflow_id=request.workflow_id,
        trace_id=request.trace_id, status=ReportStatus.COMPLETED,
        knowledge_status=KnowledgeStatus.KNOWN, uncertainty_reason=None,
        summary="Supported bounded evidence.",
    )
    decision = DecisionDraft(
        knowledge_status=KnowledgeStatus.KNOWN, uncertainty_reason=None,
        action=Action.NO_TRADE, execute_now=False, decision_confidence=0.8,
    )

    def permit(checkpoint):
        return CheckpointPermit(
            workflow_id=checkpoint.workflow_id, trace_id=checkpoint.trace_id,
            checkpoint_ordinal=checkpoint.ordinal,
            checkpoint_digest=checkpoint.canonical_digest(),
            decision=(CheckpointDecision.DEGRADE
                      if checkpoint.node is CoreNodeName.RISK
                      else CheckpointDecision.CONTINUE),
            reason_code=("budget_exhausted"
                         if checkpoint.node is CoreNodeName.RISK
                         else "checkpoint_authorized"),
        )

    observations = ExecutionObservationCollector(
        request.workflow_id, request.trace_id, checkpoint_sink=permit
    )
    result = CoordinatedWorkflow().invoke(request, WorkflowServices(
        plan=lambda _request: plan, dispatch=lambda _plan: (report,),
        recover=lambda current, reports: (current, reports),
        decide=lambda *_args: decision, technical=lambda _reports: None,
        verify=lambda *_args: decision, execution_observer=observations,
    ))

    assert result.terminal_mode is TerminalMode.UNKNOWN
    assert tuple(item.node for item in observations.checkpoints()) == (
        CoreNodeName.PLAN, CoreNodeName.DISPATCH, CoreNodeName.RECOVER,
        CoreNodeName.DECIDE, CoreNodeName.REFLECT, CoreNodeName.RISK,
    )


def test_partial_recovery_preserves_survivor_owner_and_marks_only_new_work_recovery_owned() -> None:
    request = WorkflowRequest(
        workflow_id="workflow-1", trace_id="1" * 32,
        user_query="分析市场", trigger_reason="manual_once",
    )

    def task(identifier: str, *, parent: str | None = None) -> AgentTask:
        return AgentTask(
            task_id=identifier, parent_task_id=parent, workflow_id=request.workflow_id,
            trace_id=request.trace_id, task_type=TaskType.TECHNICAL,
            objective="Analyze bounded technical evidence.", context_summary_id="summary-" + identifier,
            allowed_data=("context_summary",), allowed_tools=(), expected_output="technical-v1",
            acceptance_criteria=("Return one report.",), difficulty=TaskDifficulty.NORMAL,
            model_tier=ModelTier.TERRA, prompt_version="technical-v1",
            attempt_timeout_seconds=30, maximum_retries=1, reserved_cost=0.05,
            remaining_workflow_cost=0.10,
            analysis_steps=("Inspect evidence.", "Assess setup.", "Return conclusion."),
            escalation_rule="return_to_coordinator", conflict_return_rule="return_typed_conflict",
        )

    survivor, replaced, recovery = task("survivor"), task("replaced"), task("recovery", parent="replaced")
    initial = CoordinatorPlan(
        workflow_id=request.workflow_id, trace_id=request.trace_id, revision=0,
        mode=WorkflowMode.ACTIVE, tasks=(survivor, replaced),
    )
    recovered = CoordinatorPlan(
        workflow_id=request.workflow_id, trace_id=request.trace_id, revision=1,
        mode=WorkflowMode.ACTIVE, tasks=(survivor, recovery),
    )

    def report(task_value: AgentTask) -> AgentReport:
        return AgentReport(
            task_id=task_value.task_id, workflow_id=request.workflow_id,
            trace_id=request.trace_id, status=ReportStatus.COMPLETED,
            knowledge_status=KnowledgeStatus.KNOWN, uncertainty_reason=None,
            summary="Bounded evidence is supported.",
        )

    observations = ExecutionObservationCollector(request.workflow_id, request.trace_id)
    CoordinatedWorkflow().invoke(request, WorkflowServices(
        plan=lambda _request: initial,
        dispatch=lambda _plan: (report(survivor), report(replaced)),
        recover=lambda _plan, _reports: (recovered, (report(survivor), report(recovery))),
        decide=lambda *_args: None, technical=lambda _reports: None,
        execution_observer=observations,
    ))

    recovery_checkpoint = observations.checkpoints()[2]
    owners = {item.task_id: item.owner_node for item in recovery_checkpoint.work_items}
    assert owners == {
        "survivor": CoreNodeName.DISPATCH,
        "replaced": CoreNodeName.DISPATCH,
        "recovery": CoreNodeName.RECOVER,
    }
    assert recovery_checkpoint.task_ids == ("survivor", "replaced", "recovery")
