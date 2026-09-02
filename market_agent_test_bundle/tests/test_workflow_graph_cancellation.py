from market_agent.workflow_contracts import (
    KnowledgeStatus,
    TerminalMode,
    WorkflowRequest,
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
