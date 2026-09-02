from __future__ import annotations

from types import SimpleNamespace

import pytest

from market_agent.backend.settings import BackendSettings
from market_agent.workflow_production_application import (
    ProductionDependencies,
    ProductionWorkflowApplication,
)

from market_agent.workflow_contracts import (
    Action,
    InformationalAnswer,
    KnowledgeStatus,
    TerminalMode,
    WorkflowRequest,
    WorkflowResult,
)

TRACE_ID = "1" * 32
RUN_ID = "workflow-1"


def _request() -> WorkflowRequest:
    return WorkflowRequest(
        workflow_id=RUN_ID,
        trace_id=TRACE_ID,
        user_query="系统会自动下单吗",
        trigger_reason="manual_once",
    )


def _result() -> WorkflowResult:
    return WorkflowResult(
        workflow_id=RUN_ID,
        trace_id=TRACE_ID,
        terminal_mode=TerminalMode.INFORMATIONAL,
        final_action=Action.NO_TRADE,
        knowledge_status=KnowledgeStatus.KNOWN,
        uncertainty_reason=None,
        informational_answer=InformationalAnswer(
            knowledge_status=KnowledgeStatus.KNOWN,
            uncertainty_reason=None,
            answer="不会。系统没有下单权限。",
        ),
    )


def _application(committed: list[object]) -> ProductionWorkflowApplication:
    dependencies = ProductionDependencies(
        settings=BackendSettings(environment="test"),
        driver_factory=lambda _tenant: object(),
        audit_writer=object(),
        memory_repository=None,
        embedding_client=None,
        completion_hook=committed.append,
        prompt_release_manager=SimpleNamespace(
            current=lambda: SimpleNamespace(release_digest="a" * 64)
        ),
    )
    return ProductionWorkflowApplication(lambda: dependencies)


def test_accepted_result_commit_is_explicit_and_identity_bound() -> None:
    committed: list[object] = []
    application = _application(committed)
    request = _request()
    result = _result()

    application.commit_accepted_result(request, result)
    assert committed == [result]

    with pytest.raises(RuntimeError, match="identity"):
        application.commit_accepted_result(
            request,
            result.model_copy(update={"workflow_id": "workflow-other"}),
        )
    assert committed == [result]
