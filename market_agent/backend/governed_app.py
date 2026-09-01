"""Explicit production composition entrypoint for the governed workflow API."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import FastAPI

from market_agent.backend.api import create_app
from market_agent.backend.container import BackendContainer
from market_agent.backend.settings import BackendSettings
from market_agent.workflow_contracts import WorkflowRequest, WorkflowResult
from market_agent.workflow_harness import HarnessKernel
from market_agent.workflow_harness_contracts import HarnessSessionView


CompletionCandidateFactory = Callable[
    [WorkflowRequest, WorkflowResult, HarnessSessionView], dict[str, object]
]


def create_governed_app(
    *,
    harness_kernel: HarnessKernel,
    completion_candidate_factory: CompletionCandidateFactory | None = None,
    settings: BackendSettings | None = None,
) -> FastAPI:
    """Build an API whose workflow path is tied to one trusted Harness host.

    The receipt issuer belongs to the supplied kernel.  This function neither
    reads private signing material nor substitutes a development authority.
    """

    if type(harness_kernel) is not HarnessKernel:
        raise TypeError("governed application requires an exact HarnessKernel")
    if completion_candidate_factory is not None and not callable(completion_candidate_factory):
        raise TypeError("completion candidate factory must be callable")
    container = BackendContainer.create(
        settings=settings,
        harness_kernel=harness_kernel,
        harness_completion_candidate_factory=completion_candidate_factory,
    )
    return create_app(container)
