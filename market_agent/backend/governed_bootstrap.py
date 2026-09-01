"""Load deployment-owned Harness authority without handling signing secrets."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Callable

from fastapi import FastAPI

from market_agent.backend.governed_app import CompletionCandidateFactory, create_governed_app
from market_agent.backend.settings import BackendSettings
from market_agent.workflow_harness import HarnessKernel


@dataclass(frozen=True, slots=True)
class HarnessHostBindings:
    kernel: HarnessKernel
    completion_candidate_factory: CompletionCandidateFactory | None = None


def _load_factory(reference: str) -> Callable[[], HarnessHostBindings]:
    module_name, separator, attribute = reference.rpartition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("MARKET_AGENT_HARNESS_HOST_FACTORY must be module:callable")
    factory = getattr(import_module(module_name), attribute, None)
    if not callable(factory):
        raise TypeError("configured Harness host factory is not callable")
    return factory


def create_governed_app_from_environment(
    settings: BackendSettings | None = None,
) -> FastAPI:
    resolved = (settings or BackendSettings.from_env()).validate()
    factory = _load_factory(resolved.harness_host_factory)
    bindings = factory()
    if type(bindings) is not HarnessHostBindings or type(bindings.kernel) is not HarnessKernel:
        raise TypeError("Harness host factory returned invalid bindings")
    if bindings.completion_candidate_factory is not None and not callable(bindings.completion_candidate_factory):
        raise TypeError("Harness completion candidate factory must be callable")
    return create_governed_app(
        settings=resolved,
        harness_kernel=bindings.kernel,
        completion_candidate_factory=bindings.completion_candidate_factory,
    )
