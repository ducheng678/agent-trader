"""Immutable, fail-closed worker specifications for the Harness."""

from __future__ import annotations

from collections.abc import Iterable
from types import MappingProxyType

from pydantic import ValidationError

from market_agent.workflow_harness_contracts import WorkerSpec


class WorkerRegistryError(ValueError):
    """Base error for invalid worker registry operations."""


class DuplicateWorkerError(WorkerRegistryError):
    """Raised when a registry declares one worker identifier more than once."""


class InvalidWorkerError(WorkerRegistryError):
    """Raised when a worker specification bypasses strict contract validation."""


class UnknownWorkerError(WorkerRegistryError):
    """Raised when a plan refers to a worker that the registry does not declare."""


class WorkerRegistry:
    """A fixed ordered set of immutable ``WorkerSpec`` values."""

    def __init__(self, specs: Iterable[WorkerSpec]) -> None:
        try:
            materialized = tuple(
                WorkerSpec.model_validate(spec.__dict__) for spec in specs
            )
        except (AttributeError, ValidationError) as error:
            raise InvalidWorkerError("invalid worker specification") from error
        by_id = {spec.worker_id: spec for spec in materialized}
        if len(by_id) != len(materialized):
            raise DuplicateWorkerError("worker identifiers must be unique")
        self._all = materialized
        self._specs = MappingProxyType(by_id)

    def get(self, worker_id: str) -> WorkerSpec:
        try:
            return self._specs[worker_id]
        except KeyError as error:
            raise UnknownWorkerError(f"unknown worker identifier: {worker_id}") from error

    def all(self) -> tuple[WorkerSpec, ...]:
        return self._all
