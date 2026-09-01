"""Deterministic, immutable Harness plan templates and compiler."""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from types import MappingProxyType
from typing import Annotated

from pydantic import Field, StrictBool, model_validator
from pydantic import ValidationError

from market_agent.workflow_contracts import (
    ContractModel,
    ShortText,
    Text,
    WorkflowMode,
)
from market_agent.workflow_harness_contracts import (
    HarnessPlan,
    OutcomeKind,
    PinnedVersions,
    ProgressTargetSet,
    RiskClass,
    SourceCoverageWeight,
    StageSpec,
    TaskKind,
    WorkItemSpec,
)
from market_agent.workflow_contracts import WorkflowRequest
from market_agent.workflow_worker_registry import UnknownWorkerError, WorkerRegistry


TargetIds = Annotated[tuple[ShortText, ...], Field(max_length=64)]
Stages = Annotated[tuple[StageSpec, ...], Field(min_length=1, max_length=64)]
WorkerIds = Annotated[tuple[ShortText, ...], Field(min_length=1, max_length=64)]
CoverageWeights = Annotated[tuple[SourceCoverageWeight, ...], Field(max_length=64)]
class PlanRegistryError(ValueError):
    """Base error for invalid immutable plan-template registry operations."""


class DuplicateTemplateError(PlanRegistryError):
    """Raised when a template identifier or admission key is declared twice."""


class UnknownTemplateError(PlanRegistryError):
    """Raised when no declared template is available for a deterministic admission."""


class InconsistentTemplateError(PlanRegistryError):
    """Raised when a template graph or worker reference is not self-consistent."""


class PlanTemplate(ContractModel):
    """A fully declared template whose graph is frozen before any worker executes."""

    template_id: ShortText
    version: ShortText
    mode: WorkflowMode
    task_kind: TaskKind
    risk_class: RiskClass
    stages: Stages
    worker_ids: WorkerIds
    work_item_id: ShortText
    work_item_stage_id: ShortText
    work_item_worker_id: ShortText
    objective: Text
    work_item_dependencies: TargetIds = ()
    progress_output_fields: TargetIds = ()
    progress_evidence_slots: TargetIds = ()
    source_coverage_weights: CoverageWeights = ()
    known_conflict_slots: TargetIds = ()
    risk_invariant_ids: TargetIds = ()
    allows_side_effects: StrictBool

    @model_validator(mode="after")
    def validate_template_graph(self) -> PlanTemplate:
        stage_ids = tuple(stage.stage_id for stage in self.stages)
        if len(stage_ids) != len(set(stage_ids)):
            raise ValueError("template stage identifiers must be unique")
        if len(self.worker_ids) != len(set(self.worker_ids)):
            raise ValueError("template worker identifiers must be unique")
        if self.work_item_stage_id not in stage_ids:
            raise ValueError("work item stage must be declared")
        if self.work_item_worker_id not in self.worker_ids:
            raise ValueError("work item worker must be declared")
        stage = next(stage for stage in self.stages if stage.stage_id == self.work_item_stage_id)
        if self.task_kind not in stage.allowed_task_kinds:
            raise ValueError("work item task kind must be allowed by its stage")
        if any(stage.allows_side_effects for stage in self.stages) != self.allows_side_effects:
            raise ValueError("template side-effect policy must match its stages")
        return self


class PlanTemplateRegistry:
    """A fixed, uniquely selectable set of fully declared plan templates."""

    def __init__(self, templates: Iterable[PlanTemplate]) -> None:
        materialized = tuple(_revalidate_template(template) for template in templates)
        by_id = {template.template_id: template for template in materialized}
        if len(by_id) != len(materialized):
            raise DuplicateTemplateError("template identifiers must be unique")
        by_admission = {
            (template.mode, template.task_kind, template.risk_class): template
            for template in materialized
        }
        if len(by_admission) != len(materialized):
            raise DuplicateTemplateError("template admission keys must be unique")
        self._all = materialized
        self._templates = MappingProxyType(by_id)
        self._admissions = MappingProxyType(by_admission)

    def get(self, template_id: str) -> PlanTemplate:
        try:
            return self._templates[template_id]
        except KeyError as error:
            raise UnknownTemplateError(f"unknown template identifier: {template_id}") from error

    def all(self) -> tuple[PlanTemplate, ...]:
        return self._all

    def select(
        self, *, mode: WorkflowMode, task_kind: TaskKind, risk_class: RiskClass
    ) -> PlanTemplate:
        key = (mode, task_kind, risk_class)
        try:
            return self._admissions[key]
        except KeyError as error:
            raise UnknownTemplateError(
                "no template for deterministic admission: "
                f"{mode.value}/{task_kind.value}/{risk_class.value}"
            ) from error


class PlanCompiler:
    """Compile one deterministic, immutable plan from validated request fields only."""

    def __init__(self, templates: PlanTemplateRegistry, workers: WorkerRegistry) -> None:
        self._templates = templates
        self._workers = workers
        for template in self._templates.all():
            self._validate_template_workers(template, self._resolve_workers(template))

    def compile(
        self, request: WorkflowRequest, pinned_versions: PinnedVersions
    ) -> HarnessPlan:
        mode, task_kind, risk_class = _admit(request)
        template = self._templates.select(
            mode=mode, task_kind=task_kind, risk_class=risk_class
        )
        workers = self._resolve_workers(template)
        self._validate_template_workers(template, workers)
        work_item = WorkItemSpec(
            work_item_id=template.work_item_id,
            stage_id=template.work_item_stage_id,
            worker_id=template.work_item_worker_id,
            task_kind=template.task_kind,
            objective=template.objective,
            dependencies=template.work_item_dependencies,
            progress_targets=ProgressTargetSet(
                required_dependency_ids=template.work_item_dependencies,
                required_output_field_paths=template.progress_output_fields,
                required_evidence_slot_ids=template.progress_evidence_slots,
                required_source_coverage_weights=template.source_coverage_weights,
                known_conflict_slot_ids=template.known_conflict_slots,
                risk_invariant_ids=template.risk_invariant_ids,
            ),
        )
        return HarnessPlan(
            plan_id=_plan_id(request.workflow_id, template.template_id),
            run_id=request.workflow_id,
            trace_id=request.trace_id,
            template_id=template.template_id,
            revision=0,
            mode=template.mode,
            task_kind=template.task_kind,
            risk_class=template.risk_class,
            pinned_versions=pinned_versions,
            stages=template.stages,
            workers=workers,
            work_items=(work_item,),
            allows_side_effects=template.allows_side_effects,
        )

    def _resolve_workers(self, template: PlanTemplate) -> tuple:
        try:
            return tuple(self._workers.get(worker_id) for worker_id in template.worker_ids)
        except UnknownWorkerError as error:
            raise InconsistentTemplateError(
                f"template references unknown worker: {error.args[0].removeprefix('unknown worker identifier: ')}"
            ) from error

    @staticmethod
    def _validate_template_workers(template: PlanTemplate, workers: tuple) -> None:
        worker_by_id = {worker.worker_id: worker for worker in workers}
        worker = worker_by_id.get(template.work_item_worker_id)
        if worker is None:
            raise InconsistentTemplateError("work item worker must be declared")
        if template.task_kind not in worker.supported_task_kinds:
            raise InconsistentTemplateError("work item task kind must be supported by its worker")


def _admit(request: WorkflowRequest) -> tuple[WorkflowMode, TaskKind, RiskClass]:
    """Fail closed until ingress carries independently trusted typed intent."""

    return WorkflowMode.PASSIVE, TaskKind.INFORMATIONAL, RiskClass.INFORMATIONAL


def _revalidate_template(template: PlanTemplate) -> PlanTemplate:
    try:
        validated = PlanTemplate.model_validate(template.__dict__)
    except (AttributeError, ValidationError) as error:
        raise InconsistentTemplateError(str(error)) from error
    _validate_template_graph(validated)
    return validated


def _validate_template_graph(template: PlanTemplate) -> None:
    stage_dependencies = {stage.stage_id: stage.dependencies for stage in template.stages}
    declared_stage_ids = frozenset(stage_dependencies)
    if any(
        dependency not in declared_stage_ids
        for dependencies in stage_dependencies.values()
        for dependency in dependencies
    ):
        raise InconsistentTemplateError(
            "stage dependencies must reference declared identifiers"
        )

    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(stage_id: str) -> None:
        if stage_id in visiting:
            raise InconsistentTemplateError("stage dependencies must be acyclic")
        if stage_id in visited:
            return
        visiting.add(stage_id)
        for dependency in stage_dependencies[stage_id]:
            visit(dependency)
        visiting.remove(stage_id)
        visited.add(stage_id)

    for stage_id in stage_dependencies:
        visit(stage_id)

    if template.work_item_dependencies:
        raise InconsistentTemplateError("work item dependencies must be empty")


def _plan_id(run_id: str, template_id: str) -> str:
    """Return a fixed-length stable identifier for a run/template compilation."""

    material = f"{run_id}\x1f{template_id}".encode("utf-8")
    return hashlib.sha256(material).hexdigest()
