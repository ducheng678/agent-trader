"""Execution projection boundary for committed deterministic Harness transitions."""

from __future__ import annotations

from collections.abc import Mapping
from threading import RLock
from typing import Protocol, TypedDict, cast, runtime_checkable

from pydantic import StrictBool

from market_agent.workflow_contracts import (
    ContractModel,
    NonNegativeInt,
    ShortText,
)
from market_agent.workflow_harness_contracts import (
    AttemptState,
    HarnessPlan,
    HarnessSessionView,
    HarnessTransition,
    RunState,
    WorkItemState,
)

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as error:  # pragma: no cover - exercised only without the extra
    END = START = StateGraph = None
    _LANGGRAPH_IMPORT_ERROR: ImportError | None = error
else:
    _LANGGRAPH_IMPORT_ERROR = None


class ExecutionBackendError(RuntimeError):
    """Base class for an execution-backend boundary failure."""


class ExecutionBackendUnavailableError(ExecutionBackendError):
    """The optional execution-engine dependency is not installed."""


class InvalidExecutionInputError(ExecutionBackendError):
    """A public input is not an exact, valid strict Harness contract."""


class ExecutionRegistrationError(ExecutionBackendError):
    """A run cannot be registered without conflicting with executor state."""


class ExecutionIdentityError(ExecutionBackendError):
    """Run, trace, or entity identity does not match the registered execution."""


class ExecutionPlanMismatchError(ExecutionBackendError):
    """The active plan identity or revision does not match."""


class ExecutionHandleMismatchError(ExecutionBackendError):
    """The caller supplied a stale, forged, or unknown execution handle."""


class UncommittedTransitionError(ExecutionBackendError):
    """A value other than an exact committed HarnessTransition reached routing."""


class StaleExecutionTransitionError(ExecutionBackendError):
    """A transition was prepared against a different authoritative revision."""


class DuplicateExecutionTransitionError(ExecutionBackendError):
    """A committed transition idempotency key was already projected."""


class CancelledExecutionError(ExecutionBackendError):
    """A cancelled executor projection cannot be resumed or advanced."""


class ExecutionProjectionError(ExecutionBackendError):
    """LangGraph failed to project an otherwise valid committed transition."""


class ExecutionHandle(ContractModel):
    """Frozen executor projection; it is never an orchestration authority."""

    run_id: ShortText
    trace_id: ShortText
    plan_id: ShortText
    plan_revision: NonNegativeInt
    state_revision: NonNegativeInt
    routed_state: ShortText | None = None
    cancelled: StrictBool = False


class BackendProjection(TypedDict, total=False):
    """Disposable LangGraph state for one committed transition projection."""

    committed_transition: HarnessTransition
    routed_state: str


@runtime_checkable
class ExecutionBackend(Protocol):
    def register(
        self, plan: HarnessPlan, view: HarnessSessionView
    ) -> ExecutionHandle: ...

    def apply_committed_transition(
        self, handle: ExecutionHandle, transition: HarnessTransition
    ) -> ExecutionHandle: ...

    def resume(
        self,
        plan: HarnessPlan,
        folded_view: HarnessSessionView,
        *,
        disposable_checkpoint: object | None = None,
    ) -> ExecutionHandle: ...

    def cancel(self, run_id: str) -> None: ...


def _reject_undeclared_model_fields(value: object) -> None:
    if isinstance(value, ContractModel):
        declared = type(value).model_fields
        if set(value.__dict__).difference(declared):
            raise ValueError("contract model contains undeclared fields")
        for item in value.__dict__.values():
            _reject_undeclared_model_fields(item)
    elif isinstance(value, Mapping):
        for item in value.values():
            _reject_undeclared_model_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_undeclared_model_fields(item)


def _fresh_contract(
    value: object,
    expected_type: type[ContractModel],
    error_type: type[ExecutionBackendError],
    message: str,
) -> ContractModel:
    if type(value) is not expected_type:
        raise error_type(message)
    try:
        _reject_undeclared_model_fields(value)
        values = value.model_dump(mode="python", exclude_unset=True)
        return expected_type.model_validate(values)
    except ExecutionBackendError:
        raise
    except Exception as error:
        raise error_type(message) from error


def _fresh_plan(value: object) -> HarnessPlan:
    return cast(
        HarnessPlan,
        _fresh_contract(
            value,
            HarnessPlan,
            InvalidExecutionInputError,
            "execution plan must be an exact valid HarnessPlan",
        ),
    )


def _fresh_view(value: object) -> HarnessSessionView:
    return cast(
        HarnessSessionView,
        _fresh_contract(
            value,
            HarnessSessionView,
            InvalidExecutionInputError,
            "execution view must be an exact valid folded HarnessSessionView",
        ),
    )


def _fresh_handle(value: object) -> ExecutionHandle:
    return cast(
        ExecutionHandle,
        _fresh_contract(
            value,
            ExecutionHandle,
            ExecutionHandleMismatchError,
            "execution handle is stale, forged, or invalid",
        ),
    )


def _fresh_transition(value: object) -> HarnessTransition:
    return cast(
        HarnessTransition,
        _fresh_contract(
            value,
            HarnessTransition,
            UncommittedTransitionError,
            "LangGraph routing requires an exact committed HarnessTransition",
        ),
    )


def _validated_route(transition: HarnessTransition) -> str:
    enum_type = {
        "run": RunState,
        "work_item": WorkItemState,
        "attempt": AttemptState,
    }[transition.entity_kind]
    try:
        enum_type(transition.to_state)
    except ValueError as error:
        raise UncommittedTransitionError(
            "committed transition has an unknown target state"
        ) from error
    return transition.to_state


def route_committed_transition(state: BackendProjection) -> str:
    """Return the only permitted edge selector: a strict committed transition."""

    if not isinstance(state, Mapping):
        raise UncommittedTransitionError(
            "LangGraph routing requires a committed-transition projection"
        )
    transition = _fresh_transition(state.get("committed_transition"))
    return _validated_route(transition)


def _accept_committed_transition(state: BackendProjection) -> BackendProjection:
    transition = _fresh_transition(state.get("committed_transition"))
    _validated_route(transition)
    return {"committed_transition": transition}


def _project_route(route: str):
    def project(state: BackendProjection) -> BackendProjection:
        transition = _fresh_transition(state.get("committed_transition"))
        if _validated_route(transition) != route:
            raise UncommittedTransitionError(
                "committed transition changed during LangGraph projection"
            )
        return {"routed_state": route}

    return project


_ALL_ROUTES = tuple(
    sorted(
        {
            *(state.value for state in RunState),
            *(state.value for state in WorkItemState),
            *(state.value for state in AttemptState),
        }
    )
)


def _build_projection_graph():
    if StateGraph is None or START is None or END is None:
        raise ExecutionBackendUnavailableError(
            "LangGraph is required for LangGraphExecutionBackend"
        ) from _LANGGRAPH_IMPORT_ERROR
    builder = StateGraph(BackendProjection)
    router_node = "accept_committed_transition"
    builder.add_node(router_node, _accept_committed_transition)
    builder.add_edge(START, router_node)
    route_nodes: dict[str, str] = {}
    for route in _ALL_ROUTES:
        node_name = f"project_{route}"
        route_nodes[route] = node_name
        builder.add_node(node_name, _project_route(route))
        builder.add_edge(node_name, END)
    builder.add_conditional_edges(router_node, route_committed_transition, route_nodes)
    return builder.compile()


def _validate_plan_view(
    plan: HarnessPlan, view: HarnessSessionView
) -> tuple[HarnessPlan, HarnessSessionView]:
    plan = _fresh_plan(plan)
    view = _fresh_view(view)
    if view.run_id is None or view.trace_id is None:
        raise ExecutionIdentityError(
            "executor registration requires folded run and trace identity"
        )
    if plan.run_id != view.run_id or plan.trace_id != view.trace_id:
        raise ExecutionIdentityError(
            "plan identity does not match the folded Harness view"
        )
    if plan.revision != view.plan_revision:
        raise ExecutionPlanMismatchError(
            "plan revision does not match the folded Harness view"
        )
    return plan, view


def _handle_from_view(
    plan: HarnessPlan, view: HarnessSessionView
) -> ExecutionHandle:
    return ExecutionHandle(
        run_id=plan.run_id,
        trace_id=plan.trace_id,
        plan_id=plan.plan_id,
        plan_revision=plan.revision,
        state_revision=view.state_revision,
        routed_state=view.run_state.value if view.run_state is not None else None,
        cancelled=False,
    )


class LangGraphExecutionBackend:
    """Disposable LangGraph projection driven only by committed transitions."""

    def __init__(self) -> None:
        self._graph = _build_projection_graph()
        self._lock = RLock()
        self._handles: dict[str, ExecutionHandle] = {}
        self._plans: dict[str, HarnessPlan] = {}
        self._applied_idempotency_keys: dict[str, set[str]] = {}
        self._cancelled_run_ids: set[str] = set()

    def register(
        self, plan: HarnessPlan, view: HarnessSessionView
    ) -> ExecutionHandle:
        plan, view = _validate_plan_view(plan, view)
        handle = _handle_from_view(plan, view)
        with self._lock:
            if plan.run_id in self._cancelled_run_ids:
                raise CancelledExecutionError(
                    "cancelled execution cannot be registered again"
                )
            existing = self._handles.get(plan.run_id)
            if existing is not None:
                if existing == handle and self._plans.get(plan.run_id) == plan:
                    return existing
                raise ExecutionRegistrationError(
                    "run is already registered with different executor state"
                )
            self._handles[plan.run_id] = handle
            self._plans[plan.run_id] = plan
            self._applied_idempotency_keys[plan.run_id] = set(
                view.applied_idempotency_keys
            )
            return handle

    def apply_committed_transition(
        self, handle: ExecutionHandle, transition: HarnessTransition
    ) -> ExecutionHandle:
        handle = _fresh_handle(handle)
        transition = _fresh_transition(transition)
        with self._lock:
            if handle.run_id in self._cancelled_run_ids:
                raise CancelledExecutionError(
                    "cancelled execution cannot project transitions"
                )
            current = self._handles.get(handle.run_id)
            if current is None or current != handle:
                raise ExecutionHandleMismatchError(
                    "execution handle is stale, forged, or unknown"
                )
            applied = self._applied_idempotency_keys[handle.run_id]
            if transition.idempotency_key in applied:
                raise DuplicateExecutionTransitionError(
                    "transition idempotency key was already projected"
                )
            if (
                transition.run_id != handle.run_id
                or transition.trace_id != handle.trace_id
                or (
                    transition.entity_kind == "run"
                    and transition.entity_id != handle.run_id
                )
            ):
                raise ExecutionIdentityError(
                    "transition identity does not match execution handle"
                )
            if transition.plan_revision != handle.plan_revision:
                raise ExecutionPlanMismatchError(
                    "transition plan revision does not match execution handle"
                )
            if transition.expected_state_revision != handle.state_revision:
                raise StaleExecutionTransitionError(
                    "transition expected state revision is stale"
                )
            try:
                projected = self._graph.invoke(
                    {"committed_transition": transition}
                )
            except UncommittedTransitionError:
                raise
            except Exception as error:
                raise ExecutionProjectionError(
                    "LangGraph failed to project committed transition"
                ) from error
            routed_state = projected.get("routed_state")
            if routed_state != transition.to_state:
                raise ExecutionProjectionError(
                    "LangGraph projected an inconsistent transition target"
                )
            advanced = ExecutionHandle(
                run_id=handle.run_id,
                trace_id=handle.trace_id,
                plan_id=handle.plan_id,
                plan_revision=handle.plan_revision,
                state_revision=handle.state_revision + 1,
                routed_state=routed_state,
                cancelled=False,
            )
            self._handles[handle.run_id] = advanced
            applied.add(transition.idempotency_key)
            return advanced

    def resume(
        self,
        plan: HarnessPlan,
        folded_view: HarnessSessionView,
        *,
        disposable_checkpoint: object | None = None,
    ) -> ExecutionHandle:
        # Checkpoints are deliberately neither read nor validated. The event-folded
        # view is the complete authority for executor reconstruction.
        plan, folded_view = _validate_plan_view(plan, folded_view)
        handle = _handle_from_view(plan, folded_view)
        with self._lock:
            if plan.run_id in self._cancelled_run_ids:
                raise CancelledExecutionError(
                    "cancelled execution cannot be resumed"
                )
            existing_plan = self._plans.get(plan.run_id)
            if existing_plan is not None and existing_plan != plan:
                raise ExecutionPlanMismatchError(
                    "resume plan differs from the registered execution plan"
                )
            self._handles[plan.run_id] = handle
            self._plans[plan.run_id] = plan
            self._applied_idempotency_keys[plan.run_id] = set(
                folded_view.applied_idempotency_keys
            )
            return handle

    def cancel(self, run_id: str) -> None:
        if type(run_id) is not str:
            raise InvalidExecutionInputError(
                "run identifier must be a strict string"
            )
        normalized = run_id.strip()
        if not normalized or len(normalized) > 256:
            raise InvalidExecutionInputError(
                "run identifier must be nonblank and at most 256 characters"
            )
        with self._lock:
            current = self._handles.get(normalized)
            if current is None or normalized in self._cancelled_run_ids:
                return
            cancelled = ExecutionHandle(
                run_id=current.run_id,
                trace_id=current.trace_id,
                plan_id=current.plan_id,
                plan_revision=current.plan_revision,
                state_revision=current.state_revision,
                routed_state=current.routed_state,
                cancelled=True,
            )
            self._handles[normalized] = cancelled
            self._cancelled_run_ids.add(normalized)
