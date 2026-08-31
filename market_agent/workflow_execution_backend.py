"""Execution projection boundary for durably committed Harness transitions."""

from __future__ import annotations

from collections.abc import Callable, Mapping
import hashlib
import json
from threading import RLock
from typing import Protocol, TypedDict, cast, final, runtime_checkable

from pydantic import StrictBool, model_validator

from market_agent.workflow_contracts import ContractModel, Digest, NonNegativeInt, PositiveInt, ShortText
from market_agent.workflow_harness_contracts import (
    AttemptState,
    AttemptWorkItemOwnershipRecord,
    HarnessPlan,
    HarnessSessionView,
    HarnessTransition,
    RunState,
    TransitionAuthorityRecord,
    WorkItemState,
)
from market_agent.workflow_state_machine import (
    AttemptTransitionAuthorization,
    GlobalTaskStateMachine,
    RunTransitionEvidence,
    StateMachineError,
    WorkItemTransitionAuthorization,
)

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as error:  # pragma: no cover
    END = START = StateGraph = None
    _LANGGRAPH_IMPORT_ERROR: ImportError | None = error
else:
    _LANGGRAPH_IMPORT_ERROR = None


class ExecutionBackendError(RuntimeError):
    """Base execution-backend boundary failure."""


class ExecutionBackendUnavailableError(ExecutionBackendError):
    """The optional execution-engine dependency is unavailable."""


class InvalidExecutionInputError(ExecutionBackendError):
    """A public value is not an exact strict Harness contract."""


class ExecutionRegistrationError(ExecutionBackendError):
    """Registration conflicts with executor state."""


class ExecutionIdentityError(ExecutionBackendError):
    """Execution or entity ownership does not match."""


class ExecutionPlanMismatchError(ExecutionBackendError):
    """Committed plan identity, digest, or revision does not match."""


class ExecutionHandleMismatchError(ExecutionBackendError):
    """An execution handle is stale, forged, or unknown."""


class UncommittedTransitionError(ExecutionBackendError):
    """Unvalidated data reached the LangGraph router."""


class UnverifiedExecutionReceiptError(ExecutionBackendError):
    """Commit authority is absent, malformed, or unverified."""


class InvalidCommittedTransitionError(ExecutionBackendError):
    """A claimed commit is inconsistent with authoritative state."""


class StaleExecutionSnapshotError(ExecutionBackendError):
    """Resume would replace authority with stale or divergent state."""


class StaleExecutionTransitionError(ExecutionBackendError):
    """A transition targets a different authoritative revision."""


class DuplicateExecutionTransitionError(ExecutionBackendError):
    """A committed idempotency key was already projected."""


class CancelledExecutionError(ExecutionBackendError):
    """A cancelled projection cannot be resumed or advanced."""


class ExecutionProjectionError(ExecutionBackendError):
    """LangGraph failed to project a validated route."""


class ExecutionHandle(ContractModel):
    run_id: ShortText
    trace_id: ShortText
    plan_id: ShortText
    plan_revision: NonNegativeInt
    state_revision: NonNegativeInt
    routed_state: ShortText | None = None
    cancelled: StrictBool = False


class CommittedExecutionSnapshot(ContractModel):
    """Host-verifiable binding of a plan to one folded event-chain head."""

    run_id: ShortText
    trace_id: ShortText
    plan_id: ShortText
    plan_digest: Digest
    plan_revision: NonNegativeInt
    sequence: PositiveInt
    state_revision: NonNegativeInt
    view_digest: Digest
    event_head_hash: Digest


class CommittedTransitionReceipt(ContractModel):
    """Host-verifiable proof connecting two committed folds."""

    pre: CommittedExecutionSnapshot
    post: CommittedExecutionSnapshot
    transition_digest: Digest

    @model_validator(mode="after")
    def validate_continuity(self) -> CommittedTransitionReceipt:
        fields = ("run_id", "trace_id", "plan_id", "plan_digest", "plan_revision")
        if any(getattr(self.pre, name) != getattr(self.post, name) for name in fields):
            raise ValueError("receipt endpoints have different execution identity")
        if self.post.sequence != self.pre.sequence + 1:
            raise ValueError("receipt sequence must advance exactly once")
        if self.post.state_revision != self.pre.state_revision + 1:
            raise ValueError("receipt state revision must advance exactly once")
        if self.post.event_head_hash == self.pre.event_head_hash:
            raise ValueError("receipt event-chain head must advance")
        return self


@final
class ExecutionReceiptVerifier:
    """Frozen host capability that verifies immutable canonical receipt bytes."""

    __slots__ = ("_verify_bytes",)

    def __init__(self, verify_bytes: Callable[[bytes], bool]) -> None:
        if not callable(verify_bytes) or isinstance(verify_bytes, Mapping):
            raise InvalidExecutionInputError("receipt verifier function must be callable")
        object.__setattr__(self, "_verify_bytes", verify_bytes)

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError("receipt verifier is immutable")

    def verify(self, payload: bytes) -> bool:
        if type(payload) is not bytes:
            raise TypeError("receipt verifier input must be immutable bytes")
        return self._verify_bytes(payload)


_ROUTE_CAPABILITY = object()


class _ValidatedRoute:
    __slots__ = ("_capability", "target")

    def __init__(self, target: str, capability: object) -> None:
        if capability is not _ROUTE_CAPABILITY:
            raise TypeError("validated routes are backend-internal")
        self._capability = capability
        self.target = target


class BackendProjection(TypedDict, total=False):
    validated_route: object
    routed_state: str


@runtime_checkable
class ExecutionBackend(Protocol):
    def register(self, plan: HarnessPlan, view: HarnessSessionView, committed_snapshot: CommittedExecutionSnapshot) -> ExecutionHandle: ...
    def apply_committed_transition(self, handle: ExecutionHandle, transition: HarnessTransition, pre_view: HarnessSessionView, post_view: HarnessSessionView, receipt: CommittedTransitionReceipt) -> ExecutionHandle: ...
    def resume(self, plan: HarnessPlan, folded_view: HarnessSessionView, committed_snapshot: CommittedExecutionSnapshot, *, disposable_checkpoint: object | None = None) -> ExecutionHandle: ...
    def cancel(self, run_id: str) -> None: ...


def _reject_undeclared_model_fields(value: object) -> None:
    if isinstance(value, ContractModel):
        if set(value.__dict__).difference(type(value).model_fields):
            raise ValueError("contract model contains undeclared fields")
        for item in value.__dict__.values():
            _reject_undeclared_model_fields(item)
    elif isinstance(value, Mapping):
        for item in value.values():
            _reject_undeclared_model_fields(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_undeclared_model_fields(item)


def _require_same_contract_tree(original: object, fresh: object) -> None:
    if isinstance(original, ContractModel) or isinstance(fresh, ContractModel):
        if type(original) is not type(fresh):
            raise TypeError("nested contract runtime type changed during validation")
        for field in type(fresh).model_fields:
            _require_same_contract_tree(
                getattr(original, field), getattr(fresh, field)
            )
        return
    if isinstance(original, Mapping) or isinstance(fresh, Mapping):
        if not isinstance(original, Mapping) or not isinstance(fresh, Mapping):
            raise TypeError("nested contract mapping shape changed")
        if original.keys() != fresh.keys():
            raise TypeError("nested contract mapping keys changed")
        for key in original:
            _require_same_contract_tree(original[key], fresh[key])
        return
    if isinstance(original, (list, tuple)) or isinstance(fresh, (list, tuple)):
        if type(original) is not type(fresh) or len(original) != len(fresh):
            raise TypeError("nested contract sequence shape changed")
        for original_item, fresh_item in zip(original, fresh, strict=True):
            _require_same_contract_tree(original_item, fresh_item)


def _fresh_contract(value: object, expected_type: type[ContractModel], error_type: type[ExecutionBackendError], message: str) -> ContractModel:
    if type(value) is not expected_type:
        raise error_type(message)
    try:
        _reject_undeclared_model_fields(value)
        fresh = expected_type.model_validate(
            value.model_dump(mode="python", exclude_unset=True)
        )
        _require_same_contract_tree(value, fresh)
        return fresh
    except ExecutionBackendError:
        raise
    except Exception as error:
        raise error_type(message) from error


def _fresh_plan(value: object) -> HarnessPlan:
    return cast(HarnessPlan, _fresh_contract(value, HarnessPlan, InvalidExecutionInputError, "execution plan must be an exact valid HarnessPlan"))


def _fresh_view(value: object) -> HarnessSessionView:
    return cast(HarnessSessionView, _fresh_contract(value, HarnessSessionView, InvalidExecutionInputError, "execution view must be an exact valid folded HarnessSessionView"))


def _fresh_handle(value: object) -> ExecutionHandle:
    return cast(ExecutionHandle, _fresh_contract(value, ExecutionHandle, ExecutionHandleMismatchError, "execution handle is stale, forged, or invalid"))


def _fresh_transition(value: object) -> HarnessTransition:
    return cast(HarnessTransition, _fresh_contract(value, HarnessTransition, UncommittedTransitionError, "transition must be an exact valid HarnessTransition"))


def _fresh_snapshot(value: object) -> CommittedExecutionSnapshot:
    return cast(CommittedExecutionSnapshot, _fresh_contract(value, CommittedExecutionSnapshot, UnverifiedExecutionReceiptError, "execution requires an exact committed snapshot"))


def _fresh_receipt(value: object) -> CommittedTransitionReceipt:
    if type(value) is not CommittedTransitionReceipt or (
        type(value.pre) is not CommittedExecutionSnapshot
        or type(value.post) is not CommittedExecutionSnapshot
    ):
        raise UnverifiedExecutionReceiptError(
            "receipt endpoints must be exact committed snapshots"
        )
    return cast(CommittedTransitionReceipt, _fresh_contract(value, CommittedTransitionReceipt, UnverifiedExecutionReceiptError, "transition requires an exact committed receipt"))


def _canonical_bytes(value: ContractModel) -> bytes:
    return json.dumps(
        value.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _canonical_digest(value: ContractModel) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def canonical_plan_digest(plan: HarnessPlan) -> str:
    return _canonical_digest(_fresh_plan(plan))


def canonical_view_digest(view: HarnessSessionView) -> str:
    return _canonical_digest(_fresh_view(view))


def canonical_transition_digest(transition: HarnessTransition) -> str:
    return _canonical_digest(_fresh_transition(transition))


def _route_from_projection(state: object) -> _ValidatedRoute:
    if not isinstance(state, Mapping):
        raise UncommittedTransitionError("LangGraph requires a validated route")
    route = state.get("validated_route")
    if type(route) is not _ValidatedRoute or route._capability is not _ROUTE_CAPABILITY:
        raise UncommittedTransitionError("LangGraph accepts only backend-validated routes")
    return route


def route_committed_transition(state: BackendProjection) -> str:
    return _route_from_projection(state).target


def _accept_validated_route(state: BackendProjection) -> BackendProjection:
    return {"validated_route": _route_from_projection(state)}


def _project_route(target: str):
    def project(state: BackendProjection) -> BackendProjection:
        route = _route_from_projection(state)
        if route.target != target:
            raise UncommittedTransitionError("validated route changed during projection")
        return {"routed_state": target}
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
        raise ExecutionBackendUnavailableError("LangGraph is required") from _LANGGRAPH_IMPORT_ERROR
    builder = StateGraph(BackendProjection)
    router_node = "accept_validated_route"
    builder.add_node(router_node, _accept_validated_route)
    builder.add_edge(START, router_node)
    route_nodes = {}
    for target in _ALL_ROUTES:
        node_name = f"project_{target}"
        route_nodes[target] = node_name
        builder.add_node(node_name, _project_route(target))
        builder.add_edge(node_name, END)
    builder.add_conditional_edges(router_node, route_committed_transition, route_nodes)
    return builder.compile()


def _validate_plan_view(plan: HarnessPlan, view: HarnessSessionView) -> tuple[HarnessPlan, HarnessSessionView]:
    plan = _fresh_plan(plan)
    view = _fresh_view(view)
    if view.run_id is None or view.trace_id is None:
        raise ExecutionIdentityError("folded run and trace identity is required")
    if plan.run_id != view.run_id or plan.trace_id != view.trace_id:
        raise ExecutionIdentityError("plan identity does not match folded view")
    if plan.revision != view.plan_revision:
        raise ExecutionPlanMismatchError("plan revision does not match folded view")
    if view.last_event_hash is None:
        raise UnverifiedExecutionReceiptError("folded view has no committed event head")
    return plan, view


def _handle_from_view(plan: HarnessPlan, view: HarnessSessionView) -> ExecutionHandle:
    return ExecutionHandle(run_id=plan.run_id, trace_id=plan.trace_id, plan_id=plan.plan_id, plan_revision=plan.revision, state_revision=view.state_revision, routed_state=view.run_state.value if view.run_state is not None else None, cancelled=False)


_APPEND_ONLY_AUTHORITY_FIELDS = (
    "transition_authorities",
    "attempt_work_item_owners",
    "reconciliation_resolutions",
)


def _is_same_revision_authority_extension(
    current: HarnessSessionView, incoming: HarnessSessionView
) -> bool:
    if (
        incoming.state_revision != current.state_revision
        or incoming.sequence <= current.sequence
        or incoming.last_event_hash == current.last_event_hash
    ):
        return False
    ignored = {"sequence", "last_event_hash", *_APPEND_ONLY_AUTHORITY_FIELDS}
    if current.model_dump(exclude=ignored) != incoming.model_dump(exclude=ignored):
        return False
    appended = False
    for field in _APPEND_ONLY_AUTHORITY_FIELDS:
        previous = getattr(current, field)
        extended = getattr(incoming, field)
        if extended[: len(previous)] != previous:
            return False
        appended = appended or len(extended) > len(previous)
    return appended


class LangGraphExecutionBackend:
    """Disposable projection driven only by verified committed Harness state."""

    def __init__(self, *, authority_verifier: ExecutionReceiptVerifier) -> None:
        if type(authority_verifier) is not ExecutionReceiptVerifier:
            raise InvalidExecutionInputError(
                "authority verifier must be an exact ExecutionReceiptVerifier"
            )
        self._authority_verifier = authority_verifier
        self._graph = _build_projection_graph()
        self._state_machine = GlobalTaskStateMachine()
        self._lock = RLock()
        self._handles: dict[str, ExecutionHandle] = {}
        self._plans: dict[str, HarnessPlan] = {}
        self._views: dict[str, HarnessSessionView] = {}
        self._snapshots: dict[str, CommittedExecutionSnapshot] = {}
        self._applied_idempotency_keys: dict[str, set[str]] = {}
        self._cancelled_run_ids: set[str] = set()

    def _verify_authority(self, value: ContractModel) -> ContractModel:
        fresh_before = _fresh_contract(
            value,
            type(value),
            UnverifiedExecutionReceiptError,
            "commit authority contract is invalid",
        )
        payload = _canonical_bytes(fresh_before)
        try:
            verified = self._authority_verifier.verify(payload)
        except Exception as error:
            raise UnverifiedExecutionReceiptError("commit authority verification failed") from error
        if type(verified) is not bool or not verified:
            raise UnverifiedExecutionReceiptError("commit authority was not verified")
        fresh_after = _fresh_contract(
            fresh_before,
            type(fresh_before),
            UnverifiedExecutionReceiptError,
            "commit authority changed during verification",
        )
        if _canonical_bytes(fresh_after) != payload:
            raise UnverifiedExecutionReceiptError(
                "commit authority changed during verification"
            )
        return fresh_after

    def _validated_snapshot(self, plan: HarnessPlan, view: HarnessSessionView, snapshot: CommittedExecutionSnapshot) -> tuple[HarnessPlan, HarnessSessionView, CommittedExecutionSnapshot]:
        plan, view = _validate_plan_view(plan, view)
        snapshot = _fresh_snapshot(snapshot)
        matches = (
            snapshot.run_id == plan.run_id == view.run_id
            and snapshot.trace_id == plan.trace_id == view.trace_id
            and snapshot.plan_id == plan.plan_id
            and snapshot.plan_revision == plan.revision == view.plan_revision
            and snapshot.plan_digest == canonical_plan_digest(plan)
            and snapshot.sequence == view.sequence
            and snapshot.state_revision == view.state_revision
            and snapshot.view_digest == canonical_view_digest(view)
            and snapshot.event_head_hash == view.last_event_hash
        )
        if not matches:
            raise UnverifiedExecutionReceiptError("snapshot does not bind the plan and folded view")
        snapshot = cast(CommittedExecutionSnapshot, self._verify_authority(snapshot))
        return plan, view, snapshot

    def register(self, plan: HarnessPlan, view: HarnessSessionView, committed_snapshot: CommittedExecutionSnapshot) -> ExecutionHandle:
        plan, view, committed_snapshot = self._validated_snapshot(plan, view, committed_snapshot)
        handle = _handle_from_view(plan, view)
        with self._lock:
            if plan.run_id in self._cancelled_run_ids:
                raise CancelledExecutionError("cancelled execution cannot be registered")
            existing = self._handles.get(plan.run_id)
            if existing is not None:
                if existing == handle and self._plans.get(plan.run_id) == plan and self._views.get(plan.run_id) == view and self._snapshots.get(plan.run_id) == committed_snapshot:
                    return existing
                raise ExecutionRegistrationError("run is already registered differently")
            self._handles[plan.run_id] = handle
            self._plans[plan.run_id] = plan
            self._views[plan.run_id] = view
            self._snapshots[plan.run_id] = committed_snapshot
            self._applied_idempotency_keys[plan.run_id] = set(view.applied_idempotency_keys)
            return handle

    @staticmethod
    def _authorization(transition: HarnessTransition, view: HarnessSessionView) -> RunTransitionEvidence | WorkItemTransitionAuthorization | AttemptTransitionAuthorization:
        matching = [record for record in view.transition_authorities if (
            record.run_id == transition.run_id
            and record.trace_id == transition.trace_id
            and record.entity_kind == transition.entity_kind
            and record.entity_id == transition.entity_id
            and record.from_state == transition.from_state
            and record.to_state == transition.to_state
            and record.expected_state_revision == transition.expected_state_revision
            and record.plan_revision == transition.plan_revision
            and record.reason_code == transition.reason_code
            and record.idempotency_key == transition.idempotency_key
            and record.lease_epoch == transition.lease_epoch
            and record.fencing_token_digest == transition.fencing_token_digest
        )]
        if len(matching) != 1:
            raise InvalidCommittedTransitionError("exactly one committed authority record must identify the transition")
        record: TransitionAuthorityRecord = matching[0]
        common: dict[str, object] = {
            "run_id": record.run_id,
            "trace_id": record.trace_id,
            "entity_id": record.entity_id,
            "expected_state_revision": record.expected_state_revision,
            "plan_revision": record.plan_revision,
            "dependency_versions": record.dependency_versions,
        }
        if transition.entity_kind == "run":
            return RunTransitionEvidence(**common)
        leased = {
            **common,
            "reservation_id": record.reservation_id,
            "grant_id": record.grant_id,
            "lease_epoch": record.lease_epoch,
            "fencing_token_digest": record.fencing_token_digest,
        }
        if transition.entity_kind == "work_item":
            return WorkItemTransitionAuthorization(**leased)
        return AttemptTransitionAuthorization(**leased)

    @staticmethod
    def _validate_entity_ownership(plan: HarnessPlan, view: HarnessSessionView, transition: HarnessTransition) -> None:
        work_item_ids = {item.work_item_id for item in plan.work_items}
        if transition.entity_kind == "run":
            if transition.entity_id != plan.run_id:
                raise ExecutionIdentityError("run transition entity is foreign")
            return
        if transition.entity_kind == "work_item":
            if transition.entity_id not in work_item_ids:
                raise ExecutionIdentityError("work item is not owned by the plan")
            return
        owners = [owner for owner in view.attempt_work_item_owners if owner.attempt_id == transition.entity_id]
        if len(owners) != 1:
            raise ExecutionIdentityError("attempt has no unique committed owner")
        owner: AttemptWorkItemOwnershipRecord = owners[0]
        if owner.run_id != plan.run_id or owner.trace_id != plan.trace_id or owner.plan_revision != plan.revision or owner.work_item_id not in work_item_ids:
            raise ExecutionIdentityError("attempt ownership is foreign to the plan")

    def apply_committed_transition(self, handle: ExecutionHandle, transition: HarnessTransition, pre_view: HarnessSessionView, post_view: HarnessSessionView, receipt: CommittedTransitionReceipt) -> ExecutionHandle:
        handle = _fresh_handle(handle)
        transition = _fresh_transition(transition)
        with self._lock:
            if handle.run_id in self._cancelled_run_ids:
                raise CancelledExecutionError("cancelled execution cannot project transitions")
            current = self._handles.get(handle.run_id)
            if current is None or current != handle:
                raise ExecutionHandleMismatchError("execution handle is stale, forged, or unknown")
            applied = self._applied_idempotency_keys[handle.run_id]
            if transition.idempotency_key in applied:
                raise DuplicateExecutionTransitionError("idempotency key was already projected")
            if transition.run_id != handle.run_id or transition.trace_id != handle.trace_id:
                raise ExecutionIdentityError("transition identity does not match execution")
            if transition.plan_revision != handle.plan_revision:
                raise ExecutionPlanMismatchError("transition plan revision does not match")
            if transition.expected_state_revision != handle.state_revision:
                raise StaleExecutionTransitionError("transition revision is stale")

            plan = self._plans[handle.run_id]
            pre_view = _fresh_view(pre_view)
            post_view = _fresh_view(post_view)
            receipt = _fresh_receipt(receipt)
            _, pre_view, pre_snapshot = self._validated_snapshot(plan, pre_view, receipt.pre)
            _, post_view, post_snapshot = self._validated_snapshot(plan, post_view, receipt.post)
            receipt = cast(CommittedTransitionReceipt, self._verify_authority(receipt))
            if canonical_transition_digest(transition) != receipt.transition_digest:
                raise InvalidCommittedTransitionError("receipt identifies another transition")
            if self._views[handle.run_id] != pre_view or self._snapshots[handle.run_id] != pre_snapshot or pre_snapshot.state_revision != handle.state_revision:
                raise StaleExecutionTransitionError("receipt pre-state is not registered authority")
            self._validate_entity_ownership(plan, pre_view, transition)
            authorization = self._authorization(transition, pre_view)
            try:
                expected_post = self._state_machine.apply(transition, pre_view, authorization=authorization)
            except StateMachineError as error:
                raise InvalidCommittedTransitionError("transition is illegal for authoritative pre-state") from error
            if expected_post.model_dump(exclude={"last_event_hash"}) != post_view.model_dump(exclude={"last_event_hash"}):
                raise InvalidCommittedTransitionError("post fold differs from state-machine result")

            target = post_view.run_state.value if transition.entity_kind == "run" and post_view.run_state is not None else transition.to_state
            route = _ValidatedRoute(target, _ROUTE_CAPABILITY)
            try:
                projected = self._graph.invoke({"validated_route": route})
            except UncommittedTransitionError:
                raise
            except Exception as error:
                raise ExecutionProjectionError("LangGraph failed to project route") from error
            routed_state = projected.get("routed_state")
            if routed_state != target:
                raise ExecutionProjectionError("LangGraph projected inconsistent target")
            advanced = ExecutionHandle(
                run_id=handle.run_id,
                trace_id=handle.trace_id,
                plan_id=handle.plan_id,
                plan_revision=handle.plan_revision,
                state_revision=post_view.state_revision,
                routed_state=routed_state,
                cancelled=False,
            )
            self._handles[handle.run_id] = advanced
            self._views[handle.run_id] = post_view
            self._snapshots[handle.run_id] = post_snapshot
            applied.add(transition.idempotency_key)
            return advanced

    def resume(self, plan: HarnessPlan, folded_view: HarnessSessionView, committed_snapshot: CommittedExecutionSnapshot, *, disposable_checkpoint: object | None = None) -> ExecutionHandle:
        # Disposable checkpoints are intentionally unread and non-authoritative.
        plan, folded_view, committed_snapshot = self._validated_snapshot(plan, folded_view, committed_snapshot)
        handle = _handle_from_view(plan, folded_view)
        incoming_keys = set(folded_view.applied_idempotency_keys)
        with self._lock:
            if plan.run_id in self._cancelled_run_ids:
                raise CancelledExecutionError("cancelled execution cannot be resumed")
            existing_plan = self._plans.get(plan.run_id)
            if existing_plan is None:
                self._plans[plan.run_id] = plan
                self._views[plan.run_id] = folded_view
                self._snapshots[plan.run_id] = committed_snapshot
                self._handles[plan.run_id] = handle
                self._applied_idempotency_keys[plan.run_id] = incoming_keys
                return handle
            if existing_plan != plan:
                raise ExecutionPlanMismatchError("resume plan differs from registered plan")
            current = self._handles[plan.run_id]
            current_view = self._views[plan.run_id]
            current_snapshot = self._snapshots[plan.run_id]
            current_keys = self._applied_idempotency_keys[plan.run_id]
            if folded_view.state_revision < current_view.state_revision:
                raise StaleExecutionSnapshotError("resume snapshot is older than authority")
            if folded_view.state_revision == current_view.state_revision:
                if (
                    folded_view == current_view
                    and committed_snapshot == current_snapshot
                    and incoming_keys == current_keys
                ):
                    return current
                if not _is_same_revision_authority_extension(
                    current_view, folded_view
                ):
                    raise StaleExecutionSnapshotError(
                        "same-revision snapshot is not an append-only authority extension"
                    )
                self._views[plan.run_id] = folded_view
                self._snapshots[plan.run_id] = committed_snapshot
                return current
            if folded_view.sequence <= current_view.sequence or committed_snapshot.event_head_hash == current_snapshot.event_head_hash or not current_keys.issubset(incoming_keys):
                raise StaleExecutionSnapshotError("new snapshot is not a monotonic extension")
            self._handles[plan.run_id] = handle
            self._views[plan.run_id] = folded_view
            self._snapshots[plan.run_id] = committed_snapshot
            self._applied_idempotency_keys[plan.run_id] = incoming_keys
            return handle

    def cancel(self, run_id: str) -> None:
        if type(run_id) is not str:
            raise InvalidExecutionInputError("run identifier must be a strict string")
        normalized = run_id.strip()
        if not normalized or len(normalized) > 256:
            raise InvalidExecutionInputError("run identifier must be nonblank and at most 256 characters")
        with self._lock:
            current = self._handles.get(normalized)
            if current is None or normalized in self._cancelled_run_ids:
                return
            self._handles[normalized] = ExecutionHandle(
                run_id=current.run_id,
                trace_id=current.trace_id,
                plan_id=current.plan_id,
                plan_revision=current.plan_revision,
                state_revision=current.state_revision,
                routed_state=current.routed_state,
                cancelled=True,
            )
            self._cancelled_run_ids.add(normalized)
