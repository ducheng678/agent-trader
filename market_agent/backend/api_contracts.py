from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StringConstraints
from market_agent.workflow_structured_logging import StructuredEvent
from market_agent.workflow_tracing import TraceId

NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
IdempotencyKey = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskSubmissionRequest(ApiModel):
    payload: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: IdempotencyKey | None = None


class GeneratePlaybookPayload(ApiModel):
    trace_id: str | None = None
    tenant_id: str = "default"
    user_query: NonEmptyText
    event_tape: list[dict[str, Any]]
    trigger_reason: NonEmptyText
    trigger_event: dict[str, Any] | None = None
    recent_events: list[dict[str, Any]] | None = None
    trade_symbol_context: dict[str, Any] | None = None
    active_symbol: str | None = None
    has_live_position: StrictBool = False
    prefetched_passive_event_judge: dict[str, Any] | None = None


class WorkflowSubmissionRequest(ApiModel):
    payload: GeneratePlaybookPayload
    idempotency_key: IdempotencyKey | None = None


class WorkflowAcceptedResponse(ApiModel):
    run_id: str
    trace_id: TraceId
    status: str
    status_url: str
    job_id: str
    job_status_url: str


class WorkflowStatusResponse(ApiModel):
    run_id: str
    trace_id: TraceId
    state: str | None
    sequence: int
    state_revision: int
    plan_revision: int
    reconciliation_required: bool


class WorkflowEventResponse(ApiModel):
    sequence: int
    event_type: str
    state_revision: int
    payload: dict[str, Any]


class WorkflowEventListResponse(ApiModel):
    items: list[WorkflowEventResponse]
    next_cursor: int
    has_more: bool


class PromptReleaseResponse(ApiModel):
    release_id: str
    release_digest: str
    output_schema_hash: str
    manifest_hash: str


class PromptReleaseActivationResponse(PromptReleaseResponse):
    action: str
    previous_release_id: str | None


class TaskAcceptedResponse(ApiModel):
    job_id: str
    status: str
    reused: bool


class TaskStatusResponse(ApiModel):
    job_id: str
    task_name: str
    status: str
    result: Any = None
    error: dict[str, Any] | None = None
    attempt_count: int
    max_attempts: int
    created_at: str
    updated_at: str


class TaskEventResponse(ApiModel):
    event_id: int
    job_id: str
    event_type: str
    payload: dict[str, Any]
    created_at: str


class TaskEventListResponse(ApiModel):
    items: list[TaskEventResponse]


class HealthResponse(ApiModel):
    status: str
    components: dict[str, str]


class ErrorResponse(ApiModel):
    error: str
    message: str
    request_id: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class TraceEventResponse(ApiModel):
    sequence: Annotated[int, Field(ge=1)]
    event: StructuredEvent


class TraceQueryResponse(ApiModel):
    trace_id: TraceId
    items: list[TraceEventResponse]
    next_cursor: Annotated[int, Field(ge=0)]
    oldest_available_sequence: Annotated[int, Field(ge=0)]
    has_more: bool
    truncated: bool
