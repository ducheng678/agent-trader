from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ApiModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TaskSubmissionRequest(ApiModel):
    payload: dict[str, Any] = Field(default_factory=dict)
    idempotency_key: str | None = Field(default=None, min_length=1, max_length=256)


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
