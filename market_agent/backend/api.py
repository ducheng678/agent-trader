from __future__ import annotations

import hmac
import logging
import re
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, Query, Request, Response
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse

from market_agent.backend.api_contracts import (
    ErrorResponse,
    HealthResponse,
    TaskAcceptedResponse,
    TaskEventListResponse,
    TaskEventResponse,
    TaskStatusResponse,
    TaskSubmissionRequest,
)
from market_agent.backend.container import BackendContainer
from market_agent.backend.database import JobRecord
from market_agent.backend.errors import AuthenticationError, BackendError, ValidationError
from market_agent.backend.observability import current_request_id, request_context

_logger = logging.getLogger("market_agent.backend.api")
_request_id_pattern = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


def _task_response(job: JobRecord) -> TaskStatusResponse:
    return TaskStatusResponse(
        job_id=job.job_id,
        task_name=job.task_name,
        status=job.status,
        result=job.result,
        error=job.error,
        attempt_count=job.attempt_count,
        max_attempts=job.max_attempts,
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


def _normalize_request_id(value: str | None) -> str:
    candidate = str(value or "").strip()
    return candidate if _request_id_pattern.fullmatch(candidate) else uuid.uuid4().hex


def _request_id(request: Request) -> str:
    stored = str(getattr(request.state, "request_id", "") or "")
    return stored or _normalize_request_id(request.headers.get("X-Request-ID"))


def _route_label(request: Request) -> str:
    route = request.scope.get("route")
    return str(getattr(route, "path", "") or "__unmatched__")


def _normalize_idempotency_key(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        raise ValidationError("idempotency key cannot be blank")
    if len(normalized) > 256:
        raise ValidationError("idempotency key cannot exceed 256 characters")
    return normalized


def _resolve_idempotency_key(body_value: str | None, header_value: str | None) -> str | None:
    body_key = _normalize_idempotency_key(body_value)
    header_key = _normalize_idempotency_key(header_value)
    if body_key is not None and header_key is not None and body_key != header_key:
        raise ValidationError("idempotency key header and body must match")
    return body_key or header_key


def _safe_validation_errors(errors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{key: value for key, value in error.items() if key not in {"input", "url"}} for error in errors]


def create_app(container: BackendContainer | None = None) -> FastAPI:
    owns_container = container is None
    resolved_container = container or BackendContainer.create()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        try:
            yield
        finally:
            if owns_container:
                resolved_container.shutdown()

    app = FastAPI(
        title="Market Agent Backend",
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
    )
    app.state.container = resolved_container

    @app.middleware("http")
    async def attach_request_context(request: Request, call_next: Any):
        request_id = _normalize_request_id(request.headers.get("X-Request-ID"))
        request.state.request_id = request_id
        started_at = time.perf_counter()
        status_code = 500
        with request_context(request_id):
            try:
                response = await call_next(request)
                status_code = response.status_code
            except Exception:
                resolved_container.metrics.increment(
                    "market_agent_http_requests_failed_total",
                    labels={"method": request.method, "path": _route_label(request)},
                )
                raise
            finally:
                labels = {"method": request.method, "path": _route_label(request)}
                resolved_container.metrics.increment(
                    "market_agent_http_requests_total",
                    labels={**labels, "status_code": status_code},
                )
                resolved_container.metrics.observe(
                    "market_agent_http_request_duration_seconds",
                    time.perf_counter() - started_at,
                    labels=labels,
                )
        response.headers["X-Request-ID"] = request_id
        return response

    def require_api_token(authorization: str | None = Header(default=None)) -> None:
        expected_token = resolved_container.settings.api_token
        if not expected_token:
            return
        scheme, separator, supplied_token = str(authorization or "").partition(" ")
        if not separator or scheme.lower() != "bearer" or not supplied_token.strip():
            raise AuthenticationError("valid bearer token required")
        if not hmac.compare_digest(supplied_token.strip(), expected_token):
            raise AuthenticationError("valid bearer token required")

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = _request_id(request)
        payload = ErrorResponse(
            error="request_validation_error",
            message="request validation failed",
            request_id=request_id,
            details={"errors": _safe_validation_errors(exc.errors())},
        )
        return JSONResponse(status_code=422, content=jsonable_encoder(payload), headers={"X-Request-ID": request_id})

    @app.exception_handler(BackendError)
    async def backend_error_handler(request: Request, exc: BackendError) -> JSONResponse:
        request_id = current_request_id() or _request_id(request)
        payload = ErrorResponse(
            error=exc.error_code,
            message=exc.message,
            request_id=request_id,
            details=exc.details,
        )
        return JSONResponse(status_code=exc.status_code, content=jsonable_encoder(payload), headers={"X-Request-ID": request_id})

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = _request_id(request)
        with request_context(request_id):
            _logger.error(
                "unhandled HTTP request error",
                exc_info=(type(exc), exc, exc.__traceback__),
            )
        payload = ErrorResponse(
            error="internal_server_error",
            message="internal server error",
            request_id=request_id,
        )
        return JSONResponse(status_code=500, content=jsonable_encoder(payload), headers={"X-Request-ID": request_id})

    @app.get("/health/live", response_model=HealthResponse, tags=["health"])
    def live() -> HealthResponse:
        return HealthResponse(status="ok", components={"service": "ok"})

    @app.get("/health/ready", response_model=HealthResponse, tags=["health"])
    def ready(response: Response) -> HealthResponse:
        components = resolved_container.readiness()
        status = "ok" if all(value == "ok" for value in components.values()) else "degraded"
        if status != "ok":
            response.status_code = 503
        return HealthResponse(status=status, components=components)

    @app.post(
        "/v1/tasks/{task_name}",
        response_model=TaskAcceptedResponse,
        status_code=202,
        tags=["tasks"],
    )
    def submit_task(
        task_name: str,
        body: TaskSubmissionRequest,
        idempotency_key_header: str | None = Header(default=None, alias="Idempotency-Key"),
        _: None = Depends(require_api_token),
    ) -> TaskAcceptedResponse:
        idempotency_key = _resolve_idempotency_key(body.idempotency_key, idempotency_key_header)
        submission = resolved_container.task_queue.submit(
            task_name,
            body.payload,
            idempotency_key=idempotency_key,
            request_id=current_request_id(),
        )
        return TaskAcceptedResponse(
            job_id=submission.job.job_id,
            status=submission.job.status,
            reused=submission.reused,
        )

    @app.get("/v1/tasks/{job_id}", response_model=TaskStatusResponse, tags=["tasks"])
    def get_task(job_id: str, _: None = Depends(require_api_token)) -> TaskStatusResponse:
        return _task_response(resolved_container.task_queue.get_job(job_id))

    @app.get("/v1/tasks/{job_id}/events", response_model=TaskEventListResponse, tags=["tasks"])
    def get_task_events(
        job_id: str,
        limit: int = Query(default=100, ge=1, le=1000),
        _: None = Depends(require_api_token),
    ) -> TaskEventListResponse:
        events = resolved_container.task_queue.list_events(job_id, limit=limit)
        return TaskEventListResponse(
            items=[
                TaskEventResponse(
                    event_id=event.event_id,
                    job_id=event.job_id,
                    event_type=event.event_type,
                    payload=event.payload,
                    created_at=event.created_at,
                )
                for event in events
            ]
        )

    @app.get("/metrics", response_class=PlainTextResponse, tags=["operations"])
    def metrics(_: None = Depends(require_api_token)) -> PlainTextResponse:
        resolved_container.readiness()
        return PlainTextResponse(resolved_container.metrics.render_prometheus(), media_type="text/plain; version=0.0.4")

    return app
