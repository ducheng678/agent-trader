from __future__ import annotations

import hmac
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, Query, Request
from fastapi.encoders import jsonable_encoder
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
from market_agent.backend.errors import AuthenticationError, BackendError
from market_agent.backend.observability import current_request_id, request_context


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
        request_id = str(request.headers.get("X-Request-ID") or uuid.uuid4().hex)
        started_at = time.perf_counter()
        with request_context(request_id):
            try:
                response = await call_next(request)
            except Exception:
                resolved_container.metrics.increment(
                    "market_agent_http_requests_failed_total",
                    labels={"method": request.method, "path": request.url.path},
                )
                raise
        response.headers["X-Request-ID"] = request_id
        resolved_container.metrics.increment(
            "market_agent_http_requests_total",
            labels={"method": request.method, "path": request.url.path, "status_code": response.status_code},
        )
        resolved_container.metrics.observe(
            "market_agent_http_request_duration_seconds",
            time.perf_counter() - started_at,
            labels={"method": request.method, "path": request.url.path},
        )
        return response

    async def require_api_token(authorization: str | None = Header(default=None)) -> None:
        expected_token = resolved_container.settings.api_token
        if not expected_token:
            return
        prefix = "Bearer "
        supplied_token = authorization[len(prefix):].strip() if authorization and authorization.startswith(prefix) else ""
        if not supplied_token or not hmac.compare_digest(supplied_token, expected_token):
            raise AuthenticationError("valid bearer token required")

    @app.exception_handler(BackendError)
    async def backend_error_handler(request: Request, exc: BackendError) -> JSONResponse:
        request_id = current_request_id() or str(request.headers.get("X-Request-ID") or "")
        payload = ErrorResponse(
            error=exc.error_code,
            message=exc.message,
            request_id=request_id or None,
            details=exc.details,
        )
        return JSONResponse(status_code=exc.status_code, content=jsonable_encoder(payload), headers={"X-Request-ID": request_id})

    @app.get("/health/live", response_model=HealthResponse, tags=["health"])
    async def live() -> HealthResponse:
        return HealthResponse(status="ok", components={"service": "ok"})

    @app.get("/health/ready", response_model=HealthResponse, tags=["health"])
    async def ready() -> HealthResponse:
        components = resolved_container.readiness()
        status = "ok" if all(value == "ok" for value in components.values()) else "degraded"
        return HealthResponse(status=status, components=components)

    @app.post(
        "/v1/tasks/{task_name}",
        response_model=TaskAcceptedResponse,
        status_code=202,
        tags=["tasks"],
    )
    async def submit_task(
        task_name: str,
        body: TaskSubmissionRequest,
        idempotency_key_header: str | None = Header(default=None, alias="Idempotency-Key"),
        _: None = Depends(require_api_token),
    ) -> TaskAcceptedResponse:
        idempotency_key = body.idempotency_key or idempotency_key_header
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
    async def get_task(job_id: str, _: None = Depends(require_api_token)) -> TaskStatusResponse:
        return _task_response(resolved_container.task_queue.get_job(job_id))

    @app.get("/v1/tasks/{job_id}/events", response_model=TaskEventListResponse, tags=["tasks"])
    async def get_task_events(
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
    async def metrics(_: None = Depends(require_api_token)) -> PlainTextResponse:
        resolved_container.readiness()
        return PlainTextResponse(resolved_container.metrics.render_prometheus(), media_type="text/plain; version=0.0.4")

    return app
