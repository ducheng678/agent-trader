from __future__ import annotations

import hmac
from hashlib import sha256
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
    TraceEventResponse,
    TraceQueryResponse,
    WorkflowAcceptedResponse,
    WorkflowEventListResponse,
    WorkflowEventResponse,
    WorkflowStatusResponse,
    WorkflowSubmissionRequest,
)
from market_agent.backend.container import BackendContainer
from market_agent.backend.database import JobRecord
from market_agent.backend.errors import (
    AuthenticationError,
    AuthorizationError,
    BackendError,
    DependencyUnavailableError,
    NotFoundError,
    ValidationError,
)
from market_agent.backend.observability import current_request_id, request_context
from market_agent.workflow_tracing import TraceContext, TraceId
from market_agent.workflow_contracts import WorkflowRequest
from market_agent.workflow_execution_backend import ExecutionRegistrationError
from market_agent.workflow_harness import InvalidHarnessInputError, UnknownHarnessRunError
from market_agent.workflow_harness_contracts import HarnessSessionView, RunState

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


def get_request_trace(request: Request) -> TraceContext:
    value = getattr(request.state, "trace_context", None)
    if value is None:
        raise DependencyUnavailableError("request trace context is unavailable")
    return TraceContext.model_validate(value)


def _response_headers(request: Request) -> dict[str, str]:
    headers = {"X-Request-ID": _request_id(request)}
    trace = getattr(request.state, "trace_context", None)
    if trace is not None:
        trace = TraceContext.model_validate(trace)
        headers.update({"X-Trace-ID": trace.trace_id, "traceparent": trace.inject()["traceparent"]})
    return headers


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
        malformed_upstream = False
        upstream_headers = request.headers.getlist("traceparent")
        try:
            if len(upstream_headers) > 1:
                raise ValueError("duplicate traceparent")
            trace = TraceContext.new_request(upstream={"traceparent": upstream_headers[0]} if upstream_headers else None)
        except ValueError:
            trace = TraceContext.new_request()
            malformed_upstream = True
        request.state.trace_context = trace
        started_at = time.perf_counter()
        status_code = 500
        method = request.method if request.method in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TRACE", "CONNECT"} else "OTHER"
        with request_context(request_id):
            response = None
            try:
                resolved_container.observability.started(trace, request_id=request_id, malformed_upstream=malformed_upstream)
            except Exception:
                response = JSONResponse(status_code=503, content=jsonable_encoder(ErrorResponse(
                    error="dependency_unavailable", message="request observability is unavailable", request_id=request_id)))
            if response is None:
                try:
                    response = await call_next(request)
                except Exception:
                    resolved_container.metrics.increment("market_agent_http_requests_failed_total",
                        labels={"method": method, "path": _route_label(request)})
                    _logger.error("unhandled HTTP request error")
                    response = JSONResponse(status_code=500, content=jsonable_encoder(ErrorResponse(
                        error="internal_server_error", message="internal server error", request_id=request_id)))
            status_code = response.status_code
            labels = {"method": method, "path": _route_label(request)}
            latency = time.perf_counter() - started_at
            resolved_container.metrics.increment(
                "market_agent_http_requests_total", labels={**labels, "status_code": status_code})
            resolved_container.metrics.observe("market_agent_http_request_duration_seconds", latency, labels=labels)
            try:
                resolved_container.observability.completed(trace, request_id=request_id, method=method,
                    route=_route_label(request), status_code=status_code, latency_seconds=latency)
            except Exception:
                response.headers["X-Observability-Degraded"] = "1"
                resolved_container.metrics.increment(
                    "market_agent_observability_failures_total", labels={"phase": "request_completion"}
                )
                _logger.error("request completion observability unavailable")
        response.headers.update(_response_headers(request))
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

    def require_harness():
        kernel = getattr(resolved_container, "harness_kernel", None)
        if kernel is None:
            raise DependencyUnavailableError("Harness kernel is not configured")
        return kernel

    def require_harness_workflow() -> Any:
        application = getattr(resolved_container, "harness_application", None)
        if application is None:
            raise DependencyUnavailableError("Harness workflow application is not configured")
        return application

    def workflow_view(kernel: Any, run_id: str) -> HarnessSessionView:
        try:
            return kernel.snapshot(run_id)
        except UnknownHarnessRunError as error:
            raise NotFoundError("workflow run was not found") from error
        except InvalidHarnessInputError as error:
            raise ValidationError("workflow run identifier is invalid") from error

    def workflow_status(view: HarnessSessionView) -> WorkflowStatusResponse:
        return WorkflowStatusResponse(
            run_id=view.run_id,
            trace_id=view.trace_id,
            state=view.run_state.value if view.run_state is not None else None,
            sequence=view.sequence,
            state_revision=view.state_revision,
            plan_revision=view.plan_revision,
            reconciliation_required=bool(
                view.external_side_effect_unknown
                or view.run_state is RunState.WAITING_RECONCILIATION
            ),
        )

    @app.exception_handler(RequestValidationError)
    async def request_validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
        request_id = _request_id(request)
        payload = ErrorResponse(
            error="request_validation_error",
            message="request validation failed",
            request_id=request_id,
            details={"errors": _safe_validation_errors(exc.errors())},
        )
        return JSONResponse(status_code=422, content=jsonable_encoder(payload), headers=_response_headers(request))

    @app.exception_handler(BackendError)
    async def backend_error_handler(request: Request, exc: BackendError) -> JSONResponse:
        request_id = current_request_id() or _request_id(request)
        payload = ErrorResponse(
            error=exc.error_code,
            message=exc.message,
            request_id=request_id,
            details=exc.details,
        )
        return JSONResponse(status_code=exc.status_code, content=jsonable_encoder(payload), headers=_response_headers(request))

    @app.exception_handler(Exception)
    async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
        request_id = _request_id(request)
        with request_context(request_id):
            _logger.error("unhandled HTTP request error")
        payload = ErrorResponse(
            error="internal_server_error",
            message="internal server error",
            request_id=request_id,
        )
        return JSONResponse(status_code=500, content=jsonable_encoder(payload), headers=_response_headers(request))

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
        request: Request,
        task_name: str,
        body: TaskSubmissionRequest,
        idempotency_key_header: str | None = Header(default=None, alias="Idempotency-Key"),
        _: None = Depends(require_api_token),
    ) -> TaskAcceptedResponse:
        idempotency_key = _resolve_idempotency_key(body.idempotency_key, idempotency_key_header)
        task_payload = dict(body.payload)
        if task_name == "generate_playbook":
            if not resolved_container.settings.legacy_playbook_api_enabled:
                raise DependencyUnavailableError(
                    "legacy generate_playbook API is disabled; use /v1/workflows"
                )
            task_payload["trace_id"] = get_request_trace(request).trace_id
        submission = resolved_container.task_queue.submit(
            task_name,
            task_payload,
            idempotency_key=idempotency_key,
            request_id=get_request_trace(request).trace_id,
        )
        return TaskAcceptedResponse(
            job_id=submission.job.job_id,
            status=submission.job.status,
            reused=submission.reused,
        )

    @app.post("/v1/workflows", response_model=WorkflowAcceptedResponse, status_code=202, tags=["workflows"])
    def create_workflow(
        request: Request,
        body: WorkflowSubmissionRequest,
        idempotency_key_header: str | None = Header(default=None, alias="Idempotency-Key"),
        _: None = Depends(require_api_token),
    ) -> WorkflowAcceptedResponse:
        kernel = require_harness()
        require_harness_workflow()
        key = _resolve_idempotency_key(body.idempotency_key, idempotency_key_header)
        trace_id = get_request_trace(request).trace_id
        # A missing key deliberately means a distinct user request.  Never use
        # the request trace as implicit idempotency: upstream retries normally
        # receive a new trace identifier.
        run_id = (
            "wf-" + sha256(f"{resolved_container.settings.tenant_id}:{key}".encode("utf-8")).hexdigest()[:32]
            if key is not None
            else "wf-" + uuid.uuid4().hex
        )
        payload = body.payload
        if payload.tenant_id != resolved_container.settings.tenant_id:
            raise AuthorizationError("workflow tenant does not match the configured host scope")
        workflow_request = WorkflowRequest(
            workflow_id=run_id, trace_id=trace_id, user_query=payload.user_query,
            event_tape=tuple(payload.event_tape), trigger_reason=payload.trigger_reason,
            trigger_event=payload.trigger_event, recent_events=tuple(payload.recent_events or ()),
            trade_symbol_context=payload.trade_symbol_context, active_symbol=payload.active_symbol,
            has_live_position=payload.has_live_position,
            prefetched_passive_event_judge=payload.prefetched_passive_event_judge,
        )
        try:
            handle = kernel.create(workflow_request)
            status = handle.run_state.value
        except ExecutionRegistrationError as error:
            if str(error) != "run already exists":
                raise
            view = workflow_view(kernel, run_id)
            trace_id = view.trace_id
            status = view.run_state.value if view.run_state is not None else "created"
        else:
            try:
                resolved_container.task_queue.submit(
                    "execute_harness_workflow",
                    workflow_request.model_dump(mode="json"),
                    idempotency_key=run_id,
                    request_id=trace_id,
                )
            except Exception:
                # Do not leave an admitted run permanently stranded when the
                # durable dispatcher cannot accept it.
                kernel.cancel(run_id, "workflow_dispatch_unavailable")
                raise
        return WorkflowAcceptedResponse(run_id=run_id, trace_id=trace_id, status=status,
                                        status_url=f"/v1/workflows/{run_id}")

    @app.get("/v1/workflows/{run_id}", response_model=WorkflowStatusResponse, tags=["workflows"])
    def get_workflow(run_id: str, _: None = Depends(require_api_token)) -> WorkflowStatusResponse:
        return workflow_status(workflow_view(require_harness(), run_id))

    @app.post("/v1/workflows/{run_id}:cancel", response_model=WorkflowStatusResponse, tags=["workflows"])
    def cancel_workflow(run_id: str, _: None = Depends(require_api_token)) -> WorkflowStatusResponse:
        kernel = require_harness()
        kernel.cancel(run_id, "api_cancellation")
        return workflow_status(workflow_view(kernel, run_id))

    @app.get("/v1/workflows/{run_id}/events", response_model=WorkflowEventListResponse, tags=["workflows"])
    def get_workflow_events(run_id: str, after_sequence: int = Query(default=0, ge=0),
                            limit: int = Query(default=100, ge=1, le=500),
                            _: None = Depends(require_api_token)) -> WorkflowEventListResponse:
        kernel = require_harness()
        # Establish existence through the authoritative projection first; a
        # direct empty event-store read cannot distinguish an unknown run.
        workflow_view(kernel, run_id)
        events = kernel.event_store.load(run_id)
        selected = [event for event in events if event.sequence > after_sequence]
        page = selected[:limit]
        return WorkflowEventListResponse(items=[WorkflowEventResponse(
            sequence=event.sequence, event_type=event.event_type,
            state_revision=event.state_revision, payload=dict(event.payload),
        ) for event in page], next_cursor=page[-1].sequence if page else after_sequence,
            has_more=len(selected) > len(page))

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
        return PlainTextResponse(resolved_container.metrics.render_prometheus() + resolved_container.observability.metrics.render_prometheus(),
                                 media_type="text/plain; version=0.0.4")

    @app.get("/v1/traces/{trace_id}", response_model=TraceQueryResponse, tags=["operations"])
    def get_trace(trace_id: TraceId, after_sequence: int = Query(default=0, ge=0),
                  limit: int = Query(default=100, ge=1, le=500),
                  _: None = Depends(require_api_token)) -> TraceQueryResponse:
        try:
            page = resolved_container.observability.sink.query(trace_id, after_sequence=after_sequence, limit=limit)
        except Exception:
            raise DependencyUnavailableError("trace query storage is unavailable") from None
        return TraceQueryResponse(trace_id=trace_id,
            items=[TraceEventResponse(sequence=item.sequence, event=item.event) for item in page.items],
            next_cursor=page.next_cursor, oldest_available_sequence=page.oldest_available_sequence,
            has_more=page.has_more, truncated=page.truncated)

    return app
