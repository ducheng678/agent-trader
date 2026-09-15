from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
from threading import RLock
import time

from market_agent.workflow_capabilities import CapabilityAuthorization
from market_agent.workflow_metrics import MetricLabels, WorkflowMetrics
from market_agent.workflow_structured_logging import StructuredEvent, StructuredLogger, identity_digest, summarize_payload
from market_agent.workflow_tracing import TraceContext


@dataclass(frozen=True, slots=True)
class ToolSpan:
    trace: TraceContext
    tool_hash: str
    workflow_id: str | None
    task_id: str
    started_at: float
    argument_hash: str
    schema_hash: str


class ToolObserver:
    def __init__(self, *, logger: StructuredLogger, metrics: WorkflowMetrics,
                 audit_sink: Callable[[StructuredEvent], None], clock: Callable[[], float] = time.monotonic,
                 authorization_clock: Callable[[], float] = time.time):
        self._logger, self._metrics, self._audit = logger, metrics, audit_sink
        self._clock, self._authorization_clock = clock, authorization_clock
        self._active: dict[str, ToolSpan] = {}
        self._lock = RLock()

    def _now(self) -> float:
        value = self._clock()
        if type(value) not in (float, int) or not math.isfinite(value):
            raise ValueError("tool observation clock must be finite")
        return float(value)

    def start(self, trace: TraceContext, *, tool: str, arguments: object, schema_hash: str,
              authorization: CapabilityAuthorization, workflow_id: str | None = None) -> ToolSpan:
        trace = TraceContext.model_validate(trace)
        if not isinstance(authorization, CapabilityAuthorization) or authorization.kind != "tool" or authorization.resource != tool:
            raise PermissionError("tool observation requires matching host authorization")
        trace.assert_same_trace(authorization.scope.trace_id)
        now = self._authorization_clock()
        if not math.isfinite(now) or not authorization.authorized_at <= now < authorization.expires_at:
            raise PermissionError("tool authorization is not current")
        child = trace.child()
        summary = summarize_payload(arguments)
        started = self._now()
        handle = ToolSpan(child, identity_digest(tool), workflow_id, authorization.scope.task_id,
                          started, summary.content_hash, schema_hash)
        event = StructuredEvent.create(child, event="tool_started", actor="tool", status="started",
            workflow_id=workflow_id, task_id=handle.task_id, subject_hash=handle.tool_hash,
            output_schema_hash=schema_hash, payload=summary)
        self._audit(event)
        self._logger.emit(event)
        with self._lock:
            self._active[child.span_id] = handle
        return handle

    def end(self, handle: ToolSpan, *, trace: TraceContext, result: object = None, failed: bool = False) -> StructuredEvent:
        trace = TraceContext.model_validate(trace)
        handle.trace.assert_same_trace(trace)
        with self._lock:
            if self._active.get(handle.trace.span_id) is not handle:
                raise ValueError("tool span is unknown or already completed")
            elapsed = self._now() - handle.started_at
            if elapsed < 0:
                raise ValueError("tool observation clock moved backwards")
            summary = summarize_payload(None if failed else result)
            event = StructuredEvent.create(handle.trace, event="tool_failed" if failed else "tool_completed", actor="tool",
                status="failed" if failed else "succeeded", workflow_id=handle.workflow_id, task_id=handle.task_id,
                subject_hash=handle.tool_hash, output_schema_hash=handle.schema_hash, payload=summary,
                latency_seconds=elapsed, reason="provider_error" if failed else "none")
            self._audit(event)
            self._logger.emit(event)
            labels = MetricLabels(component="tool", outcome="failure" if failed else "success")
            self._metrics.increment("workflow_tool_calls_total", labels=labels)
            self._metrics.observe("workflow_tool_duration_seconds", elapsed, labels, trace=handle.trace)
            del self._active[handle.trace.span_id]
            return event

    def call(self, trace: TraceContext, *, tool: str, arguments: object, schema_hash: str,
             authorize: Callable[[], CapabilityAuthorization], invoke: Callable[[], object],
             workflow_id: str | None = None) -> object:
        handle = self.start(trace, tool=tool, arguments=arguments, schema_hash=schema_hash,
                            authorization=authorize(), workflow_id=workflow_id)
        try:
            result = invoke()
        except Exception:
            self.end(handle, trace=trace, failed=True)
            raise
        self.end(handle, trace=trace, result=result)
        return result


ToolObservability = ToolObserver
