from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
from threading import RLock

from pydantic import TypeAdapter

from market_agent.workflow_metrics import MetricLabels, WorkflowMetrics
from market_agent.workflow_structured_logging import StructuredEvent, StructuredLogger, identity_digest, summarize_payload
from market_agent.workflow_tracing import TraceContext, TraceId


@dataclass(frozen=True, slots=True)
class StoredTraceEvent:
    sequence: int
    event: StructuredEvent


@dataclass(frozen=True, slots=True)
class TracePage:
    items: tuple[StoredTraceEvent, ...]
    next_cursor: int
    oldest_available_sequence: int
    has_more: bool
    truncated: bool


class BoundedTraceSink:
    def __init__(self, *, capacity: int = 10000, maximum_query: int = 100):
        if type(capacity) is not int or not 1 <= capacity <= 100000:
            raise ValueError("trace event capacity must be between 1 and 100000")
        if type(maximum_query) is not int or not 1 <= maximum_query <= min(500, capacity):
            raise ValueError("trace query limit exceeds its bounded capacity")
        self._capacity, self._maximum_query = capacity, maximum_query
        self._events: deque[StoredTraceEvent] = deque(maxlen=capacity)
        self._sequence = 0
        self._lock = RLock()

    def record(self, event: StructuredEvent) -> None:
        event = StructuredEvent.model_validate(event)
        with self._lock:
            self._sequence += 1
            self._events.append(StoredTraceEvent(self._sequence, event))

    def query(self, trace_id: str, *, after_sequence: int = 0, limit: int = 100) -> TracePage:
        trace_id = TypeAdapter(TraceId).validate_python(trace_id, strict=True)
        if type(after_sequence) is not int or after_sequence < 0 or type(limit) is not int or limit < 1:
            raise ValueError("trace query cursor and limit are invalid")
        with self._lock:
            oldest = self._events[0].sequence if self._events else 0
            matches = tuple(item for item in self._events if item.sequence > after_sequence and item.event.trace.trace_id == trace_id)
            items = matches[:min(limit, self._maximum_query)]
            return TracePage(items=items, next_cursor=items[-1].sequence if items else after_sequence,
                oldest_available_sequence=oldest, has_more=len(matches) > len(items),
                truncated=oldest > 1 and after_sequence < oldest - 1)


class BackendObservability:
    def __init__(self, *, sink: BoundedTraceSink, metrics: WorkflowMetrics, logger: StructuredLogger):
        self.sink, self.metrics, self.logger = sink, metrics, logger

    @classmethod
    def create(cls, *, event_capacity: int = 10000, maximum_query: int = 100,
               maximum_metric_series: int = 2048) -> BackendObservability:
        logger = logging.getLogger("market_agent.backend.trace")
        return cls(sink=BoundedTraceSink(capacity=event_capacity, maximum_query=maximum_query),
                   metrics=WorkflowMetrics(maximum_series=maximum_metric_series),
                   logger=StructuredLogger(lambda line: logger.info(line.rstrip("\n"))))

    def record(self, event: StructuredEvent) -> None:
        self.sink.record(event)
        self.logger.emit(event)

    def query(self, trace_id: str, *, after_sequence: int = 0, limit: int = 100) -> TracePage:
        """Unified trace repository entrypoint used by API and operators."""

        return self.sink.query(trace_id, after_sequence=after_sequence, limit=limit)

    def record_component(
        self,
        trace: TraceContext,
        *,
        event: str,
        status: str,
        component: str,
        workflow_id: str | None = None,
        task_id: str | None = None,
        **fields,
    ) -> None:
        """Record a bounded structured event for any workflow component.

        Component labels stay in the event body, while metrics remain
        low-cardinality.  The supplied TraceContext is validated before the
        event is accepted, so cross-request records cannot enter the stream.
        """

        actor = component if component in {"queue", "coordinator", "specialist", "reflection", "driver", "cache", "memory", "tool", "service", "ingress"} else "service"
        self.record(StructuredEvent.create(
            TraceContext.model_validate(trace), event=event, actor=actor, status=status,
            workflow_id=workflow_id, task_id=task_id, **fields,
        ))

    def started(self, trace: TraceContext, *, request_id: str, malformed_upstream: bool = False) -> None:
        self.record(StructuredEvent.create(trace, event="request_started", actor="ingress", status="started",
                    workflow_id=request_id, reason="invalid_input" if malformed_upstream else "none"))

    def completed(self, trace: TraceContext, *, request_id: str, method: str, route: str,
                  status_code: int, latency_seconds: float) -> None:
        failed = status_code >= 400
        self.record(StructuredEvent.create(trace, event="request_failed" if failed else "request_completed",
            actor="ingress", status="failed" if failed else "succeeded", workflow_id=request_id,
            subject_hash=identity_digest(method + ":" + route), latency_seconds=latency_seconds,
            payload=summarize_payload({"method": method, "route": route, "status_code": status_code}),
            reason="provider_error" if status_code >= 500 else "invalid_input" if failed else "none"))
        self.metrics.record_request(success=not failed, latency_seconds=latency_seconds,
                                    labels=MetricLabels(component="request"), trace=trace)
