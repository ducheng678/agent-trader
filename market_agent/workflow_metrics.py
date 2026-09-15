from __future__ import annotations

from dataclasses import dataclass
import json
import math
from threading import RLock
from typing import Literal

from market_agent.workflow_agent_contracts import StrictModel
from market_agent.workflow_tracing import TraceContext


CounterName = Literal["workflow_requests_total", "workflow_success_total", "workflow_failure_total", "workflow_unknown_total", "workflow_input_tokens_total", "workflow_output_tokens_total", "workflow_cached_tokens_total", "workflow_cost_usd_total", "workflow_tool_calls_total"]
HistogramName = Literal["workflow_request_duration_seconds", "workflow_agent_duration_seconds", "workflow_tool_duration_seconds", "workflow_request_tokens", "workflow_request_cost_usd"]
_COUNTERS = frozenset(CounterName.__args__)
_BUCKETS = {
    "workflow_request_duration_seconds": (0.01, 0.1, 1.0, 5.0, 30.0, 130.0, 300.0),
    "workflow_agent_duration_seconds": (0.01, 0.1, 1.0, 5.0, 30.0, 60.0, 300.0),
    "workflow_tool_duration_seconds": (0.001, 0.01, 0.1, 1.0, 5.0, 30.0, 300.0),
    "workflow_request_tokens": (100.0, 500.0, 1000.0, 5000.0, 20000.0, 100000.0),
    "workflow_request_cost_usd": (0.001, 0.01, 0.05, 0.1, 0.3, 0.75, 10.0),
}


class MetricLabels(StrictModel):
    mode: Literal["active", "passive", "informational", "unknown"] = "unknown"
    component: Literal["request", "coordinator", "specialist", "reflection", "driver", "tool", "cache", "memory"] = "request"
    outcome: Literal["started", "success", "failure", "degraded", "unknown", "rejected"] = "started"
    model: Literal["none", "luna", "terra", "sol", "local"] = "none"


@dataclass(frozen=True, slots=True)
class CounterSnapshot:
    name: str
    labels: tuple[tuple[str, str], ...]
    value: float


@dataclass(frozen=True, slots=True)
class HistogramSnapshot:
    name: str
    labels: tuple[tuple[str, str], ...]
    count: int
    total: float
    buckets: tuple[tuple[float, int], ...]
    exemplar_trace_id: str | None


def _number(value: float) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError("metric values must be finite and nonnegative")
    return float(value)


class WorkflowMetrics:
    def __init__(self, *, maximum_series: int = 2048):
        if type(maximum_series) is not int or maximum_series < 1:
            raise ValueError("metric series limit must be positive")
        self._maximum_series = maximum_series
        self._lock = RLock()
        self._counters = {}
        self._histograms = {}

    def _key(self, name: str, labels: MetricLabels | None):
        labels = MetricLabels.model_validate(labels or MetricLabels())
        return name, tuple(sorted(labels.model_dump(exclude={"schema_version"}).items()))

    def _admit(self, key):
        if key not in self._counters and key not in self._histograms and len(self._counters) + len(self._histograms) >= self._maximum_series:
            raise ValueError("metric series limit reached")

    def increment(self, name: CounterName, value: float = 1.0, labels: MetricLabels | None = None) -> None:
        if name not in _COUNTERS:
            raise ValueError("metric counter is not registered")
        value, key = _number(value), self._key(name, labels)
        with self._lock:
            self._admit(key)
            total = _number(self._counters.get(key, 0.0) + value)
            self._counters[key] = total

    def observe(self, name: HistogramName, value: float, labels: MetricLabels | None = None,
                *, trace: TraceContext | None = None) -> None:
        if name not in _BUCKETS:
            raise ValueError("metric histogram is not registered")
        value, key = _number(value), self._key(name, labels)
        trace = TraceContext.model_validate(trace) if trace is not None else None
        with self._lock:
            self._admit(key)
            old = self._histograms.get(key, HistogramSnapshot(name, key[1], 0, 0.0,
                tuple((bound, 0) for bound in _BUCKETS[name]), None))
            self._histograms[key] = HistogramSnapshot(name, key[1], old.count + 1,
                _number(old.total + value), tuple((bound, count + int(value <= bound)) for bound, count in old.buckets),
                trace.trace_id if trace else old.exemplar_trace_id)

    def record_request(self, *, success: bool, latency_seconds: float, labels: MetricLabels | None = None,
                       trace: TraceContext | None = None, unknown: bool = False) -> None:
        latency_seconds = _number(latency_seconds)
        labels = MetricLabels.model_validate(labels or MetricLabels())
        outcome = "unknown" if unknown else "success" if success else "failure"
        labels = labels.model_copy(update={"outcome": outcome})
        with self._lock:
            self.increment("workflow_requests_total", labels=labels)
            self.increment("workflow_unknown_total" if unknown else "workflow_success_total" if success else "workflow_failure_total", labels=labels)
            self.observe("workflow_request_duration_seconds", latency_seconds, labels, trace=trace)

    def record_usage(self, *, input_tokens: int, output_tokens: int, cached_tokens: int = 0,
                     cost_usd: float, labels: MetricLabels | None = None) -> None:
        if any(type(value) is not int or value < 0 for value in (input_tokens, output_tokens, cached_tokens)) or cached_tokens > input_tokens:
            raise ValueError("token counts must be nonnegative integers with cached tokens within input tokens")
        cost_usd = _number(cost_usd)
        labels = MetricLabels.model_validate(labels or MetricLabels(component="driver"))
        with self._lock:
            for name, value in (("workflow_input_tokens_total", input_tokens), ("workflow_output_tokens_total", output_tokens),
                                ("workflow_cached_tokens_total", cached_tokens), ("workflow_cost_usd_total", cost_usd)):
                self.increment(name, value, labels)

    def snapshot(self) -> tuple[tuple[CounterSnapshot, ...], tuple[HistogramSnapshot, ...]]:
        with self._lock:
            return (tuple(CounterSnapshot(name, labels, value) for (name, labels), value in sorted(self._counters.items())),
                    tuple(value for _, value in sorted(self._histograms.items())))

    def render_prometheus(self) -> str:
        counters, histograms = self.snapshot()
        def tags(labels):
            return "{" + ",".join(key + "=" + json.dumps(value) for key, value in labels) + "}"
        lines = [f"{item.name}{tags(item.labels)} {item.value}" for item in counters]
        for item in histograms:
            for bound, count in item.buckets:
                lines.append(f"{item.name}_bucket{tags((*item.labels, ('le', str(bound))))} {count}")
            lines.extend((f"{item.name}_bucket{tags((*item.labels, ('le', '+Inf')))} {item.count}",
                          f"{item.name}_count{tags(item.labels)} {item.count}", f"{item.name}_sum{tags(item.labels)} {item.total}"))
        return "\n".join(lines) + ("\n" if lines else "")


MetricsRegistry = WorkflowMetrics
