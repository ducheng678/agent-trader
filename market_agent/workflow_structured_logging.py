from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from hashlib import sha256
import json
import math
from threading import RLock
from typing import Literal

from pydantic import AwareDatetime, BaseModel, model_validator

from market_agent.workflow_agent_contracts import StrictModel
from market_agent.workflow_contracts import Digest, NonNegativeFinite, NonNegativeInt
from market_agent.workflow_tracing import TraceContext


EventKind = Literal["request_started", "request_completed", "request_failed", "agent_started", "agent_completed", "agent_failed", "tool_started", "tool_completed", "tool_failed", "cache_hit", "cache_miss", "retry_scheduled", "fallback_selected", "memory_completed", "permission_denied"]
EventStatus = Literal["started", "succeeded", "failed", "degraded", "unknown", "rejected", "skipped"]
Actor = Literal["ingress", "coordinator", "specialist", "reflection", "driver", "tool", "cache", "memory", "service"]


def _json_value(value: object, depth: int = 0) -> object:
    if depth > 32:
        raise ValueError("observed payload nesting exceeds the bound")
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("observed JSON keys must be strings")
        return {key: _json_value(item, depth + 1) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item, depth + 1) for item in value]
    raise ValueError("observed payload must contain finite JSON values")


def canonical_bytes(value: object) -> bytes:
    return json.dumps(_json_value(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


class PayloadSummary(StrictModel):
    content_hash: Digest
    size_bytes: NonNegativeInt
    kind: Literal["object", "array", "string", "number", "boolean", "null"]
    item_count: NonNegativeInt
    redacted: Literal[True] = True


def summarize_payload(value: object) -> PayloadSummary:
    normalized = _json_value(value)
    data = canonical_bytes(normalized)
    kind = ("null" if normalized is None else "boolean" if type(normalized) is bool else
            "object" if isinstance(normalized, dict) else "array" if isinstance(normalized, list) else
            "string" if isinstance(normalized, str) else "number")
    return PayloadSummary(content_hash=sha256(data).hexdigest(), size_bytes=len(data), kind=kind,
                          item_count=len(normalized) if isinstance(normalized, (dict, list)) else 1)


def redact_json(value: object) -> dict[str, object]:
    return summarize_payload(value).model_dump(mode="json")


def identity_digest(value: str | None) -> str | None:
    if value is None:
        return None
    if type(value) is not str or not value:
        raise ValueError("logging identity must be a nonempty string")
    return sha256(value.encode("utf-8")).hexdigest()


class StructuredEvent(StrictModel):
    occurred_at: AwareDatetime
    trace: TraceContext
    event: EventKind
    actor: Actor
    status: EventStatus
    workflow_id_hash: Digest | None = None
    task_id_hash: Digest | None = None
    subject_hash: Digest | None = None
    attempt: NonNegativeInt = 0
    latency_seconds: NonNegativeFinite = 0.0
    model: Literal["none", "luna", "terra", "sol", "local"] = "none"
    prompt_release_hash: Digest | None = None
    output_schema_hash: Digest | None = None
    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    cached_tokens: NonNegativeInt = 0
    cost_usd: NonNegativeFinite = 0.0
    payload: PayloadSummary | None = None
    reason: Literal["none", "invalid_input", "provider_error", "permission_denied", "timeout", "unavailable", "insufficient_evidence", "budget_exhausted", "invalid_output"] = "none"

    @model_validator(mode="after")
    def consistent(self):
        if self.occurred_at.utcoffset().total_seconds() != 0:
            raise ValueError("structured event timestamps must use UTC")
        if self.cached_tokens > self.input_tokens:
            raise ValueError("cached tokens cannot exceed input tokens")
        return self

    @classmethod
    def create(cls, trace: TraceContext, *, event: EventKind, actor: Actor, status: EventStatus,
               workflow_id: str | None = None, task_id: str | None = None, **fields) -> StructuredEvent:
        return cls(occurred_at=datetime.now(timezone.utc), trace=trace, event=event, actor=actor, status=status,
                   workflow_id_hash=identity_digest(workflow_id), task_id_hash=identity_digest(task_id), **fields)

    def json_line(self) -> str:
        event = StructuredEvent.model_validate(self)
        data = event.model_dump(mode="json", exclude={"trace"})
        data.update(trace_id=event.trace.trace_id, span_id=event.trace.span_id, parent_span_id=event.trace.parent_span_id,
                    trace_links=[link.model_dump(mode="json") for link in event.trace.links])
        return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False) + "\n"


class ObservabilityUnavailableError(RuntimeError):
    pass


class StructuredLogger:
    def __init__(self, sink: Callable[[str], None]):
        self._sink = sink
        self._lock = RLock()

    def emit(self, event: StructuredEvent) -> None:
        line = StructuredEvent.model_validate(event).json_line()
        with self._lock:
            try:
                self._sink(line)
            except Exception:
                raise ObservabilityUnavailableError("structured event sink failed") from None

    record = emit
