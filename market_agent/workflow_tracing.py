from __future__ import annotations

from collections.abc import Mapping
import secrets
from typing import Annotated

from pydantic import AfterValidator, Field, StringConstraints, model_validator

from market_agent.workflow_agent_contracts import StrictModel


def _nonzero(value: str) -> str:
    if not int(value, 16):
        raise ValueError("trace and span identifiers cannot be zero")
    return value


TraceId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{32}$"), AfterValidator(_nonzero)]
SpanId = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{16}$"), AfterValidator(_nonzero)]


class TraceMismatchError(ValueError):
    pass


class TraceLink(StrictModel):
    trace_id: TraceId
    span_id: SpanId


class TraceContext(StrictModel):
    trace_id: TraceId
    span_id: SpanId
    parent_span_id: SpanId | None = None
    sampled: bool = False
    links: tuple[TraceLink, ...] = Field(default_factory=tuple, max_length=8)

    @model_validator(mode="after")
    def distinct_parent(self):
        if self.span_id == self.parent_span_id:
            raise ValueError("a span cannot parent itself")
        if len({(link.trace_id, link.span_id) for link in self.links}) != len(self.links):
            raise ValueError("trace links must be unique")
        return self

    @classmethod
    def new_request(cls, *, upstream: TraceContext | Mapping[str, str] | None = None,
                    sampled: bool = False) -> TraceContext:
        source = cls.extract(upstream) if isinstance(upstream, Mapping) else upstream
        if source is not None:
            source = cls.model_validate(source)
        return cls(trace_id=_identifier(16), span_id=_identifier(8), sampled=sampled,
                   links=(TraceLink(trace_id=source.trace_id, span_id=source.span_id),) if source else ())

    def child(self) -> TraceContext:
        parent = TraceContext.model_validate(self)
        identifier = _identifier(8)
        while identifier == parent.span_id:
            identifier = _identifier(8)
        return TraceContext(trace_id=parent.trace_id, span_id=identifier,
                            parent_span_id=parent.span_id, sampled=parent.sampled)

    def inject(self, carrier: Mapping[str, str] | None = None) -> dict[str, str]:
        context = TraceContext.model_validate(self)
        values = dict(carrier or {})
        for key in tuple(values):
            if key.casefold() in {"traceparent", "x-workflow-parent-span-id"}:
                del values[key]
        values["traceparent"] = f"00-{context.trace_id}-{context.span_id}-{'01' if context.sampled else '00'}"
        if context.parent_span_id is not None:
            values["x-workflow-parent-span-id"] = context.parent_span_id
        return values

    @classmethod
    def extract(cls, carrier: Mapping[str, str]) -> TraceContext:
        headers = {}
        for key, value in carrier.items():
            if type(key) is not str or type(value) is not str:
                raise ValueError("trace carrier must contain string headers")
            normalized = key.casefold()
            if normalized in headers:
                raise ValueError("trace carrier contains duplicate headers")
            headers[normalized] = value
        parts = headers.get("traceparent", "").split("-")
        if len(parts) != 4 or parts[0] != "00" or parts[3] not in {"00", "01"}:
            raise ValueError("unsupported or invalid traceparent")
        return cls(trace_id=parts[1], span_id=parts[2], sampled=parts[3] == "01",
                   parent_span_id=headers.get("x-workflow-parent-span-id"))

    def assert_same_trace(self, other: TraceContext | str) -> None:
        expected = TraceContext.model_validate(self)
        identifier = TraceContext.model_validate(other).trace_id if isinstance(other, TraceContext) else other
        if type(identifier) is not str or identifier != expected.trace_id:
            raise TraceMismatchError("operation belongs to another trace")


def _identifier(size: int) -> str:
    value = secrets.token_hex(size)
    while not int(value, 16):
        value = secrets.token_hex(size)
    return value


def assert_same_trace(expected: TraceContext, *others: TraceContext | str) -> None:
    for other in others:
        expected.assert_same_trace(other)


new_request = TraceContext.new_request
extract = TraceContext.extract
