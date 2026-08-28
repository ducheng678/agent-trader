from __future__ import annotations

import contextlib
import contextvars
import json
import logging
import re
import threading
import time
from collections import defaultdict
from collections.abc import Iterator, Mapping
from typing import Any

_request_id = contextvars.ContextVar("market_agent_request_id", default="")
_job_id = contextvars.ContextVar("market_agent_job_id", default="")
_metric_name = re.compile(r"^[a-zA-Z_:][a-zA-Z0-9_:]*$")


def current_request_id() -> str:
    return _request_id.get()


def current_job_id() -> str:
    return _job_id.get()


@contextlib.contextmanager
def request_context(request_id: str, job_id: str = "") -> Iterator[None]:
    request_token = _request_id.set(str(request_id or ""))
    job_token = _job_id.set(str(job_id or ""))
    try:
        yield
    finally:
        _request_id.reset(request_token)
        _job_id.reset(job_token)


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        request_id = current_request_id()
        job_id = current_job_id()
        if request_id:
            payload["request_id"] = request_id
        if job_id:
            payload["job_id"] = job_id
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), default=str)


def configure_structured_logging(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger("market_agent.backend")
    if not any(getattr(handler, "_market_agent_structured", False) for handler in logger.handlers):
        handler = logging.StreamHandler()
        handler._market_agent_structured = True
        handler.setFormatter(JsonLogFormatter())
        logger.addHandler(handler)
    logger.setLevel(str(level or "INFO").upper())
    logger.propagate = False
    return logger


class MetricsRegistry:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], float] = defaultdict(float)
        self._gauges: dict[tuple[str, tuple[tuple[str, str], ...]], float] = {}
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], tuple[int, float]] = {}

    @staticmethod
    def _key(name: str, labels: Mapping[str, Any] | None = None) -> tuple[str, tuple[tuple[str, str], ...]]:
        if not _metric_name.match(name):
            raise ValueError(f"invalid metric name: {name}")
        return name, tuple(sorted((str(key), str(value)) for key, value in dict(labels or {}).items()))

    def increment(self, name: str, value: float = 1.0, labels: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            self._counters[self._key(name, labels)] += float(value)

    def set_gauge(self, name: str, value: float, labels: Mapping[str, Any] | None = None) -> None:
        with self._lock:
            self._gauges[self._key(name, labels)] = float(value)

    def observe(self, name: str, value: float, labels: Mapping[str, Any] | None = None) -> None:
        key = self._key(name, labels)
        with self._lock:
            count, total = self._histograms.get(key, (0, 0.0))
            self._histograms[key] = (count + 1, total + float(value))

    @staticmethod
    def _render_labels(labels: tuple[tuple[str, str], ...]) -> str:
        if not labels:
            return ""
        encoded = ",".join(f'{key}="{value.replace(chr(34), chr(92) + chr(34))}"' for key, value in labels)
        return "{" + encoded + "}"

    def render_prometheus(self) -> str:
        with self._lock:
            counters = dict(self._counters)
            gauges = dict(self._gauges)
            histograms = dict(self._histograms)
        lines: list[str] = []
        for (name, labels), value in sorted(counters.items()):
            lines.append(f"{name}{self._render_labels(labels)} {value}")
        for (name, labels), value in sorted(gauges.items()):
            lines.append(f"{name}{self._render_labels(labels)} {value}")
        for (name, labels), (count, total) in sorted(histograms.items()):
            label_text = self._render_labels(labels)
            lines.append(f"{name}_count{label_text} {count}")
            lines.append(f"{name}_sum{label_text} {total}")
        return "\n".join(lines) + ("\n" if lines else "")


class TimedOperation:
    def __init__(self, metrics: MetricsRegistry, metric_name: str, labels: Mapping[str, Any] | None = None) -> None:
        self._metrics = metrics
        self._metric_name = metric_name
        self._labels = labels
        self._started_at = 0.0

    def __enter__(self) -> "TimedOperation":
        self._started_at = time.perf_counter()
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self._metrics.observe(self._metric_name, time.perf_counter() - self._started_at, self._labels)
