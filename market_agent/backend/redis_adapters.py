"""Fail-closed Redis adapters; clients are injected only at backend composition."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
import math
import random
import re
import threading
from typing import Any, Protocol

from market_agent.backend.message_bus import MessageEnvelope
from market_agent.backend.database import JobRecord


class RedisUnavailableError(RuntimeError):
    pass


class RedisSerializationError(ValueError):
    pass


class RedisLike(Protocol):
    def get(self, key: str) -> object: ...
    def set(self, key: str, value: str, ex: int | None = None, nx: bool = False) -> object: ...
    def delete(self, *keys: str) -> object: ...
    def ping(self) -> object: ...
    def xadd(self, name: str, fields: dict[str, str], id: str = "*") -> object: ...
    def xreadgroup(self, groupname: str, consumername: str, streams: dict[str, str], count: int = 1, block: int | None = None) -> object: ...
    def xack(self, name: str, groupname: str, *ids: str) -> object: ...
    def xgroup_create(self, name: str, groupname: str, id: str = "0", mkstream: bool = False) -> object: ...
    def xautoclaim(self, name: str, groupname: str, consumername: str, min_idle_time: int,
                   start_id: str = "0-0", count: int | None = None) -> object: ...


_SENSITIVE = frozenset({"authorization", "credential", "password", "secret", "token", "api_key", "access_key"})
_TRACE_ID = re.compile(r"^[0-9a-fA-F]{32}$")
_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_STREAM_ID = re.compile(r"^[0-9]+-[0-9]+$")


def _sensitive_key(key: str) -> bool:
    normalized = re.sub(r"[^a-z0-9]", "", key.casefold())
    markers = tuple(re.sub(r"[^a-z0-9]", "", marker) for marker in _SENSITIVE)
    return normalized in markers or any(normalized.endswith(marker) for marker in markers)


def _tenant_namespace(namespace: str, tenant_id: str, kind: str) -> str:
    if type(namespace) is not str or type(tenant_id) is not str or not namespace.strip() or not tenant_id.strip():
        raise ValueError("Redis namespace and tenant are required")
    return f"{namespace}:tenant:{sha256(tenant_id.encode('utf-8')).hexdigest()}:{kind}:"


def _require_trace_id(value: object) -> str:
    if type(value) is not str or not _TRACE_ID.fullmatch(value) or not int(value, 16):
        raise ValueError("Redis messages require a nonzero W3C trace_id")
    return value


def _require_request_id(value: object) -> str:
    if type(value) is not str or not _REQUEST_ID.fullmatch(value):
        raise ValueError("Redis messages require a compact nonempty request_id")
    return value


def _safe_value(value: Any) -> Any:
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise RedisSerializationError("non-finite values cannot enter Redis")
        return value
    if isinstance(value, (tuple, list)):
        return [_safe_value(item) for item in value]
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise RedisSerializationError("Redis values require string object keys")
            if _sensitive_key(key):
                result[key] = "[REDACTED]"
            else:
                result[key] = _safe_value(item)
        return result
    raise RedisSerializationError("Redis values must be bounded JSON")


def _encode(value: Any, maximum_bytes: int) -> str:
    encoded = json.dumps(_safe_value(value), ensure_ascii=False, separators=(",", ":"), sort_keys=True, allow_nan=False)
    if len(encoded.encode("utf-8")) > maximum_bytes:
        raise RedisSerializationError("Redis value exceeds configured byte bound")
    return encoded


def _decode(value: object) -> Any:
    if value is None:
        return None
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if type(value) is not str:
        raise RedisUnavailableError("Redis returned an invalid value")
    try:
        return json.loads(value)
    except Exception as error:
        raise RedisUnavailableError("Redis returned malformed JSON") from error


@dataclass(frozen=True, slots=True)
class RedisHealth:
    status: str
    detail: str = ""


class RedisTenantCache:
    def __init__(self, client: RedisLike, *, tenant_id: str, namespace: str = "market-agent", default_ttl_seconds: int = 60, maximum_value_bytes: int = 65_536) -> None:
        if not tenant_id.strip() or not namespace.strip() or default_ttl_seconds < 1 or maximum_value_bytes < 128:
            raise ValueError("invalid Redis cache configuration")
        self._client = client
        self._prefix = _tenant_namespace(namespace, tenant_id, "cache")
        self._ttl = default_ttl_seconds
        self._maximum_value_bytes = maximum_value_bytes

    def get(self, key: str, default: Any = None) -> Any:
        try:
            value = _decode(self._client.get(self._key(key)))
            return default if value is None else value
        except RedisSerializationError:
            raise
        except Exception as error:
            raise RedisUnavailableError("Redis cache read failed") from error

    def set(self, key: str, value: Any, ttl_seconds: float | None = None) -> None:
        lifetime = self._ttl if ttl_seconds is None else int(ttl_seconds)
        if lifetime < 1:
            self.delete(key)
            return
        try:
            self._client.set(self._key(key), _encode(value, self._maximum_value_bytes), ex=lifetime)
        except RedisSerializationError:
            raise
        except Exception as error:
            raise RedisUnavailableError("Redis cache write failed") from error

    def set_idempotent(self, key: str, value: Any, *, ttl_seconds: float | None = None) -> bool:
        lifetime = self._ttl if ttl_seconds is None else int(ttl_seconds)
        if lifetime < 1:
            raise ValueError("idempotency TTL must be positive")
        try:
            result = self._client.set(self._key("idempotency:" + key), _encode(value, self._maximum_value_bytes), ex=lifetime, nx=True)
            return result is True or result in ("OK", b"OK")
        except RedisSerializationError:
            raise
        except Exception as error:
            raise RedisUnavailableError("Redis idempotency write failed") from error

    def delete(self, key: str) -> None:
        try:
            self._client.delete(self._key(key))
        except Exception as error:
            raise RedisUnavailableError("Redis cache delete failed") from error

    def health(self) -> RedisHealth:
        try:
            return RedisHealth("ok" if self._client.ping() else "unavailable")
        except Exception as error:
            return RedisHealth("unavailable", type(error).__name__)

    def _key(self, key: str) -> str:
        if type(key) is not str or not key.strip() or any(character.isspace() for character in key):
            raise ValueError("Redis cache key is invalid")
        return self._prefix + key


class RedisJobCache:
    """Typed task-cache facade that preserves JobRecord on Redis round trips."""

    def __init__(self, cache: RedisTenantCache) -> None:
        if type(cache) is not RedisTenantCache:
            raise TypeError("Redis job cache requires a tenant cache")
        self._cache = cache

    def get(self, key: str, default: Any = None) -> Any:
        value = self._cache.get(key, default)
        if value is default:
            return default
        if not isinstance(value, dict):
            self._cache.delete(key)
            return default
        try:
            return JobRecord(**value)
        except (TypeError, ValueError):
            self._cache.delete(key)
            return default

    def set(self, key: str, value: Any, ttl_seconds: float | None = None) -> None:
        if type(value) is not JobRecord:
            raise TypeError("Redis task cache stores only JobRecord values")
        self._cache.set(key, asdict(value), ttl_seconds)

    def delete(self, key: str) -> None:
        self._cache.delete(key)

    def health(self) -> RedisHealth:
        return self._cache.health()


@dataclass(frozen=True, slots=True)
class StreamDelivery:
    stream: str
    message_id: str
    envelope: MessageEnvelope


class RedisStreamMessageBus:
    def __init__(self, client: RedisLike, *, tenant_id: str, namespace: str = "market-agent", maximum_message_bytes: int = 65_536) -> None:
        if not tenant_id.strip() or not namespace.strip() or maximum_message_bytes < 256:
            raise ValueError("invalid Redis message bus configuration")
        self._client = client
        self._prefix = _tenant_namespace(namespace, tenant_id, "stream")
        self._maximum_message_bytes = maximum_message_bytes

    def publish(self, message: MessageEnvelope) -> str:
        if type(message) is not MessageEnvelope or not message.topic.strip():
            raise ValueError("a concrete message envelope with topic is required")
        payload = dict(message.payload)
        trace_id = _require_trace_id(payload.get("trace_id"))
        request_id = _require_request_id(message.request_id)
        encoded = _encode({"topic": message.topic, "payload": payload, "request_id": request_id, "job_id": message.job_id, "message_id": message.message_id, "occurred_at": message.occurred_at, "trace_id": trace_id}, self._maximum_message_bytes)
        try:
            identifier = self._client.xadd(self._stream(message.topic), {"envelope": encoded, "trace_id": trace_id})
            return identifier.decode("utf-8") if isinstance(identifier, bytes) else str(identifier)
        except Exception as error:
            raise RedisUnavailableError("Redis stream publish failed") from error

    def consume(self, *, topic: str, group: str, consumer: str, count: int = 1, block_ms: int | None = 1_000) -> tuple[StreamDelivery, ...]:
        if not all(type(value) is str and value.strip() for value in (topic, group, consumer)) or count < 1:
            raise ValueError("stream consume arguments are invalid")
        stream = self._stream(topic)
        try:
            batches = self._client.xreadgroup(group, consumer, {stream: ">"}, count=count, block=block_ms)
            return tuple(self._deliveries(stream, batches))
        except RedisUnavailableError:
            raise
        except Exception as error:
            raise RedisUnavailableError("Redis stream consume failed") from error

    def recover_pending(self, *, topic: str, group: str, consumer: str, min_idle_ms: int,
                        start_id: str = "0-0", count: int = 1) -> tuple[str, tuple[StreamDelivery, ...]]:
        if (not all(type(value) is str and value.strip() for value in (topic, group, consumer, start_id))
                or not _STREAM_ID.fullmatch(start_id) or min_idle_ms < 1 or count < 1):
            raise ValueError("stream pending recovery arguments are invalid")
        stream = self._stream(topic)
        try:
            result = self._client.xautoclaim(
                stream,
                group,
                consumer,
                min_idle_ms,
                start_id=start_id,
                count=count,
            )
            if not isinstance(result, (list, tuple)) or len(result) < 2:
                raise RedisUnavailableError("Redis stream returned invalid pending recovery")
            next_start, messages = result[0], result[1]
            if isinstance(next_start, bytes):
                next_start = next_start.decode("utf-8")
            if type(next_start) is not str or not _STREAM_ID.fullmatch(next_start):
                raise RedisUnavailableError("Redis stream returned invalid pending cursor")
            return next_start, tuple(self._deliveries(stream, ((stream, messages),)))
        except RedisUnavailableError:
            raise
        except Exception as error:
            raise RedisUnavailableError("Redis stream pending recovery failed") from error

    def ack(self, delivery: StreamDelivery, *, group: str) -> None:
        if type(delivery) is not StreamDelivery or not group.strip():
            raise ValueError("stream acknowledgement is invalid")
        try:
            self._client.xack(delivery.stream, group, delivery.message_id)
        except Exception as error:
            raise RedisUnavailableError("Redis stream acknowledgement failed") from error

    def dead_letter(self, delivery: StreamDelivery, *, group: str, reason: str) -> str:
        if not reason.strip():
            raise ValueError("dead-letter reason is required")
        payload = {"original_stream": delivery.stream, "original_message_id": delivery.message_id, "reason": reason, "envelope": json.loads(_encode(delivery.envelope.payload, self._maximum_message_bytes)), "trace_id": str(delivery.envelope.payload.get("trace_id") or delivery.envelope.request_id)}
        try:
            identifier = self._client.xadd(delivery.stream + ":dead", {"envelope": _encode(payload, self._maximum_message_bytes)})
            self._client.xack(delivery.stream, group, delivery.message_id)
            return identifier.decode("utf-8") if isinstance(identifier, bytes) else str(identifier)
        except Exception as error:
            raise RedisUnavailableError("Redis dead-letter operation failed") from error

    def health(self) -> RedisHealth:
        try:
            return RedisHealth("ok" if self._client.ping() else "unavailable")
        except Exception as error:
            return RedisHealth("unavailable", type(error).__name__)

    def _stream(self, topic: str) -> str:
        if type(topic) is not str or not topic.strip() or any(character.isspace() for character in topic):
            raise ValueError("stream topic is invalid")
        return self._prefix + topic

    def _deliveries(self, stream: str, batches: object):
        if not isinstance(batches, (list, tuple)):
            raise RedisUnavailableError("Redis stream returned invalid batches")
        for batch_stream, messages in batches:
            normalized_stream = batch_stream.decode("utf-8") if isinstance(batch_stream, bytes) else batch_stream
            if normalized_stream != stream or not isinstance(messages, (list, tuple)):
                raise RedisUnavailableError("Redis stream crossed tenant/topic boundary")
            for message_id, fields in messages:
                normalized_id = message_id.decode("utf-8") if isinstance(message_id, bytes) else message_id
                fields = {(key.decode("utf-8") if isinstance(key, bytes) else key): (value.decode("utf-8") if isinstance(value, bytes) else value) for key, value in fields.items()}
                decoded = _decode(fields.get("envelope"))
                if not isinstance(decoded, dict) or decoded.get("trace_id") != fields.get("trace_id"):
                    raise RedisUnavailableError("Redis stream trace propagation is invalid")
                envelope = MessageEnvelope(topic=decoded["topic"], payload=decoded["payload"], request_id=decoded["request_id"], job_id=decoded["job_id"], message_id=decoded["message_id"], occurred_at=decoded["occurred_at"])
                yield StreamDelivery(stream=stream, message_id=str(normalized_id), envelope=envelope)


class RedisMessageBusAdapter:
    """MessageBus-compatible Redis Streams consumer with ack and dead-letter handling."""

    _RETRY_BASE_SECONDS = 0.25
    _RETRY_MAX_SECONDS = 5.0
    _PENDING_IDLE_MS = 60_000
    _PENDING_BATCH_SIZE = 10

    def __init__(self, stream_bus: RedisStreamMessageBus, *, group: str = "market-agent-workers") -> None:
        if type(stream_bus) is not RedisStreamMessageBus or not group.strip():
            raise ValueError("Redis message adapter requires a stream bus and consumer group")
        self._bus = stream_bus
        self._group = group
        self._consumer = "consumer-" + sha256(f"{stream_bus._prefix}:{group}".encode("utf-8")).hexdigest()[:24]
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._health_lock = threading.Lock()
        self._consumer_failures: dict[str, str] = {}

    def publish(self, message: MessageEnvelope) -> None:
        self._bus.publish(message)

    def subscribe(self, topic: str, handler: Any):
        if not callable(handler):
            raise TypeError("message handler must be callable")
        consumer = self._consumer
        health_key = consumer + ":" + topic
        stream = self._bus._stream(topic)
        try:
            self._bus._client.xgroup_create(stream, self._group, id="0", mkstream=True)
        except Exception as error:
            if "BUSYGROUP" not in str(error).upper():
                raise RedisUnavailableError("Redis consumer group creation failed") from error
        local_stop = threading.Event()

        def consume() -> None:
            failures = 0
            pending_cursor = "0-0"
            try:
                while not self._stop.is_set() and not local_stop.is_set():
                    try:
                        pending_cursor, pending = self._bus.recover_pending(
                            topic=topic,
                            group=self._group,
                            consumer=consumer,
                            min_idle_ms=self._PENDING_IDLE_MS,
                            start_id=pending_cursor,
                            count=self._PENDING_BATCH_SIZE,
                        )
                        deliveries = pending or self._bus.consume(
                            topic=topic,
                            group=self._group,
                            consumer=consumer,
                            count=self._PENDING_BATCH_SIZE,
                            block_ms=1000,
                        )
                        for delivery in deliveries:
                            if self._stop.is_set() or local_stop.is_set():
                                return
                            try:
                                handler(delivery.envelope)
                            except Exception as error:
                                self._bus.dead_letter(
                                    delivery,
                                    group=self._group,
                                    reason=type(error).__name__[:128],
                                )
                            else:
                                self._bus.ack(delivery, group=self._group)
                    except RedisUnavailableError as error:
                        failures += 1
                        self._mark_consumer_degraded(health_key, error)
                        if self._wait_for_retry(local_stop, self._retry_delay(failures)):
                            return
                        continue

                    failures = 0
                    self._mark_consumer_healthy(health_key)
            finally:
                self._mark_consumer_healthy(health_key)

        thread = threading.Thread(target=consume, name=consumer, daemon=True)
        self._threads.append(thread)
        thread.start()

        def unsubscribe() -> None:
            local_stop.set()

        return unsubscribe

    def _retry_delay(self, failures: int) -> float:
        exponential = min(
            self._RETRY_MAX_SECONDS,
            self._RETRY_BASE_SECONDS * (2 ** min(failures - 1, 16)),
        )
        return random.uniform(exponential / 2.0, exponential)

    def _wait_for_retry(self, local_stop: threading.Event, delay_seconds: float) -> bool:
        remaining = delay_seconds
        while remaining > 0.0:
            interval = min(0.1, remaining)
            if self._stop.wait(interval) or local_stop.is_set():
                return True
            remaining -= interval
        return self._stop.is_set() or local_stop.is_set()

    def _mark_consumer_degraded(self, consumer: str, error: Exception) -> None:
        with self._health_lock:
            self._consumer_failures[consumer] = type(error).__name__

    def _mark_consumer_healthy(self, consumer: str) -> None:
        with self._health_lock:
            self._consumer_failures.pop(consumer, None)

    def close(self) -> None:
        self._stop.set()
        for thread in tuple(self._threads):
            thread.join(timeout=2.0)

    def health(self) -> RedisHealth:
        with self._health_lock:
            if self._consumer_failures:
                return RedisHealth("degraded", ",".join(sorted(set(self._consumer_failures.values()))))
        return self._bus.health()
