"""Fail-closed Redis adapters; clients are injected only at backend composition."""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
from typing import Any, Protocol

from market_agent.backend.message_bus import MessageEnvelope


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


_SENSITIVE = frozenset({"authorization", "credential", "password", "secret", "token", "api_key", "access_key"})


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
            if key.casefold() in _SENSITIVE:
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
        self._prefix = f"{namespace}:tenant:{tenant_id}:cache:"
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
            return self._client.set(self._key("idempotency:" + key), _encode(value, self._maximum_value_bytes), ex=lifetime, nx=True) is True
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
        self._prefix = f"{namespace}:tenant:{tenant_id}:stream:"
        self._maximum_message_bytes = maximum_message_bytes

    def publish(self, message: MessageEnvelope) -> str:
        if type(message) is not MessageEnvelope or not message.topic.strip():
            raise ValueError("a concrete message envelope with topic is required")
        payload = dict(message.payload)
        trace_id = str(payload.get("trace_id") or message.request_id).strip()
        if not trace_id:
            raise ValueError("messages require a trace_id or request_id")
        encoded = _encode({"topic": message.topic, "payload": payload, "request_id": message.request_id, "job_id": message.job_id, "message_id": message.message_id, "occurred_at": message.occurred_at, "trace_id": trace_id}, self._maximum_message_bytes)
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
