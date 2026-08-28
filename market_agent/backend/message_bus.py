from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Protocol


@dataclass(frozen=True)
class MessageEnvelope:
    topic: str
    payload: dict[str, Any]
    request_id: str = ""
    job_id: str = ""
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    occurred_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


MessageHandler = Callable[[MessageEnvelope], None]


class MessageBus(Protocol):
    def publish(self, message: MessageEnvelope) -> None:
        ...

    def subscribe(self, topic: str, handler: MessageHandler) -> Callable[[], None]:
        ...


class InMemoryMessageBus:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._handlers: dict[str, list[MessageHandler]] = {}

    def subscribe(self, topic: str, handler: MessageHandler) -> Callable[[], None]:
        normalized_topic = str(topic).strip()
        if not normalized_topic:
            raise ValueError("topic is required")
        with self._lock:
            self._handlers.setdefault(normalized_topic, []).append(handler)

        def unsubscribe() -> None:
            with self._lock:
                subscribers = self._handlers.get(normalized_topic, [])
                if handler in subscribers:
                    subscribers.remove(handler)

        return unsubscribe

    def publish(self, message: MessageEnvelope) -> None:
        with self._lock:
            subscribers = tuple(self._handlers.get(message.topic, ())) + tuple(self._handlers.get("*", ()))
        for handler in subscribers:
            handler(message)
