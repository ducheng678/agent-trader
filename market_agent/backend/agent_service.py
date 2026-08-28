from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any

from market_agent.backend.errors import ValidationError
from market_agent.backend.task_queue import BackgroundTaskQueue


class AgentPlaybookService:
    def __init__(self, engine_factory: Callable[[], Any] | None = None) -> None:
        self._engine_factory = engine_factory
        self._thread_local = threading.local()

    def _get_engine(self) -> Any:
        engine = getattr(self._thread_local, "engine", None)
        if engine is None:
            factory = self._engine_factory
            if factory is None:
                from market_agent.agent_runtime import DiscretionaryLLMEngine

                factory = DiscretionaryLLMEngine
            engine = factory()
            self._thread_local.engine = engine
        return engine

    def generate_playbook(self, payload: dict[str, Any]) -> dict[str, Any]:
        required_fields = ("user_query", "event_tape", "trigger_reason")
        missing = [field for field in required_fields if field not in payload]
        if missing:
            raise ValidationError("generate_playbook payload is missing required fields", {"missing": missing})
        event_tape = payload["event_tape"]
        if not isinstance(event_tape, list):
            raise ValidationError("event_tape must be a list")
        engine = self._get_engine()
        playbook, report = engine.get_playbook(
            user_query=str(payload["user_query"]),
            event_tape=event_tape,
            trigger_reason=str(payload["trigger_reason"]),
            trigger_event=payload.get("trigger_event"),
            recent_events=payload.get("recent_events"),
            trade_symbol_context=payload.get("trade_symbol_context"),
            active_symbol=payload.get("active_symbol"),
            has_live_position=bool(payload.get("has_live_position", False)),
            prefetched_passive_event_judge=payload.get("prefetched_passive_event_judge"),
        )
        return {"playbook": playbook.to_dict(), "report": report}


def register_agent_tasks(task_queue: BackgroundTaskQueue) -> AgentPlaybookService:
    service = AgentPlaybookService()
    task_queue.register("generate_playbook", service.generate_playbook)
    return service
