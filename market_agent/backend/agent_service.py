from __future__ import annotations

from collections.abc import Callable
from threading import RLock
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from market_agent.backend.api_contracts import GeneratePlaybookPayload
from market_agent.backend.errors import ValidationError
from market_agent.backend.task_queue import BackgroundTaskQueue


class AgentPlaybookService:
    def __init__(self, engine_factory: Callable[[], Any] | None = None) -> None:
        self._engine_factory = engine_factory
        self._engine: Any = None
        self._engine_lock = RLock()

    def _get_engine(self) -> Any:
        with self._engine_lock:
            if self._engine is None:
                factory = self._engine_factory
                if factory is None:
                    from market_agent.agent_runtime import DiscretionaryLLMEngine

                    factory = DiscretionaryLLMEngine
                self._engine = factory()
            return self._engine

    def generate_playbook(self, payload: dict[str, Any]) -> dict[str, Any]:
        try:
            request = GeneratePlaybookPayload.model_validate(payload)
        except PydanticValidationError as exc:
            raise ValidationError(
                "generate_playbook payload is invalid",
                {"errors": exc.errors(include_url=False, include_input=False)},
            ) from exc
        with self._engine_lock:
            engine = self._get_engine()
            playbook, report = engine.get_playbook(
                user_query=request.user_query,
                event_tape=request.event_tape,
                trigger_reason=request.trigger_reason,
                trigger_event=request.trigger_event,
                recent_events=request.recent_events,
                trade_symbol_context=request.trade_symbol_context,
                active_symbol=request.active_symbol,
                has_live_position=request.has_live_position,
                prefetched_passive_event_judge=request.prefetched_passive_event_judge,
            )
            return {"playbook": playbook.to_dict(), "report": report}


def register_agent_tasks(task_queue: BackgroundTaskQueue) -> AgentPlaybookService:
    service = AgentPlaybookService()
    task_queue.register("generate_playbook", service.generate_playbook)
    return service
