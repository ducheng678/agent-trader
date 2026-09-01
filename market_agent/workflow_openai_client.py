"""OpenAI Responses adapter for the bounded AgentDriver model boundary."""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping

from market_agent.langchain_runtime import LangChainResponsesRuntime
from market_agent.workflow_agent_contracts import AgentUsage, ModelTier
from market_agent.workflow_agent_driver import ModelRequest, ModelResponse


class OpenAIModelClient:
    def __init__(self, *, api_key: str, clock: Callable[[], float] = time.time,
                 prompt_cache_prefix: str = "market-agent",
                 model_ids: Mapping[ModelTier, str] | None = None) -> None:
        if not api_key.strip() or not prompt_cache_prefix.strip():
            raise ValueError("OpenAI adapter requires an API key and cache prefix")
        resolved_models = dict(model_ids or {
            ModelTier.SOL: "gpt-5.6-sol",
            ModelTier.TERRA: "gpt-5.6-terra",
            ModelTier.LUNA: "gpt-5.6-luna",
        })
        if set(resolved_models) != set(ModelTier) or any(
            type(value) is not str or not value.strip() for value in resolved_models.values()
        ):
            raise ValueError("OpenAI adapter requires one model identifier per tier")
        self._runtime = LangChainResponsesRuntime(api_key=api_key)
        self._clock = clock
        self._cache_prefix = prompt_cache_prefix.strip()
        self._model_ids = resolved_models

    def invoke(self, request: ModelRequest) -> ModelResponse:
        if type(request) is not ModelRequest:
            raise TypeError("OpenAI adapter requires a bounded ModelRequest")
        remaining = request.deadline_epoch - self._clock()
        if remaining <= 0:
            raise TimeoutError("model request deadline elapsed")
        messages = [
            {"role": role, "content": [{"type": "input_text", "text": content}]}
            for role, content in request.messages
        ]
        schema = json.loads(request.output_schema_json)
        stable_system = request.messages[0][1] if request.messages else ""
        prompt_hash = hashlib.sha256(stable_system.encode("utf-8")).hexdigest()[:24]
        cache_key = f"{self._cache_prefix}-{request.model_tier.value}-{prompt_hash}"
        response = self._runtime.create(
            timeout=remaining,
            model=self._model_ids[request.model_tier],
            input=messages,
            temperature=request.temperature,
            prompt_cache_key=cache_key,
            text={"format": {
                "type": "json_schema",
                "name": request.output_schema_id.replace(".", "_").replace("-", "_")[:64],
                "strict": True,
                "schema": schema,
            }},
        )
        return ModelResponse(
            content=response.output_text,
            usage=AgentUsage(
                input_tokens=max(0, int(response.usage.input_tokens)),
                output_tokens=max(0, int(response.usage.output_tokens)),
                # The driver reserves before calling. Charging the reservation
                # is conservative when provider-specific billing detail is not
                # exposed by this transport.
                cost_usd=request.cost_limit_usd,
                model_tier=request.model_tier,
            ),
        )
