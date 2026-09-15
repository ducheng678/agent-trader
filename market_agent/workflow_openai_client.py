"""OpenAI Responses adapter for the bounded AgentDriver model boundary."""

from __future__ import annotations

import hashlib
import json
import re
import time
from collections.abc import Callable, Mapping

from market_agent.langchain_runtime import LangChainResponsesRuntime
from market_agent.workflow_agent_contracts import AgentUsage, ModelTier
from market_agent.workflow_agent_driver import ModelRequest, ModelResponse
from market_agent.openai_usage import UsageTokens, estimate_workflow_usage_cost
from market_agent.workflow_model_budget import WORKFLOW_PRICING_VERSION, admit_model_cost


def _provider_nonnegative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"OpenAI response has invalid {field_name}")
    return value


def _provider_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"OpenAI response has invalid {field_name}")
    return value.strip()


def _provider_model_matches_priced_identity(provider_model: str, model_id: str) -> bool:
    """Permit only the configured model or its documented dated API snapshot."""

    return provider_model == model_id or re.fullmatch(
        re.escape(model_id) + r"-\d{4}-\d{2}-\d{2}", provider_model,
    ) is not None


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
        if any(
            resolved_models[tier] != f"gpt-5.6-{tier.value}"
            for tier in ModelTier
        ):
            raise ValueError("OpenAI adapter model identifiers must match their pricing tiers")
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
        model_id = self._model_ids[request.model_tier]
        if model_id != f"gpt-5.6-{request.model_tier.value}":
            raise ValueError("OpenAI adapter model identifier does not match the requested pricing tier")
        admission = admit_model_cost(
            model_id=model_id,
            pricing_band=request.pricing_band,
            messages=request.messages,
            output_schema_json=request.output_schema_json,
            cost_limit_usd=request.cost_limit_usd,
        )
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
            model=model_id,
            input=messages,
            temperature=request.temperature,
            max_output_tokens=admission.max_output_tokens,
            prompt_cache_key=cache_key,
            text={"format": {
                "type": "json_schema",
                "name": request.output_schema_id.replace(".", "_").replace("-", "_")[:64],
                "strict": True,
                "schema": schema,
            }},
        )
        provider_model = _provider_text(response.model, "model id")
        if not _provider_model_matches_priced_identity(provider_model, model_id):
            raise ValueError("OpenAI response model does not match the configured priced identity")
        usage = response.usage
        input_tokens = _provider_nonnegative_int(usage.input_tokens, "input tokens")
        output_tokens = _provider_nonnegative_int(usage.output_tokens, "output tokens")
        input_details = getattr(usage, "input_tokens_details", None)
        cached_input_tokens = _provider_nonnegative_int(
            input_details.get("cached_tokens", 0) if isinstance(input_details, Mapping)
            else getattr(input_details, "cached_tokens", 0), "cached input tokens"
        )
        if cached_input_tokens > input_tokens:
            raise ValueError("OpenAI cached input tokens exceed input tokens")
        output_items = getattr(response, "output", ())
        if not isinstance(output_items, (tuple, list)):
            raise ValueError("OpenAI response output is invalid")
        web_search_tool_calls = sum(
            (item.get("type") if isinstance(item, Mapping) else getattr(item, "type", None))
            == "web_search_call" for item in output_items
        )
        return ModelResponse(
            content=response.output_text,
            usage=AgentUsage(
                input_tokens=input_tokens,
                cached_input_tokens=cached_input_tokens,
                output_tokens=output_tokens,
                web_search_tool_calls=web_search_tool_calls,
                cost_usd=float(estimate_workflow_usage_cost(
                    f"gpt-5.6-{request.model_tier.value}",
                    admission.pricing_band,
                    UsageTokens(
                        input_tokens=input_tokens,
                        cached_input_tokens=cached_input_tokens,
                        output_tokens=output_tokens,
                        web_search_tool_calls=web_search_tool_calls,
                    ),
                )),
                model_tier=request.model_tier,
                provider="openai",
                provider_request_id=_provider_text(response.id, "response id"),
                model_id=provider_model,
                pricing_version=admission.pricing_version,
                pricing_model_id=f"gpt-5.6-{request.model_tier.value}",
                pricing_band=admission.pricing_band,
            ),
        )
