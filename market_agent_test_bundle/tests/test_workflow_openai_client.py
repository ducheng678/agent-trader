from __future__ import annotations

from types import SimpleNamespace
from decimal import Decimal

from market_agent.workflow_agent_contracts import ModelTier
from market_agent.workflow_agent_driver import ModelRequest
from market_agent.workflow_openai_client import OpenAIModelClient


def test_openai_adapter_preserves_provider_usage_identity_and_dimensions() -> None:
    response = SimpleNamespace(
        id="resp-123",
        model="gpt-5.6-terra-2026-08-15",
        output_text='{"answer":"known"}',
        output=(
            SimpleNamespace(type="web_search_call"),
            SimpleNamespace(type="message"),
        ),
        usage=SimpleNamespace(
            input_tokens=101,
            input_tokens_details=SimpleNamespace(cached_tokens=40),
            output_tokens=11,
        ),
    )
    client = object.__new__(OpenAIModelClient)
    client._runtime = SimpleNamespace(create=lambda **_kwargs: response)
    client._clock = lambda: 1.0
    client._cache_prefix = "test"
    client._model_ids = {
        ModelTier.LUNA: "gpt-5.6-luna",
        ModelTier.TERRA: "gpt-5.6-terra",
        ModelTier.SOL: "gpt-5.6-sol",
    }
    request = ModelRequest(
        trace_id="1" * 32,
        model_tier=ModelTier.TERRA,
        messages=(("system", "stable"), ("user", "payload")),
        temperature=0.0,
        output_schema_id="answer-v1",
        output_schema_digest="a" * 64,
        output_schema_json='{"type":"object","additionalProperties":false}',
        deadline_epoch=10.0,
        attempt=0,
        cost_limit_usd=0.05,
    )

    result = client.invoke(request)

    assert result.usage.input_tokens == 101
    assert result.usage.cached_input_tokens == 40
    assert result.usage.output_tokens == 11
    assert result.usage.web_search_tool_calls == 1
    assert result.usage.provider == "openai"
    assert result.usage.provider_request_id == "resp-123"
    assert result.usage.model_id == "gpt-5.6-terra-2026-08-15"
    assert result.usage.pricing_version == "openai-standard-2026-08-01"
    assert result.usage.pricing_model_id == "gpt-5.6-terra"
    assert result.usage.pricing_band == "short"
    assert Decimal(str(result.usage.cost_usd)) == Decimal("0.010262")


def test_openai_adapter_uses_request_pinned_long_pricing_band() -> None:
    response = SimpleNamespace(
        id="resp-long", model="gpt-5.6-terra", output_text='{"answer":"known"}',
        output=(SimpleNamespace(type="web_search_call"),),
        usage=SimpleNamespace(
            input_tokens=101,
            input_tokens_details=SimpleNamespace(cached_tokens=40),
            output_tokens=11,
        ),
    )
    client = object.__new__(OpenAIModelClient)
    client._runtime = SimpleNamespace(create=lambda **_kwargs: response)
    client._clock = lambda: 1.0
    client._cache_prefix = "test"
    client._model_ids = {tier: f"gpt-5.6-{tier.value}" for tier in ModelTier}
    request = ModelRequest(
        trace_id="1" * 32, model_tier=ModelTier.TERRA,
        messages=(("system", "stable"), ("user", "payload")), temperature=0.0,
        output_schema_id="answer-v1", output_schema_digest="a" * 64,
        output_schema_json='{"type":"object","additionalProperties":false}',
        deadline_epoch=10.0, attempt=0, cost_limit_usd=0.20,
        pricing_band="long",
    )

    result = client.invoke(request)

    assert result.usage.pricing_band == "long"
    assert Decimal(str(result.usage.cost_usd)) == Decimal("0.010458")
