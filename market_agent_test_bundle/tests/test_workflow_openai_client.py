from __future__ import annotations

import math
from types import SimpleNamespace
from decimal import Decimal

import pytest

from market_agent.openai_usage import UsageTokens, estimate_workflow_usage_cost
from market_agent.workflow_agent_contracts import ModelTier
from market_agent.workflow_agent_driver import ModelRequest, pricing_band_for_rendered_request
from market_agent.workflow_model_budget import admit_model_cost
from market_agent.workflow_openai_client import OpenAIModelClient


def _request(*, cost_limit_usd: float = 0.05, pricing_band: str = "short") -> ModelRequest:
    return ModelRequest(
        trace_id="1" * 32,
        model_tier=ModelTier.TERRA,
        messages=(("system", "stable"), ("user", "payload")),
        temperature=0.0,
        output_schema_id="answer-v1",
        output_schema_digest="a" * 64,
        output_schema_json='{"type":"object","additionalProperties":false}',
        deadline_epoch=10.0,
        attempt=0,
        cost_limit_usd=cost_limit_usd,
        pricing_band=pricing_band,
    )


def _client_with_runtime(create):
    client = object.__new__(OpenAIModelClient)
    client._runtime = SimpleNamespace(create=create)
    client._clock = lambda: 1.0
    client._cache_prefix = "test"
    client._model_ids = {tier: f"gpt-5.6-{tier.value}" for tier in ModelTier}
    return client


def test_openai_adapter_rejects_insufficient_reservation_before_provider_call() -> None:
    calls = []
    client = _client_with_runtime(lambda **kwargs: calls.append(kwargs))

    try:
        client.invoke(_request(cost_limit_usd=0.000001))
    except ValueError as error:
        assert "reservation" in str(error)
    else:
        raise AssertionError("insufficient reservation must reject before dispatch")

    assert calls == []


def test_openai_adapter_rejects_unknown_priced_model_before_provider_call() -> None:
    calls = []
    client = _client_with_runtime(lambda **kwargs: calls.append(kwargs))
    client._model_ids[ModelTier.TERRA] = "unpriced-model"

    try:
        client.invoke(_request())
    except ValueError as error:
        assert "pricing" in str(error)
    else:
        raise AssertionError("unknown model price must reject before dispatch")

    assert calls == []


def test_openai_adapter_rejects_a_known_model_mapped_to_the_wrong_tier() -> None:
    calls = []
    client = _client_with_runtime(lambda **kwargs: calls.append(kwargs))
    client._model_ids[ModelTier.TERRA] = "gpt-5.6-sol"

    try:
        client.invoke(_request())
    except ValueError as error:
        assert "tier" in str(error)
    else:
        raise AssertionError("cross-tier model mapping must reject before dispatch")

    assert calls == []


def test_openai_adapter_rejects_nonfinite_reservation_before_provider_call() -> None:
    calls = []
    client = _client_with_runtime(lambda **kwargs: calls.append(kwargs))

    try:
        client.invoke(_request(cost_limit_usd=math.nan))
    except ValueError as error:
        assert "reservation" in str(error)
    else:
        raise AssertionError("non-finite reservation must reject before dispatch")

    assert calls == []


def test_openai_adapter_caps_output_tokens_to_the_remaining_reservation() -> None:
    captured = {}
    response = SimpleNamespace(
        id="resp-cap", model="gpt-5.6-terra", output_text='{"answer":"known"}', output=(),
        usage=SimpleNamespace(input_tokens=1, input_tokens_details=SimpleNamespace(cached_tokens=0), output_tokens=1),
    )
    client = _client_with_runtime(lambda **kwargs: captured.update(kwargs) or response)

    client.invoke(_request(cost_limit_usd=0.002))

    assert 0 < captured["max_output_tokens"] < 167


def test_openai_adapter_uses_the_pinned_long_price_for_its_output_cap() -> None:
    captured = {}
    response = SimpleNamespace(
        id="resp-long-cap", model="gpt-5.6-terra", output_text='{"answer":"known"}', output=(),
        usage=SimpleNamespace(input_tokens=1, input_tokens_details=SimpleNamespace(cached_tokens=0), output_tokens=1),
    )
    client = _client_with_runtime(lambda **kwargs: captured.update(kwargs) or response)

    client.invoke(_request(cost_limit_usd=0.003, pricing_band="long"))

    assert 0 < captured["max_output_tokens"] < 167


def test_model_admission_promotes_short_band_when_its_conservative_bound_is_long() -> None:
    messages = (("user", "x" * 271_800),)
    schema = '{"type":"object","additionalProperties":false}'
    admission = admit_model_cost(
        model_id="gpt-5.6-terra",
        pricing_band="short",
        messages=messages,
        output_schema_json=schema,
        cost_limit_usd=10.0,
    )

    assert pricing_band_for_rendered_request(messages, schema) == "long"
    assert admission.pricing_band == "long"
    assert 400_000 < admission.max_output_tokens < 500_000


def test_openai_adapter_settles_usage_at_the_promoted_admission_band() -> None:
    """Settling at the requested short rate would understate a promoted request."""
    response = SimpleNamespace(
        id="resp-promoted", model="gpt-5.6-terra", output_text='{"answer":"known"}', output=(),
        usage=SimpleNamespace(
            input_tokens=100_000,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
            output_tokens=10,
        ),
    )
    client = _client_with_runtime(lambda **_kwargs: response)
    request = ModelRequest(
        trace_id="1" * 32,
        model_tier=ModelTier.TERRA,
        messages=(("user", "x" * 271_800),),
        temperature=0.0,
        output_schema_id="answer-v1",
        output_schema_digest="a" * 64,
        output_schema_json='{"type":"object","additionalProperties":false}',
        deadline_epoch=10.0,
        attempt=0,
        cost_limit_usd=10.0,
        pricing_band="short",
    )

    result = client.invoke(request)
    expected = estimate_workflow_usage_cost(
        "gpt-5.6-terra", "long", UsageTokens(input_tokens=100_000, output_tokens=10),
    )

    assert result.usage.pricing_band == "long"
    assert Decimal(str(result.usage.cost_usd)) == expected
    assert result.usage.cost_usd <= request.cost_limit_usd


def test_openai_adapter_rejects_unexpected_provider_model_after_dispatch() -> None:
    """A nonempty provider model name is not sufficient for fixed-price settlement."""
    response = SimpleNamespace(
        id="resp-substituted", model="gpt-5.6-sol", output_text='{"answer":"known"}', output=(),
        usage=SimpleNamespace(
            input_tokens=1,
            input_tokens_details=SimpleNamespace(cached_tokens=0),
            output_tokens=1,
        ),
    )
    client = _client_with_runtime(lambda **_kwargs: response)

    with pytest.raises(ValueError, match="model.*priced identity"):
        client.invoke(_request())


def test_langchain_runtime_forwards_cap_and_disables_provider_retries(monkeypatch) -> None:
    from langchain import chat_models
    from market_agent.langchain_runtime import LangChainResponsesRuntime

    captured = {}

    class FakeModel:
        def invoke(self, _messages, **kwargs):
            captured["invoke"] = kwargs
            return SimpleNamespace(
                id="resp-runtime", content="{}", content_blocks=[],
                usage_metadata={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
                response_metadata={"model_name": "gpt-5.6-terra"},
            )

    def initialize(*_args, **kwargs):
        captured["initialize"] = kwargs
        return FakeModel()

    monkeypatch.setattr(chat_models, "init_chat_model", initialize)
    LangChainResponsesRuntime(api_key="test-key").create(
        timeout=1.0, model="gpt-5.6-terra", max_output_tokens=37,
        input=[{"role": "user", "content": [{"type": "input_text", "text": "payload"}]}],
    )

    assert captured["initialize"]["max_retries"] == 0
    assert captured["invoke"]["max_output_tokens"] == 37


@pytest.mark.parametrize("usage,metadata", [
    (None, {"model_name": "gpt-5.6-terra"}),
    ({}, {"model_name": "gpt-5.6-terra"}),
    ({"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}, {}),
    ({"input_tokens": "1", "output_tokens": 1, "total_tokens": 2},
     {"model_name": "gpt-5.6-terra"}),
    ({"input_tokens": 1, "output_tokens": True, "total_tokens": 2},
     {"model_name": "gpt-5.6-terra"}),
])
def test_langchain_to_priced_adapter_rejects_unverified_metadata(monkeypatch, usage, metadata):
    from langchain import chat_models
    from market_agent.langchain_runtime import LangChainResponsesRuntime

    class FakeModel:
        def invoke(self, _messages, **_kwargs):
            return SimpleNamespace(
                id="resp-runtime", content='{"answer":"known"}', content_blocks=[],
                usage_metadata=usage, response_metadata=metadata,
            )

    monkeypatch.setattr(chat_models, "init_chat_model", lambda *_args, **_kwargs: FakeModel())
    client = _client_with_runtime(
        LangChainResponsesRuntime(api_key="test-key").create,
    )
    with pytest.raises(ValueError):
        client.invoke(_request())


def test_langchain_to_priced_adapter_preserves_real_mapping_cache_usage(monkeypatch):
    from langchain import chat_models
    from market_agent.langchain_runtime import LangChainResponsesRuntime

    class FakeModel:
        def invoke(self, _messages, **_kwargs):
            return SimpleNamespace(
                id="resp-runtime", content='{"answer":"known"}', content_blocks=[],
                usage_metadata={
                    "input_tokens": 100, "output_tokens": 10, "total_tokens": 110,
                    "input_token_details": {"cache_read": 40},
                },
                response_metadata={"model_name": "gpt-5.6-terra"},
            )

    monkeypatch.setattr(chat_models, "init_chat_model", lambda *_args, **_kwargs: FakeModel())
    client = _client_with_runtime(
        LangChainResponsesRuntime(api_key="test-key").create,
    )
    response = client.invoke(_request())
    assert response.usage.input_tokens == 100
    assert response.usage.cached_input_tokens == 40
    assert Decimal(str(response.usage.cost_usd)) == estimate_workflow_usage_cost(
        "gpt-5.6-terra", "short",
        UsageTokens(input_tokens=100, cached_input_tokens=40, output_tokens=10),
    )


def test_openai_adapter_prices_mapping_web_search_items():
    response = SimpleNamespace(
        id="resp-mapping", model="gpt-5.6-terra", output_text='{"answer":"known"}',
        output=({"type": "web_search_call"}, {"type": "message"}),
        usage=SimpleNamespace(input_tokens=100, input_tokens_details={"cached_tokens": 40}, output_tokens=10),
    )
    client = _client_with_runtime(lambda **_kwargs: response)
    result = client.invoke(_request())
    assert result.usage.cached_input_tokens == 40
    assert result.usage.web_search_tool_calls == 1
    assert Decimal(str(result.usage.cost_usd)) == estimate_workflow_usage_cost(
        "gpt-5.6-terra", "short",
        UsageTokens(input_tokens=100, cached_input_tokens=40, output_tokens=10, web_search_tool_calls=1),
    )


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
