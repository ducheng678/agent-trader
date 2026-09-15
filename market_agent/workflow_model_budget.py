"""Pre-dispatch cost admission for the bounded OpenAI model adapter."""

from __future__ import annotations

import json
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_FLOOR
from typing import Literal

from market_agent.openai_usage import workflow_model_pricing


WORKFLOW_PRICING_VERSION = "openai-standard-2026-08-01"
_TOKENS_PER_MILLION = Decimal(1_000_000)
SHORT_CONTEXT_MAX_TOKENS = 272_000
# Responses API request envelopes add provider-owned keys around the rendered
# messages and schema.  Reserve this fixed allowance in addition to the exact
# UTF-8 JSON wire representation; cache discounts are intentionally ignored.
_REQUEST_FRAMING_TOKEN_ALLOWANCE = 256


@dataclass(frozen=True, slots=True)
class ModelCostAdmission:
    pricing_version: str
    pricing_band: Literal["short", "long"]
    input_tokens_upper_bound: int
    input_cost_usd: Decimal
    max_output_tokens: int


def _finite_positive_decimal(value: object, field_name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError(f"{field_name} must be a finite positive decimal")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"{field_name} must be a finite positive decimal") from exc
    if not result.is_finite() or result <= 0:
        raise ValueError(f"{field_name} must be a finite positive decimal")
    return result


def _serialized_input_upper_bound(
    messages: tuple[tuple[str, str], ...], output_schema_json: str,
) -> int:
    if not isinstance(messages, tuple) or not isinstance(output_schema_json, str):
        raise TypeError("model admission requires canonical messages and schema")
    wire_payload = {
        "input": [
            {"role": role, "content": [{"type": "input_text", "text": content}]}
            for role, content in messages
        ],
        "text": {"format": {"type": "json_schema", "schema": output_schema_json}},
    }
    try:
        serialized = json.dumps(
            wire_payload, ensure_ascii=True, separators=(",", ":"), allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("model admission requires serializable request content") from exc
    return len(serialized.encode("utf-8")) + _REQUEST_FRAMING_TOKEN_ALLOWANCE


def pricing_band_for_cost_admission(
    messages: tuple[tuple[str, str], ...], output_schema_json: str,
    requested_band: Literal["short", "long"] = "short",
) -> Literal["short", "long"]:
    """Never price a request as short when its conservative wire bound is long."""

    if requested_band not in {"short", "long"}:
        raise ValueError("workflow pricing band must be explicit")
    if requested_band == "long":
        return "long"
    return (
        "long"
        if _serialized_input_upper_bound(messages, output_schema_json) > SHORT_CONTEXT_MAX_TOKENS
        else "short"
    )


def admit_model_cost(
    *,
    model_id: str,
    pricing_band: Literal["short", "long"],
    messages: tuple[tuple[str, str], ...],
    output_schema_json: str,
    cost_limit_usd: object,
) -> ModelCostAdmission:
    """Reserve uncached input plus a bounded output before provider dispatch."""

    if not isinstance(model_id, str) or not model_id.strip():
        raise ValueError("model pricing is unavailable")
    input_tokens = _serialized_input_upper_bound(messages, output_schema_json)
    effective_band = pricing_band_for_cost_admission(
        messages, output_schema_json, pricing_band,
    )
    try:
        pricing = workflow_model_pricing(model_id, effective_band)
    except ValueError as exc:
        raise ValueError(f"model pricing is unavailable for {model_id}") from exc
    if not all(price.is_finite() and price >= 0 for price in (
        pricing.input, pricing.cached_input, pricing.cache_write, pricing.output,
    )) or pricing.output <= 0:
        raise ValueError("model pricing must be finite with positive output pricing")
    budget = _finite_positive_decimal(cost_limit_usd, "cost reservation")
    input_cost = Decimal(input_tokens) * pricing.input / _TOKENS_PER_MILLION
    remaining = budget - input_cost
    output_tokens = int((remaining * _TOKENS_PER_MILLION / pricing.output).to_integral_value(
        rounding=ROUND_FLOOR,
    ))
    if output_tokens < 1:
        raise ValueError("cost reservation is insufficient for the rendered request")
    return ModelCostAdmission(
        pricing_version=WORKFLOW_PRICING_VERSION,
        pricing_band=effective_band,
        input_tokens_upper_bound=input_tokens,
        input_cost_usd=input_cost,
        max_output_tokens=output_tokens,
    )
