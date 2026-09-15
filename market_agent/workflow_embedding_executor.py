"""Host-owned embedding execution, cancellation and immutable usage settlement."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import replace
from decimal import Decimal
from uuid import uuid4

from market_agent.workflow_embedding_client import (
    EmbeddingCancelled,
    EmbeddingClient,
    EmbeddingNotDispatched,
    EmbeddingRequest,
    EmbeddingResponse,
    admit_embedding_cost,
    check_embedding_active,
    validate_embedding_vector,
)
from market_agent.workflow_observation import AttemptUsage, CoreNodeName, TokenUsage


class BoundedEmbeddingExecutor:
    def __init__(self, client: EmbeddingClient, *, clock: Callable[[], float] = time.time) -> None:
        self._client = client
        self._clock = clock

    def execute(
        self, request: EmbeddingRequest, *, observation_callback: Callable[[AttemptUsage], object],
    ) -> EmbeddingResponse:
        if not callable(observation_callback):
            raise ValueError("embedding execution requires a usage observation callback")
        admission = admit_embedding_cost(request)
        started = self._clock()
        check_embedding_active(request, started)
        attempt_id = f"host-embedding-{uuid4().hex}"
        identity = dict(
            workflow_id=request.workflow_id, trace_id=request.trace_id, task_id=request.task_id,
            attempt=request.attempt, node=CoreNodeName(request.node), provider="openai",
            model_id=request.model_id, model_tier=None, pricing_version=request.pricing_version,
            pricing_model_id=request.model_id, pricing_band=None,
        )
        try:
            result = self._client.invoke(request)
            if type(result) is not EmbeddingResponse:
                raise ValueError("embedding provider did not return a typed response")
            result = replace(result)
        except EmbeddingNotDispatched:
            raise
        except Exception:
            observation_callback(AttemptUsage(
                **identity, provider_request_id=attempt_id, tokens=None,
                source="embedding_usage_unavailable",
                estimated_cost_usd=float(admission.reserved_cost_usd),
                latency_ms=max(0, int((self._clock() - started) * 1000)),
            ))
            raise
        observation_callback(AttemptUsage(
            **identity, provider_request_id=result.provider_request_id,
            tokens=TokenUsage(input_tokens=result.prompt_tokens, output_tokens=0),
            estimated_cost_usd=result.estimated_cost_usd, source="embedding_response",
            latency_ms=max(0, int((self._clock() - started) * 1000)),
        ))
        # Late/cancelled responses still cost money but cannot reach retrieval or
        # a cache. Keep callbacks outside the catch to avoid double settlement.
        check_embedding_active(request, self._clock(), before_dispatch=False)
        if (result.prompt_tokens > admission.input_tokens_upper_bound
                or Decimal(str(result.estimated_cost_usd)) > admission.reserved_cost_usd):
            raise ValueError("embedding provider usage exceeded its admitted reservation")
        vector = validate_embedding_vector(result.vector, request.dimensions)
        return replace(result, vector=vector)
