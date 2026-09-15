"""Typed embedding transport and pinned pre-dispatch cost admission.

Workflow callers use BoundedEmbeddingExecutor so provider work enters the ledger.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from typing import Protocol
from uuid import uuid4

from market_agent.workflow_model_budget import _finite_positive_decimal


EMBEDDING_MODEL_ID = "text-embedding-3-small"
EMBEDDING_PRICING_VERSION = "openai-embedding-2026-09-15"
# https://developers.openai.com/api/docs/models/text-embedding-3-small
# Standard (non-batch) input price, pinned on 2026-09-15.
EMBEDDING_INPUT_PRICE_PER_MILLION = Decimal("0.02")


def embedding_cost(prompt_tokens: int) -> Decimal:
    if type(prompt_tokens) is not int or prompt_tokens <= 0:
        raise ValueError("embedding usage requires positive integer prompt tokens")
    return Decimal(prompt_tokens) * EMBEDDING_INPUT_PRICE_PER_MILLION / Decimal(1_000_000)


class EmbeddingCancelled(RuntimeError):
    """Cancellation prevents a vector from leaving the embedding executor."""


class EmbeddingNotDispatched(RuntimeError):
    """The transport guarantees no provider call occurred."""


class _CancelledBeforeDispatch(EmbeddingCancelled, EmbeddingNotDispatched):
    pass


class _ExpiredBeforeDispatch(TimeoutError, EmbeddingNotDispatched):
    pass


@dataclass(frozen=True, slots=True, kw_only=True)
class EmbeddingRequest:
    text: str
    workflow_id: str
    trace_id: str
    task_id: str
    deadline_epoch: float
    cost_limit_usd: float
    cancellation_check: Callable[[], bool] = lambda: False
    model_id: str = EMBEDDING_MODEL_ID
    dimensions: int = 1536
    pricing_version: str = EMBEDDING_PRICING_VERSION
    attempt: int = 0
    # Accept CoreNodeName (a str enum) without coupling pricing to observations.
    node: str = "plan"

    def __post_init__(self) -> None:
        if type(self.text) is not str or not self.text.strip():
            raise ValueError("embedding input must be non-empty text")
        for value in (self.workflow_id, self.trace_id, self.task_id):
            if type(value) is not str or not value.strip() or len(value) > 256:
                raise ValueError("embedding request requires bounded workflow identity")
        if self.model_id != EMBEDDING_MODEL_ID or self.pricing_version != EMBEDDING_PRICING_VERSION:
            raise ValueError("embedding request requires a supported pinned pricing identity")
        if type(self.dimensions) is not int or not 1 <= self.dimensions <= 1536:
            raise ValueError("embedding dimensions must be between 1 and 1536")
        if type(self.attempt) is not int or self.attempt < 0:
            raise ValueError("embedding attempt must be a nonnegative integer")
        if self.node not in {"plan", "dispatch", "recover", "decide", "reflect", "risk", "assemble"}:
            raise ValueError("embedding request requires a known owning graph node")
        if not callable(self.cancellation_check):
            raise ValueError("embedding request requires a cancellation check")
        _finite_positive_decimal(self.deadline_epoch, "embedding deadline")
        _finite_positive_decimal(self.cost_limit_usd, "embedding cost reservation")


@dataclass(frozen=True, slots=True, kw_only=True)
class EmbeddingResponse:
    vector: tuple[float, ...]
    prompt_tokens: int
    total_tokens: int
    model_id: str
    provider_request_id: str
    estimated_cost_usd: float
    pricing_version: str = EMBEDDING_PRICING_VERSION

    def __post_init__(self) -> None:
        cost = embedding_cost(self.prompt_tokens)
        if type(self.total_tokens) is not int or self.total_tokens != self.prompt_tokens:
            raise ValueError("embedding total tokens must equal prompt tokens")
        if self.model_id != EMBEDDING_MODEL_ID or self.pricing_version != EMBEDDING_PRICING_VERSION:
            raise ValueError("embedding response has unverified model or pricing identity")
        if (type(self.provider_request_id) is not str or not self.provider_request_id.strip()
                or len(self.provider_request_id) > 256):
            raise ValueError("embedding response requires a request identity")
        if type(self.estimated_cost_usd) is not float or Decimal(str(self.estimated_cost_usd)) != cost:
            raise ValueError("embedding response cost does not match pinned token pricing")
        # Validate vectors after recording usage: malformed vectors do not erase
        # successfully verified, billable provider usage.


@dataclass(frozen=True, slots=True)
class EmbeddingCostAdmission:
    input_tokens_upper_bound: int
    reserved_cost_usd: Decimal


def admit_embedding_cost(request: EmbeddingRequest) -> EmbeddingCostAdmission:
    if type(request) is not EmbeddingRequest:
        raise TypeError("embedding admission requires a typed request")
    # The byte-level tokenizer cannot produce more tokens than UTF-8 bytes.
    # Reject conservatively above the model's single-input context ceiling.
    tokens = len(request.text.encode("utf-8"))
    if tokens > 8192:
        raise ValueError("embedding input exceeds the bounded input allowance")
    cost = embedding_cost(tokens)
    if cost > _finite_positive_decimal(request.cost_limit_usd, "embedding cost reservation"):
        raise ValueError("embedding cost reservation is insufficient")
    return EmbeddingCostAdmission(tokens, cost)


def check_embedding_active(
    request: EmbeddingRequest, now: float, *, before_dispatch: bool = True,
) -> float:
    if request.cancellation_check():
        error = _CancelledBeforeDispatch if before_dispatch else EmbeddingCancelled
        raise error("embedding request cancelled")
    remaining = request.deadline_epoch - now
    if not math.isfinite(remaining) or remaining <= 0:
        error = _ExpiredBeforeDispatch if before_dispatch else TimeoutError
        raise error("embedding request deadline elapsed")
    return remaining


def validate_embedding_vector(vector: object, dimensions: int) -> tuple[float, ...]:
    if not isinstance(vector, tuple) or len(vector) != dimensions or any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in vector
    ) or not any(vector):
        raise ValueError("provider returned an invalid embedding")
    return tuple(float(value) for value in vector)


class EmbeddingClient(Protocol):
    def invoke(self, request: EmbeddingRequest) -> EmbeddingResponse: ...


class OpenAIEmbeddingClient:
    """Lazy single-dispatch provider transport; the executor owns settlement."""

    def __init__(self, *, api_key: str, model_id: str = EMBEDDING_MODEL_ID,
                 dimensions: int = 1536, clock: Callable[[], float] = time.time) -> None:
        if not isinstance(api_key, str) or not api_key.strip():
            raise ValueError("embedding client requires an API key")
        if model_id != EMBEDDING_MODEL_ID:
            raise ValueError("embedding model has no pinned supported price")
        if type(dimensions) is not int or not 1 <= dimensions <= 1536:
            raise ValueError("embedding dimensions must be between 1 and 1536")
        self._api_key = api_key
        self._model_id = model_id
        self._dimensions = dimensions
        self._clock = clock

    def invoke(self, request: EmbeddingRequest) -> EmbeddingResponse:
        admit_embedding_cost(request)
        if (request.model_id, request.dimensions) != (self._model_id, self._dimensions):
            raise EmbeddingNotDispatched("embedding request does not match configured transport")
        remaining = check_embedding_active(request, self._clock())
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key, timeout=remaining, max_retries=0)
        try:
            remaining = check_embedding_active(request, self._clock())
            response = client.embeddings.create(
                model=request.model_id, input=request.text, dimensions=request.dimensions,
                encoding_format="float", timeout=remaining,
            )
            model_id = getattr(response, "model", None)
            usage = getattr(response, "usage", None)
            prompt_tokens = getattr(usage, "prompt_tokens", None)
            total_tokens = getattr(usage, "total_tokens", None)
            # Embeddings have no JSON response id; the SDK exposes x-request-id.
            # If absent, identify this host attempt explicitly, without claiming
            # the generated identifier came from the provider.
            request_id = getattr(response, "_request_id", None) or f"host-embedding-{uuid4().hex}"
            data = getattr(response, "data", None)
            vector = ()
            if isinstance(data, (tuple, list)) and len(data) == 1 and getattr(data[0], "index", None) == 0:
                values = getattr(data[0], "embedding", None)
                if isinstance(values, (tuple, list)):
                    vector = tuple(values)
            return EmbeddingResponse(
                vector=vector, prompt_tokens=prompt_tokens, total_tokens=total_tokens,
                model_id=model_id, provider_request_id=request_id,
                estimated_cost_usd=float(embedding_cost(prompt_tokens)),
            )
        finally:
            client.close()

    def embed(self, text: str, *, deadline_epoch: float) -> tuple[float, ...]:
        """Legacy standalone API. Production workflows must use the executor."""
        request = EmbeddingRequest(
            text=text, workflow_id="standalone", trace_id="standalone",
            task_id="standalone-embedding", deadline_epoch=deadline_epoch,
            model_id=self._model_id, dimensions=self._dimensions,
            cost_limit_usd=float(embedding_cost(8192)),
        )
        response = self.invoke(request)
        check_embedding_active(request, self._clock(), before_dispatch=False)
        return validate_embedding_vector(response.vector, request.dimensions)
