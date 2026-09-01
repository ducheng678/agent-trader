"""Lazy, versioned embedding boundary owned by production composition."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Protocol


class EmbeddingClient(Protocol):
    def embed(self, text: str, *, deadline_epoch: float) -> tuple[float, ...]: ...


class OpenAIEmbeddingClient:
    """Create the provider client only when a bounded embedding is requested."""

    def __init__(self, *, api_key: str, model_id: str, dimensions: int,
                 clock: Callable[[], float] = time.time) -> None:
        if not api_key.strip() or not model_id.strip():
            raise ValueError("embedding client requires an API key and model identifier")
        if type(dimensions) is not int or not 1 <= dimensions <= 2000:
            raise ValueError("embedding dimensions must be between 1 and 2000")
        self._api_key = api_key
        self._model_id = model_id
        self._dimensions = dimensions
        self._clock = clock

    def embed(self, text: str, *, deadline_epoch: float) -> tuple[float, ...]:
        if type(text) is not str or not text.strip():
            raise ValueError("embedding input must be non-empty text")
        remaining = deadline_epoch - self._clock()
        if not math.isfinite(remaining) or remaining <= 0:
            raise TimeoutError("embedding request deadline elapsed")
        from openai import OpenAI

        client = OpenAI(api_key=self._api_key, timeout=remaining, max_retries=0)
        response = client.embeddings.create(
            model=self._model_id,
            input=text,
            dimensions=self._dimensions,
        )
        values = tuple(float(value) for value in response.data[0].embedding)
        if len(values) != self._dimensions or not all(math.isfinite(value) for value in values) or not any(values):
            raise ValueError("provider returned an invalid embedding")
        return values
