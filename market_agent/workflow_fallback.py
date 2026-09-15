"""One-way fallback decisions for model availability failures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from market_agent.local_knowledge_base import LocalKnowledgeAnswer, LocalKnowledgeBase
from market_agent.workflow_agent_contracts import ModelTier


@dataclass(frozen=True, slots=True)
class Downgrade:
    tier: ModelTier


@dataclass(frozen=True, slots=True)
class UseLocalKnowledge:
    pass


@dataclass(frozen=True, slots=True)
class Abstain:
    conclusion: str = "不知道"

    def __post_init__(self) -> None:
        if self.conclusion != "不知道":
            raise ValueError('fallback abstention conclusion must be exactly "不知道"')


class FallbackPolicy:
    """Return only the next lower tier, local knowledge, or the fixed abstention."""

    def __init__(
        self, permitted_tiers: Iterable[ModelTier], *, knowledge_base: LocalKnowledgeBase | None = None
    ) -> None:
        self._tiers = tuple(permitted_tiers)
        if len(set(self._tiers)) != len(self._tiers):
            raise ValueError("fallback model tiers cannot repeat")
        expected_order = tuple(tier for tier in (ModelTier.SOL, ModelTier.TERRA, ModelTier.LUNA) if tier in self._tiers)
        if self._tiers != expected_order:
            raise ValueError("fallback model tiers must be in strict descending order")
        self._knowledge_base = knowledge_base

    def next(self, current_tier: ModelTier | str | None, failure: Any) -> Downgrade | UseLocalKnowledge | Abstain:
        del failure  # Failure classification belongs to retry policy; fallback order is fixed.
        if current_tier is None or current_tier == "local_knowledge":
            return Abstain()
        try:
            current = ModelTier(current_tier)
        except ValueError:
            return Abstain()
        if current not in self._tiers:
            return Abstain()
        current_index = self._tiers.index(current)
        if current_index + 1 < len(self._tiers):
            return Downgrade(self._tiers[current_index + 1])
        if self._knowledge_base is not None:
            return UseLocalKnowledge()
        return Abstain()

    def resolve_local_knowledge(self, query: str) -> LocalKnowledgeAnswer | None:
        if self._knowledge_base is None:
            return None
        return self._knowledge_base.lookup(query)
