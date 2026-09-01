from __future__ import annotations

from market_agent.local_knowledge_base import KnowledgeDocument, LocalKnowledgeBase
from market_agent.workflow_agent_contracts import ModelTier
from market_agent.workflow_fallback import Abstain, Downgrade, FallbackPolicy, UseLocalKnowledge


def test_fallback_only_downgrades_to_the_next_lower_permitted_tier():
    """An upward fallback could spend more or bypass the allowed model policy."""
    fallback = FallbackPolicy((ModelTier.SOL, ModelTier.TERRA, ModelTier.LUNA))

    assert fallback.next(ModelTier.SOL, "unavailable") == Downgrade(ModelTier.TERRA)
    assert fallback.next(ModelTier.TERRA, "unavailable") == Downgrade(ModelTier.LUNA)


def test_fallback_uses_cited_local_knowledge_before_exact_abstention():
    """Returning uncited local text or skipping the terminal abstention hides evidence gaps."""
    knowledge = LocalKnowledgeBase(
        [KnowledgeDocument(document_id="policy-1", text="The supported answer is stable.", answer="stable")]
    )
    fallback = FallbackPolicy((ModelTier.LUNA,), knowledge_base=knowledge)

    assert fallback.next(ModelTier.LUNA, "unavailable") == UseLocalKnowledge()
    answer = fallback.resolve_local_knowledge("supported answer")
    assert answer is not None
    assert answer.citations == ("policy-1",)
    assert fallback.next("local_knowledge", "no_match") == Abstain("不知道")


def test_fallback_ends_with_the_exact_abstention_when_no_tier_remains():
    """Changing the final wording would break the schema-valid unknown conclusion."""
    fallback = FallbackPolicy(())

    assert fallback.next(None, "unavailable") == Abstain("不知道")
