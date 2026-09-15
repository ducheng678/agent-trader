from __future__ import annotations

import pytest

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


@pytest.mark.parametrize("query", ["What is the capital of France?", "the is", "supported refund deadline"])
def test_local_knowledge_rejects_stopword_and_partial_topic_overlap(query):
    """A shared filler word or unrelated topic must not become claimed evidence."""
    knowledge = LocalKnowledgeBase([KnowledgeDocument("policy-1", "The supported answer is stable.", "stable")])
    assert knowledge.lookup(query) is None


def test_local_knowledge_rejects_answer_not_supported_by_document_text():
    knowledge = LocalKnowledgeBase([KnowledgeDocument("policy-1", "The supported answer is stable.", "refunds are guaranteed")])
    assert knowledge.lookup("supported answer") is None


def test_local_knowledge_ignores_question_stopwords_when_topic_is_fully_supported():
    knowledge = LocalKnowledgeBase([KnowledgeDocument("policy-1", "The supported answer is stable.", "stable")])
    answer = knowledge.lookup("What is the supported answer?")
    assert answer is not None
    assert answer.answer == "stable"
    assert answer.citations == ("policy-1",)
