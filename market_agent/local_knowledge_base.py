"""Deterministic local-knowledge fallback with mandatory source citations."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "been", "by", "do", "does", "for",
    "from", "how", "i", "in", "is", "it", "me", "of", "on", "or", "that", "the",
    "their", "these", "this", "those", "to", "was", "were", "what", "when", "where",
    "which", "who", "why", "with", "you", "your",
})


@dataclass(frozen=True, slots=True)
class KnowledgeDocument:
    document_id: str
    text: str
    answer: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.document_id, str) or not self.document_id.strip():
            raise ValueError("knowledge document ID must be non-empty")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("knowledge document text must be non-empty")
        if self.answer is not None and (not isinstance(self.answer, str) or not self.answer.strip()):
            raise ValueError("knowledge document answer must be non-empty when present")


@dataclass(frozen=True, slots=True)
class LocalKnowledgeAnswer:
    answer: str
    citations: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.answer or not self.citations:
            raise ValueError("local knowledge answers require text and a local document citation")


class LocalKnowledgeBase:
    """Conservative lexical lookup for a local document collection.

    Every substantive query token must occur in one document, and configured
    answers must be extracts of that document. This intentionally abstains on
    paraphrases that would require semantic inference.
    """

    def __init__(self, documents: Iterable[KnowledgeDocument] = ()) -> None:
        self._documents = tuple(documents)
        identifiers = [document.document_id for document in self._documents]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("local knowledge document IDs must be unique")

    def lookup(self, query: str) -> LocalKnowledgeAnswer | None:
        query_tokens = _tokens(query)
        if not query_tokens:
            return None
        ranked = sorted(
            (
                document
                for document in self._documents
                if query_tokens <= _tokens(document.text) and _has_supported_answer(document)
            ),
            key=lambda document: document.document_id,
        )
        if not ranked:
            return None
        document = ranked[0]
        return LocalKnowledgeAnswer(answer=document.answer or document.text, citations=(document.document_id,))


def _tokens(value: str) -> frozenset[str]:
    if not isinstance(value, str):
        raise ValueError("knowledge queries must be text")
    return frozenset(re.findall(r"[a-z0-9]+", value.lower())) - _STOPWORDS


def _has_supported_answer(document: KnowledgeDocument) -> bool:
    if document.answer is None:
        return True
    answer = " ".join(document.answer.split()).casefold()
    text = " ".join(document.text.split()).casefold()
    return re.search(r"(?<!\w)" + re.escape(answer) + r"(?!\w)", text) is not None
