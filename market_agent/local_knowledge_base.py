"""Deterministic local-knowledge fallback with mandatory source citations."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable


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
    """Small local document collection; it never calls a model or external index."""

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
                (-len(query_tokens & _tokens(document.text)), document.document_id, document)
                for document in self._documents
                if query_tokens & _tokens(document.text)
            ),
            key=lambda candidate: (candidate[0], candidate[1]),
        )
        if not ranked:
            return None
        document = ranked[0][2]
        return LocalKnowledgeAnswer(answer=document.answer or document.text, citations=(document.document_id,))


def _tokens(value: str) -> frozenset[str]:
    if not isinstance(value, str):
        raise ValueError("knowledge queries must be text")
    return frozenset(re.findall(r"[a-z0-9]+", value.lower()))
