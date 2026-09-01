"""Deterministic retrieval and bounded untrusted data for the agent boundary.

No model or external embedding service runs here. A versioned query embedding
may be supplied by trusted composition; unversioned queries use normalized text.
Conflicts are explicit provenance links, not guessed semantic contradictions.
"""
from __future__ import annotations

import math
import re
import sqlite3
from datetime import datetime, timedelta
from typing import Annotated, Literal

from pydantic import AwareDatetime, Field, field_validator, model_validator

from market_agent.workflow_contracts import Digest, FiniteUnit, NonNegativeInt, PositiveInt, ShortText, Text
from market_agent.workflow_long_term_memory import (
    DecisionLesson, DecisionRecord, Embedding, EventRecord, Ids, KnowledgeRevision,
    Lifecycle, MemoryContract, MemoryIntegrityError, MemoryRepository, OutcomeRecord,
    Record, canonical_json, content_hash,
)


class MemoryQuery(MemoryContract):
    tenant_id: ShortText
    scope: ShortText = "default"
    task: Text
    applicability: Ids = ()
    now: AwareDatetime
    memory_schema_version: ShortText = "v1"
    model_version: ShortText = "none"
    vector_version: ShortText = "none"
    embedding: Embedding = ()
    max_age_seconds: PositiveInt = 86400
    top_k: Annotated[PositiveInt, Field(le=50)] = 5
    min_confidence: FiniteUnit = 0.6
    min_similarity: FiniteUnit = 0.15

    @field_validator("task", mode="before")
    @classmethod
    def normalize_task(cls, value):
        return " ".join(value.split()).casefold() if isinstance(value, str) else value

    @model_validator(mode="after")
    def embedding_versions_are_known(self):
        if self.embedding and (self.model_version == "none" or self.vector_version == "none"):
            raise ValueError("query embeddings require explicit model and vector versions")
        return self


class MemoryMatch(MemoryContract):
    record_id: ShortText
    kind: Literal["rule", "lesson"]
    text: Text
    evidence_ids: tuple[ShortText, ...]
    contradicting_evidence_ids: tuple[ShortText, ...] = ()
    confidence: FiniteUnit
    freshness_seconds: NonNegativeInt
    expires_at: AwareDatetime
    score: float


class RetrievalResult(MemoryContract):
    tenant_id: ShortText
    scope: ShortText
    now: AwareDatetime
    status: Literal["hit", "miss", "conflict", "failed"]
    matches: tuple[MemoryMatch, ...] = ()
    omissions: tuple[ShortText, ...] = ()
    omitted_count: NonNegativeInt = 0


class SummaryItem(MemoryContract):
    record_id: ShortText
    text: Text
    evidence_ids: tuple[ShortText, ...]


class CoreExperienceSummary(MemoryContract):
    """The only agent-facing memory value; render in dynamic USER content.

    ``as_dynamic_context`` is empty for every unsafe/no-memory state. No callers
    should place memory in a system prefix or treat its text as instructions.
    Trusted retrieval supplies issuance and the earliest supporting-record
    deadline. Consumers must compare these timestamps with their own clock;
    reported evidence age alone cannot authorize reuse of a cached summary.
    """
    trust: Literal["untrusted_memory"] = "untrusted_memory"
    tenant_id: ShortText
    scope: ShortText
    issued_at: AwareDatetime
    expires_at: AwareDatetime
    selected_ids: tuple[ShortText, ...] = ()
    evidence_ids: tuple[ShortText, ...] = ()
    contradicting_evidence_ids: tuple[ShortText, ...] = ()
    rules: tuple[SummaryItem, ...] = ()
    lessons: tuple[SummaryItem, ...] = ()
    confidence: FiniteUnit = 0.0
    freshness_seconds: NonNegativeInt = 0
    conflict_state: Literal["clear", "conflict", "no_memory", "failed"]
    omissions: tuple[ShortText, ...] = ()
    omitted_count: NonNegativeInt = 0
    summary_hash: Digest | None = None

    @model_validator(mode="after")
    def validate_summary_hash(self):
        if self.expires_at < self.issued_at:
            raise ValueError("summary expiry cannot precede issuance")
        digest = content_hash(self.model_dump(mode="json", exclude={"summary_hash"}))
        if self.summary_hash is not None and self.summary_hash != digest:
            raise ValueError("summary hash mismatch")
        object.__setattr__(self, "summary_hash", digest)
        return self

    def as_dynamic_context(self) -> str:
        if self.conflict_state != "clear" or not (self.rules or self.lessons):
            return ""
        return canonical_json(self.model_dump(mode="json"))


def _fresh(record: Record, query: MemoryQuery) -> bool:
    return (record.tenant_id == query.tenant_id and record.scope == query.scope
            and record.visibility == "tenant" and record.lifecycle is Lifecycle.ACTIVE
            and record.schema_version == query.memory_schema_version
            and 0 <= (query.now - record.observed_at).total_seconds() <= query.max_age_seconds
            and (record.expires_at is None or record.expires_at > query.now))


def _eligible_memory(record: KnowledgeRevision | DecisionLesson, query: MemoryQuery) -> bool:
    return (_fresh(record, query) and record.model_version == query.model_version
            and record.vector_version == query.vector_version
            and (not isinstance(record, KnowledgeRevision) or record.effective_at <= query.now)
            and {tag.casefold() for tag in record.applicability}.issubset(
                {tag.casefold() for tag in query.applicability}))


def _event_evidence(identifiers: tuple[str, ...], records: dict[str, Record],
                    query: MemoryQuery, *, observed_by: datetime | None = None) -> tuple[set[str], set[str]]:
    """Validate fresh event ancestry and collect original independent groups."""
    visiting: set[str] = set()
    roots: dict[str, set[str]] = {}
    pending = [(identifier, False) for identifier in reversed(identifiers)]
    while pending:
        identifier, expanded = pending.pop()
        evidence = records.get(identifier)
        if (not isinstance(evidence, EventRecord) or not _fresh(evidence, query)
                or (observed_by is not None and evidence.observed_at > observed_by)):
            raise MemoryIntegrityError("missing or ineligible evidence")
        if expanded:
            visiting.remove(identifier)
            groups: set[str] = set()
            if evidence.provenance.source_kind != "model":
                if evidence.provenance.derived_from:
                    for parent in evidence.provenance.derived_from:
                        groups.update(roots[parent])
                else:
                    groups.add(evidence.provenance.independent_group)
            roots[identifier] = groups
            continue
        if identifier in visiting:
            raise MemoryIntegrityError("circular evidence")
        if identifier in roots:
            continue
        visiting.add(identifier)
        pending.append((identifier, True))
        pending.extend((parent, False) for parent in reversed(evidence.provenance.derived_from))
    return set(roots), set().union(*(roots[identifier] for identifier in identifiers))


def _outcome_evidence(outcome: Record | None, records: dict[str, Record],
                      query: MemoryQuery) -> set[str]:
    if not isinstance(outcome, OutcomeRecord) or not outcome.verified or not _fresh(outcome, query):
        raise MemoryIntegrityError("outcome is not verifiable")
    decision = records.get(outcome.decision_id)
    if (not isinstance(decision, DecisionRecord) or not _fresh(decision, query)
            or decision.status != "final" or decision.observed_at > outcome.observed_at):
        raise MemoryIntegrityError("outcome decision is missing or mismatched")
    # Evidence must already exist when the outcome was verified and remain fresh
    # at retrieval time. Applying the bound during traversal includes ancestors.
    identifiers, roots = _event_evidence(outcome.evidence_ids, records, query,
                                        observed_by=outcome.observed_at)
    if not roots:
        raise MemoryIntegrityError("outcome has no independent observation")
    return identifiers


def _support(record: KnowledgeRevision | DecisionLesson, records: dict[str, Record],
             query: MemoryQuery) -> tuple[tuple[str, ...], float, bool]:
    identifiers, roots = _event_evidence(record.evidence_ids, records, query)
    verified = False
    if record.outcome_id is not None:
        outcome = records.get(record.outcome_id)
        outcome_ids = _outcome_evidence(outcome, records, query)
        if isinstance(record, DecisionLesson) and record.decision_id != outcome.decision_id:
            raise MemoryIntegrityError("outcome decision is missing or mismatched")
        if isinstance(record, KnowledgeRevision) and not set(outcome.evidence_ids) & set(record.evidence_ids):
            raise MemoryIntegrityError("outcome does not support candidate evidence")
        identifiers.update(outcome_ids)
        identifiers.add(outcome.record_id)
        verified = True
    elif len(roots) < 2:
        raise MemoryIntegrityError("independent corroboration is missing")
    authority = max((1.0 if records[identifier].provenance.source_kind == "external" else 0.8
                     for identifier in identifiers if isinstance(records[identifier], EventRecord)
                     and records[identifier].provenance.source_kind != "model"), default=0.0)
    return tuple(sorted(identifiers)), authority, verified


def _contradicting_evidence(record: KnowledgeRevision | DecisionLesson, records: dict[str, Record],
                            query: MemoryQuery) -> tuple[str, ...]:
    identifiers: set[str] = set()
    for identifier in getattr(record, "contradicting_ids", ()):
        target = records.get(identifier)
        if target is None or not _fresh(target, query):
            raise MemoryIntegrityError("missing or ineligible contradiction")
        if isinstance(target, EventRecord):
            evidence_ids, _ = _event_evidence((identifier,), records, query)
        elif isinstance(target, (KnowledgeRevision, DecisionLesson)):
            if not _eligible_memory(target, query):
                raise MemoryIntegrityError("ineligible contradiction")
            evidence_ids, _, _ = _support(target, records, query)
        elif isinstance(target, OutcomeRecord):
            evidence_ids = _outcome_evidence(target, records, query)
        else:
            raise MemoryIntegrityError("unsupported contradiction")
        identifiers.update(evidence_ids)
    return tuple(sorted(identifiers))


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.casefold()))


def _text_similarity(left: str, right: str) -> float:
    a, b = _tokens(left), _tokens(right)
    return len(a & b) / len(a | b) if a or b else 0.0


def _similarity(query: MemoryQuery, record: KnowledgeRevision | DecisionLesson, text: str) -> float | None:
    if not query.embedding:
        if query.model_version != "none" or query.vector_version != "none":
            return None
        return _text_similarity(query.task, text)
    if len(query.embedding) != len(record.embedding):
        return None
    # Normalize before multiplying to avoid overflowing finite large embeddings.
    norm_a, norm_b = math.hypot(*query.embedding), math.hypot(*record.embedding)
    if not norm_a or not norm_b or not math.isfinite(norm_a) or not math.isfinite(norm_b):
        return None
    return max(0.0, min(1.0, math.fsum((a / norm_a) * (b / norm_b)
                                       for a, b in zip(query.embedding, record.embedding))))


def retrieve_memory(query: MemoryQuery, repository: MemoryRepository) -> RetrievalResult:
    """Retrieve using ``query.now`` supplied by the trusted composition clock."""
    query = MemoryQuery.model_validate(query)
    base = dict(tenant_id=query.tenant_id, scope=query.scope, now=query.now)
    try:
        snapshot = repository.list_records(tenant_id=query.tenant_id)
    except (MemoryIntegrityError, OSError, sqlite3.DatabaseError):
        return RetrievalResult(**base, status="failed", omissions=("retrieval_failed",))
    # Recheck tenant even for adapter implementations; foreign IDs never become
    # evidence or diagnostics. A snapshot avoids per-record read inconsistencies.
    records = {record.record_id: record for record in snapshot if record.tenant_id == query.tenant_id}
    matches = []
    omissions: set[str] = set()
    omitted_count = 0
    for record in records.values():
        if not isinstance(record, (KnowledgeRevision, DecisionLesson)):
            continue
        if not _eligible_memory(record, query):
            omissions.add("ineligible")
            omitted_count += 1
            continue
        text = record.rule if isinstance(record, KnowledgeRevision) else record.lesson
        similarity = _similarity(query, record, text)
        if similarity is None or record.confidence < query.min_confidence or similarity < query.min_similarity:
            omissions.add("weak_match")
            omitted_count += 1
            continue
        try:
            evidence_ids, authority, verified = _support(record, records, query)
            contradicting = _contradicting_evidence(record, records, query)
        except MemoryIntegrityError:
            omissions.add("evidence_gap")
            omitted_count += 1
            continue
        dependencies = [record, *(records[identifier] for identifier in evidence_ids)]
        # A verified outcome depends on its final decision even when that
        # decision is not a displayed evidence citation.
        dependencies.extend(records[item.decision_id] for item in tuple(dependencies)
                            if isinstance(item, OutcomeRecord))
        oldest = min(item.observed_at for item in dependencies)
        expires_at = min(min(item.observed_at + timedelta(seconds=query.max_age_seconds),
                             item.expires_at or datetime.max.replace(tzinfo=query.now.tzinfo))
                         for item in dependencies)
        age = int((query.now - oldest).total_seconds())
        exact = query.task == " ".join(text.split()).casefold()
        score = (0.45 * similarity + 0.15 * exact + 0.1 * authority + 0.1 * record.confidence
                 + 0.1 * (1 - age / query.max_age_seconds) + 0.05 * verified
                 + 0.05 * bool(record.applicability) - 0.3 * bool(contradicting))
        matches.append(MemoryMatch(record_id=record.record_id,
                                    kind="rule" if isinstance(record, KnowledgeRevision) else "lesson",
                                    text=text, evidence_ids=evidence_ids,
                                    contradicting_evidence_ids=contradicting,
                                    confidence=record.confidence, freshness_seconds=age,
                                    expires_at=expires_at, score=round(score, 12)))
    # Inspect the full eligible set before Top-K so truncation cannot hide a
    # strong contradiction behind a higher ranked piece of advice.
    status = "conflict" if any(match.contradicting_evidence_ids for match in matches) else "hit" if matches else "miss"
    selected: list[MemoryMatch] = []
    while matches and len(selected) < query.top_k:
        def rank(match):
            duplicate = max((_text_similarity(match.text, prior.text) for prior in selected), default=0.0)
            # Preserve conflict diagnostics preferentially when no advice is safe.
            conflict_first = status == "conflict" and bool(match.contradicting_evidence_ids)
            return (-int(conflict_first), -round(match.score - 0.3 * duplicate, 12), match.record_id)
        best = min(matches, key=rank)
        selected.append(best)
        matches.remove(best)
    if matches:
        omissions.add("top_k")
        omitted_count += len(matches)
    return RetrievalResult(**base, status=status, matches=tuple(selected),
                           omissions=tuple(sorted(omissions)), omitted_count=omitted_count)


def _summary(result: RetrievalResult, matches: list[MemoryMatch], omissions: set[str],
             omitted_count: int) -> CoreExperienceSummary:
    clear = result.status == "hit" and bool(matches)
    state = "clear" if clear else "conflict" if result.status == "conflict" else "failed" if result.status == "failed" else "no_memory"
    values = dict(tenant_id=result.tenant_id, scope=result.scope,
                  issued_at=result.now, expires_at=min((m.expires_at for m in matches), default=result.now),
                  selected_ids=tuple(match.record_id for match in matches),
                  evidence_ids=tuple(sorted({identifier for match in matches for identifier in match.evidence_ids})),
                  contradicting_evidence_ids=tuple(sorted({identifier for match in matches for identifier in match.contradicting_evidence_ids})),
                  rules=tuple(SummaryItem(record_id=m.record_id, text=m.text, evidence_ids=m.evidence_ids)
                              for m in matches if clear and m.kind == "rule"),
                  lessons=tuple(SummaryItem(record_id=m.record_id, text=m.text, evidence_ids=m.evidence_ids)
                                for m in matches if clear and m.kind == "lesson"),
                  confidence=min((m.confidence for m in matches), default=0.0) if clear else 0.0,
                  freshness_seconds=max((m.freshness_seconds for m in matches), default=0),
                  conflict_state=state, omissions=tuple(sorted(omissions)), omitted_count=omitted_count)
    return CoreExperienceSummary(**values)


def build_core_experience_summary(result: RetrievalResult, token_budget: int) -> CoreExperienceSummary:
    """Keep whole cited entries within a conservative UTF-8-byte token bound.

    One UTF-8 byte per token is a safe upper estimate for byte-based tokenizers;
    metadata, escaping and the hash count too. Tiny budgets yield no injection.
    Conflict diagnostics are also truncated, although they never enter a prompt.
    """
    result = RetrievalResult.model_validate(result)
    if type(token_budget) is not int or token_budget < 0:
        raise ValueError("token_budget must be a nonnegative integer")
    omissions = set(result.omissions)
    kept = []
    omitted = result.omitted_count
    for match in result.matches:
        proposed = _summary(result, [*kept, match], omissions, omitted)
        size = len(canonical_json(proposed.model_dump(mode="json")).encode("utf-8"))
        if size <= token_budget:
            kept.append(match)
        else:
            omissions.add("token_budget")
            omitted += 1
    # An omission can grow metadata after accepting an entry; trim whole entries
    # again so the final serialization, not an earlier estimate, fits the bound.
    summary = _summary(result, kept, omissions, omitted)
    while kept and len(canonical_json(summary.model_dump(mode="json")).encode("utf-8")) > token_budget:
        kept.pop()
        omissions.add("token_budget")
        omitted += 1
        summary = _summary(result, kept, omissions, omitted)
    return summary
