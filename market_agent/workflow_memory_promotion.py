"""Trusted deterministic orchestration of separately audited promotion stages."""
from __future__ import annotations

from datetime import datetime
from itertools import islice
from typing import Callable, Iterable, Literal, Unpack

from pydantic import AwareDatetime, TypeAdapter

from market_agent.workflow_long_term_memory import (
    EventRecord, KnowledgeRevision, Lifecycle, MemoryAuthorityError, MemoryConflictError,
    MemoryPromotionError, MemoryRepository, MutationContext, OutcomeRecord,
    WriteArguments, content_hash,
)
from market_agent.workflow_contracts import ContractModel, Digest, ShortText


class PromotionEvaluation(ContractModel):
    evaluation_id: Digest
    trace_id: ShortText
    candidate_id: ShortText
    status: Literal["promoted", "rejected"]
    reason_code: Literal[
        "promoted",
        "verified_outcome_required",
        "expired_or_forgotten",
        "contradictory_evidence",
        "evidence_mismatch",
        "repository_rejected",
        "cancelled",
    ]
    evaluated_at: AwareDatetime


class PromotionScheduler:
    """Bounded host scheduler; agent execution has no promotion capability.

    It deliberately evaluates candidates separately from accepted-result
    persistence.  The repository remains the authority for provenance and
    corroboration at activation time; this scheduler adds the workflow-specific
    verified-outcome and retention gates before making that host mutation.
    """

    def __init__(self, *, repository: MemoryRepository, authority: object,
                 tenant_id: str, max_candidates_per_run: int = 10,
                 evaluation_observer: Callable[[PromotionEvaluation], object]) -> None:
        if repository is None or authority is None or not tenant_id.strip():
            raise ValueError("promotion scheduler requires host-owned dependencies")
        if type(max_candidates_per_run) is not int or not 1 <= max_candidates_per_run <= 100:
            raise ValueError("promotion scheduler candidate limit is invalid")
        if not callable(evaluation_observer):
            raise ValueError("promotion evaluation requires a host audit observer")
        self._repository = repository
        self._authority = authority
        self._tenant_id = tenant_id
        self._max_candidates = max_candidates_per_run
        self._observer = evaluation_observer
        self._promoted: dict[str, KnowledgeRevision] = {}

    def evaluate(self, candidates: Iterable[KnowledgeRevision], *, now: datetime,
                 trace_id: str,
                 cancellation_check: Callable[[], bool] = lambda: False,
                 ) -> tuple[KnowledgeRevision, ...]:
        now = TypeAdapter(AwareDatetime).validate_python(now, strict=True)
        if type(trace_id) is not str or not trace_id.strip():
            raise ValueError("promotion trace ID is invalid")
        selected = tuple(KnowledgeRevision.model_validate(value) for value in
                         islice(candidates, self._max_candidates))
        promoted: list[KnowledgeRevision] = []
        for candidate in selected:
            evaluation_id = canonical_evaluation_key(trace_id, candidate, now)
            prior = self._promoted.get(evaluation_id)
            if prior is not None:
                promoted.append(prior)
                continue
            if self._cancelled(cancellation_check):
                self._emit(candidate, trace_id, now, evaluation_id,
                           "rejected", "cancelled")
                break
            reason = self._ineligible_reason(candidate, now)
            if reason is not None:
                self._emit(candidate, trace_id, now, evaluation_id, "rejected", reason)
                continue
            if self._cancelled(cancellation_check):
                self._emit(candidate, trace_id, now, evaluation_id,
                           "rejected", "cancelled")
                break
            key = content_hash({"trace_id": trace_id, "candidate_id": candidate.record_id})
            try:
                active = promote_candidate(
                    candidate, self._repository, now=now,
                    tenant_id=self._tenant_id, trace_id=trace_id,
                    idempotency_key="scheduler-" + key, authority=self._authority,
                )
            except (MemoryPromotionError, MemoryConflictError):
                self._emit(candidate, trace_id, now, evaluation_id,
                           "rejected", "repository_rejected")
                continue
            self._emit(candidate, trace_id, now, evaluation_id,
                       "promoted", "promoted")
            self._promoted[evaluation_id] = active
            promoted.append(active)
        return tuple(promoted)

    @staticmethod
    def _cancelled(check: Callable[[], bool]) -> bool:
        try:
            return bool(check())
        except Exception:
            return True

    def _ineligible_reason(self, candidate: KnowledgeRevision, now: datetime) -> str | None:
        if candidate.tenant_id != self._tenant_id or candidate.outcome_id is None:
            return "verified_outcome_required"
        if (candidate.lifecycle is not Lifecycle.PROPOSED
                or candidate.effective_at > now
                or (candidate.expires_at is not None and candidate.expires_at <= now)):
            return "expired_or_forgotten"
        if candidate.contradicting_ids:
            return "contradictory_evidence"
        outcome = self._repository.get_by_id(candidate.outcome_id, tenant_id=self._tenant_id)
        if not isinstance(outcome, OutcomeRecord) or not outcome.verified:
            return "verified_outcome_required"
        if (outcome.lifecycle is not Lifecycle.ACTIVE or outcome.observed_at > now
                or (outcome.expires_at is not None and outcome.expires_at <= now)):
            return "expired_or_forgotten"
        # The candidate cannot upgrade a weaker or unrelated evidence set.
        if not set(outcome.evidence_ids) <= set(candidate.evidence_ids):
            return "evidence_mismatch"
        for identifier in candidate.evidence_ids:
            evidence = self._repository.get_by_id(identifier, tenant_id=self._tenant_id)
            if (evidence is None or evidence.lifecycle is not Lifecycle.ACTIVE
                    or evidence.observed_at > now
                    or (evidence.expires_at is not None and evidence.expires_at <= now)):
                return "expired_or_forgotten"
        if len(self._independent_groups(candidate.evidence_ids)) < 2:
            return "repository_rejected"
        return None

    def _independent_groups(self, identifiers: tuple[str, ...]) -> set[str]:
        groups: set[str] = set()
        pending = list(identifiers)
        seen: set[str] = set()
        while pending and len(seen) <= 128:
            identifier = pending.pop()
            if identifier in seen:
                continue
            seen.add(identifier)
            record = self._repository.get_by_id(identifier, tenant_id=self._tenant_id)
            if not isinstance(record, EventRecord):
                continue
            provenance = record.provenance
            if provenance.source_kind == "model":
                continue
            if provenance.derived_from:
                pending.extend(provenance.derived_from)
            else:
                groups.add(provenance.independent_group)
        return groups

    def _emit(self, candidate: KnowledgeRevision, trace_id: str, now: datetime,
              evaluation_id: str, status: str, reason_code: str) -> None:
        self._observer(PromotionEvaluation(
            evaluation_id=evaluation_id,
            trace_id=trace_id,
            candidate_id=candidate.record_id,
            status=status,
            reason_code=reason_code,
            evaluated_at=now,
        ))


def canonical_evaluation_key(trace_id: str, candidate: KnowledgeRevision,
                             now: datetime) -> str:
    return content_hash({
        "service": "promotion_scheduler_v1",
        "trace_id": trace_id,
        "candidate": candidate.model_dump(mode="json"),
        "evaluated_at": now.isoformat(),
    })


def promote_candidate(candidate: KnowledgeRevision, repository: MemoryRepository, *,
                      now: datetime, **context: Unpack[WriteArguments]) -> KnowledgeRevision:
    """Propose then activate using the repository's transactional evidence gates.

    A rejected activation may leave its separately audited proposal for review.
    Replaying the same trace/key/time resumes both stages without extra audit
    events. The caller must retain service authority; this grants none itself.
    """
    candidate = KnowledgeRevision.model_validate(candidate)
    now = TypeAdapter(AwareDatetime).validate_python(now, strict=True)
    mutation = MutationContext(**{name: context[name] for name in
                                  ("tenant_id", "trace_id", "idempotency_key")})
    if candidate.tenant_id != mutation.tenant_id:
        raise MemoryAuthorityError("mutation tenant does not match candidate")
    # Hashing stage namespaces supports maximum-length user keys without
    # truncation collisions and keeps raw caller keys out of repository audits.
    proposal_context = dict(context, idempotency_key=content_hash({
        "service": "memory_promotion", "stage": "proposal", "key": mutation.idempotency_key}))
    activation_context = dict(context, idempotency_key=content_hash({
        "service": "memory_promotion", "stage": "activation", "key": mutation.idempotency_key}))
    proposed = repository.propose_knowledge(candidate, **proposal_context)
    return repository.activate_knowledge(proposed.record_id, expected_revision=proposed.revision,
                                         now=now, **activation_context)
