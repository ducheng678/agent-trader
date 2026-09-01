"""Trusted deterministic orchestration of separately audited promotion stages."""
from __future__ import annotations

from datetime import datetime
from typing import Unpack

from pydantic import AwareDatetime, TypeAdapter

from market_agent.workflow_long_term_memory import (
    KnowledgeRevision, MemoryAuthorityError, MemoryRepository, MutationContext,
    WriteArguments, content_hash,
)


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
