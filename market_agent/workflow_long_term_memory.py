"""Trusted service contracts for durable memory; never agent instructions or handles."""
from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime
from enum import Enum
import hashlib
import json
import math
from types import MappingProxyType
from typing import Annotated, Any, Literal, Protocol, Self, TypedDict, Unpack

from pydantic import AwareDatetime, BeforeValidator, ConfigDict, Field, PlainSerializer, StrictFloat, model_validator

from market_agent.workflow_contracts import ContractModel, Digest, FiniteUnit, NonNegativeInt, PositiveInt, ShortText, Text


class MemoryIntegrityError(ValueError):
    """Stored memory or an artifact failed integrity verification."""


class MemoryAuthorityError(PermissionError):
    """A caller lacks service authority or the requested tenant scope."""


class MemoryConflictError(ValueError):
    """An idempotency key, immutable identity, or revision conflicts."""


class MemoryPromotionError(ValueError):
    """Evidence does not authorize a memory link or promotion."""


def freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        if any(type(key) is not str for key in value):
            raise ValueError("JSON object keys must be strings")
        return MappingProxyType({key: freeze_json(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item) for item in value)
    if value is None or type(value) in (str, int, bool):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise ValueError("memory payloads must contain only finite JSON values")


def thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def _payload(value: Any) -> Any:
    if not isinstance(value, Mapping):
        raise ValueError("payload must be a JSON object")
    return freeze_json(value)


def canonical_json(value: Any) -> str:
    return json.dumps(thaw_json(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


FrozenPayload = Annotated[Any, BeforeValidator(_payload), PlainSerializer(thaw_json)]
Ids = Annotated[tuple[ShortText, ...], Field(max_length=128)]
Embedding = Annotated[tuple[StrictFloat, ...], Field(max_length=4096)]


class MemoryContract(ContractModel):
    model_config = ConfigDict(validate_default=True)

    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        # Pydantic's default model_copy bypasses validation, including frozen fields.
        values = self.model_dump(mode="python", round_trip=True)
        values.update(update or {})
        return type(self).model_validate(values)


class Lifecycle(str, Enum):
    PROPOSED = "proposed"
    ACTIVE = "active"
    ARCHIVED = "archived"
    TOMBSTONED = "tombstoned"


class ArtifactReference(MemoryContract):
    tenant_id: ShortText
    sha256: Digest
    size_bytes: NonNegativeInt


class Provenance(MemoryContract):
    source_id: ShortText
    source_kind: Literal["external", "system", "model"]
    independent_group: ShortText
    derived_from: Ids = ()

    @model_validator(mode="after")
    def unique_sources(self) -> Self:
        if len(set(self.derived_from)) != len(self.derived_from):
            raise ValueError("provenance references must be unique")
        return self


class MemoryRecord(MemoryContract):
    record_id: ShortText
    tenant_id: ShortText
    observed_at: AwareDatetime
    expires_at: AwareDatetime | None = None
    lifecycle: Lifecycle = Lifecycle.ACTIVE
    visibility: Literal["tenant", "private"] = "tenant"
    scope: ShortText = "default"
    retention_class: Literal["standard", "short", "permanent"] = "standard"
    legal_hold: bool = False
    # Defaults preserve pre-retrieval stored JSON. Unknown versions never match
    # a query requesting a concrete embedding/model release.
    model_version: ShortText = "none"
    vector_version: ShortText = "none"
    embedding: Embedding = ()

    @model_validator(mode="after")
    def validate_times_and_links(self) -> Self:
        if self.expires_at is not None and self.expires_at <= self.observed_at:
            raise ValueError("expiry must follow observation")
        for field in ("evidence_ids", "lineage_ids"):
            ids = getattr(self, field, ())
            if self.record_id in ids or len(set(ids)) != len(ids):
                raise ValueError("memory references must be unique and cannot reference self")
        for field in ("decision_id", "outcome_id", "supersedes_id"):
            if getattr(self, field, None) == self.record_id:
                raise ValueError("memory cannot reference itself")
        return self


class EventRecord(MemoryRecord):
    source: ShortText
    payload: FrozenPayload
    payload_hash: Digest | None = None
    provenance: Provenance
    artifact: ArtifactReference | None = None

    @model_validator(mode="after")
    def validate_payload_integrity(self) -> Self:
        digest = content_hash(self.payload)
        if self.payload_hash is not None and self.payload_hash != digest:
            raise ValueError("event payload hash mismatch")
        object.__setattr__(self, "payload_hash", digest)
        if self.artifact is not None and self.artifact.tenant_id != self.tenant_id:
            raise ValueError("artifact and event must share tenant")
        if self.record_id in self.provenance.derived_from:
            raise ValueError("event provenance cannot reference itself")
        return self


class KnowledgeRevision(MemoryRecord):
    lifecycle: Lifecycle = Lifecycle.PROPOSED
    knowledge_id: ShortText
    revision: PositiveInt
    rule: Text
    applicability: Ids = ()
    confidence: FiniteUnit
    effective_at: AwareDatetime
    evidence_ids: Annotated[Ids, Field(min_length=1)]
    lineage_ids: Ids = ()
    outcome_id: ShortText | None = None
    contradicting_ids: Ids = ()

    @model_validator(mode="after")
    def validate_rule(self) -> Self:
        if self.expires_at is not None and self.effective_at >= self.expires_at:
            raise ValueError("effective time must precede expiry")
        if " ".join(self.rule.split()) != self.rule:
            raise ValueError("knowledge rules must have normalized whitespace")
        if self.record_id in self.contradicting_ids or set(self.evidence_ids) & set(self.contradicting_ids):
            raise ValueError("supporting and contradicting evidence cannot overlap or reference self")
        return self


class DecisionRecord(MemoryRecord):
    decision: Text
    status: Literal["provisional", "final"]
    evidence_ids: Ids = ()
    supersedes_id: ShortText | None = None


class OutcomeRecord(MemoryRecord):
    decision_id: ShortText
    result: Text
    verified: bool = False
    evidence_ids: Ids = ()

    @model_validator(mode="after")
    def verified_needs_evidence(self) -> Self:
        if self.verified and not self.evidence_ids:
            raise ValueError("verified outcomes require evidence")
        return self


class DecisionLesson(MemoryRecord):
    decision_id: ShortText
    outcome_id: ShortText
    lesson: Text
    evidence_ids: Annotated[Ids, Field(min_length=1)]
    applicability: Ids = ()
    confidence: FiniteUnit


Record = EventRecord | KnowledgeRevision | DecisionRecord | OutcomeRecord | DecisionLesson
RECORD_TYPES = {cls.__name__: cls for cls in (EventRecord, KnowledgeRevision, DecisionRecord, OutcomeRecord, DecisionLesson)}


class MemoryAudit(MemoryContract):
    sequence: PositiveInt
    tenant_id: ShortText
    trace_id: ShortText
    operation: ShortText
    record_id: ShortText
    idempotency_digest: Digest
    record_hash: Digest
    # No arbitrary payload, reason text, credentials, or raw idempotency key.


class MutationContext(MemoryContract):
    tenant_id: ShortText
    trace_id: ShortText
    idempotency_key: ShortText


class WriteArguments(TypedDict):
    tenant_id: str
    trace_id: str
    idempotency_key: str
    authority: object


def validate_authority(expected: object | None, authority: object, *, tenant_id: str,
                       trace_id: str, idempotency_key: str) -> MutationContext:
    if expected is None or authority is not expected:
        raise MemoryAuthorityError("memory mutation requires trusted service authority")
    return MutationContext(tenant_id=tenant_id, trace_id=trace_id, idempotency_key=idempotency_key)


class MemoryRepository(Protocol):
    def append_event(self, record: EventRecord, **context: Unpack[WriteArguments]) -> EventRecord: ...
    def propose_knowledge(self, record: KnowledgeRevision, **context: Unpack[WriteArguments]) -> KnowledgeRevision: ...
    def activate_knowledge(self, record_id: str, *, expected_revision: int, now: datetime,
                           **context: Unpack[WriteArguments]) -> KnowledgeRevision: ...
    def append_decision(self, record: DecisionRecord, **context: Unpack[WriteArguments]) -> DecisionRecord: ...
    def append_outcome(self, record: OutcomeRecord, **context: Unpack[WriteArguments]) -> OutcomeRecord: ...
    def link_lesson(self, record: DecisionLesson, **context: Unpack[WriteArguments]) -> DecisionLesson: ...
    def get_by_id(self, record_id: str, *, tenant_id: str) -> Record | None: ...
    def list_records(self, *, tenant_id: str) -> tuple[Record, ...]: ...
    def list_audit(self, *, tenant_id: str) -> tuple[MemoryAudit, ...]: ...


class ArtifactStore(Protocol):
    def put(self, data: bytes, **context: Unpack[WriteArguments]) -> ArtifactReference: ...
    def get(self, reference: ArtifactReference, *, tenant_id: str) -> bytes: ...
    def delete(self, reference: ArtifactReference, **context: Unpack[WriteArguments]) -> None: ...
