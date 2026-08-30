from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256
import json
from typing import Annotated, Iterable, Literal, Mapping

from pydantic import Field, StrictBool, StrictFloat, StringConstraints, field_validator, model_validator

from market_agent.workflow_contracts import ContextSummary, ContractModel, NonNegativeInt, OmittedSection, ShortText, SourceFact, SummaryCompleteness, Text


_MAX_CANDIDATES = 200
_MAX_INPUT_BYTES = 65536
_MAX_OMITTED_IDS = 30
_MAX_UNCERTAINTIES = 20
_POLICY_VERSION = "context-selector-v2"
ClaimText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]


def _require_utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError("context timestamps must be UTC")
    return value


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _canonical_hash(value: object) -> str:
    return sha256(_canonical_bytes(value)).hexdigest()


def _utc_text(value: datetime) -> str:
    return value.isoformat().replace("+00:00", "Z")


class NormalizedClaim(ContractModel):
    claim_id: ShortText
    source_id: ShortText
    observed_at: datetime
    value: ClaimText
    unit: ShortText | None = None
    negated: StrictBool
    untrusted_data: Literal[True]

    @field_validator("observed_at")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        return _require_utc(value)


class EvidenceReference(ContractModel):
    evidence_id: ShortText
    source_id: ShortText
    observed_at: datetime
    relation: Literal["supporting", "contradicting"]

    @field_validator("observed_at")
    @classmethod
    def require_utc_timestamp(cls, value: datetime) -> datetime:
        return _require_utc(value)


class ConflictGroup(ContractModel):
    group_id: ShortText
    description: Text
    unresolved: StrictBool
    record_ids: tuple[ShortText, ...] = Field(min_length=2, max_length=30)


class ContextRecord(ContractModel):
    record_id: ShortText
    claim: NormalizedClaim
    relevance: StrictFloat = Field(ge=0.0, le=1.0)
    uncertainty: Text | None = None
    supporting_evidence: tuple[EvidenceReference, ...] = Field(default_factory=tuple, max_length=20)
    contradicting_evidence: tuple[EvidenceReference, ...] = Field(default_factory=tuple, max_length=20)
    conflict_group_id: ShortText | None = None
    conflict_description: Text | None = None
    conflict_unresolved: StrictBool = False

    @model_validator(mode="after")
    def validate_conflict_fields(self) -> ContextRecord:
        if (self.conflict_group_id is None) != (self.conflict_description is None):
            raise ValueError("conflict identifier and description must be supplied together")
        if self.conflict_unresolved and self.conflict_group_id is None:
            raise ValueError("unresolved conflicts require a conflict group")
        return self


def _selected_hash(records: tuple[ContextRecord, ...], policy_version: str, max_records: int, max_bytes: int) -> str:
    return _canonical_hash({"policy_version": policy_version, "max_records": max_records, "max_bytes": max_bytes, "records": [record.model_dump(mode="json") for record in records]})


class ContextSelection(ContractModel):
    records: tuple[ContextRecord, ...] = Field(default_factory=tuple, max_length=30)
    selected_ids: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=30)
    omitted_ids: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=_MAX_OMITTED_IDS)
    selected_count: NonNegativeInt
    omitted_count: NonNegativeInt
    omitted_ids_truncated: StrictBool = False
    selection_policy_version: ShortText = _POLICY_VERSION
    max_records: NonNegativeInt
    max_bytes: NonNegativeInt
    selected_record_hash: ShortText
    all_input_hash: ShortText

    @property
    def input_hash(self) -> str:
        return self.selected_record_hash

    @model_validator(mode="after")
    def validate_selection(self) -> ContextSelection:
        record_ids = tuple(record.record_id for record in self.records)
        if len(set(record_ids)) != len(record_ids) or self.selected_ids != record_ids or self.selected_count != len(record_ids):
            raise ValueError("selection identifiers and counts must match selected records")
        if len(set(self.omitted_ids)) != len(self.omitted_ids) or set(self.selected_ids).intersection(self.omitted_ids):
            raise ValueError("selected and omitted identifiers must be unique and disjoint")
        if self.omitted_count < len(self.omitted_ids) or self.omitted_ids_truncated != (self.omitted_count > len(self.omitted_ids)):
            raise ValueError("omitted identifier truncation metadata is inconsistent")
        if not 1 <= self.max_records <= 30 or self.max_bytes < 1:
            raise ValueError("selection bounds are invalid")
        if self.selected_record_hash != _selected_hash(self.records, self.selection_policy_version, self.max_records, self.max_bytes):
            raise ValueError("selection hash does not match selected records")
        return self


class ContextHandoff(ContractModel):
    summary: ContextSummary
    input_hash: ShortText
    all_input_hash: ShortText
    output_hash: ShortText
    selected_ids: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=30)
    omitted_ids: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=_MAX_OMITTED_IDS)
    selected_count: NonNegativeInt
    omitted_count: NonNegativeInt
    omitted_ids_truncated: StrictBool
    unreported_omitted_count: NonNegativeInt
    uncertainty_markers: tuple[Text, ...] = Field(default_factory=tuple, max_length=_MAX_UNCERTAINTIES)
    omitted_uncertainty_count: NonNegativeInt
    conflicts: tuple[ConflictGroup, ...] = Field(default_factory=tuple, max_length=20)
    supporting_evidence: tuple[EvidenceReference, ...] = Field(default_factory=tuple, max_length=50)
    contradicting_evidence: tuple[EvidenceReference, ...] = Field(default_factory=tuple, max_length=50)
    selection_policy_version: ShortText
    untrusted_data: Literal[True]

    @model_validator(mode="after")
    def validate_handoff(self) -> ContextHandoff:
        if self.summary.source_record_hash != self.input_hash or self.selected_count != len(self.selected_ids):
            raise ValueError("handoff identity contradicts summary selection")
        if self.omitted_count < len(self.omitted_ids) or self.unreported_omitted_count != self.omitted_count - len(self.omitted_ids):
            raise ValueError("handoff omitted metadata is inconsistent")
        if self.omitted_ids_truncated != (self.unreported_omitted_count > 0):
            raise ValueError("handoff omitted truncation metadata is inconsistent")
        expected_output = _canonical_hash({key: value for key, value in self.model_dump(mode="json").items() if key != "output_hash"})
        if self.output_hash != expected_output:
            raise ValueError("handoff output hash does not match handoff content")
        return self


def _parse_legacy_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("legacy context timestamps must be strings")
    return _require_utc(datetime.fromisoformat(value.replace("Z", "+00:00")))


def _normalize_record(record: ContextRecord | Mapping[str, object]) -> ContextRecord:
    if isinstance(record, ContextRecord):
        return record
    if "claim" in record:
        return ContextRecord.model_validate(record)
    record_id = record.get("record_id")
    source_id = record.get("source_id")
    fact = record.get("fact")
    if not all(isinstance(value, str) for value in (record_id, source_id, fact)):
        raise ValueError("context records require a normalized claim")
    return ContextRecord(
        record_id=record_id,
        claim=NormalizedClaim(claim_id=record_id, source_id=source_id, observed_at=_parse_legacy_timestamp(record.get("observed_at")), value=fact, unit=None, negated=False, untrusted_data=True),
        relevance=record.get("relevance", 0.0),
        uncertainty=record.get("uncertainty"),
    )


def _record_sort_key(record: ContextRecord) -> tuple[float, str]:
    return (-record.relevance, record.record_id)


def select_context(source_records: Iterable[ContextRecord | Mapping[str, object]], *, max_records: int = 30, max_characters: int = 12000, max_bytes: int | None = None) -> ContextSelection:
    if isinstance(max_records, bool) or not 1 <= max_records <= 30:
        raise ValueError("max_records must be between 1 and 30")
    selected_max_bytes = max_characters if max_bytes is None else max_bytes
    if isinstance(selected_max_bytes, bool) or selected_max_bytes < 1:
        raise ValueError("max_bytes must be positive")
    normalized: list[ContextRecord] = []
    consumed_input_bytes = 0
    for record in source_records:
        if len(normalized) >= _MAX_CANDIDATES:
            raise ValueError("context candidate count exceeds limit")
        normalized_record = _normalize_record(record)
        consumed_input_bytes += len(_canonical_bytes(normalized_record.model_dump(mode="json")))
        if consumed_input_bytes > _MAX_INPUT_BYTES:
            raise ValueError("context input exceeds byte limit")
        normalized.append(normalized_record)
    if len({record.record_id for record in normalized}) != len(normalized):
        raise ValueError("context record identifiers must be unique")
    grouped: dict[str, list[ContextRecord]] = {}
    for record in normalized:
        grouped.setdefault(record.conflict_group_id or record.record_id, []).append(record)
    selected: list[ContextRecord] = []
    omitted: list[str] = []
    used_bytes = 0
    group_values = sorted(grouped.values(), key=lambda group: _record_sort_key(sorted(group, key=_record_sort_key)[0]))
    for group in group_values:
        ordered_group = sorted(group, key=_record_sort_key)
        group_bytes = sum(len(_canonical_bytes(record.model_dump(mode="json"))) for record in ordered_group)
        if len(selected) + len(ordered_group) <= max_records and used_bytes + group_bytes <= selected_max_bytes:
            selected.extend(ordered_group)
            used_bytes += group_bytes
        else:
            omitted.extend(record.record_id for record in ordered_group)
    selected_tuple = tuple(selected)
    reported_omitted = tuple(omitted[:_MAX_OMITTED_IDS])
    return ContextSelection(
        records=selected_tuple,
        selected_ids=tuple(record.record_id for record in selected_tuple),
        omitted_ids=reported_omitted,
        selected_count=len(selected_tuple),
        omitted_count=len(omitted),
        omitted_ids_truncated=len(omitted) > len(reported_omitted),
        selection_policy_version=_POLICY_VERSION,
        max_records=max_records,
        max_bytes=selected_max_bytes,
        selected_record_hash=_selected_hash(selected_tuple, _POLICY_VERSION, max_records, selected_max_bytes),
        all_input_hash=_canonical_hash({"policy_version": _POLICY_VERSION, "records": [record.model_dump(mode="json") for record in sorted(normalized, key=lambda item: item.record_id)]}),
    )


def _conflicts(records: tuple[ContextRecord, ...]) -> tuple[ConflictGroup, ...]:
    groups: dict[str, list[ContextRecord]] = {}
    for record in records:
        if record.conflict_group_id is not None:
            groups.setdefault(record.conflict_group_id, []).append(record)
    return tuple(ConflictGroup(group_id=group_id, description=items[0].conflict_description or "conflict", unresolved=any(item.conflict_unresolved for item in items), record_ids=tuple(item.record_id for item in sorted(items, key=_record_sort_key))) for group_id, items in sorted(groups.items()))


def _summary_id(selection: ContextSelection, workflow_id: str, trace_id: str, task_id: str, user_objective: str, immutable_constraints: tuple[str, ...], summary_version: str, uncertainty: tuple[str, ...], conflicts: tuple[ConflictGroup, ...]) -> str:
    return "summary-" + _canonical_hash({"workflow_id": workflow_id, "trace_id": trace_id, "task_id": task_id, "user_objective": user_objective, "immutable_constraints": immutable_constraints, "summary_version": summary_version, "selected_record_hash": selection.selected_record_hash, "all_input_hash": selection.all_input_hash, "selected_ids": selection.selected_ids, "omitted_ids": selection.omitted_ids, "selected_count": selection.selected_count, "omitted_count": selection.omitted_count, "uncertainty": uncertainty, "conflicts": [group.model_dump(mode="json") for group in conflicts], "policy": selection.selection_policy_version})[:16]


def summarize_context(selection: ContextSelection, *, workflow_id: str, trace_id: str, task_id: str, user_objective: str, immutable_constraints: tuple[str, ...] = (), summary_version: str = "v1") -> ContextHandoff:
    selection = ContextSelection.model_validate(selection.model_dump())
    expected_hash = _selected_hash(selection.records, selection.selection_policy_version, selection.max_records, selection.max_bytes)
    if selection.selected_record_hash != expected_hash:
        raise ValueError("selection hash does not match selected records")
    facts = tuple(SourceFact(source_id=record.claim.source_id, observed_at=_utc_text(record.claim.observed_at), fact=("not " if record.claim.negated else "") + record.claim.value + (f" {record.claim.unit}" if record.claim.unit else "")) for record in sorted(selection.records, key=lambda record: (record.claim.source_id, record.record_id)))
    all_uncertainty = tuple(sorted({record.uncertainty for record in selection.records if record.uncertainty is not None}))
    uncertainty = all_uncertainty[:_MAX_UNCERTAINTIES]
    conflicts = _conflicts(selection.records)
    unresolved = tuple(f"unresolved conflict: {group.group_id}" for group in conflicts if group.unresolved)
    incomplete = selection.omitted_count > 0 or not selection.records or bool(unresolved)
    omitted_sections = (OmittedSection(section="source_records", count=selection.omitted_count),) if incomplete else ()
    unresolved_questions = uncertainty + unresolved or (("insufficient source evidence",) if not selection.records else ())
    summary = ContextSummary(summary_id=_summary_id(selection, workflow_id, trace_id, task_id, user_objective, immutable_constraints, summary_version, uncertainty, conflicts), task_id=task_id, workflow_id=workflow_id, trace_id=trace_id, user_objective=user_objective, immutable_constraints=immutable_constraints, market_facts=facts, unresolved_questions=unresolved_questions, conflicts=tuple(f"{group.group_id}: {group.description}" for group in conflicts), omitted_sections=omitted_sections, token_estimate=sum(len(fact.fact.split()) for fact in facts), completeness=SummaryCompleteness.INCOMPLETE if incomplete else SummaryCompleteness.COMPLETE, summary_version=summary_version, source_record_hash=selection.selected_record_hash, source_references=tuple(fact.source_id for fact in facts))
    supporting = tuple(item for record in selection.records for item in record.supporting_evidence)
    explicit_contradictions = tuple(item for record in selection.records for item in record.contradicting_evidence)
    inferred_contradictions = tuple(EvidenceReference(evidence_id=record.record_id, source_id=record.claim.source_id, observed_at=record.claim.observed_at, relation="contradicting") for group in conflicts for record in selection.records if record.conflict_group_id == group.group_id and record.record_id != group.record_ids[0])
    contradicting = explicit_contradictions + inferred_contradictions
    base = {"summary": summary, "input_hash": selection.selected_record_hash, "all_input_hash": selection.all_input_hash, "selected_ids": selection.selected_ids, "omitted_ids": selection.omitted_ids, "selected_count": selection.selected_count, "omitted_count": selection.omitted_count, "omitted_ids_truncated": selection.omitted_ids_truncated, "unreported_omitted_count": selection.omitted_count - len(selection.omitted_ids), "uncertainty_markers": uncertainty, "omitted_uncertainty_count": len(all_uncertainty) - len(uncertainty), "conflicts": conflicts, "supporting_evidence": supporting, "contradicting_evidence": contradicting, "selection_policy_version": selection.selection_policy_version, "untrusted_data": True}
    provisional = ContextHandoff.model_construct(**base, output_hash="pending")
    output_hash = _canonical_hash({key: value for key, value in provisional.model_dump(mode="json").items() if key != "output_hash"})
    return ContextHandoff(**base, output_hash=output_hash)
