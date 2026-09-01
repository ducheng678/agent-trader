from __future__ import annotations

from datetime import datetime, timedelta
from hashlib import sha256
import json
from typing import Annotated, Iterable, Literal, Mapping

from pydantic import Field, StrictBool, StrictFloat, StringConstraints, field_validator, model_validator

from market_agent.workflow_contracts import ContextSummary, ContractModel, Digest, NonNegativeInt, OmittedSection, PositiveInt, ShortText, SourceFact, SummaryCompleteness, Text


_MAX_CANDIDATES = 200
_MAX_INPUT_BYTES = 524288
_MAX_OMITTED_IDS = 30
_MAX_UNCERTAINTIES = 20
_MAX_EVIDENCE = 50
_MAX_CONFLICT_QUESTIONS = 10
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
        if any(item.relation != "supporting" for item in self.supporting_evidence) or any(item.relation != "contradicting" for item in self.contradicting_evidence):
            raise ValueError("evidence must be placed in its matching relation container")
        return self


class CandidateManifestEntry(ContractModel):
    record_id: ShortText
    record_hash: Digest
    relevance: StrictFloat = Field(ge=0.0, le=1.0)
    conflict_group_id: ShortText | None = None
    canonical_byte_length: PositiveInt


def _limits_identity(max_records: int, max_bytes: int) -> dict[str, str]:
    return {"max_bytes": f"{max_bytes:012d}", "max_records": f"{max_records:02d}"}


def _manifest_envelope(entries: tuple[CandidateManifestEntry, ...], policy_version: str, max_records: int, max_bytes: int) -> dict[str, object]:
    return {"candidate_count": len(entries), "entries": [entry.model_dump(mode="json") for entry in entries], "limits": _limits_identity(max_records, max_bytes), "policy_version": policy_version}


def _manifest_hash(entries: tuple[CandidateManifestEntry, ...], policy_version: str, max_records: int, max_bytes: int) -> str:
    return _canonical_hash(_manifest_envelope(entries, policy_version, max_records, max_bytes))


def _manifest_byte_length(entries: tuple[CandidateManifestEntry, ...], policy_version: str, max_records: int, max_bytes: int) -> int:
    return len(_canonical_bytes(_manifest_envelope(entries, policy_version, max_records, max_bytes)))


def _raw_content_envelope(records: tuple[ContextRecord, ...], manifest_hash: str, policy_version: str, max_records: int, max_bytes: int, scope: Literal["full", "selected"]) -> dict[str, object]:
    return {"limits": _limits_identity(max_records, max_bytes), "manifest_hash": manifest_hash, "policy_version": policy_version, "record_count": len(records), "records": [record.model_dump(mode="json") for record in records], "scope": scope}


def _content_envelope_byte_length(entries: tuple[CandidateManifestEntry, ...], manifest_hash: str, policy_version: str, max_records: int, max_bytes: int, scope: Literal["full", "selected"]) -> int:
    empty_envelope = {"limits": _limits_identity(max_records, max_bytes), "manifest_hash": manifest_hash, "policy_version": policy_version, "record_count": len(entries), "records": [], "scope": scope}
    empty_length = len(_canonical_bytes(empty_envelope))
    if not entries:
        return empty_length
    return empty_length + sum(entry.canonical_byte_length for entry in entries) + len(entries) - 1


def _selected_byte_length(entries: tuple[CandidateManifestEntry, ...], manifest_hash: str, policy_version: str, max_records: int, max_bytes: int) -> int:
    return _content_envelope_byte_length(entries, manifest_hash, policy_version, max_records, max_bytes, "selected")


def _full_input_byte_length(entries: tuple[CandidateManifestEntry, ...], manifest_hash: str, policy_version: str, max_records: int, max_bytes: int) -> int:
    return _content_envelope_byte_length(entries, manifest_hash, policy_version, max_records, max_bytes, "full")


def _group_key(record_id: str, conflict_group_id: str | None) -> tuple[str, str]:
    return ("conflict", conflict_group_id) if conflict_group_id is not None else ("record", record_id)


def _group_entries(entries: Iterable[CandidateManifestEntry | ContextRecord]) -> dict[tuple[str, str], list[CandidateManifestEntry | ContextRecord]]:
    grouped: dict[tuple[str, str], list[CandidateManifestEntry | ContextRecord]] = {}
    for entry in entries:
        grouped.setdefault(_group_key(entry.record_id, entry.conflict_group_id), []).append(entry)
    return grouped


def _derive_inventory(entries: tuple[CandidateManifestEntry, ...], manifest_hash: str, policy_version: str, max_records: int, max_bytes: int) -> tuple[tuple[str, ...], tuple[str, ...], int]:
    grouped = _group_entries(entries)
    if any(key[0] == "conflict" and len(group) < 2 for key, group in grouped.items()):
        raise ValueError("manifest conflict groups require at least two members")
    ordered_groups = []
    for group in grouped.values():
        ordered = tuple(sorted(group, key=lambda item: (-item.relevance, item.record_id)))
        ordered_groups.append(ordered)
    ordered_groups.sort(key=lambda group: (-group[0].relevance, group[0].record_id))
    selected: list[CandidateManifestEntry] = []
    omitted: list[str] = []
    for group in ordered_groups:
        proposed = tuple(selected) + group
        proposed_length = _selected_byte_length(proposed, manifest_hash, policy_version, max_records, max_bytes)
        if len(proposed) <= max_records and proposed_length <= max_bytes:
            selected.extend(group)
        else:
            omitted.extend(entry.record_id for entry in group)
    selected_entries = tuple(selected)
    selected_length = _selected_byte_length(selected_entries, manifest_hash, policy_version, max_records, max_bytes)
    if selected_length > max_bytes:
        raise ValueError("max_bytes cannot hold the canonical empty selection envelope")
    return tuple(entry.record_id for entry in selected_entries), tuple(omitted), selected_length


def _raw_content_hash(records: tuple[ContextRecord, ...], manifest_hash: str, policy_version: str, max_records: int, max_bytes: int, scope: Literal["full", "selected"]) -> str:
    return _canonical_hash(_raw_content_envelope(records, manifest_hash, policy_version, max_records, max_bytes, scope))


class ContextSelection(ContractModel):
    records: tuple[ContextRecord, ...] = Field(default_factory=tuple, max_length=30)
    selected_ids: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=30)
    omitted_ids: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=_MAX_OMITTED_IDS)
    selected_count: NonNegativeInt
    omitted_count: NonNegativeInt
    omitted_ids_truncated: StrictBool = False
    selection_policy_version: Literal["context-selector-v2"] = _POLICY_VERSION
    max_records: NonNegativeInt
    max_bytes: NonNegativeInt
    selected_record_hash: Digest
    all_input_hash: Digest
    candidate_count: NonNegativeInt = 0
    candidate_manifest: tuple[CandidateManifestEntry, ...] = Field(default_factory=tuple, max_length=_MAX_CANDIDATES)
    manifest_hash: Digest
    manifest_byte_length: PositiveInt
    selected_byte_length: PositiveInt
    full_input_byte_length: PositiveInt

    @property
    def input_hash(self) -> str:
        return self.selected_record_hash

    @model_validator(mode="after")
    def validate_selection(self) -> ContextSelection:
        if not 1 <= self.max_records <= 30 or not 1 <= self.max_bytes <= 999999999999:
            raise ValueError("selection bounds are invalid")
        if self.candidate_count != len(self.candidate_manifest) or len({entry.record_id for entry in self.candidate_manifest}) != self.candidate_count:
            raise ValueError("candidate manifest count or identifiers are inconsistent")
        if self.candidate_manifest != tuple(sorted(self.candidate_manifest, key=lambda entry: entry.record_id)):
            raise ValueError("candidate manifest order is noncanonical")
        expected_manifest_hash = _manifest_hash(self.candidate_manifest, self.selection_policy_version, self.max_records, self.max_bytes)
        if self.manifest_hash != expected_manifest_hash or self.manifest_byte_length != _manifest_byte_length(self.candidate_manifest, self.selection_policy_version, self.max_records, self.max_bytes):
            raise ValueError("candidate manifest hash is inconsistent")
        if self.full_input_byte_length != _full_input_byte_length(self.candidate_manifest, self.manifest_hash, self.selection_policy_version, self.max_records, self.max_bytes):
            raise ValueError("full-input envelope identity is inconsistent")
        derived_selected, derived_omitted, selected_length = _derive_inventory(self.candidate_manifest, self.manifest_hash, self.selection_policy_version, self.max_records, self.max_bytes)
        canonical_omitted = derived_omitted[:_MAX_OMITTED_IDS]
        if self.selected_ids != derived_selected or self.selected_count != len(derived_selected):
            raise ValueError("selection identifiers and counts do not match policy")
        if self.omitted_ids != canonical_omitted or self.omitted_count != len(derived_omitted):
            raise ValueError("omitted identifiers and counts do not match policy")
        if self.omitted_ids_truncated != (len(derived_omitted) > len(canonical_omitted)):
            raise ValueError("omitted identifier truncation metadata is inconsistent")
        record_ids = tuple(record.record_id for record in self.records)
        if len(set(record_ids)) != len(record_ids) or record_ids != self.selected_ids:
            raise ValueError("selection identifiers and counts must match selected records")
        entries = {entry.record_id: entry for entry in self.candidate_manifest}
        if any(entries.get(record.record_id) is None or entries[record.record_id].record_hash != _canonical_hash(record.model_dump(mode="json")) or entries[record.record_id].canonical_byte_length != len(_canonical_bytes(record.model_dump(mode="json"))) or entries[record.record_id].relevance != record.relevance or entries[record.record_id].conflict_group_id != record.conflict_group_id for record in self.records):
            raise ValueError("selected records do not match candidate manifest")
        selected_bytes = _canonical_bytes(_raw_content_envelope(self.records, self.manifest_hash, self.selection_policy_version, self.max_records, self.max_bytes, "selected"))
        if self.selected_byte_length != selected_length or self.selected_byte_length != len(selected_bytes) or self.selected_record_hash != sha256(selected_bytes).hexdigest():
            raise ValueError("selection hash does not match canonical selected envelope")
        return self


_SUMMARY_IDENTITY_FIELDS = (
    "input_hash", "all_input_hash", "selected_ids", "omitted_ids", "selected_count", "omitted_count",
    "omitted_ids_truncated", "unreported_omitted_count", "selection_policy_version", "max_records", "max_bytes",
    "candidate_count", "candidate_manifest", "manifest_hash", "manifest_byte_length", "selected_byte_length", "full_input_byte_length",
    "uncertainty_markers", "omitted_uncertainty_count", "conflict_questions", "omitted_conflict_question_count", "conflicts",
    "supporting_evidence", "omitted_supporting_evidence_count", "contradicting_evidence", "omitted_contradicting_evidence_count",
)


def _summary_identity(summary: ContextSummary, values: Mapping[str, object]) -> str:
    normalized_summary = {key: value for key, value in summary.model_dump(mode="json").items() if key != "summary_id"}
    inventory = {key: values[key] for key in _SUMMARY_IDENTITY_FIELDS}
    return "summary-" + _canonical_hash({"normalized_summary": normalized_summary, "selection_inventory": inventory})[:32]


def _validate_evidence_registry(supporting_items: tuple[EvidenceReference, ...], contradicting_items: tuple[EvidenceReference, ...]) -> None:
    registry: dict[str, EvidenceReference] = {}
    for item in supporting_items + contradicting_items:
        prior = registry.get(item.evidence_id)
        if prior is not None and prior != item:
            raise ValueError("duplicate evidence identifiers must have identical contracts")
        registry[item.evidence_id] = item


class ContextHandoff(ContractModel):
    summary: ContextSummary
    input_hash: Digest
    all_input_hash: Digest
    output_hash: Digest
    selected_ids: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=30)
    omitted_ids: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=_MAX_OMITTED_IDS)
    selected_count: NonNegativeInt
    omitted_count: NonNegativeInt
    omitted_ids_truncated: StrictBool
    unreported_omitted_count: NonNegativeInt
    uncertainty_markers: tuple[Text, ...] = Field(default_factory=tuple, max_length=_MAX_UNCERTAINTIES)
    omitted_uncertainty_count: NonNegativeInt
    conflict_questions: tuple[Text, ...] = Field(default_factory=tuple, max_length=_MAX_CONFLICT_QUESTIONS)
    omitted_conflict_question_count: NonNegativeInt
    conflicts: tuple[ConflictGroup, ...] = Field(default_factory=tuple, max_length=20)
    supporting_evidence: tuple[EvidenceReference, ...] = Field(default_factory=tuple, max_length=_MAX_EVIDENCE)
    omitted_supporting_evidence_count: NonNegativeInt
    contradicting_evidence: tuple[EvidenceReference, ...] = Field(default_factory=tuple, max_length=_MAX_EVIDENCE)
    omitted_contradicting_evidence_count: NonNegativeInt
    selection_policy_version: Literal["context-selector-v2"]
    max_records: NonNegativeInt
    max_bytes: NonNegativeInt
    candidate_count: NonNegativeInt
    candidate_manifest: tuple[CandidateManifestEntry, ...] = Field(default_factory=tuple, max_length=_MAX_CANDIDATES)
    manifest_hash: Digest
    manifest_byte_length: PositiveInt
    selected_byte_length: PositiveInt
    full_input_byte_length: PositiveInt
    untrusted_data: Literal[True]

    @model_validator(mode="after")
    def validate_handoff(self) -> ContextHandoff:
        if not 1 <= self.max_records <= 30 or not 1 <= self.max_bytes <= 999999999999:
            raise ValueError("handoff selection bounds are invalid")
        if self.candidate_count != len(self.candidate_manifest) or self.candidate_manifest != tuple(sorted(self.candidate_manifest, key=lambda entry: entry.record_id)) or len({entry.record_id for entry in self.candidate_manifest}) != self.candidate_count:
            raise ValueError("handoff candidate manifest is inconsistent")
        expected_manifest_hash = _manifest_hash(self.candidate_manifest, self.selection_policy_version, self.max_records, self.max_bytes)
        if self.manifest_hash != expected_manifest_hash or self.manifest_byte_length != _manifest_byte_length(self.candidate_manifest, self.selection_policy_version, self.max_records, self.max_bytes):
            raise ValueError("handoff manifest hash is inconsistent")
        if self.full_input_byte_length != _full_input_byte_length(self.candidate_manifest, self.manifest_hash, self.selection_policy_version, self.max_records, self.max_bytes):
            raise ValueError("handoff full-input envelope is inconsistent")
        derived_selected, derived_omitted, selected_length = _derive_inventory(self.candidate_manifest, self.manifest_hash, self.selection_policy_version, self.max_records, self.max_bytes)
        if self.selected_ids != derived_selected or self.selected_count != len(derived_selected):
            raise ValueError("handoff selected inventory is inconsistent")
        if self.omitted_ids != derived_omitted[:_MAX_OMITTED_IDS] or self.omitted_count != len(derived_omitted):
            raise ValueError("handoff omitted inventory is inconsistent")
        if self.selected_byte_length != selected_length:
            raise ValueError("handoff selected envelope is inconsistent")
        if self.summary.source_record_hash != self.input_hash:
            raise ValueError("handoff identity contradicts summary selection")
        if self.omitted_count < len(self.omitted_ids) or self.unreported_omitted_count != self.omitted_count - len(self.omitted_ids):
            raise ValueError("handoff omitted metadata is inconsistent")
        if self.omitted_ids_truncated != (self.unreported_omitted_count > 0):
            raise ValueError("handoff omitted truncation metadata is inconsistent")
        group_ids = tuple(group.group_id for group in self.conflicts)
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("handoff conflict group identifiers must be unique")
        if self.conflicts != tuple(sorted(self.conflicts, key=lambda group: group.group_id)):
            raise ValueError("handoff conflict groups must use canonical order")
        expected_questions = tuple(sorted(f"unresolved conflict: {group.group_id}" for group in self.conflicts if group.unresolved))
        if self.conflict_questions != expected_questions[:_MAX_CONFLICT_QUESTIONS] or self.omitted_conflict_question_count != len(expected_questions) - len(self.conflict_questions):
            raise ValueError("handoff conflict-question aggregation is inconsistent")
        if len(self.conflict_questions) != len(set(self.conflict_questions)):
            raise ValueError("handoff conflict-question identifiers must be unique")
        manifest_entries = {entry.record_id: entry for entry in self.candidate_manifest}
        expected_conflict_members: dict[str, tuple[str, ...]] = {}
        for record_id in self.selected_ids:
            group_id = manifest_entries[record_id].conflict_group_id
            if group_id is not None:
                expected_conflict_members[group_id] = (*expected_conflict_members.get(group_id, ()), record_id)
        actual_conflict_members = {group.group_id: group.record_ids for group in self.conflicts}
        if actual_conflict_members != expected_conflict_members:
            raise ValueError("handoff conflicts do not match selected manifest groups")
        expected_summary_conflicts = tuple(f"{group.group_id}: {group.description}" for group in self.conflicts)
        expected_summary_questions = (self.uncertainty_markers + self.conflict_questions)[:_MAX_UNCERTAINTIES] or (("insufficient source evidence",) if not self.selected_ids else ())
        unresolved_count = sum(group.unresolved for group in self.conflicts)
        unresolved_sections = tuple(section for section in self.summary.omitted_sections if section.section == "unresolved_conflicts")
        expected_unresolved_sections = () if unresolved_count == 0 else (OmittedSection(section="unresolved_conflicts", count=unresolved_count),)
        if self.summary.conflicts != expected_summary_conflicts or self.summary.unresolved_questions != expected_summary_questions or unresolved_sections != expected_unresolved_sections:
            raise ValueError("handoff summary conflict inventory is inconsistent")
        for items, omitted, relation in ((self.supporting_evidence, self.omitted_supporting_evidence_count, "supporting"), (self.contradicting_evidence, self.omitted_contradicting_evidence_count, "contradicting")):
            if any(item.relation != relation for item in items) or items != tuple(sorted(items, key=lambda item: item.evidence_id)):
                raise ValueError("handoff evidence aggregation is noncanonical")
            if len({item.evidence_id for item in items}) != len(items):
                raise ValueError("handoff evidence identifiers must be unique")
            if (omitted > 0 and len(items) != _MAX_EVIDENCE) or (len(items) < _MAX_EVIDENCE and omitted != 0):
                raise ValueError("handoff evidence omission metadata is inconsistent")
        _validate_evidence_registry(self.supporting_evidence, self.contradicting_evidence)
        reserved_ids = {record_id for group in self.conflicts if group.unresolved for record_id in group.record_ids}
        if not reserved_ids.issubset({item.evidence_id for item in self.contradicting_evidence}):
            raise ValueError("handoff contradicting evidence is missing reserved conflict members")
        if self.uncertainty_markers != tuple(sorted(set(self.uncertainty_markers))) or (self.omitted_uncertainty_count > 0 and len(self.uncertainty_markers) != _MAX_UNCERTAINTIES) or (len(self.uncertainty_markers) < _MAX_UNCERTAINTIES and self.omitted_uncertainty_count != 0):
            raise ValueError("handoff uncertainty aggregation is inconsistent")
        if self.summary.summary_id != _summary_identity(self.summary, self.model_dump(mode="json")):
            raise ValueError("handoff summary identity is inconsistent")
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
        return ContextRecord.model_validate(record.model_dump(mode="python"))
    if "claim" in record:
        return ContextRecord.model_validate(record)
    raise ValueError("context records require a normalized structured claim")


def _record_sort_key(record: ContextRecord) -> tuple[float, str]:
    return (-record.relevance, record.record_id)


def select_context(source_records: Iterable[ContextRecord | Mapping[str, object]], *, max_records: int = 30, max_characters: int = 12000, max_bytes: int | None = None) -> ContextSelection:
    if isinstance(max_records, bool) or not 1 <= max_records <= 30:
        raise ValueError("max_records must be between 1 and 30")
    selected_max_bytes = max_characters if max_bytes is None else max_bytes
    if isinstance(selected_max_bytes, bool) or not 1 <= selected_max_bytes <= 999999999999:
        raise ValueError("max_bytes must be positive")
    normalized: list[ContextRecord] = []
    for record in source_records:
        if len(normalized) >= _MAX_CANDIDATES:
            raise ValueError("context candidate count exceeds limit")
        normalized_record = _normalize_record(record)
        normalized.append(normalized_record)
    if len({record.record_id for record in normalized}) != len(normalized):
        raise ValueError("context record identifiers must be unique")
    grouped = _group_entries(normalized)
    for group_key, group in grouped.items():
        if group_key[0] == "record":
            continue
        if len(group) < 2:
            raise ValueError("conflict groups require at least two members")
        if len({item.conflict_description for item in group}) != 1 or len({item.conflict_unresolved for item in group}) != 1:
            raise ValueError("conflict group metadata must be consistent")
    manifest = tuple(CandidateManifestEntry(record_id=record.record_id, record_hash=_canonical_hash(record.model_dump(mode="json")), relevance=record.relevance, conflict_group_id=record.conflict_group_id, canonical_byte_length=len(_canonical_bytes(record.model_dump(mode="json")))) for record in sorted(normalized, key=lambda item: item.record_id))
    manifest_hash = _manifest_hash(manifest, _POLICY_VERSION, max_records, selected_max_bytes)
    full_records = tuple(sorted(normalized, key=lambda item: item.record_id))
    full_input_envelope = _canonical_bytes(_raw_content_envelope(full_records, manifest_hash, _POLICY_VERSION, max_records, selected_max_bytes, "full"))
    full_input_byte_length = len(full_input_envelope)
    if full_input_byte_length > _MAX_INPUT_BYTES:
        raise ValueError("context full-input envelope exceeds byte limit")
    selected_ids, omitted, selected_byte_length = _derive_inventory(manifest, manifest_hash, _POLICY_VERSION, max_records, selected_max_bytes)
    records_by_id = {record.record_id: record for record in normalized}
    selected_tuple = tuple(records_by_id[record_id] for record_id in selected_ids)
    selected_envelope = _canonical_bytes(_raw_content_envelope(selected_tuple, manifest_hash, _POLICY_VERSION, max_records, selected_max_bytes, "selected"))
    reported_omitted = tuple(omitted[:_MAX_OMITTED_IDS])
    return ContextSelection(
        records=selected_tuple,
        selected_ids=selected_ids,
        omitted_ids=reported_omitted,
        selected_count=len(selected_tuple),
        omitted_count=len(omitted),
        omitted_ids_truncated=len(omitted) > len(reported_omitted),
        selection_policy_version=_POLICY_VERSION,
        max_records=max_records,
        max_bytes=selected_max_bytes,
        selected_record_hash=sha256(selected_envelope).hexdigest(),
        all_input_hash=sha256(full_input_envelope).hexdigest(),
        candidate_count=len(normalized),
        candidate_manifest=manifest,
        manifest_hash=manifest_hash,
        manifest_byte_length=_manifest_byte_length(manifest, _POLICY_VERSION, max_records, selected_max_bytes),
        selected_byte_length=selected_byte_length,
        full_input_byte_length=full_input_byte_length,
    )


def _conflicts(records: tuple[ContextRecord, ...]) -> tuple[ConflictGroup, ...]:
    groups: dict[str, list[ContextRecord]] = {}
    for record in records:
        if record.conflict_group_id is not None:
            groups.setdefault(record.conflict_group_id, []).append(record)
    return tuple(ConflictGroup(group_id=group_id, description=items[0].conflict_description or "conflict", unresolved=any(item.conflict_unresolved for item in items), record_ids=tuple(item.record_id for item in sorted(items, key=_record_sort_key))) for group_id, items in sorted(groups.items()))


def _dedupe_evidence(supporting_items: tuple[EvidenceReference, ...], contradicting_items: tuple[EvidenceReference, ...]) -> tuple[tuple[EvidenceReference, ...], tuple[EvidenceReference, ...]]:
    _validate_evidence_registry(supporting_items, contradicting_items)
    supporting = tuple(sorted({item.evidence_id: item for item in supporting_items}.values(), key=lambda item: item.evidence_id))
    contradicting = tuple(sorted({item.evidence_id: item for item in contradicting_items}.values(), key=lambda item: item.evidence_id))
    return supporting, contradicting


def summarize_context(selection: ContextSelection, *, workflow_id: str, trace_id: str, task_id: str, user_objective: str, immutable_constraints: tuple[str, ...] = (), summary_version: str = "v1") -> ContextHandoff:
    selection = ContextSelection.model_validate(selection.model_dump(mode="python"))
    facts = tuple(SourceFact(source_id=record.claim.source_id, observed_at=_utc_text(record.claim.observed_at), fact=("not " if record.claim.negated else "") + record.claim.value + (f" {record.claim.unit}" if record.claim.unit else "")) for record in sorted(selection.records, key=lambda record: (record.claim.source_id, record.record_id)))
    all_uncertainty = tuple(sorted({record.uncertainty for record in selection.records if record.uncertainty is not None}))
    uncertainty = all_uncertainty[:_MAX_UNCERTAINTIES]
    conflicts = _conflicts(selection.records)
    all_conflict_questions = tuple(sorted(f"unresolved conflict: {group.group_id}" for group in conflicts if group.unresolved))
    conflict_questions = all_conflict_questions[:_MAX_CONFLICT_QUESTIONS]
    incomplete = selection.omitted_count > 0 or not selection.records or bool(all_conflict_questions)
    omitted_sections_values: list[OmittedSection] = []
    if selection.omitted_count:
        omitted_sections_values.append(OmittedSection(section="source_records", count=selection.omitted_count))
    if all_conflict_questions:
        omitted_sections_values.append(OmittedSection(section="unresolved_conflicts", count=len(all_conflict_questions)))
    if not selection.records and not selection.omitted_count:
        omitted_sections_values.append(OmittedSection(section="source_records", count=0))
    omitted_sections = tuple(omitted_sections_values)
    combined_questions = uncertainty + conflict_questions
    unresolved_questions = combined_questions[:_MAX_UNCERTAINTIES] or (("insufficient source evidence",) if not selection.records else ())
    provisional_summary = ContextSummary(summary_id="pending", task_id=task_id, workflow_id=workflow_id, trace_id=trace_id, user_objective=user_objective, immutable_constraints=immutable_constraints, market_facts=facts, unresolved_questions=unresolved_questions, conflicts=tuple(f"{group.group_id}: {group.description}" for group in conflicts), omitted_sections=omitted_sections, token_estimate=sum(len(fact.fact.split()) for fact in facts), completeness=SummaryCompleteness.INCOMPLETE if incomplete else SummaryCompleteness.COMPLETE, summary_version=summary_version, source_record_hash=selection.selected_record_hash, source_references=tuple(fact.source_id for fact in facts))
    explicit_supporting = tuple(item for record in selection.records for item in record.supporting_evidence)
    explicit_contradictions = tuple(item for record in selection.records for item in record.contradicting_evidence)
    inferred_contradictions = tuple(EvidenceReference(evidence_id=record.record_id, source_id=record.claim.source_id, observed_at=record.claim.observed_at, relation="contradicting") for group in conflicts if group.unresolved for record in selection.records if record.conflict_group_id == group.group_id)
    all_supporting, all_contradicting = _dedupe_evidence(explicit_supporting, explicit_contradictions + inferred_contradictions)
    supporting = all_supporting[:_MAX_EVIDENCE]
    reserved_ids = {record_id for group in conflicts if group.unresolved for record_id in group.record_ids}
    contradiction_by_id = {item.evidence_id: item for item in all_contradicting}
    reserved = tuple(contradiction_by_id[record_id] for record_id in sorted(reserved_ids))
    other_contradictions = tuple(item for item in all_contradicting if item.evidence_id not in reserved_ids)
    contradicting = tuple(sorted(reserved + other_contradictions[:_MAX_EVIDENCE - len(reserved)], key=lambda item: item.evidence_id))
    base = {"summary": provisional_summary, "input_hash": selection.selected_record_hash, "all_input_hash": selection.all_input_hash, "selected_ids": selection.selected_ids, "omitted_ids": selection.omitted_ids, "selected_count": selection.selected_count, "omitted_count": selection.omitted_count, "omitted_ids_truncated": selection.omitted_ids_truncated, "unreported_omitted_count": selection.omitted_count - len(selection.omitted_ids), "uncertainty_markers": uncertainty, "omitted_uncertainty_count": len(all_uncertainty) - len(uncertainty), "conflict_questions": conflict_questions, "omitted_conflict_question_count": len(all_conflict_questions) - len(conflict_questions), "conflicts": conflicts, "supporting_evidence": supporting, "omitted_supporting_evidence_count": len(all_supporting) - len(supporting), "contradicting_evidence": contradicting, "omitted_contradicting_evidence_count": len(all_contradicting) - len(contradicting), "selection_policy_version": selection.selection_policy_version, "max_records": selection.max_records, "max_bytes": selection.max_bytes, "candidate_count": selection.candidate_count, "candidate_manifest": selection.candidate_manifest, "manifest_hash": selection.manifest_hash, "manifest_byte_length": selection.manifest_byte_length, "selected_byte_length": selection.selected_byte_length, "full_input_byte_length": selection.full_input_byte_length, "untrusted_data": True}
    provisional = ContextHandoff.model_construct(**base, output_hash="pending")
    summary_values = provisional.model_dump(mode="json")
    summary = provisional_summary.model_copy(update={"summary_id": _summary_identity(provisional_summary, summary_values)})
    base["summary"] = summary
    provisional = ContextHandoff.model_construct(**base, output_hash="pending")
    output_hash = _canonical_hash({key: value for key, value in provisional.model_dump(mode="json").items() if key != "output_hash"})
    handoff = ContextHandoff(**base, output_hash=output_hash)
    return ContextHandoff.model_validate(handoff.model_dump(mode="python"))
