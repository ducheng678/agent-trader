from __future__ import annotations

from hashlib import sha256
import json
from typing import Iterable, Mapping

from pydantic import Field

from market_agent.workflow_contracts import ContextSummary, ContractModel, NonNegativeInt, OmittedSection, ShortText, SourceFact, SummaryCompleteness, Text


class ContextRecord(ContractModel):
    record_id: ShortText
    source_id: ShortText
    observed_at: ShortText
    fact: Text
    relevance: float = Field(ge=0.0, le=1.0)
    uncertainty: Text | None = None


class ContextSelection(ContractModel):
    records: tuple[ContextRecord, ...] = Field(default_factory=tuple, max_length=30)
    selected_ids: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=30)
    omitted_ids: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=200)
    selected_count: NonNegativeInt
    omitted_count: NonNegativeInt
    input_hash: ShortText


class ContextHandoff(ContractModel):
    summary: ContextSummary
    input_hash: ShortText
    output_hash: ShortText
    selected_ids: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=30)
    omitted_ids: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=200)
    selected_count: NonNegativeInt
    omitted_count: NonNegativeInt


def _canonical_hash(value: object) -> str:
    rendered = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return sha256(rendered.encode("utf-8")).hexdigest()


def _record_sort_key(record: ContextRecord) -> tuple[float, str, str]:
    return (-record.relevance, record.record_id, record.source_id)


def select_context(
    source_records: Iterable[ContextRecord | Mapping[str, object]],
    *,
    max_records: int = 30,
    max_characters: int = 12000,
) -> ContextSelection:
    if isinstance(max_records, bool) or max_records < 1:
        raise ValueError("max_records must be positive")
    if isinstance(max_characters, bool) or max_characters < 1:
        raise ValueError("max_characters must be positive")
    normalized = [record if isinstance(record, ContextRecord) else ContextRecord.model_validate(record) for record in source_records]
    if len({record.record_id for record in normalized}) != len(normalized):
        raise ValueError("context record identifiers must be unique")
    ordered = sorted(normalized, key=_record_sort_key)
    selected: list[ContextRecord] = []
    omitted: list[str] = []
    consumed_characters = 0
    for record in ordered:
        next_length = consumed_characters + len(record.fact)
        if len(selected) < max_records and next_length <= max_characters:
            selected.append(record)
            consumed_characters = next_length
        else:
            omitted.append(record.record_id)
    input_hash = _canonical_hash([record.model_dump(mode="json") for record in sorted(normalized, key=lambda item: item.record_id)])
    return ContextSelection(
        records=tuple(selected),
        selected_ids=tuple(record.record_id for record in selected),
        omitted_ids=tuple(omitted),
        selected_count=len(selected),
        omitted_count=len(omitted),
        input_hash=input_hash,
    )


def summarize_context(
    selection: ContextSelection,
    *,
    workflow_id: str,
    trace_id: str,
    task_id: str,
    user_objective: str,
    immutable_constraints: tuple[str, ...] = (),
    summary_version: str = "v1",
) -> ContextHandoff:
    facts = tuple(
        SourceFact(source_id=record.source_id, observed_at=record.observed_at, fact=record.fact)
        for record in sorted(selection.records, key=lambda record: (record.source_id, record.record_id))
    )
    uncertainty = tuple(sorted({record.uncertainty for record in selection.records if record.uncertainty is not None}))
    incomplete = selection.omitted_count > 0 or not selection.records
    omitted_sections = (OmittedSection(section="source_records", count=selection.omitted_count),) if incomplete else ()
    unresolved_questions = uncertainty or (("insufficient source evidence",) if not selection.records else ())
    summary_hash_seed = {
        "workflow_id": workflow_id,
        "trace_id": trace_id,
        "task_id": task_id,
        "input_hash": selection.input_hash,
        "selected_ids": selection.selected_ids,
        "omitted_ids": selection.omitted_ids,
    }
    summary = ContextSummary(
        summary_id=f"summary-{_canonical_hash(summary_hash_seed)[:16]}",
        task_id=task_id,
        workflow_id=workflow_id,
        trace_id=trace_id,
        user_objective=user_objective,
        immutable_constraints=immutable_constraints,
        market_facts=facts,
        unresolved_questions=unresolved_questions,
        omitted_sections=omitted_sections,
        token_estimate=sum(len(fact.fact.split()) for fact in facts),
        completeness=SummaryCompleteness.INCOMPLETE if incomplete else SummaryCompleteness.COMPLETE,
        summary_version=summary_version,
        source_record_hash=selection.input_hash,
        source_references=tuple(fact.source_id for fact in facts),
    )
    output_hash = _canonical_hash(summary.model_dump(mode="json"))
    return ContextHandoff(
        summary=summary,
        input_hash=selection.input_hash,
        output_hash=output_hash,
        selected_ids=selection.selected_ids,
        omitted_ids=selection.omitted_ids,
        selected_count=selection.selected_count,
        omitted_count=selection.omitted_count,
    )
