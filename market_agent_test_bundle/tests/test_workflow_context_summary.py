from __future__ import annotations

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from market_agent.workflow_context_summary import ContextRecord, ContextSelection, NormalizedClaim, select_context, summarize_context


def records() -> list[dict[str, object]]:
    return [
        {
            "record_id": "source-b",
            "claim": {"claim_id": "claim-b", "source_id": "fed-1", "observed_at": datetime(2026, 8, 29, 10, tzinfo=timezone.utc), "value": "CPI is 3.2 percent, not 2.9 percent.", "unit": None, "negated": False, "untrusted_data": True},
            "relevance": 0.9,
            "uncertainty": "revision may follow",
        },
        {
            "record_id": "source-a",
            "claim": {"claim_id": "claim-a", "source_id": "market-1", "observed_at": datetime(2026, 8, 29, 9, tzinfo=timezone.utc), "value": "BTC traded at 61,250 USD.", "unit": None, "negated": False, "untrusted_data": True},
            "relevance": 0.9,
        },
        {
            "record_id": "source-c",
            "claim": {"claim_id": "claim-c", "source_id": "market-2", "observed_at": datetime(2026, 8, 29, 8, tzinfo=timezone.utc), "value": "Funding was 0.01 percent.", "unit": None, "negated": False, "untrusted_data": True},
            "relevance": 0.2,
        },
    ]


def test_select_context_is_deterministic_bounded_and_reports_selected_and_omitted_identifiers():
    first = select_context(records(), max_records=2)
    second = select_context(list(reversed(records())), max_records=2)

    assert first == second
    assert first.selected_ids == ("source-a", "source-b")
    assert first.omitted_ids == ("source-c",)
    assert first.selected_count == 2
    assert first.omitted_count == 1
    assert first.input_hash == second.input_hash


def test_summary_preserves_numeric_units_negation_time_provenance_and_uncertainty():
    handoff = summarize_context(
        select_context(records(), max_records=2),
        workflow_id="workflow-1",
        trace_id="trace-1",
        task_id="task-1",
        user_objective="Assess BTC direction",
        immutable_constraints=("Do not place orders.",),
    )

    facts = {fact.source_id: fact for fact in handoff.summary.market_facts}
    assert facts["fed-1"].observed_at == "2026-08-29T10:00:00Z"
    assert facts["fed-1"].fact == "CPI is 3.2 percent, not 2.9 percent."
    assert facts["market-1"].fact == "BTC traded at 61,250 USD."
    assert handoff.summary.unresolved_questions == ("revision may follow",)
    assert handoff.summary.source_references == ("fed-1", "market-1")
    assert handoff.selected_ids == ("source-a", "source-b")
    assert handoff.omitted_ids == ("source-c",)
    assert handoff.summary.completeness.value == "incomplete"
    assert handoff.summary.omitted_sections[0].count == 1


def test_summary_is_deterministic_and_represents_missing_evidence_as_insufficient():
    selected = select_context([], max_records=2)
    first = summarize_context(
        selected,
        workflow_id="workflow-1",
        trace_id="trace-1",
        task_id="task-1",
        user_objective="Assess BTC direction",
    )
    second = summarize_context(
        selected,
        workflow_id="workflow-1",
        trace_id="trace-1",
        task_id="task-1",
        user_objective="Assess BTC direction",
    )

    assert first.input_hash == first.summary.source_record_hash
    assert first.output_hash == second.output_hash
    assert first.summary.completeness.value == "incomplete"
    assert first.summary.unresolved_questions == ("insufficient source evidence",)
    assert first.summary.omitted_sections[0].section == "source_records"
    assert first.summary.omitted_sections[0].count == 0


def test_context_rejects_unbounded_inputs_and_binds_hashes_to_selected_records_and_policy():
    with pytest.raises(ValueError):
        select_context(records(), max_records=31)
    selection = select_context(records(), max_records=2)
    with pytest.raises(ValidationError):
        ContextSelection(
            records=selection.records,
            selected_ids=("source-b",),
            omitted_ids=selection.omitted_ids,
            selected_count=2,
            omitted_count=1,
            selected_record_hash=selection.selected_record_hash,
            all_input_hash=selection.all_input_hash,
            selection_policy_version=selection.selection_policy_version,
        )


def test_summary_recomputes_selection_and_marks_unresolved_conflicts_incomplete():
    claim = NormalizedClaim(
        claim_id="claim-1",
        source_id="source-1",
        observed_at=datetime(2026, 8, 29, tzinfo=timezone.utc),
        value="BTC has support at 60,000",
        unit="USD",
        negated=False,
        untrusted_data=True,
    )
    selection = select_context(
        [
            ContextRecord(record_id="source-1", claim=claim, relevance=0.9, conflict_group_id="conflict-1", conflict_description="sources disagree", conflict_unresolved=True),
            ContextRecord(record_id="source-2", claim=claim.model_copy(update={"claim_id": "claim-2", "source_id": "source-2"}), relevance=0.8, conflict_group_id="conflict-1", conflict_description="sources disagree", conflict_unresolved=True),
        ],
        max_records=2,
    )
    forged = selection.model_copy(update={"selected_record_hash": "forged"})
    forged_ids = selection.model_copy(update={"selected_ids": ("forged",)})

    with pytest.raises(ValueError, match="selection"):
        summarize_context(forged, workflow_id="workflow-1", trace_id="trace-1", task_id="task-1", user_objective="Assess BTC")
    with pytest.raises(ValidationError, match="selection"):
        summarize_context(forged_ids, workflow_id="workflow-1", trace_id="trace-1", task_id="task-1", user_objective="Assess BTC")
    handoff = summarize_context(selection, workflow_id="workflow-1", trace_id="trace-1", task_id="task-1", user_objective="Assess BTC")

    assert handoff.summary.completeness.value == "incomplete"
    assert handoff.summary.conflicts == ("conflict-1: sources disagree",)
    assert handoff.contradicting_evidence
    assert handoff.output_hash != handoff.input_hash
