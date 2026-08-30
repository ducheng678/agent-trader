from __future__ import annotations

from market_agent.workflow_context_summary import select_context, summarize_context


def records() -> list[dict[str, object]]:
    return [
        {
            "record_id": "source-b",
            "source_id": "fed-1",
            "observed_at": "2026-08-29T10:00:00Z",
            "fact": "CPI is 3.2 percent, not 2.9 percent.",
            "relevance": 0.9,
            "uncertainty": "revision may follow",
        },
        {
            "record_id": "source-a",
            "source_id": "market-1",
            "observed_at": "2026-08-29T09:00:00Z",
            "fact": "BTC traded at 61,250 USD.",
            "relevance": 0.9,
        },
        {
            "record_id": "source-c",
            "source_id": "market-2",
            "observed_at": "2026-08-29T08:00:00Z",
            "fact": "Funding was 0.01 percent.",
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
