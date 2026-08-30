from __future__ import annotations

from datetime import datetime, timezone
from hashlib import sha256
import json

import pytest
from pydantic import ValidationError

from market_agent import workflow_context_summary
from market_agent.workflow_context_summary import ContextHandoff, ContextRecord, ContextSelection, EvidenceReference, NormalizedClaim, select_context, summarize_context


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

    with pytest.raises(ValueError):
        summarize_context(forged, workflow_id="workflow-1", trace_id="trace-1", task_id="task-1", user_objective="Assess BTC")
    with pytest.raises(ValidationError, match="selection"):
        summarize_context(forged_ids, workflow_id="workflow-1", trace_id="trace-1", task_id="task-1", user_objective="Assess BTC")
    handoff = summarize_context(selection, workflow_id="workflow-1", trace_id="trace-1", task_id="task-1", user_objective="Assess BTC")

    assert handoff.summary.completeness.value == "incomplete"
    assert handoff.summary.conflicts == ("conflict-1: sources disagree",)
    assert handoff.contradicting_evidence
    assert handoff.output_hash != handoff.input_hash


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _rehash_handoff(values):
    values["output_hash"] = sha256(_canonical({key: value for key, value in values.items() if key != "output_hash"})).hexdigest()
    return values


def test_candidate_manifest_carries_no_raw_policy_metadata_and_recomputes_inventory():
    selection = select_context(records(), max_records=1, max_bytes=10000)
    entries = {entry.record_id: entry for entry in selection.candidate_manifest}
    normalized = {item["record_id"]: ContextRecord.model_validate(item) for item in records()}

    assert entries["source-a"].record_hash == sha256(_canonical(normalized["source-a"].model_dump(mode="json"))).hexdigest()
    assert entries["source-a"].relevance == 0.9
    assert entries["source-a"].conflict_group_id is None
    assert entries["source-a"].canonical_byte_length == len(_canonical(normalized["source-a"].model_dump(mode="json")))
    assert not hasattr(entries["source-a"], "record")

    low_only = select_context([records()[2]], max_records=1, max_bytes=10000)
    forged = selection.model_dump()
    forged.update(records=low_only.records, selected_ids=("source-c",), omitted_ids=("source-a", "source-b"), selected_count=1, omitted_count=2, selected_record_hash=low_only.selected_record_hash)
    with pytest.raises(ValidationError, match="selection"):
        ContextSelection(**forged)

    reversed_omissions = selection.model_dump()
    reversed_omissions["omitted_ids"] = tuple(reversed(selection.omitted_ids))
    with pytest.raises(ValidationError, match="omitted"):
        ContextSelection(**reversed_omissions)


def test_handoff_carries_and_revalidates_complete_selection_inventory():
    selection = select_context(records(), max_records=1, max_bytes=10000)
    handoff = summarize_context(selection, workflow_id="workflow-1", trace_id="trace-1", task_id="task-1", user_objective="Assess BTC")

    assert handoff.candidate_manifest == selection.candidate_manifest
    assert (handoff.max_records, handoff.max_bytes) == (selection.max_records, selection.max_bytes)
    assert handoff.selected_byte_length == selection.selected_byte_length
    assert handoff.full_input_byte_length == selection.full_input_byte_length

    forged = handoff.model_dump()
    forged["omitted_ids"] = tuple(reversed(forged["omitted_ids"]))
    with pytest.raises(ValidationError, match="omitted"):
        ContextHandoff(**_rehash_handoff(forged))


def test_summary_identity_is_recomputed_before_output_hash_acceptance():
    handoff = summarize_context(select_context(records(), max_records=2), workflow_id="workflow-1", trace_id="trace-1", task_id="task-1", user_objective="Assess BTC")
    forged = handoff.model_dump()
    forged["summary"]["user_objective"] = "Assess ETH"

    with pytest.raises(ValidationError, match="summary identity"):
        ContextHandoff(**_rehash_handoff(forged))


def _dense_records(count=30):
    observed_at = datetime(2026, 8, 29, tzinfo=timezone.utc)
    result = []
    for record_index in range(count):
        supporting = tuple(EvidenceReference(evidence_id=f"s-{record_index}-{index}", source_id=f"s{record_index}", observed_at=observed_at, relation="supporting") for index in range(10))
        contradicting = tuple(EvidenceReference(evidence_id=f"c-{record_index}-{index}", source_id=f"c{record_index}", observed_at=observed_at, relation="contradicting") for index in range(10))
        result.append(ContextRecord(record_id=f"r-{record_index:02d}", claim=NormalizedClaim(claim_id=f"claim-{record_index}", source_id=f"source-{record_index}", observed_at=observed_at, value=f"value {record_index}", negated=False, untrusted_data=True), relevance=1.0 - record_index / 100.0, uncertainty=f"uncertainty-{record_index:02d}", supporting_evidence=supporting, contradicting_evidence=contradicting))
    return result


def test_duplicate_evidence_ids_must_be_identical():
    observed_at = datetime(2026, 8, 29, tzinfo=timezone.utc)
    duplicate_a = EvidenceReference(evidence_id="evidence-1", source_id="source-a", observed_at=observed_at, relation="supporting")
    duplicate_b = EvidenceReference(evidence_id="evidence-1", source_id="source-b", observed_at=observed_at, relation="supporting")
    candidates = [
        ContextRecord(record_id="record-a", claim=NormalizedClaim(claim_id="claim-a", source_id="source-a", observed_at=observed_at, value="value a", negated=False, untrusted_data=True), relevance=0.9, supporting_evidence=(duplicate_a,)),
        ContextRecord(record_id="record-b", claim=NormalizedClaim(claim_id="claim-b", source_id="source-b", observed_at=observed_at, value="value b", negated=False, untrusted_data=True), relevance=0.8, supporting_evidence=(duplicate_b,)),
    ]

    with pytest.raises(ValueError, match="duplicate evidence"):
        summarize_context(select_context(candidates, max_records=2, max_bytes=500000), workflow_id="workflow-1", trace_id="trace-1", task_id="task-1", user_objective="Assess BTC")


def test_maximal_evidence_and_uncertainty_aggregates_are_capped_with_separate_counts():
    handoff = summarize_context(select_context(_dense_records(), max_records=30, max_bytes=500000), workflow_id="workflow-1", trace_id="trace-1", task_id="task-1", user_objective="Assess BTC")

    assert (len(handoff.supporting_evidence), handoff.omitted_supporting_evidence_count) == (50, 250)
    assert (len(handoff.contradicting_evidence), handoff.omitted_contradicting_evidence_count) == (50, 250)
    assert (len(handoff.uncertainty_markers), handoff.omitted_uncertainty_count) == (20, 10)


def test_conflict_counterevidence_is_symmetric_and_omitted_sections_are_cumulative():
    observed_at = datetime(2026, 8, 29, tzinfo=timezone.utc)
    conflict_records = [
        ContextRecord(record_id=f"record-{index}", claim=NormalizedClaim(claim_id=f"claim-{index}", source_id=f"source-{index}", observed_at=observed_at, value=f"value {index}", negated=False, untrusted_data=True), relevance=1.0 - index / 10.0, conflict_group_id="conflict-1", conflict_description="sources disagree", conflict_unresolved=True)
        for index in range(2)
    ]
    omitted = ContextRecord(record_id="record-low", claim=NormalizedClaim(claim_id="claim-low", source_id="source-low", observed_at=observed_at, value="low value", negated=False, untrusted_data=True), relevance=0.1)
    handoff = summarize_context(select_context(conflict_records + [omitted], max_records=2, max_bytes=500000), workflow_id="workflow-1", trace_id="trace-1", task_id="task-1", user_objective="Assess BTC")

    assert {item.evidence_id for item in handoff.contradicting_evidence} == {"record-0", "record-1"}
    assert {(item.section, item.count) for item in handoff.summary.omitted_sections} == {("source_records", 1), ("unresolved_conflicts", 1)}


def test_conflict_questions_have_an_independent_deterministic_cap():
    observed_at = datetime(2026, 8, 29, tzinfo=timezone.utc)
    candidates = []
    for group_index in range(15):
        for member_index in range(2):
            candidates.append(ContextRecord(record_id=f"record-{group_index:02d}-{member_index}", claim=NormalizedClaim(claim_id=f"claim-{group_index:02d}-{member_index}", source_id=f"source-{group_index:02d}-{member_index}", observed_at=observed_at, value=f"value {group_index} {member_index}", negated=False, untrusted_data=True), relevance=1.0 - group_index / 100.0, conflict_group_id=f"conflict-{group_index:02d}", conflict_description=f"conflict {group_index}", conflict_unresolved=True))
    handoff = summarize_context(select_context(candidates, max_records=30, max_bytes=500000), workflow_id="workflow-1", trace_id="trace-1", task_id="task-1", user_objective="Assess BTC")

    assert len(handoff.conflict_questions) == 10
    assert handoff.omitted_conflict_question_count == 5
    assert handoff.conflict_questions == tuple(sorted(handoff.conflict_questions))


def test_exact_canonical_envelopes_enforce_minus_at_and_plus_one_boundaries(monkeypatch):
    candidate = [records()[0]]
    baseline = select_context(candidate, max_records=1, max_bytes=500000)
    selected_limit = baseline.selected_byte_length

    assert select_context(candidate, max_records=1, max_bytes=selected_limit - 1).selected_count == 0
    assert select_context(candidate, max_records=1, max_bytes=selected_limit).selected_count == 1
    assert select_context(candidate, max_records=1, max_bytes=selected_limit + 1).selected_count == 1

    full_limit = baseline.full_input_byte_length
    monkeypatch.setattr(workflow_context_summary, "_MAX_INPUT_BYTES", full_limit - 1)
    with pytest.raises(ValueError, match="full-input envelope"):
        select_context(candidate, max_records=1, max_bytes=500000)
    monkeypatch.setattr(workflow_context_summary, "_MAX_INPUT_BYTES", full_limit)
    assert select_context(candidate, max_records=1, max_bytes=500000).candidate_count == 1
    monkeypatch.setattr(workflow_context_summary, "_MAX_INPUT_BYTES", full_limit + 1)
    assert select_context(candidate, max_records=1, max_bytes=500000).candidate_count == 1


def _resign_handoff(handoff):
    identity_fields = (
        "input_hash", "all_input_hash", "selected_ids", "omitted_ids", "selected_count", "omitted_count",
        "omitted_ids_truncated", "unreported_omitted_count", "selection_policy_version", "max_records", "max_bytes",
        "candidate_count", "candidate_manifest", "manifest_hash", "manifest_byte_length", "selected_byte_length", "full_input_byte_length",
        "uncertainty_markers", "omitted_uncertainty_count", "conflict_questions", "omitted_conflict_question_count", "conflicts",
        "supporting_evidence", "omitted_supporting_evidence_count", "contradicting_evidence", "omitted_contradicting_evidence_count",
    )
    rendered = handoff.model_dump(mode="json")
    normalized_summary = {key: value for key, value in rendered["summary"].items() if key != "summary_id"}
    inventory = {key: rendered[key] for key in identity_fields}
    summary_id = "summary-" + sha256(_canonical({"normalized_summary": normalized_summary, "selection_inventory": inventory})).hexdigest()[:32]
    updated = handoff.model_copy(update={"summary": handoff.summary.model_copy(update={"summary_id": summary_id}), "output_hash": "0" * 64})
    output_values = {key: value for key, value in updated.model_dump(mode="json").items() if key != "output_hash"}
    return updated.model_copy(update={"output_hash": sha256(_canonical(output_values)).hexdigest()})


def test_source_grouping_tags_standalone_records_separately_from_conflict_ids():
    observed_at = datetime(2026, 8, 29, tzinfo=timezone.utc)
    standalone = ContextRecord(record_id="shared", claim=NormalizedClaim(claim_id="claim-standalone", source_id="source-standalone", observed_at=observed_at, value="standalone", negated=False, untrusted_data=True), relevance=0.7)
    conflicts = [
        ContextRecord(record_id=f"conflict-member-{index}", claim=NormalizedClaim(claim_id=f"claim-{index}", source_id=f"source-{index}", observed_at=observed_at, value=f"conflict {index}", negated=False, untrusted_data=True), relevance=0.9 - index / 10.0, conflict_group_id="shared", conflict_description="sources disagree", conflict_unresolved=True)
        for index in range(2)
    ]

    for candidates in (conflicts + [standalone], [standalone] + conflicts):
        selection = select_context(candidates, max_records=3, max_bytes=500000)
        assert set(selection.selected_ids) == {"shared", "conflict-member-0", "conflict-member-1"}


def test_selection_revalidates_constructed_records_and_nested_evidence():
    observed_at = datetime(2026, 8, 29, tzinfo=timezone.utc)
    claim = NormalizedClaim(claim_id="claim-1", source_id="source-1", observed_at=observed_at, value="value", negated=False, untrusted_data=True)
    bad_relation = EvidenceReference.model_construct(evidence_id="evidence-1", source_id="source-1", observed_at=observed_at, relation="wrong")
    constructed = ContextRecord.model_construct(record_id="record-1", claim=claim, relevance=0.9, uncertainty=None, supporting_evidence=[bad_relation], contradicting_evidence=(), conflict_group_id=None, conflict_description=None, conflict_unresolved=False)

    with pytest.raises(ValidationError):
        select_context([constructed], max_records=1, max_bytes=500000)


def test_summarize_revalidates_copied_selection_with_mutable_lists():
    copied = select_context(records(), max_records=2).model_copy(update={"selected_ids": ["source-a", "source-b"]})

    with pytest.raises(ValidationError):
        summarize_context(copied, workflow_id="workflow-1", trace_id="trace-1", task_id="task-1", user_objective="Assess BTC")


def test_standalone_handoff_rejects_inconsistent_duplicate_evidence_after_resigning():
    handoff = summarize_context(select_context(records(), max_records=2), workflow_id="workflow-1", trace_id="trace-1", task_id="task-1", user_objective="Assess BTC")
    observed_at = datetime(2026, 8, 29, tzinfo=timezone.utc)
    support = EvidenceReference(evidence_id="duplicate", source_id="source-a", observed_at=observed_at, relation="supporting")
    contradiction = EvidenceReference(evidence_id="duplicate", source_id="source-b", observed_at=observed_at, relation="contradicting")
    forged = _resign_handoff(handoff.model_copy(update={"supporting_evidence": (support,), "contradicting_evidence": (contradiction,)}))

    with pytest.raises(ValidationError, match="duplicate evidence"):
        ContextHandoff.model_validate(forged)


def test_standalone_handoff_revalidates_nested_relation_and_mutable_tuple_fields():
    handoff = summarize_context(select_context(records(), max_records=2), workflow_id="workflow-1", trace_id="trace-1", task_id="task-1", user_objective="Assess BTC")
    observed_at = datetime(2026, 8, 29, tzinfo=timezone.utc)
    bad_relation = EvidenceReference.model_construct(evidence_id="evidence-1", source_id="source-1", observed_at=observed_at, relation="supporting")
    bad_evidence = _resign_handoff(handoff.model_copy(update={"contradicting_evidence": (bad_relation,)}))
    bad_list = ContextHandoff.model_construct(**{**handoff.model_dump(), "selected_ids": list(handoff.selected_ids)})

    with pytest.raises(ValidationError):
        ContextHandoff.model_validate(bad_evidence)
    with pytest.raises(ValidationError):
        ContextHandoff.model_validate(bad_list)


def test_unresolved_conflict_evidence_is_reserved_before_global_truncation():
    observed_at = datetime(2026, 8, 29, tzinfo=timezone.utc)
    candidates = []
    for index in range(30):
        explicit = EvidenceReference(evidence_id=f"a-explicit-{index:02d}", source_id=f"explicit-{index}", observed_at=observed_at, relation="contradicting")
        candidates.append(ContextRecord(record_id=f"z-member-{index:02d}", claim=NormalizedClaim(claim_id=f"claim-{index}", source_id=f"source-{index}", observed_at=observed_at, value=f"value {index}", negated=False, untrusted_data=True), relevance=1.0 - index / 100.0, contradicting_evidence=(explicit,), conflict_group_id="conflict-all", conflict_description="all sources disagree", conflict_unresolved=True))
    handoff = summarize_context(select_context(candidates, max_records=30, max_bytes=500000), workflow_id="workflow-1", trace_id="trace-1", task_id="task-1", user_objective="Assess BTC")
    reserved = {f"z-member-{index:02d}" for index in range(30)}

    assert reserved.issubset({item.evidence_id for item in handoff.contradicting_evidence})
    assert (len(handoff.contradicting_evidence), handoff.omitted_contradicting_evidence_count) == (50, 10)

    replacement = EvidenceReference(evidence_id="a-explicit-29", source_id="explicit-29", observed_at=observed_at, relation="contradicting")
    forged_items = tuple(sorted(tuple(item for item in handoff.contradicting_evidence if item.evidence_id != "z-member-00") + (replacement,), key=lambda item: item.evidence_id))
    forged = _resign_handoff(handoff.model_copy(update={"contradicting_evidence": forged_items}))
    with pytest.raises(ValidationError, match="reserved"):
        ContextHandoff.model_validate(forged)


def _raw_content_envelope(record_values, selection, scope):
    return {
        "limits": {"max_bytes": f"{selection.max_bytes:012d}", "max_records": f"{selection.max_records:02d}"},
        "manifest_hash": selection.manifest_hash,
        "policy_version": selection.selection_policy_version,
        "record_count": len(record_values),
        "records": [record.model_dump(mode="json") for record in record_values],
        "scope": scope,
    }


def test_raw_content_hashes_and_lengths_use_the_same_exact_canonical_envelopes():
    normalized = tuple(ContextRecord.model_validate(item) for item in records())
    selection = select_context(normalized, max_records=3, max_bytes=500000)
    full_records = tuple(sorted(normalized, key=lambda item: item.record_id))
    full_bytes = _canonical(_raw_content_envelope(full_records, selection, "full"))
    selected_bytes = _canonical(_raw_content_envelope(selection.records, selection, "selected"))
    manifest_bytes = _canonical({"candidate_count": len(selection.candidate_manifest), "entries": [entry.model_dump(mode="json") for entry in selection.candidate_manifest], "limits": {"max_bytes": f"{selection.max_bytes:012d}", "max_records": f"{selection.max_records:02d}"}, "policy_version": selection.selection_policy_version})

    assert (selection.all_input_hash, selection.full_input_byte_length) == (sha256(full_bytes).hexdigest(), len(full_bytes))
    assert (selection.selected_record_hash, selection.selected_byte_length) == (sha256(selected_bytes).hexdigest(), len(selected_bytes))
    assert (selection.manifest_hash, selection.manifest_byte_length) == (sha256(manifest_bytes).hexdigest(), len(manifest_bytes))

    changed_values = records()
    changed_values[0]["claim"] = {**changed_values[0]["claim"], "value": changed_values[0]["claim"]["value"] + "X"}
    changed = select_context(changed_values, max_records=3, max_bytes=500000)
    assert changed.all_input_hash != selection.all_input_hash
    assert changed.full_input_byte_length == selection.full_input_byte_length + 1
    assert changed.selected_record_hash != selection.selected_record_hash
    assert changed.selected_byte_length == selection.selected_byte_length + 1
    assert changed.manifest_hash != selection.manifest_hash


def test_handoff_carries_manifest_identity_separately_from_raw_attestations():
    selection = select_context(records(), max_records=2, max_bytes=500000)
    handoff = summarize_context(selection, workflow_id="workflow-1", trace_id="trace-1", task_id="task-1", user_objective="Assess BTC")

    assert (handoff.input_hash, handoff.selected_byte_length) == (selection.selected_record_hash, selection.selected_byte_length)
    assert (handoff.all_input_hash, handoff.full_input_byte_length) == (selection.all_input_hash, selection.full_input_byte_length)
    assert (handoff.manifest_hash, handoff.manifest_byte_length) == (selection.manifest_hash, selection.manifest_byte_length)


def _canonical_multi_group_handoff():
    observed_at = datetime(2026, 8, 29, tzinfo=timezone.utc)
    candidates = []
    for group_index, unresolved in enumerate((True, False)):
        for member_index in range(2):
            candidates.append(ContextRecord(record_id=f"member-{group_index}-{member_index}", claim=NormalizedClaim(claim_id=f"claim-{group_index}-{member_index}", source_id=f"source-{group_index}-{member_index}", observed_at=observed_at, value=f"value {group_index} {member_index}", negated=False, untrusted_data=True), relevance=1.0 - group_index / 10.0 - member_index / 100.0, conflict_group_id=f"group-{group_index}", conflict_description=f"conflict {group_index}", conflict_unresolved=unresolved))
    return summarize_context(select_context(candidates, max_records=4, max_bytes=500000), workflow_id="workflow-1", trace_id="trace-1", task_id="task-1", user_objective="Assess BTC")


def test_valid_canonical_multi_group_handoff_round_trips():
    handoff = _canonical_multi_group_handoff()

    assert ContextHandoff.model_validate(handoff.model_dump(mode="python")) == handoff
    assert tuple(group.group_id for group in handoff.conflicts) == ("group-0", "group-1")


def test_resigned_handoff_rejects_duplicate_conflict_groups_with_same_resolution():
    handoff = _canonical_multi_group_handoff()
    duplicate = handoff.conflicts[1]
    forged = _resign_handoff(handoff.model_copy(update={"conflicts": (handoff.conflicts[0], duplicate, duplicate)}))

    with pytest.raises(ValidationError, match="conflict"):
        ContextHandoff.model_validate(forged)


def test_resigned_handoff_rejects_duplicate_conflict_groups_with_conflicting_resolution():
    handoff = _canonical_multi_group_handoff()
    unresolved = handoff.conflicts[0]
    resolved_copy = unresolved.model_copy(update={"unresolved": False})
    forged = _resign_handoff(handoff.model_copy(update={"conflicts": (resolved_copy, unresolved, handoff.conflicts[1])}))

    with pytest.raises(ValidationError, match="conflict"):
        ContextHandoff.model_validate(forged)


def test_resigned_handoff_rejects_reversed_conflict_group_order():
    handoff = _canonical_multi_group_handoff()
    forged = _resign_handoff(handoff.model_copy(update={"conflicts": tuple(reversed(handoff.conflicts))}))

    with pytest.raises(ValidationError, match="conflict"):
        ContextHandoff.model_validate(forged)


def test_resigned_handoff_rejects_duplicate_conflict_questions_and_evidence():
    handoff = _canonical_multi_group_handoff()
    duplicated_question = _resign_handoff(handoff.model_copy(update={"conflict_questions": handoff.conflict_questions + handoff.conflict_questions}))
    first_evidence = handoff.contradicting_evidence[0]
    duplicated_evidence = tuple(sorted(handoff.contradicting_evidence + (first_evidence,), key=lambda item: item.evidence_id))
    duplicated_reference = _resign_handoff(handoff.model_copy(update={"contradicting_evidence": duplicated_evidence}))

    with pytest.raises(ValidationError, match="conflict-question"):
        ContextHandoff.model_validate(duplicated_question)
    with pytest.raises(ValidationError, match="evidence"):
        ContextHandoff.model_validate(duplicated_reference)


@pytest.mark.parametrize("summary_update", [{"conflicts": ("group-1: conflict 1", "group-0: conflict 0")}, {"conflicts": ("group-0: conflict 0", "group-0: conflict 0", "group-1: conflict 1")}, {"unresolved_questions": ()}])
def test_resigned_handoff_rejects_summary_conflict_inventory_divergence(summary_update):
    handoff = _canonical_multi_group_handoff()
    forged_summary = handoff.summary.model_copy(update=summary_update)
    forged = _resign_handoff(handoff.model_copy(update={"summary": forged_summary}))

    with pytest.raises(ValidationError, match="summary conflict"):
        ContextHandoff.model_validate(forged)
