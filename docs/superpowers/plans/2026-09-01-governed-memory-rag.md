# Governed Memory and RAG Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement durable governed memory, retrieval summaries, promotion, and forgetting without granting agents storage-write authority.

**Architecture:** Strict contracts and a SQLite WAL repository establish event, knowledge, and decision truth. Deterministic retrieval/promotion and lifecycle services consume repository protocols and expose only bounded summaries to drivers.

**Tech Stack:** Python 3.13, Pydantic v2, SQLite WAL, pytest.

**Spec:** `docs/superpowers/specs/2026-09-01-governed-memory-rag-design.md`

## Global Constraints

- All memory mutations require trace ID, tenant scope, idempotency key, and auditable deterministic service authority.
- Agents receive no repository or artifact-store write handle.
- Retrieval filters tenant/version/lifecycle/freshness before ranking and injects only bounded cited summaries.
- Promotion rejects circular/model-only/expired evidence; forgetting preserves active, referenced, and legally held records.

---

### Task 1: Memory Contracts and SQLite Event/Knowledge/Decision Store

**Files:**
- Create: `market_agent/workflow_long_term_memory.py`
- Create: `market_agent/workflow_memory_sqlite.py`
- Create: `market_agent/workflow_object_store.py`
- Test: `market_agent_test_bundle/tests/test_workflow_memory_storage.py`

**Interfaces:** `MemoryRepository.append_event/propose_knowledge/activate_knowledge/append_decision/append_outcome/link_lesson/get_by_id`; `ArtifactStore.put/get`.

- [ ] **Step 1: Write failing authority and transaction tests.**

```python
def test_knowledge_requires_existing_same_tenant_event_evidence(repo):
    with pytest.raises(MemoryPromotionError):
        repo.activate_knowledge(candidate(evidence_ids=("missing",)))

def test_event_idempotency_does_not_duplicate_audit_truth(repo):
    assert repo.append_event(event(), idempotency_key="one") == repo.append_event(event(), idempotency_key="one")
```

- [ ] **Step 2: Run `python -m pytest market_agent_test_bundle/tests/test_workflow_memory_storage.py -q`; confirm failure.**

- [ ] **Step 3: Implement strict immutable records, checksum artifacts, and WAL repository.**

```python
def append_event(self, record: EventRecord, *, trace_id: str, idempotency_key: str) -> EventRecord:
    return self._transactional_insert(record, trace_id, idempotency_key)
```

- [ ] **Step 4: Add hash, tenant, rollback, revision-CAS, copy/rehydration, and agent-write-denial tests; rerun.**

- [ ] **Step 5: Commit `feat: add governed memory storage`.**

### Task 2: Retrieval, Summary, and Promotion

**Files:**
- Create: `market_agent/workflow_memory_retrieval.py`
- Create: `market_agent/workflow_memory_promotion.py`
- Test: `market_agent_test_bundle/tests/test_workflow_memory_retrieval.py`
- Test: `market_agent_test_bundle/tests/test_workflow_memory_promotion.py`

**Interfaces:** `retrieve_memory(query, repository) -> RetrievalResult`, `build_core_experience_summary(result, token_budget)`, and `promote_candidate(candidate, repository)`.

- [ ] **Step 1: Write failing filtered retrieval and promotion tests.**

```python
def test_summary_preserves_citations_and_conflicts(repository):
    summary = build_core_experience_summary(retrieve_memory(query(), repository), token_budget=800)
    assert summary.evidence_ids and summary.summary_hash

def test_circular_provenance_cannot_be_promoted(repository):
    with pytest.raises(MemoryPromotionError):
        promote_candidate(circular_candidate(), repository)
```

- [ ] **Step 2: Run the two test files; confirm failure.**

- [ ] **Step 3: Implement version/freshness filters, stable ranking, bounded conflict summaries, and verified-outcome promotion.**

```python
eligible = [r for r in records if r.tenant_id == query.tenant_id and r.expires_at > query.now]
```

- [ ] **Step 4: Add tenant/version mismatch, stale/weak/conflicting miss, anti-injection, no-memory, and agent-write-denial tests; rerun.**

- [ ] **Step 5: Commit `feat: add governed memory retrieval and promotion`.**

### Task 3: Lifecycle Forgetting and Integration Gate

**Files:**
- Create: `market_agent/workflow_memory_lifecycle.py`
- Test: `market_agent_test_bundle/tests/test_workflow_memory_lifecycle.py`
- Modify: `market_agent/workflow_agent_driver.py`

**Interfaces:** `LifecycleWorker.plan(scope, now)`, `apply(plan, limits)`, and `effective_confidence(record, now, policy)`; driver accepts only `CoreExperienceSummary` as dynamic context.

- [ ] **Step 1: Write failing decay/protection/lifecycle tests.**

```python
def test_referenced_or_held_event_never_reaches_purge(worker):
    plan = worker.plan("tenant-a", now=expired_time())
    assert protected_id not in plan.purge_ids
```

- [ ] **Step 2: Run lifecycle test; confirm failure.**

- [ ] **Step 3: Implement dry-run lifecycle actions, archive/tombstone/purge guards, idempotent cleanup, and summary-only driver boundary.**

```python
if record.legal_hold or record.referenced_by or record.lifecycle is Lifecycle.ACTIVE:
    return LifecycleAction.keep(record.record_id)
```

- [ ] **Step 4: Add decay, holds, cleanup idempotency, expired-evidence gap, trace audit, and driver forbidden-import tests; run all three Task suites.**

- [ ] **Step 5: Commit `feat: add governed memory forgetting lifecycle`; run compileall, memory-targeted pytest, and `git diff --check`.**

## PostgreSQL / pgvector Adapter Note

`workflow_memory_postgres.py` provides a production DB-API boundary without a
mandatory driver dependency. It uses tenant predicates and parameterized SQL for
immutable records, knowledge heads, idempotency, and audit rows; migrations
require pgvector, and vector candidates stay repository-side. The injected
connection factory is private to the host service, so agents receive only
validated summaries rather than database handles.
