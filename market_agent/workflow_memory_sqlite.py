"""Transactional local memory. Only deterministic services retain writer authority."""
from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime, timezone
import hashlib
from pathlib import Path
import sqlite3
import threading
from typing import Unpack

from pydantic import TypeAdapter, AwareDatetime, ValidationError

from market_agent.workflow_long_term_memory import (
    RECORD_TYPES, DecisionLesson, DecisionRecord, EventRecord, KnowledgeRevision,
    Lifecycle, MemoryAudit, MemoryAuthorityError, MemoryConflictError,
    MemoryIntegrityError, MemoryPromotionError, MutationContext, OutcomeRecord,
    Record, WriteArguments, canonical_json, content_hash, validate_authority,
)
from market_agent.workflow_memory_lifecycle import (
    CleanupTask, LifecycleEntry, LifecycleLimits, LifecyclePlan, LifecyclePolicy,
    LifecycleResult, LifecycleScope, build_lifecycle_plan,
)


class SQLiteMemoryRepository:
    def __init__(self, path: str | Path, *, writer_authority: object | None = None,
                 clock: Callable[[], datetime] | None = None):
        self.path = Path(path)
        self._authority = writer_authority
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, timeout=30, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA foreign_keys=ON")
        self._db.executescript("""
            CREATE TABLE IF NOT EXISTS memory_records (
                tenant_id TEXT NOT NULL, record_id TEXT NOT NULL, kind TEXT NOT NULL,
                body TEXT NOT NULL, body_hash TEXT NOT NULL, event_hash TEXT,
                PRIMARY KEY (tenant_id, record_id), UNIQUE (tenant_id, event_hash)
            );
            CREATE INDEX IF NOT EXISTS memory_tenant_kind ON memory_records(tenant_id, kind);
            CREATE TABLE IF NOT EXISTS memory_links (
                tenant_id TEXT NOT NULL, record_id TEXT NOT NULL, target_id TEXT NOT NULL,
                PRIMARY KEY (tenant_id, record_id, target_id),
                FOREIGN KEY (tenant_id, record_id) REFERENCES memory_records(tenant_id, record_id),
                FOREIGN KEY (tenant_id, target_id) REFERENCES memory_records(tenant_id, record_id)
            );
            CREATE INDEX IF NOT EXISTS memory_references ON memory_links(tenant_id, target_id);
            CREATE TABLE IF NOT EXISTS memory_heads (
                tenant_id TEXT NOT NULL, knowledge_id TEXT NOT NULL, revision INTEGER NOT NULL,
                record_id TEXT NOT NULL, PRIMARY KEY (tenant_id, knowledge_id),
                FOREIGN KEY (tenant_id, record_id) REFERENCES memory_records(tenant_id, record_id)
            );
            CREATE TABLE IF NOT EXISTS memory_idempotency (
                tenant_id TEXT NOT NULL, key TEXT NOT NULL, request_hash TEXT NOT NULL,
                kind TEXT NOT NULL, result TEXT NOT NULL, result_hash TEXT NOT NULL,
                PRIMARY KEY (tenant_id, key)
            );
            CREATE TABLE IF NOT EXISTS memory_audit (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT, tenant_id TEXT NOT NULL,
                trace_id TEXT NOT NULL, operation TEXT NOT NULL, record_id TEXT NOT NULL,
                idempotency_digest TEXT NOT NULL, record_hash TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS memory_audit_tenant ON memory_audit(tenant_id, sequence);
            CREATE TABLE IF NOT EXISTS memory_lifecycle_state (
                tenant_id TEXT NOT NULL, record_id TEXT NOT NULL, changed_at TEXT NOT NULL,
                PRIMARY KEY(tenant_id, record_id),
                FOREIGN KEY(tenant_id, record_id) REFERENCES memory_records(tenant_id, record_id)
            );
            CREATE TABLE IF NOT EXISTS memory_lifecycle_replay (
                tenant_id TEXT NOT NULL, key TEXT NOT NULL, request_hash TEXT NOT NULL,
                PRIMARY KEY(tenant_id, key)
            );
            CREATE TABLE IF NOT EXISTS memory_purged (
                tenant_id TEXT NOT NULL, record_id TEXT NOT NULL, event_hash TEXT,
                PRIMARY KEY(tenant_id, record_id), UNIQUE(tenant_id, event_hash)
            );
            CREATE TABLE IF NOT EXISTS memory_cleanup (
                tenant_id TEXT NOT NULL, task_id TEXT NOT NULL, body TEXT NOT NULL,
                body_hash TEXT NOT NULL, done INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(tenant_id, task_id)
            );
            CREATE TABLE IF NOT EXISTS memory_cleanup_attempts (
                sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                tenant_id TEXT NOT NULL, task_id TEXT NOT NULL,
                FOREIGN KEY(tenant_id, task_id) REFERENCES memory_cleanup(tenant_id, task_id)
            );
            CREATE INDEX IF NOT EXISTS memory_cleanup_last_attempt
                ON memory_cleanup_attempts(tenant_id, task_id, sequence);
        """)

    def close(self) -> None:
        self._db.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def _context(self, *, authority: object, **context) -> MutationContext:
        return validate_authority(self._authority, authority, **context)

    def validate_mutation(self, **context: Unpack[WriteArguments]) -> MutationContext:
        return self._context(**context)

    @staticmethod
    def _body(record: Record) -> str:
        return canonical_json(record.model_dump(mode="json"))

    @staticmethod
    def _rehydrate(kind: str, body: str, digest: str) -> Record:
        try:
            if hashlib.sha256(body.encode()).hexdigest() != digest:
                raise MemoryIntegrityError("stored memory hash mismatch")
            return RECORD_TYPES[kind].model_validate_json(body)
        except (ValidationError, ValueError, KeyError) as exc:
            raise MemoryIntegrityError("invalid stored memory") from exc

    def _read(self, record_id: str, tenant_id: str) -> Record | None:
        row = self._db.execute("SELECT * FROM memory_records WHERE tenant_id=? AND record_id=?",
                               (tenant_id, record_id)).fetchone()
        if row is None:
            return None
        return self._row_record(row)

    def _row_record(self, row: sqlite3.Row) -> Record:
        record = self._rehydrate(row["kind"], row["body"], row["body_hash"])
        if record.tenant_id != row["tenant_id"] or record.record_id != row["record_id"]:
            raise MemoryIntegrityError("stored record identity mismatch")
        return record

    def get_by_id(self, record_id: str, *, tenant_id: str) -> Record | None:
        with self._lock:
            return self._read(record_id, tenant_id)

    def list_records(self, *, tenant_id: str) -> tuple[Record, ...]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM memory_records WHERE tenant_id=? ORDER BY record_id", (tenant_id,)).fetchall()
            return tuple(self._row_record(row) for row in rows)

    def list_audit(self, *, tenant_id: str) -> tuple[MemoryAudit, ...]:
        with self._lock:
            rows = self._db.execute("SELECT * FROM memory_audit WHERE tenant_id=? ORDER BY sequence", (tenant_id,)).fetchall()
            return tuple(MemoryAudit.model_validate(dict(row)) for row in rows)

    @staticmethod
    def _references(record: Record) -> tuple[str, ...]:
        links = list(getattr(record, "evidence_ids", ()))
        links.extend(getattr(record, "lineage_ids", ()))
        links.extend(getattr(record, "contradicting_ids", ()))
        if isinstance(record, EventRecord):
            links.extend(record.provenance.derived_from)
        links.extend(value for name in ("decision_id", "outcome_id", "supersedes_id")
                     if (value := getattr(record, name, None)) is not None)
        return tuple(sorted(set(links)))

    def _require(self, record_id: str, tenant_id: str, cls) -> Record:
        record = self._read(record_id, tenant_id)
        if not isinstance(record, cls):
            raise MemoryPromotionError("evidence must exist in the same tenant with the required type")
        return record

    def _check_links(self, record: Record) -> None:
        for ref in self._references(record):
            self._require(ref, record.tenant_id, tuple(RECORD_TYPES.values()))
        if isinstance(record, (KnowledgeRevision, OutcomeRecord, DecisionLesson)):
            for ref in record.evidence_ids:
                self._require(ref, record.tenant_id, EventRecord)
        if isinstance(record, KnowledgeRevision):
            self._check_evidence_ancestry(record)
            for ref in record.lineage_ids:
                prior = self._require(ref, record.tenant_id, KnowledgeRevision)
                if prior.knowledge_id != record.knowledge_id or prior.revision >= record.revision:
                    raise MemoryPromotionError("knowledge lineage must precede this rule revision")
            if record.outcome_id is not None:
                self._verified_outcome(record.outcome_id, record.tenant_id)
        if isinstance(record, DecisionRecord):
            for ref in record.evidence_ids:
                evidence = self._require(ref, record.tenant_id, (EventRecord, KnowledgeRevision))
                if evidence.lifecycle is not Lifecycle.ACTIVE:
                    raise MemoryPromotionError("decision evidence must be active")
            if record.supersedes_id is not None:
                prior = self._require(record.supersedes_id, record.tenant_id, DecisionRecord)
                if prior.status != "provisional" or record.status != "final":
                    raise MemoryPromotionError("only provisional decisions can be finalized")
                for row in self._db.execute("SELECT record_id FROM memory_records WHERE tenant_id=? AND kind='DecisionRecord'", (record.tenant_id,)):
                    existing = self._read(row[0], record.tenant_id)
                    if existing.supersedes_id == prior.record_id:
                        raise MemoryConflictError("decision already finalized")
        if isinstance(record, (OutcomeRecord, DecisionLesson)):
            decision = self._require(record.decision_id, record.tenant_id, DecisionRecord)
            if decision.status != "final":
                raise MemoryPromotionError("outcomes and lessons require a final decision")
            if decision.observed_at > record.observed_at:
                raise MemoryPromotionError("outcomes and lessons cannot predate their decision")
        if isinstance(record, OutcomeRecord) and record.verified:
            self._validate_outcome_evidence(record, record.observed_at)
        if isinstance(record, DecisionLesson):
            outcome = self._verified_outcome(record.outcome_id, record.tenant_id, now=record.observed_at)
            if outcome.decision_id != record.decision_id:
                raise MemoryPromotionError("lesson outcome must belong to its decision")
            self._evidence_roots(record.evidence_ids, record.tenant_id, record.observed_at)

    def _walk_ancestry(self, identifiers: tuple[str, ...], tenant_id: str, *,
                       events_only: bool = False) -> Iterator[Record]:
        """Visit parents before children; reject cycles without Python recursion."""
        visiting: set[str] = set()
        visited: set[str] = set()
        records: dict[str, Record] = {}
        pending = [(identifier, False) for identifier in reversed(identifiers)]
        while pending:
            identifier, expanded = pending.pop()
            if expanded:
                visiting.remove(identifier)
                visited.add(identifier)
                yield records[identifier]
                continue
            if identifier in visiting:
                raise MemoryPromotionError("circular evidence is forbidden")
            if identifier in visited:
                continue
            evidence = self._require(identifier, tenant_id,
                                     EventRecord if events_only else tuple(RECORD_TYPES.values()))
            records[identifier] = evidence
            visiting.add(identifier)
            pending.append((identifier, True))
            parents = evidence.provenance.derived_from if events_only else self._references(evidence)
            pending.extend((parent, False) for parent in reversed(parents))

    def _check_evidence_ancestry(self, candidate: KnowledgeRevision) -> None:
        identifiers = (*candidate.evidence_ids, *((candidate.outcome_id,) if candidate.outcome_id else ()))
        for evidence in self._walk_ancestry(identifiers, candidate.tenant_id):
            if isinstance(evidence, KnowledgeRevision) and evidence.knowledge_id == candidate.knowledge_id:
                raise MemoryPromotionError("a rule's descendants cannot corroborate that rule")

    def _evidence_roots(self, identifiers: tuple[str, ...], tenant_id: str, now: datetime) -> set[str]:
        """Only fresh, model-free event paths establish independent observations.

        Decision/knowledge links provide context, not observation provenance, so
        non-event parents are ambiguous here and fail the event type check.
        """
        roots: dict[str, set[str]] = {}
        for evidence in self._walk_ancestry(identifiers, tenant_id, events_only=True):
            self._fresh_evidence(evidence, now)
            provenance = evidence.provenance
            groups: set[str] = set()
            if provenance.source_kind != "model":
                if provenance.derived_from:
                    for parent in provenance.derived_from:
                        groups.update(roots[parent])
                else:
                    groups.add(provenance.independent_group)
            roots[evidence.record_id] = groups
        return set().union(*(roots[identifier] for identifier in identifiers))

    def _validate_outcome_evidence(self, record: OutcomeRecord, now: datetime) -> None:
        self._fresh_evidence(record, now)
        if not self._evidence_roots(record.evidence_ids, record.tenant_id, now):
            raise MemoryPromotionError("independent non-model root evidence is required to verify an outcome")

    def _verified_outcome(self, record_id: str, tenant_id: str, *, now: datetime | None = None) -> OutcomeRecord:
        record = self._require(record_id, tenant_id, OutcomeRecord)
        if not record.verified:
            raise MemoryPromotionError("a verified outcome is required")
        self._validate_outcome_evidence(record, record.observed_at if now is None else now)
        return record

    def _replay(self, context: MutationContext, request_hash: str) -> Record | None:
        row = self._db.execute("SELECT * FROM memory_idempotency WHERE tenant_id=? AND key=?",
                               (context.tenant_id, context.idempotency_key)).fetchone()
        if row is None:
            return None
        if row["request_hash"] != request_hash:
            raise MemoryConflictError("idempotency key already binds a different request")
        return self._rehydrate(row["kind"], row["result"], row["result_hash"])

    def _audit(self, record: Record, operation: str, context: MutationContext) -> None:
        self._db.execute("INSERT INTO memory_audit(tenant_id,trace_id,operation,record_id,idempotency_digest,record_hash) VALUES(?,?,?,?,?,?)",
                         (context.tenant_id, context.trace_id, operation, record.record_id,
                          hashlib.sha256(context.idempotency_key.encode()).hexdigest(),
                          hashlib.sha256(self._body(record).encode()).hexdigest()))

    def _remember(self, record: Record, context: MutationContext, request_hash: str) -> None:
        body = self._body(record)
        self._db.execute("INSERT INTO memory_idempotency VALUES(?,?,?,?,?,?)",
                         (context.tenant_id, context.idempotency_key, request_hash,
                          type(record).__name__, body, hashlib.sha256(body.encode()).hexdigest()))

    def _store(self, record: Record, *, update: bool = False) -> None:
        body = self._body(record)
        digest = hashlib.sha256(body.encode()).hexdigest()
        if update:
            self._db.execute("UPDATE memory_records SET body=?,body_hash=? WHERE tenant_id=? AND record_id=?",
                             (body, digest, record.tenant_id, record.record_id))
        else:
            self._db.execute("INSERT INTO memory_records VALUES(?,?,?,?,?,?)",
                             (record.tenant_id, record.record_id, type(record).__name__, body, digest,
                              record.payload_hash if isinstance(record, EventRecord) else None))
            for target in self._references(record):
                self._db.execute("INSERT INTO memory_links VALUES(?,?,?)", (record.tenant_id, record.record_id, target))

    def _append(self, record: Record, cls, operation: str, context: MutationContext) -> Record:
        if type(record) is not cls:
            raise MemoryIntegrityError("record has the wrong type")
        record = cls.model_validate(record)
        if record.tenant_id != context.tenant_id:
            raise MemoryAuthorityError("mutation tenant does not match record")
        expected_state = Lifecycle.PROPOSED if cls is KnowledgeRevision else Lifecycle.ACTIVE
        if record.lifecycle is not expected_state:
            raise MemoryPromotionError("new records must enter their initial lifecycle")
        request_hash = content_hash({"operation": operation, "trace_id": context.trace_id,
                                     "record": record.model_dump(mode="json")})
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            removed = self._db.execute(
                "SELECT 1 FROM memory_purged WHERE tenant_id=? AND (record_id=? OR event_hash=?)",
                (record.tenant_id, record.record_id, record.payload_hash if isinstance(record, EventRecord) else None)).fetchone()
            if removed is not None:
                raise MemoryConflictError("purged memory cannot be resurrected by replay or reappend")
            if isinstance(record, EventRecord) and record.artifact is not None:
                # A deferred delete must not race with a new attachment of the
                # same content address after the last old reference was purged.
                for row in self._db.execute("SELECT body,body_hash FROM memory_cleanup WHERE tenant_id=?",
                                            (record.tenant_id,)):
                    task = self._cleanup_task(row)
                    if task.kind == "artifact" and task.artifact.sha256 == record.artifact.sha256:
                        raise MemoryConflictError("artifact address has entered cleanup")
            replay = self._replay(context, request_hash)
            if replay is not None:
                return replay
            if self._read(record.record_id, record.tenant_id) is not None:
                raise MemoryConflictError("record identity is append-only")
            self._check_links(record)
            if isinstance(record, EventRecord):
                row = self._db.execute("SELECT record_id FROM memory_records WHERE tenant_id=? AND event_hash=?",
                                       (record.tenant_id, record.payload_hash)).fetchone()
                if row is not None:
                    original = self._read(row[0], record.tenant_id)
                    self._remember(original, context, request_hash)
                    return original
            if isinstance(record, KnowledgeRevision):
                head = self._db.execute("SELECT * FROM memory_heads WHERE tenant_id=? AND knowledge_id=?",
                                        (record.tenant_id, record.knowledge_id)).fetchone()
                if head is None:
                    if record.revision != 1 or record.lineage_ids:
                        raise MemoryConflictError("initial knowledge revision must be 1 with no lineage")
                elif record.revision != head["revision"] + 1 or head["record_id"] not in record.lineage_ids:
                    raise MemoryConflictError("knowledge revision must extend the current head")
            self._store(record)
            if isinstance(record, KnowledgeRevision):
                self._db.execute("INSERT INTO memory_heads VALUES(?,?,?,?) ON CONFLICT(tenant_id,knowledge_id) DO UPDATE SET revision=excluded.revision,record_id=excluded.record_id",
                                 (record.tenant_id, record.knowledge_id, record.revision, record.record_id))
            self._audit(record, operation, context)
            self._remember(record, context, request_hash)
            return record

    def append_event(self, record: EventRecord, **context: Unpack[WriteArguments]) -> EventRecord:
        return self._append(record, EventRecord, "append_event", self._context(**context))

    def propose_knowledge(self, record: KnowledgeRevision, **context: Unpack[WriteArguments]) -> KnowledgeRevision:
        return self._append(record, KnowledgeRevision, "propose_knowledge", self._context(**context))

    def append_decision(self, record: DecisionRecord, **context: Unpack[WriteArguments]) -> DecisionRecord:
        return self._append(record, DecisionRecord, "append_decision", self._context(**context))

    def append_outcome(self, record: OutcomeRecord, **context: Unpack[WriteArguments]) -> OutcomeRecord:
        return self._append(record, OutcomeRecord, "append_outcome", self._context(**context))

    def link_lesson(self, record: DecisionLesson, **context: Unpack[WriteArguments]) -> DecisionLesson:
        return self._append(record, DecisionLesson, "link_lesson", self._context(**context))

    def activate_knowledge(self, record_id: str, *, expected_revision: int, now: datetime,
                           **context: Unpack[WriteArguments]) -> KnowledgeRevision:
        ctx = self._context(**context)
        now = TypeAdapter(AwareDatetime).validate_python(now, strict=True)
        if type(expected_revision) is not int or expected_revision < 1:
            raise MemoryConflictError("activation requires a positive revision")
        request_hash = content_hash({"operation": "activate_knowledge", "trace_id": ctx.trace_id,
                                     "record_id": record_id, "revision": expected_revision, "now": now.isoformat()})
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            replay = self._replay(ctx, request_hash)
            if replay is not None:
                return replay
            record = self._require(record_id, ctx.tenant_id, KnowledgeRevision)
            head = self._db.execute("SELECT revision,record_id FROM memory_heads WHERE tenant_id=? AND knowledge_id=?",
                                    (ctx.tenant_id, record.knowledge_id)).fetchone()
            if record.revision != expected_revision or head["record_id"] != record_id or record.lifecycle is not Lifecycle.PROPOSED:
                raise MemoryConflictError("stale knowledge activation revision or state")
            self._check_links(record)
            if record.effective_at > now or (record.expires_at is not None and record.expires_at <= now):
                raise MemoryPromotionError("knowledge is not effective at activation time")
            if record.contradicting_ids:
                raise MemoryPromotionError("conflicting evidence prevents activation")
            sources = self._evidence_roots(record.evidence_ids, ctx.tenant_id, now)
            if record.outcome_id is not None:
                outcome = self._verified_outcome(record.outcome_id, ctx.tenant_id, now=now)
                if not set(outcome.evidence_ids) & set(record.evidence_ids):
                    raise MemoryPromotionError("verified outcome must support the candidate evidence")
            elif len(sources) < 2:
                raise MemoryPromotionError("activation needs independent corroboration or a verified outcome")
            # Superseding one rule revision archives the prior active revision in this transaction.
            for prior in self.list_records(tenant_id=ctx.tenant_id):
                if isinstance(prior, KnowledgeRevision) and prior.knowledge_id == record.knowledge_id and prior.lifecycle is Lifecycle.ACTIVE:
                    archived = prior.model_copy(update={"lifecycle": Lifecycle.ARCHIVED})
                    self._store(archived, update=True)
                    self._set_lifecycle_time(archived, TypeAdapter(AwareDatetime).validate_python(self._clock(), strict=True))
                    self._audit(archived, "supersede_knowledge", ctx)
            active = record.model_copy(update={"lifecycle": Lifecycle.ACTIVE})
            self._store(active, update=True)
            self._audit(active, "activate_knowledge", ctx)
            self._remember(active, ctx, request_hash)
            return active

    @staticmethod
    def _fresh_evidence(record: Record, now: datetime) -> None:
        if record.lifecycle is not Lifecycle.ACTIVE or record.observed_at > now or (record.expires_at is not None and record.expires_at <= now):
            raise MemoryPromotionError("evidence must be active, observed, and unexpired")

    def _set_lifecycle_time(self, record: Record, now: datetime) -> None:
        self._db.execute("INSERT INTO memory_lifecycle_state VALUES(?,?,?) "
                         "ON CONFLICT(tenant_id,record_id) DO UPDATE SET changed_at=excluded.changed_at",
                         (record.tenant_id, record.record_id, now.isoformat()))

    def lifecycle_snapshot(self, scope: LifecycleScope) -> tuple[LifecycleEntry, ...]:
        scope = LifecycleScope.model_validate(scope)
        with self._lock:
            own_transaction = not self._db.in_transaction
            if own_transaction:
                self._db.execute("BEGIN")
            try:
                entries = []
                for record in self.list_records(tenant_id=scope.tenant_id):
                    if scope.scope is not None and record.scope != scope.scope:
                        continue
                    refs = self._db.execute("SELECT record_id FROM memory_links WHERE tenant_id=? AND target_id=? ORDER BY record_id",
                                             (record.tenant_id, record.record_id)).fetchall()
                    state = self._db.execute("SELECT changed_at FROM memory_lifecycle_state WHERE tenant_id=? AND record_id=?",
                                              (record.tenant_id, record.record_id)).fetchone()
                    head = self._db.execute("SELECT 1 FROM memory_heads WHERE tenant_id=? AND record_id=?",
                                            (record.tenant_id, record.record_id)).fetchone()
                    entries.append(LifecycleEntry(record=record, referenced_by=tuple(row[0] for row in refs),
                        changed_at=datetime.fromisoformat(state[0]) if state else None,
                        is_knowledge_head=head is not None))
                return tuple(entries)
            finally:
                if own_transaction:
                    self._db.rollback()

    def _queue_cleanup(self, record: Record, kind: str, ctx: MutationContext) -> None:
        task = CleanupTask(task_id=content_hash({"tenant": record.tenant_id, "record": record.record_id, "kind": kind}),
            tenant_id=record.tenant_id, scope=record.scope, trace_id=ctx.trace_id, record_id=record.record_id,
            record_hash=content_hash(record.model_dump(mode="json")), kind=kind,
            artifact=record.artifact if kind == "artifact" else None)
        body = task.model_dump_json()
        self._db.execute("INSERT OR IGNORE INTO memory_cleanup(tenant_id,task_id,body,body_hash) VALUES(?,?,?,?)",
                          (record.tenant_id, task.task_id, body, hashlib.sha256(body.encode()).hexdigest()))

    @staticmethod
    def _cleanup_task(row) -> CleanupTask:
        try:
            if hashlib.sha256(row["body"].encode()).hexdigest() != row["body_hash"]:
                raise MemoryIntegrityError("cleanup task checksum mismatch")
            return CleanupTask.model_validate_json(row["body"])
        except (ValidationError, ValueError) as exc:
            raise MemoryIntegrityError("invalid cleanup task") from exc

    def list_cleanup(self, *, tenant_id: str, scope: str | None = None) -> tuple[CleanupTask, ...]:
        with self._lock:
            result = []
            for row in self._db.execute("SELECT c.* FROM memory_cleanup c WHERE c.tenant_id=? AND c.done=0 "
                    "ORDER BY COALESCE((SELECT MAX(a.sequence) FROM memory_cleanup_attempts a "
                    "WHERE a.tenant_id=c.tenant_id AND a.task_id=c.task_id), 0), c.task_id", (tenant_id,)):
                task = self._cleanup_task(row)
                if task.tenant_id != tenant_id or task.task_id != row["task_id"]:
                    raise MemoryIntegrityError("cleanup task identity mismatch")
                if scope is None or task.scope == scope:
                    result.append(task)
            return tuple(result)

    def apply_lifecycle(self, plan: LifecyclePlan, policy: LifecyclePolicy, limits: LifecycleLimits,
                        **context: Unpack[WriteArguments]) -> LifecycleResult:
        ctx = self._context(**context)
        plan, policy, limits = (LifecyclePlan.model_validate(plan), LifecyclePolicy.model_validate(policy),
                                LifecycleLimits.model_validate(limits))
        if plan.scope.tenant_id != ctx.tenant_id:
            raise MemoryAuthorityError("lifecycle tenant does not match mutation scope")
        if plan.policy_hash != content_hash(policy.model_dump(mode="json")):
            raise MemoryConflictError("lifecycle policy does not match plan")
        applied, skipped = [], []
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            # Selection and guards share the write transaction, including links
            # created through other SQLite connections after the dry run.
            applied_at = TypeAdapter(AwareDatetime).validate_python(self._clock(), strict=True)
            if plan.now > applied_at:
                raise MemoryConflictError("lifecycle plan is ahead of the trusted apply clock")
            current = build_lifecycle_plan(self.lifecycle_snapshot(plan.scope), plan.scope, applied_at, policy)
            eligible = {action.record_id: action for action in current.actions}
            examined = 0
            for action in plan.actions:
                if examined >= limits.max_actions:
                    break
                key = content_hash({"key": ctx.idempotency_key, "plan": plan.plan_hash, "record": action.record_id})
                request_hash = content_hash({"trace": ctx.trace_id, "plan": plan.plan_hash, "action": action.model_dump(mode="json")})
                replay = self._db.execute("SELECT request_hash FROM memory_lifecycle_replay WHERE tenant_id=? AND key=?",
                                           (ctx.tenant_id, key)).fetchone()
                if replay is not None:
                    if replay[0] != request_hash:
                        raise MemoryConflictError("lifecycle replay context changed")
                    continue
                if eligible.get(action.record_id) != action:
                    skipped.append(action.record_id)
                    continue
                examined += 1
                record = self._read(action.record_id, ctx.tenant_id)
                action_ctx = ctx.model_copy(update={"idempotency_key": key})
                if action.kind == "purge":
                    # Planner permits only unreferenced, unheld, non-head
                    # tombstones. Keep the destructive gate explicit as well.
                    refs = self._db.execute("SELECT 1 FROM memory_links WHERE tenant_id=? AND target_id=?",
                                             (ctx.tenant_id, record.record_id)).fetchone()
                    if record.lifecycle is not Lifecycle.TOMBSTONED or record.legal_hold or refs:
                        skipped.append(action.record_id)
                        continue
                    if isinstance(record, EventRecord) and record.artifact is not None:
                        shared = any(isinstance(other, EventRecord) and other.record_id != record.record_id
                                     and other.artifact is not None and other.artifact.sha256 == record.artifact.sha256
                                     for other in self.list_records(tenant_id=ctx.tenant_id))
                        if not shared:
                            self._queue_cleanup(record, "artifact", action_ctx)
                    self._audit(record, "lifecycle_purge", action_ctx)
                    self._db.execute("INSERT INTO memory_purged VALUES(?,?,?)",
                                      (ctx.tenant_id, record.record_id, record.payload_hash if isinstance(record, EventRecord) else None))
                    self._db.execute("DELETE FROM memory_links WHERE tenant_id=? AND record_id=?", (ctx.tenant_id, record.record_id))
                    self._db.execute("DELETE FROM memory_lifecycle_state WHERE tenant_id=? AND record_id=?", (ctx.tenant_id, record.record_id))
                    self._db.execute("DELETE FROM memory_idempotency WHERE tenant_id=? AND json_extract(result, '$.record_id')=?",
                                      (ctx.tenant_id, record.record_id))
                    self._db.execute("DELETE FROM memory_records WHERE tenant_id=? AND record_id=?", (ctx.tenant_id, record.record_id))
                else:
                    target = Lifecycle.ARCHIVED if action.kind == "archive" else Lifecycle.TOMBSTONED
                    record = record.model_copy(update={"lifecycle": target})
                    self._store(record, update=True)
                    self._set_lifecycle_time(record, applied_at)
                    self._audit(record, "lifecycle_" + action.kind, action_ctx)
                    if action.kind == "tombstone":
                        for kind in ("vector", "cache"):
                            self._queue_cleanup(record, kind, action_ctx)
                self._db.execute("INSERT INTO memory_lifecycle_replay VALUES(?,?,?)", (ctx.tenant_id, key, request_hash))
                applied.append(action.record_id)
        return LifecycleResult(applied_ids=tuple(applied), skipped_ids=tuple(skipped))

    def begin_cleanup(self, task: CleanupTask, **context: Unpack[WriteArguments]) -> bool:
        """Persist a fair retry position before external work, including crashes."""
        ctx = self._context(**context)
        task = CleanupTask.model_validate(task)
        if task.tenant_id != ctx.tenant_id or task.trace_id != ctx.trace_id or task.task_id != ctx.idempotency_key:
            raise MemoryAuthorityError("cleanup attempt must match its original mutation context")
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            row = self._db.execute("SELECT * FROM memory_cleanup WHERE tenant_id=? AND task_id=?", (ctx.tenant_id, task.task_id)).fetchone()
            if row is None or self._cleanup_task(row) != task:
                raise MemoryIntegrityError("cleanup task is not registered")
            if row["done"]:
                return False
            attempt = self._db.execute("INSERT INTO memory_cleanup_attempts(tenant_id,task_id) VALUES(?,?)",
                                      (ctx.tenant_id, task.task_id)).lastrowid
            self._db.execute("INSERT INTO memory_audit(tenant_id,trace_id,operation,record_id,idempotency_digest,record_hash) VALUES(?,?,?,?,?,?)",
                (ctx.tenant_id, ctx.trace_id, "lifecycle_cleanup_attempt", task.record_id,
                 content_hash({"task": task.task_id, "attempt": attempt}), task.record_hash))
            return True

    def finish_cleanup(self, task: CleanupTask, **context: Unpack[WriteArguments]) -> None:
        ctx = self._context(**context)
        task = CleanupTask.model_validate(task)
        if task.tenant_id != ctx.tenant_id or task.trace_id != ctx.trace_id or task.task_id != ctx.idempotency_key:
            raise MemoryAuthorityError("cleanup acknowledgement must match its original mutation context")
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            row = self._db.execute("SELECT * FROM memory_cleanup WHERE tenant_id=? AND task_id=?", (ctx.tenant_id, task.task_id)).fetchone()
            if row is None or self._cleanup_task(row) != task:
                raise MemoryIntegrityError("cleanup task is not registered")
            if row["done"]:
                return
            self._db.execute("UPDATE memory_cleanup SET done=1 WHERE tenant_id=? AND task_id=?", (ctx.tenant_id, task.task_id))
            self._db.execute("INSERT INTO memory_audit(tenant_id,trace_id,operation,record_id,idempotency_digest,record_hash) VALUES(?,?,?,?,?,?)",
                              (ctx.tenant_id, ctx.trace_id, "cleanup_" + task.kind, task.record_id,
                               hashlib.sha256(ctx.idempotency_key.encode()).hexdigest(), task.record_hash))
