"""Host-owned durable result journal bridging Harness success and side effects.

Staging is not acceptance. Only a verified terminal receipt authorizes commit.
Commit callbacks must be idempotent: a crash after their writes can replay them
with the exact same proof before the journal's committed marker is persisted.
"""

from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import sqlite3
from typing import Callable, Protocol

from market_agent.workflow_contracts import WorkflowRequest, canonical_workflow_request_digest
from market_agent.workflow_memory_result_writer import AcceptedOutcomeProof
from market_agent.workflow_observation import WorkflowExecution


@dataclass(frozen=True, slots=True)
class PreparedWorkflowResult:
    request_digest: str
    execution: WorkflowExecution
    proof: AcceptedOutcomeProof | None
    committed: bool


class WorkflowResultStore(Protocol):
    def stage(self, request: WorkflowRequest, execution: WorkflowExecution) -> PreparedWorkflowResult: ...
    def load(self, request: WorkflowRequest) -> PreparedWorkflowResult | None: ...
    def bind_proof(self, request: WorkflowRequest, proof: AcceptedOutcomeProof) -> AcceptedOutcomeProof: ...
    def mark_committed(self, request: WorkflowRequest, proof: AcceptedOutcomeProof) -> None: ...


class _SqlWorkflowResultStore:
    """Shared SQL behavior; subclasses supply connection/locking dialect only."""

    def __init__(self, factory: Callable, *, namespace: str, postgres: bool) -> None:
        if not namespace.strip():
            raise ValueError("result store namespace is required")
        self._factory = factory
        self._namespace = namespace
        self._postgres = postgres
        with self._transaction() as cursor:
            cursor.execute("""CREATE TABLE IF NOT EXISTS market_agent_workflow_results (
                namespace TEXT NOT NULL, workflow_id TEXT NOT NULL,
                trace_id TEXT NOT NULL, request_digest TEXT NOT NULL,
                execution_json TEXT NOT NULL, proof_json TEXT,
                committed INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(namespace, workflow_id))""")

    @contextmanager
    def _transaction(self):
        connection = self._factory()
        cursor = connection.cursor()
        try:
            if not self._postgres:
                cursor.execute("BEGIN IMMEDIATE")
            yield cursor
            connection.commit()
        except BaseException:
            connection.rollback()
            raise
        finally:
            cursor.close()
            connection.close()

    def _execute(self, cursor, sql: str, values: tuple):
        cursor.execute(sql.replace("?", "%s") if self._postgres else sql, values)

    def _select(self, cursor, request: WorkflowRequest) -> PreparedWorkflowResult | None:
        self._execute(cursor,
            "SELECT trace_id,request_digest,execution_json,proof_json,committed "
            "FROM market_agent_workflow_results WHERE namespace=? AND workflow_id=?"
            + (" FOR UPDATE" if self._postgres else ""),
            (self._namespace, request.workflow_id))
        row = cursor.fetchone()
        if row is None:
            return None
        if row[0] != request.trace_id or row[1] != canonical_workflow_request_digest(request):
            raise ValueError("durable result request binding does not match")
        execution = WorkflowExecution.model_validate_json(row[2])
        if (execution.result.workflow_id, execution.result.trace_id) != (request.workflow_id, request.trace_id):
            raise ValueError("durable result identity does not match request")
        if execution.prompt_release_digest is None:
            raise ValueError("durable result has no immutable prompt binding")
        proof = AcceptedOutcomeProof.model_validate_json(row[3]) if row[3] else None
        if proof is not None:
            proof.verify(request, execution.result)
            if proof.prompt_release_digest != execution.prompt_release_digest:
                raise ValueError("durable proof prompt binding does not match execution")
        if row[4] not in (0, 1) or (row[4] == 1 and proof is None):
            raise ValueError("durable commit marker has no verified proof")
        return PreparedWorkflowResult(row[1], execution, proof, bool(row[4]))

    def load(self, request: WorkflowRequest) -> PreparedWorkflowResult | None:
        request = WorkflowRequest.model_validate(request)
        with self._transaction() as cursor:
            return self._select(cursor, request)

    def stage(self, request: WorkflowRequest, execution: WorkflowExecution) -> PreparedWorkflowResult:
        request = WorkflowRequest.model_validate(request)
        execution = WorkflowExecution.model_validate(execution)
        if (execution.result.workflow_id, execution.result.trace_id) != (request.workflow_id, request.trace_id):
            raise ValueError("staged result identity does not match request")
        if execution.prompt_release_digest is None:
            raise ValueError("staged result requires immutable prompt binding")
        with self._transaction() as cursor:
            self._execute(cursor,
                "INSERT INTO market_agent_workflow_results "
                "(namespace,workflow_id,trace_id,request_digest,execution_json) VALUES (?,?,?,?,?) "
                "ON CONFLICT(namespace,workflow_id) DO NOTHING",
                (self._namespace, request.workflow_id, request.trace_id,
                 canonical_workflow_request_digest(request), execution.model_dump_json()))
            saved = self._select(cursor, request)
            if saved is None or saved.execution != execution:
                raise ValueError("durable execution cannot be replaced")
            return saved

    def bind_proof(self, request: WorkflowRequest, proof: AcceptedOutcomeProof) -> AcceptedOutcomeProof:
        request = WorkflowRequest.model_validate(request)
        proof = AcceptedOutcomeProof.model_validate(proof)
        with self._transaction() as cursor:
            saved = self._select(cursor, request)
            if saved is None:
                raise ValueError("accepted result must be staged before proof binding")
            proof.verify(request, saved.execution.result)
            if proof.prompt_release_digest != saved.execution.prompt_release_digest:
                raise ValueError("accepted proof does not bind staged prompt")
            if saved.proof is not None:
                if saved.proof.receipt_digest != proof.receipt_digest:
                    raise ValueError("accepted terminal receipt cannot be replaced")
                return saved.proof
            self._execute(cursor,
                "UPDATE market_agent_workflow_results SET proof_json=? WHERE namespace=? AND workflow_id=?",
                (proof.model_dump_json(), self._namespace, request.workflow_id))
            return proof

    def mark_committed(self, request: WorkflowRequest, proof: AcceptedOutcomeProof) -> None:
        request = WorkflowRequest.model_validate(request)
        proof = AcceptedOutcomeProof.model_validate(proof)
        with self._transaction() as cursor:
            saved = self._select(cursor, request)
            if saved is None or saved.proof != proof:
                raise ValueError("commit marker requires the bound durable proof")
            self._execute(cursor,
                "UPDATE market_agent_workflow_results SET committed=1 WHERE namespace=? AND workflow_id=?",
                (self._namespace, request.workflow_id))

    def healthcheck(self) -> bool:
        with self._transaction() as cursor:
            cursor.execute("SELECT 1 FROM market_agent_workflow_results LIMIT 1")
        return True


class SqliteWorkflowResultStore(_SqlWorkflowResultStore):
    def __init__(self, path: str | Path, *, namespace: str = "default") -> None:
        path = Path(path)
        if str(path) == ":memory:":
            raise ValueError("result recovery requires a durable SQLite file")
        path.parent.mkdir(parents=True, exist_ok=True)
        super().__init__(lambda: sqlite3.connect(path, timeout=30), namespace=namespace, postgres=False)


class PostgresWorkflowResultStore(_SqlWorkflowResultStore):
    def __init__(self, connection_factory: Callable, *, namespace: str) -> None:
        super().__init__(connection_factory, namespace=namespace, postgres=True)
