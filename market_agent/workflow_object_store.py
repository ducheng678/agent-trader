"""Tenant-scoped immutable filesystem blobs with checksum verification."""
from __future__ import annotations

from contextlib import closing
import hashlib
import os
from pathlib import Path
import sqlite3
import tempfile
from typing import Unpack

from market_agent.workflow_long_term_memory import (
    ArtifactReference, ArtifactStore, MemoryAuthorityError, MemoryConflictError,
    MemoryIntegrityError, WriteArguments, validate_authority,
)


class FileArtifactStore:
    def __init__(self, root: str | Path, *, writer_authority: object | None = None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._authority = writer_authority
        self._manifest = self.root / "manifest.sqlite3"
        with closing(sqlite3.connect(self._manifest)) as db, db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS artifacts (
                    tenant_id TEXT NOT NULL, sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                    PRIMARY KEY(tenant_id, sha256)
                );
                CREATE TABLE IF NOT EXISTS artifact_audit (
                    tenant_id TEXT NOT NULL, key_digest TEXT NOT NULL, trace_id TEXT NOT NULL,
                    sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                    PRIMARY KEY(tenant_id, key_digest)
                );
                CREATE TABLE IF NOT EXISTS artifact_deletion_audit (
                    tenant_id TEXT NOT NULL, sha256 TEXT NOT NULL, key_digest TEXT NOT NULL,
                    trace_id TEXT NOT NULL, size_bytes INTEGER NOT NULL,
                    PRIMARY KEY(tenant_id, sha256), UNIQUE(tenant_id, key_digest)
                );
            """)

    def _path(self, reference: ArtifactReference) -> Path:
        tenant = hashlib.sha256(reference.tenant_id.encode()).hexdigest()
        return self.root / tenant / reference.sha256

    def put(self, data: bytes, **context: Unpack[WriteArguments]) -> ArtifactReference:
        ctx = validate_authority(self._authority, **context)
        if type(data) is not bytes:
            raise TypeError("artifact contents must be immutable bytes")
        ref = ArtifactReference(tenant_id=ctx.tenant_id, sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
        key_digest = hashlib.sha256(ctx.idempotency_key.encode()).hexdigest()
        with closing(sqlite3.connect(self._manifest, timeout=30)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT 1 FROM artifact_deletion_audit WHERE tenant_id=? AND sha256=?",
                           (ctx.tenant_id, ref.sha256)).fetchone():
                raise MemoryConflictError("deleted artifact content cannot be resurrected by put replay")
            replay = db.execute("SELECT trace_id,sha256,size_bytes FROM artifact_audit WHERE tenant_id=? AND key_digest=?",
                                (ctx.tenant_id, key_digest)).fetchone()
            if replay is not None:
                if replay != (ctx.trace_id, ref.sha256, ref.size_bytes):
                    raise MemoryConflictError("artifact idempotency key is already bound")
                self.get(ref, tenant_id=ctx.tenant_id)
                return ref
            target = self._path(ref)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                # Publish only complete, synced bytes. A crash may leave an unreferenced
                # blob, never a manifest entry pointing at a partial blob.
                temporary = None
                try:
                    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as pending:
                        temporary = Path(pending.name)
                        pending.write(data)
                        pending.flush()
                        os.fsync(pending.fileno())
                    try:
                        os.link(temporary, target)
                    except FileExistsError:
                        pass
                finally:
                    if temporary is not None:
                        temporary.unlink(missing_ok=True)
            actual = target.read_bytes()
            if len(actual) != ref.size_bytes or hashlib.sha256(actual).hexdigest() != ref.sha256:
                raise MemoryIntegrityError("artifact checksum mismatch")
            db.execute("INSERT OR IGNORE INTO artifacts VALUES(?,?,?)", (ctx.tenant_id, ref.sha256, ref.size_bytes))
            db.execute("INSERT INTO artifact_audit VALUES(?,?,?,?,?)",
                       (ctx.tenant_id, key_digest, ctx.trace_id, ref.sha256, ref.size_bytes))
        return ref

    def get(self, reference: ArtifactReference, *, tenant_id: str) -> bytes:
        reference = ArtifactReference.model_validate(reference)
        if reference.tenant_id != tenant_id:
            raise MemoryAuthorityError("artifact access must match tenant scope")
        with closing(sqlite3.connect(self._manifest)) as db, db:
            row = db.execute("SELECT size_bytes FROM artifacts WHERE tenant_id=? AND sha256=?",
                             (tenant_id, reference.sha256)).fetchone()
        if row is None:
            raise FileNotFoundError("artifact is not registered in this tenant")
        data = self._path(reference).read_bytes()
        if row[0] != reference.size_bytes or len(data) != reference.size_bytes or hashlib.sha256(data).hexdigest() != reference.sha256:
            raise MemoryIntegrityError("artifact checksum mismatch")
        return data

    def delete(self, reference: ArtifactReference, **context: Unpack[WriteArguments]) -> None:
        """Idempotent derivative removal; the repository checks live references.

        Delete intent commits before unlink so a crash cannot permit a new put
        to resurrect the address. Replays finish a previously interrupted unlink.
        """
        ctx = validate_authority(self._authority, **context)
        reference = ArtifactReference.model_validate(reference)
        if reference.tenant_id != ctx.tenant_id:
            raise MemoryAuthorityError("artifact deletion must share tenant scope")
        key = hashlib.sha256(ctx.idempotency_key.encode()).hexdigest()
        with closing(sqlite3.connect(self._manifest, timeout=30)) as db, db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute("SELECT sha256,trace_id,size_bytes FROM artifact_deletion_audit WHERE tenant_id=? AND key_digest=?",
                                (ctx.tenant_id, key)).fetchone()
            if prior is not None and prior != (reference.sha256, ctx.trace_id, reference.size_bytes):
                raise MemoryConflictError("artifact cleanup idempotency key is already bound")
            other = db.execute("SELECT key_digest,trace_id,size_bytes FROM artifact_deletion_audit WHERE tenant_id=? AND sha256=?",
                                (ctx.tenant_id, reference.sha256)).fetchone()
            if other is not None and other != (key, ctx.trace_id, reference.size_bytes):
                raise MemoryConflictError("artifact cleanup is already owned by another task")
            db.execute("INSERT OR IGNORE INTO artifact_deletion_audit VALUES(?,?,?,?,?)",
                        (ctx.tenant_id, reference.sha256, key, ctx.trace_id, reference.size_bytes))
            db.execute("DELETE FROM artifacts WHERE tenant_id=? AND sha256=?", (ctx.tenant_id, reference.sha256))
        target = self._path(reference)
        root = self.root.resolve()
        if not target.resolve().is_relative_to(root):
            raise MemoryIntegrityError("artifact path escapes the configured store")
        target.unlink(missing_ok=True)


__all__ = ["ArtifactStore", "FileArtifactStore"]
