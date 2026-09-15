from __future__ import annotations

from market_agent.backend.prompt_activation_store import (
    PostgresPromptActivationStore,
    PromptReleasePointer,
    SQLitePromptActivationStore,
)


def _pointer(release_id: str) -> PromptReleasePointer:
    return PromptReleasePointer(
        release_id=release_id,
        release_digest=("a" * 64),
        manifest_hash=("b" * 64),
    )


def test_acknowledging_only_existing_audit_revision_preserves_future_delivery(tmp_path):
    store = SQLitePromptActivationStore(tmp_path / "shared.sqlite3")
    initial = store.bootstrap(_pointer("release-a"))

    assert store.acknowledge_audit(1) is True
    assert store.acknowledge_audit(2) is False
    changed = store.compare_and_swap(initial.revision, _pointer("release-b"), initial.active, "activate")
    assert changed is not None
    assert [audit.revision for audit in store.pending_audits()] == [2]
    assert store.acknowledge_audit(2) is True
    assert store.acknowledge_audit(2) is False
    assert store.pending_audits() == ()


def test_acknowledgement_is_namespace_isolated(tmp_path):
    path = tmp_path / "shared.sqlite3"
    tenant_a = SQLitePromptActivationStore(path, namespace="tenant-a")
    tenant_b = SQLitePromptActivationStore(path, namespace="tenant-b")
    tenant_a.bootstrap(_pointer("release-a"))
    tenant_b.bootstrap(_pointer("release-b"))

    assert tenant_a.acknowledge_audit(1) is True
    assert tenant_b.pending_audits()[0].revision == 1


def test_postgres_acknowledgement_uses_existing_audit_insert_select_boundary():
    statements: list[tuple[str, tuple[object, ...]]] = []

    class Cursor:
        rowcount = 1

        def execute(self, sql, values=()):
            statements.append((sql, values))

        def close(self):
            pass

    class Connection:
        def cursor(self):
            return Cursor()

        def commit(self):
            pass

        def rollback(self):
            pass

        def close(self):
            pass

    assert PostgresPromptActivationStore(lambda: Connection()).acknowledge_audit(7) is True
    sql, values = statements[0]
    assert "INSERT INTO market_agent_prompt_activation_delivery (namespace,revision)" in sql
    assert "SELECT namespace,revision FROM market_agent_prompt_activation_audit" in sql
    assert "WHERE namespace=%s AND revision=%s" in sql
    assert "ON CONFLICT(namespace,revision) DO NOTHING" in sql
    assert values == ("default", 7)
