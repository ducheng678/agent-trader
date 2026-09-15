from __future__ import annotations

import json
import sqlite3
from hashlib import sha256

import pytest

from market_agent.backend.prompt_activation_store import (
    PromptActivationState,
    PromptReleasePointer,
    PostgresPromptActivationStore,
    SQLitePromptActivationStore,
)
from market_agent.workflow_agent_contracts import ModelTier
from market_agent.workflow_prompt_config import (
    PromptConfigurationError,
    PromptReleaseManager,
    PromptReleaseManifest,
)
from market_agent.workflow_prompt_release import PromptRelease, canonical_json


def _release(identifier: str, prefix: str) -> PromptRelease:
    values = {
        "schema_version": "v1",
        "release_id": identifier,
        "stable_system_prefix": prefix,
        "supported_task_kinds": ("extract",),
        "supported_model_tiers": (ModelTier.LUNA,),
        "temperature_profile": ((ModelTier.LUNA, 0.0),),
    }
    return PromptRelease(
        digest=sha256(canonical_json(values).encode("utf-8")).hexdigest(),
        **values,
    )


def _manifest(release: PromptRelease) -> PromptReleaseManifest:
    values = {
        "schema_version": "v1",
        "release": release.model_dump(mode="python"),
        "output_schema_hash": "0" * 64,
    }
    return PromptReleaseManifest(
        manifest_hash=sha256(canonical_json(values).encode("utf-8")).hexdigest(),
        **values,
    )


def _pointer(manifest: PromptReleaseManifest) -> PromptReleasePointer:
    return PromptReleasePointer(
        release_id=manifest.release.release_id,
        release_digest=manifest.release.digest,
        manifest_hash=manifest.manifest_hash,
    )


def _manager(tmp_path, name: str, manifests, store) -> PromptReleaseManager:
    return PromptReleaseManager(
        manifests=tuple(manifests),
        registry_path=tmp_path / f"{name}-outbox.sqlite3",
        activation_store=store,
    )


def test_two_managers_observe_shared_activation_and_rollback_without_changing_existing_pins(tmp_path):
    """Reading a process-local pointer or mutating a captured pin would fail this test."""
    first = _manifest(_release("release-a", "Stable release A."))
    second = _manifest(_release("release-b", "Stable release B."))
    shared_path = tmp_path / "shared-prompts.sqlite3"
    manager_a = _manager(
        tmp_path, "a", (first, second), SQLitePromptActivationStore(shared_path)
    )
    manager_b = _manager(
        tmp_path, "b", (first, second), SQLitePromptActivationStore(shared_path)
    )

    assert manager_a.activation_store is not None
    manager_a.activate("release-a")
    original_pin = manager_b.current()
    manager_b.activate("release-b")
    in_flight_pin = manager_a.current()
    rolled_back = manager_a.rollback_previous()

    assert rolled_back.active_release_id == "release-a"
    assert manager_b.current().release_id == "release-a"
    assert original_pin.release_id == "release-a"
    assert original_pin.release.stable_system_prefix == "Stable release A."
    assert in_flight_pin.release_id == "release-b"
    assert in_flight_pin.release.stable_system_prefix == "Stable release B."


def test_stale_compare_and_swap_cannot_overwrite_a_concurrent_activation(tmp_path):
    """Dropping the revision predicate would let a stale writer erase a newer activation."""
    first = _manifest(_release("release-a", "Stable release A."))
    second = _manifest(_release("release-b", "Stable release B."))
    store = SQLitePromptActivationStore(tmp_path / "shared-prompts.sqlite3")
    initial = store.bootstrap(_pointer(first))

    changed = store.compare_and_swap(
        initial.revision,
        _pointer(second),
        initial.active,
        "activate",
    )
    stale = store.compare_and_swap(
        initial.revision,
        initial.active,
        None,
        "rollback",
    )

    assert changed == PromptActivationState(
        active=_pointer(second), previous=_pointer(first), revision=2
    )
    assert stale is None
    assert store.read() == changed


def test_current_fails_closed_when_shared_release_has_no_local_manifest(tmp_path):
    """Trusting only a shared release ID would run a release absent from the local checkout."""
    first = _manifest(_release("release-a", "Stable release A."))
    second = _manifest(_release("release-b", "Stable release B."))
    store = SQLitePromptActivationStore(tmp_path / "shared-prompts.sqlite3")
    store.bootstrap(_pointer(second))
    manager = _manager(tmp_path, "only-a", (first,), store)

    with pytest.raises(PromptConfigurationError, match="manifest.*unavailable"):
        manager.current()


def test_current_fails_closed_when_shared_digest_does_not_match_local_manifest(tmp_path):
    """Checking only a release ID would accept locally different prompt content."""
    manifest = _manifest(_release("release-a", "Stable release A."))
    store = SQLitePromptActivationStore(tmp_path / "shared-prompts.sqlite3")
    store.bootstrap(PromptReleasePointer(
        release_id="release-a",
        release_digest="f" * 64,
        manifest_hash=manifest.manifest_hash,
    ))
    manager = _manager(tmp_path, "digest", (manifest,), store)

    with pytest.raises(PromptConfigurationError, match="digest.*does not match"):
        manager.current()


class _UnavailableStore:
    def read(self):
        raise OSError("shared database unavailable")

    def bootstrap(self, active):
        raise OSError("shared database unavailable")

    def compare_and_swap(self, expected_revision, active, previous, action):
        raise OSError("shared database unavailable")


def test_injected_shared_store_outage_never_falls_back_to_local_registry(tmp_path):
    """Falling back to the local SQLite pointer would split production authority."""
    manifest = _manifest(_release("release-a", "Stable release A."))
    local = PromptReleaseManager(
        manifests=(manifest,), registry_path=tmp_path / "outbox.sqlite3"
    )
    local.activate("release-a")
    manager = _manager(tmp_path, "outbox", (manifest,), _UnavailableStore())

    with pytest.raises(PromptConfigurationError, match="shared prompt activation store"):
        manager.current()
    with pytest.raises(PromptConfigurationError, match="shared prompt activation store"):
        manager.activate("release-a")


def test_sqlite_state_change_and_append_only_audit_are_committed_together(tmp_path):
    """Updating state without its audit row, or allowing audit mutation, would fail this test."""
    first = _manifest(_release("release-a", "Stable release A."))
    second = _manifest(_release("release-b", "Stable release B."))
    path = tmp_path / "shared-prompts.sqlite3"
    store = SQLitePromptActivationStore(path, namespace="tenant-a")
    initial = store.bootstrap(_pointer(first))
    store.compare_and_swap(initial.revision, _pointer(second), initial.active, "activate")

    with sqlite3.connect(path) as connection:
        rows = connection.execute(
            "SELECT revision,action,active_release_id,previous_release_id "
            "FROM market_agent_prompt_activation_audit "
            "WHERE namespace=? ORDER BY revision",
            ("tenant-a",),
        ).fetchall()
        assert rows == [
            (1, "activate", "release-a", None),
            (2, "activate", "release-b", "release-a"),
        ]
        with pytest.raises(sqlite3.DatabaseError, match="append-only"):
            connection.execute(
                "DELETE FROM market_agent_prompt_activation_audit WHERE namespace=?",
                ("tenant-a",),
            )


def test_shared_outbox_replays_after_the_committed_cas_loses_its_local_audit(tmp_path, monkeypatch):
    """A second manager must recover an audit that the committing manager could not persist."""
    first = _manifest(_release("release-a", "Stable release A."))
    second = _manifest(_release("release-b", "Stable release B."))
    store = SQLitePromptActivationStore(tmp_path / "shared-prompts.sqlite3")
    initial = store.bootstrap(_pointer(first))
    assert store.acknowledge_audit(initial.revision) is True
    observed = []
    manager_a = _manager(tmp_path, "a", (first, second), store)
    manager_b = PromptReleaseManager(
        manifests=(first, second),
        registry_path=tmp_path / "b-outbox.sqlite3",
        activation_store=store,
        audit_hook=lambda activation, _pin: observed.append(activation),
    )

    def fail_local_audit(*_args, **_kwargs):
        raise OSError("local audit disk unavailable")

    monkeypatch.setattr(manager_a, "_record_local_audit", fail_local_audit)
    with pytest.raises(OSError, match="local audit"):
        manager_a.activate("release-b")

    assert manager_b.current().release_id == "release-b"
    assert manager_b.replay_pending_audit() == 1
    assert len(observed) == 1
    assert observed[0].active_release_id == "release-b"
    assert observed[0].previous_release_id == "release-a"
    assert observed[0].action == "activate"
    assert observed[0].revision == 2
    assert manager_b.replay_pending_audit() == 0


def test_shared_outbox_ack_is_separate_from_the_immutable_audit(tmp_path):
    """Delivery acknowledgement must not mutate or delete the shared audit row."""
    manifest = _manifest(_release("release-a", "Stable release A."))
    store = SQLitePromptActivationStore(tmp_path / "shared-prompts.sqlite3")
    state = store.bootstrap(_pointer(manifest))

    assert [record.revision for record in store.pending_audits()] == [state.revision]
    assert store.acknowledge_audit(state.revision) is True
    assert store.acknowledge_audit(state.revision) is False
    assert store.pending_audits() == ()
    with sqlite3.connect(tmp_path / "shared-prompts.sqlite3") as connection:
        assert connection.execute(
            "SELECT action FROM market_agent_prompt_activation_audit WHERE namespace=? AND revision=?",
            ("default", state.revision),
        ).fetchone() == ("activate",)


def test_rollback_audit_replay_preserves_the_immediate_previous_release(tmp_path):
    first = _manifest(_release("release-a", "Stable release A."))
    second = _manifest(_release("release-b", "Stable release B."))
    store = SQLitePromptActivationStore(tmp_path / "shared-prompts.sqlite3")
    seen = []
    manager = PromptReleaseManager(
        manifests=(first, second), registry_path=tmp_path / "outbox.sqlite3",
        activation_store=store, audit_hook=lambda activation, _pin: seen.append(activation),
    )
    manager.activate("release-a")
    manager.activate("release-b")
    rollback = manager.rollback_previous()
    assert rollback.previous_release_id == "release-b"
    assert store.read().previous is None
    assert seen[-1] == rollback
    # Simulate acknowledgement loss: the immutable event must replay identically.
    with sqlite3.connect(tmp_path / "shared-prompts.sqlite3") as connection:
        connection.execute(
            "DELETE FROM market_agent_prompt_activation_delivery WHERE namespace=? AND revision=?",
            ("default", rollback.revision),
        )
    assert manager.replay_pending_audit() == 1
    assert seen[-1] == rollback


def test_postgres_migration_guards_shared_audit_against_updates_and_deletes():
    """Production SQL must enforce the same immutable-audit policy as SQLite."""
    statements = []

    class Cursor:
        def execute(self, sql, _values=()):
            statements.append(sql)

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

    PostgresPromptActivationStore(lambda: Connection()).migrate()

    sql = "\n".join(statements)
    assert "RETURNS trigger" in sql
    assert "BEFORE UPDATE OR DELETE ON market_agent_prompt_activation_audit" in sql
    assert "prompt activation audit is append-only" in sql
