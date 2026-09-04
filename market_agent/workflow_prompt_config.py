"""Git-tracked prompt release loading and atomic local activation management."""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from hashlib import sha256
import json
from pathlib import Path
import re
import sqlite3
import subprocess
from threading import RLock
from typing import Callable, Iterable, Iterator, Protocol

from pydantic import Field, model_validator

from market_agent.workflow_agent_contracts import ModelTier
from market_agent.workflow_contracts import ContractModel, Digest, ShortText
from market_agent.workflow_prompt_release import PromptRelease, PromptReleaseRegistry, canonical_json


_PLACEHOLDER = re.compile(r"\{\{|\}\}|\$\{|\{(?:trace_id|task_id|tenant_id|workflow_id|run_id|now|timestamp)\}")


class PromptConfigurationError(RuntimeError):
    pass


class PromptReleaseManifest(ContractModel):
    release: PromptRelease
    output_schema_hash: Digest
    manifest_hash: Digest

    @model_validator(mode="after")
    def validate_manifest(self) -> PromptReleaseManifest:
        content = self.model_dump(mode="json", exclude={"manifest_hash"})
        if self.manifest_hash != sha256(canonical_json(content).encode("utf-8")).hexdigest():
            raise ValueError("prompt manifest hash does not match canonical content")
        if _PLACEHOLDER.search(self.release.stable_system_prefix):
            raise ValueError("stable system prefix cannot contain dynamic runtime placeholders")
        return self


@dataclass(frozen=True, slots=True)
class PromptPin:
    release_id: str
    release_digest: str
    output_schema_hash: str
    manifest_hash: str
    release: PromptRelease


@dataclass(frozen=True, slots=True)
class PromptReleaseComponent:
    """One agent-specific release frozen into a workflow-wide prompt pin."""

    profile_id: str
    output_schema_digest: str
    release: PromptRelease


@dataclass(frozen=True, slots=True)
class WorkflowPromptPin:
    """Immutable base release plus every prompt component used by one run."""

    base: PromptPin
    components: tuple[PromptReleaseComponent, ...]
    release_digest: str

    @classmethod
    def capture(
        cls,
        base: PromptPin,
        components: Iterable[tuple[PromptRelease, str]],
    ) -> WorkflowPromptPin:
        base = base if isinstance(base, PromptPin) else PromptPin(**base)
        frozen: list[PromptReleaseComponent] = []
        for component, schema_digest in components:
            component = PromptRelease.model_validate(component)
            if type(schema_digest) is not str or not re.fullmatch(r"[0-9a-f]{64}", schema_digest):
                raise ValueError("prompt component schema digest is invalid")
            frozen.append(PromptReleaseComponent(
                profile_id=component.release_id,
                output_schema_digest=schema_digest,
                release=component,
            ))
        if not frozen or len({item.profile_id for item in frozen}) != len(frozen):
            raise ValueError("workflow prompt components must be non-empty and unique")
        frozen.sort(key=lambda item: item.profile_id)
        digest = sha256(canonical_json({
            "base_release_id": base.release_id,
            "base_release_digest": base.release_digest,
            "base_manifest_hash": base.manifest_hash,
            "components": tuple({
                "profile_id": item.profile_id,
                "release_digest": item.release.digest,
                "output_schema_digest": item.output_schema_digest,
            } for item in frozen),
        }).encode("utf-8")).hexdigest()
        return cls(base=base, components=tuple(frozen), release_digest=digest)

    @property
    def release_id(self) -> str:
        return self.base.release_id

    def component(self, profile_id: str, output_schema_digest: str) -> PromptReleaseComponent:
        match = next((item for item in self.components if item.profile_id == profile_id), None)
        if match is None or match.output_schema_digest != output_schema_digest:
            raise PromptConfigurationError("invocation is not bound to the workflow prompt pin")
        return match

    def registry(self) -> PromptReleaseRegistry:
        return PromptReleaseRegistry(releases=tuple(item.release for item in self.components))

    def system_prefix(self, profile_id: str, output_schema_digest: str) -> str:
        component = self.component(profile_id, output_schema_digest)
        return (
            self.base.release.stable_system_prefix.rstrip()
            + "\n"
            + component.release.stable_system_prefix.lstrip()
        )


@dataclass(frozen=True, slots=True)
class PromptActivation:
    active_release_id: str
    previous_release_id: str | None
    action: str


@dataclass(frozen=True, slots=True)
class PendingPromptAudit:
    sequence: int
    activation: PromptActivation
    release_id: str
    release_digest: str
    manifest_hash: str


class ReleaseGate(Protocol):
    def __call__(self, pin: PromptPin, action: str) -> bool: ...


class ReleaseHook(Protocol):
    def __call__(self, activation: PromptActivation, pin: PromptPin) -> object: ...


def load_git_tracked_releases(*, prompts_root: Path, git_root: Path) -> tuple[PromptReleaseManifest, ...]:
    root = prompts_root.resolve(strict=True)
    repository = git_root.resolve(strict=True)
    if root != repository and repository not in root.parents:
        raise PromptConfigurationError("prompts root must be inside the git root")
    manifests: list[PromptReleaseManifest] = []
    for path in sorted(root.rglob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise PromptConfigurationError("prompt manifest must be a regular tracked file")
        relative = path.resolve(strict=True).relative_to(repository).as_posix()
        completed = subprocess.run(
            ("git", "-C", str(repository), "ls-files", "--error-unmatch", "--", relative),
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        if completed.returncode != 0:
            raise PromptConfigurationError(f"prompt manifest is not git tracked: {relative}")
        try:
            decoded = json.loads(path.read_text(encoding="utf-8"))
            release = decoded.get("release") if isinstance(decoded, dict) else None
            if isinstance(release, dict):
                for field_name in ("supported_task_kinds", "supported_model_tiers", "temperature_profile"):
                    if isinstance(release.get(field_name), list):
                        release[field_name] = tuple(
                            tuple(item) if field_name == "temperature_profile" and isinstance(item, list) else item
                            for item in release[field_name]
                        )
                if isinstance(release.get("supported_model_tiers"), tuple):
                    release["supported_model_tiers"] = tuple(ModelTier(item) for item in release["supported_model_tiers"])
                if isinstance(release.get("temperature_profile"), tuple):
                    release["temperature_profile"] = tuple((ModelTier(item[0]), item[1]) for item in release["temperature_profile"])
            manifests.append(PromptReleaseManifest.model_validate(decoded))
        except Exception as error:
            raise PromptConfigurationError(f"invalid prompt manifest: {relative}") from error
    if not manifests:
        raise PromptConfigurationError("no git-tracked prompt manifests were found")
    identifiers = tuple(item.release.release_id for item in manifests)
    if len(identifiers) != len(set(identifiers)):
        raise PromptConfigurationError("prompt release IDs must be unique")
    return tuple(manifests)


class PromptReleaseManager:
    """Pins immutable releases while atomically switching a durable active pointer."""

    def __init__(
        self,
        *,
        manifests: tuple[PromptReleaseManifest, ...],
        registry_path: Path,
        release_gate: ReleaseGate | None = None,
        audit_hook: ReleaseHook | None = None,
        metric_hook: ReleaseHook | None = None,
    ) -> None:
        if not manifests:
            raise ValueError("at least one prompt manifest is required")
        copied = tuple(PromptReleaseManifest.model_validate(item.model_dump(mode="python")) for item in manifests)
        ids = tuple(item.release.release_id for item in copied)
        if len(ids) != len(set(ids)):
            raise ValueError("prompt release IDs must be unique")
        self._manifests = {item.release.release_id: item for item in copied}
        self._registry_path = Path(registry_path)
        self._release_gate = release_gate
        self._audit_hook = audit_hook
        self._metric_hook = metric_hook
        self._lock = RLock()
        self._registry_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_registry()

    @classmethod
    def from_git(
        cls,
        *,
        prompts_root: Path,
        git_root: Path,
        registry_path: Path,
        release_gate: ReleaseGate | None = None,
        audit_hook: ReleaseHook | None = None,
        metric_hook: ReleaseHook | None = None,
    ) -> PromptReleaseManager:
        return cls(
            manifests=load_git_tracked_releases(prompts_root=prompts_root, git_root=git_root),
            registry_path=registry_path,
            release_gate=release_gate,
            audit_hook=audit_hook,
            metric_hook=metric_hook,
        )

    def current(self) -> PromptPin:
        with self._lock, self._connection() as connection:
            active, _ = self._state(connection)
        if active is None:
            raise PromptConfigurationError("no active prompt release")
        return self._pin(active)

    def pin(self, release_id: str | None = None) -> PromptPin:
        if release_id is None:
            return self.current()
        return self._pin(release_id)

    def registry(self) -> PromptReleaseRegistry:
        """Return immutable copies of every Git-validated release for a driver.

        Selection remains an ingress-time operation through ``current``.  The
        registry only provides the driver's immutable digest and capability
        validation boundary.
        """
        with self._lock:
            return PromptReleaseRegistry(releases=tuple(
                PromptRelease.model_validate(manifest.release.model_dump(mode="python"))
                for _, manifest in sorted(self._manifests.items())
            ))

    def activate(self, release_id: str) -> PromptActivation:
        pin = self._pin(release_id)
        self._require_gate(pin, "activate")
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active, _ = self._state(connection)
            if active == pin.release_id:
                connection.commit()
                return PromptActivation(active_release_id=pin.release_id, previous_release_id=active, action="activate_noop")
            connection.execute("UPDATE prompt_release_registry SET active_release_id = ?, previous_release_id = ? WHERE registry_id = 1", (pin.release_id, active))
            self._record_activation(connection, PromptActivation(
                active_release_id=pin.release_id, previous_release_id=active, action="activate"), pin)
            connection.commit()
        activation = PromptActivation(active_release_id=pin.release_id, previous_release_id=active, action="activate")
        self._emit(activation, pin)
        return activation

    def rollback_previous(self) -> PromptActivation:
        with self._lock, self._connection() as connection:
            connection.execute("BEGIN IMMEDIATE")
            active, previous = self._state(connection)
            if active is None or previous is None:
                connection.rollback()
                raise PromptConfigurationError("no previous prompt release is available for rollback")
            pin = self._pin(previous)
            self._require_gate(pin, "rollback")
            connection.execute("UPDATE prompt_release_registry SET active_release_id = ?, previous_release_id = NULL WHERE registry_id = 1", (previous,))
            self._record_activation(connection, PromptActivation(
                active_release_id=previous, previous_release_id=active, action="rollback"), pin)
            connection.commit()
        activation = PromptActivation(active_release_id=previous, previous_release_id=active, action="rollback")
        self._emit(activation, pin)
        return activation

    def replay_pending_audit(self, *, limit: int = 100) -> int:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("prompt audit replay limit is invalid")
        if self._audit_hook is None:
            return 0
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT sequence,active_release_id,previous_release_id,action,release_digest,manifest_hash "
                "FROM prompt_release_audit WHERE external_audit_delivered=0 ORDER BY sequence LIMIT ?",
                (limit,),
            ).fetchall()
        delivered = 0
        for sequence, active, previous, action, digest, manifest_hash in rows:
            pin = self._pin(str(active))
            if pin.release_digest != digest or pin.manifest_hash != manifest_hash:
                raise PromptConfigurationError("pending prompt audit no longer matches Git release")
            activation = PromptActivation(active_release_id=str(active), previous_release_id=previous, action=str(action))
            try:
                self._audit_hook(activation, pin)
            except Exception:
                continue
            with self._lock, self._connection() as connection:
                changed = connection.execute(
                    "UPDATE prompt_release_audit SET external_audit_delivered=1 "
                    "WHERE sequence=? AND external_audit_delivered=0",
                    (sequence,),
                ).rowcount
            delivered += int(changed == 1)
        return delivered

    def pending_audits(self, *, limit: int = 100) -> tuple[PendingPromptAudit, ...]:
        if type(limit) is not int or not 1 <= limit <= 1000:
            raise ValueError("prompt pending audit limit is invalid")
        with self._lock, self._connection() as connection:
            rows = connection.execute(
                "SELECT sequence,active_release_id,previous_release_id,action,release_digest,manifest_hash "
                "FROM prompt_release_audit WHERE external_audit_delivered=0 ORDER BY sequence LIMIT ?",
                (limit,),
            ).fetchall()
        return tuple(PendingPromptAudit(
            sequence=int(sequence),
            activation=PromptActivation(active_release_id=str(active), previous_release_id=previous, action=str(action)),
            release_id=str(active), release_digest=str(digest), manifest_hash=str(manifest_hash),
        ) for sequence, active, previous, action, digest, manifest_hash in rows)

    def _initialize_registry(self) -> None:
        with self._connection() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS prompt_release_registry (registry_id INTEGER PRIMARY KEY CHECK (registry_id = 1), active_release_id TEXT, previous_release_id TEXT)")
            connection.execute("INSERT OR IGNORE INTO prompt_release_registry (registry_id, active_release_id, previous_release_id) VALUES (1, NULL, NULL)")
            connection.execute("CREATE TABLE IF NOT EXISTS prompt_release_audit (sequence INTEGER PRIMARY KEY AUTOINCREMENT, active_release_id TEXT NOT NULL, previous_release_id TEXT, action TEXT NOT NULL, release_digest TEXT NOT NULL, manifest_hash TEXT NOT NULL, external_audit_delivered INTEGER NOT NULL DEFAULT 0 CHECK (external_audit_delivered IN (0,1)))")

    @contextmanager
    def _connection(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self._registry_path, isolation_level=None)
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            yield connection
        finally:
            connection.close()

    @staticmethod
    def _state(connection: sqlite3.Connection) -> tuple[str | None, str | None]:
        row = connection.execute("SELECT active_release_id, previous_release_id FROM prompt_release_registry WHERE registry_id = 1").fetchone()
        if row is None or any(value is not None and type(value) is not str for value in row):
            raise PromptConfigurationError("prompt release registry is malformed")
        return row[0], row[1]

    def _pin(self, release_id: str) -> PromptPin:
        if type(release_id) is not str or not release_id:
            raise PromptConfigurationError("prompt release ID is invalid")
        manifest = self._manifests.get(release_id)
        if manifest is None:
            raise PromptConfigurationError("unknown prompt release")
        release = PromptRelease.model_validate(manifest.release.model_dump(mode="python"))
        return PromptPin(
            release_id=release.release_id,
            release_digest=release.digest,
            output_schema_hash=manifest.output_schema_hash,
            manifest_hash=manifest.manifest_hash,
            release=release,
        )

    def _require_gate(self, pin: PromptPin, action: str) -> None:
        if self._release_gate is not None and self._release_gate(pin, action) is not True:
            raise PromptConfigurationError("prompt release gate denied activation")

    def _emit(self, activation: PromptActivation, pin: PromptPin) -> None:
        if self._audit_hook is not None:
            try:
                self._audit_hook(activation, pin)
            except Exception:
                pass
            else:
                with self._lock, self._connection() as connection:
                    connection.execute(
                        "UPDATE prompt_release_audit SET external_audit_delivered = 1 WHERE sequence = (SELECT MAX(sequence) FROM prompt_release_audit WHERE active_release_id = ? AND action = ?)",
                        (activation.active_release_id, activation.action),
                    )
        if self._metric_hook is not None:
            try:
                self._metric_hook(activation, pin)
            except Exception:
                pass

    @staticmethod
    def _record_activation(connection: sqlite3.Connection, activation: PromptActivation,
                           pin: PromptPin) -> None:
        connection.execute(
            "INSERT INTO prompt_release_audit (active_release_id,previous_release_id,action,release_digest,manifest_hash) VALUES (?,?,?,?,?)",
            (activation.active_release_id, activation.previous_release_id, activation.action,
             pin.release_digest, pin.manifest_hash),
        )


def default_prompt_manager(*, registry_path: Path, git_root: Path | None = None,
                           release_gate: ReleaseGate | None = None,
                           audit_hook: ReleaseHook | None = None,
                           metric_hook: ReleaseHook | None = None) -> PromptReleaseManager:
    root = (git_root or Path(__file__).resolve().parents[1]).resolve()
    manager = PromptReleaseManager.from_git(
        prompts_root=root / "prompts",
        git_root=root,
        registry_path=registry_path,
        release_gate=release_gate,
        audit_hook=audit_hook,
        metric_hook=metric_hook,
    )
    try:
        manager.current()
    except PromptConfigurationError as error:
        if str(error) != "no active prompt release":
            raise
        manager.activate("default-v1")
    return manager
