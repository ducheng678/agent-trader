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
from typing import Callable, Iterator, Protocol

from pydantic import Field, model_validator

from market_agent.workflow_agent_contracts import ModelTier
from market_agent.workflow_contracts import ContractModel, Digest, ShortText
from market_agent.workflow_prompt_release import PromptRelease, canonical_json


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
class PromptActivation:
    active_release_id: str
    previous_release_id: str | None
    action: str


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
            connection.commit()
        activation = PromptActivation(active_release_id=previous, previous_release_id=active, action="rollback")
        self._emit(activation, pin)
        return activation

    def _initialize_registry(self) -> None:
        with self._connection() as connection:
            connection.execute("CREATE TABLE IF NOT EXISTS prompt_release_registry (registry_id INTEGER PRIMARY KEY CHECK (registry_id = 1), active_release_id TEXT, previous_release_id TEXT)")
            connection.execute("INSERT OR IGNORE INTO prompt_release_registry (registry_id, active_release_id, previous_release_id) VALUES (1, NULL, NULL)")

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
        for hook in (self._audit_hook, self._metric_hook):
            if hook is not None:
                try:
                    hook(activation, pin)
                except Exception:
                    continue


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
