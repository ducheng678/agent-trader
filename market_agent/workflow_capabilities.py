"""Short-lived, least-privilege authority grants for workflow adapters."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import math
import secrets
from threading import RLock
from typing import Callable, Literal, NoReturn

from pydantic import Field, StrictFloat, model_validator

from market_agent.workflow_contracts import ContractModel, ShortText


CapabilityKind = Literal["read", "tool", "state_write", "service"]
CapabilityIds = tuple[ShortText, ...]
_MAX_GRANT_SECONDS = 300.0
_RESERVED_AUTHORITY = frozenset(
    {
        "audit",
        "audit.write",
        "audit.read",
        "durable_memory",
        "durable_memory.read",
        "durable_memory.write",
        "exchange",
        "exchange.read",
        "exchange.trade",
        "queue",
        "queue.read",
        "queue.write",
    }
)
_STATE_WRITE_PREFIXES = ("invocation.", "ephemeral.")


def _is_reserved_authority(value: str) -> bool:
    return any(
        value == reserved or value.startswith(reserved + ".") or value.startswith(reserved + ":")
        for reserved in _RESERVED_AUTHORITY
    )


def _validate_ids(values: CapabilityIds, field_name: str) -> CapabilityIds:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must be unique")
    for value in values:
        if _is_reserved_authority(value):
            raise ValueError(f"{field_name} cannot grant reserved authority")
    return values


class CapabilityScope(ContractModel):
    actor_id: ShortText
    task_id: ShortText
    tenant_id: ShortText
    trace_id: ShortText


class CapabilityGrant(ContractModel):
    grant_id: ShortText
    credential: ShortText
    scope: CapabilityScope
    issued_at: StrictFloat
    expires_at: StrictFloat
    readable_resources: CapabilityIds = Field(default_factory=tuple, max_length=64)
    tool_capabilities: CapabilityIds = Field(default_factory=tuple, max_length=32)
    writable_state_keys: CapabilityIds = Field(default_factory=tuple, max_length=16)
    service_capabilities: CapabilityIds = Field(default_factory=tuple, max_length=16)

    @model_validator(mode="after")
    def validate_grant(self) -> CapabilityGrant:
        if not math.isfinite(self.issued_at) or not math.isfinite(self.expires_at):
            raise ValueError("grant times must be finite")
        duration = self.expires_at - self.issued_at
        if duration <= 0.0 or duration > _MAX_GRANT_SECONDS:
            raise ValueError("grant lifetime is outside the permitted bound")
        _validate_ids(self.readable_resources, "readable resources")
        _validate_ids(self.tool_capabilities, "tool capabilities")
        _validate_ids(self.service_capabilities, "service capabilities")
        _validate_ids(self.writable_state_keys, "writable state keys")
        if any(not value.startswith(_STATE_WRITE_PREFIXES) for value in self.writable_state_keys):
            raise ValueError("state writes are limited to invocation or ephemeral state")
        return self


class CapabilityDenial(ContractModel):
    code: ShortText
    kind: CapabilityKind
    resource: ShortText
    scope: CapabilityScope
    grant_id: ShortText | None = None


class CapabilityDeniedError(PermissionError):
    def __init__(self, denial: CapabilityDenial) -> None:
        self.denial = denial
        super().__init__(f"capability denied: {denial.code}")


@dataclass(frozen=True, slots=True)
class CapabilityAuthorization:
    grant_id: str
    kind: CapabilityKind
    resource: str
    scope: CapabilityScope
    authorized_at: float
    expires_at: float


Clock = Callable[[], float]
CredentialFactory = Callable[[], str]


class CapabilityIssuer:
    """Host-owned issuer; only grants issued by this instance may authorize work."""

    def __init__(
        self,
        *,
        clock: Clock,
        credential_factory: CredentialFactory | None = None,
        maximum_grant_seconds: float = _MAX_GRANT_SECONDS,
    ) -> None:
        if not math.isfinite(maximum_grant_seconds) or not 0.0 < maximum_grant_seconds <= _MAX_GRANT_SECONDS:
            raise ValueError("maximum grant lifetime is outside the permitted bound")
        self._clock = clock
        self._credential_factory = credential_factory or (lambda: secrets.token_urlsafe(32))
        self._maximum_grant_seconds = maximum_grant_seconds
        self._active: dict[str, tuple[str, CapabilityGrant]] = {}
        self._lock = RLock()

    def issue(
        self,
        *,
        scope: CapabilityScope,
        ttl_seconds: float,
        readable_resources: CapabilityIds = (),
        tool_capabilities: CapabilityIds = (),
        writable_state_keys: CapabilityIds = (),
        service_capabilities: CapabilityIds = (),
    ) -> CapabilityGrant:
        if type(scope) is not CapabilityScope:
            raise TypeError("capability scope must be an exact CapabilityScope")
        if not math.isfinite(ttl_seconds) or not 0.0 < ttl_seconds <= self._maximum_grant_seconds:
            raise ValueError("grant lifetime is outside the issuer policy")
        now = self._now()
        credential = self._credential_factory()
        if type(credential) is not str or not credential.strip():
            raise ValueError("credential factory returned an invalid credential")
        grant = CapabilityGrant(
            grant_id=secrets.token_urlsafe(18),
            credential=credential,
            scope=scope,
            issued_at=now,
            expires_at=now + ttl_seconds,
            readable_resources=readable_resources,
            tool_capabilities=tool_capabilities,
            writable_state_keys=writable_state_keys,
            service_capabilities=service_capabilities,
        )
        frozen = self._copy_grant(grant)
        with self._lock:
            self._active[frozen.grant_id] = (self._credential_digest(frozen.credential), frozen)
        return frozen

    def revoke(self, grant: CapabilityGrant) -> None:
        if type(grant) is not CapabilityGrant:
            raise TypeError("grant must be an exact CapabilityGrant")
        with self._lock:
            self._active.pop(grant.grant_id, None)

    def authorize_read(self, grant: CapabilityGrant, *, scope: CapabilityScope, resource: str) -> CapabilityAuthorization:
        return self._authorize(grant, scope=scope, kind="read", resource=resource)

    def authorize_tool(self, grant: CapabilityGrant, *, scope: CapabilityScope, tool: str) -> CapabilityAuthorization:
        return self._authorize(grant, scope=scope, kind="tool", resource=tool)

    def authorize_state_write(self, grant: CapabilityGrant, *, scope: CapabilityScope, state_key: str) -> CapabilityAuthorization:
        return self._authorize(grant, scope=scope, kind="state_write", resource=state_key)

    def authorize_service_request(self, grant: CapabilityGrant, *, scope: CapabilityScope, service: str) -> CapabilityAuthorization:
        return self._authorize(grant, scope=scope, kind="service", resource=service)

    def _authorize(self, grant: CapabilityGrant, *, scope: CapabilityScope, kind: CapabilityKind, resource: str) -> CapabilityAuthorization:
        if type(scope) is not CapabilityScope:
            raise TypeError("authorization scope must be an exact CapabilityScope")
        if type(resource) is not str or not resource.strip():
            return self._deny("invalid_resource", kind, str(resource), scope, None)
        if _is_reserved_authority(resource):
            return self._deny("reserved_authority", kind, resource, scope, getattr(grant, "grant_id", None))
        if kind == "state_write" and not resource.startswith(_STATE_WRITE_PREFIXES):
            return self._deny("state_write_outside_ephemeral_scope", kind, resource, scope, getattr(grant, "grant_id", None))
        if type(grant) is not CapabilityGrant:
            return self._deny("unissued_grant", kind, resource, scope, None)
        try:
            candidate = self._copy_grant(grant)
        except Exception:
            return self._deny("invalid_grant", kind, resource, scope, grant.grant_id)
        now = self._now()
        with self._lock:
            active = self._active.get(candidate.grant_id)
        if active is None:
            return self._deny("revoked_or_unknown_grant", kind, resource, scope, candidate.grant_id)
        credential_digest, issued = active
        if not secrets.compare_digest(credential_digest, self._credential_digest(candidate.credential)) or candidate != issued:
            return self._deny("grant_integrity_mismatch", kind, resource, scope, candidate.grant_id)
        if candidate.scope != scope:
            return self._deny("scope_mismatch", kind, resource, scope, candidate.grant_id)
        if now >= candidate.expires_at:
            return self._deny("grant_expired", kind, resource, scope, candidate.grant_id)
        allowed = {
            "read": candidate.readable_resources,
            "tool": candidate.tool_capabilities,
            "state_write": candidate.writable_state_keys,
            "service": candidate.service_capabilities,
        }[kind]
        if resource not in allowed:
            return self._deny("resource_not_allowlisted", kind, resource, scope, candidate.grant_id)
        return CapabilityAuthorization(
            grant_id=candidate.grant_id,
            kind=kind,
            resource=resource,
            scope=scope,
            authorized_at=now,
            expires_at=candidate.expires_at,
        )

    @staticmethod
    def _copy_grant(grant: CapabilityGrant) -> CapabilityGrant:
        return CapabilityGrant.model_validate(grant.model_dump(mode="python"))

    @staticmethod
    def _credential_digest(credential: str) -> str:
        return sha256(credential.encode("utf-8")).hexdigest()

    def _now(self) -> float:
        value = self._clock()
        if type(value) not in (int, float) or isinstance(value, bool) or not math.isfinite(value):
            raise ValueError("clock returned an invalid time")
        return float(value)

    @staticmethod
    def _deny(code: str, kind: CapabilityKind, resource: str, scope: CapabilityScope, grant_id: str | None) -> NoReturn:
        raise CapabilityDeniedError(CapabilityDenial(
            code=code, kind=kind, resource=resource or "invalid", scope=scope, grant_id=grant_id,
        ))
