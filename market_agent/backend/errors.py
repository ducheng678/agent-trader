from __future__ import annotations

from typing import Any


class BackendError(Exception):
    status_code = 500
    error_code = "backend_error"

    def __init__(self, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.message = message
        self.details = dict(details or {})


class ConfigurationError(BackendError):
    status_code = 500
    error_code = "configuration_error"


class ValidationError(BackendError):
    status_code = 422
    error_code = "validation_error"


class NotFoundError(BackendError):
    status_code = 404
    error_code = "not_found"


class ConflictError(BackendError):
    status_code = 409
    error_code = "conflict"


class AuthenticationError(BackendError):
    status_code = 401
    error_code = "authentication_error"


class DependencyUnavailableError(BackendError):
    status_code = 503
    error_code = "dependency_unavailable"


class UnknownTaskError(NotFoundError):
    error_code = "unknown_task"


class JobNotFoundError(NotFoundError):
    error_code = "job_not_found"


class IdempotencyConflictError(ConflictError):
    error_code = "idempotency_conflict"
