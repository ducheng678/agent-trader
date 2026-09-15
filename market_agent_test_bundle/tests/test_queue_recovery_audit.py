from __future__ import annotations

import pytest

from market_agent.backend.database import JobRepository, PostgresJobRepository


def _job(repository: JobRepository):
    return repository.create_or_get_job("echo", {"value": 1}, None, 2, "request")[0]


def test_successful_recovery_claim_appends_one_audit_event(tmp_path):
    repository = JobRepository(tmp_path / "jobs.db")
    job = _job(repository)

    claimed = repository.claim_job(job.job_id, "recovery-owner", 30, recovery=True)

    assert claimed is not None
    assert [event.event_type for event in repository.list_events(job.job_id)] == [
        "task_accepted",
        "task_recovery_queued",
    ]


def test_failed_recovery_claim_does_not_append_an_audit_event(tmp_path):
    repository = JobRepository(tmp_path / "jobs.db")
    job = _job(repository)
    assert repository.claim_job(job.job_id, "healthy-owner", 30) is not None

    assert repository.claim_job(job.job_id, "recovery-owner", 30, recovery=True) is None
    assert [event.event_type for event in repository.list_events(job.job_id)] == ["task_accepted"]


@pytest.mark.parametrize("claim_succeeds", [True, False])
def test_postgres_recovery_audit_is_written_only_for_a_successful_claim(claim_succeeds):
    statements: list[str] = []
    claimed_row = (
        "job-1", "echo", "accepted", "{}", None, "fingerprint", None, None,
        0, 2, "request", "2026-09-09T00:00:00+00:00", "2026-09-09T00:00:00+00:00",
        "owner", "2026-09-09T00:00:30+00:00",
    )

    class Cursor:
        def __init__(self):
            self.returned = None

        def execute(self, statement, parameters=None):
            statements.append(" ".join(statement.split()))
            if statement.startswith("UPDATE market_agent_jobs"):
                self.returned = claimed_row if claim_succeeds else None

        def fetchone(self):
            return self.returned

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

    claimed = PostgresJobRepository(Connection).claim_job("job-1", "owner", 30, recovery=True)

    assert (claimed is not None) is claim_succeeds
    assert any("INSERT INTO market_agent_job_events" in statement for statement in statements) is claim_succeeds
