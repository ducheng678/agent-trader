from __future__ import annotations

from dataclasses import asdict

from market_agent.workflow_historical_answer_cache import (
    HistoricalAnswerMetadata,
    HistoricalAnswerRecord,
    InMemoryHistoricalAnswerCache,
    PostgresHistoricalAnswerCache,
)
from market_agent.workflow_contracts import WorkflowMode, WorkflowRequest
from market_agent.workflow_production_application import _is_static_information


class _Cursor:
    def __init__(self, rows: tuple[tuple[object, ...], ...] = ()) -> None:
        self.query = ""
        self.params: tuple[object, ...] = ()
        self.rows = rows

    def __enter__(self) -> "_Cursor":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def execute(self, query: str, params: tuple[object, ...]) -> None:
        self.query = query
        self.params = params

    def fetchall(self) -> tuple[tuple[object, ...], ...]:
        return self.rows


class _Connection:
    def __init__(self, cursor: _Cursor) -> None:
        self._cursor = cursor

    def __enter__(self) -> "_Connection":
        return self

    def __exit__(self, *_: object) -> None:
        return None

    def cursor(self) -> _Cursor:
        return self._cursor


def _metadata(*, expires_at: float = 999.0) -> HistoricalAnswerMetadata:
    return HistoricalAnswerMetadata(
        tenant_scope="tenant-a",
        model_id="gpt-5.6-terra",
        model_version="terra-v1",
        embedding_model="text-embedding-3-small",
        embedding_version="embedding-v1",
        prompt_release_digest="a" * 64,
        output_schema_digest="b" * 64,
        safety_policy_version="safe-v1",
        locale="zh-CN",
        context_fingerprint="static-information-v1",
        knowledge_fingerprint="knowledge-v1",
        evidence_references=("evidence-a",),
        expires_at=expires_at,
    )


def _record(request_text: str) -> HistoricalAnswerRecord:
    return HistoricalAnswerRecord(
        entry_id="history-1",
        request_text=request_text,
        request_vector=(1.0, 0.0),
        response={"answer": "不会。系统没有下单权限。"},
        request_timestamp=1.0,
        response_timestamp=2.0,
        metadata=_metadata(),
    )


def test_postgres_lookup_excludes_dynamic_metadata_from_compatibility_predicate() -> None:
    """Expiry and citations change per request and must not make compatible rows unreachable."""
    cursor = _Cursor()
    cache = PostgresHistoricalAnswerCache(
        lambda: _Connection(cursor),
        embedding_dimension=2,
    )

    assert cache.lookup(
        (1.0, 0.0), _metadata(), now=10.0, query_text="系统会自动下单吗"
    ) is None

    assert "metadata=%s::jsonb" not in cursor.query
    rendered_parameters = repr(cursor.params)
    assert "evidence-a" not in rendered_parameters
    assert "999.0" not in rendered_parameters


def test_hybrid_lookup_accepts_semantic_paraphrase_with_compatible_keywords() -> None:
    """Rewording the same intent should still reuse a highly similar safe answer."""
    cache = InMemoryHistoricalAnswerCache()
    cache.put(_record("系统会自动下单吗"))

    hit = cache.lookup(
        (1.0, 0.0),
        _metadata(expires_at=500.0),
        now=10.0,
        query_text="这个系统会自动交易吗",
    )

    assert hit is not None
    assert hit.entry_id == "history-1"


def test_hybrid_lookup_rejects_negation_entity_and_number_conflicts() -> None:
    """Semantic proximity cannot override negation, symbol, or numeric differences."""
    cases = (
        ("系统会自动下单吗", "系统不会自动下单吗"),
        ("BTC 现在能买吗", "ETH 现在能买吗"),
        ("持有 5 分钟安全吗", "持有 15 分钟安全吗"),
    )
    for stored, query in cases:
        cache = InMemoryHistoricalAnswerCache()
        cache.put(_record(stored))
        assert cache.lookup(
            (1.0, 0.0),
            _metadata(),
            now=10.0,
            query_text=query,
        ) is None


def test_postgres_hybrid_gate_skips_conflicting_nearest_candidate() -> None:
    """A lexical conflict in the nearest vector must not hide the next valid neighbour."""
    saved_metadata = asdict(_metadata())
    rows = (
        ("conflict", "系统不会自动下单吗", "[1.0,0.0]", {"answer": "conflict"}, saved_metadata, 1.0, 1.0),
        ("compatible", "系统会自动下单吗", "[1.0,0.0]", {"answer": "safe"}, saved_metadata, 1.0, 2.0),
    )
    cursor = _Cursor(rows)
    cache = PostgresHistoricalAnswerCache(lambda: _Connection(cursor), embedding_dimension=2)

    hit = cache.lookup(
        (1.0, 0.0),
        _metadata(expires_at=500.0),
        now=10.0,
        query_text="这个系统会自动交易吗",
    )

    assert hit is not None
    assert hit.entry_id == "compatible"


def test_static_information_admission_is_independent_of_passive_event_mode() -> None:
    """Ordinary user questions must reach safe answer caches without pretending to be events."""
    request = WorkflowRequest(
        workflow_id="workflow-1",
        trace_id="1" * 32,
        user_query="这个系统会自动下单吗",
        trigger_reason="manual_once",
    )

    assert _is_static_information(request, WorkflowMode.ACTIVE)
    assert not _is_static_information(
        request.model_copy(update={"event_tape": ({"source": "market"},)}),
        WorkflowMode.ACTIVE,
    )
