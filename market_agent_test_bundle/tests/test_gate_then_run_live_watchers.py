from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from gate_then_run_live_watchers import count_fail_rounds, should_start_live


def test_count_fail_rounds_sums_per_source_failures():
    summary = {
        "sources": {
            "a": {"fail_rounds": 0},
            "b": {"fail_rounds": 2},
            "c": {"fail_rounds": 1},
        }
    }

    assert count_fail_rounds(summary) == 3


def test_should_start_live_requires_clean_soak_and_authenticity():
    summary = {"sources": {"a": {"fail_rounds": 0}, "b": {"fail_rounds": 0}}}
    authenticity = {"mismatch_count": 0, "error_count": 0}

    assert should_start_live(summary, authenticity) is True
    assert should_start_live({"sources": {"a": {"fail_rounds": 1}}}, authenticity) is False
    assert should_start_live(summary, {"mismatch_count": 1, "error_count": 0}) is False
    assert should_start_live(summary, {"mismatch_count": 0, "error_count": 1}) is False
