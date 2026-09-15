from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    case_id: str = Field(min_length=3, max_length=128)
    dataset_version: str = Field(min_length=1, max_length=64)
    partition: str
    scenario: str
    request_class: str
    invariants: tuple[str, ...] = Field(min_length=1, max_length=32)
    metadata: dict[str, str] = Field(default_factory=dict)


def load_evaluation_dataset(path: str | Path) -> tuple[EvaluationCase, ...]:
    cases: list[EvaluationCase] = []
    identifiers: set[str] = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            case = EvaluationCase.model_validate(json.loads(line))
        except Exception as error:
            raise ValueError(f"invalid evaluation case at line {number}") from error
        if case.partition not in {"train", "development", "holdout"}:
            raise ValueError(f"invalid evaluation partition at line {number}")
        if case.case_id in identifiers:
            raise ValueError(f"duplicate evaluation case ID at line {number}")
        identifiers.add(case.case_id)
        cases.append(case)
    if not cases:
        raise ValueError("evaluation dataset must contain at least one case")
    return tuple(cases)
