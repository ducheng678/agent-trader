from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import re
from typing import Literal

from pydantic import Field, model_validator

from market_agent.workflow_contracts import Digest, NonNegativeFinite, NonNegativeInt, PositiveInt, ShortText, Text
from market_agent.workflow_long_term_memory import FrozenPayload, MemoryContract, canonical_json, content_hash
from market_agent.workflow_tracing import TraceId


Violation = Literal["order_execution", "risk_bypass", "cross_trace", "permission_bypass", "secret_exposure", "unsupported_claim", "unsafe_cache", "missing_reflection", "invalid_schema", "evidence_fabrication"]


class CaseProvenance(MemoryContract):
    source_kind: Literal["synthetic", "sanitized_recording"]
    reference: ShortText
    license: Literal["CC0-1.0", "internal-approved"]
    sanitized: Literal[True]
    description: Text
    provenance_hash: Digest | None = None

    @model_validator(mode="after")
    def bind_provenance(self):
        digest = content_hash(self.model_dump(mode="json", exclude={"provenance_hash"}))
        if self.provenance_hash is not None and self.provenance_hash != digest:
            raise ValueError("evaluation provenance hash mismatch")
        object.__setattr__(self, "provenance_hash", digest)
        return self


class ExpectedBehavior(MemoryContract):
    conclusion: Literal["answer", "不知道", "no_trade"]
    action: Literal["no_trade", "long", "short"] = "no_trade"
    required_facts: tuple[Text, ...] = Field(default_factory=tuple, max_length=20)
    forbidden_facts: tuple[Text, ...] = Field(default_factory=tuple, max_length=20)
    allowed_facts: tuple[Text, ...] = Field(default_factory=tuple, max_length=20)
    required_evidence_ids: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=20)
    maximum_latency_seconds: NonNegativeFinite = 30.0
    maximum_tokens: PositiveInt = 4096
    maximum_cost_usd: NonNegativeFinite = 0.1

    @model_validator(mode="after")
    def safe_unknown(self):
        if self.conclusion in ("不知道", "no_trade") and self.action != "no_trade":
            raise ValueError("abstention cases cannot expect a trade")
        if set(self.required_facts) & set(self.forbidden_facts):
            raise ValueError("required and forbidden assertions overlap")
        return self


class RecordedObservation(MemoryContract):
    output: FrozenPayload
    latency_seconds: NonNegativeFinite
    input_tokens: NonNegativeInt
    output_tokens: NonNegativeInt
    cost_usd: NonNegativeFinite
    observed_violations: tuple[Violation, ...] = ()


class EvaluationAnswer(MemoryContract):
    conclusion: Literal["answer", "不知道", "no_trade"]
    action: Literal["no_trade", "long", "short"]
    facts: tuple[Text, ...] = Field(default_factory=tuple, max_length=20)
    evidence_ids: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=20)
    trace_id: TraceId


def _normalized_input(value):
    if isinstance(value, dict):
        return {key: _normalized_input(item) for key, item in value.items() if key not in {"trace_id", "request_id", "run_id"}}
    if isinstance(value, (tuple, list)):
        return [_normalized_input(item) for item in value]
    return " ".join(value.split()).casefold() if isinstance(value, str) else value


class EvaluationCase(MemoryContract):
    schema_version: Literal["evaluation-case-v1"] = "evaluation-case-v1"
    case_id: ShortText
    suite: Literal["regression", "abstention", "security", "resilience", "cache", "retrieval", "memory", "reflection", "permission", "trace"]
    leakage_group: ShortText
    trace_id: TraceId
    task_input: FrozenPayload
    allowed_evidence_ids: tuple[ShortText, ...] = Field(default_factory=tuple, max_length=50)
    provenance: CaseProvenance
    expected: ExpectedBehavior
    recording: RecordedObservation
    input_fingerprint: Digest | None = None
    case_hash: Digest | None = None

    @model_validator(mode="after")
    def integrity(self):
        if not set(self.expected.required_evidence_ids) <= set(self.allowed_evidence_ids):
            raise ValueError("required evidence is outside the case evidence inventory")
        fingerprint = content_hash(_normalized_input(self.model_dump(mode="json")["task_input"]))
        if self.input_fingerprint is not None and self.input_fingerprint != fingerprint:
            raise ValueError("evaluation input fingerprint mismatch")
        object.__setattr__(self, "input_fingerprint", fingerprint)
        digest = content_hash(self.model_dump(mode="json", exclude={"case_hash"}))
        if self.case_hash is not None and self.case_hash != digest:
            raise ValueError("evaluation case hash mismatch")
        object.__setattr__(self, "case_hash", digest)
        return self


class DatasetManifest(MemoryContract):
    schema_version: Literal["evaluation-dataset-v1"] = "evaluation-dataset-v1"
    dataset_id: ShortText
    dataset_version: Literal["v1"] = "v1"
    split: Literal["train", "development", "holdout"]
    corpus_file: ShortText
    corpus_sha256: Digest
    schema_file: ShortText
    schema_sha256: Digest
    case_count: PositiveInt
    manifest_hash: Digest | None = None

    @model_validator(mode="after")
    def integrity(self):
        digest = content_hash(self.model_dump(mode="json", exclude={"manifest_hash"}))
        if self.manifest_hash is not None and self.manifest_hash != digest:
            raise ValueError("evaluation manifest hash mismatch")
        object.__setattr__(self, "manifest_hash", digest)
        return self


class EvaluationDataset(MemoryContract):
    manifest: DatasetManifest
    cases: tuple[EvaluationCase, ...] = Field(min_length=1, max_length=10000)

    @model_validator(mode="after")
    def unique_cases(self):
        if len(self.cases) != self.manifest.case_count:
            raise ValueError("evaluation case count does not match the manifest")
        for name in ("case_id", "input_fingerprint", "case_hash"):
            if len({getattr(case, name) for case in self.cases}) != len(self.cases):
                raise ValueError("duplicate evaluation case or normalized input")
        return self

    @property
    def dataset_hash(self) -> str:
        return content_hash({"manifest": self.manifest.manifest_hash, "cases": [case.case_hash for case in self.cases]})

    def validate(self, other_datasets: tuple[EvaluationDataset, ...] = ()) -> EvaluationDataset:
        current = EvaluationDataset.model_validate(self)
        for other in other_datasets:
            other = EvaluationDataset.model_validate(other)
            if current.manifest.split == other.manifest.split:
                continue
            for field in ("case_id", "input_fingerprint", "leakage_group"):
                if {getattr(case, field) for case in current.cases} & {getattr(case, field) for case in other.cases}:
                    raise ValueError("evaluation split leakage detected")
        return current

    @classmethod
    def load(cls, manifest_path: str | Path, *, other_datasets: tuple[EvaluationDataset, ...] = ()) -> EvaluationDataset:
        path = Path(manifest_path).resolve()
        manifest = DatasetManifest.model_validate_json(_read(path))
        root = path.parent.parent
        corpus_path = (path.parent / manifest.corpus_file).resolve()
        schema_path = (path.parent / manifest.schema_file).resolve()
        if not corpus_path.is_relative_to(root) or not schema_path.is_relative_to(root):
            raise ValueError("evaluation manifest paths escape their root")
        corpus, schema = _read(corpus_path), _read(schema_path)
        if sha256(corpus).hexdigest() != manifest.corpus_sha256 or sha256(schema).hexdigest() != manifest.schema_sha256:
            raise ValueError("evaluation corpus or schema checksum mismatch")
        if json.loads(schema) != EvaluationCase.model_json_schema():
            raise ValueError("evaluation case schema is incompatible")
        text = corpus.decode("utf-8")
        if re.search(r"(?:sk-(?:live|proj|test)-[A-Za-z0-9]{8,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|Bearer [A-Za-z0-9._-]{12,})", text):
            raise ValueError("evaluation corpus contains credential-like material")
        lines = text.splitlines()
        if any(not line.strip() for line in lines):
            raise ValueError("evaluation JSONL cannot contain blank records")
        cases = tuple(EvaluationCase.model_validate_json(line) for line in lines)
        return cls(manifest=manifest, cases=cases).validate(other_datasets)


def _read(path: Path) -> bytes:
    if path.stat().st_size > 16 * 1024 * 1024:
        raise ValueError("evaluation input exceeds the 16 MiB file limit")
    return path.read_bytes().replace(b"\r\n", b"\n")
