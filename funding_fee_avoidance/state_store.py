from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterator, Mapping, Optional

from .file_locking import exclusive_file_lock
from .models import HedgeCycleState


class CycleTransaction:
    def __init__(self, cycles: Mapping[str, HedgeCycleState], flush_callback) -> None:
        self.cycles: Dict[str, HedgeCycleState] = dict(cycles)
        self.dirty = False
        self._flush_callback = flush_callback

    def get(self, symbol: str) -> Optional[HedgeCycleState]:
        return self.cycles.get(symbol)

    def put(self, cycle: HedgeCycleState) -> None:
        self.cycles[cycle.symbol] = cycle
        self.dirty = True

    def flush(self) -> None:
        if self.dirty:
            self._flush_callback(self.cycles)
            self.dirty = False


class CycleStateStore:
    """Schema-checked, atomic state with a process lock.

    The coordinator holds this lock across intent persistence and order
    submission, so two watcher processes cannot own the same cycle.
    """

    SCHEMA_VERSION = 1

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def _read_unlocked(self) -> Dict[str, HedgeCycleState]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise ValueError("hedge state must be a JSON object")
        if int(payload.get("schema_version", 0)) != self.SCHEMA_VERSION:
            raise ValueError("unsupported hedge-state schema version")
        raw_cycles = payload.get("cycles", {})
        if not isinstance(raw_cycles, dict):
            raise ValueError("hedge state cycles must be an object")
        result: Dict[str, HedgeCycleState] = {}
        for symbol, raw_cycle in raw_cycles.items():
            if not isinstance(raw_cycle, dict):
                raise ValueError(f"invalid hedge cycle for {symbol}")
            cycle = HedgeCycleState.from_mapping(raw_cycle)
            if cycle.symbol != symbol:
                raise ValueError(f"hedge cycle key mismatch for {symbol}")
            result[symbol] = cycle
        return result

    def _write_unlocked(self, cycles: Mapping[str, HedgeCycleState]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": self.SCHEMA_VERSION,
            "cycles": {
                symbol: cycle.to_dict() for symbol, cycle in sorted(cycles.items())
            },
        }
        fd, temporary_name = tempfile.mkstemp(
            prefix=self.path.name + ".",
            suffix=".tmp",
            dir=str(self.path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_name, self.path)
        finally:
            if os.path.exists(temporary_name):
                os.unlink(temporary_name)

    @contextmanager
    def transaction(self) -> Iterator[CycleTransaction]:
        with exclusive_file_lock(self.lock_path):
            transaction = CycleTransaction(self._read_unlocked(), self._write_unlocked)
            yield transaction
            transaction.flush()

    def load_all(self) -> Dict[str, HedgeCycleState]:
        with self.transaction() as transaction:
            return dict(transaction.cycles)
