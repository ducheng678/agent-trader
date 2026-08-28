from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class CacheStats:
    hits: int
    misses: int
    evictions: int
    size: int


class CacheBackend(Protocol):
    def get(self, key: str, default: T | None = None) -> T | None:
        ...

    def set(self, key: str, value: Any, ttl_seconds: float | None = None) -> None:
        ...

    def delete(self, key: str) -> None:
        ...


@dataclass
class _CacheEntry:
    value: Any
    expires_at: float


class TTLCache:
    def __init__(self, max_entries: int, default_ttl_seconds: float) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be positive")
        if default_ttl_seconds <= 0:
            raise ValueError("default_ttl_seconds must be positive")
        self._max_entries = int(max_entries)
        self._default_ttl_seconds = float(default_ttl_seconds)
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
        self._evictions = 0

    def get(self, key: str, default: T | None = None) -> T | None:
        normalized_key = str(key)
        with self._lock:
            entry = self._entries.get(normalized_key)
            if entry is None or entry.expires_at <= time.monotonic():
                if entry is not None:
                    self._entries.pop(normalized_key, None)
                self._misses += 1
                return default
            self._entries.move_to_end(normalized_key)
            self._hits += 1
            return entry.value

    def set(self, key: str, value: Any, ttl_seconds: float | None = None) -> None:
        lifetime = self._default_ttl_seconds if ttl_seconds is None else float(ttl_seconds)
        if lifetime <= 0:
            self.delete(str(key))
            return
        normalized_key = str(key)
        with self._lock:
            self._entries[normalized_key] = _CacheEntry(value=value, expires_at=time.monotonic() + lifetime)
            self._entries.move_to_end(normalized_key)
            while len(self._entries) > self._max_entries:
                self._entries.popitem(last=False)
                self._evictions += 1

    def delete(self, key: str) -> None:
        with self._lock:
            self._entries.pop(str(key), None)

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def stats(self) -> CacheStats:
        with self._lock:
            self._purge_expired()
            return CacheStats(self._hits, self._misses, self._evictions, len(self._entries))

    def _purge_expired(self) -> None:
        now = time.monotonic()
        for key in tuple(self._entries):
            if self._entries[key].expires_at <= now:
                self._entries.pop(key, None)
