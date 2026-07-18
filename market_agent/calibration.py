import os
import re
from typing import Any, List, Optional

from market_agent.symbols import canonicalize_execution_symbol, normalize_candidate_key, split_execution_symbol
from market_agent.utils import safe_float


DEFAULT_TRIGGER_CONFIDENCE_RELEVANCE_THRESHOLD = 0.20
DEFAULT_TRIGGER_CONFIDENCE_FULL_SCALE = 0.70
DEFAULT_OPPOSITE_EVENT_TRIM_THRESHOLD = 0.40
DEFAULT_OPPOSITE_EVENT_REVERSE_EDGE = 0.15
DEFAULT_OPPOSITE_EVENT_MIN_REVERSE_CONFIDENCE = 0.60
DEFAULT_OPPOSITE_EVENT_UNKNOWN_BASIS_REVERSE_THRESHOLD = 0.85
DEFAULT_OPPOSITE_EVENT_FULL_CONFIDENCE = 0.85
DEFAULT_OPPOSITE_EVENT_FLATTEN_CLOSE_FRACTION = 0.85
DEFAULT_OPPOSITE_EVENT_BASIS_VALIDITY_DECAY = 0.50


def _symbol_env_suffix_candidates(symbol: Any) -> List[str]:
    raw = str(symbol or "").strip()
    if not raw:
        return []
    exact = canonicalize_execution_symbol(raw)
    dex, asset = split_execution_symbol(exact)
    market_name = asset or exact
    display_symbol = market_name
    if display_symbol and not re.search(r"[-_]USDC$", display_symbol, re.IGNORECASE):
        display_symbol = f"{display_symbol}-USDC"
    candidates: List[str] = []
    seen: set = set()
    for value in (
        raw,
        exact,
        market_name,
        display_symbol,
        normalize_candidate_key(raw),
        normalize_candidate_key(exact),
        normalize_candidate_key(market_name),
        normalize_candidate_key(display_symbol),
    ):
        suffix = re.sub(r"[^A-Z0-9]+", "_", str(value or "").strip().upper()).strip("_")
        if not suffix or suffix in seen:
            continue
        seen.add(suffix)
        candidates.append(suffix)
    return candidates


def _trigger_confidence_symbol_env_suffix(symbol: Any) -> str:
    candidates = _symbol_env_suffix_candidates(symbol)
    if not candidates:
        return ""
    for value in candidates:
        if value.endswith("_USDC"):
            return value
    return candidates[0]


def get_trigger_confidence_calibration(symbol: Any = "") -> dict:
    suffix = _trigger_confidence_symbol_env_suffix(symbol)
    threshold = None
    full_scale = None
    if suffix:
        threshold = safe_float(os.getenv(f"TRIGGER_CONFIDENCE_RELEVANCE_THRESHOLD_{suffix}"), None)
        full_scale = safe_float(os.getenv(f"TRIGGER_CONFIDENCE_FULL_SCALE_{suffix}"), None)
    if threshold is None:
        threshold = safe_float(
            os.getenv("TRIGGER_CONFIDENCE_RELEVANCE_THRESHOLD", str(DEFAULT_TRIGGER_CONFIDENCE_RELEVANCE_THRESHOLD)),
            DEFAULT_TRIGGER_CONFIDENCE_RELEVANCE_THRESHOLD,
        )
    if full_scale is None:
        full_scale = safe_float(
            os.getenv("TRIGGER_CONFIDENCE_FULL_SCALE", str(DEFAULT_TRIGGER_CONFIDENCE_FULL_SCALE)),
            DEFAULT_TRIGGER_CONFIDENCE_FULL_SCALE,
        )
    threshold = min(max(float(threshold or 0.0), 0.0), 1.0)
    full_scale = min(max(float(full_scale or 0.0), 0.0), 1.0)
    if full_scale <= threshold:
        full_scale = min(1.0, threshold + 0.01)
    return {
        "relevance_threshold": threshold,
        "full_scale": full_scale,
    }


def _symbol_float_env(symbol: Any, name: str, default: float) -> float:
    value = None
    for suffix in _symbol_env_suffix_candidates(symbol):
        value = safe_float(os.getenv(f"{name}_{suffix}"), None)
        if value is not None:
            break
    if value is None:
        value = safe_float(os.getenv(name, str(default)), default)
    return float(value if value is not None else default)


def get_opposite_event_config(symbol: Any = "") -> dict:
    trim_threshold = min(max(_symbol_float_env(symbol, "OPPOSITE_EVENT_TRIM_THRESHOLD", DEFAULT_OPPOSITE_EVENT_TRIM_THRESHOLD), 0.0), 1.0)
    reverse_edge = max(0.0, _symbol_float_env(symbol, "OPPOSITE_EVENT_REVERSE_EDGE", DEFAULT_OPPOSITE_EVENT_REVERSE_EDGE))
    min_reverse_confidence = min(
        max(_symbol_float_env(symbol, "OPPOSITE_EVENT_MIN_REVERSE_CONFIDENCE", DEFAULT_OPPOSITE_EVENT_MIN_REVERSE_CONFIDENCE), 0.0),
        1.0,
    )
    unknown_basis_reverse_threshold = min(
        max(
            _symbol_float_env(
                symbol,
                "OPPOSITE_EVENT_UNKNOWN_BASIS_REVERSE_THRESHOLD",
                DEFAULT_OPPOSITE_EVENT_UNKNOWN_BASIS_REVERSE_THRESHOLD,
            ),
            0.0,
        ),
        1.0,
    )
    full_confidence = min(max(_symbol_float_env(symbol, "OPPOSITE_EVENT_FULL_CONFIDENCE", DEFAULT_OPPOSITE_EVENT_FULL_CONFIDENCE), 0.0), 1.0)
    if full_confidence <= trim_threshold:
        full_confidence = min(1.0, trim_threshold + 0.01)
    flatten_close_fraction = min(
        max(_symbol_float_env(symbol, "OPPOSITE_EVENT_FLATTEN_CLOSE_FRACTION", DEFAULT_OPPOSITE_EVENT_FLATTEN_CLOSE_FRACTION), 0.0),
        1.0,
    )
    basis_validity_decay = min(
        max(_symbol_float_env(symbol, "OPPOSITE_EVENT_BASIS_VALIDITY_DECAY", DEFAULT_OPPOSITE_EVENT_BASIS_VALIDITY_DECAY), 0.0),
        1.0,
    )
    return {
        "trim_threshold": trim_threshold,
        "reverse_edge": reverse_edge,
        "min_reverse_confidence": min_reverse_confidence,
        "unknown_basis_reverse_threshold": unknown_basis_reverse_threshold,
        "full_confidence": full_confidence,
        "flatten_close_fraction": flatten_close_fraction,
        "basis_validity_decay": basis_validity_decay,
    }


def normalize_confidence_value(value: Any, symbol: Any = "") -> Optional[float]:
    numeric = safe_float(value, None)
    if numeric is None:
        return None
    calibration = get_trigger_confidence_calibration(symbol)
    threshold = float(calibration["relevance_threshold"])
    full_scale = float(calibration["full_scale"])
    span = max(full_scale - threshold, 0.01)
    return min(max((float(numeric) - threshold) / span, 0.0), 1.0)


def extract_raw_confidence_value(value: Any) -> Optional[float]:
    return safe_float(value, None)
