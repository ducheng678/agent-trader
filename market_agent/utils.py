import math
from typing import Any, Optional


def safe_float(value: Any, default: Optional[float] = 0.0) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def format_query_amount(value: Optional[float]) -> str:
    if value is None:
        return "unknown"
    value = float(value)
    if abs(value) >= 1000:
        return f"{value:.2f}"
    return f"{value:.2f}".rstrip("0").rstrip(".")


def format_display_price(value: Optional[float], sig_digits: int = 5) -> str:
    if value is None:
        return ""
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        return ""
    magnitude = math.floor(math.log10(abs(numeric))) if numeric != 0 else 0
    decimals = sig_digits - 1 - magnitude
    rounded = round(numeric, decimals)
    if decimals > 0:
        return f"{rounded:.{decimals}f}".rstrip("0").rstrip(".")
    return f"{rounded:.0f}"


def format_query_value(value: Any) -> str:
    if value is None:
        return "unknown"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return format_query_amount(float(value))
    return str(value)


def clamp_int(value: Any, minimum: int, maximum: int, fallback: int) -> int:
    try:
        parsed = int(float(value))
    except Exception:
        parsed = int(fallback)
    return max(int(minimum), min(int(maximum), parsed))
