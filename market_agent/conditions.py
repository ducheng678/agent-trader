import time
from typing import Deque, List, Optional, Tuple

from market_agent.models import Condition


def history_slice(
    history: Deque[Tuple[float, float]],
    timer_seconds: int,
    *,
    now: Optional[float] = None,
    since_ts: Optional[float] = None,
) -> List[Tuple[float, float]]:
    if now is None:
        now = time.time()
    window_start = since_ts if since_ts is not None else float("-inf")
    if timer_seconds > 0:
        window_start = max(window_start, now - timer_seconds)
    return [(ts, px) for ts, px in history if ts >= window_start]


def effective_condition_tolerance_bps(condition: Condition, default: float = 0.0) -> float:
    return max(0.0, float(condition.tolerance_bps or 0.0) or default)


def effective_condition_min_ratio(condition: Condition, default: float = 0.85) -> float:
    raw = float(condition.min_ratio or 0.0)
    if raw <= 0.0:
        raw = default
    return min(max(raw, 0.0), 1.0)


def level_floor(level: float, tolerance_bps: float) -> float:
    return float(level) * (1.0 - max(0.0, tolerance_bps) / 10000.0)


def level_ceiling(level: float, tolerance_bps: float) -> float:
    return float(level) * (1.0 + max(0.0, tolerance_bps) / 10000.0)


def band_with_tolerance(low: float, high: float, tolerance_bps: float) -> Tuple[float, float]:
    lo, hi = sorted([float(low or 0.0), float(high or 0.0)])
    if lo <= 0.0 and hi <= 0.0:
        return lo, hi
    ref = max(abs(lo), abs(hi), 1.0)
    pad = ref * max(0.0, tolerance_bps) / 10000.0
    return lo - pad, hi + pad


def sample_ratio(samples: List[Tuple[float, float]], predicate) -> float:
    if not samples:
        return 0.0
    hits = sum(1 for _, px in samples if predicate(px))
    return hits / float(len(samples))


def crossed_above_in_samples(samples: List[Tuple[float, float]], level: float) -> bool:
    prev_px: Optional[float] = None
    for _, px in samples:
        if prev_px is not None and prev_px < level <= px:
            return True
        prev_px = px
    return False


def crossed_below_in_samples(samples: List[Tuple[float, float]], level: float) -> bool:
    prev_px: Optional[float] = None
    for _, px in samples:
        if prev_px is not None and prev_px > level >= px:
            return True
        prev_px = px
    return False


def evaluate_condition(
    condition: Condition,
    price: float,
    prev_price: Optional[float],
    history: Deque[Tuple[float, float]],
    *,
    now: Optional[float] = None,
    since_ts: Optional[float] = None,
) -> bool:
    if now is None:
        now = time.time()
    t = condition.type
    tol_bps = effective_condition_tolerance_bps(condition)
    if t == "price_ge":
        return price >= level_floor(condition.level, tol_bps)
    if t == "price_le":
        return price <= level_ceiling(condition.level, tol_bps)
    if t == "price_between":
        lo, hi = band_with_tolerance(condition.low, condition.high, tol_bps)
        return lo <= price <= hi
    if t == "cross_above":
        crossed_now = prev_price is not None and prev_price < condition.level <= price
        if condition.timer_seconds <= 0:
            return crossed_now
        samples = history_slice(history, condition.timer_seconds, now=now, since_ts=since_ts)
        if len(samples) < 2:
            return crossed_now
        soft_floor = level_floor(condition.level, tol_bps)
        latest_on_side = price >= soft_floor
        return crossed_above_in_samples(samples, condition.level) and latest_on_side
    if t == "cross_below":
        crossed_now = prev_price is not None and prev_price > condition.level >= price
        if condition.timer_seconds <= 0:
            return crossed_now
        samples = history_slice(history, condition.timer_seconds, now=now, since_ts=since_ts)
        if len(samples) < 2:
            return crossed_now
        soft_ceiling = level_ceiling(condition.level, tol_bps)
        latest_on_side = price <= soft_ceiling
        return crossed_below_in_samples(samples, condition.level) and latest_on_side
    if since_ts is not None and condition.timer_seconds > 0 and now - since_ts < condition.timer_seconds:
        return False
    samples = history_slice(history, condition.timer_seconds, now=now, since_ts=since_ts)
    if not samples:
        return False
    if condition.timer_seconds > 0 and now - samples[0][0] + 1e-9 < condition.timer_seconds:
        return False
    prices = [px for _, px in samples]
    if t == "sustained_ge":
        soft_floor = level_floor(condition.level, tol_bps)
        hard = all(px >= condition.level for px in prices)
        min_ratio = effective_condition_min_ratio(condition)
        ratio_ok = sample_ratio(samples, lambda px: px >= soft_floor) >= min_ratio
        return price >= soft_floor and (hard or ratio_ok)
    if t == "sustained_le":
        soft_ceiling = level_ceiling(condition.level, tol_bps)
        hard = all(px <= condition.level for px in prices)
        min_ratio = effective_condition_min_ratio(condition)
        ratio_ok = sample_ratio(samples, lambda px: px <= soft_ceiling) >= min_ratio
        return price <= soft_ceiling and (hard or ratio_ok)
    if t == "sustained_between":
        lo, hi = sorted([condition.low, condition.high])
        soft_lo, soft_hi = band_with_tolerance(lo, hi, tol_bps)
        hard = all(lo <= px <= hi for px in prices)
        min_ratio = effective_condition_min_ratio(condition)
        ratio_ok = sample_ratio(samples, lambda px: soft_lo <= px <= soft_hi) >= min_ratio
        return soft_lo <= price <= soft_hi and (hard or ratio_ok)
    return False
