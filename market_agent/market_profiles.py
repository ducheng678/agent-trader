from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Tuple


def _time_tuple_seconds(value: Tuple[int, int, int]) -> int:
    hour, minute, second = value
    return int(hour) * 3600 + int(minute) * 60 + int(second)


@dataclass(frozen=True)
class LocalTimeWindow:
    start: Tuple[int, int, int]
    end: Tuple[int, int, int]

    def contains(self, local_dt: datetime) -> bool:
        value = local_dt.hour * 3600 + local_dt.minute * 60 + local_dt.second
        start = _time_tuple_seconds(self.start)
        end = _time_tuple_seconds(self.end)
        if start == end:
            return True
        if start < end:
            return start <= value < end
        return value >= start or value < end


@dataclass(frozen=True)
class InstrumentMarketProfile:
    name: str
    timezone_name: str
    helper_reset_time: Optional[Tuple[int, int, int]]
    low_liquidity_windows: Tuple[LocalTimeWindow, ...]
    normal_liquidity_windows: Tuple[LocalTimeWindow, ...]
    pre_disabled_weekday_reset_time: Optional[Tuple[int, int, int]] = None
    helper_reset_timezone_name: Optional[str] = None
    low_liquidity_weekdays: Tuple[int, ...] = ()
    low_liquidity_trade_disabled_weekdays: Tuple[int, ...] = ()
    reset_skip_weekdays: Tuple[int, ...] = ()
    atr_ref_timeframe: str = "15m"
    atr_ref_period: int = 14
    atr_ref_lookback_days: int = 5
    atr_ref_lookback_bars: int = 700
    normal_liquidity_r_min_atr_multiple: float = 1.5
    normal_liquidity_r_max_atr_multiple: float = 2.5
    low_liquidity_r_min_atr_multiple: float = 2.5
    low_liquidity_r_max_atr_multiple: float = 3.0
    normal_liquidity_tp1_r_multiple: float = 1.0
    normal_liquidity_tp2_r_multiple: float = 2.0
    normal_liquidity_tp1_close_fraction: float = 0.30
    normal_liquidity_tp2_close_fraction: float = 0.40
    normal_liquidity_post_tp1_stop_r_multiple: float = -0.40
    normal_liquidity_post_tp2_locked_r_multiple: float = 1.0
    normal_liquidity_trailing_soft_atr_multiple: float = 2.5
    normal_liquidity_trailing_hard_atr_multiple: float = 3.5
    low_liquidity_tp1_r_multiple: float = 0.75
    low_liquidity_tp2_r_multiple: float = 1.25
    low_liquidity_tp1_close_fraction: float = 0.40
    low_liquidity_tp2_close_fraction: float = 0.40
    low_liquidity_post_tp1_stop_r_multiple: float = -0.40
    low_liquidity_post_tp2_locked_r_multiple: float = 0.75
    low_liquidity_trailing_soft_atr_multiple: float = 1.5
    low_liquidity_trailing_hard_atr_multiple: float = 2.5


WEEKDAY_NAME_TO_INDEX = {
    "monday": 0,
    "mon": 0,
    "tuesday": 1,
    "tue": 1,
    "wednesday": 2,
    "wed": 2,
    "thursday": 3,
    "thu": 3,
    "friday": 4,
    "fri": 4,
    "saturday": 5,
    "sat": 5,
    "sunday": 6,
    "sun": 6,
}
