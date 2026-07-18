import io
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from market_agent.constants import (
    DEFAULT_CHART_IMAGE_DETAIL,
    DEFAULT_CHART_IMAGE_LAYOUT_HEIGHT_PX,
    DEFAULT_CHART_IMAGE_LAYOUT_WIDTH_PX,
    DEFAULT_CHART_IMAGE_WINDOW_HOURS_BY_TIMEFRAME,
)
from market_agent.utils import format_query_amount, safe_float


def _build_chart_image_timeframe_specs(env_value: str, default_timeframes: Tuple[str, ...]) -> Tuple[Dict[str, Any], ...]:
    requested = [str(item or "").strip().lower() for item in str(env_value or "").split(",") if str(item or "").strip()]
    if not requested:
        requested = [str(item or "").strip().lower() for item in list(default_timeframes or ()) if str(item or "").strip()]
    resolved: List[Dict[str, Any]] = []
    seen: set[str] = set()
    for timeframe in requested:
        if timeframe in seen:
            continue
        window_hours = DEFAULT_CHART_IMAGE_WINDOW_HOURS_BY_TIMEFRAME.get(timeframe)
        if window_hours is None:
            continue
        resolved.append({"timeframe": timeframe, "window_hours": float(window_hours)})
        seen.add(timeframe)
    if resolved:
        return tuple(resolved)
    fallback = []
    for timeframe in list(default_timeframes or ()):
        key = str(timeframe or "").strip().lower()
        window_hours = DEFAULT_CHART_IMAGE_WINDOW_HOURS_BY_TIMEFRAME.get(key)
        if not key or window_hours is None:
            continue
        fallback.append({"timeframe": key, "window_hours": float(window_hours)})
    return tuple(fallback)


def _candle_timestamp_ms(candle: Dict[str, Any]) -> int:
    for key in ("t", "T", "time", "timestamp", "ts"):
        raw = candle.get(key)
        if raw in (None, ""):
            continue
        try:
            value = int(float(raw))
            if value > 0:
                return value
        except Exception:
            continue
    return 0


def _sorted_candles(candles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return sorted(
        [item for item in (candles or []) if isinstance(item, dict)],
        key=lambda item: (_candle_timestamp_ms(item), safe_float(item.get("c"), 0.0) or 0.0),
    )


def _window_change_pct(candles: List[Dict[str, Any]]) -> float:
    candles = _sorted_candles(candles)
    if not candles:
        return 0.0
    first_open = safe_float(candles[0].get("o"), 0.0) or 0.0
    last_close = safe_float(candles[-1].get("c"), 0.0) or 0.0
    if first_open <= 0 or last_close <= 0:
        return 0.0
    return ((last_close / first_open) - 1.0) * 100.0


def _window_high_low(candles: List[Dict[str, Any]]) -> Tuple[float, float]:
    candles = _sorted_candles(candles)
    highs = [safe_float(item.get("h"), 0.0) or 0.0 for item in candles]
    lows = [safe_float(item.get("l"), 0.0) or 0.0 for item in candles]
    valid_highs = [value for value in highs if value > 0]
    valid_lows = [value for value in lows if value > 0]
    return (
        max(valid_highs, default=0.0),
        min(valid_lows, default=0.0),
    )


def _range_position_pct(price: Optional[float], low: float, high: float) -> float:
    px = safe_float(price, 0.0) or 0.0
    lo = max(0.0, float(low or 0.0))
    hi = max(0.0, float(high or 0.0))
    if px <= 0 or lo <= 0 or hi <= 0 or hi <= lo:
        return 0.0
    return max(0.0, min(100.0, ((px - lo) / (hi - lo)) * 100.0))


def _average_candle_range_pct(candles: List[Dict[str, Any]]) -> float:
    ranges: List[float] = []
    for item in _sorted_candles(candles):
        high = safe_float(item.get("h"), 0.0) or 0.0
        low = safe_float(item.get("l"), 0.0) or 0.0
        close = safe_float(item.get("c"), 0.0) or 0.0
        if close > 0 and high > 0 and low > 0:
            ranges.append(((high - low) / close) * 100.0)
    return (sum(ranges) / len(ranges)) if ranges else 0.0


def _build_chart_debug_record(
    *,
    timeframe: str,
    window_hours: float,
    width_px: int,
    height_px: int,
    detail: str,
    candle_count: int,
    image_bytes: int,
    data_url_chars: int,
) -> Dict[str, Any]:
    return {
        "timeframe": timeframe,
        "window_hours": float(window_hours),
        "width_px": int(width_px),
        "height_px": int(height_px),
        "detail": str(detail or DEFAULT_CHART_IMAGE_DETAIL),
        "candle_count": int(candle_count),
        "image_bytes": int(image_bytes),
        "data_url_chars": int(data_url_chars),
    }


def _build_chart_summary_record(
    *,
    candles: List[Dict[str, Any]],
    timeframe: str,
    window_hours: float,
    current_price: Optional[float],
) -> Dict[str, Any]:
    sorted_candles = _sorted_candles(candles)
    window_high, window_low = _window_high_low(sorted_candles)
    return {
        "timeframe": str(timeframe or ""),
        "window_hours": float(window_hours),
        "current_price": safe_float(current_price, None),
        "window_change_pct": _window_change_pct(sorted_candles),
        "window_high": window_high,
        "window_low": window_low,
        "range_position_pct": _range_position_pct(current_price, window_low, window_high),
        "avg_candle_range_pct": _average_candle_range_pct(sorted_candles),
    }


def _build_chart_tick_positions(candle_count: int, tick_count: int) -> List[int]:
    candle_count = max(0, int(candle_count or 0))
    tick_count = max(0, int(tick_count or 0))
    if candle_count <= 0 or tick_count <= 0:
        return []
    tick_step = max(1, candle_count // max(1, tick_count - 1))
    positions = [idx for idx in range(0, candle_count, tick_step) if idx < max(0, candle_count - 1)]
    if not positions:
        return [0] if candle_count == 1 else [0, max(0, candle_count // 2)]
    if positions[0] != 0:
        positions.insert(0, 0)
    if candle_count > 2 and len(positions) == 1:
        midpoint = max(1, min(candle_count - 2, candle_count // 2))
        if midpoint not in positions:
            positions.append(midpoint)
    return sorted(set(positions))


def _format_chart_price_label(value: Any) -> str:
    price = safe_float(value, None)
    if price is None:
        return ""
    magnitude = abs(price)
    if magnitude >= 1000:
        decimals = 0
    elif magnitude >= 100:
        decimals = 2
    elif magnitude >= 1:
        decimals = 3
    elif magnitude >= 0.1:
        decimals = 4
    else:
        decimals = 5
    formatted = f"{price:.{decimals}f}"
    if "." in formatted:
        formatted = formatted.rstrip("0").rstrip(".")
    return formatted


def _resize_png_bytes(png_bytes: bytes, width_px: int, height_px: int) -> bytes:
    if not png_bytes:
        return png_bytes
    try:
        from PIL import Image
    except Exception:
        return png_bytes
    buffer = io.BytesIO(png_bytes)
    try:
        image = Image.open(buffer)
        if image.size == (width_px, height_px):
            return png_bytes
        resized = image.resize((max(1, int(width_px)), max(1, int(height_px))), Image.Resampling.LANCZOS)
        out = io.BytesIO()
        resized.save(out, format="PNG")
        return out.getvalue()
    except Exception:
        return png_bytes


def render_candles_chart_png(
    *,
    candles: List[Dict[str, Any]],
    symbol_label: str,
    timeframe: str,
    window_hours: float,
    width_px: int,
    height_px: int,
    current_price: Optional[float],
) -> Optional[bytes]:
    candles = _sorted_candles(candles)
    if len(candles) < 2:
        return None
    try:
        import matplotlib

        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        from matplotlib.ticker import FuncFormatter, MaxNLocator
    except Exception:
        return None

    timestamps = [_candle_timestamp_ms(item) for item in candles]
    opens = [safe_float(item.get("o"), 0.0) or 0.0 for item in candles]
    highs = [safe_float(item.get("h"), 0.0) or 0.0 for item in candles]
    lows = [safe_float(item.get("l"), 0.0) or 0.0 for item in candles]
    closes = [safe_float(item.get("c"), 0.0) or 0.0 for item in candles]
    x_values = list(range(len(candles)))
    price_floor = min(value for value in lows if value > 0) if any(value > 0 for value in lows) else 0.0
    price_ceiling = max(value for value in highs if value > 0) if any(value > 0 for value in highs) else 0.0
    if price_floor <= 0 or price_ceiling <= 0 or price_ceiling <= price_floor:
        return None

    dpi = 120
    render_width_px = max(DEFAULT_CHART_IMAGE_LAYOUT_WIDTH_PX, int(width_px or 0))
    render_height_px = max(DEFAULT_CHART_IMAGE_LAYOUT_HEIGHT_PX, int(height_px or 0))
    fig = None
    try:
        fig = plt.figure(
            figsize=(max(320, render_width_px) / dpi, max(240, render_height_px) / dpi),
            dpi=dpi,
            facecolor="white",
        )
        ax_price = fig.add_subplot(1, 1, 1)
        ax_price.set_facecolor("#ffffff")
        ax_price.grid(True, color="#d7dde5", alpha=0.45, linewidth=0.6)
        body_width = min(0.82, max(0.28, 18.0 / max(1, len(candles))))
        min_body_height = max((price_ceiling - price_floor) * 0.0015, price_ceiling * 0.0002)
        up_color = "#1f9d61"
        down_color = "#d43f3a"
        for x, open_px, high_px, low_px, close_px in zip(x_values, opens, highs, lows, closes):
            color = up_color if close_px >= open_px else down_color
            ax_price.vlines(x, low_px, high_px, color=color, linewidth=1.0, alpha=0.95)
            lower = min(open_px, close_px)
            height = max(abs(close_px - open_px), min_body_height)
            if close_px < open_px:
                lower = close_px
            rect = Rectangle(
                (x - body_width / 2.0, lower),
                body_width,
                height,
                facecolor=color,
                edgecolor=color,
                linewidth=0.8,
                alpha=0.92,
            )
            ax_price.add_patch(rect)
        close_line_color = "#0f172a"
        ax_price.plot(x_values, closes, color=close_line_color, linewidth=1.0, alpha=0.35)
        current_px = safe_float(current_price, None)
        current_label = ""
        if current_px is not None and current_px > 0:
            ax_price.axhline(current_px, color="#2563eb", linestyle="--", linewidth=0.9, alpha=0.7)
            current_label = _format_chart_price_label(current_px)
        ax_price.set_ylabel("Price", fontsize=8)
        ax_price.yaxis.set_major_locator(MaxNLocator(nbins=6))
        ax_price.yaxis.set_major_formatter(FuncFormatter(lambda val, _pos: _format_chart_price_label(val)))
        ax_price.tick_params(axis="y", labelsize=6, pad=1)
        tick_count = min(6, max(3, len(candles) // 12))
        tick_positions = _build_chart_tick_positions(len(candles), tick_count)
        tick_labels = []
        for idx in tick_positions:
            dt = datetime.fromtimestamp(max(0, timestamps[idx]) / 1000.0, tz=timezone.utc)
            tick_labels.append(dt.strftime("%m-%d\n%H:%M"))
        ax_price.set_xticks(tick_positions)
        ax_price.set_xticklabels(tick_labels, fontsize=6, rotation=0, linespacing=0.95)
        ax_price.tick_params(axis="x", pad=2)
        ax_price.set_xlim(-0.75, len(candles) - 0.25)
        pad = max((price_ceiling - price_floor) * 0.05, price_ceiling * 0.0025)
        ax_price.set_ylim(price_floor - pad, price_ceiling + pad)
        title = f"{symbol_label} | {timeframe} | recent {format_query_amount(window_hours)}h"
        if current_label:
            title += f" | current price {current_label}"
        ax_price.set_title(
            title,
            fontsize=10,
            loc="left",
        )
        fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.20)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", facecolor="white", dpi=dpi)
        png_bytes = buffer.getvalue()
        if render_width_px != int(width_px or 0) or render_height_px != int(height_px or 0):
            return _resize_png_bytes(png_bytes, int(width_px or render_width_px), int(height_px or render_height_px))
        return png_bytes
    finally:
        if fig is not None:
            plt.close(fig)
