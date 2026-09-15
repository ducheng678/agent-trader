from __future__ import annotations

import argparse
import json
import os
import re
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence

import requests

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - optional dependency during isolated execution.
    load_dotenv = None


if callable(load_dotenv):
    load_dotenv()


SUPPORTED_TIMEFRAMES = ("1m", "5m", "15m", "30m", "1h")
TIMEFRAME_TO_MINUTES = {
    "1m": 1,
    "5m": 5,
    "15m": 15,
    "30m": 30,
    "1h": 60,
}
DEFAULT_BARS_PER_TIMEFRAME = 48
MAX_BASE_1M_BARS = 5000
DEFAULT_FLAT_THRESHOLD_PCT = 0.05
HYPERLIQUID_TIMEOUT_SECONDS = max(2.0, float(os.getenv("TREND_FETCHER_HYPERLIQUID_TIMEOUT_SECONDS", "15") or 15.0))
TWELVEDATA_TIMEOUT_SECONDS = max(2.0, float(os.getenv("TREND_FETCHER_TWELVEDATA_TIMEOUT_SECONDS", "20") or 20.0))
DATABENTO_TIMEOUT_SECONDS = max(2.0, float(os.getenv("TREND_FETCHER_DATABENTO_TIMEOUT_SECONDS", "30") or 30.0))
FINNHUB_TIMEOUT_SECONDS = max(2.0, float(os.getenv("TREND_FETCHER_FINNHUB_TIMEOUT_SECONDS", "15") or 15.0))
FINNHUB_MAX_QUOTE_AGE_SECONDS = max(
    0.0,
    float(os.getenv("TREND_FETCHER_FINNHUB_MAX_QUOTE_AGE_SECONDS", "0") or 0.0),
)
HYPERLIQUID_QUOTE_MAX_REFERENCE_DEVIATION_PCT = max(
    0.0,
    float(os.getenv("TREND_FETCHER_HYPERLIQUID_QUOTE_MAX_REFERENCE_DEVIATION_PCT", "5") or 5.0),
)
TREND_FETCHER_FLAT_THRESHOLD_PCT = max(
    0.0,
    float(os.getenv("TREND_FETCHER_FLAT_THRESHOLD_PCT", str(DEFAULT_FLAT_THRESHOLD_PCT)) or DEFAULT_FLAT_THRESHOLD_PCT),
)

# Lightweight dollar-strength proxy for frequent refreshes. This intentionally
# uses only the two most informative FX legs instead of the full 6-pair DXY basket.
SYNTHETIC_DXY_SCALE = 50.14348112
SYNTHETIC_DXY_COMPONENTS = {
    "EUR/USD": -0.576,
    "USD/JPY": 0.136,
}

FINNHUB_QUOTE_SYMBOL_ALIASES = {
    "CPER": "CPER",
    "JETS": "JETS",
    "HYG": "HYG",
    "LQD": "LQD",
    "IWM": "IWM",
    "GLD": "GLD",
    "SLV": "SLV",
    "XLE": "XLE",
    "VIXY": "VIXY",
    "VXX": "VXX",
    "UVXY": "UVXY",
    "SVIX": "SVIX",
}

FINNHUB_QUOTE_SYMBOL_META = {
    "IWM": {
        "proxy_for": "Russell 2000 / RTY small-cap risk appetite proxy",
        "proxy_note": "ETF quote proxy, not CME RTY futures.",
    },
    "VIXY": {
        "proxy_for": "VIX short-term futures / equity volatility sentiment",
        "proxy_note": "Not spot VIX. Tracks short-term VIX futures exposure.",
        "risk_direction": "up implies volatility demand rising / risk-off; down implies volatility demand falling / risk-on",
    },
    "VXX": {
        "proxy_for": "VIX short-term futures / equity volatility sentiment",
        "proxy_note": "Not spot VIX. ETN exposure to short-term VIX futures.",
        "risk_direction": "up implies volatility demand rising / risk-off; down implies volatility demand falling / risk-on",
    },
    "UVXY": {
        "proxy_for": "leveraged VIX short-term futures / volatility spike signal",
        "proxy_note": "Not spot VIX. Leveraged product, so moves are amplified and decay-sensitive.",
        "risk_direction": "up implies volatility demand rising / risk-off; down implies volatility demand falling / risk-on",
    },
    "SVIX": {
        "proxy_for": "inverse short VIX futures / volatility-down risk-on proxy",
        "proxy_note": "Not spot VIX. Inverse volatility product; direction is opposite VIX futures demand.",
        "risk_direction": "up usually implies volatility falling / risk-on; down usually implies volatility rising / risk-off",
    },
}

TWELVEDATA_FOREX_SYMBOL_ALIASES = {
    "USDCAD": "USD/CAD",
    "USDCHF": "USD/CHF",
    "USDNOK": "USD/NOK",
    "GBPUSD": "GBP/USD",
}

SPECIAL_SOURCE_ALIASES = {
    "dxy": "dxy",
    "usdx": "dxy",
    "usdindex": "dxy",
    "vix": "vix_spot_unsupported",
    "vixusdc": "vix_spot_unsupported",
    "us2y": "us2y",
    "us2yr": "us2y",
    "ust2y": "us2y",
    "zt": "us2y",
    "ztv0": "us2y",
    "ztn0": "us2y",
    "us10y": "us10y",
    "us10yr": "us10y",
    "ust10y": "us10y",
    "zn": "us10y",
    "znv0": "us10y",
    "znn0": "us10y",
    **{key.lower(): "twelvedata_forex" for key in TWELVEDATA_FOREX_SYMBOL_ALIASES},
    **{key.lower(): "finnhub_quote" for key in FINNHUB_QUOTE_SYMBOL_ALIASES},
}

DEFAULT_MARKET_HINTS = {
    "oil": ("BRENTOIL", "BRENT", "WTI", "CRUDE"),
    "brent": ("BRENTOIL", "BRENT"),
    "btc": ("BTC",),
    "eth": ("ETH",),
    "spx": ("SPX", "SP500", "S&P", "ES", "US500"),
    "sp500": ("SPX", "SP500", "S&P", "ES", "US500"),
    "nasdaq": ("NASDAQ", "NDX", "NAS100", "US100", "NQ"),
    "ndx": ("NDX", "NASDAQ", "NAS100", "US100", "NQ"),
    "nq": ("NQ", "NDX", "NASDAQ", "NAS100", "US100"),
    "dow": ("DOW", "DJI", "US30", "YM"),
    "dji": ("DJI", "DOW", "US30", "YM"),
    "rut": ("RUT", "RUSSELL", "US2000"),
    "russell": ("RUSSELL", "RUT", "US2000"),
}

BUILTIN_HYPERLIQUID_MARKET_ALIASES = {
    "NDX": "XYZ100",
    "NQ": "XYZ100",
    "NQF": "XYZ100",
    "NAS100": "XYZ100",
    "NASDAQ100": "XYZ100",
    "US100": "XYZ100",
    "GOLD": "GOLD",
    "XAU": "GOLD",
    "XAUUSD": "GOLD",
    "GC": "GOLD",
    "GCF": "GOLD",
    "SILVER": "SILVER",
    "XAG": "SILVER",
    "XAGUSD": "SILVER",
    "SI": "SILVER",
    "SIF": "SILVER",
    "EUR": "EUR",
    "EURUSD": "EUR",
    "JPY": "JPY",
    "USDJPY": "JPY",
    "CL": "CL",
    "CLF": "CL",
    "WTI": "CL",
    "BRENTOIL": "BRENTOIL",
    "BZ": "BRENTOIL",
    "BZF": "BRENTOIL",
    "BRENT": "BRENTOIL",
    "NATGAS": "NATGAS",
    "NG": "NATGAS",
    "NGF": "NATGAS",
    "NATURALGAS": "NATGAS",
    "COPPER": "COPPER",
    "HG": "COPPER",
    "HGF": "COPPER",
    "SPX": "SP500",
    "SP500": "SP500",
    "US500": "SP500",
    "ES": "SP500",
    "ESF": "SP500",
    "RTY": "SMALL2000",
    "RUT": "SMALL2000",
    "RUSSELL": "SMALL2000",
    "RUSSELL2000": "SMALL2000",
    "US2000": "SMALL2000",
    "SMALL2000": "SMALL2000",
}

BLOCKED_GENERIC_FALLBACK_TOKENS = {
    "HO",
    "HOF",
    "RB",
    "RBF",
}
BLOCKED_HYPERLIQUID_VIX_TOKENS = {
    "VIX",
    "VIXUSDC",
}

PREFERRED_HYPERLIQUID_DEX_ORDER = ("xyz", "", "km", "cash", "flx", "hyna", "vntl")
PREFERRED_HYPERLIQUID_DEX_RANK = {item: index for index, item in enumerate(PREFERRED_HYPERLIQUID_DEX_ORDER)}
HYPERLIQUID_QUOTE_PRICE_FIELDS = ("markPx", "midPx", "oraclePx", "prevDayPx")


@dataclass
class CandleBar:
    ts_ms: int
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    source_symbol: str = ""
    quote_only: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ts_ms": self.ts_ms,
            "ts": ms_to_iso(self.ts_ms),
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
            "source_symbol": self.source_symbol,
            "quote_only": self.quote_only,
        }


def _safe_float(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _normalize_lookup_token(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", "", str(value or "").strip().upper())


def _dedupe(values: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for item in values:
        token = str(item or "").strip()
        if not token or token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def ms_to_iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(int(ts_ms) / 1000.0, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _parse_iso_or_utc_datetime(value: str) -> int:
    text = str(value or "").strip()
    if not text:
        raise ValueError("Missing datetime value")
    if text.endswith("Z"):
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    elif "T" in text and any(marker in text[-6:] for marker in ["+", "-"]):
        dt = datetime.fromisoformat(text)
    else:
        fmt = "%Y-%m-%d %H:%M:%S" if " " in text else "%Y-%m-%d"
        dt = datetime.strptime(text, fmt).replace(tzinfo=timezone.utc)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def normalize_timeframes(timeframes: Optional[Sequence[str]] = None) -> List[str]:
    raw = list(timeframes or SUPPORTED_TIMEFRAMES)
    normalized: List[str] = []
    seen = set()
    for item in raw:
        timeframe = str(item or "").strip().lower()
        if timeframe not in TIMEFRAME_TO_MINUTES:
            raise ValueError(f"Unsupported timeframe: {item!r}. Supported: {', '.join(SUPPORTED_TIMEFRAMES)}")
        if timeframe not in seen:
            seen.add(timeframe)
            normalized.append(timeframe)
    return normalized


def detect_special_source(symbol_name: str) -> str:
    token = _normalize_lookup_token(symbol_name)
    alias_key = token.lower()
    if alias_key in SPECIAL_SOURCE_ALIASES:
        return SPECIAL_SOURCE_ALIASES[alias_key]
    if token.startswith("ZT") and token not in {"ZTR"}:
        return "us2y"
    if token.startswith("ZN") and token not in {"ZNT", "ZNTR"}:
        return "us10y"
    return "hyperliquid"


def resolve_finnhub_quote_symbol(symbol_name: str) -> str:
    token = _normalize_lookup_token(symbol_name)
    resolved = FINNHUB_QUOTE_SYMBOL_ALIASES.get(token)
    if not resolved:
        raise KeyError(f"Could not resolve Finnhub quote symbol for {symbol_name!r}")
    return resolved


def resolve_twelvedata_forex_symbol(symbol_name: str) -> str:
    token = _normalize_lookup_token(symbol_name)
    resolved = TWELVEDATA_FOREX_SYMBOL_ALIASES.get(token)
    if not resolved:
        raise KeyError(f"Could not resolve TwelveData forex symbol for {symbol_name!r}")
    return resolved


def invert_direction(direction: str) -> str:
    if direction == "up":
        return "down"
    if direction == "down":
        return "up"
    return direction


def _direction_from_change_pct(change_pct: Optional[float], flat_threshold_pct: float) -> str:
    if change_pct is None:
        return "unknown"
    if change_pct > flat_threshold_pct:
        return "up"
    if change_pct < -flat_threshold_pct:
        return "down"
    return "flat"


def _floor_ts_ms(ts_ms: int, interval_minutes: int) -> int:
    interval_ms = int(interval_minutes * 60_000)
    return int(ts_ms // interval_ms) * interval_ms


def resample_bars(bars: Sequence[CandleBar], timeframe: str) -> List[CandleBar]:
    interval_minutes = TIMEFRAME_TO_MINUTES[str(timeframe or "").strip().lower()]
    if interval_minutes == 1:
        return [CandleBar(**bar.__dict__) for bar in sorted(bars, key=lambda item: item.ts_ms)]

    buckets: Dict[int, List[CandleBar]] = {}
    for bar in sorted(bars, key=lambda item: item.ts_ms):
        bucket_start = _floor_ts_ms(bar.ts_ms, interval_minutes)
        buckets.setdefault(bucket_start, []).append(bar)

    output: List[CandleBar] = []
    for bucket_start in sorted(buckets):
        chunk = buckets[bucket_start]
        output.append(
            CandleBar(
                ts_ms=bucket_start,
                open=chunk[0].open,
                high=max(item.high for item in chunk),
                low=min(item.low for item in chunk),
                close=chunk[-1].close,
                volume=sum(float(item.volume or 0.0) for item in chunk),
                source_symbol=chunk[-1].source_symbol,
                quote_only=all(item.quote_only for item in chunk),
            )
        )
    return output


def summarize_bars(
    bars: Sequence[CandleBar],
    *,
    bars_requested: int,
    flat_threshold_pct: float,
    include_bars: bool,
    inverse_price_relation: bool = False,
) -> Dict[str, Any]:
    ordered = sorted(bars, key=lambda item: item.ts_ms)
    sample = ordered[-int(bars_requested or 0):] if bars_requested else ordered
    summary: Dict[str, Any] = {
        "bars_requested": int(bars_requested or 0),
        "bar_count": len(sample),
        "bar_completeness": 0.0,
        "start_ts": "",
        "end_ts": "",
        "open": None,
        "close": None,
        "high": None,
        "low": None,
        "change_pct": None,
        "direction": "unknown",
        "avg_bar_range_pct": None,
        "realized_range_pct": None,
    }
    if bars_requested:
        summary["bar_completeness"] = round(len(sample) / float(bars_requested), 6)
    if not sample:
        if inverse_price_relation:
            summary["inverse_change_pct"] = None
            summary["inverse_direction"] = "unknown"
        if include_bars:
            summary["bars"] = []
        return summary

    first_open = sample[0].open
    last_close = sample[-1].close
    quote_only = len(sample) == 1 and bool(getattr(sample[0], "quote_only", False))
    window_high = max(item.high for item in sample)
    window_low = min(item.low for item in sample)
    change_pct = None
    if first_open > 0 and not quote_only:
        change_pct = ((last_close / first_open) - 1.0) * 100.0
    avg_range_samples = [((item.high - item.low) / item.open) * 100.0 for item in sample if item.open > 0]
    avg_bar_range_pct = (sum(avg_range_samples) / len(avg_range_samples)) if avg_range_samples else None
    realized_range_pct = (((window_high - window_low) / first_open) * 100.0) if first_open > 0 else None
    summary.update(
        {
            "start_ts": ms_to_iso(sample[0].ts_ms),
            "end_ts": ms_to_iso(sample[-1].ts_ms),
            "open": first_open,
            "close": last_close,
            "high": window_high,
            "low": window_low,
            "change_pct": change_pct,
            "direction": "unknown" if quote_only else _direction_from_change_pct(change_pct, flat_threshold_pct),
            "avg_bar_range_pct": avg_bar_range_pct,
            "realized_range_pct": realized_range_pct,
        }
    )
    if quote_only:
        summary["quote_only"] = True
    if inverse_price_relation:
        summary["inverse_change_pct"] = (-change_pct) if change_pct is not None else None
        summary["inverse_direction"] = invert_direction(summary["direction"])
    if include_bars:
        summary["bars"] = [item.to_dict() for item in sample]
    return summary


def _dxy_price_from_component_values(component_values: Dict[str, float]) -> float:
    level = SYNTHETIC_DXY_SCALE
    for pair, weight in SYNTHETIC_DXY_COMPONENTS.items():
        px = float(component_values[pair])
        if px <= 0:
            raise ValueError(f"Invalid component price for {pair}: {px}")
        level *= px ** weight
    return level


def build_synthetic_dxy_bars(series_by_symbol: Dict[str, Sequence[CandleBar]]) -> List[CandleBar]:
    expected = set(SYNTHETIC_DXY_COMPONENTS)
    if set(series_by_symbol) != expected:
        missing = sorted(expected - set(series_by_symbol))
        extra = sorted(set(series_by_symbol) - expected)
        raise ValueError(f"Synthetic DXY requires exact component set; missing={missing} extra={extra}")

    bar_maps = {
        pair: {bar.ts_ms: bar for bar in bars if isinstance(bar, CandleBar)}
        for pair, bars in series_by_symbol.items()
    }
    common_timestamps = set.intersection(*(set(item.keys()) for item in bar_maps.values()))
    output: List[CandleBar] = []
    for ts_ms in sorted(common_timestamps):
        open_values = {pair: bar_maps[pair][ts_ms].open for pair in SYNTHETIC_DXY_COMPONENTS}
        close_values = {pair: bar_maps[pair][ts_ms].close for pair in SYNTHETIC_DXY_COMPONENTS}
        high_inputs = {
            pair: (bar_maps[pair][ts_ms].high if weight >= 0 else bar_maps[pair][ts_ms].low)
            for pair, weight in SYNTHETIC_DXY_COMPONENTS.items()
        }
        low_inputs = {
            pair: (bar_maps[pair][ts_ms].low if weight >= 0 else bar_maps[pair][ts_ms].high)
            for pair, weight in SYNTHETIC_DXY_COMPONENTS.items()
        }
        open_px = _dxy_price_from_component_values(open_values)
        close_px = _dxy_price_from_component_values(close_values)
        high_px = _dxy_price_from_component_values(high_inputs)
        low_px = _dxy_price_from_component_values(low_inputs)
        high_px = max(high_px, open_px, close_px)
        low_px = min(low_px, open_px, close_px)
        output.append(
            CandleBar(
                ts_ms=ts_ms,
                open=open_px,
                high=high_px,
                low=low_px,
                close=close_px,
                volume=0.0,
                source_symbol="DXY.synthetic",
            )
        )
    return output


def _market_search_tokens(spec: Dict[str, Any]) -> List[str]:
    return [
        token
        for token in {
            str(spec.get("execution_symbol", "") or "").strip().upper(),
            str(spec.get("market_name", "") or "").strip().upper(),
            str(spec.get("display_name", "") or "").strip().upper(),
            _normalize_lookup_token(spec.get("execution_symbol", "")),
            _normalize_lookup_token(spec.get("market_name", "")),
            _normalize_lookup_token(spec.get("display_name", "")),
        }
        if token
    ]


def pick_best_market_match(
    query: str,
    market_specs: Sequence[Dict[str, Any]],
    *,
    hint_tokens: Sequence[str] = (),
) -> Optional[Dict[str, Any]]:
    raw_query = str(query or "").strip().upper()
    normalized_query = _normalize_lookup_token(query)
    if not raw_query and not normalized_query:
        return None

    def score(spec: Dict[str, Any], probe: str) -> Optional[tuple]:
        tokens = _market_search_tokens(spec)
        normalized_probe = _normalize_lookup_token(probe)
        best: Optional[tuple] = None
        for token in tokens:
            normalized_token = _normalize_lookup_token(token)
            rank: Optional[tuple] = None
            if token == raw_query or normalized_token == normalized_query:
                rank = (0, len(normalized_token))
            elif normalized_probe and normalized_token.startswith(normalized_probe):
                rank = (1, len(normalized_token))
            elif normalized_probe and normalized_probe in normalized_token:
                rank = (2, len(normalized_token))
            if rank is not None and (best is None or rank < best):
                best = rank
        return best

    probes = [raw_query, normalized_query] + list(hint_tokens or [])
    ranked: List[tuple] = []
    for spec in market_specs:
        best_probe_score: Optional[tuple] = None
        for probe in probes:
            probe_score = score(spec, probe)
            if probe_score is not None and (best_probe_score is None or probe_score < best_probe_score):
                best_probe_score = probe_score
        if best_probe_score is not None:
            ranked.append((best_probe_score, spec))
    if not ranked:
        return None
    ranked.sort(key=lambda item: item[0])
    return dict(ranked[0][1])


def build_trend_snapshot(
    *,
    requested_symbol: str,
    resolved_symbol: str,
    source: str,
    source_symbol: str,
    base_1m_bars: Sequence[CandleBar],
    timeframes: Sequence[str],
    bars_per_timeframe: int,
    include_bars: bool,
    flat_threshold_pct: float,
    notes: Optional[Sequence[str]] = None,
    market_meta: Optional[Dict[str, Any]] = None,
    inverse_price_relation: bool = False,
) -> Dict[str, Any]:
    normalized_timeframes = normalize_timeframes(timeframes)
    ordered_base = sorted(base_1m_bars, key=lambda item: item.ts_ms)
    payload: Dict[str, Any] = {
        "generated_at": ms_to_iso(int(time.time() * 1000)),
        "requested_symbol": str(requested_symbol or "").strip(),
        "resolved_symbol": str(resolved_symbol or "").strip(),
        "source": source,
        "source_symbol": source_symbol,
        "latest_price": ordered_base[-1].close if ordered_base else None,
        "bars_per_timeframe": int(bars_per_timeframe or 0),
        "available_base_1m_bars": len(ordered_base),
        "notes": [str(item).strip() for item in list(notes or []) if str(item).strip()],
        "market_meta": dict(market_meta or {}),
        "timeframes": {},
    }
    for timeframe in normalized_timeframes:
        tf_bars = ordered_base if timeframe == "1m" else resample_bars(ordered_base, timeframe)
        payload["timeframes"][timeframe] = summarize_bars(
            tf_bars,
            bars_requested=bars_per_timeframe,
            flat_threshold_pct=flat_threshold_pct,
            include_bars=include_bars,
            inverse_price_relation=inverse_price_relation,
        )
    return payload


def _parse_env_alias_map(raw: str) -> Dict[str, str]:
    alias_map: Dict[str, str] = {}
    for item in re.split(r"[;,\n]+", str(raw or "")):
        chunk = str(item or "").strip()
        if not chunk or "=" not in chunk:
            continue
        key, value = chunk.split("=", 1)
        normalized_key = _normalize_lookup_token(key)
        normalized_value = str(value or "").strip()
        if normalized_key and normalized_value:
            alias_map[normalized_key] = normalized_value
    return alias_map


def _normalize_hyperliquid_market_identity(raw_name: str, dex: str = "") -> tuple[str, str]:
    text = str(raw_name or "").strip()
    if not text:
        return "", ""
    dex_name = str(dex or "").strip().lower()
    if ":" in text:
        prefix, suffix = text.split(":", 1)
        market_name = str(suffix or "").strip().upper()
        execution_dex = str(prefix or dex_name).strip().lower() or dex_name
        if execution_dex:
            return f"{execution_dex}:{market_name}", market_name
        return market_name, market_name
    market_name = text.upper()
    if dex_name:
        return f"{dex_name}:{market_name}", market_name
    return market_name, market_name


def _is_blocked_hyperliquid_vix_symbol(raw_symbol: Any) -> bool:
    text = str(raw_symbol or "").strip()
    if not text:
        return False
    probes = [text]
    if ":" in text:
        probes.append(text.split(":", 1)[1])
    return any(_normalize_lookup_token(probe) in BLOCKED_HYPERLIQUID_VIX_TOKENS for probe in probes)


def _raise_if_blocked_hyperliquid_vix(raw_symbol: Any) -> None:
    if _is_blocked_hyperliquid_vix_symbol(raw_symbol):
        raise KeyError(
            "Hyperliquid VIX market is disabled because it is not a reliable realtime VIX spot source. "
            "Use Finnhub VIXY as the preferred volatility proxy or VXX as a backup."
        )


def _raise_if_blocked_hyperliquid_market(market: Dict[str, Any]) -> None:
    _raise_if_blocked_hyperliquid_vix(market.get("execution_symbol", ""))
    _raise_if_blocked_hyperliquid_vix(market.get("market_name", ""))


def _market_preference_key(spec: Dict[str, Any]) -> tuple[int, str, str]:
    dex = str(spec.get("dex", "") or "").strip().lower()
    execution_symbol = str(spec.get("execution_symbol", "") or "").strip().upper()
    display_name = str(spec.get("display_name", "") or "").strip().upper()
    return (
        PREFERRED_HYPERLIQUID_DEX_RANK.get(dex, len(PREFERRED_HYPERLIQUID_DEX_RANK)),
        display_name,
        execution_symbol,
    )


def _find_market_candidates_by_alias_target(catalog: Dict[str, Dict[str, Any]], alias_target: str) -> List[Dict[str, Any]]:
    normalized_target = _normalize_lookup_token(alias_target)
    candidates: List[Dict[str, Any]] = []
    for spec in catalog.values():
        execution_symbol = str(spec.get("execution_symbol", "") or "").strip()
        execution_suffix = execution_symbol.split(":", 1)[-1]
        spec_tokens = {
            _normalize_lookup_token(execution_symbol),
            _normalize_lookup_token(execution_suffix),
            _normalize_lookup_token(spec.get("market_name", "")),
            _normalize_lookup_token(spec.get("display_name", "")),
        }
        if normalized_target in spec_tokens:
            candidates.append(dict(spec))
    return sorted(candidates, key=_market_preference_key)


class HyperliquidInfoClient:
    def __init__(self) -> None:
        network = (os.getenv("HYPERLIQUID_NETWORK", "mainnet") or "").strip().lower()
        if network == "mainnet":
            self.base_url = "https://api.hyperliquid.xyz"
        elif network == "testnet":
            self.base_url = "https://api.hyperliquid-testnet.xyz"
        else:
            raise ValueError("HYPERLIQUID_NETWORK must be mainnet or testnet")
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self._perp_dex_names: Optional[List[str]] = None
        self._meta_cache_by_dex: Dict[str, Dict[str, Any]] = {}
        self._market_catalog: Optional[Dict[str, Dict[str, Any]]] = None
        self._alias_index: Optional[Dict[str, str]] = None
        self._env_aliases = _parse_env_alias_map(os.getenv("TREND_FETCHER_HYPERLIQUID_ALIASES", ""))

    def _post_info(self, payload: Dict[str, Any]) -> Any:
        response = self.session.post(self.base_url.rstrip("/") + "/info", json=payload, timeout=HYPERLIQUID_TIMEOUT_SECONDS)
        response.raise_for_status()
        return response.json()

    def _get_perp_dex_names(self) -> List[str]:
        if self._perp_dex_names is not None:
            return list(self._perp_dex_names)
        try:
            payload = self._post_info({"type": "perpDexs"})
            self._perp_dex_names = [
                str(item.get("name", "") or "").strip().lower()
                for item in list(payload or [])
                if isinstance(item, dict) and str(item.get("name", "") or "").strip()
            ]
        except Exception:
            self._perp_dex_names = []
        return list(self._perp_dex_names)

    def _get_perp_meta(self, dex: str = "") -> Dict[str, Any]:
        key = str(dex or "").strip().lower()
        if key not in self._meta_cache_by_dex:
            payload: Dict[str, Any] = {"type": "meta"}
            if key:
                payload["dex"] = key
            self._meta_cache_by_dex[key] = dict(self._post_info(payload) or {})
        return dict(self._meta_cache_by_dex[key])

    def _get_perp_meta_and_asset_ctxs(self, dex: str = "") -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        payload: Dict[str, Any] = {"type": "metaAndAssetCtxs"}
        dex_name = str(dex or "").strip().lower()
        if dex_name:
            payload["dex"] = dex_name
        response = self._post_info(payload)
        if not isinstance(response, list) or len(response) < 2:
            raise RuntimeError(f"Unexpected metaAndAssetCtxs response for dex={dex_name!r}")
        meta = dict(response[0] or {})
        asset_ctxs = [dict(item or {}) for item in list(response[1] or [])]
        return meta, asset_ctxs

    def _iter_market_specs(self) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for dex in [""] + self._get_perp_dex_names():
            try:
                meta = self._get_perp_meta(dex)
            except Exception:
                continue
            for entry in list(meta.get("universe") or []):
                if not isinstance(entry, dict):
                    continue
                execution_symbol, market_name = _normalize_hyperliquid_market_identity(entry.get("name", ""), dex)
                if not execution_symbol or not market_name:
                    continue
                items.append(
                    {
                        "execution_symbol": execution_symbol,
                        "dex": dex,
                        "market_name": market_name,
                        "display_name": f"{market_name}-USDC",
                        "sz_decimals": int(entry.get("szDecimals", 0) or 0),
                        "max_leverage": int(entry.get("maxLeverage", 0) or 0),
                        "only_isolated": bool(entry.get("onlyIsolated", False)),
                    }
                )
        return items

    def get_market_catalog(self) -> Dict[str, Dict[str, Any]]:
        if self._market_catalog is None:
            self._market_catalog = {
                str(item.get("execution_symbol", "") or "").strip().upper(): item
                for item in self._iter_market_specs()
                if str(item.get("execution_symbol", "") or "").strip()
            }
        return {key: dict(value) for key, value in self._market_catalog.items()}

    def _build_alias_index(self) -> Dict[str, str]:
        alias_index: Dict[str, str] = {}
        for execution_symbol, spec in self.get_market_catalog().items():
            for token in _market_search_tokens(spec):
                alias_index.setdefault(token, execution_symbol)
        return alias_index

    def resolve_market(self, raw_symbol: str) -> Dict[str, Any]:
        token = str(raw_symbol or "").strip()
        if not token:
            raise ValueError("Empty market symbol")
        _raise_if_blocked_hyperliquid_vix(token)
        catalog = self.get_market_catalog()
        uppercase_token = token.upper()
        normalized_token = _normalize_lookup_token(token)

        alias_target = self._env_aliases.get(normalized_token)
        if alias_target:
            env_alias_matches = _find_market_candidates_by_alias_target(catalog, alias_target)
            if env_alias_matches:
                _raise_if_blocked_hyperliquid_market(env_alias_matches[0])
                return env_alias_matches[0]

        builtin_alias_target = BUILTIN_HYPERLIQUID_MARKET_ALIASES.get(normalized_token)
        if builtin_alias_target:
            builtin_alias_matches = _find_market_candidates_by_alias_target(catalog, builtin_alias_target)
            if builtin_alias_matches:
                _raise_if_blocked_hyperliquid_market(builtin_alias_matches[0])
                return builtin_alias_matches[0]

        if uppercase_token in catalog:
            market = dict(catalog[uppercase_token])
            _raise_if_blocked_hyperliquid_market(market)
            return market

        if self._alias_index is None:
            self._alias_index = self._build_alias_index()

        exact_target = self._alias_index.get(uppercase_token) or self._alias_index.get(normalized_token)
        if exact_target and exact_target in catalog:
            market = dict(catalog[exact_target])
            _raise_if_blocked_hyperliquid_market(market)
            return market

        if normalized_token in BLOCKED_GENERIC_FALLBACK_TOKENS:
            raise KeyError(
                f"Could not resolve Hyperliquid market for {raw_symbol!r}. "
                "This shorthand is too ambiguous for generic matching; add an explicit alias instead."
            )

        hint_tokens = DEFAULT_MARKET_HINTS.get(normalized_token.lower(), ())
        fallback_match = pick_best_market_match(token, list(catalog.values()), hint_tokens=hint_tokens)
        if fallback_match is not None:
            _raise_if_blocked_hyperliquid_market(fallback_match)
            return fallback_match

        available = sorted(spec.get("display_name", spec.get("execution_symbol", "")) for spec in catalog.values())
        raise KeyError(f"Could not resolve Hyperliquid market for {raw_symbol!r}. Available sample: {available[:20]}")

    @staticmethod
    def _quote_price_from_asset_ctx(asset_ctx: Dict[str, Any]) -> tuple[Optional[float], str]:
        for field in HYPERLIQUID_QUOTE_PRICE_FIELDS:
            price = _safe_float(asset_ctx.get(field))
            if price is not None and price > 0:
                return price, field
        return None, ""

    @staticmethod
    def _quote_reference_prices(asset_ctx: Dict[str, Any], selected_field: str) -> Dict[str, float]:
        references: Dict[str, float] = {}
        for field in HYPERLIQUID_QUOTE_PRICE_FIELDS:
            if field == selected_field:
                continue
            price = _safe_float(asset_ctx.get(field))
            if price is not None and price > 0:
                references[field] = price
        return references

    @staticmethod
    def _max_quote_reference_deviation_pct(price: float, references: Dict[str, float]) -> Optional[float]:
        deviations = [
            abs((price / reference) - 1.0) * 100.0
            for reference in references.values()
            if reference > 0
        ]
        return max(deviations) if deviations else None

    def _fetch_quote_snapshot_bar(self, market: Dict[str, Any], ts_ms: int) -> tuple[List[CandleBar], Dict[str, Any]]:
        execution_symbol = str(market.get("execution_symbol", "") or "").strip()
        market_name = str(market.get("market_name", "") or execution_symbol).strip().upper()
        dex = str(market.get("dex", "") or "").strip().lower()
        meta, asset_ctxs = self._get_perp_meta_and_asset_ctxs(dex)
        for index, entry in enumerate(list(meta.get("universe") or [])):
            if not isinstance(entry, dict):
                continue
            entry_execution_symbol, entry_market_name = _normalize_hyperliquid_market_identity(entry.get("name", ""), dex)
            if entry_execution_symbol != execution_symbol and entry_market_name != market_name:
                continue
            asset_ctx = asset_ctxs[index] if index < len(asset_ctxs) else {}
            price, price_field = self._quote_price_from_asset_ctx(asset_ctx)
            if price is None:
                raise RuntimeError(f"No usable quote price in metaAndAssetCtxs for {execution_symbol!r}")
            quote_reference_prices = self._quote_reference_prices(asset_ctx, price_field)
            max_deviation_pct = self._max_quote_reference_deviation_pct(price, quote_reference_prices)
            if (
                max_deviation_pct is not None
                and max_deviation_pct > HYPERLIQUID_QUOTE_MAX_REFERENCE_DEVIATION_PCT
            ):
                raise RuntimeError(
                    f"Rejected quote fallback for {execution_symbol!r}: {price_field}={price} differs from "
                    f"reference quote fields by up to {max_deviation_pct:.6f}%"
                )
            enriched_market = dict(market)
            enriched_market.update(
                {
                    "price_source": "metaAndAssetCtxs",
                    "quote_only": True,
                    "quote_price_field": price_field,
                    "quote_price": price,
                    "quote_reference_prices": quote_reference_prices,
                    "quote_max_reference_deviation_pct": max_deviation_pct,
                    "quote_max_reference_deviation_threshold_pct": HYPERLIQUID_QUOTE_MAX_REFERENCE_DEVIATION_PCT,
                    "quote_open_interest": _safe_float(asset_ctx.get("openInterest")),
                    "quote_day_base_volume": _safe_float(asset_ctx.get("dayBaseVlm")),
                    "quote_day_notional_volume": _safe_float(asset_ctx.get("dayNtlVlm")),
                    "quote_is_delisted": bool(entry.get("isDelisted", False)),
                }
            )
            bar = CandleBar(
                ts_ms=_floor_ts_ms(ts_ms, 1),
                open=price,
                high=price,
                low=price,
                close=price,
                volume=_safe_float(asset_ctx.get("dayBaseVlm"), 0.0) or 0.0,
                source_symbol=execution_symbol,
                quote_only=True,
            )
            return [bar], enriched_market
        raise RuntimeError(f"Could not find metaAndAssetCtxs quote for {execution_symbol!r}")

    def fetch_1m_bars(self, raw_symbol: str, bar_count: int) -> tuple[List[CandleBar], Dict[str, Any]]:
        _raise_if_blocked_hyperliquid_vix(raw_symbol)
        market = self.resolve_market(raw_symbol)
        execution_symbol = str(market.get("execution_symbol", "") or "").strip()
        market_name = str(market.get("market_name", "") or execution_symbol).strip().upper()
        coins_to_try = _dedupe([execution_symbol, market_name, str(raw_symbol or "").strip().upper().replace("-USDC", "")])
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - int(max(1, bar_count) * 60_000)
        last_error: Optional[Exception] = None
        for coin in coins_to_try:
            try:
                payload = {
                    "type": "candleSnapshot",
                    "req": {
                        "coin": coin,
                        "interval": "1m",
                        "startTime": start_ms,
                        "endTime": end_ms,
                    },
                }
                data = self._post_info(payload)
                bars = [
                    CandleBar(
                        ts_ms=int(item.get("t", 0) or 0),
                        open=float(item.get("o")),
                        high=float(item.get("h")),
                        low=float(item.get("l")),
                        close=float(item.get("c")),
                        volume=float(item.get("v", 0.0) or 0.0),
                        source_symbol=execution_symbol,
                    )
                    for item in list(data or [])
                    if isinstance(item, dict) and item.get("t") is not None
                ]
                if bars:
                    return bars, market
            except Exception as exc:
                last_error = exc
        try:
            return self._fetch_quote_snapshot_bar(market, end_ms)
        except Exception as exc:
            last_error = exc
        raise RuntimeError(f"Failed to fetch Hyperliquid 1m candles for {raw_symbol!r}: {last_error}")


class TwelveDataClient:
    def __init__(self) -> None:
        self.api_key = (os.getenv("TWELVEDATA_API_KEY", "") or "").strip()
        self.session = requests.Session()

    def _require_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError("TWELVEDATA_API_KEY is required for synthetic DXY")
        return self.api_key

    def fetch_1m_bars(self, symbol: str, bar_count: int) -> List[CandleBar]:
        params = {
            "symbol": symbol,
            "interval": "1min",
            "outputsize": min(MAX_BASE_1M_BARS, max(1, int(bar_count))),
            "order": "asc",
            "timezone": "UTC",
            "apikey": self._require_api_key(),
        }
        response = self.session.get("https://api.twelvedata.com/time_series", params=params, timeout=TWELVEDATA_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
        status = str(payload.get("status", "ok") or "ok").strip().lower()
        if status != "ok":
            raise RuntimeError(f"Twelve Data error for {symbol}: {payload.get('message') or payload}")
        bars: List[CandleBar] = []
        for item in list(payload.get("values") or []):
            if not isinstance(item, dict):
                continue
            bars.append(
                CandleBar(
                    ts_ms=_parse_iso_or_utc_datetime(item.get("datetime", "")),
                    open=float(item.get("open")),
                    high=float(item.get("high")),
                    low=float(item.get("low")),
                    close=float(item.get("close")),
                    volume=float(item.get("volume", 0.0) or 0.0),
                    source_symbol=symbol,
                )
            )
        return bars


class FinnhubQuoteClient:
    def __init__(self) -> None:
        self.api_key = (os.getenv("FINNHUB_API_KEY", "") or os.getenv("FINNHUB_TOKEN", "") or "").strip()
        self.session = requests.Session()

    def _require_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError("FINNHUB_API_KEY or FINNHUB_TOKEN is required for Finnhub ETF quotes")
        return self.api_key

    def fetch_quote_bar(self, symbol: str) -> tuple[List[CandleBar], Dict[str, Any]]:
        normalized_symbol = str(symbol or "").strip().upper()
        if not normalized_symbol:
            raise ValueError("Missing Finnhub quote symbol")
        response = self.session.get(
            "https://finnhub.io/api/v1/quote",
            params={"symbol": normalized_symbol, "token": self._require_api_key()},
            timeout=FINNHUB_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = dict(response.json() or {})
        price = _safe_float(payload.get("c"))
        if price is None or price <= 0:
            raise RuntimeError(f"Finnhub quote for {normalized_symbol!r} has no usable current price: {payload}")
        ts_seconds = _safe_float(payload.get("t"), time.time()) or time.time()
        ts_ms = _floor_ts_ms(int(float(ts_seconds) * 1000), 1)
        quote_age_seconds = max(0.0, time.time() - float(ts_seconds))
        if FINNHUB_MAX_QUOTE_AGE_SECONDS and quote_age_seconds > FINNHUB_MAX_QUOTE_AGE_SECONDS:
            raise RuntimeError(
                f"Finnhub quote for {normalized_symbol!r} is stale: "
                f"{quote_age_seconds:.3f}s > {FINNHUB_MAX_QUOTE_AGE_SECONDS:.3f}s"
            )
        bar = CandleBar(
            ts_ms=ts_ms,
            open=price,
            high=price,
            low=price,
            close=price,
            volume=0.0,
            source_symbol=normalized_symbol,
            quote_only=True,
        )
        meta = {
            "provider": "finnhub",
            "price_source": "quote",
            "quote_only": True,
            "quote_symbol": normalized_symbol,
            "quote_price": price,
            "quote_timestamp": ms_to_iso(ts_ms),
            "quote_age_seconds": quote_age_seconds,
            "quote_open": _safe_float(payload.get("o")),
            "quote_high": _safe_float(payload.get("h")),
            "quote_low": _safe_float(payload.get("l")),
            "quote_previous_close": _safe_float(payload.get("pc")),
            "quote_change": _safe_float(payload.get("d")),
            "quote_change_pct": _safe_float(payload.get("dp")),
            "quote_freshness_threshold_seconds": FINNHUB_MAX_QUOTE_AGE_SECONDS or None,
        }
        return [bar], meta


class DatabentoHistoricalClient:
    def __init__(self) -> None:
        self.api_key = (os.getenv("DATABENTO_API_KEY", "") or "").strip()
        self.session = requests.Session()

    def _require_api_key(self) -> str:
        if not self.api_key:
            raise RuntimeError("DATABENTO_API_KEY is required for the US10Y proxy")
        return self.api_key

    @staticmethod
    def _request_payload(symbol: str, start_dt: datetime, end_dt: datetime) -> Dict[str, str]:
        return {
            "dataset": "GLBX.MDP3",
            "symbols": symbol,
            "stype_in": "continuous",
            "schema": "ohlcv-1m",
            "start": start_dt.isoformat().replace("+00:00", "Z"),
            "end": end_dt.isoformat().replace("+00:00", "Z"),
            "encoding": "json",
            "pretty_px": "true",
            "pretty_ts": "true",
            "map_symbols": "true",
        }

    def fetch_1m_bars(self, symbol: str, bar_count: int) -> List[CandleBar]:
        lookback_minutes = max(1, int(bar_count))
        request_end = datetime.now(timezone.utc)
        response = None
        for _ in range(4):
            request_start = request_end - timedelta(minutes=lookback_minutes)
            request_data = self._request_payload(symbol, request_start, request_end)
            response = self.session.post(
                "https://hist.databento.com/v0/timeseries.get_range",
                data=request_data,
                auth=(self._require_api_key(), ""),
                timeout=DATABENTO_TIMEOUT_SECONDS,
            )
            if response.status_code != 422:
                break
            try:
                error_payload = response.json()
            except Exception:
                error_payload = {}
            detail = dict((error_payload or {}).get("detail") or {})
            available_end = str(((detail.get("payload") or {}).get("available_end") or detail.get("available_end") or "")).strip()
            if not available_end:
                break
            adjusted_end = datetime.fromisoformat(available_end.replace("Z", "+00:00"))
            if adjusted_end >= request_end:
                break
            request_end = adjusted_end
        if response is None:
            raise RuntimeError(f"Failed to request Databento bars for {symbol!r}")
        response.raise_for_status()
        bars: List[CandleBar] = []
        for line in response.text.splitlines():
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            header = item.get("hd") or item
            ts_value = header.get("ts_event")
            open_px = item.get("open")
            if ts_value is None or open_px is None:
                continue
            bars.append(
                CandleBar(
                    ts_ms=_parse_iso_or_utc_datetime(ts_value),
                    open=float(open_px),
                    high=float(item.get("high")),
                    low=float(item.get("low")),
                    close=float(item.get("close")),
                    volume=float(item.get("volume", 0.0) or 0.0),
                    source_symbol=str(item.get("symbol", symbol) or symbol),
                )
            )
        return bars


class CrossAssetTrendFetcher:
    def __init__(
        self,
        *,
        hyperliquid_client: Optional[HyperliquidInfoClient] = None,
        twelvedata_client: Optional[TwelveDataClient] = None,
        databento_client: Optional[DatabentoHistoricalClient] = None,
        finnhub_client: Optional[FinnhubQuoteClient] = None,
    ) -> None:
        self.hyperliquid = hyperliquid_client or HyperliquidInfoClient()
        self.twelvedata = twelvedata_client or TwelveDataClient()
        self.databento = databento_client or DatabentoHistoricalClient()
        self.finnhub = finnhub_client or FinnhubQuoteClient()

    @staticmethod
    def _base_1m_bars_needed(timeframes: Sequence[str], bars_per_timeframe: int) -> int:
        normalized_timeframes = normalize_timeframes(timeframes)
        max_minutes = max(TIMEFRAME_TO_MINUTES[item] for item in normalized_timeframes)
        needed = max_minutes * max(1, int(bars_per_timeframe))
        if needed > MAX_BASE_1M_BARS:
            raise ValueError(
                f"Requested lookback requires {needed} base 1m bars, exceeding the 5000-bar cap. "
                f"Reduce bars_per_timeframe or remove the largest timeframe."
            )
        return needed

    def _fetch_synthetic_dxy_1m_bars(self, bar_count: int) -> List[CandleBar]:
        component_series = {
            pair: self.twelvedata.fetch_1m_bars(pair, bar_count)
            for pair in SYNTHETIC_DXY_COMPONENTS
        }
        bars = build_synthetic_dxy_bars(component_series)
        if not bars:
            raise RuntimeError("No common 1m bars available across the synthetic DXY component set")
        return bars

    def fetch_symbol_trends(
        self,
        symbol_name: str,
        *,
        timeframes: Optional[Sequence[str]] = None,
        bars_per_timeframe: int = DEFAULT_BARS_PER_TIMEFRAME,
        include_bars: bool = True,
    ) -> Dict[str, Any]:
        normalized_timeframes = normalize_timeframes(timeframes)
        base_needed = self._base_1m_bars_needed(normalized_timeframes, bars_per_timeframe)
        # Small buffer helps when a source has sparse minutes.
        fetch_count = min(MAX_BASE_1M_BARS, base_needed + max(10, base_needed // 20))
        source_kind = detect_special_source(symbol_name)
        notes: List[str] = []
        market_meta: Dict[str, Any] = {}
        inverse_price_relation = False

        if source_kind == "vix_spot_unsupported":
            raise RuntimeError(
                "Reliable realtime VIX spot is not available through the configured free sources. "
                "Request VIXY for the preferred volatility proxy or VXX as a backup; these are not spot VIX."
            )
        if source_kind == "dxy":
            bars = self._fetch_synthetic_dxy_1m_bars(fetch_count)
            resolved_symbol = "DXY"
            source = "synthetic_dxy"
            source_symbol = "DXY.synthetic"
            notes.append(
                "Lightweight DXY proxy built from EUR/USD and USD/JPY using the corresponding classic DXY weights."
            )
            notes.append(
                "This proxy is for frequent dollar-strength direction checks and is not the official full-basket DXY level."
            )
            notes.append(
                "Synthetic candle highs/lows are an approximation because component extremes are not time-synchronized within the bar."
            )
            market_meta = {
                "formula_scale": SYNTHETIC_DXY_SCALE,
                "components": dict(SYNTHETIC_DXY_COMPONENTS),
            }
        elif source_kind in {"us2y", "us10y"}:
            if source_kind == "us2y":
                proxy_symbol = (os.getenv("TREND_FETCHER_ZT_SYMBOL", "ZT.v.0") or "ZT.v.0").strip()
                canonical_symbol = "US2Y"
                note_tenor = "2Y"
                futures_root = "ZT"
            else:
                proxy_symbol = (os.getenv("TREND_FETCHER_ZN_SYMBOL", "ZN.v.0") or "ZN.v.0").strip()
                canonical_symbol = "US10Y"
                note_tenor = "10Y"
                futures_root = "ZN"
            bars = self.databento.fetch_1m_bars(proxy_symbol, fetch_count)
            normalized_symbol = _normalize_lookup_token(symbol_name)
            resolved_symbol = canonical_symbol if normalized_symbol.startswith(canonical_symbol) or normalized_symbol.startswith("UST" + canonical_symbol[2:]) else str(symbol_name or proxy_symbol).strip().upper()
            source = "databento"
            source_symbol = proxy_symbol
            inverse_price_relation = True
            notes.append(f"{canonical_symbol} is proxied with CME/CBOT {note_tenor} Treasury Note futures ({futures_root}) minute bars from Databento.")
            notes.append(f"{futures_root} price up usually implies {canonical_symbol} yield down; inverse_direction and inverse_change_pct expose that relation.")
            market_meta = {
                "proxy_symbol": proxy_symbol,
                "dataset": "GLBX.MDP3",
                "schema": "ohlcv-1m",
                "stype_in": "continuous",
            }
        elif source_kind == "twelvedata_forex":
            forex_symbol = resolve_twelvedata_forex_symbol(symbol_name)
            bars = self.twelvedata.fetch_1m_bars(forex_symbol, fetch_count)
            resolved_symbol = _normalize_lookup_token(forex_symbol)
            source = "twelvedata_forex"
            source_symbol = forex_symbol
            notes.append(
                f"{resolved_symbol} is fetched from TwelveData forex 1min time_series because no preferred Hyperliquid market is configured."
            )
            market_meta = {
                "provider": "twelvedata",
                "source_symbol": forex_symbol,
                "interval": "1min",
            }
        elif source_kind == "finnhub_quote":
            finnhub_symbol = resolve_finnhub_quote_symbol(symbol_name)
            bars, quote_meta = self.finnhub.fetch_quote_bar(finnhub_symbol)
            resolved_symbol = finnhub_symbol
            source = "finnhub_quote"
            source_symbol = finnhub_symbol
            market_meta = dict(FINNHUB_QUOTE_SYMBOL_META.get(finnhub_symbol, {}))
            market_meta.update(dict(quote_meta))
            notes.append(
                "Finnhub quote endpoint is used for this ETF/proxy symbol; this path returns the current quote and day quote fields only."
            )
            notes.append(
                "No intraday historical candles are inferred from quote data, so timeframe summaries are marked quote_only with unknown direction."
            )
            if market_meta.get("proxy_note"):
                notes.append(str(market_meta["proxy_note"]))
        else:
            bars, market = self.hyperliquid.fetch_1m_bars(symbol_name, fetch_count)
            resolved_symbol = str(market.get("display_name", symbol_name) or symbol_name).strip().upper()
            source = "hyperliquid"
            source_symbol = str(market.get("execution_symbol", "") or "").strip()
            market_meta = dict(market)
            if market_meta.get("price_source") == "metaAndAssetCtxs":
                notes.append(
                    "Hyperliquid candleSnapshot was unavailable; using a metaAndAssetCtxs quote fallback for the current price only."
                )
                notes.append(
                    "Quote fallback validates the selected price against other available quote fields, but it is not historical candle data."
                )
            else:
                notes.append("Primary trade instrument fetched from Hyperliquid /info candleSnapshot on 1m, then resampled locally.")

        return build_trend_snapshot(
            requested_symbol=symbol_name,
            resolved_symbol=resolved_symbol,
            source=source,
            source_symbol=source_symbol,
            base_1m_bars=bars,
            timeframes=normalized_timeframes,
            bars_per_timeframe=bars_per_timeframe,
            include_bars=include_bars,
            flat_threshold_pct=TREND_FETCHER_FLAT_THRESHOLD_PCT,
            notes=notes,
            market_meta=market_meta,
            inverse_price_relation=inverse_price_relation,
        )

    def fetch_trade_bundle(
        self,
        trade_symbol: str,
        *,
        timeframes: Optional[Sequence[str]] = None,
        bars_per_timeframe: int = DEFAULT_BARS_PER_TIMEFRAME,
        include_bars: bool = True,
        include_macro: bool = True,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "generated_at": ms_to_iso(int(time.time() * 1000)),
            "trade_symbol": str(trade_symbol or "").strip(),
            "primary": self.fetch_symbol_trends(
                trade_symbol,
                timeframes=timeframes,
                bars_per_timeframe=bars_per_timeframe,
                include_bars=include_bars,
            ),
            "macro": {},
        }
        if include_macro:
            requested_normalized = _normalize_lookup_token(trade_symbol)
            for macro_symbol in ["DXY", "US2Y", "US10Y"]:
                if _normalize_lookup_token(macro_symbol) == requested_normalized:
                    continue
                payload["macro"][macro_symbol] = self.fetch_symbol_trends(
                    macro_symbol,
                    timeframes=timeframes,
                    bars_per_timeframe=bars_per_timeframe,
                    include_bars=include_bars,
                )
        return payload


def _parse_timeframe_csv(raw: str) -> List[str]:
    return normalize_timeframes([item.strip() for item in str(raw or "").split(",") if item.strip()])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fetch multi-timeframe trends for a Hyperliquid trade instrument, with optional DXY, Treasury, and Finnhub ETF quote proxies."
        )
    )
    parser.add_argument("symbol", help="Trade symbol or macro alias. Examples: BTC-USDC, ETH-USDC, BRENTOIL-USDC, DXY, US2Y, US10Y, CPER, JETS")
    parser.add_argument("--timeframes", default=",".join(SUPPORTED_TIMEFRAMES), help="Comma-separated timeframes. Default: 1m,5m,15m,30m,1h")
    parser.add_argument("--bars-per-timeframe", type=int, default=DEFAULT_BARS_PER_TIMEFRAME, help="How many bars to keep per timeframe. Default: 48")
    parser.add_argument("--no-bars", action="store_true", help="Only return summaries, not per-timeframe bar arrays")
    parser.add_argument("--bundle", action="store_true", help="Fetch the trade symbol plus the DXY, US2Y, and US10Y proxies together")
    args = parser.parse_args()

    fetcher = CrossAssetTrendFetcher()
    timeframes = _parse_timeframe_csv(args.timeframes)
    if args.bundle:
        payload = fetcher.fetch_trade_bundle(
            args.symbol,
            timeframes=timeframes,
            bars_per_timeframe=args.bars_per_timeframe,
            include_bars=not args.no_bars,
            include_macro=True,
        )
    else:
        payload = fetcher.fetch_symbol_trends(
            args.symbol,
            timeframes=timeframes,
            bars_per_timeframe=args.bars_per_timeframe,
            include_bars=not args.no_bars,
        )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
