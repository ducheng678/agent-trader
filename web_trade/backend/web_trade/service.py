from __future__ import annotations

import json
import os
import time
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from market_agent.exchange import HyperliquidExecutor, HyperliquidRestReader
from market_agent.symbols import canonicalize_execution_symbol, split_execution_symbol

from web_trade.backend.web_trade.ledger import (
    SyntheticPositionLedger,
    current_margin_basis_usd,
    snapshot_has_open_position,
)
from web_trade.backend.web_trade.margin import calculate_margin_limits


ExecutorFactory = Callable[[Any, str], Any]


RESOLUTION_TO_INTERVAL = {
    "1": "1m",
    "5": "5m",
    "15": "15m",
    "30": "30m",
    "60": "1h",
    "240": "4h",
    "1D": "1d",
    "D": "1d",
    "1W": "1w",
    "W": "1w",
}

INTERVAL_SECONDS = {
    "1m": 60,
    "5m": 300,
    "15m": 900,
    "30m": 1800,
    "1h": 3600,
    "4h": 14400,
    "1d": 86400,
    "1w": 604800,
}


def mask_address(address: str) -> str:
    value = str(address or "").strip()
    if len(value) <= 12:
        return value
    return f"{value[:6]}...{value[-4:]}"


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:
        return default
    return result


def _payload_has_exchange_error(payload: Any) -> bool:
    if isinstance(payload, dict):
        status = str(payload.get("status", "") or "").strip().lower()
        if status == "err":
            return True
        error = payload.get("error")
        if isinstance(error, str) and error.strip():
            return True
        return any(_payload_has_exchange_error(value) for value in payload.values())
    if isinstance(payload, list):
        return any(_payload_has_exchange_error(item) for item in payload)
    return False


def _result_accepted(result: Any) -> bool:
    if not isinstance(result, dict):
        return False
    if result.get("accepted") is False:
        return False
    if _payload_has_exchange_error(result):
        return False
    status = str(result.get("status", "") or "").lower()
    if status == "err":
        return False
    return True


def _configured_min_trade_notional_usd() -> float:
    raw = os.getenv("WEB_TRADE_MIN_TRADE_NOTIONAL_USD", os.getenv("HYPERLIQUID_MIN_TRADE_NOTIONAL_USD", "10"))
    return max(0.0, _as_float(raw, 10.0))


def _configured_main_perp_exceptions() -> set[str]:
    raw = os.getenv("WEB_TRADE_MAIN_PERP_EXCEPTIONS", "BTC,ETH")
    assets: set[str] = set()
    for item in str(raw or "").split(","):
        _, asset = split_execution_symbol(canonicalize_execution_symbol(item.strip()))
        asset = asset.removesuffix("-USDC").strip().upper()
        if asset:
            assets.add(asset)
    return assets


def _market_filter_parts(market: Dict[str, Any]) -> tuple[str, str]:
    symbol = str(market.get("symbol") or market.get("execution_symbol") or "").strip()
    symbol_dex, symbol_asset = split_execution_symbol(symbol)
    dex = str(market.get("dex", symbol_dex) or "").strip().lower()
    asset = str(market.get("market_name") or symbol_asset or symbol).strip().upper().removesuffix("-USDC")
    return dex, asset


def _effective_open_notional_after_size_rounding(reader: Any, symbol: str, target_notional: float, mid_price: float) -> float:
    if target_notional <= 0.0 or mid_price <= 0.0:
        return max(0.0, target_notional)
    get_decimals = getattr(reader, "get_sz_decimals", None)
    if not callable(get_decimals):
        return max(0.0, target_notional)
    try:
        decimals = max(0, int(get_decimals(symbol) or 0))
        notional = Decimal(str(max(0.0, float(target_notional or 0.0))))
        mid = Decimal(str(max(float(mid_price or 0.0), 1e-12)))
        quantum = Decimal(1).scaleb(-decimals)
        qty = (notional / mid).quantize(quantum, rounding=ROUND_DOWN)
        if qty <= 0:
            return 0.0
        return float(qty * mid)
    except (InvalidOperation, ValueError, TypeError, ZeroDivisionError):
        return max(0.0, target_notional)


def _leverage_rebalance_target_from_position(snapshot: Dict[str, Any], target_leverage: int) -> tuple[float, float]:
    current_notional = abs(_as_float((snapshot or {}).get("notional_usd"), 0.0))
    current_leverage = max(0.0, _as_float((snapshot or {}).get("leverage"), 0.0))
    target = max(1, int(target_leverage or 1))
    if current_notional > 0.0 and current_leverage > 0.0:
        target_notional = current_notional * (float(target) / current_leverage)
        return target_notional, target_notional / float(target)
    current_margin = current_margin_basis_usd(snapshot)
    if current_margin > 0.0:
        target_notional = current_margin * float(target)
        return target_notional, current_margin
    return 0.0, 0.0


def _synthetic_flat_snapshot_after_close(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    flat = dict(snapshot or {})
    flat.update(
        {
            "side": "flat",
            "size": 0.0,
            "notional_usd": 0.0,
            "leverage": 0.0,
            "margin_used": 0.0,
            "unrealized_pnl": 0.0,
        }
    )
    return flat


def _expected_position_after_rebalance(
    snapshot: Dict[str, Any],
    open_result: Dict[str, Any],
    *,
    side: str,
    target_leverage: int,
    target_notional_usd: float,
    rebalance_margin_usd: float,
) -> Dict[str, Any]:
    view = dict(snapshot or {})
    mid = _as_float(open_result.get("mid"), _as_float(snapshot.get("mid_price"), 0.0))
    qty = _as_float(open_result.get("open_qty"), 0.0)
    if qty <= 0.0:
        qty = _as_float(open_result.get("final_qty"), _as_float(open_result.get("requested_qty"), 0.0))
    notional = abs(qty) * mid if qty > 0.0 and mid > 0.0 else max(0.0, _as_float(target_notional_usd, 0.0))
    signed_qty = abs(qty)
    if str(side or "").lower() == "short":
        signed_qty = -signed_qty
    implied_margin = (
        notional / float(target_leverage)
        if target_leverage > 0 and notional > 0.0
        else max(0.0, _as_float(rebalance_margin_usd, 0.0))
    )
    view.update(
        {
            "side": side,
            "size": signed_qty,
            "notional_usd": notional,
            "leverage": float(target_leverage or 0),
            "margin_used": implied_margin,
            "unrealized_pnl": 0.0,
            "return_on_equity": 0.0,
        }
    )
    if mid > 0.0:
        view["mid_price"] = mid
        view["entry_price"] = mid
    return view


def _fresh_liquidation_price_after_rebalance(snapshot: Dict[str, Any], *, side: str, target_leverage: int) -> float:
    if not snapshot_has_open_position(snapshot):
        return 0.0
    if str((snapshot or {}).get("side", "") or "").lower() != str(side or "").lower():
        return 0.0
    leverage = _as_float((snapshot or {}).get("leverage"), 0.0)
    if target_leverage > 0 and abs(leverage - float(target_leverage)) > 1e-9:
        return 0.0
    return max(0.0, _as_float((snapshot or {}).get("liquidation_price"), 0.0))


def _normalize_candle(raw: Dict[str, Any]) -> Dict[str, float] | None:
    def pick(*keys: str) -> Any:
        for key in keys:
            if key in raw:
                return raw.get(key)
        return None

    timestamp = _as_float(pick("time", "t", "T"), 0.0)
    if timestamp <= 0.0:
        return None
    if timestamp > 10_000_000_000:
        timestamp = timestamp / 1000.0
    open_px = _as_float(pick("open", "o"), 0.0)
    high_px = _as_float(pick("high", "h"), 0.0)
    low_px = _as_float(pick("low", "l"), 0.0)
    close_px = _as_float(pick("close", "c"), 0.0)
    if min(open_px, high_px, low_px, close_px) <= 0.0:
        return None
    return {
        "time": int(timestamp),
        "open": open_px,
        "high": high_px,
        "low": low_px,
        "close": close_px,
        "volume": _as_float(pick("volume", "v"), 0.0),
    }


def _normalize_book_side(levels: Any) -> List[Dict[str, float]]:
    rows: List[Dict[str, float]] = []
    running_total = 0.0
    for item in list(levels or []):
        if not isinstance(item, dict):
            continue
        price_value = _as_float(item.get("price", item.get("px")), 0.0)
        size_value = _as_float(item.get("size", item.get("sz")), 0.0)
        if price_value <= 0.0 or size_value <= 0.0:
            continue
        running_total += size_value
        rows.append(
            {
                "price": price_value,
                "size": size_value,
                "total": running_total,
                "orders": int(_as_float(item.get("orders", item.get("n")), 0.0)),
            }
        )
    return rows


def _history_time_ms(item: Any) -> int:
    if not isinstance(item, dict):
        return 0
    for key in ("statusTimestamp", "time", "timestamp", "startTime", "endTime"):
        value = _as_float(item.get(key), 0.0)
        if value > 0:
            return int(value)
    order = item.get("order")
    if isinstance(order, dict):
        for key in ("timestamp", "time"):
            value = _as_float(order.get(key), 0.0)
            if value > 0:
                return int(value)
    return 0


def _strip_private_history_fields(item: Any) -> Any:
    if isinstance(item, list):
        return [_strip_private_history_fields(entry) for entry in item]
    if not isinstance(item, dict):
        return item
    return {
        key: _strip_private_history_fields(value)
        for key, value in item.items()
        if key not in {"user", "account_address"}
    }


class WebTradeService:
    def __init__(
        self,
        *,
        reader: Optional[Any] = None,
        ledger: Optional[SyntheticPositionLedger] = None,
        executor_factory: ExecutorFactory = HyperliquidExecutor,
        favorites_path: Optional[Path] = None,
    ):
        self.reader = reader if reader is not None else HyperliquidRestReader()
        default_ledger_path = Path(__file__).resolve().parents[2] / "runtime" / "positions.json"
        self.ledger = ledger if ledger is not None else SyntheticPositionLedger(default_ledger_path)
        self.executor_factory = executor_factory
        self.favorites_path = favorites_path or Path(__file__).resolve().parents[2] / "runtime" / "favorite_markets.json"
        self.preferred_perp_dex = str(os.getenv("WEB_TRADE_PERP_DEX", "xyz") or "").strip().lower()
        self.main_perp_exceptions = _configured_main_perp_exceptions()

    def _symbol(self, symbol: str) -> str:
        return canonicalize_execution_symbol(symbol)

    def _market_symbol(self, raw_symbol: str, markets_by_symbol: Dict[str, Dict[str, Any]]) -> str:
        symbol = self._symbol(raw_symbol)
        if symbol in markets_by_symbol:
            return symbol
        dex, market_name = split_execution_symbol(symbol)
        if self.preferred_perp_dex and market_name and not dex:
            preferred_symbol = canonicalize_execution_symbol(f"{self.preferred_perp_dex}:{market_name}")
            if preferred_symbol in markets_by_symbol:
                return preferred_symbol
        return ""

    def _display_name(self, symbol: str, payload: Dict[str, Any]) -> str:
        raw_display = str(payload.get("display_name") or "").strip()
        dex, market_name = split_execution_symbol(symbol)
        if dex and market_name:
            return f"{market_name}-USDC"
        if raw_display and ":" in raw_display:
            return raw_display.split(":", 1)[1]
        return raw_display or f"{symbol}-USDC"

    def _executor(self, symbol: str) -> Any:
        return self.executor_factory(self.reader, self._symbol(symbol))

    def _position_snapshot(self, symbol: str) -> Dict[str, Any]:
        symbol = self._symbol(symbol)
        if hasattr(self.reader, "get_selected_symbol_position_context"):
            try:
                context = self.reader.get_selected_symbol_position_context(symbol)
                snapshot = context.get("position_snapshot") if isinstance(context, dict) else None
                if isinstance(snapshot, dict):
                    return dict(snapshot)
            except Exception:
                pass
        return dict(self.reader.get_position_snapshot(symbol))

    def session(self) -> Dict[str, Any]:
        address = str(getattr(self.reader, "account_address", "") or "")
        return {
            "network": str(getattr(self.reader, "network", "") or ""),
            "account_address": address,
            "account_address_masked": mask_address(address),
            "live_trading": os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true",
        }

    def markets(self) -> List[Dict[str, Any]]:
        catalog = self.reader.get_market_catalog()
        mids_by_dex: Dict[str, Dict[str, Any]] = {}
        if hasattr(self.reader, "get_mids"):
            mids_by_dex[""] = dict(self.reader.get_mids() or {})
            if hasattr(self.reader, "list_perp_dex_names"):
                for dex_name in list(self.reader.list_perp_dex_names() or []):
                    try:
                        mids_by_dex[str(dex_name or "").strip().lower()] = dict(self.reader.get_mids(dex=dex_name) or {})
                    except Exception:
                        mids_by_dex[str(dex_name or "").strip().lower()] = {}
        markets = []
        for symbol, item in sorted(catalog.items()):
            payload = dict(item)
            payload.setdefault("symbol", symbol)
            payload.setdefault("execution_symbol", symbol)
            payload.setdefault("display_name", f"{symbol}-USDC")
            dex, market_name = split_execution_symbol(symbol)
            payload.setdefault("dex", dex)
            payload.setdefault("market_name", market_name)
            payload["display_name"] = self._display_name(symbol, payload)
            if hasattr(self.reader, "get_ws_symbol"):
                try:
                    payload["ws_symbol"] = str(self.reader.get_ws_symbol(symbol) or symbol)
                except Exception:
                    payload["ws_symbol"] = symbol
            else:
                payload.setdefault("ws_symbol", payload.get("execution_symbol", symbol))
            dex_mids = mids_by_dex.get(dex, {}) or {}
            live_mid = _as_float(dex_mids.get(symbol) or dex_mids.get(market_name), 0.0)
            payload["mid_price"] = live_mid if live_mid > 0.0 else _as_float(payload.get("mid_price"), 0.0)
            markets.append(payload)
        if self.preferred_perp_dex:
            main_exception_assets = {
                asset
                for market in markets
                for dex, asset in [_market_filter_parts(market)]
                if not dex and asset in self.main_perp_exceptions
            }
            filtered_markets = []
            for market in markets:
                dex, asset = _market_filter_parts(market)
                is_main_exception = not dex and asset in self.main_perp_exceptions
                is_preferred = dex == self.preferred_perp_dex and asset not in main_exception_assets
                if is_main_exception or is_preferred:
                    filtered_markets.append(market)
            if filtered_markets:
                return filtered_markets
        return markets

    def account(self) -> Dict[str, Any]:
        payload = dict(self.reader.get_all_positions())
        positions = []
        for position in list(payload.get("positions", []) or []):
            position_payload = dict(position)
            position_payload.setdefault("available_margin_usd", payload.get("available_margin_usd", 0.0))
            position_payload.setdefault("withdrawable_usd", payload.get("withdrawable_usd", 0.0))
            position_payload.setdefault("remaining_capital_usd", payload.get("remaining_capital_usd", 0.0))
            view = self.ledger.overlay_position(position_payload)
            view["margin_limits"] = calculate_margin_limits(position_payload)
            positions.append(view)
        payload["positions"] = positions
        try:
            payload["open_orders"] = list(self.reader.get_frontend_open_orders())
        except Exception:
            payload["open_orders"] = []
        return payload

    def account_history(self, window_days: int = 90) -> Dict[str, Any]:
        days = max(1, int(window_days or 90))
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - days * 24 * 60 * 60 * 1000
        address = str(getattr(self.reader, "account_address", "") or "").strip()

        def read_list(method_name: str, *args: Any, **kwargs: Any) -> List[Dict[str, Any]]:
            method = getattr(self.reader, method_name, None)
            if not method or not address:
                return []
            try:
                return [dict(item) for item in list(method(*args, **kwargs) or []) if isinstance(item, dict)]
            except Exception:
                return []

        trade_history = read_list("get_user_fills_by_time", address, start_ms, end_ms, aggregate_by_time=False)
        funding_history = read_list("get_user_funding_history", address, start_ms, end_ms)
        raw_order_history = read_list("get_historical_orders", address)
        order_history = [
            item
            for item in raw_order_history
            if _history_time_ms(item) <= 0 or _history_time_ms(item) >= start_ms
        ]
        order_history.sort(key=_history_time_ms, reverse=True)

        return {
            "window_days": days,
            "start_time_ms": start_ms,
            "end_time_ms": end_ms,
            "trade_history": _strip_private_history_fields(trade_history),
            "funding_history": _strip_private_history_fields(funding_history),
            "order_history": _strip_private_history_fields(order_history),
        }

    def market_snapshot(self, symbol: str, interval: str = "1m", window_seconds: int = 3600) -> Dict[str, Any]:
        symbol = self._symbol(symbol)
        dex, _ = split_execution_symbol(symbol)
        end_ms = int(time.time() * 1000)
        start_ms = end_ms - max(60, int(window_seconds or 3600)) * 1000
        raw_candles = list(self.reader.get_candles_snapshot(symbol, interval, start_ms, end_ms))
        candles = [item for item in (_normalize_candle(dict(candle)) for candle in raw_candles if isinstance(candle, dict)) if item]
        mids = self.reader.get_mids(dex=dex) if dex and hasattr(self.reader, "get_mids") else self.reader.get_mids() if hasattr(self.reader, "get_mids") else {}
        return {
            "symbol": symbol,
            "interval": interval,
            "candles": candles,
            "mid_price": _as_float((mids or {}).get(symbol), 0.0) if isinstance(mids, dict) else 0.0,
        }

    def market_bars(
        self,
        symbol: str,
        *,
        resolution: str = "1",
        from_s: int | None = None,
        to_s: int | None = None,
        count_back: int | None = None,
    ) -> Dict[str, Any]:
        symbol = self._symbol(symbol)
        resolution_key = str(resolution or "1")
        interval = RESOLUTION_TO_INTERVAL.get(resolution_key, resolution_key)
        if interval not in INTERVAL_SECONDS:
            raise ValueError(f"Unsupported chart resolution: {resolution}")
        interval_seconds = INTERVAL_SECONDS[interval]
        end_s = int(to_s or time.time())
        requested_count = max(0, int(count_back or 0))
        if from_s is not None and int(from_s) > 0:
            start_s = int(from_s)
        elif requested_count > 0:
            start_s = end_s - requested_count * interval_seconds
        else:
            start_s = end_s - 500 * interval_seconds
        start_s = max(0, min(start_s, end_s - interval_seconds))
        raw_candles = list(self.reader.get_candles_snapshot(symbol, interval, start_s * 1000, end_s * 1000))
        bars = [
            item
            for item in (_normalize_candle(dict(candle)) for candle in raw_candles if isinstance(candle, dict))
            if item and start_s <= int(item["time"]) <= end_s
        ]
        if requested_count > 0:
            bars = bars[-requested_count:]
        return {
            "symbol": symbol,
            "resolution": resolution_key,
            "interval": interval,
            "bars": bars,
            "no_data": len(bars) == 0,
        }

    def market_book(self, symbol: str) -> Dict[str, Any]:
        symbol = self._symbol(symbol)
        raw = self.reader.get_l2_book_snapshot(symbol) if hasattr(self.reader, "get_l2_book_snapshot") else {}
        levels = list(raw.get("levels", []) or []) if isinstance(raw, dict) else []
        bids = _normalize_book_side(levels[0] if len(levels) > 0 else [])
        asks = _normalize_book_side(levels[1] if len(levels) > 1 else [])
        if bids and asks:
            mid_price = (bids[0]["price"] + asks[0]["price"]) / 2.0
        else:
            dex, _ = split_execution_symbol(symbol)
            mids = (
                self.reader.get_mids(dex=dex)
                if dex and hasattr(self.reader, "get_mids")
                else self.reader.get_mids()
                if hasattr(self.reader, "get_mids")
                else {}
            )
            mid_price = _as_float((mids or {}).get(symbol), 0.0) if isinstance(mids, dict) else 0.0
        return {
            "symbol": symbol,
            "time": int(_as_float(raw.get("time"), 0.0)) if isinstance(raw, dict) else 0,
            "mid_price": mid_price,
            "bids": bids,
            "asks": asks,
        }

    def _read_favorite_symbols(self) -> List[str]:
        try:
            payload = json.loads(self.favorites_path.read_text())
        except (OSError, json.JSONDecodeError):
            return []
        raw_symbols = payload.get("symbols", []) if isinstance(payload, dict) else payload
        if not isinstance(raw_symbols, list):
            return []
        return [str(symbol) for symbol in raw_symbols]

    def favorite_markets(self) -> List[Dict[str, Any]]:
        markets_by_symbol = {str(market.get("symbol")): market for market in self.markets()}
        favorites: List[Dict[str, Any]] = []
        seen: set[str] = set()
        for raw_symbol in self._read_favorite_symbols():
            symbol = self._market_symbol(raw_symbol, markets_by_symbol)
            if symbol in markets_by_symbol and symbol not in seen:
                favorites.append(dict(markets_by_symbol[symbol]))
                seen.add(symbol)
        return favorites

    def update_favorite_markets(self, symbols: List[str]) -> List[Dict[str, Any]]:
        markets_by_symbol = {str(market.get("symbol")): market for market in self.markets()}
        clean_symbols: List[str] = []
        seen: set[str] = set()
        for raw_symbol in list(symbols or []):
            raw_value = str(raw_symbol or "").strip()
            if not raw_value:
                continue
            symbol = self._market_symbol(raw_value, markets_by_symbol)
            if symbol in markets_by_symbol and symbol not in seen:
                clean_symbols.append(symbol)
                seen.add(symbol)
        self.favorites_path.parent.mkdir(parents=True, exist_ok=True)
        self.favorites_path.write_text(json.dumps({"symbols": clean_symbols}, indent=2, sort_keys=True) + "\n")
        return [dict(markets_by_symbol[symbol]) for symbol in clean_symbols]

    def margin_limits(self, symbol: str, safety_buffer_usd: float | None = None) -> Dict[str, Any]:
        symbol = self._symbol(symbol)
        snapshot = self._position_snapshot(symbol)
        return calculate_margin_limits(snapshot, safety_buffer_usd=safety_buffer_usd)

    def place_order(
        self,
        *,
        symbol: str,
        order_type: str,
        side: str,
        margin_usd: float,
        leverage: int,
        limit_price: float = 0.0,
        reduce_only: bool = False,
        close_all: bool = False,
        position_action: str = "open",
    ) -> Dict[str, Any]:
        symbol = self._symbol(symbol)
        raw_type = str(order_type or "").strip().lower()
        raw_side = str(side or "").strip().lower()
        raw_position_action = str(position_action or "open").strip().lower()
        if raw_position_action in {"", "order"}:
            raw_position_action = "open"
        if raw_position_action not in {"open", "reverse"}:
            raise ValueError("position_action must be open or reverse")
        target_side = {"buy": "long", "long": "long", "sell": "short", "short": "short"}.get(raw_side, "")
        if target_side not in {"long", "short"}:
            raise ValueError("side must be long or short")
        if raw_type not in {"market", "limit"}:
            raise ValueError("order_type must be market or limit")
        is_reverse_action = raw_position_action == "reverse"
        if is_reverse_action and raw_type != "market":
            raise ValueError("reverse position action requires a market order")
        if is_reverse_action and reduce_only:
            raise ValueError("reverse position action cannot be reduce_only")
        requested_margin = max(0.0, _as_float(margin_usd, 0.0))
        if requested_margin <= 0.0:
            raise ValueError("margin_usd must be positive")
        market = next((item for item in self.markets() if item.get("symbol") == symbol), {})
        max_leverage = int(market.get("max_leverage", 0) or 0)
        requested_leverage = int(leverage or 0)
        if requested_leverage <= 0:
            requested_leverage = 1
        if max_leverage > 0:
            requested_leverage = min(requested_leverage, max_leverage)
        requested_notional = requested_margin * requested_leverage

        snapshot = self._position_snapshot(symbol)
        if is_reverse_action:
            current_side = str(snapshot.get("side", "") or "").lower()
            if not snapshot_has_open_position(snapshot) or current_side not in {"long", "short"}:
                raise ValueError("No open position to reverse.")
            if current_side == target_side:
                raise ValueError("reverse side must be opposite to the current open position")
        if not reduce_only and not is_reverse_action and "available_margin_usd" in snapshot:
            available_margin = max(0.0, _as_float(snapshot.get("available_margin_usd"), 0.0))
            if requested_margin > available_margin + 1e-9:
                return {
                    "accepted": False,
                    "symbol": symbol,
                    "order_type": raw_type,
                    "side": target_side,
                    "margin_usd": requested_margin,
                    "requested_margin_usd": requested_margin,
                    "available_margin_usd": available_margin,
                    "target_notional_usd": requested_notional,
                    "leverage": requested_leverage,
                    "close_all": bool(close_all),
                    "position_action": raw_position_action,
                    "stage": "insufficient_available_margin",
                    "message": (
                        f"Insufficient available margin. Requested {requested_margin:.6f} USDC, "
                        f"available {available_margin:.6f} USDC."
                    ),
                }
        if not reduce_only:
            min_trade_notional = _configured_min_trade_notional_usd()
            notional_reference_price = (
                _as_float(limit_price, 0.0)
                if raw_type == "limit" and _as_float(limit_price, 0.0) > 0.0
                else _as_float(snapshot.get("mid_price"), 0.0)
            )
            effective_notional = _effective_open_notional_after_size_rounding(
                self.reader,
                symbol,
                requested_notional,
                notional_reference_price,
            )
            if min_trade_notional > 0.0 and (
                0.0 < requested_notional < min_trade_notional or 0.0 <= effective_notional < min_trade_notional
            ):
                return {
                    "accepted": False,
                    "symbol": symbol,
                    "order_type": raw_type,
                    "side": target_side,
                    "margin_usd": requested_margin,
                    "requested_margin_usd": requested_margin,
                    "target_notional_usd": requested_notional,
                    "effective_target_notional_usd": effective_notional,
                    "min_trade_notional_usd": min_trade_notional,
                    "leverage": requested_leverage,
                    "close_all": bool(close_all),
                    "position_action": raw_position_action,
                    "stage": "below_min_trade_notional",
                    "message": (
                        f"Order notional is below the exchange minimum. Requested {requested_notional:.6f} USDC, "
                        f"minimum {min_trade_notional:.6f} USDC."
                    ),
                }
        executor = self._executor(symbol)
        if reduce_only:
            current_side = str(snapshot.get("side", "") or "").lower()
            current_size = abs(_as_float(snapshot.get("size"), 0.0))
            current_notional = max(_as_float(snapshot.get("notional_usd"), 0.0), 1e-12)
            if current_side not in {"long", "short"} or current_side == target_side:
                raise ValueError("reduce_only order must be opposite to the current open position")
            close_size = current_size if close_all and raw_type == "market" else current_size * min(1.0, requested_notional / current_notional)
            if raw_type == "limit":
                rounded_limit = _as_float(limit_price, 0.0)
                if rounded_limit <= 0.0:
                    raise ValueError("limit_price must be positive for limit orders")
                if not hasattr(executor, "place_reduce_only_limit_order"):
                    raise ValueError("reduce_only limit orders are not supported by this executor")
                result = executor.place_reduce_only_limit_order(
                    side=current_side,
                    close_size=close_size,
                    limit_price=rounded_limit,
                    plan_name="web_trade",
                    leg_name="web_reduce_only_limit_order",
                )
            else:
                result = executor.reduce_position(
                    current_side,
                    close_size,
                    "web_reduce_only_order",
                    "web_trade",
                    position_before=snapshot,
                )
        elif is_reverse_action:
            result = executor.execute_position_target(
                target_side=target_side,
                target_notional_usd=requested_notional,
                requested_leverage=requested_leverage,
                reason="web_position_reverse",
                plan_name="web_trade",
                position_before=snapshot,
                execution_mid_price=_as_float(snapshot.get("mid_price"), 0.0),
            )
        elif raw_type == "limit":
            rounded_limit = _as_float(limit_price, 0.0)
            if rounded_limit <= 0.0:
                raise ValueError("limit_price must be positive for limit orders")
            result = executor.place_entry_limit_order(
                side=target_side,
                notional_usd=requested_notional,
                requested_leverage=requested_leverage,
                limit_price=rounded_limit,
                reason="web_limit_order",
                plan_name="web_trade",
            )
        else:
            result = executor.place_market_order(
                side=target_side,
                notional_usd=requested_notional,
                requested_leverage=requested_leverage,
                reason="web_market_order",
                plan_name="web_trade",
                position_before=snapshot,
                execution_mid_price=_as_float(snapshot.get("mid_price"), 0.0),
            )
        response = {
            "accepted": _result_accepted(result),
            "symbol": symbol,
            "order_type": raw_type,
            "side": target_side,
            "margin_usd": requested_margin,
            "target_notional_usd": requested_notional,
            "leverage": requested_leverage,
            "close_all": bool(close_all),
            "position_action": raw_position_action,
            "result": result,
        }
        if is_reverse_action and _result_accepted(result):
            after_snapshot = self._position_snapshot(symbol)
            if snapshot_has_open_position(after_snapshot):
                response["position"] = self.ledger.overlay_position(after_snapshot)
        if isinstance(result, dict):
            message = str(result.get("message", "") or "").strip()
            if message:
                response["message"] = message
            for key in ("partial_fill", "requested_qty", "filled_qty", "filled_notional_usd"):
                if key in result:
                    response[key] = result[key]
        return response

    def set_position_tpsl(
        self,
        *,
        symbol: str,
        take_profit_price: float = 0.0,
        stop_loss_price: float = 0.0,
    ) -> Dict[str, Any]:
        symbol = self._symbol(symbol)
        snapshot = self._position_snapshot(symbol)
        if not snapshot_has_open_position(snapshot):
            raise ValueError("No open position to set TP/SL.")
        current_side = str(snapshot.get("side", "") or "").lower()
        close_size = abs(_as_float(snapshot.get("size"), 0.0))
        if current_side not in {"long", "short"} or close_size <= 0.0:
            raise ValueError("No open position to set TP/SL.")
        tp_price = _as_float(take_profit_price, 0.0)
        sl_price = _as_float(stop_loss_price, 0.0)
        if tp_price <= 0.0 and sl_price <= 0.0:
            raise ValueError("take_profit_price or stop_loss_price must be positive")

        executor = self._executor(symbol)
        if not hasattr(executor, "place_reduce_only_tpsl_order"):
            raise ValueError("TP/SL orders are not supported by this executor")

        orders: List[Dict[str, Any]] = []
        if tp_price > 0.0:
            orders.append(
                executor.place_reduce_only_tpsl_order(
                    side=current_side,
                    close_size=close_size,
                    trigger_price=tp_price,
                    tpsl="tp",
                    plan_name="web_trade",
                    leg_name="web_take_profit",
                )
            )
        if sl_price > 0.0:
            orders.append(
                executor.place_reduce_only_tpsl_order(
                    side=current_side,
                    close_size=close_size,
                    trigger_price=sl_price,
                    tpsl="sl",
                    plan_name="web_trade",
                    leg_name="web_stop_loss",
                )
            )
        return {
            "accepted": bool(orders) and all(_result_accepted(order) for order in orders),
            "symbol": symbol,
            "side": current_side,
            "close_size": close_size,
            "take_profit_price": tp_price,
            "stop_loss_price": sl_price,
            "orders": orders,
        }

    def rebalance_leverage(self, symbol: str, target_leverage: int) -> Dict[str, Any]:
        symbol = self._symbol(symbol)
        before = self._position_snapshot(symbol)
        if not snapshot_has_open_position(before):
            raise ValueError("No open position to rebalance.")
        before_view = self.ledger.overlay_position(before)
        target = max(1, min(int(target_leverage or 0), int(before.get("max_leverage", 0) or target_leverage or 1)))
        side = str(before.get("side", "") or "").lower()
        target_notional, rebalance_margin = _leverage_rebalance_target_from_position(before, target)
        if target_notional <= 0.0:
            raise ValueError("Could not determine current position notional and leverage for leverage rebalance.")
        min_trade_notional = _configured_min_trade_notional_usd()
        effective_target_notional = _effective_open_notional_after_size_rounding(
            self.reader,
            symbol,
            target_notional,
            _as_float(before.get("mid_price"), 0.0),
        )
        if min_trade_notional > 0.0 and (
            0.0 < target_notional < min_trade_notional or 0.0 <= effective_target_notional < min_trade_notional
        ):
            return {
                "accepted": False,
                "stage": "target_below_min_trade_notional",
                "symbol": symbol,
                "target_leverage": target,
                "rebalance_margin_usd": rebalance_margin,
                "target_notional_usd": target_notional,
                "effective_target_notional_usd": effective_target_notional,
                "min_trade_notional_usd": min_trade_notional,
                "position_before": before_view,
                "message": (
                    "Leverage rebalance skipped before closing because the target reopen "
                    "notional is below the exchange minimum trade notional."
                ),
            }
        realized_pnl = _as_float(before.get("unrealized_pnl"), 0.0)

        executor = self._executor(symbol)
        close_result = executor.close_position(
            side,
            "web_leverage_rebalance_close",
            "web_trade",
            position_before=before,
        )
        if not _result_accepted(close_result):
            return {"accepted": False, "stage": "close_failed", "close_result": close_result, "position_before": before}

        leverage_result = executor.apply_requested_leverage(target)
        flat_snapshot = _synthetic_flat_snapshot_after_close(before)
        open_result = executor.execute_position_target(
            target_side=side,
            target_notional_usd=target_notional,
            requested_leverage=target,
            reason="web_leverage_rebalance_open",
            plan_name="web_trade",
            position_before=flat_snapshot,
            execution_mid_price=_as_float(before.get("mid_price"), 0.0),
        )
        if not _result_accepted(open_result):
            self.ledger.record_hidden_rebalance(
                symbol=symbol,
                realized_pnl_usd=realized_pnl,
                target_leverage=target,
                target_notional_usd=target_notional,
            )
            return {
                "accepted": False,
                "stage": "reopen_failed",
                "close_result": close_result,
                "leverage_result": leverage_result,
                "open_result": open_result,
            }

        self.ledger.record_hidden_rebalance(
            symbol=symbol,
            realized_pnl_usd=realized_pnl,
            target_leverage=target,
            target_notional_usd=target_notional,
        )
        expected_after = _expected_position_after_rebalance(
            before,
            open_result,
            side=side,
            target_leverage=target,
            target_notional_usd=target_notional,
            rebalance_margin_usd=rebalance_margin,
        )
        fresh_liquidation_price = _fresh_liquidation_price_after_rebalance(
            self._position_snapshot(symbol),
            side=side,
            target_leverage=target,
        )
        if fresh_liquidation_price > 0.0:
            expected_after["liquidation_price"] = fresh_liquidation_price
        after = self.ledger.overlay_position(expected_after)
        return {
            "accepted": True,
            "stage": "complete",
            "symbol": symbol,
            "target_leverage": target,
            "rebalance_margin_usd": rebalance_margin,
            "target_notional_usd": target_notional,
            "position_before": before_view,
            "position": after,
            "close_result": close_result,
            "leverage_result": leverage_result,
            "open_result": open_result,
        }

    def update_isolated_margin(
        self,
        symbol: str,
        direction: str,
        amount_usd: float,
        *,
        safety_buffer_usd: float | None = None,
    ) -> Dict[str, Any]:
        symbol = self._symbol(symbol)
        snapshot = self._position_snapshot(symbol)
        self.ledger.overlay_position(snapshot)
        limits = calculate_margin_limits(snapshot, safety_buffer_usd=safety_buffer_usd)
        if not limits.get("enabled"):
            raise ValueError(str(limits.get("reason") or "margin_adjustment_disabled"))
        requested = max(0.0, _as_float(amount_usd, 0.0))
        raw_direction = str(direction or "").lower().strip()
        if raw_direction == "add":
            applied = min(requested, _as_float(limits.get("max_add_margin_usd"), 0.0))
        elif raw_direction == "remove":
            applied = -min(requested, _as_float(limits.get("max_remove_margin_usd"), 0.0))
        else:
            raise ValueError("direction must be add or remove")
        if abs(applied) <= 0.0:
            raise ValueError("Requested margin amount is not available.")

        executor = self._executor(symbol)
        exchange_result: Dict[str, Any]
        if not bool(getattr(executor, "enabled", False)):
            exchange_result = {
                "status": "dry_run",
                "message": "ENABLE_LIVE_TRADING=false, so isolated margin update is dry-run only.",
            }
        else:
            executor._ensure_exchange()
            exchange_result = executor._exchange.update_isolated_margin(applied, symbol)
        if isinstance(exchange_result, dict) and str(exchange_result.get("status", "") or "").lower() == "err":
            return {"accepted": False, "exchange": exchange_result, "limits": limits}
        self.ledger.apply_margin_delta(symbol, applied)
        position = self.ledger.overlay_position(self._position_snapshot(symbol))
        return {
            "accepted": True,
            "symbol": symbol,
            "direction": raw_direction,
            "requested_amount_usd": requested,
            "applied_amount_usd": applied,
            "limits": limits,
            "exchange": exchange_result,
            "position": position,
        }
