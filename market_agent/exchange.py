import base64
import os
import time
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal, InvalidOperation, ROUND_DOWN, ROUND_UP
from typing import Any, Dict, List, Optional, Tuple

import requests
from requests.adapters import HTTPAdapter
from market_agent.charting import (
    _build_chart_debug_record,
    _build_chart_image_timeframe_specs,
    _build_chart_summary_record,
    _sorted_candles,
    render_candles_chart_png,
)
from market_agent.constants import (
    DEFAULT_CHART_IMAGE_DETAIL,
    DEFAULT_CHART_IMAGE_HEIGHT_PX,
    DEFAULT_CHART_IMAGE_WIDTH_PX,
    MANAGEMENT_EXPOSURE_ACTION_VALUES,
)
from market_agent.models import ManagementDecision, StrategyDecision
from market_agent.positions import normalize_spot_user_state, snapshot_has_open_position
from market_agent.symbols import base_url, canonicalize_execution_symbol, normalize_candidate_key, split_execution_symbol
from market_agent.utils import clamp_int, format_query_amount, safe_float


def active_chart_image_timeframe_specs() -> Tuple[Dict[str, Any], ...]:
    return _build_chart_image_timeframe_specs(
        os.getenv("OPENAI_ACTIVE_CHART_IMAGE_TIMEFRAMES", ""),
        ("1m", "5m", "15m"),
    )


def _cumulative_funding_fields(position: Dict[str, Any]) -> Dict[str, float]:
    raw = (position or {}).get("cumFunding")
    if not isinstance(raw, dict):
        return {}
    mapping = {
        "allTime": "funding_all_time_usd",
        "sinceOpen": "funding_since_open_usd",
        "sinceChange": "funding_since_change_usd",
    }
    return {
        output_key: safe_float(raw.get(input_key), 0.0) or 0.0
        for input_key, output_key in mapping.items()
        if input_key in raw
    }


class HyperliquidRestReader:
    def __init__(self):
        self.network = (os.getenv("HYPERLIQUID_NETWORK", "mainnet") or "").strip().lower()
        self.account_address = (os.getenv("HL_ACCOUNT_ADDRESS", "") or "").strip()
        if not self.account_address:
            raise RuntimeError("HL_ACCOUNT_ADDRESS 未设置")
        self.base = base_url(self.network)
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        pool_connections = max(10, int(os.getenv("HYPERLIQUID_HTTP_POOL_CONNECTIONS", "16")))
        pool_maxsize = max(pool_connections, int(os.getenv("HYPERLIQUID_HTTP_POOL_MAXSIZE", "64")))
        http_adapter = HTTPAdapter(pool_connections=pool_connections, pool_maxsize=pool_maxsize)
        self.session.mount("https://", http_adapter)
        self.session.mount("http://", http_adapter)
        self.info_timeout_seconds = max(1.0, float(os.getenv("HYPERLIQUID_INFO_TIMEOUT_SECONDS", "15")))
        self.info_max_retries = max(0, int(os.getenv("HYPERLIQUID_INFO_MAX_RETRIES", "3")))
        self.info_retry_delay_seconds = max(0.1, float(os.getenv("HYPERLIQUID_INFO_RETRY_DELAY_SECONDS", "1.5")))
        self._meta_cache_by_dex: Dict[str, dict] = {}
        self._perp_dexs_cache: Optional[List[dict]] = None
        self._meta_asset_context_cache_by_dex: Dict[str, Tuple[dict, List[dict]]] = {}
        self._spot_meta_cache: Optional[dict] = None
        self._safe_spot_meta_cache: Optional[dict] = None
        self._mids_cache_by_dex: Dict[str, Tuple[float, Dict[str, str]]] = {}
        self._market_catalog_cache: Optional[Dict[str, dict]] = None
        self._market_alias_index_cache: Optional[Dict[str, str]] = None
        self._info_client: Any = None
        self._ws_info_client: Any = None
        self._market_chart_cache: Dict[str, Tuple[float, Dict[str, Any]]] = {}
        self._user_fee_rates_cache: Tuple[float, Dict[str, Any]] = (0.0, {})

    @staticmethod
    def _payload_with_dex(payload_type: str, dex: str = "", **extra: Any) -> Dict[str, Any]:
        payload: Dict[str, Any] = {"type": payload_type, **extra}
        if str(dex or "").strip():
            payload["dex"] = str(dex).strip().lower()
        return payload

    def post_info(self, payload: dict) -> Any:
        attempts = max(1, self.info_max_retries + 1)
        payload_type = str(payload.get("type", "unknown"))
        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                resp = self.session.post(
                    self.base.rstrip("/") + "/info",
                    json=payload,
                    timeout=self.info_timeout_seconds,
                )
                resp.raise_for_status()
                return resp.json()
            except KeyboardInterrupt:
                raise
            except requests.exceptions.HTTPError as exc:
                last_error = exc
                status_code = exc.response.status_code if exc.response is not None else None
                should_retry = status_code is None or status_code >= 500 or status_code == 429
            except (requests.exceptions.RequestException, ValueError) as exc:
                last_error = exc
                should_retry = True
            if not should_retry or attempt >= attempts:
                raise last_error if last_error is not None else RuntimeError(f"Hyperliquid info request failed: {payload_type}")
            delay = self.info_retry_delay_seconds * attempt
            print(
                f"[warn] Hyperliquid /info retry attempt={attempt}/{attempts - 1} "
                f"type={payload_type} delay={format_query_amount(delay)}s error={last_error}"
            )
            time.sleep(delay)
        raise RuntimeError(f"Hyperliquid info request exhausted retries: {payload_type}")

    def get_perp_dexs(self) -> List[dict]:
        if self._perp_dexs_cache is None:
            raw = self.post_info({"type": "perpDexs"})
            self._perp_dexs_cache = [item for item in (raw or []) if isinstance(item, dict) and str(item.get("name", "")).strip()]
        return list(self._perp_dexs_cache)

    def list_perp_dex_names(self) -> List[str]:
        return [str(item.get("name", "")).strip().lower() for item in self.get_perp_dexs() if str(item.get("name", "")).strip()]

    def get_meta(self) -> dict:
        return self.get_perp_meta("")

    def get_perp_meta(self, dex: str = "") -> dict:
        key = str(dex or "").strip().lower()
        if key not in self._meta_cache_by_dex:
            self._meta_cache_by_dex[key] = self.post_info(self._payload_with_dex("meta", key))
        return self._meta_cache_by_dex[key]

    def get_perp_meta_and_asset_contexts(self, dex: str = "") -> Tuple[dict, List[dict]]:
        key = str(dex or "").strip().lower()
        cache = getattr(self, "_meta_asset_context_cache_by_dex", None)
        if cache is None:
            cache = {}
            self._meta_asset_context_cache_by_dex = cache
        if key not in cache:
            response = self.post_info(self._payload_with_dex("metaAndAssetCtxs", key))
            meta = response[0] if isinstance(response, list) and len(response) >= 1 and isinstance(response[0], dict) else {}
            contexts = response[1] if isinstance(response, list) and len(response) >= 2 and isinstance(response[1], list) else []
            if not meta:
                meta = self.get_perp_meta(key)
            cache[key] = (meta, [ctx for ctx in contexts if isinstance(ctx, dict)])
        return cache[key]

    def _iter_market_specs(self) -> List[dict]:
        items: List[dict] = []
        for dex_name in [""] + self.list_perp_dex_names():
            try:
                meta, contexts = self.get_perp_meta_and_asset_contexts(dex_name)
            except Exception:
                meta, contexts = self.get_perp_meta(dex_name), []
            for idx, entry in enumerate(meta.get("universe", []) or []):
                raw_name = canonicalize_execution_symbol(str(entry.get("name", "") or "").strip())
                if not raw_name:
                    continue
                dex, market_name = split_execution_symbol(raw_name)
                if dex_name and not dex:
                    dex = dex_name
                    raw_name = canonicalize_execution_symbol(f"{dex}:{market_name}")
                market_name = market_name or raw_name
                items.append(
                    {
                        "execution_symbol": raw_name,
                        "dex": dex,
                        "market_name": market_name,
                        "display_name": f"{raw_name}-USDC",
                        "symbol": raw_name,
                        "sz_decimals": int(entry.get("szDecimals", 0) or 0),
                        "max_leverage": int(entry.get("maxLeverage", 0) or 0),
                        "only_isolated": bool(entry.get("onlyIsolated", False)),
                    }
                )
                ctx = contexts[idx] if idx < len(contexts) and isinstance(contexts[idx], dict) else {}
                if ctx:
                    items[-1].update(
                        {
                            "mid_price": safe_float(ctx.get("midPx"), 0.0) or 0.0,
                            "mark_price": safe_float(ctx.get("markPx"), 0.0) or 0.0,
                            "prev_day_price": safe_float(ctx.get("prevDayPx"), 0.0) or 0.0,
                            "day_volume_usd": safe_float(ctx.get("dayNtlVlm"), 0.0) or 0.0,
                        }
                    )
        return items

    def get_market_catalog(self) -> Dict[str, dict]:
        if self._market_catalog_cache is None:
            self._market_catalog_cache = {
                canonicalize_execution_symbol(str(item.get("execution_symbol", "") or "")): item
                for item in self._iter_market_specs()
            }
        return dict(self._market_catalog_cache)

    def _build_market_alias_index(self) -> Dict[str, str]:
        counts: Dict[str, int] = {}
        alias_targets: Dict[str, str] = {}
        for item in self.get_market_catalog().values():
            execution_symbol = canonicalize_execution_symbol(str(item.get("execution_symbol", "") or "").strip())
            if not execution_symbol:
                continue
            aliases = {
                execution_symbol,
                normalize_candidate_key(execution_symbol),
                str(item.get("market_name", "") or "").strip().upper(),
                normalize_candidate_key(item.get("market_name", "")),
                str(item.get("display_name", "") or "").strip().upper(),
                normalize_candidate_key(item.get("display_name", "")),
            }
            for alias in {alias for alias in aliases if alias}:
                counts[alias] = counts.get(alias, 0) + 1
                alias_targets.setdefault(alias, execution_symbol)
        return {alias: alias_targets[alias] for alias, count in counts.items() if count == 1}

    def resolve_execution_symbol(self, raw_symbol: str) -> str:
        token = str(raw_symbol or "").strip()
        if not token:
            return ""
        if self._market_alias_index_cache is None:
            self._market_alias_index_cache = self._build_market_alias_index()
        exact = canonicalize_execution_symbol(token)
        resolved = self._market_alias_index_cache.get(exact) or self._market_alias_index_cache.get(exact.upper())
        if resolved:
            return resolved
        normalized = normalize_candidate_key(token)
        return self._market_alias_index_cache.get(normalized, "")

    def get_mids(self, ttl_seconds: float = 2.0, dex: str = "") -> Dict[str, str]:
        key = str(dex or "").strip().lower()
        cache_by_dex = getattr(self, "_mids_cache_by_dex", None)
        if cache_by_dex is None:
            cache_by_dex = {}
            legacy_ts, legacy_mids = getattr(self, "_mids_cache", (0.0, {}))
            if legacy_mids:
                cache_by_dex[""] = (legacy_ts, legacy_mids)
            self._mids_cache_by_dex = cache_by_dex
        ts, mids = cache_by_dex.get(key, (0.0, {}))
        now = time.time()
        if now - ts > ttl_seconds or not mids:
            try:
                mids = self.post_info(self._payload_with_dex("allMids", key))
                normalized_mids = {}
                for name, value in (mids or {}).items():
                    symbol = canonicalize_execution_symbol(str(name or "").strip())
                    if not symbol:
                        continue
                    dex, market_name = split_execution_symbol(symbol)
                    if key and not dex:
                        symbol = canonicalize_execution_symbol(f"{key}:{market_name}")
                    normalized_mids[symbol] = value
                mids = normalized_mids
                self._mids_cache_by_dex[key] = (now, mids)
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                if mids:
                    age = max(0.0, now - ts)
                    print(
                        f"[warn] allMids refresh failed; using stale cache age={format_query_amount(age)}s error={exc}"
                    )
                    return mids
                raise
        return mids

    def get_spot_meta(self) -> dict:
        if self._spot_meta_cache is None:
            self._spot_meta_cache = self.post_info({"type": "spotMeta"})
        return self._spot_meta_cache

    def get_safe_spot_meta(self) -> dict:
        if self._safe_spot_meta_cache is not None:
            return self._safe_spot_meta_cache
        raw = self.get_spot_meta()
        tokens = list(raw.get("tokens", []) or [])
        max_index = len(tokens) - 1
        safe_universe: List[dict] = []
        dropped = 0
        for item in raw.get("universe", []) or []:
            pair = item.get("tokens", []) or []
            if len(pair) != 2:
                dropped += 1
                continue
            base_idx, quote_idx = pair
            if not isinstance(base_idx, int) or not isinstance(quote_idx, int):
                dropped += 1
                continue
            if base_idx < 0 or quote_idx < 0 or base_idx > max_index or quote_idx > max_index:
                dropped += 1
                continue
            safe_universe.append(item)
        self._safe_spot_meta_cache = {"tokens": tokens, "universe": safe_universe}
        if dropped > 0:
            self._safe_spot_meta_cache["_dropped_invalid_universe_entries"] = dropped
        return self._safe_spot_meta_cache

    def get_mid_price(self, symbol: str) -> Optional[float]:
        execution_symbol = self.resolve_execution_symbol(symbol) or canonicalize_execution_symbol(symbol)
        dex, _ = split_execution_symbol(execution_symbol)
        return safe_float(self.get_mids(dex=dex).get(execution_symbol), None)

    def get_ws_symbol(self, symbol: str) -> str:
        execution_symbol = self.resolve_execution_symbol(symbol) or canonicalize_execution_symbol(symbol)
        execution_symbol = canonicalize_execution_symbol(execution_symbol)
        if not execution_symbol:
            return ""
        try:
            return str(self.get_info_client().name_to_coin.get(execution_symbol, execution_symbol) or execution_symbol)
        except Exception:
            return execution_symbol

    def get_market_asset_context(self, symbol: str) -> Dict[str, Any]:
        execution_symbol = self.resolve_execution_symbol(symbol) or canonicalize_execution_symbol(symbol)
        execution_symbol = canonicalize_execution_symbol(execution_symbol)
        if not execution_symbol:
            return {}
        dex, market_name = split_execution_symbol(execution_symbol)
        meta, ctxs = self.get_perp_meta_and_asset_contexts(dex)
        universe = list(meta.get("universe", []) or [])
        targets = {
            execution_symbol,
            canonicalize_execution_symbol(market_name or execution_symbol),
            canonicalize_execution_symbol(f"{dex}:{market_name}") if dex and market_name else "",
        }
        for idx, asset in enumerate(universe):
            if idx >= len(ctxs) or not isinstance(asset, dict) or not isinstance(ctxs[idx], dict):
                continue
            asset_name = canonicalize_execution_symbol(str(asset.get("name", "") or "").strip())
            if asset_name not in targets:
                continue
            payload = dict(ctxs[idx])
            payload.update(
                {
                    "source": "metaAndAssetCtxs",
                    "execution_symbol": execution_symbol,
                    "dex": dex,
                    "market_name": market_name,
                    "asset_name": asset_name,
                    "asset_index": idx,
                }
            )
            return payload
        return {}

    def align_price_to_wire(self, symbol: str, price: float) -> float:
        raw = float(price or 0.0)
        if raw <= 0.0:
            return 0.0
        execution_symbol = self.resolve_execution_symbol(symbol) or canonicalize_execution_symbol(symbol)
        is_spot = False
        sz_decimals = 0
        try:
            info = self.get_info_client()
            coin = str(info.name_to_coin.get(execution_symbol, execution_symbol) or execution_symbol)
            asset = int(info.coin_to_asset[coin])
            sz_decimals = int(info.asset_to_sz_decimals[asset] or 0)
            is_spot = 10000 <= asset < 110000
        except Exception:
            sz_decimals = int(self.get_market_spec(execution_symbol).get("sz_decimals", 0) or 0)
        decimals = max(0, (8 if is_spot else 6) - sz_decimals)
        from hyperliquid.utils.signing import float_to_wire

        base = round(float(f"{raw:.5g}"), decimals)
        for places in range(decimals, -1, -1):
            candidate = round(base, places)
            try:
                float_to_wire(candidate)
                return candidate
            except ValueError:
                continue
        raise ValueError("Could not align price to Hyperliquid wire format", execution_symbol, raw)

    def get_sz_decimals(self, symbol: str) -> int:
        return int(self.get_market_spec(symbol).get("sz_decimals", 0) or 0)

    def list_perp_symbols(self) -> List[str]:
        return sorted(self.get_market_catalog().keys())

    def get_market_spec(self, symbol: str) -> dict:
        execution_symbol = self.resolve_execution_symbol(symbol) or canonicalize_execution_symbol(symbol)
        execution_symbol = canonicalize_execution_symbol(execution_symbol)
        spec = self.get_market_catalog().get(execution_symbol)
        if spec is None:
            raise KeyError(f"Could not find {symbol} in Hyperliquid perp meta universe")
        return dict(spec)

    def get_info_client(self) -> Any:
        if self._info_client is None:
            from hyperliquid.info import Info

            self._info_client = Info(
                base_url=self.base,
                skip_ws=True,
                meta=self.get_meta(),
                spot_meta=self.get_safe_spot_meta(),
                perp_dexs=[""] + self.list_perp_dex_names(),
                timeout=self.info_timeout_seconds,
            )
        return self._info_client

    def get_ws_info_client(self) -> Any:
        if self._ws_info_client is None:
            from hyperliquid.info import Info

            self._ws_info_client = Info(
                base_url=self.base,
                skip_ws=False,
                meta=self.get_meta(),
                spot_meta=self.get_safe_spot_meta(),
                perp_dexs=[""] + self.list_perp_dex_names(),
                timeout=self.info_timeout_seconds,
            )
        return self._ws_info_client

    def subscribe_user_fills(self, address: str, callback: Any) -> int:
        user = str(address or self.account_address or "").strip()
        if not user:
            raise RuntimeError("No user address configured for userFills websocket subscription")
        return int(self.get_ws_info_client().subscribe({"type": "userFills", "user": user}, callback))

    def unsubscribe_user_fills(self, address: str, subscription_id: int) -> bool:
        if self._ws_info_client is None:
            return False
        user = str(address or self.account_address or "").strip()
        if not user:
            return False
        return bool(self.get_ws_info_client().unsubscribe({"type": "userFills", "user": user}, int(subscription_id)))

    def disconnect_ws(self) -> None:
        client = self._ws_info_client
        self._ws_info_client = None
        if client is None:
            return
        try:
            client.disconnect_websocket()
        except Exception:
            pass

    def user_fills_ws_is_healthy(self) -> bool:
        client = self._ws_info_client
        manager = getattr(client, "ws_manager", None) if client is not None else None
        if manager is None:
            return False
        ws = getattr(manager, "ws", None)
        return bool(
            getattr(manager, "ws_ready", False)
            and manager.is_alive()
            and bool(getattr(ws, "keep_running", False))
        )

    def get_user_fills_by_time(
        self,
        address: str,
        start_time_ms: int,
        end_time_ms: Optional[int] = None,
        *,
        aggregate_by_time: bool = False,
    ) -> List[dict]:
        user = str(address or self.account_address or "").strip()
        if not user:
            raise RuntimeError("No user address configured for userFillsByTime query")
        return list(
            self.get_info_client().user_fills_by_time(
                user,
                int(start_time_ms),
                int(end_time_ms) if end_time_ms is not None else None,
                bool(aggregate_by_time),
            )
            or []
        )

    def get_user_funding_history(
        self,
        address: str,
        start_time_ms: int,
        end_time_ms: Optional[int] = None,
    ) -> List[dict]:
        user = str(address or self.account_address or "").strip()
        if not user:
            raise RuntimeError("No user address configured for userFunding query")
        return list(
            self.get_info_client().user_funding_history(
                user,
                int(start_time_ms),
                int(end_time_ms) if end_time_ms is not None else None,
            )
            or []
        )

    def get_historical_orders(self, address: Optional[str] = None) -> List[dict]:
        user = str(address or self.account_address or "").strip()
        if not user:
            raise RuntimeError("No user address configured for historicalOrders query")
        return list(self.get_info_client().historical_orders(user) or [])

    def get_frontend_open_orders(self, symbol: Optional[str] = None) -> List[dict]:
        orders: List[dict] = []
        target_symbol = canonicalize_execution_symbol(symbol or "") if symbol else ""
        for dex_name in [""] + self.list_perp_dex_names():
            raw = self.get_info_client().frontend_open_orders(self.account_address, dex=dex_name)
            for item in list(raw or []):
                if not isinstance(item, dict):
                    continue
                coin = canonicalize_execution_symbol(item.get("coin") or "")
                if target_symbol and coin != target_symbol:
                    continue
                orders.append(dict(item))
        return orders

    def get_user_fee_rates(self, ttl_seconds: float = 300.0) -> Dict[str, Any]:
        now = time.time()
        cached_ts, cached_payload = self._user_fee_rates_cache
        if cached_payload and now - cached_ts <= ttl_seconds:
            return dict(cached_payload)
        try:
            raw = self.post_info({"type": "userFees", "user": self.account_address})
            taker_fee_rate = max(0.0, safe_float((raw or {}).get("userCrossRate"), 0.0) or 0.0)
            maker_fee_rate = max(0.0, safe_float((raw or {}).get("userAddRate"), 0.0) or 0.0)
            payload = {
                "known": bool(taker_fee_rate > 0 or maker_fee_rate > 0),
                "taker_fee_rate": taker_fee_rate,
                "maker_fee_rate": maker_fee_rate,
                "source": "hyperliquid_userFees",
            }
            self._user_fee_rates_cache = (now, payload)
            return dict(payload)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            if cached_payload:
                age = max(0.0, now - cached_ts)
                print(
                    f"[warn] userFees refresh failed; using stale cache age={format_query_amount(age)}s error={exc}"
                )
                return dict(cached_payload)
            return {
                "known": False,
                "taker_fee_rate": 0.0,
                "maker_fee_rate": 0.0,
                "source": "unavailable",
                "error": str(exc),
            }

    def get_candles_snapshot(self, symbol: str, interval: str, start_ms: int, end_ms: int) -> List[dict]:
        execution_symbol = self.resolve_execution_symbol(symbol) or canonicalize_execution_symbol(symbol)
        try:
            data = self.get_info_client().candles_snapshot(execution_symbol, interval, int(start_ms), int(end_ms))
        except Exception:
            return []
        return [item for item in (data or []) if isinstance(item, dict)]

    def get_l2_book_snapshot(self, symbol: str) -> dict:
        execution_symbol = self.resolve_execution_symbol(symbol) or canonicalize_execution_symbol(symbol)
        execution_symbol = canonicalize_execution_symbol(execution_symbol)
        dex, market_name = split_execution_symbol(execution_symbol)
        coin = market_name or execution_symbol
        try:
            data = self.post_info(self._payload_with_dex("l2Book", dex, coin=coin))
        except Exception:
            return {}
        return dict(data) if isinstance(data, dict) else {}

    def get_funding_history(self, symbol: str, start_ms: int, end_ms: int) -> List[dict]:
        execution_symbol = self.resolve_execution_symbol(symbol) or canonicalize_execution_symbol(symbol)
        try:
            data = self.get_info_client().funding_history(execution_symbol, int(start_ms), int(end_ms))
        except Exception:
            return []
        return [item for item in (data or []) if isinstance(item, dict)]

    def get_market_chart_context(
        self,
        symbol: str,
        display_name: str = "",
        ttl_seconds: float = 120.0,
        timeframe_specs: Optional[Tuple[Dict[str, Any], ...]] = None,
    ) -> Dict[str, Any]:
        execution_symbol = self.resolve_execution_symbol(symbol) or canonicalize_execution_symbol(symbol)
        resolved_specs: List[Tuple[str, float]] = []
        default_timeframe_specs = active_chart_image_timeframe_specs()
        for spec in list(timeframe_specs or default_timeframe_specs):
            if not isinstance(spec, dict):
                continue
            timeframe = str(spec.get("timeframe", "") or "").strip()
            window_hours = safe_float(spec.get("window_hours"), None)
            if timeframe and window_hours and window_hours > 0:
                resolved_specs.append((timeframe, float(window_hours)))
        if not resolved_specs:
            resolved_specs = [(str(item["timeframe"]), float(item["window_hours"])) for item in default_timeframe_specs]
        specs_key = tuple(resolved_specs)
        cache_key = (execution_symbol, specs_key)
        cache = getattr(self, "_market_chart_cache", {})
        now = time.time()
        cached_ts, cached_payload = cache.get(cache_key, (0.0, {}))
        if cached_payload and now - cached_ts <= ttl_seconds:
            payload = dict(cached_payload)
            payload["input_images"] = [dict(item) for item in (cached_payload.get("input_images") or []) if isinstance(item, dict)]
            payload["debug_images"] = [dict(item) for item in (cached_payload.get("debug_images") or []) if isinstance(item, dict)]
            payload["chart_summaries"] = [dict(item) for item in (cached_payload.get("chart_summaries") or []) if isinstance(item, dict)]
            return payload

        width_px = clamp_int(os.getenv("OPENAI_CHART_IMAGE_WIDTH_PX", str(DEFAULT_CHART_IMAGE_WIDTH_PX)), 320, 1400, DEFAULT_CHART_IMAGE_WIDTH_PX)
        height_px = clamp_int(os.getenv("OPENAI_CHART_IMAGE_HEIGHT_PX", str(DEFAULT_CHART_IMAGE_HEIGHT_PX)), 240, 900, DEFAULT_CHART_IMAGE_HEIGHT_PX)
        detail = str(os.getenv("OPENAI_CHART_IMAGE_DETAIL", DEFAULT_CHART_IMAGE_DETAIL) or DEFAULT_CHART_IMAGE_DETAIL).strip().lower()
        if detail not in {"low", "high", "auto", "original"}:
            detail = DEFAULT_CHART_IMAGE_DETAIL
        input_images: List[Dict[str, Any]] = []
        debug_images: List[Dict[str, Any]] = []
        chart_summaries: List[Dict[str, Any]] = []
        now_ms = int(now * 1000)
        for timeframe, window_hours in specs_key:
            start_ms = now_ms - int(window_hours * 60 * 60 * 1000)
            candles = self.get_candles_snapshot(execution_symbol, timeframe, start_ms, now_ms)
            sorted_candles = _sorted_candles(candles)
            current_price = safe_float(sorted_candles[-1].get("c"), None) if sorted_candles else None
            png_bytes = render_candles_chart_png(
                candles=sorted_candles,
                symbol_label=str(display_name or execution_symbol),
                timeframe=timeframe,
                window_hours=window_hours,
                width_px=width_px,
                height_px=height_px,
                current_price=current_price,
            )
            if not png_bytes:
                continue
            data_url = "data:image/png;base64," + base64.b64encode(png_bytes).decode("ascii")
            input_images.append({"type": "input_image", "detail": detail, "image_url": data_url})
            debug_images.append(
                _build_chart_debug_record(
                    timeframe=timeframe,
                    window_hours=window_hours,
                    width_px=width_px,
                    height_px=height_px,
                    detail=detail,
                    candle_count=len(sorted_candles),
                    image_bytes=len(png_bytes),
                    data_url_chars=len(data_url),
                )
            )
            chart_summaries.append(
                _build_chart_summary_record(
                    candles=sorted_candles,
                    timeframe=timeframe,
                    window_hours=window_hours,
                    current_price=current_price,
                )
            )
        payload = {
            "display_name": str(display_name or execution_symbol),
            "execution_symbol": execution_symbol,
            "detail": detail,
            "input_images": input_images,
            "debug_images": debug_images,
            "chart_summaries": chart_summaries,
            "image_count": len(debug_images),
            "note": (
                "这些图表截图只用于让模型理解结构和形态；精确价格、资金费率、杠杆和风控约束仍以本地数值上下文为准。"
            ),
        }
        cache[cache_key] = (now, payload)
        self._market_chart_cache = cache
        return {
            **payload,
            "input_images": [dict(item) for item in input_images],
            "debug_images": [dict(item) for item in debug_images],
            "chart_summaries": [dict(item) for item in chart_summaries],
        }

    def get_all_positions(self) -> dict:
        spot_user_state = self.post_info({"type": "spotClearinghouseState", "user": self.account_address})
        spot_summary = normalize_spot_user_state(spot_user_state)
        try:
            user_abstraction_raw = self.post_info({"type": "userAbstraction", "user": self.account_address})
        except Exception:
            user_abstraction_raw = None
        user_abstraction = str(user_abstraction_raw or "").strip()
        user_states: List[Tuple[str, dict]] = []
        for dex_name in [""] + self.list_perp_dex_names():
            state = self.post_info(self._payload_with_dex("clearinghouseState", dex_name, user=self.account_address))
            user_states.append((dex_name, state if isinstance(state, dict) else {}))
        withdrawable_usd = sum(safe_float(state.get("withdrawable"), 0.0) or 0.0 for _, state in user_states)
        spot_usdc_total = float(spot_summary.get("spot_usdc_total", 0.0) or 0.0)
        spot_available_usdc = float(spot_summary.get("spot_available_usdc", 0.0) or 0.0)
        universe_specs = self.get_market_catalog()
        mids_by_dex = {dex_name: self.get_mids(dex=dex_name) for dex_name in [""] + self.list_perp_dex_names()}
        positions: List[dict] = []
        total_notional = 0.0
        total_margin_used_usd = 0.0
        for dex_name, user_state in user_states:
            mids = mids_by_dex.get(dex_name, {})
            for entry in user_state.get("assetPositions", []) or []:
                position = entry.get("position", {}) or {}
                coin = canonicalize_execution_symbol(position.get("coin") or "")
                if not coin:
                    continue
                szi = safe_float(position.get("szi", 0.0), 0.0) or 0.0
                entry_px = safe_float(position.get("entryPx", 0.0), 0.0) or 0.0
                mid = safe_float(mids.get(coin), None)
                side = "flat"
                if szi > 0:
                    side = "long"
                elif szi < 0:
                    side = "short"
                notional = abs(szi) * (mid if mid is not None else entry_px)
                total_notional += notional
                spec = universe_specs.get(coin, {})
                margin_used = safe_float(position.get("marginUsed", 0.0), 0.0) or 0.0
                total_margin_used_usd += margin_used
                positions.append(
                    {
                        "symbol": coin,
                        "side": side,
                        "size": szi,
                        "entry_price": entry_px,
                        "mid_price": mid,
                        "notional_usd": notional,
                        "unrealized_pnl": safe_float(position.get("unrealizedPnl", 0.0), 0.0) or 0.0,
                        "return_on_equity": safe_float(position.get("returnOnEquity", 0.0), 0.0) or 0.0,
                        "leverage": safe_float((position.get("leverage") or {}).get("value"), 0.0) or 0.0,
                        "max_leverage": int(spec.get("max_leverage", 0) or 0),
                        "only_isolated": bool(spec.get("only_isolated", False)),
                        "liquidation_price": safe_float(position.get("liquidationPx", 0.0), 0.0) or 0.0,
                        "margin_used": margin_used,
                        **_cumulative_funding_fields(position),
                    }
                )
        positions.sort(key=lambda x: abs(float(x.get("notional_usd", 0.0) or 0.0)), reverse=True)
        if user_abstraction in {"unifiedAccount", "portfolioMargin"}:
            perp_account_equity_usd = spot_usdc_total
            isolated_margin_basis_usd = spot_usdc_total
            cross_margin_basis_usd = spot_usdc_total
            isolated_available_margin_usd = 0.0
            cross_available_margin_usd = 0.0
            account_equity_usd = spot_usdc_total
        else:
            perp_account_equity_usd = total_margin_used_usd + max(0.0, withdrawable_usd)
            isolated_margin_basis_usd = perp_account_equity_usd
            cross_margin_basis_usd = perp_account_equity_usd
            isolated_available_margin_usd = 0.0
            cross_available_margin_usd = max(0.0, withdrawable_usd)
            account_equity_usd = perp_account_equity_usd + spot_usdc_total
        available_margin_usd = cross_available_margin_usd + spot_available_usdc
        withdrawable_total_usd = withdrawable_usd + spot_available_usdc
        if spot_available_usdc > 0 and withdrawable_usd > 0:
            remaining_capital_source = "spot_available_plus_perp_withdrawable"
        elif spot_available_usdc > 0:
            remaining_capital_source = "spot_available_after_maintenance"
        elif withdrawable_usd > 0:
            remaining_capital_source = "withdrawable"
        else:
            remaining_capital_source = "used_plus_available"
        remaining_capital_usd = (
            withdrawable_total_usd if withdrawable_total_usd > 0 else available_margin_usd
        )
        result = {
            "known": True,
            "account_address": self.account_address,
            "network": self.network,
            "user_abstraction": user_abstraction,
            "spot_summary": spot_summary,
            "perp_account_equity_usd": perp_account_equity_usd,
            "isolated_margin_basis_usd": isolated_margin_basis_usd,
            "cross_margin_basis_usd": cross_margin_basis_usd,
            "spot_usdc_total": spot_usdc_total,
            "spot_usdc_hold": float(spot_summary.get("spot_usdc_hold", 0.0) or 0.0),
            "spot_available_usdc": spot_available_usdc,
            "account_equity_usd": account_equity_usd,
            "total_margin_used_usd": float(total_margin_used_usd or 0.0),
            "available_margin_usd": available_margin_usd,
            "withdrawable_usd": withdrawable_total_usd,
            "remaining_capital_usd": float(remaining_capital_usd or 0.0),
            "remaining_capital_source": remaining_capital_source,
            "positions": positions,
            "positions_count": len(positions),
            "total_notional_usd": total_notional,
        }
        if user_abstraction not in {"unifiedAccount", "portfolioMargin"}:
            result.update(
                {
                    "isolated_available_margin_usd": isolated_available_margin_usd,
                    "cross_available_margin_usd": cross_available_margin_usd,
                    "perp_withdrawable_usd": withdrawable_usd,
                }
            )
        return result

    def get_selected_symbol_position_context(self, symbol: str) -> dict:
        execution_symbol = self.resolve_execution_symbol(symbol) or canonicalize_execution_symbol(symbol)
        execution_symbol = canonicalize_execution_symbol(execution_symbol)
        if not execution_symbol:
            raise ValueError("symbol is required for selected symbol position context")
        dex, market_name = split_execution_symbol(execution_symbol)
        dex_names = [""] + self.list_perp_dex_names()
        spec = self.get_market_spec(execution_symbol)

        def _get_user_abstraction() -> str:
            try:
                return str(self.post_info({"type": "userAbstraction", "user": self.account_address}) or "").strip()
            except Exception:
                return ""

        with ThreadPoolExecutor(max_workers=max(4, len(dex_names) + 3), thread_name_prefix="selected-position-context") as executor:
            spot_future = executor.submit(self.post_info, {"type": "spotClearinghouseState", "user": self.account_address})
            user_abstraction_future = executor.submit(_get_user_abstraction)
            state_futures = {
                dex_name: executor.submit(
                    self.post_info,
                    self._payload_with_dex("clearinghouseState", dex_name, user=self.account_address),
                )
                for dex_name in dex_names
            }
            mids_future = executor.submit(self.get_mids, dex=dex)

            spot_user_state = spot_future.result()
            user_abstraction = user_abstraction_future.result()
            user_states: List[Tuple[str, dict]] = []
            for dex_name, future in state_futures.items():
                state = future.result()
                user_states.append((dex_name, state if isinstance(state, dict) else {}))
            mids = mids_future.result()

        spot_summary = normalize_spot_user_state(spot_user_state)
        withdrawable_usd = sum(safe_float(state.get("withdrawable"), 0.0) or 0.0 for _, state in user_states)
        spot_usdc_total = float(spot_summary.get("spot_usdc_total", 0.0) or 0.0)
        spot_available_usdc = float(spot_summary.get("spot_available_usdc", 0.0) or 0.0)
        total_margin_used_usd = 0.0
        selected_position: Optional[dict] = None
        selected_mid = safe_float(
            mids.get(execution_symbol)
            or mids.get(canonicalize_execution_symbol(market_name))
            or mids.get(canonicalize_execution_symbol(f"{dex}:{market_name}") if dex and market_name else ""),
            None,
        )
        targets = {
            execution_symbol,
            canonicalize_execution_symbol(market_name),
            canonicalize_execution_symbol(f"{dex}:{market_name}") if dex and market_name else "",
        }
        for dex_name, user_state in user_states:
            for entry in user_state.get("assetPositions", []) or []:
                position = entry.get("position", {}) or {}
                coin = canonicalize_execution_symbol(position.get("coin") or "")
                if not coin:
                    continue
                margin_used = safe_float(position.get("marginUsed", 0.0), 0.0) or 0.0
                total_margin_used_usd += margin_used
                dex_coin = canonicalize_execution_symbol(f"{dex_name}:{coin}") if dex_name and ":" not in coin else coin
                if selected_position is not None:
                    continue
                if coin not in targets and dex_coin not in targets:
                    continue
                szi = safe_float(position.get("szi", 0.0), 0.0) or 0.0
                entry_px = safe_float(position.get("entryPx", 0.0), 0.0) or 0.0
                side = "long" if szi > 0 else "short" if szi < 0 else "flat"
                notional = abs(szi) * (selected_mid if selected_mid is not None else entry_px)
                selected_position = {
                    "symbol": execution_symbol,
                    "side": side,
                    "size": szi,
                    "entry_price": entry_px,
                    "mid_price": selected_mid,
                    "notional_usd": notional,
                    "unrealized_pnl": safe_float(position.get("unrealizedPnl", 0.0), 0.0) or 0.0,
                    "return_on_equity": safe_float(position.get("returnOnEquity", 0.0), 0.0) or 0.0,
                    "leverage": safe_float((position.get("leverage") or {}).get("value"), 0.0) or 0.0,
                    "max_leverage": int(spec.get("max_leverage", 0) or 0),
                    "only_isolated": bool(spec.get("only_isolated", False)),
                    "liquidation_price": safe_float(position.get("liquidationPx", 0.0), 0.0) or 0.0,
                    "margin_used": margin_used,
                    **_cumulative_funding_fields(position),
                }

        if user_abstraction in {"unifiedAccount", "portfolioMargin"}:
            perp_account_equity_usd = spot_usdc_total
            isolated_margin_basis_usd = spot_usdc_total
            cross_margin_basis_usd = spot_usdc_total
            isolated_available_margin_usd = 0.0
            cross_available_margin_usd = 0.0
            account_equity_usd = spot_usdc_total
        else:
            perp_account_equity_usd = total_margin_used_usd + max(0.0, withdrawable_usd)
            isolated_margin_basis_usd = perp_account_equity_usd
            cross_margin_basis_usd = perp_account_equity_usd
            isolated_available_margin_usd = 0.0
            cross_available_margin_usd = max(0.0, withdrawable_usd)
            account_equity_usd = perp_account_equity_usd + spot_usdc_total
        available_margin_usd = cross_available_margin_usd + spot_available_usdc
        withdrawable_total_usd = withdrawable_usd + spot_available_usdc
        if spot_available_usdc > 0 and withdrawable_usd > 0:
            remaining_capital_source = "spot_available_plus_perp_withdrawable"
        elif spot_available_usdc > 0:
            remaining_capital_source = "spot_available_after_maintenance"
        elif withdrawable_usd > 0:
            remaining_capital_source = "withdrawable"
        else:
            remaining_capital_source = "used_plus_available"
        remaining_capital_usd = withdrawable_total_usd if withdrawable_total_usd > 0 else available_margin_usd
        all_positions = {
            "known": True,
            "partial": True,
            "partial_scope": "selected_symbol_position_context",
            "account_address": self.account_address,
            "network": self.network,
            "user_abstraction": user_abstraction,
            "spot_summary": spot_summary,
            "perp_account_equity_usd": perp_account_equity_usd,
            "isolated_margin_basis_usd": isolated_margin_basis_usd,
            "cross_margin_basis_usd": cross_margin_basis_usd,
            "spot_usdc_total": spot_usdc_total,
            "spot_usdc_hold": float(spot_summary.get("spot_usdc_hold", 0.0) or 0.0),
            "spot_available_usdc": spot_available_usdc,
            "account_equity_usd": account_equity_usd,
            "total_margin_used_usd": float(total_margin_used_usd or 0.0),
            "available_margin_usd": available_margin_usd,
            "withdrawable_usd": withdrawable_total_usd,
            "remaining_capital_usd": float(remaining_capital_usd or 0.0),
            "remaining_capital_source": remaining_capital_source,
            "positions": [selected_position] if selected_position is not None else [],
            "positions_count": 1 if selected_position is not None else 0,
            "total_notional_usd": float((selected_position or {}).get("notional_usd", 0.0) or 0.0),
        }
        if user_abstraction not in {"unifiedAccount", "portfolioMargin"}:
            all_positions.update(
                {
                    "isolated_available_margin_usd": isolated_available_margin_usd,
                    "cross_available_margin_usd": cross_available_margin_usd,
                    "perp_withdrawable_usd": withdrawable_usd,
                }
            )
        position_snapshot = self.get_position_snapshot(
            execution_symbol,
            all_positions=all_positions,
            current_price=selected_mid,
        )
        return {
            "all_positions": all_positions,
            "position_snapshot": position_snapshot,
            "mid_price": selected_mid,
        }

    def get_position_snapshot(
        self,
        symbol: str,
        all_positions: Optional[dict] = None,
        current_price: Optional[float] = None,
    ) -> dict:
        symbol = self.resolve_execution_symbol(symbol) or canonicalize_execution_symbol(symbol)
        all_positions = all_positions or self.get_all_positions()
        spec = self.get_market_spec(symbol)
        snapshot = {
            "known": True,
            "account_address": self.account_address,
            "network": self.network,
            "user_abstraction": all_positions.get("user_abstraction", ""),
            "symbol": symbol,
            "side": "flat",
            "size": 0.0,
            "entry_price": 0.0,
            "mid_price": current_price if current_price is not None else self.get_mid_price(symbol),
            "notional_usd": 0.0,
            "leverage": 0.0,
            "max_leverage": int(spec.get("max_leverage", 0) or 0),
            "only_isolated": bool(spec.get("only_isolated", False)),
            "account_equity_usd": float(all_positions.get("account_equity_usd", 0.0) or 0.0),
            "perp_account_equity_usd": float(all_positions.get("perp_account_equity_usd", 0.0) or 0.0),
            "isolated_margin_basis_usd": float(all_positions.get("isolated_margin_basis_usd", 0.0) or 0.0),
            "cross_margin_basis_usd": float(all_positions.get("cross_margin_basis_usd", 0.0) or 0.0),
            "available_margin_usd": float(all_positions.get("available_margin_usd", 0.0) or 0.0),
            "withdrawable_usd": float(all_positions.get("withdrawable_usd", 0.0) or 0.0),
            "remaining_capital_usd": float(all_positions.get("remaining_capital_usd", 0.0) or 0.0),
        }
        if "isolated_available_margin_usd" in all_positions:
            snapshot["isolated_available_margin_usd"] = float(all_positions.get("isolated_available_margin_usd", 0.0) or 0.0)
        if "cross_available_margin_usd" in all_positions:
            snapshot["cross_available_margin_usd"] = float(all_positions.get("cross_available_margin_usd", 0.0) or 0.0)
        if "perp_withdrawable_usd" in all_positions:
            snapshot["perp_withdrawable_usd"] = float(all_positions.get("perp_withdrawable_usd", 0.0) or 0.0)
        for pos in all_positions.get("positions", []):
            if pos.get("symbol") == symbol:
                snapshot.update(pos)
                return snapshot
        return snapshot

    @staticmethod
    def format_all_positions(snapshot: dict) -> str:
        lines = [
            f"address={snapshot.get('account_address')}",
            f"network={snapshot.get('network')}",
            f"account_equity_usd≈{float(snapshot.get('account_equity_usd', 0.0) or 0.0):.2f}",
            f"withdrawable_usd≈{float(snapshot.get('withdrawable_usd', 0.0) or 0.0):.2f}",
            f"available_margin_usd≈{float(snapshot.get('available_margin_usd', 0.0) or 0.0):.2f}",
            f"remaining_capital_usd≈{float(snapshot.get('remaining_capital_usd', 0.0) or 0.0):.2f}",
            f"remaining_capital_source={snapshot.get('remaining_capital_source')}",
            f"perp_account_equity_usd≈{float(snapshot.get('perp_account_equity_usd', 0.0) or 0.0):.2f}",
            f"spot_usdc_total≈{float(snapshot.get('spot_usdc_total', 0.0) or 0.0):.2f}",
            f"spot_usdc_hold≈{float(snapshot.get('spot_usdc_hold', 0.0) or 0.0):.2f}",
            f"spot_available_usdc≈{float(snapshot.get('spot_available_usdc', 0.0) or 0.0):.2f}",
            f"total_margin_used_usd≈{float(snapshot.get('total_margin_used_usd', 0.0) or 0.0):.2f}",
            f"positions_count={snapshot.get('positions_count', 0)}",
            f"total_notional_usd≈{float(snapshot.get('total_notional_usd', 0.0) or 0.0):.2f}",
        ]
        if "cross_available_margin_usd" in snapshot:
            lines.append(f"cross_available_margin_usd≈{float(snapshot.get('cross_available_margin_usd', 0.0) or 0.0):.2f}")
        if "perp_withdrawable_usd" in snapshot:
            lines.append(f"perp_withdrawable_usd≈{float(snapshot.get('perp_withdrawable_usd', 0.0) or 0.0):.2f}")
        positions = snapshot.get("positions", []) or []
        if not positions:
            lines.append("positions=<empty>")
            return "\n".join(lines)
        lines.append("positions=")
        for i, pos in enumerate(positions, start=1):
            mid = pos.get("mid_price")
            mid_txt = "null" if mid is None else f"{float(mid):.6f}"
            lines.append(
                "  "
                + f"[{i}] {pos.get('symbol')} side={pos.get('side')} "
                + f"size={float(pos.get('size', 0.0) or 0.0):.8f} "
                + f"entry={float(pos.get('entry_price', 0.0) or 0.0):.6f} "
                + f"mid={mid_txt} "
                + f"notional≈{float(pos.get('notional_usd', 0.0) or 0.0):.2f} "
                + f"upnl={float(pos.get('unrealized_pnl', 0.0) or 0.0):.6f} "
                + f"lev={float(pos.get('leverage', 0.0) or 0.0):.2f} "
                + f"max_lev={int(pos.get('max_leverage', 0) or 0)}"
            )
        return "\n".join(lines)

    @staticmethod
    def format_symbol_position(snapshot: dict) -> str:
        mid = snapshot.get("mid_price")
        mid_txt = "null" if mid is None else f"{float(mid):.6f}"
        return "\n".join(
            [
                f"address={snapshot.get('account_address')}",
                f"network={snapshot.get('network')}",
                f"symbol={snapshot.get('symbol')}",
                f"side={snapshot.get('side')}",
                f"size={float(snapshot.get('size', 0.0) or 0.0):.8f}",
                f"entry_price={float(snapshot.get('entry_price', 0.0) or 0.0):.6f}",
                f"mid_price={mid_txt}",
                f"notional_usd≈{float(snapshot.get('notional_usd', 0.0) or 0.0):.2f}",
                f"leverage={float(snapshot.get('leverage', 0.0) or 0.0):.2f}",
                f"max_leverage={int(snapshot.get('max_leverage', 0) or 0)}",
                f"only_isolated={bool(snapshot.get('only_isolated', False))}",
                f"remaining_capital_usd≈{float(snapshot.get('remaining_capital_usd', 0.0) or 0.0):.2f}",
                f"available_margin_usd≈{float(snapshot.get('available_margin_usd', 0.0) or 0.0):.2f}",
            ]
        )


class HyperliquidExecutor:
    def __init__(self, reader: HyperliquidRestReader, symbol: str):
        self.reader = reader
        self.symbol = canonicalize_execution_symbol(symbol)
        self.enabled = os.getenv("ENABLE_LIVE_TRADING", "false").lower() == "true"
        self.slippage = float(os.getenv("HYPERLIQUID_SLIPPAGE", "0.01"))
        self._exchange = None
        self._wallet = None
        self._risk_order_nonce = 0

    def _ensure_exchange(self) -> None:
        if self._exchange is not None:
            return
        secret_key = (os.getenv("HL_SECRET_KEY", "") or "").strip()
        if not secret_key:
            raise RuntimeError("HL_SECRET_KEY is required when ENABLE_LIVE_TRADING=true")
        try:
            import eth_account
            from hyperliquid.exchange import Exchange
        except ImportError as exc:
            raise RuntimeError(
                "Missing live-trading dependencies. Install: pip install hyperliquid-python-sdk eth-account"
            ) from exc
        self._wallet = eth_account.Account.from_key(secret_key)
        exchange_kwargs = {"account_address": self.reader.account_address}
        if hasattr(self.reader, "get_meta"):
            exchange_kwargs["meta"] = self.reader.get_meta()
        if hasattr(self.reader, "get_safe_spot_meta"):
            exchange_kwargs["spot_meta"] = self.reader.get_safe_spot_meta()
        if hasattr(self.reader, "list_perp_dex_names"):
            exchange_kwargs["perp_dexs"] = [""] + list(self.reader.list_perp_dex_names())
        self._exchange = Exchange(self._wallet, self.reader.base, **exchange_kwargs)

    def usd_to_size(self, notional_usd: float, mid_price: float) -> float:
        decimals = self.reader.get_sz_decimals(self.symbol)
        try:
            notional = Decimal(str(max(0.0, float(notional_usd or 0.0))))
            price = Decimal(str(max(float(mid_price or 0.0), 1e-12)))
            return self._quantize_size_decimal(notional / price, decimals)
        except (InvalidOperation, ValueError, TypeError, ZeroDivisionError):
            return 0.0

    @staticmethod
    def _quantize_size_decimal(size: Decimal, decimals: int) -> float:
        decimals = max(0, int(decimals or 0))
        if not size.is_finite() or size <= 0:
            return 0.0
        quantum = Decimal(1).scaleb(-decimals)
        epsilon = quantum * Decimal("1e-9")
        return float((size + epsilon).quantize(quantum, rounding=ROUND_DOWN))

    def _round_size_to_precision(self, size: float) -> float:
        decimals = max(0, int(self.reader.get_sz_decimals(self.symbol) or 0))
        try:
            raw = Decimal(str(max(0.0, float(size or 0.0))))
            return self._quantize_size_decimal(raw, decimals)
        except (InvalidOperation, ValueError, TypeError):
            return 0.0

    def _round_price_to_wire_precision(self, price: float) -> float:
        raw = float(price or 0.0)
        if raw <= 0.0:
            return 0.0
        if hasattr(self.reader, "align_price_to_wire"):
            return float(self.reader.align_price_to_wire(self.symbol, raw) or 0.0)
        if hasattr(self.reader, "get_sz_decimals"):
            try:
                sz_decimals = int(self.reader.get_sz_decimals(self.symbol) or 0)
                return round(float(f"{raw:.5g}"), max(0, 6 - sz_decimals))
            except Exception:
                pass
        return raw

    def _result_has_exchange_error(self, payload: Any) -> bool:
        if isinstance(payload, dict):
            status = str(payload.get("status", "") or "").strip().lower()
            if status == "err":
                return True
            error = payload.get("error")
            if isinstance(error, str) and error.strip():
                return True
            return any(self._result_has_exchange_error(value) for value in payload.values())
        if isinstance(payload, list):
            return any(self._result_has_exchange_error(item) for item in payload)
        return False

    def _payload_contains_text(self, payload: Any, needles: Tuple[str, ...]) -> bool:
        if isinstance(payload, dict):
            return any(self._payload_contains_text(value, needles) for value in payload.values())
        if isinstance(payload, list):
            return any(self._payload_contains_text(item, needles) for item in payload)
        if isinstance(payload, str):
            text = payload.lower()
            return any(needle in text for needle in needles)
        return False

    @staticmethod
    def _filled_qty_from_exchange_result(payload: Any) -> float:
        statuses = (((payload or {}).get("response") or {}).get("data") or {}).get("statuses") or []
        filled_qty = 0.0
        for status in statuses:
            filled = status.get("filled") if isinstance(status, dict) else {}
            if not isinstance(filled, dict):
                continue
            filled_qty += safe_float(filled.get("totalSz", filled.get("sz")), 0.0) or 0.0
        return max(0.0, filled_qty)

    def _is_partial_fill(self, requested_qty: float, filled_qty: float) -> bool:
        requested = max(0.0, float(requested_qty or 0.0))
        filled = max(0.0, float(filled_qty or 0.0))
        if requested <= 0.0 or filled <= 0.0:
            return False
        decimals = max(0, int(self.reader.get_sz_decimals(self.symbol) or 0))
        tolerance = float(Decimal(1).scaleb(-decimals)) / 2.0
        return filled < max(0.0, requested - tolerance)

    def _market_open_with_qty_backoff(self, *, is_buy: bool, qty: float) -> Dict[str, Any]:
        requested_qty = self._round_size_to_precision(qty)
        attempt_qty = requested_qty
        attempted_qtys: List[float] = []
        last_exchange_result: Dict[str, Any] = {}
        max_attempts = 4
        for _ in range(max_attempts):
            if attempt_qty <= 0:
                break
            attempted_qtys.append(attempt_qty)
            exchange_result = self._exchange.market_open(self.symbol, is_buy, attempt_qty, None, self.slippage)
            last_exchange_result = exchange_result
            if not self._result_has_exchange_error(exchange_result):
                filled_qty = self._filled_qty_from_exchange_result(exchange_result)
                partial_fill = self._is_partial_fill(attempt_qty, filled_qty)
                message = ""
                if partial_fill:
                    message = f"Market order partially filled: filled {filled_qty:g} / requested {attempt_qty:g} {self.symbol}."
                return {
                    "exchange_result": exchange_result,
                    "requested_qty": requested_qty,
                    "final_qty": attempt_qty,
                    "filled_qty": filled_qty,
                    "partial_fill": partial_fill,
                    "attempt_count": len(attempted_qtys),
                    "attempted_qtys": attempted_qtys,
                    "accepted": True,
                    "message": message,
                }
            if self._payload_contains_text(
                exchange_result,
                ("mintradentlrejected", "minimum trade", "min trade", "minimum notional"),
            ):
                return {
                    "exchange_result": exchange_result,
                    "requested_qty": requested_qty,
                    "final_qty": attempt_qty,
                    "attempt_count": len(attempted_qtys),
                    "attempted_qtys": attempted_qtys,
                    "accepted": False,
                    "message": "Exchange rejected market_open: below minimum trade notional.",
                }
            if self._payload_contains_text(
                exchange_result,
                ("perpmarginrejected", "insufficient margin", "not enough margin", "margin rejected"),
            ):
                return {
                    "exchange_result": exchange_result,
                    "requested_qty": requested_qty,
                    "final_qty": attempt_qty,
                    "attempt_count": len(attempted_qtys),
                    "attempted_qtys": attempted_qtys,
                    "accepted": False,
                    "message": "Exchange rejected market_open: insufficient margin.",
                }
            next_qty = self._round_size_to_precision(attempt_qty * 0.99)
            if next_qty <= 0 or next_qty >= attempt_qty:
                break
            attempt_qty = next_qty
        return {
            "exchange_result": last_exchange_result,
            "requested_qty": requested_qty,
            "final_qty": attempt_qty,
            "attempt_count": len(attempted_qtys),
            "attempted_qtys": attempted_qtys,
            "accepted": False,
            "message": "Exchange rejected market_open after qty backoff retries.",
        }

    def _entry_limit_with_qty_backoff(self, *, is_buy: bool, qty: float, limit_price: float) -> Dict[str, Any]:
        from hyperliquid.utils.types import Cloid

        requested_qty = self._round_size_to_precision(qty)
        attempt_qty = requested_qty
        attempted_qtys: List[float] = []
        last_exchange_result: Dict[str, Any] = {}
        order_type = {"limit": {"tif": "Gtc"}}
        max_attempts = 4
        for _ in range(max_attempts):
            if attempt_qty <= 0:
                break
            attempted_qtys.append(attempt_qty)
            cloid_raw = self._next_risk_order_cloid()
            exchange_result = self._exchange.order(
                self.symbol,
                is_buy,
                attempt_qty,
                limit_price,
                order_type,
                reduce_only=False,
                cloid=Cloid.from_str(cloid_raw),
            )
            last_exchange_result = exchange_result
            if not self._result_has_exchange_error(exchange_result):
                result = {
                    "exchange_result": exchange_result,
                    "requested_qty": requested_qty,
                    "final_qty": attempt_qty,
                    "attempt_count": len(attempted_qtys),
                    "attempted_qtys": attempted_qtys,
                    "accepted": True,
                    "cloid": cloid_raw,
                }
                try:
                    statuses = (((exchange_result or {}).get("response") or {}).get("data") or {}).get("statuses") or []
                    first_status = statuses[0] if statuses else {}
                    resting = first_status.get("resting") if isinstance(first_status, dict) else {}
                    if isinstance(resting, dict):
                        oid_value = resting.get("oid")
                        if oid_value is not None:
                            result["oid"] = int(oid_value)
                        resting_cloid = str(resting.get("cloid", "") or "").strip()
                        if resting_cloid:
                            result["cloid"] = resting_cloid
                except Exception:
                    pass
                return result
            next_qty = self._round_size_to_precision(attempt_qty * 0.99)
            if next_qty <= 0 or next_qty >= attempt_qty:
                break
            attempt_qty = next_qty
        return {
            "exchange_result": last_exchange_result,
            "requested_qty": requested_qty,
            "final_qty": attempt_qty,
            "attempt_count": len(attempted_qtys),
            "attempted_qtys": attempted_qtys,
            "accepted": False,
            "message": "Exchange rejected entry limit order after qty backoff retries.",
        }

    def place_market_order(
        self,
        *,
        side: str,
        notional_usd: float,
        requested_leverage: int,
        reason: str,
        plan_name: Optional[str] = None,
        position_before: Optional[dict] = None,
        execution_mid_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        snapshot = position_before if isinstance(position_before, dict) else self.reader.get_position_snapshot(self.symbol)
        result = self._position_result(snapshot, plan_name, reason)
        side = str(side or "").strip().lower()
        requested_notional_usd = max(0.0, float(notional_usd or 0.0))
        result.update(
            {
                "target_side": side,
                "order_notional_usd": requested_notional_usd,
                "requested_leverage": int(requested_leverage or 0),
            }
        )
        if side not in {"long", "short"}:
            result["message"] = f"Unsupported target side for market order: {side or '<empty>'}."
            result["accepted"] = False
            return result
        if requested_notional_usd <= 0:
            result["message"] = "Market order skipped due to non-positive order notional."
            result["accepted"] = False
            return result
        mid = safe_float(execution_mid_price, None)
        if mid is not None and mid > 0:
            result["mid_source"] = "execution_context"
        else:
            mid = self.reader.get_mid_price(self.symbol)
            result["mid_source"] = "fresh_mid_price"
        result["mid"] = mid
        if mid is None or mid <= 0:
            result["message"] = "Could not fetch valid mid price; skipped market order."
            result["accepted"] = False
            return result
        leverage_update = self.apply_requested_leverage(int(requested_leverage or 0))
        if leverage_update:
            result["leverage_update"] = leverage_update
        qty = self.usd_to_size(requested_notional_usd, mid)
        result["requested_qty"] = qty
        result["open_qty"] = qty
        if qty <= 0:
            result["message"] = "Market order size rounded to 0 after precision handling; skipped."
            result["accepted"] = False
            return result
        if not self.enabled:
            result["message"] = "ENABLE_LIVE_TRADING=false, so market order is dry-run only."
            return result
        self._ensure_exchange()
        is_buy = side == "long"
        open_attempt = self._market_open_with_qty_backoff(is_buy=is_buy, qty=qty)
        exchange_result = open_attempt.get("exchange_result", {})
        final_qty = float(open_attempt.get("final_qty", qty) or 0.0)
        filled_qty = float(open_attempt.get("filled_qty", 0.0) or 0.0)
        result["requested_qty"] = float(open_attempt.get("requested_qty", qty) or 0.0)
        result["open_qty"] = final_qty
        result["filled_qty"] = filled_qty
        result["partial_fill"] = bool(open_attempt.get("partial_fill", False))
        if filled_qty > 0.0 and mid and mid > 0:
            result["filled_notional_usd"] = filled_qty * float(mid)
        result["attempt_count"] = int(open_attempt.get("attempt_count", 0) or 0)
        result["attempted_qtys"] = [float(item or 0.0) for item in list(open_attempt.get("attempted_qtys", []) or [])]
        result["accepted"] = bool(open_attempt.get("accepted", False))
        result["actions"].append(
            {
                "market_open": exchange_result,
                "symbol": self.symbol,
                "is_buy": is_buy,
                "qty": final_qty,
                "requested_qty": result["requested_qty"],
                "filled_qty": filled_qty,
                "attempt_count": result["attempt_count"],
                "attempted_qtys": result["attempted_qtys"],
                "mid": mid,
            }
        )
        if result["accepted"]:
            result["message"] = str(open_attempt.get("message", "") or "Placed market order.")
        else:
            result["message"] = str(open_attempt.get("message", "") or "Exchange rejected market order.")
        return result

    def place_entry_limit_order(
        self,
        *,
        side: str,
        notional_usd: float,
        requested_leverage: int,
        limit_price: float,
        reason: str,
        plan_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        snapshot = self.reader.get_position_snapshot(self.symbol)
        result = self._position_result(snapshot, plan_name, reason)
        side = str(side or "").strip().lower()
        rounded_limit_price = self._round_price_to_wire_precision(limit_price)
        requested_notional_usd = max(0.0, float(notional_usd or 0.0))
        result.update(
            {
                "target_side": side,
                "target_notional_usd": requested_notional_usd,
                "requested_leverage": int(requested_leverage or 0),
                "requested_limit_price": float(limit_price or 0.0),
                "limit_price": rounded_limit_price,
            }
        )
        if side not in {"long", "short"}:
            result["message"] = f"Unsupported target side for entry limit order: {side or '<empty>'}."
            result["accepted"] = False
            result["entry_order_pending"] = False
            return result
        if requested_notional_usd <= 0 or rounded_limit_price <= 0:
            result["message"] = "Entry limit order skipped due to non-positive target notional or limit price."
            result["accepted"] = False
            result["entry_order_pending"] = False
            return result
        leverage_update = self.apply_requested_leverage(int(requested_leverage or 0))
        if leverage_update:
            result["leverage_update"] = leverage_update
        qty = self.usd_to_size(requested_notional_usd, rounded_limit_price)
        result["requested_qty"] = qty
        result["open_qty"] = qty
        if qty <= 0:
            result["message"] = "Entry limit size rounded to 0 after precision handling; skipped."
            result["accepted"] = False
            result["entry_order_pending"] = False
            return result
        cloid_raw = self._next_risk_order_cloid()
        result["cloid"] = cloid_raw
        if not self.enabled:
            result["message"] = "ENABLE_LIVE_TRADING=false, so entry limit order is dry-run only."
            result["accepted"] = False
            result["entry_order_pending"] = False
            return result
        self._ensure_exchange()
        is_buy = side == "long"
        open_attempt = self._entry_limit_with_qty_backoff(is_buy=is_buy, qty=qty, limit_price=rounded_limit_price)
        exchange_result = open_attempt.get("exchange_result", {})
        final_qty = float(open_attempt.get("final_qty", qty) or 0.0)
        result["requested_qty"] = float(open_attempt.get("requested_qty", qty) or 0.0)
        result["open_qty"] = final_qty
        result["attempt_count"] = int(open_attempt.get("attempt_count", 0) or 0)
        result["attempted_qtys"] = [float(item or 0.0) for item in list(open_attempt.get("attempted_qtys", []) or [])]
        result["actions"].append(
            {
                "entry_limit": exchange_result,
                "symbol": self.symbol,
                "is_buy": is_buy,
                "qty": final_qty,
                "requested_qty": result["requested_qty"],
                "attempt_count": result["attempt_count"],
                "attempted_qtys": result["attempted_qtys"],
                "limit_price": rounded_limit_price,
            }
        )
        result["accepted"] = bool(open_attempt.get("accepted", False))
        result["entry_order_pending"] = bool(result["accepted"])
        if result["accepted"]:
            oid_value = open_attempt.get("oid")
            if oid_value is not None:
                result["oid"] = int(oid_value)
            cloid_value = str(open_attempt.get("cloid", "") or "").strip()
            if cloid_value:
                result["cloid"] = cloid_value
            result["message"] = "Placed resting entry limit order."
            return result
        if not str(result.get("message", "") or "").strip():
            result["message"] = str(open_attempt.get("message", "") or "Exchange rejected entry limit order.")
        return result

    def cancel_entry_order(
        self,
        *,
        oid: Optional[int] = None,
        cloid: str = "",
        reason: str,
        plan_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        snapshot = self.reader.get_position_snapshot(self.symbol)
        result = self._position_result(snapshot, plan_name, reason)
        result["oid"] = int(oid) if oid is not None else None
        result["cloid"] = str(cloid or "")
        if oid is None and not str(cloid or "").strip():
            result["message"] = "No entry order identifier available for cancellation."
            result["accepted"] = False
            return result
        if not self.enabled:
            result["message"] = "ENABLE_LIVE_TRADING=false, so entry order cancel is dry-run only."
            result["accepted"] = False
            return result
        self._ensure_exchange()
        actions = []
        if oid is not None:
            actions.append({"cancel_entry_order": self._exchange.cancel(self.symbol, int(oid))})
        elif str(cloid or "").strip():
            from hyperliquid.utils.types import Cloid

            cancel_result = self._exchange.bulk_cancel_by_cloid([
                {"coin": self.symbol, "cloid": Cloid.from_str(str(cloid).strip())}
            ])
            actions.append({"cancel_entry_order_by_cloid": cancel_result})
        result["actions"] = actions
        result["accepted"] = not self._result_has_exchange_error(actions)
        if result["accepted"]:
            result["message"] = "Cancelled pending entry order."
        elif not str(result.get("message", "") or "").strip():
            result["message"] = "Exchange rejected entry order cancellation."
        return result

    def apply_requested_leverage(self, requested_leverage: int) -> Dict[str, Any]:
        spec = self.reader.get_market_spec(self.symbol)
        allowed_max = int(spec.get("max_leverage", 0) or 0)
        if requested_leverage <= 0 or allowed_max <= 0:
            return {}
        leverage = max(1, min(int(requested_leverage), allowed_max))
        is_cross = not bool(spec.get("only_isolated", False))
        result = {
            "requested_leverage": int(requested_leverage),
            "applied_leverage": leverage,
            "max_leverage": allowed_max,
            "is_cross": is_cross,
        }
        if not self.enabled:
            result["message"] = "ENABLE_LIVE_TRADING=false, so leverage update is dry-run only."
            return result
        self._ensure_exchange()
        result["exchange"] = self._exchange.update_leverage(leverage, self.symbol, is_cross=is_cross)
        return result

    def _leverage_update_has_exchange_error(self, update: Any) -> bool:
        exchange = update.get("exchange") if isinstance(update, dict) else {}
        if isinstance(exchange, dict) and str(exchange.get("status", "") or "").strip().lower() == "err":
            return True
        return self._result_has_exchange_error(exchange)

    @staticmethod
    def _leverage_update_needs_more_margin(update: Any) -> bool:
        exchange = update.get("exchange") if isinstance(update, dict) else {}
        response = str((exchange or {}).get("response", "") or "").strip().lower()
        if not response:
            return False
        return "sufficient margin" in response or "add margin" in response

    def _estimate_isolated_margin_top_up_usd(self, snapshot: dict, requested_leverage: int) -> float:
        if not isinstance(snapshot, dict) or not snapshot_has_open_position(snapshot):
            return 0.0
        if not bool(snapshot.get("only_isolated", False)):
            return 0.0
        target_leverage = max(1, int(requested_leverage or 0))
        current_leverage = max(0.0, float(snapshot.get("leverage", 0.0) or 0.0))
        if current_leverage <= 0.0 or target_leverage >= current_leverage:
            return 0.0
        notional_usd = max(0.0, float(snapshot.get("notional_usd", 0.0) or 0.0))
        margin_used = max(0.0, float(snapshot.get("margin_used", 0.0) or 0.0))
        if notional_usd <= 0.0 or margin_used <= 0.0:
            return 0.0
        required_margin = notional_usd / max(float(target_leverage), 1e-12)
        shortfall = required_margin - margin_used
        if shortfall <= 0.0:
            return 0.0
        buffer_usd = max(0.10, required_margin * 0.01)
        return max(0.0, shortfall + buffer_usd)

    @staticmethod
    def _quantize_usd_amount_for_exchange(amount: float) -> float:
        normalized = max(0.0, float(amount or 0.0))
        if normalized <= 0.0:
            return 0.0
        return float(Decimal(str(normalized)).quantize(Decimal("0.000001"), rounding=ROUND_UP))

    def _add_isolated_margin_if_needed(self, snapshot: dict, requested_leverage: int) -> Dict[str, Any]:
        raw_requested_amount = self._estimate_isolated_margin_top_up_usd(snapshot, requested_leverage)
        requested_amount = self._quantize_usd_amount_for_exchange(raw_requested_amount)
        if requested_amount <= 0.0:
            return {}
        result = {
            "raw_requested_amount_usd": raw_requested_amount,
            "requested_amount_usd": requested_amount,
            "symbol": self.symbol,
        }
        if not self.enabled:
            result["message"] = "ENABLE_LIVE_TRADING=false, so isolated margin update is dry-run only."
            return result
        self._ensure_exchange()
        result["exchange"] = self._exchange.update_isolated_margin(requested_amount, self.symbol)
        return result

    def reconcile_requested_leverage_after_execution(self, position_after: dict, requested_leverage: int) -> Dict[str, Any]:
        def refresh_symbol_snapshot() -> dict:
            if hasattr(self.reader, "get_selected_symbol_position_context"):
                try:
                    context = self.reader.get_selected_symbol_position_context(self.symbol)
                    snapshot = context.get("position_snapshot") if isinstance(context, dict) else None
                    if isinstance(snapshot, dict):
                        return snapshot
                except Exception:
                    pass
            return self.reader.get_position_snapshot(self.symbol)

        target_leverage = max(0, int(requested_leverage or 0))
        snapshot = dict(position_after or {}) if isinstance(position_after, dict) else {}
        if target_leverage <= 0 or not snapshot_has_open_position(snapshot):
            return {}
        current_leverage = max(0.0, float(snapshot.get("leverage", 0.0) or 0.0))
        if current_leverage > 0.0 and abs(current_leverage - float(target_leverage)) < 1e-9:
            return {}
        result: Dict[str, Any] = {}
        only_isolated = bool(snapshot.get("only_isolated", False))
        if only_isolated and current_leverage > 0.0 and target_leverage < current_leverage:
            isolated_margin_update = self._add_isolated_margin_if_needed(snapshot, target_leverage)
            if isolated_margin_update:
                result["isolated_margin_update"] = isolated_margin_update
                snapshot = refresh_symbol_snapshot()
        leverage_update = self.apply_requested_leverage(target_leverage)
        if leverage_update:
            result["leverage_update"] = leverage_update
        if (
            only_isolated
            and target_leverage < current_leverage
            and self._leverage_update_has_exchange_error(leverage_update)
            and self._leverage_update_needs_more_margin(leverage_update)
        ):
            retry_snapshot = refresh_symbol_snapshot()
            isolated_margin_retry = self._add_isolated_margin_if_needed(retry_snapshot, target_leverage)
            if isolated_margin_retry:
                result["isolated_margin_update_retry"] = isolated_margin_retry
                retry_snapshot = refresh_symbol_snapshot()
            leverage_retry = self.apply_requested_leverage(target_leverage)
            if leverage_retry:
                result["leverage_update_retry"] = leverage_retry
                result["leverage_update"] = leverage_retry
            snapshot = refresh_symbol_snapshot()
        else:
            snapshot = refresh_symbol_snapshot()
        result["position_after"] = snapshot
        return result

    def _next_risk_order_cloid(self) -> str:
        self._risk_order_nonce += 1
        raw = (int(time.time_ns()) + self._risk_order_nonce) & ((1 << 128) - 1)
        return f"0x{raw:032x}"

    def place_reduce_only_tpsl_order(
        self,
        *,
        side: str,
        close_size: float,
        trigger_price: float,
        tpsl: str,
        plan_name: Optional[str] = None,
        leg_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        snapshot = self.reader.get_position_snapshot(self.symbol)
        result = self._position_result(snapshot, plan_name, f"risk_tpsl_{tpsl}_order")
        requested_qty = max(0.0, float(close_size or 0.0))
        qty = self._round_size_to_precision(requested_qty)
        trigger_price = float(trigger_price or 0.0)
        rounded_trigger_price = self._round_price_to_wire_precision(trigger_price)
        cloid_raw = self._next_risk_order_cloid()
        result.update({
            "leg_name": str(leg_name or ""),
            "requested_close_size": requested_qty,
            "close_size": qty,
            "trigger_price": rounded_trigger_price,
            "requested_trigger_price": trigger_price,
            "tpsl": str(tpsl or ""),
            "order_kind": "trigger",
            "is_trigger": True,
            "cloid": cloid_raw,
        })
        if qty <= 0 or rounded_trigger_price <= 0:
            result["message"] = "Reduce-only TP/SL order skipped due to non-positive size or trigger price."
            result["accepted"] = False
            return result
        if not self.enabled:
            result["message"] = "ENABLE_LIVE_TRADING=false, so reduce-only trigger order is dry-run only."
            result["accepted"] = False
            return result
        self._ensure_exchange()
        from hyperliquid.utils.types import Cloid

        is_buy = side == "short"
        order_type = {"trigger": {"triggerPx": rounded_trigger_price, "isMarket": True, "tpsl": str(tpsl or "sl")}}
        exchange_result = self._exchange.order(
            self.symbol,
            is_buy,
            qty,
            rounded_trigger_price,
            order_type,
            reduce_only=True,
            cloid=Cloid.from_str(cloid_raw),
        )
        result["actions"].append({"reduce_only_tpsl_order": exchange_result})
        result["accepted"] = not self._result_has_exchange_error(exchange_result)
        if result["accepted"]:
            try:
                statuses = (((exchange_result or {}).get("response") or {}).get("data") or {}).get("statuses") or []
                first_status = statuses[0] if statuses else {}
                resting = first_status.get("resting") if isinstance(first_status, dict) else {}
                if isinstance(resting, dict):
                    oid_value = resting.get("oid")
                    if oid_value is not None:
                        result["oid"] = int(oid_value)
                    resting_cloid = str(resting.get("cloid", "") or "").strip()
                    if resting_cloid:
                        result["cloid"] = resting_cloid
            except Exception:
                pass
        if not result["accepted"] and not str(result.get("message", "") or "").strip():
            result["message"] = "Exchange rejected reduce-only trigger order."
        return result

    def place_reduce_only_limit_order(
        self,
        *,
        side: str,
        close_size: float,
        limit_price: float,
        plan_name: Optional[str] = None,
        leg_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        snapshot = self.reader.get_position_snapshot(self.symbol)
        result = self._position_result(snapshot, plan_name, "risk_take_profit_limit_order")
        requested_qty = max(0.0, float(close_size or 0.0))
        qty = self._round_size_to_precision(requested_qty)
        limit_price = float(limit_price or 0.0)
        rounded_limit_price = self._round_price_to_wire_precision(limit_price)
        cloid_raw = self._next_risk_order_cloid()
        result.update(
            {
                "leg_name": str(leg_name or ""),
                "requested_close_size": requested_qty,
                "close_size": qty,
                "limit_price": rounded_limit_price,
                "requested_limit_price": limit_price,
                "trigger_price": rounded_limit_price,
                "requested_trigger_price": limit_price,
                "tpsl": "tp",
                "order_kind": "limit",
                "is_trigger": False,
                "cloid": cloid_raw,
            }
        )
        if qty <= 0 or rounded_limit_price <= 0:
            result["message"] = "Reduce-only limit order skipped due to non-positive size or limit price."
            result["accepted"] = False
            return result
        if not self.enabled:
            result["message"] = "ENABLE_LIVE_TRADING=false, so reduce-only limit order is dry-run only."
            result["accepted"] = False
            return result
        self._ensure_exchange()
        from hyperliquid.utils.types import Cloid

        is_buy = side == "short"
        order_type = {"limit": {"tif": "Gtc"}}
        exchange_result = self._exchange.order(
            self.symbol,
            is_buy,
            qty,
            rounded_limit_price,
            order_type,
            reduce_only=True,
            cloid=Cloid.from_str(cloid_raw),
        )
        result["actions"].append({"reduce_only_limit_order": exchange_result})
        result["accepted"] = not self._result_has_exchange_error(exchange_result)
        if result["accepted"]:
            try:
                statuses = (((exchange_result or {}).get("response") or {}).get("data") or {}).get("statuses") or []
                first_status = statuses[0] if statuses else {}
                resting = first_status.get("resting") if isinstance(first_status, dict) else {}
                filled = first_status.get("filled") if isinstance(first_status, dict) else {}
                order_status = resting if isinstance(resting, dict) and resting else filled if isinstance(filled, dict) else {}
                if isinstance(order_status, dict):
                    oid_value = order_status.get("oid")
                    if oid_value is not None:
                        result["oid"] = int(oid_value)
                    resting_cloid = str(order_status.get("cloid", "") or "").strip()
                    if resting_cloid:
                        result["cloid"] = resting_cloid
            except Exception:
                pass
        if not result["accepted"] and not str(result.get("message", "") or "").strip():
            result["message"] = "Exchange rejected reduce-only limit order."
        return result

    def cancel_reduce_only_tpsl_orders(self, order_refs: List[Dict[str, Any]], plan_name: Optional[str] = None) -> Dict[str, Any]:
        snapshot = self.reader.get_position_snapshot(self.symbol)
        result = self._position_result(snapshot, plan_name, "risk_tpsl_order_cancel")
        refs = [dict(item) for item in list(order_refs or []) if isinstance(item, dict)]
        result["order_refs"] = refs
        if not refs:
            result["message"] = "No reduce-only exit orders to cancel."
            result["accepted"] = True
            return result
        if not self.enabled:
            result["message"] = "ENABLE_LIVE_TRADING=false, so reduce-only order cancel is dry-run only."
            result["accepted"] = False
            return result
        self._ensure_exchange()
        oid_cancel_requests = []
        cloid_cancel_requests = []
        for ref in refs:
            oid_value = ref.get("oid")
            if oid_value is not None:
                try:
                    oid_cancel_requests.append({"coin": self.symbol, "oid": int(oid_value)})
                    continue
                except Exception:
                    pass
            cloid_raw = str(ref.get("cloid", "") or "").strip()
            if cloid_raw:
                from hyperliquid.utils.types import Cloid

                cloid_cancel_requests.append({"coin": self.symbol, "cloid": Cloid.from_str(cloid_raw)})
        if oid_cancel_requests:
            exchange_result = self._exchange.bulk_cancel(oid_cancel_requests)
            result["actions"].append({"cancel_reduce_only_tpsl_orders": exchange_result})
        if cloid_cancel_requests:
            cloid_cancel_result = self._exchange.bulk_cancel_by_cloid(cloid_cancel_requests)
            result["actions"].append({"cancel_reduce_only_tpsl_orders_by_cloid": cloid_cancel_result})
        result["accepted"] = not self._result_has_exchange_error(result.get("actions", []))
        if not result["accepted"] and not str(result.get("message", "") or "").strip():
            result["message"] = "Exchange rejected risk order cancellation."
        return result

    def modify_reduce_only_tpsl_orders(
        self,
        order_updates: List[Dict[str, Any]],
        *,
        side: str,
        plan_name: Optional[str] = None,
    ) -> Dict[str, Any]:
        snapshot = self.reader.get_position_snapshot(self.symbol)
        result = self._position_result(snapshot, plan_name, "risk_tpsl_order_modify")
        updates = [dict(item) for item in list(order_updates or []) if isinstance(item, dict)]
        result["order_updates"] = updates
        if not updates:
            result["message"] = "No reduce-only TP/SL orders to modify."
            result["accepted"] = True
            result["updated_refs"] = []
            return result
        if not self.enabled:
            result["message"] = "ENABLE_LIVE_TRADING=false, so reduce-only order modify is dry-run only."
            result["accepted"] = False
            result["updated_refs"] = []
            return result
        self._ensure_exchange()
        from hyperliquid.utils.types import Cloid

        is_buy = side == "short"
        modify_requests = []
        updated_refs: List[Dict[str, Any]] = []
        for item in updates:
            requested_qty = max(0.0, float(item.get("close_size", 0.0) or 0.0))
            qty = self._round_size_to_precision(requested_qty)
            trigger_price = float(item.get("trigger_price", 0.0) or 0.0)
            rounded_trigger_price = self._round_price_to_wire_precision(trigger_price)
            tpsl = str(item.get("tpsl", "") or "sl")
            oid_or_cloid: Any = None
            oid_value = item.get("oid")
            if oid_value is not None:
                try:
                    oid_or_cloid = int(oid_value)
                except Exception:
                    oid_or_cloid = None
            if oid_or_cloid is None:
                cloid_raw = str(item.get("cloid", "") or "").strip()
                if cloid_raw:
                    oid_or_cloid = Cloid.from_str(cloid_raw)
            if oid_or_cloid is None or qty <= 0 or rounded_trigger_price <= 0:
                continue
            order_type = {"trigger": {"triggerPx": rounded_trigger_price, "isMarket": True, "tpsl": tpsl}}
            modify_requests.append(
                {
                    "oid": oid_or_cloid,
                    "order": {
                        "coin": self.symbol,
                        "is_buy": is_buy,
                        "sz": qty,
                        "limit_px": rounded_trigger_price,
                        "order_type": order_type,
                        "reduce_only": True,
                    },
                }
            )
            updated_refs.append(
                {
                    "key": str(item.get("key", "") or ""),
                    "name": str(item.get("name", "") or ""),
                    "leg_type": str(item.get("leg_type", "") or ""),
                    "tpsl": tpsl,
                    "trigger_price": rounded_trigger_price,
                    "close_size": qty,
                    "cloid": str(item.get("cloid", "") or ""),
                    "oid": oid_value,
                    "order_kind": "trigger",
                    "is_trigger": True,
                }
            )
        if not modify_requests:
            result["message"] = "Reduce-only TP/SL modify skipped due to missing identifiers or non-positive size/trigger price."
            result["accepted"] = False
            result["updated_refs"] = []
            return result
        exchange_result = self._exchange.bulk_modify_orders_new(modify_requests)
        result["actions"].append({"modify_reduce_only_tpsl_orders": exchange_result})
        result["accepted"] = not self._result_has_exchange_error(exchange_result)
        if result["accepted"]:
            try:
                statuses = (((exchange_result or {}).get("response") or {}).get("data") or {}).get("statuses") or []
                for idx, first_status in enumerate(statuses):
                    if idx >= len(updated_refs):
                        break
                    resting = first_status.get("resting") if isinstance(first_status, dict) else {}
                    if isinstance(resting, dict):
                        oid_value = resting.get("oid")
                        if oid_value is not None:
                            updated_refs[idx]["oid"] = int(oid_value)
                        resting_cloid = str(resting.get("cloid", "") or "").strip()
                        if resting_cloid:
                            updated_refs[idx]["cloid"] = resting_cloid
            except Exception:
                pass
        if not result["accepted"] and not str(result.get("message", "") or "").strip():
            result["message"] = "Exchange rejected risk order modification."
        result["updated_refs"] = updated_refs
        return result

    def _position_result(self, snapshot: dict, plan_name: Optional[str], reason: str) -> Dict[str, Any]:
        return {
            "mode": "live" if self.enabled else "dry_run",
            "symbol": self.symbol,
            "plan_name": plan_name,
            "reason": reason,
            "position_before": snapshot,
            "actions": [],
        }

    def reduce_position(
        self,
        side: str,
        close_size: float,
        reason: str,
        plan_name: Optional[str] = None,
        position_before: Optional[dict] = None,
    ) -> Dict[str, Any]:
        snapshot = position_before if isinstance(position_before, dict) else self.reader.get_position_snapshot(self.symbol)
        result = self._position_result(snapshot, plan_name, reason)
        current_sz = float(snapshot.get("size", 0.0) or 0.0)
        expected = 1 if side == "long" else -1
        if current_sz == 0 or (current_sz > 0 and expected < 0) or (current_sz < 0 and expected > 0):
            result["message"] = "No matching position to reduce."
            return result
        requested_qty = min(abs(current_sz), max(0.0, float(close_size or 0.0)))
        qty = self._round_size_to_precision(requested_qty)
        result["requested_close_size"] = requested_qty
        result["close_size"] = qty
        if qty <= 0:
            result["message"] = "Requested close size rounded to 0 after precision handling."
            return result
        if not self.enabled:
            result["message"] = "ENABLE_LIVE_TRADING=false, so reduce is dry-run only."
            return result
        self._ensure_exchange()
        result["actions"].append({"market_close": self._exchange.market_close(self.symbol, sz=qty, slippage=self.slippage)})
        return result

    def close_position(
        self,
        side: str,
        reason: str,
        plan_name: Optional[str] = None,
        position_before: Optional[dict] = None,
    ) -> Dict[str, Any]:
        snapshot = position_before if isinstance(position_before, dict) else self.reader.get_position_snapshot(self.symbol)
        current_sz = abs(float(snapshot.get("size", 0.0) or 0.0))
        return self.reduce_position(side, current_sz, reason, plan_name, position_before=snapshot)

    def _snapshot_side_notional(self, snapshot: dict) -> Tuple[str, float, float]:
        current_sz = float(snapshot.get("size", 0.0) or 0.0)
        current_side = "long" if current_sz > 0 else "short" if current_sz < 0 else "flat"
        current_notional = abs(float(snapshot.get("notional_usd", 0.0) or 0.0))
        return current_side, current_sz, current_notional

    def execute_position_target(
        self,
        *,
        target_side: str,
        target_notional_usd: float,
        requested_leverage: int,
        reason: str,
        plan_name: Optional[str] = None,
        position_before: Optional[dict] = None,
        execution_mid_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        snapshot = position_before if isinstance(position_before, dict) else self.reader.get_position_snapshot(self.symbol)
        current_side, current_sz, current_notional = self._snapshot_side_notional(snapshot)
        target_side = str(target_side or "flat").strip().lower()
        requested_target_notional_usd = max(0.0, float(target_notional_usd or 0.0))
        target_notional_usd = requested_target_notional_usd
        result = self._position_result(snapshot, plan_name, reason)
        result.update(
            {
                "current_side": current_side,
                "current_notional_usd": current_notional,
                "target_side": target_side,
                "requested_target_notional_usd": requested_target_notional_usd,
                "target_notional_usd": target_notional_usd,
                "requested_leverage": int(requested_leverage or 0),
            }
        )
        if target_side not in {"long", "short", "flat"}:
            result["message"] = f"Unsupported target side: {target_side or '<empty>'}."
            return result

        def apply_leverage_if_needed() -> None:
            leverage_update = self.apply_requested_leverage(int(requested_leverage or 0))
            if leverage_update:
                result["leverage_update"] = leverage_update

        def open_notional_delta(open_side: str, open_notional_usd: float) -> bool:
            result["open_notional_usd"] = max(0.0, float(open_notional_usd or 0.0))
            if result["open_notional_usd"] <= 0:
                result["message"] = "Target notional is 0, so no open order is needed."
                return False
            if not self.enabled:
                apply_leverage_if_needed()
                result["message"] = "ENABLE_LIVE_TRADING=false, so target adjustment is dry-run only."
                return False
            mid = safe_float(execution_mid_price, None)
            if mid is not None and mid > 0:
                result["mid_source"] = "execution_context"
            else:
                mid = self.reader.get_mid_price(self.symbol)
                result["mid_source"] = "fresh_mid_price"
            result["mid"] = mid
            if mid is None or mid <= 0:
                result["message"] = "Could not fetch valid mid price; skipped target open order."
                return False
            apply_leverage_if_needed()
            self._ensure_exchange()
            qty = self.usd_to_size(result["open_notional_usd"], mid)
            result["open_qty"] = qty
            if qty <= 0:
                result["message"] = "Target size rounded to 0 after precision handling; skipped."
                return False
            is_buy = open_side == "long"
            open_attempt = self._market_open_with_qty_backoff(is_buy=is_buy, qty=qty)
            final_qty = float(open_attempt.get("final_qty", qty) or 0.0)
            result["open_requested_qty"] = float(open_attempt.get("requested_qty", qty) or 0.0)
            result["open_qty"] = final_qty
            result["open_attempt_count"] = int(open_attempt.get("attempt_count", 0) or 0)
            result["open_attempted_qtys"] = [float(item or 0.0) for item in list(open_attempt.get("attempted_qtys", []) or [])]
            result["accepted"] = bool(open_attempt.get("accepted", False))
            if not result["accepted"]:
                result["message"] = str(open_attempt.get("message", "") or "Exchange rejected market open.")
            open_res = open_attempt.get("exchange_result", {})
            result["actions"].append({
                "open": open_res,
                "symbol": self.symbol,
                "is_buy": is_buy,
                "qty": final_qty,
                "requested_qty": result["open_requested_qty"],
                "attempt_count": result["open_attempt_count"],
                "attempted_qtys": result["open_attempted_qtys"],
                "mid": mid,
            })
            return True

        if target_side == "flat" or target_notional_usd <= 0:
            if current_side == "flat":
                result["message"] = "Already flat; no target adjustment needed."
                return result
            close_result = self.close_position(current_side, reason, plan_name, position_before=snapshot)
            close_result.update(
                {
                    "current_side": current_side,
                    "current_notional_usd": current_notional,
                    "target_side": "flat",
                    "target_notional_usd": 0.0,
                    "requested_leverage": int(requested_leverage or 0),
                }
            )
            return close_result

        if current_side == "flat":
            open_notional_delta(target_side, target_notional_usd)
            return result

        if current_side != target_side:
            result["reverse_order_notional_usd"] = current_notional + target_notional_usd
            open_notional_delta(target_side, result["reverse_order_notional_usd"])
            return result

        result["notional_delta_usd"] = target_notional_usd - current_notional
        if result["notional_delta_usd"] > 0:
            result["additional_notional_usd"] = result["notional_delta_usd"]
            open_notional_delta(target_side, result["notional_delta_usd"])
            return result
        if result["notional_delta_usd"] < 0:
            current_notional_safe = max(current_notional, 1e-12)
            target_fraction = max(0.0, min(1.0, target_notional_usd / current_notional_safe))
            requested_close_size = abs(current_sz) * max(0.0, 1.0 - target_fraction)
            reduce_result = self.reduce_position(current_side, requested_close_size, reason, plan_name, position_before=snapshot)
            reduce_result.update(
                {
                    "current_side": current_side,
                    "current_notional_usd": current_notional,
                    "target_side": target_side,
                    "target_notional_usd": target_notional_usd,
                    "requested_leverage": int(requested_leverage or 0),
                    "notional_delta_usd": result["notional_delta_usd"],
                }
            )
            return reduce_result

        apply_leverage_if_needed()
        result["message"] = "Position already matches target exposure."
        return result

    def execute(self, decision: StrategyDecision, plan_name: Optional[str] = None, trigger_confidence_raw: Optional[float] = None) -> Dict[str, Any]:
        position = self.reader.get_position_snapshot(self.symbol)
        current_side, _, _ = self._snapshot_side_notional(position)
        result: Dict[str, Any] = {
            "mode": "live" if self.enabled else "dry_run",
            "symbol": self.symbol,
            "plan_name": plan_name,
            "position_before": position,
            "decision": decision.to_dict(),
            "actions": [],
        }
        if decision.action == "no_trade":
            result["message"] = "No trade action."
            return result
        if decision.action not in {"long", "short"}:
            result["message"] = "Unsupported trade action."
            return result
        if current_side == "flat":
            initial_entry_price = decision.entry_price
            target_result = self.execute_position_target(
                target_side=decision.action,
                target_notional_usd=decision.suggested_notional_usd,
                requested_leverage=decision.requested_leverage,
                reason="entry_target_adjustment_from_flat",
                plan_name=plan_name,
            )
            target_result["decision"] = decision.to_dict()
            target_result["risk_plan"] = {
                "initial_entry_price": float(initial_entry_price or 0.0),
                "stop_loss_price": float(decision.stop_loss_price or 0.0),
            }
            return target_result
        initial_entry_price = position.get("mid_price")
        target_result = self.execute_position_target(
            target_side=decision.action,
            target_notional_usd=decision.suggested_notional_usd,
            requested_leverage=decision.requested_leverage,
            reason="entry_target_adjustment",
            plan_name=plan_name,
        )
        target_result["decision"] = decision.to_dict()
        target_result["risk_plan"] = {
            "initial_entry_price": float(initial_entry_price or 0.0),
            "stop_loss_price": float(decision.stop_loss_price or 0.0),
        }
        return target_result

    def execute_management(
        self,
        decision: ManagementDecision,
        plan_name: Optional[str] = None,
        trigger_confidence_raw: Optional[float] = None,
        position_before: Optional[dict] = None,
        execution_mid_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        snapshot = position_before if isinstance(position_before, dict) else self.reader.get_position_snapshot(self.symbol)
        current_side, _, current_notional = self._snapshot_side_notional(snapshot)
        result: Dict[str, Any] = {
            "mode": "live" if self.enabled else "dry_run",
            "symbol": self.symbol,
            "plan_name": plan_name,
            "position_before": snapshot,
            "decision": decision.to_dict(),
            "actions": [],
        }
        if decision.action == "no_change":
            leverage_update = self.apply_requested_leverage(int(decision.leverage or 0))
            if leverage_update:
                result["leverage_update"] = leverage_update
                result["message"] = "Leverage updated without changing position size."
            else:
                result["message"] = "No immediate management action."
            return result
        if decision.action in {"long", "short"}:
            target_result = self.execute_position_target(
                target_side=decision.action,
                target_notional_usd=decision.new_notional_usd,
                requested_leverage=decision.leverage,
                reason="management_target_adjustment_from_flat",
                plan_name=plan_name,
                position_before=snapshot,
                execution_mid_price=execution_mid_price,
            )
            target_result["decision"] = decision.to_dict()
            return target_result
        if current_side == "flat" and decision.action not in MANAGEMENT_EXPOSURE_ACTION_VALUES:
            result["message"] = "No open position to manage."
            return result
        if decision.action == "close":
            target_result = self.execute_position_target(
                target_side="flat",
                target_notional_usd=0.0,
                requested_leverage=0,
                reason="management_close",
                plan_name=plan_name,
                position_before=snapshot,
                execution_mid_price=execution_mid_price,
            )
            target_result["decision"] = decision.to_dict()
            return target_result
        if decision.action == "trim":
            target_notional = current_notional * max(0.0, 1.0 - float(decision.close_fraction or 0.0))
            target_result = self.execute_position_target(
                target_side=current_side,
                target_notional_usd=target_notional,
                requested_leverage=decision.leverage,
                reason="management_trim",
                plan_name=plan_name,
                position_before=snapshot,
                execution_mid_price=execution_mid_price,
            )
            target_result["decision"] = decision.to_dict()
            target_result["trim_close_fraction"] = float(decision.close_fraction or 0.0)
            return target_result
        if decision.action in {"add_to_long", "add_to_short"}:
            target_side = "long" if decision.action == "add_to_long" else "short"
            if current_side != target_side:
                result["message"] = f"Cannot {decision.action} without an existing {target_side} position."
                return result
            target_result = self.execute_position_target(
                target_side=target_side,
                target_notional_usd=decision.new_notional_usd,
                requested_leverage=decision.leverage,
                reason="management_add_on",
                plan_name=plan_name,
                position_before=snapshot,
                execution_mid_price=execution_mid_price,
            )
            target_result["decision"] = decision.to_dict()
            return target_result
        if decision.action in {"reverse_to_long", "reverse_to_short"}:
            target_side = "long" if decision.action == "reverse_to_long" else "short"
            target_result = self.execute_position_target(
                target_side=target_side,
                target_notional_usd=decision.new_notional_usd,
                requested_leverage=decision.leverage,
                reason="management_reverse",
                plan_name=plan_name,
                position_before=snapshot,
                execution_mid_price=execution_mid_price,
            )
            target_result["decision"] = decision.to_dict()
            return target_result
        result["message"] = "Unsupported management action."
        return result
