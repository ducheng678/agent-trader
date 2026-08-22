from __future__ import annotations

import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cross_asset.trend_fetcher import (
    CandleBar,
    CrossAssetTrendFetcher,
    DatabentoHistoricalClient,
    FinnhubQuoteClient,
    HyperliquidInfoClient,
    SYNTHETIC_DXY_COMPONENTS,
    SYNTHETIC_DXY_SCALE,
    build_synthetic_dxy_bars,
    build_trend_snapshot,
    detect_special_source,
    pick_best_market_match,
    resample_bars,
)


def _bar(ts_ms: int, open_px: float, high_px: float, low_px: float, close_px: float, volume: float = 0.0, source_symbol: str = "TEST") -> CandleBar:
    return CandleBar(
        ts_ms=ts_ms,
        open=open_px,
        high=high_px,
        low=low_px,
        close=close_px,
        volume=volume,
        source_symbol=source_symbol,
    )


def _dxy_price(values: dict[str, float]) -> float:
    level = SYNTHETIC_DXY_SCALE
    for pair, weight in SYNTHETIC_DXY_COMPONENTS.items():
        level *= float(values[pair]) ** weight
    return level


def test_resample_bars_aggregates_ohlcv_and_volume() -> None:
    bars = [
        _bar(0, 100.0, 101.0, 99.5, 100.8, 1.0),
        _bar(60_000, 100.8, 102.2, 100.4, 101.9, 2.0),
        _bar(120_000, 101.9, 103.1, 101.1, 102.4, 3.0),
        _bar(180_000, 102.4, 104.7, 102.0, 104.0, 4.0),
        _bar(240_000, 104.0, 105.5, 103.8, 105.1, 5.0),
    ]

    out = resample_bars(bars, "5m")

    assert len(out) == 1
    merged = out[0]
    assert merged.open == pytest.approx(100.0)
    assert merged.high == pytest.approx(105.5)
    assert merged.low == pytest.approx(99.5)
    assert merged.close == pytest.approx(105.1)
    assert merged.volume == pytest.approx(15.0)


def test_build_synthetic_dxy_bars_uses_weight_aware_extremes() -> None:
    ts_ms = 1_700_000_000_000
    series = {
        "EUR/USD": [_bar(ts_ms, 1.10, 1.20, 1.00, 1.15, source_symbol="EUR/USD")],
        "USD/JPY": [_bar(ts_ms, 150.0, 151.0, 149.0, 150.5, source_symbol="USD/JPY")],
    }

    out = build_synthetic_dxy_bars(series)

    assert len(out) == 1
    bar = out[0]
    expected_open = _dxy_price({pair: candles[0].open for pair, candles in series.items()})
    expected_close = _dxy_price({pair: candles[0].close for pair, candles in series.items()})
    expected_high = _dxy_price(
        {
            pair: (candles[0].high if SYNTHETIC_DXY_COMPONENTS[pair] >= 0 else candles[0].low)
            for pair, candles in series.items()
        }
    )
    expected_low = _dxy_price(
        {
            pair: (candles[0].low if SYNTHETIC_DXY_COMPONENTS[pair] >= 0 else candles[0].high)
            for pair, candles in series.items()
        }
    )

    assert bar.open == pytest.approx(expected_open)
    assert bar.close == pytest.approx(expected_close)
    assert bar.high == pytest.approx(max(expected_high, expected_open, expected_close))
    assert bar.low == pytest.approx(min(expected_low, expected_open, expected_close))


def test_pick_best_market_match_prefers_brent_for_generic_oil() -> None:
    market_specs = [
        {"execution_symbol": "xyz:BRENTOIL", "market_name": "BRENTOIL", "display_name": "BRENTOIL-USDC"},
        {"execution_symbol": "ETH", "market_name": "ETH", "display_name": "ETH-USDC"},
    ]

    match = pick_best_market_match("oil", market_specs, hint_tokens=("BRENTOIL", "BRENT"))

    assert match is not None
    assert match["execution_symbol"] == "xyz:BRENTOIL"


def test_build_trend_snapshot_adds_inverse_fields_for_us10y_proxy() -> None:
    bars = [
        _bar(0, 100.0, 100.2, 99.9, 100.1),
        _bar(60_000, 100.1, 100.6, 100.0, 100.4),
        _bar(120_000, 100.4, 100.9, 100.3, 100.8),
    ]

    snapshot = build_trend_snapshot(
        requested_symbol="US10Y",
        resolved_symbol="US10Y",
        source="databento",
        source_symbol="ZN.v.0",
        base_1m_bars=bars,
        timeframes=["1m"],
        bars_per_timeframe=3,
        include_bars=False,
        flat_threshold_pct=0.01,
        inverse_price_relation=True,
    )

    summary = snapshot["timeframes"]["1m"]
    assert summary["direction"] == "up"
    assert summary["inverse_direction"] == "down"
    assert summary["inverse_change_pct"] == pytest.approx(-summary["change_pct"])


def test_build_trend_snapshot_marks_quote_only_resamples_unknown() -> None:
    bars = [
        _bar(60_000, 20.0, 20.0, 20.0, 20.0),
    ]
    bars[0].quote_only = True

    snapshot = build_trend_snapshot(
        requested_symbol="QUOTEONLY",
        resolved_symbol="QUOTEONLY",
        source="test_quote",
        source_symbol="QUOTEONLY",
        base_1m_bars=bars,
        timeframes=["1m", "5m", "1h"],
        bars_per_timeframe=3,
        include_bars=False,
        flat_threshold_pct=0.01,
    )

    for timeframe in ["1m", "5m", "1h"]:
        summary = snapshot["timeframes"][timeframe]
        assert summary["quote_only"] is True
        assert summary["change_pct"] is None
        assert summary["direction"] == "unknown"


def test_detect_special_source_routes_macro_aliases() -> None:
    assert detect_special_source("DXY") == "dxy"
    assert detect_special_source("US2Y") == "us2y"
    assert detect_special_source("UST2Y") == "us2y"
    assert detect_special_source("ZT.v.0") == "us2y"
    assert detect_special_source("US10Y") == "us10y"
    assert detect_special_source("ZN.v.0") == "us10y"
    assert detect_special_source("USDCAD") == "twelvedata_forex"
    assert detect_special_source("CPER") == "finnhub_quote"
    assert detect_special_source("IWM") == "finnhub_quote"
    assert detect_special_source("VIXY") == "finnhub_quote"
    assert detect_special_source("^VIX") == "vix_spot_unsupported"
    assert detect_special_source("VIX") == "vix_spot_unsupported"
    assert detect_special_source("BTC-USDC") == "hyperliquid"


def test_finnhub_quote_client_builds_quote_only_bar() -> None:
    class FakeResponse:
        def json(self):
            return {
                "c": 80.4,
                "d": -0.11,
                "dp": -0.1366,
                "h": 80.43,
                "l": 80.325,
                "o": 80.3425,
                "pc": 80.51,
                "t": 1_700_000_123,
            }

        def raise_for_status(self) -> None:
            return None

    class FakeSession:
        def __init__(self) -> None:
            self.calls: list[dict] = []

        def get(self, url: str, params: dict, timeout=None):
            self.calls.append({"url": url, "params": dict(params), "timeout": timeout})
            return FakeResponse()

    client = FinnhubQuoteClient()
    client.api_key = "test-token"
    client.session = FakeSession()

    bars, meta = client.fetch_quote_bar("hyg")

    assert client.session.calls[0]["params"]["symbol"] == "HYG"
    assert client.session.calls[0]["params"]["token"] == "test-token"
    assert len(bars) == 1
    assert bars[0].close == pytest.approx(80.4)
    assert bars[0].quote_only is True
    assert meta["provider"] == "finnhub"
    assert meta["quote_only"] is True
    assert meta["quote_change_pct"] == pytest.approx(-0.1366)
    assert meta["quote_previous_close"] == pytest.approx(80.51)


def test_fetch_symbol_trends_uses_finnhub_quote_for_etf_symbols() -> None:
    class FakeFinnhub:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def fetch_quote_bar(self, symbol: str):
            self.seen.append(symbol)
            bar = _bar(60_000, 25.25, 25.25, 25.25, 25.25, source_symbol=symbol)
            bar.quote_only = True
            return [bar], {"provider": "finnhub", "quote_only": True, "quote_symbol": symbol}

    finnhub = FakeFinnhub()
    fetcher = CrossAssetTrendFetcher(
        hyperliquid_client=object(),
        twelvedata_client=object(),
        databento_client=object(),
        finnhub_client=finnhub,
    )

    snapshot = fetcher.fetch_symbol_trends("JETS", timeframes=["1m", "5m"], bars_per_timeframe=3, include_bars=False)

    assert finnhub.seen == ["JETS"]
    assert snapshot["source"] == "finnhub_quote"
    assert snapshot["source_symbol"] == "JETS"
    assert snapshot["market_meta"]["quote_symbol"] == "JETS"
    assert snapshot["timeframes"]["1m"]["quote_only"] is True
    assert snapshot["timeframes"]["1m"]["direction"] == "unknown"
    assert snapshot["timeframes"]["5m"]["quote_only"] is True
    assert snapshot["timeframes"]["5m"]["change_pct"] is None


def test_fetch_symbol_trends_uses_twelvedata_forex_for_usdcad() -> None:
    class FakeTwelveData:
        def __init__(self) -> None:
            self.seen: list[tuple[str, int]] = []

        def fetch_1m_bars(self, symbol: str, bar_count: int) -> list[CandleBar]:
            self.seen.append((symbol, bar_count))
            return [
                _bar(0, 1.37, 1.371, 1.369, 1.3705, source_symbol=symbol),
                _bar(60_000, 1.3705, 1.372, 1.370, 1.3715, source_symbol=symbol),
            ]

    twelvedata = FakeTwelveData()
    fetcher = CrossAssetTrendFetcher(
        hyperliquid_client=object(),
        twelvedata_client=twelvedata,
        databento_client=object(),
        finnhub_client=object(),
    )

    snapshot = fetcher.fetch_symbol_trends("USDCAD", timeframes=["1m"], bars_per_timeframe=2, include_bars=False)

    assert twelvedata.seen[0][0] == "USD/CAD"
    assert snapshot["source"] == "twelvedata_forex"
    assert snapshot["source_symbol"] == "USD/CAD"
    assert snapshot["resolved_symbol"] == "USDCAD"
    assert snapshot["market_meta"]["provider"] == "twelvedata"


def test_fetch_symbol_trends_uses_finnhub_quote_for_vixy_proxy() -> None:
    class FakeFinnhub:
        def __init__(self) -> None:
            self.seen: list[str] = []

        def fetch_quote_bar(self, symbol: str):
            self.seen.append(symbol)
            bar = _bar(60_000, 18.75, 18.75, 18.75, 18.75, source_symbol=symbol)
            bar.quote_only = True
            return [bar], {"provider": "finnhub", "quote_only": True, "quote_symbol": symbol}

    finnhub = FakeFinnhub()
    fetcher = CrossAssetTrendFetcher(
        hyperliquid_client=object(),
        twelvedata_client=object(),
        databento_client=object(),
        finnhub_client=finnhub,
    )

    snapshot = fetcher.fetch_symbol_trends("VIXY", timeframes=["1m"], bars_per_timeframe=3, include_bars=False)

    assert finnhub.seen == ["VIXY"]
    assert snapshot["source"] == "finnhub_quote"
    assert snapshot["market_meta"]["proxy_for"] == "VIX short-term futures / equity volatility sentiment"
    assert snapshot["market_meta"]["proxy_note"] == "Not spot VIX. Tracks short-term VIX futures exposure."
    assert "Not spot VIX" in " ".join(snapshot["notes"])


def test_fetch_symbol_trends_rejects_vix_spot_without_proxy() -> None:
    fetcher = CrossAssetTrendFetcher(
        hyperliquid_client=object(),
        twelvedata_client=object(),
        databento_client=object(),
        finnhub_client=object(),
    )

    with pytest.raises(RuntimeError, match="Request VIXY"):
        fetcher.fetch_symbol_trends("VIX", timeframes=["1m"], bars_per_timeframe=3, include_bars=False)


def test_fetch_symbol_trends_uses_zt_for_us2y_proxy() -> None:
    class FakeDatabento:
        def __init__(self) -> None:
            self.seen: list[tuple[str, int]] = []

        def fetch_1m_bars(self, symbol: str, bar_count: int) -> list[CandleBar]:
            self.seen.append((symbol, bar_count))
            return [
                _bar(0, 100.0, 100.2, 99.9, 100.1, source_symbol=symbol),
                _bar(60_000, 100.1, 100.5, 100.0, 100.4, source_symbol=symbol),
            ]

    databento = FakeDatabento()
    fetcher = CrossAssetTrendFetcher(databento_client=databento)

    snapshot = fetcher.fetch_symbol_trends("UST2Y", timeframes=["1m"], bars_per_timeframe=2, include_bars=False)

    assert databento.seen[0][0] == "ZT.v.0"
    assert snapshot["resolved_symbol"] == "US2Y"
    assert snapshot["source_symbol"] == "ZT.v.0"
    assert snapshot["market_meta"]["proxy_symbol"] == "ZT.v.0"
    assert snapshot["timeframes"]["1m"]["inverse_direction"] == "down"


def test_fetch_trade_bundle_includes_macro_refs(monkeypatch: pytest.MonkeyPatch) -> None:
    fetcher = CrossAssetTrendFetcher()

    def fake_fetch(symbol_name: str, **kwargs):
        return {"requested_symbol": symbol_name, "kwargs": kwargs}

    monkeypatch.setattr(fetcher, "fetch_symbol_trends", fake_fetch)

    bundle = fetcher.fetch_trade_bundle("BTC-USDC", include_macro=True, include_bars=False)

    assert bundle["primary"]["requested_symbol"] == "BTC-USDC"
    assert bundle["macro"]["DXY"]["requested_symbol"] == "DXY"
    assert bundle["macro"]["US2Y"]["requested_symbol"] == "US2Y"
    assert bundle["macro"]["US10Y"]["requested_symbol"] == "US10Y"


def test_databento_client_retries_into_accessible_window() -> None:
    class FakeResponse:
        def __init__(self, status_code: int, payload: dict | None = None, text: str = "") -> None:
            self.status_code = status_code
            self._payload = payload or {}
            self.text = text

        def json(self):
            return self._payload

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"http {self.status_code}")

    class FakeSession:
        def __init__(self) -> None:
            self.calls: list[dict[str, str]] = []

        def post(self, url: str, data: dict[str, str], auth=None, timeout=None):
            self.calls.append(dict(data))
            if len(self.calls) == 1:
                return FakeResponse(
                    422,
                    {
                        "detail": {
                            "payload": {
                                "available_end": "2026-04-17T17:40:00.000000000Z",
                            }
                        }
                    },
                )
            if len(self.calls) == 2:
                return FakeResponse(
                    422,
                    {
                        "detail": {
                            "payload": {
                                "available_end": "2026-04-17T09:49:36.825715000Z",
                            }
                        }
                    },
                )
            return FakeResponse(
                200,
                text='{"symbol":"ZN.v.0","ts_event":"2026-04-17T09:48:00Z","open":111.0,"high":111.5,"low":110.9,"close":111.3,"volume":100}\n'
                     '{"symbol":"ZN.v.0","ts_event":"2026-04-17T09:49:00Z","open":111.3,"high":111.4,"low":111.2,"close":111.35,"volume":120}',
            )

    client = DatabentoHistoricalClient()
    client.api_key = "test-key"
    client.session = FakeSession()

    bars = client.fetch_1m_bars("ZN.v.0", 2)

    assert len(client.session.calls) == 3
    assert len(bars) == 2
    assert bars[-1].close == pytest.approx(111.35)


def test_hyperliquid_catalog_normalizes_dex_market_names() -> None:
    client = HyperliquidInfoClient()
    client._get_perp_dex_names = lambda: ["xyz"]
    client._get_perp_meta = lambda dex="": {"universe": [{"name": "xyz:BRENTOIL", "szDecimals": 2, "maxLeverage": 20, "onlyIsolated": True}]}

    catalog = client.get_market_catalog()

    assert "xyz:BRENTOIL".upper() in catalog
    item = catalog["xyz:BRENTOIL".upper()]
    assert item["execution_symbol"] == "xyz:BRENTOIL"
    assert item["execution_symbol"] != "xyz:XYZ:BRENTOIL"
    assert item["market_name"] == "BRENTOIL"
    assert item["display_name"] == "BRENTOIL-USDC"


def test_hyperliquid_fetch_preserves_dex_symbol_case() -> None:
    client = HyperliquidInfoClient()
    seen_coins: list[str] = []
    client.resolve_market = lambda raw_symbol: {
        "execution_symbol": "xyz:BRENTOIL",
        "market_name": "BRENTOIL",
        "display_name": "BRENTOIL-USDC",
    }

    def fake_post_info(payload: dict):
        seen_coins.append(str(((payload or {}).get("req") or {}).get("coin") or ""))
        if seen_coins[-1] == "xyz:BRENTOIL":
            return [{"t": 1, "o": "1", "h": "2", "l": "0.5", "c": "1.5", "v": "10"}]
        raise AssertionError(f"unexpected coin probe: {seen_coins[-1]}")

    client._post_info = fake_post_info

    bars, market = client.fetch_1m_bars("BRENTOIL-USDC", 2)

    assert seen_coins[0] == "xyz:BRENTOIL"
    assert len(bars) == 1
    assert market["execution_symbol"] == "xyz:BRENTOIL"


def test_hyperliquid_fetch_falls_back_to_validated_quote_snapshot() -> None:
    client = HyperliquidInfoClient()
    client.resolve_market = lambda raw_symbol: {
        "execution_symbol": "xyz:TESTQUOTE",
        "dex": "xyz",
        "market_name": "TESTQUOTE",
        "display_name": "TESTQUOTE-USDC",
    }

    def fake_post_info(payload: dict):
        if payload.get("type") == "candleSnapshot":
            raise RuntimeError("candle unavailable")
        if payload.get("type") == "metaAndAssetCtxs" and payload.get("dex") == "xyz":
            return [
                {"universe": [{"name": "xyz:TESTQUOTE"}]},
                [
                    {
                        "markPx": "20.0",
                        "oraclePx": "20.0",
                        "prevDayPx": "20.0",
                        "openInterest": "0.0",
                        "dayBaseVlm": "0.0",
                        "dayNtlVlm": "0.0",
                    }
                ],
            ]
        raise AssertionError(f"unexpected payload: {payload}")

    client._post_info = fake_post_info

    bars, market = client.fetch_1m_bars("TESTQUOTE", 2)

    assert len(bars) == 1
    assert bars[0].close == pytest.approx(20.0)
    assert bars[0].quote_only is True
    assert market["price_source"] == "metaAndAssetCtxs"
    assert market["quote_only"] is True
    assert market["quote_price_field"] == "markPx"
    assert market["quote_is_delisted"] is False
    assert market["quote_max_reference_deviation_pct"] == pytest.approx(0.0)


def test_hyperliquid_quote_fallback_rejects_inconsistent_prices() -> None:
    client = HyperliquidInfoClient()
    client.resolve_market = lambda raw_symbol: {
        "execution_symbol": "xyz:TESTQUOTE",
        "dex": "xyz",
        "market_name": "TESTQUOTE",
        "display_name": "TESTQUOTE-USDC",
    }

    def fake_post_info(payload: dict):
        if payload.get("type") == "candleSnapshot":
            raise RuntimeError("candle unavailable")
        return [
            {"universe": [{"name": "xyz:TESTQUOTE"}]},
            [{"markPx": "30.0", "oraclePx": "20.0"}],
        ]

    client._post_info = fake_post_info

    with pytest.raises(RuntimeError, match="Rejected quote fallback"):
        client.fetch_1m_bars("TESTQUOTE", 2)


def test_hyperliquid_rejects_vix_market_directly() -> None:
    client = HyperliquidInfoClient()
    client._market_catalog = {
        "XYZ:VIX": {"execution_symbol": "xyz:VIX", "dex": "xyz", "market_name": "VIX", "display_name": "VIX-USDC"},
    }

    with pytest.raises(KeyError, match="Hyperliquid VIX market is disabled"):
        client.resolve_market("VIX")

    with pytest.raises(KeyError, match="Hyperliquid VIX market is disabled"):
        client.resolve_market("xyz:VIX")

    with pytest.raises(KeyError, match="Hyperliquid VIX market is disabled"):
        client.fetch_1m_bars("VIX", 2)


def test_hyperliquid_resolve_market_uses_builtin_aliases_and_prefers_xyz() -> None:
    client = HyperliquidInfoClient()
    client._market_catalog = {
        "CASH:GOLD": {"execution_symbol": "cash:GOLD", "dex": "cash", "market_name": "GOLD", "display_name": "GOLD-USDC"},
        "XYZ:GOLD": {"execution_symbol": "xyz:GOLD", "dex": "xyz", "market_name": "GOLD", "display_name": "GOLD-USDC"},
        "XYZ:SILVER": {"execution_symbol": "xyz:SILVER", "dex": "xyz", "market_name": "SILVER", "display_name": "SILVER-USDC"},
        "XYZ:EUR": {"execution_symbol": "xyz:EUR", "dex": "xyz", "market_name": "EUR", "display_name": "EUR-USDC"},
        "XYZ:XYZ100": {"execution_symbol": "xyz:XYZ100", "dex": "xyz", "market_name": "XYZ100", "display_name": "XYZ100-USDC"},
        "XYZ:CL": {"execution_symbol": "xyz:CL", "dex": "xyz", "market_name": "CL", "display_name": "CL-USDC"},
        "XYZ:BRENTOIL": {"execution_symbol": "xyz:BRENTOIL", "dex": "xyz", "market_name": "BRENTOIL", "display_name": "BRENTOIL-USDC"},
        "XYZ:NATGAS": {"execution_symbol": "xyz:NATGAS", "dex": "xyz", "market_name": "NATGAS", "display_name": "NATGAS-USDC"},
        "XYZ:COPPER": {"execution_symbol": "xyz:COPPER", "dex": "xyz", "market_name": "COPPER", "display_name": "COPPER-USDC"},
        "SPX": {"execution_symbol": "SPX", "dex": "", "market_name": "SPX", "display_name": "SPX-USDC"},
        "XYZ:SP500": {"execution_symbol": "xyz:SP500", "dex": "xyz", "market_name": "SP500", "display_name": "SP500-USDC"},
        "KM:SMALL2000": {"execution_symbol": "km:SMALL2000", "dex": "km", "market_name": "SMALL2000", "display_name": "SMALL2000-USDC"},
    }

    assert client.resolve_market("XAUUSD")["execution_symbol"] == "xyz:GOLD"
    assert client.resolve_market("GC=F")["execution_symbol"] == "xyz:GOLD"
    assert client.resolve_market("XAGUSD")["execution_symbol"] == "xyz:SILVER"
    assert client.resolve_market("EURUSD")["execution_symbol"] == "xyz:EUR"
    assert client.resolve_market("CL=F")["execution_symbol"] == "xyz:CL"
    assert client.resolve_market("BZ=F")["execution_symbol"] == "xyz:BRENTOIL"
    assert client.resolve_market("NG")["execution_symbol"] == "xyz:NATGAS"
    assert client.resolve_market("HG")["execution_symbol"] == "xyz:COPPER"
    assert client.resolve_market("ES=F")["execution_symbol"] == "xyz:SP500"
    assert client.resolve_market("SPX")["execution_symbol"] == "xyz:SP500"
    assert client.resolve_market("NDX")["execution_symbol"] == "xyz:XYZ100"
    assert client.resolve_market("NQ=F")["execution_symbol"] == "xyz:XYZ100"
    assert client.resolve_market("RTY")["execution_symbol"] == "km:SMALL2000"
    assert client.resolve_market("GOLD")["execution_symbol"] == "xyz:GOLD"


def test_hyperliquid_resolve_market_rejects_ambiguous_shorthands() -> None:
    client = HyperliquidInfoClient()
    client._market_catalog = {
        "CASH:HOOD": {"execution_symbol": "cash:HOOD", "dex": "cash", "market_name": "HOOD", "display_name": "HOOD-USDC"},
        "ARB": {"execution_symbol": "ARB", "dex": "", "market_name": "ARB", "display_name": "ARB-USDC"},
    }

    with pytest.raises(KeyError):
        client.resolve_market("HO")

    with pytest.raises(KeyError):
        client.resolve_market("RB")
