from __future__ import annotations

import json
import io
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _minimal_playbook_payload():
    return {
        "trigger_event_relevance": "not_applicable",
        "trigger_confidence": None,
        "playbook": {
            "entry_plan": {
                "execute_now": False,
                "action_decision": {
                    "action": "no_trade",
                    "entry_price": 0.0,
                    "stop_loss_price": 0.0,
                },
                "scenario": None,
            },
        },
    }


def _passive_event_judge_payload(*, relevance="relevant", confidence=0.42, action="long"):
    return {
        "trigger_event_relevance": relevance,
        "trigger_confidence": confidence,
        "action": action,
    }


def _passive_technical_pricing_payload(*, entry_price=100000.0, stop_loss_price=99200.0):
    return {
        "entry_price": entry_price,
        "stop_loss_price": stop_loss_price,
    }


def _build_candles(start_ms: int, interval_ms: int, count: int, base_price: float):
    candles = []
    for idx in range(count):
        open_px = base_price + idx * 0.2
        close_px = open_px + (0.15 if idx % 2 == 0 else -0.1)
        high_px = max(open_px, close_px) + 0.25
        low_px = min(open_px, close_px) - 0.2
        candles.append(
            {
                "t": start_ms + idx * interval_ms,
                "o": f"{open_px:.4f}",
                "h": f"{high_px:.4f}",
                "l": f"{low_px:.4f}",
                "c": f"{close_px:.4f}",
                "v": f"{1000 + idx * 10:.2f}",
            }
        )
    return candles


def test_passive_relevance_threshold_env_controls_local_gate_without_prompt_coupling(uma, monkeypatch):
    class FakeOpenAI:
        def __init__(self, api_key):
            self.api_key = api_key
            self.responses = SimpleNamespace()

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("PASSIVE_RELEVANCE_THRESHOLD", "0.35")
    monkeypatch.setattr(sys.modules["openai"], "OpenAI", FakeOpenAI)

    engine = uma.DiscretionaryLLMEngine()
    prompt = engine._build_passive_event_judge_prompt("fast")

    assert engine.passive_relevance_threshold == pytest.approx(0.35)
    assert "If trigger_event has no direct effect on trade_symbol" in prompt
    assert "If trigger_confidence is below 0.35" not in prompt


def test_validate_passive_event_judge_leaves_relevance_threshold_to_local_gate():
    from market_agent.llm_engine import validate_passive_event_judge

    assert validate_passive_event_judge(
        _passive_event_judge_payload(relevance="relevant", confidence=0.34, action="long"),
        relevance_threshold=0.35,
    )["action"] == "long"
    assert validate_passive_event_judge(
        _passive_event_judge_payload(relevance="unrelated", confidence=0.35, action="no_trade"),
        relevance_threshold=0.35,
    )["trigger_event_relevance"] == "unrelated"
    with pytest.raises(ValueError, match="must set action to no_trade"):
        validate_passive_event_judge(
            _passive_event_judge_payload(relevance="unrelated", confidence=0.35, action="long"),
            relevance_threshold=0.35,
        )


def test_hyperliquid_executor_size_precision_uses_decimal_floor(uma, monkeypatch):
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "false")
    reader = SimpleNamespace(
        get_sz_decimals=lambda symbol: 2,
        get_position_snapshot=lambda symbol: {
            "symbol": symbol,
            "side": "short",
            "size": -9.2,
            "entry_price": 107.24,
            "mid_price": 108.12,
            "notional_usd": 994.704,
        },
    )
    executor = uma.HyperliquidExecutor(reader, "xyz:BRENTOIL")

    assert executor._round_size_to_precision(9.2) == pytest.approx(9.2)
    assert executor._round_size_to_precision(abs(-9.2)) == pytest.approx(9.2)
    assert executor._round_size_to_precision(9.209) == pytest.approx(9.20)
    assert executor.usd_to_size(986.6999999999999, 107.25) == pytest.approx(9.20)

    stop_order = executor.place_reduce_only_tpsl_order(
        side="short",
        close_size=9.2,
        trigger_price=108.12,
        tpsl="sl",
        plan_name="position_management",
        leg_name="stage_initial_stop",
    )

    assert stop_order["requested_close_size"] == pytest.approx(9.2)
    assert stop_order["close_size"] == pytest.approx(9.2)


def test_hyperliquid_executor_places_reduce_only_limit_take_profit(uma, monkeypatch):
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")
    reader = SimpleNamespace(
        get_sz_decimals=lambda symbol: 2,
        get_position_snapshot=lambda symbol: {
            "symbol": symbol,
            "side": "long",
            "size": 8.2,
            "entry_price": 110.0,
            "mid_price": 110.8,
            "notional_usd": 908.56,
        },
    )
    executor = uma.HyperliquidExecutor(reader, "xyz:BRENTOIL")

    class FakeExchange:
        def __init__(self):
            self.orders = []

        def order(self, coin, is_buy, qty, limit_price, order_type, reduce_only=False, cloid=None):
            self.orders.append(
                {
                    "coin": coin,
                    "is_buy": is_buy,
                    "qty": qty,
                    "limit_price": limit_price,
                    "order_type": order_type,
                    "reduce_only": reduce_only,
                    "cloid": str(cloid),
                }
            )
            return {
                "status": "ok",
                "response": {
                    "data": {
                        "statuses": [
                            {
                                "resting": {
                                    "oid": 12345,
                                    "cloid": str(cloid),
                                }
                            }
                        ]
                    }
                },
            }

    fake_exchange = FakeExchange()
    executor._exchange = fake_exchange

    take_profit_order = executor.place_reduce_only_limit_order(
        side="long",
        close_size=1.234,
        limit_price=111.13,
        plan_name="position_management",
        leg_name="stage_tp1",
    )

    assert take_profit_order["accepted"] is True
    assert take_profit_order["order_kind"] == "limit"
    assert take_profit_order["is_trigger"] is False
    assert take_profit_order["close_size"] == pytest.approx(1.23)
    assert take_profit_order["limit_price"] == pytest.approx(111.13)
    assert take_profit_order["oid"] == 12345
    assert fake_exchange.orders == [
        {
            "coin": "xyz:BRENTOIL",
            "is_buy": False,
            "qty": pytest.approx(1.23),
            "limit_price": pytest.approx(111.13),
            "order_type": {"limit": {"tif": "Gtc"}},
            "reduce_only": True,
            "cloid": take_profit_order["cloid"],
        }
    ]


def test_event_file_watcher_skips_malformed_lines_and_recovers_after_truncate(uma, tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text('{"id": 1, "headline": "old"}\n', encoding="utf-8")

    watcher = uma.EventFileWatcher(path, start_from="end", max_recent=10)

    with path.open("a", encoding="utf-8") as handle:
        handle.write("not-json\n")
        handle.write('{"id": 2, "headline": "new"}\n')

    polled = watcher.poll()

    assert [event["id"] for event in polled] == [2]
    assert [event["id"] for event in watcher.recent()] == [1, 2]

    path.write_text('{"id": 3, "headline": "after-truncate"}\n', encoding="utf-8")
    polled_after_truncate = watcher.poll()

    assert [event["id"] for event in polled_after_truncate] == [3]


def test_event_file_watcher_recent_uses_time_window_and_enriches_timestamps(uma, tmp_path, monkeypatch):
    monkeypatch.setattr(uma, "current_utc_iso", lambda: "2026-03-31T12:00:00Z")
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps({"id": 1, "headline": "stale", "published_at": "2026-03-27T11:00:00Z"}, ensure_ascii=False),
                json.dumps({"id": 2, "headline": "fresh", "published_at": "2026-03-30T13:30:00.000Z"}, ensure_ascii=False),
                json.dumps({"id": 3, "headline": "fallback-ingest-only"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    watcher = uma.EventFileWatcher(
        path,
        start_from="beginning",
        max_recent=50,
        recent_window_hours=72,
        max_context_items=50,
    )

    recent = watcher.recent()

    assert [event["id"] for event in recent] == [2, 3]
    assert recent[0]["event_timestamp"] == "2026-03-30T13:30:00Z"
    assert recent[0]["event_time_source"] == "published_at"
    assert recent[0]["seen_at"] == "2026-03-31T12:00:00Z"
    assert recent[0]["is_within_recent_window"] is True
    assert recent[0]["event_age_hours"] == pytest.approx(22.5)
    assert recent[1]["event_timestamp"] == "2026-03-31T12:00:00Z"
    assert recent[1]["event_time_source"] == "event_timestamp"
    assert recent[1]["event_age_hours"] == pytest.approx(0.0)


def test_rest_reader_get_all_positions_extracts_account_and_position_fields(uma):
    reader = object.__new__(uma.HyperliquidRestReader)
    reader.account_address = "0xabc"
    reader.network = "mainnet"
    reader.base = "https://example.invalid"
    reader._perp_dexs_cache = None
    reader._meta_cache_by_dex = {}
    reader._market_catalog_cache = None
    reader._market_alias_index_cache = None
    reader._mids_cache_by_dex = {}
    def fake_post_info(payload):
        if payload["type"] == "userAbstraction":
            return "user"
        if payload["type"] == "perpDexs":
            return [{"name": "xyz"}]
        if payload["type"] == "meta":
            if payload.get("dex") == "xyz":
                return {"universe": [{"name": "xyz:SILVER", "szDecimals": 2, "maxLeverage": 25, "onlyIsolated": True}]}
            return {"universe": [{"name": "BTC", "szDecimals": 3, "maxLeverage": 40, "onlyIsolated": False}]}
        if payload["type"] == "clearinghouseState" and payload.get("dex") == "xyz":
            return {
                "assetPositions": [],
                "withdrawable": "0",
            }
        if payload["type"] == "clearinghouseState":
            return {
                "assetPositions": [
                    {
                        "position": {
                            "coin": "BTC",
                            "szi": "0.5",
                            "entryPx": "99000",
                            "unrealizedPnl": "10",
                            "returnOnEquity": "0.05",
                            "leverage": {"value": "5"},
                            "liquidationPx": "80000",
                            "marginUsed": "100",
                        }
                    }
                ],
                "withdrawable": "250",
            }
        if payload["type"] == "spotClearinghouseState":
            return {
                "balances": [
                    {"coin": "USDC", "token": 0, "total": "700", "hold": "50", "entryNtl": "0"}
                ],
                "tokenToAvailableAfterMaintenance": [[0, "650"]],
            }
        if payload["type"] == "allMids":
            if payload.get("dex") == "xyz":
                return {"xyz:SILVER": "74.75"}
            return {"BTC": "100000"}
        raise AssertionError(f"Unexpected payload: {payload}")

    reader.post_info = fake_post_info

    all_positions = reader.get_all_positions()
    snapshot = reader.get_position_snapshot("BTC", all_positions=all_positions, current_price=100000.0)

    assert all_positions["perp_account_equity_usd"] == pytest.approx(350.0)
    assert all_positions["spot_usdc_total"] == pytest.approx(700.0)
    assert all_positions["spot_available_usdc"] == pytest.approx(650.0)
    assert all_positions["account_equity_usd"] == pytest.approx(1050.0)
    assert all_positions["isolated_available_margin_usd"] == pytest.approx(0.0)
    assert all_positions["cross_available_margin_usd"] == pytest.approx(250.0)
    assert all_positions["perp_withdrawable_usd"] == pytest.approx(250.0)
    assert all_positions["available_margin_usd"] == pytest.approx(900.0)
    assert all_positions["withdrawable_usd"] == pytest.approx(900.0)
    assert all_positions["remaining_capital_usd"] == pytest.approx(900.0)
    assert all_positions["remaining_capital_source"] == "spot_available_plus_perp_withdrawable"
    assert all_positions["total_margin_used_usd"] == pytest.approx(100.0)
    assert all_positions["positions"][0]["max_leverage"] == 40
    assert snapshot["side"] == "long"
    assert snapshot["max_leverage"] == 40
    assert snapshot["remaining_capital_usd"] == pytest.approx(900.0)


def test_rest_reader_market_catalog_preserves_same_symbol_across_perp_dexs(uma):
    reader = object.__new__(uma.HyperliquidRestReader)
    reader._perp_dexs_cache = None
    reader._meta_cache_by_dex = {}
    reader._market_catalog_cache = None
    reader._market_alias_index_cache = None
    reader._mids_cache_by_dex = {}

    def fake_post_info(payload):
        if payload["type"] == "perpDexs":
            return [{"name": "testdex"}]
        if payload["type"] == "meta" and payload.get("dex") == "testdex":
            return {"universe": [{"name": "BTC", "szDecimals": 4, "maxLeverage": 10, "onlyIsolated": True}]}
        if payload["type"] == "meta":
            return {"universe": [{"name": "BTC", "szDecimals": 3, "maxLeverage": 40, "onlyIsolated": False}]}
        if payload["type"] == "allMids" and payload.get("dex") == "testdex":
            return {"BTC": "102000"}
        if payload["type"] == "allMids":
            return {"BTC": "101000"}
        raise AssertionError(f"Unexpected payload: {payload}")

    reader.post_info = fake_post_info

    catalog = reader.get_market_catalog()
    testdex_mids = reader.get_mids(dex="testdex")

    assert sorted(catalog) == ["BTC", "testdex:BTC"]
    assert catalog["BTC"]["max_leverage"] == 40
    assert catalog["testdex:BTC"]["dex"] == "testdex"
    assert catalog["testdex:BTC"]["display_name"] == "testdex:BTC-USDC"
    assert testdex_mids["testdex:BTC"] == "102000"


def test_rest_reader_get_all_positions_uses_spot_usdc_total_for_unified_account_margin_basis(uma):
    reader = object.__new__(uma.HyperliquidRestReader)
    reader.account_address = "0xabc"
    reader.network = "mainnet"
    reader.base = "https://example.invalid"
    reader._perp_dexs_cache = None
    reader._meta_cache_by_dex = {}
    reader._market_catalog_cache = None
    reader._market_alias_index_cache = None
    reader._mids_cache_by_dex = {}

    def fake_post_info(payload):
        if payload["type"] == "userAbstraction":
            return "unifiedAccount"
        if payload["type"] == "perpDexs":
            return [{"name": "xyz"}]
        if payload["type"] == "meta":
            if payload.get("dex") == "xyz":
                return {"universe": []}
            return {"universe": [{"name": "BTC", "szDecimals": 3, "maxLeverage": 40, "onlyIsolated": False}]}
        if payload["type"] == "clearinghouseState":
            return {
                "assetPositions": [],
                "marginSummary": {
                    "accountValue": "0",
                    "totalMarginUsed": "0",
                    "totalNtlPos": "0",
                    "totalRawUsd": "0",
                },
                "crossMarginSummary": {
                    "accountValue": "0",
                    "totalMarginUsed": "0",
                    "totalNtlPos": "0",
                    "totalRawUsd": "0",
                },
                "withdrawable": "0",
            }
        if payload["type"] == "spotClearinghouseState":
            return {
                "balances": [
                    {"coin": "USDC", "token": 0, "total": "700", "hold": "50", "entryNtl": "0"}
                ],
                "tokenToAvailableAfterMaintenance": [[0, "650"]],
            }
        if payload["type"] == "allMids":
            return {"BTC": "100000"}
        raise AssertionError(f"Unexpected payload: {payload}")

    reader.post_info = fake_post_info

    all_positions = reader.get_all_positions()
    snapshot = reader.get_position_snapshot("BTC", all_positions=all_positions, current_price=100000.0)

    assert all_positions["user_abstraction"] == "unifiedAccount"
    assert all_positions["perp_account_equity_usd"] == pytest.approx(700.0)
    assert all_positions["isolated_margin_basis_usd"] == pytest.approx(700.0)
    assert all_positions["cross_margin_basis_usd"] == pytest.approx(700.0)
    assert all_positions["account_equity_usd"] == pytest.approx(700.0)
    assert "isolated_available_margin_usd" not in all_positions
    assert "cross_available_margin_usd" not in all_positions
    assert "perp_withdrawable_usd" not in all_positions
    assert snapshot["user_abstraction"] == "unifiedAccount"
    assert snapshot["cross_margin_basis_usd"] == pytest.approx(700.0)
    assert "isolated_available_margin_usd" not in snapshot
    assert "cross_available_margin_usd" not in snapshot
    assert "perp_withdrawable_usd" not in snapshot


def test_rest_reader_resolves_ui_display_symbols_to_execution_symbols(uma):
    reader = object.__new__(uma.HyperliquidRestReader)
    reader.account_address = "0xabc"
    reader.network = "mainnet"
    reader.base = "https://example.invalid"
    reader._perp_dexs_cache = None
    reader._meta_cache_by_dex = {}
    reader._market_catalog_cache = None
    reader._market_alias_index_cache = None
    reader._mids_cache_by_dex = {}
    reader._spot_meta_cache = None
    reader._safe_spot_meta_cache = None
    reader.post_info = lambda payload: (
        [{"name": "xyz"}]
        if payload["type"] == "perpDexs"
        else {"universe": [{"name": "xyz:SILVER", "szDecimals": 2, "maxLeverage": 25, "onlyIsolated": True}]}
        if payload["type"] == "meta" and payload.get("dex") == "xyz"
        else {"universe": [{"name": "BTC", "szDecimals": 5, "maxLeverage": 40, "onlyIsolated": False}]}
    )

    assert reader.resolve_execution_symbol("BTC-USDC") == "BTC"
    assert reader.resolve_execution_symbol("xyz:SILVER") == "xyz:SILVER"
    assert reader.list_perp_symbols() == ["BTC", "xyz:SILVER"]
    assert reader.get_market_spec("BTC-USDC")["max_leverage"] == 40
    assert reader.get_market_spec("xyz:SILVER")["only_isolated"] is True


def test_normalize_spot_user_state_extracts_usdc_balances(uma):
    summary = uma.normalize_spot_user_state(
        {
            "balances": [
                {"coin": "USDC", "token": 0, "total": "998.5", "hold": "12.25", "entryNtl": "0"},
                {"coin": "ABC", "token": 42, "total": "1", "hold": "0", "entryNtl": "0"},
            ],
            "tokenToAvailableAfterMaintenance": [[0, "986.25"]],
        }
    )

    assert summary["spot_usdc_total"] == pytest.approx(998.5)
    assert summary["spot_usdc_hold"] == pytest.approx(12.25)
    assert summary["spot_available_usdc"] == pytest.approx(986.25)


def test_rest_reader_get_safe_spot_meta_filters_invalid_universe_entries(uma):
    reader = object.__new__(uma.HyperliquidRestReader)
    reader._spot_meta_cache = None
    reader._safe_spot_meta_cache = None
    reader.post_info = lambda payload: {
        "tokens": [{"name": "USDC", "szDecimals": 6}, {"name": "PURR", "szDecimals": 2}],
        "universe": [
            {"name": "PURR/USDC", "tokens": [1, 0], "index": 0, "isCanonical": True},
            {"name": "BROKEN/USDC", "tokens": [2, 0], "index": 1, "isCanonical": False},
        ],
    }

    safe_spot = reader.get_safe_spot_meta()

    assert len(safe_spot["tokens"]) == 2
    assert len(safe_spot["universe"]) == 1
    assert safe_spot["universe"][0]["name"] == "PURR/USDC"
    assert safe_spot["_dropped_invalid_universe_entries"] == 1


def test_executor_usd_to_size_rounds_down_to_exchange_precision(uma, monkeypatch):
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "false")
    reader = SimpleNamespace(get_sz_decimals=lambda symbol: 3)
    executor = uma.HyperliquidExecutor(reader, "BTC")

    size = executor.usd_to_size(100.0, 3.0)

    assert size == pytest.approx(33.333)


def test_apply_requested_leverage_clamps_to_symbol_max_in_dry_run(uma, monkeypatch):
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "false")
    reader = SimpleNamespace(get_market_spec=lambda symbol: {"max_leverage": 20, "only_isolated": True})
    executor = uma.HyperliquidExecutor(reader, "BTC")

    result = executor.apply_requested_leverage(50)

    assert result["requested_leverage"] == 50
    assert result["applied_leverage"] == 20
    assert result["max_leverage"] == 20
    assert result["is_cross"] is False
    assert "dry-run" in result["message"]


def test_rest_reader_build_market_chart_context_returns_three_chart_images(uma, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("OPENAI_CHART_IMAGE_WIDTH_PX", "510")
    monkeypatch.setenv("OPENAI_CHART_IMAGE_HEIGHT_PX", "240")
    monkeypatch.setenv("OPENAI_CHART_IMAGE_DETAIL", "low")

    reader = object.__new__(uma.HyperliquidRestReader)
    reader._market_chart_cache = {}
    reader.resolve_execution_symbol = lambda symbol: symbol
    reader.get_mid_price = lambda symbol: 104.75

    def fake_candles(symbol, interval, start_ms, end_ms):
        if interval == "1m":
            return _build_candles(start_ms, 60 * 1000, 120, 103.6)
        if interval == "5m":
            return _build_candles(start_ms, 5 * 60 * 1000, 72, 103.2)
        if interval == "15m":
            return _build_candles(start_ms, 15 * 60 * 1000, 96, 102.8)
        return []

    reader.get_candles_snapshot = fake_candles

    chart_context = reader.get_market_chart_context(
        "xyz:BRENTOIL",
        display_name="BRENTOIL-USDC",
        ttl_seconds=0.0,
        timeframe_specs=(
            {"timeframe": "1m", "window_hours": 2.0},
            {"timeframe": "5m", "window_hours": 6.0},
            {"timeframe": "15m", "window_hours": 24.0},
        ),
    )

    assert chart_context["execution_symbol"] == "xyz:BRENTOIL"
    assert chart_context["image_count"] == 3
    assert len(chart_context["input_images"]) == 3
    assert all(item["type"] == "input_image" for item in chart_context["input_images"])
    assert all(item["detail"] == "low" for item in chart_context["input_images"])
    assert all(item["image_url"].startswith("data:image/png;base64,") for item in chart_context["input_images"])
    assert [item["timeframe"] for item in chart_context["debug_images"]] == ["1m", "5m", "15m"]
    assert all(item["width_px"] == 510 for item in chart_context["debug_images"])
    assert all(item["height_px"] == 240 for item in chart_context["debug_images"])
    assert all(item["image_bytes"] > 0 for item in chart_context["debug_images"])


def test_rest_reader_build_passive_market_chart_context_returns_one_1m_chart_image(uma, monkeypatch):
    pytest.importorskip("matplotlib")
    monkeypatch.setenv("OPENAI_CHART_IMAGE_WIDTH_PX", "510")
    monkeypatch.setenv("OPENAI_CHART_IMAGE_HEIGHT_PX", "240")
    monkeypatch.setenv("OPENAI_CHART_IMAGE_DETAIL", "low")

    reader = object.__new__(uma.HyperliquidRestReader)
    reader._market_chart_cache = {}
    reader.resolve_execution_symbol = lambda symbol: symbol
    reader.get_mid_price = lambda symbol: 104.75

    def fake_candles(symbol, interval, start_ms, end_ms):
        if interval == "1m":
            return _build_candles(start_ms, 60 * 1000, 120, 103.2)
        return []

    reader.get_candles_snapshot = fake_candles

    chart_context = reader.get_market_chart_context(
        "xyz:BRENTOIL",
        display_name="BRENTOIL-USDC",
        ttl_seconds=0.0,
        timeframe_specs=({"timeframe": "1m", "window_hours": 2.0},),
    )

    assert chart_context["execution_symbol"] == "xyz:BRENTOIL"
    assert chart_context["image_count"] == 1
    assert len(chart_context["input_images"]) == 1
    assert chart_context["debug_images"][0]["timeframe"] == "1m"
    assert chart_context["debug_images"][0]["window_hours"] == 2.0


def test_render_candles_chart_png_resizes_after_base_layout_render(uma):
    pytest.importorskip("matplotlib")
    from PIL import Image

    candles = _build_candles(0, 5 * 60 * 1000, 72, 103.2)
    png = uma.render_candles_chart_png(
        candles=candles,
        symbol_label="BRENTOIL-USDC",
        timeframe="5m",
        window_hours=6.0,
        width_px=480,
        height_px=270,
        current_price=104.75,
    )
    assert png
    image = Image.open(io.BytesIO(png))
    assert image.size == (480, 270)


def test_build_chart_tick_positions_excludes_latest_candle(uma):
    positions = uma._build_chart_tick_positions(73, 6)
    assert positions
    assert positions[0] == 0
    assert positions[-1] < 72
    assert 72 not in positions


def test_format_chart_price_label_compacts_decimals(uma):
    assert uma._format_chart_price_label(67680.1234) == "67680"
    assert uma._format_chart_price_label(104.715) == "104.72"
    assert uma._format_chart_price_label(74.773) == "74.773"


def test_build_trade_symbol_context_strips_account_level_position_fields(uma):
    agent = object.__new__(uma.UnifiedMarketAgent)

    class Reader:
        def get_mid_price(self, symbol):
            return 100000.0 if symbol == "BTC" else 74.5

        def get_position_snapshot(self, symbol, all_positions=None, current_price=None):
            return {
                "known": True,
                "symbol": symbol,
                "side": "flat",
                "size": 0.0,
                "entry_price": 0.0,
                "mid_price": current_price,
                "notional_usd": 0.0,
                "leverage": 0.0,
                "max_leverage": 40 if symbol == "BTC" else 25,
                "only_isolated": False,
                "available_margin_usd": 321.5,
                "withdrawable_usd": 300.0,
                "remaining_capital_usd": 300.0,
                "unrealized_pnl": 0.0,
                "margin_used": 0.0,
            }

        def get_market_spec(self, symbol):
            return {"symbol": symbol, "max_leverage": 40 if symbol == "BTC" else 25, "only_isolated": False}

    agent.reader = Reader()

    context = agent._build_trade_symbol_context(
        {"remaining_capital_usd": 300.0, "available_margin_usd": 321.5, "withdrawable_usd": 300.0},
        {
            "trade_symbol_key": "BTC_USDC",
            "display_name": "BTC-USDC",
            "configured_execution_symbol": "BTC",
            "execution_symbol": "BTC",
        },
    )

    assert "symbol_position" not in context
    assert "market_technical_context" not in context
    assert context["current_price"] == pytest.approx(100000.0)
    assert context["market_spec"]["symbol"] == "BTC"


def test_execute_live_mode_flat_open_uses_market_adjustment_and_skips_without_mid_price(uma, monkeypatch):
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")

    class Reader:
        def get_position_snapshot(self, symbol):
            return {"symbol": symbol, "side": "flat", "size": 0.0, "entry_price": 0.0, "mid_price": None}

        def get_mid_price(self, symbol):
            return None

        def get_sz_decimals(self, symbol):
            return 3

        def get_market_spec(self, symbol):
            return {"max_leverage": 40, "only_isolated": False}

    executor = uma.HyperliquidExecutor(Reader(), "BTC")
    executor._ensure_exchange = lambda: None
    executor.apply_requested_leverage = lambda requested_leverage: {}
    decision = uma.StrategyDecision(
        action="long",
        suggested_notional_usd=1000.0,
        entry_price=100000.0,
        stop_loss_price=99000.0,
        planned_margin_used_usd=100.0,
        planned_max_loss_usd=10.0,
        requested_leverage=10,
    )

    result = executor.execute(decision, plan_name="open")

    assert result["reason"] == "entry_target_adjustment_from_flat"
    assert result["target_side"] == "long"
    assert result["open_notional_usd"] == pytest.approx(1000.0)
    assert result["mid"] is None
    assert result["actions"] == []
    assert "skipped target open order" in result["message"].lower()


def test_entry_limit_order_retries_with_qty_backoff_until_success(uma, monkeypatch):
    monkeypatch.setenv("ENABLE_LIVE_TRADING", "true")

    class Reader:
        def get_position_snapshot(self, symbol):
            return {"symbol": symbol, "side": "flat", "size": 0.0, "entry_price": 0.0, "mid_price": None}

        def get_sz_decimals(self, symbol):
            return 1

        def get_market_spec(self, symbol):
            return {"max_leverage": 40, "only_isolated": False}

    order_calls = []

    def fake_order(symbol, is_buy, qty, limit_price, order_type, reduce_only=False, cloid=None):
        order_calls.append(float(qty))
        if len(order_calls) < 3:
            return {"error": "Rejected"}
        return {"response": {"data": {"statuses": [{"resting": {"oid": 777777}}]}}}

    executor = uma.HyperliquidExecutor(Reader(), "BTC")
    executor._ensure_exchange = lambda: None
    executor._exchange = SimpleNamespace(order=fake_order)
    executor.apply_requested_leverage = lambda requested_leverage: {}

    result = executor.place_entry_limit_order(
        side="long",
        notional_usd=1000.0,
        requested_leverage=10,
        limit_price=100.0,
        reason="test_limit_retry",
        plan_name="plan",
    )

    assert result["accepted"] is True
    assert result["entry_order_pending"] is True
    assert result["attempt_count"] == 3
    assert result["attempted_qtys"] == pytest.approx([10.0, 9.9, 9.8])
    assert order_calls == pytest.approx([10.0, 9.9, 9.8])
    assert result["oid"] == 777777


def test_engine_get_playbook_attaches_chart_images_and_sanitizes_request_debug(uma):
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="resp_with_images",
                model="gpt-5.4-mini",
                output_text=json.dumps(_minimal_playbook_payload(), ensure_ascii=False),
                usage=SimpleNamespace(
                    input_tokens=2400,
                    input_tokens_details=SimpleNamespace(cached_tokens=0),
                    output_tokens=500,
                    output_tokens_details=SimpleNamespace(reasoning_tokens=90),
                    total_tokens=2900,
                ),
                output=[SimpleNamespace(type="message")],
            )

    engine = object.__new__(uma.DiscretionaryLLMEngine)
    engine.client = SimpleNamespace(responses=FakeResponses())
    engine.active_model = "gpt-5.4-mini"
    engine.passive_model = "gpt-5.4-mini"
    engine.symbol = "BTC"
    engine.default_search_mode = "off"
    engine.active_search_mode = "off"
    engine.passive_search_mode = "off"
    engine.active_reasoning_effort = "high"
    engine.passive_reasoning_effort = "medium"
    engine.execute_now_confidence_threshold = 0.7
    engine.include_chart_images = True
    engine.chart_context_builder = lambda candidate: {
        "display_name": "BTC-USDC",
        "execution_symbol": "BTC",
        "detail": "low",
        "input_images": [
            {"type": "input_image", "detail": "low", "image_url": "data:image/png;base64,AAAA"},
            {"type": "input_image", "detail": "low", "image_url": "data:image/png;base64,BBBB"},
            {"type": "input_image", "detail": "low", "image_url": "data:image/png;base64,CCCC"},
        ],
        "debug_images": [
            {"timeframe": "1m", "window_hours": 2.0, "width_px": 510, "height_px": 240, "detail": "low", "candle_count": 120, "image_bytes": 900, "data_url_chars": 26},
            {"timeframe": "5m", "window_hours": 12.0, "width_px": 510, "height_px": 240, "detail": "low", "candle_count": 144, "image_bytes": 1000, "data_url_chars": 26},
            {"timeframe": "15m", "window_hours": 24.0, "width_px": 510, "height_px": 240, "detail": "low", "candle_count": 96, "image_bytes": 1200, "data_url_chars": 26},
        ],
        "note": "charts",
    }

    playbook, mode = engine.get_playbook(
        user_query="test query",
        event_tape=[{"id": 1, "headline": "headline"}],
        recent_events=[{"id": 1, "headline": "headline"}],
        trigger_reason="startup",
        trigger_event=None,
        has_live_position=False,
        trade_symbol_context={
            "trade_symbol_key": "BTC_USDC",
            "display_name": "BTC-USDC",
            "execution_symbol": "BTC",
            "tradable_on_hyperliquid": True,
            "current_price": 100000.0,
            "symbol_position": {"symbol": "BTC", "side": "flat", "max_leverage": 40},
            "market_spec": {"symbol": "BTC", "max_leverage": 40},
        },
        active_symbol="BTC",
    )

    assert mode == "raw_context_only"
    assert playbook.selected_symbol == "BTC-USDC"
    user_payload = json.loads(captured["input"][1]["content"][0]["text"])
    assert "chart_image_context" not in user_payload
    assert len(captured["input"][1]["content"]) == 5
    assert captured["input"][1]["content"][1]["type"] == "input_text"
    assert "chart screenshot" in captured["input"][1]["content"][1]["text"].lower()
    assert captured["input"][1]["content"][2]["type"] == "input_image"
    assert captured["input"][1]["content"][2]["image_url"].startswith("data:image/png;base64,")
    assert engine.last_call_debug["request_messages"][1]["content"][2]["image_url"].startswith("<data-url:")
    assert engine.last_call_debug["usage_cost"]["image_inputs"]["count"] == 3
    assert engine.last_call_debug["chart_screenshot_debug"]["image_count"] == 3


def test_engine_get_playbook_passive_attaches_single_1m_chart_image_when_enabled(uma):
    captured = {}
    requests = []
    chart_modes = []

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            requests.append(kwargs)
            schema_name = kwargs.get("text", {}).get("format", {}).get("name", "")
            if schema_name == "llm_passive_event_judge":
                output_text = json.dumps(_passive_event_judge_payload(action="long", confidence=0.62), ensure_ascii=False)
                response_id = "resp_passive_judge"
            else:
                output_text = json.dumps(_passive_technical_pricing_payload(entry_price=110.0, stop_loss_price=108.0), ensure_ascii=False)
                response_id = "resp_passive_price"
            return SimpleNamespace(
                id=response_id,
                model="gpt-5.4-mini",
                output_text=output_text,
                usage=SimpleNamespace(
                    input_tokens=1200,
                    input_tokens_details=SimpleNamespace(cached_tokens=0),
                    output_tokens=300,
                    output_tokens_details=SimpleNamespace(reasoning_tokens=60),
                    total_tokens=1500,
                ),
                output=[SimpleNamespace(type="message")],
            )

    def build_chart(candidate):
        chart_modes.append(candidate.get("_chart_mode"))
        return {
            "display_name": "BRENTOIL-USDC",
            "execution_symbol": "xyz:BRENTOIL",
            "detail": "low",
            "input_images": [
                {"type": "input_image", "detail": "low", "image_url": "data:image/png;base64,AAAA"},
            ],
            "debug_images": [
                {"timeframe": "1m", "window_hours": 2.0, "width_px": 510, "height_px": 240, "detail": "low", "candle_count": 120, "image_bytes": 1000, "data_url_chars": 26},
            ],
            "note": "charts",
            "image_count": 1,
        }

    engine = object.__new__(uma.DiscretionaryLLMEngine)
    engine.client = SimpleNamespace(responses=FakeResponses())
    engine.active_model = "gpt-5.4-mini"
    engine.passive_model = "gpt-5.4-mini"
    engine.symbol = "BTC"
    engine.default_search_mode = "off"
    engine.active_search_mode = "off"
    engine.passive_search_mode = "off"
    engine.active_reasoning_effort = "high"
    engine.passive_reasoning_effort = "medium"
    engine.passive_relevance_threshold = 0.35
    engine.execute_now_confidence_threshold = 0.7
    engine.include_chart_images = True
    engine.include_passive_chart_images = True
    engine.chart_context_builder = build_chart

    playbook, mode = engine.get_playbook(
        user_query="test query",
        event_tape=[{"headline": "headline"}],
        recent_events=[],
        trigger_reason="passive_event_trigger",
        trigger_event={"headline": "headline"},
        has_live_position=False,
        trade_symbol_context={
            "trade_symbol_key": "BRENTOIL_USDC",
            "display_name": "BRENTOIL-USDC",
            "execution_symbol": "xyz:BRENTOIL",
            "tradable_on_hyperliquid": True,
            "current_price": 110.0,
            "symbol_position": {"symbol": "xyz:BRENTOIL", "side": "flat", "max_leverage": 20},
            "market_spec": {"symbol": "xyz:BRENTOIL", "max_leverage": 20},
        },
        active_symbol="xyz:BRENTOIL",
    )

    assert mode == "raw_context_only"
    assert playbook.selected_symbol == "BRENTOIL-USDC"
    assert chart_modes == ["passive"]
    assert len(requests) == 3
    pricing_request = requests[-1]
    user_payload = json.loads(pricing_request["input"][1]["content"][0]["text"])
    assert user_payload["action"] == "long"
    assert "chart_image_context" not in user_payload
    assert len(pricing_request["input"][1]["content"]) == 3
    assert pricing_request["input"][1]["content"][1]["type"] == "input_text"
    assert "chart screenshot" in pricing_request["input"][1]["content"][1]["text"].lower()
    assert pricing_request["input"][1]["content"][2]["type"] == "input_image"
    assert engine.last_call_debug["chart_screenshot_debug"]["image_count"] == 1
    assert engine.last_call_debug["chart_screenshot_debug"]["images"][0]["timeframe"] == "1m"


def test_engine_get_playbook_passive_keeps_chart_summary_when_images_disabled(uma):
    requests = []
    chart_modes = []

    class FakeResponses:
        def create(self, **kwargs):
            requests.append(kwargs)
            schema_name = kwargs.get("text", {}).get("format", {}).get("name", "")
            if schema_name == "llm_passive_event_judge":
                output_text = json.dumps(_passive_event_judge_payload(action="long", confidence=0.62), ensure_ascii=False)
                response_id = "resp_passive_summary_judge"
            else:
                output_text = json.dumps(_passive_technical_pricing_payload(entry_price=110.0, stop_loss_price=108.0), ensure_ascii=False)
                response_id = "resp_passive_summary_price"
            return SimpleNamespace(
                id=response_id,
                model="gpt-5.4-mini",
                output_text=output_text,
                usage=SimpleNamespace(
                    input_tokens=1200,
                    input_tokens_details=SimpleNamespace(cached_tokens=0),
                    output_tokens=300,
                    output_tokens_details=SimpleNamespace(reasoning_tokens=60),
                    total_tokens=1500,
                ),
                output=[SimpleNamespace(type="message")],
            )

    def build_chart(candidate):
        chart_modes.append(candidate.get("_chart_mode"))
        return {
            "display_name": "BRENTOIL-USDC",
            "execution_symbol": "xyz:BRENTOIL",
            "detail": "low",
            "input_images": [
                {"type": "input_image", "detail": "low", "image_url": "data:image/png;base64,AAAA"},
            ],
            "debug_images": [
                {"timeframe": "1m", "window_hours": 2.0, "width_px": 510, "height_px": 240, "detail": "low", "candle_count": 120, "image_bytes": 1000, "data_url_chars": 26},
            ],
            "chart_summaries": [
                {"timeframe": "1m", "window_hours": 2.0, "current_price": 110.0},
            ],
            "note": "charts",
            "image_count": 1,
        }

    engine = object.__new__(uma.DiscretionaryLLMEngine)
    engine.client = SimpleNamespace(responses=FakeResponses())
    engine.active_model = "gpt-5.4-mini"
    engine.passive_model = "gpt-5.4-mini"
    engine.symbol = "BTC"
    engine.default_search_mode = "off"
    engine.active_search_mode = "off"
    engine.passive_search_mode = "off"
    engine.active_reasoning_effort = "high"
    engine.passive_reasoning_effort = "medium"
    engine.passive_relevance_threshold = 0.35
    engine.execute_now_confidence_threshold = 0.7
    engine.include_chart_images = True
    engine.include_passive_chart_images = False
    engine.chart_context_builder = build_chart

    playbook, mode = engine.get_playbook(
        user_query="test query",
        event_tape=[{"headline": "headline"}],
        recent_events=[],
        trigger_reason="passive_event_trigger",
        trigger_event={"headline": "headline"},
        has_live_position=False,
        trade_symbol_context={
            "trade_symbol_key": "BRENTOIL_USDC",
            "display_name": "BRENTOIL-USDC",
            "execution_symbol": "xyz:BRENTOIL",
            "tradable_on_hyperliquid": True,
            "current_price": 110.0,
            "symbol_position": {"symbol": "xyz:BRENTOIL", "side": "flat", "max_leverage": 20},
            "market_spec": {"symbol": "xyz:BRENTOIL", "max_leverage": 20},
        },
        active_symbol="xyz:BRENTOIL",
    )

    assert mode == "raw_context_only"
    assert playbook.selected_symbol == "BRENTOIL-USDC"
    assert chart_modes == ["passive"]
    assert len(requests) == 3
    pricing_request = requests[-1]
    pricing_payload = json.loads(pricing_request["input"][1]["content"][0]["text"])
    assert pricing_payload["chart_summaries"][0]["current_price"] == pytest.approx(110.0)
    assert len(pricing_request["input"][1]["content"]) == 1
    assert engine.last_call_debug["chart_screenshot_debug"]["image_count"] == 0
    assert "image_inputs" not in engine.last_call_debug["usage_cost"]


def test_engine_get_playbook_context_only_includes_web_search_and_payload_fields(uma):
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="resp_test_123",
                model="gpt-5.4",
                output_text=json.dumps(_passive_event_judge_payload(relevance="unrelated", confidence=0.0, action="no_trade"), ensure_ascii=False),
                usage=SimpleNamespace(
                    input_tokens=1200,
                    input_tokens_details=SimpleNamespace(cached_tokens=200),
                    output_tokens=400,
                    output_tokens_details=SimpleNamespace(reasoning_tokens=80),
                    total_tokens=1600,
                ),
                output=[SimpleNamespace(type="message")],
            )

    engine = object.__new__(uma.DiscretionaryLLMEngine)
    engine.client = SimpleNamespace(responses=FakeResponses())
    engine.active_model = "gpt-5.4"
    engine.passive_model = "gpt-5.4"
    engine.symbol = "BTC"
    engine.default_search_mode = "context_only"
    engine.active_search_mode = "context_only"
    engine.passive_search_mode = "context_only"
    engine.active_reasoning_effort = "high"
    engine.passive_reasoning_effort = "medium"
    engine.passive_relevance_threshold = 0.35
    engine.execute_now_confidence_threshold = 0.7

    playbook, mode = engine.get_playbook(
        user_query="test query",
        event_tape=[{"headline": "headline"}, {"headline": "follow-up"}],
        recent_events=[{"id": 1, "headline": "headline"}],
        trigger_reason="passive_event_trigger",
        trigger_event={"headline": "headline"},
        has_live_position=False,
        trade_symbol_context={
            "trade_symbol_key": "BTC_USDC",
            "display_name": "BTC-USDC",
            "execution_symbol": "BTC",
            "tradable_on_hyperliquid": True,
            "current_price": 100000.0,
            "symbol_position": {"symbol": "BTC", "side": "flat", "max_leverage": 40},
            "market_spec": {"symbol": "BTC", "max_leverage": 40},
        },
        active_symbol="BTC",
    )

    user_payload = json.loads(captured["input"][1]["content"][0]["text"])

    assert captured["tools"] == []
    assert mode == "context_enriched_with_web"
    assert "account_summary" not in user_payload
    assert "risk_constraints" not in user_payload
    assert "all_positions" not in user_payload
    assert "symbol_position" not in user_payload
    assert [item["headline"] for item in user_payload["recent_events"]] == ["headline"]
    assert user_payload["trigger_event"]["headline"] == "headline"
    assert "recent_event_context" not in user_payload
    assert "market_news_context" not in user_payload
    assert user_payload["trade_symbol"] == "BTC-USDC"
    assert "symbol" not in user_payload
    assert "active_symbol" not in user_payload
    assert "current_price" not in user_payload
    assert "trade_symbol_context" not in user_payload
    assert "chart_summaries" not in user_payload
    assert engine.last_call_debug["llm_payload_market_only"] is True
    assert playbook.entry_plan.execute_now is False
    assert playbook.selected_symbol == "BTC-USDC"
    assert engine.last_call_debug["response_id"] == "resp_test_123+resp_test_123"
    assert engine.last_call_debug["usage"]["input_tokens"] == 2400
    assert engine.last_call_debug["web_search_tool_calls"] == 0
    assert captured["reasoning"]["effort"] == "medium"
    assert engine.last_call_debug["usage_cost"]["known"] is True
    assert engine.last_call_debug["usage_cost"]["total_cost_usd"] > 0
    assert engine.last_call_debug["capped_playbook"]["entry_plan"]["action_decision"]["entry_price"] == pytest.approx(0.0)


def test_engine_get_playbook_omits_recent_events_from_active_pricing_payload(uma):
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="resp_test_active_recent_events",
                model="gpt-5.4-mini",
                output_text=json.dumps(
                    {
                        "trigger_event_relevance": "not_applicable",
                        "trigger_confidence": None,
                        "playbook": {
                            "entry_plan": {
                                "execute_now": False,
                                "action_decision": {
                                    "action": "no_trade",
                                    "entry_price": 0.0,
                                    "stop_loss_price": 0.0,
                                },
                                "scenario": None,
                            }
                        },
                    },
                    ensure_ascii=False,
                ),
                usage=SimpleNamespace(
                    input_tokens=500,
                    input_tokens_details=SimpleNamespace(cached_tokens=0),
                    output_tokens=120,
                    output_tokens_details=SimpleNamespace(reasoning_tokens=10),
                    total_tokens=620,
                ),
                output=[SimpleNamespace(type="message")],
            )

    engine = object.__new__(uma.DiscretionaryLLMEngine)
    engine.client = SimpleNamespace(responses=FakeResponses())
    engine.active_model = "gpt-5.4-mini"
    engine.passive_model = "gpt-5.4-mini"
    engine.default_search_mode = "off"
    engine.active_search_mode = "off"
    engine.passive_search_mode = "off"
    engine.include_chart_images = False
    engine.include_passive_chart_images = False
    engine.execute_now_confidence_threshold = 0.7
    engine.active_reasoning_effort = "medium"
    engine.passive_reasoning_effort = "medium"
    engine.active_openai_request_timeout_seconds = 60.0
    engine.passive_openai_request_timeout_seconds = 60.0
    engine.openai_max_attempts = 1
    engine.openai_retry_delay_seconds = 0.0
    engine.last_call_debug = {}
    engine._resolve_model = lambda trigger_reason: "gpt-5.4-mini"
    engine._resolve_request_timeout_seconds = lambda trigger_reason: 60.0
    engine._responses_create_with_retry = lambda **kwargs: engine.client.responses.create(**kwargs)

    engine.get_playbook(
        user_query="trade BTC",
        event_tape=[],
        trigger_reason="active_periodic_refresh",
        trigger_event=None,
        recent_events=[{"source": "reuters", "title": "recent", "event_timestamp": "2026-04-14T00:00:00Z"}],
        trade_symbol_context={
            "display_name": "BTC-USDC",
            "trade_symbol_key": "BTC_USDC",
            "execution_symbol": "BTC",
            "current_price": 100000.0,
        },
        active_symbol="BTC",
    )

    user_payload = json.loads(captured["input"][1]["content"][0]["text"])

    assert "recent_events" not in user_payload



def test_engine_get_playbook_omits_leverage_related_fields_from_management_query_payload(uma):
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                id="resp_test_mgmt_omit_leverage",
                model="gpt-5.4",
                output_text=json.dumps(_minimal_playbook_payload(), ensure_ascii=False),
                usage=SimpleNamespace(
                    input_tokens=900,
                    input_tokens_details=SimpleNamespace(cached_tokens=0),
                    output_tokens=220,
                    output_tokens_details=SimpleNamespace(reasoning_tokens=40),
                    total_tokens=1120,
                ),
                output=[SimpleNamespace(type="message")],
            )

    engine = object.__new__(uma.DiscretionaryLLMEngine)
    engine.client = SimpleNamespace(responses=FakeResponses())
    engine.active_model = "gpt-5.4"
    engine.passive_model = "gpt-5.4"
    engine.symbol = "BTC"
    engine.default_search_mode = "off"
    engine.active_search_mode = "off"
    engine.passive_search_mode = "off"
    engine.active_reasoning_effort = "high"
    engine.passive_reasoning_effort = "medium"
    engine.execute_now_confidence_threshold = 0.7

    open_position = {
        "symbol": "BTC",
        "side": "long",
        "size": 1.5,
        "entry_price": 100000.0,
        "mid_price": 100250.0,
        "notional_usd": 150375.0,
        "unrealized_pnl": 375.0,
        "return_on_equity": 12.5,
        "leverage": 8.0,
        "max_leverage": 40,
        "only_isolated": False,
        "liquidation_price": 90123.0,
        "margin_used": 18796.88,
    }

    playbook, mode = engine.get_playbook(
        user_query="test management query",
        event_tape=[{"id": 1, "headline": "headline"}],
        recent_events=[{"id": 1, "headline": "headline"}],
        trigger_reason="manual_once",
        trigger_event=None,
        has_live_position=True,
        trade_symbol_context={
            "trade_symbol_key": "BTC_USDC",
            "display_name": "BTC-USDC",
            "execution_symbol": "BTC",
            "tradable_on_hyperliquid": True,
            "current_price": 100250.0,
            "symbol_position": dict(open_position),
            "market_spec": {"symbol": "BTC", "max_leverage": 40, "only_isolated": False},
        },
        active_symbol="BTC",
    )

    user_payload = json.loads(captured["input"][1]["content"][0]["text"])

    assert mode == "raw_context_only"
    assert playbook.selected_symbol == "BTC-USDC"
    assert "symbol_position" not in user_payload
    assert "all_positions" not in user_payload
    assert "account_summary" not in user_payload
    assert "trade_symbol_context" not in user_payload
    assert user_payload["chart_summaries"] == []
    assert engine.last_call_debug["llm_payload_market_only"] is True

def test_engine_resolve_search_mode_rejects_unknown_values(uma):
    engine = object.__new__(uma.DiscretionaryLLMEngine)
    engine.default_search_mode = "context_only"
    engine.active_search_mode = "invalid_mode"
    engine.passive_search_mode = "context_only"

    with pytest.raises(ValueError):
        engine._resolve_search_mode("active_periodic_refresh")


def test_analyze_web_search_calls_detects_budget_and_duplicates_for_fixed_symbol(uma):
    trade_symbol_context = {
        "display_name": "BTC-USDC",
        "trade_symbol_key": "BTC_USDC",
        "execution_symbol": "BTC",
    }
    calls = [
        {"action": {"query": "Reuters March 2026 bitcoin ETF inflows Morgan Stanley enters bitcoin ETF race March 2026"}},
        {"action": {"query": "Reuters Morgan Stanley bitcoin ETF March 2026"}},
        {"action": {"query": "Reuters March 2026 bitcoin ETF inflows and bank demand"}},
    ]

    analysis = uma.analyze_web_search_calls(
        calls,
        trade_symbol_context,
        max_total_calls=2,
        max_calls_per_topic=2,
    )

    assert analysis["actual_calls"] == 3
    assert analysis["over_budget"] is True
    assert analysis["calls_per_topic"]["BTC-USDC"] == 3
    assert analysis["topic_budget_violations"] == {"BTC-USDC": 3}
    assert analysis["duplicate_call_count"] >= 1
    assert any(item["topic"] == "BTC-USDC" for item in analysis["duplicate_calls"])


def test_analyze_web_search_calls_flags_calculator_usage_as_non_news(uma):
    analysis = uma.analyze_web_search_calls(
        [{"action": {"query": "calculator: 300/((104.5-99.4)/104.5)"}}],
        {
            "display_name": "BRENTOIL-USDC",
            "trade_symbol_key": "BRENTOIL_USDC",
            "execution_symbol": "xyz:BRENTOIL",
        },
        max_total_calls=4,
        max_calls_per_topic=2,
    )

    assert analysis["non_news_call_count"] == 1
    assert analysis["non_news_calls"][0]["query_kind"] == "calculator"


def test_infer_search_call_topic_prefers_silver_for_silver_queries(uma):
    trade_symbol_context = {
        "display_name": "SILVER-USDC",
        "trade_symbol_key": "SILVER_USDC",
        "execution_symbol": "xyz:SILVER",
    }

    topic = uma._infer_search_call_topic(
        "Reuters March 2026 silver market supply demand inventory geopolitics ETF silver",
        trade_symbol_context,
    )

    assert topic == "SILVER-USDC"


def test_engine_forced_news_context_emits_audit_callback_before_final_playbook(uma, tmp_path):
    calls = []
    request_log = []

    class FakeResponses:
        def __init__(self):
            self.count = 0

        def create(self, **kwargs):
            self.count += 1
            request_log.append(kwargs)
            if self.count == 1:
                return SimpleNamespace(
                    id="resp_news_1",
                    model="gpt-5.4-mini",
                    output_text=json.dumps(
                        {
                            "market_mainline_context": {
                                "current_move_logic_mainline": "ETF-led crypto risk-on tone is supporting BTC.",
                                "diagnostic_instruments": ["NDX", "DXY"],
                            },
                            "materially_new_first_events": [
                                {
                                    "event_timestamp": "2026-04-20T00:00:00Z",
                                    "source": "reuters",
                                    "title": "ETF demand accelerates for BTC",
                                }
                            ],
                        },
                        ensure_ascii=False,
                    ),
                    usage=SimpleNamespace(
                        input_tokens=100,
                        input_tokens_details=SimpleNamespace(cached_tokens=0),
                        output_tokens=50,
                        output_tokens_details=SimpleNamespace(reasoning_tokens=10),
                        total_tokens=150,
                    ),
                    output=[SimpleNamespace(type="web_search_call")],
                )
            return SimpleNamespace(
                id="resp_playbook_2",
                model="gpt-5.4-mini",
                output_text=json.dumps(_minimal_playbook_payload(), ensure_ascii=False),
                usage=SimpleNamespace(
                    input_tokens=200,
                    input_tokens_details=SimpleNamespace(cached_tokens=0),
                    output_tokens=80,
                    output_tokens_details=SimpleNamespace(reasoning_tokens=20),
                    total_tokens=280,
                ),
                output=[SimpleNamespace(type="message")],
            )

    engine = object.__new__(uma.DiscretionaryLLMEngine)
    engine.client = SimpleNamespace(responses=FakeResponses())
    engine.active_model = "gpt-5.4-mini"
    engine.passive_model = "gpt-5.4-mini"
    engine.symbol = "BTC"
    engine.default_search_mode = "always"
    engine.active_search_mode = "always"
    engine.passive_search_mode = "context_only"
    engine.active_reasoning_effort = "high"
    engine.passive_reasoning_effort = "medium"
    engine.execute_now_confidence_threshold = 0.7
    engine.force_active_news_context = True
    engine.force_passive_news_context = False
    engine.event_recent_window_hours = 72.0
    engine.audit_callback = lambda event, payload: calls.append((event, payload))
    engine.last_call_debug = {}
    engine.helper_market_mainline_latest_path = tmp_path / "latest_helper_market_mainline.json"
    engine.helper_materially_new_first_events_path = tmp_path / "helper_materially_new_first_events.jsonl"
    engine.latest_helper_market_mainline_context = {}
    engine.latest_helper_market_mainline_debug = {}
    engine.latest_helper_materially_new_first_events = []

    playbook, mode = engine.get_playbook(
        user_query="test query",
        event_tape=[
            {
                "source": "reuters",
                "item_id": "btc123",
                "title": "ETF demand accelerates for BTC",
                "url": "https://example.com/btc",
                "event_timestamp": "2026-04-20T00:00:00Z",
            }
        ],
        recent_events=[],
        trigger_reason="manual_once",
        trigger_event=None,
        has_live_position=False,
        trade_symbol_context={
            "trade_symbol_key": "BTC_USDC",
            "display_name": "BTC-USDC",
            "execution_symbol": "BTC",
            "tradable_on_hyperliquid": True,
            "current_price": 100000.0,
            "symbol_position": {"symbol": "BTC", "side": "flat", "max_leverage": 40},
            "market_spec": {"symbol": "BTC", "max_leverage": 40},
        },
        active_symbol="BTC",
    )

    assert mode == "verified_with_web"
    assert playbook.selected_symbol == "BTC-USDC"
    assert calls
    assert calls[0][0] == "market_mainline_call_debug"
    assert calls[0][1]["web_search_tool_calls"] == 1
    assert "selected_instrument" not in calls[0][1]["market_mainline_context"]
    assert calls[0][1]["market_mainline_context"]["current_move_logic_mainline"] == "ETF-led crypto risk-on tone is supporting BTC."
    assert calls[0][1]["market_mainline_context"]["diagnostic_instruments"] == ["NDX", "DXY"]
    assert calls[0][1]["web_search_budget"]["max_total_calls"] is None
    assert calls[0][1]["web_search_analysis"]["actual_calls"] == 1
    assert engine.last_call_debug["market_mainline_web_search_budget"]["max_total_calls"] is None
    assert engine.last_call_debug["market_mainline_web_search_analysis"]["over_budget"] is False
    first_prompt = request_log[0]["input"][0]["content"][0]["text"]
    first_payload = json.loads(request_log[0]["input"][1]["content"][0]["text"])
    assert "trade_symbol is the only trading instrument this helper serves" in first_prompt
    assert "Use diagnostic_instrument_universe only for diagnostic_instruments" in first_prompt
    assert "Use prior_materially_new_events only as background for already-known facts." in first_prompt
    assert "but do not add web_search-only events to materially_new_first_events" in first_prompt
    assert "Combine materially_new_first_events with web_search to produce current_move_logic_mainline for trade_symbol" in first_prompt
    assert "Then choose diagnostic_instruments only from diagnostic_instrument_universe" in first_prompt
    assert "diagnostic_instruments are instruments whose reaction to the same news flow or regime change helps diagnose" in first_prompt
    assert "ranking by interpretive value rather than raw correlation" in first_prompt
    assert "Return 3 diagnostic_instruments when possible" in first_prompt
    assert "Avoid redundant selections that express the same signal" in first_prompt
    assert "Do not return trade_symbol, its base asset, obvious aliases or equivalents of trade_symbol" in first_prompt
    assert "For oil/Brent" not in first_prompt
    assert "From recent_events, keep only thoroughly materially new first fact-level events after comparing within recent_events and against prior_materially_new_events" in first_prompt
    assert sorted(first_payload.keys()) == ["diagnostic_instrument_universe", "prior_materially_new_events", "recent_events", "trade_symbol"]
    assert first_payload["trade_symbol"] == "BTC-USDC"
    assert "NDX" in first_payload["diagnostic_instrument_universe"]
    assert "DXY" in first_payload["diagnostic_instrument_universe"]
    assert "ETH" in first_payload["diagnostic_instrument_universe"]
    assert first_payload["prior_materially_new_events"] == []
    assert first_payload["recent_events"][0] == {
        "source": "reuters",
        "title": "ETF demand accelerates for BTC",
        "event_timestamp": "2026-04-20T00:00:00Z",
    }
    assert request_log[0]["reasoning"]["effort"] == "xhigh"
    assert request_log[-1]["reasoning"]["effort"] == "high"
    assert request_log[-1]["tools"] == []
    assert "market_mainline_context" not in json.loads(request_log[-1]["input"][1]["content"][0]["text"])
    persisted_lines = [line for line in engine.helper_market_mainline_latest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    persisted_payload = json.loads(persisted_lines[-1])
    assert "selected_instrument" not in persisted_payload["market_mainline_context"]
    assert persisted_payload["market_mainline_context"]["current_move_logic_mainline"] == "ETF-led crypto risk-on tone is supporting BTC."
    assert persisted_payload["market_mainline_context"]["diagnostic_instruments"] == ["NDX", "DXY"]
    assert persisted_payload["market_mainline_call_debug"]["winner_display_name"] == "BTC-USDC"
    materiality_lines = [line for line in engine.helper_materially_new_first_events_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    materiality_payload = json.loads(materiality_lines[-1])
    assert materiality_payload["materially_new_first_events"][0] == {
        "event_timestamp": "2026-04-20T00:00:00Z",
        "source": "reuters",
        "title": "ETF demand accelerates for BTC",
    }


def test_engine_passive_playbook_includes_helper_analysis_even_with_live_position(uma):
    request_log = []

    class FakeResponses:
        def create(self, **kwargs):
            request_log.append(kwargs)
            schema_name = kwargs.get("text", {}).get("format", {}).get("name", "")
            if schema_name == "llm_passive_event_judge":
                output_text = json.dumps(_passive_event_judge_payload(action="long", confidence=0.62), ensure_ascii=False)
                response_id = "resp_playbook_passive_judge"
            else:
                output_text = json.dumps(_passive_technical_pricing_payload(entry_price=100000.0, stop_loss_price=99200.0), ensure_ascii=False)
                response_id = "resp_playbook_passive_price"
            return SimpleNamespace(
                id=response_id,
                model="gpt-5.4-mini",
                output_text=output_text,
                usage=SimpleNamespace(
                    input_tokens=240,
                    input_tokens_details=SimpleNamespace(cached_tokens=0),
                    output_tokens=90,
                    output_tokens_details=SimpleNamespace(reasoning_tokens=24),
                    total_tokens=330,
                ),
                output=[SimpleNamespace(type="message")],
            )

    engine = object.__new__(uma.DiscretionaryLLMEngine)
    engine.client = SimpleNamespace(responses=FakeResponses())
    engine.active_model = "gpt-5.4-mini"
    engine.passive_model = "gpt-5.4-mini"
    engine.symbol = "BTC"
    engine.default_search_mode = "context_only"
    engine.active_search_mode = "always"
    engine.passive_search_mode = "context_only"
    engine.active_reasoning_effort = "high"
    engine.passive_reasoning_effort = "medium"
    engine.passive_relevance_threshold = 0.35
    engine.execute_now_confidence_threshold = 0.7
    engine.force_active_news_context = True
    engine.force_passive_news_context = False
    engine.event_recent_window_hours = 72.0
    engine.audit_callback = None
    engine.last_call_debug = {}
    engine.latest_helper_market_mainline_context = {
        "current_move_logic_mainline": "BTC is trading on a ceasefire-driven risk-on unwind in energy and macro hedges.",
        "diagnostic_instruments": ["Brent crude", "Nasdaq 100", "DXY"],
    }
    engine.latest_helper_market_mainline_debug = {
        "winner_display_name": "BTC-USDC",
        "phase": "market_news_context",
    }

    playbook, mode = engine.get_playbook(
        user_query="test passive query",
        event_tape=[{"headline": "ceasefire headline"}],
        recent_events=[{"headline": "ceasefire headline"}],
        trigger_reason="passive_event_trigger",
        trigger_event={"headline": "ceasefire headline"},
        has_live_position=True,
        trade_symbol_context={
            "trade_symbol_key": "BTC_USDC",
            "display_name": "BTC-USDC",
            "execution_symbol": "BTC",
            "tradable_on_hyperliquid": True,
            "current_price": 100000.0,
            "symbol_position": {"symbol": "BTC", "side": "long", "size": 0.1, "max_leverage": 40},
            "market_spec": {"symbol": "BTC", "max_leverage": 40},
        },
        active_symbol="BTC",
    )

    assert mode == "context_enriched_with_web"
    assert playbook.selected_symbol == "BTC-USDC"
    assert len(request_log) == 3
    passive_payload = json.loads(request_log[0]["input"][1]["content"][0]["text"])
    assert "selected_instrument" not in passive_payload["market_mainline_context"]
    assert passive_payload["market_mainline_context"]["current_move_logic_mainline"] == (
        "BTC is trading on a ceasefire-driven risk-on unwind in energy and macro hedges."
    )
    assert passive_payload["market_mainline_context"]["diagnostic_instruments"] == ["DXY"]
    assert passive_payload["trigger_event"]["headline"] == "ceasefire headline"

def test_engine_adds_stable_prompt_cache_key_for_request_family(uma):
    captured = {}

    class FakeResponses:
        def create(self, **kwargs):
            captured.clear()
            captured.update(kwargs)
            return SimpleNamespace(id="resp_prompt_cache")

    engine = object.__new__(uma.DiscretionaryLLMEngine)
    engine.client = SimpleNamespace(responses=FakeResponses())
    engine.openai_max_attempts = 1
    engine.openai_retry_delay_seconds = 0.0
    engine.prompt_cache_enabled = True
    engine.prompt_cache_key_prefix = "market-agent"

    def request(system_prompt, user_prompt):
        return engine._responses_create_with_retry(
            phase="playbook",
            timeout_seconds=1.0,
            model="gpt-5.4",
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
        )

    request("shared system prompt", "first user payload")
    first_key = captured["prompt_cache_key"]
    assert captured["input"][0]["role"] == "system"
    assert first_key == "market-agent-playbook-gpt-5-4"

    request("shared system prompt", "second user payload")
    assert captured["prompt_cache_key"] == first_key

    request("changed system prompt", "third user payload")
    assert captured["prompt_cache_key"] == first_key

    engine.prompt_cache_enabled = False
    request("shared system prompt", "cache disabled")
    assert "prompt_cache_key" not in captured


def test_playbook_system_prompt_keeps_variant_guidance_after_shared_prefix(uma):
    engine = object.__new__(uma.DiscretionaryLLMEngine)
    engine.execute_now_confidence_threshold = 0.65

    active_prompt = engine._build_system_prompt("verified", "manual_request")
    passive_prompt = engine._build_system_prompt("verified", "passive_event_trigger")
    shared_prefix_length = next(
        (
            index
            for index, (active_char, passive_char) in enumerate(
                zip(active_prompt, passive_prompt)
            )
            if active_char != passive_char
        ),
        min(len(active_prompt), len(passive_prompt)),
    )

    assert shared_prefix_length > 2000
    assert active_prompt.index("Think the probability") == shared_prefix_length
    assert passive_prompt.index("When recent_events is not empty") == shared_prefix_length


def test_langchain_runtime_forwards_prompt_cache_key(monkeypatch):
    from langchain import chat_models
    from market_agent.langchain_runtime import LangChainResponsesRuntime

    captured = {}

    class FakeModel:
        def invoke(self, messages, **kwargs):
            captured["messages"] = messages
            captured["kwargs"] = kwargs
            return SimpleNamespace(
                id="resp_prompt_cache",
                content="{}",
                content_blocks=[],
                usage_metadata={},
                response_metadata={},
            )

    monkeypatch.setattr(chat_models, "init_chat_model", lambda *args, **kwargs: FakeModel())

    runtime = LangChainResponsesRuntime(api_key="test-key")
    runtime.create(
        timeout=1.0,
        model="gpt-5.4",
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": "shared system prompt"}]},
            {"role": "user", "content": [{"type": "input_text", "text": "dynamic payload"}]},
        ],
        prompt_cache_key="market-agent-playbook-deadbeef",
    )

    assert captured["messages"][0].type == "system"
    assert captured["kwargs"]["prompt_cache_key"] == "market-agent-playbook-deadbeef"
