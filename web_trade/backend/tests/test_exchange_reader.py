from __future__ import annotations

import pytest

from market_agent.exchange import HyperliquidRestReader


def _reader_with_position() -> HyperliquidRestReader:
    reader = HyperliquidRestReader.__new__(HyperliquidRestReader)
    reader.account_address = "0x1234567890abcdef1234567890abcdef12345678"
    reader.network = "testnet"

    def post_info(payload):
        payload_type = payload.get("type")
        if payload_type == "spotClearinghouseState":
            return {
                "balances": [{"coin": "USDC", "token": 0, "total": "1000", "hold": "0"}],
                "tokenToAvailableAfterMaintenance": [[0, "900"]],
            }
        if payload_type == "userAbstraction":
            return ""
        if payload_type == "clearinghouseState":
            return {
                "withdrawable": "500",
                "assetPositions": [
                    {
                        "position": {
                            "coin": "BTC",
                            "szi": "0.1",
                            "entryPx": "100000",
                            "unrealizedPnl": "120",
                            "returnOnEquity": "0.12",
                            "leverage": {"value": 10},
                            "liquidationPx": "92000",
                            "marginUsed": "1000",
                            "cumFunding": {
                                "allTime": "-3.5",
                                "sinceOpen": "-1.25",
                                "sinceChange": "-0.5",
                            },
                        }
                    }
                ],
            }
        raise AssertionError(f"unexpected payload: {payload}")

    reader.post_info = post_info
    reader.list_perp_dex_names = lambda: []
    reader.get_market_catalog = lambda: {"BTC": {"max_leverage": 40, "only_isolated": True}}
    reader.get_market_spec = lambda symbol: {"max_leverage": 40, "only_isolated": True}
    reader.get_mids = lambda dex="": {"BTC": "101000"}
    reader.resolve_execution_symbol = lambda symbol: symbol
    return reader


def test_all_positions_maps_hyperliquid_cumulative_funding_fields():
    account = _reader_with_position().get_all_positions()

    position = account["positions"][0]

    assert position["funding_all_time_usd"] == pytest.approx(-3.5)
    assert position["funding_since_open_usd"] == pytest.approx(-1.25)
    assert position["funding_since_change_usd"] == pytest.approx(-0.5)


def test_selected_position_context_maps_hyperliquid_cumulative_funding_fields():
    context = _reader_with_position().get_selected_symbol_position_context("BTC")

    position = context["position_snapshot"]

    assert position["funding_all_time_usd"] == pytest.approx(-3.5)
    assert position["funding_since_open_usd"] == pytest.approx(-1.25)
    assert position["funding_since_change_usd"] == pytest.approx(-0.5)


def test_market_catalog_maps_asset_context_fields_for_favorites_bar():
    reader = HyperliquidRestReader.__new__(HyperliquidRestReader)
    reader._market_catalog_cache = None
    reader._meta_cache_by_dex = {}
    reader.list_perp_dex_names = lambda: []

    def post_info(payload):
        payload_type = payload.get("type")
        if payload_type == "metaAndAssetCtxs":
            return [
                {
                    "universe": [
                        {
                            "name": "BTC",
                            "szDecimals": 5,
                            "maxLeverage": 40,
                            "onlyIsolated": False,
                        }
                    ]
                },
                [
                    {
                        "midPx": "101000",
                        "markPx": "101050",
                        "prevDayPx": "100000",
                        "dayNtlVlm": "123456",
                    }
                ],
            ]
        if payload_type == "meta":
            return {
                "universe": [
                    {
                        "name": "BTC",
                        "szDecimals": 5,
                        "maxLeverage": 40,
                        "onlyIsolated": False,
                    }
                ]
            }
        raise AssertionError(f"unexpected payload: {payload}")

    reader.post_info = post_info

    catalog = reader.get_market_catalog()

    assert catalog["BTC"]["mid_price"] == pytest.approx(101000.0)
    assert catalog["BTC"]["mark_price"] == pytest.approx(101050.0)
    assert catalog["BTC"]["prev_day_price"] == pytest.approx(100000.0)
    assert catalog["BTC"]["day_volume_usd"] == pytest.approx(123456.0)
