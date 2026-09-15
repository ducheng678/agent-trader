from __future__ import annotations

from fastapi.testclient import TestClient

from web_trade.backend.web_trade.api import create_app


class FakeService:
    def __init__(self):
        self.tpsl_calls = []

    def session(self):
        return {
            "network": "testnet",
            "account_address": "0x1234567890abcdef1234567890abcdef12345678",
            "account_address_masked": "0x1234...5678",
            "live_trading": False,
        }

    def markets(self):
        return [{"symbol": "BTC", "display_name": "BTC-USDC"}]

    def account(self):
        return {"positions": [], "open_orders": []}

    def account_history(self, window_days=90):
        return {
            "account_address": "0x1234567890abcdef1234567890abcdef12345678",
            "window_days": window_days,
            "trade_history": [{"coin": "BTC", "oid": 1}],
            "funding_history": [],
            "order_history": [],
        }

    def margin_limits(self, symbol, safety_buffer_usd=None):
        return {"enabled": True, "symbol": symbol, "max_add_margin_usd": 100.0, "max_remove_margin_usd": 50.0}

    def market_book(self, symbol):
        return {
            "symbol": symbol,
            "time": 1700000123000,
            "mid_price": 100.5,
            "bids": [{"price": 100.0, "size": 2.0, "total": 2.0, "orders": 3}],
            "asks": [{"price": 101.0, "size": 1.5, "total": 1.5, "orders": 2}],
        }

    def market_bars(self, symbol, resolution="1", from_s=None, to_s=None, count_back=None):
        return {
            "symbol": symbol,
            "resolution": resolution,
            "interval": "1m",
            "bars": [{"time": 1700000000, "open": 100.0, "high": 110.0, "low": 95.0, "close": 105.0, "volume": 12.5}],
            "no_data": False,
        }

    def favorite_markets(self):
        return [{"symbol": "BTC", "display_name": "BTC-USDC"}]

    def update_favorite_markets(self, symbols):
        return [{"symbol": symbol, "display_name": f"{symbol}-USDC"} for symbol in symbols if symbol == "BTC"]

    def place_order(self, **kwargs):
        return {"accepted": True, **kwargs}

    def set_position_tpsl(self, **kwargs):
        self.tpsl_calls.append(kwargs)
        return {"accepted": True, **kwargs}


def test_api_requires_bearer_token(monkeypatch):
    monkeypatch.setenv("WEB_ADMIN_TOKEN", "secret-token")
    client = TestClient(create_app(service=FakeService()))

    unauthenticated = client.get("/api/session")
    authenticated = client.get("/api/session", headers={"Authorization": "Bearer secret-token"})

    assert unauthenticated.status_code == 401
    assert authenticated.status_code == 200
    assert authenticated.json()["account_address"] is None
    assert authenticated.json()["account_address_masked"] == "0x1234...5678"


def test_api_exposes_margin_limits_and_order_route(monkeypatch):
    monkeypatch.setenv("WEB_ADMIN_TOKEN", "secret-token")
    client = TestClient(create_app(service=FakeService()))
    headers = {"Authorization": "Bearer secret-token"}

    limits = client.get("/api/positions/BTC/margin-limits", headers=headers)
    order = client.post(
        "/api/orders",
        headers=headers,
        json={
            "symbol": "BTC",
            "order_type": "market",
            "side": "long",
            "margin_usd": 200,
            "leverage": 5,
            "close_all": True,
        },
    )

    assert limits.status_code == 200
    assert limits.json()["max_add_margin_usd"] == 100.0
    assert order.status_code == 200
    assert order.json()["accepted"] is True
    assert order.json()["close_all"] is True


def test_api_exposes_position_tpsl_route(monkeypatch):
    monkeypatch.setenv("WEB_ADMIN_TOKEN", "secret-token")
    service = FakeService()
    client = TestClient(create_app(service=service))

    response = client.post(
        "/api/positions/BTC/tpsl",
        headers={"Authorization": "Bearer secret-token"},
        json={"take_profit_price": 105000, "stop_loss_price": 99000},
    )

    assert response.status_code == 200
    assert response.json()["accepted"] is True
    assert service.tpsl_calls == [
        {"symbol": "BTC", "take_profit_price": 105000.0, "stop_loss_price": 99000.0}
    ]


def test_api_exposes_market_book_route(monkeypatch):
    monkeypatch.setenv("WEB_ADMIN_TOKEN", "secret-token")
    client = TestClient(create_app(service=FakeService()))

    response = client.get("/api/market/BTC/book", headers={"Authorization": "Bearer secret-token"})

    assert response.status_code == 200
    assert response.json()["mid_price"] == 100.5
    assert response.json()["bids"][0]["total"] == 2.0


def test_api_exposes_account_history_without_real_address(monkeypatch):
    monkeypatch.setenv("WEB_ADMIN_TOKEN", "secret-token")
    client = TestClient(create_app(service=FakeService()))

    response = client.get("/api/account/history?window_days=30", headers={"Authorization": "Bearer secret-token"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["window_days"] == 30
    assert payload["trade_history"][0]["coin"] == "BTC"
    assert "account_address" not in payload


def test_api_exposes_tradingview_bars_route(monkeypatch):
    monkeypatch.setenv("WEB_ADMIN_TOKEN", "secret-token")
    client = TestClient(create_app(service=FakeService()))

    response = client.get(
        "/api/market/BTC/bars?resolution=60&from=1700000000&to=1700003600&count_back=100",
        headers={"Authorization": "Bearer secret-token"},
    )

    assert response.status_code == 200
    assert response.json()["symbol"] == "BTC"
    assert response.json()["resolution"] == "60"
    assert response.json()["bars"][0]["volume"] == 12.5


def test_api_exposes_file_backed_favorites_routes(monkeypatch):
    monkeypatch.setenv("WEB_ADMIN_TOKEN", "secret-token")
    client = TestClient(create_app(service=FakeService()))
    headers = {"Authorization": "Bearer secret-token"}

    before = client.get("/api/favorites/markets", headers=headers)
    after = client.put("/api/favorites/markets", headers=headers, json={"symbols": ["BTC", "DOGE"]})

    assert before.status_code == 200
    assert before.json()[0]["symbol"] == "BTC"
    assert after.status_code == 200
    assert after.json() == [{"symbol": "BTC", "display_name": "BTC-USDC"}]


def test_create_app_defers_default_service_initialization(monkeypatch):
    from web_trade.backend.web_trade import api as api_module

    calls = {"count": 0}

    class FailingService:
        def __init__(self):
            calls["count"] += 1
            raise RuntimeError("service initialized")

    monkeypatch.setattr(api_module, "WebTradeService", FailingService)
    application = api_module.create_app()
    assert application.title == "Private Hyperliquid Trade Web"
    assert calls["count"] == 0
