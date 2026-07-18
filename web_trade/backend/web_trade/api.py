from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from web_trade.backend.web_trade.service import WebTradeService


class LeverageRequest(BaseModel):
    leverage: int


class MarginRequest(BaseModel):
    direction: str
    amount_usd: float
    safety_buffer_usd: Optional[float] = None


class OrderRequest(BaseModel):
    symbol: str
    order_type: str
    side: str
    margin_usd: float = 0.0
    leverage: int = 0
    limit_price: float = 0.0
    reduce_only: bool = False
    close_all: bool = False
    position_action: str = "open"


class TpslRequest(BaseModel):
    take_profit_price: float = 0.0
    stop_loss_price: float = 0.0


class FavoritesRequest(BaseModel):
    symbols: list[str]


def _require_token(authorization: str = Header(default="")) -> None:
    expected = str(os.getenv("WEB_ADMIN_TOKEN", "") or "").strip()
    if not expected:
        raise HTTPException(status_code=500, detail="WEB_ADMIN_TOKEN is required")
    prefix = "Bearer "
    if not authorization.startswith(prefix) or authorization[len(prefix) :].strip() != expected:
        raise HTTPException(status_code=401, detail="Unauthorized")


def _public_session(payload: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(payload)
    result["account_address"] = None
    return result


def _bad_request(exc: ValueError) -> HTTPException:
    return HTTPException(status_code=400, detail=str(exc))


def create_app(service: WebTradeService | Any | None = None) -> FastAPI:
    app = FastAPI(title="Private Hyperliquid Trade Web", version="0.1.0")
    trade_service = service if service is not None else WebTradeService()

    @app.get("/api/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/session", dependencies=[Depends(_require_token)])
    def session() -> Dict[str, Any]:
        return _public_session(trade_service.session())

    @app.get("/api/markets", dependencies=[Depends(_require_token)])
    def markets() -> Any:
        return trade_service.markets()

    @app.get("/api/account", dependencies=[Depends(_require_token)])
    def account() -> Any:
        payload = trade_service.account()
        if isinstance(payload, dict):
            payload = dict(payload)
            payload["account_address"] = None
        return payload

    @app.get("/api/account/history", dependencies=[Depends(_require_token)])
    def account_history(window_days: int = 90) -> Any:
        payload = trade_service.account_history(window_days=window_days)
        if isinstance(payload, dict):
            payload = dict(payload)
            payload.pop("account_address", None)
        return payload

    @app.get("/api/market/{symbol}/snapshot", dependencies=[Depends(_require_token)])
    def market_snapshot(symbol: str, interval: str = "1m", window_seconds: int = 3600) -> Any:
        return trade_service.market_snapshot(symbol, interval=interval, window_seconds=window_seconds)

    @app.get("/api/market/{symbol}/book", dependencies=[Depends(_require_token)])
    def market_book(symbol: str) -> Any:
        return trade_service.market_book(symbol)

    @app.get("/api/market/{symbol}/bars", dependencies=[Depends(_require_token)])
    def market_bars(
        symbol: str,
        resolution: str = "1",
        from_s: Optional[int] = Query(default=None, alias="from"),
        to_s: Optional[int] = Query(default=None, alias="to"),
        count_back: Optional[int] = None,
    ) -> Any:
        try:
            return trade_service.market_bars(
                symbol,
                resolution=resolution,
                from_s=from_s,
                to_s=to_s,
                count_back=count_back,
            )
        except ValueError as exc:
            raise _bad_request(exc) from exc

    @app.get("/api/favorites/markets", dependencies=[Depends(_require_token)])
    def favorite_markets() -> Any:
        return trade_service.favorite_markets()

    @app.put("/api/favorites/markets", dependencies=[Depends(_require_token)])
    def update_favorite_markets(request: FavoritesRequest) -> Any:
        return trade_service.update_favorite_markets(request.symbols)

    @app.get("/api/positions/{symbol}/margin-limits", dependencies=[Depends(_require_token)])
    def margin_limits(symbol: str, safety_buffer_usd: Optional[float] = None) -> Any:
        try:
            return trade_service.margin_limits(symbol, safety_buffer_usd=safety_buffer_usd)
        except ValueError as exc:
            raise _bad_request(exc) from exc

    @app.post("/api/positions/{symbol}/leverage", dependencies=[Depends(_require_token)])
    def rebalance_leverage(symbol: str, request: LeverageRequest) -> Any:
        try:
            return trade_service.rebalance_leverage(symbol, request.leverage)
        except ValueError as exc:
            raise _bad_request(exc) from exc

    @app.post("/api/positions/{symbol}/margin", dependencies=[Depends(_require_token)])
    def update_margin(symbol: str, request: MarginRequest) -> Any:
        try:
            return trade_service.update_isolated_margin(
                symbol,
                request.direction,
                request.amount_usd,
                safety_buffer_usd=request.safety_buffer_usd,
            )
        except ValueError as exc:
            raise _bad_request(exc) from exc

    @app.post("/api/positions/{symbol}/tpsl", dependencies=[Depends(_require_token)])
    def set_position_tpsl(symbol: str, request: TpslRequest) -> Any:
        try:
            return trade_service.set_position_tpsl(
                symbol=symbol,
                take_profit_price=request.take_profit_price,
                stop_loss_price=request.stop_loss_price,
            )
        except ValueError as exc:
            raise _bad_request(exc) from exc

    @app.post("/api/orders", dependencies=[Depends(_require_token)])
    def place_order(request: OrderRequest) -> Any:
        try:
            return trade_service.place_order(
                symbol=request.symbol,
                order_type=request.order_type,
                side=request.side,
                margin_usd=request.margin_usd,
                leverage=request.leverage,
                limit_price=request.limit_price,
                reduce_only=request.reduce_only,
                close_all=request.close_all,
                position_action=request.position_action,
            )
        except ValueError as exc:
            raise _bad_request(exc) from exc

    frontend_dist = Path(__file__).resolve().parents[2] / "frontend" / "dist"
    index_html = frontend_dist / "index.html"
    assets_dir = frontend_dist / "assets"
    if index_html.exists():
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

        @app.get("/", include_in_schema=False)
        def frontend_index() -> FileResponse:
            return FileResponse(index_html)

        @app.get("/{full_path:path}", include_in_schema=False)
        def frontend_fallback(full_path: str) -> FileResponse:
            if full_path.startswith("api/"):
                raise HTTPException(status_code=404, detail="Not Found")
            requested = (frontend_dist / full_path).resolve()
            if frontend_dist.resolve() in requested.parents and requested.is_file():
                return FileResponse(requested)
            return FileResponse(index_html)

    return app


app = create_app()
