from __future__ import annotations

import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Callable, Mapping, Optional

from market_agent.exchange import HyperliquidRestReader
from market_agent.symbols import canonicalize_execution_symbol, split_execution_symbol

from .config import StrategyConfig


@dataclass(frozen=True)
class ExecutionResult:
    accepted: bool
    requested_size: Decimal
    reported_filled_size: Decimal
    raw: Any
    error: str = ""


def _exchange_error(payload: Any) -> str:
    if isinstance(payload, Mapping):
        if str(payload.get("status", "") or "").lower() == "err":
            return str(payload.get("response") or payload.get("error") or "exchange error")
        error = payload.get("error")
        if isinstance(error, str) and error.strip():
            return error.strip()
        for value in payload.values():
            nested = _exchange_error(value)
            if nested:
                return nested
    elif isinstance(payload, list):
        for value in payload:
            nested = _exchange_error(value)
            if nested:
                return nested
    return ""


def _reported_fill(payload: Any) -> Decimal:
    total = Decimal("0")
    if isinstance(payload, Mapping):
        filled = payload.get("filled")
        if isinstance(filled, Mapping):
            value = filled.get("totalSz", filled.get("sz"))
            try:
                total += Decimal(str(value))
            except Exception:
                pass
        for value in payload.values():
            total += _reported_fill(value)
    elif isinstance(payload, list):
        for value in payload:
            total += _reported_fill(value)
    return total


def _accepted_response(payload: Any) -> bool:
    return (
        isinstance(payload, Mapping)
        and str(payload.get("status", "") or "").strip().lower() == "ok"
        and not _exchange_error(payload)
    )


class HyperliquidHedgeExecutor:
    """The only write boundary; it is permanently bound to the hedge address."""

    def __init__(
        self,
        config: StrategyConfig,
        reader: HyperliquidRestReader,
        *,
        exchange_factory: Optional[Callable[..., Any]] = None,
        wallet_factory: Optional[Callable[[str], Any]] = None,
        env: Optional[Mapping[str, str]] = None,
    ) -> None:
        self.config = config
        self.reader = reader
        self.env = os.environ if env is None else env
        self.exchange_factory = exchange_factory
        self.wallet_factory = wallet_factory
        self._exchange: Any = None
        self._loaded_dexes: set[str] = set()
        primary = config.primary_account_address or str(
            getattr(reader, "account_address", "") or ""
        )
        if not primary or not config.hedge_account_address:
            raise ValueError("both primary and hedge account addresses are required")
        if primary.lower() == config.hedge_account_address.lower():
            raise ValueError("executor cannot route orders to the primary account")

    @property
    def hedge_address(self) -> str:
        return self.config.hedge_account_address

    def _ensure_exchange(self, required_symbol: str) -> Any:
        required_dex, _ = split_execution_symbol(required_symbol)
        if self._exchange is not None and required_dex in self._loaded_dexes:
            return self._exchange
        secret = str(self.env.get("FUNDING_HEDGE_SECRET_KEY", "") or "").strip()
        if not secret:
            raise RuntimeError("FUNDING_HEDGE_SECRET_KEY is required for execution")

        if self.wallet_factory is None:
            try:
                import eth_account
            except ImportError as exc:
                raise RuntimeError("eth-account is required for execution") from exc
            wallet_factory = eth_account.Account.from_key
        else:
            wallet_factory = self.wallet_factory
        wallet = wallet_factory(secret)

        if self.config.hedge_account_kind == "wallet":
            signer = str(getattr(wallet, "address", "") or "")
            if signer.lower() != self.hedge_address.lower():
                raise RuntimeError(
                    "independent-wallet signer does not match FUNDING_HEDGE_ACCOUNT_ADDRESS"
                )

        if self.exchange_factory is None:
            try:
                from hyperliquid.exchange import Exchange
            except ImportError as exc:
                raise RuntimeError(
                    "hyperliquid-python-sdk is required for execution"
                ) from exc
            exchange_factory = Exchange
        else:
            exchange_factory = self.exchange_factory

        dexes = [""]
        for loaded in sorted(self._loaded_dexes):
            if loaded and loaded not in dexes:
                dexes.append(loaded)
        for symbol in self.config.symbols:
            dex, _ = split_execution_symbol(symbol)
            if dex and dex not in dexes:
                dexes.append(dex)
        if required_dex and required_dex not in dexes:
            dexes.append(required_dex)
        kwargs = {
            "account_address": self.hedge_address,
            "perp_dexs": dexes,
            "timeout": self.config.http_timeout_seconds,
        }
        if self.config.hedge_account_kind == "subaccount":
            # Hyperliquid's SDK names the action-routing field vault_address;
            # _post_action serializes it as vaultAddress for subaccounts/vaults.
            kwargs["vault_address"] = self.hedge_address
        else:
            kwargs["vault_address"] = None
        self._exchange = exchange_factory(wallet, self.reader.base, **kwargs)
        self._loaded_dexes = set(dexes)
        return self._exchange

    def _validate_order(
        self,
        symbol: str,
        size: Decimal,
        cloid: str,
        *,
        require_allowlisted: bool,
    ) -> str:
        canonical = canonicalize_execution_symbol(symbol)
        if require_allowlisted and canonical not in self.config.symbols:
            raise ValueError(f"symbol {canonical!r} is not in FUNDING_HEDGE_SYMBOLS")
        if size <= 0:
            raise ValueError("order size must be positive")
        if not cloid.startswith("0x") or len(cloid) != 34:
            raise ValueError("cloid must be a 16-byte 0x-prefixed hex string")
        return canonical

    @staticmethod
    def _deadline_ms(submit_deadline: datetime) -> int:
        if submit_deadline.tzinfo is None or submit_deadline.utcoffset() is None:
            raise ValueError("submit_deadline must include a timezone")
        return int(submit_deadline.astimezone(timezone.utc).timestamp() * 1000)

    def _set_action_expiry(
        self, exchange: Any, submit_deadline: Optional[datetime] = None
    ) -> None:
        now_ms = int(time.time() * 1000)
        expiry_ms = now_ms + self.config.action_expiry_seconds * 1000
        if submit_deadline is not None:
            deadline_ms = self._deadline_ms(submit_deadline)
            if now_ms >= deadline_ms:
                raise RuntimeError("hedge open deadline has already passed")
            expiry_ms = min(expiry_ms, deadline_ms)
        exchange.set_expires_after(expiry_ms)

    def _aggressive_ioc_price(
        self, exchange: Any, symbol: str, is_buy: bool
    ) -> float:
        price_fn = getattr(exchange, "_slippage_price", None)
        if not callable(price_fn):
            raise RuntimeError(
                "installed Hyperliquid SDK does not expose the required IOC price helper"
            )
        return float(
            price_fn(
                symbol,
                is_buy,
                float(self.config.max_order_slippage),
                None,
            )
        )

    @staticmethod
    def _cloid(raw: str) -> Any:
        from hyperliquid.utils.types import Cloid

        return Cloid.from_str(raw)

    def open_hedge(
        self,
        symbol: str,
        signed_size: Decimal,
        cloid: str,
        *,
        submit_deadline: datetime,
    ) -> ExecutionResult:
        canonical = self._validate_order(
            symbol, abs(signed_size), cloid, require_allowlisted=True
        )
        exchange = self._ensure_exchange(canonical)
        self._set_action_expiry(exchange, submit_deadline)
        leverage_result = exchange.update_leverage(
            int(self.config.assumed_hedge_leverage),
            canonical,
            is_cross=self.config.hedge_margin_mode == "cross",
        )
        leverage_error = _exchange_error(leverage_result)
        if not _accepted_response(leverage_result):
            raise RuntimeError(
                "failed to set hedge leverage/margin mode: "
                + (leverage_error or "unrecognized exchange response")
            )
        is_buy = signed_size > 0
        limit_price = self._aggressive_ioc_price(exchange, canonical, is_buy)
        # Metadata/mids and updateLeverage can consume the remaining window.
        # Start action expiry only after those reads, immediately before POST.
        self._set_action_expiry(exchange, submit_deadline)
        raw = exchange.order(
            canonical,
            is_buy,
            float(abs(signed_size)),
            limit_price,
            {"limit": {"tif": "Ioc"}},
            reduce_only=False,
            cloid=self._cloid(cloid),
        )
        error = _exchange_error(raw)
        accepted = _accepted_response(raw)
        if not accepted and not error:
            raise RuntimeError("unrecognized exchange response after hedge open")
        return ExecutionResult(
            accepted=accepted,
            requested_size=abs(signed_size),
            reported_filled_size=_reported_fill(raw),
            raw=raw,
            error=error,
        )

    def close_hedge(
        self,
        symbol: str,
        current_signed_size: Decimal,
        size: Decimal,
        cloid: str,
    ) -> ExecutionResult:
        if current_signed_size == 0:
            raise ValueError("current_signed_size must be non-zero for reduce-only close")
        canonical = self._validate_order(
            symbol, size, cloid, require_allowlisted=False
        )
        exchange = self._ensure_exchange(canonical)
        is_buy = current_signed_size < 0
        limit_price = self._aggressive_ioc_price(exchange, canonical, is_buy)
        # Set expiry after the price read so slow /info calls cannot consume
        # the action's entire validity window.
        self._set_action_expiry(exchange)
        raw = exchange.order(
            canonical,
            is_buy,
            float(size),
            limit_price,
            {"limit": {"tif": "Ioc"}},
            reduce_only=True,
            cloid=self._cloid(cloid),
        )
        error = _exchange_error(raw)
        accepted = _accepted_response(raw)
        if not accepted and not error:
            raise RuntimeError("unrecognized exchange response after hedge close")
        return ExecutionResult(
            accepted=accepted,
            requested_size=size,
            reported_filled_size=_reported_fill(raw),
            raw=raw,
            error=error,
        )
