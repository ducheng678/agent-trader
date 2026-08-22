from __future__ import annotations

import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Mapping, Optional, Tuple

from market_agent.symbols import canonicalize_execution_symbol


DEFAULT_STATE_PATH = Path(__file__).resolve().parent / "runtime" / "state.json"


def _decimal_env(env: Mapping[str, str], name: str, default: str) -> Decimal:
    raw = str(env.get(name, default) or default).strip()
    try:
        value = Decimal(raw)
    except InvalidOperation as exc:
        raise ValueError(f"{name} must be a decimal number, got {raw!r}") from exc
    if not value.is_finite():
        raise ValueError(f"{name} must be finite")
    return value


def _int_env(env: Mapping[str, str], name: str, default: int) -> int:
    raw = str(env.get(name, default) or default).strip()
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _bool_env(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = str(env.get(name, str(default)) or "").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false, got {raw!r}")


def _symbols_env(env: Mapping[str, str]) -> Tuple[str, ...]:
    raw = str(env.get("FUNDING_HEDGE_SYMBOLS", "xyz:SKHX") or "")
    symbols = []
    for token in raw.split(","):
        symbol = canonicalize_execution_symbol(token)
        if symbol and symbol not in symbols:
            symbols.append(symbol)
    return tuple(symbols)


@dataclass(frozen=True)
class StrategyConfig:
    """Configuration for an independent-account temporary funding hedge.

    Rates are decimals, not percentages.  For example, ``0.00045`` is
    4.5 basis points.  A zero ``max_hedge_notional_usd`` means no configured
    cap; exchange/account limits still apply.
    """

    symbols: Tuple[str, ...] = ("xyz:SKHX",)
    network: str = "mainnet"
    primary_account_address: str = ""
    hedge_account_address: str = ""
    hedge_account_kind: str = "subaccount"
    hedge_ratio: Decimal = Decimal("1")
    hedge_open_lead_seconds: int = 120
    latest_open_cutoff_seconds: int = 15
    funding_confirmation_timeout_seconds: int = 120
    max_funding_data_age_seconds: int = 30
    max_clock_skew_seconds: int = 5
    min_position_notional_usd: Decimal = Decimal("10")
    min_hedge_notional_usd: Decimal = Decimal("10")
    max_hedge_notional_usd: Decimal = Decimal("0")
    min_expected_saving_usd: Decimal = Decimal("0.50")
    min_available_margin_usd: Decimal = Decimal("10")
    assumed_hedge_leverage: Decimal = Decimal("1")
    hedge_margin_mode: str = "cross"
    margin_safety_multiplier: Decimal = Decimal("1.20")
    slippage_rate_per_order: Decimal = Decimal("0.0002")
    hip3_extra_fee_rate_per_order: Decimal = Decimal("0")
    hip3_extra_fee_rate_known: bool = False
    risk_buffer_rate: Decimal = Decimal("0.0002")
    cost_safety_multiplier: Decimal = Decimal("1.25")
    fallback_taker_fee_rate: Decimal = Decimal("0.00045")
    max_order_slippage: Decimal = Decimal("0.002")
    require_verified_subaccount: bool = True
    execution_enabled: bool = False
    ambiguous_order_retry_seconds: int = 15
    open_reconciliation_timeout_seconds: int = 30
    http_timeout_seconds: int = 5
    action_expiry_seconds: int = 15
    poll_seconds: int = 5
    state_path: Path = DEFAULT_STATE_PATH

    def __post_init__(self) -> None:
        canonical_symbols = tuple(
            dict.fromkeys(
                canonicalize_execution_symbol(symbol)
                for symbol in self.symbols
                if canonicalize_execution_symbol(symbol)
            )
        )
        object.__setattr__(self, "symbols", canonical_symbols)
        object.__setattr__(
            self, "primary_account_address", self.primary_account_address.strip()
        )
        object.__setattr__(
            self, "hedge_account_address", self.hedge_account_address.strip()
        )
        kind = self.hedge_account_kind.strip().lower()
        object.__setattr__(self, "hedge_account_kind", kind)
        network = self.network.strip().lower()
        object.__setattr__(self, "network", network)
        margin_mode = self.hedge_margin_mode.strip().lower()
        object.__setattr__(self, "hedge_margin_mode", margin_mode)

        if network not in {"mainnet", "testnet"}:
            raise ValueError("network must be mainnet or testnet")
        if kind not in {"subaccount", "wallet"}:
            raise ValueError("hedge_account_kind must be subaccount or wallet")
        if margin_mode not in {"cross", "isolated"}:
            raise ValueError("hedge_margin_mode must be cross or isolated")
        if not (Decimal("0") < self.hedge_ratio <= Decimal("1")):
            raise ValueError("hedge_ratio must be greater than 0 and at most 1")
        if self.hedge_open_lead_seconds <= 0:
            raise ValueError("hedge_open_lead_seconds must be positive")
        if self.latest_open_cutoff_seconds < 0:
            raise ValueError("latest_open_cutoff_seconds must be non-negative")
        if self.latest_open_cutoff_seconds >= self.hedge_open_lead_seconds:
            raise ValueError(
                "latest_open_cutoff_seconds must be less than hedge_open_lead_seconds"
            )
        if self.funding_confirmation_timeout_seconds <= 0:
            raise ValueError("funding_confirmation_timeout_seconds must be positive")
        if self.max_funding_data_age_seconds <= 0:
            raise ValueError("max_funding_data_age_seconds must be positive")
        if self.max_clock_skew_seconds < 0:
            raise ValueError("max_clock_skew_seconds must be non-negative")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be positive")
        for name in (
            "min_position_notional_usd",
            "min_hedge_notional_usd",
            "max_hedge_notional_usd",
            "min_expected_saving_usd",
            "min_available_margin_usd",
            "slippage_rate_per_order",
            "hip3_extra_fee_rate_per_order",
            "risk_buffer_rate",
            "fallback_taker_fee_rate",
            "max_order_slippage",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.assumed_hedge_leverage <= 0:
            raise ValueError("assumed_hedge_leverage must be positive")
        if self.assumed_hedge_leverage != self.assumed_hedge_leverage.to_integral_value():
            raise ValueError("assumed_hedge_leverage must be an integer")
        if self.margin_safety_multiplier < 1:
            raise ValueError("margin_safety_multiplier must be at least 1")
        if self.cost_safety_multiplier < 1:
            raise ValueError("cost_safety_multiplier must be at least 1")
        for name in (
            "ambiguous_order_retry_seconds",
            "open_reconciliation_timeout_seconds",
            "http_timeout_seconds",
            "action_expiry_seconds",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.action_expiry_seconds <= self.http_timeout_seconds:
            raise ValueError(
                "action_expiry_seconds must exceed http_timeout_seconds"
            )
        if self.execution_enabled:
            if not self.primary_account_address:
                raise ValueError("HL_ACCOUNT_ADDRESS is required for execution")
            if not self.hedge_account_address:
                raise ValueError(
                    "FUNDING_HEDGE_ACCOUNT_ADDRESS is required for execution"
                )
            if self.primary_account_address.lower() == self.hedge_account_address.lower():
                raise ValueError("primary and hedge account addresses must differ")

    @classmethod
    def from_env(cls, env: Optional[Mapping[str, str]] = None) -> "StrategyConfig":
        source = os.environ if env is None else env
        return cls(
            symbols=_symbols_env(source),
            network=str(source.get("HYPERLIQUID_NETWORK", "mainnet") or "mainnet"),
            primary_account_address=str(source.get("HL_ACCOUNT_ADDRESS", "") or ""),
            hedge_account_address=str(
                source.get("FUNDING_HEDGE_ACCOUNT_ADDRESS", "") or ""
            ),
            hedge_account_kind=str(
                source.get("FUNDING_HEDGE_ACCOUNT_KIND", "subaccount")
                or "subaccount"
            ),
            hedge_ratio=_decimal_env(source, "FUNDING_HEDGE_RATIO", "1"),
            hedge_open_lead_seconds=_int_env(
                source, "FUNDING_HEDGE_OPEN_LEAD_SECONDS", 120
            ),
            latest_open_cutoff_seconds=_int_env(
                source, "FUNDING_HEDGE_LATEST_OPEN_CUTOFF_SECONDS", 15
            ),
            funding_confirmation_timeout_seconds=_int_env(
                source, "FUNDING_HEDGE_CONFIRMATION_TIMEOUT_SECONDS", 120
            ),
            max_funding_data_age_seconds=_int_env(
                source, "FUNDING_HEDGE_MAX_DATA_AGE_SECONDS", 30
            ),
            max_clock_skew_seconds=_int_env(
                source, "FUNDING_HEDGE_MAX_CLOCK_SKEW_SECONDS", 5
            ),
            min_position_notional_usd=_decimal_env(
                source, "FUNDING_HEDGE_MIN_POSITION_NOTIONAL_USD", "10"
            ),
            min_hedge_notional_usd=_decimal_env(
                source, "FUNDING_HEDGE_MIN_ORDER_NOTIONAL_USD", "10"
            ),
            max_hedge_notional_usd=_decimal_env(
                source, "FUNDING_HEDGE_MAX_NOTIONAL_USD", "0"
            ),
            min_expected_saving_usd=_decimal_env(
                source, "FUNDING_HEDGE_MIN_EXPECTED_SAVING_USD", "0.50"
            ),
            min_available_margin_usd=_decimal_env(
                source, "FUNDING_HEDGE_MIN_AVAILABLE_MARGIN_USD", "10"
            ),
            assumed_hedge_leverage=_decimal_env(
                source, "FUNDING_HEDGE_ASSUMED_LEVERAGE", "1"
            ),
            hedge_margin_mode=str(
                source.get("FUNDING_HEDGE_MARGIN_MODE", "cross") or "cross"
            ),
            margin_safety_multiplier=_decimal_env(
                source, "FUNDING_HEDGE_MARGIN_SAFETY_MULTIPLIER", "1.20"
            ),
            slippage_rate_per_order=_decimal_env(
                source, "FUNDING_HEDGE_SLIPPAGE_RATE_PER_ORDER", "0.0002"
            ),
            hip3_extra_fee_rate_per_order=_decimal_env(
                source, "FUNDING_HEDGE_HIP3_EXTRA_FEE_RATE_PER_ORDER", "0"
            ),
            hip3_extra_fee_rate_known=_bool_env(
                source, "FUNDING_HEDGE_HIP3_EXTRA_FEE_RATE_KNOWN", False
            ),
            risk_buffer_rate=_decimal_env(
                source, "FUNDING_HEDGE_RISK_BUFFER_RATE", "0.0002"
            ),
            cost_safety_multiplier=_decimal_env(
                source, "FUNDING_HEDGE_COST_SAFETY_MULTIPLIER", "1.25"
            ),
            fallback_taker_fee_rate=_decimal_env(
                source, "FUNDING_HEDGE_FALLBACK_TAKER_FEE_RATE", "0.00045"
            ),
            max_order_slippage=_decimal_env(
                source, "FUNDING_HEDGE_MAX_ORDER_SLIPPAGE", "0.002"
            ),
            require_verified_subaccount=_bool_env(
                source, "FUNDING_HEDGE_REQUIRE_VERIFIED_SUBACCOUNT", True
            ),
            execution_enabled=_bool_env(
                source, "FUNDING_HEDGE_EXECUTE", False
            ),
            ambiguous_order_retry_seconds=_int_env(
                source, "FUNDING_HEDGE_AMBIGUOUS_RETRY_SECONDS", 15
            ),
            open_reconciliation_timeout_seconds=_int_env(
                source, "FUNDING_HEDGE_OPEN_RECONCILIATION_TIMEOUT_SECONDS", 30
            ),
            http_timeout_seconds=_int_env(
                source, "FUNDING_HEDGE_HTTP_TIMEOUT_SECONDS", 5
            ),
            action_expiry_seconds=_int_env(
                source, "FUNDING_HEDGE_ACTION_EXPIRY_SECONDS", 15
            ),
            poll_seconds=_int_env(source, "FUNDING_HEDGE_POLL_SECONDS", 5),
            state_path=Path(
                str(
                    source.get(
                        "FUNDING_HEDGE_STATE_PATH",
                        str(DEFAULT_STATE_PATH),
                    )
                    or str(DEFAULT_STATE_PATH)
                )
            ),
        )
