from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple

from market_agent.symbols import canonicalize_execution_symbol


UTC = timezone.utc


def parse_decimal(value: Any, field_name: str) -> Decimal:
    try:
        parsed = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a decimal number") from exc
    if not parsed.is_finite():
        raise ValueError(f"{field_name} must be finite")
    return parsed


def parse_utc_datetime(value: Any, field_name: str) -> datetime:
    if isinstance(value, datetime):
        parsed = value
    else:
        raw = str(value or "").strip()
        if raw.endswith("Z"):
            raw = raw[:-1] + "+00:00"
        try:
            parsed = datetime.fromisoformat(raw)
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an ISO-8601 datetime") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field_name} must include a timezone")
    return parsed.astimezone(UTC)


def decimal_text(value: Decimal) -> str:
    return format(value, "f")


def utc_text(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _canonical_symbol(value: Any) -> str:
    symbol = canonicalize_execution_symbol(str(value or ""))
    if not symbol:
        raise ValueError("symbol is required")
    return symbol


class HedgeAction(str, Enum):
    HOLD = "hold"
    OPEN_HEDGE = "open_hedge"
    WAIT_FOR_FUNDING = "wait_for_funding"
    CLOSE_HEDGE = "close_hedge"
    ADJUST_HEDGE = "adjust_hedge"
    RECOVERY_REQUIRED = "recovery_required"


class HedgeCycleStatus(str, Enum):
    ARMED = "armed"
    OPEN_SUBMITTED = "open_submitted"
    OPEN_PARTIAL = "open_partial"
    HEDGED = "hedged"
    AWAITING_FUNDING = "awaiting_funding"
    CLOSE_SUBMITTED = "close_submitted"
    CLOSE_PARTIAL = "close_partial"
    COMPLETED = "completed"
    ABORTED = "aborted"
    RECOVERY_REQUIRED = "recovery_required"

    @property
    def terminal(self) -> bool:
        return self in {
            HedgeCycleStatus.COMPLETED,
            HedgeCycleStatus.ABORTED,
            HedgeCycleStatus.RECOVERY_REQUIRED,
        }


@dataclass(frozen=True)
class AccountPositionSnapshot:
    account_address: str
    symbol: str
    size: Decimal
    margin_used_usd: Decimal = Decimal("0")

    def __post_init__(self) -> None:
        object.__setattr__(self, "account_address", self.account_address.strip())
        object.__setattr__(self, "symbol", _canonical_symbol(self.symbol))
        if not self.account_address:
            raise ValueError("account_address is required")
        if self.margin_used_usd < 0:
            raise ValueError("margin_used_usd must be non-negative")

    @property
    def side(self) -> str:
        if self.size > 0:
            return "long"
        if self.size < 0:
            return "short"
        return "flat"

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "AccountPositionSnapshot":
        return cls(
            account_address=str(payload.get("account_address", "") or ""),
            symbol=str(payload.get("symbol", "") or ""),
            size=parse_decimal(payload.get("size", "0"), "size"),
            margin_used_usd=parse_decimal(
                payload.get("margin_used_usd", "0"), "margin_used_usd"
            ),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "account_address": self.account_address,
            "symbol": self.symbol,
            "side": self.side,
            "size": decimal_text(self.size),
            "margin_used_usd": decimal_text(self.margin_used_usd),
        }


@dataclass(frozen=True)
class FundingObservation:
    symbol: str
    oracle_price: Decimal
    mark_price: Decimal
    funding_rate: Decimal
    observed_at: datetime
    next_funding_at: datetime
    size_decimals: int
    market_data_available: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _canonical_symbol(self.symbol))
        if self.oracle_price <= 0:
            raise ValueError("oracle_price must be positive")
        if self.mark_price <= 0:
            raise ValueError("mark_price must be positive")
        if self.size_decimals < 0:
            raise ValueError("size_decimals must be non-negative")
        for name in ("observed_at", "next_funding_at"):
            value = getattr(self, name)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must include a timezone")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "FundingObservation":
        oracle = parse_decimal(payload.get("oracle_price"), "oracle_price")
        return cls(
            symbol=str(payload.get("symbol", "") or ""),
            oracle_price=oracle,
            mark_price=parse_decimal(
                payload.get("mark_price", oracle), "mark_price"
            ),
            funding_rate=parse_decimal(payload.get("funding_rate"), "funding_rate"),
            observed_at=parse_utc_datetime(payload.get("observed_at"), "observed_at"),
            next_funding_at=parse_utc_datetime(
                payload.get("next_funding_at"), "next_funding_at"
            ),
            size_decimals=int(payload.get("size_decimals", 0)),
            market_data_available=bool(payload.get("market_data_available", True)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "oracle_price": decimal_text(self.oracle_price),
            "mark_price": decimal_text(self.mark_price),
            "funding_rate": decimal_text(self.funding_rate),
            "observed_at": utc_text(self.observed_at),
            "next_funding_at": utc_text(self.next_funding_at),
            "size_decimals": self.size_decimals,
            "market_data_available": self.market_data_available,
        }


@dataclass(frozen=True)
class HedgeAccountSnapshot:
    position: AccountPositionSnapshot
    withdrawable_usd: Decimal
    taker_fee_rate: Decimal
    fee_rate_source: str
    hip3_fee_formula_verified: bool = False
    account_kind: str = "subaccount"
    ownership_verified: bool = False
    margin_available_verified: bool = True
    funding_confirmed: bool = False
    funding_record_time: Optional[datetime] = None
    funding_delta_usd: Optional[Decimal] = None
    open_order_cloids: Tuple[str, ...] = ()
    unknown_open_orders: bool = False
    open_order_status: str = ""
    close_order_status: str = ""
    adjust_order_status: str = ""

    def __post_init__(self) -> None:
        if self.withdrawable_usd < 0:
            raise ValueError("withdrawable_usd must be non-negative")
        if self.taker_fee_rate < 0:
            raise ValueError("taker_fee_rate must be non-negative")
        if self.account_kind not in {"subaccount", "wallet"}:
            raise ValueError("account_kind must be subaccount or wallet")
        if self.funding_record_time is not None:
            value = self.funding_record_time
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError("funding_record_time must include a timezone")

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "HedgeAccountSnapshot":
        position_payload = payload.get("position")
        if not isinstance(position_payload, Mapping):
            raise ValueError("hedge_account.position is required")
        record_time = payload.get("funding_record_time")
        delta = payload.get("funding_delta_usd")
        return cls(
            position=AccountPositionSnapshot.from_mapping(position_payload),
            withdrawable_usd=parse_decimal(
                payload.get("withdrawable_usd", "0"), "withdrawable_usd"
            ),
            taker_fee_rate=parse_decimal(
                payload.get("taker_fee_rate", "0"), "taker_fee_rate"
            ),
            fee_rate_source=str(payload.get("fee_rate_source", "input") or "input"),
            hip3_fee_formula_verified=bool(
                payload.get("hip3_fee_formula_verified", False)
            ),
            account_kind=str(payload.get("account_kind", "subaccount") or "subaccount"),
            ownership_verified=bool(payload.get("ownership_verified", False)),
            margin_available_verified=bool(
                payload.get("margin_available_verified", True)
            ),
            funding_confirmed=bool(payload.get("funding_confirmed", False)),
            funding_record_time=(
                parse_utc_datetime(record_time, "funding_record_time")
                if record_time
                else None
            ),
            funding_delta_usd=(
                parse_decimal(delta, "funding_delta_usd") if delta is not None else None
            ),
            open_order_cloids=tuple(
                str(item) for item in payload.get("open_order_cloids", []) or []
            ),
            unknown_open_orders=bool(payload.get("unknown_open_orders", False)),
            open_order_status=str(payload.get("open_order_status", "") or ""),
            close_order_status=str(payload.get("close_order_status", "") or ""),
            adjust_order_status=str(payload.get("adjust_order_status", "") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "position": self.position.to_dict(),
            "withdrawable_usd": decimal_text(self.withdrawable_usd),
            "taker_fee_rate": decimal_text(self.taker_fee_rate),
            "fee_rate_source": self.fee_rate_source,
            "hip3_fee_formula_verified": self.hip3_fee_formula_verified,
            "account_kind": self.account_kind,
            "ownership_verified": self.ownership_verified,
            "margin_available_verified": self.margin_available_verified,
            "funding_confirmed": self.funding_confirmed,
            "funding_record_time": (
                utc_text(self.funding_record_time) if self.funding_record_time else None
            ),
            "funding_delta_usd": (
                decimal_text(self.funding_delta_usd)
                if self.funding_delta_usd is not None
                else None
            ),
            "open_order_cloids": list(self.open_order_cloids),
            "unknown_open_orders": self.unknown_open_orders,
            "open_order_status": self.open_order_status,
            "close_order_status": self.close_order_status,
            "adjust_order_status": self.adjust_order_status,
        }


@dataclass(frozen=True)
class HedgeSnapshot:
    primary_position: AccountPositionSnapshot
    hedge_account: HedgeAccountSnapshot
    funding: FundingObservation

    def __post_init__(self) -> None:
        symbols = {
            self.primary_position.symbol,
            self.hedge_account.position.symbol,
            self.funding.symbol,
        }
        if len(symbols) != 1:
            raise ValueError("primary, hedge, and funding symbols must match exactly")

    @property
    def symbol(self) -> str:
        return self.funding.symbol

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "HedgeSnapshot":
        primary = payload.get("primary_position")
        hedge = payload.get("hedge_account")
        funding = payload.get("funding")
        if not isinstance(primary, Mapping):
            raise ValueError("primary_position is required")
        if not isinstance(hedge, Mapping):
            raise ValueError("hedge_account is required")
        if not isinstance(funding, Mapping):
            raise ValueError("funding is required")
        return cls(
            primary_position=AccountPositionSnapshot.from_mapping(primary),
            hedge_account=HedgeAccountSnapshot.from_mapping(hedge),
            funding=FundingObservation.from_mapping(funding),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "primary_position": self.primary_position.to_dict(),
            "hedge_account": self.hedge_account.to_dict(),
            "funding": self.funding.to_dict(),
        }


@dataclass(frozen=True)
class HedgeCycleState:
    cycle_key: str
    status: HedgeCycleStatus
    symbol: str
    primary_account_address: str
    hedge_account_address: str
    settlement_at: datetime
    target_hedge_size: Decimal
    open_cloid: str
    close_cloid: str
    created_at: datetime
    updated_at: datetime
    network: str = "mainnet"
    hedge_account_kind: str = "subaccount"
    actual_hedge_size: Decimal = Decimal("0")
    close_attempt_count: int = 0
    adjustment_count: int = 0
    pending_adjust_cloid: str = ""
    pre_adjust_hedge_size: Decimal = Decimal("0")
    open_submitted_at: Optional[datetime] = None
    close_submitted_at: Optional[datetime] = None
    adjust_submitted_at: Optional[datetime] = None
    funding_confirmed_at: Optional[datetime] = None
    funding_delta_usd: Optional[Decimal] = None
    last_error: str = ""

    SCHEMA_VERSION = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "symbol", _canonical_symbol(self.symbol))
        for name in ("settlement_at", "created_at", "updated_at"):
            value = getattr(self, name)
            if value.tzinfo is None or value.utcoffset() is None:
                raise ValueError(f"{name} must include a timezone")
        if self.close_attempt_count < 0:
            raise ValueError("close_attempt_count must be non-negative")
        if self.adjustment_count < 0:
            raise ValueError("adjustment_count must be non-negative")
        if self.network not in {"mainnet", "testnet"}:
            raise ValueError("cycle network must be mainnet or testnet")
        if self.hedge_account_kind not in {"subaccount", "wallet"}:
            raise ValueError("invalid cycle hedge_account_kind")
        for name in (
            "open_submitted_at",
            "close_submitted_at",
            "adjust_submitted_at",
        ):
            value = getattr(self, name)
            if value is not None and (value.tzinfo is None or value.utcoffset() is None):
                raise ValueError(f"{name} must include a timezone")

    @property
    def active(self) -> bool:
        return not self.status.terminal

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> "HedgeCycleState":
        if int(payload.get("schema_version", 0)) != cls.SCHEMA_VERSION:
            raise ValueError("unsupported hedge-cycle schema version")
        confirmed_at = payload.get("funding_confirmed_at")
        delta = payload.get("funding_delta_usd")
        return cls(
            cycle_key=str(payload.get("cycle_key", "") or ""),
            status=HedgeCycleStatus(str(payload.get("status", ""))),
            symbol=str(payload.get("symbol", "") or ""),
            primary_account_address=str(
                payload.get("primary_account_address", "") or ""
            ),
            hedge_account_address=str(payload.get("hedge_account_address", "") or ""),
            settlement_at=parse_utc_datetime(
                payload.get("settlement_at"), "settlement_at"
            ),
            target_hedge_size=parse_decimal(
                payload.get("target_hedge_size", "0"), "target_hedge_size"
            ),
            actual_hedge_size=parse_decimal(
                payload.get("actual_hedge_size", "0"), "actual_hedge_size"
            ),
            close_attempt_count=int(payload.get("close_attempt_count", 0)),
            adjustment_count=int(payload.get("adjustment_count", 0)),
            pending_adjust_cloid=str(
                payload.get("pending_adjust_cloid", "") or ""
            ),
            pre_adjust_hedge_size=parse_decimal(
                payload.get("pre_adjust_hedge_size", "0"),
                "pre_adjust_hedge_size",
            ),
            open_cloid=str(payload.get("open_cloid", "") or ""),
            close_cloid=str(payload.get("close_cloid", "") or ""),
            created_at=parse_utc_datetime(payload.get("created_at"), "created_at"),
            updated_at=parse_utc_datetime(payload.get("updated_at"), "updated_at"),
            network=str(payload.get("network", "mainnet") or "mainnet"),
            hedge_account_kind=str(
                payload.get("hedge_account_kind", "subaccount") or "subaccount"
            ),
            open_submitted_at=(
                parse_utc_datetime(payload.get("open_submitted_at"), "open_submitted_at")
                if payload.get("open_submitted_at")
                else None
            ),
            close_submitted_at=(
                parse_utc_datetime(payload.get("close_submitted_at"), "close_submitted_at")
                if payload.get("close_submitted_at")
                else None
            ),
            adjust_submitted_at=(
                parse_utc_datetime(payload.get("adjust_submitted_at"), "adjust_submitted_at")
                if payload.get("adjust_submitted_at")
                else None
            ),
            funding_confirmed_at=(
                parse_utc_datetime(confirmed_at, "funding_confirmed_at")
                if confirmed_at
                else None
            ),
            funding_delta_usd=(
                parse_decimal(delta, "funding_delta_usd") if delta is not None else None
            ),
            last_error=str(payload.get("last_error", "") or ""),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "cycle_key": self.cycle_key,
            "status": self.status.value,
            "symbol": self.symbol,
            "primary_account_address": self.primary_account_address,
            "hedge_account_address": self.hedge_account_address,
            "network": self.network,
            "hedge_account_kind": self.hedge_account_kind,
            "settlement_at": utc_text(self.settlement_at),
            "target_hedge_size": decimal_text(self.target_hedge_size),
            "actual_hedge_size": decimal_text(self.actual_hedge_size),
            "close_attempt_count": self.close_attempt_count,
            "adjustment_count": self.adjustment_count,
            "pending_adjust_cloid": self.pending_adjust_cloid,
            "pre_adjust_hedge_size": decimal_text(self.pre_adjust_hedge_size),
            "open_submitted_at": (
                utc_text(self.open_submitted_at) if self.open_submitted_at else None
            ),
            "close_submitted_at": (
                utc_text(self.close_submitted_at) if self.close_submitted_at else None
            ),
            "adjust_submitted_at": (
                utc_text(self.adjust_submitted_at) if self.adjust_submitted_at else None
            ),
            "open_cloid": self.open_cloid,
            "close_cloid": self.close_cloid,
            "created_at": utc_text(self.created_at),
            "updated_at": utc_text(self.updated_at),
            "funding_confirmed_at": (
                utc_text(self.funding_confirmed_at)
                if self.funding_confirmed_at
                else None
            ),
            "funding_delta_usd": (
                decimal_text(self.funding_delta_usd)
                if self.funding_delta_usd is not None
                else None
            ),
            "last_error": self.last_error,
        }


@dataclass(frozen=True)
class HedgeDecision:
    snapshot: HedgeSnapshot
    action: HedgeAction
    reason: str
    evaluated_at: datetime
    seconds_to_funding: Decimal
    target_hedge_size: Decimal
    current_hedge_size: Decimal
    order_size_delta: Decimal
    reduce_only: bool
    estimated_primary_funding_debit_usd: Decimal
    estimated_hedge_funding_credit_usd: Decimal
    estimated_round_trip_cost_usd: Decimal
    estimated_net_saving_usd: Decimal
    estimated_required_margin_usd: Decimal
    cycle_key: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action.value,
            "reason": self.reason,
            "cycle_key": self.cycle_key,
            "evaluated_at": utc_text(self.evaluated_at),
            "seconds_to_funding": decimal_text(self.seconds_to_funding),
            "target_hedge_size": decimal_text(self.target_hedge_size),
            "current_hedge_size": decimal_text(self.current_hedge_size),
            "order_size_delta": decimal_text(self.order_size_delta),
            "reduce_only": self.reduce_only,
            "estimated_primary_funding_debit_usd": decimal_text(
                self.estimated_primary_funding_debit_usd
            ),
            "estimated_hedge_funding_credit_usd": decimal_text(
                self.estimated_hedge_funding_credit_usd
            ),
            "estimated_round_trip_cost_usd": decimal_text(
                self.estimated_round_trip_cost_usd
            ),
            "estimated_net_saving_usd": decimal_text(self.estimated_net_saving_usd),
            "estimated_required_margin_usd": decimal_text(
                self.estimated_required_margin_usd
            ),
            "snapshot": self.snapshot.to_dict(),
        }
