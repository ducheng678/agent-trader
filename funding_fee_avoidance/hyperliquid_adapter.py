from __future__ import annotations

from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from typing import Any, Dict, List, Mapping, Optional, Tuple

from market_agent.exchange import HyperliquidRestReader
from market_agent.symbols import (
    canonicalize_execution_symbol,
    split_execution_symbol,
)

from .config import StrategyConfig
from .models import (
    AccountPositionSnapshot,
    FundingObservation,
    HedgeAccountSnapshot,
    HedgeCycleState,
    HedgeSnapshot,
)


UTC = timezone.utc


def _decimal(value: Any) -> Optional[Decimal]:
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError):
        return None
    return result if result.is_finite() else None


def _payload_with_dex(payload_type: str, dex: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"type": payload_type, **extra}
    if dex:
        payload["dex"] = dex
    return payload


def _canonical_market_name(raw: Any, dex: str) -> str:
    value = canonicalize_execution_symbol(str(raw or ""))
    if not value:
        return ""
    raw_dex, asset = split_execution_symbol(value)
    return canonicalize_execution_symbol(f"{dex}:{asset}") if dex and not raw_dex else value


def _next_hour(now: datetime) -> datetime:
    floor = now.astimezone(UTC).replace(minute=0, second=0, microsecond=0)
    return floor + timedelta(hours=1)


def _position_from_state(
    state: Any, address: str, symbol: str
) -> Tuple[AccountPositionSnapshot, Decimal]:
    dex, _ = split_execution_symbol(symbol)
    size = Decimal("0")
    margin_used = Decimal("0")
    if isinstance(state, Mapping):
        entries = state.get("assetPositions", [])
        for wrapper in entries if isinstance(entries, list) else []:
            position = wrapper.get("position", {}) if isinstance(wrapper, Mapping) else {}
            if not isinstance(position, Mapping):
                continue
            coin = _canonical_market_name(position.get("coin"), dex)
            if coin != symbol:
                continue
            parsed_size = _decimal(position.get("szi"))
            parsed_margin = _decimal(position.get("marginUsed"))
            size = parsed_size if parsed_size is not None else Decimal("0")
            margin_used = (
                max(Decimal("0"), parsed_margin)
                if parsed_margin is not None
                else Decimal("0")
            )
            break
        withdrawable = _decimal(state.get("withdrawable"))
    else:
        withdrawable = None
    return (
        AccountPositionSnapshot(
            account_address=address,
            symbol=symbol,
            size=size,
            margin_used_usd=margin_used,
        ),
        max(Decimal("0"), withdrawable or Decimal("0")),
    )


def _market_context(raw: Any, symbol: str) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
    if not isinstance(raw, list) or len(raw) < 2:
        return {}, {}
    meta = raw[0] if isinstance(raw[0], Mapping) else {}
    contexts = raw[1] if isinstance(raw[1], list) else []
    dex, _ = split_execution_symbol(symbol)
    universe = meta.get("universe", []) if isinstance(meta, Mapping) else []
    for index, asset in enumerate(universe if isinstance(universe, list) else []):
        if not isinstance(asset, Mapping) or index >= len(contexts):
            continue
        if _canonical_market_name(asset.get("name"), dex) != symbol:
            continue
        context = contexts[index] if isinstance(contexts[index], Mapping) else {}
        return asset, context
    return {}, {}


def _subaccount_addresses(raw: Any) -> Tuple[str, ...]:
    addresses: List[str] = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, Mapping):
            continue
        address = str(
            item.get("subAccountUser")
            or item.get("subAccountAddress")
            or item.get("user")
            or ""
        ).strip()
        if address:
            addresses.append(address.lower())
    return tuple(addresses)


def _order_status(raw: Any) -> str:
    if not isinstance(raw, Mapping):
        return "unknown"
    status = str(raw.get("status", "") or "").strip().lower()
    order = raw.get("order")
    if isinstance(order, Mapping):
        nested = str(order.get("status", "") or "").strip().lower()
        return nested or status or "unknown"
    return status or "unknown"


def _funding_record(
    raw: Any, symbol: str, settlement_at: datetime
) -> Tuple[bool, Optional[datetime], Optional[Decimal]]:
    dex, _ = split_execution_symbol(symbol)
    target_hour = settlement_at.astimezone(UTC).replace(
        minute=0, second=0, microsecond=0
    )
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, Mapping):
            continue
        delta = item.get("delta") if isinstance(item.get("delta"), Mapping) else item
        coin = _canonical_market_name(delta.get("coin"), dex)
        time_ms = _decimal(item.get("time"))
        if coin != symbol or time_ms is None:
            continue
        record_time = datetime.fromtimestamp(
            float(time_ms / Decimal("1000")), tz=UTC
        )
        record_hour = record_time.replace(minute=0, second=0, microsecond=0)
        if record_hour != target_hour:
            continue
        amount = _decimal(delta.get("usdc"))
        return True, record_time, amount
    return False, None, None


def _perp_dex_info(raw: Any, dex: str) -> Mapping[str, Any]:
    for item in raw if isinstance(raw, list) else []:
        if isinstance(item, Mapping) and str(item.get("name", "") or "").lower() == dex:
            return item
    return {}


def hip3_fee_rates(
    user_fees: Mapping[str, Any],
    dex_info: Mapping[str, Any],
    asset_meta: Mapping[str, Any],
    *,
    is_aligned_quote_token: bool = False,
) -> Tuple[Decimal, Decimal]:
    """Official HIP-3 maker/taker formula, returned as decimal rates."""

    zero = Decimal("0")
    one = Decimal("1")
    taker_base = _decimal(user_fees.get("userCrossRate"))
    maker_base = _decimal(user_fees.get("userAddRate"))
    referral = _decimal(user_fees.get("activeReferralDiscount"))
    deployer_scale = _decimal(dex_info.get("deployerFeeScale"))
    if None in {taker_base, maker_base, referral, deployer_scale}:
        raise ValueError("incomplete userFees/perpDexs fee inputs")
    assert taker_base is not None
    assert maker_base is not None
    assert referral is not None
    assert deployer_scale is not None
    if taker_base < zero or not (zero <= referral <= one):
        raise ValueError("invalid user fee inputs")
    if not (zero <= deployer_scale <= Decimal("3")):
        raise ValueError("invalid deployerFeeScale")
    growth_value = asset_meta.get("growthMode")
    if growth_value == "enabled":
        growth = True
    elif growth_value in {None, "disabled"}:
        growth = False
    else:
        raise ValueError(f"unknown growthMode {growth_value!r}")
    if growth and deployer_scale > one:
        raise ValueError("growth-mode deployerFeeScale cannot exceed 1")

    hip3_scale = (
        one + deployer_scale
        if deployer_scale < one
        else Decimal("2") * deployer_scale
    )
    deployer_share = (
        deployer_scale / (one + deployer_scale)
        if deployer_scale < one
        else Decimal("0.5")
    )
    growth_scale = Decimal("0.1") if growth else one
    taker = taker_base * hip3_scale * growth_scale * (one - referral)
    if is_aligned_quote_token:
        taker *= (one - deployer_share) * Decimal("0.8") + deployer_share

    maker = maker_base * growth_scale
    if maker > zero:
        maker *= hip3_scale * (one - referral)
    elif is_aligned_quote_token:
        maker *= (one - deployer_share) * Decimal("1.5") + deployer_share
    return maker, taker


class HyperliquidSnapshotAdapter:
    """Fresh, explicit-address reads for both sides of the hedge.

    This class deliberately exposes no exchange/order method.  It does not use
    the repository reader's cached ``get_market_asset_context`` because stale
    funding data must never authorize an opening order.
    """

    def __init__(
        self,
        config: StrategyConfig,
        reader: Optional[HyperliquidRestReader] = None,
    ) -> None:
        self.config = config
        self.reader = reader or HyperliquidRestReader()

    def _account_addresses(self) -> Tuple[str, str]:
        primary = self.config.primary_account_address or str(
            getattr(self.reader, "account_address", "") or ""
        )
        hedge = self.config.hedge_account_address
        if not primary:
            raise RuntimeError("HL_ACCOUNT_ADDRESS is required")
        if not hedge:
            raise RuntimeError("FUNDING_HEDGE_ACCOUNT_ADDRESS is required")
        if primary.lower() == hedge.lower():
            raise RuntimeError("primary and hedge account addresses must differ")
        return primary, hedge

    def _fresh_state(self, address: str, dex: str) -> Any:
        return self.reader.post_info(
            _payload_with_dex("clearinghouseState", dex, user=address)
        )

    def _fresh_context(self, symbol: str) -> Tuple[Mapping[str, Any], Mapping[str, Any]]:
        dex, _ = split_execution_symbol(symbol)
        return _market_context(
            self.reader.post_info(_payload_with_dex("metaAndAssetCtxs", dex)),
            symbol,
        )

    def _ownership_verified(self, primary: str, hedge: str, errors: List[str]) -> bool:
        if self.config.hedge_account_kind == "wallet":
            # The executor separately proves this by matching the signer address.
            return True
        try:
            raw = self.reader.post_info({"type": "subAccounts", "user": primary})
            return hedge.lower() in _subaccount_addresses(raw)
        except Exception as exc:
            errors.append(f"subaccount ownership lookup failed: {exc}")
            return False

    def _user_fees(self, hedge: str, errors: List[str]) -> Mapping[str, Any]:
        try:
            raw = self.reader.post_info({"type": "userFees", "user": hedge})
            if isinstance(raw, Mapping):
                return raw
        except Exception as exc:
            errors.append(f"hedge account fee lookup failed: {exc}")
        errors.append("hedge account fee inputs are unavailable")
        return {}

    def _orders(
        self,
        hedge: str,
        symbol: str,
        dex: str,
        cycle: Optional[HedgeCycleState],
        errors: List[str],
    ) -> Tuple[Tuple[str, ...], bool, str, str, str]:
        expected = {
            value.lower()
            for value in (
                cycle.open_cloid if cycle else "",
                cycle.close_cloid if cycle else "",
                cycle.pending_adjust_cloid if cycle else "",
            )
            if value
        }
        cloids: List[str] = []
        unknown = False
        try:
            raw = self.reader.post_info(
                _payload_with_dex("frontendOpenOrders", dex, user=hedge)
            )
            for order in raw if isinstance(raw, list) else []:
                if not isinstance(order, Mapping):
                    continue
                if _canonical_market_name(order.get("coin"), dex) != symbol:
                    continue
                cloid = str(order.get("cloid", "") or "").strip()
                if cloid:
                    cloids.append(cloid)
                if not cloid or cloid.lower() not in expected:
                    unknown = True
        except Exception as exc:
            errors.append(f"{symbol}: open-order lookup failed: {exc}")
            unknown = True

        statuses: Dict[str, str] = {"open": "", "close": "", "adjust": ""}
        if cycle:
            for phase, cloid in (
                ("open", cycle.open_cloid),
                ("close", cycle.close_cloid),
                ("adjust", cycle.pending_adjust_cloid),
            ):
                if not cloid:
                    continue
                try:
                    statuses[phase] = _order_status(
                        self.reader.post_info(
                            {"type": "orderStatus", "user": hedge, "oid": cloid}
                        )
                    )
                except Exception as exc:
                    errors.append(f"{symbol}: {phase} order-status lookup failed: {exc}")
                    statuses[phase] = "unknown"
        return (
            tuple(cloids),
            unknown,
            statuses["open"],
            statuses["close"],
            statuses["adjust"],
        )

    def _funding_confirmation(
        self,
        hedge: str,
        symbol: str,
        cycle: Optional[HedgeCycleState],
        now: datetime,
        errors: List[str],
    ) -> Tuple[bool, Optional[datetime], Optional[Decimal]]:
        if cycle is None or not cycle.active or now < cycle.settlement_at:
            return False, None, None
        start_ms = int(
            (cycle.settlement_at - timedelta(minutes=1)).timestamp() * 1000
        )
        end_ms = int(now.timestamp() * 1000)
        try:
            raw = self.reader.post_info(
                {
                    "type": "userFunding",
                    "user": hedge,
                    "startTime": start_ms,
                    "endTime": end_ms,
                }
            )
            return _funding_record(raw, symbol, cycle.settlement_at)
        except Exception as exc:
            errors.append(f"{symbol}: userFunding lookup failed: {exc}")
            return False, None, None

    def load_snapshots(
        self,
        observed_at: Optional[datetime] = None,
        cycles: Optional[Mapping[str, HedgeCycleState]] = None,
    ) -> Tuple[List[HedgeSnapshot], List[str]]:
        now = (observed_at or datetime.now(tz=UTC)).astimezone(UTC)
        errors: List[str] = []
        primary, hedge = self._account_addresses()
        ownership_verified = self._ownership_verified(primary, hedge, errors)
        user_fees = self._user_fees(hedge, errors)
        try:
            perp_dexs = self.reader.post_info({"type": "perpDexs"})
        except Exception as exc:
            perp_dexs = []
            errors.append(f"perpDexs fee-scale lookup failed: {exc}")
        try:
            abstraction = str(
                self.reader.post_info({"type": "userAbstraction", "user": hedge})
                or ""
            ).strip()
        except Exception as exc:
            abstraction = "unknown"
            errors.append(f"hedge account abstraction lookup failed: {exc}")
        # In Standard mode, the dex-scoped clearinghouseState.withdrawable is
        # meaningful.  Unified/portfolio balances require activeAssetData;
        # this one-shot REST adapter therefore fails closed for new orders.
        margin_available_verified = abstraction == "disabled"
        if not margin_available_verified:
            errors.append(
                "hedge available margin is not verified for this account abstraction; "
                "new hedges are blocked"
            )
        snapshots: List[HedgeSnapshot] = []

        symbols = list(self.config.symbols)
        for cycle_symbol, stored_cycle in (cycles or {}).items():
            if stored_cycle.active and cycle_symbol not in symbols:
                symbols.append(cycle_symbol)

        for raw_symbol in symbols:
            symbol = canonicalize_execution_symbol(raw_symbol)
            dex, _ = split_execution_symbol(symbol)
            cycle = (cycles or {}).get(symbol)
            try:
                hedge_state = self._fresh_state(hedge, dex)
            except Exception as exc:
                errors.append(
                    f"{symbol}: hedge account state unavailable; no order can be sent: {exc}"
                )
                continue
            try:
                primary_state = self._fresh_state(primary, dex)
                primary_position, _ = _position_from_state(
                    primary_state, primary, symbol
                )
            except Exception as exc:
                if cycle is None or not cycle.active:
                    errors.append(f"{symbol}: primary account state unavailable: {exc}")
                    continue
                errors.append(
                    f"{symbol}: primary account state unavailable; tracked hedge will be shut down: {exc}"
                )
                primary_position = AccountPositionSnapshot(
                    account_address=primary,
                    symbol=symbol,
                    size=Decimal("0"),
                )

            hedge_position, withdrawable = _position_from_state(
                hedge_state, hedge, symbol
            )
            market_data_available = True
            asset: Mapping[str, Any] = {}
            try:
                asset, context = self._fresh_context(symbol)
                oracle = _decimal(context.get("oraclePx"))
                mark = _decimal(context.get("markPx"))
                rate = _decimal(context.get("funding"))
                if oracle is None or oracle <= 0 or rate is None:
                    raise ValueError("incomplete fresh metaAndAssetCtxs data")
                mark = mark if mark is not None and mark > 0 else oracle
                try:
                    size_decimals = int(asset.get("szDecimals", 0))
                except (TypeError, ValueError):
                    raise ValueError("invalid size precision")
            except Exception as exc:
                if cycle is None or not cycle.active:
                    errors.append(f"{symbol}: market data unavailable; skipped: {exc}")
                    continue
                market_data_available = False
                errors.append(
                    f"{symbol}: market data unavailable; only tracked-hedge shutdown is allowed: {exc}"
                )
                oracle = Decimal("1")
                mark = Decimal("1")
                rate = Decimal("0")
                size_decimals = max(
                    0, -int(cycle.target_hedge_size.as_tuple().exponent)
                )

            taker_fee = self.config.fallback_taker_fee_rate
            fee_source = "configured_fallback"
            hip3_fee_verified = False
            try:
                if dex:
                    _, taker_fee = hip3_fee_rates(
                        user_fees,
                        _perp_dex_info(perp_dexs, dex),
                        asset,
                    )
                    fee_source = "official_hip3_fee_formula"
                    hip3_fee_verified = True
                else:
                    base_rate = _decimal(user_fees.get("userCrossRate"))
                    referral = _decimal(user_fees.get("activeReferralDiscount"))
                    if base_rate is None or referral is None or not (
                        Decimal("0") <= referral <= Decimal("1")
                    ):
                        raise ValueError("invalid main-dex fee inputs")
                    taker_fee = base_rate * (Decimal("1") - referral)
                    fee_source = "hyperliquid_userFees"
            except Exception as exc:
                errors.append(
                    f"{symbol}: dynamic fee calculation unavailable; configured "
                    f"fallback used and new HIP-3 opens remain blocked: {exc}"
                )

            try:
                (
                    cloids,
                    unknown_orders,
                    open_status,
                    close_status,
                    adjust_status,
                ) = self._orders(hedge, symbol, dex, cycle, errors)
                confirmed, record_time, funding_delta = self._funding_confirmation(
                    hedge, symbol, cycle, now, errors
                )
                snapshots.append(
                    HedgeSnapshot(
                        primary_position=primary_position,
                        hedge_account=HedgeAccountSnapshot(
                            position=hedge_position,
                            withdrawable_usd=withdrawable,
                            taker_fee_rate=taker_fee,
                            fee_rate_source=fee_source,
                            hip3_fee_formula_verified=hip3_fee_verified,
                            account_kind=self.config.hedge_account_kind,
                            ownership_verified=ownership_verified,
                            margin_available_verified=margin_available_verified,
                            funding_confirmed=confirmed,
                            funding_record_time=record_time,
                            funding_delta_usd=funding_delta,
                            open_order_cloids=cloids,
                            unknown_open_orders=unknown_orders,
                            open_order_status=open_status,
                            close_order_status=close_status,
                            adjust_order_status=adjust_status,
                        ),
                        funding=FundingObservation(
                            symbol=symbol,
                            oracle_price=oracle,
                            mark_price=mark,
                            funding_rate=rate,
                            observed_at=now,
                            next_funding_at=_next_hour(now),
                            size_decimals=size_decimals,
                            market_data_available=market_data_available,
                        ),
                    )
                )
            except Exception as exc:
                errors.append(f"{symbol}: snapshot read failed: {exc}")
        return snapshots, errors

    # Kept as a descriptive compatibility shim for callers of the first
    # report-only prototype.  The returned objects now contain both accounts.
    def load_positions(
        self, observed_at: Optional[datetime] = None
    ) -> Tuple[List[HedgeSnapshot], List[str]]:
        return self.load_snapshots(observed_at=observed_at)
