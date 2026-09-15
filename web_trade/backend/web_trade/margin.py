from __future__ import annotations

from typing import Any, Dict

from web_trade.backend.web_trade.ledger import snapshot_has_open_position


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:
        return default
    return result


def calculate_margin_limits(snapshot: Dict[str, Any], safety_buffer_usd: float | None = None) -> Dict[str, Any]:
    position = dict(snapshot or {})
    if not snapshot_has_open_position(position):
        return {
            "enabled": False,
            "reason": "no_open_position",
            "max_add_margin_usd": 0.0,
            "max_remove_margin_usd": 0.0,
        }
    if not bool(position.get("only_isolated", False)):
        return {
            "enabled": False,
            "reason": "not_isolated",
            "max_add_margin_usd": 0.0,
            "max_remove_margin_usd": 0.0,
        }

    notional = abs(_as_float(position.get("notional_usd"), 0.0))
    max_leverage = max(0.0, _as_float(position.get("max_leverage"), 0.0))
    margin_used = max(0.0, _as_float(position.get("margin_used"), 0.0))
    unrealized = _as_float(position.get("unrealized_pnl"), 0.0)
    available = max(
        0.0,
        _as_float(position.get("available_margin_usd"), 0.0),
        _as_float(position.get("remaining_capital_usd"), 0.0),
        _as_float(position.get("withdrawable_usd"), 0.0),
    )
    buffer_usd = (
        max(0.0, _as_float(safety_buffer_usd, 0.0))
        if safety_buffer_usd is not None
        else max(1.0, notional * 0.001)
    )
    initial_margin_required = notional / max(max_leverage, 1e-12) if max_leverage > 0.0 else notional
    transfer_margin_floor = 0.10 * notional
    required_remaining_margin = max(initial_margin_required, transfer_margin_floor)
    isolated_position_equity = margin_used + unrealized
    return {
        "enabled": True,
        "reason": "",
        "safety_buffer_usd": buffer_usd,
        "current_position_value_usd": notional,
        "isolated_position_equity_usd": isolated_position_equity,
        "initial_margin_required_usd": initial_margin_required,
        "transfer_margin_floor_usd": transfer_margin_floor,
        "required_remaining_margin_usd": required_remaining_margin,
        "max_add_margin_usd": max(0.0, available - buffer_usd),
        "max_remove_margin_usd": max(0.0, isolated_position_equity - required_remaining_margin - buffer_usd),
    }
