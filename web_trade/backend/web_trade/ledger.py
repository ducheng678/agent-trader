from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    if result != result:
        return default
    return result


def _symbol_key(symbol: str) -> str:
    return str(symbol or "").strip().upper()


def snapshot_has_open_position(snapshot: Dict[str, Any]) -> bool:
    side = str((snapshot or {}).get("side", "") or "").strip().lower()
    size = abs(_as_float((snapshot or {}).get("size"), 0.0))
    notional = abs(_as_float((snapshot or {}).get("notional_usd"), 0.0))
    return side in {"long", "short"} and (size > 0.0 or notional > 0.0)


def current_margin_basis_usd(snapshot: Dict[str, Any]) -> float:
    margin_used = max(0.0, _as_float((snapshot or {}).get("margin_used"), 0.0))
    if margin_used > 0.0:
        return margin_used
    notional = abs(_as_float((snapshot or {}).get("notional_usd"), 0.0))
    leverage = max(0.0, _as_float((snapshot or {}).get("leverage"), 0.0))
    if notional > 0.0 and leverage > 0.0:
        return notional / leverage
    return 0.0


class SyntheticPositionLedger:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def _load(self) -> Dict[str, Any]:
        if not self.path.exists():
            return {"positions": {}}
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"positions": {}}
        if not isinstance(payload, dict):
            return {"positions": {}}
        positions = payload.get("positions")
        if not isinstance(positions, dict):
            payload["positions"] = {}
        return payload

    def _save(self, payload: Dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
        tmp_path.replace(self.path)

    def _get_state(self, payload: Dict[str, Any], symbol: str) -> Dict[str, Any] | None:
        state = payload.setdefault("positions", {}).get(_symbol_key(symbol))
        return state if isinstance(state, dict) else None

    def _initialize_state(self, payload: Dict[str, Any], snapshot: Dict[str, Any]) -> Dict[str, Any]:
        symbol = _symbol_key(str(snapshot.get("symbol", "")))
        now = time.time()
        state = {
            "symbol": symbol,
            "side": str(snapshot.get("side", "") or "").lower(),
            "display_entry_price": max(0.0, _as_float(snapshot.get("entry_price"), 0.0)),
            "lifecycle_roi_basis_usd": max(0.0, current_margin_basis_usd(snapshot)),
            "carried_realized_pnl_usd": 0.0,
            "created_at": now,
            "updated_at": now,
            "last_hidden_rebalance": {},
        }
        payload.setdefault("positions", {})[symbol] = state
        return state

    def state_for_snapshot(self, snapshot: Dict[str, Any]) -> Dict[str, Any] | None:
        if not snapshot_has_open_position(snapshot):
            return None
        payload = self._load()
        symbol = _symbol_key(str(snapshot.get("symbol", "")))
        state = self._get_state(payload, symbol)
        side = str(snapshot.get("side", "") or "").lower()
        if state is None or str(state.get("side", "") or "").lower() != side:
            state = self._initialize_state(payload, snapshot)
            self._save(payload)
        return dict(state)

    def overlay_position(self, snapshot: Dict[str, Any]) -> Dict[str, Any]:
        view = dict(snapshot or {})
        state = self.state_for_snapshot(view)
        if not state:
            return view
        basis = max(0.0, _as_float(state.get("lifecycle_roi_basis_usd"), 0.0))
        carried = _as_float(state.get("carried_realized_pnl_usd"), 0.0)
        unrealized = _as_float(view.get("unrealized_pnl"), 0.0)
        synthetic_pnl = carried + unrealized
        view.update(
            {
                "display_entry_price": _as_float(state.get("display_entry_price"), 0.0),
                "lifecycle_roi_basis_usd": basis,
                "carried_realized_pnl_usd": carried,
                "synthetic_pnl_usd": synthetic_pnl,
                "synthetic_pnl_pct": synthetic_pnl / basis if basis > 0.0 else 0.0,
                "ledger": state,
            }
        )
        return view

    def apply_margin_delta(self, symbol: str, amount_usd: float) -> Dict[str, Any]:
        payload = self._load()
        state = self._get_state(payload, symbol)
        if state is None:
            raise KeyError(f"No synthetic ledger state for {symbol}")
        old_basis = max(0.0, _as_float(state.get("lifecycle_roi_basis_usd"), 0.0))
        new_basis = max(0.0, old_basis + _as_float(amount_usd, 0.0))
        state["lifecycle_roi_basis_usd"] = new_basis
        state["updated_at"] = time.time()
        state.setdefault("margin_events", []).append(
            {
                "amount_usd": _as_float(amount_usd, 0.0),
                "basis_before_usd": old_basis,
                "basis_after_usd": new_basis,
                "ts": state["updated_at"],
            }
        )
        self._save(payload)
        return dict(state)

    def record_hidden_rebalance(
        self,
        *,
        symbol: str,
        realized_pnl_usd: float,
        target_leverage: int,
        target_notional_usd: float,
    ) -> Dict[str, Any]:
        payload = self._load()
        state = self._get_state(payload, symbol)
        if state is None:
            raise KeyError(f"No synthetic ledger state for {symbol}")
        carried_before = _as_float(state.get("carried_realized_pnl_usd"), 0.0)
        carried_after = carried_before + _as_float(realized_pnl_usd, 0.0)
        now = time.time()
        event = {
            "realized_pnl_usd": _as_float(realized_pnl_usd, 0.0),
            "carried_before_usd": carried_before,
            "carried_after_usd": carried_after,
            "target_leverage": int(target_leverage or 0),
            "target_notional_usd": max(0.0, _as_float(target_notional_usd, 0.0)),
            "ts": now,
        }
        state["carried_realized_pnl_usd"] = carried_after
        state["last_hidden_rebalance"] = event
        state.setdefault("hidden_rebalance_events", []).append(event)
        state["updated_at"] = now
        self._save(payload)
        return dict(state)
