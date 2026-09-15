from typing import Any, Dict

from market_agent.utils import safe_float


def normalize_spot_user_state(state: Any) -> dict:
    state = state if isinstance(state, dict) else {}
    balances = state.get("balances", []) or []
    usdc_balance = None
    for entry in balances:
        if not isinstance(entry, dict):
            continue
        coin = str(entry.get("coin", "") or "").upper()
        token = entry.get("token")
        if coin == "USDC" or token == 0:
            usdc_balance = entry
            break
    token_to_available = {}
    for item in state.get("tokenToAvailableAfterMaintenance", []) or []:
        if isinstance(item, (list, tuple)) and len(item) == 2:
            token_to_available[int(item[0])] = safe_float(item[1], 0.0) or 0.0
    usdc_total = safe_float((usdc_balance or {}).get("total"), 0.0) or 0.0
    usdc_hold = safe_float((usdc_balance or {}).get("hold"), 0.0) or 0.0
    usdc_entry_ntl = safe_float((usdc_balance or {}).get("entryNtl"), 0.0) or 0.0
    usdc_available_after_maintenance = token_to_available.get(0, max(0.0, usdc_total - usdc_hold))
    return {
        "balances": balances,
        "token_to_available_after_maintenance": token_to_available,
        "spot_usdc_total": usdc_total,
        "spot_usdc_hold": usdc_hold,
        "spot_usdc_entry_ntl": usdc_entry_ntl,
        "spot_available_usdc": max(0.0, usdc_available_after_maintenance),
    }


def snapshot_has_open_position(snapshot: Dict[str, Any]) -> bool:
    return abs(float(snapshot.get("size", 0.0) or 0.0)) > 0.0
