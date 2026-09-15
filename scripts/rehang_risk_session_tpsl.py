#!/usr/bin/env python3
"""Rebuild and rehang the current staged risk-session TP/SL orders.

Default mode is dry-run. Use --execute only after stopping the live agent,
because the live process keeps an in-memory RiskSession that will otherwise
disagree with the rewritten state file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from unified_market_agent import UnifiedMarketAgent  # noqa: E402
from market_agent.positions import snapshot_has_open_position  # noqa: E402
from market_agent.symbols import canonicalize_execution_symbol  # noqa: E402
from market_agent.utils import safe_float  # noqa: E402


def _json_dump(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _load_state(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"risk state file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"risk state payload is not an object: {path}")
    return payload


def _running_agent_processes() -> List[str]:
    try:
        result = subprocess.run(
            ["pgrep", "-af", r"python .*unified_market_agent\.py"],
            check=False,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return []
    current_pid = os.getpid()
    rows = []
    for line in result.stdout.splitlines():
        raw = line.strip()
        if not raw:
            continue
        try:
            pid = int(raw.split(None, 1)[0])
        except Exception:
            pid = 0
        if pid and pid != current_pid:
            rows.append(raw)
    return rows


def _state_stage(payload: Dict[str, Any]) -> str:
    stage = str(payload.get("stage", "") or "").strip()
    if stage:
        return stage
    if bool(payload.get("tp2_hit", False)):
        return "tail"
    if bool(payload.get("tp1_hit", False)):
        return "post_tp1"
    return "initial"


def _price_close(a: float, b: float) -> bool:
    if a <= 0.0 or b <= 0.0:
        return False
    return abs(a - b) <= max(1e-8, abs(a) * 1e-5)


def _order_refs_match_current_exchange(agent: UnifiedMarketAgent, state_refs: List[Dict[str, Any]], current_refs: List[Dict[str, Any]], symbol: str) -> bool:
    qty_tol = max(agent._risk_session_order_qty_tolerance(symbol), max([abs(float(ref.get("close_size", 0.0) or 0.0)) for ref in state_refs] or [0.0]) * 0.01)
    unmatched = [dict(ref) for ref in current_refs]
    for state_ref in state_refs:
        found_index = None
        for idx, current_ref in enumerate(unmatched):
            if not agent._order_ref_identity_matches(state_ref, current_ref):
                continue
            if str(state_ref.get("tpsl", "") or "") != str(current_ref.get("tpsl", "") or ""):
                continue
            if not _price_close(float(state_ref.get("trigger_price", 0.0) or 0.0), float(current_ref.get("trigger_price", 0.0) or 0.0)):
                continue
            if abs(float(state_ref.get("close_size", 0.0) or 0.0) - float(current_ref.get("close_size", 0.0) or 0.0)) > qty_tol:
                continue
            found_index = idx
            break
        if found_index is None:
            return False
        unmatched.pop(found_index)
    return True


def _summarize_specs(specs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            "key": str(item.get("key", "") or ""),
            "name": str(item.get("name", "") or ""),
            "leg_type": str(item.get("leg_type", "") or ""),
            "tpsl": str(item.get("tpsl", "") or ""),
            "trigger_price": float(item.get("trigger_price", 0.0) or 0.0),
            "close_size": float(item.get("close_size", 0.0) or 0.0),
        }
        for item in specs
    ]


def _ms_to_utc_datetime(value_ms: int) -> datetime:
    if value_ms > 0:
        return datetime.fromtimestamp(value_ms / 1000.0, tz=timezone.utc)
    return datetime.now(timezone.utc)


def _infer_clip_time_ms(state_payload: Dict[str, Any], order_refs: List[Dict[str, Any]]) -> int:
    order_times = [
        int(safe_float(ref.get("created_time_ms"), 0.0) or 0)
        for ref in list(order_refs or [])
        if int(safe_float(ref.get("created_time_ms"), 0.0) or 0) > 0
    ]
    if order_times:
        return min(order_times)
    start_time = safe_float(state_payload.get("start_time"), 0.0) or 0.0
    if start_time > 0.0:
        return int(start_time * 1000)
    updated_at_ms = int(safe_float(state_payload.get("updated_at_ms"), 0.0) or 0)
    return updated_at_ms if updated_at_ms > 0 else 0


def _profile_clipped_stop_for_band(
    agent: UnifiedMarketAgent,
    *,
    symbol: str,
    side: str,
    entry_price: float,
    stop_price: float,
    target_band: str,
    clip_time_ms: int,
) -> Dict[str, Any]:
    original_stop = max(0.0, float(stop_price or 0.0))
    entry = max(0.0, float(entry_price or 0.0))
    if side not in {"long", "short"} or entry <= 0.0 or original_stop <= 0.0:
        return {"applied": False, "code": "invalid_input", "stop_loss_price": original_stop}
    if side == "long" and original_stop >= entry:
        return {"applied": False, "code": "invalid_long_stop", "stop_loss_price": original_stop}
    if side == "short" and original_stop <= entry:
        return {"applied": False, "code": "invalid_short_stop", "stop_loss_price": original_stop}
    profile = agent._market_profile_for_symbol(symbol)
    if profile is None:
        return {"applied": False, "code": "no_profile", "stop_loss_price": original_stop}
    if target_band not in {"normal_liquidity", "low_liquidity"}:
        return {"applied": False, "code": f"invalid_target_band:{target_band}", "stop_loss_price": original_stop}
    clip_time = _ms_to_utc_datetime(clip_time_ms)
    atr_ref = agent._profile_normal_liquidity_atr_ref(symbol=symbol, profile=profile, now_utc=clip_time)
    if not atr_ref.get("available"):
        return {"applied": False, "code": str(atr_ref.get("code") or "atr_ref_unavailable"), "stop_loss_price": original_stop, "atr_ref": atr_ref}
    atr_value = max(0.0, float(atr_ref.get("atr_ref", 0.0) or 0.0))
    min_multiple, max_multiple = agent._profile_r_clip_multiples(profile, target_band)
    r_min = max(0.0, min_multiple * atr_value)
    r_max = max(0.0, max_multiple * atr_value)
    if r_max > 0.0:
        r_max = max(r_min, r_max)
    r_raw = abs(entry - original_stop)
    r_clipped = max(r_raw, r_min)
    if r_max > 0.0:
        r_clipped = min(r_clipped, r_max)
    corrected = entry - r_clipped if side == "long" else entry + r_clipped
    corrected = agent._align_price_for_symbol(symbol, corrected)
    return {
        "applied": abs(corrected - original_stop) > max(1.0, entry) * 1e-12,
        "code": f"profile_{target_band}_r_clip",
        "profile": profile.name,
        "symbol": canonicalize_execution_symbol(symbol or ""),
        "liquidity_band": target_band,
        "side": side,
        "entry_price": entry,
        "original_stop_loss_price": original_stop,
        "stop_loss_price": corrected,
        "r_raw": r_raw,
        "r_min": r_min,
        "r_max": r_max,
        "r_clipped": abs(entry - corrected),
        "r_min_atr_multiple": min_multiple,
        "r_max_atr_multiple": max_multiple,
        "atr_ref": atr_ref,
        "clip_time_ms": int(clip_time_ms or 0),
        "clip_time": clip_time.isoformat().replace("+00:00", "Z"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Rehang current staged risk-session TP/SL orders using profile liquidity-band parameters.")
    parser.add_argument("--state-path", default=os.getenv("RISK_SESSION_STATE_PATH", "runtime/risk_session_state.json"))
    parser.add_argument("--symbol", default="", help="Execution symbol. Defaults to symbol in the state file.")
    parser.add_argument("--target-band", choices=["low_liquidity", "normal_liquidity"], default="low_liquidity")
    parser.add_argument("--execute", action="store_true", help="Cancel old TP/SL, place new TP/SL, and rewrite the risk state file.")
    parser.add_argument("--keep-stop", action="store_true", help="Keep the existing initial SL instead of re-clipping it for the target liquidity band.")
    parser.add_argument("--allow-running-agent", action="store_true", help="Allow execution while unified_market_agent.py is running. Not recommended.")
    parser.add_argument("--allow-non-initial", action="store_true", help="Bypass the initial-stage guard. Not recommended.")
    args = parser.parse_args()

    state_path = Path(args.state_path)
    state_payload = _load_state(state_path)
    symbol = canonicalize_execution_symbol(args.symbol or state_payload.get("symbol", ""))
    if not symbol:
        raise ValueError("missing symbol; pass --symbol or include symbol in state")

    stage = _state_stage(state_payload)
    if stage != "initial" and not args.allow_non_initial:
        raise RuntimeError(f"refusing to rehang non-initial risk session stage={stage!r}; pass --allow-non-initial only after manual review")
    if bool(state_payload.get("tp1_hit", False)) or bool(state_payload.get("tp2_hit", False)):
        if not args.allow_non_initial:
            raise RuntimeError("refusing to rehang after TP hit flags; this script is intended for initial-stage sessions")

    running = _running_agent_processes()
    if args.execute and running and not args.allow_running_agent:
        raise RuntimeError(
            "unified_market_agent.py appears to be running. Stop it before --execute, or pass --allow-running-agent if you accept stale in-memory risk-session risk:\n"
            + "\n".join(running)
        )
    if args.execute and os.getenv("ENABLE_LIVE_TRADING", "false").strip().lower() not in {"1", "true", "yes", "on"}:
        raise RuntimeError("ENABLE_LIVE_TRADING is not true; refusing --execute")

    # The script only needs REST reads/writes. Keep the user-fills websocket off
    # so dry-runs and failures exit cleanly without background subscriptions.
    os.environ["ENABLE_HYPERLIQUID_USER_FILLS_WEBSOCKET"] = "false"
    agent = UnifiedMarketAgent("")
    agent._set_active_symbol(symbol, reason="rehang_risk_session_tpsl")
    snapshot = agent.reader.get_position_snapshot(symbol)
    if not snapshot_has_open_position(snapshot):
        raise RuntimeError(f"no open position for {symbol}")
    if canonicalize_execution_symbol(snapshot.get("symbol", "")) != symbol:
        raise RuntimeError(f"position snapshot symbol mismatch: expected={symbol} snapshot={snapshot.get('symbol')}")

    current_refs, _, _, _ = agent._exchange_reduce_only_order_refs_from_snapshot(
        snapshot,
        entry_price=safe_float(state_payload.get("initial_entry_price"), 0.0) or safe_float(snapshot.get("entry_price"), 0.0) or 0.0,
    )
    old_refs = [dict(item) for item in list(state_payload.get("resting_exit_orders") or []) if isinstance(item, dict)]
    if not old_refs:
        raise RuntimeError("state has no resting_exit_orders to replace")
    if not _order_refs_match_current_exchange(agent, old_refs, current_refs, symbol):
        raise RuntimeError(
            "state resting_exit_orders do not match current exchange open TP/SL orders; refusing to cancel/replace\n"
            f"state_refs={_json_dump(old_refs)}\ncurrent_refs={_json_dump(current_refs)}"
        )

    restored_session = agent._risk_session_from_state_payload(state_payload, snapshot)
    if restored_session is None:
        raise RuntimeError("could not load RiskSession from state payload")
    target_params = agent._staged_exit_params_for_profile_band(symbol, args.target_band)
    entry_price = float(getattr(restored_session, "initial_entry_price", 0.0) or state_payload.get("initial_entry_price", 0.0) or 0.0)
    if entry_price <= 0.0:
        raise RuntimeError("missing risk anchor entry; refusing to rehang without initial_entry_price")
    stop_price = float(getattr(restored_session, "initial_stop_price", 0.0) or state_payload.get("initial_stop_price", 0.0) or state_payload.get("stop_loss_price", 0.0) or 0.0)
    if stop_price <= 0.0:
        raise RuntimeError("missing initial stop; refusing to rehang without initial_stop_price")
    clip_time_ms = _infer_clip_time_ms(state_payload, old_refs)
    stop_clip = _profile_clipped_stop_for_band(
        agent,
        symbol=symbol,
        side=str(snapshot.get("side", "") or ""),
        entry_price=entry_price,
        stop_price=stop_price,
        target_band=args.target_band,
        clip_time_ms=clip_time_ms,
    )
    target_stop_price = stop_price if args.keep_stop else float(stop_clip.get("stop_loss_price", stop_price) or stop_price)
    new_session = agent._build_staged_risk_session_from_stop(
        position_after=snapshot,
        plan_name=str(getattr(restored_session, "plan_name", "") or state_payload.get("plan_name", "position_management") or "position_management"),
        initial_entry_price=entry_price,
        stop_loss_price=target_stop_price,
        position_management=getattr(restored_session, "position_management", None),
        risk_entry_source=str(getattr(restored_session, "risk_entry_source", "") or state_payload.get("risk_entry_source", "")),
        staged_exit_params_override=target_params,
    )
    if new_session is None:
        raise RuntimeError("could not rebuild target staged RiskSession")
    if getattr(new_session, "position_management", None) is not None and getattr(new_session.position_management, "action_decision", None) is not None:
        new_session.position_management.action_decision.entry_price = entry_price
        new_session.position_management.action_decision.stop_loss_price = target_stop_price
    new_session.start_time = float(getattr(restored_session, "start_time", 0.0) or time.time())
    new_session.initial_size_abs = max(float(getattr(restored_session, "initial_size_abs", 0.0) or 0.0), abs(float(snapshot.get("size", 0.0) or 0.0)))
    new_session.staged_exit_size_basis_abs = max(float(getattr(restored_session, "staged_exit_size_basis_abs", 0.0) or 0.0), new_session.initial_size_abs)
    new_session.resting_exit_orders = old_refs
    new_session.use_resting_exit_orders = True

    old_specs = _summarize_specs(agent._iter_risk_session_exit_order_specs(restored_session))
    new_specs = _summarize_specs(agent._iter_risk_session_exit_order_specs(new_session))
    print("[rehang] symbol=", symbol)
    print("[rehang] stage=", stage, "target_band=", args.target_band, "execute=", args.execute)
    print("[rehang] snapshot=", _json_dump({k: snapshot.get(k) for k in ["symbol", "side", "size", "entry_price", "mid_price", "leverage"]}))
    print("[rehang] old_specs=", _json_dump(old_specs))
    print("[rehang] new_specs=", _json_dump(new_specs))
    print("[rehang] target_params=", _json_dump(target_params))
    print("[rehang] stop_clip=", _json_dump(stop_clip))

    if not args.execute:
        print("[rehang] dry-run only. Re-run with --execute after stopping unified_market_agent.py to cancel/re-place orders and rewrite state.")
        agent.shutdown()
        return 0

    agent.risk_session = new_session
    agent._sync_risk_session_resting_orders(new_session)
    if not bool(getattr(new_session, "use_resting_exit_orders", False)) or not list(getattr(new_session, "resting_exit_orders", []) or []):
        raise RuntimeError("order rehang failed; new session has no resting_exit_orders after sync")
    agent._persist_risk_session_state()
    print("[rehang] placed_refs=", _json_dump(list(getattr(new_session, "resting_exit_orders", []) or [])))
    print(f"[rehang] wrote state: {state_path}")
    print("[rehang] restart unified_market_agent.py so it loads the rewritten risk session.")
    agent.shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
