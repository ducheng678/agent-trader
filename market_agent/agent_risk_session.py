import json
import os
import statistics
import threading
import time
from collections import deque
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_DOWN
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from market_agent.calibration import extract_raw_confidence_value
from market_agent.constants import MANAGEMENT_EXPOSURE_ACTION_VALUES
from market_agent.exchange import HyperliquidExecutor
from market_agent.market_profiles import InstrumentMarketProfile
from market_agent.models import (
    SCENARIO_RUNTIME_KEY,
    Condition,
    ExitLeg,
    ManagementDecision,
    PendingEntryOrderSession,
    PositionManagementPlan,
    PositionManagementSession,
    RiskSession,
    ScenarioRuntime,
    StrategyDecision,
)
from market_agent.playbook import GenericPlaybook
from market_agent.positions import snapshot_has_open_position
from market_agent.presentation import _status_format_price
from market_agent.runtime_views import (
    build_effective_target_position,
    build_empty_management_decision,
    build_management_exposure_entry_decision,
    compare_position_management_plans,
    position_management_plan_has_content,
)
from market_agent.symbols import canonicalize_execution_symbol
from market_agent.utils import format_query_amount, safe_float


class RiskSessionMixin:
    def _cross_asset_soft_stop_symbol(self) -> str:
        return canonicalize_execution_symbol(
            str(getattr(self, "risk_cross_asset_soft_stop_symbol", "") or "").strip()
        )

    def _cross_asset_soft_stop_cache_snapshot(self) -> Dict[str, Any]:
        lock = getattr(self, "_cross_asset_soft_stop_lock", None)
        cache = getattr(self, "_cross_asset_soft_stop_cache", None)
        if lock is None or cache is None:
            return {}
        with lock:
            return dict(cache)

    def _ensure_cross_asset_soft_stop_poller(self) -> None:
        if not bool(getattr(self, "risk_cross_asset_soft_stop_enabled", False)):
            return
        symbol = self._cross_asset_soft_stop_symbol()
        if not symbol:
            return
        if getattr(self, "_cross_asset_soft_stop_lock", None) is None:
            self._cross_asset_soft_stop_lock = threading.Lock()
        if getattr(self, "_cross_asset_soft_stop_stop_event", None) is None:
            self._cross_asset_soft_stop_stop_event = threading.Event()
        if getattr(self, "_cross_asset_soft_stop_cache", None) is None:
            self._cross_asset_soft_stop_cache = {}
        thread = getattr(self, "_cross_asset_soft_stop_thread", None)
        if thread is not None and thread.is_alive():
            return
        stop_event = self._cross_asset_soft_stop_stop_event
        stop_event.clear()
        thread = threading.Thread(
            target=self._cross_asset_soft_stop_poll_loop,
            name="cross-asset-soft-stop",
            daemon=True,
        )
        self._cross_asset_soft_stop_thread = thread
        thread.start()

    def _shutdown_cross_asset_soft_stop_poller(self) -> None:
        stop_event = getattr(self, "_cross_asset_soft_stop_stop_event", None)
        if stop_event is not None:
            stop_event.set()
        thread = getattr(self, "_cross_asset_soft_stop_thread", None)
        if thread is not None and thread.is_alive():
            try:
                thread.join(timeout=1.0)
            except Exception:
                pass
        self._cross_asset_soft_stop_thread = None

    def _cross_asset_soft_stop_poll_loop(self) -> None:
        interval = max(1.0, float(getattr(self, "risk_cross_asset_soft_stop_poll_seconds", 10.0) or 10.0))
        stop_event = getattr(self, "_cross_asset_soft_stop_stop_event", None)
        while stop_event is not None and not stop_event.is_set():
            symbol = self._cross_asset_soft_stop_symbol()
            now = time.time()
            if not symbol:
                if stop_event.wait(interval):
                    break
                continue
            try:
                price = safe_float(self.reader.get_mid_price(symbol), None)
                if price is not None and price > 0.0:
                    payload = {
                        "symbol": symbol,
                        "price": float(price),
                        "updated_at": now,
                        "error": "",
                    }
                else:
                    payload = {
                        "symbol": symbol,
                        "price": 0.0,
                        "updated_at": 0.0,
                        "error": "missing_price",
                        "error_at": now,
                    }
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                payload = {
                    "symbol": symbol,
                    "price": 0.0,
                    "updated_at": 0.0,
                    "error": str(exc),
                    "error_at": now,
                }
            lock = getattr(self, "_cross_asset_soft_stop_lock", None)
            if lock is not None:
                with lock:
                    if payload.get("error"):
                        previous = dict(getattr(self, "_cross_asset_soft_stop_cache", {}) or {})
                        if safe_float(previous.get("price"), 0.0) and float(previous.get("updated_at", 0.0) or 0.0) > 0.0:
                            previous["error"] = payload.get("error", "")
                            previous["error_at"] = payload.get("error_at", now)
                            previous["symbol"] = symbol
                            payload = previous
                    self._cross_asset_soft_stop_cache = payload
            if stop_event.wait(interval):
                break

    def _initialize_risk_session_cross_asset_reference(self, session: Optional[RiskSession]) -> None:
        if session is None or not bool(getattr(self, "risk_cross_asset_soft_stop_enabled", False)):
            return
        symbol = self._cross_asset_soft_stop_symbol()
        if not symbol:
            return
        current_symbol = canonicalize_execution_symbol(getattr(session, "cross_asset_soft_stop_symbol", "") or "")
        entry_price = safe_float(getattr(session, "cross_asset_entry_price", 0.0), 0.0) or 0.0
        if current_symbol == symbol and entry_price > 0.0:
            return
        cache = self._cross_asset_soft_stop_cache_snapshot()
        if canonicalize_execution_symbol(cache.get("symbol", "") or "") != symbol:
            return
        price = safe_float(cache.get("price"), None)
        updated_at = float(cache.get("updated_at", 0.0) or 0.0)
        max_age = max(
            float(getattr(self, "risk_cross_asset_soft_stop_poll_seconds", 10.0) or 10.0) * 3.0,
            float(getattr(self, "risk_cross_asset_soft_stop_cache_max_age_seconds", 30.0) or 30.0),
        )
        now = time.time()
        if price is None or price <= 0.0 or updated_at <= 0.0 or now - updated_at > max_age:
            return
        session.cross_asset_soft_stop_symbol = symbol
        session.cross_asset_entry_price = float(price)
        session.cross_asset_entry_time = updated_at
        session.cross_asset_peak_adverse_pct = 0.0

    def _risk_session_cross_asset_soft_stop_adjustment(
        self,
        rs: RiskSession,
        *,
        base_soft_stop: float,
        side: str,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "enabled": False,
            "base_soft_stop_price": float(base_soft_stop or 0.0),
            "effective_soft_stop_price": float(base_soft_stop or 0.0),
            "buffer_r": 0.0,
            "buffer_usd": 0.0,
        }
        if not bool(getattr(self, "risk_cross_asset_soft_stop_enabled", False)):
            return payload
        if side not in {"long", "short"} or base_soft_stop <= 0.0:
            return payload
        self._initialize_risk_session_cross_asset_reference(rs)
        symbol = self._cross_asset_soft_stop_symbol()
        entry_price = safe_float(getattr(rs, "cross_asset_entry_price", 0.0), 0.0) or 0.0
        session_symbol = canonicalize_execution_symbol(getattr(rs, "cross_asset_soft_stop_symbol", "") or "")
        if not symbol or session_symbol != symbol or entry_price <= 0.0:
            return payload
        cache = self._cross_asset_soft_stop_cache_snapshot()
        current_price = safe_float(cache.get("price"), None)
        updated_at = float(cache.get("updated_at", 0.0) or 0.0)
        max_age = max(1.0, float(getattr(self, "risk_cross_asset_soft_stop_cache_max_age_seconds", 30.0) or 30.0))
        now = time.time()
        if (
            canonicalize_execution_symbol(cache.get("symbol", "") or "") != symbol
            or current_price is None
            or current_price <= 0.0
            or updated_at <= 0.0
            or now - updated_at > max_age
        ):
            payload.update(
                {
                    "enabled": True,
                    "symbol": symbol,
                    "entry_price": entry_price,
                    "current_price": current_price or 0.0,
                    "cache_age_seconds": max(0.0, now - updated_at) if updated_at > 0.0 else 0.0,
                    "status": "cache_unavailable",
                }
            )
            return payload
        move_pct = (float(current_price) / entry_price - 1.0) * 100.0
        current_adverse_pct = max(0.0, move_pct if side == "long" else -move_pct)
        previous_peak = max(0.0, float(getattr(rs, "cross_asset_peak_adverse_pct", 0.0) or 0.0))
        peak_adverse_pct = max(previous_peak, current_adverse_pct)
        if peak_adverse_pct > previous_peak:
            rs.cross_asset_peak_adverse_pct = peak_adverse_pct
            last_persist = float(getattr(self, "_cross_asset_soft_stop_peak_persist_at", 0.0) or 0.0)
            if now - last_persist >= 30.0:
                self._cross_asset_soft_stop_peak_persist_at = now
                persist_state = getattr(self, "_persist_risk_session_state", None)
                if callable(persist_state):
                    try:
                        persist_state()
                    except Exception:
                        pass
        release_pct = max(0.0, float(getattr(self, "risk_cross_asset_soft_stop_release_pct", 0.05) or 0.0))
        if release_pct > 0.0 and current_adverse_pct <= max(0.0, peak_adverse_pct - release_pct):
            buffer_basis_pct = min(peak_adverse_pct, current_adverse_pct + release_pct * 0.5)
        else:
            buffer_basis_pct = peak_adverse_pct
        start_pct = max(0.0, float(getattr(self, "risk_cross_asset_soft_stop_start_pct", 0.04) or 0.0))
        full_pct = max(start_pct + 1e-9, float(getattr(self, "risk_cross_asset_soft_stop_full_pct", 0.14) or 0.14))
        max_buffer_r = max(0.0, float(getattr(self, "risk_cross_asset_soft_stop_max_buffer_r", 0.20) or 0.0))
        strength = min(1.0, max(0.0, (buffer_basis_pct - start_pct) / (full_pct - start_pct)))
        buffer_r = strength * max_buffer_r
        risk_distance = max(0.0, float(getattr(rs, "initial_risk_price_distance", 0.0) or 0.0))
        buffer_usd = buffer_r * risk_distance
        effective_soft_stop = float(base_soft_stop or 0.0)
        if buffer_usd > 0.0:
            if side == "long":
                effective_soft_stop = max(0.0, effective_soft_stop - buffer_usd)
            else:
                effective_soft_stop = effective_soft_stop + buffer_usd
        payload.update(
            {
                "enabled": True,
                "status": "ok",
                "symbol": symbol,
                "entry_price": entry_price,
                "entry_time": float(getattr(rs, "cross_asset_entry_time", 0.0) or 0.0),
                "current_price": float(current_price),
                "current_price_updated_at": updated_at,
                "cache_age_seconds": max(0.0, now - updated_at),
                "move_pct": move_pct,
                "current_adverse_pct": current_adverse_pct,
                "peak_adverse_pct": peak_adverse_pct,
                "buffer_basis_pct": buffer_basis_pct,
                "release_pct": release_pct,
                "start_pct": start_pct,
                "full_pct": full_pct,
                "max_buffer_r": max_buffer_r,
                "buffer_r": buffer_r,
                "buffer_usd": buffer_usd,
                "effective_soft_stop_price": effective_soft_stop,
            }
        )
        return payload

    def _risk_session_state_file_path(self) -> Optional[Path]:
        if not bool(getattr(self, "risk_session_state_enabled", False)):
            return None
        raw_path = getattr(self, "risk_session_state_path", None)
        if raw_path is None:
            return None
        try:
            return Path(raw_path)
        except TypeError:
            return None

    @staticmethod
    def _risk_condition_to_dict(condition: Condition) -> dict:
        return {
            "type": str(getattr(condition, "type", "") or ""),
            "level": float(getattr(condition, "level", 0.0) or 0.0),
            "low": float(getattr(condition, "low", 0.0) or 0.0),
            "high": float(getattr(condition, "high", 0.0) or 0.0),
            "timer_seconds": int(getattr(condition, "timer_seconds", 0) or 0),
            "tolerance_bps": float(getattr(condition, "tolerance_bps", 0.0) or 0.0),
            "min_ratio": float(getattr(condition, "min_ratio", 0.0) or 0.0),
            "note": str(getattr(condition, "note", "") or ""),
        }

    @classmethod
    def _risk_exit_leg_to_dict(cls, leg: ExitLeg) -> dict:
        return {
            "name": str(getattr(leg, "name", "") or ""),
            "note": str(getattr(leg, "note", "") or ""),
            "when_all": [cls._risk_condition_to_dict(item) for item in list(getattr(leg, "when_all", []) or [])],
            "close_fraction": float(getattr(leg, "close_fraction", 0.0) or 0.0),
        }

    @staticmethod
    def _risk_condition_from_dict(payload: Any) -> Condition:
        data = payload if isinstance(payload, dict) else {}
        return Condition(
            type=str(data.get("type", "") or ""),
            level=float(data.get("level", 0.0) or 0.0),
            low=float(data.get("low", 0.0) or 0.0),
            high=float(data.get("high", 0.0) or 0.0),
            timer_seconds=int(data.get("timer_seconds", data.get("seconds", 0)) or 0),
            tolerance_bps=float(data.get("tolerance_bps", 0.0) or 0.0),
            min_ratio=float(data.get("min_ratio", 0.0) or 0.0),
            note=str(data.get("note", "") or ""),
        )

    @classmethod
    def _risk_exit_leg_from_dict(cls, payload: Any) -> ExitLeg:
        data = payload if isinstance(payload, dict) else {}
        return ExitLeg(
            name=str(data.get("name", "") or ""),
            note=str(data.get("note", "") or ""),
            when_all=[cls._risk_condition_from_dict(item) for item in list(data.get("when_all") or [])],
            close_fraction=float(data.get("close_fraction", 0.0) or 0.0),
        )

    @staticmethod
    def _risk_management_plan_from_dict(payload: Any, fallback_entry: float, fallback_stop: float, fallback_leverage: int = 0) -> PositionManagementPlan:
        data = payload if isinstance(payload, dict) else {}
        decision_data = data.get("action_decision") if isinstance(data.get("action_decision"), dict) else {}
        decision = ManagementDecision(
            action=str(decision_data.get("action", "no_change") or "no_change"),
            close_fraction=float(decision_data.get("close_fraction", 0.0) or 0.0),
            new_notional_usd=float(decision_data.get("new_notional_usd", 0.0) or 0.0),
            entry_price=float(decision_data.get("entry_price", fallback_entry) or fallback_entry or 0.0),
            stop_loss_price=float(decision_data.get("stop_loss_price", fallback_stop) or fallback_stop or 0.0),
            planned_max_loss_usd=float(decision_data.get("planned_max_loss_usd", 0.0) or 0.0),
            leverage=int(decision_data.get("leverage", fallback_leverage) or fallback_leverage or 0),
            margin_basis_usd=float(decision_data.get("margin_basis_usd", 0.0) or 0.0),
            continue_entry_plan_after_close=bool(decision_data.get("continue_entry_plan_after_close", False)),
        )
        return PositionManagementPlan(
            execute_now=False,
            action_decision=decision,
            scenario=None,
        )

    @staticmethod
    def _risk_session_stage_name(session: RiskSession) -> str:
        if bool(getattr(session, "tp2_hit", False)):
            return "tail"
        if bool(getattr(session, "tp1_hit", False)):
            return "post_tp1"
        return "initial"

    def _risk_session_state_payload(self, session: RiskSession) -> dict:
        return {
            "version": 2,
            "updated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "updated_at_ms": int(time.time() * 1000),
            "symbol": canonicalize_execution_symbol(getattr(self, "symbol", "") or ""),
            "plan_name": str(getattr(session, "plan_name", "") or ""),
            "side": str(getattr(session, "side", "") or ""),
            "anchor_source": str(getattr(session, "risk_entry_source", "") or ""),
            "risk_entry_source": str(getattr(session, "risk_entry_source", "") or ""),
            "stage": self._risk_session_stage_name(session),
            "stop_loss_price": float(getattr(session, "stop_loss_price", 0.0) or 0.0),
            "start_time": float(getattr(session, "start_time", 0.0) or 0.0),
            "baseline_size": float(getattr(session, "baseline_size", 0.0) or 0.0),
            "expected_size": float(getattr(session, "expected_size", 0.0) or 0.0),
            "initial_size_abs": float(getattr(session, "initial_size_abs", 0.0) or 0.0),
            "take_profit_legs": [self._risk_exit_leg_to_dict(item) for item in list(getattr(session, "take_profit_legs", []) or [])],
            "stop_loss_legs": [self._risk_exit_leg_to_dict(item) for item in list(getattr(session, "stop_loss_legs", []) or [])],
            "completed_leg_keys": sorted(str(item) for item in set(getattr(session, "executed_leg_names", set()) or set())),
            "executed_leg_names": sorted(str(item) for item in set(getattr(session, "executed_leg_names", set()) or set())),
            "resting_exit_orders": self._normalized_risk_session_order_refs(session),
            "use_resting_exit_orders": bool(getattr(session, "use_resting_exit_orders", False)),
            "take_profit_legs_scale_from_initial_size": bool(getattr(session, "take_profit_legs_scale_from_initial_size", False)),
            "staged_exit_enabled": bool(getattr(session, "staged_exit_enabled", False)),
            "staged_exit_size_basis_abs": float(getattr(session, "staged_exit_size_basis_abs", 0.0) or 0.0),
            "tp1_completed_size_abs": self._aligned_risk_completed_size_for_session(session, "stage_tp1"),
            "tp2_completed_size_abs": self._aligned_risk_completed_size_for_session(session, "stage_tp2"),
            "initial_entry_price": float(getattr(session, "initial_entry_price", 0.0) or 0.0),
            "initial_stop_price": float(getattr(session, "initial_stop_price", 0.0) or 0.0),
            "initial_risk_price_distance": float(getattr(session, "initial_risk_price_distance", 0.0) or 0.0),
            "tp1_price": float(getattr(session, "tp1_price", 0.0) or 0.0),
            "tp2_price": float(getattr(session, "tp2_price", 0.0) or 0.0),
            "tp1_hit": bool(getattr(session, "tp1_hit", False)),
            "tp2_hit": bool(getattr(session, "tp2_hit", False)),
            "tp1_hit_at": float(getattr(session, "tp1_hit_at", 0.0) or 0.0),
            "tp2_hit_at": float(getattr(session, "tp2_hit_at", 0.0) or 0.0),
            "staged_exit_liquidity_band": str(getattr(session, "staged_exit_liquidity_band", "") or ""),
            "max_favorable_excursion_r": float(getattr(session, "max_favorable_excursion_r", 0.0) or 0.0),
            "tp1_no_follow_through_applied": bool(getattr(session, "tp1_no_follow_through_applied", False)),
            "tp1_no_follow_through_at": float(getattr(session, "tp1_no_follow_through_at", 0.0) or 0.0),
            "tp2_no_continuation_applied": bool(getattr(session, "tp2_no_continuation_applied", False)),
            "post_tp1_stop_price": float(getattr(session, "post_tp1_stop_price", 0.0) or 0.0),
            "locked_floor_price": float(getattr(session, "locked_floor_price", 0.0) or 0.0),
            "active_soft_stop_price": float(getattr(session, "active_soft_stop_price", 0.0) or 0.0),
            "active_hard_stop_price": float(getattr(session, "active_hard_stop_price", 0.0) or 0.0),
            "cross_asset_soft_stop_symbol": str(getattr(session, "cross_asset_soft_stop_symbol", "") or ""),
            "cross_asset_entry_price": float(getattr(session, "cross_asset_entry_price", 0.0) or 0.0),
            "cross_asset_entry_time": float(getattr(session, "cross_asset_entry_time", 0.0) or 0.0),
            "cross_asset_peak_adverse_pct": max(0.0, float(getattr(session, "cross_asset_peak_adverse_pct", 0.0) or 0.0)),
            "soft_stop_breach_since": float(getattr(session, "soft_stop_breach_since", 0.0) or 0.0),
            "soft_stop_last_breach_price": float(getattr(session, "soft_stop_last_breach_price", 0.0) or 0.0),
            "exchange_hard_stop_buffer_usd": float(getattr(session, "exchange_hard_stop_buffer_usd", 0.0) or 0.0),
            "exchange_hard_stop_min_buffer_usd": float(getattr(session, "exchange_hard_stop_min_buffer_usd", 0.0) or 0.0),
            "exchange_hard_stop_atr_buffer_usd": float(getattr(session, "exchange_hard_stop_atr_buffer_usd", 0.0) or 0.0),
            "exchange_hard_stop_r_buffer_usd": float(getattr(session, "exchange_hard_stop_r_buffer_usd", 0.0) or 0.0),
            "exchange_hard_stop_atr_value": float(getattr(session, "exchange_hard_stop_atr_value", 0.0) or 0.0),
            "trailing_timeframe": str(getattr(session, "trailing_timeframe", "15m") or "15m"),
            "trailing_atr_period": int(getattr(session, "trailing_atr_period", 14) or 14),
            "trailing_atr_lookback_bars": int(getattr(session, "trailing_atr_lookback_bars", 200) or 200),
            "trailing_soft_atr_mult": float(getattr(session, "trailing_soft_atr_mult", 2.5) or 2.5),
            "trailing_hard_atr_mult": float(getattr(session, "trailing_hard_atr_mult", 3.5) or 3.5),
            "trailing_highest_close": float(getattr(session, "trailing_highest_close", 0.0) or 0.0),
            "trailing_lowest_close": float(getattr(session, "trailing_lowest_close", 0.0) or 0.0),
            "trailing_soft_stop_price": float(getattr(session, "trailing_soft_stop_price", 0.0) or 0.0),
            "trailing_hard_stop_price": float(getattr(session, "trailing_hard_stop_price", 0.0) or 0.0),
            "trailing_last_bar_ms": int(getattr(session, "trailing_last_bar_ms", 0) or 0),
            "trailing_last_close_price": float(getattr(session, "trailing_last_close_price", 0.0) or 0.0),
            "position_basis_confidence_raw": extract_raw_confidence_value(getattr(session, "position_basis_confidence_raw", None)),
            "position_basis_validity": max(0.0, min(1.0, float(getattr(session, "position_basis_validity", 0.0) or 0.0))),
            "basis_profit_observation_active": bool(getattr(session, "basis_profit_observation_active", False)),
            "basis_profit_observation_started_at": float(getattr(session, "basis_profit_observation_started_at", 0.0) or 0.0),
            "basis_profit_observation_basis_start": float(getattr(session, "basis_profit_observation_basis_start", 0.0) or 0.0),
            "position_management": session.position_management.to_dict() if isinstance(getattr(session, "position_management", None), PositionManagementPlan) else None,
        }

    def _risk_session_from_state_payload(self, payload: Any, position_snapshot: dict) -> Optional[RiskSession]:
        data = payload if isinstance(payload, dict) else {}
        if not data or not snapshot_has_open_position(position_snapshot):
            return None
        state_symbol = canonicalize_execution_symbol(data.get("symbol", "") or "")
        snapshot_symbol = canonicalize_execution_symbol(position_snapshot.get("symbol", "") or getattr(self, "symbol", "") or "")
        if state_symbol and snapshot_symbol and state_symbol != snapshot_symbol:
            return None
        state_side = str(data.get("side", "") or "")
        snapshot_side = str(position_snapshot.get("side", "") or "")
        if state_side not in {"long", "short"} or snapshot_side not in {"long", "short"} or state_side != snapshot_side:
            return None
        position_management = self._risk_management_plan_from_dict(
            data.get("position_management"),
            float(data.get("initial_entry_price", 0.0) or 0.0),
            float(data.get("stop_loss_price", data.get("initial_stop_price", 0.0)) or 0.0),
            int(position_snapshot.get("leverage", 0) or 0),
        )
        completed_keys = list(data.get("completed_leg_keys") or data.get("executed_leg_names") or [])
        session = RiskSession(
            plan_name=str(data.get("plan_name", "") or "startup_persisted_restore"),
            side=state_side,
            stop_loss_price=float(data.get("stop_loss_price", 0.0) or 0.0),
            start_time=float(data.get("start_time", 0.0) or time.time()),
            baseline_size=float(data.get("baseline_size", position_snapshot.get("size", 0.0)) or 0.0),
            position_management=position_management,
            expected_size=float(data.get("expected_size", position_snapshot.get("size", 0.0)) or 0.0),
            initial_size_abs=float(data.get("initial_size_abs", 0.0) or abs(float(position_snapshot.get("size", 0.0) or 0.0))),
            take_profit_legs=[self._risk_exit_leg_from_dict(item) for item in list(data.get("take_profit_legs") or [])],
            stop_loss_legs=[self._risk_exit_leg_from_dict(item) for item in list(data.get("stop_loss_legs") or [])],
            runtimes={},
            history_seconds=float(data.get("history_seconds", getattr(self, "price_history_seconds", 1800)) or getattr(self, "price_history_seconds", 1800)),
            executed_leg_names=set(str(item) for item in completed_keys if str(item or "")),
            resting_exit_orders=[dict(item) for item in list(data.get("resting_exit_orders") or []) if isinstance(item, dict)],
            use_resting_exit_orders=bool(data.get("use_resting_exit_orders", False)),
            take_profit_legs_scale_from_initial_size=bool(data.get("take_profit_legs_scale_from_initial_size", False)),
            staged_exit_enabled=bool(data.get("staged_exit_enabled", False)),
            staged_exit_size_basis_abs=float(data.get("staged_exit_size_basis_abs", 0.0) or 0.0),
            tp1_completed_size_abs=float(data.get("tp1_completed_size_abs", 0.0) or 0.0),
            tp2_completed_size_abs=float(data.get("tp2_completed_size_abs", 0.0) or 0.0),
            initial_entry_price=float(data.get("initial_entry_price", 0.0) or 0.0),
            initial_stop_price=float(data.get("initial_stop_price", 0.0) or 0.0),
            initial_risk_price_distance=float(data.get("initial_risk_price_distance", 0.0) or 0.0),
            tp1_price=float(data.get("tp1_price", 0.0) or 0.0),
            tp2_price=float(data.get("tp2_price", 0.0) or 0.0),
            tp1_hit=bool(data.get("tp1_hit", False)),
            tp2_hit=bool(data.get("tp2_hit", False)),
            tp1_hit_at=float(data.get("tp1_hit_at", 0.0) or 0.0),
            tp2_hit_at=float(data.get("tp2_hit_at", 0.0) or 0.0),
            staged_exit_liquidity_band=str(data.get("staged_exit_liquidity_band", "") or ""),
            max_favorable_excursion_r=max(0.0, float(data.get("max_favorable_excursion_r", 0.0) or 0.0)),
            tp1_no_follow_through_applied=bool(data.get("tp1_no_follow_through_applied", False)),
            tp1_no_follow_through_at=float(data.get("tp1_no_follow_through_at", 0.0) or 0.0),
            tp2_no_continuation_applied=bool(data.get("tp2_no_continuation_applied", False)),
            post_tp1_stop_price=float(data.get("post_tp1_stop_price", 0.0) or 0.0),
            locked_floor_price=float(data.get("locked_floor_price", 0.0) or 0.0),
            active_soft_stop_price=float(data.get("active_soft_stop_price", 0.0) or 0.0),
            active_hard_stop_price=float(data.get("active_hard_stop_price", 0.0) or 0.0),
            cross_asset_soft_stop_symbol=str(data.get("cross_asset_soft_stop_symbol", "") or ""),
            cross_asset_entry_price=float(data.get("cross_asset_entry_price", 0.0) or 0.0),
            cross_asset_entry_time=float(data.get("cross_asset_entry_time", 0.0) or 0.0),
            cross_asset_peak_adverse_pct=max(0.0, float(data.get("cross_asset_peak_adverse_pct", 0.0) or 0.0)),
            soft_stop_breach_since=float(data.get("soft_stop_breach_since", 0.0) or 0.0),
            soft_stop_last_breach_price=float(data.get("soft_stop_last_breach_price", 0.0) or 0.0),
            exchange_hard_stop_buffer_usd=float(data.get("exchange_hard_stop_buffer_usd", 0.0) or 0.0),
            exchange_hard_stop_min_buffer_usd=float(data.get("exchange_hard_stop_min_buffer_usd", 0.0) or 0.0),
            exchange_hard_stop_atr_buffer_usd=float(data.get("exchange_hard_stop_atr_buffer_usd", 0.0) or 0.0),
            exchange_hard_stop_r_buffer_usd=float(data.get("exchange_hard_stop_r_buffer_usd", 0.0) or 0.0),
            exchange_hard_stop_atr_value=float(data.get("exchange_hard_stop_atr_value", 0.0) or 0.0),
            trailing_timeframe=str(data.get("trailing_timeframe", "15m") or "15m"),
            trailing_atr_period=int(data.get("trailing_atr_period", 14) or 14),
            trailing_atr_lookback_bars=int(data.get("trailing_atr_lookback_bars", 200) or 200),
            trailing_soft_atr_mult=float(data.get("trailing_soft_atr_mult", 2.5) or 2.5),
            trailing_hard_atr_mult=float(data.get("trailing_hard_atr_mult", 3.5) or 3.5),
            trailing_highest_close=float(data.get("trailing_highest_close", 0.0) or 0.0),
            trailing_lowest_close=float(data.get("trailing_lowest_close", 0.0) or 0.0),
            trailing_soft_stop_price=float(data.get("trailing_soft_stop_price", 0.0) or 0.0),
            trailing_hard_stop_price=float(data.get("trailing_hard_stop_price", 0.0) or 0.0),
            trailing_last_bar_ms=int(data.get("trailing_last_bar_ms", 0) or 0),
            trailing_last_close_price=float(data.get("trailing_last_close_price", 0.0) or 0.0),
            position_basis_confidence_raw=extract_raw_confidence_value(data.get("position_basis_confidence_raw")),
            position_basis_validity=max(0.0, min(1.0, float(data.get("position_basis_validity", 0.0) or 0.0))),
            basis_profit_observation_active=bool(data.get("basis_profit_observation_active", False)),
            basis_profit_observation_started_at=float(data.get("basis_profit_observation_started_at", 0.0) or 0.0),
            basis_profit_observation_basis_start=float(data.get("basis_profit_observation_basis_start", 0.0) or 0.0),
        )
        if bool(getattr(session, "staged_exit_enabled", False)) and not bool(getattr(session, "tp2_hit", False)):
            apply_stop_buffer = True
            if bool(getattr(session, "tp1_hit", False)):
                soft_stop = max(0.0, float(getattr(session, "post_tp1_stop_price", 0.0) or getattr(session, "active_soft_stop_price", 0.0) or 0.0))
                stop_name = "stage_post_tp1_stop"
                stop_note = "staged_post_tp1_soft_stop"
            else:
                active_soft = max(0.0, float(getattr(session, "active_soft_stop_price", 0.0) or 0.0))
                if bool(getattr(session, "tp1_no_follow_through_applied", False)) and active_soft > 0.0:
                    soft_stop = active_soft
                    apply_stop_buffer = False
                else:
                    soft_stop = max(0.0, float(getattr(session, "initial_stop_price", 0.0) or active_soft or getattr(session, "stop_loss_price", 0.0) or 0.0))
                stop_name = "stage_initial_stop"
                stop_note = "staged_initial_soft_stop"
            if soft_stop > 0.0:
                self._set_stage_soft_hard_stop(
                    session,
                    soft_stop,
                    name=stop_name,
                    note=stop_note,
                    apply_hard_buffer=apply_stop_buffer,
                )
        self._normalize_risk_session_size_state(session)
        setattr(session, "risk_entry_source", str(data.get("risk_entry_source", data.get("anchor_source", "")) or "persisted_state"))
        return session

    def _load_risk_session_state_payload(self) -> Optional[dict]:
        path = self._risk_session_state_file_path()
        if path is None or not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            self._audit_event("risk_session_state_load_failed", {"path": str(path), "error": str(exc)})
            return None
        return payload if isinstance(payload, dict) else None

    def _persist_risk_session_state(self) -> None:
        path = self._risk_session_state_file_path()
        session = getattr(self, "risk_session", None)
        if path is None:
            return
        if session is None:
            self._clear_risk_session_state()
            return
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = path.with_name(f"{path.name}.tmp")
            tmp_path.write_text(json.dumps(self._risk_session_state_payload(session), ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
            tmp_path.replace(path)
        except Exception as exc:
            self._audit_event("risk_session_state_persist_failed", {"path": str(path), "error": str(exc)})

    def _clear_risk_session_state(self) -> None:
        path = self._risk_session_state_file_path()
        if path is None:
            return
        try:
            if path.exists():
                path.unlink()
        except Exception as exc:
            self._audit_event("risk_session_state_clear_failed", {"path": str(path), "error": str(exc)})

    def _build_fallback_staged_risk_session(self, decision: StrategyDecision, position_after: dict, plan_name: str) -> Optional[RiskSession]:
        entry_price = safe_float(position_after.get("entry_price"), 0.0) or safe_float(position_after.get("mid_price"), 0.0) or 0.0
        management_plan = PositionManagementPlan(
            execute_now=False,
            action_decision=ManagementDecision(
                **{
                    **build_empty_management_decision().to_dict(),
                    "entry_price": entry_price,
                    "stop_loss_price": float(decision.stop_loss_price or 0.0),
                    "leverage": int(decision.requested_leverage or 0),
                }
            ),
            scenario=None,
        )
        return self._build_staged_risk_session_from_stop(
            position_after=position_after,
            plan_name=plan_name,
            initial_entry_price=entry_price,
            stop_loss_price=float(decision.stop_loss_price or 0.0),
            position_management=management_plan,
            risk_entry_source="position_entry_price",
        )
    def _build_management_session(self, plan: PositionManagementPlan, position_after: dict, plan_name: str) -> Optional[RiskSession]:
        if not snapshot_has_open_position(position_after):
            return None
        decision = getattr(plan, "action_decision", None)
        action = str(getattr(decision, "action", "") or "")
        plan_entry = safe_float(getattr(decision, "entry_price", 0.0), 0.0) or 0.0
        position_entry = safe_float(position_after.get("entry_price"), 0.0) or safe_float(position_after.get("mid_price"), 0.0) or 0.0
        ref = plan_entry if action in {"no_change", "add_to_long", "add_to_short"} and plan_entry > 0.0 else position_entry
        risk_entry_source = "strategy_entry_price" if action in {"no_change", "add_to_long", "add_to_short"} and plan_entry > 0.0 else "position_entry_price"
        stop_loss_price = max(0.0, float(getattr(decision, "stop_loss_price", 0.0) or 0.0))
        tp1_protect_price = position_entry if action == "no_change" and plan_entry > 0.0 and position_entry > 0.0 else 0.0
        return self._build_staged_risk_session_from_stop(
            position_after=position_after,
            plan_name=plan_name,
            initial_entry_price=ref,
            stop_loss_price=stop_loss_price,
            position_management=plan,
            risk_entry_source=risk_entry_source,
            tp1_position_entry_protect_price=tp1_protect_price,
        )
    def _protect_stage_tp1_against_position_entry(
        self,
        *,
        side: str,
        take_profit_legs: List[ExitLeg],
        tp1_price: float,
        position_entry_price: float,
    ) -> float:
        protected_price = max(0.0, float(position_entry_price or 0.0))
        if protected_price <= 0.0:
            return tp1_price
        protected_price = self._align_price_for_symbol(self.symbol, protected_price)
        adjusted_price = tp1_price
        if side == "long" and tp1_price < protected_price:
            adjusted_price = protected_price
        elif side == "short" and tp1_price > protected_price:
            adjusted_price = protected_price
        else:
            return tp1_price
        for leg in take_profit_legs:
            if str(getattr(leg, "name", "") or "") != "stage_tp1":
                continue
            for condition in list(getattr(leg, "when_all", []) or []):
                if str(getattr(condition, "type", "") or "") in {"price_ge", "price_le"}:
                    condition.level = adjusted_price
        return adjusted_price
    def _build_stage_stop_leg_for_session(self, side: str, stop_price: float, *, name: str, note: str) -> List[ExitLeg]:
        leg = self._build_single_exit_leg(
            side=side,
            trigger_price=stop_price,
            exit_kind="stop_loss",
            name=name,
            note=note,
            close_fraction=1.0,
        )
        return [leg] if leg is not None else []

    def _risk_exchange_hard_stop_buffer_details(self, session: RiskSession) -> Dict[str, float]:
        min_buffer = max(
            0.0,
            float(
                getattr(
                    self,
                    "risk_soft_stop_min_buffer_usd",
                    getattr(self, "risk_exchange_hard_stop_min_buffer_usd", 0.0),
                )
                or 0.0
            ),
        )
        atr_multiple = max(
            0.0,
            float(
                getattr(
                    self,
                    "risk_soft_stop_atr_multiple",
                    getattr(self, "risk_exchange_hard_stop_atr_multiple", 0.0),
                )
                or 0.0
            ),
        )
        r_multiple = max(
            0.0,
            float(
                getattr(
                    self,
                    "risk_soft_stop_r_multiple",
                    getattr(self, "risk_exchange_hard_stop_r_multiple", 0.0),
                )
                or 0.0
            ),
        )
        risk_distance = max(0.0, float(getattr(session, "initial_risk_price_distance", 0.0) or 0.0))
        r_buffer = risk_distance * r_multiple
        atr_value = 0.0
        atr_buffer = 0.0
        if atr_multiple > 0.0:
            try:
                candles = self._latest_completed_candles_for_risk_session(
                    session,
                    now=time.time(),
                    interval=getattr(session, "trailing_timeframe", "15m"),
                    min_bars=max(int(getattr(session, "trailing_atr_period", 14) or 14) + 5, 24),
                )
                atr = self._atr_from_completed_candles(candles, int(getattr(session, "trailing_atr_period", 14) or 14))
                if atr is not None and atr > 0.0:
                    atr_value = float(atr)
                    atr_buffer = atr_value * atr_multiple
            except Exception:
                atr_value = 0.0
                atr_buffer = 0.0
        buffer_usd = max(min_buffer, atr_buffer, r_buffer)
        return {
            "buffer_usd": buffer_usd,
            "min_buffer_usd": min_buffer,
            "atr_buffer_usd": atr_buffer,
            "r_buffer_usd": r_buffer,
            "atr_value": atr_value,
        }

    def _risk_exchange_hard_stop_price(self, session: RiskSession, soft_stop_price: float) -> float:
        soft = max(0.0, float(soft_stop_price or 0.0))
        if soft <= 0.0:
            return 0.0
        details = self._risk_exchange_hard_stop_buffer_details(session)
        buffer_usd = max(0.0, float(details.get("buffer_usd", 0.0) or 0.0))
        session.exchange_hard_stop_buffer_usd = buffer_usd
        session.exchange_hard_stop_min_buffer_usd = max(0.0, float(details.get("min_buffer_usd", 0.0) or 0.0))
        session.exchange_hard_stop_atr_buffer_usd = max(0.0, float(details.get("atr_buffer_usd", 0.0) or 0.0))
        session.exchange_hard_stop_r_buffer_usd = max(0.0, float(details.get("r_buffer_usd", 0.0) or 0.0))
        session.exchange_hard_stop_atr_value = max(0.0, float(details.get("atr_value", 0.0) or 0.0))
        if session.side == "long":
            hard = max(0.0, soft - buffer_usd)
        elif session.side == "short":
            hard = soft + buffer_usd
        else:
            hard = soft
        return self._align_price_for_symbol(self.symbol, hard)

    def _set_stage_soft_hard_stop(
        self,
        session: RiskSession,
        soft_stop_price: float,
        *,
        name: str,
        note: str,
        apply_hard_buffer: bool = True,
    ) -> None:
        strategy_stop = self._align_price_for_symbol(self.symbol, max(0.0, float(soft_stop_price or 0.0)))
        if apply_hard_buffer:
            soft = self._risk_exchange_hard_stop_price(session, strategy_stop)
        else:
            soft = strategy_stop
            session.exchange_hard_stop_buffer_usd = 0.0
            session.exchange_hard_stop_min_buffer_usd = 0.0
            session.exchange_hard_stop_atr_buffer_usd = 0.0
            session.exchange_hard_stop_r_buffer_usd = 0.0
            session.exchange_hard_stop_atr_value = 0.0
        session.stop_loss_price = soft
        session.active_soft_stop_price = soft
        session.active_hard_stop_price = 0.0
        session.soft_stop_breach_since = 0.0
        session.soft_stop_last_breach_price = 0.0
        session.stop_loss_legs = []

    def _build_staged_risk_session_from_stop(
        self,
        *,
        position_after: dict,
        plan_name: str,
        initial_entry_price: float,
        stop_loss_price: float,
        position_management: Optional[PositionManagementPlan] = None,
        risk_entry_source: str = "",
        staged_exit_params_override: Optional[Dict[str, Any]] = None,
        tp1_position_entry_protect_price: float = 0.0,
        apply_stop_hard_buffer: bool = True,
    ) -> Optional[RiskSession]:
        if not snapshot_has_open_position(position_after):
            return None
        side = str(position_after.get("side", "flat") or "flat")
        entry_price = max(0.0, float(initial_entry_price or 0.0))
        stop_price = max(0.0, float(stop_loss_price or 0.0))
        if side not in {"long", "short"} or entry_price <= 0.0 or stop_price <= 0.0:
            return None
        risk_distance = abs(entry_price - stop_price)
        if risk_distance <= 0.0:
            return None
        symbol = canonicalize_execution_symbol(position_after.get("symbol", "") or getattr(self, "symbol", "") or "")
        staged_exit_params = dict(staged_exit_params_override) if isinstance(staged_exit_params_override, dict) else self._staged_exit_params_for_symbol(symbol)
        staged_legs = self._build_risk_session_stage_exit_legs(
            side=side,
            entry_price=entry_price,
            stop_loss_price=stop_price,
            staged_exit_params=staged_exit_params,
        )
        if staged_legs is None:
            return None
        take_profit_legs, stop_loss_legs = staged_legs
        tp1_price = self._price_from_r_multiple(side, entry_price, risk_distance, float(staged_exit_params.get("tp1_r_multiple", 1.0) or 1.0))
        tp2_price = self._price_from_r_multiple(side, entry_price, risk_distance, float(staged_exit_params.get("tp2_r_multiple", 2.0) or 2.0))
        if tp1_price <= 0.0 or tp2_price <= 0.0:
            return None
        tp1_price = self._protect_stage_tp1_against_position_entry(
            side=side,
            take_profit_legs=take_profit_legs,
            tp1_price=tp1_price,
            position_entry_price=tp1_position_entry_protect_price,
        )
        post_tp1_stop_r_multiple = float(staged_exit_params.get("post_tp1_stop_r_multiple", -0.40) or -0.40)
        post_tp2_locked_r_multiple = float(staged_exit_params.get("post_tp2_locked_r_multiple", 1.0) or 1.0)
        post_tp1_stop_price = self._price_from_r_multiple(side, entry_price, risk_distance, post_tp1_stop_r_multiple)
        locked_floor_price = self._price_from_r_multiple(side, entry_price, risk_distance, post_tp2_locked_r_multiple)
        baseline_size = float(position_after.get("size", 0.0) or 0.0)
        session = RiskSession(
            plan_name=plan_name,
            side=side,
            stop_loss_price=stop_price,
            start_time=time.time(),
            baseline_size=baseline_size,
            position_management=position_management,
            expected_size=baseline_size,
            initial_size_abs=abs(baseline_size),
            take_profit_legs=take_profit_legs,
            stop_loss_legs=stop_loss_legs,
            runtimes={},
            history_seconds=float(getattr(self, "price_history_seconds", 1800) or 1800),
            take_profit_legs_scale_from_initial_size=True,
            staged_exit_enabled=True,
            staged_exit_size_basis_abs=abs(baseline_size),
            initial_entry_price=entry_price,
            initial_stop_price=stop_price,
            initial_risk_price_distance=risk_distance,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            staged_exit_liquidity_band=str(staged_exit_params.get("liquidity_band", "") or ""),
            post_tp1_stop_price=post_tp1_stop_price,
            locked_floor_price=locked_floor_price,
            trailing_timeframe=str(getattr(self, "risk_trailing_timeframe", "15m") or "15m"),
            trailing_atr_period=int(getattr(self, "risk_trailing_atr_period", 14) or 14),
            trailing_atr_lookback_bars=int(getattr(self, "risk_trailing_atr_lookback_bars", 200) or 200),
            trailing_soft_atr_mult=float(staged_exit_params.get("trailing_soft_atr_multiple", 2.5) or 2.5),
            trailing_hard_atr_mult=float(staged_exit_params.get("trailing_hard_atr_multiple", 3.5) or 3.5),
        )
        self._set_stage_soft_hard_stop(
            session,
            stop_price,
            name="stage_initial_stop",
            note="staged_initial_soft_stop",
            apply_hard_buffer=apply_stop_hard_buffer,
        )
        basis_state = self._current_position_basis_state(side) if hasattr(self, "_current_position_basis_state") else {}
        basis_raw = extract_raw_confidence_value((basis_state or {}).get("position_basis_confidence_raw"))
        if basis_raw is not None:
            session.position_basis_confidence_raw = basis_raw
            session.position_basis_validity = max(0.0, min(1.0, float((basis_state or {}).get("position_basis_validity", 1.0) or 0.0)))
        setattr(session, "risk_entry_source", str(risk_entry_source or ""))
        self._ensure_cross_asset_soft_stop_poller()
        self._initialize_risk_session_cross_asset_reference(session)
        return session
    def _build_position_management_session(self, plan: PositionManagementPlan, position_snapshot: dict, plan_name: str) -> Optional[PositionManagementSession]:
        if not position_management_plan_has_content(plan) or plan.scenario is None:
            return None
        snapshot_side = str(position_snapshot.get("side", "flat") or "flat")
        side = snapshot_side if snapshot_side in {"long", "short"} else "flat"
        if side == "flat":
            action = str(getattr(plan.action_decision, "action", "") or "")
            if action in {"long", "add_to_long", "reverse_to_long"}:
                side = "long"
            elif action in {"short", "add_to_short", "reverse_to_short"}:
                side = "short"
        ref, _ = self._resolve_decision_entry_reference(getattr(plan, "action_decision", None), position_snapshot)
        if not ref or ref <= 0:
            ref = safe_float(position_snapshot.get("entry_price"), 0.0) or safe_float(position_snapshot.get("mid_price"), 0.0) or 0.0
        baseline_size = float(position_snapshot.get("size", 0.0) or 0.0)
        basis_state = self._current_position_basis_state(side) if hasattr(self, "_current_position_basis_state") else {}
        return PositionManagementSession(
            plan_name=plan_name,
            side=side,
            playbook_reason=str(getattr(self, "current_playbook_reason", "") or ""),
            trigger_confidence_raw=extract_raw_confidence_value(getattr(getattr(self, "current_playbook", None), "trigger_confidence_raw", None)),
            position_basis_confidence_raw=extract_raw_confidence_value((basis_state or {}).get("position_basis_confidence_raw")),
            position_basis_validity=max(0.0, min(1.0, float((basis_state or {}).get("position_basis_validity", 0.0) or 0.0))),
            position_management=plan,
            start_time=time.time(),
            baseline_size=baseline_size,
            expected_size=baseline_size,
            initial_size_abs=abs(baseline_size),
            runtimes={SCENARIO_RUNTIME_KEY: ScenarioRuntime()} if plan.scenario is not None else {},
            history_seconds=self.price_history_seconds,
        )
    def _set_position_management_session_from_plan(self, plan: PositionManagementPlan, position_snapshot: dict, plan_name: str) -> None:
        current_session = getattr(self, "position_management_session", None)
        current_plan = getattr(current_session, "position_management", None) if current_session is not None else None
        if (
            current_session is not None
            and current_plan is not None
            and not current_session.scenarios_completed()
        ):
            compare_result = compare_position_management_plans(current_plan, plan)
            if not compare_result.get("should_replace", True):
                if isinstance(getattr(self, "current_playbook", None), GenericPlaybook):
                    self.current_playbook.position_management = current_plan
                    self.current_playbook.target_position = build_effective_target_position(self.current_playbook, position_snapshot)
                self._audit_event(
                    "position_management_session_retained",
                    {
                        "plan_name": plan_name,
                        "position_management": current_plan.to_dict(),
                        "incoming_position_management": plan.to_dict(),
                        "position_snapshot": position_snapshot,
                        "compare_result": compare_result,
                    },
                )
                return
            self._audit_event(
                "position_management_session_replaced",
                {
                    "plan_name": plan_name,
                    "position_management": plan.to_dict(),
                    "previous_position_management": current_plan.to_dict(),
                    "position_snapshot": position_snapshot,
                    "compare_result": compare_result,
                },
            )
        self.position_management_session = self._build_position_management_session(plan, position_snapshot, plan_name)
    @staticmethod
    def _risk_session_leg_trigger_price(leg: ExitLeg) -> float:
        for condition in list(getattr(leg, "when_all", []) or []):
            level = max(0.0, float(getattr(condition, "level", 0.0) or 0.0))
            if level > 0.0:
                return level
        return 0.0
    def _risk_session_leg_log_items(self, legs: List[ExitLeg]) -> List[Dict[str, Any]]:
        items: List[Dict[str, Any]] = []
        for leg in list(legs or []):
            if leg is None:
                continue
            items.append(
                {
                    "name": str(getattr(leg, "name", "") or ""),
                    "trigger_price": self._risk_session_leg_trigger_price(leg),
                    "close_fraction": max(0.0, float(getattr(leg, "close_fraction", 0.0) or 0.0)),
                }
            )
        return items
    @staticmethod
    def _risk_session_leg_brief(items: List[Dict[str, Any]]) -> str:
        parts: List[str] = []
        for item in list(items or []):
            name = str(item.get("name", "") or "unknown")
            price = _status_format_price(item.get("trigger_price"))
            fraction = max(0.0, float(item.get("close_fraction", 0.0) or 0.0))
            part = name
            if price:
                part += f"@{price}"
            if fraction > 0.0:
                part += f"x{format_query_amount(fraction)}"
            parts.append(part)
        return ",".join(parts) if parts else "none"
    def _log_risk_session_ready(
        self,
        session: Optional[RiskSession],
        *,
        reason: str,
        position_after: Optional[dict] = None,
    ) -> None:
        if session is None:
            return
        tp_items = self._risk_session_leg_log_items(getattr(session, "take_profit_legs", []) or [])
        sl_items = self._risk_session_leg_log_items(getattr(session, "stop_loss_legs", []) or [])
        payload = {
            "reason": str(reason or "unknown"),
            "plan_name": str(getattr(session, "plan_name", "") or ""),
            "side": str(getattr(session, "side", "") or ""),
            "expected_size": float(getattr(session, "expected_size", 0.0) or 0.0),
            "baseline_size": float(getattr(session, "baseline_size", 0.0) or 0.0),
            "initial_size_abs": float(getattr(session, "initial_size_abs", 0.0) or 0.0),
            "staged_exit_size_basis_abs": float(getattr(session, "staged_exit_size_basis_abs", 0.0) or 0.0),
            "initial_entry_price": float(getattr(session, "initial_entry_price", 0.0) or 0.0),
            "initial_stop_price": float(getattr(session, "initial_stop_price", 0.0) or 0.0),
            "initial_risk_price_distance": float(getattr(session, "initial_risk_price_distance", 0.0) or 0.0),
            "tp1_price": float(getattr(session, "tp1_price", 0.0) or 0.0),
            "tp2_price": float(getattr(session, "tp2_price", 0.0) or 0.0),
            "stop_loss_price": float(getattr(session, "stop_loss_price", 0.0) or 0.0),
            "active_soft_stop_price": float(getattr(session, "active_soft_stop_price", 0.0) or 0.0),
            "active_hard_stop_price": float(getattr(session, "active_hard_stop_price", 0.0) or 0.0),
            "cross_asset_soft_stop_symbol": str(getattr(session, "cross_asset_soft_stop_symbol", "") or ""),
            "cross_asset_entry_price": float(getattr(session, "cross_asset_entry_price", 0.0) or 0.0),
            "cross_asset_entry_time": float(getattr(session, "cross_asset_entry_time", 0.0) or 0.0),
            "cross_asset_peak_adverse_pct": max(0.0, float(getattr(session, "cross_asset_peak_adverse_pct", 0.0) or 0.0)),
            "soft_stop_breach_since": float(getattr(session, "soft_stop_breach_since", 0.0) or 0.0),
            "soft_stop_last_breach_price": float(getattr(session, "soft_stop_last_breach_price", 0.0) or 0.0),
            "exchange_hard_stop_buffer_usd": float(getattr(session, "exchange_hard_stop_buffer_usd", 0.0) or 0.0),
            "exchange_hard_stop_min_buffer_usd": float(getattr(session, "exchange_hard_stop_min_buffer_usd", 0.0) or 0.0),
            "exchange_hard_stop_atr_buffer_usd": float(getattr(session, "exchange_hard_stop_atr_buffer_usd", 0.0) or 0.0),
            "exchange_hard_stop_r_buffer_usd": float(getattr(session, "exchange_hard_stop_r_buffer_usd", 0.0) or 0.0),
            "exchange_hard_stop_atr_value": float(getattr(session, "exchange_hard_stop_atr_value", 0.0) or 0.0),
            "tp1_hit": bool(getattr(session, "tp1_hit", False)),
            "tp2_hit": bool(getattr(session, "tp2_hit", False)),
            "tp1_no_follow_through_applied": bool(getattr(session, "tp1_no_follow_through_applied", False)),
            "tp1_no_follow_through_at": float(getattr(session, "tp1_no_follow_through_at", 0.0) or 0.0),
            "position_basis_confidence_raw": extract_raw_confidence_value(getattr(session, "position_basis_confidence_raw", None)),
            "position_basis_validity": max(0.0, min(1.0, float(getattr(session, "position_basis_validity", 0.0) or 0.0))),
            "take_profit_legs": tp_items,
            "stop_loss_legs": sl_items,
            "resting_exit_orders_count": len(list(getattr(session, "resting_exit_orders", []) or [])),
            "use_resting_exit_orders": bool(getattr(session, "use_resting_exit_orders", False)),
            "trailing": {
                "timeframe": str(getattr(session, "trailing_timeframe", "") or ""),
                "atr_period": int(getattr(session, "trailing_atr_period", 0) or 0),
                "atr_lookback_bars": int(getattr(session, "trailing_atr_lookback_bars", 0) or 0),
                "soft_atr_mult": float(getattr(session, "trailing_soft_atr_mult", 0.0) or 0.0),
                "hard_atr_mult": float(getattr(session, "trailing_hard_atr_mult", 0.0) or 0.0),
                "soft_stop_price": float(getattr(session, "trailing_soft_stop_price", 0.0) or 0.0),
                "hard_stop_price": float(getattr(session, "trailing_hard_stop_price", 0.0) or 0.0),
            },
        }
        if isinstance(position_after, dict):
            payload["position_after_side"] = str(position_after.get("side", "") or "")
            payload["position_after_size"] = float(position_after.get("size", 0.0) or 0.0)
            payload["position_after_entry_price"] = float(position_after.get("entry_price", 0.0) or 0.0)
            payload["position_after_mid_price"] = float(position_after.get("mid_price", 0.0) or 0.0)

        print(
            "[risk_session_ready] "
            f"reason={payload['reason']} "
            f"plan={payload['plan_name'] or 'unknown'} "
            f"side={payload['side'] or 'unknown'} "
            f"size={format_query_amount(abs(payload['expected_size']))} "
            f"entry0={_status_format_price(payload['initial_entry_price']) or 'n/a'} "
            f"stop0={_status_format_price(payload['initial_stop_price']) or 'n/a'} "
            f"R={_status_format_price(payload['initial_risk_price_distance']) or 'n/a'} "
            f"tp1={_status_format_price(payload['tp1_price']) or 'n/a'} "
            f"tp2={_status_format_price(payload['tp2_price']) or 'n/a'} "
            f"sl={_status_format_price(payload['stop_loss_price']) or 'n/a'} "
            f"soft_sl={_status_format_price(payload['active_soft_stop_price']) or 'n/a'} "
            f"soft_buf={_status_format_price(payload['exchange_hard_stop_buffer_usd']) or 'n/a'} "
            f"hits=tp1:{payload['tp1_hit']}/tp2:{payload['tp2_hit']} "
            f"tp_legs={self._risk_session_leg_brief(tp_items)} "
            f"sl_legs={self._risk_session_leg_brief(sl_items)} "
            f"orders={payload['resting_exit_orders_count']} "
            f"trailing={payload['trailing']['timeframe']}/ATR{payload['trailing']['atr_period']}"
            f"/{format_query_amount(payload['trailing']['soft_atr_mult'])}x"
            f"/{format_query_amount(payload['trailing']['hard_atr_mult'])}x"
        )
        self._print_json_block("risk_session_ready", payload)

    def _maybe_audit_risk_session_market_context(self, snapshot: dict, now: float) -> None:
        session = getattr(self, "risk_session", None)
        if session is None:
            return
        try:
            interval = max(0.0, float(os.getenv("RISK_SESSION_MARK_PRICE_AUDIT_INTERVAL_SECONDS", "3") or 3.0))
        except (TypeError, ValueError):
            interval = 3.0
        if interval <= 0.0:
            return
        if not hasattr(self.reader, "get_market_asset_context"):
            return
        last_at = float(getattr(self, "_last_risk_session_market_context_audit_at", 0.0) or 0.0)
        if now - last_at < interval:
            return
        setattr(self, "_last_risk_session_market_context_audit_at", now)
        payload: Dict[str, Any] = {
            "plan_name": str(getattr(session, "plan_name", "") or ""),
            "symbol": canonicalize_execution_symbol(getattr(self, "symbol", "") or ""),
            "side": str(getattr(session, "side", "") or ""),
            "expected_size": float(getattr(session, "expected_size", 0.0) or 0.0),
            "snapshot_mid_price": safe_float((snapshot or {}).get("mid_price"), None),
            "active_soft_stop_price": float(getattr(session, "active_soft_stop_price", 0.0) or 0.0),
            "active_hard_stop_price": float(getattr(session, "active_hard_stop_price", 0.0) or 0.0),
            "soft_stop_breach_since": float(getattr(session, "soft_stop_breach_since", 0.0) or 0.0),
            "exchange_hard_stop_buffer_usd": float(getattr(session, "exchange_hard_stop_buffer_usd", 0.0) or 0.0),
            "exchange_hard_stop_min_buffer_usd": float(getattr(session, "exchange_hard_stop_min_buffer_usd", 0.0) or 0.0),
            "exchange_hard_stop_atr_buffer_usd": float(getattr(session, "exchange_hard_stop_atr_buffer_usd", 0.0) or 0.0),
            "exchange_hard_stop_r_buffer_usd": float(getattr(session, "exchange_hard_stop_r_buffer_usd", 0.0) or 0.0),
            "exchange_hard_stop_atr_value": float(getattr(session, "exchange_hard_stop_atr_value", 0.0) or 0.0),
            "source": "metaAndAssetCtxs",
            "at": now,
        }
        try:
            market_context = self.reader.get_market_asset_context(getattr(self, "symbol", ""))
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            payload["error"] = str(exc)
            self._audit_event("risk_session_market_context", payload)
            return
        for key in ("markPx", "oraclePx", "midPx", "impactPxs", "premium", "openInterest", "funding"):
            if key in market_context:
                payload[key] = market_context.get(key)
        for key in ("execution_symbol", "dex", "market_name", "asset_name", "asset_index"):
            if key in market_context:
                payload[key] = market_context.get(key)
        self._audit_event("risk_session_market_context", payload)

    def _market_basis_context_for_side(
        self,
        *,
        symbol: str,
        side: str,
        snapshot_mid_price: Optional[float] = None,
    ) -> Dict[str, Any]:
        target_symbol = canonicalize_execution_symbol(symbol or getattr(self, "symbol", "") or "")
        target_side = str(side or "").strip().lower()
        payload: Dict[str, Any] = {
            "available": False,
            "symbol": target_symbol,
            "side": target_side,
        }
        if target_side not in {"long", "short"} or not target_symbol:
            payload["reason"] = "missing_side_or_symbol"
            return payload
        if not hasattr(self.reader, "get_market_asset_context"):
            payload["reason"] = "market_asset_context_unavailable"
            return payload
        try:
            market_context = self.reader.get_market_asset_context(target_symbol)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            payload["reason"] = "market_asset_context_error"
            payload["error"] = str(exc)
            return payload
        if not isinstance(market_context, dict):
            payload["reason"] = "market_asset_context_not_dict"
            return payload
        oracle_px = safe_float(market_context.get("oraclePx"), None)
        mid_px = safe_float(market_context.get("midPx"), None)
        if mid_px is None or mid_px <= 0.0:
            mid_px = safe_float(snapshot_mid_price, None)
        if oracle_px is None or oracle_px <= 0.0 or mid_px is None or mid_px <= 0.0:
            payload["reason"] = "missing_oracle_or_mid"
            payload["oraclePx"] = oracle_px
            payload["midPx"] = mid_px
            return payload
        basis = float(oracle_px) - float(mid_px)
        favorable_basis = basis if target_side == "short" else -basis
        payload.update(
            {
                "available": True,
                "reason": "ok",
                "oraclePx": float(oracle_px),
                "midPx": float(mid_px),
                "basis": basis,
                "favorable_basis": favorable_basis,
                "raw_market_context": {
                    key: market_context.get(key)
                    for key in ("markPx", "oraclePx", "midPx", "impactPxs", "premium", "openInterest", "funding")
                    if key in market_context
                },
            }
        )
        return payload

    @staticmethod
    def _basis_history_slope_usd_per_min(
        samples: List[Tuple[float, float]],
        *,
        now: float,
        lookback_seconds: float,
    ) -> Optional[Dict[str, Any]]:
        lookback = max(1.0, float(lookback_seconds or 0.0))
        usable = [
            (float(ts), float(value))
            for ts, value in list(samples or [])
            if float(ts or 0.0) >= float(now or 0.0) - lookback
        ]
        if len(usable) < 3:
            return None
        usable = sorted(usable, key=lambda item: item[0])
        duration_seconds = max(0.0, usable[-1][0] - usable[0][0])
        if duration_seconds < min(60.0, lookback * 0.5):
            return None
        xs = [(ts - usable[0][0]) / 60.0 for ts, _ in usable]
        ys = [value for _, value in usable]
        x_mean = sum(xs) / len(xs)
        y_mean = sum(ys) / len(ys)
        denom = sum((x - x_mean) ** 2 for x in xs)
        if denom <= 0.0:
            return None
        slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
        return {
            "slope_usd_per_min": slope,
            "sample_count": len(usable),
            "duration_seconds": duration_seconds,
            "first_basis": usable[0][1],
            "last_basis": usable[-1][1],
            "max_basis": max(ys),
            "min_basis": min(ys),
        }

    def _clear_risk_session_basis_profit_observation(
        self,
        rs: RiskSession,
        *,
        reason: str,
        basis_context: Optional[Dict[str, Any]] = None,
        current_profit_r: Optional[float] = None,
        now: Optional[float] = None,
    ) -> None:
        if not bool(getattr(rs, "basis_profit_observation_active", False)):
            return
        payload = {
            "plan_name": str(getattr(rs, "plan_name", "") or ""),
            "side": str(getattr(rs, "side", "") or ""),
            "reason": str(reason or ""),
            "started_at": float(getattr(rs, "basis_profit_observation_started_at", 0.0) or 0.0),
            "basis_start": float(getattr(rs, "basis_profit_observation_basis_start", 0.0) or 0.0),
            "current_profit_r": current_profit_r,
            "basis_context": dict(basis_context or {}),
            "at": float(now or time.time()),
        }
        rs.basis_profit_observation_active = False
        rs.basis_profit_observation_started_at = 0.0
        rs.basis_profit_observation_basis_start = 0.0
        if getattr(rs, "basis_profit_history", None) is not None:
            rs.basis_profit_history.clear()
        self._audit_event("risk_session_basis_profit_observation_cleared", payload)
        self._persist_risk_session_state()

    def _replace_risk_session(self, session: Optional[RiskSession]) -> None:
        previous = getattr(self, "risk_session", None)
        if previous is not None and previous is not session:
            self._cancel_risk_session_resting_orders(previous)
        self.risk_session = session
        if self.risk_session is not None:
            self._ensure_cross_asset_soft_stop_poller()
            self._initialize_risk_session_cross_asset_reference(self.risk_session)
            sync_basis = getattr(self, "_sync_position_basis_from_session", None)
            if callable(sync_basis):
                sync_basis(self.risk_session, reason="risk_session_replace")
            self._sync_risk_session_resting_orders(self.risk_session)
            self._log_risk_session_ready(self.risk_session, reason="replace")
            self._persist_risk_session_state()
        else:
            self._clear_risk_session_state()
    def _pending_entry_order_matches_open_orders(
        self,
        session: PendingEntryOrderSession,
        open_orders: List[Dict[str, Any]],
    ) -> bool:
        target_oid = session.oid
        target_cloid = str(session.cloid or "").strip()
        for item in list(open_orders or []):
            if not isinstance(item, dict):
                continue
            if target_oid is not None:
                try:
                    if int(item.get("oid")) == int(target_oid):
                        return True
                except Exception:
                    pass
            item_cloid = str(item.get("cloid", "") or "").strip()
            if target_cloid and item_cloid and item_cloid == target_cloid:
                return True
        return False
    def _set_pending_entry_order_session(
        self,
        *,
        plan_name: str,
        management_decision: ManagementDecision,
        position_management: PositionManagementPlan,
        post_fill_risk_template: Optional[PositionManagementPlan],
        execution_result: Dict[str, Any],
    ) -> None:
        self.pending_entry_order_session = PendingEntryOrderSession(
            plan_name=plan_name,
            symbol=canonicalize_execution_symbol(self.symbol or ""),
            side="long" if str(management_decision.action or "") == "long" else "short",
            management_decision=management_decision,
            position_management=position_management,
            post_fill_risk_template=post_fill_risk_template,
            oid=int(execution_result.get("oid")) if execution_result.get("oid") is not None else None,
            cloid=str(execution_result.get("cloid", "") or "").strip(),
            limit_price=float(execution_result.get("limit_price", 0.0) or 0.0),
            requested_qty=float(execution_result.get("open_qty", 0.0) or 0.0),
            created_at=time.time(),
        )
        self._audit_event(
            "pending_entry_order_created",
            {
                "plan_name": plan_name,
                "symbol": self.symbol,
                "decision": management_decision.to_dict(),
                "execution_result": execution_result,
            },
        )
    def _cancel_pending_entry_order(self, reason: str) -> Optional[Dict[str, Any]]:
        session = getattr(self, "pending_entry_order_session", None)
        if session is None:
            return None
        executor = self.executor if canonicalize_execution_symbol(self.symbol or "") == canonicalize_execution_symbol(session.symbol or "") else HyperliquidExecutor(self.reader, session.symbol)
        cancel_result = executor.cancel_entry_order(
            oid=session.oid,
            cloid=session.cloid,
            reason=reason,
            plan_name=session.plan_name,
        )
        self._audit_event(
            "pending_entry_order_cancelled",
            {
                "reason": reason,
                "symbol": session.symbol,
                "plan_name": session.plan_name,
                "result": cancel_result,
            },
        )
        self.pending_entry_order_session = None
        return cancel_result
    def _finalize_pending_entry_order_fill(self, position_after: dict, *, source: str) -> None:
        session = getattr(self, "pending_entry_order_session", None)
        if session is None:
            return
        open_orders = self.reader.get_frontend_open_orders(session.symbol)
        if self._pending_entry_order_matches_open_orders(session, open_orders):
            executor = self.executor if canonicalize_execution_symbol(self.symbol or "") == canonicalize_execution_symbol(session.symbol or "") else HyperliquidExecutor(self.reader, session.symbol)
            cancel_result = executor.cancel_entry_order(
                oid=session.oid,
                cloid=session.cloid,
                reason="pending_entry_filled_cancel_residual",
                plan_name=session.plan_name,
            )
            self._audit_event(
                "pending_entry_order_residual_cancelled",
                {
                    "symbol": session.symbol,
                    "plan_name": session.plan_name,
                    "result": cancel_result,
                },
            )
        self.pending_entry_order_session = None
        self._set_risk_session_after_management_decision(
            session.management_decision,
            session.position_management,
            session.post_fill_risk_template,
            position_after,
            session.plan_name,
        )
        self._schedule_next_active_query(position_after)
        self._audit_event(
            "pending_entry_order_filled",
            {
                "source": source,
                "symbol": session.symbol,
                "plan_name": session.plan_name,
                "position_after": position_after,
            },
        )
    def step_pending_entry_order_session(self, now: float) -> Optional[str]:
        session = getattr(self, "pending_entry_order_session", None)
        if session is None:
            return None
        symbol = canonicalize_execution_symbol(session.symbol or "")
        if not symbol:
            self.pending_entry_order_session = None
            return None
        snapshot = self.reader.get_position_snapshot(symbol)
        if snapshot_has_open_position(snapshot):
            self._finalize_pending_entry_order_fill(snapshot, source="position_opened")
            return None
        open_orders = self.reader.get_frontend_open_orders(symbol)
        if self._pending_entry_order_matches_open_orders(session, open_orders):
            return None
        self._audit_event(
            "pending_entry_order_missing",
            {
                "symbol": symbol,
                "plan_name": session.plan_name,
                "at": now,
            },
        )
        self.pending_entry_order_session = None
        self._schedule_next_active_query(snapshot)
        return None
    def _risk_session_order_qty_tolerance(self, symbol: str = "") -> float:
        base_tol = max(1e-9, float(getattr(self, "position_size_change_tol", 0.0) or 0.0))
        target_symbol = canonicalize_execution_symbol(symbol or getattr(self, "symbol", "") or "")
        if not target_symbol:
            return base_tol
        reader = getattr(self, "reader", None)
        if reader is None or not hasattr(reader, "get_sz_decimals"):
            return base_tol
        try:
            sz_decimals = int(reader.get_sz_decimals(target_symbol) or 0)
        except Exception:
            return base_tol
        quantum = 10 ** (-sz_decimals) if sz_decimals > 0 else 1.0
        return max(base_tol, quantum / 2.0)
    @staticmethod
    def _timeframe_to_seconds(timeframe: str) -> int:
        key = str(timeframe or "").strip().lower()
        mapping = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}
        return mapping.get(key, 0)
    def _completed_candles_for_risk_session(
        self,
        rs: RiskSession,
        *,
        now: float,
        interval: Optional[str] = None,
        min_bars: int = 0,
    ) -> List[Dict[str, Any]]:
        timeframe = str(interval or getattr(rs, "trailing_timeframe", "15m") or "15m").strip().lower() or "15m"
        interval_seconds = self._timeframe_to_seconds(timeframe)
        if interval_seconds <= 0 or not hasattr(self.reader, "get_candles_snapshot"):
            return []
        entry_time = max(0.0, float(getattr(rs, "start_time", 0.0) or 0.0))
        lookback_bars = max(
            int(min_bars or 0),
            int(getattr(rs, "trailing_atr_lookback_bars", 200) or 200),
            int(getattr(rs, "trailing_atr_period", 14) or 14) + 10,
        )
        start_ms = max(
            0,
            int((entry_time - (lookback_bars * interval_seconds)) * 1000),
        )
        end_ms = max(start_ms + (interval_seconds * 1000), int(float(now or time.time()) * 1000))
        try:
            rows = list(self.reader.get_candles_snapshot(self.symbol, timeframe, start_ms, end_ms) or [])
        except Exception:
            return []
        completed: List[Dict[str, Any]] = []
        now_ms = int(float(now or time.time()) * 1000)
        for row in rows:
            try:
                open_ms = int(row.get("t", 0) or 0)
                close_ms = open_ms + (interval_seconds * 1000)
                if close_ms > now_ms:
                    continue
                completed.append(
                    {
                        "open_ms": open_ms,
                        "close_ms": close_ms,
                        "high": float(row.get("h", 0.0) or 0.0),
                        "low": float(row.get("l", 0.0) or 0.0),
                        "close": float(row.get("c", 0.0) or 0.0),
                    }
                )
            except Exception:
                continue
        return sorted(completed, key=lambda item: int(item.get("close_ms", 0) or 0))
    def _latest_completed_candles_for_risk_session(
        self,
        rs: RiskSession,
        *,
        now: float,
        interval: Optional[str] = None,
        min_bars: int = 0,
        use_trailing_lookback: bool = True,
    ) -> List[Dict[str, Any]]:
        timeframe = str(interval or getattr(rs, "trailing_timeframe", "15m") or "15m").strip().lower() or "15m"
        interval_seconds = self._timeframe_to_seconds(timeframe)
        if interval_seconds <= 0 or not hasattr(self.reader, "get_candles_snapshot"):
            return []
        if use_trailing_lookback:
            lookback_bars = max(
                int(min_bars or 0),
                int(getattr(rs, "trailing_atr_lookback_bars", 200) or 200),
                int(getattr(rs, "trailing_atr_period", 14) or 14) + 10,
            )
        else:
            lookback_bars = max(int(min_bars or 0), 2)
        end_ms = int(float(now or time.time()) * 1000)
        start_ms = max(0, end_ms - ((lookback_bars + 2) * interval_seconds * 1000))
        try:
            rows = list(self.reader.get_candles_snapshot(self.symbol, timeframe, start_ms, end_ms) or [])
        except Exception:
            return []
        now_ms = int(float(now or time.time()) * 1000)
        completed: List[Dict[str, Any]] = []
        for row in rows:
            try:
                open_ms = int(row.get("t", 0) or 0)
                close_ms = open_ms + (interval_seconds * 1000)
                if close_ms > now_ms:
                    continue
                completed.append(
                    {
                        "open_ms": open_ms,
                        "close_ms": close_ms,
                        "high": float(row.get("h", 0.0) or 0.0),
                        "low": float(row.get("l", 0.0) or 0.0),
                        "close": float(row.get("c", 0.0) or 0.0),
                    }
                )
            except Exception:
                continue
        completed = sorted(completed, key=lambda item: int(item.get("close_ms", 0) or 0))
        if len(completed) > lookback_bars:
            completed = completed[-lookback_bars:]
        return completed
    @staticmethod
    def _atr_from_completed_candles(candles: List[Dict[str, Any]], period: int) -> Optional[float]:
        usable = sorted(
            [dict(item) for item in list(candles or []) if isinstance(item, dict)],
            key=lambda item: int(item.get("close_ms", 0) or 0),
        )
        if len(usable) < max(2, int(period or 0) + 1):
            return None
        trs: List[float] = []
        prev_close: Optional[float] = None
        for candle in usable:
            high = float(candle.get("high", 0.0) or 0.0)
            low = float(candle.get("low", 0.0) or 0.0)
            close = float(candle.get("close", 0.0) or 0.0)
            if high <= 0.0 or low <= 0.0 or close <= 0.0:
                prev_close = close if close > 0.0 else prev_close
                continue
            if prev_close is None:
                tr = high - low
            else:
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(max(0.0, tr))
            prev_close = close
        period_value = max(1, int(period or 0))
        if len(trs) < period_value:
            return None
        atr = sum(trs[:period_value]) / period_value
        for tr in trs[period_value:]:
            atr = ((atr * (period_value - 1)) + tr) / period_value
        return atr
    @staticmethod
    def _completed_bar_close_ms(now_utc: datetime, timeframe: str) -> int:
        interval_seconds = RiskSessionMixin._timeframe_to_seconds(timeframe)
        if interval_seconds <= 0:
            return 0
        now_ms = int(now_utc.timestamp() * 1000)
        interval_ms = interval_seconds * 1000
        return (now_ms // interval_ms) * interval_ms
    @staticmethod
    def _normalize_completed_candle_rows(
        rows: List[Dict[str, Any]],
        *,
        interval_seconds: int,
        now_ms: int,
    ) -> List[Dict[str, Any]]:
        completed: List[Dict[str, Any]] = []
        for row in list(rows or []):
            try:
                open_ms = int(row.get("t", 0) or 0)
                close_ms = open_ms + (interval_seconds * 1000)
                if close_ms > now_ms:
                    continue
                high = float(row.get("h", 0.0) or 0.0)
                low = float(row.get("l", 0.0) or 0.0)
                close = float(row.get("c", 0.0) or 0.0)
                if high <= 0.0 or low <= 0.0 or close <= 0.0:
                    continue
                completed.append(
                    {
                        "open_ms": open_ms,
                        "close_ms": close_ms,
                        "high": high,
                        "low": low,
                        "close": close,
                    }
                )
            except Exception:
                continue
        return sorted(completed, key=lambda item: int(item.get("close_ms", 0) or 0))
    @staticmethod
    def _wilder_atr_rows(candles: List[Dict[str, Any]], period: int) -> List[Dict[str, Any]]:
        usable = sorted(
            [dict(item) for item in list(candles or []) if isinstance(item, dict)],
            key=lambda item: int(item.get("close_ms", 0) or 0),
        )
        rows: List[Dict[str, Any]] = []
        trs: List[float] = []
        prev_close: Optional[float] = None
        period_value = max(1, int(period or 0))
        for candle in usable:
            high = float(candle.get("high", 0.0) or 0.0)
            low = float(candle.get("low", 0.0) or 0.0)
            close = float(candle.get("close", 0.0) or 0.0)
            if high <= 0.0 or low <= 0.0 or close <= 0.0:
                continue
            if prev_close is None:
                tr = high - low
            else:
                tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            trs.append(max(0.0, tr))
            prev_close = close
            atr: Optional[float]
            if len(trs) < period_value:
                atr = None
            elif len(trs) == period_value:
                atr = sum(trs[:period_value]) / period_value
            else:
                prev_atr = float(rows[-1].get("atr", 0.0) or 0.0)
                atr = ((prev_atr * (period_value - 1)) + tr) / period_value
            rows.append({**candle, "tr": tr, "atr": atr})
        return rows
    def _profile_normal_liquidity_atr_ref(
        self,
        *,
        symbol: str,
        profile: InstrumentMarketProfile,
        now_utc: datetime,
    ) -> Dict[str, Any]:
        timeframe = str(profile.atr_ref_timeframe or "15m").strip().lower() or "15m"
        interval_seconds = self._timeframe_to_seconds(timeframe)
        period = max(1, int(profile.atr_ref_period or 14))
        completed_close_ms = self._completed_bar_close_ms(now_utc, timeframe)
        cache_key = (
            canonicalize_execution_symbol(symbol or ""),
            profile.name,
            timeframe,
            period,
            int(profile.atr_ref_lookback_days or 0),
            completed_close_ms,
        )
        cached = dict(getattr(self, "atr_ref_cache", {}).get(cache_key) or {})
        if cached:
            return cached
        if interval_seconds <= 0 or completed_close_ms <= 0:
            return {"available": False, "code": "invalid_timeframe"}
        start_fetch = time.perf_counter()
        end_ms = int(now_utc.timestamp() * 1000)
        lookback_bars = max(period + 10, int(profile.atr_ref_lookback_bars or 700))
        start_ms = max(0, end_ms - ((lookback_bars + 2) * interval_seconds * 1000))
        try:
            rows = list(self.reader.get_candles_snapshot(symbol, timeframe, start_ms, end_ms) or [])
        except Exception as exc:
            return {"available": False, "code": "fetch_failed", "error": str(exc)}
        fetch_seconds = time.perf_counter() - start_fetch
        candles = self._normalize_completed_candle_rows(rows, interval_seconds=interval_seconds, now_ms=end_ms)
        atr_rows = self._wilder_atr_rows(candles, period)
        zone = self._profile_timezone(profile)
        local_now = now_utc.astimezone(zone)
        sample_dates: List[Any] = []
        for row in reversed(atr_rows):
            atr = row.get("atr")
            if atr is None:
                continue
            local_close = datetime.fromtimestamp(int(row.get("close_ms", 0) or 0) / 1000, tz=timezone.utc).astimezone(zone)
            if local_close >= local_now:
                continue
            if local_close.weekday() >= 5:
                continue
            if not any(window.contains(local_close) for window in list(profile.normal_liquidity_windows or ())):
                continue
            if local_close.date() not in sample_dates:
                sample_dates.append(local_close.date())
            if len(sample_dates) >= max(1, int(profile.atr_ref_lookback_days or 5)):
                break
        sample_date_set = set(sample_dates)
        samples: List[float] = []
        for row in atr_rows:
            atr = row.get("atr")
            if atr is None:
                continue
            local_close = datetime.fromtimestamp(int(row.get("close_ms", 0) or 0) / 1000, tz=timezone.utc).astimezone(zone)
            if local_close.date() not in sample_date_set:
                continue
            if local_close.weekday() >= 5:
                continue
            if any(window.contains(local_close) for window in list(profile.normal_liquidity_windows or ())):
                samples.append(float(atr))
        if not samples:
            result = {
                "available": False,
                "code": "no_normal_liquidity_samples",
                "raw_rows": len(rows),
                "completed_candles": len(candles),
                "fetch_seconds": fetch_seconds,
            }
        else:
            result = {
                "available": True,
                "code": "ok",
                "atr_ref": float(statistics.median(samples)),
                "timeframe": timeframe,
                "period": period,
                "sample_count": len(samples),
                "sample_dates": [str(item) for item in sorted(sample_date_set)],
                "raw_rows": len(rows),
                "completed_candles": len(candles),
                "fetch_seconds": fetch_seconds,
                "completed_close_ms": completed_close_ms,
            }
        self.atr_ref_cache[cache_key] = dict(result)
        return result
    def _profile_liquidity_band_for_symbol(
        self,
        symbol: str,
        now_utc: datetime,
    ) -> Tuple[Optional[InstrumentMarketProfile], str]:
        profile = self._market_profile_for_symbol(symbol)
        if profile is None:
            return None, "no_profile"
        if self._profile_low_liquidity_weekday_contains(profile, now_utc):
            return profile, "low_liquidity"
        if self._profile_time_windows_contain(profile, profile.low_liquidity_windows, now_utc):
            return profile, "low_liquidity"
        if self._profile_time_windows_contain(profile, profile.normal_liquidity_windows, now_utc):
            return profile, "normal_liquidity"
        return profile, "outside_configured_liquidity_windows"

    @staticmethod
    def _profile_r_clip_multiples(profile: InstrumentMarketProfile, liquidity_band: str) -> Tuple[float, float]:
        if liquidity_band == "low_liquidity":
            return (
                max(0.0, float(profile.low_liquidity_r_min_atr_multiple or 0.0)),
                max(0.0, float(profile.low_liquidity_r_max_atr_multiple or 0.0)),
            )
        if liquidity_band == "normal_liquidity":
            return (
                max(0.0, float(profile.normal_liquidity_r_min_atr_multiple or 0.0)),
                max(0.0, float(profile.normal_liquidity_r_max_atr_multiple or 0.0)),
            )
        return 0.0, 0.0

    def _fallback_staged_exit_params(self) -> Dict[str, Any]:
        tp1_r_multiple = max(0.0, float(getattr(self, "risk_tp1_r_multiple", 1.0) or 0.0))
        tp2_r_multiple = max(tp1_r_multiple, float(getattr(self, "risk_tp2_r_multiple", 2.0) or 0.0))
        tp1_close_fraction = min(max(0.0, float(getattr(self, "risk_tp1_close_fraction", 0.30) or 0.0)), 1.0)
        tp2_close_fraction = min(max(0.0, float(getattr(self, "risk_tp2_close_fraction", 0.40) or 0.0)), 1.0)
        soft_atr_multiple = max(0.1, float(getattr(self, "risk_trailing_soft_atr_multiple", 2.5) or 2.5))
        hard_atr_multiple = max(soft_atr_multiple, float(getattr(self, "risk_trailing_hard_atr_multiple", 3.5) or 3.5))
        return {
            "source": "global_fallback",
            "profile_name": "",
            "liquidity_band": "no_profile",
            "tp1_r_multiple": tp1_r_multiple,
            "tp2_r_multiple": tp2_r_multiple,
            "tp1_close_fraction": tp1_close_fraction,
            "tp2_close_fraction": tp2_close_fraction,
            "post_tp1_stop_r_multiple": float(getattr(self, "risk_post_tp1_stop_r_multiple", -0.40) or -0.40),
            "post_tp2_locked_r_multiple": float(getattr(self, "risk_post_tp2_locked_r_multiple", 1.0) or 1.0),
            "trailing_soft_atr_multiple": soft_atr_multiple,
            "trailing_hard_atr_multiple": hard_atr_multiple,
        }

    def _staged_exit_params_for_symbol(
        self,
        symbol: str,
        now_utc: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        params = self._fallback_staged_exit_params()
        now = now_utc or datetime.now(timezone.utc)
        profile, liquidity_band = self._profile_liquidity_band_for_symbol(symbol, now)
        if profile is None or liquidity_band not in {"normal_liquidity", "low_liquidity"}:
            params["liquidity_band"] = liquidity_band
            return params
        return self._staged_exit_params_for_profile_band(symbol, liquidity_band)

    def _staged_exit_params_for_profile_band(
        self,
        symbol: str,
        liquidity_band: str,
    ) -> Dict[str, Any]:
        params = self._fallback_staged_exit_params()
        profile = self._market_profile_for_symbol(symbol)
        if profile is None or liquidity_band not in {"normal_liquidity", "low_liquidity"}:
            params["liquidity_band"] = liquidity_band or "no_profile"
            return params
        prefix = "low_liquidity" if liquidity_band == "low_liquidity" else "normal_liquidity"
        params.update(
            {
                "source": "instrument_market_profile",
                "profile_name": profile.name,
                "liquidity_band": liquidity_band,
                "tp1_r_multiple": max(0.0, float(getattr(profile, f"{prefix}_tp1_r_multiple"))),
                "tp2_r_multiple": max(0.0, float(getattr(profile, f"{prefix}_tp2_r_multiple"))),
                "tp1_close_fraction": min(max(0.0, float(getattr(profile, f"{prefix}_tp1_close_fraction"))), 1.0),
                "tp2_close_fraction": min(max(0.0, float(getattr(profile, f"{prefix}_tp2_close_fraction"))), 1.0),
                "post_tp1_stop_r_multiple": float(getattr(profile, f"{prefix}_post_tp1_stop_r_multiple")),
                "post_tp2_locked_r_multiple": float(getattr(profile, f"{prefix}_post_tp2_locked_r_multiple")),
                "trailing_soft_atr_multiple": max(0.1, float(getattr(profile, f"{prefix}_trailing_soft_atr_multiple"))),
                "trailing_hard_atr_multiple": max(0.1, float(getattr(profile, f"{prefix}_trailing_hard_atr_multiple"))),
            }
        )
        params["tp2_r_multiple"] = max(float(params["tp1_r_multiple"]), float(params["tp2_r_multiple"]))
        params["trailing_hard_atr_multiple"] = max(
            float(params["trailing_soft_atr_multiple"]),
            float(params["trailing_hard_atr_multiple"]),
        )
        return params

    def _maybe_clip_profile_stop_loss(
        self,
        *,
        side: str,
        entry_price: float,
        stop_price: float,
        symbol: str,
        now_utc: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        original_stop = max(0.0, float(stop_price or 0.0))
        entry = max(0.0, float(entry_price or 0.0))
        if side not in {"long", "short"} or entry <= 0.0 or original_stop <= 0.0:
            return {"applied": False, "code": "invalid_input", "stop_loss_price": original_stop}
        if side == "long" and original_stop >= entry:
            return {"applied": False, "code": "invalid_long_stop", "stop_loss_price": original_stop}
        if side == "short" and original_stop <= entry:
            return {"applied": False, "code": "invalid_short_stop", "stop_loss_price": original_stop}
        now = now_utc or datetime.now(timezone.utc)
        profile, liquidity_band = self._profile_liquidity_band_for_symbol(symbol, now)
        if profile is None:
            return {"applied": False, "code": "no_profile", "stop_loss_price": original_stop}
        if liquidity_band not in {"normal_liquidity", "low_liquidity"}:
            return {
                "applied": False,
                "code": liquidity_band,
                "profile": profile.name,
                "stop_loss_price": original_stop,
            }
        atr_ref = self._profile_normal_liquidity_atr_ref(symbol=symbol, profile=profile, now_utc=now)
        if not atr_ref.get("available"):
            return {
                "applied": False,
                "code": str(atr_ref.get("code") or "atr_ref_unavailable"),
                "stop_loss_price": original_stop,
                "atr_ref": atr_ref,
            }
        atr_value = max(0.0, float(atr_ref.get("atr_ref", 0.0) or 0.0))
        min_multiple, max_multiple = self._profile_r_clip_multiples(profile, liquidity_band)
        r_min = max(0.0, min_multiple * atr_value)
        r_max = max(0.0, max_multiple * atr_value)
        if r_max > 0.0:
            r_max = max(r_min, r_max)
        r_raw = abs(entry - original_stop)
        if r_min <= 0.0 and r_max <= 0.0:
            return {
                "applied": False,
                "code": "profile_r_clip_disabled",
                "stop_loss_price": original_stop,
                "profile": profile.name,
                "liquidity_band": liquidity_band,
                "entry_price": entry,
                "r_raw": r_raw,
                "r_min": r_min,
                "r_max": r_max,
                "atr_ref": atr_ref,
            }
        r_clipped = max(r_raw, r_min)
        if r_max > 0.0:
            r_clipped = min(r_clipped, r_max)
        if abs(r_clipped - r_raw) <= max(1.0, entry) * 1e-12:
            return {
                "applied": False,
                "code": "already_in_profile_r_range",
                "stop_loss_price": original_stop,
                "profile": profile.name,
                "liquidity_band": liquidity_band,
                "entry_price": entry,
                "r_raw": r_raw,
                "r_min": r_min,
                "r_max": r_max,
                "r_clipped": r_clipped,
                "atr_ref": atr_ref,
            }
        clip_reason = "r_min_applied" if r_clipped > r_raw else "r_max_applied"
        corrected = entry - r_clipped if side == "long" else entry + r_clipped
        corrected = self._align_price_for_symbol(symbol, corrected)
        if side == "long" and corrected >= entry:
            return {"applied": False, "code": "corrected_long_stop_invalid", "stop_loss_price": original_stop, "atr_ref": atr_ref}
        if side == "short" and corrected <= entry:
            return {"applied": False, "code": "corrected_short_stop_invalid", "stop_loss_price": original_stop, "atr_ref": atr_ref}
        if abs(corrected - original_stop) <= max(1.0, entry) * 1e-12:
            return {
                "applied": False,
                "code": "aligned_stop_unchanged",
                "stop_loss_price": original_stop,
                "profile": profile.name,
                "liquidity_band": liquidity_band,
                "entry_price": entry,
                "r_raw": r_raw,
                "r_min": r_min,
                "r_max": r_max,
                "r_clipped": r_clipped,
                "atr_ref": atr_ref,
            }
        return {
            "applied": True,
            "code": f"profile_{liquidity_band}_{clip_reason}",
            "profile": profile.name,
            "symbol": canonicalize_execution_symbol(symbol or ""),
            "liquidity_band": liquidity_band,
            "side": side,
            "entry_price": entry,
            "original_stop_loss_price": original_stop,
            "stop_loss_price": corrected,
            "r_raw": r_raw,
            "r_min": r_min,
            "r_max": r_max,
            "r_clipped": r_clipped,
            "r_min_atr_multiple": min_multiple,
            "r_max_atr_multiple": max_multiple,
            "atr_ref": atr_ref,
        }

    @staticmethod
    def _staged_tp_leg_fraction(rs: RiskSession, leg_name: str) -> float:
        target_name = str(leg_name or "")
        for leg in list(getattr(rs, "take_profit_legs", []) or []):
            if str(getattr(leg, "name", "") or "") == target_name:
                return min(max(0.0, float(getattr(leg, "close_fraction", 0.0) or 0.0)), 1.0)
        if target_name == "stage_tp1":
            return 0.30
        if target_name == "stage_tp2":
            return 0.40
        return 0.0
    def _align_risk_exit_size_to_precision(self, size: float, symbol: str = "") -> float:
        raw_size = max(0.0, float(size or 0.0))
        if raw_size <= 0.0:
            return 0.0
        target_symbol = canonicalize_execution_symbol(symbol or getattr(self, "symbol", "") or "")
        reader = getattr(self, "reader", None)
        if reader is None or not hasattr(reader, "get_sz_decimals"):
            return raw_size
        try:
            decimals = max(0, int(reader.get_sz_decimals(target_symbol) or 0))
            quantum = Decimal(1).scaleb(-decimals)
            raw_decimal = Decimal(str(raw_size))
        except (InvalidOperation, ValueError, TypeError):
            return raw_size
        if not raw_decimal.is_finite() or raw_decimal <= 0 or quantum <= 0:
            return 0.0
        epsilon = quantum * Decimal("1e-9")
        return float((raw_decimal + epsilon).quantize(quantum, rounding=ROUND_DOWN))

    @staticmethod
    def _decimal_size_delta_abs(before_abs: float, after_abs: float) -> float:
        try:
            before_decimal = Decimal(str(max(0.0, float(before_abs or 0.0))))
            after_decimal = Decimal(str(max(0.0, float(after_abs or 0.0))))
            delta = before_decimal - after_decimal
        except (InvalidOperation, ValueError, TypeError):
            return max(0.0, float(before_abs or 0.0) - float(after_abs or 0.0))
        if not delta.is_finite() or delta <= 0:
            return 0.0
        return float(delta)

    @staticmethod
    def _clamp_accounting_size_abs(size: float, *, max_size: Optional[float] = None) -> float:
        try:
            size_decimal = Decimal(str(max(0.0, float(size or 0.0))))
            if max_size is not None:
                max_decimal = Decimal(str(max(0.0, float(max_size or 0.0))))
                size_decimal = min(size_decimal, max_decimal)
        except (InvalidOperation, ValueError, TypeError):
            raw_size = max(0.0, float(size or 0.0))
            if max_size is not None:
                raw_size = min(raw_size, max(0.0, float(max_size or 0.0)))
            return raw_size
        if not size_decimal.is_finite() or size_decimal <= 0:
            return 0.0
        return float(size_decimal)
    def _staged_tp_target_size_abs(self, rs: RiskSession, leg_name: str) -> float:
        basis = abs(float(getattr(rs, "staged_exit_size_basis_abs", 0.0) or getattr(rs, "initial_size_abs", 0.0) or 0.0))
        fraction = self._staged_tp_leg_fraction(rs, leg_name)
        return self._align_risk_exit_size_to_precision(max(0.0, basis * fraction), str(getattr(rs, "symbol", "") or ""))

    def _align_risk_close_size_for_session(self, rs: RiskSession, size: float, *, max_size: Optional[float] = None) -> float:
        raw_size = max(0.0, float(size or 0.0))
        if max_size is not None:
            raw_size = min(max(0.0, float(max_size or 0.0)), raw_size)
        return self._align_risk_exit_size_to_precision(raw_size, str(getattr(rs, "symbol", "") or ""))

    def _aligned_risk_completed_size_for_session(self, rs: RiskSession, leg_name: str) -> float:
        attr_name = "tp1_completed_size_abs" if str(leg_name or "") == "stage_tp1" else "tp2_completed_size_abs"
        raw_completed = max(0.0, float(getattr(rs, attr_name, 0.0) or 0.0))
        target_size = self._staged_tp_target_size_abs(rs, leg_name)
        max_size = target_size if target_size > 0.0 else None
        return self._align_risk_close_size_for_session(rs, raw_completed, max_size=max_size)

    def _normalized_risk_session_order_ref(self, rs: RiskSession, ref: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(ref) if isinstance(ref, dict) else {}
        if "close_size" in item:
            item["close_size"] = self._align_risk_close_size_for_session(
                rs,
                abs(float(safe_float(item.get("close_size"), 0.0) or 0.0)),
            )
        if "filled_size" in item:
            max_size = float(item.get("close_size", 0.0) or 0.0) if "close_size" in item else None
            item["filled_size"] = self._align_risk_close_size_for_session(
                rs,
                abs(float(safe_float(item.get("filled_size"), 0.0) or 0.0)),
                max_size=max_size,
            )
        return item

    def _normalized_risk_session_order_refs(self, rs: RiskSession, refs: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        source_refs = list(getattr(rs, "resting_exit_orders", []) or []) if refs is None else list(refs or [])
        return [
            self._normalized_risk_session_order_ref(rs, item)
            for item in source_refs
            if isinstance(item, dict)
        ]

    def _normalize_risk_session_size_state(self, rs: RiskSession) -> None:
        rs.resting_exit_orders = self._normalized_risk_session_order_refs(rs)
        if bool(getattr(rs, "staged_exit_enabled", False)):
            rs.tp1_completed_size_abs = self._aligned_risk_completed_size_for_session(rs, "stage_tp1")
            rs.tp2_completed_size_abs = self._aligned_risk_completed_size_for_session(rs, "stage_tp2")

    def _staged_tp_completed_size_abs(self, rs: RiskSession, leg_name: str) -> float:
        if str(leg_name or "") == "stage_tp1":
            completed = max(0.0, float(getattr(rs, "tp1_completed_size_abs", 0.0) or 0.0))
            return max(completed, self._staged_tp_target_size_abs(rs, "stage_tp1") if bool(getattr(rs, "tp1_hit", False)) else 0.0)
        if str(leg_name or "") == "stage_tp2":
            completed = max(0.0, float(getattr(rs, "tp2_completed_size_abs", 0.0) or 0.0))
            return max(completed, self._staged_tp_target_size_abs(rs, "stage_tp2") if bool(getattr(rs, "tp2_hit", False)) else 0.0)
        return 0.0
    def _risk_session_tp1_completed(self, rs: RiskSession) -> bool:
        if bool(getattr(rs, "tp1_hit", False)):
            return True
        if "take_profit::stage_tp1" in set(getattr(rs, "executed_leg_names", set()) or set()):
            return True
        target_size = self._staged_tp_target_size_abs(rs, "stage_tp1")
        qty_tol = self._risk_session_order_qty_tolerance()
        if target_size <= qty_tol:
            return True
        completed_size = max(0.0, float(getattr(rs, "tp1_completed_size_abs", 0.0) or 0.0))
        return completed_size >= max(0.0, target_size - qty_tol)
    def _risk_session_tp2_completed(self, rs: RiskSession) -> bool:
        if bool(getattr(rs, "tp2_hit", False)):
            return True
        if "take_profit::stage_tp2" in set(getattr(rs, "executed_leg_names", set()) or set()):
            return True
        target_size = self._staged_tp_target_size_abs(rs, "stage_tp2")
        qty_tol = self._risk_session_order_qty_tolerance()
        if target_size <= qty_tol:
            return True
        completed_size = max(0.0, float(getattr(rs, "tp2_completed_size_abs", 0.0) or 0.0))
        return completed_size >= max(0.0, target_size - qty_tol)
    def _update_staged_risk_session_after_completed_keys(self, rs: RiskSession, completed_keys: List[str], *, now: float) -> bool:
        if not bool(getattr(rs, "staged_exit_enabled", False)):
            return False
        completed = {str(item or "") for item in list(completed_keys or []) if str(item or "")}
        changed = False
        if "take_profit::stage_tp1" in completed and not bool(getattr(rs, "tp1_hit", False)):
            rs.tp1_hit = True
            if float(getattr(rs, "tp1_hit_at", 0.0) or 0.0) <= 0.0:
                rs.tp1_hit_at = float(now or time.time())
            rs.tp1_completed_size_abs = max(
                float(getattr(rs, "tp1_completed_size_abs", 0.0) or 0.0),
                self._staged_tp_target_size_abs(rs, "stage_tp1"),
            )
            rs.take_profit_legs = [
                leg
                for leg in list(getattr(rs, "take_profit_legs", []) or [])
                if str(getattr(leg, "name", "") or "") != "stage_tp1"
            ]
            if float(getattr(rs, "post_tp1_stop_price", 0.0) or 0.0) > 0.0:
                self._set_stage_soft_hard_stop(
                    rs,
                    float(rs.post_tp1_stop_price),
                    name="stage_post_tp1_stop",
                    note="staged_post_tp1_soft_stop",
                )
            changed = True
        if "take_profit::stage_tp2" in completed and not bool(getattr(rs, "tp2_hit", False)):
            rs.tp2_hit = True
            if float(getattr(rs, "tp2_hit_at", 0.0) or 0.0) <= 0.0:
                rs.tp2_hit_at = float(now or time.time())
            rs.tp2_completed_size_abs = max(
                float(getattr(rs, "tp2_completed_size_abs", 0.0) or 0.0),
                self._staged_tp_target_size_abs(rs, "stage_tp2"),
            )
            rs.take_profit_legs = []
            self._update_staged_risk_session_trailing_state(rs, now=now)
            soft_stop_price = max(0.0, float(getattr(rs, "trailing_soft_stop_price", 0.0) or 0.0))
            if soft_stop_price <= 0.0:
                soft_stop_price = max(0.0, float(getattr(rs, "locked_floor_price", 0.0) or 0.0))
            rs.stop_loss_price = soft_stop_price
            rs.stop_loss_legs = []
            rs.active_soft_stop_price = 0.0
            rs.active_hard_stop_price = 0.0
            rs.soft_stop_breach_since = 0.0
            rs.soft_stop_last_breach_price = 0.0
            rs.exchange_hard_stop_buffer_usd = 0.0
            rs.exchange_hard_stop_min_buffer_usd = 0.0
            rs.exchange_hard_stop_atr_buffer_usd = 0.0
            rs.exchange_hard_stop_r_buffer_usd = 0.0
            rs.exchange_hard_stop_atr_value = 0.0
            changed = True
        return changed

    @staticmethod
    def _risk_session_profit_r(rs: RiskSession, price: float) -> Optional[float]:
        side = str(getattr(rs, "side", "") or "")
        entry = max(0.0, float(getattr(rs, "initial_entry_price", 0.0) or 0.0))
        risk_distance = max(0.0, float(getattr(rs, "initial_risk_price_distance", 0.0) or 0.0))
        mark = max(0.0, float(price or 0.0))
        if side not in {"long", "short"} or entry <= 0.0 or risk_distance <= 0.0 or mark <= 0.0:
            return None
        if side == "long":
            return (mark - entry) / risk_distance
        return (entry - mark) / risk_distance

    def _risk_session_locked_floor_r(self, rs: RiskSession) -> Optional[float]:
        side = str(getattr(rs, "side", "") or "")
        entry = max(0.0, float(getattr(rs, "initial_entry_price", 0.0) or 0.0))
        risk_distance = max(0.0, float(getattr(rs, "initial_risk_price_distance", 0.0) or 0.0))
        locked_price = max(0.0, float(getattr(rs, "locked_floor_price", 0.0) or 0.0))
        if side not in {"long", "short"} or entry <= 0.0 or risk_distance <= 0.0 or locked_price <= 0.0:
            return None
        if side == "long":
            return max(0.0, (locked_price - entry) / risk_distance)
        return max(0.0, (entry - locked_price) / risk_distance)

    def _adjust_time_decay_tp2_tail_locked_floor(
        self,
        rs: RiskSession,
        candidate: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        current_r = safe_float(candidate.get("current_profit_r"), None)
        profile_locked_r = self._risk_session_locked_floor_r(rs)
        if current_r is None or profile_locked_r is None:
            return None
        buffer_r = max(0.0, float(getattr(self, "risk_time_decay_tp2_tail_lock_buffer_r", 0.15) or 0.0))
        adjusted_locked_r = max(0.0, min(float(profile_locked_r), float(current_r) - buffer_r))
        old_locked_price = max(0.0, float(getattr(rs, "locked_floor_price", 0.0) or 0.0))
        new_locked_price = self._price_from_r_multiple(
            str(getattr(rs, "side", "") or ""),
            max(0.0, float(getattr(rs, "initial_entry_price", 0.0) or 0.0)),
            max(0.0, float(getattr(rs, "initial_risk_price_distance", 0.0) or 0.0)),
            adjusted_locked_r,
        )
        if new_locked_price <= 0.0:
            return None
        rs.locked_floor_price = new_locked_price
        rs.trailing_soft_stop_price = 0.0
        rs.trailing_hard_stop_price = 0.0
        rs.trailing_highest_close = 0.0
        rs.trailing_lowest_close = 0.0
        rs.trailing_last_bar_ms = 0
        rs.trailing_last_close_price = 0.0
        payload = {
            "plan_name": str(getattr(rs, "plan_name", "") or ""),
            "side": str(getattr(rs, "side", "") or ""),
            "current_profit_r": float(current_r),
            "buffer_r": buffer_r,
            "profile_locked_r": float(profile_locked_r),
            "adjusted_locked_r": adjusted_locked_r,
            "old_locked_floor_price": old_locked_price,
            "new_locked_floor_price": new_locked_price,
            "leg_name": str(candidate.get("leg_name", "") or ""),
            "liquidity_band": str(candidate.get("liquidity_band", "") or getattr(rs, "staged_exit_liquidity_band", "") or ""),
        }
        self._audit_event("risk_session_time_decay_tp2_tail_lock_adjusted", payload)
        return payload

    def _update_risk_session_mfe(self, rs: RiskSession, price: float) -> Optional[float]:
        current_r = self._risk_session_profit_r(rs, price)
        if current_r is None:
            return None
        prior_mfe = max(0.0, float(getattr(rs, "max_favorable_excursion_r", 0.0) or 0.0))
        rs.max_favorable_excursion_r = max(prior_mfe, current_r)
        return current_r

    def _risk_time_decay_tp_params_for_session(self, rs: RiskSession) -> Dict[str, float]:
        band = str(getattr(rs, "staged_exit_liquidity_band", "") or "").strip().lower()
        prefix = "low" if band == "low_liquidity" else "normal"
        return {
            "tp1_bars": max(1.0, float(getattr(self, f"risk_time_decay_{prefix}_tp1_bars", 4.0 if prefix == "low" else 8.0) or 1.0)),
            "tp1_mfe_r": max(0.0, float(getattr(self, f"risk_time_decay_{prefix}_tp1_mfe_r", 0.50 if prefix == "low" else 0.60) or 0.0)),
            "tp1_current_r": max(0.0, float(getattr(self, f"risk_time_decay_{prefix}_tp1_current_r", 0.25 if prefix == "low" else 0.30) or 0.0)),
            "tp2_bars": max(1.0, float(getattr(self, f"risk_time_decay_{prefix}_tp2_bars", 18.0 if prefix == "low" else 6.0) or 1.0)),
            "tp2_mfe_r": max(0.0, float(getattr(self, f"risk_time_decay_{prefix}_tp2_mfe_r", 1.00 if prefix == "low" else 1.50) or 0.0)),
            "tp2_current_r": max(0.0, float(getattr(self, f"risk_time_decay_{prefix}_tp2_current_r", 0.75 if prefix == "low" else 1.00) or 0.0)),
        }

    def _risk_session_time_decay_tp_candidate(
        self,
        rs: RiskSession,
        snapshot: dict,
        *,
        price: float,
        now: float,
    ) -> Optional[Dict[str, Any]]:
        if not bool(getattr(self, "risk_time_decay_tp_enabled", False)):
            return None
        if not bool(getattr(rs, "staged_exit_enabled", False)):
            return None
        if not snapshot_has_open_position(snapshot):
            return None
        side = str(getattr(rs, "side", "") or "")
        snapshot_side = str(snapshot.get("side", "") or "")
        if side not in {"long", "short"} or snapshot_side != side:
            return None
        current_r = self._update_risk_session_mfe(rs, price)
        if current_r is None:
            return None
        mfe_r = max(0.0, float(getattr(rs, "max_favorable_excursion_r", 0.0) or 0.0))
        params = self._risk_time_decay_tp_params_for_session(rs)
        timeframe_seconds = max(1.0, float(getattr(self, "risk_time_decay_tp_timeframe_seconds", 300.0) or 300.0))
        qty_tol = self._risk_session_order_qty_tolerance()
        position_abs = abs(float(snapshot.get("size", 0.0) or 0.0))
        if position_abs <= qty_tol:
            return None

        leg_name = ""
        leg_key = ""
        anchor_source = ""
        elapsed_seconds = 0.0
        required_seconds = 0.0
        required_mfe_r = 0.0
        required_current_r = 0.0
        if not self._risk_session_tp1_completed(rs):
            leg_name = "stage_tp1"
            leg_key = "take_profit::stage_tp1"
            anchor_source = "entry_start"
            elapsed_seconds = max(0.0, float(now or 0.0) - float(getattr(rs, "start_time", 0.0) or 0.0))
            required_seconds = float(params["tp1_bars"]) * timeframe_seconds
            required_mfe_r = float(params["tp1_mfe_r"])
            required_current_r = float(params["tp1_current_r"])
        elif not self._risk_session_tp2_completed(rs):
            tp1_no_follow_applied = bool(getattr(rs, "tp1_no_follow_through_applied", False))
            tp1_no_follow_at = float(getattr(rs, "tp1_no_follow_through_at", 0.0) or 0.0)
            tp1_hit_at = float(getattr(rs, "tp1_hit_at", 0.0) or 0.0)
            if tp1_no_follow_applied and tp1_no_follow_at > 0.0:
                anchor = tp1_no_follow_at
                anchor_source = "tp1_no_follow_through"
            else:
                anchor = tp1_hit_at
                anchor_source = "tp1_hit"
            if anchor <= 0.0:
                return None
            leg_name = "stage_tp2"
            leg_key = "take_profit::stage_tp2"
            elapsed_seconds = max(0.0, float(now or 0.0) - anchor)
            required_seconds = float(params["tp2_bars"]) * timeframe_seconds
            required_mfe_r = float(params["tp2_mfe_r"])
            required_current_r = float(params["tp2_current_r"])
        else:
            return None

        if leg_key in set(getattr(rs, "executed_leg_names", set()) or set()):
            return None
        trigger_mode = "elapsed_full_threshold"
        if leg_name == "stage_tp1":
            current_threshold_met = current_r >= required_current_r
            mfe_threshold_met = mfe_r >= required_mfe_r
            elapsed_threshold_met = elapsed_seconds >= required_seconds
            tp1_no_follow_applied = bool(getattr(rs, "tp1_no_follow_through_applied", False))
            if not mfe_threshold_met:
                return None
            if not elapsed_threshold_met and not current_threshold_met:
                return None
            if elapsed_threshold_met and tp1_no_follow_applied and not current_threshold_met:
                return None
            if not elapsed_threshold_met:
                trigger_mode = "early_profit_threshold"
            elif not current_threshold_met:
                trigger_mode = "elapsed_mfe_threshold"
        else:
            current_threshold_met = current_r >= required_current_r
            mfe_threshold_met = mfe_r >= required_mfe_r
            elapsed_threshold_met = elapsed_seconds >= required_seconds
            tp2_no_continuation_applied = bool(getattr(rs, "tp2_no_continuation_applied", False))
            if not mfe_threshold_met:
                return None
            if not elapsed_threshold_met and not current_threshold_met:
                return None
            if elapsed_threshold_met and tp2_no_continuation_applied and not current_threshold_met:
                return None
            if not elapsed_threshold_met:
                trigger_mode = "early_profit_threshold"
            elif not current_threshold_met:
                trigger_mode = "elapsed_mfe_threshold"
        target_size = self._staged_tp_target_size_abs(rs, leg_name)
        completed_size = self._staged_tp_completed_size_abs(rs, leg_name)
        close_size = min(position_abs, max(0.0, target_size - completed_size))
        if close_size <= qty_tol:
            return None
        return {
            "leg_name": leg_name,
            "leg_key": leg_key,
            "close_size": close_size,
            "current_profit_r": current_r,
            "max_favorable_excursion_r": mfe_r,
            "elapsed_seconds": elapsed_seconds,
            "time_decay_anchor_source": anchor_source,
            "tp1_no_follow_through_at": float(getattr(rs, "tp1_no_follow_through_at", 0.0) or 0.0),
            "tp1_hit_at": float(getattr(rs, "tp1_hit_at", 0.0) or 0.0),
            "required_seconds": required_seconds,
            "required_mfe_r": required_mfe_r,
            "required_current_r": required_current_r,
            "time_decay_trigger_mode": trigger_mode,
            "liquidity_band": str(getattr(rs, "staged_exit_liquidity_band", "") or ""),
            "price": float(price or 0.0),
        }

    def _risk_session_basis_profit_lock_candidate(
        self,
        rs: RiskSession,
        snapshot: dict,
        *,
        price: float,
        now: float,
    ) -> Optional[Dict[str, Any]]:
        if not bool(getattr(self, "risk_basis_profit_lock_enabled", False)):
            return None
        if not bool(getattr(rs, "staged_exit_enabled", False)):
            return None
        if not snapshot_has_open_position(snapshot):
            self._clear_risk_session_basis_profit_observation(rs, reason="position_not_open", now=now)
            return None
        side = str(getattr(rs, "side", "") or "")
        snapshot_side = str(snapshot.get("side", "") or "")
        if side not in {"long", "short"} or snapshot_side != side:
            self._clear_risk_session_basis_profit_observation(rs, reason="side_mismatch", now=now)
            return None
        current_r = self._update_risk_session_mfe(rs, price)
        if current_r is None or current_r <= 0.0:
            self._clear_risk_session_basis_profit_observation(rs, reason="not_profitable", current_profit_r=current_r, now=now)
            return None
        basis_context = self._market_basis_context_for_side(
            symbol=str(getattr(self, "symbol", "") or getattr(rs, "symbol", "") or ""),
            side=side,
            snapshot_mid_price=price,
        )
        if not bool(basis_context.get("available")):
            return None
        current_basis = float(basis_context.get("favorable_basis", 0.0) or 0.0)
        min_basis = max(0.0, float(getattr(self, "risk_basis_profit_lock_min_basis_usd", 1.0) or 1.0))
        observe_threshold = max(min_basis, float(getattr(self, "risk_basis_profit_lock_observe_threshold_usd", 1.5) or 1.5))
        if current_basis < min_basis:
            self._clear_risk_session_basis_profit_observation(
                rs,
                reason="basis_below_min",
                basis_context=basis_context,
                current_profit_r=current_r,
                now=now,
            )
            return None

        history = getattr(rs, "basis_profit_history", None)
        if history is None:
            history = deque()
            rs.basis_profit_history = history
        history.append((float(now or time.time()), current_basis))
        keep_seconds = max(
            float(getattr(self, "risk_basis_profit_lock_slope_lookback_seconds", 180.0) or 180.0) + 120.0,
            float(getattr(self, "risk_basis_profit_lock_min_observation_seconds", 300.0) or 300.0) + 60.0,
        )
        while history and float(history[0][0]) < float(now or time.time()) - keep_seconds:
            history.popleft()

        if not bool(getattr(rs, "basis_profit_observation_active", False)):
            if current_basis < observe_threshold:
                return None
            rs.basis_profit_observation_active = True
            rs.basis_profit_observation_started_at = float(now or time.time())
            rs.basis_profit_observation_basis_start = current_basis
            payload = {
                "plan_name": str(getattr(rs, "plan_name", "") or ""),
                "side": side,
                "current_profit_r": current_r,
                "observe_threshold_usd": observe_threshold,
                "min_basis_usd": min_basis,
                "basis_context": dict(basis_context),
                "at": float(now or time.time()),
            }
            self._audit_event("risk_session_basis_profit_observation_started", payload)
            self._persist_risk_session_state()
            return None

        started_at = float(getattr(rs, "basis_profit_observation_started_at", 0.0) or 0.0)
        elapsed = max(0.0, float(now or time.time()) - started_at)
        required_elapsed = max(1.0, float(getattr(self, "risk_basis_profit_lock_min_observation_seconds", 300.0) or 300.0))
        if elapsed < required_elapsed:
            return None

        lookback_seconds = max(30.0, float(getattr(self, "risk_basis_profit_lock_slope_lookback_seconds", 180.0) or 180.0))
        slope_debug = self._basis_history_slope_usd_per_min(list(history), now=float(now or time.time()), lookback_seconds=lookback_seconds)
        if slope_debug is None:
            return None
        trigger_slope = float(getattr(self, "risk_basis_profit_lock_trigger_slope_usd_per_min", -0.01) or -0.01)
        if float(slope_debug.get("slope_usd_per_min", 0.0) or 0.0) <= trigger_slope:
            return None

        position_abs = abs(float(snapshot.get("size", 0.0) or 0.0))
        qty_tol = self._risk_session_order_qty_tolerance()
        if position_abs <= qty_tol:
            return None
        leg_name = ""
        leg_key = ""
        close_size = 0.0
        action = "tighten_tail_stop"
        if not self._risk_session_tp1_completed(rs):
            leg_name = "stage_tp1"
            leg_key = "take_profit::stage_tp1"
        elif not self._risk_session_tp2_completed(rs):
            leg_name = "stage_tp2"
            leg_key = "take_profit::stage_tp2"
        if leg_name:
            target_size = self._staged_tp_target_size_abs(rs, leg_name)
            completed_size = self._staged_tp_completed_size_abs(rs, leg_name)
            close_size = min(position_abs, max(0.0, target_size - completed_size))
            if close_size <= qty_tol:
                return None
            action = "take_profit_like_reduce"
        target_stop = 0.0
        if not leg_name:
            entry = max(0.0, float(getattr(rs, "initial_entry_price", 0.0) or 0.0))
            risk_distance = max(0.0, float(getattr(rs, "initial_risk_price_distance", 0.0) or 0.0))
            buffer_r = max(0.0, float(getattr(self, "risk_basis_profit_lock_tail_buffer_r", 0.15) or 0.15))
            if entry <= 0.0 or risk_distance <= 0.0:
                return None
            raw_stop = float(price or 0.0) - (buffer_r * risk_distance) if side == "long" else float(price or 0.0) + (buffer_r * risk_distance)
            raw_stop = self._align_price_for_symbol(self.symbol, raw_stop)
            current_stop = max(
                0.0,
                float(getattr(rs, "trailing_soft_stop_price", 0.0) or 0.0)
                or float(getattr(rs, "stop_loss_price", 0.0) or 0.0)
                or float(getattr(rs, "locked_floor_price", 0.0) or 0.0),
            )
            if side == "long":
                target_stop = max(current_stop, raw_stop) if current_stop > 0.0 else raw_stop
                if target_stop <= current_stop + 1e-12:
                    return None
            else:
                target_stop = min(current_stop, raw_stop) if current_stop > 0.0 else raw_stop
                if current_stop > 0.0 and target_stop >= current_stop - 1e-12:
                    return None

        return {
            "action": action,
            "leg_name": leg_name,
            "leg_key": leg_key,
            "close_size": close_size,
            "current_profit_r": current_r,
            "max_favorable_excursion_r": max(0.0, float(getattr(rs, "max_favorable_excursion_r", 0.0) or 0.0)),
            "price": float(price or 0.0),
            "position_size_before": position_abs,
            "basis_context": dict(basis_context),
            "basis_observation_elapsed_seconds": elapsed,
            "basis_observation_required_seconds": required_elapsed,
            "basis_observation_started_at": started_at,
            "basis_observation_basis_start": float(getattr(rs, "basis_profit_observation_basis_start", 0.0) or 0.0),
            "basis_min_required_usd": min_basis,
            "basis_observe_threshold_usd": observe_threshold,
            "basis_slope": slope_debug,
            "trigger_slope_usd_per_min": trigger_slope,
            "target_trailing_soft_stop_price": target_stop,
            "tail_buffer_r": max(0.0, float(getattr(self, "risk_basis_profit_lock_tail_buffer_r", 0.15) or 0.15)),
            "liquidity_band": str(getattr(rs, "staged_exit_liquidity_band", "") or ""),
        }

    def _risk_session_tp1_no_follow_through_candidate(
        self,
        rs: RiskSession,
        snapshot: dict,
        *,
        price: float,
        now: float,
    ) -> Optional[Dict[str, Any]]:
        if not bool(getattr(self, "risk_tp1_no_follow_through_enabled", True)):
            return None
        if not bool(getattr(rs, "staged_exit_enabled", False)):
            return None
        if bool(getattr(rs, "tp1_no_follow_through_applied", False)):
            return None
        if self._risk_session_tp1_completed(rs):
            return None
        if not snapshot_has_open_position(snapshot):
            return None
        side = str(getattr(rs, "side", "") or "")
        snapshot_side = str(snapshot.get("side", "") or "")
        if side not in {"long", "short"} or snapshot_side != side:
            return None

        current_r = self._update_risk_session_mfe(rs, price)
        if current_r is None:
            return None
        mfe_r = max(0.0, float(getattr(rs, "max_favorable_excursion_r", 0.0) or 0.0))
        params = self._risk_time_decay_tp_params_for_session(rs)
        timeframe_seconds = max(1.0, float(getattr(self, "risk_time_decay_tp_timeframe_seconds", 300.0) or 300.0))
        elapsed_seconds = max(0.0, float(now or 0.0) - float(getattr(rs, "start_time", 0.0) or 0.0))
        required_seconds = float(params["tp1_bars"]) * timeframe_seconds
        required_mfe_r = float(params["tp1_mfe_r"])
        if elapsed_seconds < required_seconds or mfe_r >= required_mfe_r:
            return None

        qty_tol = self._risk_session_order_qty_tolerance()
        position_abs = abs(float(snapshot.get("size", 0.0) or 0.0))
        if position_abs <= qty_tol:
            return None

        action = "trim"
        close_fraction = min(
            1.0,
            max(0.0, float(getattr(self, "risk_tp1_no_follow_through_normal_close_fraction", 0.50) or 0.0)),
        )
        close_size = self._align_risk_close_size_for_session(rs, position_abs * close_fraction, max_size=position_abs)
        entry = max(0.0, float(getattr(rs, "initial_entry_price", 0.0) or 0.0))
        risk_distance = max(0.0, float(getattr(rs, "initial_risk_price_distance", 0.0) or 0.0))
        soft_stop_r = max(0.0, float(getattr(self, "risk_tp1_no_follow_through_normal_soft_stop_r", 0.40) or 0.0))
        if entry > 0.0 and risk_distance > 0.0 and soft_stop_r > 0.0:
            raw_stop = entry - (soft_stop_r * risk_distance) if side == "long" else entry + (soft_stop_r * risk_distance)
            tightened_stop_price = self._align_price_for_symbol(self.symbol, raw_stop)
        else:
            tightened_stop_price = 0.0
        if close_size <= qty_tol:
            return None

        target_size = self._staged_tp_target_size_abs(rs, "stage_tp1")
        completed_size = self._staged_tp_completed_size_abs(rs, "stage_tp1")
        return {
            "action": action,
            "close_size": close_size,
            "close_fraction": close_fraction,
            "current_profit_r": current_r,
            "max_favorable_excursion_r": mfe_r,
            "elapsed_seconds": elapsed_seconds,
            "required_seconds": required_seconds,
            "required_mfe_r": required_mfe_r,
            "required_current_r": float(params["tp1_current_r"]),
            "liquidity_band": str(getattr(rs, "staged_exit_liquidity_band", "") or ""),
            "price": float(price or 0.0),
            "tp1_completed": False,
            "tp1_target_size_abs": target_size,
            "tp1_completed_size_abs": completed_size,
            "position_size_before": position_abs,
            "initial_entry_price": float(getattr(rs, "initial_entry_price", 0.0) or 0.0),
            "initial_sl_price": float(getattr(rs, "initial_stop_price", 0.0) or 0.0),
            "initial_risk_price_distance": float(getattr(rs, "initial_risk_price_distance", 0.0) or 0.0),
            "tightened_strategy_stop_price": tightened_stop_price,
        }

    def _risk_session_tp2_no_continuation_candidate(
        self,
        rs: RiskSession,
        snapshot: dict,
        *,
        price: float,
        now: float,
    ) -> Optional[Dict[str, Any]]:
        if not bool(getattr(self, "risk_tp2_no_continuation_enabled", True)):
            return None
        if not bool(getattr(rs, "staged_exit_enabled", False)):
            return None
        if bool(getattr(rs, "tp2_no_continuation_applied", False)):
            return None
        tp1_completed = self._risk_session_tp1_completed(rs)
        tp1_no_follow_applied = bool(getattr(rs, "tp1_no_follow_through_applied", False))
        if not tp1_completed and not tp1_no_follow_applied:
            return None
        if self._risk_session_tp2_completed(rs):
            return None
        if not snapshot_has_open_position(snapshot):
            return None
        side = str(getattr(rs, "side", "") or "")
        snapshot_side = str(snapshot.get("side", "") or "")
        if side not in {"long", "short"} or snapshot_side != side:
            return None

        tp1_no_follow_at = float(getattr(rs, "tp1_no_follow_through_at", 0.0) or 0.0)
        tp1_hit_at = float(getattr(rs, "tp1_hit_at", 0.0) or 0.0)
        if tp1_no_follow_applied and tp1_no_follow_at > 0.0:
            anchor = tp1_no_follow_at
            anchor_source = "tp1_no_follow_through"
        else:
            anchor = tp1_hit_at
            anchor_source = "tp1_hit"
        if anchor <= 0.0:
            return None
        current_r = self._update_risk_session_mfe(rs, price)
        if current_r is None:
            return None
        mfe_r = max(0.0, float(getattr(rs, "max_favorable_excursion_r", 0.0) or 0.0))
        params = self._risk_time_decay_tp_params_for_session(rs)
        timeframe_seconds = max(1.0, float(getattr(self, "risk_time_decay_tp_timeframe_seconds", 300.0) or 300.0))
        elapsed_seconds = max(0.0, float(now or 0.0) - anchor)
        required_seconds = float(params["tp2_bars"]) * timeframe_seconds
        required_mfe_r = float(params["tp2_mfe_r"])
        if elapsed_seconds < required_seconds or mfe_r >= required_mfe_r:
            return None

        qty_tol = self._risk_session_order_qty_tolerance()
        position_abs = abs(float(snapshot.get("size", 0.0) or 0.0))
        if position_abs <= qty_tol:
            return None

        action = "flatten"
        close_fraction = 1.0
        close_size = position_abs
        tightened_stop_price = 0.0
        stop_already_crossed = False
        entry = max(0.0, float(getattr(rs, "initial_entry_price", 0.0) or 0.0))
        risk_distance = max(0.0, float(getattr(rs, "initial_risk_price_distance", 0.0) or 0.0))
        soft_stop_r = max(0.0, float(getattr(self, "risk_tp2_no_continuation_normal_soft_stop_r", 0.25) or 0.0))
        if entry > 0.0 and risk_distance > 0.0 and soft_stop_r > 0.0:
            target_stop = entry + (soft_stop_r * risk_distance) if side == "long" else entry - (soft_stop_r * risk_distance)
            target_stop = self._align_price_for_symbol(self.symbol, target_stop)
            current_soft = max(0.0, float(self._risk_session_active_soft_stop_price(rs) or 0.0))
            if side == "long":
                tightened_stop_price = max(current_soft, target_stop) if current_soft > 0.0 else target_stop
                stop_already_crossed = float(price or 0.0) <= tightened_stop_price
            else:
                tightened_stop_price = min(current_soft, target_stop) if current_soft > 0.0 else target_stop
                stop_already_crossed = float(price or 0.0) >= tightened_stop_price
        if not stop_already_crossed:
            action = "trim"
            close_fraction = min(
                1.0,
                max(0.0, float(getattr(self, "risk_tp2_no_continuation_normal_close_fraction", 0.50) or 0.0)),
            )
            close_size = self._align_risk_close_size_for_session(rs, position_abs * close_fraction, max_size=position_abs)
        if close_size <= qty_tol:
            return None

        return {
            "action": action,
            "close_size": close_size,
            "close_fraction_of_remaining": close_fraction,
            "current_profit_r": current_r,
            "max_favorable_excursion_r": mfe_r,
            "elapsed_since_tp1_seconds": elapsed_seconds,
            "continuation_anchor_source": anchor_source,
            "tp1_no_follow_through_at": tp1_no_follow_at,
            "tp1_hit_at": tp1_hit_at,
            "required_seconds": required_seconds,
            "required_mfe_r": required_mfe_r,
            "required_current_r": float(params["tp2_current_r"]),
            "liquidity_band": str(getattr(rs, "staged_exit_liquidity_band", "") or ""),
            "price": float(price or 0.0),
            "tp1_completed": tp1_completed,
            "tp1_no_follow_through_applied": tp1_no_follow_applied,
            "tp2_completed": False,
            "position_size_before": position_abs,
            "initial_entry_price": float(getattr(rs, "initial_entry_price", 0.0) or 0.0),
            "initial_sl_price": float(getattr(rs, "initial_stop_price", 0.0) or 0.0),
            "initial_risk_price_distance": float(getattr(rs, "initial_risk_price_distance", 0.0) or 0.0),
            "new_soft_stop_price": tightened_stop_price,
            "tightened_strategy_stop_price": tightened_stop_price,
            "stop_already_crossed": stop_already_crossed,
        }

    def _risk_executor_result_accepted(self, result: Any) -> bool:
        if isinstance(result, dict) and "accepted" in result:
            return bool(result.get("accepted"))
        if not isinstance(result, dict):
            return False
        if hasattr(self.executor, "_result_has_exchange_error") and self.executor._result_has_exchange_error(result):
            return False
        return bool(result.get("actions"))

    def _cancel_risk_session_resting_refs_for_keys(
        self,
        session: RiskSession,
        keys: set,
        *,
        reason: str,
    ) -> bool:
        target_keys = {str(item or "") for item in set(keys or set()) if str(item or "")}
        refs = [
            dict(item)
            for item in list(getattr(session, "resting_exit_orders", []) or [])
            if str(item.get("key", "") or "") in target_keys
        ]
        if not refs:
            return True
        reason_key = str(reason or "")
        if reason_key == "tp1_no_follow_through":
            event_prefix = "risk_session_tp1_no_follow_through"
        elif reason_key == "tp2_no_continuation":
            event_prefix = "risk_session_tp2_no_continuation"
        elif reason_key == "basis_profit_lock":
            event_prefix = "risk_session_basis_profit_lock"
        else:
            event_prefix = "risk_session_time_decay_take_profit"
        if not hasattr(self.executor, "cancel_reduce_only_tpsl_orders"):
            self._audit_event(
                f"{event_prefix}_cancel_skipped",
                {
                    "plan_name": session.plan_name,
                    "reason": "cancel_method_unavailable",
                    "keys": sorted(target_keys),
                },
            )
            return False
        cancel_result = self.executor.cancel_reduce_only_tpsl_orders(refs, plan_name=session.plan_name)
        accepted = self._risk_executor_result_accepted(cancel_result)
        self._audit_event(
            f"{event_prefix}_cancel",
            {
                "plan_name": session.plan_name,
                "reason": reason,
                "keys": sorted(target_keys),
                "result": cancel_result,
                "accepted": accepted,
            },
        )
        if not accepted:
            return False
        session.resting_exit_orders = [
            ref
            for ref in list(getattr(session, "resting_exit_orders", []) or [])
            if str(ref.get("key", "") or "") not in target_keys
        ]
        session.use_resting_exit_orders = bool(session.resting_exit_orders)
        return True

    def _maybe_execute_risk_session_basis_profit_lock(
        self,
        rs: RiskSession,
        snapshot: dict,
        *,
        price: Optional[float],
        now: float,
    ) -> Optional[str]:
        mark = safe_float(price, None)
        if mark is None or mark <= 0.0:
            return None
        candidate = self._risk_session_basis_profit_lock_candidate(rs, snapshot, price=mark, now=now)
        if candidate is None:
            return None

        action = str(candidate.get("action", "") or "")
        self._audit_event(
            "risk_session_basis_profit_lock",
            {
                "plan_name": rs.plan_name,
                "side": rs.side,
                "candidate": candidate,
                "position_before": snapshot,
            },
        )

        if action == "tighten_tail_stop":
            target_stop = max(0.0, float(candidate.get("target_trailing_soft_stop_price", 0.0) or 0.0))
            if target_stop <= 0.0:
                return None
            previous_stop = max(
                0.0,
                float(getattr(rs, "trailing_soft_stop_price", 0.0) or 0.0)
                or float(getattr(rs, "stop_loss_price", 0.0) or 0.0),
            )
            previous_locked_floor = max(0.0, float(getattr(rs, "locked_floor_price", 0.0) or 0.0))
            rs.trailing_soft_stop_price = target_stop
            rs.stop_loss_price = target_stop
            if str(getattr(rs, "side", "") or "") == "long":
                rs.locked_floor_price = max(previous_locked_floor, target_stop)
            elif str(getattr(rs, "side", "") or "") == "short":
                rs.locked_floor_price = (
                    min(previous_locked_floor, target_stop)
                    if previous_locked_floor > 0.0
                    else target_stop
                )
            rs.basis_profit_observation_active = False
            rs.basis_profit_observation_started_at = 0.0
            rs.basis_profit_observation_basis_start = 0.0
            if getattr(rs, "basis_profit_history", None) is not None:
                rs.basis_profit_history.clear()
            self._persist_risk_session_state()
            self._audit_event(
                "risk_session_basis_profit_lock_applied",
                {
                    "plan_name": rs.plan_name,
                    "action": action,
                    "candidate": candidate,
                    "previous_trailing_soft_stop_price": previous_stop,
                    "new_trailing_soft_stop_price": target_stop,
                    "previous_locked_floor_price": previous_locked_floor,
                    "new_locked_floor_price": float(getattr(rs, "locked_floor_price", 0.0) or 0.0),
                    "remaining_size_abs": abs(float(snapshot.get("size", 0.0) or 0.0)),
                    "position_after": snapshot,
                },
            )
            return "basis_profit_lock"

        if action != "take_profit_like_reduce":
            return None
        leg_key = str(candidate.get("leg_key", "") or "")
        if not leg_key:
            return None
        cancel_ok = self._cancel_risk_session_resting_refs_for_keys(
            rs,
            {leg_key},
            reason="basis_profit_lock",
        )
        if not cancel_ok:
            self._audit_event(
                "risk_session_basis_profit_lock_skipped",
                {
                    "plan_name": rs.plan_name,
                    "candidate": candidate,
                    "reason": "resting_tp_cancel_failed",
                },
            )
            return None

        close_size = float(candidate.get("close_size", 0.0) or 0.0)
        reduce_result = self.executor.reduce_position(
            rs.side,
            close_size,
            "basis_profit_lock",
            rs.plan_name,
            position_before=snapshot,
        )
        accepted = self._risk_executor_result_accepted(reduce_result)
        self._audit_event(
            "risk_session_basis_profit_lock_result",
            {
                "plan_name": rs.plan_name,
                "side": rs.side,
                "candidate": candidate,
                "result": reduce_result,
                "accepted": accepted,
            },
        )
        if not accepted:
            self._sync_risk_session_resting_orders(rs)
            self._persist_risk_session_state()
            return None

        refreshed_snapshot = self.reader.get_position_snapshot(self.symbol)
        before_abs = abs(float(snapshot.get("size", 0.0) or 0.0))
        after_abs = abs(float(refreshed_snapshot.get("size", 0.0) or 0.0))
        qty_tol = self._risk_session_order_qty_tolerance()
        position_delta_closed_abs = self._decimal_size_delta_abs(before_abs, after_abs)
        exchange_filled_abs = self._clamp_accounting_size_abs(
            self._extract_filled_size_from_execution_result(reduce_result),
            max_size=before_abs,
        )
        closed_abs = position_delta_closed_abs
        closed_size_source = "position_delta"
        if exchange_filled_abs > qty_tol:
            closed_abs = exchange_filled_abs
            closed_size_source = "exchange_fill"
            expected_after_abs = self._decimal_size_delta_abs(before_abs, closed_abs)
            if abs(after_abs - expected_after_abs) > qty_tol:
                after_abs = expected_after_abs
                refreshed_snapshot = dict(refreshed_snapshot)
                refreshed_snapshot["size"] = after_abs if rs.side == "long" else -after_abs
                refreshed_snapshot["side"] = rs.side if after_abs > qty_tol else "flat"
                self._audit_event(
                    "risk_session_basis_profit_lock_position_synthesized_from_fill",
                    {
                        "plan_name": rs.plan_name,
                        "candidate": candidate,
                        "exchange_filled_size_abs": exchange_filled_abs,
                        "position_delta_closed_size_abs": position_delta_closed_abs,
                        "position_before": snapshot,
                        "position_after": refreshed_snapshot,
                    },
                )
        if closed_abs <= qty_tol:
            fallback_closed_abs = min(before_abs, close_size)
            if fallback_closed_abs > qty_tol:
                closed_abs = fallback_closed_abs
                closed_size_source = "requested_size_fallback"
                after_abs = self._decimal_size_delta_abs(before_abs, closed_abs)
                refreshed_snapshot = dict(refreshed_snapshot)
                refreshed_snapshot["size"] = after_abs if rs.side == "long" else -after_abs
                refreshed_snapshot["side"] = rs.side if after_abs > qty_tol else "flat"
                self._audit_event(
                    "risk_session_basis_profit_lock_size_fallback",
                    {
                        "plan_name": rs.plan_name,
                        "candidate": candidate,
                        "position_before": snapshot,
                        "position_after": refreshed_snapshot,
                    },
                )

        if after_abs <= qty_tol or not snapshot_has_open_position(refreshed_snapshot):
            self._clear_position_basis_state(reason="basis_profit_lock_flat", position_snapshot=refreshed_snapshot)
            self._replace_risk_session(None)
            self._audit_event(
                "risk_session_basis_profit_lock_applied",
                {
                    "plan_name": rs.plan_name,
                    "action": action,
                    "closed_size_abs": before_abs,
                    "closed_size_source": closed_size_source,
                    "exchange_filled_size_abs": exchange_filled_abs,
                    "position_delta_closed_size_abs": position_delta_closed_abs,
                    "remaining_size_abs": 0.0,
                    "candidate": candidate,
                    "position_after": refreshed_snapshot,
                },
            )
            return "basis_profit_lock"
        if closed_abs <= qty_tol:
            self._audit_event(
                "risk_session_basis_profit_lock_no_size_change",
                {
                    "plan_name": rs.plan_name,
                    "candidate": candidate,
                    "position_before": snapshot,
                    "position_after": refreshed_snapshot,
                },
            )
            self._sync_risk_session_resting_orders(rs)
            self._persist_risk_session_state()
            return None

        rs.expected_size = float(refreshed_snapshot.get("size", 0.0) or 0.0)
        rs.baseline_size = rs.expected_size
        rs.side = str(refreshed_snapshot.get("side", rs.side) or rs.side)
        rs.prev_price = safe_float(refreshed_snapshot.get("mid_price"), mark) or mark
        tail_lock_adjustment = None
        if str(candidate.get("leg_name", "") or "") == "stage_tp2":
            tail_lock_adjustment = self._adjust_time_decay_tp2_tail_locked_floor(rs, candidate)
        completed_keys = self._apply_staged_exit_early_take_profit_trim(rs, closed_abs, now=now)
        rs.basis_profit_observation_active = False
        rs.basis_profit_observation_started_at = 0.0
        rs.basis_profit_observation_basis_start = 0.0
        if getattr(rs, "basis_profit_history", None) is not None:
            rs.basis_profit_history.clear()
        self._sync_risk_session_resting_orders(rs)
        self._persist_risk_session_state()
        self._log_risk_session_ready(rs, reason="basis_profit_lock", position_after=refreshed_snapshot)
        self._audit_event(
            "risk_session_basis_profit_lock_applied",
            {
                "plan_name": rs.plan_name,
                "action": action,
                "closed_size_abs": closed_abs,
                "closed_size_source": closed_size_source,
                "exchange_filled_size_abs": exchange_filled_abs,
                "position_delta_closed_size_abs": position_delta_closed_abs,
                "remaining_size_abs": after_abs,
                "completed_keys": completed_keys,
                "candidate": candidate,
                "tail_lock_adjustment": tail_lock_adjustment,
                "position_after": refreshed_snapshot,
            },
        )
        return "basis_profit_lock"

    def _maybe_execute_risk_session_time_decay_take_profit(
        self,
        rs: RiskSession,
        snapshot: dict,
        *,
        price: Optional[float],
        now: float,
    ) -> Optional[str]:
        mark = safe_float(price, None)
        if mark is None or mark <= 0.0:
            return None
        candidate = self._risk_session_time_decay_tp_candidate(rs, snapshot, price=mark, now=now)
        if candidate is None:
            return None
        leg_key = str(candidate.get("leg_key", "") or "")
        cancel_ok = self._cancel_risk_session_resting_refs_for_keys(
            rs,
            {leg_key},
            reason="time_decay_take_profit",
        )
        if not cancel_ok:
            self._audit_event(
                "risk_session_time_decay_take_profit_skipped",
                {
                    "plan_name": rs.plan_name,
                    "candidate": candidate,
                    "reason": "resting_tp_cancel_failed",
                },
            )
            return None

        close_size = float(candidate.get("close_size", 0.0) or 0.0)
        reduce_result = self.executor.reduce_position(
            rs.side,
            close_size,
            "time_decay_take_profit",
            rs.plan_name,
            position_before=snapshot,
        )
        accepted = self._risk_executor_result_accepted(reduce_result)
        self._audit_event(
            "risk_session_time_decay_take_profit",
            {
                "plan_name": rs.plan_name,
                "side": rs.side,
                "candidate": candidate,
                "result": reduce_result,
                "accepted": accepted,
            },
        )
        if not accepted:
            self._sync_risk_session_resting_orders(rs)
            self._persist_risk_session_state()
            return None

        refreshed_snapshot = self.reader.get_position_snapshot(self.symbol)
        before_abs = abs(float(snapshot.get("size", 0.0) or 0.0))
        after_abs = abs(float(refreshed_snapshot.get("size", 0.0) or 0.0))
        qty_tol = self._risk_session_order_qty_tolerance()
        position_delta_closed_abs = self._decimal_size_delta_abs(before_abs, after_abs)
        exchange_filled_abs = self._clamp_accounting_size_abs(
            self._extract_filled_size_from_execution_result(reduce_result),
            max_size=before_abs,
        )
        closed_abs = position_delta_closed_abs
        closed_size_source = "position_delta"
        if exchange_filled_abs > qty_tol:
            closed_abs = exchange_filled_abs
            closed_size_source = "exchange_fill"
            expected_after_abs = self._decimal_size_delta_abs(before_abs, closed_abs)
            if abs(after_abs - expected_after_abs) > qty_tol:
                after_abs = expected_after_abs
                refreshed_snapshot = dict(refreshed_snapshot)
                refreshed_snapshot["size"] = after_abs if rs.side == "long" else -after_abs
                refreshed_snapshot["side"] = rs.side if after_abs > qty_tol else "flat"
                self._audit_event(
                    "risk_session_time_decay_take_profit_position_synthesized_from_fill",
                    {
                        "plan_name": rs.plan_name,
                        "candidate": candidate,
                        "exchange_filled_size_abs": exchange_filled_abs,
                        "position_delta_closed_size_abs": position_delta_closed_abs,
                        "position_before": snapshot,
                        "position_after": refreshed_snapshot,
                    },
                )
        if closed_abs <= qty_tol:
            fallback_closed_abs = min(before_abs, close_size)
            if fallback_closed_abs > qty_tol:
                closed_abs = fallback_closed_abs
                closed_size_source = "requested_size_fallback"
                after_abs = self._decimal_size_delta_abs(before_abs, closed_abs)
                refreshed_snapshot = dict(refreshed_snapshot)
                refreshed_snapshot["size"] = after_abs if rs.side == "long" else -after_abs
                refreshed_snapshot["side"] = rs.side if after_abs > qty_tol else "flat"
                self._audit_event(
                    "risk_session_time_decay_take_profit_size_fallback",
                    {
                        "plan_name": rs.plan_name,
                        "candidate": candidate,
                        "position_before": snapshot,
                        "position_after": refreshed_snapshot,
                    },
                )
        if after_abs <= qty_tol or not snapshot_has_open_position(refreshed_snapshot):
            self._clear_position_basis_state(reason="time_decay_take_profit_flat", position_snapshot=refreshed_snapshot)
            self._replace_risk_session(None)
            return "take_profit_hit"
        if closed_abs <= qty_tol:
            self._audit_event(
                "risk_session_time_decay_take_profit_no_size_change",
                {
                    "plan_name": rs.plan_name,
                    "candidate": candidate,
                    "position_before": snapshot,
                    "position_after": refreshed_snapshot,
                },
            )
            self._sync_risk_session_resting_orders(rs)
            self._persist_risk_session_state()
            return None

        rs.expected_size = float(refreshed_snapshot.get("size", 0.0) or 0.0)
        rs.baseline_size = rs.expected_size
        rs.side = str(refreshed_snapshot.get("side", rs.side) or rs.side)
        rs.prev_price = safe_float(refreshed_snapshot.get("mid_price"), mark) or mark
        tail_lock_adjustment = None
        if str(candidate.get("leg_name", "") or "") == "stage_tp2":
            tail_lock_adjustment = self._adjust_time_decay_tp2_tail_locked_floor(rs, candidate)
        completed_keys = self._apply_staged_exit_early_take_profit_trim(rs, closed_abs, now=now)
        self._sync_risk_session_resting_orders(rs)
        self._persist_risk_session_state()
        self._log_risk_session_ready(rs, reason="time_decay_take_profit", position_after=refreshed_snapshot)
        self._audit_event(
            "risk_session_time_decay_take_profit_applied",
            {
                "plan_name": rs.plan_name,
                "closed_size_abs": closed_abs,
                "closed_size_source": closed_size_source,
                "exchange_filled_size_abs": exchange_filled_abs,
                "position_delta_closed_size_abs": position_delta_closed_abs,
                "remaining_size_abs": after_abs,
                "completed_keys": completed_keys,
                "candidate": candidate,
                "tail_lock_adjustment": tail_lock_adjustment,
            },
        )
        return "take_profit_hit"

    def _maybe_execute_risk_session_tp1_no_follow_through(
        self,
        rs: RiskSession,
        snapshot: dict,
        *,
        price: Optional[float],
        now: float,
    ) -> Optional[str]:
        mark = safe_float(price, None)
        if mark is None or mark <= 0.0:
            return None
        candidate = self._risk_session_tp1_no_follow_through_candidate(rs, snapshot, price=mark, now=now)
        if candidate is None:
            return None

        action = str(candidate.get("action", "") or "")
        close_size = float(candidate.get("close_size", 0.0) or 0.0)
        self._audit_event(
            "risk_session_tp1_no_follow_through",
            {
                "plan_name": rs.plan_name,
                "side": rs.side,
                "candidate": candidate,
                "position_before": snapshot,
            },
        )

        if action == "flatten":
            close_result = self.executor.close_position(
                rs.side,
                "tp1_no_follow_through",
                rs.plan_name,
                position_before=snapshot,
            )
        elif action == "trim":
            resting_keys = {
                str(ref.get("key", "") or "")
                for ref in list(getattr(rs, "resting_exit_orders", []) or [])
                if isinstance(ref, dict) and str(ref.get("key", "") or "")
            }
            cancel_ok = self._cancel_risk_session_resting_refs_for_keys(
                rs,
                resting_keys,
                reason="tp1_no_follow_through",
            )
            if not cancel_ok:
                self._audit_event(
                    "risk_session_tp1_no_follow_through_skipped",
                    {
                        "plan_name": rs.plan_name,
                        "candidate": candidate,
                        "reason": "resting_exit_cancel_failed",
                    },
                )
                return None
            close_result = self.executor.reduce_position(
                rs.side,
                close_size,
                "tp1_no_follow_through",
                rs.plan_name,
                position_before=snapshot,
            )
        else:
            return None

        accepted = self._risk_executor_result_accepted(close_result)
        self._audit_event(
            "risk_session_tp1_no_follow_through_result",
            {
                "plan_name": rs.plan_name,
                "side": rs.side,
                "candidate": candidate,
                "result": close_result,
                "accepted": accepted,
            },
        )
        if not accepted:
            self._sync_risk_session_resting_orders(rs)
            self._persist_risk_session_state()
            return None

        refreshed_snapshot = self.reader.get_position_snapshot(self.symbol)
        before_abs = abs(float(snapshot.get("size", 0.0) or 0.0))
        after_abs = abs(float(refreshed_snapshot.get("size", 0.0) or 0.0))
        qty_tol = self._risk_session_order_qty_tolerance()
        position_delta_closed_abs = self._decimal_size_delta_abs(before_abs, after_abs)
        exchange_filled_abs = self._clamp_accounting_size_abs(
            self._extract_filled_size_from_execution_result(close_result),
            max_size=before_abs,
        )
        closed_abs = position_delta_closed_abs
        closed_size_source = "position_delta"
        if exchange_filled_abs > qty_tol:
            closed_abs = exchange_filled_abs
            closed_size_source = "exchange_fill"
            expected_after_abs = self._decimal_size_delta_abs(before_abs, closed_abs)
            if abs(after_abs - expected_after_abs) > qty_tol:
                after_abs = expected_after_abs
                refreshed_snapshot = dict(refreshed_snapshot)
                refreshed_snapshot["size"] = after_abs if rs.side == "long" else -after_abs
                refreshed_snapshot["side"] = rs.side if after_abs > qty_tol else "flat"
                self._audit_event(
                    "risk_session_tp1_no_follow_through_position_synthesized_from_fill",
                    {
                        "plan_name": rs.plan_name,
                        "candidate": candidate,
                        "exchange_filled_size_abs": exchange_filled_abs,
                        "position_delta_closed_size_abs": position_delta_closed_abs,
                        "position_before": snapshot,
                        "position_after": refreshed_snapshot,
                    },
                )
        if closed_abs <= qty_tol and action == "trim":
            fallback_closed_abs = min(before_abs, close_size)
            if fallback_closed_abs > qty_tol:
                closed_abs = fallback_closed_abs
                closed_size_source = "requested_size_fallback"
                after_abs = self._decimal_size_delta_abs(before_abs, closed_abs)
                refreshed_snapshot = dict(refreshed_snapshot)
                refreshed_snapshot["size"] = after_abs if rs.side == "long" else -after_abs
                refreshed_snapshot["side"] = rs.side if after_abs > qty_tol else "flat"
                self._audit_event(
                    "risk_session_tp1_no_follow_through_size_fallback",
                    {
                        "plan_name": rs.plan_name,
                        "candidate": candidate,
                        "position_before": snapshot,
                        "position_after": refreshed_snapshot,
                    },
                )

        if after_abs <= qty_tol or not snapshot_has_open_position(refreshed_snapshot):
            self._clear_position_basis_state(reason="tp1_no_follow_through_flat", position_snapshot=refreshed_snapshot)
            self._replace_risk_session(None)
            self._audit_event(
                "risk_session_tp1_no_follow_through_applied",
                {
                    "plan_name": rs.plan_name,
                    "action": action,
                    "closed_size_abs": before_abs,
                    "closed_size_source": closed_size_source,
                    "exchange_filled_size_abs": exchange_filled_abs,
                    "position_delta_closed_size_abs": position_delta_closed_abs,
                    "remaining_size_abs": 0.0,
                    "position_size_before": before_abs,
                    "position_size_after": 0.0,
                    "initial_entry_price": float(getattr(rs, "initial_entry_price", 0.0) or 0.0),
                    "initial_sl_price": float(getattr(rs, "initial_stop_price", 0.0) or 0.0),
                    "initial_risk_price_distance": float(getattr(rs, "initial_risk_price_distance", 0.0) or 0.0),
                    "candidate": candidate,
                    "position_after": refreshed_snapshot,
                },
            )
            return "tp1_no_follow_through"

        if closed_abs <= qty_tol:
            self._audit_event(
                "risk_session_tp1_no_follow_through_no_size_change",
                {
                    "plan_name": rs.plan_name,
                    "candidate": candidate,
                    "position_before": snapshot,
                    "position_after": refreshed_snapshot,
                },
            )
            self._sync_risk_session_resting_orders(rs)
            self._persist_risk_session_state()
            return None

        rs.expected_size = float(refreshed_snapshot.get("size", 0.0) or 0.0)
        rs.baseline_size = rs.expected_size
        rs.side = str(refreshed_snapshot.get("side", rs.side) or rs.side)
        rs.prev_price = safe_float(refreshed_snapshot.get("mid_price"), mark) or mark
        rs.tp1_no_follow_through_applied = True
        if float(getattr(rs, "tp1_no_follow_through_at", 0.0) or 0.0) <= 0.0:
            rs.tp1_no_follow_through_at = float(now or time.time())
        self._retarget_staged_exit_size_basis_after_risk_reduce(rs, after_abs)

        tightened_stop = max(0.0, float(candidate.get("tightened_strategy_stop_price", 0.0) or 0.0))
        if tightened_stop > 0.0:
            self._set_stage_soft_hard_stop(
                rs,
                tightened_stop,
                name="stage_initial_stop",
                note="tp1_no_follow_through_tightened_soft_stop",
            )

        if isinstance(getattr(rs, "position_management", None), PositionManagementPlan):
            rs.position_management.action_decision.action = "no_change"
            rs.position_management.action_decision.close_fraction = 0.0
            rs.position_management.action_decision.entry_price = max(0.0, float(getattr(rs, "initial_entry_price", 0.0) or 0.0))
            rs.position_management.action_decision.stop_loss_price = max(0.0, float(getattr(rs, "stop_loss_price", 0.0) or 0.0))

        self._sync_risk_session_resting_orders(rs)
        self._persist_risk_session_state()
        self._log_risk_session_ready(rs, reason="tp1_no_follow_through", position_after=refreshed_snapshot)
        self._audit_event(
            "risk_session_tp1_no_follow_through_applied",
            {
                "plan_name": rs.plan_name,
                "action": action,
                "closed_size_abs": closed_abs,
                "closed_size_source": closed_size_source,
                "exchange_filled_size_abs": exchange_filled_abs,
                "position_delta_closed_size_abs": position_delta_closed_abs,
                "remaining_size_abs": after_abs,
                "tp1_no_follow_through_at": float(getattr(rs, "tp1_no_follow_through_at", 0.0) or 0.0),
                "tp1_hit_at": float(getattr(rs, "tp1_hit_at", 0.0) or 0.0),
                "position_size_before": before_abs,
                "position_size_after": after_abs,
                "initial_entry_price": float(getattr(rs, "initial_entry_price", 0.0) or 0.0),
                "initial_sl_price": float(getattr(rs, "initial_stop_price", 0.0) or 0.0),
                "initial_risk_price_distance": float(getattr(rs, "initial_risk_price_distance", 0.0) or 0.0),
                "active_soft_stop_price": float(getattr(rs, "active_soft_stop_price", 0.0) or 0.0),
                "candidate": candidate,
                "position_after": refreshed_snapshot,
            },
        )
        return "tp1_no_follow_through"

    def _maybe_execute_risk_session_tp2_no_continuation(
        self,
        rs: RiskSession,
        snapshot: dict,
        *,
        price: Optional[float],
        now: float,
    ) -> Optional[str]:
        mark = safe_float(price, None)
        if mark is None or mark <= 0.0:
            return None
        candidate = self._risk_session_tp2_no_continuation_candidate(rs, snapshot, price=mark, now=now)
        if candidate is None:
            return None

        action = str(candidate.get("action", "") or "")
        close_size = float(candidate.get("close_size", 0.0) or 0.0)
        self._audit_event(
            "risk_session_tp2_no_continuation",
            {
                "plan_name": rs.plan_name,
                "side": rs.side,
                "candidate": candidate,
                "position_before": snapshot,
            },
        )

        if action == "flatten":
            close_result = self.executor.close_position(
                rs.side,
                "tp2_no_continuation",
                rs.plan_name,
                position_before=snapshot,
            )
        elif action == "trim":
            resting_keys = {
                str(ref.get("key", "") or "")
                for ref in list(getattr(rs, "resting_exit_orders", []) or [])
                if isinstance(ref, dict) and str(ref.get("key", "") or "")
            }
            cancel_ok = self._cancel_risk_session_resting_refs_for_keys(
                rs,
                resting_keys,
                reason="tp2_no_continuation",
            )
            if not cancel_ok:
                self._audit_event(
                    "risk_session_tp2_no_continuation_skipped",
                    {
                        "plan_name": rs.plan_name,
                        "candidate": candidate,
                        "reason": "resting_exit_cancel_failed",
                    },
                )
                return None
            close_result = self.executor.reduce_position(
                rs.side,
                close_size,
                "tp2_no_continuation",
                rs.plan_name,
                position_before=snapshot,
            )
        else:
            return None

        accepted = self._risk_executor_result_accepted(close_result)
        self._audit_event(
            "risk_session_tp2_no_continuation_result",
            {
                "plan_name": rs.plan_name,
                "side": rs.side,
                "candidate": candidate,
                "result": close_result,
                "accepted": accepted,
            },
        )
        if not accepted:
            self._sync_risk_session_resting_orders(rs)
            self._persist_risk_session_state()
            return None

        refreshed_snapshot = self.reader.get_position_snapshot(self.symbol)
        before_abs = abs(float(snapshot.get("size", 0.0) or 0.0))
        after_abs = abs(float(refreshed_snapshot.get("size", 0.0) or 0.0))
        qty_tol = self._risk_session_order_qty_tolerance()
        position_delta_closed_abs = self._decimal_size_delta_abs(before_abs, after_abs)
        exchange_filled_abs = self._clamp_accounting_size_abs(
            self._extract_filled_size_from_execution_result(close_result),
            max_size=before_abs,
        )
        closed_abs = position_delta_closed_abs
        closed_size_source = "position_delta"
        if exchange_filled_abs > qty_tol:
            closed_abs = exchange_filled_abs
            closed_size_source = "exchange_fill"
            expected_after_abs = self._decimal_size_delta_abs(before_abs, closed_abs)
            if abs(after_abs - expected_after_abs) > qty_tol:
                after_abs = expected_after_abs
                refreshed_snapshot = dict(refreshed_snapshot)
                refreshed_snapshot["size"] = after_abs if rs.side == "long" else -after_abs
                refreshed_snapshot["side"] = rs.side if after_abs > qty_tol else "flat"
                self._audit_event(
                    "risk_session_tp2_no_continuation_position_synthesized_from_fill",
                    {
                        "plan_name": rs.plan_name,
                        "candidate": candidate,
                        "exchange_filled_size_abs": exchange_filled_abs,
                        "position_delta_closed_size_abs": position_delta_closed_abs,
                        "position_before": snapshot,
                        "position_after": refreshed_snapshot,
                    },
                )
        if closed_abs <= qty_tol and action == "trim":
            fallback_closed_abs = min(before_abs, close_size)
            if fallback_closed_abs > qty_tol:
                closed_abs = fallback_closed_abs
                closed_size_source = "requested_size_fallback"
                after_abs = self._decimal_size_delta_abs(before_abs, closed_abs)
                refreshed_snapshot = dict(refreshed_snapshot)
                refreshed_snapshot["size"] = after_abs if rs.side == "long" else -after_abs
                refreshed_snapshot["side"] = rs.side if after_abs > qty_tol else "flat"
                self._audit_event(
                    "risk_session_tp2_no_continuation_size_fallback",
                    {
                        "plan_name": rs.plan_name,
                        "candidate": candidate,
                        "position_before": snapshot,
                        "position_after": refreshed_snapshot,
                    },
                )

        if after_abs <= qty_tol or not snapshot_has_open_position(refreshed_snapshot):
            self._clear_position_basis_state(reason="tp2_no_continuation_flat", position_snapshot=refreshed_snapshot)
            self._replace_risk_session(None)
            self._audit_event(
                "risk_session_tp2_no_continuation_applied",
                {
                    "plan_name": rs.plan_name,
                    "action": action,
                    "closed_size_abs": before_abs,
                    "closed_size_source": closed_size_source,
                    "exchange_filled_size_abs": exchange_filled_abs,
                    "position_delta_closed_size_abs": position_delta_closed_abs,
                    "remaining_size_abs": 0.0,
                    "position_size_before": before_abs,
                    "position_size_after": 0.0,
                    "initial_entry_price": float(getattr(rs, "initial_entry_price", 0.0) or 0.0),
                    "initial_sl_price": float(getattr(rs, "initial_stop_price", 0.0) or 0.0),
                    "initial_risk_price_distance": float(getattr(rs, "initial_risk_price_distance", 0.0) or 0.0),
                    "candidate": candidate,
                    "position_after": refreshed_snapshot,
                },
            )
            return "tp2_no_continuation"

        if closed_abs <= qty_tol:
            self._audit_event(
                "risk_session_tp2_no_continuation_no_size_change",
                {
                    "plan_name": rs.plan_name,
                    "candidate": candidate,
                    "position_before": snapshot,
                    "position_after": refreshed_snapshot,
                },
            )
            self._sync_risk_session_resting_orders(rs)
            self._persist_risk_session_state()
            return None

        rs.expected_size = float(refreshed_snapshot.get("size", 0.0) or 0.0)
        rs.baseline_size = rs.expected_size
        rs.side = str(refreshed_snapshot.get("side", rs.side) or rs.side)
        rs.prev_price = safe_float(refreshed_snapshot.get("mid_price"), mark) or mark
        rs.tp2_no_continuation_applied = True
        self._retarget_staged_exit_size_basis_after_risk_reduce(rs, after_abs)

        tightened_stop = max(0.0, float(candidate.get("tightened_strategy_stop_price", 0.0) or 0.0))
        if tightened_stop > 0.0:
            rs.post_tp1_stop_price = tightened_stop
            self._set_stage_soft_hard_stop(
                rs,
                tightened_stop,
                name="stage_post_tp1_stop",
                note="tp2_no_continuation_tightened_soft_stop",
            )

        if isinstance(getattr(rs, "position_management", None), PositionManagementPlan):
            rs.position_management.action_decision.action = "no_change"
            rs.position_management.action_decision.close_fraction = 0.0
            rs.position_management.action_decision.entry_price = max(0.0, float(getattr(rs, "initial_entry_price", 0.0) or 0.0))
            rs.position_management.action_decision.stop_loss_price = max(0.0, float(getattr(rs, "stop_loss_price", 0.0) or 0.0))

        self._sync_risk_session_resting_orders(rs)
        self._persist_risk_session_state()
        self._log_risk_session_ready(rs, reason="tp2_no_continuation", position_after=refreshed_snapshot)
        self._audit_event(
            "risk_session_tp2_no_continuation_applied",
            {
                "plan_name": rs.plan_name,
                "action": action,
                "closed_size_abs": closed_abs,
                "closed_size_source": closed_size_source,
                "exchange_filled_size_abs": exchange_filled_abs,
                "position_delta_closed_size_abs": position_delta_closed_abs,
                "remaining_size_abs": after_abs,
                "position_size_before": before_abs,
                "position_size_after": after_abs,
                "initial_entry_price": float(getattr(rs, "initial_entry_price", 0.0) or 0.0),
                "initial_sl_price": float(getattr(rs, "initial_stop_price", 0.0) or 0.0),
                "initial_risk_price_distance": float(getattr(rs, "initial_risk_price_distance", 0.0) or 0.0),
                "active_soft_stop_price": float(getattr(rs, "active_soft_stop_price", 0.0) or 0.0),
                "candidate": candidate,
                "position_after": refreshed_snapshot,
            },
        )
        return "tp2_no_continuation"

    def _risk_session_active_soft_stop_price(self, rs: RiskSession) -> float:
        explicit = max(0.0, float(getattr(rs, "active_soft_stop_price", 0.0) or 0.0))
        if explicit > 0.0:
            return explicit
        if bool(getattr(rs, "tp2_hit", False)):
            return 0.0
        if bool(getattr(rs, "tp1_hit", False)):
            return max(0.0, float(getattr(rs, "post_tp1_stop_price", 0.0) or 0.0))
        return max(0.0, float(getattr(rs, "initial_stop_price", 0.0) or getattr(rs, "stop_loss_price", 0.0) or 0.0))

    @staticmethod
    def _risk_session_stop_breach_amount(side: str, price: float, stop_price: float) -> float:
        if side == "long":
            return max(0.0, float(stop_price or 0.0) - float(price or 0.0))
        if side == "short":
            return max(0.0, float(price or 0.0) - float(stop_price or 0.0))
        return 0.0

    def _risk_session_soft_stop_candidate(
        self,
        rs: RiskSession,
        *,
        price: Optional[float],
        now: float,
    ) -> Optional[Dict[str, Any]]:
        if not bool(getattr(self, "risk_soft_stop_enabled", True)):
            return None
        if not bool(getattr(rs, "staged_exit_enabled", False)) or bool(getattr(rs, "tp2_hit", False)):
            return None
        mark = safe_float(price, None)
        side = str(getattr(rs, "side", "") or "")
        soft_stop = self._risk_session_active_soft_stop_price(rs)
        if side not in {"long", "short"} or soft_stop <= 0.0:
            return None
        timeframe = str(getattr(self, "risk_soft_stop_confirm_timeframe", "1m") or "1m").strip().lower() or "1m"
        try:
            candles = self._latest_completed_candles_for_risk_session(
                rs,
                now=now,
                interval=timeframe,
                min_bars=2,
                use_trailing_lookback=False,
            )
        except Exception:
            return None
        if not candles:
            return None
        latest_bar = candles[-1]
        close_price = safe_float(latest_bar.get("close"), None)
        if close_price is None or close_price <= 0.0:
            return None
        cross_asset_adjustment = self._risk_session_cross_asset_soft_stop_adjustment(
            rs,
            base_soft_stop=soft_stop,
            side=side,
        )
        effective_soft_stop = safe_float(cross_asset_adjustment.get("effective_soft_stop_price"), soft_stop) or soft_stop
        breach_amount = self._risk_session_stop_breach_amount(side, close_price, effective_soft_stop)
        if breach_amount <= 0.0:
            rs.soft_stop_breach_since = 0.0
            rs.soft_stop_last_breach_price = 0.0
            return None
        rs.soft_stop_breach_since = 0.0
        rs.soft_stop_last_breach_price = close_price
        current_breach_amount = self._risk_session_stop_breach_amount(side, mark, effective_soft_stop) if mark is not None else 0.0
        return {
            "plan_name": rs.plan_name,
            "side": side,
            "soft_stop_price": effective_soft_stop,
            "base_soft_stop_price": soft_stop,
            "effective_soft_stop_price": effective_soft_stop,
            "cross_asset_soft_stop": cross_asset_adjustment,
            "hard_stop_price": float(getattr(rs, "active_hard_stop_price", 0.0) or 0.0),
            "soft_stop_buffer_usd": float(getattr(rs, "exchange_hard_stop_buffer_usd", 0.0) or 0.0),
            "hard_stop_buffer_usd": float(getattr(rs, "exchange_hard_stop_buffer_usd", 0.0) or 0.0),
            "hard_stop_min_buffer_usd": float(getattr(rs, "exchange_hard_stop_min_buffer_usd", 0.0) or 0.0),
            "hard_stop_atr_buffer_usd": float(getattr(rs, "exchange_hard_stop_atr_buffer_usd", 0.0) or 0.0),
            "hard_stop_r_buffer_usd": float(getattr(rs, "exchange_hard_stop_r_buffer_usd", 0.0) or 0.0),
            "hard_stop_atr_value": float(getattr(rs, "exchange_hard_stop_atr_value", 0.0) or 0.0),
            "price": mark if mark is not None else close_price,
            "current_price": mark,
            "breach_amount": breach_amount,
            "current_breach_amount": current_breach_amount,
            "confirm_timeframe": timeframe,
            "candle_close_price": close_price,
            "candle_close_ms": int(latest_bar.get("close_ms", 0) or 0),
            "confirmed_by": "candle_close",
            "stage": self._risk_session_stage_name(rs),
        }

    def _maybe_execute_risk_session_soft_stop(
        self,
        rs: RiskSession,
        snapshot: dict,
        *,
        price: Optional[float],
        now: float,
    ) -> Optional[str]:
        candidate = self._risk_session_soft_stop_candidate(rs, price=price, now=now)
        if candidate is None:
            return None
        close_result = self.executor.close_position(rs.side, "soft_stop", rs.plan_name, position_before=snapshot)
        accepted = self._risk_executor_result_accepted(close_result)
        self._audit_event(
            "risk_session_soft_stop_triggered",
            {
                "plan_name": rs.plan_name,
                "side": rs.side,
                "candidate": candidate,
                "position_before": snapshot,
                "result": close_result,
                "accepted": accepted,
            },
        )
        if not accepted:
            return None
        refreshed_snapshot = self.reader.get_position_snapshot(self.symbol)
        if snapshot_has_open_position(refreshed_snapshot):
            rs.expected_size = float(refreshed_snapshot.get("size", 0.0) or 0.0)
            rs.side = str(refreshed_snapshot.get("side", rs.side) or rs.side)
            rs.pending_fill_reconcile_since = now
            rs.soft_stop_breach_since = 0.0
            rs.soft_stop_last_breach_price = 0.0
            self._persist_risk_session_state()
        else:
            self._clear_position_basis_state(reason="soft_stop_flat", position_snapshot=refreshed_snapshot)
            self._replace_risk_session(None)
        return "soft_stop_hit"

    def _update_staged_risk_session_trailing_state(self, rs: RiskSession, *, now: float) -> bool:
        if not bool(getattr(rs, "staged_exit_enabled", False)) or not bool(getattr(rs, "tp2_hit", False)):
            return False
        anchor_candles = self._completed_candles_for_risk_session(
            rs,
            now=now,
            interval=getattr(rs, "trailing_timeframe", "15m"),
            min_bars=max(int(getattr(rs, "trailing_atr_period", 14) or 14) + 5, 24),
        )
        atr_candles = self._latest_completed_candles_for_risk_session(
            rs,
            now=now,
            interval=getattr(rs, "trailing_timeframe", "15m"),
            min_bars=max(int(getattr(rs, "trailing_atr_period", 14) or 14) + 5, 24),
        )
        if not anchor_candles or not atr_candles:
            return False
        latest_bar = atr_candles[-1]
        latest_bar_ms = int(latest_bar.get("close_ms", 0) or 0)
        if latest_bar_ms <= int(getattr(rs, "trailing_last_bar_ms", 0) or 0):
            return False
        atr = self._atr_from_completed_candles(atr_candles, int(getattr(rs, "trailing_atr_period", 14) or 14))
        if atr is None or atr <= 0.0:
            return False
        completed_since_entry = [
            candle for candle in anchor_candles
            if int(candle.get("close_ms", 0) or 0) >= int(float(getattr(rs, "start_time", 0.0) or 0.0) * 1000)
        ]
        if not completed_since_entry:
            return False
        if rs.side == "long":
            highest_close = max(float(candle.get("close", 0.0) or 0.0) for candle in completed_since_entry)
            rs.trailing_highest_close = max(float(getattr(rs, "trailing_highest_close", 0.0) or 0.0), highest_close)
            soft_atr_stop = self._align_price_for_symbol(
                self.symbol,
                rs.trailing_highest_close - (float(getattr(rs, "trailing_soft_atr_mult", 2.5) or 2.5) * atr),
            )
            hard_atr_stop = self._align_price_for_symbol(
                self.symbol,
                rs.trailing_highest_close - (float(getattr(rs, "trailing_hard_atr_mult", 3.5) or 3.5) * atr),
            )
            current_soft_stop = max(0.0, float(getattr(rs, "trailing_soft_stop_price", 0.0) or 0.0))
            locked_floor = max(
                0.0,
                float(getattr(rs, "locked_floor_price", 0.0) or 0.0),
                current_soft_stop,
            )
            rs.trailing_soft_stop_price = max(locked_floor, soft_atr_stop)
            rs.trailing_hard_stop_price = max(locked_floor, hard_atr_stop)
        elif rs.side == "short":
            lowest_close = min(float(candle.get("close", 0.0) or 0.0) for candle in completed_since_entry)
            prior_lowest = float(getattr(rs, "trailing_lowest_close", 0.0) or 0.0)
            rs.trailing_lowest_close = lowest_close if prior_lowest <= 0.0 else min(prior_lowest, lowest_close)
            soft_atr_stop = self._align_price_for_symbol(
                self.symbol,
                rs.trailing_lowest_close + (float(getattr(rs, "trailing_soft_atr_mult", 2.5) or 2.5) * atr),
            )
            hard_atr_stop = self._align_price_for_symbol(
                self.symbol,
                rs.trailing_lowest_close + (float(getattr(rs, "trailing_hard_atr_mult", 3.5) or 3.5) * atr),
            )
            current_soft_stop = max(0.0, float(getattr(rs, "trailing_soft_stop_price", 0.0) or 0.0))
            locked_ceiling = max(0.0, float(getattr(rs, "locked_floor_price", 0.0) or 0.0))
            if current_soft_stop > 0.0:
                locked_ceiling = min(locked_ceiling, current_soft_stop) if locked_ceiling > 0.0 else current_soft_stop
            rs.trailing_soft_stop_price = min(locked_ceiling, soft_atr_stop) if locked_ceiling > 0.0 else soft_atr_stop
            rs.trailing_hard_stop_price = min(locked_ceiling, hard_atr_stop) if locked_ceiling > 0.0 else hard_atr_stop
        else:
            return False
        rs.trailing_last_bar_ms = latest_bar_ms
        rs.trailing_last_close_price = float(latest_bar.get("close", 0.0) or 0.0)
        if float(getattr(rs, "trailing_soft_stop_price", 0.0) or 0.0) > 0.0:
            rs.stop_loss_price = float(rs.trailing_soft_stop_price)
            rs.stop_loss_legs = []
        return True
    def _staged_risk_session_soft_stop_trigger_candidate(self, rs: RiskSession, *, now: float) -> Optional[Dict[str, Any]]:
        if not bool(getattr(rs, "staged_exit_enabled", False)) or not bool(getattr(rs, "tp2_hit", False)):
            return None
        soft_stop = float(getattr(rs, "trailing_soft_stop_price", 0.0) or 0.0)
        if soft_stop <= 0.0:
            return None
        confirm_timeframe = str(getattr(self, "risk_soft_stop_confirm_timeframe", "1m") or "1m").strip().lower() or "1m"
        try:
            candles = self._latest_completed_candles_for_risk_session(
                rs,
                now=now,
                interval=confirm_timeframe,
                min_bars=2,
                use_trailing_lookback=False,
            )
        except Exception:
            return None
        if not candles:
            return None
        latest_bar = candles[-1]
        close_price = safe_float(latest_bar.get("close"), None)
        if close_price is None or close_price <= 0.0:
            return None
        breached = False
        if rs.side == "long":
            breached = close_price <= soft_stop
        elif rs.side == "short":
            breached = close_price >= soft_stop
        if not breached:
            return None
        return {
            "soft_stop_price": soft_stop,
            "hard_stop_price": float(getattr(rs, "trailing_hard_stop_price", 0.0) or 0.0),
            "last_close_price": close_price,
            "confirm_timeframe": confirm_timeframe,
            "candle_close_price": close_price,
            "candle_close_ms": int(latest_bar.get("close_ms", 0) or 0),
            "confirmed_by": "candle_close",
            "trailing_timeframe": str(getattr(rs, "trailing_timeframe", "") or ""),
            "trailing_last_bar_ms": int(getattr(rs, "trailing_last_bar_ms", 0) or 0),
            "trailing_last_close_price": float(getattr(rs, "trailing_last_close_price", 0.0) or 0.0),
        }
    def _cancel_risk_session_resting_orders(self, session: Optional[RiskSession]) -> None:
        if session is None:
            return
        order_refs = [dict(item) for item in list(getattr(session, "resting_exit_orders", []) or []) if isinstance(item, dict)]
        session.resting_exit_orders = []
        session.use_resting_exit_orders = False
        if not order_refs or not hasattr(self.executor, "cancel_reduce_only_tpsl_orders"):
            return
        cancel_result = self.executor.cancel_reduce_only_tpsl_orders(order_refs, plan_name=session.plan_name)
        self._audit_event(
            "risk_session_tpsl_orders_cancelled",
            {
                "plan_name": session.plan_name,
                "result": cancel_result,
            },
        )
    def _iter_risk_session_exit_order_specs(self, session: RiskSession, leg_type_filter: Optional[str] = None) -> List[Dict[str, Any]]:
        specs: List[Dict[str, Any]] = []
        remaining_size_abs = abs(float(session.expected_size or 0.0))
        if remaining_size_abs <= 0.0:
            return specs

        def append_specs(leg_type: str, legs: List[ExitLeg]) -> None:
            if leg_type_filter and leg_type != leg_type_filter:
                return
            active: List[Dict[str, Any]] = []
            for leg in list(legs or []):
                key = f"{leg_type}::{leg.name}"
                if key in session.executed_leg_names:
                    continue
                if len(list(leg.when_all or [])) != 1:
                    continue
                condition = leg.when_all[0]
                cond_type = str(getattr(condition, "type", "") or "")
                trigger_price = float(getattr(condition, "level", 0.0) or 0.0)
                if cond_type not in {"price_ge", "price_le"} or trigger_price <= 0.0:
                    continue
                active.append({
                    "key": key,
                    "name": leg.name,
                    "trigger_price": trigger_price,
                    "close_fraction": float(leg.close_fraction or 0.0),
                    "tpsl": "tp" if leg_type == "take_profit" else "sl",
                    "leg_type": leg_type,
                })
            total_fraction = sum(max(0.0, float(item.get("close_fraction", 0.0) or 0.0)) for item in active)
            if total_fraction <= 0.0:
                return
            if (
                leg_type == "take_profit"
                and bool(getattr(session, "staged_exit_enabled", False))
                and bool(getattr(session, "take_profit_legs_scale_from_initial_size", False))
            ):
                allocated_size = 0.0
                qty_tol = self._risk_session_order_qty_tolerance()
                for item in active:
                    close_fraction = float(item.get("close_fraction", 0.0) or 0.0)
                    target_size = (
                        self._staged_tp_target_size_abs(session, str(item.get("name", "") or ""))
                        if str(item.get("name", "") or "").startswith("stage_tp")
                        else self._align_risk_exit_size_to_precision(
                            abs(float(getattr(session, "staged_exit_size_basis_abs", 0.0) or getattr(session, "initial_size_abs", 0.0) or 0.0)) * close_fraction,
                            str(getattr(session, "symbol", "") or ""),
                        )
                    )
                    completed_size = self._staged_tp_completed_size_abs(session, str(item.get("name", "") or ""))
                    remaining_target = max(0.0, target_size - completed_size)
                    remaining_capacity = max(0.0, remaining_size_abs - allocated_size)
                    close_size = min(remaining_target, remaining_capacity)
                    if close_size <= qty_tol:
                        continue
                    item["close_size"] = close_size
                    allocated_size += close_size
                    specs.append(item)
                return
            for item in active:
                close_fraction = float(item.get("close_fraction", 0.0) or 0.0)
                if leg_type == "take_profit" and bool(getattr(session, "take_profit_legs_scale_from_initial_size", False)):
                    item["close_size"] = min(
                        remaining_size_abs,
                        self._align_risk_exit_size_to_precision(
                            abs(float(session.initial_size_abs or 0.0)) * close_fraction,
                            str(getattr(session, "symbol", "") or ""),
                        ),
                    )
                else:
                    item["close_size"] = self._align_risk_close_size_for_session(
                        session,
                        remaining_size_abs * (close_fraction / total_fraction),
                        max_size=remaining_size_abs,
                    )
                specs.append(item)

        append_specs("take_profit", session.take_profit_legs)
        append_specs("stop_loss", session.stop_loss_legs)
        return specs
    def _resync_risk_session_counterpart_orders(self, session: RiskSession, hit_leg_type: str) -> None:
        counterpart_leg_type = "stop_loss" if hit_leg_type == "take_profit" else "take_profit" if hit_leg_type == "stop_loss" else ""
        if counterpart_leg_type not in {"take_profit", "stop_loss"}:
            session.use_resting_exit_orders = bool(session.resting_exit_orders)
            return

        specs = self._iter_risk_session_exit_order_specs(session, leg_type_filter=counterpart_leg_type)
        if not specs:
            counterpart_refs = [
                dict(item)
                for item in list(session.resting_exit_orders or [])
                if str(item.get("leg_type", "") or "") == counterpart_leg_type
            ]
            if counterpart_refs and hasattr(self.executor, "cancel_reduce_only_tpsl_orders"):
                self.executor.cancel_reduce_only_tpsl_orders(counterpart_refs, plan_name=session.plan_name)
                remove_keys = {str(item.get("key", "") or "") for item in counterpart_refs}
                session.resting_exit_orders = [
                    ref
                    for ref in list(session.resting_exit_orders or [])
                    if str(ref.get("key", "") or "") not in remove_keys
                ]
            session.use_resting_exit_orders = bool(session.resting_exit_orders)
            return

        counterpart_refs = [
            dict(item)
            for item in list(session.resting_exit_orders or [])
            if str(item.get("leg_type", "") or "") == counterpart_leg_type
        ]
        spec_by_key = {str(item.get("key", "") or ""): item for item in specs}
        ref_by_key = {str(item.get("key", "") or ""): item for item in counterpart_refs}
        keys_match = set(spec_by_key.keys()) == set(ref_by_key.keys())
        single_order_replacement = len(specs) == 1 and len(counterpart_refs) == 1
        counterpart_refs_have_limit = any(str(item.get("order_kind", "") or "").lower() == "limit" for item in counterpart_refs)

        can_modify_existing_orders = (
            counterpart_refs
            and not counterpart_refs_have_limit
            and hasattr(self.executor, "modify_reduce_only_tpsl_orders")
            and (keys_match or single_order_replacement)
        )
        if can_modify_existing_orders:
            order_updates = []
            if keys_match:
                spec_ref_pairs = [
                    (spec_by_key[key], ref_by_key[key])
                    for key in spec_by_key.keys()
                ]
            else:
                spec_ref_pairs = list(zip(specs, counterpart_refs))
            for spec, current_ref in spec_ref_pairs:
                order_updates.append(
                    {
                        "key": str(spec.get("key", "") or current_ref.get("key", "") or ""),
                        "name": str(spec.get("name", "") or current_ref.get("name", "") or ""),
                        "leg_type": counterpart_leg_type,
                        "tpsl": str(spec.get("tpsl", "") or current_ref.get("tpsl", "") or ""),
                        "trigger_price": float(spec.get("trigger_price", 0.0) or current_ref.get("trigger_price", 0.0) or 0.0),
                        "close_size": float(spec.get("close_size", 0.0) or 0.0),
                        "oid": current_ref.get("oid"),
                        "cloid": current_ref.get("cloid"),
                        "order_kind": str(current_ref.get("order_kind", "") or "trigger"),
                        "is_trigger": bool(current_ref.get("is_trigger", True)),
                    }
                )
            modify_result = self.executor.modify_reduce_only_tpsl_orders(
                order_updates,
                side=session.side,
                plan_name=session.plan_name,
            )
            self._audit_event(
                "risk_session_tpsl_counterpart_orders_modified",
                {
                    "plan_name": session.plan_name,
                    "hit_leg_type": hit_leg_type,
                    "counterpart_leg_type": counterpart_leg_type,
                    "result": modify_result,
                },
            )
            if modify_result.get("accepted", False):
                updated_ref_map = {
                    str(item.get("key", "") or ""): dict(item)
                    for item in list(modify_result.get("updated_refs") or [])
                    if isinstance(item, dict)
                }
                refreshed_refs: List[Dict[str, Any]] = []
                counterpart_ref_keys = {
                    str(item.get("key", "") or "")
                    for item in counterpart_refs
                    if isinstance(item, dict)
                }
                for ref in list(session.resting_exit_orders or []):
                    key = str(ref.get("key", "") or "")
                    if key in counterpart_ref_keys:
                        desired_key = None
                        if len(spec_ref_pairs) == 1:
                            desired_key = str(spec_ref_pairs[0][0].get("key", "") or "")
                        replacement = updated_ref_map.get(desired_key or key, None)
                        refreshed_refs.append(replacement or ref)
                    else:
                        refreshed_refs.append(ref)
                session.resting_exit_orders = refreshed_refs
                session.use_resting_exit_orders = bool(session.resting_exit_orders)
                return

        session.use_resting_exit_orders = bool(session.resting_exit_orders)
        if counterpart_refs_have_limit:
            self._audit_event(
                "risk_session_tpsl_counterpart_orders_rebuild_required",
                {
                    "plan_name": session.plan_name,
                    "hit_leg_type": hit_leg_type,
                    "counterpart_leg_type": counterpart_leg_type,
                    "reason": "limit_counterpart_requires_rebuild",
                    "spec_keys": sorted(spec_by_key.keys()),
                    "ref_keys": sorted(ref_by_key.keys()),
                },
            )
            if (
                bool(getattr(self.executor, "enabled", False))
                and hasattr(self.executor, "cancel_reduce_only_tpsl_orders")
                and hasattr(self.executor, "place_reduce_only_tpsl_order")
            ):
                self._sync_risk_session_resting_orders(session)
                return
        if counterpart_refs and not keys_match and not single_order_replacement:
            self._audit_event(
                "risk_session_tpsl_counterpart_orders_rebuild_required",
                {
                    "plan_name": session.plan_name,
                    "hit_leg_type": hit_leg_type,
                    "counterpart_leg_type": counterpart_leg_type,
                    "reason": "multi_order_key_mismatch",
                    "spec_keys": sorted(spec_by_key.keys()),
                    "ref_keys": sorted(ref_by_key.keys()),
                },
            )
            if (
                bool(getattr(self.executor, "enabled", False))
                and hasattr(self.executor, "cancel_reduce_only_tpsl_orders")
                and hasattr(self.executor, "place_reduce_only_tpsl_order")
            ):
                self._sync_risk_session_resting_orders(session)
                return
        self._audit_event(
            "risk_session_tpsl_counterpart_orders_modify_skipped",
            {
                "plan_name": session.plan_name,
                "hit_leg_type": hit_leg_type,
                "counterpart_leg_type": counterpart_leg_type,
                "reason": "modify_unavailable_or_key_mismatch_or_rejected",
                "spec_keys": sorted(spec_by_key.keys()),
                "ref_keys": sorted(ref_by_key.keys()),
                "has_modify": bool(hasattr(self.executor, "modify_reduce_only_tpsl_orders")),
            },
        )
    def _sync_risk_session_resting_orders(self, session: Optional[RiskSession]) -> None:
        if session is None:
            return
        self._cancel_risk_session_resting_orders(session)
        if not bool(getattr(self.executor, "enabled", False)):
            return
        if not hasattr(self.executor, "place_reduce_only_tpsl_order"):
            return
        specs = self._iter_risk_session_exit_order_specs(session)
        if not specs:
            return
        placed_refs: List[Dict[str, Any]] = []
        for spec in specs:
            leg_type = str(spec.get("leg_type", "") or "")
            if leg_type == "take_profit" and hasattr(self.executor, "place_reduce_only_limit_order"):
                placement = self.executor.place_reduce_only_limit_order(
                    side=session.side,
                    close_size=float(spec.get("close_size", 0.0) or 0.0),
                    limit_price=float(spec.get("trigger_price", 0.0) or 0.0),
                    plan_name=session.plan_name,
                    leg_name=str(spec.get("name", "") or ""),
                )
            else:
                placement = self.executor.place_reduce_only_tpsl_order(
                    side=session.side,
                    close_size=float(spec.get("close_size", 0.0) or 0.0),
                    trigger_price=float(spec.get("trigger_price", 0.0) or 0.0),
                    tpsl=str(spec.get("tpsl", "") or "sl"),
                    plan_name=session.plan_name,
                    leg_name=str(spec.get("name", "") or ""),
                )
            if not placement.get("accepted", False):
                if placed_refs and hasattr(self.executor, "cancel_reduce_only_tpsl_orders"):
                    self.executor.cancel_reduce_only_tpsl_orders(placed_refs, plan_name=session.plan_name)
                session.resting_exit_orders = []
                session.use_resting_exit_orders = False
                self._audit_event(
                    "risk_session_tpsl_orders_place_failed",
                    {
                        "plan_name": session.plan_name,
                        "spec": spec,
                        "result": placement,
                    },
                )
                return
            order_kind = str(placement.get("order_kind", "") or ("limit" if leg_type == "take_profit" and hasattr(self.executor, "place_reduce_only_limit_order") else "trigger"))
            trigger_price = float(placement.get("trigger_price", spec.get("trigger_price", 0.0)) or 0.0)
            placed_refs.append({
                "key": str(spec.get("key", "") or ""),
                "name": str(spec.get("name", "") or ""),
                "leg_type": leg_type,
                "tpsl": str(spec.get("tpsl", "") or ""),
                "trigger_price": trigger_price,
                "limit_price": float(placement.get("limit_price", trigger_price) or 0.0) if order_kind == "limit" else 0.0,
                "close_size": float(placement.get("close_size", spec.get("close_size", 0.0)) or 0.0),
                "cloid": str(placement.get("cloid", "") or ""),
                "oid": placement.get("oid"),
                "order_kind": order_kind,
                "is_trigger": bool(placement.get("is_trigger", order_kind != "limit")),
            })
        session.resting_exit_orders = placed_refs
        session.use_resting_exit_orders = bool(placed_refs)
        if placed_refs:
            self._audit_event(
                "risk_session_tpsl_orders_placed",
                {
                    "plan_name": session.plan_name,
                    "orders": placed_refs,
                },
            )
    def _find_resting_exit_order_ref_for_fill(self, session: RiskSession, fill: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        fill_coin = canonicalize_execution_symbol(fill.get("coin", ""))
        active_symbol = canonicalize_execution_symbol(getattr(self, "symbol", ""))
        if fill_coin and active_symbol and fill_coin != active_symbol:
            return None
        fill_oid: Optional[int] = None
        try:
            if fill.get("oid") is not None:
                fill_oid = int(fill.get("oid"))
        except Exception:
            fill_oid = None
        fill_cloid = str(fill.get("cloid", "") or "").strip().lower()
        for ref in list(getattr(session, "resting_exit_orders", []) or []):
            oid_value = ref.get("oid")
            if fill_oid is not None and oid_value is not None:
                try:
                    if int(oid_value) == fill_oid:
                        return ref
                except Exception:
                    pass
            cloid_value = str(ref.get("cloid", "") or "").strip().lower()
            if fill_cloid and cloid_value and cloid_value == fill_cloid:
                return ref
        return None
    @staticmethod
    def _order_status_text(payload: Any) -> str:
        try:
            return json.dumps(payload, ensure_ascii=False, sort_keys=True).lower()
        except Exception:
            return str(payload).lower()
    @classmethod
    def _order_status_indicates_exchange_hit(cls, payload: Any) -> bool:
        text = cls._order_status_text(payload)
        return bool(text and ("filled" in text or "triggered" in text))
    def _reconcile_risk_session_exchange_fill(self, snapshot: dict, now: float) -> Optional[str]:
        if self.risk_session is None:
            return None
        rs = self.risk_session
        order_refs = [dict(item) for item in list(getattr(rs, "resting_exit_orders", []) or []) if isinstance(item, dict)]
        if not order_refs:
            return None

        try:
            open_orders = list(self.reader.get_frontend_open_orders(self.symbol) or [])
            info_client = self.reader.get_info_client()
        except Exception:
            return None

        open_oids: set = set()
        for item in open_orders:
            if not isinstance(item, dict):
                continue
            try:
                if item.get("oid") is not None:
                    open_oids.add(int(item.get("oid")))
            except Exception:
                continue

        user = str(getattr(self.reader, "account_address", "") or "").strip()
        synthetic_fills: List[Dict[str, Any]] = []
        matched_refs: List[Dict[str, Any]] = []
        status_summaries: List[Dict[str, Any]] = []
        for ref in order_refs:
            oid_value = ref.get("oid")
            try:
                oid_int = int(oid_value) if oid_value is not None else None
            except Exception:
                oid_int = None
            if oid_int is None or oid_int in open_oids:
                continue
            try:
                status_payload = info_client.query_order_by_oid(user, oid_int)
            except Exception:
                continue
            if not self._order_status_indicates_exchange_hit(status_payload):
                continue
            matched_refs.append(ref)
            synthetic_fills.append(
                {
                    "coin": self.symbol,
                    "oid": oid_int,
                    "cloid": ref.get("cloid"),
                    "sz": self._align_risk_close_size_for_session(rs, float(ref.get("close_size", 0.0) or 0.0)),
                    "px": snapshot.get("mid_price"),
                    "time": int(now * 1000),
                }
            )
            status_summaries.append(
                {
                    "key": str(ref.get("key", "") or ""),
                    "oid": oid_int,
                    "status": status_payload,
                }
            )

        if not synthetic_fills:
            return None

        self._audit_event(
            "risk_session_tpsl_exchange_reconciled",
            {
                "plan_name": rs.plan_name,
                "matched_refs": matched_refs,
                "statuses": status_summaries,
                "remaining_size": float(snapshot.get("size", 0.0) or 0.0),
                "at": now,
            },
        )
        return self._process_risk_session_user_fill_events(
            snapshot,
            synthetic_fills,
            now,
            source="exchange_reconcile",
        )
    def _process_risk_session_user_fill_events(
        self,
        snapshot: dict,
        fill_events: List[Dict[str, Any]],
        now: float,
        *,
        source: str = "user_fills_ws",
    ) -> Optional[str]:
        if self.risk_session is None or not fill_events:
            return None
        rs = self.risk_session
        qty_tol = self._risk_session_order_qty_tolerance()
        matched_by_key: Dict[str, Dict[str, Any]] = {}
        leg_type_sequence: List[str] = []
        for fill in list(fill_events or []):
            ref = self._find_resting_exit_order_ref_for_fill(rs, fill)
            if ref is None:
                continue
            fill_size = abs(float(safe_float(fill.get("sz"), 0.0) or 0.0))
            if fill_size <= 0.0:
                continue
            key = str(ref.get("key", "") or "")
            leg_type = str(ref.get("leg_type", "") or "")
            ref["filled_size"] = abs(float(ref.get("filled_size", 0.0) or 0.0)) + fill_size
            bucket = matched_by_key.setdefault(
                key,
                {
                    "ref": ref,
                    "leg_type": leg_type,
                    "leg_name": str(ref.get("name", "") or ""),
                    "fills": [],
                    "filled_size": 0.0,
                },
            )
            bucket["fills"].append(dict(fill))
            bucket["filled_size"] += fill_size
            if leg_type and leg_type not in leg_type_sequence:
                leg_type_sequence.append(leg_type)
        if not matched_by_key:
            return None

        size = float(snapshot.get("size", 0.0) or 0.0)
        price = safe_float(snapshot.get("mid_price"), None)
        completed_keys: set = set()
        matched_keys = list(matched_by_key.keys())
        leg_names: List[str] = []
        primary_leg_type = ""
        remaining_refs: List[Dict[str, Any]] = []
        for ref in list(rs.resting_exit_orders or []):
            key = str(ref.get("key", "") or "")
            if key not in matched_by_key:
                remaining_refs.append(ref)
                continue
            if not primary_leg_type:
                primary_leg_type = str(ref.get("leg_type", "") or "")
            leg_name = str(ref.get("name", "") or "")
            if leg_name:
                leg_names.append(leg_name)
            close_size = self._align_risk_close_size_for_session(rs, abs(float(ref.get("close_size", 0.0) or 0.0)))
            ref["close_size"] = close_size
            filled_size = abs(float(ref.get("filled_size", 0.0) or 0.0))
            if filled_size + qty_tol >= close_size:
                completed_keys.add(key)
            else:
                remaining_refs.append(ref)
        if completed_keys:
            rs.executed_leg_names.update(completed_keys)
        rs.resting_exit_orders = remaining_refs
        rs.expected_size = size
        rs.side = str(snapshot.get("side", rs.side))
        if price is not None:
            rs.prev_price = price
        print(f"[{primary_leg_type or leg_type_sequence[0]}_leg_hit] {rs.plan_name} legs={','.join(leg_names)} size={size:.8f}")
        self._audit_event(
            "management_exit_leg_hit",
            {
                "plan_name": rs.plan_name,
                "leg_type": primary_leg_type or (leg_type_sequence[0] if leg_type_sequence else ""),
                "leg_names": leg_names,
                "matched_keys": matched_keys,
                "completed_keys": list(completed_keys),
                "price": price,
                "at": now,
                "source": str(source or "user_fills_ws"),
                "remaining_size": size,
            },
        )
        if size == 0.0:
            print(f"[risk_monitor_done] {rs.plan_name} position already flat")
            self._emit_status_line("risk_monitor_done", f"持仓已平，结束风险监控: {rs.plan_name}")
            rs.pending_fill_reconcile_since = None
            clear_basis = getattr(self, "_clear_position_basis_state", None)
            if callable(clear_basis):
                clear_basis(reason="risk_session_fill_position_flat", position_snapshot=snapshot)
            self._replace_risk_session(None)
            return "position_flat"
        self._update_staged_risk_session_after_completed_keys(rs, list(completed_keys), now=now)
        for leg_type in leg_type_sequence:
            self._resync_risk_session_counterpart_orders(rs, leg_type)
        rs.pending_fill_reconcile_since = None
        rs.use_resting_exit_orders = bool(rs.resting_exit_orders)
        self._persist_risk_session_state()
        return f"{primary_leg_type or leg_type_sequence[0]}_hit"
    def _match_risk_session_reduction(
        self,
        session: RiskSession,
        new_size: float,
    ) -> Optional[Tuple[str, List[str]]]:
        delta_abs = abs(float(session.expected_size or 0.0)) - abs(float(new_size or 0.0))
        if delta_abs <= 0.0:
            return None
        qty_tol = self._risk_session_order_qty_tolerance()
        best_match: Optional[Tuple[str, List[str], float]] = None
        for leg_type in ("take_profit", "stop_loss"):
            matching_refs = [ref for ref in list(getattr(session, "resting_exit_orders", []) or []) if str(ref.get("leg_type", "") or "") == leg_type]
            cumulative = 0.0
            keys: List[str] = []
            for ref in matching_refs:
                cumulative += self._align_risk_close_size_for_session(session, abs(float(ref.get("close_size", 0.0) or 0.0)))
                keys.append(str(ref.get("key", "") or ""))
                error = abs(cumulative - delta_abs)
                if error <= qty_tol and (best_match is None or error < best_match[2]):
                    best_match = (leg_type, list(keys), error)
        if best_match is None:
            return None
        return best_match[0], best_match[1]
    def _refresh_position_management_session_from_current_playbook(
        self,
        position_snapshot: dict,
        all_positions: Optional[Dict[str, Any]] = None,
    ) -> None:
        playbook = getattr(self, "current_playbook", None)
        if not isinstance(playbook, GenericPlaybook):
            self.position_management_session = None
            return
        if all_positions is None:
            symbol = canonicalize_execution_symbol(position_snapshot.get("symbol", "") or getattr(self, "symbol", "") or "")
            if symbol and hasattr(self.reader, "get_selected_symbol_position_context"):
                context = self.reader.get_selected_symbol_position_context(symbol)
                if isinstance(context, dict):
                    all_positions = context.get("all_positions") if isinstance(context.get("all_positions"), dict) else {}
                    refreshed_snapshot = context.get("position_snapshot")
                    if isinstance(refreshed_snapshot, dict):
                        position_snapshot = refreshed_snapshot
            if all_positions is None:
                all_positions = self.reader.get_all_positions()
        self.current_playbook = self._materialize_live_position_management_from_entry_plan(
            playbook,
            position_snapshot,
            all_positions,
        )
        self._set_position_management_session_from_plan(
            self.current_playbook.position_management,
            position_snapshot,
            "position_management",
        )
        if getattr(self, "position_management_session", None) is not None:
            self._audit_event(
                "position_management_session_refreshed",
                {
                    "plan_name": "position_management",
                    "position_management": self.current_playbook.position_management.to_dict(),
                    "position_snapshot": position_snapshot,
                },
            )
    def _resolve_decision_entry_reference(
        self,
        decision: Any,
        position_snapshot: dict,
    ) -> Tuple[Optional[float], str]:
        explicit_price = safe_float(getattr(decision, "entry_price", 0.0), None) if decision is not None else None
        if explicit_price is not None and explicit_price > 0:
            return explicit_price, "llm_entry_price"
        market_mid = safe_float(position_snapshot.get("mid_price"), None)
        if market_mid is not None and market_mid > 0:
            return market_mid, "market_mid_price"
        current_entry = safe_float(position_snapshot.get("entry_price"), None)
        if current_entry is not None and current_entry > 0:
            return current_entry, "position_entry_price"
        return None, "unavailable"
    @staticmethod
    def _management_immediate_reduce_fraction(plan: PositionManagementPlan) -> float:
        if not isinstance(plan, PositionManagementPlan) or not plan.execute_now:
            return 0.0
        action_decision = getattr(plan, "action_decision", None)
        action = str(getattr(action_decision, "action", "") or "").strip()
        if action == "close":
            return 1.0
        if action == "trim":
            return min(1.0, max(0.0, float(getattr(action_decision, "close_fraction", 0.0) or 0.0)))
        return 0.0
    def _build_stop_profile_from_management(
        self,
        side: str,
        entry_price: float,
        plan: PositionManagementPlan,
        *,
        apply_immediate_reduction: bool = False,
    ) -> dict:
        immediate_reduce_fraction = self._management_immediate_reduce_fraction(plan) if apply_immediate_reduction else 0.0
        required_coverage = max(0.0, 1.0 - immediate_reduce_fraction)
        action_decision = getattr(plan, "action_decision", None)
        stop_price = max(0.0, float(getattr(action_decision, "stop_loss_price", 0.0) or 0.0))
        if stop_price <= 0.0:
            if required_coverage <= 1e-9:
                return {
                    "allowed": True,
                    "code": "no_stop_needed_after_immediate_close",
                    "stop_coverage_fraction": 0.0,
                    "required_stop_coverage_fraction": required_coverage,
                    "immediate_reduce_fraction": immediate_reduce_fraction,
                    "weighted_loss_fraction": 0.0,
                    "legs": [],
                    "management_stop_mode": "current_managed_position" if apply_immediate_reduction else "new_exposure",
                }
            return {"allowed": False, "code": "missing_stop_loss", "message": "Position management plan is missing stop_loss_price."}
        if side == "long" and stop_price >= entry_price:
            return {"allowed": False, "code": "invalid_long_stop", "message": "Long management stop-loss must be below the assumed entry price."}
        if side == "short" and stop_price <= entry_price:
            return {"allowed": False, "code": "invalid_short_stop", "message": "Short management stop-loss must be above the assumed entry price."}
        coverage = 1.0
        loss_fraction = abs(entry_price - stop_price) / max(entry_price, 1e-12)
        weighted_loss_fraction = loss_fraction
        leg_details: List[dict] = [
            {
                "name": "management_stop",
                "close_fraction": 1.0,
                "stop_price": stop_price,
                "loss_fraction": loss_fraction,
            }
        ]
        if coverage + 1e-9 < required_coverage:
            requirement_message = (
                f"after the immediate management reduction, required stop coverage is {required_coverage * 100.0:.2f}%."
                if apply_immediate_reduction
                else f"new or reversed exposure requires stop coverage of {required_coverage * 100.0:.2f}%."
            )
            return {
                "allowed": False,
                "code": "partial_stop_coverage",
                "message": (
                    f"Stop-loss legs cover only {coverage * 100.0:.2f}% of the managed position; "
                    f"{requirement_message}"
                ),
                "stop_coverage_fraction": coverage,
                "required_stop_coverage_fraction": required_coverage,
                "immediate_reduce_fraction": immediate_reduce_fraction,
                "legs": leg_details,
                "management_stop_mode": "current_managed_position" if apply_immediate_reduction else "new_exposure",
            }
        if weighted_loss_fraction <= 0:
            return {"allowed": False, "code": "zero_stop_distance", "message": "Computed stop distance is zero.", "legs": leg_details}
        return {
            "allowed": True,
            "code": "ok",
            "stop_coverage_fraction": coverage,
            "required_stop_coverage_fraction": required_coverage,
            "immediate_reduce_fraction": immediate_reduce_fraction,
            "weighted_loss_fraction": weighted_loss_fraction,
            "legs": leg_details,
            "management_stop_mode": "current_managed_position" if apply_immediate_reduction else "new_exposure",
        }
    def _build_stop_profile_from_entry_decision(self, decision: StrategyDecision, entry_price: float) -> dict:
        stop_price = safe_float(decision.stop_loss_price, 0.0) or 0.0
        if stop_price <= 0:
            return {"allowed": False, "code": "missing_stop_loss", "message": "Entry decision is missing a usable stop-loss price."}
        side = decision.action
        if side == "long" and stop_price >= entry_price:
            return {"allowed": False, "code": "invalid_long_stop", "message": "Long entry stop-loss must be below the assumed entry price."}
        if side == "short" and stop_price <= entry_price:
            return {"allowed": False, "code": "invalid_short_stop", "message": "Short entry stop-loss must be above the assumed entry price."}
        loss_fraction = abs(entry_price - stop_price) / max(entry_price, 1e-12)
        if loss_fraction <= 0:
            return {"allowed": False, "code": "zero_stop_distance", "message": "Computed stop distance is zero."}
        return {
            "allowed": True,
            "code": "ok",
            "stop_coverage_fraction": 1.0,
            "weighted_loss_fraction": loss_fraction,
            "legs": [
                {
                    "name": "entry_stop",
                    "close_fraction": 1.0,
                    "stop_price": stop_price,
                    "loss_fraction": loss_fraction,
                }
            ],
        }
    def _build_stop_profile_from_management_decision(self, decision: ManagementDecision, entry_price: float) -> dict:
        if decision.action not in MANAGEMENT_EXPOSURE_ACTION_VALUES:
            return {"allowed": False, "code": "missing_stop_plan", "message": "Management decision does not define an exposure-increasing stop-loss plan."}
        synthetic_entry = build_management_exposure_entry_decision(decision)
        return self._build_stop_profile_from_entry_decision(synthetic_entry, entry_price)
    def _apply_risk_session_passthrough_from_management(
        self,
        plan: PositionManagementPlan,
        position_after: dict,
        plan_name: str,
    ) -> bool:
        current_session = getattr(self, "risk_session", None)
        if current_session is None or not snapshot_has_open_position(position_after):
            return False
        after_side, after_size = self._snapshot_side_and_size(position_after)
        if after_side not in {"long", "short"}:
            return False
        if str(getattr(current_session, "side", "") or "") != after_side:
            return False
        qty_tol = self._risk_session_order_qty_tolerance(str(position_after.get("symbol", "") or getattr(self, "symbol", "") or ""))
        if abs(float(after_size or 0.0)) <= qty_tol:
            return False
        self._audit_event(
            "risk_session_passthrough",
            {
                "plan_name": plan_name,
                "position_management": plan.to_dict(),
                "position_after": position_after,
                "expected_size": float(getattr(current_session, "expected_size", 0.0) or 0.0),
                "baseline_size": float(getattr(current_session, "baseline_size", 0.0) or 0.0),
            },
        )
        self._log_risk_session_ready(current_session, reason="passthrough", position_after=position_after)
        return True
    def _set_risk_session_from_management(self, plan: PositionManagementPlan, position_after: dict, plan_name: str) -> None:
        if bool(getattr(plan, "risk_session_passthrough", False)) and self._apply_risk_session_passthrough_from_management(plan, position_after, plan_name):
            return
        if not position_management_plan_has_content(plan):
            self._replace_risk_session(None)
            return
        session = self._build_management_session(plan, position_after, plan_name)
        self._replace_risk_session(session)
        if self.risk_session is not None:
            self._audit_event(
                "risk_session_created",
                {
                    "plan_name": plan_name,
                    "position_management": plan.to_dict(),
                    "position_after": position_after,
                },
            )
    def _set_risk_session_after_management_decision(
        self,
        decision: ManagementDecision,
        management_plan: Optional[PositionManagementPlan],
        post_fill_risk_template: Optional[PositionManagementPlan],
        position_after: dict,
        plan_name: str,
        add_fill_entry_price: float = 0.0,
    ) -> None:
        if not snapshot_has_open_position(position_after):
            self._replace_risk_session(None)
            return
        if decision.action in MANAGEMENT_EXPOSURE_ACTION_VALUES:
            if decision.action in {"add_to_long", "add_to_short"}:
                add_plan = management_plan
                if add_plan is None or not position_management_plan_has_content(add_plan):
                    add_plan = PositionManagementPlan(
                        execute_now=False,
                        action_decision=decision,
                        scenario=None,
                    )
                risk_entry_price = safe_float(add_fill_entry_price, 0.0) or 0.0
                if risk_entry_price > 0.0:
                    self._replace_risk_session(
                        self._build_staged_risk_session_from_stop(
                            position_after=position_after,
                            plan_name=plan_name,
                            initial_entry_price=risk_entry_price,
                            stop_loss_price=max(0.0, float(getattr(decision, "stop_loss_price", 0.0) or 0.0)),
                            position_management=add_plan,
                            risk_entry_source="add_fill_avg_price",
                        )
                    )
                    risk_entry_source = "add_fill_avg_price"
                else:
                    self._set_risk_session_from_management(add_plan, position_after, plan_name)
                    risk_entry_price = safe_float(getattr(add_plan.action_decision, "entry_price", 0.0), 0.0) or 0.0
                    risk_entry_source = "strategy_entry_price"
                self._audit_event(
                    "risk_session_created" if self.risk_session is not None else "risk_session_create_skipped",
                    {
                        "plan_name": plan_name,
                        "exposure_management_decision": decision.to_dict(),
                        "position_management": add_plan.to_dict(),
                        "position_after": position_after,
                        "risk_entry_source": risk_entry_source,
                        "risk_entry_price": risk_entry_price,
                    },
                )
                return
            exposure_entry = build_management_exposure_entry_decision(decision)
            self._replace_risk_session(self._build_fallback_staged_risk_session(exposure_entry, position_after, plan_name))
            self._audit_event(
                "risk_session_created" if self.risk_session is not None else "risk_session_create_skipped",
                {
                    "plan_name": plan_name,
                    "exposure_management_decision": decision.to_dict(),
                    "position_after": position_after,
                },
            )
            return
        if management_plan is not None:
            self._set_risk_session_from_management(management_plan, position_after, plan_name)
            return
        self._replace_risk_session(None)
    @staticmethod
    def _snapshot_side_and_size(snapshot: dict) -> Tuple[str, float]:
        size = float((snapshot or {}).get("size", 0.0) or 0.0)
        side = "long" if size > 0.0 else "short" if size < 0.0 else "flat"
        return side, size
    def _requested_leverage_is_satisfied(self, position_after: dict, requested_leverage: int) -> bool:
        target_leverage = max(0, int(requested_leverage or 0))
        if not snapshot_has_open_position(position_after):
            return target_leverage <= 0
        if target_leverage <= 0:
            return False
        current_leverage = max(0.0, float((position_after or {}).get("leverage", 0.0) or 0.0))
        return abs(current_leverage - float(target_leverage)) < 1e-9
    def _management_decision_reached_effective_state(
        self,
        decision: ManagementDecision,
        position_before: dict,
        position_after: dict,
    ) -> bool:
        action = str(getattr(decision, "action", "") or "").strip()
        if not action:
            return False
        if action == "no_change":
            return self._requested_leverage_is_satisfied(position_after, int(getattr(decision, "leverage", 0) or 0))
        before_side, before_size = self._snapshot_side_and_size(position_before)
        after_side, after_size = self._snapshot_side_and_size(position_after)
        size_tol = self._risk_session_order_qty_tolerance()
        before_abs = abs(before_size)
        after_abs = abs(after_size)
        if action == "close":
            return after_side == "flat" or after_abs <= size_tol
        if action == "trim":
            return after_side == before_side and after_side in {"long", "short"} and after_abs < before_abs - size_tol
        if action in {"long", "short"}:
            return before_abs <= size_tol and after_side == action and after_abs > size_tol
        if action in {"add_to_long", "add_to_short"}:
            target_side = "long" if action == "add_to_long" else "short"
            return before_side == target_side and after_side == target_side and after_abs > before_abs + size_tol
        if action in {"reverse_to_long", "reverse_to_short"}:
            target_side = "long" if action == "reverse_to_long" else "short"
            return after_side == target_side and after_abs > size_tol
        return False
    @staticmethod
    def _extract_filled_avg_price_from_execution_result(result: Any) -> float:
        stack = [result]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                filled = item.get("filled")
                if isinstance(filled, dict):
                    for key in ("avgPx", "avg_px", "average_price", "price", "px"):
                        value = safe_float(filled.get(key), 0.0)
                        if value and value > 0.0:
                            return value
                for key in ("avgPx", "avg_px", "average_price", "price", "px"):
                    value = safe_float(item.get(key), 0.0)
                    if value and value > 0.0:
                        return value
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
        return 0.0

    @staticmethod
    def _extract_filled_size_from_execution_result(result: Any) -> float:
        stack = [result]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                filled = item.get("filled")
                if isinstance(filled, dict):
                    for key in ("totalSz", "total_sz", "total_size", "size", "sz", "qty"):
                        value = safe_float(filled.get(key), 0.0)
                        if value and value > 0.0:
                            return abs(float(value))
                if (
                    any(key in item for key in ("totalSz", "total_sz", "total_size"))
                    and any(key in item for key in ("avgPx", "avg_px", "average_price", "price", "px", "oid"))
                ):
                    for key in ("totalSz", "total_sz", "total_size"):
                        value = safe_float(item.get(key), 0.0)
                        if value and value > 0.0:
                            return abs(float(value))
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
        return 0.0

    def _retarget_staged_exit_size_basis_after_risk_reduce(self, rs: RiskSession, remaining_size_abs: float) -> None:
        old_basis = max(0.0, float(getattr(rs, "staged_exit_size_basis_abs", 0.0) or getattr(rs, "initial_size_abs", 0.0) or 0.0))
        if old_basis <= 0.0:
            rs.staged_exit_size_basis_abs = max(0.0, remaining_size_abs)
            rs.tp1_completed_size_abs = 0.0
            rs.tp2_completed_size_abs = 0.0
            return
        tp1_fraction = self._staged_tp_leg_fraction(rs, "stage_tp1")
        tp2_fraction = self._staged_tp_leg_fraction(rs, "stage_tp2")
        tp1_completed_fraction = tp1_fraction if bool(getattr(rs, "tp1_hit", False)) else min(tp1_fraction, max(0.0, float(getattr(rs, "tp1_completed_size_abs", 0.0) or 0.0)) / old_basis)
        tp2_completed_fraction = tp2_fraction if bool(getattr(rs, "tp2_hit", False)) else min(tp2_fraction, max(0.0, float(getattr(rs, "tp2_completed_size_abs", 0.0) or 0.0)) / old_basis)
        completed_fraction = min(1.0, max(0.0, tp1_completed_fraction + tp2_completed_fraction))
        active_fraction = max(1e-12, 1.0 - completed_fraction)
        new_basis = max(0.0, remaining_size_abs) / active_fraction
        rs.staged_exit_size_basis_abs = new_basis
        rs.tp1_completed_size_abs = self._align_risk_close_size_for_session(rs, new_basis * tp1_completed_fraction)
        rs.tp2_completed_size_abs = self._align_risk_close_size_for_session(rs, new_basis * tp2_completed_fraction)
    def _apply_staged_exit_early_take_profit_trim(self, rs: RiskSession, closed_size_abs: float, *, now: float) -> List[str]:
        remaining_credit = max(0.0, float(closed_size_abs or 0.0))
        completed_keys: List[str] = []
        qty_tol = self._risk_session_order_qty_tolerance()
        for leg_name, attr_name, hit_attr in [
            ("stage_tp1", "tp1_completed_size_abs", "tp1_hit"),
            ("stage_tp2", "tp2_completed_size_abs", "tp2_hit"),
        ]:
            if remaining_credit <= qty_tol:
                break
            target_size = self._staged_tp_target_size_abs(rs, leg_name)
            if target_size <= qty_tol:
                continue
            prior_completed = max(0.0, float(getattr(rs, attr_name, 0.0) or 0.0))
            if bool(getattr(rs, hit_attr, False)):
                setattr(rs, attr_name, max(prior_completed, target_size))
                continue
            unfilled_target = max(0.0, target_size - prior_completed)
            credited_size = min(remaining_credit, unfilled_target)
            updated_completed = self._align_risk_close_size_for_session(rs, prior_completed + credited_size, max_size=target_size)
            setattr(rs, attr_name, updated_completed)
            remaining_credit = max(0.0, remaining_credit - max(0.0, updated_completed - prior_completed))
            if updated_completed >= target_size - qty_tol:
                completed_keys.append(f"take_profit::{leg_name}")
        for key in completed_keys:
            rs.executed_leg_names.add(key)
        if completed_keys:
            self._update_staged_risk_session_after_completed_keys(rs, completed_keys, now=now)
        return completed_keys
    def _reuse_staged_risk_session_after_trim(
        self,
        *,
        decision: ManagementDecision,
        execution_result: Dict[str, Any],
        position_before: dict,
        position_after: dict,
    ) -> bool:
        if str(getattr(decision, "action", "") or "") != "trim":
            return False
        rs = getattr(self, "risk_session", None)
        if rs is None or not bool(getattr(rs, "staged_exit_enabled", False)):
            return False
        before_side, before_size = self._snapshot_side_and_size(position_before)
        after_side, after_size = self._snapshot_side_and_size(position_after)
        if before_side != after_side or after_side not in {"long", "short"}:
            return False
        if str(getattr(rs, "side", "") or "") != after_side:
            return False
        before_abs = abs(float(before_size or 0.0))
        after_abs = abs(float(after_size or 0.0))
        qty_tol = self._risk_session_order_qty_tolerance()
        if after_abs <= qty_tol or before_abs <= after_abs + qty_tol:
            return False
        position_delta_closed_abs = self._decimal_size_delta_abs(before_abs, after_abs)
        exchange_filled_abs = self._clamp_accounting_size_abs(
            self._extract_filled_size_from_execution_result(execution_result),
            max_size=before_abs,
        )
        closed_size_source = "exchange_fill" if exchange_filled_abs > qty_tol else "position_delta"
        closed_abs = exchange_filled_abs if exchange_filled_abs > qty_tol else position_delta_closed_abs
        fill_price = self._extract_filled_avg_price_from_execution_result(execution_result)
        if fill_price <= 0.0:
            fill_price = safe_float(position_after.get("mid_price"), 0.0) or safe_float(position_before.get("mid_price"), 0.0) or 0.0
        entry0 = max(
            0.0,
            float(getattr(rs, "initial_entry_price", 0.0) or 0.0)
            or safe_float(position_after.get("entry_price"), 0.0)
            or safe_float(position_before.get("entry_price"), 0.0)
            or 0.0,
        )
        if entry0 <= 0.0:
            self._audit_event(
                "risk_session_staged_trim_missing_initial_entry",
                {
                    "plan_name": str(getattr(rs, "plan_name", "") or ""),
                    "side": str(getattr(rs, "side", "") or ""),
                    "position_before": position_before,
                    "position_after": position_after,
                },
            )
            return False
        if float(getattr(rs, "initial_entry_price", 0.0) or 0.0) <= 0.0:
            self._audit_event(
                "risk_session_staged_trim_fallback_entry",
                {
                    "fallback_entry_price": entry0,
                    "position_before_entry_price": safe_float(position_before.get("entry_price"), 0.0),
                    "position_after_entry_price": safe_float(position_after.get("entry_price"), 0.0),
                    "plan_name": str(getattr(rs, "plan_name", "") or ""),
                    "side": str(getattr(rs, "side", "") or ""),
                },
            )
        trim_kind = "risk_reduce"
        if fill_price > 0.0 and entry0 > 0.0:
            if after_side == "long" and fill_price > entry0:
                trim_kind = "early_take_profit"
            elif after_side == "short" and fill_price < entry0:
                trim_kind = "early_take_profit"

        rs.expected_size = after_size
        rs.baseline_size = after_size
        rs.prev_price = safe_float(position_after.get("mid_price"), 0.0) or rs.prev_price
        if trim_kind == "early_take_profit":
            completed_keys = self._apply_staged_exit_early_take_profit_trim(rs, closed_abs, now=time.time())
        else:
            completed_keys = []
            self._retarget_staged_exit_size_basis_after_risk_reduce(rs, after_abs)

        if isinstance(getattr(rs, "position_management", None), PositionManagementPlan):
            rs.position_management.action_decision.action = "no_change"
            rs.position_management.action_decision.close_fraction = 0.0
            rs.position_management.action_decision.entry_price = max(0.0, float(getattr(rs, "initial_entry_price", 0.0) or 0.0))
            rs.position_management.action_decision.stop_loss_price = max(0.0, float(getattr(rs, "stop_loss_price", 0.0) or 0.0))

        self._sync_risk_session_resting_orders(rs)
        self._log_risk_session_ready(rs, reason=f"trim_{trim_kind}", position_after=position_after)
        self._persist_risk_session_state()
        self._audit_event(
            "risk_session_staged_trim_reused",
            {
                "trim_kind": trim_kind,
                "closed_size_abs": closed_abs,
                "closed_size_source": closed_size_source,
                "exchange_filled_size_abs": exchange_filled_abs,
                "position_delta_closed_size_abs": position_delta_closed_abs,
                "remaining_size_abs": after_abs,
                "fill_price": fill_price,
                "initial_entry_price": float(getattr(rs, "initial_entry_price", 0.0) or 0.0),
                "initial_stop_price": float(getattr(rs, "initial_stop_price", 0.0) or 0.0),
                "staged_exit_size_basis_abs": float(getattr(rs, "staged_exit_size_basis_abs", 0.0) or 0.0),
                "tp1_completed_size_abs": float(getattr(rs, "tp1_completed_size_abs", 0.0) or 0.0),
                "tp2_completed_size_abs": float(getattr(rs, "tp2_completed_size_abs", 0.0) or 0.0),
                "completed_keys": completed_keys,
            },
        )
        return True
