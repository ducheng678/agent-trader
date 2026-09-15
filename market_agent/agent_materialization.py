import math
import os
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from market_agent.calibration import (
    extract_raw_confidence_value,
    get_opposite_event_config,
    normalize_confidence_value,
)
from market_agent.constants import MANAGEMENT_EXPOSURE_ACTION_VALUES
from market_agent.models import (
    Condition,
    EntryScenario,
    ExecuteWhenAll,
    ExitLeg,
    ManagementDecision,
    PositionManagementPlan,
    RiskSession,
    Scenario,
    StrategyDecision,
    _coerce_observe_when_all,
)
from market_agent.playbook import GenericPlaybook
from market_agent.positions import snapshot_has_open_position
from market_agent.presentation import default_observation_starts_when, normalize_entry_price
from market_agent.runtime_views import (
    build_empty_management_decision,
    build_empty_position_management_plan,
    build_empty_strategy_decision,
)
from market_agent.symbols import canonicalize_execution_symbol
from market_agent.utils import format_query_amount, safe_float


class MaterializationMixin:
    def _derive_local_sizing(
        self,
        *,
        side: str,
        decision_context: Any,
        position_snapshot: dict,
        management_plan: Optional[PositionManagementPlan] = None,
    ) -> dict:
        entry_price, entry_reference_source = self._resolve_decision_entry_reference(decision_context, position_snapshot)
        if entry_price is None or entry_price <= 0:
            return {"allowed": False, "code": "missing_entry_price", "message": "Could not determine a valid entry reference price."}
        def build_stop_profile() -> dict:
            if isinstance(decision_context, ManagementDecision) and decision_context.action in MANAGEMENT_EXPOSURE_ACTION_VALUES:
                return self._build_stop_profile_from_management_decision(decision_context, entry_price)
            if isinstance(decision_context, StrategyDecision):
                return self._build_stop_profile_from_entry_decision(decision_context, entry_price)
            if management_plan is not None:
                return self._build_stop_profile_from_management(side, entry_price, management_plan, apply_immediate_reduction=False)
            return {"allowed": False, "code": "missing_stop_plan", "message": "No stop-loss plan is available for local sizing."}

        stop_profile = build_stop_profile()
        if not stop_profile.get("allowed"):
            return dict(stop_profile)
        symbol_for_risk = canonicalize_execution_symbol(position_snapshot.get("symbol", "") or getattr(self, "symbol", "") or "")
        stop_price = safe_float((stop_profile.get("legs") or [{}])[0].get("stop_price"), 0.0) or 0.0
        correction = self._maybe_clip_profile_stop_loss(
            side=side,
            entry_price=float(entry_price),
            stop_price=stop_price,
            symbol=symbol_for_risk,
        )
        if correction.get("applied"):
            corrected_stop = float(correction.get("stop_loss_price", 0.0) or 0.0)
            if isinstance(decision_context, (StrategyDecision, ManagementDecision)):
                decision_context.stop_loss_price = corrected_stop
            if management_plan is not None and getattr(management_plan, "action_decision", None) is not None:
                management_plan.action_decision.stop_loss_price = corrected_stop
            stop_profile = build_stop_profile()
            if not stop_profile.get("allowed"):
                failed = dict(stop_profile)
                failed["profile_r_clip_correction"] = correction
                if correction.get("liquidity_band") == "low_liquidity":
                    failed["low_liquidity_r_correction"] = correction
                return failed
            self._audit_event("profile_stop_loss_clipped", correction)
        if correction:
            stop_profile["profile_r_clip_correction"] = correction
            if correction.get("liquidity_band") == "low_liquidity":
                stop_profile["low_liquidity_r_correction"] = correction
        weighted_stop_loss_fraction = float(stop_profile.get("weighted_loss_fraction", 0.0) or 0.0)
        if weighted_stop_loss_fraction <= 0:
            return {"allowed": False, "code": "zero_stop_distance", "message": "Computed stop distance is zero."}
        fee_profile = self._build_fee_loss_profile(
            include_entry_fee=True,
            include_exit_fee=True,
        )
        weighted_fee_fraction = float(fee_profile.get("total_fee_fraction", 0.0) or 0.0)
        weighted_total_loss_fraction = weighted_stop_loss_fraction + weighted_fee_fraction
        remaining_capital_usd = self._available_perp_margin_for_symbol(position_snapshot)
        margin_basis_usd = self._margin_basis_for_target_position(position_snapshot)
        max_leverage = int(position_snapshot.get("max_leverage", 0) or 0)
        if max_leverage <= 0:
            max_leverage = int((self.reader.get_market_spec(position_snapshot.get("symbol", self.symbol)) or {}).get("max_leverage", 0) or 0)
        if max_leverage <= 0:
            return {"allowed": False, "code": "missing_max_leverage", "message": "Could not determine symbol max leverage for local sizing."}
        max_planned_loss_usd = self._current_max_planned_loss_usd(position_snapshot=position_snapshot)
        max_notional_by_loss_usd = (
            max_planned_loss_usd / weighted_total_loss_fraction if max_planned_loss_usd > 0 and weighted_total_loss_fraction > 0 else 0.0
        )
        max_notional_by_margin_usd = margin_basis_usd * max_leverage
        max_allowed_notional_usd = min(max_notional_by_loss_usd, max_notional_by_margin_usd)
        if max_allowed_notional_usd <= 0:
            return {"allowed": False, "code": "zero_notional", "message": "Local sizing produced zero allowed notional."}
        if margin_basis_usd <= 0:
            return {"allowed": False, "code": "zero_capital", "message": "No capital is available for local sizing."}
        recommended_leverage = max(1, min(max_leverage, int(math.ceil(max_allowed_notional_usd / max(margin_basis_usd, 1e-12) - 1e-12))))
        estimated_margin_used_usd = margin_basis_usd
        return {
            "allowed": True,
            "code": "ok",
            "entry_price": entry_price,
            "entry_reference_source": entry_reference_source,
            "weighted_stop_loss_fraction": weighted_stop_loss_fraction,
            "weighted_fee_fraction": weighted_fee_fraction,
            "weighted_total_loss_fraction": weighted_total_loss_fraction,
            "fee_profile": fee_profile,
            "max_notional_by_loss_usd": max_notional_by_loss_usd,
            "max_notional_by_margin_usd": max_notional_by_margin_usd,
            "max_allowed_notional_usd": max_allowed_notional_usd,
            "suggested_notional_usd": max_allowed_notional_usd,
            "requested_leverage": recommended_leverage,
            "planned_margin_used_usd": estimated_margin_used_usd,
            "planned_max_loss_usd": max_planned_loss_usd,
            "max_planned_loss_usd": max_planned_loss_usd,
            "remaining_capital_usd": remaining_capital_usd,
            "margin_basis_usd": margin_basis_usd,
            "current_margin_credit_usd": 0.0,
            "released_margin_usd": 0.0,
            "max_leverage": max_leverage,
            "stop_profile": stop_profile,
            "source": "system_risk_bounded_from_entry_and_stop",
        }
    def _build_fee_loss_profile(self, *, include_entry_fee: bool, include_exit_fee: bool) -> dict:
        getter = getattr(self.reader, "get_user_fee_rates", None)
        fee_rates = {}
        if callable(getter):
            try:
                fee_rates = dict(getter() or {})
            except KeyboardInterrupt:
                raise
            except Exception as exc:
                fee_rates = {"known": False, "error": str(exc)}
        taker_fee_rate = max(0.0, safe_float((fee_rates or {}).get("taker_fee_rate"), 0.0) or 0.0)
        entry_fee_fraction = taker_fee_rate if include_entry_fee else 0.0
        exit_fee_fraction = taker_fee_rate if include_exit_fee else 0.0
        total_fee_fraction = entry_fee_fraction + exit_fee_fraction
        return {
            "known": bool((fee_rates or {}).get("known", False) or total_fee_fraction > 0),
            "source": str((fee_rates or {}).get("source", "unavailable") or "unavailable"),
            "taker_fee_rate": taker_fee_rate,
            "entry_fee_fraction": entry_fee_fraction,
            "exit_fee_fraction": exit_fee_fraction,
            "total_fee_fraction": total_fee_fraction,
            "include_entry_fee": bool(include_entry_fee),
            "include_exit_fee": bool(include_exit_fee),
        }
    def _position_side_and_notional(self, position_snapshot: dict) -> Tuple[str, float]:
        current_sz = float(position_snapshot.get("size", 0.0) or 0.0)
        current_side = "long" if current_sz > 0 else "short" if current_sz < 0 else "flat"
        current_notional = abs(float(position_snapshot.get("notional_usd", 0.0) or 0.0))
        return current_side, current_notional
    def _position_used_leverage(self, position_snapshot: dict) -> float:
        leverage = max(0.0, float(safe_float(position_snapshot.get("leverage"), 0.0) or 0.0))
        if leverage > 0:
            return leverage
        current_notional = abs(float(position_snapshot.get("notional_usd", 0.0) or 0.0))
        margin_used = max(0.0, float(position_snapshot.get("margin_used", 0.0) or 0.0))
        if current_notional > 0.0 and margin_used > 0.0:
            derived = current_notional / max(margin_used, 1e-12)
            if math.isfinite(derived) and derived > 0.0:
                return float(derived)
        return 0.0
    @staticmethod
    def _clamp_float(value: Any, lower: float, upper: float) -> float:
        numeric = safe_float(value, None)
        if numeric is None or not math.isfinite(float(numeric)):
            return lower
        return min(max(float(numeric), lower), upper)
    def _audit_position_basis_event(self, event_type: str, payload: Dict[str, Any]) -> None:
        audit = getattr(self, "_audit_event", None)
        if callable(audit):
            audit(event_type, payload)
    def _set_position_basis_state(
        self,
        *,
        side: str,
        confidence_raw: Any,
        validity: Any,
        reason: str,
        position_snapshot: Optional[dict] = None,
    ) -> Dict[str, Any]:
        normalized_side = str(side or "").strip().lower()
        raw = extract_raw_confidence_value(confidence_raw)
        if normalized_side not in {"long", "short"} or raw is None:
            return self._clear_position_basis_state(reason=reason, position_snapshot=position_snapshot)
        raw = self._clamp_float(raw, 0.0, 1.0)
        validity_value = self._clamp_float(validity, 0.0, 1.0)
        self.position_basis_side = normalized_side
        self.position_basis_confidence_raw = raw
        self.position_basis_validity = validity_value
        for session_name in ("risk_session", "position_management_session"):
            session = getattr(self, session_name, None)
            if session is None or str(getattr(session, "side", "") or "").strip().lower() != normalized_side:
                continue
            setattr(session, "position_basis_confidence_raw", raw)
            setattr(session, "position_basis_validity", validity_value)
        payload = {
            "reason": str(reason or ""),
            "side": normalized_side,
            "position_basis_confidence_raw": raw,
            "position_basis_validity": validity_value,
            "basis_eff": raw * validity_value,
            "position_snapshot": position_snapshot if isinstance(position_snapshot, dict) else None,
        }
        self._audit_position_basis_event("position_basis_updated", payload)
        return payload
    def _clear_position_basis_state(
        self,
        *,
        reason: str,
        position_snapshot: Optional[dict] = None,
    ) -> Dict[str, Any]:
        previous = {
            "position_basis_side": str(getattr(self, "position_basis_side", "") or ""),
            "position_basis_confidence_raw": extract_raw_confidence_value(getattr(self, "position_basis_confidence_raw", None)),
            "position_basis_validity": safe_float(getattr(self, "position_basis_validity", None), None),
        }
        self.position_basis_side = ""
        self.position_basis_confidence_raw = None
        self.position_basis_validity = 0.0
        for session_name in ("risk_session", "position_management_session"):
            session = getattr(self, session_name, None)
            if session is None:
                continue
            try:
                setattr(session, "position_basis_confidence_raw", None)
                setattr(session, "position_basis_validity", 0.0)
            except Exception:
                pass
        payload = {
            "reason": str(reason or ""),
            "previous": previous,
            "position_snapshot": position_snapshot if isinstance(position_snapshot, dict) else None,
        }
        self._audit_position_basis_event("position_basis_cleared", payload)
        return payload
    def _sync_position_basis_from_session(self, session: Any, *, reason: str = "session_sync") -> Optional[Dict[str, Any]]:
        if session is None:
            return None
        side = str(getattr(session, "side", "") or "").strip().lower()
        raw = extract_raw_confidence_value(getattr(session, "position_basis_confidence_raw", None))
        if raw is None:
            raw = extract_raw_confidence_value(getattr(session, "trigger_confidence_raw", None))
        if side not in {"long", "short"} or raw is None:
            return None
        validity = safe_float(getattr(session, "position_basis_validity", None), None)
        if validity is None:
            validity = 1.0
        return self._set_position_basis_state(
            side=side,
            confidence_raw=raw,
            validity=validity,
            reason=reason,
        )
    def _current_position_basis_state(self, current_side: str) -> Dict[str, Any]:
        side = str(current_side or "").strip().lower()
        if side not in {"long", "short"}:
            return {
                "position_basis_confidence_raw": None,
                "position_basis_validity": 0.0,
                "basis_eff": None,
                "basis_source": "flat",
            }
        candidates: List[Tuple[str, Any]] = [
            ("agent", self),
            ("risk_session", getattr(self, "risk_session", None)),
            ("position_management_session", getattr(self, "position_management_session", None)),
        ]
        for source, holder in candidates:
            if holder is None:
                continue
            holder_side = str(getattr(holder, "position_basis_side", "") or getattr(holder, "side", "") or "").strip().lower()
            if holder_side and holder_side != side:
                continue
            raw = extract_raw_confidence_value(getattr(holder, "position_basis_confidence_raw", None))
            if raw is None and source == "position_management_session":
                raw = extract_raw_confidence_value(getattr(holder, "trigger_confidence_raw", None))
            if raw is None:
                continue
            validity = safe_float(getattr(holder, "position_basis_validity", None), None)
            if validity is None:
                validity = 1.0
            raw = self._clamp_float(raw, 0.0, 1.0)
            validity = self._clamp_float(validity, 0.0, 1.0)
            return {
                "position_basis_confidence_raw": raw,
                "position_basis_validity": validity,
                "basis_eff": raw * validity,
                "basis_source": source,
            }
        return {
            "position_basis_confidence_raw": None,
            "position_basis_validity": 0.0,
            "basis_eff": None,
            "basis_source": "missing",
        }
    def _decide_opposite_event_action(
        self,
        *,
        current_side: str,
        target_side: str,
        opposite_conf_raw: Optional[float],
    ) -> Dict[str, Any]:
        symbol = getattr(self, "symbol", "") or ""
        config = get_opposite_event_config(symbol)
        basis_state = self._current_position_basis_state(current_side)
        basis_eff = safe_float(basis_state.get("basis_eff"), None)
        if opposite_conf_raw is None:
            return {
                **config,
                **basis_state,
                "position_side": current_side,
                "signal_action": target_side,
                "opposite_conf_raw": None,
                "decision": "legacy_missing_confidence",
                "close_fraction": 0.0,
                "open_reverse": True,
                "reason": "missing_opposite_confidence_raw",
            }
        opposite_conf = self._clamp_float(opposite_conf_raw, 0.0, 1.0)
        trim_threshold = float(config["trim_threshold"])
        if opposite_conf < trim_threshold:
            return {
                **config,
                **basis_state,
                "position_side": current_side,
                "signal_action": target_side,
                "opposite_conf_raw": opposite_conf,
                "decision": "no_change",
                "close_fraction": 0.0,
                "open_reverse": False,
                "reason": "opposite_conf_below_trim_threshold",
            }
        if basis_eff is not None:
            reverse_threshold = max(
                float(basis_eff) + float(config["reverse_edge"]),
                float(config["min_reverse_confidence"]),
            )
            should_reverse = opposite_conf >= reverse_threshold
            reverse_reason = "opposite_conf_stronger_than_current_basis"
        else:
            reverse_threshold = float(config["unknown_basis_reverse_threshold"])
            should_reverse = opposite_conf >= reverse_threshold
            reverse_reason = "opposite_conf_exceeds_unknown_basis_reverse_threshold"
        if should_reverse:
            return {
                **config,
                **basis_state,
                "position_side": current_side,
                "signal_action": target_side,
                "opposite_conf_raw": opposite_conf,
                "reverse_threshold": reverse_threshold,
                "decision": "reverse",
                "close_fraction": 1.0,
                "open_reverse": True,
                "reason": reverse_reason,
            }
        basis_for_trim = float(basis_eff) if basis_eff is not None else 0.50
        strength = self._clamp_float(
            (opposite_conf - trim_threshold)
            / max(float(config["full_confidence"]) - trim_threshold, 0.01),
            0.0,
            1.0,
        )
        close_fraction = self._clamp_float(
            0.15 + (0.85 * strength) - (0.10 * basis_for_trim),
            0.0,
            0.95,
        )
        if close_fraction >= float(config["flatten_close_fraction"]):
            return {
                **config,
                **basis_state,
                "position_side": current_side,
                "signal_action": target_side,
                "opposite_conf_raw": opposite_conf,
                "reverse_threshold": reverse_threshold,
                "basis_for_trim": basis_for_trim,
                "strength": strength,
                "decision": "flatten",
                "close_fraction": 1.0,
                "open_reverse": False,
                "reason": "opposite_signal_strong_enough_to_flatten_but_not_reverse",
            }
        return {
            **config,
            **basis_state,
            "position_side": current_side,
            "signal_action": target_side,
            "opposite_conf_raw": opposite_conf,
            "reverse_threshold": reverse_threshold,
            "basis_for_trim": basis_for_trim,
            "strength": strength,
            "decision": "trim",
            "close_fraction": close_fraction,
            "open_reverse": False,
            "reason": "opposite_signal_valid_but_not_stronger_than_current_basis",
        }
    def _update_position_basis_after_management_execution(
        self,
        *,
        decision: ManagementDecision,
        trigger_confidence_raw: Any,
        position_before: dict,
        position_after: dict,
        accepted: bool,
    ) -> Optional[Dict[str, Any]]:
        if not accepted:
            return None
        before_side, before_notional = self._position_side_and_notional(position_before or {})
        after_side, after_notional = self._position_side_and_notional(position_after or {})
        action = str(getattr(decision, "action", "") or "").strip()
        raw = extract_raw_confidence_value(trigger_confidence_raw)
        if after_side == "flat" or not snapshot_has_open_position(position_after):
            return self._clear_position_basis_state(reason=f"management_{action or 'unknown'}_flat", position_snapshot=position_after)
        if action in {"long", "short", "reverse_to_long", "reverse_to_short"}:
            if raw is None:
                return None
            return self._set_position_basis_state(
                side=after_side,
                confidence_raw=raw,
                validity=1.0,
                reason=f"management_{action}_basis_reset",
                position_snapshot=position_after,
            )
        if action in {"add_to_long", "add_to_short"} and raw is not None:
            basis_state = self._current_position_basis_state(after_side)
            old_raw = extract_raw_confidence_value(basis_state.get("position_basis_confidence_raw"))
            old_validity = safe_float(basis_state.get("position_basis_validity"), None)
            added_notional = max(0.0, float(after_notional or 0.0) - max(0.0, float(before_notional or 0.0)))
            if old_raw is not None and before_notional > 0.0 and added_notional > 0.0:
                new_raw = ((float(old_raw) * float(before_notional)) + (float(raw) * added_notional)) / max(float(before_notional) + added_notional, 1e-12)
                new_validity = 1.0 if old_validity is None else old_validity
            else:
                new_raw = raw
                new_validity = 1.0
            return self._set_position_basis_state(
                side=after_side,
                confidence_raw=new_raw,
                validity=new_validity,
                reason=f"management_{action}_basis_weighted",
                position_snapshot=position_after,
            )
        opposite_decision = getattr(decision, "opposite_event_decision", None)
        if action == "trim" and isinstance(opposite_decision, dict):
            basis_state = self._current_position_basis_state(before_side)
            old_raw = extract_raw_confidence_value(basis_state.get("position_basis_confidence_raw"))
            if old_raw is None:
                return None
            old_validity = self._clamp_float(basis_state.get("position_basis_validity"), 0.0, 1.0)
            close_fraction = self._clamp_float(getattr(decision, "close_fraction", 0.0), 0.0, 1.0)
            decay = self._clamp_float(
                opposite_decision.get("basis_validity_decay", get_opposite_event_config(getattr(self, "symbol", ""))["basis_validity_decay"]),
                0.0,
                1.0,
            )
            new_validity = self._clamp_float(old_validity * (1.0 - decay * close_fraction), 0.0, 1.0)
            return self._set_position_basis_state(
                side=after_side,
                confidence_raw=old_raw,
                validity=new_validity,
                reason="opposite_event_trim_basis_decay",
                position_snapshot=position_after,
            )
        return None
    def _adjust_same_side_target_notional_for_comparison(
        self,
        *,
        target_side: str,
        target_notional: float,
        decision: StrategyDecision,
        position_snapshot: dict,
    ) -> float:
        adjusted_target = max(0.0, float(target_notional or 0.0))
        if target_side not in {"long", "short"} or adjusted_target <= 0.0:
            return adjusted_target
        current_entry_price = safe_float(position_snapshot.get("entry_price"), None)
        new_entry_price = safe_float(getattr(decision, "entry_price", 0.0), None)
        if current_entry_price is None or current_entry_price <= 0.0 or new_entry_price is None or new_entry_price <= 0.0:
            return adjusted_target
        raw_change_fraction = (new_entry_price - current_entry_price) / current_entry_price
        comparison_change_fraction = raw_change_fraction if target_side == "long" else -raw_change_fraction
        if comparison_change_fraction >= 0.0:
            return adjusted_target
        used_leverage = self._position_used_leverage(position_snapshot)
        if used_leverage <= 0.0:
            return adjusted_target
        numerator = 1.0 + comparison_change_fraction
        denominator = 1.0 + (used_leverage * comparison_change_fraction)
        if numerator <= 0.0 or denominator <= 0.0:
            return adjusted_target
        adjusted = adjusted_target * numerator / denominator
        if not math.isfinite(adjusted) or adjusted < 0.0:
            return adjusted_target
        return float(adjusted)
    def _current_max_planned_loss_usd(
        self,
        *,
        position_snapshot: Optional[dict] = None,
        all_positions: Optional[dict] = None,
    ) -> float:
        account_equity_usd = 0.0
        if isinstance(position_snapshot, dict):
            account_equity_usd = max(0.0, float(position_snapshot.get("account_equity_usd", 0.0) or 0.0))
        if account_equity_usd <= 0.0 and isinstance(all_positions, dict):
            account_equity_usd = max(0.0, float(all_positions.get("account_equity_usd", 0.0) or 0.0))
        max_planned_loss_ratio_raw = getattr(self, "max_planned_loss_ratio", None)
        max_planned_loss_usd_fallback_raw = getattr(
            self,
            "max_planned_loss_usd_fallback",
            getattr(self, "max_planned_loss_usd", None),
        )
        if max_planned_loss_ratio_raw is None and max_planned_loss_usd_fallback_raw is None:
            max_planned_loss_ratio_raw = os.getenv("MAX_PLANNED_LOSS_RATIO", os.getenv("MAX_PLANNED_LOSS_FRACTION", "0.33"))
        if max_planned_loss_usd_fallback_raw is None:
            max_planned_loss_usd_fallback_raw = os.getenv(
                "MAX_PLANNED_LOSS_USD",
                os.getenv("MAX_TRADE_LOSS_USD", os.getenv("PLANNED_MAX_LOSS_USD", "100")),
            )
        max_planned_loss_ratio = max(0.0, float(max_planned_loss_ratio_raw or 0.0))
        max_planned_loss_usd_fallback = max(0.0, float(max_planned_loss_usd_fallback_raw or 0.0))
        if max_planned_loss_ratio > 0.0 and account_equity_usd > 0.0:
            return account_equity_usd * max_planned_loss_ratio
        return max_planned_loss_usd_fallback
    @staticmethod
    def _available_perp_margin_for_symbol(position_snapshot: dict) -> float:
        snapshot = position_snapshot if isinstance(position_snapshot, dict) else {}
        symbol_available_margin_usd = max(
            0.0,
            float(
                snapshot.get(
                    "isolated_available_margin_usd" if bool(snapshot.get("only_isolated", False)) else "cross_available_margin_usd",
                    0.0,
                )
                or 0.0
            ),
        )
        if symbol_available_margin_usd > 0:
            return symbol_available_margin_usd
        account_available_margin_usd = max(0.0, float(snapshot.get("available_margin_usd", 0.0) or 0.0))
        if account_available_margin_usd > 0:
            return account_available_margin_usd
        return max(0.0, float(snapshot.get("remaining_capital_usd", 0.0) or 0.0))
    @staticmethod
    def _margin_basis_for_target_position(position_snapshot: dict) -> float:
        snapshot = position_snapshot if isinstance(position_snapshot, dict) else {}
        symbol_margin_basis_usd = max(
            0.0,
            float(
                snapshot.get(
                    "isolated_margin_basis_usd" if bool(snapshot.get("only_isolated", False)) else "cross_margin_basis_usd",
                    0.0,
                )
                or 0.0
            ),
        )
        if symbol_margin_basis_usd > 0:
            return symbol_margin_basis_usd
        perp_margin_basis_usd = max(0.0, float(snapshot.get("perp_account_equity_usd", 0.0) or 0.0))
        if perp_margin_basis_usd > 0:
            return perp_margin_basis_usd
        account_margin_basis_usd = max(0.0, float(snapshot.get("account_equity_usd", 0.0) or 0.0))
        if account_margin_basis_usd > 0:
            return account_margin_basis_usd
        legacy_available_margin_usd = MaterializationMixin._available_perp_margin_for_symbol(snapshot)
        legacy_current_margin_usd = max(0.0, float(snapshot.get("margin_used", 0.0) or 0.0))
        return legacy_available_margin_usd + legacy_current_margin_usd
    def _risk_direction_sign(self, side: str) -> int:
        return 1 if side == "long" else -1 if side == "short" else 0
    def _price_from_r_multiple(self, side: str, entry_price: float, risk_distance: float, r_multiple: float) -> float:
        sign = self._risk_direction_sign(side)
        if sign == 0 or entry_price <= 0.0 or risk_distance <= 0.0:
            return 0.0
        return self._align_price_for_symbol(self.symbol, entry_price + (sign * risk_distance * float(r_multiple or 0.0)))
    def _build_single_exit_leg(self, *, side: str, trigger_price: float, exit_kind: str, name: str, note: str, close_fraction: float) -> Optional[ExitLeg]:
        trigger_price = self._align_price_for_symbol(self.symbol, trigger_price)
        if trigger_price <= 0.0:
            return None
        if exit_kind == "take_profit":
            condition_type = "price_ge" if side == "long" else "price_le"
        else:
            condition_type = "price_le" if side == "long" else "price_ge"
        return ExitLeg(
            name=name,
            note=note,
            when_all=[Condition(type=condition_type, level=trigger_price, note=note)],
            close_fraction=min(max(float(close_fraction or 0.0), 0.0), 1.0),
        )
    def _build_risk_session_stage_exit_legs(
        self,
        *,
        side: str,
        entry_price: float,
        stop_loss_price: float,
        staged_exit_params: Optional[Dict[str, Any]] = None,
    ) -> Optional[Tuple[List[ExitLeg], List[ExitLeg]]]:
        if side not in {"long", "short"} or entry_price <= 0.0 or stop_loss_price <= 0.0:
            return None
        if side == "long" and stop_loss_price >= entry_price:
            return None
        if side == "short" and stop_loss_price <= entry_price:
            return None
        risk_distance = abs(float(entry_price - stop_loss_price))
        if risk_distance <= 0.0:
            return None
        params = dict(staged_exit_params or {})
        tp1_r_multiple = max(0.0, float(params.get("tp1_r_multiple", getattr(self, "risk_tp1_r_multiple", 1.0)) or 0.0))
        tp2_r_multiple = max(tp1_r_multiple, float(params.get("tp2_r_multiple", getattr(self, "risk_tp2_r_multiple", 2.0)) or 0.0))
        tp1_close_fraction = min(max(0.0, float(params.get("tp1_close_fraction", getattr(self, "risk_tp1_close_fraction", 0.30)) or 0.0)), 1.0)
        tp2_close_fraction = min(max(0.0, float(params.get("tp2_close_fraction", getattr(self, "risk_tp2_close_fraction", 0.40)) or 0.0)), 1.0)
        tp1_price = self._price_from_r_multiple(side, entry_price, risk_distance, tp1_r_multiple)
        tp2_price = self._price_from_r_multiple(side, entry_price, risk_distance, tp2_r_multiple)
        tp1_leg = self._build_single_exit_leg(
            side=side,
            trigger_price=tp1_price,
            exit_kind="take_profit",
            name="stage_tp1",
            note="staged_take_profit_tp1",
            close_fraction=tp1_close_fraction,
        )
        tp2_leg = self._build_single_exit_leg(
            side=side,
            trigger_price=tp2_price,
            exit_kind="take_profit",
            name="stage_tp2",
            note="staged_take_profit_tp2",
            close_fraction=tp2_close_fraction,
        )
        initial_stop_leg = self._build_single_exit_leg(
            side=side,
            trigger_price=stop_loss_price,
            exit_kind="stop_loss",
            name="stage_initial_stop",
            note="staged_initial_stop",
            close_fraction=1.0,
        )
        take_profit_legs = [leg for leg in [tp1_leg, tp2_leg] if leg is not None and leg.close_fraction > 0.0]
        stop_loss_legs = [leg for leg in [initial_stop_leg] if leg is not None]
        if not take_profit_legs or not stop_loss_legs:
            return None
        return take_profit_legs, stop_loss_legs
    def _fee_adjusted_exit_target_price(
        self,
        *,
        side: str,
        entry_price: float,
        target_price: float,
        include_entry_fee: bool,
    ) -> float:
        adjusted_target_price = max(0.0, float(target_price or 0.0))
        if side not in {"long", "short"} or adjusted_target_price <= 0.0 or entry_price <= 0.0:
            return adjusted_target_price
        fee_profile = self._build_fee_loss_profile(
            include_entry_fee=include_entry_fee,
            include_exit_fee=True,
        )
        total_fee_fraction = max(0.0, float(fee_profile.get("total_fee_fraction", 0.0) or 0.0))
        if total_fee_fraction <= 0.0:
            return adjusted_target_price
        fee_price_delta = entry_price * total_fee_fraction
        if side == "long":
            adjusted_target_price += fee_price_delta
        else:
            adjusted_target_price -= fee_price_delta
        return max(0.0, adjusted_target_price)
    def _align_price_for_symbol(self, symbol: str, price: float) -> float:
        raw = max(0.0, float(price or 0.0))
        if raw <= 0.0:
            return 0.0
        reader = getattr(self, "reader", None)
        if reader is not None and hasattr(reader, "align_price_to_wire"):
            try:
                return float(reader.align_price_to_wire(symbol, raw) or 0.0)
            except Exception:
                pass
        if reader is not None and hasattr(reader, "get_sz_decimals"):
            try:
                sz_decimals = int(reader.get_sz_decimals(symbol) or 0)
                return round(float(f"{raw:.5g}"), max(0, 6 - sz_decimals))
            except Exception:
                pass
        return raw
    def _normalize_condition_prices_for_symbol(self, condition: Condition, symbol: str) -> Condition:
        condition.level = self._align_price_for_symbol(symbol, condition.level)
        condition.low = self._align_price_for_symbol(symbol, condition.low)
        condition.high = self._align_price_for_symbol(symbol, condition.high)
        return condition
    def _normalize_strategy_decision_prices_for_symbol(self, decision: StrategyDecision, symbol: str) -> StrategyDecision:
        decision.entry_price = self._align_price_for_symbol(symbol, decision.entry_price)
        decision.entry_price = normalize_entry_price(decision.entry_price)
        decision.stop_loss_price = self._align_price_for_symbol(symbol, decision.stop_loss_price)
        return decision
    def _normalize_management_decision_prices_for_symbol(self, decision: ManagementDecision, symbol: str) -> ManagementDecision:
        decision.entry_price = self._align_price_for_symbol(symbol, decision.entry_price)
        decision.entry_price = normalize_entry_price(decision.entry_price)
        decision.stop_loss_price = self._align_price_for_symbol(symbol, decision.stop_loss_price)
        return decision
    def _normalize_playbook_prices_for_symbol(self, playbook: GenericPlaybook, symbol: str) -> GenericPlaybook:
        self._normalize_strategy_decision_prices_for_symbol(playbook.entry_plan.action_decision, symbol)
        for scenario in ([playbook.entry_plan.scenario] if playbook.entry_plan.scenario is not None else []):
            scenario.observe_when_all = self._normalize_observe_when_all_for_symbol(scenario.observe_when_all, symbol)
            if scenario.execute_when_all.condition is not None:
                self._normalize_condition_prices_for_symbol(scenario.execute_when_all.condition, symbol)
        self._normalize_management_decision_prices_for_symbol(playbook.position_management.action_decision, symbol)
        for scenario in ([playbook.position_management.scenario] if playbook.position_management.scenario is not None else []):
            scenario.observe_when_all = self._normalize_observe_when_all_for_symbol(scenario.observe_when_all, symbol)
            if scenario.execute_when_all.condition is not None:
                self._normalize_condition_prices_for_symbol(scenario.execute_when_all.condition, symbol)
        self._normalize_management_decision_prices_for_symbol(playbook.post_fill_risk_template.action_decision, symbol)
        for scenario in ([playbook.post_fill_risk_template.scenario] if playbook.post_fill_risk_template.scenario is not None else []):
            scenario.observe_when_all = self._normalize_observe_when_all_for_symbol(scenario.observe_when_all, symbol)
            if scenario.execute_when_all.condition is not None:
                self._normalize_condition_prices_for_symbol(scenario.execute_when_all.condition, symbol)
        return playbook
    def _estimate_live_position_target_notional_from_entry(
        self,
        decision: StrategyDecision,
        position_snapshot: dict,
        all_positions: Optional[Dict[str, Any]],
    ) -> Tuple[float, Dict[str, Any]]:
        snapshot = dict(position_snapshot or {})
        all_positions = all_positions if isinstance(all_positions, dict) else {}
        for key in ("account_equity_usd", "available_margin_usd", "withdrawable_usd", "remaining_capital_usd"):
            if key not in snapshot and key in all_positions:
                snapshot[key] = all_positions.get(key)
        fallback_notional = max(0.0, float(decision.suggested_notional_usd or 0.0))
        if decision.action not in {"long", "short"}:
            return fallback_notional, {"allowed": False, "code": "not_exposure_action"}
        sizing = self._derive_local_sizing(
            side=decision.action,
            decision_context=decision,
            position_snapshot=snapshot,
            management_plan=None,
        )
        if sizing.get("allowed"):
            return max(0.0, float(sizing.get("suggested_notional_usd", 0.0) or 0.0)), sizing
        current_notional = abs(float(snapshot.get("notional_usd", 0.0) or 0.0))
        if fallback_notional > 0:
            return fallback_notional, sizing
        return current_notional, sizing
    def _convert_entry_decision_to_management_decision(
        self,
        decision: StrategyDecision,
        position_snapshot: dict,
        all_positions: Optional[Dict[str, Any]] = None,
        allow_immediate_reverse: bool = False,
        trigger_confidence_raw: Optional[float] = None,
        debug_context: Optional[Dict[str, Any]] = None,
    ) -> ManagementDecision:
        current_side, current_notional = self._position_side_and_notional(position_snapshot)
        target_notional, sizing = self._estimate_live_position_target_notional_from_entry(decision, position_snapshot, all_positions)
        tolerance_usd = max(1.0, float(self.local_risk_tolerance_usd or 0.0))
        close_fraction_no_change_tolerance = min(max(float(getattr(self, "local_no_change_close_fraction_tolerance", 0.01) or 0.01), 0.0), 1.0)
        normalized_confidence = normalize_confidence_value(trigger_confidence_raw, symbol=self.symbol)
        action = "no_change"
        close_fraction = 0.0
        new_notional_usd = current_notional if current_side in {"long", "short"} else 0.0
        continue_entry_plan_after_close = False
        entry_action = str(decision.action or "no_trade")
        comparison_target_notional = target_notional
        target_side = entry_action if entry_action in {"long", "short"} else ""
        same_side_comparison_applied = False
        trigger_confidence_scaling_mode = "none"
        confidence_gated_no_change = False
        opposite_event_decision: Optional[Dict[str, Any]] = None
        no_change_reason = ""
        if entry_action == "no_trade":
            action = "no_change"
            no_change_reason = "entry_no_trade"
        elif entry_action in {"long", "short"}:
            is_opposite_signal = current_side in {"long", "short"} and target_side in {"long", "short"} and current_side != target_side
            if normalized_confidence is not None and normalized_confidence <= 0.0 and not is_opposite_signal:
                confidence_gated_no_change = True
                action = "no_change"
                new_notional_usd = current_notional if current_side in {"long", "short"} else 0.0
                no_change_reason = "confidence_gate"
            else:
                if current_side == target_side:
                    same_side_comparison_applied = True
                    comparison_target_notional = self._adjust_same_side_target_notional_for_comparison(
                        target_side=target_side,
                        target_notional=target_notional,
                        decision=decision,
                        position_snapshot=position_snapshot,
                    )
                    margin_cap_notional = max(0.0, float((sizing.get("max_notional_by_margin_usd") if sizing.get("allowed") else 0.0) or 0.0))
                    if margin_cap_notional > 0.0:
                        comparison_target_notional = min(comparison_target_notional, margin_cap_notional)
                    if normalized_confidence is not None:
                        trigger_confidence_scaling_mode = "same_side_delta"
                        comparison_target_notional = max(
                            0.0,
                            current_notional + normalized_confidence * (comparison_target_notional - current_notional),
                        )
                if current_side == "flat":
                    if normalized_confidence is not None:
                        trigger_confidence_scaling_mode = "flat_open_target"
                        comparison_target_notional = max(0.0, normalized_confidence * target_notional)
                    if target_notional > tolerance_usd:
                        if comparison_target_notional > tolerance_usd:
                            action = target_side
                            new_notional_usd = comparison_target_notional
                        else:
                            action = "no_change"
                            new_notional_usd = 0.0
                            no_change_reason = "flat_target_too_small"
                    else:
                        action = "no_change"
                        new_notional_usd = 0.0
                        no_change_reason = "flat_target_too_small"
                elif current_side == target_side:
                    if comparison_target_notional > current_notional + tolerance_usd:
                        action = "add_to_long" if target_side == "long" else "add_to_short"
                        new_notional_usd = comparison_target_notional
                    elif current_notional > tolerance_usd and comparison_target_notional <= tolerance_usd:
                        close_fraction = 1.0
                        action = "close"
                        new_notional_usd = 0.0
                    elif current_notional > tolerance_usd and comparison_target_notional < current_notional - tolerance_usd:
                        close_fraction = min(max((current_notional - comparison_target_notional) / max(current_notional, 1e-12), 0.0), 1.0)
                        if close_fraction >= 0.999999:
                            action = "close"
                            new_notional_usd = 0.0
                        elif close_fraction < close_fraction_no_change_tolerance:
                            action = "no_change"
                            close_fraction = 0.0
                            new_notional_usd = current_notional
                            no_change_reason = "same_side_delta_too_small"
                        else:
                            action = "trim"
                            new_notional_usd = comparison_target_notional
                    else:
                        action = "no_change"
                        new_notional_usd = current_notional
                        no_change_reason = "same_side_delta_too_small"
                else:
                    if normalized_confidence is not None:
                        trigger_confidence_scaling_mode = "reverse_new_side"
                        comparison_target_notional = max(0.0, normalized_confidence * target_notional)
                    raw_opposite_confidence = extract_raw_confidence_value(trigger_confidence_raw)
                    opposite_event_decision = self._decide_opposite_event_action(
                        current_side=current_side,
                        target_side=target_side,
                        opposite_conf_raw=raw_opposite_confidence,
                    )
                    opposite_decision = str((opposite_event_decision or {}).get("decision", "") or "")
                    if opposite_decision == "no_change":
                        action = "no_change"
                        new_notional_usd = current_notional
                        no_change_reason = str((opposite_event_decision or {}).get("reason", "") or "opposite_event_no_change")
                    elif opposite_decision == "reverse":
                        if comparison_target_notional > tolerance_usd and allow_immediate_reverse:
                            action = "reverse_to_long" if target_side == "long" else "reverse_to_short"
                            new_notional_usd = comparison_target_notional
                        elif comparison_target_notional > tolerance_usd:
                            action = "close"
                            close_fraction = 1.0
                            new_notional_usd = 0.0
                            continue_entry_plan_after_close = True
                        else:
                            action = "close"
                            close_fraction = 1.0
                            new_notional_usd = 0.0
                            continue_entry_plan_after_close = False
                            no_change_reason = "opposite_reverse_target_too_small_flatten"
                    elif opposite_decision == "flatten":
                        action = "close"
                        close_fraction = 1.0
                        new_notional_usd = 0.0
                        continue_entry_plan_after_close = False
                    elif opposite_decision == "trim":
                        close_fraction = min(max(float((opposite_event_decision or {}).get("close_fraction", 0.0) or 0.0), 0.0), 1.0)
                        if close_fraction >= 0.999999:
                            action = "close"
                            new_notional_usd = 0.0
                            continue_entry_plan_after_close = False
                        elif close_fraction < close_fraction_no_change_tolerance:
                            action = "no_change"
                            close_fraction = 0.0
                            new_notional_usd = current_notional
                            no_change_reason = "opposite_trim_too_small"
                        else:
                            action = "trim"
                            new_notional_usd = max(0.0, current_notional * (1.0 - close_fraction))
                    else:
                        if comparison_target_notional > tolerance_usd and allow_immediate_reverse:
                            action = "reverse_to_long" if target_side == "long" else "reverse_to_short"
                            new_notional_usd = comparison_target_notional
                        elif comparison_target_notional > tolerance_usd:
                            action = "close"
                            close_fraction = 1.0
                            new_notional_usd = 0.0
                            continue_entry_plan_after_close = True
                        else:
                            action = "close"
                            close_fraction = 1.0
                            new_notional_usd = 0.0
                            continue_entry_plan_after_close = False
        low_liquidity_trade_block: Optional[Dict[str, Any]] = None
        if action in MANAGEMENT_EXPOSURE_ACTION_VALUES:
            symbol_for_profile = canonicalize_execution_symbol(
                position_snapshot.get("symbol", "") or getattr(self, "symbol", "") or ""
            )
            low_liquidity_profile = self._low_liquidity_trade_disabled_profile_for_symbol(symbol_for_profile)
            if low_liquidity_profile is not None:
                low_liquidity_trade_block = {
                    "code": "low_liquidity_trading_disabled",
                    "profile": low_liquidity_profile.name,
                    "symbol": symbol_for_profile,
                    "low_liquidity_source": "trade_disabled_weekday",
                    "low_liquidity_trade_disabled_weekdays": list(
                        low_liquidity_profile.low_liquidity_trade_disabled_weekdays or ()
                    ),
                    "original_action": action,
                    "current_side": current_side,
                    "current_notional_usd": current_notional,
                    "target_notional_usd": target_notional,
                    "comparison_target_notional_usd": comparison_target_notional,
                    "entry_action": entry_action,
                }
                self._audit_event("low_liquidity_trade_blocked", low_liquidity_trade_block)
                action = "no_change"
                close_fraction = 0.0
                new_notional_usd = current_notional if current_side in {"long", "short"} else 0.0
                continue_entry_plan_after_close = False
                no_change_reason = "low_liquidity_trading_disabled"
        if debug_context is not None:
            debug_context.update(
                {
                    "current_side": current_side,
                    "current_notional_usd": current_notional,
                    "target_side": target_side,
                    "target_notional_usd": target_notional,
                    "trigger_confidence_raw": extract_raw_confidence_value(trigger_confidence_raw),
                    "trigger_confidence": normalized_confidence,
                    "trigger_confidence_scaling_mode": trigger_confidence_scaling_mode,
                    "comparison_target_notional_usd": comparison_target_notional,
                    "confidence_gated_no_change": confidence_gated_no_change,
                    "opposite_event_decision": opposite_event_decision,
                    "no_change_reason": no_change_reason,
                    "no_change_should_refresh": no_change_reason == "same_side_delta_too_small",
                    "low_liquidity_trade_block": low_liquidity_trade_block,
                    "same_side_comparison_applied": same_side_comparison_applied,
                    "tolerance_usd": tolerance_usd,
                    "materialized_action": action,
                }
            )
        if action == "close" and current_side in {"long", "short"}:
            continue_entry_plan_after_close = True
        margin_basis_usd = float((sizing.get("margin_basis_usd") if sizing.get("allowed") else 0.0) or 0.0)
        leverage = (
            max(1, int(math.ceil(max(0.0, float(new_notional_usd or 0.0)) / max(margin_basis_usd, 1e-12) - 1e-12)))
            if max(0.0, float(new_notional_usd or 0.0)) > 0 and margin_basis_usd > 0
            else 0
        )
        max_leverage = int((sizing.get("max_leverage") if sizing.get("allowed") else 0) or 0)
        if leverage > 0 and max_leverage > 0:
            leverage = min(leverage, max_leverage)
        planned_max_loss_usd = float((sizing.get("max_planned_loss_usd") if sizing.get("allowed") else decision.planned_max_loss_usd) or 0.0)
        materialized_decision = ManagementDecision(
            action=action,
            close_fraction=close_fraction,
            new_notional_usd=max(0.0, float(new_notional_usd or 0.0)),
            entry_price=float(decision.entry_price or 0.0),
            stop_loss_price=float(decision.stop_loss_price or 0.0),
            planned_max_loss_usd=planned_max_loss_usd,
            leverage=leverage,
            margin_basis_usd=margin_basis_usd,
            continue_entry_plan_after_close=continue_entry_plan_after_close,
        )
        if isinstance(opposite_event_decision, dict):
            setattr(materialized_decision, "opposite_event_decision", dict(opposite_event_decision))
        return materialized_decision
    def _convert_entry_scenario_to_management_scenario(
        self,
        scenario: EntryScenario,
    ) -> Optional[Scenario]:
        return Scenario(
            observe_when_all=_coerce_observe_when_all(scenario.observe_when_all),
            execute_when_all=ExecuteWhenAll(
                condition=scenario.execute_when_all.condition,
                timeout_seconds=int(scenario.execute_when_all.timeout_seconds or 0),
            ),
            observation_starts_when=default_observation_starts_when(scenario.observe_when_all, scenario.execute_when_all.condition),
        )
    @staticmethod
    def _extract_open_order_trigger_price(item: Dict[str, Any]) -> float:
        trigger_price = safe_float((item or {}).get("triggerPx"), 0.0) or 0.0
        if trigger_price > 0.0:
            return trigger_price
        order_type = (item or {}).get("orderType")
        if isinstance(order_type, dict):
            trigger_payload = dict(order_type.get("trigger") or {})
            return safe_float(trigger_payload.get("triggerPx"), 0.0) or 0.0
        trigger_payload = dict((item or {}).get("trigger") or {})
        return safe_float(trigger_payload.get("triggerPx"), 0.0) or 0.0
    @staticmethod
    def _extract_open_order_limit_price(item: Dict[str, Any]) -> float:
        for key in ("limitPx", "limit_px", "px", "price"):
            limit_price = safe_float((item or {}).get(key), 0.0) or 0.0
            if limit_price > 0.0:
                return limit_price
        order_payload = dict((item or {}).get("order") or {})
        for key in ("limitPx", "limit_px", "px", "price"):
            limit_price = safe_float(order_payload.get(key), 0.0) or 0.0
            if limit_price > 0.0:
                return limit_price
        return 0.0
    @classmethod
    def _extract_open_order_exit_price(cls, item: Dict[str, Any]) -> float:
        if bool((item or {}).get("isTrigger", False)):
            return cls._extract_open_order_trigger_price(item)
        return cls._extract_open_order_limit_price(item)
    @staticmethod
    def _extract_open_order_time_ms(item: Dict[str, Any]) -> int:
        for key in ("timestamp", "time", "createdAt", "created_at", "createdTime", "orderTime", "statusTimestamp"):
            value = int(safe_float((item or {}).get(key), 0.0) or 0)
            if value > 0:
                return value
        order_payload = dict((item or {}).get("order") or {})
        for key in ("timestamp", "time", "createdAt", "created_at", "createdTime", "orderTime", "statusTimestamp"):
            value = int(safe_float(order_payload.get(key), 0.0) or 0)
            if value > 0:
                return value
        return 0
    @staticmethod
    def _extract_open_order_tpsl(item: Dict[str, Any]) -> str:
        direct = str((item or {}).get("tpsl", "") or "").strip().lower()
        if direct in {"tp", "sl"}:
            return direct
        order_type = (item or {}).get("orderType")
        if isinstance(order_type, dict):
            trigger_payload = dict(order_type.get("trigger") or {})
            nested = str(trigger_payload.get("tpsl", "") or "").strip().lower()
            if nested in {"tp", "sl"}:
                return nested
        trigger_payload = dict((item or {}).get("trigger") or {})
        nested = str(trigger_payload.get("tpsl", "") or "").strip().lower()
        if nested in {"tp", "sl"}:
            return nested
        order_type_text = str(order_type or "").strip().lower()
        if order_type_text == "take profit market":
            return "tp"
        if order_type_text == "stop market":
            return "sl"
        return ""
    def _infer_tpsl_from_trigger_price(self, side: str, trigger_price: float, entry_price: float) -> str:
        if side not in {"long", "short"} or trigger_price <= 0.0 or entry_price <= 0.0:
            return ""
        if side == "long":
            return "tp" if trigger_price >= entry_price else "sl"
        return "tp" if trigger_price <= entry_price else "sl"
    @staticmethod
    def _trigger_price_is_loss_side(side: str, trigger_price: float, entry_price: float) -> bool:
        if side == "long":
            return 0.0 < trigger_price < entry_price
        if side == "short":
            return trigger_price > entry_price > 0.0
        return False
    @staticmethod
    def _restore_price_matches(expected: float, actual: float) -> bool:
        if expected <= 0.0 or actual <= 0.0:
            return False
        return abs(float(expected) - float(actual)) <= max(1e-8, abs(float(expected)) * 1e-5)
    def _match_exchange_order_refs_subset_to_risk_session_specs(
        self,
        session: RiskSession,
        exchange_order_refs: List[Dict[str, Any]],
        symbol: str = "",
        *,
        check_sizes: bool = True,
    ) -> Optional[List[Dict[str, Any]]]:
        desired_specs = self._iter_risk_session_exit_order_specs(session)
        remaining_specs = [dict(item) for item in list(desired_specs or []) if isinstance(item, dict)]
        refs = [dict(item) for item in list(exchange_order_refs or []) if isinstance(item, dict)]
        if not remaining_specs or len(refs) > len(remaining_specs):
            return None
        matched_refs: List[Dict[str, Any]] = []
        qty_tol = max(self._risk_session_order_qty_tolerance(symbol), abs(float(session.initial_size_abs or 0.0)) * 0.01)
        for ref in refs:
            match_index: Optional[int] = None
            for index, spec in enumerate(remaining_specs):
                if str(ref.get("tpsl", "") or "") != str(spec.get("tpsl", "") or ""):
                    continue
                if not self._restore_price_matches(
                    float(spec.get("trigger_price", 0.0) or 0.0),
                    float(ref.get("trigger_price", 0.0) or 0.0),
                ):
                    continue
                spec_size = max(0.0, float(spec.get("close_size", 0.0) or 0.0))
                ref_size = max(0.0, float(ref.get("close_size", 0.0) or 0.0))
                if check_sizes and ref_size > 0.0 and spec_size > 0.0 and abs(ref_size - spec_size) > qty_tol:
                    continue
                match_index = index
                break
            if match_index is None:
                return None
            matched_spec = dict(remaining_specs.pop(match_index))
            matched_ref = dict(ref)
            matched_ref.update(
                {
                    "key": str(matched_spec.get("key", "") or ""),
                    "name": str(matched_spec.get("name", "") or ""),
                    "leg_type": str(matched_spec.get("leg_type", "") or ""),
                    "tpsl": str(matched_spec.get("tpsl", "") or ""),
                    "trigger_price": float(matched_spec.get("trigger_price", 0.0) or 0.0),
                    "close_size": float(matched_spec.get("close_size", 0.0) or matched_ref.get("close_size", 0.0) or 0.0),
                }
            )
            matched_refs.append(matched_ref)
        return matched_refs
    def _exchange_take_profit_refs_match_risk_session_specs(
        self,
        session: RiskSession,
        exchange_order_refs: List[Dict[str, Any]],
    ) -> bool:
        specs = [dict(item) for item in list(self._iter_risk_session_exit_order_specs(session) or []) if isinstance(item, dict)]
        refs = [dict(item) for item in list(exchange_order_refs or []) if isinstance(item, dict)]
        if not specs:
            return bool(refs)
        take_profit_refs = [ref for ref in refs if str(ref.get("leg_type", "") or "") == "take_profit"]
        if not take_profit_refs:
            return False
        take_profit_specs = [spec for spec in specs if str(spec.get("leg_type", "") or "") == "take_profit"]
        if not take_profit_specs:
            return False
        for ref in take_profit_refs:
            matched = False
            for spec in take_profit_specs:
                if str(ref.get("tpsl", "") or "") != str(spec.get("tpsl", "") or ""):
                    continue
                if self._restore_price_matches(
                    float(spec.get("trigger_price", 0.0) or 0.0),
                    float(ref.get("trigger_price", 0.0) or 0.0),
                ):
                    matched = True
                    break
            if not matched:
                return False
        return True
    @staticmethod
    def _restored_exit_order_refs_need_resync(order_refs: List[Dict[str, Any]]) -> bool:
        return any(
            str(item.get("leg_type", "") or "") == "take_profit"
            and str(item.get("order_kind", "") or "").lower() != "limit"
            for item in list(order_refs or [])
            if isinstance(item, dict)
        )
    def _finalize_restored_risk_session_order_refs(
        self,
        session: RiskSession,
        exchange_order_refs: List[Dict[str, Any]],
        symbol: str,
        *,
        allow_resync: bool,
    ) -> Optional[RiskSession]:
        matched_refs = self._match_exchange_order_refs_subset_to_risk_session_specs(
            session,
            exchange_order_refs,
            symbol,
            check_sizes=True,
        )
        loose_matched_refs = matched_refs
        if loose_matched_refs is None and allow_resync:
            loose_matched_refs = self._match_exchange_order_refs_subset_to_risk_session_specs(
                session,
                exchange_order_refs,
                symbol,
                check_sizes=False,
            )
        if loose_matched_refs is None:
            if allow_resync and self._exchange_take_profit_refs_match_risk_session_specs(session, exchange_order_refs):
                session.resting_exit_orders = [dict(item) for item in list(exchange_order_refs or []) if isinstance(item, dict)]
                session.use_resting_exit_orders = bool(session.resting_exit_orders)
                setattr(session, "_startup_restore_needs_order_resync", bool(session.resting_exit_orders))
                return session
            return None
        desired_count = len(self._iter_risk_session_exit_order_specs(session))
        if matched_refs is not None and len(matched_refs) == desired_count:
            needs_order_resync = self._restored_exit_order_refs_need_resync(matched_refs)
            if needs_order_resync and not allow_resync:
                return None
            session.resting_exit_orders = matched_refs
            session.use_resting_exit_orders = True
            setattr(session, "_startup_restore_needs_order_resync", needs_order_resync)
            return session
        if not allow_resync:
            return None
        session.resting_exit_orders = [dict(item) for item in list(exchange_order_refs or []) if isinstance(item, dict)]
        session.use_resting_exit_orders = bool(session.resting_exit_orders)
        setattr(session, "_startup_restore_needs_order_resync", bool(session.resting_exit_orders))
        return session
    def _build_startup_post_tp1_risk_session(
        self,
        position_snapshot: dict,
        *,
        side: str,
        initial_entry_price: float,
        post_tp1_stop_price: float,
        tp2_order_ref: Dict[str, Any],
        staged_exit_params_override: Optional[Dict[str, Any]] = None,
    ) -> Optional[RiskSession]:
        symbol = canonicalize_execution_symbol(position_snapshot.get("symbol", "") or getattr(self, "symbol", "") or "")
        staged_exit_params = dict(staged_exit_params_override) if isinstance(staged_exit_params_override, dict) else (self._staged_exit_params_for_symbol(symbol) if hasattr(self, "_staged_exit_params_for_symbol") else {})
        post_tp1_multiple = float(staged_exit_params.get("post_tp1_stop_r_multiple", getattr(self, "risk_post_tp1_stop_r_multiple", -0.40)) or -0.40)
        if post_tp1_multiple == 0.0:
            return None
        risk_distance = abs(float(initial_entry_price - post_tp1_stop_price)) / abs(post_tp1_multiple)
        if risk_distance <= 0.0:
            return None
        initial_stop_price = self._price_from_r_multiple(side, initial_entry_price, risk_distance, -1.0)
        tp1_r_multiple = float(staged_exit_params.get("tp1_r_multiple", getattr(self, "risk_tp1_r_multiple", 1.0)) or 1.0)
        tp2_r_multiple = float(staged_exit_params.get("tp2_r_multiple", getattr(self, "risk_tp2_r_multiple", 2.0)) or 2.0)
        tp1_price = self._price_from_r_multiple(side, initial_entry_price, risk_distance, tp1_r_multiple)
        tp2_price = self._price_from_r_multiple(side, initial_entry_price, risk_distance, tp2_r_multiple)
        if not self._restore_price_matches(tp2_price, float(tp2_order_ref.get("trigger_price", 0.0) or 0.0)):
            return None
        tp1_fraction = min(max(0.0, float(staged_exit_params.get("tp1_close_fraction", getattr(self, "risk_tp1_close_fraction", 0.30)) or 0.0)), 1.0)
        tp2_fraction = min(max(0.0, float(staged_exit_params.get("tp2_close_fraction", getattr(self, "risk_tp2_close_fraction", 0.40)) or 0.0)), 1.0)
        remaining_size_abs = abs(float(position_snapshot.get("size", 0.0) or 0.0))
        initial_size_abs = remaining_size_abs / max(1.0 - tp1_fraction, 1e-12)
        tp2_ref_size = max(0.0, float(tp2_order_ref.get("close_size", 0.0) or 0.0))
        if tp2_ref_size > 0.0 and tp2_fraction > 0.0:
            initial_size_abs = max(initial_size_abs, tp2_ref_size / tp2_fraction)
        tp2_leg = self._build_single_exit_leg(
            side=side,
            trigger_price=tp2_price,
            exit_kind="take_profit",
            name="stage_tp2",
            note="staged_take_profit_tp2",
            close_fraction=tp2_fraction,
        )
        take_profit_legs = [tp2_leg] if tp2_leg is not None else []
        if not take_profit_legs:
            return None
        risk_plan = PositionManagementPlan(
            execute_now=False,
            action_decision=ManagementDecision(
                action="no_change",
                close_fraction=0.0,
                new_notional_usd=max(0.0, float(position_snapshot.get("notional_usd", 0.0) or 0.0)),
                entry_price=max(0.0, float(initial_entry_price or 0.0)),
                stop_loss_price=post_tp1_stop_price,
                planned_max_loss_usd=0.0,
                leverage=max(0, int(position_snapshot.get("leverage", 0) or 0)),
                margin_basis_usd=max(0.0, float(position_snapshot.get("margin_used", 0.0) or 0.0)),
            ),
            scenario=None,
        )
        session = RiskSession(
            plan_name="startup_exchange_restore",
            side=side,
            stop_loss_price=post_tp1_stop_price,
            start_time=time.time(),
            baseline_size=float(position_snapshot.get("size", 0.0) or 0.0),
            position_management=risk_plan,
            expected_size=float(position_snapshot.get("size", 0.0) or 0.0),
            initial_size_abs=initial_size_abs,
            take_profit_legs=list(take_profit_legs),
            stop_loss_legs=[],
            runtimes={},
            history_seconds=float(getattr(self, "price_history_seconds", 1800) or 1800),
            executed_leg_names={"take_profit::stage_tp1"},
            take_profit_legs_scale_from_initial_size=True,
            staged_exit_enabled=True,
            staged_exit_size_basis_abs=initial_size_abs,
            tp1_completed_size_abs=self._align_risk_exit_size_to_precision(initial_size_abs * tp1_fraction, symbol),
            initial_entry_price=initial_entry_price,
            initial_stop_price=initial_stop_price,
            initial_risk_price_distance=risk_distance,
            tp1_price=tp1_price,
            tp2_price=tp2_price,
            tp1_hit=True,
            tp2_hit=False,
            tp1_hit_at=max(0.0, float(int(safe_float(tp2_order_ref.get("created_time_ms"), 0.0) or 0) / 1000.0)),
            active_soft_stop_price=post_tp1_stop_price,
            active_hard_stop_price=0.0,
            post_tp1_stop_price=post_tp1_stop_price,
            locked_floor_price=self._price_from_r_multiple(
                side,
                initial_entry_price,
                risk_distance,
                float(staged_exit_params.get("post_tp2_locked_r_multiple", getattr(self, "risk_post_tp2_locked_r_multiple", 1.0)) or 1.0),
            ),
            trailing_timeframe=str(getattr(self, "risk_trailing_timeframe", "15m") or "15m"),
            trailing_atr_period=int(getattr(self, "risk_trailing_atr_period", 14) or 14),
            trailing_atr_lookback_bars=int(getattr(self, "risk_trailing_atr_lookback_bars", 200) or 200),
            trailing_soft_atr_mult=float(staged_exit_params.get("trailing_soft_atr_multiple", getattr(self, "risk_trailing_soft_atr_multiple", 2.5)) or 2.5),
            trailing_hard_atr_mult=float(staged_exit_params.get("trailing_hard_atr_multiple", getattr(self, "risk_trailing_hard_atr_multiple", 3.5)) or 3.5),
        )
        self._normalize_risk_session_size_state(session)
        return session
    def _build_startup_tail_risk_session(
        self,
        position_snapshot: dict,
        *,
        side: str,
        initial_entry_price: float,
        hard_stop_price: float,
        staged_exit_params_override: Optional[Dict[str, Any]] = None,
    ) -> Optional[RiskSession]:
        symbol = canonicalize_execution_symbol(position_snapshot.get("symbol", "") or getattr(self, "symbol", "") or "")
        staged_exit_params = dict(staged_exit_params_override) if isinstance(staged_exit_params_override, dict) else (self._staged_exit_params_for_symbol(symbol) if hasattr(self, "_staged_exit_params_for_symbol") else {})
        risk_plan = PositionManagementPlan(
            execute_now=False,
            action_decision=ManagementDecision(
                action="no_change",
                close_fraction=0.0,
                new_notional_usd=max(0.0, float(position_snapshot.get("notional_usd", 0.0) or 0.0)),
                entry_price=max(0.0, float(initial_entry_price or 0.0)),
                stop_loss_price=hard_stop_price,
                planned_max_loss_usd=0.0,
                leverage=max(0, int(position_snapshot.get("leverage", 0) or 0)),
                margin_basis_usd=max(0.0, float(position_snapshot.get("margin_used", 0.0) or 0.0)),
            ),
            scenario=None,
        )
        baseline_size = float(position_snapshot.get("size", 0.0) or 0.0)
        return RiskSession(
            plan_name="startup_exchange_restore",
            side=side,
            stop_loss_price=hard_stop_price,
            start_time=time.time(),
            baseline_size=baseline_size,
            position_management=risk_plan,
            expected_size=baseline_size,
            initial_size_abs=abs(baseline_size),
            take_profit_legs=[],
            stop_loss_legs=[],
            runtimes={},
            history_seconds=float(getattr(self, "price_history_seconds", 1800) or 1800),
            take_profit_legs_scale_from_initial_size=True,
            staged_exit_enabled=True,
            staged_exit_size_basis_abs=abs(baseline_size),
            initial_entry_price=initial_entry_price,
            initial_stop_price=hard_stop_price,
            initial_risk_price_distance=abs(float(initial_entry_price - hard_stop_price)),
            tp1_hit=True,
            tp2_hit=True,
            locked_floor_price=hard_stop_price,
            trailing_soft_stop_price=hard_stop_price,
            trailing_hard_stop_price=0.0,
            trailing_timeframe=str(getattr(self, "risk_trailing_timeframe", "15m") or "15m"),
            trailing_atr_period=int(getattr(self, "risk_trailing_atr_period", 14) or 14),
            trailing_atr_lookback_bars=int(getattr(self, "risk_trailing_atr_lookback_bars", 200) or 200),
            trailing_soft_atr_mult=float(staged_exit_params.get("trailing_soft_atr_multiple", getattr(self, "risk_trailing_soft_atr_multiple", 2.5)) or 2.5),
            trailing_hard_atr_mult=float(staged_exit_params.get("trailing_hard_atr_multiple", getattr(self, "risk_trailing_hard_atr_multiple", 3.5)) or 3.5),
        )

    def _exchange_reduce_only_order_refs_from_snapshot(
        self,
        position_snapshot: dict,
        *,
        entry_price: Optional[float] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not snapshot_has_open_position(position_snapshot):
            return [], [], [], []
        side = str(position_snapshot.get("side", "flat") or "flat")
        if side not in {"long", "short"}:
            return [], [], [], []
        symbol = canonicalize_execution_symbol(position_snapshot.get("symbol", self.symbol))
        ref_entry_price = (
            safe_float(entry_price, 0.0)
            or safe_float(position_snapshot.get("entry_price"), 0.0)
            or safe_float(position_snapshot.get("mid_price"), 0.0)
            or 0.0
        )
        try:
            open_orders = list(self.reader.get_frontend_open_orders(symbol) or [])
        except Exception:
            return [], [], [], []

        exchange_order_refs: List[Dict[str, Any]] = []
        take_profit_refs: List[Dict[str, Any]] = []
        stop_refs: List[Dict[str, Any]] = []
        stop_loss_refs: List[Dict[str, Any]] = []
        for raw_item in open_orders:
            if not isinstance(raw_item, dict):
                continue
            if canonicalize_execution_symbol(raw_item.get("coin") or "") != symbol:
                continue
            if not bool(raw_item.get("reduceOnly", False)):
                continue
            is_trigger = bool(raw_item.get("isTrigger", False))
            trigger_price = self._extract_open_order_exit_price(raw_item)
            if trigger_price <= 0.0:
                continue
            tpsl = self._extract_open_order_tpsl(raw_item) or self._infer_tpsl_from_trigger_price(side, trigger_price, ref_entry_price)
            if tpsl not in {"tp", "sl"}:
                continue
            if not is_trigger and tpsl != "tp":
                continue
            leg_type = "take_profit" if tpsl == "tp" else "stop_loss"
            close_size = max(
                0.0,
                float(
                    safe_float(raw_item.get("sz"), None)
                    or safe_float(raw_item.get("origSz"), None)
                    or safe_float(raw_item.get("size"), 0.0)
                    or 0.0
                ),
            )
            ref = {
                "key": f"exchange_restore::{leg_type}::{len(exchange_order_refs) + 1}",
                "name": f"exchange_restore_{'tp' if tpsl == 'tp' else 'sl'}_{len(exchange_order_refs) + 1}",
                "leg_type": leg_type,
                "tpsl": tpsl,
                "trigger_price": float(trigger_price),
                "limit_price": float(trigger_price) if not is_trigger else 0.0,
                "close_size": float(close_size),
                "oid": raw_item.get("oid"),
                "cloid": str(raw_item.get("cloid", "") or "").strip(),
                "created_time_ms": self._extract_open_order_time_ms(raw_item),
                "order_kind": "trigger" if is_trigger else "limit",
                "is_trigger": is_trigger,
            }
            exchange_order_refs.append(ref)
            if leg_type == "take_profit":
                take_profit_refs.append(ref)
            else:
                stop_refs.append(ref)
                if self._trigger_price_is_loss_side(side, trigger_price, ref_entry_price):
                    stop_loss_refs.append(ref)
        return exchange_order_refs, take_profit_refs, stop_refs, stop_loss_refs

    @staticmethod
    def _order_ref_identity_matches(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        left_oid = left.get("oid")
        right_oid = right.get("oid")
        if left_oid is not None and right_oid is not None:
            try:
                if int(left_oid) == int(right_oid):
                    return True
            except Exception:
                pass
        left_cloid = str(left.get("cloid", "") or "").strip().lower()
        right_cloid = str(right.get("cloid", "") or "").strip().lower()
        return bool(left_cloid and right_cloid and left_cloid == right_cloid)

    @staticmethod
    def _startup_restore_fill_time_ms(fill: Dict[str, Any]) -> int:
        return int(safe_float((fill or {}).get("time"), 0.0) or 0.0)
    @staticmethod
    def _startup_restore_ms_to_utc_datetime(value_ms: Any) -> Optional[datetime]:
        value = int(safe_float(value_ms, 0.0) or 0)
        if value <= 0:
            return None
        try:
            return datetime.fromtimestamp(value / 1000.0, tz=timezone.utc)
        except Exception:
            return None

    @staticmethod
    def _startup_restore_order_refs_time_ms(order_refs: List[Dict[str, Any]]) -> int:
        values = [
            int(safe_float((item or {}).get("created_time_ms"), 0.0) or 0)
            for item in list(order_refs or [])
            if int(safe_float((item or {}).get("created_time_ms"), 0.0) or 0) > 0
        ]
        return min(values) if values else 0

    @staticmethod
    def _startup_restore_fill_price(fill: Dict[str, Any]) -> float:
        for key in ("px", "price", "avgPx", "avg_px", "average_price"):
            value = safe_float((fill or {}).get(key), 0.0)
            if value and value > 0.0:
                return float(value)
        return 0.0

    @staticmethod
    def _startup_restore_fill_reduce_only_hint(fill: Dict[str, Any]) -> bool:
        if bool((fill or {}).get("reduceOnly", False) or (fill or {}).get("reduce_only", False)):
            return True
        text = " ".join(
            str((fill or {}).get(key, "") or "")
            for key in ("dir", "side", "crossed", "feeToken")
        ).lower()
        if "close" in text or "reduce" in text:
            return True
        return False

    @staticmethod
    def _startup_restore_state_time_seconds(state_payload: Optional[dict]) -> float:
        payload = state_payload if isinstance(state_payload, dict) else {}
        updated_at_ms = int(safe_float(payload.get("updated_at_ms"), 0.0) or 0)
        if updated_at_ms > 0:
            return updated_at_ms / 1000.0
        updated_at = str(payload.get("updated_at", "") or "").strip()
        if updated_at:
            try:
                parsed = datetime.fromisoformat(updated_at.replace("Z", "+00:00"))
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.timestamp()
            except Exception:
                return 0.0
        return 0.0

    def _startup_restore_latest_reduction_fill_seconds(self, fills: List[Dict[str, Any]]) -> float:
        values = [
            self._startup_restore_fill_time_ms(fill) / 1000.0
            for fill in list(fills or [])
            if isinstance(fill, dict)
            and self._startup_restore_fill_reduce_only_hint(fill)
            and self._startup_restore_fill_time_ms(fill) > 0
        ]
        return max(values) if values else 0.0

    def _startup_restore_active_tp_order_seconds(self, session: RiskSession, exchange_order_refs: List[Dict[str, Any]], leg_name: str) -> float:
        price = float(getattr(session, f"{leg_name.replace('stage_', '')}_price", 0.0) or 0.0)
        if price <= 0.0:
            for spec in self._iter_risk_session_exit_order_specs(session):
                if str(spec.get("name", "") or "") == leg_name:
                    price = float(spec.get("trigger_price", 0.0) or 0.0)
                    break
        values = [
            int(safe_float((ref or {}).get("created_time_ms"), 0.0) or 0) / 1000.0
            for ref in list(exchange_order_refs or [])
            if isinstance(ref, dict)
            and str(ref.get("leg_type", "") or "") == "take_profit"
            and self._restore_price_matches(price, float(ref.get("trigger_price", 0.0) or 0.0))
            and int(safe_float(ref.get("created_time_ms"), 0.0) or 0) > 0
        ]
        return min(values) if values else 0.0

    def _startup_restore_tp1_completed_at_seconds(
        self,
        session: RiskSession,
        exchange_order_refs: List[Dict[str, Any]],
        recent_fills: List[Dict[str, Any]],
        state_payload: Optional[dict],
        now: float,
    ) -> Tuple[float, str]:
        existing = float(getattr(session, "tp1_hit_at", 0.0) or 0.0)
        if existing > 0.0:
            return existing, "persisted_tp1_hit_at"
        reduction_fill_seconds = self._startup_restore_latest_reduction_fill_seconds(recent_fills)
        if reduction_fill_seconds > 0.0:
            return reduction_fill_seconds, "recent_reduction_fill"
        tp2_order_seconds = self._startup_restore_active_tp_order_seconds(session, exchange_order_refs, "stage_tp2")
        if tp2_order_seconds > 0.0:
            return tp2_order_seconds, "active_tp2_order_created_time"
        state_seconds = self._startup_restore_state_time_seconds(state_payload)
        if state_seconds > 0.0:
            return state_seconds, "persisted_state_updated_at"
        return float(now or time.time()), "startup_now_fallback"

    def _startup_restore_open_take_profit_ref_for_leg(
        self,
        session: RiskSession,
        exchange_order_refs: List[Dict[str, Any]],
        leg_name: str,
    ) -> Optional[Dict[str, Any]]:
        price_attr = "tp1_price" if leg_name == "stage_tp1" else "tp2_price"
        price = float(getattr(session, price_attr, 0.0) or 0.0)
        if price <= 0.0:
            for spec in self._iter_risk_session_exit_order_specs(session):
                if str(spec.get("name", "") or "") == leg_name:
                    price = float(spec.get("trigger_price", 0.0) or 0.0)
                    break
        if price <= 0.0:
            return None
        matches = [
            dict(ref)
            for ref in list(exchange_order_refs or [])
            if isinstance(ref, dict)
            and str(ref.get("leg_type", "") or "") == "take_profit"
            and self._restore_price_matches(price, float(ref.get("trigger_price", 0.0) or 0.0))
        ]
        if not matches:
            return None
        return min(matches, key=lambda item: abs(float(item.get("close_size", 0.0) or 0.0)))

    def _startup_restore_reconcile_staged_progress(
        self,
        session: RiskSession,
        position_snapshot: dict,
        exchange_order_refs: List[Dict[str, Any]],
        completed_keys: set,
        recent_fills: List[Dict[str, Any]],
        state_payload: Optional[dict],
        now: float,
    ) -> Tuple[set, float]:
        completed = set(completed_keys or set())
        if not bool(getattr(session, "staged_exit_enabled", False)):
            return completed, float(now or time.time())
        initial_size_abs = abs(float(getattr(session, "staged_exit_size_basis_abs", 0.0) or getattr(session, "initial_size_abs", 0.0) or 0.0))
        current_size_abs = abs(float(position_snapshot.get("size", 0.0) or 0.0))
        if initial_size_abs <= 0.0:
            return completed, float(now or time.time())
        qty_tol = max(self._risk_session_order_qty_tolerance(), initial_size_abs * 0.01)
        total_closed_abs = max(0.0, initial_size_abs - current_size_abs)
        tp1_target = self._staged_tp_target_size_abs(session, "stage_tp1")
        tp2_target = self._staged_tp_target_size_abs(session, "stage_tp2")
        original = {
            "tp1_completed_size_abs": float(getattr(session, "tp1_completed_size_abs", 0.0) or 0.0),
            "tp2_completed_size_abs": float(getattr(session, "tp2_completed_size_abs", 0.0) or 0.0),
            "tp1_hit": bool(getattr(session, "tp1_hit", False)),
            "tp2_hit": bool(getattr(session, "tp2_hit", False)),
            "tp1_hit_at": float(getattr(session, "tp1_hit_at", 0.0) or 0.0),
        }

        def completed_from_open_order(leg_name: str, target_size: float) -> Optional[float]:
            ref = self._startup_restore_open_take_profit_ref_for_leg(session, exchange_order_refs, leg_name)
            if ref is None or target_size <= 0.0:
                return None
            open_size = max(0.0, float(ref.get("close_size", 0.0) or 0.0))
            return self._align_risk_close_size_for_session(
                session,
                max(0.0, min(target_size, target_size - open_size)),
                max_size=target_size,
            )

        tp1_completed = max(0.0, float(getattr(session, "tp1_completed_size_abs", 0.0) or 0.0))
        open_tp1_completed = completed_from_open_order("stage_tp1", tp1_target)
        if open_tp1_completed is not None:
            tp1_completed = max(tp1_completed, open_tp1_completed)
        elif tp1_target > 0.0:
            if bool(getattr(session, "tp1_hit", False)) or "take_profit::stage_tp1" in completed or total_closed_abs >= tp1_target - qty_tol:
                tp1_completed = max(tp1_completed, tp1_target)
            else:
                tp1_completed = max(tp1_completed, min(tp1_target, total_closed_abs))
        tp1_completed = self._align_risk_close_size_for_session(session, max(0.0, min(tp1_target, tp1_completed)), max_size=tp1_target) if tp1_target > 0.0 else 0.0
        if tp1_target > 0.0 and tp1_completed >= tp1_target - qty_tol:
            completed.add("take_profit::stage_tp1")

        tp2_completed = max(0.0, float(getattr(session, "tp2_completed_size_abs", 0.0) or 0.0))
        open_tp2_completed = completed_from_open_order("stage_tp2", tp2_target)
        if open_tp2_completed is not None:
            tp2_completed = max(tp2_completed, open_tp2_completed)
        elif tp2_target > 0.0:
            if bool(getattr(session, "tp2_hit", False)) or "take_profit::stage_tp2" in completed or total_closed_abs >= tp1_target + tp2_target - qty_tol:
                tp2_completed = max(tp2_completed, tp2_target)
            elif "take_profit::stage_tp1" in completed or tp1_completed >= tp1_target - qty_tol:
                tp2_completed = max(tp2_completed, min(tp2_target, max(0.0, total_closed_abs - tp1_target)))
        tp2_completed = self._align_risk_close_size_for_session(session, max(0.0, min(tp2_target, tp2_completed)), max_size=tp2_target) if tp2_target > 0.0 else 0.0
        if tp2_target > 0.0 and tp2_completed >= tp2_target - qty_tol:
            completed.add("take_profit::stage_tp2")

        session.tp1_completed_size_abs = tp1_completed
        session.tp2_completed_size_abs = tp2_completed
        tp1_completed_at, tp1_completed_at_source = self._startup_restore_tp1_completed_at_seconds(
            session,
            exchange_order_refs,
            recent_fills,
            state_payload,
            now,
        )
        if "take_profit::stage_tp1" in completed and float(getattr(session, "tp1_hit_at", 0.0) or 0.0) <= 0.0:
            session.tp1_hit_at = tp1_completed_at

        changed = (
            abs(original["tp1_completed_size_abs"] - float(getattr(session, "tp1_completed_size_abs", 0.0) or 0.0)) > qty_tol
            or abs(original["tp2_completed_size_abs"] - float(getattr(session, "tp2_completed_size_abs", 0.0) or 0.0)) > qty_tol
            or original["tp1_hit_at"] != float(getattr(session, "tp1_hit_at", 0.0) or 0.0)
            or ("take_profit::stage_tp1" in completed and not original["tp1_hit"])
            or ("take_profit::stage_tp2" in completed and not original["tp2_hit"])
        )
        if changed:
            self._audit_event(
                "startup_live_tpsl_restore_staged_progress_reconciled",
                {
                    "completed_keys": sorted(completed),
                    "tp1_completed_size_abs": float(getattr(session, "tp1_completed_size_abs", 0.0) or 0.0),
                    "tp2_completed_size_abs": float(getattr(session, "tp2_completed_size_abs", 0.0) or 0.0),
                    "tp1_hit_at": float(getattr(session, "tp1_hit_at", 0.0) or 0.0),
                    "tp1_completed_at_source": tp1_completed_at_source if "take_profit::stage_tp1" in completed else "",
                    "initial_size_abs": initial_size_abs,
                    "current_size_abs": current_size_abs,
                    "total_closed_abs": total_closed_abs,
                    "exchange_order_refs": exchange_order_refs,
                    "position_snapshot": position_snapshot,
                },
            )
        apply_time = tp1_completed_at if "take_profit::stage_tp1" in completed else float(now or time.time())
        return completed, apply_time

    def _startup_restore_recent_user_fills(
        self,
        state_payload: Optional[dict],
        now: float,
        position_snapshot: Optional[dict] = None,
    ) -> List[Dict[str, Any]]:
        if not hasattr(self.reader, "get_user_fills_by_time"):
            return []
        address = str(getattr(self, "user_fills_address", "") or getattr(self.reader, "account_address", "") or "").strip()
        if not address:
            return []
        payload = state_payload if isinstance(state_payload, dict) else {}
        updated_at_ms = int(safe_float(payload.get("updated_at_ms"), 0.0) or 0.0)
        base_lookback = max(float(getattr(self, "user_fills_backfill_lookback_seconds", 120.0) or 120.0), 1.0)
        restore_lookback = max(float(getattr(self, "risk_session_restore_fill_lookback_seconds", 21600.0) or 21600.0), base_lookback)
        grace_ms = int(max(float(getattr(self, "user_fills_reconcile_grace_seconds", 3.0) or 3.0), 0.0) * 1000)
        end_ms = int(now * 1000)
        start_ms = max(0, (updated_at_ms - grace_ms) if updated_at_ms > 0 else (end_ms - int(restore_lookback * 1000)))
        try:
            fills = [
                dict(item)
                for item in list(self.reader.get_user_fills_by_time(address, start_ms, end_ms, aggregate_by_time=False) or [])
                if isinstance(item, dict)
            ]
        except Exception as exc:
            self._audit_event(
                "startup_live_tpsl_restore_fills_backfill_failed",
                {"start_time_ms": start_ms, "end_time_ms": end_ms, "error": str(exc)},
            )
            return []
        symbol = canonicalize_execution_symbol((position_snapshot or {}).get("symbol", "") or getattr(self, "symbol", "") or "")
        if not symbol:
            return sorted(fills, key=self._startup_restore_fill_time_ms)
        return sorted(
            [
                fill
                for fill in fills
                if not canonicalize_execution_symbol(fill.get("coin", "") or "") or canonicalize_execution_symbol(fill.get("coin", "") or "") == symbol
            ],
            key=self._startup_restore_fill_time_ms,
        )

    def _startup_restore_open_order_keys(self, session: RiskSession, exchange_order_refs: List[Dict[str, Any]]) -> set:
        open_keys: set = set()
        saved_refs = [dict(item) for item in list(getattr(session, "resting_exit_orders", []) or []) if isinstance(item, dict)]
        for exchange_ref in list(exchange_order_refs or []):
            for saved_ref in saved_refs:
                if self._order_ref_identity_matches(exchange_ref, saved_ref):
                    key = str(saved_ref.get("key", "") or "")
                    if key:
                        open_keys.add(key)
                    break
        if open_keys:
            return open_keys
        desired_specs = self._iter_risk_session_exit_order_specs(session)
        for exchange_ref in list(exchange_order_refs or []):
            for saved_ref in saved_refs + desired_specs:
                if str(exchange_ref.get("tpsl", "") or "") != str(saved_ref.get("tpsl", "") or ""):
                    continue
                if self._restore_price_matches(float(saved_ref.get("trigger_price", 0.0) or 0.0), float(exchange_ref.get("trigger_price", 0.0) or 0.0)):
                    key = str(saved_ref.get("key", "") or "")
                    if key:
                        open_keys.add(key)
                    break
        return open_keys

    def _startup_restore_key_for_fill(self, session: RiskSession, fill: Dict[str, Any]) -> Tuple[str, float]:
        saved_refs = [dict(item) for item in list(getattr(session, "resting_exit_orders", []) or []) if isinstance(item, dict)]
        for ref in saved_refs:
            if self._order_ref_identity_matches(fill, ref):
                return str(ref.get("key", "") or ""), self._align_risk_close_size_for_session(session, abs(float(ref.get("close_size", 0.0) or 0.0)))
        fill_price = self._startup_restore_fill_price(fill)
        if fill_price <= 0.0:
            return "", 0.0
        for spec in self._iter_risk_session_exit_order_specs(session):
            if self._restore_price_matches(float(spec.get("trigger_price", 0.0) or 0.0), fill_price):
                return str(spec.get("key", "") or ""), abs(float(spec.get("close_size", 0.0) or 0.0))
        return "", 0.0

    def _startup_restore_completed_keys_from_fills(self, session: RiskSession, fills: List[Dict[str, Any]]) -> set:
        completed: set = set()
        fill_sizes_by_key: Dict[str, float] = {}
        for fill in sorted(list(fills or []), key=self._startup_restore_fill_time_ms):
            if not isinstance(fill, dict):
                continue
            key, close_size = self._startup_restore_key_for_fill(session, fill)
            if not key:
                continue
            fill_size = abs(float(safe_float(fill.get("sz"), 0.0) or 0.0))
            if fill_size <= 0.0:
                continue
            fill_sizes_by_key[key] = fill_sizes_by_key.get(key, 0.0) + fill_size
            if close_size <= 0.0 or fill_sizes_by_key[key] + self._risk_session_order_qty_tolerance() >= close_size:
                completed.add(key)
        return completed

    def _startup_restore_infer_completed_keys(
        self,
        session: RiskSession,
        position_snapshot: dict,
        exchange_order_refs: List[Dict[str, Any]],
        fill_completed_keys: set,
    ) -> set:
        completed = set(fill_completed_keys or set())
        current_size_abs = abs(float(position_snapshot.get("size", 0.0) or 0.0))
        initial_size_abs = abs(float(getattr(session, "initial_size_abs", 0.0) or getattr(session, "staged_exit_size_basis_abs", 0.0) or 0.0))
        qty_tol = max(self._risk_session_order_qty_tolerance(), initial_size_abs * 0.01)
        open_keys = self._startup_restore_open_order_keys(session, exchange_order_refs)
        tp1_key = "take_profit::stage_tp1"
        tp2_key = "take_profit::stage_tp2"
        stop_keys = {
            str(ref.get("key", "") or "")
            for ref in list(getattr(session, "resting_exit_orders", []) or [])
            if str(ref.get("leg_type", "") or "") == "stop_loss"
        }
        if current_size_abs <= qty_tol:
            completed.update(stop_keys or {"stop_loss::stage_initial_stop", "stop_loss::stage_post_tp1_stop", "stop_loss::stage_tail_hard_stop"})
            return completed
        tp1_target = self._staged_tp_target_size_abs(session, "stage_tp1")
        tp2_target = self._staged_tp_target_size_abs(session, "stage_tp2")
        if tp1_key not in open_keys and initial_size_abs > 0.0 and tp1_target > 0.0:
            if current_size_abs <= initial_size_abs - tp1_target + qty_tol:
                completed.add(tp1_key)
        if tp2_key not in open_keys and initial_size_abs > 0.0 and tp2_target > 0.0:
            if current_size_abs <= initial_size_abs - tp1_target - tp2_target + qty_tol:
                completed.add(tp1_key)
                completed.add(tp2_key)
        return completed

    def _startup_restore_apply_completed_keys(self, session: RiskSession, completed_keys: set, position_snapshot: dict, now: float) -> Optional[RiskSession]:
        stop_completed = [key for key in set(completed_keys or set()) if str(key).startswith("stop_loss::")]
        if stop_completed:
            return None
        ordered_keys = []
        if "take_profit::stage_tp1" in completed_keys:
            ordered_keys.append("take_profit::stage_tp1")
        if "take_profit::stage_tp2" in completed_keys:
            ordered_keys.append("take_profit::stage_tp2")
        if ordered_keys:
            self._update_staged_risk_session_after_completed_keys(session, ordered_keys, now=now)
            session.executed_leg_names.update(ordered_keys)
        session.expected_size = float(position_snapshot.get("size", 0.0) or 0.0)
        session.baseline_size = float(position_snapshot.get("size", 0.0) or 0.0)
        session.side = str(position_snapshot.get("side", session.side) or session.side)
        return session

    def _restore_risk_session_from_persisted_state(self, position_snapshot: dict) -> Optional[RiskSession]:
        state_payload = self._load_risk_session_state_payload()
        if not state_payload:
            return None
        now = time.time()
        session = self._risk_session_from_state_payload(state_payload, position_snapshot)
        if session is None:
            return None
        symbol = canonicalize_execution_symbol(position_snapshot.get("symbol", "") or getattr(self, "symbol", "") or "")
        exchange_order_refs, _, _, _ = self._exchange_reduce_only_order_refs_from_snapshot(
            position_snapshot,
            entry_price=float(getattr(session, "initial_entry_price", 0.0) or 0.0),
        )
        recent_fills = self._startup_restore_recent_user_fills(state_payload, now, position_snapshot)
        fill_completed_keys = self._startup_restore_completed_keys_from_fills(session, recent_fills)
        completed_keys = self._startup_restore_infer_completed_keys(session, position_snapshot, exchange_order_refs, fill_completed_keys)
        completed_keys, apply_time = self._startup_restore_reconcile_staged_progress(
            session,
            position_snapshot,
            exchange_order_refs,
            completed_keys,
            recent_fills,
            state_payload,
            now,
        )
        restored_session = self._startup_restore_apply_completed_keys(session, completed_keys, position_snapshot, apply_time)
        if restored_session is None:
            self._clear_risk_session_state()
            self._audit_event(
                "startup_live_tpsl_restore_state_cleared",
                {
                    "reason": "stop_loss_completed_or_position_flat",
                    "completed_keys": sorted(completed_keys),
                    "position_snapshot": position_snapshot,
                },
            )
            return None
        finalized = self._finalize_restored_risk_session_order_refs(restored_session, exchange_order_refs, symbol, allow_resync=True)
        if finalized is not None:
            return finalized
        subset = self._match_exchange_order_refs_subset_to_risk_session_specs(restored_session, exchange_order_refs, symbol, check_sizes=False)
        if subset is not None:
            restored_session.resting_exit_orders = [dict(item) for item in list(exchange_order_refs or []) if isinstance(item, dict)]
            restored_session.use_resting_exit_orders = bool(restored_session.resting_exit_orders)
            setattr(restored_session, "_startup_restore_needs_order_resync", True)
            return restored_session
        open_keys = self._startup_restore_open_order_keys(session, exchange_order_refs)
        if completed_keys and (open_keys or not exchange_order_refs):
            restored_session.resting_exit_orders = [dict(item) for item in list(exchange_order_refs or []) if isinstance(item, dict)]
            restored_session.use_resting_exit_orders = bool(restored_session.resting_exit_orders)
            setattr(restored_session, "_startup_restore_needs_order_resync", True)
            return restored_session
        if not exchange_order_refs:
            restored_session.resting_exit_orders = []
            restored_session.use_resting_exit_orders = False
            setattr(restored_session, "_startup_restore_needs_order_resync", True)
            return restored_session
        self._audit_event(
            "startup_live_tpsl_restore_persisted_state_mismatch",
            {
                "completed_keys": sorted(completed_keys),
                "open_order_keys": sorted(open_keys),
                "exchange_order_refs": exchange_order_refs,
                "state_resting_exit_orders": list(getattr(session, "resting_exit_orders", []) or []),
                "position_snapshot": position_snapshot,
            },
        )
        return None

    def _startup_restore_infer_strategy_entry_candidates(
        self,
        *,
        side: str,
        take_profit_refs: List[Dict[str, Any]],
        stop_refs: List[Dict[str, Any]],
        staged_exit_params: Dict[str, Any],
    ) -> List[float]:
        tp1_mult = float(staged_exit_params.get("tp1_r_multiple", getattr(self, "risk_tp1_r_multiple", 1.0)) or 1.0)
        tp2_mult = float(staged_exit_params.get("tp2_r_multiple", getattr(self, "risk_tp2_r_multiple", 2.0)) or 2.0)
        if tp1_mult <= 0.0 or tp2_mult <= tp1_mult:
            return []
        candidates: List[float] = []
        tp_prices = sorted([float(ref.get("trigger_price", 0.0) or 0.0) for ref in take_profit_refs if float(ref.get("trigger_price", 0.0) or 0.0) > 0.0])
        if side == "short":
            tp_prices = list(reversed(tp_prices))
        stop_prices = [float(ref.get("trigger_price", 0.0) or 0.0) for ref in stop_refs if float(ref.get("trigger_price", 0.0) or 0.0) > 0.0]
        for stop_price in stop_prices:
            for tp_price in tp_prices:
                for multiple in (tp1_mult, tp2_mult):
                    entry = (tp_price + (multiple * stop_price)) / (1.0 + multiple)
                    if entry > 0.0:
                        candidates.append(entry)
        if len(tp_prices) >= 2:
            tp1_price = tp_prices[0]
            tp2_price = tp_prices[1]
            entry = ((tp2_mult * tp1_price) - (tp1_mult * tp2_price)) / (tp2_mult - tp1_mult)
            if entry > 0.0:
                candidates.append(entry)
        deduped: List[float] = []
        for price in candidates:
            if any(self._restore_price_matches(existing, price) for existing in deduped):
                continue
            deduped.append(price)
        return deduped

    def _startup_restore_staged_exit_params_at_ms(self, symbol: str, value_ms: Any) -> Optional[Dict[str, Any]]:
        when = self._startup_restore_ms_to_utc_datetime(value_ms)
        if when is None or not hasattr(self, "_staged_exit_params_for_symbol"):
            return None
        return self._staged_exit_params_for_symbol(symbol, now_utc=when)

    def _startup_restore_strategy_param_variants(self, symbol: str) -> List[Dict[str, Any]]:
        variants: List[Dict[str, Any]] = []
        seen: set = set()
        if hasattr(self, "_staged_exit_params_for_profile_band"):
            for liquidity_band in ("normal_liquidity", "low_liquidity"):
                params = self._staged_exit_params_for_profile_band(symbol, liquidity_band)
                key = (
                    float(params.get("tp1_r_multiple", 0.0) or 0.0),
                    float(params.get("tp2_r_multiple", 0.0) or 0.0),
                    float(params.get("tp1_close_fraction", 0.0) or 0.0),
                    float(params.get("tp2_close_fraction", 0.0) or 0.0),
                    float(params.get("post_tp1_stop_r_multiple", 0.0) or 0.0),
                    float(params.get("post_tp2_locked_r_multiple", 0.0) or 0.0),
                    float(params.get("trailing_soft_atr_multiple", 0.0) or 0.0),
                    float(params.get("trailing_hard_atr_multiple", 0.0) or 0.0),
                )
                if key in seen:
                    continue
                seen.add(key)
                variants.append(params)
        if not variants:
            variants.append(self._staged_exit_params_for_symbol(symbol) if hasattr(self, "_staged_exit_params_for_symbol") else {})
        return variants

    def _startup_restore_anchor_candidates(
        self,
        position_snapshot: dict,
        exchange_order_refs: List[Dict[str, Any]],
        take_profit_refs: List[Dict[str, Any]],
        stop_refs: List[Dict[str, Any]],
        recent_fills: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        candidates: List[Dict[str, Any]] = []

        def add_candidate(
            source: str,
            price: float,
            evidence: str = "",
            *,
            anchor_time_ms: int = 0,
            staged_exit_params: Optional[Dict[str, Any]] = None,
            requires_unique_match: bool = False,
        ) -> None:
            price = safe_float(price, 0.0) or 0.0
            if price <= 0.0:
                return
            liquidity_band = str((staged_exit_params or {}).get("liquidity_band", "") or "")
            if any(
                self._restore_price_matches(float(item.get("entry_price", 0.0) or 0.0), price)
                and str(item.get("source", "")) == source
                and int(item.get("anchor_time_ms", 0) or 0) == int(anchor_time_ms or 0)
                and str((item.get("staged_exit_params") or {}).get("liquidity_band", "") or "") == liquidity_band
                for item in candidates
            ):
                return
            item: Dict[str, Any] = {"source": source, "entry_price": float(price), "evidence": evidence}
            if anchor_time_ms > 0:
                item["anchor_time_ms"] = int(anchor_time_ms)
            if isinstance(staged_exit_params, dict):
                item["staged_exit_params"] = dict(staged_exit_params)
            if requires_unique_match:
                item["requires_unique_match"] = True
            candidates.append(item)

        side = str(position_snapshot.get("side", "") or "")
        symbol = canonicalize_execution_symbol(position_snapshot.get("symbol", self.symbol))
        matching_open_fills: List[Tuple[Dict[str, Any], float, int]] = []
        for fill in reversed(list(recent_fills or [])):
            if self._startup_restore_fill_reduce_only_hint(fill):
                continue
            price = self._startup_restore_fill_price(fill)
            text = " ".join(str(fill.get(key, "") or "") for key in ("dir", "side")).lower()
            if "open long" in text and side == "short":
                continue
            if "open short" in text and side == "long":
                continue
            fill_time_ms = self._startup_restore_fill_time_ms(fill)
            if price > 0.0:
                matching_open_fills.append((fill, price, fill_time_ms))
            if len(matching_open_fills) >= 20:
                break

        latest_open_fill_time_ms = next((fill_time_ms for _, _, fill_time_ms in matching_open_fills if fill_time_ms > 0), 0)
        position_entry = safe_float(position_snapshot.get("entry_price"), 0.0) or 0.0
        add_candidate("position_entry_price", position_entry, "position_snapshot.entry_price", anchor_time_ms=latest_open_fill_time_ms)

        for _, price, fill_time_ms in matching_open_fills:
            add_candidate("add_fill_avg_price", price, "recent_non_reduce_only_fill", anchor_time_ms=fill_time_ms)

        order_time_ms = self._startup_restore_order_refs_time_ms(exchange_order_refs)
        strategy_param_variants: List[Tuple[Dict[str, Any], int, str, bool]] = []
        if order_time_ms > 0:
            params_at_order_time = self._startup_restore_staged_exit_params_at_ms(symbol, order_time_ms)
            if isinstance(params_at_order_time, dict):
                strategy_param_variants.append((params_at_order_time, order_time_ms, "open_tpsl_price_inference_order_time", False))
        else:
            for params in self._startup_restore_strategy_param_variants(symbol):
                strategy_param_variants.append((params, 0, "open_tpsl_price_inference_no_order_time", True))

        for staged_exit_params, anchor_time_ms, evidence, requires_unique_match in strategy_param_variants:
            for inferred_entry in self._startup_restore_infer_strategy_entry_candidates(
                side=side,
                take_profit_refs=take_profit_refs,
                stop_refs=stop_refs,
                staged_exit_params=staged_exit_params,
            ):
                add_candidate(
                    "strategy_entry_price",
                    inferred_entry,
                    evidence,
                    anchor_time_ms=anchor_time_ms,
                    staged_exit_params=staged_exit_params,
                    requires_unique_match=requires_unique_match,
                )
        return candidates

    def _startup_restore_seed_initial_size_from_orders(self, session: RiskSession, exchange_order_refs: List[Dict[str, Any]]) -> None:
        current_size_abs = abs(float(getattr(session, "expected_size", 0.0) or 0.0))
        max_stop_size = max(
            [abs(float(ref.get("close_size", 0.0) or 0.0)) for ref in list(exchange_order_refs or []) if str(ref.get("leg_type", "") or "") == "stop_loss"] or [0.0]
        )
        seeded_initial_size = max(abs(float(getattr(session, "initial_size_abs", 0.0) or 0.0)), current_size_abs, max_stop_size)
        if seeded_initial_size > 0.0:
            session.initial_size_abs = seeded_initial_size
            session.staged_exit_size_basis_abs = max(float(getattr(session, "staged_exit_size_basis_abs", 0.0) or 0.0), seeded_initial_size)

    def _startup_restore_try_anchor_candidate(
        self,
        *,
        position_snapshot: dict,
        candidate: Dict[str, Any],
        recent_fills: List[Dict[str, Any]],
    ) -> Optional[RiskSession]:
        side = str(position_snapshot.get("side", "flat") or "flat")
        symbol = canonicalize_execution_symbol(position_snapshot.get("symbol", self.symbol))
        entry_price = safe_float(candidate.get("entry_price"), 0.0) or 0.0
        source = str(candidate.get("source", "") or "")
        if side not in {"long", "short"} or entry_price <= 0.0:
            return None
        staged_exit_params_override = candidate.get("staged_exit_params") if isinstance(candidate.get("staged_exit_params"), dict) else None
        if staged_exit_params_override is None:
            params_at_anchor_time = self._startup_restore_staged_exit_params_at_ms(symbol, candidate.get("anchor_time_ms"))
            if isinstance(params_at_anchor_time, dict):
                staged_exit_params_override = params_at_anchor_time
        exchange_order_refs, take_profit_refs, stop_refs, stop_loss_refs = self._exchange_reduce_only_order_refs_from_snapshot(position_snapshot, entry_price=entry_price)
        if not exchange_order_refs:
            return None
        for stop_loss_ref in list(stop_loss_refs or []):
            stop_loss_price = float(stop_loss_ref.get("trigger_price", 0.0) or 0.0)
            risk_plan = PositionManagementPlan(
                execute_now=False,
                action_decision=ManagementDecision(
                    action="no_change",
                    close_fraction=0.0,
                    new_notional_usd=max(0.0, float(position_snapshot.get("notional_usd", 0.0) or 0.0)),
                    entry_price=max(0.0, float(entry_price or 0.0)),
                    stop_loss_price=stop_loss_price,
                    planned_max_loss_usd=0.0,
                    leverage=max(0, int(position_snapshot.get("leverage", 0) or 0)),
                    margin_basis_usd=max(0.0, float(position_snapshot.get("margin_used", 0.0) or 0.0)),
                ),
                scenario=None,
            )
            initial_session = self._build_staged_risk_session_from_stop(
                position_after=position_snapshot,
                plan_name="startup_exchange_restore",
                initial_entry_price=entry_price,
                stop_loss_price=stop_loss_price,
                position_management=risk_plan,
                risk_entry_source=source,
                staged_exit_params_override=staged_exit_params_override,
                apply_stop_hard_buffer=False,
            )
            if initial_session is None:
                continue
            self._startup_restore_seed_initial_size_from_orders(initial_session, exchange_order_refs)
            open_keys_before = self._startup_restore_open_order_keys(initial_session, exchange_order_refs)
            fill_completed_keys = self._startup_restore_completed_keys_from_fills(initial_session, recent_fills)
            completed_keys = self._startup_restore_infer_completed_keys(initial_session, position_snapshot, exchange_order_refs, fill_completed_keys)
            completed_keys, apply_time = self._startup_restore_reconcile_staged_progress(
                initial_session,
                position_snapshot,
                exchange_order_refs,
                completed_keys,
                recent_fills,
                None,
                time.time(),
            )
            restored_session = self._startup_restore_apply_completed_keys(initial_session, completed_keys, position_snapshot, apply_time)
            if restored_session is None:
                self._audit_event(
                    "startup_live_tpsl_restore_stop_completed",
                    {
                        "anchor_source": source,
                        "entry_price": entry_price,
                        "completed_keys": sorted(completed_keys),
                        "position_snapshot": position_snapshot,
                    },
                )
                return None
            setattr(restored_session, "risk_entry_source", source)
            finalized = self._finalize_restored_risk_session_order_refs(restored_session, exchange_order_refs, symbol, allow_resync=True)
            if finalized is not None:
                return finalized
            take_profit_overlap = any(str(key).startswith("take_profit::") for key in open_keys_before)
            if completed_keys and open_keys_before and (take_profit_overlap or fill_completed_keys):
                restored_session.resting_exit_orders = [dict(item) for item in list(exchange_order_refs or []) if isinstance(item, dict)]
                restored_session.use_resting_exit_orders = bool(restored_session.resting_exit_orders)
                setattr(restored_session, "_startup_restore_needs_order_resync", True)
                return restored_session

        if len(stop_loss_refs) == 1 and len(take_profit_refs) == 1:
            post_tp1_session = self._build_startup_post_tp1_risk_session(
                position_snapshot,
                side=side,
                initial_entry_price=entry_price,
                post_tp1_stop_price=float(stop_loss_refs[0].get("trigger_price", 0.0) or 0.0),
                tp2_order_ref=take_profit_refs[0],
                staged_exit_params_override=staged_exit_params_override,
            )
            if post_tp1_session is not None:
                setattr(post_tp1_session, "risk_entry_source", source)
                finalized = self._finalize_restored_risk_session_order_refs(post_tp1_session, exchange_order_refs, symbol, allow_resync=True)
                if finalized is not None:
                    return finalized

        if len(stop_refs) == 1 and not take_profit_refs:
            hard_stop_price = float(stop_refs[0].get("trigger_price", 0.0) or 0.0)
            if not self._trigger_price_is_loss_side(side, hard_stop_price, entry_price):
                tail_session = self._build_startup_tail_risk_session(
                    position_snapshot,
                    side=side,
                    initial_entry_price=entry_price,
                    hard_stop_price=hard_stop_price,
                    staged_exit_params_override=staged_exit_params_override,
                )
                if tail_session is not None:
                    setattr(tail_session, "risk_entry_source", source)
                    finalized = self._finalize_restored_risk_session_order_refs(tail_session, exchange_order_refs, symbol, allow_resync=True)
                    if finalized is not None:
                        return finalized
        return None

    def _rebuild_risk_session_from_exchange_reduce_only_orders(self, position_snapshot: dict) -> Optional[RiskSession]:
        if not snapshot_has_open_position(position_snapshot):
            return None
        side = str(position_snapshot.get("side", "flat") or "flat")
        if side not in {"long", "short"}:
            return None
        symbol = canonicalize_execution_symbol(position_snapshot.get("symbol", self.symbol))
        exchange_order_refs, take_profit_refs, stop_refs, _ = self._exchange_reduce_only_order_refs_from_snapshot(position_snapshot)
        if not exchange_order_refs:
            return None
        recent_fills = self._startup_restore_recent_user_fills(None, time.time(), position_snapshot)
        candidates = self._startup_restore_anchor_candidates(position_snapshot, exchange_order_refs, take_profit_refs, stop_refs, recent_fills)
        deferred_unique_matches: List[Tuple[Dict[str, Any], RiskSession]] = []
        for candidate in candidates:
            restored = self._startup_restore_try_anchor_candidate(
                position_snapshot=position_snapshot,
                candidate=candidate,
                recent_fills=recent_fills,
            )
            if restored is not None:
                if bool(candidate.get("requires_unique_match", False)):
                    deferred_unique_matches.append((candidate, restored))
                    continue
                self._audit_event(
                    "startup_live_tpsl_restore_fallback_anchor_selected",
                    {
                        "anchor_source": candidate.get("source"),
                        "entry_price": candidate.get("entry_price"),
                        "evidence": candidate.get("evidence"),
                        "anchor_time_ms": candidate.get("anchor_time_ms"),
                        "liquidity_band": (candidate.get("staged_exit_params") or {}).get("liquidity_band") if isinstance(candidate.get("staged_exit_params"), dict) else None,
                        "symbol": symbol,
                    },
                )
                return restored
        if len(deferred_unique_matches) == 1:
            candidate, restored = deferred_unique_matches[0]
            self._audit_event(
                "startup_live_tpsl_restore_fallback_anchor_selected",
                {
                    "anchor_source": candidate.get("source"),
                    "entry_price": candidate.get("entry_price"),
                    "evidence": candidate.get("evidence"),
                    "anchor_time_ms": candidate.get("anchor_time_ms"),
                    "liquidity_band": (candidate.get("staged_exit_params") or {}).get("liquidity_band") if isinstance(candidate.get("staged_exit_params"), dict) else None,
                    "symbol": symbol,
                },
            )
            return restored
        if len(deferred_unique_matches) > 1:
            self._audit_event(
                "startup_live_tpsl_restore_fallback_ambiguous",
                {
                    "symbol": symbol,
                    "matches": [
                        {
                            "anchor_source": item.get("source"),
                            "entry_price": item.get("entry_price"),
                            "evidence": item.get("evidence"),
                            "liquidity_band": (item.get("staged_exit_params") or {}).get("liquidity_band") if isinstance(item.get("staged_exit_params"), dict) else None,
                        }
                        for item, _ in deferred_unique_matches[:20]
                    ],
                    "exchange_order_refs": exchange_order_refs,
                },
            )
            return None
        self._audit_event(
            "startup_live_tpsl_restore_fallback_failed",
            {
                "symbol": symbol,
                "candidate_count": len(candidates),
                "candidates": candidates[:20],
                "exchange_order_refs": exchange_order_refs,
            },
        )
        return None
    def _maybe_restore_startup_live_tpsl(self, position_snapshot: dict) -> bool:
        if bool(getattr(self, "_startup_live_tpsl_restore_attempted", False)):
            return False
        if self.risk_session is not None:
            return False
        if not snapshot_has_open_position(position_snapshot):
            self._clear_risk_session_state()
            return False
        self._startup_live_tpsl_restore_attempted = True
        restore_symbol = canonicalize_execution_symbol(position_snapshot.get("symbol", "") or "")
        configured_context = dict(getattr(self, "trade_symbol_context", {}) or {})
        configured_symbols = {canonicalize_execution_symbol(configured_context.get("execution_symbol", "") or "")}
        configured_symbols.discard("")
        runtime_symbol = canonicalize_execution_symbol(getattr(self, "symbol", "") or "")
        if not configured_symbols and runtime_symbol:
            configured_symbols.add(runtime_symbol)
        if restore_symbol and configured_symbols and restore_symbol not in configured_symbols:
            self._audit_event(
                "startup_live_tpsl_restore_skipped_unconfigured_symbol",
                {
                    "configured_symbols": sorted(configured_symbols),
                    "position_symbol": restore_symbol,
                    "position_snapshot": position_snapshot,
                },
            )
            return False
        if restore_symbol:
            self._set_active_symbol(restore_symbol, reason="startup_live_tpsl_restore")
        restore_source = "persisted_state"
        restored_session = self._restore_risk_session_from_persisted_state(position_snapshot)
        if restored_session is None:
            restore_source = "exchange_open_orders"
            restored_session = self._rebuild_risk_session_from_exchange_reduce_only_orders(position_snapshot)
        if restored_session is None:
            exchange_order_refs, _, _, _ = self._exchange_reduce_only_order_refs_from_snapshot(position_snapshot)
            self._audit_event(
                "startup_live_tpsl_restore_missing",
                {
                    "symbol": position_snapshot.get("symbol", self.symbol),
                    "position_snapshot": position_snapshot,
                    "exchange_order_refs": exchange_order_refs,
                },
            )
            return False
        self.risk_session = restored_session
        if hasattr(self, "_ensure_cross_asset_soft_stop_poller"):
            self._ensure_cross_asset_soft_stop_poller()
        if hasattr(self, "_initialize_risk_session_cross_asset_reference"):
            self._initialize_risk_session_cross_asset_reference(restored_session)
        self._sync_position_basis_from_session(restored_session, reason="startup_restore")
        needs_order_resync = bool(getattr(restored_session, "_startup_restore_needs_order_resync", False))
        order_resync_attempted = False
        if needs_order_resync and bool(getattr(self.executor, "enabled", False)):
            order_resync_attempted = True
            self._sync_risk_session_resting_orders(restored_session)
        self._log_risk_session_ready(restored_session, reason="startup_restore", position_after=position_snapshot)
        self._persist_risk_session_state()
        self._audit_event(
            "startup_live_tpsl_restore_succeeded",
            {
                "symbol": position_snapshot.get("symbol", self.symbol),
                "position_snapshot": position_snapshot,
                "position_management": restored_session.position_management.to_dict() if restored_session.position_management is not None else None,
                "resting_exit_orders": list(restored_session.resting_exit_orders or []),
                "order_resync_needed": needs_order_resync,
                "order_resync_attempted": order_resync_attempted,
                "restore_source": restore_source,
            },
        )
        return True
    def _materialize_live_position_management_from_entry_plan(
        self,
        playbook: GenericPlaybook,
        position_snapshot: dict,
        all_positions: Optional[Dict[str, Any]] = None,
    ) -> GenericPlaybook:
        current_side, current_notional = self._position_side_and_notional(position_snapshot)
        entry_action_decision = playbook.entry_plan.action_decision
        materialization_debug: Dict[str, Any] = {}
        converted_action_decision = self._convert_entry_decision_to_management_decision(
            entry_action_decision,
            position_snapshot,
            all_positions,
            allow_immediate_reverse=bool(playbook.entry_plan.execute_now),
            trigger_confidence_raw=playbook.trigger_confidence_raw,
            debug_context=materialization_debug,
        )
        converted_scenario = (
            self._convert_entry_scenario_to_management_scenario(playbook.entry_plan.scenario)
            if playbook.entry_plan.scenario is not None
            else None
        )
        forced_immediate_close_decision: Optional[ManagementDecision] = None
        forced_immediate_close_payload: Optional[Dict[str, Any]] = None
        if current_side in {"long", "short"} and converted_scenario is not None:
            if (
                str(getattr(converted_action_decision, "action", "") or "") == "close"
                and bool(getattr(converted_action_decision, "continue_entry_plan_after_close", False))
            ):
                source_action = str(getattr(entry_action_decision, "action", "") or "")
                if source_action == "long" and current_side == "short":
                    forced_reason = "trigger_reverse_long"
                elif source_action == "short" and current_side == "long":
                    forced_reason = "trigger_reverse_short"
                else:
                    forced_reason = "trigger_close"
                forced_immediate_close_decision = converted_action_decision
                forced_immediate_close_payload = {
                    "selected_symbol": playbook.selected_symbol,
                    "source_entry_action": source_action,
                    "forced_reason": forced_reason,
                    "forced_now_action": converted_action_decision.to_dict(),
                }
        no_change_should_refresh = bool(materialization_debug.get("no_change_should_refresh", False))
        immediate_same_side_refresh = bool(
            current_side in {"long", "short"}
            and playbook.entry_plan.execute_now
            and str(entry_action_decision.action or "") == current_side
            and converted_action_decision.action == "no_change"
            and no_change_should_refresh
            and (
                converted_action_decision.leverage > 0
                or float(converted_action_decision.stop_loss_price or 0.0) > 0.0
            )
        )
        execute_now = bool(
            playbook.entry_plan.execute_now
            and (
                converted_action_decision.action not in {"no_change"}
                or immediate_same_side_refresh
            )
        )
        materialized_action_decision = converted_action_decision
        materialized_scenario = None if execute_now else converted_scenario
        if forced_immediate_close_decision is not None:
            self._audit_event("position_management_forced_immediate_close", forced_immediate_close_payload)
            execute_now = True
            materialized_action_decision = forced_immediate_close_decision
            materialized_scenario = None
        risk_session_passthrough = bool(
            current_side in {"long", "short"}
            and str(getattr(materialized_action_decision, "action", "") or "") == "no_change"
            and not no_change_should_refresh
        )
        if str(getattr(materialized_action_decision, "action", "") or "") == "no_change" and not no_change_should_refresh:
            materialized_action_decision = ManagementDecision(
                **{
                    **materialized_action_decision.to_dict(),
                    "close_fraction": 0.0,
                    "entry_price": 0.0,
                    "stop_loss_price": 0.0,
                    "planned_max_loss_usd": 0.0,
                    "leverage": 0,
                    "margin_basis_usd": 0.0,
                    "continue_entry_plan_after_close": False,
                }
            )
        if bool(playbook.entry_plan.execute_now) and str(materialized_action_decision.action or "") == "close":
            materialized_action_decision = ManagementDecision(**{**materialized_action_decision.to_dict(), "continue_entry_plan_after_close": False})
        comparison_target_notional = safe_float(materialization_debug.get("comparison_target_notional_usd"), None)
        if comparison_target_notional is not None and current_side in {"long", "short"} and str(entry_action_decision.action or "") == current_side:
            raw_target_notional = safe_float(materialization_debug.get("target_notional_usd"), 0.0) or 0.0
            print(
                "[materialize_notional_compare] "
                f"current_notional_usd={format_query_amount(current_notional)} | "
                f"comparison_target_notional={format_query_amount(comparison_target_notional)} | "
                f"target_notional_usd={format_query_amount(raw_target_notional)} | "
                f"materialized_action={converted_action_decision.action}"
            )
        playbook.position_management = PositionManagementPlan(
            execute_now=execute_now,
            action_decision=materialized_action_decision,
            scenario=materialized_scenario,
        )
        if risk_session_passthrough:
            setattr(playbook.position_management, "risk_session_passthrough", True)
        if converted_action_decision.action in MANAGEMENT_EXPOSURE_ACTION_VALUES:
            playbook.post_fill_risk_template = PositionManagementPlan(
                execute_now=False,
                action_decision=ManagementDecision(
                    **{
                        **build_empty_management_decision().to_dict(),
                        "entry_price": float(entry_action_decision.entry_price or 0.0),
                        "stop_loss_price": float(entry_action_decision.stop_loss_price or 0.0),
                        "leverage": int(entry_action_decision.requested_leverage or 0),
                    }
                ),
                scenario=None,
            )
        self._audit_event(
            "entry_plan_materialized",
            {
                "selected_symbol": playbook.selected_symbol,
                "current_side": current_side,
                "current_notional_usd": current_notional,
                "target_notional_usd": safe_float(materialization_debug.get("target_notional_usd"), 0.0) or 0.0,
                "comparison_target_notional_usd": safe_float(materialization_debug.get("comparison_target_notional_usd"), 0.0) or 0.0,
                "same_side_comparison_applied": bool(materialization_debug.get("same_side_comparison_applied", False)),
                "opposite_event_decision": materialization_debug.get("opposite_event_decision"),
                "materialization_debug": materialization_debug,
                "entry_plan": playbook.entry_plan.to_dict(),
                "materialized_position_management": playbook.position_management.to_dict(),
                "materialized_post_fill_risk_template": playbook.post_fill_risk_template.to_dict(),
            },
        )
        return playbook
    def _disable_nontradable_entry(self, playbook: GenericPlaybook, candidate: Dict[str, Any]) -> GenericPlaybook:
        note = (
            f"当前最值得关注的候选是 {playbook.selected_symbol or candidate.get('trade_symbol_key') or candidate.get('candidate_key')}, "
            "但该候选当前不能直接在 Hyperliquid 执行，因此本地已拦截自动执行。"
        )
        playbook.entry_plan.execute_now = False
        playbook.entry_plan.action_decision = build_empty_strategy_decision()
        playbook.entry_plan.scenario = None
        playbook.position_management = build_empty_position_management_plan()
        playbook.post_fill_risk_template = build_empty_position_management_plan()
        playbook.display_answer = f"{playbook.display_answer}\n\n[本地执行校验] {note}".strip()
        return playbook
