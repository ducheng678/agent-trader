import json
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

from market_agent.market_profiles import (
    InstrumentMarketProfile,
    LocalTimeWindow,
    WEEKDAY_NAME_TO_INDEX,
)
from market_agent.positions import snapshot_has_open_position
from market_agent.symbols import (
    canonicalize_execution_symbol,
    normalize_candidate_key,
    split_execution_symbol,
)


class HelperResetMixin:
    def _schedule_next_active_query(self, position_snapshot: Optional[dict] = None) -> None:
        snapshot = position_snapshot
        if snapshot is None:
            runtime_symbol = self._runtime_symbol()
            snapshot = self.reader.get_position_snapshot(runtime_symbol) if runtime_symbol else self._empty_runtime_snapshot()
        interval = self.active_management_query_interval_seconds if snapshot_has_open_position(snapshot) else self.active_query_interval_seconds
        self.next_active_query_due_at = time.time() + interval
    @staticmethod
    def _parse_profile_time(value: Any, default: Optional[Tuple[int, int, int]] = None) -> Optional[Tuple[int, int, int]]:
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            hour = int(value[0] or 0)
            minute = int(value[1] or 0)
            second = int(value[2] or 0) if len(value) >= 3 else 0
        else:
            raw = str(value or "").strip()
            if not raw:
                return default
            parts = raw.split(":")
            if len(parts) not in {2, 3}:
                return default
            hour = int(parts[0])
            minute = int(parts[1])
            second = int(parts[2]) if len(parts) == 3 else 0
        if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
            return default
        return hour, minute, second
    @classmethod
    def _parse_profile_window(cls, value: Any) -> Optional[LocalTimeWindow]:
        if isinstance(value, dict):
            start = cls._parse_profile_time(value.get("start"))
            end = cls._parse_profile_time(value.get("end"))
        else:
            raw = str(value or "").strip()
            if "-" not in raw:
                return None
            left, right = raw.split("-", 1)
            start = cls._parse_profile_time(left.strip())
            end = cls._parse_profile_time(right.strip())
        if start is None or end is None:
            return None
        return LocalTimeWindow(start, end)
    @classmethod
    def _parse_profile_windows(
        cls,
        value: Any,
        default: Tuple[LocalTimeWindow, ...] = (),
    ) -> Tuple[LocalTimeWindow, ...]:
        if value is None:
            return default
        raw_items = value if isinstance(value, list) else [value]
        windows = [window for window in (cls._parse_profile_window(item) for item in raw_items) if window is not None]
        return tuple(windows) if windows else default
    @staticmethod
    def _parse_profile_weekdays(value: Any, default: Tuple[int, ...] = ()) -> Tuple[int, ...]:
        if value is None:
            return default
        raw_items = value if isinstance(value, list) else [value]
        parsed: List[int] = []
        for item in raw_items:
            if isinstance(item, int):
                weekday = int(item)
            else:
                raw = str(item or "").strip().lower()
                if raw.isdigit():
                    weekday = int(raw)
                else:
                    weekday = WEEKDAY_NAME_TO_INDEX.get(raw, -1)
            if 0 <= weekday <= 6 and weekday not in parsed:
                parsed.append(weekday)
        return tuple(parsed) if parsed else default
    @staticmethod
    def _parse_profile_nonnegative_float(value: Any, default: float) -> float:
        try:
            return max(0.0, float(value if value is not None else default))
        except Exception:
            return max(0.0, float(default or 0.0))
    @staticmethod
    def _parse_profile_float(value: Any, default: float) -> float:
        try:
            return float(value if value is not None else default)
        except Exception:
            return float(default or 0.0)
    @classmethod
    def _parse_profile_fraction(cls, value: Any, default: float) -> float:
        return min(max(0.0, cls._parse_profile_float(value, default)), 1.0)
    @staticmethod
    def _profile_lookup_keys(symbol: str) -> List[str]:
        canonical = canonicalize_execution_symbol(symbol or "")
        _, asset = split_execution_symbol(canonical)
        raw = str(symbol or "").strip()
        base_tokens: List[str] = []
        for token in [raw, canonical, asset]:
            text = str(token or "").strip()
            if not text:
                continue
            base_tokens.append(text)
            upper = text.upper()
            if upper.endswith("-USDC"):
                base_tokens.append(text[:-5])
        keys = [
            raw.upper(),
            canonical.upper(),
            asset.upper(),
            normalize_candidate_key(raw),
            normalize_candidate_key(canonical),
            normalize_candidate_key(asset),
        ]
        for token in base_tokens:
            keys.append(str(token).upper())
            keys.append(normalize_candidate_key(token))
        seen: set[str] = set()
        return [key for key in keys if key and not (key in seen or seen.add(key))]
    @classmethod
    def _coerce_instrument_market_profile(
        cls,
        key: str,
        value: Any,
    ) -> Optional[InstrumentMarketProfile]:
        if not isinstance(value, dict):
            return None
        timezone_name = str(value.get("timezone_name") or value.get("timezone") or "").strip()
        if not timezone_name:
            return None
        try:
            ZoneInfo(timezone_name)
        except Exception:
            return None
        helper_reset_timezone_name = str(
            value.get("helper_reset_timezone_name")
            or value.get("helper_reset_timezone")
            or value.get("reset_timezone")
            or ""
        ).strip()
        if helper_reset_timezone_name:
            try:
                ZoneInfo(helper_reset_timezone_name)
            except Exception:
                return None
        else:
            helper_reset_timezone_name = None
        helper_reset_time = cls._parse_profile_time(
            value.get("helper_reset_time", value.get("reset_time")),
        )
        pre_disabled_weekday_reset_time = cls._parse_profile_time(
            value.get("pre_disabled_weekday_reset_time"),
        )
        low_liquidity_windows = cls._parse_profile_windows(
            value.get("low_liquidity_windows", value.get("low_liquidity")),
            (),
        )
        normal_liquidity_windows = cls._parse_profile_windows(
            value.get("normal_liquidity_windows", value.get("normal_liquidity")),
            (),
        )
        return InstrumentMarketProfile(
            name=str(value.get("name") or normalize_candidate_key(key).lower() or key).strip(),
            timezone_name=timezone_name,
            helper_reset_time=helper_reset_time,
            low_liquidity_windows=low_liquidity_windows,
            normal_liquidity_windows=normal_liquidity_windows,
            pre_disabled_weekday_reset_time=pre_disabled_weekday_reset_time,
            helper_reset_timezone_name=helper_reset_timezone_name,
            low_liquidity_weekdays=cls._parse_profile_weekdays(
                value.get("low_liquidity_weekdays", value.get("low_liquidity_days")),
                (),
            ),
            low_liquidity_trade_disabled_weekdays=cls._parse_profile_weekdays(
                value.get(
                    "low_liquidity_trade_disabled_weekdays",
                    value.get("low_liquidity_no_trade_weekdays", value.get("no_trade_weekdays")),
                ),
                (),
            ),
            reset_skip_weekdays=cls._parse_profile_weekdays(
                value.get("reset_skip_weekdays", value.get("reset_skip_days")),
                (),
            ),
            atr_ref_timeframe=str(value.get("atr_ref_timeframe") or "15m").strip().lower(),
            atr_ref_period=max(1, int(value.get("atr_ref_period", 14) or 14)),
            atr_ref_lookback_days=max(1, int(value.get("atr_ref_lookback_days", 5) or 5)),
            atr_ref_lookback_bars=max(50, int(value.get("atr_ref_lookback_bars", 700) or 700)),
            normal_liquidity_r_min_atr_multiple=cls._parse_profile_nonnegative_float(
                value.get("normal_liquidity_r_min_atr_multiple"),
                1.5,
            ),
            normal_liquidity_r_max_atr_multiple=cls._parse_profile_nonnegative_float(
                value.get("normal_liquidity_r_max_atr_multiple"),
                2.5,
            ),
            low_liquidity_r_min_atr_multiple=cls._parse_profile_nonnegative_float(
                value.get("low_liquidity_r_min_atr_multiple"),
                2.5,
            ),
            low_liquidity_r_max_atr_multiple=cls._parse_profile_nonnegative_float(
                value.get("low_liquidity_r_max_atr_multiple"),
                3.0,
            ),
            normal_liquidity_tp1_r_multiple=cls._parse_profile_nonnegative_float(
                value.get("normal_liquidity_tp1_r_multiple"),
                1.0,
            ),
            normal_liquidity_tp2_r_multiple=cls._parse_profile_nonnegative_float(
                value.get("normal_liquidity_tp2_r_multiple"),
                2.0,
            ),
            normal_liquidity_tp1_close_fraction=cls._parse_profile_fraction(
                value.get("normal_liquidity_tp1_close_fraction"),
                0.30,
            ),
            normal_liquidity_tp2_close_fraction=cls._parse_profile_fraction(
                value.get("normal_liquidity_tp2_close_fraction"),
                0.40,
            ),
            normal_liquidity_post_tp1_stop_r_multiple=cls._parse_profile_float(
                value.get("normal_liquidity_post_tp1_stop_r_multiple"),
                -0.40,
            ),
            normal_liquidity_post_tp2_locked_r_multiple=cls._parse_profile_float(
                value.get("normal_liquidity_post_tp2_locked_r_multiple"),
                1.0,
            ),
            normal_liquidity_trailing_soft_atr_multiple=cls._parse_profile_nonnegative_float(
                value.get("normal_liquidity_trailing_soft_atr_multiple"),
                2.5,
            ),
            normal_liquidity_trailing_hard_atr_multiple=cls._parse_profile_nonnegative_float(
                value.get("normal_liquidity_trailing_hard_atr_multiple"),
                3.5,
            ),
            low_liquidity_tp1_r_multiple=cls._parse_profile_nonnegative_float(
                value.get("low_liquidity_tp1_r_multiple"),
                0.75,
            ),
            low_liquidity_tp2_r_multiple=cls._parse_profile_nonnegative_float(
                value.get("low_liquidity_tp2_r_multiple"),
                1.25,
            ),
            low_liquidity_tp1_close_fraction=cls._parse_profile_fraction(
                value.get("low_liquidity_tp1_close_fraction"),
                0.40,
            ),
            low_liquidity_tp2_close_fraction=cls._parse_profile_fraction(
                value.get("low_liquidity_tp2_close_fraction"),
                0.40,
            ),
            low_liquidity_post_tp1_stop_r_multiple=cls._parse_profile_float(
                value.get("low_liquidity_post_tp1_stop_r_multiple"),
                -0.40,
            ),
            low_liquidity_post_tp2_locked_r_multiple=cls._parse_profile_float(
                value.get("low_liquidity_post_tp2_locked_r_multiple"),
                0.75,
            ),
            low_liquidity_trailing_soft_atr_multiple=cls._parse_profile_nonnegative_float(
                value.get("low_liquidity_trailing_soft_atr_multiple"),
                1.5,
            ),
            low_liquidity_trailing_hard_atr_multiple=cls._parse_profile_nonnegative_float(
                value.get("low_liquidity_trailing_hard_atr_multiple"),
                2.5,
            ),
        )
    @staticmethod
    def _load_profile_payload_from_file() -> Dict[str, Any]:
        raw_path = str(os.getenv("INSTRUMENT_MARKET_PROFILES_PATH", "config/instrument_market_profiles.json") or "").strip()
        if not raw_path:
            return {}
        candidate_paths = [raw_path] if os.path.isabs(raw_path) else [
            os.path.abspath(raw_path),
            os.path.join(os.path.dirname(os.path.dirname(__file__)), raw_path),
        ]
        path = next((item for item in candidate_paths if os.path.exists(item)), candidate_paths[0])
        if not os.path.exists(path):
            print(f"[warn] instrument market profile file not found: {path}")
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("profiles"), dict):
                data = data.get("profiles")
            return dict(data) if isinstance(data, dict) else {}
        except Exception as exc:
            print(f"[warn] failed to parse instrument market profile file {path}: {exc}")
            return {}
    def _load_instrument_market_profiles(self) -> Dict[str, InstrumentMarketProfile]:
        profiles: Dict[str, InstrumentMarketProfile] = {}

        def register(key: str, profile: InstrumentMarketProfile) -> None:
            for lookup_key in self._profile_lookup_keys(key):
                profiles[lookup_key] = profile

        for key, raw_profile in self._load_profile_payload_from_file().items():
            profile = self._coerce_instrument_market_profile(str(key), raw_profile)
            if profile is not None:
                register(str(key), profile)
                if isinstance(raw_profile, dict):
                    for alias in list(raw_profile.get("aliases") or []):
                        register(str(alias), profile)
            else:
                print(f"[warn] ignored invalid instrument market profile: {key}")
        return profiles
    def _market_profile_for_symbol(self, symbol: str = "") -> Optional[InstrumentMarketProfile]:
        profiles = getattr(self, "instrument_market_profiles", {}) or {}
        for key in self._profile_lookup_keys(symbol):
            if key in profiles:
                return profiles[key]
        return None
    def _helper_reset_symbol_hint(self) -> str:
        current = canonicalize_execution_symbol(getattr(self, "symbol", "") or "")
        if current:
            return current
        context = dict(getattr(self, "trade_symbol_context", {}) or {})
        execution_symbol = canonicalize_execution_symbol(context.get("execution_symbol", ""))
        if execution_symbol:
            return execution_symbol
        return ""
    @staticmethod
    def _profile_timezone(profile: InstrumentMarketProfile) -> timezone:
        try:
            return ZoneInfo(str(profile.timezone_name or "UTC"))
        except Exception:
            return timezone.utc
    @staticmethod
    def _profile_helper_reset_timezone(profile: InstrumentMarketProfile) -> timezone:
        try:
            return ZoneInfo(str(profile.helper_reset_timezone_name or profile.timezone_name or "UTC"))
        except Exception:
            return timezone.utc
    @staticmethod
    def _profile_time_windows_contain(
        profile: InstrumentMarketProfile,
        windows: Tuple[LocalTimeWindow, ...],
        now_utc: datetime,
    ) -> bool:
        zone = HelperResetMixin._profile_timezone(profile)
        local_now = now_utc.astimezone(zone)
        return any(window.contains(local_now) for window in list(windows or ()))
    @staticmethod
    def _profile_low_liquidity_weekday_contains(
        profile: InstrumentMarketProfile,
        now_utc: datetime,
    ) -> bool:
        zone = HelperResetMixin._profile_timezone(profile)
        local_now = now_utc.astimezone(zone)
        return local_now.weekday() in set(int(item) for item in list(profile.low_liquidity_weekdays or ()))
    def _low_liquidity_profile_for_symbol(self, symbol: str, now_utc: Optional[datetime] = None) -> Optional[InstrumentMarketProfile]:
        profile = self._market_profile_for_symbol(symbol)
        if profile is None:
            return None
        now = now_utc or datetime.now(timezone.utc)
        if self._profile_low_liquidity_weekday_contains(profile, now):
            return profile
        if self._profile_time_windows_contain(profile, profile.low_liquidity_windows, now):
            return profile
        return None
    def _low_liquidity_trade_disabled_profile_for_symbol(self, symbol: str, now_utc: Optional[datetime] = None) -> Optional[InstrumentMarketProfile]:
        profile = self._market_profile_for_symbol(symbol)
        if profile is None:
            return None
        now = now_utc or datetime.now(timezone.utc)
        zone = self._profile_timezone(profile)
        local_now = now.astimezone(zone)
        disabled_weekdays = set(int(item) for item in list(profile.low_liquidity_trade_disabled_weekdays or ()))
        if local_now.weekday() in disabled_weekdays:
            return profile
        return None
    def _next_profile_reset_at(
        self,
        profile: InstrumentMarketProfile,
        now_utc: datetime,
    ) -> Optional[datetime]:
        return self._next_profile_local_time_at(
            profile,
            now_utc,
            profile.helper_reset_time,
            zone=self._profile_helper_reset_timezone(profile),
            skip_weekdays=profile.reset_skip_weekdays,
        )
    def _next_profile_pre_disabled_weekday_reset_at(
        self,
        profile: InstrumentMarketProfile,
        now_utc: datetime,
    ) -> Optional[datetime]:
        if not profile.low_liquidity_trade_disabled_weekdays:
            return None
        return self._next_profile_local_time_at(
            profile,
            now_utc,
            profile.pre_disabled_weekday_reset_time,
            only_weekdays=profile.low_liquidity_trade_disabled_weekdays,
        )
    def _next_profile_local_time_at(
        self,
        profile: InstrumentMarketProfile,
        now_utc: datetime,
        reset_time: Optional[Tuple[int, int, int]],
        *,
        zone: Optional[timezone] = None,
        only_weekdays: Tuple[int, ...] = (),
        skip_weekdays: Tuple[int, ...] = (),
    ) -> Optional[datetime]:
        if reset_time is None:
            return None
        zone = zone or self._profile_timezone(profile)
        local_now = now_utc.astimezone(zone)
        hour, minute, second = reset_time
        target_local = local_now.replace(hour=hour, minute=minute, second=second, microsecond=0)
        if local_now > target_local:
            target_local = target_local + timedelta(days=1)
        only_weekdays_set = set(int(item) for item in list(only_weekdays or ()))
        skip_weekdays_set = set(int(item) for item in list(skip_weekdays or ()))
        guard = 0
        while (
            (only_weekdays_set and target_local.weekday() not in only_weekdays_set)
            or target_local.weekday() in skip_weekdays_set
        ):
            target_local = target_local + timedelta(days=1)
            guard += 1
            if guard > 7:
                return None
        return target_local.astimezone(timezone.utc)
    def _reschedule_helper_reset(self, reason: str = "") -> None:
        if not hasattr(self, "next_helper_reset_at"):
            return
        self.next_helper_reset_at = self._compute_next_helper_reset_at()
        self._audit_event(
            "helper_reset_rescheduled",
            {
                "reason": reason,
                "symbol": self._helper_reset_symbol_hint(),
                "next_helper_reset_at": self.next_helper_reset_at.isoformat() if isinstance(self.next_helper_reset_at, datetime) else None,
            },
        )
    def _compute_next_helper_reset_at(self, now_utc: Optional[datetime] = None) -> Optional[datetime]:
        now = now_utc or datetime.now(timezone.utc)
        profile = self._market_profile_for_symbol(self._helper_reset_symbol_hint())
        if profile is None:
            return None
        candidates = [
            candidate
            for candidate in (
                self._next_profile_reset_at(profile, now),
                self._next_profile_pre_disabled_weekday_reset_at(profile, now),
            )
            if isinstance(candidate, datetime)
        ]
        return min(candidates) if candidates else None
    def _helper_reset_due(self, now_utc: datetime) -> bool:
        target = getattr(self, "next_helper_reset_at", None)
        return isinstance(target, datetime) and now_utc >= target
    def _advance_next_helper_reset_at(self) -> None:
        target = getattr(self, "next_helper_reset_at", None)
        if isinstance(target, datetime):
            self.next_helper_reset_at = self._compute_next_helper_reset_at(target + timedelta(seconds=1))
    def _clear_state_for_helper_reset(self) -> None:
        self._cancel_pending_entry_order("helper_reset_time")
        self.position_management_session = None
        clear_basis = getattr(self, "_clear_position_basis_state", None)
        if callable(clear_basis):
            clear_basis(reason="helper_reset_time")
        self._replace_risk_session(None)
        self.current_playbook = None
        self.current_mode = None
        self.current_playbook_reason = ""
    def _perform_helper_reset(self, now_utc: datetime) -> bool:
        if not self._helper_reset_due(now_utc):
            return False
        scheduled_for = getattr(self, "next_helper_reset_at", None)
        all_positions = self.reader.get_all_positions()
        had_open_position = any(
            snapshot_has_open_position(pos)
            for pos in list((all_positions or {}).get("positions") or [])
            if isinstance(pos, dict)
        )
        risk_session = getattr(self, "risk_session", None)
        tp2_no_continuation_applied = bool(
            getattr(risk_session, "tp2_no_continuation_applied", False)
        ) if risk_session is not None else False
        if had_open_position and not tp2_no_continuation_applied:
            payload = {
                "scheduled_for": scheduled_for.isoformat() if isinstance(scheduled_for, datetime) else None,
                "checked_at": now_utc.isoformat(),
                "had_open_position": True,
                "tp2_no_continuation_applied": tp2_no_continuation_applied,
                "has_risk_session": risk_session is not None,
            }
            deferred_key = str(payload["scheduled_for"] or "")
            if getattr(self, "_last_helper_reset_deferred_waiting_tp2_key", "") != deferred_key:
                self._last_helper_reset_deferred_waiting_tp2_key = deferred_key
                print("[helper_reset] deferred; waiting for tp2_no_continuation before reset")
                self._audit_event("helper_reset_deferred_waiting_tp2_no_continuation", payload)
            return False

        self._advance_next_helper_reset_at()
        flatten_outcome = {"results": [], "all_accepted": True}
        if had_open_position:
            flatten_outcome = self._flatten_unselected_positions("", all_positions, reason="helper_reset_time")
            all_positions = self.reader.get_all_positions()
        has_open_position_after = any(
            snapshot_has_open_position(pos)
            for pos in list((all_positions or {}).get("positions") or [])
            if isinstance(pos, dict)
        )
        payload = {
            "scheduled_for": scheduled_for.isoformat() if isinstance(scheduled_for, datetime) else None,
            "executed_at": now_utc.isoformat(),
            "had_open_position": had_open_position,
            "has_open_position_after": has_open_position_after,
            "tp2_no_continuation_applied": tp2_no_continuation_applied,
            "has_risk_session": risk_session is not None,
            "flatten_outcome": flatten_outcome,
        }
        if has_open_position_after:
            print("[helper_reset] positions remain open after reset flatten attempt; helper skipped")
            self._audit_event("helper_reset_skipped_positions_remain_open", payload)
            return False
        self._clear_state_for_helper_reset()
        print("[helper_reset] state cleared; running helper refresh")
        self._audit_event("helper_reset_triggered", payload)
        return True
