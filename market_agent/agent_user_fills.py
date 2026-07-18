from typing import Any, Dict, List, Optional, Tuple
import time

from market_agent.symbols import canonicalize_execution_symbol
from market_agent.utils import safe_float


class UserFillsMixin:
    def _ensure_user_fills_subscription(
        self,
        *,
        now: Optional[float] = None,
        force_reconnect: bool = False,
    ) -> bool:
        if not bool(getattr(self, "enable_user_fills_websocket", False)):
            return False
        now_ts = float(time.time() if now is None else now)
        address = str(getattr(self, "user_fills_address", "") or getattr(self.reader, "account_address", "")).strip()
        if not address or not hasattr(self.reader, "subscribe_user_fills"):
            return False
        subscription_id = getattr(self, "user_fills_subscription_id", None)
        health_check = getattr(self.reader, "user_fills_ws_is_healthy", None)
        healthy = bool(health_check()) if callable(health_check) else subscription_id is not None
        if subscription_id is not None and healthy and not force_reconnect:
            return True
        retry_seconds = max(1.0, float(getattr(self, "user_fills_reconnect_retry_seconds", 10.0) or 10.0))
        last_attempt = float(getattr(self, "user_fills_last_subscribe_attempt_at", 0.0) or 0.0)
        if now_ts - last_attempt < retry_seconds:
            return False
        reconnecting = subscription_id is not None
        if reconnecting and hasattr(self.reader, "disconnect_ws"):
            try:
                self.reader.disconnect_ws()
            except Exception:
                pass
            self.user_fills_subscription_id = None
        try:
            self.user_fills_last_subscribe_attempt_at = now_ts
            self.user_fills_subscription_id = self.reader.subscribe_user_fills(address, self._on_user_fills_ws_message)
            self.user_fills_address = address
            event_name = "user_fills_ws_resubscribed" if reconnecting else "user_fills_ws_subscribed"
            print(f"[{event_name}] user={address}")
            self._audit_event(event_name, {"user": address})
            return True
        except Exception as exc:
            event_name = "user_fills_ws_resubscribe_failed" if reconnecting else "user_fills_ws_subscribe_failed"
            print(f"[warn] {event_name} user={address} error={exc}")
            self._audit_event(event_name, {"user": address, "message": str(exc)})
            return False

    def _maintain_user_fills_subscription(self, now: Optional[float] = None) -> bool:
        if not bool(getattr(self, "enable_user_fills_websocket", False)):
            return False
        return self._ensure_user_fills_subscription(now=now)

    def shutdown(self) -> None:
        shutdown_cross_asset_poller = getattr(self, "_shutdown_cross_asset_soft_stop_poller", None)
        if callable(shutdown_cross_asset_poller):
            shutdown_cross_asset_poller()
        shutdown_passive_prefetch = getattr(self, "_shutdown_passive_event_judge_prefetch", None)
        if callable(shutdown_passive_prefetch):
            shutdown_passive_prefetch()
        subscription_id = getattr(self, "user_fills_subscription_id", None)
        address = str(getattr(self, "user_fills_address", "") or getattr(self.reader, "account_address", "")).strip()
        if subscription_id is not None and hasattr(self.reader, "unsubscribe_user_fills"):
            try:
                self.reader.unsubscribe_user_fills(address, int(subscription_id))
            except Exception:
                pass
            self.user_fills_subscription_id = None
        if hasattr(self.reader, "disconnect_ws"):
            self.reader.disconnect_ws()

    def _make_user_fill_event_key(self, fill: Dict[str, Any]) -> Tuple[Any, ...]:
        return (
            canonicalize_execution_symbol(fill.get("coin", "")),
            int(safe_float(fill.get("oid"), 0.0) or 0.0),
            int(safe_float(fill.get("tid"), 0.0) or 0.0),
            int(safe_float(fill.get("time"), 0.0) or 0.0),
            str(fill.get("sz", "") or ""),
            str(fill.get("px", "") or ""),
            str(fill.get("hash", "") or ""),
        )

    def _remember_user_fill_event_key(self, key: Tuple[Any, ...]) -> None:
        seen_set = getattr(self, "_user_fills_seen_key_set", None)
        seen_queue = getattr(self, "_user_fills_seen_keys", None)
        if seen_set is None or seen_queue is None:
            return
        if key in seen_set:
            return
        seen_set.add(key)
        seen_queue.append(key)
        capacity = max(100, int(getattr(self, "user_fills_seen_capacity", 4000) or 4000))
        while len(seen_queue) > capacity:
            expired = seen_queue.popleft()
            seen_set.discard(expired)

    def _on_user_fills_ws_message(self, ws_msg: Dict[str, Any]) -> None:
        self.user_fills_last_message_at = time.time()
        data = dict((ws_msg or {}).get("data") or {}) if isinstance(ws_msg, dict) else {}
        fills = [dict(item) for item in list(data.get("fills") or []) if isinstance(item, dict)]
        buffer = getattr(self, "user_fills_event_buffer", None)
        if buffer is None:
            return
        for fill in fills:
            fill_time_ms = int(safe_float(fill.get("time"), 0.0) or 0.0)
            if fill_time_ms > int(getattr(self, "user_fills_last_fill_time_ms", 0) or 0):
                self.user_fills_last_fill_time_ms = fill_time_ms
            key = self._make_user_fill_event_key(fill)
            if key in getattr(self, "_user_fills_seen_key_set", set()):
                continue
            self._remember_user_fill_event_key(key)
            buffer.append(fill)

    def _drain_pending_user_fill_events(self) -> List[Dict[str, Any]]:
        buffer = getattr(self, "user_fills_event_buffer", None)
        if buffer is None:
            return []
        events: List[Dict[str, Any]] = []
        while buffer:
            events.append(dict(buffer.popleft()))
        return events

    def _backfill_recent_user_fills(self, now: float, *, force: bool = False) -> List[Dict[str, Any]]:
        if not bool(getattr(self, "enable_user_fills_websocket", False)):
            return []
        if not hasattr(self.reader, "get_user_fills_by_time"):
            return []
        poll_seconds = max(0.5, float(getattr(self, "user_fills_backfill_poll_seconds", 5.0) or 5.0))
        last_backfill_at = float(getattr(self, "user_fills_last_backfill_at", 0.0) or 0.0)
        if not force and now - last_backfill_at < poll_seconds:
            return []
        address = str(getattr(self, "user_fills_address", "") or getattr(self.reader, "account_address", "")).strip()
        if not address:
            return []
        end_time_ms = max(0, int(now * 1000))
        lookback_seconds = max(
            poll_seconds,
            float(getattr(self, "user_fills_backfill_lookback_seconds", 120.0) or 120.0),
        )
        overlap_ms = max(
            1000,
            int(float(getattr(self, "user_fills_reconcile_grace_seconds", 3.0) or 3.0) * 1000),
        )
        last_fill_time_ms = int(getattr(self, "user_fills_last_fill_time_ms", 0) or 0)
        start_time_ms = (
            max(0, last_fill_time_ms - overlap_ms)
            if last_fill_time_ms > 0
            else max(0, end_time_ms - int(lookback_seconds * 1000))
        )
        self.user_fills_last_backfill_at = now
        try:
            raw_fills = list(
                self.reader.get_user_fills_by_time(
                    address,
                    start_time_ms,
                    end_time_ms,
                    aggregate_by_time=False,
                )
                or []
            )
        except Exception:
            return []
        recovered: List[Dict[str, Any]] = []
        max_fill_time_ms = last_fill_time_ms
        for item in sorted(raw_fills, key=lambda fill: int(safe_float((fill or {}).get("time"), 0.0) or 0.0)):
            if not isinstance(item, dict):
                continue
            fill = dict(item)
            fill_time_ms = int(safe_float(fill.get("time"), 0.0) or 0.0)
            if fill_time_ms > max_fill_time_ms:
                max_fill_time_ms = fill_time_ms
            key = self._make_user_fill_event_key(fill)
            if key in getattr(self, "_user_fills_seen_key_set", set()):
                continue
            self._remember_user_fill_event_key(key)
            recovered.append(fill)
        if max_fill_time_ms > last_fill_time_ms:
            self.user_fills_last_fill_time_ms = max_fill_time_ms
        if recovered:
            self._audit_event(
                "user_fills_rest_backfilled",
                {
                    "count": len(recovered),
                    "start_time_ms": start_time_ms,
                    "end_time_ms": end_time_ms,
                },
            )
        return recovered

    def _user_fills_ws_is_active(self) -> bool:
        if not bool(getattr(self, "enable_user_fills_websocket", False)):
            return False
        if getattr(self, "user_fills_subscription_id", None) is None:
            return False
        health_check = getattr(self.reader, "user_fills_ws_is_healthy", None)
        if callable(health_check):
            return bool(health_check())
        return True
