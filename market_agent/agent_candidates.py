from typing import Any, Dict, List, Optional, Tuple
import os
import time

from market_agent.charting import _build_chart_image_timeframe_specs
from market_agent.constants import PM_SCENARIO_REQUERY_LOCK_REASONS
from market_agent.exchange import HyperliquidExecutor
from market_agent.playbook import GenericPlaybook
from market_agent.positions import snapshot_has_open_position
from market_agent.symbols import (
    canonicalize_execution_symbol,
    normalize_candidate_key,
    parse_trade_symbol_context,
    split_execution_symbol,
)


def active_chart_image_timeframe_specs() -> Tuple[Dict[str, Any], ...]:
    return _build_chart_image_timeframe_specs(
        os.getenv("OPENAI_ACTIVE_CHART_IMAGE_TIMEFRAMES", ""),
        ("1m", "5m", "15m"),
    )


def passive_chart_image_timeframe_specs() -> Tuple[Dict[str, Any], ...]:
    return _build_chart_image_timeframe_specs(
        os.getenv("OPENAI_PASSIVE_CHART_IMAGE_TIMEFRAMES", ""),
        ("1m",),
    )


class TradeSymbolContextMixin:
    def _load_trade_symbol_context(self) -> Dict[str, Any]:
        configured = os.getenv("TRADE_SYMBOL", "")
        raw_context = parse_trade_symbol_context(configured)
        if not raw_context:
            raise RuntimeError("TRADE_SYMBOL must configure exactly one tradable symbol.")
        tradable_symbols = set(self.reader.list_perp_symbols())
        trade_symbol_key = str(raw_context.get("trade_symbol_key", "") or "").strip().upper()
        requested_execution_symbol = canonicalize_execution_symbol(raw_context.get("configured_execution_symbol", ""))
        display_name = str(raw_context.get("display_name", "") or trade_symbol_key).strip()
        inferred_symbol = requested_execution_symbol or self.reader.resolve_execution_symbol(display_name) or self.reader.resolve_execution_symbol(trade_symbol_key)
        execution_symbol = inferred_symbol if inferred_symbol in tradable_symbols else ""
        market_spec = self.reader.get_market_spec(execution_symbol) if execution_symbol else {}
        market_name = str((market_spec or {}).get("market_name", "") or "").strip()
        if not market_name:
            _, market_name = split_execution_symbol(execution_symbol or display_name)
        display_symbol = str((market_spec or {}).get("display_name", "") or "").strip()
        if not display_symbol:
            display_symbol = display_name or (f"{market_name}-USDC" if market_name else "")
        canonical_symbol_key = normalize_candidate_key(display_symbol or market_name or trade_symbol_key)
        context = {
            "trade_symbol_key": canonical_symbol_key or trade_symbol_key,
            "canonical_symbol_key": canonical_symbol_key or trade_symbol_key,
            "market_name": market_name,
            "display_symbol": display_symbol or display_name,
            "display_name": display_symbol or display_name,
            "configured_execution_symbol": requested_execution_symbol,
            "execution_symbol": execution_symbol,
            "tradable_on_hyperliquid": bool(execution_symbol),
        }
        if not context.get("execution_symbol"):
            raise RuntimeError("No tradable Hyperliquid candidate symbols are configured.")
        return context

    @staticmethod
    def _trade_symbol_matches_selected_symbol(trade_symbol_context: Dict[str, Any], selected_symbol: str) -> bool:
        raw = str(selected_symbol or "").strip().upper()
        normalized = normalize_candidate_key(raw)
        context = trade_symbol_context if isinstance(trade_symbol_context, dict) else {}
        tokens = {
            str(context.get("trade_symbol_key", "") or context.get("candidate_key", "") or "").strip().upper(),
            normalize_candidate_key(context.get("trade_symbol_key", "") or context.get("candidate_key", "")),
            str(context.get("canonical_symbol_key", "") or "").strip().upper(),
            normalize_candidate_key(context.get("canonical_symbol_key", "")),
            str(context.get("market_name", "") or "").strip().upper(),
            normalize_candidate_key(context.get("market_name", "")),
            str(context.get("display_symbol", "") or "").strip().upper(),
            normalize_candidate_key(context.get("display_symbol", "")),
            str(context.get("display_name", "") or "").strip().upper(),
            normalize_candidate_key(context.get("display_name", "")),
            str(context.get("configured_execution_symbol", "") or "").strip().upper(),
            normalize_candidate_key(context.get("configured_execution_symbol", "")),
            str(context.get("execution_symbol", "") or "").strip().upper(),
            normalize_candidate_key(context.get("execution_symbol", "")),
        }
        return bool(raw and (raw in tokens or normalized in tokens))

    def _assert_open_positions_match_trade_symbol(self, all_positions: dict) -> None:
        positions = [pos for pos in (all_positions.get("positions", []) or []) if snapshot_has_open_position(pos)]
        if not positions:
            return
        configured_context = dict(getattr(self, "trade_symbol_context", {}) or {})
        configured_symbol = canonicalize_execution_symbol(configured_context.get("execution_symbol", ""))
        if not configured_symbol:
            raise RuntimeError("TRADE_SYMBOL context is not configured.")
        mismatches: List[str] = []
        for pos in positions:
            symbol = canonicalize_execution_symbol(pos.get("symbol", ""))
            if symbol and symbol != configured_symbol:
                side = str(pos.get("side", "") or "").strip()
                size = pos.get("size")
                notional = pos.get("notional_usd")
                details = symbol
                if side:
                    details += f" side={side}"
                if size not in (None, ""):
                    details += f" size={size}"
                if notional not in (None, ""):
                    details += f" notional_usd={notional}"
                mismatches.append(details)
        if mismatches:
            raise RuntimeError(
                "Open position symbol does not match configured TRADE_SYMBOL "
                f"{configured_symbol}: {', '.join(mismatches)}"
            )

    def _set_active_symbol(self, symbol: str, reason: str = "") -> None:
        target = canonicalize_execution_symbol(symbol)
        if not target:
            if hasattr(self, "engine") and self.engine is not None:
                self.engine.symbol = self.symbol or ""
            return
        old_symbol = canonicalize_execution_symbol(self.symbol or "")
        current_executor_symbol = canonicalize_execution_symbol(getattr(getattr(self, "executor", None), "symbol", "") or "")
        if target == old_symbol and current_executor_symbol == target:
            if hasattr(self, "engine") and self.engine is not None:
                self.engine.symbol = target
            return
        self.symbol = target
        if hasattr(self, "engine") and self.engine is not None:
            self.engine.symbol = target
        if current_executor_symbol != target:
            executor = getattr(self, "executor", None)
            if executor is not None and hasattr(executor, "symbol"):
                executor.symbol = target
            else:
                self.executor = HyperliquidExecutor(self.reader, target)
        old_label = canonicalize_execution_symbol(old_symbol or "") or "<none>"
        if old_label != target:
            print(f"[execution_symbol_switch] {old_label} -> {target} reason={reason or 'llm_selection'}")
            self._reschedule_helper_reset(reason=reason or "symbol_switch")

    def _find_management_symbol(self, all_positions: dict) -> str:
        self._assert_open_positions_match_trade_symbol(all_positions)
        positions = [pos for pos in (all_positions.get("positions", []) or []) if snapshot_has_open_position(pos)]
        if not positions:
            return ""
        if any(canonicalize_execution_symbol(pos.get("symbol", "")) == self.symbol for pos in positions):
            return self.symbol
        largest = max(positions, key=lambda pos: abs(float(pos.get("notional_usd", 0.0) or 0.0)))
        return canonicalize_execution_symbol(largest.get("symbol", ""))

    def _runtime_symbol(self, all_positions: Optional[dict] = None) -> str:
        snapshot = all_positions if isinstance(all_positions, dict) else self.reader.get_all_positions()
        return self._find_management_symbol(snapshot) or canonicalize_execution_symbol(self.symbol or "")

    def _empty_runtime_snapshot(self, all_positions: Optional[dict] = None, symbol: str = "") -> Dict[str, Any]:
        snapshot = all_positions if isinstance(all_positions, dict) else self.reader.get_all_positions()
        return {
            "known": True,
            "account_address": getattr(self.reader, "account_address", str(snapshot.get("account_address", "") or "")),
            "network": getattr(self.reader, "network", str(snapshot.get("network", "") or "")),
            "symbol": canonicalize_execution_symbol(symbol),
            "side": "flat",
            "size": 0.0,
            "entry_price": 0.0,
            "mid_price": None,
            "notional_usd": 0.0,
            "account_equity_usd": float(snapshot.get("account_equity_usd", 0.0) or 0.0),
            "available_margin_usd": float(snapshot.get("available_margin_usd", 0.0) or 0.0),
            "withdrawable_usd": float(snapshot.get("withdrawable_usd", 0.0) or 0.0),
            "remaining_capital_usd": float(snapshot.get("remaining_capital_usd", 0.0) or 0.0),
        }

    def _resolve_query_trade_symbol_context(self, all_positions: dict, reason: str = "") -> Tuple[Dict[str, Any], str]:
        self._assert_open_positions_match_trade_symbol(all_positions)
        configured_context = dict(getattr(self, "trade_symbol_context", {}) or {})
        current_selected_symbol = ""
        if getattr(self, "current_playbook", None) is not None:
            current_selected_symbol = str(getattr(self.current_playbook, "selected_symbol", "") or "").strip()
        sticky_selected_symbol = bool(current_selected_symbol) and (
            getattr(self, "position_management_session", None) is not None or reason in PM_SCENARIO_REQUERY_LOCK_REASONS
        )
        if sticky_selected_symbol:
            if self._trade_symbol_matches_selected_symbol(configured_context, current_selected_symbol):
                return configured_context, ""
        return configured_context, ""

    def _flatten_unselected_positions(
        self,
        selected_execution_symbol: str,
        all_positions: dict,
        *,
        reason: str,
    ) -> Dict[str, Any]:
        keep_symbol = canonicalize_execution_symbol(selected_execution_symbol)
        results: List[Dict[str, Any]] = []
        all_accepted = True
        for pos in (all_positions.get("positions", []) or []):
            if not snapshot_has_open_position(pos):
                continue
            symbol = canonicalize_execution_symbol(pos.get("symbol", ""))
            if not symbol or symbol == keep_symbol:
                continue
            side = str(pos.get("side", "flat") or "flat")
            executor = self.executor if symbol == self.symbol else HyperliquidExecutor(self.reader, symbol)
            result = executor.close_position(side, reason, f"{reason}:{symbol}")
            result["accepted"] = not executor._result_has_exchange_error(result)
            result["position_after"] = self.reader.get_position_snapshot(symbol)
            results.append(result)
            if executor.enabled and not result.get("accepted", False):
                all_accepted = False
            if executor.enabled:
                time.sleep(0.5)
        if results:
            self._audit_event(
                "unselected_positions_flattened",
                {
                    "selected_execution_symbol": keep_symbol,
                    "reason": reason,
                    "results": results,
                    "all_accepted": all_accepted,
                },
            )
        return {"results": results, "all_accepted": all_accepted}

    def _build_trade_symbol_context(self, all_positions: dict, trade_symbol_context: Dict[str, Any]) -> Dict[str, Any]:
        item = dict(trade_symbol_context or {})
        execution_symbol = canonicalize_execution_symbol(item.get("execution_symbol", ""))
        context = {
            "trade_symbol_key": str(item.get("trade_symbol_key", "") or item.get("candidate_key", "") or "").strip().upper(),
            "canonical_symbol_key": str(item.get("canonical_symbol_key", "") or item.get("trade_symbol_key", "") or item.get("candidate_key", "") or "").strip().upper(),
            "market_name": str(item.get("market_name", "") or "").strip(),
            "display_symbol": str(item.get("display_symbol", "") or item.get("display_name", "") or "").strip(),
            "display_name": str(item.get("display_name", "") or "").strip(),
            "configured_execution_symbol": canonicalize_execution_symbol(item.get("configured_execution_symbol", "")),
            "execution_symbol": execution_symbol,
            "tradable_on_hyperliquid": bool(execution_symbol),
        }
        if execution_symbol:
            context["current_price"] = self.reader.get_mid_price(execution_symbol)
            context["market_spec"] = self.reader.get_market_spec(execution_symbol)
        else:
            context["current_price"] = None
            context["market_spec"] = {}
        return context

    def _build_chart_context_for_trade_symbol(self, trade_symbol_context: Dict[str, Any]) -> Dict[str, Any]:
        context = trade_symbol_context if isinstance(trade_symbol_context, dict) else {}
        execution_symbol = canonicalize_execution_symbol(context.get("execution_symbol", ""))
        if not execution_symbol:
            return {}
        display_name = str(
            context.get("display_name", "")
            or context.get("trade_symbol_key", "")
            or context.get("candidate_key", "")
            or execution_symbol
        ).strip()
        get_market_chart_context = getattr(self.reader, "get_market_chart_context", None)
        if not callable(get_market_chart_context):
            return {}
        chart_mode = str(context.get("_chart_mode", "") or "").strip().lower()
        timeframe_specs = passive_chart_image_timeframe_specs() if chart_mode == "passive" else active_chart_image_timeframe_specs()
        chart_context = get_market_chart_context(execution_symbol, display_name=display_name, timeframe_specs=timeframe_specs)
        if not isinstance(chart_context, dict):
            return {}
        return chart_context

    @staticmethod
    def _select_playbook_trade_symbol_context(
        self,
        trade_symbol_context: Dict[str, Any],
        *,
        active_symbol: str,
        market_mainline_context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        context = dict(trade_symbol_context or {}) if isinstance(trade_symbol_context, dict) else {}
        if not context:
            return None
        active = canonicalize_execution_symbol(active_symbol)
        if active and not self._trade_symbol_matches_selected_symbol(context, active):
            return None
        return context

    def _normalize_selected_symbol(self, playbook: GenericPlaybook, trade_symbol_context: Dict[str, Any]) -> str:
        chosen = str(playbook.selected_symbol or "").strip().upper()
        context = dict(trade_symbol_context or {})
        matched = context if (not chosen or self._trade_symbol_matches_selected_symbol(context, chosen)) else None
        if matched is None and chosen:
            print(f"[warn] playbook selected_symbol={chosen} is unknown; local execution disabled.")
        canonical = (
            str((matched or {}).get("display_name", "") or "").strip()
            or str((matched or {}).get("trade_symbol_key", "") or (matched or {}).get("candidate_key", "") or "").strip().upper()
        )
        playbook.selected_symbol = canonical
        if not playbook.selection_reason:
            playbook.selection_reason = "LLM 未单独解释选标原因，已沿用当前主方案。"
        return canonical

    def _find_trade_symbol_by_selected_symbol(self, selected_symbol: str, trade_symbol_context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        target = str(selected_symbol or "").strip().upper()
        if not target:
            return None
        context = dict(trade_symbol_context or {})
        return context if self._trade_symbol_matches_selected_symbol(context, target) else None
