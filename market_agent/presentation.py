from typing import Any, Dict, List, Optional

from market_agent.models import Condition, _coerce_observe_when_all
from market_agent.utils import format_display_price, format_query_amount, safe_float


def normalize_entry_price(entry_price: float) -> float:
    return max(0.0, float(entry_price or 0.0))


def describe_entry_window(entry_price: float) -> dict:
    entry_price = normalize_entry_price(entry_price)
    if entry_price > 0:
        return {
            "entry_price": entry_price,
            "entry_zone_text": format_display_price(entry_price),
        }
    return {"entry_price": 0.0, "entry_zone_text": ""}


def default_observation_starts_when(observe_when_all: Any, execute_when_all: Optional[Condition] = None) -> str:
    observe = _coerce_observe_when_all(observe_when_all)
    if observe.low > 0.0 and observe.high > 0.0:
        return f"当价格进入 {format_display_price(observe.low)}-{format_display_price(observe.high)} 区间后开始观察。"
    if execute_when_all is None:
        return "立即开始观察。"
    return "当 execute_when_all.condition 满足后立即执行。"


def _status_format_price(value: Any) -> str:
    numeric = safe_float(value, None)
    if numeric is None or numeric <= 0:
        return ""
    return format_display_price(numeric)


def _status_condition_brief(condition: Dict[str, Any]) -> str:
    if not isinstance(condition, dict):
        return ""
    kind = str(condition.get("type", "") or "").strip()
    level = safe_float(condition.get("level"), 0.0) or 0.0
    low = safe_float(condition.get("low"), 0.0) or 0.0
    high = safe_float(condition.get("high"), 0.0) or 0.0
    timer_seconds = int(condition.get("timer_seconds", 0) or 0)
    tolerance_bps = safe_float(condition.get("tolerance_bps"), 0.0) or 0.0
    min_ratio = safe_float(condition.get("min_ratio"), 0.0) or 0.0
    if kind in {"price_between", "sustained_between"}:
        base = f"{_status_format_price(low)}-{_status_format_price(high)}"
    elif kind in {"price_ge", "sustained_ge", "cross_above"}:
        base = f">={_status_format_price(level)}"
    elif kind in {"price_le", "sustained_le", "cross_below"}:
        base = f"<={_status_format_price(level)}"
    else:
        base = _status_format_price(level) or kind
    extras: List[str] = []
    if timer_seconds > 0:
        extras.append(f"{timer_seconds}s")
    if min_ratio > 0:
        extras.append(f"ratio>={min_ratio:.2f}")
    if tolerance_bps > 0:
        tol_txt = str(int(tolerance_bps)) if float(tolerance_bps).is_integer() else format_query_amount(tolerance_bps)
        extras.append(f"tol={tol_txt}bp")
    if extras:
        return f"{base} ({', '.join(extras)})"
    return base


def _status_conditions_brief(conditions: Any) -> str:
    items = conditions if isinstance(conditions, list) else []
    briefs = [_status_condition_brief(item) for item in items if _status_condition_brief(item)]
    return " & ".join(briefs)


def _status_observe_when_all_brief(observe_when_all: Any) -> str:
    if isinstance(observe_when_all, dict):
        low = safe_float(observe_when_all.get("low"), 0.0) or 0.0
        high = safe_float(observe_when_all.get("high"), 0.0) or 0.0
        if low > 0.0 or high > 0.0:
            return f"{_status_format_price(low)}-{_status_format_price(high)}"
    observe = _coerce_observe_when_all(observe_when_all)
    if observe.low > 0.0 or observe.high > 0.0:
        return f"{_status_format_price(observe.low)}-{_status_format_price(observe.high)}"
    return _status_conditions_brief(observe_when_all)


def _status_decision_brief(decision: Dict[str, Any]) -> str:
    decision = decision if isinstance(decision, dict) else {}
    action = str(decision.get("action", "") or "").strip() or "unknown"
    entry_window = describe_entry_window(
        safe_float(decision.get("entry_price"), 0.0) or 0.0,
    )
    entry_zone = entry_window.get("entry_zone_text") or (_status_format_price(entry_window.get("entry_price")) if entry_window.get("entry_price") else "")
    notional = safe_float(decision.get("suggested_notional_usd"), 0.0) or safe_float(decision.get("new_notional_usd"), 0.0) or 0.0
    leverage = int(decision.get("leverage", decision.get("requested_leverage", 0)) or 0)
    planned_loss = safe_float(decision.get("planned_max_loss_usd"), 0.0) or 0.0
    margin_basis = safe_float(decision.get("margin_basis_usd"), 0.0) or 0.0
    confidence_raw = safe_float(decision.get("trigger_confidence_raw"), None)
    confidence = safe_float(decision.get("trigger_confidence"), None)
    stop_loss_price = safe_float(decision.get("stop_loss_price"), 0.0) or 0.0
    close_fraction = safe_float(decision.get("close_fraction"), 0.0) or 0.0
    parts = [action]
    if entry_zone:
        parts.append(f"entry {entry_zone}")
    if stop_loss_price > 0:
        parts.append(f"sl {_status_format_price(stop_loss_price)}")
    if notional > 0:
        parts.append(f"notional {format_query_amount(notional)}")
    if margin_basis > 0:
        parts.append(f"margin {format_query_amount(margin_basis)}")
    if planned_loss > 0:
        parts.append(f"risk {format_query_amount(planned_loss)}")
    if confidence_raw is not None:
        parts.append(f"trigger_confidence {format_query_amount(confidence_raw)}")
    elif confidence is not None:
        parts.append(f"trigger_confidence {format_query_amount(confidence)}")
    if leverage > 0:
        parts.append(f"{leverage}x")
    if close_fraction > 0:
        close_pct = close_fraction * 100.0
        if close_pct < 1.0:
            parts.append(f"close {close_pct:.1f}%")
        else:
            parts.append(f"close {round(close_pct)}%")
    return " | ".join(parts)


def _status_trade_symbol_price_brief(trade_symbol_context: Any) -> str:
    context = trade_symbol_context if isinstance(trade_symbol_context, dict) else {}
    label = str(context.get("display_name", "") or context.get("trade_symbol_key", "") or context.get("candidate_key", "") or "").strip()
    price = _status_format_price(context.get("current_price"))
    if label and price:
        return f"{label} {price}"
    return ""


def _status_event_brief(event: Any) -> str:
    event = event if isinstance(event, dict) else {}
    source = str(event.get("source", "") or "").strip()
    title = str(event.get("title", "") or "").strip()
    if source and title:
        return f"{source}: {title}"
    return title or source


def _status_position_brief(snapshot: Any) -> str:
    snapshot = snapshot if isinstance(snapshot, dict) else {}
    side = str(snapshot.get("side", "") or "").strip()
    size = safe_float(snapshot.get("size"), 0.0) or 0.0
    entry_price = safe_float(snapshot.get("entry_price"), 0.0) or 0.0
    mid_price = safe_float(snapshot.get("mid_price"), 0.0) or 0.0
    if not side or side == "flat" or abs(size) <= 1e-12:
        return "flat"
    parts = [f"{side} {format_query_amount(size)}"]
    if entry_price > 0:
        parts.append(f"entry {_status_format_price(entry_price)}")
    if mid_price > 0:
        parts.append(f"mid {_status_format_price(mid_price)}")
    return " | ".join(parts)


def _status_execution_result_brief(result: Any) -> str:
    result = result if isinstance(result, dict) else {}
    mode = str(result.get("mode", "") or "").strip() or "unknown"
    accepted = result.get("accepted")
    parts = [mode]
    if accepted is True:
        parts.append("accepted")
    elif accepted is False:
        parts.append("rejected")
    position_after = result.get("position_after") if isinstance(result.get("position_after"), dict) else None
    if position_after is not None:
        position_brief = _status_position_brief(position_after)
        if position_brief:
            parts.append(f"position {position_brief}")
    leverage_update = result.get("leverage_update") if isinstance(result.get("leverage_update"), dict) else {}
    applied_leverage = int(leverage_update.get("applied_leverage", 0) or 0)
    if applied_leverage > 0:
        parts.append(f"applied {applied_leverage}x")
    message = str(result.get("message", "") or "").strip()
    if message:
        parts.append(message)
    return " | ".join(parts)


def _status_execution_result_from_payload(payload: Any) -> str:
    payload = payload if isinstance(payload, dict) else {}
    result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
    if "accepted" not in result and "accepted" in payload:
        result = dict(result)
        result["accepted"] = payload.get("accepted")
    return _status_execution_result_brief(result)


def _status_risk_candidate_brief(candidate: Any) -> str:
    candidate = candidate if isinstance(candidate, dict) else {}
    parts: List[str] = []

    action = str(candidate.get("action", "") or "").strip()
    if action:
        parts.append(action)

    leg_name = str(candidate.get("leg_name", "") or "").strip()
    leg_key = str(candidate.get("leg_key", "") or "").strip()
    if leg_name:
        parts.append(leg_name)
    elif leg_key:
        parts.append(leg_key)

    liquidity_band = str(candidate.get("liquidity_band", "") or "").strip()
    if liquidity_band:
        parts.append(liquidity_band)

    price = (
        safe_float(candidate.get("price"), None)
        or safe_float(candidate.get("current_price"), None)
        or safe_float(candidate.get("candle_close_price"), None)
    )
    price_txt = _status_format_price(price)
    if price_txt:
        parts.append(f"price {price_txt}")

    close_size = safe_float(candidate.get("close_size"), 0.0) or 0.0
    if close_size > 0:
        parts.append(f"close_size {format_query_amount(close_size)}")

    close_fraction = (
        safe_float(candidate.get("close_fraction"), None)
        if "close_fraction" in candidate
        else safe_float(candidate.get("close_fraction_of_remaining"), None)
    )
    if close_fraction is not None and close_fraction > 0:
        parts.append(f"close {format_query_amount(close_fraction * 100.0)}%")

    elapsed_label = "elapsed"
    elapsed_seconds = safe_float(candidate.get("elapsed_seconds"), None)
    if elapsed_seconds is None:
        elapsed_seconds = safe_float(candidate.get("elapsed_since_tp1_seconds"), None)
        elapsed_label = "elapsed_since_tp1"
    required_seconds = safe_float(candidate.get("required_seconds"), None)
    if elapsed_seconds is not None or required_seconds is not None:
        elapsed_value = format_query_amount(float(elapsed_seconds or 0.0))
        required_value = format_query_amount(float(required_seconds or 0.0))
        parts.append(f"{elapsed_label} {elapsed_value}s/{required_value}s")
    anchor_source = str(
        candidate.get("time_decay_anchor_source")
        or candidate.get("continuation_anchor_source")
        or ""
    ).strip()
    if anchor_source:
        parts.append(f"anchor {anchor_source}")

    mfe_r = safe_float(candidate.get("max_favorable_excursion_r"), None)
    required_mfe_r = safe_float(candidate.get("required_mfe_r"), None)
    if mfe_r is not None:
        if required_mfe_r is not None:
            parts.append(f"MFE {format_query_amount(mfe_r)}R/{format_query_amount(required_mfe_r)}R")
        else:
            parts.append(f"MFE {format_query_amount(mfe_r)}R")

    current_r = safe_float(candidate.get("current_profit_r"), None)
    required_current_r = safe_float(candidate.get("required_current_r"), None)
    if current_r is not None:
        if required_current_r is not None:
            parts.append(f"current {format_query_amount(current_r)}R/{format_query_amount(required_current_r)}R")
        else:
            parts.append(f"current {format_query_amount(current_r)}R")

    soft_stop = safe_float(candidate.get("soft_stop_price"), 0.0) or 0.0
    if soft_stop > 0:
        parts.append(f"soft_sl {_status_format_price(soft_stop)}")

    hard_stop = safe_float(candidate.get("hard_stop_price"), 0.0) or 0.0
    if hard_stop > 0:
        parts.append(f"hard_sl {_status_format_price(hard_stop)}")

    confirmed_by = str(candidate.get("confirmed_by", "") or "").strip()
    if confirmed_by:
        parts.append(f"confirmed_by {confirmed_by}")

    stage = str(candidate.get("stage", "") or "").strip()
    if stage:
        parts.append(f"stage {stage}")

    return " | ".join(parts)


def _status_risk_size_brief(payload: Any) -> str:
    payload = payload if isinstance(payload, dict) else {}
    closed_size = safe_float(payload.get("closed_size_abs"), 0.0) or 0.0
    remaining_size = safe_float(payload.get("remaining_size_abs"), 0.0) or 0.0
    parts: List[str] = []
    if closed_size > 0:
        parts.append(f"closed {format_query_amount(closed_size)}")
    if remaining_size > 0:
        parts.append(f"remaining {format_query_amount(remaining_size)}")
    elif "remaining_size_abs" in payload:
        parts.append("remaining 0")
    return " | ".join(parts)


def build_status_summary(event_type: str, payload: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
    payload = payload if isinstance(payload, dict) else {}

    def summarize_plan(plan: Dict[str, Any], *, label: str) -> str:
        action_decision = plan.get("action_decision") if isinstance(plan.get("action_decision"), dict) else {}
        primary_decision = action_decision
        if plan.get("execute_now"):
            return f"{label}: 立即 {_status_decision_brief(primary_decision)}"
        scenario = plan.get("scenario") if isinstance(plan.get("scenario"), dict) else {}
        if scenario:
            observe = _status_observe_when_all_brief(scenario.get("observe_when_all"))
            execute_when_all = scenario.get("execute_when_all") if isinstance(scenario.get("execute_when_all"), dict) else {}
            execute = _status_condition_brief(execute_when_all.get("condition"))
            action_decision = primary_decision
            parts = [label]
            if observe:
                parts.append(f"观察 {observe}")
            if execute:
                parts.append(f"执行 {execute}")
            if action_decision:
                parts.append(f"动作 {_status_decision_brief(action_decision)}")
            return " | ".join(str(part) for part in parts if part)
        return f"{label}: {_status_decision_brief(primary_decision)}"

    if event_type == "startup_ready":
        count = int(payload.get("passive_recent_events_count") or 0)
        symbol = str(payload.get("passive_recent_events_symbol", "") or "").strip()
        source = str(payload.get("passive_recent_events_source", "") or "").strip()
        hydrated = bool(payload.get("passive_recent_events_state_hydrated"))
        next_reset = str(payload.get("next_helper_reset_at", "") or "").strip()
        summary = f"启动完成 | passive tape {count}条"
        if symbol:
            summary += f" {symbol}"
        if source:
            summary += f" source={source}"
        summary += f" | state_hydrated={hydrated}"
        if next_reset:
            summary += f" | next_helper_reset={next_reset}"
        return {"kind": "startup", "summary": summary}

    if event_type == "playbook_query_requested":
        reason = str(payload.get("reason", "") or "").strip()
        if reason == "passive_event_trigger":
            return None
        summary = f"发起新query reason={reason or 'unknown'}"
        trade_symbol_price = _status_trade_symbol_price_brief(payload.get("trade_symbol_context"))
        if trade_symbol_price:
            summary += f" | trade_symbol {trade_symbol_price}"
        else:
            price_txt = _status_format_price(payload.get("current_price"))
            if price_txt:
                summary += f" price={price_txt}"
        return {"kind": "query", "summary": summary}

    if event_type == "playbook_selected":
        execution_view = payload.get("execution_view") if isinstance(payload.get("execution_view"), dict) else {}
        selected_symbol = str(execution_view.get("selected_symbol") or (payload.get("playbook") or {}).get("selected_symbol", "") or "").strip()
        if payload.get("had_open_position"):
            management = execution_view.get("position_management") if isinstance(execution_view.get("position_management"), dict) else {}
            summary = summarize_plan(management, label="新管理策略")
        else:
            runtime_view = execution_view.get("runtime_view") if isinstance(execution_view.get("runtime_view"), dict) else {}
            ignored_sources = list(runtime_view.get("ignored_immediate_sources") or [])
            ignored_entry = next(
                (
                    item for item in ignored_sources
                    if isinstance(item, dict) and str(item.get("source", "") or "").strip() == "entry_plan"
                ),
                None,
            )
            if ignored_entry is not None:
                entry_decision = ignored_entry.get("decision_view") if isinstance(ignored_entry.get("decision_view"), dict) else {}
                materialized_management = execution_view.get("position_management") if isinstance(execution_view.get("position_management"), dict) else {}
                materialized_decision = materialized_management.get("action_decision") if isinstance(materialized_management.get("action_decision"), dict) else {}
                entry_action = str(entry_decision.get("action", "") or "none").strip() or "none"
                summary = f"新策略: entry_plan.{entry_action} -> materialized {_status_decision_brief(materialized_decision)}"
            else:
                entry_plan = execution_view.get("entry_plan") if isinstance(execution_view.get("entry_plan"), dict) else {}
                summary = summarize_plan(entry_plan, label="新策略")
        if selected_symbol:
            summary = f"选中 {selected_symbol} | {summary}"
        return {"kind": "playbook", "summary": summary}

    if event_type == "position_management_forced_immediate_close":
        forced_reason = str(payload.get("forced_reason", "") or "trigger_close").strip()
        decision = payload.get("forced_now_action") if isinstance(payload.get("forced_now_action"), dict) else {}
        summary = f"管理策略改为立即平仓: {forced_reason} | {_status_decision_brief(decision)}"
        return {"kind": "management", "summary": summary}

    if event_type in {"position_management_session_created", "position_management_session_refreshed", "position_management_session_retained", "position_management_session_replaced"}:
        management = payload.get("position_management") if isinstance(payload.get("position_management"), dict) else {}
        if event_type == "position_management_session_retained":
            compare_result = payload.get("compare_result") if isinstance(payload.get("compare_result"), dict) else {}
            reasons = list(compare_result.get("soft_reasons") or [])
            reason_txt = ", ".join(reasons[:3]) if reasons else "差异较小"
            summary = summarize_plan(management, label=f"保留当前管理策略({reason_txt})")
        elif event_type == "position_management_session_replaced":
            compare_result = payload.get("compare_result") if isinstance(payload.get("compare_result"), dict) else {}
            reasons = list(compare_result.get("hard_reasons") or []) + list(compare_result.get("soft_reasons") or [])
            reason_txt = ", ".join(reasons[:3]) if reasons else "差异较大"
            summary = summarize_plan(management, label=f"替换管理策略({reason_txt})")
        else:
            summary = summarize_plan(management, label="刷新管理策略" if event_type == "position_management_session_refreshed" else "新管理策略")
        return {"kind": "management", "summary": summary}

    if event_type == "llm_call_debug":
        usage_cost = payload.get("usage_cost") if isinstance(payload.get("usage_cost"), dict) else {}
        cost_rollup = payload.get("cost_rollup") if isinstance(payload.get("cost_rollup"), dict) else {}
        if usage_cost.get("known"):
            total_cost_usd = float(usage_cost.get("estimated_total_cost_usd", usage_cost.get("total_cost_usd", 0.0)) or 0.0)
            model = str(usage_cost.get("model", "") or payload.get("response_model", "") or "").strip()
            parts = [f"本次LLM估算成本≈${total_cost_usd:.4f}"]
            if model:
                parts.append(model)
            image_inputs = usage_cost.get("image_inputs") if isinstance(usage_cost.get("image_inputs"), dict) else {}
            if int(image_inputs.get("count", 0) or 0) > 0:
                dims = sorted(
                    {
                        f"{int(item.get('width_px', 0) or 0)}x{int(item.get('height_px', 0) or 0)}"
                        for item in (image_inputs.get("rendered_images") or [])
                        if int(item.get("width_px", 0) or 0) > 0 and int(item.get("height_px", 0) or 0) > 0
                    }
                )
                image_summary = f"含{int(image_inputs.get('count', 0) or 0)}张图"
                if dims:
                    image_summary += f"({','.join(dims)})"
                parts.append(image_summary)
            last_24h = cost_rollup.get("last_24h") if isinstance(cost_rollup.get("last_24h"), dict) else {}
            last_30d = cost_rollup.get("last_30d") if isinstance(cost_rollup.get("last_30d"), dict) else {}
            if last_24h:
                parts.append(
                    f"24h≈${float(last_24h.get('total_cost_usd', last_24h.get('estimated_total_cost_usd', 0.0)) or 0.0):.4f}/{int(last_24h.get('known_cost_queries', 0) or 0)}次"
                )
            if last_30d:
                parts.append(
                    f"30d≈${float(last_30d.get('total_cost_usd', last_30d.get('estimated_total_cost_usd', 0.0)) or 0.0):.4f}/{int(last_30d.get('known_cost_queries', 0) or 0)}次"
                )
            return {"kind": "cost", "summary": " | ".join(parts)}
        return None

    if event_type == "playbook_nontradable_selection":
        selected_symbol = str(payload.get("selected_symbol", "") or "").strip()
        summary = f"选中 {selected_symbol or 'unknown'}，但当前不可在 Hyperliquid 执行，已拦截自动执行"
        return {"kind": "playbook", "summary": summary}

    if event_type in {"entry_scenario_observing", "management_scenario_observing"}:
        scenario = payload.get("scenario") if isinstance(payload.get("scenario"), dict) else {}
        observe = _status_observe_when_all_brief(scenario.get("observe_when_all"))
        price_txt = _status_format_price(payload.get("price"))
        summary = "进入观察"
        if price_txt:
            summary += f" price={price_txt}"
        if observe:
            summary += f" | 观察条件 {observe}"
        return {"kind": "observe", "summary": summary}

    if event_type in {"entry_scenario_armed", "management_scenario_armed"}:
        scenario = payload.get("scenario") if isinstance(payload.get("scenario"), dict) else {}
        execute_when_all = scenario.get("execute_when_all") if isinstance(scenario.get("execute_when_all"), dict) else {}
        execute = _status_condition_brief(execute_when_all.get("condition"))
        price_txt = _status_format_price(payload.get("price"))
        summary = "执行条件成立"
        if price_txt:
            summary += f" price={price_txt}"
        if execute:
            summary += f" | 执行条件 {execute}"
        return {"kind": "armed", "summary": summary}

    if event_type in {"entry_scenario_execute", "management_scenario_execute"}:
        price_txt = _status_format_price(payload.get("price"))
        summary = "触发执行"
        if price_txt:
            summary += f" price={price_txt}"
        return {"kind": "execute", "summary": summary}

    if event_type in {"entry_execution_result", "management_execution_result"}:
        result = payload.get("result") if isinstance(payload.get("result"), dict) else {}
        decision_view = payload.get("decision_view") if isinstance(payload.get("decision_view"), dict) else {}
        mode = str(result.get("mode", "unknown"))
        accepted = bool(result.get("accepted", False))
        position_after = result.get("position_after") if isinstance(result.get("position_after"), dict) else {}
        after_side = str(position_after.get("side", "flat") or "flat")
        after_size = abs(safe_float(position_after.get("size"), 0.0) or 0.0)
        summary = f"{'实盘' if mode == 'live' else '模拟'}执行{'已提交' if accepted else '被拒绝'}: {_status_decision_brief(decision_view)}"
        if after_side != "flat" and after_size > 0:
            summary += f" | 仓位 {after_side} {format_query_amount(after_size)}"
        elif after_side == "flat":
            summary += " | 仓位 flat"
        return {"kind": "execution", "summary": summary}

    if event_type == "risk_session_created":
        position_after = payload.get("position_after") if isinstance(payload.get("position_after"), dict) else {}
        side = str(position_after.get("side", "flat") or "flat")
        size = abs(safe_float(position_after.get("size"), 0.0) or 0.0)
        if payload.get("post_fill_risk_template") is not None:
            summary = f"持仓风控切换: {payload.get('plan_name', 'unknown')}"
        else:
            summary = f"持仓管理启动: {payload.get('plan_name', 'unknown')}"
        if side != "flat" and size > 0:
            summary += f" | {side} {format_query_amount(size)}"
        return {"kind": "risk", "summary": summary}

    if event_type == "risk_session_time_decay_take_profit":
        candidate = payload.get("candidate") if isinstance(payload.get("candidate"), dict) else {}
        summary = f"风控触发 time-decay TP: {payload.get('plan_name', 'unknown')}"
        candidate_txt = _status_risk_candidate_brief(candidate)
        if candidate_txt:
            summary += f" | {candidate_txt}"
        result_txt = _status_execution_result_from_payload(payload)
        if result_txt:
            summary += f" | {result_txt}"
        return {"kind": "risk", "summary": summary}

    if event_type == "risk_session_time_decay_take_profit_applied":
        summary = f"风控完成 time-decay TP: {payload.get('plan_name', 'unknown')}"
        size_txt = _status_risk_size_brief(payload)
        if size_txt:
            summary += f" | {size_txt}"
        candidate_txt = _status_risk_candidate_brief(payload.get("candidate"))
        if candidate_txt:
            summary += f" | {candidate_txt}"
        return {"kind": "risk", "summary": summary}

    if event_type == "basis_chase_guard_blocked":
        side = str(payload.get("target_side", "") or "").strip()
        basis = safe_float(payload.get("favorable_basis_usd"), None)
        threshold = safe_float(payload.get("threshold_usd"), None)
        summary = "执行前 basis chase guard 拦截"
        if side:
            summary += f": {side}"
        if basis is not None and threshold is not None:
            summary += f" | basis {format_query_amount(basis)} >= {format_query_amount(threshold)}"
        return {"kind": "execution", "summary": summary}

    if event_type == "risk_session_basis_profit_observation_started":
        ctx = payload.get("basis_context") if isinstance(payload.get("basis_context"), dict) else {}
        basis = safe_float(ctx.get("favorable_basis"), None)
        summary = f"风控开始 basis profit observation: {payload.get('plan_name', 'unknown')}"
        if basis is not None:
            summary += f" | favorable_basis {format_query_amount(basis)}"
        current_r = safe_float(payload.get("current_profit_r"), None)
        if current_r is not None:
            summary += f" | current {format_query_amount(current_r)}R"
        return {"kind": "risk", "summary": summary}

    if event_type == "risk_session_basis_profit_observation_cleared":
        reason = str(payload.get("reason", "") or "unknown").strip()
        summary = f"风控清除 basis profit observation: {payload.get('plan_name', 'unknown')} | {reason}"
        return {"kind": "risk", "summary": summary}

    if event_type in {"risk_session_basis_profit_lock", "risk_session_basis_profit_lock_result"}:
        label = "触发" if event_type == "risk_session_basis_profit_lock" else "执行"
        summary = f"风控{label} basis profit lock: {payload.get('plan_name', 'unknown')}"
        candidate_txt = _status_risk_candidate_brief(payload.get("candidate"))
        if candidate_txt:
            summary += f" | {candidate_txt}"
        if event_type.endswith("_result"):
            result_txt = _status_execution_result_from_payload(payload)
            if result_txt:
                summary += f" | {result_txt}"
        return {"kind": "risk", "summary": summary}

    if event_type == "risk_session_basis_profit_lock_applied":
        summary = f"风控完成 basis profit lock: {payload.get('plan_name', 'unknown')}"
        action = str(payload.get("action", "") or "").strip()
        if action:
            summary += f" | {action}"
        size_txt = _status_risk_size_brief(payload)
        if size_txt:
            summary += f" | {size_txt}"
        new_stop = safe_float(payload.get("new_trailing_soft_stop_price"), 0.0) or 0.0
        if new_stop > 0:
            summary += f" | trailing_soft_sl {_status_format_price(new_stop)}"
        candidate_txt = _status_risk_candidate_brief(payload.get("candidate"))
        if candidate_txt:
            summary += f" | {candidate_txt}"
        return {"kind": "risk", "summary": summary}

    if event_type in {"risk_session_tp1_no_follow_through", "risk_session_tp2_no_continuation"}:
        label = "tp1_no_follow_through" if event_type == "risk_session_tp1_no_follow_through" else "tp2_no_continuation"
        summary = f"风控触发 {label}: {payload.get('plan_name', 'unknown')}"
        candidate_txt = _status_risk_candidate_brief(payload.get("candidate"))
        if candidate_txt:
            summary += f" | {candidate_txt}"
        position_txt = _status_position_brief(payload.get("position_before"))
        if position_txt:
            summary += f" | position {position_txt}"
        return {"kind": "risk", "summary": summary}

    if event_type in {"risk_session_tp1_no_follow_through_result", "risk_session_tp2_no_continuation_result"}:
        label = "tp1_no_follow_through" if event_type == "risk_session_tp1_no_follow_through_result" else "tp2_no_continuation"
        summary = f"风控执行 {label}: {payload.get('plan_name', 'unknown')}"
        candidate_txt = _status_risk_candidate_brief(payload.get("candidate"))
        if candidate_txt:
            summary += f" | {candidate_txt}"
        result_txt = _status_execution_result_from_payload(payload)
        if result_txt:
            summary += f" | {result_txt}"
        return {"kind": "risk", "summary": summary}

    if event_type in {"risk_session_tp1_no_follow_through_applied", "risk_session_tp2_no_continuation_applied"}:
        label = "tp1_no_follow_through" if event_type == "risk_session_tp1_no_follow_through_applied" else "tp2_no_continuation"
        summary = f"风控完成 {label}: {payload.get('plan_name', 'unknown')}"
        action = str(payload.get("action", "") or "").strip()
        if action:
            summary += f" | {action}"
        size_txt = _status_risk_size_brief(payload)
        if size_txt:
            summary += f" | {size_txt}"
        soft_stop = safe_float(payload.get("active_soft_stop_price"), 0.0) or 0.0
        if soft_stop > 0:
            summary += f" | soft_sl {_status_format_price(soft_stop)}"
        candidate_txt = _status_risk_candidate_brief(payload.get("candidate"))
        if candidate_txt:
            summary += f" | {candidate_txt}"
        return {"kind": "risk", "summary": summary}

    if event_type == "risk_session_soft_stop_triggered":
        summary = f"风控触发 soft SL: {payload.get('plan_name', 'unknown')}"
        candidate_txt = _status_risk_candidate_brief(payload.get("candidate"))
        if candidate_txt:
            summary += f" | {candidate_txt}"
        result_txt = _status_execution_result_from_payload(payload)
        if result_txt:
            summary += f" | {result_txt}"
        return {"kind": "risk", "summary": summary}

    if event_type == "risk_session_soft_trailing_stop_triggered":
        summary = f"风控触发 trailing soft SL: {payload.get('plan_name', 'unknown')}"
        soft_stop = _status_format_price(payload.get("soft_stop_price"))
        if soft_stop:
            summary += f" | soft_sl {soft_stop}"
        last_close = _status_format_price(payload.get("last_close_price"))
        if last_close:
            summary += f" | last_close {last_close}"
        confirm_timeframe = str(payload.get("confirm_timeframe", "") or "").strip()
        if confirm_timeframe:
            summary += f" | confirmed_by {confirm_timeframe} close"
        result_txt = _status_execution_result_from_payload(payload)
        if result_txt:
            summary += f" | {result_txt}"
        return {"kind": "risk", "summary": summary}

    if event_type in {"risk_session_time_decay_take_profit_skipped", "risk_session_tp1_no_follow_through_skipped", "risk_session_tp2_no_continuation_skipped", "risk_session_basis_profit_lock_skipped"}:
        reason = str(payload.get("reason", "") or "unknown").strip()
        summary = f"风控触发已跳过: {payload.get('plan_name', 'unknown')} | {reason}"
        candidate_txt = _status_risk_candidate_brief(payload.get("candidate"))
        if candidate_txt:
            summary += f" | {candidate_txt}"
        return {"kind": "risk", "summary": summary}

    if event_type == "management_exit_leg_hit":
        leg = payload.get("leg") if isinstance(payload.get("leg"), dict) else {}
        leg_names = payload.get("leg_names") if isinstance(payload.get("leg_names"), list) else []
        leg_name = str(leg.get("name", "") or "").strip()
        if not leg_name and leg_names:
            leg_name = str(leg_names[0] or "").strip()
        price_txt = _status_format_price(payload.get("price"))
        summary = f"{payload.get('leg_type', 'unknown')} 触发: {payload.get('plan_name', 'unknown')}::{leg_name or 'unknown'}"
        if price_txt:
            summary += f" price={price_txt}"
        return {"kind": "exit_leg", "summary": summary}

    if event_type in {"entry_scenario_cancelled", "entry_scenario_timeout", "management_scenario_cancelled", "management_scenario_timeout"}:
        price_txt = _status_format_price(payload.get("price"))
        if event_type.startswith("entry_"):
            label = "入场场景取消" if event_type.endswith("cancelled") else "入场场景超时"
        else:
            label = "管理场景取消" if event_type.endswith("cancelled") else "管理场景超时"
        summary = label
        if price_txt:
            summary += f" price={price_txt}"
        return {"kind": "management", "summary": summary}

    if event_type == "runtime_error":
        stage = str(payload.get("stage", "") or "runtime")
        exc_type = str(payload.get("error_type", "") or "Exception")
        message = str(payload.get("message", "") or "")
        retry_after_seconds = safe_float(payload.get("retry_after_seconds"), 0.0) or 0.0
        summary = f"运行时异常[{stage}] {exc_type}"
        if message:
            summary += f": {message}"
        if retry_after_seconds > 0:
            summary += f" | {format_query_amount(retry_after_seconds)}s 后重试"
        return {"kind": "runtime_error", "summary": summary}

    return None
