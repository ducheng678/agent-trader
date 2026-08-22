# ruff: noqa: F401
import argparse
import base64
import json
import math
import os
import re
import statistics
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple
from zoneinfo import ZoneInfo

import requests
from dotenv import load_dotenv

_MODULE_DIR = Path(__file__).resolve().parent
if str(_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(_MODULE_DIR))

from market_agent.constants import (
    ACTION_VALUES,
    CONDITION_TYPES,
    DEFAULT_CHART_IMAGE_DETAIL,
    DEFAULT_CHART_IMAGE_HEIGHT_PX,
    DEFAULT_CHART_IMAGE_WIDTH_PX,
    DEFAULT_DIAGNOSTIC_INSTRUMENT_UNIVERSE,
    ENTRY_ACTION_VALUES,
    MANAGEMENT_EXPOSURE_ACTION_VALUES,
    MANAGEMENT_QUERY_OMIT_MARKET_SPEC_FIELDS,
    PM_SCENARIO_REQUERY_LOCK_REASONS,
    SEARCH_MODES,
    TARGET_POSITION_IMMEDIATE_ACTION_VALUES,
    TARGET_POSITION_MODE_VALUES,
    TARGET_POSITION_SIDE_VALUES,
    TARGET_POSITION_SOURCE_VALUES,
    TARGET_POSITION_STATE_VALUES,
)
from market_agent.charting import (
    _build_chart_debug_record,
    _build_chart_image_timeframe_specs,
    _build_chart_summary_record,
    _build_chart_tick_positions,
    _candle_timestamp_ms,
    _format_chart_price_label,
    _average_candle_range_pct,
    _range_position_pct,
    _resize_png_bytes,
    _sorted_candles,
    _window_change_pct,
    _window_high_low,
    render_candles_chart_png,
)
from market_agent.calibration import (
    _symbol_env_suffix_candidates,
    _trigger_confidence_symbol_env_suffix,
    extract_raw_confidence_value,
    get_trigger_confidence_calibration,
    normalize_confidence_value,
)
from market_agent.agent_candidates import TradeSymbolContextMixin
from market_agent.agent_execution_loop import ExecutionLoopMixin
from market_agent.agent_helper_reset import HelperResetMixin
from market_agent.agent_materialization import MaterializationMixin
from market_agent.agent_passive_events import PassiveEventsMixin
from market_agent.agent_runtime_io import AgentRuntimeIOMixin
from market_agent.agent_risk_session import RiskSessionMixin
from market_agent.agent_user_fills import UserFillsMixin
from market_agent.conditions import (
    band_with_tolerance,
    crossed_above_in_samples,
    crossed_below_in_samples,
    effective_condition_min_ratio,
    effective_condition_tolerance_bps,
    evaluate_condition,
    history_slice,
    level_ceiling,
    level_floor,
    sample_ratio,
)
from market_agent.events import (
    EventFileWatcher as _BaseEventFileWatcher,
    _iter_jsonl_lines_reverse,
    build_recent_event_context,
    current_utc_iso,
    normalize_event_record,
    parse_utc_iso,
    strip_item_id_for_llm,
)
from market_agent.exchange import HyperliquidExecutor, HyperliquidRestReader
from market_agent.market_profiles import (
    InstrumentMarketProfile,
    LocalTimeWindow,
    WEEKDAY_NAME_TO_INDEX,
)
from market_agent.llm_engine import (
    DiscretionaryLLMEngine,
    bump_reasoning_effort_one_level,
    normalize_reasoning_effort,
    validate_condition,
    validate_decision,
    validate_entry_scenario,
    validate_execute_when_all,
    validate_observe_when_all,
    validate_playbook,
)
from market_agent.models import (
    SCENARIO_RUNTIME_KEY,
    Condition,
    EntryPlan,
    EntryScenario,
    ExecuteWhenAll,
    ExitLeg,
    ManagementDecision,
    ObserveWhenAll,
    PendingEntryOrderSession,
    PositionManagementPlan,
    PositionManagementSession,
    RiskSession,
    Scenario,
    ScenarioRuntime,
    StrategyDecision,
    TargetPositionImmediateAction,
    TargetPositionPlan,
    _coerce_observe_when_all,
    _coerce_single_condition,
    condition_to_dict,
    entry_scenario_to_dict,
    observe_when_all_contains_price,
    observe_when_all_to_dict,
    scenario_to_dict,
)
from market_agent.openai_usage import (
    _classify_search_query_kind,
    _infer_search_call_topic,
    _response_attr,
    _response_to_primitive,
    _search_query_similarity,
    _tokenize_search_query,
    analyze_web_search_calls,
    count_web_search_tool_calls,
    estimate_openai_usage_cost,
    extract_response_usage,
    extract_web_search_call_details,
    get_openai_model_pricing,
    get_openai_web_search_tool_price_usd_per_1k,
    merge_usage_costs,
    merge_usage_dicts,
    normalize_image_input_context,
    sanitize_response_input_messages,
)
from market_agent.playbook import GenericPlaybook
from market_agent.presentation import (
    _status_decision_brief,
    _status_event_brief,
    _status_execution_result_brief,
    _status_format_price,
    _status_position_brief,
    build_status_summary,
    default_observation_starts_when,
    describe_entry_window,
    normalize_entry_price,
)
from market_agent.positions import normalize_spot_user_state, snapshot_has_open_position
from market_agent.runtime_views import (
    _pm_compare_price_bps,
    build_decision_execution_view,
    build_effective_target_position,
    build_empty_management_decision,
    build_empty_position_management_plan,
    build_empty_strategy_decision,
    build_entry_scenario_execution_view,
    build_management_exposure_entry_decision,
    build_playbook_execution_view,
    build_playbook_runtime_view,
    build_position_management_view,
    build_runtime_target_view,
    build_scenario_execution_view,
    build_target_position_plan_from_runtime_view,
    compare_position_management_plans,
    position_management_plan_has_content,
    synthetic_symbol_position_for_target_state,
)
from market_agent.schemas import HELPER_MARKET_NEWS_CONTEXT_SCHEMA, PLAYBOOK_SCHEMA
from market_agent.symbols import (
    base_url,
    build_default_query,
    candidate_display_name,
    canonicalize_execution_symbol,
    normalize_candidate_key,
    parse_symbol_universe,
    parse_trade_symbol_context,
    render_query_template,
    split_execution_symbol,
)
from market_agent.utils import (
    clamp_int,
    format_display_price,
    format_query_amount,
    format_query_value,
    safe_float,
)

load_dotenv()


CHART_IMAGE_TIMEFRAME_SPECS = _build_chart_image_timeframe_specs(
    os.getenv("OPENAI_ACTIVE_CHART_IMAGE_TIMEFRAMES", ""),
    ("1m", "5m", "15m"),
)
PASSIVE_CHART_IMAGE_TIMEFRAME_SPECS = _build_chart_image_timeframe_specs(
    os.getenv("OPENAI_PASSIVE_CHART_IMAGE_TIMEFRAMES", ""),
    ("1m",),
)


class EventFileWatcher(_BaseEventFileWatcher):
    def current_utc_iso(self) -> str:
        return current_utc_iso()


class UnifiedMarketAgent(UserFillsMixin, TradeSymbolContextMixin, ExecutionLoopMixin, HelperResetMixin, MaterializationMixin, AgentRuntimeIOMixin, PassiveEventsMixin, RiskSessionMixin):
    def __init__(self, user_query: str):
        self.reader = HyperliquidRestReader()
        self.trade_symbol_context = self._load_trade_symbol_context()
        self.instrument_market_profiles = self._load_instrument_market_profiles()
        self.symbol = None
        self.engine = DiscretionaryLLMEngine()
        self.engine.symbol = ""
        self.engine.audit_callback = self._audit_event
        self.engine.chart_context_builder = self._build_chart_context_for_trade_symbol
        self.executor = HyperliquidExecutor(self.reader, "")
        self.user_query_template = (user_query or "").strip()
        self.max_planned_loss_ratio = max(
            0.0,
            float(
                os.getenv(
                    "MAX_PLANNED_LOSS_RATIO",
                    os.getenv("MAX_PLANNED_LOSS_FRACTION", "0.33"),
                )
            ),
        )
        self.max_planned_loss_usd_fallback = max(
            0.0,
            float(
                os.getenv(
                    "MAX_PLANNED_LOSS_USD",
                    os.getenv("MAX_TRADE_LOSS_USD", os.getenv("PLANNED_MAX_LOSS_USD", "100")),
                )
            ),
        )
        # Legacy attribute retained for compatibility with older tests and payloads.
        self.max_planned_loss_usd = self.max_planned_loss_usd_fallback
        self.local_size_from_stop = str(os.getenv("LOCAL_SIZE_FROM_STOP", "true")).strip().lower() in {"1", "true", "yes", "on"}
        self.local_risk_tolerance_usd = max(0.0, float(os.getenv("LOCAL_RISK_TOLERANCE_USD", "1")))
        self.local_no_change_close_fraction_tolerance = min(max(0.0, float(os.getenv("LOCAL_NO_CHANGE_CLOSE_FRACTION_TOLERANCE", "0.01") or 0.01)), 1.0)
        self.risk_tp1_r_multiple = max(0.0, float(os.getenv("RISK_TP1_R_MULTIPLE", "1.0") or 1.0))
        self.risk_tp2_r_multiple = max(self.risk_tp1_r_multiple, float(os.getenv("RISK_TP2_R_MULTIPLE", "2.0") or 2.0))
        self.risk_tp1_close_fraction = min(max(0.0, float(os.getenv("RISK_TP1_CLOSE_FRACTION", "0.30") or 0.30)), 1.0)
        self.risk_tp2_close_fraction = min(max(0.0, float(os.getenv("RISK_TP2_CLOSE_FRACTION", "0.40") or 0.40)), 1.0)
        self.risk_post_tp1_stop_r_multiple = float(os.getenv("RISK_POST_TP1_STOP_R_MULTIPLE", "-0.40") or -0.40)
        self.risk_post_tp2_locked_r_multiple = float(os.getenv("RISK_POST_TP2_LOCKED_R_MULTIPLE", "1.0") or 1.0)
        self.risk_trailing_timeframe = str(os.getenv("RISK_TRAILING_TIMEFRAME", "15m") or "15m").strip().lower() or "15m"
        self.risk_trailing_atr_period = max(1, int(os.getenv("RISK_TRAILING_ATR_PERIOD", "14") or 14))
        self.risk_trailing_atr_lookback_bars = max(
            self.risk_trailing_atr_period + 10,
            int(os.getenv("RISK_TRAILING_ATR_LOOKBACK_BARS", "200") or 200),
        )
        self.risk_trailing_soft_atr_multiple = max(0.1, float(os.getenv("RISK_TRAILING_SOFT_ATR_MULTIPLE", "2.5") or 2.5))
        self.risk_trailing_hard_atr_multiple = max(
            self.risk_trailing_soft_atr_multiple,
            float(os.getenv("RISK_TRAILING_HARD_ATR_MULTIPLE", "3.5") or 3.5),
        )
        self.risk_soft_stop_enabled = os.getenv("RISK_SOFT_STOP_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.risk_soft_stop_confirm_timeframe = str(
            os.getenv("RISK_SOFT_STOP_CONFIRM_TIMEFRAME", "1m") or "1m"
        ).strip().lower() or "1m"
        self.risk_soft_stop_min_buffer_usd = max(
            0.0,
            float(
                os.getenv(
                    "RISK_SOFT_STOP_MIN_BUFFER_USD",
                    os.getenv("RISK_EXCHANGE_HARD_STOP_MIN_BUFFER_USD", "0.05"),
                )
                or 0.05
            ),
        )
        self.risk_soft_stop_atr_multiple = max(
            0.0,
            float(
                os.getenv(
                    "RISK_SOFT_STOP_ATR_MULTIPLE",
                    os.getenv("RISK_EXCHANGE_HARD_STOP_ATR_MULTIPLE", "0.10"),
                )
                or 0.10
            ),
        )
        self.risk_soft_stop_r_multiple = max(
            0.0,
            float(
                os.getenv(
                    "RISK_SOFT_STOP_R_MULTIPLE",
                    os.getenv("RISK_EXCHANGE_HARD_STOP_R_MULTIPLE", "0.05"),
                )
                or 0.05
            ),
        )
        self.risk_exchange_hard_stop_min_buffer_usd = self.risk_soft_stop_min_buffer_usd
        self.risk_exchange_hard_stop_atr_multiple = self.risk_soft_stop_atr_multiple
        self.risk_exchange_hard_stop_r_multiple = self.risk_soft_stop_r_multiple
        self.risk_cross_asset_soft_stop_enabled = os.getenv("RISK_CROSS_ASSET_SOFT_STOP_ENABLED", "false").strip().lower() in {"1", "true", "yes", "on"}
        self.risk_cross_asset_soft_stop_symbol = str(os.getenv("RISK_CROSS_ASSET_SOFT_STOP_SYMBOL", "SP500") or "SP500").strip()
        self.risk_cross_asset_soft_stop_poll_seconds = max(
            1.0,
            float(os.getenv("RISK_CROSS_ASSET_SOFT_STOP_POLL_SECONDS", "10") or 10.0),
        )
        self.risk_cross_asset_soft_stop_cache_max_age_seconds = max(
            self.risk_cross_asset_soft_stop_poll_seconds,
            float(os.getenv("RISK_CROSS_ASSET_SOFT_STOP_CACHE_MAX_AGE_SECONDS", "30") or 30.0),
        )
        self.risk_cross_asset_soft_stop_start_pct = max(
            0.0,
            float(os.getenv("RISK_CROSS_ASSET_SOFT_STOP_START_PCT", "0.04") or 0.04),
        )
        self.risk_cross_asset_soft_stop_full_pct = max(
            self.risk_cross_asset_soft_stop_start_pct + 1e-9,
            float(os.getenv("RISK_CROSS_ASSET_SOFT_STOP_FULL_PCT", "0.14") or 0.14),
        )
        self.risk_cross_asset_soft_stop_max_buffer_r = max(
            0.0,
            float(os.getenv("RISK_CROSS_ASSET_SOFT_STOP_MAX_BUFFER_R", "0.20") or 0.20),
        )
        self.risk_cross_asset_soft_stop_release_pct = max(
            0.0,
            float(os.getenv("RISK_CROSS_ASSET_SOFT_STOP_RELEASE_PCT", "0.05") or 0.05),
        )
        self._cross_asset_soft_stop_lock = threading.Lock()
        self._cross_asset_soft_stop_stop_event = threading.Event()
        self._cross_asset_soft_stop_thread = None
        self._cross_asset_soft_stop_cache: Dict[str, Any] = {}
        self._cross_asset_soft_stop_peak_persist_at: float = 0.0
        self._ensure_cross_asset_soft_stop_poller()
        self.risk_basis_chase_guard_enabled = os.getenv("RISK_BASIS_CHASE_GUARD_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.risk_basis_chase_first_entry_threshold_usd = max(
            0.0,
            float(os.getenv("RISK_BASIS_CHASE_FIRST_ENTRY_THRESHOLD_USD", "1.5") or 1.5),
        )
        self.risk_basis_chase_add_reverse_threshold_usd = max(
            0.0,
            float(os.getenv("RISK_BASIS_CHASE_ADD_REVERSE_THRESHOLD_USD", "1.0") or 1.0),
        )
        self.risk_basis_profit_lock_enabled = os.getenv("RISK_BASIS_PROFIT_LOCK_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.risk_basis_profit_lock_observe_threshold_usd = max(
            0.0,
            float(os.getenv("RISK_BASIS_PROFIT_LOCK_OBSERVE_THRESHOLD_USD", "1.5") or 1.5),
        )
        self.risk_basis_profit_lock_min_basis_usd = max(
            0.0,
            float(os.getenv("RISK_BASIS_PROFIT_LOCK_MIN_BASIS_USD", "1.0") or 1.0),
        )
        self.risk_basis_profit_lock_min_observation_seconds = max(
            1.0,
            float(os.getenv("RISK_BASIS_PROFIT_LOCK_MIN_OBSERVATION_SECONDS", "300") or 300.0),
        )
        self.risk_basis_profit_lock_slope_lookback_seconds = max(
            30.0,
            float(os.getenv("RISK_BASIS_PROFIT_LOCK_SLOPE_LOOKBACK_SECONDS", "180") or 180.0),
        )
        self.risk_basis_profit_lock_trigger_slope_usd_per_min = float(
            os.getenv("RISK_BASIS_PROFIT_LOCK_TRIGGER_SLOPE_USD_PER_MIN", "-0.01") or -0.01
        )
        self.risk_basis_profit_lock_tail_buffer_r = max(
            0.0,
            float(os.getenv("RISK_BASIS_PROFIT_LOCK_TAIL_BUFFER_R", "0.15") or 0.15),
        )
        self.risk_time_decay_tp_enabled = os.getenv("RISK_TIME_DECAY_TP_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.risk_time_decay_tp_timeframe_seconds = max(1.0, float(os.getenv("RISK_TIME_DECAY_TP_TIMEFRAME_SECONDS", "300") or 300.0))
        self.risk_time_decay_normal_tp1_bars = max(1.0, float(os.getenv("RISK_TIME_DECAY_NORMAL_TP1_BARS", "6") or 6.0))
        self.risk_time_decay_normal_tp1_mfe_r = max(0.0, float(os.getenv("RISK_TIME_DECAY_NORMAL_TP1_MFE_R", "0.60") or 0.60))
        self.risk_time_decay_normal_tp1_current_r = max(0.0, float(os.getenv("RISK_TIME_DECAY_NORMAL_TP1_CURRENT_R", "0.30") or 0.30))
        self.risk_time_decay_normal_tp2_bars = max(1.0, float(os.getenv("RISK_TIME_DECAY_NORMAL_TP2_BARS", "6") or 6.0))
        self.risk_time_decay_normal_tp2_mfe_r = max(0.0, float(os.getenv("RISK_TIME_DECAY_NORMAL_TP2_MFE_R", "1.50") or 1.50))
        self.risk_time_decay_normal_tp2_current_r = max(0.0, float(os.getenv("RISK_TIME_DECAY_NORMAL_TP2_CURRENT_R", "1.00") or 1.00))
        self.risk_time_decay_low_tp1_bars = max(1.0, float(os.getenv("RISK_TIME_DECAY_LOW_TP1_BARS", "18") or 18.0))
        self.risk_time_decay_low_tp1_mfe_r = max(0.0, float(os.getenv("RISK_TIME_DECAY_LOW_TP1_MFE_R", "0.30") or 0.30))
        self.risk_time_decay_low_tp1_current_r = max(0.0, float(os.getenv("RISK_TIME_DECAY_LOW_TP1_CURRENT_R", "0.15") or 0.15))
        self.risk_time_decay_low_tp2_bars = max(1.0, float(os.getenv("RISK_TIME_DECAY_LOW_TP2_BARS", "18") or 18.0))
        self.risk_time_decay_low_tp2_mfe_r = max(0.0, float(os.getenv("RISK_TIME_DECAY_LOW_TP2_MFE_R", "0.75") or 0.75))
        self.risk_time_decay_low_tp2_current_r = max(0.0, float(os.getenv("RISK_TIME_DECAY_LOW_TP2_CURRENT_R", "0.50") or 0.50))
        self.risk_time_decay_tp2_tail_lock_buffer_r = max(0.0, float(os.getenv("RISK_TIME_DECAY_TP2_TAIL_LOCK_BUFFER_R", "0.15") or 0.15))
        self.risk_tp1_no_follow_through_enabled = os.getenv("RISK_TP1_NO_FOLLOW_THROUGH_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.risk_tp1_no_follow_through_normal_close_fraction = min(
            1.0,
            max(0.0, float(os.getenv("RISK_TP1_NO_FOLLOW_THROUGH_NORMAL_CLOSE_FRACTION", "0.50") or 0.50)),
        )
        self.risk_tp1_no_follow_through_normal_soft_stop_r = max(
            0.0,
            float(os.getenv("RISK_TP1_NO_FOLLOW_THROUGH_NORMAL_SOFT_STOP_R", "0.40") or 0.40),
        )
        self.risk_tp2_no_continuation_enabled = os.getenv("RISK_TP2_NO_CONTINUATION_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.risk_tp2_no_continuation_normal_close_fraction = min(
            1.0,
            max(0.0, float(os.getenv("RISK_TP2_NO_CONTINUATION_NORMAL_CLOSE_FRACTION", "0.50") or 0.50)),
        )
        self.risk_tp2_no_continuation_normal_soft_stop_r = max(
            0.0,
            float(os.getenv("RISK_TP2_NO_CONTINUATION_NORMAL_SOFT_STOP_R", "0.25") or 0.25),
        )
        self.loop_sleep_seconds = float(os.getenv("MAIN_LOOP_SLEEP_SECONDS", "1"))
        self.playbook_poll_seconds = float(os.getenv("PLAYBOOK_POLL_SECONDS", "5"))
        self.price_history_seconds = int(os.getenv("PRICE_HISTORY_SECONDS", "1800"))
        self.risk_poll_seconds = float(os.getenv("RISK_POLL_SECONDS", "2"))
        self.position_size_change_tol = float(os.getenv("POSITION_SIZE_CHANGE_TOL", "0.00000001"))
        self.risk_session_state_enabled = os.getenv("RISK_SESSION_STATE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
        self.risk_session_state_path = Path(os.getenv("RISK_SESSION_STATE_PATH", "runtime/risk_session_state.json"))
        self.risk_session_restore_fill_lookback_seconds = max(
            1.0,
            float(os.getenv("RISK_SESSION_RESTORE_FILL_LOOKBACK_SECONDS", "21600") or 21600.0),
        )
        self.enable_monitor = os.getenv("ENABLE_PLAYBOOK_MONITOR", "true").lower() == "true"
        self.enable_active_query = os.getenv("ENABLE_ACTIVE_QUERY", "true").lower() == "true"
        self.enable_active_playbook = os.getenv("ENABLE_ACTIVE_PLAYBOOK", "true").lower() == "true"
        self.enable_active_auto_requery = os.getenv("ENABLE_ACTIVE_AUTO_REQUERY", "true").lower() == "true"
        self.atr_ref_cache: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        self.active_query_interval_seconds = float(
            os.getenv("ACTIVE_QUERY_INTERVAL_SECONDS", os.getenv("LLM_QUERY_INTERVAL_SECONDS", "180"))
        )
        self.active_management_query_interval_seconds = float(
            os.getenv("ACTIVE_MANAGEMENT_QUERY_INTERVAL_SECONDS", str(max(self.active_query_interval_seconds * 2.0, self.active_query_interval_seconds)))
        )
        self.enable_passive_event_query = os.getenv("ENABLE_PASSIVE_EVENT_QUERY", "false").lower() == "true"
        self.passive_event_relevance_filter = os.getenv("PASSIVE_EVENT_RELEVANCE_FILTER", "true").lower() == "true"
        self.passive_max_published_age_on_seen_hours = max(
            0.0,
            float(os.getenv("PASSIVE_MAX_PUBLISHED_AGE_ON_SEEN_HOURS", "0.05") or 0.05),
        )
        self.fast_replan_delay_seconds = float(os.getenv("FAST_REPLAN_DELAY_SECONDS", "2"))
        self.loop_exception_sleep_seconds = max(0.5, float(os.getenv("MAIN_LOOP_EXCEPTION_SLEEP_SECONDS", "5")))
        self.hyperliquid_transient_error_sleep_seconds = max(
            self.loop_exception_sleep_seconds,
            float(os.getenv("HYPERLIQUID_TRANSIENT_ERROR_SLEEP_SECONDS", "30") or 30.0),
        )
        self.requery_on_playbook_end = os.getenv("REQUERY_ON_PLAYBOOK_END", "true").lower() == "true"
        self.events_path = Path(os.getenv("EVENTS_JSONL_PATH", os.getenv("WATCH_EVENTS_PATH", "data/free_sources_watch/events.jsonl")))
        self.start_from = os.getenv("START_FROM", "end").lower()
        self.event_recent_window_hours = max(0.0, float(os.getenv("RECENT_EVENT_WINDOW_HOURS", "72")))
        self.event_context_max_items = int(os.getenv("RECENT_EVENT_MAX_ITEMS", "20"))
        self.passive_recent_materially_new_event_limit = int(os.getenv("PASSIVE_RECENT_MATERIALLY_NEW_EVENT_LIMIT", "0"))
        self.passive_llm_recent_events: List[Dict[str, Any]] = []
        self.passive_llm_recent_events_symbol: str = ""
        self.passive_llm_recent_events_source: str = ""
        self.passive_llm_recent_events_state_path = Path(
            os.getenv(
                "PASSIVE_LLM_RECENT_EVENTS_STATE_PATH",
                "data/free_sources_watch/passive_llm_recent_events_state.json",
            )
        )
        self.passive_llm_recent_events_state_path.parent.mkdir(parents=True, exist_ok=True)
        self.passive_llm_relevant_event_buffer_size = max(
            self.passive_recent_materially_new_event_limit,
            int(
                os.getenv(
                    "PASSIVE_LLM_RELEVANT_EVENT_BUFFER_SIZE",
                    str(max(self.passive_recent_materially_new_event_limit * 10, 200)),
                )
                or max(self.passive_recent_materially_new_event_limit * 10, 200)
            ),
        )
        self.event_buffer_size = int(
            os.getenv(
                "RECENT_EVENT_BUFFER_SIZE",
                str(max(self.event_context_max_items * 10, 200)),
            )
        )
        self.events = EventFileWatcher(
            self.events_path,
            self.start_from,
            self.event_buffer_size,
            recent_window_hours=self.event_recent_window_hours,
            max_context_items=self.event_context_max_items,
        )
        self.engine.event_recent_window_hours = self.event_recent_window_hours
        self.enable_audit_log = os.getenv("ENABLE_PLAYBOOK_AUDIT_LOG", "true").lower() == "true"
        self.audit_log_path = Path(os.getenv("PLAYBOOK_AUDIT_LOG_PATH", "logs/unified_market_agent_audit.jsonl"))
        if self.enable_audit_log:
            self.audit_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.enable_status_log = os.getenv("ENABLE_PLAYBOOK_STATUS_LOG", "true").lower() == "true"
        self.console_verbose_json = os.getenv("CONSOLE_VERBOSE_JSON", "false").lower() == "true"
        self.status_log_path = Path(os.getenv("PLAYBOOK_STATUS_LOG_PATH", "logs/unified_market_agent_status.jsonl"))
        if self.enable_status_log:
            self.status_log_path.parent.mkdir(parents=True, exist_ok=True)
        self.passive_relevant_events_log_base_path = Path(
            os.getenv(
                "PASSIVE_RELEVANT_EVENTS_LOG_PATH",
                "data/free_sources_watch/passive_relevant_events.jsonl",
            )
        )
        self.passive_relevant_events_log_base_path.parent.mkdir(parents=True, exist_ok=True)
        self.passive_relevant_events_log_path = self.passive_relevant_events_log_base_path

        self.current_playbook: Optional[GenericPlaybook] = None
        self.current_mode: Optional[str] = None
        self.current_playbook_reason: str = ""
        self.llm_relevant_passive_events_by_symbol: Dict[str, Deque[Dict[str, Any]]] = {}
        self._hydrate_llm_relevant_passive_event_buffer_from_log()
        self.passive_llm_recent_events_state_hydrated = self._hydrate_passive_llm_recent_events_state()
        self.position_management_session: Optional[PositionManagementSession] = None
        self.pending_entry_order_session: Optional[PendingEntryOrderSession] = None
        self.risk_session: Optional[RiskSession] = None
        self.position_basis_side: str = ""
        self.position_basis_confidence_raw: Optional[float] = None
        self.position_basis_validity: float = 0.0
        self._startup_live_tpsl_restore_attempted: bool = False
        self._market_catalog_warmup_attempted: bool = False
        self._market_catalog_warmup_succeeded: bool = False
        self.next_active_query_due_at: float = time.time()
        self.next_helper_reset_at: Optional[datetime] = self._compute_next_helper_reset_at()
        self.last_playbook_query_at: Optional[float] = None
        self.last_playbook_tick_at: float = 0.0
        self.last_position_management_tick_at: float = 0.0
        self.last_risk_tick_at: float = 0.0
        self.enable_user_fills_websocket = os.getenv("ENABLE_HYPERLIQUID_USER_FILLS_WEBSOCKET", "true").lower() == "true"
        self.user_fills_address = (os.getenv("HL_USER_FILLS_ADDRESS", "") or self.reader.account_address).strip()
        self.user_fills_subscription_id: Optional[int] = None
        self.user_fills_event_buffer: Deque[Dict[str, Any]] = deque()
        self._user_fills_seen_keys: Deque[Tuple[Any, ...]] = deque()
        self._user_fills_seen_key_set: set = set()
        self.user_fills_seen_capacity = max(100, int(os.getenv("HYPERLIQUID_USER_FILLS_SEEN_CAPACITY", "4000") or 4000))
        self.user_fills_reconcile_grace_seconds = max(0.1, float(os.getenv("HYPERLIQUID_USER_FILLS_RECONCILE_GRACE_SECONDS", "3") or 3.0))
        self.user_fills_reconnect_retry_seconds = max(
            1.0,
            float(os.getenv("HYPERLIQUID_USER_FILLS_RECONNECT_RETRY_SECONDS", "10") or 10.0),
        )
        self.user_fills_backfill_poll_seconds = max(
            0.5,
            float(os.getenv("HYPERLIQUID_USER_FILLS_BACKFILL_POLL_SECONDS", "5") or 5.0),
        )
        self.user_fills_backfill_lookback_seconds = max(
            self.user_fills_backfill_poll_seconds,
            float(os.getenv("HYPERLIQUID_USER_FILLS_BACKFILL_LOOKBACK_SECONDS", "120") or 120.0),
        )
        self.user_fills_last_subscribe_attempt_at: float = 0.0
        self.user_fills_last_message_at: float = 0.0
        self.user_fills_last_backfill_at: float = 0.0
        self.user_fills_last_fill_time_ms: int = 0
        self._ensure_user_fills_subscription()


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified active/passive LLM market agent for Hyperliquid")
    parser.add_argument("--symbol", type=str, default=None, help="Single trade symbol mapping, e.g. BRENTOIL-USDC:xyz:BRENTOIL")
    parser.add_argument("--symbols", type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--query", type=str, default=(os.getenv("MANUAL_STRATEGY_QUERY", "") or "").strip())
    parser.add_argument("--once", action="store_true", help="Run a single query only")
    args = parser.parse_args()

    configured_trade_symbol = str(os.getenv("TRADE_SYMBOL", "") or "").strip()
    legacy_trade_symbols = str(os.getenv("TRADE_SYMBOLS", "") or "").strip()
    if args.symbol is not None and str(args.symbol or "").strip():
        configured_trade_symbol = str(args.symbol or "").strip()
    elif args.symbols is not None and str(args.symbols or "").strip():
        configured_trade_symbol = str(args.symbols or "").strip()
    elif not configured_trade_symbol and legacy_trade_symbols:
        configured_trade_symbol = legacy_trade_symbols
    if configured_trade_symbol:
        os.environ["TRADE_SYMBOL"] = configured_trade_symbol

    print(f"[configured_trade_symbol] {os.getenv('TRADE_SYMBOL', '')}")
    print(f"[active_search_mode] {os.getenv('OPENAI_ACTIVE_SEARCH_MODE', os.getenv('OPENAI_SEARCH_MODE', 'context_only'))}")
    print(f"[passive_search_mode] {os.getenv('OPENAI_PASSIVE_SEARCH_MODE', os.getenv('OPENAI_SEARCH_MODE', 'context_only'))}")
    agent = UnifiedMarketAgent(args.query)
    try:
        if args.once:
            agent.run_once()
        else:
            agent.run_forever()
    finally:
        agent.shutdown()


if __name__ == "__main__":
    main()
