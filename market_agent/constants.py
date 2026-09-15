ACTION_VALUES = {"long", "short", "no_trade"}
ENTRY_ACTION_VALUES = {"long", "short", "no_trade"}
MANAGEMENT_ACTION_VALUES = {"no_change", "close", "trim", "long", "short", "reverse_to_long", "reverse_to_short", "add_to_long", "add_to_short"}
MANAGEMENT_EXPOSURE_ACTION_VALUES = {"long", "short", "reverse_to_long", "reverse_to_short", "add_to_long", "add_to_short"}
TARGET_POSITION_STATE_VALUES = {"open", "flat", "unknown"}
TARGET_POSITION_SOURCE_VALUES = {"entry_plan", "position_management", "post_fill_risk_template", "none"}
TARGET_POSITION_SIDE_VALUES = {"long", "short", "flat", "current_position", "none"}
TARGET_POSITION_MODE_VALUES = {"explicit_total_notional", "retain_fraction_of_current_position", "keep_current_position", "no_immediate_trade", "flat", "none"}
TARGET_POSITION_IMMEDIATE_ACTION_VALUES = ACTION_VALUES | MANAGEMENT_ACTION_VALUES | {"none"}
SEARCH_MODES = {"off", "context_only", "always"}

DEFAULT_DIAGNOSTIC_INSTRUMENT_UNIVERSE = [
    "SPX",
    "NDX",
    "RTY",
    "VIXY",
    "DXY",
    "USDJPY",
    "USDCAD",
    "UST2Y",
    "UST10Y",
    "GLD",
    "SLV",
    "CPER",
    "JETS",
    "HYG",
    "LQD",
    "BTC",
    "ETH",
]

MANAGEMENT_QUERY_OMIT_MARKET_SPEC_FIELDS = {
    "max_leverage",
    "only_isolated",
    "margin_used",
}

DEFAULT_OPENAI_MODEL_PRICING_USD_PER_1M = {
    "gpt-5.4": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
    "gpt-5.4-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5-mini": {"input": 0.25, "cached_input": 0.025, "output": 2.00},
    "gpt-5": {"input": 1.25, "cached_input": 0.125, "output": 10.00},
}
DEFAULT_OPENAI_WEB_SEARCH_TOOL_PRICE_USD_PER_1K = 10.00

DEFAULT_CHART_IMAGE_LAYOUT_WIDTH_PX = 680
DEFAULT_CHART_IMAGE_LAYOUT_HEIGHT_PX = 320
DEFAULT_CHART_IMAGE_WIDTH_PX = 510
DEFAULT_CHART_IMAGE_HEIGHT_PX = 240
DEFAULT_CHART_IMAGE_DETAIL = "low"
DEFAULT_CHART_IMAGE_WINDOW_HOURS_BY_TIMEFRAME = {
    "1m": 2.0,
    "5m": 12.0,
    "15m": 24.0,
    "1h": 72.0,
}

CONDITION_TYPES = {
    "price_ge",
    "price_le",
    "price_between",
    "sustained_ge",
    "sustained_le",
    "sustained_between",
    "cross_above",
    "cross_below",
}
SEARCH_QUERY_NOISE_TOKENS = {
    "a",
    "an",
    "and",
    "ap",
    "com",
    "fed",
    "for",
    "latest",
    "march",
    "news",
    "on",
    "of",
    "reuters",
    "site",
    "the",
    "through",
    "to",
    "us",
    "usa",
    "utm",
    "with",
    "www",
    "2026",
    "2027",
}
PM_SCENARIO_REQUERY_LOCK_REASONS = {"management_scenario_cancelled", "management_scenario_timeout"}

EVENT_TIME_KEYS = (
    "published_at",
    "event_timestamp",
    "timestamp",
    "ts",
    "created_at",
    "updated_at",
    "event_time",
    "time",
    "seen_at",
)
