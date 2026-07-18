from __future__ import annotations

from .constants import (
    CONDITION_TYPES,
    ENTRY_ACTION_VALUES,
    TARGET_POSITION_IMMEDIATE_ACTION_VALUES,
)


PLAYBOOK_SCHEMA = {
    "type": "json_schema",
    "name": "llm_generated_discretionary_playbook",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "trigger_event_relevance": {"type": "string", "enum": ["not_applicable", "relevant", "unrelated", "duplicate"]},
            "trigger_confidence": {
                "anyOf": [
                    {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    {"type": "null"},
                ]
            },
            "playbook": {
                "anyOf": [
                    {"type": "null"},
                    {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "entry_plan": {"$ref": "#/$defs/entry_plan"},
                        },
                        "required": [
                            "entry_plan",
                        ],
                    },
                ],
            },
        },
        "required": ["trigger_event_relevance", "trigger_confidence", "playbook"],
        "$defs": {
            "entry_decision": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "action": {"type": "string", "enum": sorted(ENTRY_ACTION_VALUES)},
                    "entry_price": {"type": "number", "minimum": 0.0, "maximum": 1000000000.0},
                    "stop_loss_price": {"type": "number", "minimum": 0.0, "maximum": 1000000000.0},
                },
                "required": [
                    "action",
                    "entry_price",
                    "stop_loss_price",
                ],
            },
            "target_position_immediate_action": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "action": {"type": "string", "enum": sorted(TARGET_POSITION_IMMEDIATE_ACTION_VALUES)},
                    "target_side": {"type": "string", "enum": ["long", "short", "flat", "current_position", "none"]},
                    "target_notional_usd": {"type": "number", "minimum": 0.0, "maximum": 50000.0},
                    "target_notional_mode": {"type": "string", "enum": ["explicit_total_notional", "retain_fraction_of_current_position", "keep_current_position", "no_immediate_trade", "flat", "none"]},
                    "retain_fraction_of_current_position": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                },
                "required": [
                    "action",
                    "target_side",
                    "target_notional_usd",
                    "target_notional_mode",
                    "retain_fraction_of_current_position"
                ]
            },
            "target_position": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "position_state": {"type": "string", "enum": ["open", "flat"]},
                    "immediate_action_source": {"type": "string", "enum": ["entry_plan", "position_management", "none"]},
                    "immediate_action": {"$ref": "#/$defs/target_position_immediate_action"},
                    "observation_source": {"type": "string", "enum": ["entry_plan", "position_management", "none"]},
                    "observation_plan_names": {"type": "array", "items": {"type": "string"}},
                    "active_management_source": {"type": "string", "enum": ["position_management", "none"]},
                    "active_management_summary": {"type": "string"},
                    "successor_management_source": {"type": "string", "enum": ["position_management", "post_fill_risk_template", "none"]},
                    "successor_management_summary": {"type": "string"}
                },
                "required": [
                    "position_state",
                    "immediate_action_source",
                    "immediate_action",
                    "observation_source",
                    "observation_plan_names",
                    "active_management_source",
                    "active_management_summary",
                    "successor_management_source",
                    "successor_management_summary"
                ]
            },
            "condition": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "enum": sorted(CONDITION_TYPES)},
                    "level": {"type": "number"},
                    "low": {"type": "number"},
                    "high": {"type": "number"},
                    "timer_seconds": {"type": "integer", "minimum": 0, "maximum": 86400},
                    "tolerance_bps": {"type": "number", "minimum": 0.0, "maximum": 1000.0},
                    "min_ratio": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                },
                "required": ["type", "level", "low", "high", "timer_seconds", "tolerance_bps", "min_ratio"],
            },
            "execute_when_all": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "condition": {
                        "anyOf": [
                            {"$ref": "#/$defs/condition"},
                            {"type": "null"},
                        ]
                    },
                    "timeout_seconds": {"type": "integer", "minimum": 1},
                },
                "required": ["condition", "timeout_seconds"],
            },
            "observe_when_all": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "low": {"type": "number"},
                    "high": {"type": "number"},
                },
                "required": ["low", "high"],
            },
            "entry_scenario": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "observe_when_all": {"$ref": "#/$defs/observe_when_all"},
                    "execute_when_all": {"$ref": "#/$defs/execute_when_all"},
                },
                "required": [
                    "observe_when_all",
                    "execute_when_all",
                ],
            },
            "entry_plan": {
                "type": "object",
                "additionalProperties": False,
                "properties": {

                    "execute_now": {"type": "boolean"},
                    "action_decision": {"$ref": "#/$defs/entry_decision"},
                    "scenario": {
                        "anyOf": [
                            {"$ref": "#/$defs/entry_scenario"},
                            {"type": "null"},
                        ]
                    },
                },
                "required": ["execute_now", "action_decision", "scenario"],
            },
        },
    },
}

PASSIVE_EVENT_JUDGE_SCHEMA = {
    "type": "json_schema",
    "name": "llm_passive_event_judge",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "trigger_event_relevance": {"type": "string", "enum": ["relevant", "unrelated", "duplicate"]},
            "trigger_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "action": {"type": "string", "enum": sorted(ENTRY_ACTION_VALUES)},
        },
        "required": ["trigger_event_relevance", "trigger_confidence", "action"],
    },
}

PASSIVE_TECHNICAL_PRICING_SCHEMA = {
    "type": "json_schema",
    "name": "llm_passive_technical_pricing",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "entry_price": {"type": "number", "minimum": 0.0, "maximum": 1000000000.0},
            "stop_loss_price": {"type": "number", "minimum": 0.0, "maximum": 1000000000.0},
        },
        "required": ["entry_price", "stop_loss_price"],
    },
}

HELPER_MARKET_NEWS_CONTEXT_SCHEMA = {
    "type": "json_schema",
    "name": "llm_helper_market_news_context",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "market_mainline_context": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "current_move_logic_mainline": {"type": "string"},
                    "diagnostic_instruments": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["current_move_logic_mainline", "diagnostic_instruments"],
            },
            "materially_new_first_events": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "event_timestamp": {"type": "string"},
                        "source": {"type": "string"},
                        "title": {"type": "string"},
                    },
                    "required": ["event_timestamp", "source", "title"],
                },
            },
        },
        "required": ["market_mainline_context", "materially_new_first_events"],
    },
}
