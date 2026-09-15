import re
from typing import Any, Dict, List, Optional, Tuple

from market_agent.utils import format_query_value


def normalize_candidate_key(raw: str) -> str:
    text = " ".join(str(raw or "").strip().split())
    if not text:
        return ""
    text = text.replace("-", "_").replace("/", "_").replace(":", "_")
    text = re.sub(r"\s+", "_", text)
    return re.sub(r"[^A-Z0-9_]+", "", text.upper())


def candidate_display_name(raw: str, fallback_key: str = "") -> str:
    text = " ".join(str(raw or "").strip().split())
    if not text:
        text = str(fallback_key or "").replace("_", " ").strip()
    if not text:
        return ""
    if re.fullmatch(r"[A-Z0-9:_/\-]{1,64}", text.upper()):
        return text.upper()
    words = []
    for part in text.replace("_", " ").split():
        if re.fullmatch(r"[A-Z0-9]{1,6}", part.upper()):
            words.append(part.upper())
        else:
            words.append(part.capitalize())
    return " ".join(words)


def split_execution_symbol(symbol: str) -> Tuple[str, str]:
    raw = str(symbol or "").strip()
    if ":" not in raw:
        return "", raw.upper()
    dex, asset = raw.split(":", 1)
    return dex.strip().lower(), asset.strip().upper()


def canonicalize_execution_symbol(symbol: str) -> str:
    dex, asset = split_execution_symbol(symbol)
    return f"{dex}:{asset}" if dex else asset


def parse_trade_symbol_context(raw_value: str) -> Dict[str, str]:
    raw_text = str(raw_value or "").strip()
    if not raw_text:
        return {}
    raw_items = [item.strip() for item in raw_text.split(",") if item.strip()]
    if len(raw_items) != 1:
        raise ValueError("TRADE_SYMBOL must contain exactly one symbol mapping.")
    item = raw_items[0]
    if "=" in item:
        raise ValueError("TRADE_SYMBOL must use ':' to separate display label from execution symbol.")
    label_raw = item
    explicit_symbol = ""
    if ":" in item:
        label_raw, explicit_symbol = item.split(":", 1)
    label_raw = label_raw.strip()
    explicit_symbol = canonicalize_execution_symbol(explicit_symbol.strip())
    trade_symbol_key = normalize_candidate_key(label_raw or explicit_symbol)
    if not trade_symbol_key:
        return {}
    return {
        "trade_symbol_key": trade_symbol_key,
        "display_name": candidate_display_name(label_raw or explicit_symbol or trade_symbol_key, trade_symbol_key),
        "configured_execution_symbol": explicit_symbol,
    }


def parse_symbol_universe(raw_value: str, default_items: Optional[List[str]] = None) -> List[str]:
    source = str(raw_value or "").strip()
    raw_items = source.split(",") if source else list(default_items or [])
    parsed: List[str] = []
    seen: set = set()
    for item in raw_items:
        symbol = " ".join(str(item or "").strip().split()).upper()
        if not symbol or symbol in seen:
            continue
        seen.add(symbol)
        parsed.append(symbol)
    return parsed


def base_url(network: str) -> str:
    network = (network or "").lower()
    if network == "mainnet":
        return "https://api.hyperliquid.xyz"
    if network == "testnet":
        return "https://api.hyperliquid-testnet.xyz"
    raise ValueError("HYPERLIQUID_NETWORK must be mainnet or testnet")


def build_default_query(
    symbol: str,
) -> str:
    symbol = (symbol or "").upper()
    if symbol:
        trading_prompt = (
            f"If you were an aggressive but disciplined short-term perpetual futures trader, how would you trade {symbol} right now based on the current market events and price action?"
        )
    else:
        trading_prompt = (
            "If you were an aggressive but disciplined short-term perpetual futures trader, what is the best trade right now based on the current market events and price action?"
        )
    return (
        f"{trading_prompt} "
    )


def render_query_template(raw_query: str, symbol: str, variables: Optional[Dict[str, Any]] = None) -> str:
    symbol = (symbol or "").upper()
    variables = dict(variables or {})
    raw_query = (raw_query or "").strip()
    if not raw_query:
        return build_default_query(symbol)
    replacements = {"symbol": symbol}
    for key, value in variables.items():
        replacements[str(key)] = format_query_value(value)
    rendered = raw_query
    for key, value in replacements.items():
        rendered = rendered.replace("{" + key + "}", value)
    return rendered
