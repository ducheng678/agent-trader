import os
import re
from typing import Any, Dict, List, Optional

from market_agent.constants import (
    DEFAULT_CHART_IMAGE_DETAIL,
    DEFAULT_OPENAI_MODEL_PRICING_USD_PER_1M,
    DEFAULT_OPENAI_WEB_SEARCH_TOOL_PRICE_USD_PER_1K,
    SEARCH_QUERY_NOISE_TOKENS,
)


def get_openai_model_pricing(model: str) -> Optional[Dict[str, float]]:
    override_input = os.getenv("OPENAI_PRICE_INPUT_PER_1M_USD", "").strip()
    override_cached = os.getenv("OPENAI_PRICE_CACHED_INPUT_PER_1M_USD", "").strip()
    override_output = os.getenv("OPENAI_PRICE_OUTPUT_PER_1M_USD", "").strip()
    if override_input and override_cached and override_output:
        return {
            "input": float(override_input),
            "cached_input": float(override_cached),
            "output": float(override_output),
        }
    model_key = str(model or "").strip().lower()
    if not model_key:
        return None
    if model_key in DEFAULT_OPENAI_MODEL_PRICING_USD_PER_1M:
        return dict(DEFAULT_OPENAI_MODEL_PRICING_USD_PER_1M[model_key])
    if model_key.startswith("gpt-5.4-mini"):
        return dict(DEFAULT_OPENAI_MODEL_PRICING_USD_PER_1M["gpt-5.4-mini"])
    if model_key.startswith("gpt-5-mini") or ("mini" in model_key and model_key.startswith("gpt-5")):
        return dict(DEFAULT_OPENAI_MODEL_PRICING_USD_PER_1M["gpt-5-mini"])
    if model_key.startswith("gpt-5.4"):
        return dict(DEFAULT_OPENAI_MODEL_PRICING_USD_PER_1M["gpt-5.4"])
    return None


def get_openai_web_search_tool_price_usd_per_1k() -> float:
    return max(
        0.0,
        float(os.getenv("OPENAI_WEB_SEARCH_TOOL_PRICE_PER_1K_USD", str(DEFAULT_OPENAI_WEB_SEARCH_TOOL_PRICE_USD_PER_1K))),
    )


def _response_attr(value: Any, key: str, default: Any = None) -> Any:
    if value is None:
        return default
    if isinstance(value, dict):
        return value.get(key, default)
    return getattr(value, key, default)


def extract_response_usage(response: Any) -> Dict[str, Any]:
    usage = _response_attr(response, "usage")
    if usage is None:
        return {}
    input_details = _response_attr(usage, "input_tokens_details", {}) or {}
    output_details = _response_attr(usage, "output_tokens_details", {}) or {}
    return {
        "input_tokens": int(_response_attr(usage, "input_tokens", 0) or 0),
        "cached_input_tokens": int(_response_attr(input_details, "cached_tokens", 0) or 0),
        "output_tokens": int(_response_attr(usage, "output_tokens", 0) or 0),
        "reasoning_tokens": int(_response_attr(output_details, "reasoning_tokens", 0) or 0),
        "total_tokens": int(_response_attr(usage, "total_tokens", 0) or 0),
    }


def count_web_search_tool_calls(response: Any) -> int:
    output = _response_attr(response, "output", []) or []
    count = 0
    for item in output:
        item_type = str(_response_attr(item, "type", "") or "").strip()
        if item_type == "web_search_call":
            count += 1
    return count


def _response_to_primitive(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _response_to_primitive(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_response_to_primitive(v) for v in value]
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        try:
            return _response_to_primitive(model_dump())
        except Exception:
            pass
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        try:
            return _response_to_primitive(to_dict())
        except Exception:
            pass
    attrs = {}
    for key in dir(value):
        if key.startswith("_"):
            continue
        try:
            attr = getattr(value, key)
        except Exception:
            continue
        if callable(attr):
            continue
        attrs[key] = _response_to_primitive(attr)
    return attrs if attrs else str(value)


def extract_web_search_call_details(response: Any) -> List[Dict[str, Any]]:
    details: List[Dict[str, Any]] = []
    output = _response_attr(response, "output", []) or []
    for item in output:
        item_type = str(_response_attr(item, "type", "") or "").strip()
        if item_type != "web_search_call":
            continue
        raw = _response_to_primitive(item)
        record: Dict[str, Any] = {
            "type": item_type,
            "id": str(_response_attr(item, "id", "") or ""),
            "status": str(_response_attr(item, "status", "") or ""),
            "raw": raw,
        }
        for field in ("query", "search_query", "action", "queries"):
            value = None
            if isinstance(raw, dict):
                value = raw.get(field)
            if value not in (None, "", [], {}):
                record[field] = value
        details.append(record)
    return details


def sanitize_response_input_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sanitized: List[Dict[str, Any]] = []
    for message in messages or []:
        clean_message: Dict[str, Any] = {"role": message.get("role", ""), "content": []}
        for item in message.get("content", []) or []:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "") or "").strip()
            if item_type != "input_image":
                clean_message["content"].append(dict(item))
                continue
            image_url = str(item.get("image_url", "") or "")
            clean_message["content"].append(
                {
                    "type": "input_image",
                    "detail": str(item.get("detail", "") or DEFAULT_CHART_IMAGE_DETAIL),
                    "image_url": f"<data-url:{len(image_url)} chars>" if image_url.startswith("data:") else image_url,
                }
            )
        sanitized.append(clean_message)
    return sanitized


def normalize_image_input_context(image_input_context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    context = image_input_context if isinstance(image_input_context, dict) else {}
    raw_images = context.get("debug_images")
    if not raw_images:
        raw_images = context.get("rendered_images")
    debug_images = [dict(item) for item in (raw_images or []) if isinstance(item, dict)]
    widths = sorted({int(item.get("width_px", 0) or 0) for item in debug_images if int(item.get("width_px", 0) or 0) > 0})
    heights = sorted({int(item.get("height_px", 0) or 0) for item in debug_images if int(item.get("height_px", 0) or 0) > 0})
    return {
        "count": len(debug_images),
        "symbol": str(context.get("execution_symbol", "") or ""),
        "display_name": str(context.get("display_name", "") or ""),
        "detail": str(context.get("detail", "") or DEFAULT_CHART_IMAGE_DETAIL),
        "rendered_images": debug_images,
        "total_image_bytes": sum(int(item.get("image_bytes", 0) or 0) for item in debug_images),
        "total_data_url_chars": sum(int(item.get("data_url_chars", 0) or 0) for item in debug_images),
        "widths_px": widths,
        "heights_px": heights,
        "note": (
            "Image-bearing query cost is included inside input token usage for the final Responses API estimate. "
            "The current Responses usage schema does not expose separate image token counts."
        )
        if debug_images
        else "",
    }


def _tokenize_search_query(text: str) -> List[str]:
    cleaned = re.sub(r"site:[^\s]+", " ", str(text or "").lower())
    cleaned = re.sub(r"https?://\S+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned)
    tokens: List[str] = []
    for raw_token in cleaned.split():
        token = raw_token.strip()
        if not token or token in SEARCH_QUERY_NOISE_TOKENS or token.isdigit():
            continue
        if len(token) > 4 and token.endswith("s"):
            token = token[:-1]
        if token and token not in SEARCH_QUERY_NOISE_TOKENS:
            tokens.append(token)
    return tokens


def _trade_symbol_topic_aliases(trade_symbol_context: Dict[str, Any]) -> Dict[str, set]:
    aliases: Dict[str, set] = {}
    item = trade_symbol_context if isinstance(trade_symbol_context, dict) else {}
    label = str(item.get("display_name", "") or item.get("trade_symbol_key", "") or item.get("candidate_key", "") or item.get("execution_symbol", "") or "").strip()
    if not label:
        return aliases
    topic_aliases = aliases.setdefault(label, set())
    for source in (
        label,
        item.get("trade_symbol_key", "") or item.get("candidate_key", ""),
        item.get("execution_symbol", ""),
        ((item.get("market_spec") or {}).get("market_name", "")),
    ):
        for token in _tokenize_search_query(str(source or "")):
            topic_aliases.add(token)
    label_upper = label.upper()
    if "BTC" in label_upper:
        topic_aliases.update({"btc", "bitcoin", "crypto", "ibit", "blackrock", "morgan", "stanley", "bank", "banks", "banking", "debank", "debanking"})
    if "SILVER" in label_upper or "XAG" in label_upper:
        topic_aliases.update({"silver", "xag", "precious", "metal", "metals", "bullion", "slv", "lbma", "silverinstitute"})
    if "BRENTOIL" in label_upper or "BRENT" in label_upper:
        topic_aliases.update({"brent", "oil", "crude", "iran", "hormuz", "epa", "fuel", "rfs", "shipping"})
    return aliases


def _infer_search_call_topic(query: str, trade_symbol_context: Dict[str, Any]) -> str:
    query_tokens = set(_tokenize_search_query(query))
    if not query_tokens:
        return "other"
    best_topic = "other"
    best_score = 0
    for topic, aliases in _trade_symbol_topic_aliases(trade_symbol_context).items():
        matched = query_tokens & aliases
        score = len(matched)
        if score and any(token in matched for token in _tokenize_search_query(topic)):
            score += 2
        if score > best_score:
            best_score = score
            best_topic = topic
    return best_topic


def _search_query_similarity(lhs: str, rhs: str) -> float:
    left_tokens = set(_tokenize_search_query(lhs))
    right_tokens = set(_tokenize_search_query(rhs))
    if not left_tokens or not right_tokens:
        return 0.0
    shared = len(left_tokens & right_tokens)
    union = len(left_tokens | right_tokens)
    containment = shared / max(1, min(len(left_tokens), len(right_tokens)))
    jaccard = shared / max(1, union)
    if shared < 2:
        return 0.0
    return max(containment, jaccard)


def _classify_search_query_kind(query: str) -> str:
    text = str(query or "").strip().lower()
    if text.startswith("calculator:"):
        return "calculator"
    if text.startswith("finance:"):
        return "finance_lookup"
    return "search"


def analyze_web_search_calls(
    web_search_calls: List[Dict[str, Any]],
    trade_symbol_context: Dict[str, Any],
    *,
    max_total_calls: int,
    max_calls_per_topic: int,
) -> Dict[str, Any]:
    per_call: List[Dict[str, Any]] = []
    calls_per_topic: Dict[str, int] = {}
    seen_by_topic: Dict[str, List[Dict[str, Any]]] = {}
    duplicate_calls: List[Dict[str, Any]] = []
    for index, item in enumerate(web_search_calls, 1):
        action = item.get("action") or {}
        query = str(action.get("query") or item.get("query") or "").strip()
        query_kind = _classify_search_query_kind(query)
        topic = _infer_search_call_topic(query, trade_symbol_context)
        calls_per_topic[topic] = calls_per_topic.get(topic, 0) + 1
        similar_to: Optional[int] = None
        similarity = 0.0
        if query_kind == "search":
            for previous in seen_by_topic.get(topic, []):
                score = _search_query_similarity(query, previous["query"])
                if score >= 0.5:
                    similar_to = int(previous["index"])
                    similarity = score
                    break
        call_info = {
            "index": index,
            "topic": topic,
            "query_kind": query_kind,
            "query": query,
            "duplicate_of": similar_to,
            "similarity": round(similarity, 3) if similar_to else 0.0,
        }
        per_call.append(call_info)
        seen_by_topic.setdefault(topic, []).append({"index": index, "query": query})
        if similar_to:
            duplicate_calls.append(call_info)
    non_news_calls = [item for item in per_call if item["query_kind"] != "search"]
    topic_budget_violations = {
        topic: count
        for topic, count in calls_per_topic.items()
        if count > max(1, int(max_calls_per_topic or 1))
    }
    return {
        "max_total_calls": max(1, int(max_total_calls or 1)),
        "max_calls_per_topic": max(1, int(max_calls_per_topic or 1)),
        "actual_calls": len(web_search_calls),
        "unique_topics": sorted(topic for topic in calls_per_topic.keys() if topic),
        "calls_per_topic": calls_per_topic,
        "over_budget": len(web_search_calls) > max(1, int(max_total_calls or 1)),
        "topic_budget_violations": topic_budget_violations,
        "duplicate_call_count": len(duplicate_calls),
        "duplicate_calls": duplicate_calls,
        "non_news_call_count": len(non_news_calls),
        "non_news_calls": non_news_calls,
        "per_call": per_call,
    }


def estimate_openai_usage_cost(
    *,
    model: str,
    usage: Dict[str, Any],
    web_search_tool_calls: int = 0,
    image_input_context: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    pricing = get_openai_model_pricing(model)
    normalized_images = normalize_image_input_context(image_input_context)
    if not pricing:
        return {
            "known": False,
            "model": model,
            "image_inputs": normalized_images,
            "message": f"No pricing table is configured for model {model or '<unknown>'}.",
        }
    input_tokens = int(usage.get("input_tokens", 0) or 0)
    cached_input_tokens = min(input_tokens, int(usage.get("cached_input_tokens", 0) or 0))
    uncached_input_tokens = max(0, input_tokens - cached_input_tokens)
    output_tokens = int(usage.get("output_tokens", 0) or 0)
    input_cost_usd = (uncached_input_tokens / 1_000_000.0) * float(pricing["input"])
    cached_input_cost_usd = (cached_input_tokens / 1_000_000.0) * float(pricing["cached_input"])
    output_cost_usd = (output_tokens / 1_000_000.0) * float(pricing["output"])
    web_search_tool_cost_usd = (max(0, int(web_search_tool_calls or 0)) / 1000.0) * get_openai_web_search_tool_price_usd_per_1k()
    estimated_total_cost_usd = input_cost_usd + cached_input_cost_usd + output_cost_usd + web_search_tool_cost_usd
    return {
        "known": True,
        "model": model,
        "pricing_usd_per_1m": pricing,
        "usage": {
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "uncached_input_tokens": uncached_input_tokens,
            "output_tokens": output_tokens,
            "reasoning_tokens": int(usage.get("reasoning_tokens", 0) or 0),
            "total_tokens": int(usage.get("total_tokens", 0) or 0),
        },
        "web_search_tool_calls": max(0, int(web_search_tool_calls or 0)),
        "image_inputs": normalized_images,
        "web_search_tool_price_usd_per_1k": get_openai_web_search_tool_price_usd_per_1k(),
        "cost_breakdown_usd": {
            "input_cost_usd": input_cost_usd,
            "cached_input_cost_usd": cached_input_cost_usd,
            "output_cost_usd": output_cost_usd,
            "web_search_tool_cost_usd": web_search_tool_cost_usd,
        },
        "estimated_total_cost_usd": estimated_total_cost_usd,
        "total_cost_usd": estimated_total_cost_usd,
        "approximation_note": (
            "Search content tokens are treated as part of input token usage. "
            "This is an estimate based on OpenAI Responses usage fields and the current configured pricing table. "
            "If your exact model alias is not listed on OpenAI's public pricing page, the nearest configured family price is used unless you override it with env vars."
        ),
    }


def merge_usage_dicts(*usages: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = {
        "input_tokens": 0,
        "cached_input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "total_tokens": 0,
    }
    for usage in usages:
        if not isinstance(usage, dict):
            continue
        for key in merged:
            merged[key] += int(usage.get(key, 0) or 0)
    return merged


def merge_usage_costs(*costs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    parts: List[Dict[str, Any]] = []
    total = 0.0
    known_parts = 0
    merged_images: List[Dict[str, Any]] = []
    merged_image_bytes = 0
    merged_data_url_chars = 0
    merged_image_symbol = ""
    merged_image_display_name = ""
    merged_image_detail = ""
    for idx, cost in enumerate(costs):
        if not isinstance(cost, dict):
            continue
        part = dict(cost)
        part["component_index"] = idx
        parts.append(part)
        image_inputs = normalize_image_input_context(cost.get("image_inputs"))
        if image_inputs.get("count", 0):
            merged_images.extend(list(image_inputs.get("rendered_images") or []))
            merged_image_bytes += int(image_inputs.get("total_image_bytes", 0) or 0)
            merged_data_url_chars += int(image_inputs.get("total_data_url_chars", 0) or 0)
            merged_image_symbol = merged_image_symbol or str(image_inputs.get("symbol", "") or "")
            merged_image_display_name = merged_image_display_name or str(image_inputs.get("display_name", "") or "")
            merged_image_detail = merged_image_detail or str(image_inputs.get("detail", "") or "")
        if cost.get("known"):
            known_parts += 1
            total += float(cost.get("estimated_total_cost_usd", cost.get("total_cost_usd", 0.0)) or 0.0)
    if not parts:
        return {}
    result = {
        "known": known_parts == len(parts),
        "estimated_total_cost_usd": total if known_parts else 0.0,
        "total_cost_usd": total if known_parts else 0.0,
        "components": parts,
        "message": "" if known_parts == len(parts) else "One or more cost components could not be estimated.",
    }
    if merged_images:
        result["image_inputs"] = {
            "count": len(merged_images),
            "symbol": merged_image_symbol,
            "display_name": merged_image_display_name,
            "detail": merged_image_detail or DEFAULT_CHART_IMAGE_DETAIL,
            "rendered_images": merged_images,
            "total_image_bytes": merged_image_bytes,
            "total_data_url_chars": merged_data_url_chars,
            "note": (
                "Image-bearing query cost is included inside input token usage for the final Responses API estimate. "
                "The current Responses usage schema does not expose separate image token counts."
            ),
        }
    return result
