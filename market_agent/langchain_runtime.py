from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, Dict, List


@dataclass
class LangChainUsage:
    input_tokens: int = 0
    input_tokens_details: Dict[str, Any] = field(default_factory=dict)
    output_tokens: int = 0
    output_tokens_details: Dict[str, Any] = field(default_factory=dict)
    total_tokens: int = 0


@dataclass
class LangChainResponse:
    id: str
    model: str
    output_text: str
    usage: LangChainUsage
    output: List[Dict[str, Any]] = field(default_factory=list)


def _to_langchain_messages(messages: List[Dict[str, Any]]) -> List[Any]:
    from langchain_core.messages import HumanMessage, SystemMessage

    converted: List[Any] = []
    for message in messages or []:
        role = str(message.get("role", "") or "").strip().lower()
        content: List[Dict[str, Any]] = []
        for item in message.get("content", []) or []:
            if not isinstance(item, dict):
                continue
            item_type = str(item.get("type", "") or "").strip()
            if item_type == "input_text":
                content.append(
                    {
                        "type": "text",
                        "text": str(item.get("text", "") or ""),
                    }
                )
            elif item_type == "input_image":
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": str(item.get("image_url", "") or ""),
                            "detail": str(item.get("detail", "auto") or "auto"),
                        },
                    }
                )
        message_type = SystemMessage if role == "system" else HumanMessage
        converted.append(message_type(content=content))
    return converted


def _message_text(message: Any) -> str:
    blocks = getattr(message, "content_blocks", []) or []
    text = "".join(
        str(block.get("text", "") or "")
        for block in blocks
        if isinstance(block, dict) and block.get("type") == "text"
    )
    if text:
        return text
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    return "".join(
        str(block.get("text", "") or "")
        for block in content or []
        if isinstance(block, dict) and block.get("type") in {"text", "output_text"}
    )


def _web_search_output(message: Any) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    seen = set()
    blocks = list(getattr(message, "content_blocks", []) or [])
    content = getattr(message, "content", [])
    if isinstance(content, list):
        blocks.extend(content)
    for raw_block in blocks:
        block = raw_block
        if not isinstance(block, dict):
            continue
        if block.get("type") == "non_standard" and isinstance(block.get("value"), dict):
            block = block["value"]
        block_type = str(block.get("type", "") or "")
        if block_type == "server_tool_call" and block.get("name") == "web_search":
            action = dict(block.get("args") or {})
        elif block_type == "web_search_call":
            action = dict(block.get("action") or {})
        else:
            continue
        block_id = str(block.get("id", "") or "")
        identity = (block_id, repr(action))
        if identity in seen:
            continue
        seen.add(identity)
        output.append(
            {
                "type": "web_search_call",
                "id": block_id,
                "status": str(block.get("status", "") or ""),
                "action": action,
            }
        )
    return output


def _verified_tokens(value: Any, field: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"LangChain response has unverified {field}")
    return value


def _verified_details(metadata: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    details = metadata.get(key)
    if details is None:
        return {}
    if not isinstance(details, Mapping):
        raise ValueError(f"LangChain response has invalid {key}")
    return details


def _detail_tokens(details: Mapping[str, Any], names: tuple[str, ...], suffix: str, field: str) -> int:
    values = [details[name] for name in names if name in details]
    if not values:
        values = [value for key, value in details.items() if isinstance(key, str) and key.endswith(suffix)]
    if not values:
        return 0
    verified = [_verified_tokens(value, field) for value in values]
    if len(set(verified)) > 1:
        raise ValueError(f"LangChain response has conflicting {field}")
    return verified[0]


def _normalize_response(message: Any) -> LangChainResponse:
    # The requested model is not evidence of the provider's actual priced model.
    usage_metadata = getattr(message, "usage_metadata", None)
    response_metadata = getattr(message, "response_metadata", None)
    if not isinstance(usage_metadata, Mapping) or not isinstance(response_metadata, Mapping):
        raise ValueError("LangChain response has unverified usage or model metadata")
    input_tokens = _verified_tokens(usage_metadata.get("input_tokens"), "input tokens")
    output_tokens = _verified_tokens(usage_metadata.get("output_tokens"), "output tokens")
    total_tokens = _verified_tokens(usage_metadata.get("total_tokens"), "total tokens")
    if total_tokens != input_tokens + output_tokens:
        raise ValueError("LangChain response token totals are inconsistent")
    input_details = _verified_details(usage_metadata, "input_token_details")
    output_details = _verified_details(usage_metadata, "output_token_details")
    cached_tokens = _detail_tokens(input_details, ("cache_read", "cached_tokens"), "_cache_read", "cached tokens")
    reasoning_tokens = _detail_tokens(output_details, ("reasoning", "reasoning_tokens"), "_reasoning", "reasoning tokens")
    if cached_tokens > input_tokens or reasoning_tokens > output_tokens:
        raise ValueError("LangChain response token details exceed provider totals")
    response_id = str(
        response_metadata.get("id")
        or response_metadata.get("response_id")
        or getattr(message, "id", "")
        or ""
    )
    response_model = response_metadata.get("model_name") or response_metadata.get("model")
    if not isinstance(response_model, str) or not response_model.strip():
        raise ValueError("LangChain response has unverified provider model")
    return LangChainResponse(
        id=response_id,
        model=response_model.strip(),
        output_text=_message_text(message),
        usage=LangChainUsage(
            input_tokens=input_tokens,
            input_tokens_details={"cached_tokens": cached_tokens},
            output_tokens=output_tokens,
            output_tokens_details={"reasoning_tokens": reasoning_tokens},
            total_tokens=total_tokens,
        ),
        output=_web_search_output(message),
    )


class LangChainResponsesRuntime:
    def __init__(self, *, api_key: str):
        self.api_key = api_key

    def create(self, *, timeout: float, **create_kwargs: Any) -> LangChainResponse:
        from langchain.chat_models import init_chat_model

        model_name = str(create_kwargs["model"])
        model_options: Dict[str, Any] = {
            "api_key": self.api_key,
            "use_responses_api": True,
            "output_version": "responses/v1",
            "timeout": float(timeout),
            "max_retries": 0,
        }
        if "temperature" in create_kwargs:
            model_options["temperature"] = float(create_kwargs["temperature"])
        model = init_chat_model(f"openai:{model_name}", **model_options)
        invoke_kwargs: Dict[str, Any] = {}
        reasoning = dict(create_kwargs.get("reasoning") or {})
        text = dict(create_kwargs.get("text") or {})
        tools = list(create_kwargs.get("tools") or [])
        if reasoning:
            invoke_kwargs["reasoning"] = reasoning
        if text:
            invoke_kwargs["text"] = text
        if "max_output_tokens" in create_kwargs:
            max_output_tokens = create_kwargs["max_output_tokens"]
            if isinstance(max_output_tokens, bool) or not isinstance(max_output_tokens, int) or max_output_tokens <= 0:
                raise ValueError("max output tokens must be a positive integer")
            invoke_kwargs["max_output_tokens"] = max_output_tokens
        if tools:
            invoke_kwargs["tools"] = tools
            if "parallel_tool_calls" in create_kwargs:
                invoke_kwargs["parallel_tool_calls"] = bool(
                    create_kwargs["parallel_tool_calls"]
                )
        prompt_cache_key = str(create_kwargs.get("prompt_cache_key") or "").strip()
        if prompt_cache_key:
            invoke_kwargs["prompt_cache_key"] = prompt_cache_key
        message = model.invoke(
            _to_langchain_messages(list(create_kwargs.get("input") or [])),
            **invoke_kwargs,
        )
        return _normalize_response(message)
