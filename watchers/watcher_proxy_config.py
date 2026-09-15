from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


URL_ATTRS = ("url", "feed_url", "target_url", "base_url")
URL_LIST_ATTRS = ("feed_urls", "urls")
SOURCE_LABEL_OVERRIDES = {
    "aaa": "AAA",
    "adp": "ADP",
    "bls": "BLS",
    "dol": "DOL",
    "fed": "Fed",
    "irna": "IRNA",
    "mni": "MNI",
    "nyt": "NYT",
    "sec": "SEC",
    "wsj": "WSJ",
}


@dataclass(frozen=True)
class WatcherProxyTarget:
    source_base: str
    selector_group: str
    auto_group: str
    healthcheck_url: str
    rule_domains: tuple[str, ...]


def source_base(source_name: str) -> str:
    return str(source_name or "").split(":", 1)[0].strip()


def env_prefix_variants(source_name: str) -> list[str]:
    base = source_base(source_name)
    if not base:
        return []
    normalized = re.sub(r"[^A-Za-z0-9]+", "_", base).strip("_").upper()
    collapsed = normalized.replace("_", "")
    variants: list[str] = []
    for item in (normalized, collapsed):
        if item and item not in variants:
            variants.append(item)
    return variants


def env_lookup(source_name: str, suffix: str, default: str = "") -> str:
    for prefix in env_prefix_variants(source_name):
        value = os.getenv(f"{prefix}_{suffix}")
        if value is not None and str(value).strip():
            return str(value).strip()
    return default


def source_label(source_name: str) -> str:
    base = source_base(source_name)
    if base.lower() in SOURCE_LABEL_OVERRIDES:
        return SOURCE_LABEL_OVERRIDES[base.lower()]
    words = re.split(r"[^A-Za-z0-9]+", base)
    return "".join(word[:1].upper() + word[1:] for word in words if word) or "Source"


def default_source_auto_group(source_name: str) -> str:
    return f"{source_label(source_name)} Auto"


def default_source_selector_group(source_name: str) -> str:
    return f"{source_label(source_name)} Proxy"


def configured_proxy_sources() -> set[str]:
    raw = os.getenv("PROXY_WATCHER_SOURCES", "").strip()
    if not raw:
        return set()
    return {item.strip().lower() for item in raw.split(",") if item.strip()}


def source_proxy_requested(source_name: str) -> bool:
    base = source_base(source_name).lower()
    configured = configured_proxy_sources()
    if "*" in configured or base in configured:
        return True
    if env_lookup(source_name, "HTTP_PROXY", "") or env_lookup(source_name, "HTTPS_PROXY", ""):
        return True
    selector_group = env_lookup(source_name, "PROXY_SELECTOR_GROUP", "")
    return bool(selector_group and selector_group != "PROXY")


def source_selector_group(source_name: str, *, default_when_unproxied: str = "PROXY") -> str:
    default = default_source_selector_group(source_name) if source_proxy_requested(source_name) else default_when_unproxied
    return env_lookup(source_name, "PROXY_SELECTOR_GROUP", default).strip() or default


def source_auto_group(source_name: str) -> str:
    default = default_source_auto_group(source_name)
    auto_group = env_lookup(source_name, "PROXY_AUTO_GROUP", default).strip() or default
    selector_group = source_selector_group(source_name)
    if auto_group == selector_group:
        return f"{selector_group} UrlTest"
    return auto_group


def _iter_url_values(value: Any) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value if str(item).strip()]
    return []


def extract_watcher_urls(watcher: Any) -> list[str]:
    urls: list[str] = []
    seen: set[str] = set()
    for attr in URL_ATTRS:
        for url in _iter_url_values(getattr(watcher, attr, "")):
            cleaned = url.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                urls.append(cleaned)
    for attr in URL_LIST_ATTRS:
        for url in _iter_url_values(getattr(watcher, attr, [])):
            cleaned = url.strip()
            if cleaned and cleaned not in seen:
                seen.add(cleaned)
                urls.append(cleaned)
    return urls


def normalize_healthcheck_url(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return ""
    if not parsed.path:
        return f"{parsed.scheme}://{parsed.netloc}/"
    return parsed.geturl()


def domain_suffix(url: str) -> str:
    parsed = urlparse(str(url or "").strip())
    host = (parsed.hostname or "").strip(".").lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def build_proxy_targets_from_watchers(watchers: list[Any]) -> list[WatcherProxyTarget]:
    grouped_urls: dict[str, list[str]] = {}
    grouped_domains: dict[str, list[str]] = {}
    for watcher in watchers:
        source_name = str(getattr(watcher, "source_name", "") or "")
        if not source_name or not source_proxy_requested(source_name):
            continue
        base = source_base(source_name)
        if not base:
            continue
        grouped_urls.setdefault(base, [])
        grouped_domains.setdefault(base, [])
        for url in extract_watcher_urls(watcher):
            root = normalize_healthcheck_url(url)
            domain = domain_suffix(url)
            if root and root not in grouped_urls[base]:
                grouped_urls[base].append(root)
            if domain and domain not in grouped_domains[base]:
                grouped_domains[base].append(domain)

    targets: list[WatcherProxyTarget] = []
    for base in sorted(grouped_urls):
        explicit_healthcheck = env_lookup(base, "PROXY_HEALTHCHECK_URL", "")
        healthcheck_url = explicit_healthcheck or (grouped_urls[base][0] if grouped_urls[base] else "")
        explicit_domains = env_lookup(base, "PROXY_RULE_DOMAINS", "")
        rule_domains = (
            tuple(item.strip() for item in explicit_domains.split(",") if item.strip())
            if explicit_domains
            else tuple(grouped_domains.get(base, ()))
        )
        if not healthcheck_url or not rule_domains:
            continue
        selector_group = source_selector_group(base)
        if selector_group == "PROXY":
            continue
        targets.append(
            WatcherProxyTarget(
                source_base=base,
                selector_group=selector_group,
                auto_group=source_auto_group(base),
                healthcheck_url=healthcheck_url,
                rule_domains=rule_domains,
            )
        )
    return targets
