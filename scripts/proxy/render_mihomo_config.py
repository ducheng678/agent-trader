#!/usr/bin/env python3
from __future__ import annotations

import base64
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlsplit

import requests
from dotenv import load_dotenv


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from watchers.watcher_proxy_config import WatcherProxyTarget, build_proxy_targets_from_watchers, env_lookup

RUNTIME_DIR = ROOT / "runtime" / "proxy" / "mihomo"
CONFIG_PATH = RUNTIME_DIR / "config.yaml"
NODES_PATH = RUNTIME_DIR / "nodes.json"
FETCH_UA = os.getenv("TROJANFLARE_SUBSCRIPTION_USER_AGENT", "ClashMetaForAndroid/2.11.5")


@dataclass(frozen=True)
class TrojanNode:
    name: str
    server: str
    port: int
    password: str
    sni: str
    skip_cert_verify: bool

    def dedupe_key(self) -> tuple[str, int, str, str, bool]:
        return (self.server, self.port, self.password, self.sni, self.skip_cert_verify)


def _subscription_urls() -> list[str]:
    urls: list[str] = []
    combined = os.getenv("TROJANFLARE_SUBSCRIPTION_URLS", "")
    if combined.strip():
        urls.extend(item.strip() for item in combined.split(",") if item.strip())
    for key in (
        "TROJANFLARE_SUBSCRIPTION_URL_1",
        "TROJANFLARE_SUBSCRIPTION_URL_2",
        "TROJANFLARE_SUBSCRIPTION_URL_3",
    ):
        value = os.getenv(key, "").strip()
        if value:
            urls.append(value)
    deduped: list[str] = []
    seen: set[str] = set()
    for url in urls:
        if url in seen:
            continue
        seen.add(url)
        deduped.append(url)
    if not deduped:
        raise RuntimeError("No subscription URLs configured")
    return deduped


def _bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _fetch_subscription_lines(url: str) -> list[str]:
    response = requests.get(
        url,
        timeout=20,
        headers={
            "User-Agent": FETCH_UA,
            "Accept": "*/*",
        },
    )
    response.raise_for_status()
    payload = response.text.strip()
    decoded = base64.b64decode(payload + "=" * (-len(payload) % 4)).decode("utf-8", "replace")
    return [line.strip() for line in decoded.splitlines() if line.strip()]


def _parse_bool(value: str) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _safe_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {"-", "_", ".", " "} else "_" for ch in name).strip()
    return cleaned[:96] or "node"


def _parse_trojan_uri(uri: str, sub_idx: int, node_idx: int) -> TrojanNode:
    parsed = urlsplit(uri)
    if parsed.scheme != "trojan":
        raise ValueError(f"Unsupported proxy scheme: {parsed.scheme}")
    query = parse_qs(parsed.query or "", keep_blank_values=True)
    label = unquote(parsed.fragment or f"sub{sub_idx}-{node_idx}")
    return TrojanNode(
        name=f"{_safe_name(label)} [S{sub_idx}]",
        server=parsed.hostname or "",
        port=int(parsed.port or 443),
        password=unquote(parsed.username or ""),
        sni=(query.get("sni") or query.get("peer") or [parsed.hostname or ""])[0],
        skip_cert_verify=_parse_bool((query.get("allowInsecure") or query.get("skip-cert-verify") or ["0"])[0]),
    )


def _load_nodes(urls: list[str]) -> list[TrojanNode]:
    nodes: list[TrojanNode] = []
    seen: set[tuple[str, int, str, str, bool]] = set()
    for sub_idx, url in enumerate(urls, start=1):
        for node_idx, line in enumerate(_fetch_subscription_lines(url), start=1):
            node = _parse_trojan_uri(line, sub_idx, node_idx)
            if node.dedupe_key() in seen:
                continue
            seen.add(node.dedupe_key())
            nodes.append(node)
    if not nodes:
        raise RuntimeError("No usable Trojan nodes parsed from subscriptions")
    return nodes


def _source_proxy_blocks(
    proxy_name_lines: str,
    default_interval: int,
    targets: list[WatcherProxyTarget],
) -> tuple[list[str], list[str]]:
    group_lines: list[str] = []
    rule_lines: list[str] = []
    for target in targets:
        health_interval = int(env_lookup(target.source_base, "PROXY_HEALTHCHECK_INTERVAL_SECONDS", str(default_interval)))
        group_lines.extend(
            [
                f"  - name: {target.auto_group}",
                "    type: url-test",
                f"    url: {_quote(target.healthcheck_url)}",
                f"    interval: {health_interval}",
                "    tolerance: 50",
                "    proxies:",
                proxy_name_lines,
                f"  - name: {target.selector_group}",
                "    type: select",
                "    proxies:",
                f"      - {target.auto_group}",
                "      - DIRECT",
                proxy_name_lines,
            ]
        )
        rule_lines.extend(f"  - DOMAIN-SUFFIX,{domain},{target.selector_group}" for domain in target.rule_domains)
    return group_lines, rule_lines


def main() -> int:
    load_dotenv(ROOT / ".env")
    from watch_free_sources_modular import build_watchers

    urls = _subscription_urls()
    nodes = _load_nodes(urls)
    mixed_port = int(os.getenv("MIHOMO_MIXED_PORT", "7897"))
    controller = os.getenv("MIHOMO_EXTERNAL_CONTROLLER", "127.0.0.1:9097").strip()
    secret = os.getenv("MIHOMO_EXTERNAL_CONTROLLER_SECRET", "").strip()
    allow_lan = _bool_env("MIHOMO_ALLOW_LAN", False)
    log_level = os.getenv("MIHOMO_LOG_LEVEL", "info").strip() or "info"
    health_url = os.getenv("MIHOMO_HEALTHCHECK_URL", "https://www.gstatic.com/generate_204").strip()
    health_interval = int(os.getenv("MIHOMO_HEALTHCHECK_INTERVAL_SECONDS", "600"))
    proxy_blocks: list[str] = []
    proxy_names: list[str] = []
    for node in nodes:
        proxy_names.append(node.name)
        proxy_blocks.extend(
            [
                f"  - name: {_quote(node.name)}",
                "    type: trojan",
                f"    server: {_quote(node.server)}",
                f"    port: {node.port}",
                f"    password: {_quote(node.password)}",
                "    udp: true",
                f"    sni: {_quote(node.sni)}",
                f"    skip-cert-verify: {'true' if node.skip_cert_verify else 'false'}",
            ]
        )
    proxy_name_lines = "\n".join(f"      - {_quote(name)}" for name in proxy_names)
    watcher_proxy_targets = build_proxy_targets_from_watchers(build_watchers())
    source_group_lines, source_rule_lines = _source_proxy_blocks(proxy_name_lines, health_interval, watcher_proxy_targets)
    config = "\n".join(
        [
            f"mixed-port: {mixed_port}",
            f"allow-lan: {'true' if allow_lan else 'false'}",
            "mode: rule",
            f"log-level: {log_level}",
            f"external-controller: {_quote(controller)}",
            f"secret: {_quote(secret)}",
            "",
            "proxies:",
            *proxy_blocks,
            "",
            "proxy-groups:",
            "  - name: TrojanFlare Auto",
            "    type: url-test",
            f"    url: {_quote(health_url)}",
            f"    interval: {health_interval}",
            "    tolerance: 50",
            "    proxies:",
            proxy_name_lines,
            *source_group_lines,
            "  - name: PROXY",
            "    type: select",
            "    proxies:",
            "      - TrojanFlare Auto",
            "      - DIRECT",
            proxy_name_lines,
            "",
            "rules:",
            *source_rule_lines,
            "  - MATCH,PROXY",
            "",
        ]
    )

    RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(config, encoding="utf-8")
    NODES_PATH.write_text(
        __import__("json").dumps(
            {
                "subscription_count": len(urls),
                "node_count": len(nodes),
                "nodes": [node.__dict__ for node in nodes],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(CONFIG_PATH)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
