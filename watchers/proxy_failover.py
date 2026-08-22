from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

import requests

from .watcher_proxy_config import env_lookup, source_selector_group


@dataclass(frozen=True)
class SourceProxyConfig:
    http_proxy: str = ""
    https_proxy: str = ""
    failover_enabled: bool = False
    selector_group: str = "PROXY"
    max_rotations: int = 0
    sleep_seconds: float = 1.0
    healthcheck_url: str = ""
    probe_timeout_ms: int = 5000
    probe_concurrency: int = 8

    @property
    def has_proxy(self) -> bool:
        return bool(self.http_proxy or self.https_proxy)


def load_source_proxy_config(source_name: str) -> SourceProxyConfig:
    def _bool(value: str, default: bool) -> bool:
        if value == "":
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    http_proxy = env_lookup(source_name, "HTTP_PROXY", "")
    https_proxy = env_lookup(source_name, "HTTPS_PROXY", "")
    selector_group = source_selector_group(source_name)
    if selector_group != "PROXY" and not (http_proxy or https_proxy):
        mixed_port = int(os.getenv("MIHOMO_MIXED_PORT", "7897") or 7897)
        local_proxy = f"http://127.0.0.1:{mixed_port}"
        http_proxy = local_proxy
        https_proxy = local_proxy
    failover_enabled = _bool(env_lookup(source_name, "PROXY_FAILOVER_ENABLED", ""), True)
    max_rotations = max(0, int(env_lookup(source_name, "PROXY_FAILOVER_MAX_ROTATIONS", "3") or 3))
    sleep_seconds = max(0.0, float(env_lookup(source_name, "PROXY_FAILOVER_SLEEP_SECONDS", "1.0") or 1.0))
    healthcheck_url = env_lookup(source_name, "PROXY_HEALTHCHECK_URL", "")
    probe_timeout_ms = max(1, int(env_lookup(source_name, "PROXY_FAILOVER_PROBE_TIMEOUT_MS", "5000") or 5000))
    probe_concurrency = max(1, int(env_lookup(source_name, "PROXY_FAILOVER_PROBE_CONCURRENCY", "8") or 8))
    return SourceProxyConfig(
        http_proxy=http_proxy,
        https_proxy=https_proxy,
        failover_enabled=failover_enabled,
        selector_group=selector_group,
        max_rotations=max_rotations,
        sleep_seconds=sleep_seconds,
        healthcheck_url=healthcheck_url,
        probe_timeout_ms=probe_timeout_ms,
        probe_concurrency=probe_concurrency,
    )


class MihomoProxyFailover:
    def __init__(
        self,
        controller: str,
        secret: str = "",
        selector_group: str = "PROXY",
        *,
        healthcheck_url: str = "",
        probe_timeout_ms: int = 5000,
        probe_concurrency: int = 8,
    ):
        self.controller = controller.strip().rstrip("/")
        self.secret = secret.strip()
        self.selector_group = selector_group.strip() or "PROXY"
        self.healthcheck_url = str(healthcheck_url or "").strip()
        self.probe_timeout_ms = max(1, int(probe_timeout_ms or 5000))
        self.probe_concurrency = max(1, int(probe_concurrency or 8))
        self.last_probe_summary = ""
        self.session = requests.Session()

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if self.secret:
            headers["Authorization"] = f"Bearer {self.secret}"
        return headers

    def _url(self, path: str) -> str:
        return f"http://{self.controller}{path}"

    def get_proxy_detail(self, name: str) -> dict[str, Any]:
        response = self.session.get(
            self._url(f"/proxies/{requests.utils.quote(name, safe='')}"),
            headers=self._headers(),
            timeout=5,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError(f"Unexpected Mihomo proxy detail payload for {name}: {type(payload)}")
        return payload

    def set_selector(self, selector_name: str, target_name: str) -> None:
        response = self.session.put(
            self._url(f"/proxies/{requests.utils.quote(selector_name, safe='')}"),
            headers={**self._headers(), "Content-Type": "application/json"},
            json={"name": target_name},
            timeout=5,
        )
        response.raise_for_status()

    @staticmethod
    def _is_concrete_node(name: str) -> bool:
        raw = str(name or "").strip()
        if not raw:
            return False
        return raw.upper() not in {"DIRECT", "REJECT"}

    @staticmethod
    def _is_proxy_group(detail: dict[str, Any]) -> bool:
        proxy_type = str(detail.get("type") or "").strip().lower().replace("-", "").replace(" ", "")
        return isinstance(detail.get("all"), list) or proxy_type in {
            "selector",
            "urltest",
            "fallback",
            "loadbalance",
            "relay",
        }

    def _is_leaf_proxy(self, name: str) -> bool:
        try:
            detail = self.get_proxy_detail(name)
        except Exception:
            return True
        return not self._is_proxy_group(detail)

    def _resolve_nested_current_selection(self, current_selection: str) -> str:
        current_selection = str(current_selection or "").strip()
        if not current_selection:
            return ""
        try:
            detail = self.get_proxy_detail(current_selection)
        except Exception:
            return current_selection
        if not self._is_proxy_group(detail):
            return current_selection
        nested = str(detail.get("now") or "").strip()
        return nested or current_selection

    @staticmethod
    def _is_valid_healthcheck_url(url: str) -> bool:
        parsed = urlparse(str(url or "").strip())
        return parsed.scheme in {"http", "https"} and bool(parsed.netloc)

    def _probe_node_delay(self, node_name: str) -> Optional[int]:
        if not self._is_valid_healthcheck_url(self.healthcheck_url):
            return None
        response = requests.get(
            self._url(f"/proxies/{requests.utils.quote(node_name, safe='')}/delay"),
            headers=self._headers(),
            params={"timeout": self.probe_timeout_ms, "url": self.healthcheck_url},
            timeout=max(1.0, self.probe_timeout_ms / 1000.0 + 2.0),
        )
        response.raise_for_status()
        payload = response.json()
        delay = payload.get("delay")
        try:
            delay_ms = int(delay)
        except (TypeError, ValueError):
            return None
        return delay_ms if delay_ms >= 0 else None

    def _ordered_failover_candidates(self, concrete_nodes: list[str], current_selection: str) -> list[str]:
        if not concrete_nodes:
            return []
        if current_selection in concrete_nodes:
            idx = concrete_nodes.index(current_selection)
            return concrete_nodes[idx + 1 :] + concrete_nodes[:idx]
        return list(concrete_nodes)

    def _select_fastest_healthy_node(self, candidates: list[str]) -> Optional[tuple[str, int]]:
        if not candidates or not self._is_valid_healthcheck_url(self.healthcheck_url):
            return None
        results: list[tuple[int, int, str]] = []
        max_workers = min(self.probe_concurrency, len(candidates))
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {
                pool.submit(self._probe_node_delay, node_name): (idx, node_name)
                for idx, node_name in enumerate(candidates)
            }
            for future in as_completed(futures):
                idx, node_name = futures[future]
                try:
                    delay_ms = future.result()
                except Exception:
                    continue
                if delay_ms is not None:
                    results.append((delay_ms, idx, node_name))
        if not results:
            return None
        delay_ms, _, node_name = min(results)
        return node_name, delay_ms

    def rotate_to_next_node(self) -> Optional[str]:
        self.last_probe_summary = ""
        group_detail = self.get_proxy_detail(self.selector_group)
        all_names = [str(item).strip() for item in list(group_detail.get("all") or []) if str(item).strip()]
        concrete_nodes = [name for name in all_names if self._is_concrete_node(name) and self._is_leaf_proxy(name)]
        if not concrete_nodes:
            return None
        raw_selection = str(group_detail.get("now") or "").strip()
        current_selection = self._resolve_nested_current_selection(raw_selection)
        candidates = self._ordered_failover_candidates(concrete_nodes, current_selection)
        probed = self._select_fastest_healthy_node(candidates)
        if probed is not None:
            target, delay_ms = probed
            self.last_probe_summary = f"probe={delay_ms}ms"
        elif raw_selection != current_selection:
            target = candidates[0] if candidates else concrete_nodes[0]
        elif current_selection in concrete_nodes:
            target = candidates[0] if candidates else concrete_nodes[0]
        else:
            target = concrete_nodes[0]
        self.set_selector(self.selector_group, target)
        return target


def build_source_proxy_failover(
    source_name: str,
    config: SourceProxyConfig,
    *,
    healthcheck_url: str = "",
) -> Optional[MihomoProxyFailover]:
    if not config.failover_enabled or not config.has_proxy:
        return None
    controller = str(os.getenv("MIHOMO_EXTERNAL_CONTROLLER", "") or "").strip()
    if not controller:
        return None
    secret = str(os.getenv("MIHOMO_EXTERNAL_CONTROLLER_SECRET", "") or "").strip()
    return MihomoProxyFailover(
        controller=controller,
        secret=secret,
        selector_group=config.selector_group,
        healthcheck_url=config.healthcheck_url or healthcheck_url,
        probe_timeout_ms=config.probe_timeout_ms,
        probe_concurrency=config.probe_concurrency,
    )
