# hyperliquid_account_debug.py
import json
import os
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv

load_dotenv()


def post_info(base: str, payload: dict) -> dict:
    r = requests.post(f"{base}/info", json=payload, timeout=20)
    r.raise_for_status()
    return r.json()


def safe_get(d: Any, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k)
        if cur is None:
            return default
    return cur


def base_url(network: str) -> str:
    network = (network or "").lower()
    if network == "mainnet":
        return "https://api.hyperliquid.xyz"
    if network == "testnet":
        return "https://api.hyperliquid-testnet.xyz"
    raise ValueError("HYPERLIQUID_NETWORK must be mainnet or testnet")


def pretty(title: str, obj: Any) -> None:
    print(f"[{title}]")
    print(json.dumps(obj, ensure_ascii=False, indent=2))


def try_call(base: str, payload: dict) -> dict:
    try:
        return {"ok": True, "data": post_info(base, payload)}
    except Exception as e:
        return {"ok": False, "error": repr(e), "payload": payload}


def extract_vault_addresses(user_vault_equities: Any) -> List[str]:
    out = []
    if isinstance(user_vault_equities, list):
        for item in user_vault_equities:
            if isinstance(item, dict):
                addr = item.get("vaultAddress") or item.get("address")
                if isinstance(addr, str) and addr.startswith("0x"):
                    out.append(addr)
    return out


def summarize_positions(ch_state: Any) -> List[dict]:
    res = []
    if not isinstance(ch_state, dict):
        return res
    for entry in ch_state.get("assetPositions", []) or []:
        pos = (entry or {}).get("position", {}) or {}
        coin = pos.get("coin")
        szi = pos.get("szi")
        entry_px = pos.get("entryPx")
        upnl = pos.get("unrealizedPnl")
        if coin is not None:
            res.append(
                {
                    "coin": coin,
                    "szi": szi,
                    "entryPx": entry_px,
                    "unrealizedPnl": upnl,
                }
            )
    return res


def main() -> None:
    network = os.getenv("HYPERLIQUID_NETWORK", "mainnet").lower().strip()
    address = (os.getenv("HL_ACCOUNT_ADDRESS", "") or "").strip()

    if not address:
        raise RuntimeError("HL_ACCOUNT_ADDRESS 未设置")

    base = base_url(network)

    print("[config]")
    print(f"network={network}")
    print(f"address={address}")
    print(f"base={base}")

    # 1) 角色检查：是不是 agent / user / vault / subAccount
    role = try_call(base, {"type": "userRole", "user": address})
    pretty("userRole", role)

    # 2) 主账户 perp 状态
    ch = try_call(base, {"type": "clearinghouseState", "user": address})
    pretty("clearinghouseState", ch)

    # 3) 组合/统一账户相关
    portfolio = try_call(base, {"type": "portfolio", "user": address})
    pretty("portfolio", portfolio)

    # 4) 子账户
    subs = try_call(base, {"type": "subAccounts", "user": address})
    pretty("subAccounts", subs)

    # 5) vault
    uve = try_call(base, {"type": "userVaultEquities", "user": address})
    pretty("userVaultEquities", uve)

    # 6) 如果有 vault，再逐个看
    vault_addresses: List[str] = []
    if uve.get("ok"):
        vault_addresses = extract_vault_addresses(uve["data"])

    vault_details: Dict[str, Any] = {}
    for vaddr in vault_addresses:
        vault_details[vaddr] = try_call(base, {"type": "vaultDetails", "vaultAddress": vaddr})
    pretty("vaultDetails", vault_details)

    # 简要结论
    print("[summary]")

    role_data = role.get("data") if role.get("ok") else None
    role_name = None
    if isinstance(role_data, dict):
        role_name = role_data.get("role")

    ch_positions = summarize_positions(ch.get("data")) if ch.get("ok") else []
    sub_data = subs.get("data") if subs.get("ok") else None
    sub_count = len(sub_data) if isinstance(sub_data, list) else 0
    vault_count = len(vault_addresses)

    print(f"role={role_name}")
    print(f"main_clearinghouse_positions={len(ch_positions)}")
    print(f"subaccounts_count={sub_count}")
    print(f"vaults_count={vault_count}")

    if role_name == "agent":
        print("结论：你填的很可能是 agent/API wallet 地址，不是实际账户地址。")
    elif len(ch_positions) > 0:
        print("结论：主账户 perp 仓位可直接读取。")
    elif sub_count > 0:
        print("结论：主账户下可能有 subaccount，仓位可能在子账户里。")
    elif vault_count > 0:
        print("结论：仓位/资金可能和 vault 相关。")
    else:
        print("结论：主账户 clearinghouseState 为空；若网页里仍有仓位，重点排查 unified/portfolio margin、子账户、或地址类型。")


if __name__ == "__main__":
    main()
