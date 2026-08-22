import json
import os
import sys
from typing import Any

import requests
from dotenv import load_dotenv

load_dotenv()


def safe_float(value: Any, default: float = 0.0):
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def base_url(network: str) -> str:
    network = (network or "").lower()
    if network == "mainnet":
        return "https://api.hyperliquid.xyz"
    if network == "testnet":
        return "https://api.hyperliquid-testnet.xyz"
    raise ValueError("HYPERLIQUID_NETWORK must be testnet or mainnet")


def post_info(network: str, payload: dict) -> Any:
    url = base_url(network).rstrip("/") + "/info"
    resp = requests.post(url, json=payload, headers={"Content-Type": "application/json"}, timeout=15)
    resp.raise_for_status()
    return resp.json()


def get_all_positions(address: str, network: str) -> dict:
    snapshot = {
        "known": False,
        "account_address": address,
        "network": network,
        "positions": [],
        "positions_count": 0,
        "total_notional_usd": 0.0,
    }
    user_state = post_info(network, {"type": "clearinghouseState", "user": address})
    mids = post_info(network, {"type": "allMids"})
    positions = []
    total_notional = 0.0

    for entry in user_state.get("assetPositions", []) or []:
        position = entry.get("position", {}) or {}
        coin = (position.get("coin") or "").upper()
        if not coin:
            continue
        szi = safe_float(position.get("szi", 0.0))
        entry_px = safe_float(position.get("entryPx", 0.0))
        mid = safe_float(mids.get(coin), None)
        side = "flat"
        if szi > 0:
            side = "long"
        elif szi < 0:
            side = "short"
        notional = abs(szi) * (mid if mid is not None else entry_px)
        total_notional += notional
        positions.append(
            {
                "symbol": coin,
                "side": side,
                "size": szi,
                "entry_price": entry_px,
                "mid_price": mid,
                "notional_usd": notional,
                "unrealized_pnl": safe_float(position.get("unrealizedPnl", 0.0)),
                "return_on_equity": safe_float(position.get("returnOnEquity", 0.0)),
                "leverage": safe_float((position.get("leverage") or {}).get("value"), 0.0),
                "position_value": safe_float(position.get("positionValue", 0.0)),
                "liquidation_price": safe_float(position.get("liquidationPx", 0.0), 0.0),
                "margin_used": safe_float(position.get("marginUsed", 0.0)),
            }
        )

    positions.sort(key=lambda x: abs(float(x.get("notional_usd", 0.0) or 0.0)), reverse=True)
    snapshot.update(
        {
            "known": True,
            "positions": positions,
            "positions_count": len(positions),
            "total_notional_usd": total_notional,
        }
    )
    return snapshot


def format_all_positions_lines(snapshot: dict) -> str:
    address = snapshot.get("account_address") or "<not set>"
    network = snapshot.get("network") or "<unknown>"
    positions = snapshot.get("positions", []) or []
    total_notional = float(snapshot.get("total_notional_usd", 0.0) or 0.0)
    lines = [
        f"address={address}",
        f"network={network}",
        f"positions_count={len(positions)}",
        f"total_notional_usd≈{total_notional:.2f}",
    ]
    if not positions:
        lines.append("positions=<empty>")
        return "\n".join(lines)
    lines.append("positions=")
    for idx, pos in enumerate(positions, start=1):
        mid_price = pos.get("mid_price")
        mid_text = "null" if mid_price is None else f"{float(mid_price):.6f}"
        lines.append(
            "  "
            + f"[{idx}] {pos.get('symbol','')} side={pos.get('side','flat')} "
            + f"size={float(pos.get('size', 0.0) or 0.0):.8f} "
            + f"entry={float(pos.get('entry_price', 0.0) or 0.0):.6f} "
            + f"mid={mid_text} "
            + f"notional≈{float(pos.get('notional_usd', 0.0) or 0.0):.2f} "
            + f"upnl={float(pos.get('unrealized_pnl', 0.0) or 0.0):.6f} "
            + f"lev={float(pos.get('leverage', 0.0) or 0.0):.2f}"
        )
    return "\n".join(lines)


def main() -> None:
    network = os.getenv("HYPERLIQUID_NETWORK", "testnet").lower()
    address = (os.getenv("HL_ACCOUNT_ADDRESS", "") or "").strip()

    if not address:
        raise RuntimeError("HL_ACCOUNT_ADDRESS 未设置，请先在 .env 里填你的 0x 地址")

    snapshot = get_all_positions(address, network)

    print("[config]")
    print(f"network={network}")
    print(f"address={address}")
    print("[all_positions]")
    print(format_all_positions_lines(snapshot))
    print("[all_positions_json]")
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))

    if not snapshot.get("known"):
        print("[hint] 地址未设置或查询失败。")
    elif snapshot.get("positions_count", 0) == 0:
        print("[hint] 这个地址当前没有任何持仓。如果你明明有仓位，通常是地址不对、网络不对，或者你看的不是这个账户。")
    else:
        print("[hint] 已查到全部持仓，地址大概率填对了。")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[shutdown] interrupted by user")
        sys.exit(0)
