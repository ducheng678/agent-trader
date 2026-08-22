# Private Hyperliquid Web Trade

This folder contains the private web trade UI and API. It uses the existing project Hyperliquid config and keys; it does not bind a wallet in the browser.

## Run Locally

From the repository root:

```bash
python -m pip install -r web_trade/backend/requirements.txt
cd web_trade/frontend && npm install && npm run build
cd /root/auto_trade
export WEB_ADMIN_TOKEN="change-this-token"
python -m web_trade.backend.web_trade
```

Open `http://127.0.0.1:8787` and unlock with `WEB_ADMIN_TOKEN`.

## Cloudflare Quick Tunnel

```bash
cd /root/auto_trade
WEB_ADMIN_TOKEN="change-this-token" web_trade/scripts/run_with_quick_tunnel.sh
```

If `cloudflared` is not installed, the script downloads it into `web_trade/runtime/bin/`. If `WEB_ADMIN_TOKEN` is not set, the script generates a random token for that run and prints it once.

## Position Semantics

- New orders take margin input in USD and compute position value as `margin_usd * leverage`.
- Existing-position leverage changes use a hidden close -> leverage update -> reopen flow.
- The target notional is `current adjustment-time margin basis * target leverage`.
- Lower leverage keeps margin basis fixed and reduces size; higher leverage keeps margin basis fixed and increases size.
- Display entry price is carried across hidden rebalances.
- Synthetic PnL is carried realized PnL plus current exchange unrealized PnL.
- ROI/PnL% uses net lifecycle invested capital as the denominator.
- Liquidation price is read from the exchange snapshot after the position is reopened.
- Add Margin and Remove Margin are preserved for isolated positions, with max amounts calculated server-side and margin deltas applied to net lifecycle invested capital.
