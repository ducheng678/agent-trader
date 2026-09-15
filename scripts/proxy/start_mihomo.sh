#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="$ROOT/runtime/proxy/mihomo"
BIN_PATH="$RUNTIME_DIR/mihomo"
PID_PATH="$RUNTIME_DIR/mihomo.pid"
LOG_PATH="$RUNTIME_DIR/mihomo.log"
CONFIG_PATH="$RUNTIME_DIR/config.yaml"
REFRESH_STAMP_PATH="$RUNTIME_DIR/provider_refresh_at"

cd "$ROOT"

if [[ ! -x "$BIN_PATH" ]]; then
  "$ROOT/scripts/proxy/download_mihomo.sh" >/dev/null
fi

python3 "$ROOT/scripts/proxy/render_mihomo_config.py" >/dev/null

if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
  kill "$(cat "$PID_PATH")"
  sleep 1
fi

setsid "$BIN_PATH" -d "$RUNTIME_DIR" -f "$CONFIG_PATH" </dev/null >"$LOG_PATH" 2>&1 &
echo $! > "$PID_PATH"
date +%s > "$REFRESH_STAMP_PATH"
echo "$PID_PATH"
