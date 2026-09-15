#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PID_PATH="$ROOT/runtime/proxy/mihomo/mihomo.pid"

if [[ ! -f "$PID_PATH" ]]; then
  exit 0
fi

pid="$(cat "$PID_PATH")"
if kill -0 "$pid" 2>/dev/null; then
  kill "$pid"
fi
rm -f "$PID_PATH"
