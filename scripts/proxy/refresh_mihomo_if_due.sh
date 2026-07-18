#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="$ROOT/runtime/proxy/mihomo"
STAMP_PATH="$RUNTIME_DIR/provider_refresh_at"
LOCK_PATH="$RUNTIME_DIR/provider_refresh.lock"
ENV_PATH="$ROOT/.env"
CONFIG_PATH="$RUNTIME_DIR/config.yaml"

env_value() {
  local key="$1"
  if [[ ! -f "$ENV_PATH" ]]; then
    return 0
  fi
  grep -E "^[[:space:]]*${key}=" "$ENV_PATH" | tail -1 | cut -d= -f2- | sed -e 's/^[[:space:]]*//' -e 's/[[:space:]]*$//' || true
}

provider_interval_seconds="43200"
configured="$(env_value MIHOMO_PROVIDER_INTERVAL_SECONDS | tr -d '[:space:]')"
if [[ -n "$configured" ]]; then
  provider_interval_seconds="$configured"
fi

if ! [[ "$provider_interval_seconds" =~ ^[0-9]+$ ]]; then
  echo "[mihomo_provider_refresh] invalid MIHOMO_PROVIDER_INTERVAL_SECONDS=$provider_interval_seconds" >&2
  exit 1
fi

if [[ "$provider_interval_seconds" -le 0 ]]; then
  exit 0
fi

mkdir -p "$RUNTIME_DIR"
exec 9>"$LOCK_PATH"
flock -n 9 || exit 0

now="$(date +%s)"
last_refresh="0"
if [[ -f "$STAMP_PATH" ]]; then
  last_refresh="$(cat "$STAMP_PATH" 2>/dev/null || echo 0)"
fi
if ! [[ "$last_refresh" =~ ^[0-9]+$ ]]; then
  last_refresh="0"
fi

if (( now - last_refresh < provider_interval_seconds )); then
  exit 0
fi

python3 "$ROOT/scripts/proxy/render_mihomo_config.py" >/dev/null

controller="$(env_value MIHOMO_EXTERNAL_CONTROLLER)"
controller="${controller:-127.0.0.1:9097}"
secret="$(env_value MIHOMO_EXTERNAL_CONTROLLER_SECRET)"

if python3 - "$controller" "$secret" "$CONFIG_PATH" <<'PY'
import json
import sys
import urllib.error
import urllib.request

controller, secret, config_path = sys.argv[1:4]
url = f"http://{controller.rstrip('/')}/configs?force=true"
payload = json.dumps({"path": config_path}).encode("utf-8")
headers = {"Content-Type": "application/json"}
if secret:
    headers["Authorization"] = f"Bearer {secret}"
request = urllib.request.Request(url, data=payload, headers=headers, method="PUT")
try:
    with urllib.request.urlopen(request, timeout=10) as response:
        if 200 <= response.status < 300:
            sys.exit(0)
        print(f"unexpected reload status={response.status}", file=sys.stderr)
except (OSError, urllib.error.URLError, urllib.error.HTTPError) as exc:
    print(exc, file=sys.stderr)
sys.exit(1)
PY
then
  date +%s > "$STAMP_PATH"
  echo "[mihomo_provider_refresh] reloaded at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
else
  echo "[mihomo_provider_refresh] reload failed; restarting Mihomo" >&2
  "$ROOT/scripts/proxy/start_mihomo.sh" >/dev/null
  echo "[mihomo_provider_refresh] restarted at $(date -u +%Y-%m-%dT%H:%M:%SZ)"
fi
