#!/usr/bin/env bash
set -euo pipefail

WEB_TRADE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJECT_ROOT="$(cd "${WEB_TRADE_ROOT}/.." && pwd)"

if [[ -f "${PROJECT_ROOT}/.env" ]]; then
  eval "$(
    PROJECT_ROOT="${PROJECT_ROOT}" python - <<'PY'
import os
import shlex
from pathlib import Path

try:
    from dotenv import dotenv_values
except Exception:
    dotenv_values = None

if dotenv_values is not None:
    values = dotenv_values(Path(os.environ["PROJECT_ROOT"]) / ".env")
else:
    values = {}

for key in ("WEB_ADMIN_TOKEN", "WEB_TRADE_HOST", "WEB_TRADE_PORT", "ENABLE_LIVE_TRADING"):
    value = values.get(key)
    if key not in os.environ and value not in (None, ""):
        print(f"export {key}={shlex.quote(str(value))}")
PY
  )"
fi

HOST="${WEB_TRADE_HOST:-127.0.0.1}"
PORT="${WEB_TRADE_PORT:-8787}"

if [[ -z "${WEB_ADMIN_TOKEN:-}" ]]; then
  WEB_ADMIN_TOKEN="$(python - <<'PY'
import secrets
print(secrets.token_urlsafe(24))
PY
)"
  export WEB_ADMIN_TOKEN
  echo "Generated WEB_ADMIN_TOKEN for this run: ${WEB_ADMIN_TOKEN}"
fi

if [[ ! -d "${WEB_TRADE_ROOT}/frontend/node_modules" ]]; then
  (cd "${WEB_TRADE_ROOT}/frontend" && npm install)
fi

(cd "${WEB_TRADE_ROOT}/frontend" && npm run build)

cloudflared_bin="${CLOUDFLARED_BIN:-}"
if [[ -z "${cloudflared_bin}" ]]; then
  cloudflared_bin="$(command -v cloudflared || true)"
fi

if [[ -z "${cloudflared_bin}" ]]; then
  mkdir -p "${WEB_TRADE_ROOT}/runtime/bin"
  cloudflared_bin="${WEB_TRADE_ROOT}/runtime/bin/cloudflared"
  if [[ ! -x "${cloudflared_bin}" ]]; then
    machine="$(uname -m)"
    case "${machine}" in
      x86_64|amd64)
        cloudflared_url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64"
        ;;
      aarch64|arm64)
        cloudflared_url="https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64"
        ;;
      *)
        echo "Unsupported machine architecture for auto-download: ${machine}" >&2
        exit 1
        ;;
    esac
    echo "Downloading cloudflared to ${cloudflared_bin}"
    curl -fsSL "${cloudflared_url}" -o "${cloudflared_bin}"
    chmod +x "${cloudflared_bin}"
  fi
fi

cd "${PROJECT_ROOT}"
export WEB_TRADE_HOST="${HOST}"
export WEB_TRADE_PORT="${PORT}"
export WEB_ADMIN_TOKEN

python -m web_trade.backend.web_trade &
app_pid="$!"

cleanup() {
  kill "${app_pid}" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

for _ in $(seq 1 60); do
  if curl -fsS "http://${HOST}:${PORT}/api/health" >/dev/null 2>&1; then
    break
  fi
  sleep 0.5
done

if ! curl -fsS "http://${HOST}:${PORT}/api/health" >/dev/null 2>&1; then
  echo "Backend did not become healthy on http://${HOST}:${PORT}" >&2
  exit 1
fi

echo "Backend: http://${HOST}:${PORT}"
echo "Starting Cloudflare Quick Tunnel. Use Authorization token in the web unlock screen."
exec "${cloudflared_bin}" tunnel --url "http://${HOST}:${PORT}" --no-autoupdate
