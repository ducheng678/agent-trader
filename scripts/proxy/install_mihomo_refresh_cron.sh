#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MARKER="# AUTO_TROJANFLARE_MIHOMO"
CRON_CMD="17 * * * * cd $ROOT && $ROOT/scripts/proxy/refresh_mihomo_if_due.sh >/dev/null 2>&1 $MARKER"

current="$(crontab -l 2>/dev/null || true)"
filtered="$(printf '%s\n' "$current" | grep -v "$MARKER" || true)"
{
  printf '%s\n' "$filtered"
  printf '%s\n' "$CRON_CMD"
} | sed '/^[[:space:]]*$/N;/^\n$/D' | crontab -

echo "installed"
