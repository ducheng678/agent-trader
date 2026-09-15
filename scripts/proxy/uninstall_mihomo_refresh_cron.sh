#!/usr/bin/env bash
set -euo pipefail

MARKER="# AUTO_TROJANFLARE_MIHOMO"
current="$(crontab -l 2>/dev/null || true)"
printf '%s\n' "$current" | grep -v "$MARKER" | crontab -
echo "removed"
