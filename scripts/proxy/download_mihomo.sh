#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNTIME_DIR="$ROOT/runtime/proxy/mihomo"
BIN_PATH="$RUNTIME_DIR/mihomo"
VERSION="${MIHOMO_VERSION:-v1.19.24}"
ASSET="${MIHOMO_ASSET:-mihomo-linux-amd64-v1.19.24.gz}"
URL="https://github.com/MetaCubeX/mihomo/releases/download/${VERSION}/${ASSET}"

mkdir -p "$RUNTIME_DIR"
tmp_gz="$(mktemp)"
trap 'rm -f "$tmp_gz"' EXIT

curl -fsSL "$URL" -o "$tmp_gz"
gzip -dc "$tmp_gz" > "$BIN_PATH"
chmod +x "$BIN_PATH"
echo "$BIN_PATH"
