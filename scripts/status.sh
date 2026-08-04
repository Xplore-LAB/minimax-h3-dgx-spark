#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/common.sh"

echo "=== Containers ==="
docker compose ps

echo
echo "=== Resource snapshot ==="
docker stats --no-stream minimax-h3-comfyui 2>/dev/null || true

echo
echo "=== Health endpoint ==="
port="${COMFYUI_PORT:-8188}"
if curl -fsS "http://127.0.0.1:${port}/system_stats" >/tmp/h3-system-stats.json 2>/dev/null; then
  jq . /tmp/h3-system-stats.json 2>/dev/null || cat /tmp/h3-system-stats.json
else
  echo "Health endpoint is not ready."
fi

echo
echo "=== Host memory ==="
free -h

echo
echo "=== Data disk ==="
df -hT "${H3_DATA_ROOT}"
