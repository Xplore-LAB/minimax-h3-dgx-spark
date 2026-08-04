#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/common.sh"

stamp="$(date +%Y%m%d_%H%M%S)"
out="${PROJECT_DIR}/diagnostics/h3_diagnostics_${stamp}"
mkdir -p "${out}"

{
  date -Is
  uname -a
  echo
  cat /etc/os-release 2>/dev/null || true
} >"${out}/system.txt" 2>&1

{
  free -h
  echo
  cat /proc/meminfo
} >"${out}/memory.txt" 2>&1

lsblk -o NAME,SIZE,FSTYPE,MOUNTPOINTS,MODEL >"${out}/lsblk.txt" 2>&1 || true
df -hT >"${out}/df.txt" 2>&1 || true
nvidia-smi -L >"${out}/nvidia-smi-L.txt" 2>&1 || true
nvidia-smi >"${out}/nvidia-smi.txt" 2>&1 || true
docker version >"${out}/docker-version.txt" 2>&1 || true
docker compose version >"${out}/compose-version.txt" 2>&1 || true
docker compose ps >"${out}/compose-ps.txt" 2>&1 || true
docker compose logs --tail=500 comfyui >"${out}/comfyui-logs.txt" 2>&1 || true
docker stats --no-stream >"${out}/docker-stats.txt" 2>&1 || true

if [[ -f .env ]]; then
  sed -E 's/^(HF_TOKEN=).*/\1***REDACTED***/' .env >"${out}/env-redacted.txt"
fi

find "${H3_DATA_ROOT}/models" -maxdepth 3 -type f -printf '%p\t%s bytes\n' \
  >"${out}/model-files.txt" 2>&1 || true

tarball="${out}.tar.gz"
tar -C "$(dirname "${out}")" -czf "${tarball}" "$(basename "${out}")"
rm -rf "${out}"
echo "Diagnostics package: ${tarball}"
