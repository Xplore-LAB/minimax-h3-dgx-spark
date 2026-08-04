#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/common.sh"

[[ -f .env ]] || die ".env not found. Run: make init DATA_ROOT=/mnt/<4tb>/minimax-h3"
[[ -n "${H3_DATA_ROOT:-}" ]] || die "H3_DATA_ROOT is empty."
[[ -d "${H3_DATA_ROOT}" ]] || die "Data root does not exist: ${H3_DATA_ROOT}"
[[ -w "${H3_DATA_ROOT}" ]] || die "Data root is not writable: ${H3_DATA_ROOT}"

arch="$(uname -m)"
info "Architecture: ${arch}"
[[ "${arch}" == "aarch64" ]] || warn "Expected aarch64 for DGX Spark; got ${arch}."

mem_kb="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
mem_gib=$((mem_kb / 1024 / 1024))
info "Physical memory: ~${mem_gib} GiB"
(( mem_gib >= 115 )) || warn "Less than 115 GiB visible; verify this is the 128 GB GB10 system."

command -v docker >/dev/null || die "docker not found."
docker info >/dev/null 2>&1 || die "Docker daemon is unavailable or current user lacks permission."
docker compose version >/dev/null 2>&1 || die "docker compose plugin not found."
info "Docker: $(docker --version)"
info "Compose: $(docker compose version --short)"

if command -v nvidia-smi >/dev/null; then
  info "NVIDIA GPU:"
  nvidia-smi -L || warn "nvidia-smi -L failed. UMA memory columns may be unsupported, but GPU listing should normally work."
else
  warn "nvidia-smi not found."
fi

if command -v nvidia-container-cli >/dev/null; then
  info "NVIDIA Container Toolkit found: $(nvidia-container-cli --version 2>/dev/null | head -n1)"
else
  warn "nvidia-container-cli not found; GPU container launch may fail."
fi

avail_kb="$(df -Pk "${H3_DATA_ROOT}" | awk 'NR==2 {print $4}')"
avail_gib=$((avail_kb / 1024 / 1024))
info "Free space at H3_DATA_ROOT: ${avail_gib} GiB"
(( avail_gib >= 100 )) || die "At least 100 GiB free is required."
(( avail_gib >= 200 )) || warn "Less than 200 GiB free; baseline fits, but outputs and R2V will reduce headroom."

data_src="$(findmnt -no SOURCE --target "${H3_DATA_ROOT}" 2>/dev/null || true)"
root_src="$(findmnt -no SOURCE / 2>/dev/null || true)"
info "Data filesystem: ${data_src:-unknown}"
if [[ -n "${data_src}" && "${data_src}" == "${root_src}" ]]; then
  warn "H3_DATA_ROOT appears to be on the root filesystem, not the external 4 TB NVMe."
fi

port="${COMFYUI_PORT:-8188}"
if command -v ss >/dev/null && ss -ltn "sport = :${port}" | tail -n +2 | grep -q .; then
  warn "TCP port ${port} is already in use."
else
  info "TCP port ${port} is available."
fi

swap_gib="$(awk '/SwapTotal/ {print int($2/1024/1024)}' /proc/meminfo)"
info "Swap configured: ${swap_gib} GiB"
warn "Do not treat swap as usable GPU memory; sustained swap activity will make video generation unstable."

echo
echo "Preflight complete."
echo "Next: make build"
