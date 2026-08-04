#!/usr/bin/env bash
set -Eeuo pipefail

cd /opt/ComfyUI

args=(
  python main.py
  --listen 0.0.0.0
  --port 8188
  --disable-auto-launch
  --reserve-vram "${RESERVE_VRAM_GB:-12}"
  --vram-headroom "${VRAM_HEADROOM_GB:-3}"
  --async-offload "${ASYNC_OFFLOAD_STREAMS:-2}"
  --mmap-torch-files
  --verbose INFO
)


case "${CACHE_MODE:-none}" in
  none)
    args+=(--cache-none)
    ;;
  lru)
    args+=(--cache-lru "${CACHE_LRU:-1}")
    ;;
  classic)
    args+=(--cache-classic)
    ;;
  *)
    echo "ERROR: unsupported CACHE_MODE=${CACHE_MODE}; use none, lru, or classic." >&2
    exit 2
    ;;
esac

if [[ -n "${COMFYUI_EXTRA_ARGS:-}" ]]; then
  # 该变量仅由管理员在 .env 中设置；用于附加官方 ComfyUI 命令行参数。
  read -r -a extra_args <<< "${COMFYUI_EXTRA_ARGS}"
  args+=("${extra_args[@]}")
fi

echo "[H3] Launching: ${args[*]}"
exec "${args[@]}"
