#!/usr/bin/env bash
# 下载 ComfyUI 源码到 comfyui-src/（本仓库不携带上游源码，构建前先跑一次）
# 用 codeload 镜像：容器/构建机 GitHub 直连常 TLS 断连，codeload 更稳。
set -Eeuo pipefail

REF="${COMFYUI_REF:-v0.30.0}"
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$DIR/comfyui-src"

if [[ -d "$SRC/comfy" ]]; then
  echo "[prepare] comfyui-src/ 已存在，跳过下载"
  exit 0
fi

echo "[prepare] 下载 ComfyUI ${REF} (codeload)..."
mkdir -p "$SRC"
curl -L --retry 3 -o /tmp/comfyui-src.tar.gz \
  "https://codeload.github.com/comfyanonymous/ComfyUI/tar.gz/refs/tags/${REF}"
tar xzf /tmp/comfyui-src.tar.gz -C "$SRC" --strip-components=1
rm -f /tmp/comfyui-src.tar.gz
echo "[prepare] 完成: ${SRC}"
