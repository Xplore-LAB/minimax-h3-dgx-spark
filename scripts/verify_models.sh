#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/common.sh"

required=(
  "models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
  "models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
  "models/vae/minimax_h3_video_vae_fp16.safetensors"
  "models/vae/minimax_h3_audio_vae_fp32.safetensors"
)

failed=0
echo "Model verification:"
for rel in "${required[@]}"; do
  file="${H3_DATA_ROOT}/${rel}"
  if [[ -s "${file}" ]]; then
    size="$(du -h "${file}" | awk "{print \$1}")"
    printf "  OK   %-8s %s\n" "${size}" "${rel}"
  else
    printf "  MISS %-8s %s\n" "-" "${rel}"
    failed=1
  fi
done

optional="${H3_DATA_ROOT}/models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
if [[ -s "${optional}" ]]; then
  size="$(du -h "${optional}" | awk "{print \$1}")"
  printf "  OK   %-8s %s\n" "${size}" "models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
else
  echo "  INFO Ref2VA not installed (optional)."
fi

(( failed == 0 )) || exit 1
echo "Required FL2VA files are present."
