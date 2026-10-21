#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/common.sh"

[[ -n "${H3_DATA_ROOT:-}" ]] || die "H3_DATA_ROOT is empty. Run: make init DATA_ROOT=/mnt/<4tb>/minimax-h3"
command -v python3 >/dev/null || die "python3 is required for the model header precheck."

required=(
  "models/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors"
  "models/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors"
  "models/vae/minimax_h3_video_vae_fp16.safetensors"
  "models/vae/minimax_h3_audio_vae_fp32.safetensors"
)

failed=0
echo "Model header/offset precheck (no hashes or tensor values checked):"
for rel in "${required[@]}"; do
  file="${H3_DATA_ROOT}/${rel}"
  if ! python3 "${PROJECT_DIR}/scripts/verify_safetensors.py" "${file}"; then
    failed=1
  fi
done

optional="${H3_DATA_ROOT}/models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
if [[ -e "${optional}" ]]; then
  if ! python3 "${PROJECT_DIR}/scripts/verify_safetensors.py" "${optional}"; then
    failed=1
  fi
else
  echo "  INFO Ref2VA not installed (optional)."
fi

(( failed == 0 )) || exit 1
echo "Required FL2VA files passed the header/offset precheck."
