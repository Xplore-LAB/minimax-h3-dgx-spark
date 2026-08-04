#!/usr/bin/env bash
set -Eeuo pipefail
source "$(dirname "$0")/common.sh"

[[ -f .env ]] || die ".env not found."
[[ -d "${H3_DATA_ROOT:-}" ]] || die "Invalid H3_DATA_ROOT."

info "Downloading MiniMax H3 Ref2VA diffusion model..."
info "Additional storage: approximately 21 GB."
info "Files are written directly to the external model volume."

docker compose run --rm --no-deps \
  --entrypoint python comfyui /usr/local/bin/h3-download-models r2v

"$(dirname "$0")/verify_models.sh"
