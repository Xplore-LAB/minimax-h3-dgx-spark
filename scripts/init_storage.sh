#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"

DATA_ROOT="${1:-}"
[[ -n "${DATA_ROOT}" ]] || { echo "Usage: $0 /mnt/<4tb>/minimax-h3"; exit 2; }
[[ "${DATA_ROOT}" = /* ]] || { echo "ERROR: DATA_ROOT must be an absolute path."; exit 2; }
[[ "${DATA_ROOT}" != *" "* ]] || { echo "ERROR: DATA_ROOT must not contain spaces."; exit 2; }

parent="$(dirname "${DATA_ROOT}")"
[[ -d "${parent}" ]] || { echo "ERROR: parent mount path does not exist: ${parent}"; exit 2; }

create_cmd=(mkdir -p)
if [[ ! -w "${parent}" ]]; then
  command -v sudo >/dev/null || { echo "ERROR: ${parent} is not writable and sudo is unavailable."; exit 2; }
  create_cmd=(sudo mkdir -p)
fi

"${create_cmd[@]}" "${DATA_ROOT}"/{models/diffusion_models,models/text_encoders,models/vae,input,output,temp,user,custom_nodes,hf-cache,logs}

if [[ ! -w "${DATA_ROOT}" ]]; then
  sudo chown -R "$(id -u):$(id -g)" "${DATA_ROOT}"
fi

cp -n .env.example .env

python3 - "${DATA_ROOT}" "$(id -u)" "$(id -g)" <<'PY'
from pathlib import Path
import sys, re
path = Path(".env")
data_root, uid, gid = sys.argv[1:]
text = path.read_text()
replacements = {
    "H3_DATA_ROOT": data_root,
    "HOST_UID": uid,
    "HOST_GID": gid,
}
for key, value in replacements.items():
    pattern = rf"(?m)^{re.escape(key)}=.*$"
    line = f"{key}={value}"
    if re.search(pattern, text):
        text = re.sub(pattern, line, text)
    else:
        text += f"\n{line}\n"
path.write_text(text)
PY

chmod 600 .env
echo "Initialized: ${DATA_ROOT}"
echo "Configuration: ${PROJECT_DIR}/.env"
echo
echo "Next: make preflight"
