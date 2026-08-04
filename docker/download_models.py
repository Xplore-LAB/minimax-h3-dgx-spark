#!/usr/bin/env python3
"""Download only the official Comfy-Org MiniMax H3 files needed by this deployment.

Files are downloaded directly into /opt/ComfyUI/models, which is a bind mount
backed by the user's external NVMe. No 40+ GB staging copy is written into the
container layer or host system disk.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import shutil
import sys

from huggingface_hub import hf_hub_download

REPO_ID = "Comfy-Org/MiniMax-H3"
MODEL_ROOT = Path("/opt/ComfyUI/models")

GROUPS = {
    "fl2va": [
        (
            "split_files/diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
            MODEL_ROOT / "diffusion_models/minimax_h3_fl2va_pruned_int8_convrot.safetensors",
        ),
        (
            "split_files/text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
            MODEL_ROOT / "text_encoders/qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors",
        ),
        (
            "split_files/vae/minimax_h3_video_vae_fp16.safetensors",
            MODEL_ROOT / "vae/minimax_h3_video_vae_fp16.safetensors",
        ),
        (
            "split_files/vae/minimax_h3_audio_vae_fp32.safetensors",
            MODEL_ROOT / "vae/minimax_h3_audio_vae_fp32.safetensors",
        ),
    ],
    "r2v": [
        (
            "split_files/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
            MODEL_ROOT / "diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors",
        )
    ],
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=sorted(GROUPS))
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    for remote_name, destination in GROUPS[args.group]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.is_file() and destination.stat().st_size > 1024 * 1024:
            print(f"[skip] {destination.name} already exists ({destination.stat().st_size} bytes)")
            continue

        print(f"[download] {remote_name}")
        downloaded = Path(
            hf_hub_download(
                repo_id=REPO_ID,
                filename=remote_name,
                local_dir=str(MODEL_ROOT),
                token=token,
            )
        )
        if not downloaded.is_file():
            raise FileNotFoundError(downloaded)

        # Source and target are both inside the external model bind mount.
        os.replace(downloaded, destination)
        print(f"[ready] {destination} ({destination.stat().st_size} bytes)")

    # Only remove the now-empty repository-style directory tree. The small
    # local Hugging Face metadata is retained for safe resume/update behavior.
    split_dir = MODEL_ROOT / "split_files"
    if split_dir.exists():
        shutil.rmtree(split_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
