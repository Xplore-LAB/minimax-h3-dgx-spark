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
import sys

from huggingface_hub import hf_hub_download
from safetensors import safe_open

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


def validate_model(path: Path) -> None:
    """Use the official parser without materializing any tensors in RAM/GPU."""
    with safe_open(str(path), framework="np") as model:
        if not model.keys():
            raise ValueError(f"model file has no tensors: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("group", choices=sorted(GROUPS))
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN") or None
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    for remote_name, destination in GROUPS[args.group]:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists():
            try:
                validate_model(destination)
            except Exception as exc:
                print(f"[invalid] {destination.name}: {exc}; downloading a replacement")
            else:
                print(f"[skip] {destination.name} passed safetensors validation ({destination.stat().st_size} bytes)")
                continue

        print(f"[download] {remote_name}")
        # Local Hub metadata can still mark a corrupt candidate as current.
        # Bypass that cache once if format validation fails, then fail closed.
        for force_download in (False, True):
            downloaded = Path(
                hf_hub_download(
                    repo_id=REPO_ID,
                    filename=remote_name,
                    local_dir=str(MODEL_ROOT),
                    token=token,
                    force_download=force_download,
                )
            )
            if not downloaded.is_file():
                raise FileNotFoundError(downloaded)

            # A failed check must leave the old destination intact. safe_open
            # validates headers, shapes and data bounds without get_tensor.
            try:
                validate_model(downloaded)
            except Exception as exc:
                if force_download:
                    raise
                print(f"[retry] {remote_name}: {exc}; forcing one fresh download")
            else:
                break

        # Source and target are both inside the external model bind mount.
        os.replace(downloaded, destination)
        print(f"[ready] {destination} ({destination.stat().st_size} bytes)")

    # Only remove empty repository-style directories; retain any partial files
    # from an interrupted or concurrent download for resume/inspection.
    split_dir = MODEL_ROOT / "split_files"
    if split_dir.exists():
        for directory in sorted(split_dir.rglob("*"), reverse=True):
            if directory.is_dir():
                try:
                    directory.rmdir()
                except OSError:
                    pass
        try:
            split_dir.rmdir()
        except OSError:
            pass

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
