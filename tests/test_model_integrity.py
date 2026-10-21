"""Offline regressions for corrupt model acceptance; no real .env or GPU used."""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from huggingface_hub import file_download as hub_file_download
from huggingface_hub._local_folder import write_download_metadata
from safetensors import SafetensorError

ROOT = Path(__file__).resolve().parents[1]


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


header_check = load_module("header_check", ROOT / "scripts/verify_safetensors.py")
downloader = load_module("model_downloader", ROOT / "docker/download_models.py")


def tensor_file(payload: bytes = b"\x00\x00\x80\x3f", *, header: dict | None = None) -> bytes:
    if header is None:
        header = {"weight": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}}
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    return struct.pack("<Q", len(encoded)) + encoded + payload


class ModelIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.destination = self.root / "models/diffusion_models/model.safetensors"
        self.destination.parent.mkdir(parents=True)
        self.candidate = self.root / "models/split_files/diffusion_models/model.safetensors"
        self.candidate.parent.mkdir(parents=True)

    def write(self, path: Path, data: bytes) -> Path:
        path.write_bytes(data)
        return path

    def download(self, hub_download, *, remote_name="split_files/model.safetensors"):
        with (
            patch.object(downloader, "MODEL_ROOT", self.root / "models"),
            patch.object(downloader, "GROUPS", {"fl2va": [(remote_name, self.destination)]}),
            patch.object(downloader, "hf_hub_download", hub_download),
            patch.object(sys, "argv", ["download_models.py", "fl2va"]),
        ):
            return downloader.main()

    def test_host_rejects_one_byte_large_garbage_and_truncated_models(self):
        cases = {
            "one byte": b"X",
            "large garbage": b"\x00" * (1024 * 1024 + 1),
            "truncated header": struct.pack("<Q", 20) + b"{}",
            "truncated tensor": tensor_file()[:-1],
            "trailing bytes": tensor_file() + b"X",
            "oversized header": struct.pack("<Q", header_check.MAX_HEADER_BYTES + 1),
        }
        for label, content in cases.items():
            with self.subTest(label=label):
                self.write(self.destination, content)
                with self.assertRaises(ValueError):
                    header_check.check_header(self.destination)

    def test_host_accepts_valid_tiny_model_and_rejects_overlapping_offsets(self):
        self.write(self.destination, tensor_file())
        self.assertEqual(header_check.check_header(self.destination), self.destination.stat().st_size)
        overlapping = {
            "a": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
            "b": {"dtype": "U8", "shape": [4], "data_offsets": [0, 4]},
        }
        self.write(self.destination, tensor_file(header=overlapping))
        with self.assertRaises(ValueError):
            header_check.check_header(self.destination)

    def test_official_parser_rejects_shape_size_mismatch(self):
        invalid_shape = {"weight": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}}
        self.write(self.destination, tensor_file(header=invalid_shape))
        with self.assertRaises(SafetensorError):
            downloader.validate_model(self.destination)

    def test_small_valid_existing_file_skips_network(self):
        self.write(self.destination, tensor_file())
        def no_network(**kwargs):
            self.fail("a valid existing model should not be downloaded")
        self.assertEqual(self.download(no_network), 0)

    def test_large_corrupt_existing_file_is_replaced_only_after_valid_download(self):
        self.write(self.destination, b"\x00" * (1024 * 1024 + 1))
        valid = tensor_file()
        self.write(self.candidate, valid)
        calls = []
        def fake_hub(**kwargs):
            calls.append(kwargs)
            return str(self.candidate)
        self.assertEqual(self.download(fake_hub), 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(self.destination.read_bytes(), valid)
        self.assertFalse(self.candidate.exists())

    def test_invalid_download_preserves_previous_file_and_never_reports_ready(self):
        previous = b"old interrupted download"
        self.write(self.destination, previous)
        self.write(self.candidate, tensor_file()[:-1])
        output = io.StringIO()
        with contextlib.redirect_stdout(output), self.assertRaises(SafetensorError):
            self.download(lambda **kwargs: str(self.candidate))
        self.assertEqual(self.destination.read_bytes(), previous)
        self.assertTrue(self.candidate.exists())
        self.assertNotIn("[ready]", output.getvalue())

    def test_missing_download_does_not_create_destination(self):
        with self.assertRaises(FileNotFoundError):
            self.download(lambda **kwargs: str(self.candidate))
        self.assertFalse(self.destination.exists())

    def run_cached_candidate_case(self, replacement: bytes):
        """Exercise real Hub local_dir metadata caching with transfer mocked."""
        local_dir = self.root / "models"
        remote_name = self.candidate.relative_to(local_dir).as_posix()
        self.write(self.candidate, tensor_file()[:-1])
        commit_hash = "a" * 40
        etag = "synthetic-test-etag"
        write_download_metadata(local_dir, remote_name, commit_hash, etag)

        def transfer(**kwargs):
            self.assertTrue(kwargs["force_download"])
            # Real Hub chooses/unlinks the cached candidate; only its byte
            # transfer is replaced. This is not a fake hf_hub_download result.
            Path(kwargs["destination_path"]).write_bytes(replacement)

        real_hub_download = downloader.hf_hub_download
        with (
            patch.dict(os.environ, {"HF_TOKEN": ""}),
            patch("huggingface_hub.utils._headers.get_token_to_send", return_value=None),
            patch.object(
                hub_file_download, "_get_metadata_or_catch_error",
                return_value=("https://example.invalid/model", etag, commit_hash, len(replacement), None, None),
            ),
            patch.object(hub_file_download, "_download_to_tmp_and_move", side_effect=transfer) as transfers,
            patch.object(downloader, "hf_hub_download", wraps=real_hub_download) as requests,
        ):
            output = io.StringIO()
            try:
                with contextlib.redirect_stdout(output):
                    result = self.download(requests, remote_name=remote_name)
            finally:
                self.assertEqual([call.kwargs["force_download"] for call in requests.call_args_list], [False, True])
                self.assertEqual(transfers.call_count, 1)
                self.assertIn("[retry]", output.getvalue())
        return result

    def test_valid_metadata_bad_cached_candidate_forces_one_real_hub_transfer(self):
        self.write(self.destination, b"old interrupted target")
        valid = tensor_file()
        self.assertEqual(self.run_cached_candidate_case(valid), 0)
        self.assertEqual(self.destination.read_bytes(), valid)
        self.assertFalse(self.candidate.exists())

    def test_cached_candidate_retry_remains_bad_preserves_target_and_stops(self):
        previous = b"old target must remain intact"
        self.write(self.destination, previous)
        with self.assertRaises(SafetensorError):
            self.run_cached_candidate_case(tensor_file()[:-1])
        self.assertEqual(self.destination.read_bytes(), previous)
        self.assertTrue(self.candidate.exists())

    def test_cleanup_preserves_other_partial_downloads(self):
        self.write(self.destination, tensor_file())
        partial = self.candidate.parent / "other-model.incomplete"
        self.write(partial, b"partial")
        self.assertEqual(self.download(lambda **kwargs: self.fail("unexpected download")), 0)
        self.assertEqual(partial.read_bytes(), b"partial")

    def test_host_shell_fails_for_corrupt_files_and_passes_valid_files(self):
        # Copy only the scripts under test and use a synthetic .env. The real
        # project's private configuration is never sourced by this test.
        project = self.root / "isolated-project"
        scripts = project / "scripts"
        scripts.mkdir(parents=True)
        for name in ["common.sh", "verify_models.sh", "verify_safetensors.py"]:
            shutil.copyfile(ROOT / "scripts" / name, scripts / name)
        data_root = self.root / "test-data"
        (project / ".env").write_text(f"H3_DATA_ROOT={data_root}\n")
        files = []
        for _, destination in downloader.GROUPS["fl2va"]:
            local = data_root / "models" / destination.relative_to(downloader.MODEL_ROOT)
            local.parent.mkdir(parents=True, exist_ok=True)
            files.append(local)
        for content, expected_exit in [(b"X", 1), (tensor_file(), 0)]:
            for path in files:
                path.write_bytes(content)
            result = subprocess.run(
                ["bash", str(scripts / "verify_models.sh")],
                cwd=self.root,
                env={"PATH": os.environ["PATH"]},
                text=True, capture_output=True, check=False,
            )
            self.assertEqual(result.returncode, expected_exit, result.stdout + result.stderr)
        optional = data_root / "models/diffusion_models/minimax_h3_ref2va_pruned_int8_convrot.safetensors"
        optional.write_bytes(b"")
        result = subprocess.run(
            ["bash", str(scripts / "verify_models.sh")],
            env={"PATH": os.environ["PATH"]}, capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 1, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
